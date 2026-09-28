"""Optional, streaming simulation video with synchronized camera and mosaic files.

The simulation clock sets the frame cadence; wall-clock pacing has no effect.
Only this module's encoder preflight imports the optional imageio-ffmpeg package.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np


def ensure_video_dependencies() -> str:
    """Return a working FFmpeg executable, or fail before simulation startup."""
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "Video recording requires the optional video dependency: "
            "install it with python -m pip install -e '.[video]'"
        ) from error
    try:
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        available = subprocess.run([executable, "-hide_banner", "-encoders"], check=True, timeout=10,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise RuntimeError("Video recording requires a working FFmpeg executable") from error
    if not re.search(r"\blibx264\b", available.stdout):
        raise RuntimeError("Video recording requires FFmpeg with the libx264 H.264 encoder")
    return executable


class _FFmpegWriter:
    """Write raw RGB to FFmpeg and check its exit status when finalizing MP4.

    A direct process is used because imageio_ffmpeg.write_frames does not raise
    on every nonzero encoder exit during generator.close(). Stderr goes to a
    temporary file so an encoder failure cannot deadlock a full stderr pipe.
    """

    def __init__(self, path, *, executable, fps, width, height):
        self._closed = False
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-", "-an",
            "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL,
                                             stderr=self._stderr, start_new_session=True)
        except BaseException:
            self._stderr.close()
            raise

    def _error(self, prefix):
        self._stderr.seek(0, 2)
        self._stderr.seek(max(0, self._stderr.tell() - 8192))
        detail = self._stderr.read().decode("utf-8", errors="replace").strip()
        return RuntimeError(f"{prefix}: {detail}" if detail else prefix)

    def write(self, frame):
        if self._closed:
            raise ValueError("Cannot write to a closed video encoder")
        if self._process.poll() is not None:
            raise self._error(f"FFmpeg stopped with exit code {self._process.returncode}")
        try:
            self._process.stdin.write(frame.tobytes())
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise self._error("FFmpeg could not accept a video frame") from error

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                # The exit status and stderr below identify the actual failure.
                pass
            try:
                code = self._process.wait(timeout=30)
            except subprocess.TimeoutExpired as error:
                self._process.kill()
                self._process.wait()
                raise self._error("FFmpeg timed out while finalizing video") from error
            if code:
                raise self._error(f"FFmpeg failed with exit code {code}")
        except BaseException:
            # Also reap the process if finalization itself is interrupted.
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait()
            raise
        finally:
            self._stderr.close()


def _positive_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"recording.{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"recording.{name} must be a finite positive number")
    return value


def _even_size(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < 2 or value % 2):
        raise ValueError(f"recording.{name} must be a positive even integer")
    return int(value)


class MultiViewRecorder:
    """Stream one MP4 per camera plus ``combined.mp4`` into an existing run.

    ``config`` is the recording section, with fps, width, height and a list of
    one to three camera names. The optional writer factory is called as
    ``factory(path, fps=..., width=..., height=...)`` and returns an object with
    ``write(rgb_uint8_array)`` and ``close()`` methods. It must consume each frame
    during write, without retaining mutable input buffers.

    Capture validates all views before writing any. If a simulation update
    crosses several frame instants, the same snapshot fills those instants;
    ``video_frames.jsonl`` records their actual simulation sample time. A write
    failure poisons the recorder to avoid continuing with divergent files.
    """

    def __init__(self, directory, config, *, writer_factory=None):
        self.path = Path(directory)
        self.fps = _positive_number(config["fps"], "fps")
        self.width = _even_size(config["width"], "width")
        self.height = _even_size(config["height"], "height")
        cameras = config["cameras"]
        if (not isinstance(cameras, list) or not 1 <= len(cameras) <= 3
                or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name)
                       or name == "combined" for name in cameras)
                or len(set(cameras)) != len(cameras)):
            raise ValueError("recording.cameras must be a list of one to three unique camera names")
        if not self.path.is_dir():
            raise ValueError("Video output directory must be an existing run directory")
        self.cameras = tuple(cameras)
        count = len(self.cameras)
        self._combined_width = self.width * (2 if count == 2 else 1)
        combined_height = self.height + (self.height // 2 if count == 3 else 0)
        self._combined_height = combined_height + combined_height % 2
        self._layout = {1: "single", 2: "side_by_side", 3: "overview_above_two"}[count]
        self._writers = {}
        self._video_frames = {name: 0 for name in (*self.cameras, "combined")}
        self._frames = 0
        self._closed = False
        self._failed = False
        self._errors = []
        self._last_capture_time = None
        self._index_stream = None
        if writer_factory is None:
            executable = ensure_video_dependencies()

            def writer_factory(path, **settings):
                return _FFmpegWriter(path, executable=executable, **settings)

        try:
            # Exclusive creation avoids overwriting another recorder's output.
            self._index_stream = (self.path / "video_frames.jsonl").open("x", encoding="utf-8")
            for name in self._video_frames:
                width, height = self._dimensions(name)
                self._writers[name] = writer_factory(self.path / f"{name}.mp4", fps=self.fps,
                                                     width=width, height=height)
        except BaseException:
            self._failed = True
            try:
                self.close()
            except Exception:
                pass
            raise

    def _dimensions(self, name):
        return ((self._combined_width, self._combined_height) if name == "combined"
                else (self.width, self.height))

    def _check_open(self):
        if self._closed:
            raise ValueError("Cannot record video after close")
        if self._failed:
            raise RuntimeError("Video recorder had an I/O failure; close it before continuing")

    @staticmethod
    def _time(time_s):
        if isinstance(time_s, (bool, np.bool_)) or not isinstance(time_s, Real):
            raise ValueError("Video simulation time must be finite and nonnegative")
        time_s = float(time_s)
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Video simulation time must be finite and nonnegative")
        return time_s

    def _is_due(self, time_s, end_time_s=None):
        next_time = self._frames / self.fps
        if end_time_s is not None and (next_time >= end_time_s
                                      or math.isclose(next_time, end_time_s, rel_tol=0, abs_tol=1e-9)):
            return False
        return time_s >= next_time or math.isclose(time_s, next_time, rel_tol=1e-12, abs_tol=1e-12)

    def due(self, time_s, *, end_time_s=None) -> bool:
        """Whether a frame is due, excluding scheduled times at/after the end.

        The optional simulation endpoint is exclusive for video timestamps.
        The final physics state may supply a last frame scheduled before it.
        """
        self._check_open()
        end_time_s = None if end_time_s is None else self._time(end_time_s)
        return self._is_due(self._time(time_s), end_time_s)

    def _prepare_frames(self, frames):
        if not isinstance(frames, Mapping) or set(frames) != set(self.cameras):
            raise ValueError("Video frames must contain exactly the configured camera names")
        prepared = {}
        for name in self.cameras:
            frame = np.asarray(frames[name])
            if (frame.dtype != np.uint8 or frame.ndim != 3
                    or frame.shape[:2] != (self.height, self.width) or frame.shape[2] not in (3, 4)):
                raise ValueError(f"Video frame {name!r} must be uint8 RGB/RGBA "
                                 f"with shape ({self.height}, {self.width}, 3 or 4)")
            prepared[name] = np.ascontiguousarray(frame[:, :, :3])
        prepared["combined"] = self._combine(prepared)
        return prepared

    def _combine(self, frames):
        ordered = [frames[name] for name in self.cameras]
        if len(ordered) == 1:
            return ordered[0]
        if len(ordered) == 2:
            return np.concatenate(ordered, axis=1)
        mosaic = np.zeros((self._combined_height, self._combined_width, 3), dtype=np.uint8)
        mosaic[:self.height] = ordered[0]
        for index, frame in enumerate(ordered[1:]):
            # Average each 2x2 pixel box using uint16 to avoid uint8 overflow.
            small = frame.astype(np.uint16).reshape(self.height // 2, 2,
                                                    self.width // 2, 2, 3).sum(axis=(1, 3)) // 4
            left = index * (self.width // 2)
            mosaic[self.height:self.height + self.height // 2,
                   left:left + self.width // 2] = small
        return mosaic

    def capture(self, time_s, frames, *, end_time_s=None) -> int:
        """Write all due slots from this synchronized snapshot; return their count."""
        self._check_open()
        time_s = self._time(time_s)
        end_time_s = None if end_time_s is None else self._time(end_time_s)
        if self._last_capture_time is not None and time_s < self._last_capture_time:
            raise ValueError("Video simulation time cannot move backwards")
        if not self._is_due(time_s, end_time_s):
            return 0
        prepared = self._prepare_frames(frames)
        captured = 0
        try:
            while self._is_due(time_s, end_time_s):
                for name, writer in self._writers.items():
                    writer.write(prepared[name])
                    self._video_frames[name] += 1
                self._index_stream.write(json.dumps({
                    "frame_index": self._frames,
                    "video_time_s": self._frames / self.fps,
                    "simulation_time_s": time_s,
                }, allow_nan=False) + "\n")
                self._index_stream.flush()
                self._frames += 1
                captured += 1
            self._last_capture_time = time_s
        except BaseException as error:
            self._failed = True
            self._errors.append(f"capture: {type(error).__name__}: {error}")
            raise
        return captured

    def close(self):
        """Finalize every video, even after one fails; repeated calls are harmless."""
        if self._closed:
            return
        self._closed = True
        failures = []
        for name, writer in self._writers.items():
            try:
                writer.close()
            except BaseException as error:
                failures.append((name, error))
        if self._index_stream is not None:
            try:
                self._index_stream.close()
            except BaseException as error:
                failures.append(("video_frames.jsonl", error))
        if failures:
            self._failed = True
            self._errors.extend(f"{name}: {type(error).__name__}: {error}" for name, error in failures)
            description = "; ".join(f"{name}: {error}" for name, error in failures)
            raise RuntimeError(f"Could not finalize video recording: {description}") from failures[0][1]

    def summary(self) -> dict:
        """Return JSON-safe counts and relative paths, including failure state."""
        return {
            "enabled": True, "fps": self.fps, "width": self.width, "height": self.height,
            "frames": self._frames, "encoded_duration_s": self._frames / self.fps,
            "closed": self._closed, "failed": self._failed, "errors": list(self._errors),
            "camera_order": list(self.cameras), "layout": self._layout,
            "frame_index_file": "video_frames.jsonl",
            "videos": {
                name: {"filename": f"{name}.mp4", "frames": frames, "fps": self.fps,
                       "width": self._dimensions(name)[0], "height": self._dimensions(name)[1],
                       "encoded_duration_s": frames / self.fps}
                for name, frames in self._video_frames.items()
            },
        }

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
