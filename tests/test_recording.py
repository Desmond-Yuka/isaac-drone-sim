"""Video clock, synchronized views, mosaic pixels and encoder failure handling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from isaac_drone.viz.video import MultiViewRecorder, _FFmpegWriter, ensure_video_dependencies


def config(**changes):
    result = {"enabled": True, "fps": 60, "width": 8, "height": 6,
              "cameras": ["overview", "follow", "top"]}
    result.update(changes)
    return result


def views(settings=None):
    settings = settings or config()
    return {name: np.full((settings["height"], settings["width"], 4), index * 70,
                          dtype=np.uint8)
            for index, name in enumerate(settings["cameras"], start=1)}


class FakeWriter:
    def __init__(self, path, **settings):
        self.path = Path(path)
        self.settings = settings
        self.frames = []
        self.close_count = 0
        self.write_error = None
        self.close_error = None

    def write(self, frame):
        if self.write_error:
            raise self.write_error
        assert frame.dtype == np.uint8 and frame.flags.c_contiguous
        assert frame.shape == (self.settings["height"], self.settings["width"], 3)
        self.frames.append(frame.copy())

    def close(self):
        self.close_count += 1
        if self.close_error:
            raise self.close_error


def make_recorder(path, settings=None):
    writers = {}

    def factory(destination, **kwargs):
        writer = FakeWriter(destination, **kwargs)
        writers[Path(destination).stem] = writer
        return writer

    return MultiViewRecorder(path, settings or config(), writer_factory=factory), writers


def frame_index(path):
    return [json.loads(line) for line in (path / "video_frames.jsonl").read_text().splitlines()]


def test_simulation_clock_produces_exact_duration_and_synchronizes_all_outputs(tmp_path):
    recorder, writers = make_recorder(tmp_path)
    snapshot = views()
    # 200 Hz physics for 20 seconds; t=duration is deliberately excluded.
    assert recorder.due(0)
    for step in range(4000):
        time_s = step / 200
        if recorder.due(time_s):
            assert recorder.capture(time_s, snapshot) == 1
    recorder.close()
    assert set(writers) == {"overview", "follow", "top", "combined"}
    assert all(len(writer.frames) == 1200 for writer in writers.values())
    assert all(writer.close_count == 1 for writer in writers.values())
    summary = recorder.summary()
    assert summary["frames"] == 1200
    assert summary["encoded_duration_s"] == 20
    assert summary["closed"] and not summary["failed"]
    assert summary["camera_order"] == ["overview", "follow", "top"]
    assert summary["layout"] == "overview_above_two"
    assert all(not Path(video["filename"]).is_absolute() for video in summary["videos"].values())
    assert summary["videos"]["combined"]["width"] == 8
    assert summary["videos"]["combined"]["height"] == 10
    rows = frame_index(tmp_path)
    assert len(rows) == 1200
    assert rows[0] == {"frame_index": 0, "video_time_s": 0, "simulation_time_s": 0}
    assert rows[-1]["video_time_s"] == pytest.approx(1199 / 60)
    assert all(0 <= row["simulation_time_s"] - row["video_time_s"] < .005 + 1e-12 for row in rows)


def test_cadence_repeated_queries_and_clock_gaps_are_explicit_in_frame_index(tmp_path):
    recorder, writers = make_recorder(tmp_path, config(fps=10))
    assert recorder.due(0) and recorder.due(0)
    assert recorder.capture(0, views()) == 1
    assert not recorder.due(0)
    assert recorder.capture(.05, views()) == 0
    assert recorder.capture(.3, views()) == 3
    assert not recorder.due(.3)
    assert recorder.due(.4)
    assert all(len(writer.frames) == 4 for writer in writers.values())
    assert [row["simulation_time_s"] for row in frame_index(tmp_path)] == [0, .3, .3, .3]
    with pytest.raises(ValueError, match="backwards"):
        recorder.capture(.1, views())
    recorder.close()


@pytest.mark.parametrize("end, expected", [(20, 1200), (.035, 3), (.1, 6), (0, 0)])
def test_exclusive_end_time_includes_partial_last_interval_without_an_extra_frame(tmp_path, end, expected):
    recorder, writers = make_recorder(tmp_path)
    # A single final sample also exercises filling all pending video instants.
    assert recorder.capture(end, views(), end_time_s=end) == expected
    assert not recorder.due(end, end_time_s=end)
    assert recorder.capture(end + .01, views(), end_time_s=end) == 0
    recorder.close()
    assert all(len(writer.frames) == expected for writer in writers.values())
    rows = frame_index(tmp_path)
    assert all(row["video_time_s"] < end for row in rows)
    assert all(row["simulation_time_s"] == end for row in rows)


def test_exclusive_end_boundary_tolerates_floating_point_roundoff(tmp_path):
    recorder, _ = make_recorder(tmp_path, config(fps=10))
    assert recorder.capture(.30000000000000004, views(), end_time_s=.30000000000000004) == 3
    assert not recorder.due(.30000000000000004, end_time_s=.30000000000000004)
    recorder.close()


def test_mosaic_preserves_camera_order_averages_pixels_and_pads_to_even_height(tmp_path):
    recorder, writers = make_recorder(tmp_path)
    snapshot = views()
    # One 2x2 block should average to 6, not sample its upper-left pixel (0).
    snapshot["follow"][:2, :2, :3] = np.array([[0, 4], [8, 12]])[:, :, None]
    recorder.capture(0, snapshot)
    recorder.close()
    mosaic = writers["combined"].frames[0]
    np.testing.assert_array_equal(mosaic[:6], snapshot["overview"][:, :, :3])
    np.testing.assert_array_equal(mosaic[6, 0], [6, 6, 6])
    assert np.all(mosaic[6:9, 4:] == 210)
    assert np.all(mosaic[-1] == 0)
    assert writers["follow"].frames[0][0, 0, 0] == 0


@pytest.mark.parametrize("cameras, shape, layout", [
    (["follow"], (6, 8, 3), "single"),
    (["top", "overview"], (6, 16, 3), "side_by_side"),
])
def test_one_and_two_camera_mosaics(tmp_path, cameras, shape, layout):
    settings = config(cameras=cameras)
    recorder, writers = make_recorder(tmp_path, settings)
    recorder.capture(0, views(settings))
    recorder.close()
    mosaic = writers["combined"].frames[0]
    assert mosaic.shape == shape
    assert recorder.summary()["layout"] == layout
    assert np.all(mosaic[:, :8] == 70)
    if len(cameras) == 2:
        assert np.all(mosaic[:, 8:] == 140)


@pytest.mark.parametrize("bad_frame", [
    np.zeros((6, 8, 3), dtype=float),
    np.zeros((6, 8, 2), dtype=np.uint8),
    np.zeros((8, 6, 3), dtype=np.uint8),
    np.zeros((6, 8), dtype=np.uint8),
])
def test_bad_view_is_rejected_before_any_encoder_advances(tmp_path, bad_frame):
    recorder, writers = make_recorder(tmp_path)
    snapshot = views()
    snapshot["top"] = bad_frame
    with pytest.raises(ValueError, match="uint8 RGB/RGBA"):
        recorder.capture(0, snapshot)
    assert all(not writer.frames for writer in writers.values())
    assert frame_index(tmp_path) == []
    assert recorder.due(0)
    assert not recorder.summary()["failed"]
    recorder.capture(0, views())
    recorder.close()


def test_missing_or_extra_view_is_rejected_without_partial_writes(tmp_path):
    recorder, writers = make_recorder(tmp_path)
    snapshot = views()
    del snapshot["follow"]
    with pytest.raises(ValueError, match="exactly"):
        recorder.capture(0, snapshot)
    snapshot = views()
    snapshot["unconfigured"] = snapshot["top"]
    with pytest.raises(ValueError, match="exactly"):
        recorder.capture(0, snapshot)
    assert all(not writer.frames for writer in writers.values())
    recorder.close()


@pytest.mark.parametrize("bad_time", [True, -1, "1", float("nan"), float("inf")])
def test_invalid_clock_values_are_rejected(tmp_path, bad_time):
    recorder, _ = make_recorder(tmp_path)
    with pytest.raises(ValueError, match="simulation time"):
        recorder.due(bad_time)
    with pytest.raises(ValueError, match="simulation time"):
        recorder.capture(bad_time, views())
    recorder.close()


@pytest.mark.parametrize("changes", [
    {"fps": 0}, {"fps": True}, {"fps": np.nan}, {"width": 7}, {"height": 5},
    {"width": False}, {"height": 0}, {"cameras": []}, {"cameras": "follow"},
    {"cameras": ["follow", "follow"]}, {"cameras": ["../escape"]},
    {"cameras": ["combined"]}, {"cameras": ["a", "b", "c", "d"]},
])
def test_bad_configuration_creates_no_files(tmp_path, changes):
    with pytest.raises(ValueError):
        make_recorder(tmp_path, config(**changes))
    assert list(tmp_path.iterdir()) == []


def test_requires_existing_run_directory(tmp_path):
    with pytest.raises(ValueError, match="existing run directory"):
        make_recorder(tmp_path / "not-created")
    assert list(tmp_path.iterdir()) == []


def test_encoder_write_failure_stops_recording_and_still_finalizes_every_stream(tmp_path):
    recorder, writers = make_recorder(tmp_path)
    recorder.capture(0, views())
    writers["follow"].write_error = OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        recorder.capture(1 / 60, views())
    assert recorder.summary()["failed"]
    assert recorder.summary()["frames"] == 1
    assert recorder.summary()["videos"]["overview"]["frames"] == 2
    with pytest.raises(RuntimeError, match="I/O failure"):
        recorder.capture(2 / 60, views())
    recorder.close()
    recorder.close()
    assert all(writer.close_count == 1 for writer in writers.values())
    assert len(frame_index(tmp_path)) == 1


def test_close_failure_reports_error_after_closing_every_stream(tmp_path):
    recorder, writers = make_recorder(tmp_path)
    writers["overview"].close_error = RuntimeError("encoder failed")
    with pytest.raises(RuntimeError, match="overview: encoder failed"):
        recorder.close()
    assert all(writer.close_count == 1 for writer in writers.values())
    assert recorder._index_stream.closed
    recorder.close()
    assert recorder.summary()["failed"] and recorder.summary()["closed"]
    with pytest.raises(ValueError, match="close"):
        recorder.due(0)


def test_constructor_failure_closes_already_created_writers(tmp_path):
    writers = []

    def factory(path, **settings):
        if Path(path).stem == "top":
            raise OSError("cannot start encoder")
        writer = FakeWriter(path, **settings)
        writers.append(writer)
        return writer

    with pytest.raises(OSError, match="cannot start encoder"):
        MultiViewRecorder(tmp_path, config(), writer_factory=factory)
    assert len(writers) == 2 and all(writer.close_count == 1 for writer in writers)


def test_keyboard_interrupt_finalizes_recorded_frames(tmp_path):
    recorder, writers = make_recorder(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with recorder:
            recorder.capture(0, views())
            raise KeyboardInterrupt()
    assert all(writer.close_count == 1 and len(writer.frames) == 1 for writer in writers.values())
    assert len(frame_index(tmp_path)) == 1


def test_dependency_preflight_is_lazy_and_gives_install_command(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    recorder, _ = make_recorder(tmp_path)
    recorder.close()
    with pytest.raises(RuntimeError, match=r"pip install -e '.\[video\]'"):
        ensure_video_dependencies()


def test_preflight_checks_executable_failure(monkeypatch):
    module = SimpleNamespace(get_ffmpeg_exe=lambda: "/not/an/ffmpeg")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", module)
    with pytest.raises(RuntimeError, match="working FFmpeg"):
        ensure_video_dependencies()


def test_preflight_requires_h264_encoder(monkeypatch):
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=" V..... libvpx"))
    with pytest.raises(RuntimeError, match="libx264"):
        ensure_video_dependencies()


def test_ffmpeg_exit_failure_is_not_silently_accepted(tmp_path, monkeypatch):
    # A fake process exercises the final exit check without optional dependencies.
    class Process:
        returncode = None

        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True
            self.stdin = SimpleNamespace(close=lambda: None)
            kwargs["stderr"].write(b"encoder initialization failed")

        def wait(self, timeout=None):
            self.returncode = 1
            return 1

        def poll(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", Process)
    writer = _FFmpegWriter(tmp_path / "failed.mp4", executable="ffmpeg", fps=60, width=8, height=6)
    with pytest.raises(RuntimeError, match="encoder initialization failed"):
        writer.close()
    assert writer._stderr.closed
    writer.close()


def test_real_mp4_streams_have_matching_frame_counts_and_exact_dimensions(tmp_path):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    settings = config(fps=10, width=18, height=14)
    with MultiViewRecorder(tmp_path, settings) as recorder:
        for index in range(5):
            recorder.capture(index / 10, views(settings))
    for video in recorder.summary()["videos"].values():
        path = tmp_path / video["filename"]
        count, duration = ffmpeg.count_frames_and_secs(str(path))
        assert count == 5
        assert duration == pytest.approx(.5, abs=.01)
        reader = ffmpeg.read_frames(str(path))
        metadata = next(reader)
        assert metadata["size"] == (video["width"], video["height"])
        assert metadata["fps"] == 10
        reader.close()
