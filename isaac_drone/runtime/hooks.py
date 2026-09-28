"""Standard run hooks: flown-path overlay, synchronized video, paced display, figures.

Hooks observe completed physics steps. None of them changes physics, control
timing or telemetry, so a run records identical data with or without them.
Simulator-specific objects (overlay prims, camera rigs, draw callbacks) are
injected, keeping this module importable without Isaac Sim.
"""
from __future__ import annotations

from isaac_drone.runtime.runner import RunHook


class PathOverlayHook(RunHook):
    """Accumulate the flown CoM path and push it, with the current setpoint, to an overlay.

    ``overlay`` needs ``update(flown_points, setpoint_position_w)``; ``trail``
    needs ``add(position_w)`` and ``points``. ``refresh`` is also what the
    display and video hooks call right before drawing a frame.
    """

    def __init__(self, overlay, trail):
        self.overlay, self.trail = overlay, trail
        self._loop = None

    def start(self, run):
        self._loop = run.loop

    def after_step(self, run, record):
        self.trail.add(record["post_step_state"]["position_w"])

    def refresh(self):
        loop = self._loop
        self.overlay.update(self.trail.points, loop.trajectory.sample(loop.time_s).position_w)


class VideoHook(RunHook):
    """Synchronized multi-camera recording sampled on the simulation clock.

    ``rig_factory(width, height, poses)`` returns an object with ``read(poses)``
    (RGB frames by camera name) and ``close()``. Frames are captured at t=0 and
    at every due sampling instant strictly before the run end; display pacing
    never drops video frames. Closing releases the rig before the encoders.
    """

    name = "video"

    def __init__(self, recording_config: dict, reference_points, start_position_w, rig_factory, *,
                 before_capture=None, log=print):
        self.config = recording_config
        self.reference_points = reference_points
        self.start_position_w = start_position_w
        self.rig_factory = rig_factory
        self.before_capture = before_capture
        self.log = log
        self.video = self.rig = self.views = None

    def start(self, run):
        if not run.driver.is_running():
            return
        from isaac_drone.viz.camera_views import RecordingViews
        from isaac_drone.viz.video import MultiViewRecorder

        config = self.config
        self.video = MultiViewRecorder(run.run_dir, config)
        self.views = RecordingViews(config["cameras"], self.reference_points, config["width"], config["height"])
        self.rig = self.rig_factory(config["width"], config["height"], self.views.poses(self.start_position_w))
        self.video.capture(run.loop.time_s, self.rig.read(self.views.poses(self.start_position_w)),
                           end_time_s=run.end_time_s)
        self.log(f"Video recording: {', '.join(config['cameras'])} + combined at {config['fps']} FPS -> {run.run_dir}")

    def after_step(self, run, record):
        if self.video is None or not self.video.due(run.loop.time_s, end_time_s=run.end_time_s):
            return
        if self.before_capture is not None:
            self.before_capture()
        poses = self.views.poses(record["post_step_state"]["position_w"])
        self.video.capture(run.loop.time_s, self.rig.read(poses), end_time_s=run.end_time_s)

    def summary(self):
        return None if self.video is None else self.video.summary()

    def close(self):
        try:
            if self.rig is not None:
                self.rig.close()
        finally:
            if self.video is not None:
                self.video.close()


class DisplayHook(RunHook):
    """Draw every ``render_interval`` physics steps, optionally paced to the wall clock.

    With a pacer, a frame waits while the simulation is ahead of the playback
    schedule, or is skipped so physics can catch up; physics is never altered.
    """

    name = "playback"

    def __init__(self, draw, render_interval: int, pacer=None):
        self.draw, self.render_interval, self.pacer = draw, render_interval, pacer

    def start(self, run):
        if self.pacer is not None:
            self.pacer.start(run.loop.time_s)

    def after_step(self, run, record):
        if run.loop.step_index % self.render_interval:
            return
        if self.pacer is not None:
            self.pacer.frame(run.loop.time_s, self.draw)
        else:
            self.draw()

    def paused(self, run):
        if self.pacer is not None:
            self.pacer.resync(run.loop.time_s)

    def summary(self):
        return None if self.pacer is None else self.pacer.summary()


class FiguresHook(RunHook):
    """Write the standard PNG figures after the run; failures only print a retry hint."""

    def __init__(self, log=print):
        self.log = log

    def finalize(self, result):
        try:
            from isaac_drone.analysis.plots import plot_run
            figures = plot_run(result.run_dir)
            self.log(f"Plots: {figures[0].parent} ({len(figures)} PNG)")
        except Exception as error:  # figures are a convenience and never change the run result
            self.log(f"Plots not written ({error}); retry: python -m isaac_drone plot {result.run_dir}")
