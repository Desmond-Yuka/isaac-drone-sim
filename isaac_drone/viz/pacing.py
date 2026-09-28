"""Wall-clock pacing so a displayed run plays back at a chosen speed.

Pacing only decides when to wait and whether to draw a frame. It never changes
the physics step or the control schedule, so a paced run records exactly the
same telemetry as an unpaced one.
"""

from __future__ import annotations

import math
import time


class RealTimePacer:
    """Keep simulated time aligned with wall-clock time scaled by ``speed``.

    At each frame opportunity: if simulation is ahead of the wall clock, sleep
    until they agree and draw; if simulation has fallen more than
    ``lag_tolerance_s`` behind (drawing is slower than the frame period), skip
    the frame so physics can catch up, but still draw at least every
    ``max_frame_gap_s`` so the view never freezes. ``speed=1`` is real time.
    When physics alone is slower than real time the lag keeps growing; the
    summary reports the achieved real-time factor instead of hiding it.
    """

    def __init__(
        self, speed=1.0, *, lag_tolerance_s=0.05, max_frame_gap_s=0.25, clock=time.perf_counter, sleep=time.sleep
    ):
        for name, value in (
            ("speed", speed),
            ("lag_tolerance_s", lag_tolerance_s),
            ("max_frame_gap_s", max_frame_gap_s),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        self.speed = float(speed)
        self.lag_tolerance_s = float(lag_tolerance_s)
        self.max_frame_gap_s = float(max_frame_gap_s)
        self._clock, self._sleep = clock, sleep
        self._wall0 = None

    def start(self, sim_time_s: float) -> None:
        self._wall0 = self._last_frame = self._clock()
        self._sim0 = float(sim_time_s)
        self._sim_last = self._sim0
        self.frames = self.skipped = 0
        self.draw_time_s = self.sleep_time_s = 0.0

    def resync(self, sim_time_s: float) -> None:
        """Treat sim_time_s as on schedule now, e.g. after a pause; counters are kept."""
        if self._wall0 is None:
            raise RuntimeError("start the pacer before resyncing")
        self._wall0 = self._clock() - (sim_time_s - self._sim0) / self.speed
        self._sim_last = float(sim_time_s)

    def lag_s(self, sim_time_s: float) -> float:
        """Wall-clock seconds the simulation is behind its playback schedule (negative = ahead)."""
        if self._wall0 is None:
            raise RuntimeError("start the pacer before pacing frames")
        return (self._clock() - self._wall0) - (sim_time_s - self._sim0) / self.speed

    def frame(self, sim_time_s: float, draw) -> bool:
        """Wait if ahead, then call ``draw()`` unless the frame is skipped to catch up."""
        lag = self.lag_s(sim_time_s)
        self._sim_last = float(sim_time_s)
        if lag < 0:
            self._sleep(-lag)
            self.sleep_time_s += -lag
        elif lag > self.lag_tolerance_s and self._clock() - self._last_frame < self.max_frame_gap_s:
            self.skipped += 1
            return False
        begin = self._clock()
        draw()
        self._last_frame = self._clock()
        self.draw_time_s += self._last_frame - begin
        self.frames += 1
        return True

    def summary(self) -> dict:
        if self._wall0 is None:
            raise RuntimeError("start the pacer before requesting a summary")
        wall = self._clock() - self._wall0
        simulated = self._sim_last - self._sim0
        return {
            "playback_speed": self.speed,
            "wall_time_s": wall,
            "simulated_time_s": simulated,
            "achieved_speed": simulated / wall if wall > 0 else None,
            "frames_drawn": self.frames,
            "frames_skipped": self.skipped,
            "mean_draw_time_s": self.draw_time_s / self.frames if self.frames else None,
            "sleep_time_s": self.sleep_time_s,
        }
