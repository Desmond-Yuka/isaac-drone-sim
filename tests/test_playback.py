"""Wall-clock pacing with a fake clock: waits when ahead, skips frames when behind."""
import pytest

from isaac_drone.playback import RealTimePacer


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def pacer(speed=1.0, draw_cost=0.0, **kwargs):
    clock = FakeClock()
    result = RealTimePacer(speed, clock=clock, sleep=clock.sleep, **kwargs)
    result.start(0.0)

    def draw():
        clock.now += draw_cost
    return result, clock, draw


def test_waits_for_wall_clock_when_simulation_is_ahead():
    result, clock, draw = pacer()
    clock.now += 0.004
    assert result.frame(0.02, draw)
    assert clock.sleeps == [pytest.approx(0.016)]
    assert result.lag_s(0.02) == pytest.approx(0.0)


def test_playback_speed_scales_the_schedule():
    result, clock, draw = pacer(speed=2.0)
    assert result.frame(1.0, draw)
    assert clock.now-100.0 == pytest.approx(0.5)
    half, _, draw = pacer(speed=0.5)
    assert half.frame(1.0, draw)
    assert half.summary()["wall_time_s"] == pytest.approx(2.0)


def test_skips_frames_while_behind_then_draws_after_catching_up():
    result, clock, draw = pacer(draw_cost=0.1, lag_tolerance_s=0.05, max_frame_gap_s=1.0)
    assert result.frame(0.02, draw)            # sleeps 0.02, then drawing costs 0.1
    for sim_time in (0.04, 0.06, 0.08):        # physics costs 0.005 s per 0.02 s period
        clock.now += 0.005
        assert not result.frame(sim_time, draw)  # 0.085, 0.07, 0.055 s behind: skipped
    clock.now += 0.005
    assert result.frame(0.10, draw)            # 0.04 s behind is within tolerance
    summary = result.summary()
    assert summary["frames_drawn"] == 2 and summary["frames_skipped"] == 3
    assert summary["mean_draw_time_s"] == pytest.approx(0.1)


def test_draws_at_least_every_max_gap_even_when_physics_is_too_slow():
    result, clock, draw = pacer(lag_tolerance_s=0.01, max_frame_gap_s=0.25)
    drawn = []
    for index in range(1, 21):
        clock.now += 0.0625                    # physics alone costs 2.5x real time
        drawn.append(result.frame(index*0.025, draw))
    assert sum(drawn) == 5                     # one frame per 0.25 s of wall time
    assert result.summary()["achieved_speed"] == pytest.approx(0.4)


def test_resync_discards_paused_wall_time():
    result, clock, draw = pacer()
    assert result.frame(0.5, draw)
    clock.now += 30.0                          # paused
    result.resync(0.5)
    assert result.lag_s(0.5) == pytest.approx(0.0)
    assert result.frame(0.52, draw)
    assert result.summary()["achieved_speed"] == pytest.approx(1.0)


@pytest.mark.parametrize("speed", [0.0, -1.0, float("nan"), float("inf"), True])
def test_rejects_invalid_speed(speed):
    with pytest.raises(ValueError):
        RealTimePacer(speed)


def test_requires_start():
    with pytest.raises(RuntimeError):
        RealTimePacer().frame(0.0, lambda: None)
