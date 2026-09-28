"""Minimum-time timing of the takeoff and helix phases from actual vehicle limits.

Each phase whose configured duration is null gets an accelerate/cruise/
decelerate ``CruiseTimeLaw`` over its unchanged geometric path. The planner
searches the cruise rate and the shortest acceleration/deceleration ramps
such that, at every sample, the nominal ideal-tracking rotor thrusts stay in
the central ``thrust_utilization`` band of each rotor's actual thrust range::

    reserve = (1 - utilization) * (upper - lower)
    lower + reserve <= f_i <= upper - reserve

The reserve on both sides is left to feedback: no thrust would remain for
correcting errors if the plan itself used the full range. The configured
max_tilt, max_acceleration and max_yaw_rate limits are respected too.

First-order motor lag is included: for the rotor to actually produce f(t),
its command must lead by about tau * df/dt, so ``f + tau * df/dt`` (tau_up
while rising, tau_down while falling; df/dt from consecutive samples) must
also stay in the band. Without this, short ramps demand thrust changes the
motors cannot follow and the vehicle falls behind the reference.

Like ``validate_helix_feasibility`` this is a sampled rigid-body feedforward
screen, not a proof: disturbances and tracking errors are not included, and a
sample grid can miss peaks between samples.
"""
from __future__ import annotations

import math

import numpy as np

from isaac_drone.core.types import MassProperties
from isaac_drone.core.validation import finite_array, finite_scalar
from isaac_drone.trajectories.helix import SpiralTrajectory

from .feasibility import nominal_requirement
from .timing import CruiseTimeLaw

_RAMP_SAMPLES = 33
_RAMP_TOLERANCE = 2e-3   # relative, on ramp durations and the steady rate limit
_RATE_TOLERANCE = 5e-3   # relative, on the cruise rate; total time is flat near its optimum
_LONGEST_RAMP_S = 1e3


class _PhaseLimits:
    """Sampled nominal-feasibility test of progress states on one phase path."""

    def __init__(self, setpoint, mass, inertia, gravity, allocation_inverse, band_lower, band_upper,
                 max_tilt, max_acceleration, motor_time_constants_s):
        self.setpoint = setpoint
        self.mass, self.inertia, self.gravity = mass, inertia, gravity
        self.allocation_inverse = allocation_inverse
        self.band_lower, self.band_upper = band_lower, band_upper
        self.max_tilt, self.max_acceleration = max_tilt, max_acceleration
        self.tau_up, self.tau_down = motor_time_constants_s

    def _outside(self, rotors) -> bool:
        return bool(np.any(rotors < self.band_lower) or np.any(rotors > self.band_upper))

    def feasible(self, samples) -> bool:
        """``samples`` yields (time [s], progress) in time order, either direction."""
        previous = None
        for time, progress in samples:
            reference = self.setpoint(progress)
            if (self.max_acceleration is not None
                    and np.linalg.norm(reference.acceleration_w) > self.max_acceleration):
                return False
            thrust, torque, tilt, _ = nominal_requirement(reference, self.mass, self.inertia, self.gravity)
            if tilt > self.max_tilt:
                return False
            rotors = self.allocation_inverse@np.r_[thrust, torque]
            if self._outside(rotors):
                return False
            if previous is not None and time != previous[0]:
                slope = (rotors-previous[1])/(time-previous[0])
                lead = 0.5*(rotors+previous[1])+np.where(slope > 0, self.tau_up, self.tau_down)*slope
                if self._outside(lead):
                    return False
            previous = time, rotors
        return True


def _bisect(ok, feasible: float, infeasible: float) -> float:
    """Return a feasible value within the relative tolerance of the boundary."""
    while abs(feasible-infeasible) > _RAMP_TOLERANCE*min(abs(feasible), abs(infeasible)):
        middle = 0.5*(feasible+infeasible)
        if ok(middle):
            feasible = middle
        else:
            infeasible = middle
    return feasible


def _shortest_ramp(limits: _PhaseLimits, rate: float, accelerate: bool, guess: float = 0.25) -> float:
    """Shortest ramp between rest and ``rate`` whose samples all pass; inf if none."""
    def ok(duration):
        # Fastest (most demanding) end first, so an infeasible ramp exits early.
        times = np.linspace(duration, 0.0, _RAMP_SAMPLES)
        if accelerate:
            samples = ((t, CruiseTimeLaw.accelerate(rate, duration, t)) for t in times)
        else:  # t is the time remaining to the end of the ramp
            samples = ((duration-t, CruiseTimeLaw.decelerate(rate, duration, t)) for t in times)
        return limits.feasible(samples)

    duration, step = guess, 1.25
    if ok(duration):
        while duration > 1e-4 and ok(duration/step):
            duration /= step
        return _bisect(ok, duration, duration/step) if duration > 1e-4 else duration
    while not ok(step*duration):
        duration *= step
        if duration > _LONGEST_RAMP_S:
            return math.inf
    return _bisect(ok, step*duration, duration)


def _plan_phase(limits: _PhaseLimits, path_length: float, steady_samples: int, rate_cap: float) -> tuple:
    """Minimize T(r) = T_a + T_c + T_d over the cruise progress rate r [1/s]."""
    sigmas = np.linspace(0.0, 1.0, steady_samples)

    def steady(rate):
        return limits.feasible((sigma/rate, np.array([sigma, rate, 0., 0., 0.])) for sigma in sigmas)

    # Largest constant rate the path allows (e.g. helix centripetal thrust/tilt).
    steady_limit = math.inf
    rate = 1.0/max(path_length, 1e-3)  # 1 m/s along the path
    if not steady(rate):
        raise ValueError("Even 1 m/s steady motion exceeds the thrust band; lower the speed or raise thrust_utilization")
    while rate < rate_cap and path_length*rate < 1e3:
        if not steady(2.0*rate):
            steady_limit = _bisect(steady, rate, 2.0*rate)
            break
        rate *= 2.0
    upper_rate = min(rate_cap, steady_limit)
    cache = {}

    def guess(rate, index):
        # Warm start from the nearest evaluated rate; ramps scale roughly with the rate.
        known = [(abs(math.log(other/rate)), other, item[index]) for other, item in cache.items()
                 if math.isfinite(item[index])]
        return max(1e-3, min(known)[2]*rate/min(known)[1]) if known else 0.25

    def total(rate):
        if rate not in cache:
            accelerate = _shortest_ramp(limits, rate, True, guess(rate, 1))
            decelerate = (_shortest_ramp(limits, rate, False, guess(rate, 2))
                          if math.isfinite(accelerate) else math.inf)
            cruise = CruiseTimeLaw.cruise_duration(rate, accelerate, decelerate)
            cache[rate] = (math.inf if not math.isfinite(cruise) or cruise < 0
                           else accelerate+cruise+decelerate, accelerate, decelerate)
        return cache[rate][0]

    if upper_rate == steady_limit:
        upper_rate *= 1.0-_RAMP_TOLERANCE  # the exact steady limit needs infinite ramps
    # The ramps alone also bound the rate (e.g. vertical takeoff has no steady limit):
    # beyond it no cruise time is left.
    rate = 1.0/max(path_length, 1e-3)
    while rate < upper_rate and math.isfinite(total(rate)):
        rate *= 2.0
    upper_rate = min(upper_rate, rate)
    # Golden-section search; infeasible rates (inf) lie above the feasible interval.
    low, high = 0.02*upper_rate, upper_rate
    ratio = (math.sqrt(5.0)-1.0)/2.0
    left, right = high-ratio*(high-low), low+ratio*(high-low)
    while high-low > _RATE_TOLERANCE*high:
        if total(left) <= total(right):
            high, right = right, left
            left = high-ratio*(high-low)
        else:
            low, left = left, right
            right = low+ratio*(high-low)
    candidates = [rate for rate in cache if math.isfinite(cache[rate][0])]
    if not candidates:
        raise ValueError("No accelerate/cruise/decelerate timing keeps the nominal thrusts inside the band")
    best = min(candidates, key=lambda rate: cache[rate][0])
    _, accelerate, decelerate = cache[best]
    law = CruiseTimeLaw(best, accelerate, decelerate)
    return law, {**law.describe(), "path_length_m": path_length, "peak_speed_m_s": path_length*best,
                 "steady_rate_limit_per_s": steady_limit if math.isfinite(steady_limit) else None,
                 "yaw_rate_cap_per_s": rate_cap if math.isfinite(rate_cap) else None,
                 "evaluated_rates": len(cache)}


def plan_minimum_time(
    trajectory: SpiralTrajectory,
    control_config: dict,
    mass_properties: MassProperties,
    gravity_w: np.ndarray,
    allocation_matrix_b: np.ndarray,
    lower_thrust_n: np.ndarray,
    upper_thrust_n: np.ndarray,
    motor_time_constants_s: tuple[float, float] = (0.0, 0.0),
) -> dict:
    """Assign minimum-time laws to the trajectory's null-duration phases.

    The trajectory must already be reset to the actual initial state. Inputs
    are the same actual vehicle quantities used by validate_helix_feasibility,
    plus the slowest (rising, falling) first-order motor time constants [s].
    """
    phases = trajectory.planned_phases
    if not phases:
        return {"planned_phases": []}
    utilization = float(trajectory.config["thrust_utilization"])
    gravity = finite_array(gravity_w, (3,), "gravity_w")
    matrix = finite_array(allocation_matrix_b, (6, 4), "allocation_matrix_b")
    lower = finite_array(lower_thrust_n, (4,), "lower_thrust_n")
    upper = finite_array(upper_thrust_n, (4,), "upper_thrust_n")
    if np.any(lower < 0) or np.any(upper < lower):
        raise ValueError("Minimum-time planning needs thrust bounds 0 <= lower <= upper")
    if not np.allclose(matrix[:3], np.tile([[0.], [0.], [1.]], (1, 4)), rtol=0, atol=1e-7):
        raise ValueError("Minimum-time planning requires every thrust axis along body +Z")
    collective_torque = matrix[[2, 3, 4, 5]]
    if np.linalg.matrix_rank(collective_torque) != 4:
        raise ValueError("Minimum-time planning needs four independent collective/attitude channels")
    # Four rotors and four channels: the nominal allocation is the unique exact solve.
    inverse = np.linalg.inv(collective_torque)
    mass = finite_scalar(mass_properties.mass_kg, "mass_kg")
    inertia = finite_array(mass_properties.inertia_com_b, (3, 3), "inertia_com_b")
    reserve = (1.0-utilization)*(upper-lower)
    band_lower, band_upper = lower+reserve, upper-reserve
    hover = inverse@np.r_[mass*np.linalg.norm(gravity), 0., 0., 0.]
    if np.any(hover < band_lower) or np.any(hover > band_upper):
        raise ValueError(f"Hover thrusts {hover.tolist()} N lie outside the planning band "
                         f"[{band_lower.tolist()}, {band_upper.tolist()}] N; raise thrust_utilization")
    maximum_tilt = finite_scalar(control_config["max_tilt_rad"], "max_tilt_rad")
    maximum_yaw_rate = finite_scalar(control_config["max_yaw_rate_rad_s"], "max_yaw_rate_rad_s")
    tau_up, tau_down = (finite_scalar(value, "motor time constant") for value in motor_time_constants_s)
    if tau_up < 0 or tau_down < 0:
        raise ValueError("Motor time constants must be nonnegative")
    maximum_acceleration = control_config["max_acceleration_m_s2"]
    if maximum_acceleration is not None:
        maximum_acceleration = finite_scalar(maximum_acceleration, "max_acceleration_m_s2")
    lengths = trajectory.path_lengths_m
    turns = abs(trajectory.config["turns"])
    yaw_per_progress = {
        "takeoff": abs(trajectory.takeoff_yaw_change_rad),
        "spiral": abs(trajectory.turn_angle_rad) if trajectory.config["yaw_mode"] == "tangent" else 0.0,
    }
    setpoints = {"takeoff": trajectory.takeoff_setpoint, "spiral": trajectory.spiral_setpoint}
    steady_samples = {"takeoff": 33, "spiral": max(33, math.ceil(16*turns)+1)}
    laws, report = {}, {}
    for phase in phases:
        limits = _PhaseLimits(setpoints[phase], mass, inertia, gravity, inverse, band_lower, band_upper,
                              maximum_tilt, maximum_acceleration, (tau_up, tau_down))
        rate_cap = maximum_yaw_rate/yaw_per_progress[phase] if yaw_per_progress[phase] > 0 else math.inf
        laws[phase], report[phase] = _plan_phase(limits, lengths[phase], steady_samples[phase], rate_cap)
    trajectory.set_time_laws(**laws)
    return {"planned_phases": list(phases), "thrust_utilization": utilization,
            "planning_band_n": [band_lower.tolist(), band_upper.tolist()], "hover_thrusts_n": hover.tolist(),
            "motor_time_constants_s": [tau_up, tau_down],
            "phases": report,
            "limitations": ["Sampled ideal-tracking feedforward with first-order motor lag only; feedback errors "
                            "and disturbances use the reserve outside the band, which is not proven sufficient."]}


def timing_summary(feasibility_report: dict) -> str:
    """One console line: phase durations, peak speeds and peak rotor utilization."""
    timing = feasibility_report["timing"]
    parts = [f"{phase} {item['duration_s']:.2f} s{' (min-time)' if item['planned'] else ''}, "
             f"peak {item['peak_speed_m_s']:.2f} m/s" for phase, item in timing["phases"].items()]
    thrust = max(feasibility_report["sampled_max_thrust_fraction_of_max"])
    speed = max(feasibility_report["sampled_max_motor_speed_fraction_of_max"])
    return (f"{'; '.join(parts)}; motion ends at {timing['mission_duration_s']:.2f} s; nominal peak rotor "
            f"thrust {100*thrust:.0f}% / speed {100*speed:.0f}% of max, tilt "
            f"{math.degrees(feasibility_report['sampled_max_required_tilt_rad']):.0f} deg")
