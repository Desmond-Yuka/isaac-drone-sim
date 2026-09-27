"""Physics-rate orchestration of trajectory, control, motor commands and effects.

The simulator owns rigid-body integration; the backend advances motor dynamics.
This module never teleports the vehicle to a commanded position. One prepare/finish
pair brackets one external sim.step() followed by robot.update(dt).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Protocol
import numpy as np

from .config import assert_flight_ready
from .control import BoundedAllocator, GeometricController
from .control.allocation import AllocationResult
from .mission import SpiralMissionMonitor
from .trajectories.spiral import _progress_derivatives
from .disturbances import build_effects
from .power import build_power
from .measurements import build_basic_record
from .trajectories import build_trajectory
from .types import VehicleState, Wrench, finite_array


class StateProvider(Protocol):
    """Future sensor/estimator boundary; must use the shared CoM/frame contract."""
    def reset(self) -> None: ...
    def read_state(self, time_s: float) -> VehicleState: ...


class MotionControlLoop:
    def __init__(self, config, backend, *, trajectory=None, effects=None,
                 power=None, state_provider=None):
        self.config = deepcopy(config)
        self.backend = backend
        self.state_provider = state_provider or backend
        self.trajectory = trajectory or build_trajectory(config["trajectory"])
        self.effects = effects if effects is not None else build_effects(config["effects"])
        self.power = power if power is not None else build_power(config["power"])
        if bool(config["power"]["enabled"]) != (self.power is not None):
            raise ValueError("power.enabled must agree with the supplied calibrated power system")
        self.dt_s = config["simulation"]["dt"]
        self.decimation = config["simulation"]["control_decimation"]
        self.is_helix = config["trajectory"]["kind"] in ("helix", "spiral")
        self.mission = SpiralMissionMonitor(config["trajectory"]["completion"]) if self.is_helix else None
        self.feasibility = None
        self._pending = False
        self._initialized = False

    def reset(self, *, initial_current_a=None, temperature_c=None):
        self._initialized, self._pending = False, False
        launch = self.config["vehicle"]["launch"]
        if launch["from_ground"]:
            start = self.backend.ground_start_position(launch["ground_z_m"], launch["clearance_m"])
            self.backend.reset(initial_position_w=start)
        else:
            self.backend.reset()
        if self.state_provider is not self.backend:
            self.state_provider.reset()
        matrix = np.asarray(self.backend.allocation_matrix_b, dtype=float)
        assert_flight_ready(self.config, matrix)
        if hasattr(self.backend, "assert_control_compatible"):
            self.backend.assert_control_compatible()
        self.controller = GeometricController(self.config["control"], self.backend.mass_properties(),
                                              np.array(self.config["simulation"]["gravity"]))
        self.allocator = BoundedAllocator(matrix, np.array(self.config["allocation"]["weights"]),
                                          self.config["allocation"]["regularization"])
        self.controller.reset()
        self.step_index = 0
        self._reference_sample_time_s = None
        self._pre_step_truth = None
        self._interval_reference = None
        self._previous_thrust = None
        self._requested_wrench = Wrench.zero()
        self.effects.reset(self.config["simulation"]["seed"])
        self.trajectory.reset(self.backend.read_state(0.0) if self.is_helix else self.state_provider.read_state(0.0))
        if self.mission is not None:
            from .trajectories.feasibility import validate_helix_feasibility
            self.mission.reset()
            parameters = self.backend.motor_parameters()
            self.feasibility = validate_helix_feasibility(
                self.trajectory, self.config["control"], self.backend.mass_properties(),
                np.array(self.config["simulation"]["gravity"]), matrix,
                parameters["thrust_min_n"], parameters["thrust_max_n"],
            )
        if self.power is not None:
            self.power.reset()
            if initial_current_a is None:
                raise ValueError("Power coupling needs an explicit initial current measurement/model operating point")
            self.power.initialize(initial_current_a, temperature_c=temperature_c)
        self._pending, self._initialized = False, True

    @property
    def time_s(self):
        return self.step_index * self.dt_s

    def prepare_step(self):
        if not self._initialized or self._pending:
            raise RuntimeError("reset first, and finish each prepared step exactly once")
        try:
            return self._prepare_step()
        except BaseException:
            # A plugin/controller may already have advanced internal state.
            self._initialized, self._pending = False, False
            raise

    def _prepare_step(self):
        """Evaluate commands/effects at t[k], then apply one native motor step."""
        if not self._initialized or self._pending:
            raise RuntimeError("reset first, and finish each prepared physics step exactly once")
        truth = self.backend.read_state(self.time_s)
        self._pre_step_truth = deepcopy(truth)
        state = truth if self.state_provider is self.backend else self.state_provider.read_state(self.time_s)
        if self.step_index % self.decimation == 0:
            self.setpoint = self.trajectory.sample(self.time_s)
            self._reference_sample_time_s = self.time_s
            self._requested_wrench = self.controller.compute(state, self.setpoint, self.dt_s*self.decimation)
        self._interval_reference = deepcopy(self.setpoint)
        motor_parameters = self.backend.motor_parameters()
        lower = finite_array(motor_parameters["thrust_min_n"], (4,), "motor minimum thrust")
        upper = finite_array(motor_parameters["thrust_max_n"], (4,), "motor maximum thrust")
        if self.power is not None:
            bounds = self.power.thrust_bounds_n(truth, self.backend.telemetry())
            lower = np.maximum(lower, finite_array(bounds.lower_n, (4,), "power lower thrust bounds"))
            upper = np.minimum(upper, finite_array(bounds.upper_n, (4,), "power upper thrust bounds"))
            if np.any(lower > upper):
                raise RuntimeError("Battery command envelope is incompatible with native motor thrust limits; an explicit shutdown actuator model is required")
        allocation = self.allocator.allocate(self._requested_wrench, lower, upper, self._previous_thrust)
        target_rps = self.backend.thrust_to_motor_speeds(allocation.thrusts_n)
        phase = self.trajectory.phase(self.time_s) if self.is_helix else "hold"
        startup_fraction = 1.0
        if self.is_helix and phase == "delay":
            delay = self.config["trajectory"]["spiral"]["start_delay_s"]
            startup_fraction = float(_progress_derivatives(self.time_s/delay, delay)[0])
            target_rps = target_rps * startup_fraction
        # Preserve the native minimum for an enabled motor; explicit zero is
        # stopped. Dynamics still make the actual rotor spin-up continuous.
        target_rps = np.where(target_rps == 0, 0.0, np.clip(
            target_rps, motor_parameters["rps_min"], motor_parameters["rps_max"]))
        effective_thrust = finite_array(motor_parameters["sampled_kf"], (4,), "sampled kf") * target_rps**2
        effective_wrench = Wrench.from_vector(self.allocator.matrix @ effective_thrust)
        self.controller.notify_allocation(AllocationResult(
            effective_thrust, effective_wrench, self._requested_wrench.vector-effective_wrench.vector,
            allocation.saturated or startup_fraction < 1.0))
        external = self.effects.evaluate(truth, self.dt_s)
        self.backend.apply_motor_speeds(target_rps, external)
        self._previous_thrust = effective_thrust.copy()
        self._pending = True
        self._record = {"step": self.step_index, "time_s": self.time_s,
                        "state_source": "simulation_truth" if self.state_provider is self.backend else "injected_state_provider",
                        "state": asdict(state), "setpoint": asdict(self.setpoint),
                        "mission_phase": phase, "startup_fraction": startup_fraction,
                        "motor_command_rps": target_rps,
                        "motor_command_thrust_n": effective_thrust,
                        "controller_force_derivative_source": self.controller.force_derivative_source,
                        "requested_wrench_b": self._requested_wrench.vector,
                        "allocated_thrust_n": allocation.thrusts_n,
                        "allocated_wrench_b": allocation.achieved_wrench.vector,
                        "allocation_residual": allocation.residual,
                        "allocation_saturated": allocation.saturated,
                        "external_wrench_b": external.vector}
        return deepcopy(self._record)

    def finish_step(self, *, measured_current_a=None, temperature_c=None):
        if not self._initialized or not self._pending:
            raise RuntimeError("finish_step requires a prepared step; reset after any plugin failure")
        try:
            return self._finish_step(measured_current_a=measured_current_a, temperature_c=temperature_c)
        except BaseException:
            # Physics has already advanced. Retrying this interval would double
            # count time/energy, so an explicit full reset is required.
            self._initialized, self._pending = False, False
            raise

    def _finish_step(self, *, measured_current_a=None, temperature_c=None):
        """Call only after physics and backend state buffers advance by dt.

        Electrical updates use endpoint load samples held over this interval.
        Supply measured current and temperature, or an explicitly calibrated
        load plugin. The motor integrates RPS with native lag/rate limits;
        the power envelope constrains commands, not an unmodelled ESC circuit.
        """
        if not self._pending:
            raise RuntimeError("finish_step requires one preceding prepare_step")
        self.step_index += 1
        truth = self.backend.read_state(self.time_s)
        telemetry = self.backend.telemetry()
        self._record["post_step_state"] = asdict(truth)
        self._record["backend"] = telemetry
        if self.mission is not None:
            self._record["mission"] = self.mission.update(
                truth, self.trajectory.sample(truth.time_s), self.trajectory.phase(truth.time_s))
        else:
            self._record["mission"] = None
        elapsed = truth.time_s - self._pre_step_truth.time_s
        if not np.isclose(elapsed, self.dt_s, rtol=1e-9, atol=1e-12):
            raise ValueError("Actual state timestamps must span exactly one physics step")
        self._record["basic"] = build_basic_record(
            self._pre_step_truth, truth, self._interval_reference, self._reference_sample_time_s,
            self.backend.mass_properties(), telemetry,
            Wrench.from_vector(self._record["requested_wrench_b"]),
            Wrench.from_vector(self._record["allocated_wrench_b"]),
            Wrench.from_vector(self._record["external_wrench_b"]),
        )
        if self.power is not None:
            power_state = self.power.update(truth, telemetry, self.dt_s,
                                             measured_current_a=measured_current_a,
                                             temperature_c=temperature_c)
            self._record["battery"] = asdict(power_state)
        else:
            self._record["battery"] = None
        self._pending = False
        return deepcopy(self._record)
