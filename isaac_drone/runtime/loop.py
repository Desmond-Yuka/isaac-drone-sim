"""Physics-rate orchestration of trajectory, control, motor commands and effects.

The simulator owns rigid-body integration; the backend advances motor dynamics.
This module never teleports the vehicle to a commanded position. One prepare/finish
pair brackets one external sim.step() followed by robot.update(dt).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from isaac_drone.config import assert_flight_ready
from isaac_drone.control import BoundedAllocator, FlightLimits, VehicleModel, build_controller
from isaac_drone.control.allocation import AllocationResult
from isaac_drone.core.types import Wrench
from isaac_drone.core.validation import finite_array
from isaac_drone.effects import build_effects
from isaac_drone.power import build_power
from isaac_drone.telemetry.measurements import build_basic_record
from isaac_drone.trajectories import SegmentedTrajectory, build_trajectory
from isaac_drone.trajectories.completion import CompletionMonitor
from isaac_drone.trajectories.feasibility import validate_trajectory_feasibility
from isaac_drone.trajectories.min_time import plan_minimum_time
from isaac_drone.trajectories.timing import _progress_derivatives

SPIN_UP_PHASE = "spin_up"


class MotionControlLoop:
    """Trajectory -> controller -> allocation -> motor commands -> backend, once per physics step.

    Components come from the config (``trajectory``, ``controller``, ``limits``,
    ``completion``, ``effects``, ``power``) unless injected. The controller is
    built at reset from the backend's actual mass properties and geometry; an
    injected ``controller`` instance is reset instead. During the launch spin-up
    (``vehicle.launch.spin_up_s``) the reference holds the start pose and motor
    commands are scaled by a smooth 0..1 envelope; the trajectory starts after it.
    """

    def __init__(
        self, config, backend, *, trajectory=None, controller=None, effects=None, power=None, state_provider=None
    ):
        self.config = deepcopy(config)
        self.backend = backend
        self.state_provider = state_provider or backend
        self.trajectory = trajectory or build_trajectory(config["trajectory"])
        self.limits = FlightLimits.from_config(config["limits"])
        self._injected_controller = controller
        self.controller = None
        self.effects = effects if effects is not None else build_effects(config["effects"])
        self.power = power if power is not None else build_power(config["power"])
        if bool(config["power"]["enabled"]) != (self.power is not None):
            raise ValueError("power.enabled must agree with the supplied calibrated power system")
        self.dt_s = config["simulation"]["dt"]
        self.decimation = config["simulation"]["control_decimation"]
        self.spin_up_s = float(config["vehicle"]["launch"]["spin_up_s"])
        completion = config["completion"]
        self.mission = None if completion is None else CompletionMonitor(completion)
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
        parameters = self.backend.motor_parameters()
        gravity = np.array(self.config["simulation"]["gravity"], dtype=float)
        self.vehicle = VehicleModel(
            self.backend.mass_properties(),
            gravity,
            matrix,
            finite_array(parameters["thrust_min_n"], (4,), "motor minimum thrust"),
            finite_array(parameters["thrust_max_n"], (4,), "motor maximum thrust"),
            self.dt_s * self.decimation,
        )
        if self._injected_controller is not None:
            self.controller = self._injected_controller
            self.controller_kind = f"injected:{type(self.controller).__name__}"
        else:
            self.controller = build_controller(self.config["controller"], vehicle=self.vehicle, limits=self.limits)
            self.controller_kind = self.config["controller"]["kind"]
        self.allocator = BoundedAllocator(
            matrix, np.array(self.config["allocation"]["weights"]), self.config["allocation"]["regularization"]
        )
        self.controller.reset()
        self.step_index = 0
        self._reference_sample_time_s = None
        self._pre_step_truth = None
        self._interval_reference = None
        self._interval_diagnostics = None
        self._previous_thrust = None
        self._requested_wrench = Wrench.zero()
        self._requested_thrust = None
        self.effects.reset(self.config["simulation"]["seed"])
        # A ground launch anchors the reference at the physical placement (backend truth); an
        # airborne start anchors it at the state the controller observes (possibly an estimate).
        anchor = self.backend if launch["from_ground"] else self.state_provider
        initial = anchor.read_state(0.0)
        self.trajectory.reset(initial, start_time_s=initial.time_s + self.spin_up_s)
        self._plan_and_check()
        if self.mission is not None:
            self.mission.reset()
        if self.power is not None:
            self.power.reset()
            if initial_current_a is None:
                raise ValueError("Power coupling needs an explicit initial current measurement/model operating point")
            self.power.initialize(initial_current_a, temperature_c=temperature_c)
        self._pending, self._initialized = False, True

    @property
    def _segmented(self) -> bool:
        return isinstance(self.trajectory, SegmentedTrajectory)

    def _plan_and_check(self):
        """Plan null-duration segments for minimum time, then screen the nominal reference."""
        if not self._segmented:
            self.feasibility = None
            return
        vehicle = (
            self.limits,
            self.vehicle.mass_properties,
            self.vehicle.gravity_w,
            self.vehicle.allocation_matrix_b,
            self.vehicle.thrust_min_n,
            self.vehicle.thrust_max_n,
        )
        plan = None
        if self.trajectory.planned_phases:
            # Motor lag uses the slowest time constant of the sampling range (+dt for the discrete mixing).
            thrusters = self.config["vehicle"]["thrusters"]
            lag = self.dt_s if thrusters["use_discrete_approximation"] else 0.0
            plan = plan_minimum_time(
                self.trajectory,
                *vehicle,
                motor_time_constants_s=(max(thrusters["tau_inc_range"]) + lag, max(thrusters["tau_dec_range"]) + lag),
            )
        self.feasibility = {**validate_trajectory_feasibility(self.trajectory, *vehicle), "minimum_time_plan": plan}
        if self.mission is not None:
            end = self.trajectory.phase_boundaries_s[-1][0]
            required = end + self.mission.params.dwell_time_s + 2 * self.dt_s
            if self.config["simulation"]["duration_s"] < required:
                raise ValueError(
                    f"simulation.duration_s must allow the complete trajectory and measured hover dwell: "
                    f"motion ends at {end:.3f} s, so duration_s >= {required:.3f} s is needed"
                )

    def phase(self, time_s: float) -> str:
        """Mission phase at an absolute time: ``spin_up`` before the trajectory starts, then its phases."""
        return SPIN_UP_PHASE if time_s < self.trajectory.start_time_s else self.trajectory.phase(time_s)

    def reference(self, time_s: float):
        """Trajectory reference; the start pose is held during spin-up."""
        return self.trajectory.sample(max(time_s, self.trajectory.start_time_s))

    def plan_summary(self) -> str | None:
        """One console line describing the planned motion, or None for a pure hold."""
        if not self._segmented or not self.trajectory.segments:
            return None
        from isaac_drone.trajectories.min_time import timing_summary

        return timing_summary(self.feasibility)

    def run_metadata(self) -> dict:
        """Backend-independent metadata for the run directory."""
        return {
            "controller_kind": self.controller_kind,
            "trajectory_feasibility": self.feasibility,
            "mission_phases": self.phase_schedule(),
        }

    def phase_schedule(self) -> list[tuple[float, str]]:
        """(absolute start [s], phase) of every mission phase, for metadata and figures."""
        start = self.trajectory.start_time_s
        schedule = [(0.0, SPIN_UP_PHASE)] if start > 0 else []
        if self._segmented:
            return schedule + self.trajectory.phase_boundaries_s
        return schedule + [(start, self.trajectory.phase(start))]

    def reference_path_points(self) -> np.ndarray:
        """Planned reference CoM path (for overlays and camera framing)."""
        if self._segmented:
            return self.trajectory.path_points()
        return self.trajectory.sample(self.trajectory.start_time_s).position_w[None, :]

    @property
    def time_s(self):
        return self.step_index * self.dt_s

    @property
    def mission_status(self):
        """Latest completion-monitor status, or None when the trajectory has no criterion."""
        return None if self.mission is None else self.mission.status

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
            self.setpoint = self.reference(self.time_s)
            self._reference_sample_time_s = self.time_s
            command = self.controller.compute(state, self.setpoint, self.dt_s * self.decimation)
            if self.controller.output == "wrench":
                self._requested_wrench = command
            else:
                self._requested_thrust = finite_array(command, (4,), "controller rotor thrust command")
                self._requested_wrench = Wrench.from_vector(self.allocator.matrix @ self._requested_thrust)
        self._interval_reference = deepcopy(self.setpoint)
        # Held with the setpoint: the controller only changes them in compute().
        self._interval_diagnostics = self.controller.diagnostics()
        motor_parameters = self.backend.motor_parameters()
        lower = finite_array(motor_parameters["thrust_min_n"], (4,), "motor minimum thrust")
        upper = finite_array(motor_parameters["thrust_max_n"], (4,), "motor maximum thrust")
        if self.power is not None:
            bounds = self.power.thrust_bounds_n(truth, self.backend.telemetry())
            lower = np.maximum(lower, finite_array(bounds.lower_n, (4,), "power lower thrust bounds"))
            upper = np.minimum(upper, finite_array(bounds.upper_n, (4,), "power upper thrust bounds"))
            if np.any(lower > upper):
                raise RuntimeError(
                    "Battery command envelope is incompatible with native motor thrust limits; "
                    "an explicit shutdown actuator model is required"
                )
        if self.controller.output == "wrench":
            allocation = self.allocator.allocate(self._requested_wrench, lower, upper, self._previous_thrust)
        else:
            thrusts = np.clip(self._requested_thrust, lower, upper)
            achieved = Wrench.from_vector(self.allocator.matrix @ thrusts)
            allocation = AllocationResult(
                thrusts,
                achieved,
                self._requested_wrench.vector - achieved.vector,
                bool(np.any(thrusts <= lower) or np.any(thrusts >= upper)),
            )
        target_rps = self.backend.thrust_to_motor_speeds(allocation.thrusts_n)
        phase = self.phase(self.time_s)
        startup_fraction = 1.0
        if phase == SPIN_UP_PHASE:
            startup_fraction = float(_progress_derivatives(self.time_s / self.spin_up_s, self.spin_up_s)[0])
            target_rps = target_rps * startup_fraction
        # Preserve the native minimum for an enabled motor; explicit zero is
        # stopped. Dynamics still make the actual rotor spin-up continuous.
        target_rps = np.where(
            target_rps == 0, 0.0, np.clip(target_rps, motor_parameters["rps_min"], motor_parameters["rps_max"])
        )
        effective_thrust = finite_array(motor_parameters["sampled_kf"], (4,), "sampled kf") * target_rps**2
        effective_wrench = Wrench.from_vector(self.allocator.matrix @ effective_thrust)
        self.controller.notify_allocation(
            AllocationResult(
                effective_thrust,
                effective_wrench,
                self._requested_wrench.vector - effective_wrench.vector,
                allocation.saturated or startup_fraction < 1.0,
            )
        )
        external = self.effects.evaluate(truth, self.dt_s)
        self.backend.apply_motor_speeds(target_rps, external)
        self._previous_thrust = effective_thrust.copy()
        self._pending = True
        self._record = {
            "step": self.step_index,
            "time_s": self.time_s,
            "state_source": "simulation_truth" if self.state_provider is self.backend else "injected_state_provider",
            "state": asdict(state),
            "setpoint": asdict(self.setpoint),
            "mission_phase": phase,
            "startup_fraction": startup_fraction,
            "motor_command_rps": target_rps,
            "motor_command_thrust_n": effective_thrust,
            "controller": {"kind": self.controller_kind, **self._interval_diagnostics.details},
            "requested_wrench_b": self._requested_wrench.vector,
            "allocated_thrust_n": allocation.thrusts_n,
            "allocated_wrench_b": allocation.achieved_wrench.vector,
            "allocation_residual": allocation.residual,
            "allocation_saturated": allocation.saturated,
            "external_wrench_b": external.vector,
        }
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
            self._record["mission"] = self.mission.update(truth, self.reference(truth.time_s), self.phase(truth.time_s))
        else:
            self._record["mission"] = None
        elapsed = truth.time_s - self._pre_step_truth.time_s
        if not np.isclose(elapsed, self.dt_s, rtol=1e-9, atol=1e-12):
            raise ValueError("Actual state timestamps must span exactly one physics step")
        self._record["basic"] = build_basic_record(
            self._pre_step_truth,
            truth,
            self._interval_reference,
            self._reference_sample_time_s,
            self.backend.mass_properties(),
            telemetry,
            Wrench.from_vector(self._record["requested_wrench_b"]),
            Wrench.from_vector(self._record["allocated_wrench_b"]),
            Wrench.from_vector(self._record["external_wrench_b"]),
            desired_rotation_w=self._interval_diagnostics.desired_rotation_w,
            desired_angular_velocity_d=self._interval_diagnostics.desired_angular_velocity_d,
            motor_command_rps=self._record["motor_command_rps"],
        )
        if self.power is not None:
            power_state = self.power.update(
                truth, telemetry, self.dt_s, measured_current_a=measured_current_a, temperature_c=temperature_c
            )
            self._record["battery"] = asdict(power_state)
        else:
            self._record["battery"] = None
        self._pending = False
        return deepcopy(self._record)
