"""CPU rigid-body plant for fast controller/trajectory iteration without Isaac Sim.

This is NOT Isaac Sim, an ARL asset model, hardware calibration, or proof that
ARL will fly. It drives the real trajectory/controller/allocator/runtime stack
through ``NativeRpsActuator`` and the NumPy transcription of the native motor
integrators. The body is synthetic: full inertia, displaced CoM, unequal motor
coefficients/time constants and a frictionless unilateral CoM support. The
support is a minimal contact fixture, not landing gear. No target
position/orientation/velocity is ever written into the plant while flying.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from types import MethodType, SimpleNamespace

import numpy as np

from isaac_drone.core.rotations import quaternion_to_matrix
from isaac_drone.core.types import MassProperties, VehicleState, Wrench
from isaac_drone.sim.actuation import NativeRpsActuator, thrust_to_motor_speeds
from isaac_drone.sim.native_thruster import TRANSCRIBED_METHODS

ROTOR_NAMES = ("back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop")


@dataclass(frozen=True)
class SyntheticPlantParameters:
    """Deterministic test values, deliberately asymmetric; not ARL measurements."""

    mass_kg: float = 1.24
    inertia_com_b_kg_m2: tuple = ((.0080, .0007, -.0004), (.0007, .0090, .0003), (-.0004, .0003, .0140))
    com_b_m: tuple = (.006, -.004, .002)
    rotor_positions_b_m: tuple = ((-.1, .1, 0.), (-.1, -.1, 0.), (.1, .1, 0.), (.1, -.1, 0.))
    thrust_coefficients_n_per_rps2: tuple = (1.1e-5, 1.3e-5, 1.6e-5, 1.8e-5)
    tau_inc_s: tuple = (.050, .061, .073, .080)
    tau_dec_s: tuple = (.0050, .0060, .0045, .0055)
    thrust_range_n: tuple = (.1, 10.)
    torque_to_thrust_ratio_m: float = .07
    max_rps_rate: float = 100000.0
    com_support_height_m: float = .08  # CoM height above the ground when resting on the support


class TranscribedThruster:
    """A native-``Thruster``-shaped object whose integrators are the NumPy transcription."""

    def __init__(self, parameters: SyntheticPlantParameters, dt_s: float, thrusters_config: dict):
        self.cfg = SimpleNamespace(dt=dt_s, integration_scheme=thrusters_config["integration_scheme"],
                                   thrust_range=tuple(parameters.thrust_range_n))
        self.thrust_const = np.array([parameters.thrust_coefficients_n_per_rps2], dtype=float)
        self.tau_inc_s = np.array([parameters.tau_inc_s], dtype=float)
        self.tau_dec_s = np.array([parameters.tau_dec_s], dtype=float)
        self.curr_thrust = np.zeros((1, 4))
        self.max_rate = parameters.max_rps_rate
        for name, method in TRANSCRIBED_METHODS.items():
            setattr(self, name, MethodType(method, self))
        self.mixing_factor_function = (self.discrete_mixing_factor if thrusters_config["use_discrete_approximation"]
                                       else self.continuous_mixing_factor)


class SyntheticBackend:
    """Newton--Euler body whose only control input is the four motor RPS commands.

    Position/velocity are whole-body CoM world coordinates; the quaternion is
    body-to-world wxyz; angular rate and full inertia are body-frame. The floor
    applies only a vertical inelastic impulse at the CoM: no contact torque,
    friction, drag, ground effect or battery. These are limitations of this
    plant, not assumptions of the Isaac Lab backend.
    """

    names = list(ROTOR_NAMES)

    def __init__(self, config: dict, parameters: SyntheticPlantParameters | None = None):
        self.config = deepcopy(config)
        self.parameters = parameters or SyntheticPlantParameters()
        p = self.parameters
        self.dt = config["simulation"]["dt"]
        self.gravity = np.asarray(config["simulation"]["gravity"], dtype=float)
        self.mass = MassProperties(p.mass_kg, np.array(p.inertia_com_b_kg_m2), np.array(p.com_b_m))
        self.inv_inertia = np.linalg.inv(self.mass.inertia_com_b)
        origins = np.array(p.rotor_positions_b_m)
        axes = np.tile([0., 0., 1.], (4, 1))
        reaction = p.torque_to_thrust_ratio_m * np.asarray(config["vehicle"]["rotor_directions"], dtype=float)
        moments = np.cross(origins - self.mass.com_b, axes) + reaction[:, None] * axes
        self.allocation_matrix_b = np.vstack((axes.T, moments.T))
        self.native = TranscribedThruster(p, self.dt, config["vehicle"]["thrusters"])
        self.motor = NativeRpsActuator(self.native, np.zeros((1, 4)))
        self.contact_floor_z = p.com_support_height_m
        self.reset()

    def ground_start_position(self, ground_z_m, clearance_m):
        self.contact_floor_z = ground_z_m + self.parameters.com_support_height_m
        initial = self.config["vehicle"]["initial_state"]
        root = np.asarray(initial["pos"], dtype=float).copy()
        rotation = quaternion_to_matrix(np.asarray(initial["quaternion_wxyz"]))
        root[2] = self.contact_floor_z + clearance_m - (rotation @ self.mass.com_b)[2]
        return root

    def reset(self, initial_position_w=None):
        initial = self.config["vehicle"]["initial_state"]
        root = np.asarray(initial["pos"] if initial_position_w is None else initial_position_w, dtype=float)
        quaternion = np.asarray(initial["quaternion_wxyz"], dtype=float)
        rotation = quaternion_to_matrix(quaternion)
        self.position = root + rotation @ self.mass.com_b
        angular_velocity_w = np.asarray(initial["ang_vel"], dtype=float)
        self.velocity = (np.asarray(initial["lin_vel"], dtype=float)
                         + np.cross(angular_velocity_w, rotation @ self.mass.com_b))
        self.quaternion = quaternion.copy()
        self.omega = rotation.T @ angular_velocity_w
        self.motor.reset(np.array([[initial["rps"][name] for name in self.names]]))
        self.external = Wrench.zero()
        self.submitted = Wrench.zero()
        self.has_submitted = False
        self.contact_impulse_n_s = 0.
        self.motor_step_count = 0
        self.physics_step_count = 0

    def mass_properties(self):
        return self.mass

    def read_state(self, time_s):
        return VehicleState(time_s, self.position.copy(), self.quaternion.copy(),
                            self.velocity.copy(), self.omega.copy())

    def motor_parameters(self):
        kf, low, high = self.motor.parameters()
        lower, upper = self.parameters.thrust_range_n
        return {"sampled_kf": kf[0].copy(), "rps_min": low[0].copy(), "rps_max": high[0].copy(),
                "thrust_min_n": np.full(4, float(lower)), "thrust_max_n": np.full(4, float(upper)),
                "current_rps": self.motor.rps[0].copy(), "thruster_names": self.names.copy()}

    def thrust_to_motor_speeds(self, thrust):
        params = self.motor_parameters()
        return thrust_to_motor_speeds(thrust, params["sampled_kf"], params["thrust_min_n"], params["thrust_max_n"])

    def apply_motor_speeds(self, targets, external):
        if self.motor_step_count != self.physics_step_count:
            raise RuntimeError("Motor input already prepared; advance physics once")
        self.motor.step(np.asarray(targets)[None, :])
        self.external = external
        self.submitted = Wrench.from_vector(self.allocation_matrix_b @ self.native.curr_thrust[0] + external.vector)
        self.has_submitted = True
        self.motor_step_count += 1

    def _derivative(self, packed):
        q, omega = packed[6:10], packed[10:13]
        # Intermediate RK4 stages need normalized quaternions for R only.
        rotation = quaternion_to_matrix(q / np.linalg.norm(q))
        force = rotation @ self.submitted.force_b
        angular_acceleration = self.inv_inertia @ (
            self.submitted.torque_b - np.cross(omega, self.mass.inertia_com_b @ omega))
        qdot = .5 * np.r_[-q[1:] @ omega, q[0]*omega + np.cross(q[1:], omega)]
        return np.r_[packed[3:6], self.gravity + force/self.mass.mass_kg, qdot, angular_acceleration]

    def step(self):
        """Advance the rigid body by one physics step (RK4) under the submitted wrench."""
        if self.motor_step_count != self.physics_step_count + 1:
            raise RuntimeError("Prepare exactly one motor input before physics")
        y = np.r_[self.position, self.velocity, self.quaternion, self.omega]
        dt = self.dt
        k1 = self._derivative(y)
        k2 = self._derivative(y + .5*dt*k1)
        k3 = self._derivative(y + .5*dt*k2)
        k4 = self._derivative(y + dt*k3)
        y += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        if not np.isfinite(y).all():
            raise FloatingPointError("Synthetic plant diverged")
        self.position, self.velocity = y[:3].copy(), y[3:6].copy()
        self.quaternion, self.omega = y[6:10]/np.linalg.norm(y[6:10]), y[10:13].copy()
        self.contact_impulse_n_s = 0.
        if self.position[2] < self.contact_floor_z:
            self.position[2] = self.contact_floor_z
            if self.velocity[2] < 0:
                self.contact_impulse_n_s = -self.mass.mass_kg*self.velocity[2]
                self.velocity[2] = 0.
        self.physics_step_count += 1

    def telemetry(self):
        speed = self.motor.rps[0].copy()
        return {"motor_speed_rps": speed, "motor_speed_rpm": speed*60,
                "motor_speed_rad_s": speed*2*np.pi,
                "commanded_thrust_n": self.native.thrust_const[0]*self.motor.target_rps[0]**2,
                "applied_thrust_n": self.native.curr_thrust[0].copy(),
                "motor_wrench_b": self.allocation_matrix_b @ self.native.curr_thrust[0],
                "total_wrench_b": self.submitted.vector,
                "has_submitted_wrench": self.has_submitted,
                "motor_speed_source": "synthetic_plant_native_rps_adapter_state",
                "linear_acceleration_source": "unavailable_in_synthetic_plant",
                "contact_impulse_n_s": self.contact_impulse_n_s}

    def describe(self) -> dict:
        """Static plant description for run metadata."""
        return {"backend": "synthetic", "parameters": asdict(self.parameters),
                "allocation_matrix_b": self.allocation_matrix_b.tolist(),
                "mass_kg": self.mass.mass_kg, "com_b": self.mass.com_b.tolist(),
                "inertia_com_b": self.mass.inertia_com_b.tolist(),
                "limitations": ["synthetic mass/inertia/motor coefficients, not ARL data",
                                "frictionless CoM contact support",
                                "no CFD, ground effect, drag, battery or sensor noise",
                                "native motor integrators transcribed to NumPy"]}


class SyntheticDriver:
    """SimulationDriver for the synthetic plant: never paused, never closed externally."""

    def __init__(self, backend: SyntheticBackend):
        self.backend = backend

    def step(self) -> None:
        self.backend.step()

    def is_running(self) -> bool:
        return True

    def is_paused(self) -> bool:
        return False

    def idle(self) -> None:
        pass


def run_simulation(config: dict, *, figures: bool = True, log=print, log_root=None, hooks=()):
    """Fly the configured mission once on the synthetic plant and record a standard run directory."""
    from isaac_drone.runtime.hooks import FiguresHook, MetricsHook
    from isaac_drone.runtime.loop import MotionControlLoop
    from isaac_drone.runtime.runner import run_experiment

    config = deepcopy(config)
    if config["recording"]["enabled"]:
        raise ValueError("Video recording needs rendered cameras; use --backend isaaclab or disable recording")
    config["simulation"]["device"] = "cpu"
    backend = SyntheticBackend(config)
    loop = MotionControlLoop(config, backend)
    loop.reset()
    if loop.plan_summary():
        log(f"Trajectory plan: {loop.plan_summary()}")
    description = backend.describe()
    metadata = {"backend": description, "backend_kind": "synthetic", "state_source": "simulation_truth",
                "motor_model": "RPS commands through NativeRpsActuator with transcribed native integrators",
                **loop.run_metadata(), "limitations": description["limitations"]}
    hooks = [*hooks, MetricsHook(log), *([FiguresHook(log)] if figures else [])]
    return run_experiment(loop, SyntheticDriver(backend), config, metadata=metadata, hooks=hooks,
                          log_root=log_root, log=log)
