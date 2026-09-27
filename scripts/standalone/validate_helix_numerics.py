"""CPU regression of the flight stack against an explicitly synthetic plant.

Run: python3 scripts/standalone/validate_helix_numerics.py

This is NOT Isaac Sim, an ARL asset model, hardware calibration, or proof that
ARL will fly. It exercises the real trajectory/controller/allocator/runtime
and NativeRpsActuator with a NumPy transcription of native motor integrators.
The plant is a synthetic rigid body with full inertia, displaced CoM, unequal
motor coefficients/time constants and a frictionless unilateral CoM support.
The latter is deliberately a minimal test contact fixture, not landing gear.
No target position/orientation/velocity is written into the plant while flying.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isaac_drone.actuation import NativeRpsActuator, thrust_to_motor_speeds
from isaac_drone.config import load_config
from isaac_drone.control.math import quaternion_to_matrix
from isaac_drone.runtime import MotionControlLoop
from isaac_drone.types import MassProperties, VehicleState, Wrench


class NumpyNativeMotorFixture:
    """Native thruster.py's lag/rate/RK4 equations transcribed into NumPy.

    The production adapter still calls these through NativeRpsActuator. This
    fixture avoids importing torch/Warp/Isaac Sim and does not test their ABI.
    Coefficients below are deterministic test values, not ARL measurements.
    """

    def __init__(self, dt):
        self.cfg = SimpleNamespace(dt=dt, integration_scheme="rk4", thrust_range=(.1, 10.))
        self.thrust_const = np.array([[1.1e-5, 1.3e-5, 1.6e-5, 1.8e-5]])
        self.tau_inc_s = np.array([[.050, .061, .073, .080]])
        self.tau_dec_s = np.array([[.0050, .0060, .0045, .0055]])
        self.curr_thrust = np.zeros((1, 4))
        self.max_rate = 100000.0

    def mixing_factor_function(self, tau):
        return 1.0 / (self.cfg.dt + tau)

    def motor_model_rate(self, error, mixing):
        return np.clip(mixing * error, -self.max_rate, self.max_rate)

    def rk4_integration(self, error, mixing):
        dt = self.cfg.dt
        k1 = self.motor_model_rate(error, mixing)
        k2 = self.motor_model_rate(error - .5 * dt * k1, mixing)
        k3 = self.motor_model_rate(error - .5 * dt * k2, mixing)
        k4 = self.motor_model_rate(error - dt * k3, mixing)
        return dt * (k1 + 2*k2 + 2*k3 + k4) / 6


class SyntheticRigidBody:
    """Test-only Newton--Euler body; four motor RPS values are its only input.

    p and v are whole-body CoM world coordinates. q is body-to-world WXYZ.
    Angular rates and full J are in body coordinates. A floor applies only a
    vertical inelastic contact impulse at the CoM: it has no contact torque,
    friction, drag, ground effect or battery. These are explicit limitations
    of this regression plant, not assumptions in the real ARL backend.
    """

    names = ["back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop"]

    def __init__(self, config):
        self.config = deepcopy(config)
        self.dt = config["simulation"]["dt"]
        self.gravity = np.asarray(config["simulation"]["gravity"], dtype=float)
        self.mass = MassProperties(1.24, np.array([
            [.0080, .0007, -.0004], [.0007, .0090, .0003], [-.0004, .0003, .0140],
        ]), np.array([.006, -.004, .002]))
        self.inv_inertia = np.linalg.inv(self.mass.inertia_com_b)
        origins = np.array([[-.1, .1, 0], [-.1, -.1, 0], [.1, .1, 0], [.1, -.1, 0]])
        axes = np.tile([0., 0., 1.], (4, 1))
        reaction = .07 * np.array([-1., 1., 1., -1.])
        moments = np.cross(origins - self.mass.com_b, axes) + reaction[:, None] * axes
        self.allocation_matrix_b = np.vstack((axes.T, moments.T))
        self.native = NumpyNativeMotorFixture(self.dt)
        self.motor = NativeRpsActuator(self.native, np.zeros((1, 4)))
        self.contact_floor_z = .08  # Synthetic CoM support plane, in world m.
        self.reset()

    def ground_start_position(self, ground_z_m, clearance_m):
        self.contact_floor_z = ground_z_m + .08
        root = np.asarray(self.config["vehicle"]["initial_state"]["pos"], dtype=float).copy()
        rotation = quaternion_to_matrix(np.asarray(self.config["vehicle"]["initial_state"]["quaternion_wxyz"]))
        root[2] = self.contact_floor_z + clearance_m - (rotation @ self.mass.com_b)[2]
        return root

    def reset(self, initial_position_w=None):
        initial = self.config["vehicle"]["initial_state"]
        root = np.asarray(initial["pos"] if initial_position_w is None else initial_position_w, dtype=float)
        quaternion = np.asarray(initial["quaternion_wxyz"], dtype=float)
        rotation = quaternion_to_matrix(quaternion)
        self.position = root + rotation @ self.mass.com_b
        angular_velocity_w = np.asarray(initial["ang_vel"], dtype=float)
        self.velocity = np.asarray(initial["lin_vel"], dtype=float) + np.cross(angular_velocity_w, rotation @ self.mass.com_b)
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
        return {"sampled_kf": kf[0].copy(), "rps_min": low[0].copy(), "rps_max": high[0].copy(),
                "thrust_min_n": np.full(4, .1), "thrust_max_n": np.full(4, 10.),
                "current_rps": self.motor.rps[0].copy(), "thruster_names": self.names.copy()}

    def thrust_to_motor_speeds(self, thrust):
        params = self.motor_parameters()
        return thrust_to_motor_speeds(thrust, params["sampled_kf"],
                                      params["thrust_min_n"], params["thrust_max_n"])

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
                "motor_speed_source": "synthetic_fixture_native_rps_adapter_state",
                "linear_acceleration_source": "unavailable_in_reference_plant",
                "contact_impulse_n_s": self.contact_impulse_n_s}


def run_reference(config_path=ROOT / "configs/arl_robot_1_helix.yaml", duration_s=None):
    config = load_config(config_path)
    if config["trajectory"]["kind"] not in ("helix", "spiral"):
        raise ValueError("Reference regression requires a helix task config")
    config["simulation"]["device"] = "cpu"
    duration = config["simulation"]["duration_s"] if duration_s is None else float(duration_s)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    plant = SyntheticRigidBody(config)
    loop = MotionControlLoop(config, plant)
    loop.reset()
    count = int(np.ceil(duration / plant.dt))
    errors, speeds, heights, phases, max_rotor = [], [], [], set(), 0.
    first_success_s = None
    initial = plant.read_state(0.)
    initial_motor_rps = plant.motor.rps[0].copy()
    for index in range(count):
        loop.prepare_step()
        plant.step()
        record = loop.finish_step()
        errors.append(record["basic"]["position_error_norm_m"])
        speeds.append(float(np.linalg.norm(plant.velocity)))
        heights.append(float(plant.position[2]))
        phases.add(record["mission_phase"])
        max_rotor = max(max_rotor, float(np.max(plant.motor.rps)))
        if record["mission"]["achieved"] and first_success_s is None:
            first_success_s = loop.time_s
    # Endpoint success is based on actual full-state dwell, not just passing
    # the reference endpoint. A short diagnostic run legitimately fails it.
    last = record["mission"]
    success = bool(last["achieved"] and not last["timed_out"])
    return {
        "validation_scope": "synthetic_CPU_regression_NOT_Isaac_Sim_or_ARL_flight_validation",
        "plant_limitations": ["synthetic mass/inertia/motor coefficients", "frictionless CoM contact support",
                              "no CFD, ground effect, drag, battery or sensor noise", "native integrators transcribed to NumPy"],
        "config": str(config_path), "dt_s": plant.dt, "duration_s": loop.time_s,
        "steps": count, "phases_observed": sorted(phases), "success": success,
        "initial_position_w_m": initial.position_w.tolist(),
        "initial_motor_rps": initial_motor_rps.tolist(),
        "final_position_w_m": plant.position.tolist(), "final_velocity_w_m_s": plant.velocity.tolist(),
        "endpoint_position_w_m": loop.trajectory.endpoint_position_w.tolist(),
        "maximum_position_error_m": float(np.max(errors)),
        "rms_position_error_m": float(np.sqrt(np.mean(np.square(errors)))),
        "maximum_speed_m_s": float(np.max(speeds)), "maximum_altitude_w_m": float(np.max(heights)),
        "maximum_motor_rps": max_rotor, "first_hover_success_time_s": first_success_s,
        "final_mission_status": last,
        "mass_kg": plant.mass.mass_kg, "com_b_m": plant.mass.com_b.tolist(),
        "inertia_com_b_kg_m2": plant.mass.inertia_com_b.tolist(),
        "sampled_kf_n_per_rps2": plant.native.thrust_const[0].tolist(),
        "tau_inc_s": plant.native.tau_inc_s[0].tolist(), "tau_dec_s": plant.native.tau_dec_s[0].tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/arl_robot_1_helix.yaml")
    parser.add_argument("--duration", type=float, help="Override seconds; a run ending before hover dwell fails")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args(argv)
    result = run_reference(args.config, args.duration)
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
