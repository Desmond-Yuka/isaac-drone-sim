#!/usr/bin/env python3
"""Validate, inspect, or run ARL motion control in an Isaac Lab Python environment.

Ordinary Python supports --validate. Simulator imports occur only after
AppLauncher starts. This script never installs or silently selects a simulator.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from isaac_drone.config import DEFAULT_CONFIG, load_config, resolve_asset, validate_config
from isaac_drone.telemetry import dump_json, asset_hashes


def parser_for_task():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate", action="store_true", help="Check configuration and local asset without launching simulation")
    parser.add_argument("--inspect", action="store_true", help="Load actual PhysX mass/geometry, print diagnostics, and exit without flying")
    parser.add_argument("--trajectory", choices=("hold", "helix", "spiral"))
    parser.add_argument("--duration", type=float, help="Override simulation duration [s]")
    return parser


def main(argv=None):
    parser = parser_for_task()
    preliminary, unknown = parser.parse_known_args(argv)
    config = load_config(preliminary.config)
    if preliminary.trajectory:
        config["trajectory"]["kind"] = preliminary.trajectory
    if preliminary.duration is not None:
        config["simulation"]["duration_s"] = preliminary.duration
    validate_config(config)
    asset_path = resolve_asset(config)
    if preliminary.validate:
        if unknown:
            parser.error("--validate does not accept simulator arguments: " + " ".join(unknown))
        print(dump_json({"configuration": "valid", "asset_path": asset_path,
                         "rotor_direction_source": config["vehicle"]["rotor_direction_source"],
                         "rotor_directions": config["vehicle"]["rotor_directions"],
                         "physics_dt_s": config["simulation"]["dt"],
                         "control_dt_s": config["simulation"]["dt"]*config["simulation"]["control_decimation"],
                         "asset_layer_sha256": asset_hashes(asset_path),
                         "runtime_geometry_check": "requires --inspect in Isaac Lab",
                         "trajectory": config["trajectory"]["kind"],
                         "power_enabled": config["power"]["enabled"]}))
        return 0
    if config["power"]["enabled"] and not preliminary.inspect:
        raise ValueError("The standalone CLI has no calibrated current/envelope plugins. Inject a PowerSystem into MotionControlLoop from a custom entry point")
    try:
        from isaaclab.app import AppLauncher
    except ImportError as error:
        raise RuntimeError("Use the installed Isaac Lab / Isaac Sim Python environment for flight; ordinary Python supports --validate") from error
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device=config["simulation"]["device"])
    args = parser.parse_args(argv)
    config["simulation"]["device"] = args.device
    app = AppLauncher(args).app
    try:
        return run_simulation(app, config, asset_path, inspect_only=args.inspect)
    finally:
        app.close()


def run_simulation(app, config, asset_path, *, inspect_only=False):
    import random
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab_contrib.assets import Multirotor
    from isaac_drone.backends.isaaclab import (
        ARLBackend, make_robot_cfg, make_simulation_cfg, apply_usd_overrides, _override,
    )
    from isaac_drone.runtime import MotionControlLoop
    from isaac_drone.telemetry import RunRecorder

    seed = config["simulation"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sim = sim_utils.SimulationContext(make_simulation_cfg(config))
    ground = config["scene"]["ground"]
    if ground["enabled"]:
        cfg = sim_utils.GroundPlaneCfg()
        _override(cfg, ground["native_overrides"], "scene.ground")
        cfg.func(ground["prim_path"], cfg, translation=(0.0, 0.0, config["vehicle"]["launch"]["ground_z_m"]))
    light = config["scene"]["light"]
    if light["enabled"]:
        cfg = sim_utils.DomeLightCfg(intensity=light["intensity"])
        cfg.func(light["prim_path"], cfg)
    robot_cfg = make_robot_cfg(config, asset_path)
    robot = Multirotor(robot_cfg)
    apply_usd_overrides(robot_cfg, config["vehicle"]["usd_overrides"])
    sim.reset()
    robot.update(0.0)
    backend = ARLBackend(sim, robot, config)
    if inspect_only:
        backend.reset()
        print(dump_json({"config": config, "asset_layer_sha256": asset_hashes(asset_path), "backend": backend.telemetry()}))
        return 0
    loop = MotionControlLoop(config, backend)
    loop.reset()
    steps = math.ceil(config["simulation"]["duration_s"] / loop.dt_s)
    directory = Path(config["logging"]["directory"])
    if not directory.is_absolute():
        directory = ROOT / directory
    metadata = {"asset_layer_sha256": asset_hashes(asset_path), "backend": backend.telemetry(),
                "state_source": "simulation_truth", "motor_model": "RPS commands and native first-order motor integrators",
                "trajectory_feasibility": loop.feasibility,
                "limitations": ["Simulation rotor directions are an explicit engineering choice, not measured ARL data",
                                "Disabled effects/power are not modelled; they are not evidence of zero drag or unlimited energy",
                                "Coefficient-based propulsion does not simulate full airflow or ESC electrical dynamics"]}
    with RunRecorder(directory, config, metadata) as recorder:
        print(f"ARL run records: {recorder.path}")
        try:
            while app.is_running() and loop.step_index < steps:
                if sim.is_stopped():
                    break
                if not sim.is_playing():
                    sim.render()
                    continue
                loop.prepare_step()
                sim.step(render=(loop.step_index+1) % config["simulation"]["render_interval"] == 0)
                robot.update(loop.dt_s)
                record = loop.finish_step()
                if record["step"] % config["logging"]["every_n_steps"] == 0 or loop.step_index == steps:
                    recorder.write(record)
            mission = loop.mission.status if loop.mission is not None else None
            success = loop.step_index == steps and (mission is None or mission["achieved"])
            recorder.write({"event": "finished", "physics_steps": loop.step_index,
                            "simulated_time_s": loop.time_s, "requested_steps": steps,
                            "success": success, "mission": mission})
            print(dump_json({"success": success, "mission": mission}))
        except BaseException as error:
            recorder.write({"event": "aborted", "time_s": loop.time_s, "error": str(error)})
            raise
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
