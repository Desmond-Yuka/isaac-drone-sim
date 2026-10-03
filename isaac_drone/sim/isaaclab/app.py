"""Isaac Lab application: scene construction, the physics driver, UI streaming and the run.

Everything here imports Isaac Lab / Kit lazily, only after ``AppLauncher`` has
started. Control, logging and hooks are the backend-independent ones in
``isaac_drone.runtime``; this module only supplies Isaac-specific pieces.
"""

from __future__ import annotations

import select
import sys

import numpy as np

from isaac_drone.telemetry.recorder import asset_hashes, dump_json

LIMITATIONS = [
    "Simulation rotor directions are an explicit engineering choice, not measured ARL data",
    "Disabled effects/power are not modelled; they are not evidence of zero drag or unlimited energy",
    "Coefficient-based propulsion does not simulate full airflow or ESC electrical dynamics",
]


def pump_kit_ui():
    """Refresh the Kit UI/viewport, and thus the WebRTC stream, without stepping physics.

    Isaac Lab 3.0 treats a livestream host as headless and never calls
    app.update() while stepping, so a connected client would receive no frames.
    playSimulations is disabled around the update, as in Isaac Lab's KitVisualizer.
    """
    import carb.settings
    import omni.kit.app

    settings = carb.settings.get_settings()
    previous = settings.get("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    try:
        omni.kit.app.get_app().update()
    finally:
        settings.set_bool("/app/player/playSimulations", True if previous is None else bool(previous))


def aim_viewport(eye, target):
    """Point the active viewport camera; purely cosmetic, so failures only warn."""
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf

        viewport = get_active_viewport()
        state = ViewportCameraState(viewport.get_active_camera(), viewport)
        state.set_position_world(Gf.Vec3d(*map(float, eye)), False)
        state.set_target_world(Gf.Vec3d(*map(float, target)), True)
    except Exception as error:
        print(f"Viewport camera not set: {error}", flush=True)


class IsaacDriver:
    """SimulationDriver: one ``sim.step`` plus ``robot.update(dt)`` per physics step."""

    def __init__(self, app, sim, robot, dt_s: float):
        self.app, self.sim, self.robot, self.dt_s = app, sim, robot, dt_s

    def step(self) -> None:
        self.sim.step(render=False)
        self.robot.update(self.dt_s)

    def is_running(self) -> bool:
        return self.app.is_running() and not self.sim.is_stopped()

    def is_paused(self) -> bool:
        return not self.sim.is_playing()

    def idle(self) -> None:
        self.sim.render()


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scene(config: dict, asset_path):
    """Create the simulation context, ground, light and robot; return (sim, robot, backend)."""
    import isaaclab.sim as sim_utils
    from isaaclab_contrib.assets import Multirotor

    from isaac_drone.sim.isaaclab.backend import (
        ARLBackend,
        apply_usd_overrides,
        make_robot_cfg,
        make_simulation_cfg,
        override_native,
    )

    sim = sim_utils.SimulationContext(make_simulation_cfg(config))
    ground = config["scene"]["ground"]
    if ground["enabled"]:
        cfg = sim_utils.GroundPlaneCfg()
        override_native(cfg, ground["native_overrides"], "scene.ground")
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
    return sim, robot, ARLBackend(sim, robot, config)


def wait_for_enter(app, sim, pump_ui: bool) -> None:
    """Keep the UI/stream live without stepping physics until Enter is pressed."""
    print("Scene ready: connect the stream client, then press Enter here to start.", flush=True)
    while app.is_running() and not sim.is_stopped():
        if pump_ui:
            pump_kit_ui()
        else:
            sim.render()
        if select.select([sys.stdin], [], [], 0.02)[0]:
            sys.stdin.readline()
            break


def run_simulation(
    app,
    config,
    asset_path,
    *,
    inspect_only=False,
    pump_ui=False,
    wait_for_start=False,
    playback_speed=1.0,
    path_overlay=True,
) -> int:
    """Inspect the vehicle, or fly the configured mission once; return the process exit code."""
    import isaaclab.sim as sim_utils

    from isaac_drone.runtime.hooks import DisplayHook, FiguresHook, MetricsHook, PathOverlayHook, VideoHook
    from isaac_drone.runtime.loop import MotionControlLoop
    from isaac_drone.runtime.runner import run_experiment

    seed_everything(config["simulation"]["seed"])
    sim, robot, backend = build_scene(config, asset_path)
    if inspect_only:
        backend.reset()
        print(
            dump_json(
                {"config": config, "asset_layer_sha256": asset_hashes(asset_path), "backend": backend.telemetry()}
            )
        )
        return 0
    loop = MotionControlLoop(config, backend)
    loop.reset()
    if loop.plan_summary():
        print("Trajectory plan:", loop.plan_summary(), flush=True)
    start = backend.read_state(0.0).position_w
    reference_points = loop.reference_path_points()
    recording = config["recording"]
    framing = np.vstack([start, reference_points])
    if pump_ui:
        target = (framing.min(axis=0) + framing.max(axis=0)) / 2
        aim_viewport(target + np.array([5.0, -5.0, 2.0]), target)
    displayed = pump_ui or sim.is_rendering

    hooks = []
    overlay_hook = None
    if (displayed or recording["enabled"]) and path_overlay:
        from isaac_drone.viz.overlay import PathOverlay, PathTrail

        overlay = PathOverlay(sim_utils.get_current_stage())
        if np.ptp(reference_points, axis=0).max() > 1e-6:
            overlay.set_reference(reference_points)
        overlay_hook = PathOverlayHook(overlay, PathTrail())
        overlay_hook.trail.add(start)
        overlay.update(overlay_hook.trail.points, loop.reference(loop.time_s).position_w)
        hooks.append(overlay_hook)
    if recording["enabled"]:
        from isaac_drone.sim.isaaclab.camera import IsaacCameraRig

        hooks.append(
            VideoHook(
                recording,
                framing,
                start,
                lambda width, height, poses: IsaacCameraRig(sim, width, height, poses),
                before_capture=overlay_hook.refresh if overlay_hook else None,
                log=lambda text: print(text, flush=True),
            )
        )
    if displayed:

        def draw():
            if overlay_hook is not None:
                overlay_hook.refresh()
            if sim.is_rendering:
                sim.render()
            if pump_ui:
                pump_kit_ui()

        pacer = None
        if playback_speed > 0:
            from isaac_drone.viz.pacing import RealTimePacer

            pacer = RealTimePacer(playback_speed)
        hooks.append(DisplayHook(draw, config["simulation"]["render_interval"], pacer))
    hooks.append(MetricsHook(log=lambda text: print(text, flush=True)))
    hooks.append(FiguresHook(log=lambda text: print(text, flush=True)))

    if wait_for_start:
        wait_for_enter(app, sim, pump_ui)
    metadata = {
        "asset_layer_sha256": asset_hashes(asset_path),
        "backend": backend.telemetry(),
        "backend_kind": "isaaclab",
        "state_source": "simulation_truth",
        "motor_model": "RPS commands and native first-order motor integrators",
        **loop.run_metadata(),
        "limitations": LIMITATIONS,
    }
    result = run_experiment(
        loop,
        IsaacDriver(app, sim, robot, loop.dt_s),
        config,
        metadata=metadata,
        hooks=hooks,
        log=lambda text: print(text, flush=True),
    )
    print(dump_json({"success": result.success, "mission": result.mission, **result.summaries}))
    return result.exit_code


def launch(args, config, asset_path, *, inspect_only: bool) -> int:
    """Start Kit through AppLauncher with parsed CLI ``args`` and run; always closes the app."""
    import os

    from isaaclab.app import AppLauncher

    livestream = args.livestream if args.livestream >= 0 else int(os.environ.get("LIVESTREAM", 0))
    if config["recording"]["enabled"] and not inspect_only:
        from isaac_drone.viz.video import ensure_video_dependencies

        ensure_video_dependencies()
        # Isaac Lab 3.0 selects a rendering-capable Kit experience, including
        # --visualizer none. The independent rig does not need its gym video path.
        args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        return run_simulation(
            app,
            config,
            asset_path,
            inspect_only=inspect_only,
            pump_ui=livestream >= 1,
            wait_for_start=getattr(args, "wait_for_start", False),
            playback_speed=getattr(args, "playback_speed", 0.0),
            path_overlay=not getattr(args, "no_path_overlay", False),
        )
    finally:
        app.close()
