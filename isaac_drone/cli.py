"""Command line entry point: ``python -m isaac_drone <command>``.

Commands that need Isaac Sim (``inspect``, ``run --backend isaaclab``) import
it only after all project arguments are validated; the rest run in plain
Python with NumPy/PyYAML (plus matplotlib for figures).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaac_drone.config import DEFAULT_CONFIG, load_config, resolve_asset, validate_config

CAMERAS = ("overview", "follow", "top")


def _config_arguments(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment YAML (default: %(default)s)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="Override one config value (YAML syntax), e.g. --set controller.position_kp=[10,10,6]; repeatable",
    )
    parser.add_argument("--duration", type=float, help="Override simulation.duration_s [s]")


def _recording_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        default=None,
        help="Record selected cameras and a synchronized composite MP4 in the run directory",
    )
    group.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
        help="Disable video recording even when enabled in YAML",
    )
    parser.add_argument("--video-fps", type=int, help="Video frames per simulated second (up to physics frequency)")
    parser.add_argument("--video-width", type=int, help="Each camera's width in pixels (positive even integer)")
    parser.add_argument("--video-height", type=int, help="Each camera's height in pixels (positive even integer)")
    parser.add_argument(
        "--video-cameras",
        nargs="+",
        choices=CAMERAS,
        help="Cameras to record, in composite layout order (default: overview follow top)",
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m isaac_drone", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    subparsers = {}

    validate = commands.add_parser("validate", help="Check a configuration and the local asset (no simulator)")
    _config_arguments(validate)
    _recording_arguments(validate)
    subparsers["validate"] = validate

    run = commands.add_parser(
        "run",
        help="Fly one mission and record a run directory",
        description="Fly one mission. Isaac Lab AppLauncher options are accepted with "
        "--backend isaaclab (e.g. --visualizer none, --livestream 1).",
    )
    run.add_argument(
        "--backend",
        choices=("synthetic", "isaaclab"),
        required=True,
        help="synthetic: CPU rigid-body plant (seconds, no Isaac Sim); isaaclab: PhysX in Isaac Sim",
    )
    _config_arguments(run)
    run.add_argument("--no-figures", action="store_true", help="Do not write PNG figures after the run")
    run.add_argument(
        "--wait-for-start",
        action="store_true",
        help="isaaclab: after loading the scene keep the UI/stream live and start on Enter",
    )
    run.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="isaaclab displayed runs: simulated seconds per wall-clock second (0 = unpaced)",
    )
    run.add_argument(
        "--no-path-overlay",
        action="store_true",
        help="isaaclab: do not draw the reference path, flown path and setpoint",
    )
    _recording_arguments(run)
    subparsers["run"] = run

    inspect = commands.add_parser("inspect", help="Isaac Sim: load PhysX mass/geometry, print diagnostics, exit")
    _config_arguments(inspect)
    subparsers["inspect"] = inspect

    audit = commands.add_parser("audit-usd", help="Offline authored-USD audit (needs usd-core, no simulator)")
    _config_arguments(audit)
    audit.add_argument("--asset", type=Path, help="Inspect this USD instead of vehicle.asset_path")

    sweep = commands.add_parser(
        "sweep",
        help="Grid of synthetic runs over config values, ranked by a metric",
        description="Cartesian product of --grid axes on the synthetic backend, run in "
        "parallel; each point is a full run directory.",
    )
    _config_arguments(sweep)
    sweep.add_argument(
        "--grid",
        action="append",
        required=True,
        metavar="KEY.PATH=[V1,V2,...]",
        help="One sweep axis (YAML list); repeat for more axes",
    )
    sweep.add_argument("--workers", type=int, help="Parallel worker processes (default: CPU count)")
    sweep.add_argument(
        "--rank",
        default="position_error_rms_m",
        help="Metric to sort by, ascending (default: %(default)s; per phase: helix.position_error_rms_m)",
    )
    sweep.add_argument("--output", type=Path, help="Sweep directory (default: runs/sweep_<UTC time>)")
    sweep.add_argument("--figures", action="store_true", help="Also write PNG figures for every point")

    compare = commands.add_parser("compare", help="Metrics table and overlaid figures for several runs")
    compare.add_argument("run_dirs", nargs="+", type=Path, help="Run directories to compare")
    compare.add_argument("--labels", nargs="+", help="One label per run (default: directory names)")
    compare.add_argument("--phase", help="Compare one mission phase (e.g. helix) instead of the whole run")
    compare.add_argument("--output", type=Path, help="Output directory (default: runs/compare_<UTC time>)")
    compare.add_argument("--no-figures", action="store_true", help="Only write the table")

    for name, text in (("plot", "Write PNG figures for a run"), ("summarize", "Per-phase table for a run")):
        command = commands.add_parser(name, help=text)
        command.add_argument("run_dir", nargs="?", type=Path, help="Run directory (default: newest under runs/)")
        command.add_argument("--runs-dir", type=Path, help="Where to look for the newest run (default: runs/)")
        if name == "plot":
            command.add_argument("--output", type=Path, help="Output directory (default: <run>/plots)")
    return parser, subparsers


def configure(args) -> dict:
    """Load the experiment config and apply command-line overrides, then validate again."""
    config = load_config(args.config, overrides=args.overrides)
    if args.duration is not None:
        config["simulation"]["duration_s"] = args.duration
    if getattr(args, "record_video", None) is not None:
        config["recording"]["enabled"] = args.record_video
    for name in ("fps", "width", "height", "cameras"):
        value = getattr(args, "video_" + name, None)
        if value is not None:
            config["recording"][name] = value
    validate_config(config)
    return config


def _default_runs_dir(args) -> Path:
    from isaac_drone.runtime.runner import REPO_ROOT

    return getattr(args, "runs_dir", None) or REPO_ROOT / "runs"


def _utc_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_report(config, asset_path) -> dict:
    from isaac_drone.telemetry.recorder import asset_hashes

    simulation = config["simulation"]
    return {
        "configuration": "valid",
        "asset_path": asset_path,
        "rotor_direction_source": config["vehicle"]["rotor_direction_source"],
        "rotor_directions": config["vehicle"]["rotor_directions"],
        "physics_dt_s": simulation["dt"],
        "control_dt_s": simulation["dt"] * simulation["control_decimation"],
        "asset_layer_sha256": asset_hashes(asset_path),
        "runtime_geometry_check": "requires `inspect` in Isaac Lab",
        "trajectory": config["trajectory"]["kind"],
        "power_enabled": config["power"]["enabled"],
        "recording": config["recording"],
    }


def main(argv=None) -> int:
    parser, subparsers = build_parser()
    args, unknown = parser.parse_known_args(argv)
    command = args.command

    if command in ("plot", "summarize"):
        if unknown:
            parser.error("unrecognized arguments: " + " ".join(unknown))
        if command == "plot":
            from isaac_drone.analysis.plots import latest_run, plot_run

            for path in plot_run(args.run_dir or latest_run(_default_runs_dir(args)), args.output):
                print(path)
        else:
            from isaac_drone.analysis.summary import latest_telemetry, summarize_run

            print(summarize_run(args.run_dir or latest_telemetry(_default_runs_dir(args))))
        return 0

    if command == "compare":
        if unknown:
            parser.error("unrecognized arguments: " + " ".join(unknown))
        from isaac_drone.analysis.compare import compare_runs

        output = args.output or _default_runs_dir(args) / f"compare_{_utc_stamp()}"
        report = compare_runs(args.run_dirs, output, labels=args.labels, phase=args.phase, figures=not args.no_figures)
        print(report["table"])
        print(f"\nWritten: {', '.join(str(path) for path in report['files'])}")
        return 0

    if command == "sweep":
        if unknown:
            parser.error("unrecognized arguments: " + " ".join(unknown))
        from isaac_drone.analysis.sweep import parse_grid, run_sweep

        base = list(args.overrides) + ([f"simulation.duration_s={args.duration}"] if args.duration else [])
        report = run_sweep(
            args.config,
            parse_grid(args.grid),
            base_overrides=base,
            output_dir=args.output,
            workers=args.workers,
            rank=args.rank,
            figures=args.figures,
        )
        print(report["table"])
        print(f"\nSummary: {report['output_dir'] / 'summary.csv'}")
        return 0

    config = configure(args)
    if command == "audit-usd":
        if unknown:
            parser.error("unrecognized arguments: " + " ".join(unknown))
        import json

        from isaac_drone.runtime.runner import REPO_ROOT
        from isaac_drone.sim.usd_audit import inspect as audit

        path = args.asset or Path(config["vehicle"]["asset_path"])
        try:
            report = audit(config, path if path.is_absolute() else REPO_ROOT / path)
        except ImportError as error:
            print(f"Optional OpenUSD Python bindings are required (usd-core): {error}", file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0

    asset_path = resolve_asset(config)
    if command == "validate":
        if unknown:
            parser.error("validate does not accept simulator arguments: " + " ".join(unknown))
        from isaac_drone.telemetry.recorder import dump_json

        print(dump_json(_validate_report(config, asset_path)))
        return 0

    if command == "run":
        if not math.isfinite(args.playback_speed) or args.playback_speed < 0:
            parser.error("--playback-speed must be finite and nonnegative")
        if args.backend == "synthetic":
            if unknown:
                parser.error("the synthetic backend does not accept simulator arguments: " + " ".join(unknown))
            from isaac_drone.sim.synthetic import run_simulation

            return run_simulation(config, figures=not args.no_figures).exit_code

    # Isaac Lab: inspect, or run --backend isaaclab.
    if config["power"]["enabled"] and command == "run":
        raise ValueError(
            "The CLI has no calibrated current/envelope plugins. "
            "Inject a PowerSystem into MotionControlLoop from a custom entry point"
        )
    try:
        from isaaclab.app import AppLauncher
    except ImportError as error:
        raise RuntimeError(
            "Use the installed Isaac Lab / Isaac Sim Python environment for this command; "
            "plain Python supports validate, run --backend synthetic, plot and summarize"
        ) from error
    AppLauncher.add_app_launcher_args(subparsers[command])
    subparsers[command].set_defaults(device=config["simulation"]["device"])
    args = parser.parse_args(argv)
    config["simulation"]["device"] = args.device
    from isaac_drone.sim.isaaclab.app import launch

    return launch(args, config, asset_path, inspect_only=command == "inspect")
