"""What each run was: a README.md card in every run directory and the ``runs`` table.

Run directory names only say when, which config, which backend and whether
video was recorded; the physics settings are described here. Everything is
read back from files every run already writes (config.json, metadata.json,
metrics.json, the final telemetry event, plots/, the videos), so runs recorded
before cards existed are listed too. Settings come from the effective
config.json, never from today's YAML defaults, which change over time.
"""

from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path

from isaac_drone.config import DEFAULT_CONFIG
from isaac_drone.telemetry.naming import BACKEND_LABELS, VIDEO_DIR, run_clock

CARD_FILE = "README.md"
_OLD_NAME = re.compile(r"(\d{8}T\d{6}Z)_[0-9a-f]{8}")  # directory names before 2026-10-19
_FIGURES = (
    "trajectory_3d",
    "position",
    "velocity",
    "acceleration",
    "attitude",
    "angular_velocity",
    "motor_speed",
    "rotor_thrust",
    "force",
    "torque",
)
_MISMATCH = (
    ("mass_ratio", "mass"),
    ("inertia_ratio", "inertia"),
    ("thrust_coefficient_ratio", "thrust coeff"),
    ("reaction_torque_ratio", "reaction torque"),
)
_CARD_METRICS = (
    ("position_error_rms_m", "Position error RMS [m]", 3),
    ("position_error_max_m", "Position error max [m]", 3),
    ("position_error_final_m", "Position error at end [m]", 3),
    ("velocity_error_rms_m_s", "Velocity error RMS [m/s]", 3),
    ("attitude_error_max_deg", "Attitude error max [deg]", 1),
    ("tilt_max_deg", "Tilt max [deg]", 1),
    ("speed_max_m_s", "Speed max [m/s]", 2),
    ("saturation_fraction", "Motor saturation fraction", 3),
)
_TABLE = (
    ("started", "Started"),
    ("config", "Config"),
    ("backend", "Backend"),
    ("video", "Video"),
    ("settings", "Physics settings"),
    ("result", "Result"),
    ("rms", "Pos RMS [m]"),
    ("name", "Directory"),
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _final_event(run: Path) -> dict:
    """The finished/aborted event on the last line of telemetry.jsonl ({} if the run never ended)."""
    try:
        with (run / "telemetry.jsonl").open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 262144))
            lines = stream.read().splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a blank line, or the first line cut by the read window
        return record if isinstance(record, dict) and record.get("event") in ("finished", "aborted") else {}
    return {}


def _started(run: Path, metadata: dict) -> datetime:
    """UTC start time: metadata, else the old ``<UTC>Z_<hex>`` directory name, else the directory mtime."""
    text = (metadata.get("provenance") or {}).get("started_utc")
    if text:
        return datetime.fromisoformat(text)
    match = _OLD_NAME.fullmatch(run.name)
    if match:
        return datetime.strptime(match[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(run.stat().st_mtime, timezone.utc)


def _g(value) -> str:
    return "?" if value is None else f"{value:g}"


def _number(value, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _vector(values, unit: str = "", sep: str = ", ") -> str:
    """``0.5`` for a uniform vector, else ``(1, 0, 0)``; then the unit."""
    numbers = [float(value) for value in (values if isinstance(values, list) else [values])]
    text = f"{numbers[0]:g}" if len(set(numbers)) == 1 else "(" + sep.join(f"{value:g}" for value in numbers) + ")"
    return f"{text} {unit}".rstrip()


def _ou(ou: dict, kind: str, sep: str) -> tuple[str, str, str, str] | None:
    """(mean, SD, unit, time constant) of the OU force or torque, None when mean and SD are zero."""
    keys = {"force": ("force_mean_w_n", "force_std_w_n", "N"), "torque": ("torque_mean_b_nm", "torque_std_b_nm", "N·m")}
    mean_key, std_key, unit = keys[kind]
    mean, std = ou.get(mean_key) or [0.0], ou.get(std_key) or [0.0]
    if not any(mean) and not any(std):
        return None
    return _vector(mean, "", sep), _vector(std, "", sep), unit, _g(ou.get(f"{kind}_time_constant_s"))


def _constant(item: dict, sep: str) -> str:
    force, torque = item.get("force_n") or [0.0], item.get("torque_nm") or [0.0]
    pieces = ([f"force {_vector(force, 'N', sep)}"] if any(force) else []) + (
        [f"torque {_vector(torque, 'N·m', sep)}"] if any(torque) else []
    )
    start, end = item.get("start_time_s") or 0.0, item.get("end_time_s")
    window = f" {start:g}–{end:g} s" if end is not None else (f" from {start:g} s" if start else "")
    frame = " (body)" if item.get("frame") == "body" else ""
    return "const " + (" + ".join(pieces) or "0") + frame + window


def settings_line(config: dict) -> str | None:
    """One line of the physics settings that differ from an ideal plant, None for pre-schema-2 configs.

    E.g. ``OU force (1,0,0)±0.5 N τ 1 s; mass ×1.2``. The ideal plant has no
    disturbance, no model mismatch, no rotor gyroscopic torque and seed 0.
    """
    if config.get("schema_version") != 2:
        return None
    effects = config.get("effects") or {}
    parts = []
    ou = effects.get("ou_wrench") or {}
    if ou.get("enabled"):
        for kind in ("force", "torque"):
            values = _ou(ou, kind, ",")
            if values is not None:
                mean, std, unit, tau = values
                parts.append(f"OU {kind} {mean}±{std} {unit} τ {tau} s")
    parts += [_constant(item, ",") for item in effects.get("disturbances") or [] if item.get("enabled", True)]
    wind = effects.get("wind") or {}
    if wind.get("enabled"):
        parts.append("wind" + (" + turbulence" if (wind.get("turbulence") or {}).get("enabled") else ""))
    if (effects.get("aerodynamic") or {}).get("enabled"):
        parts.append("drag")
    mismatch = config.get("model_mismatch") or {}
    for key, label in _MISMATCH:
        value = mismatch.get(key)
        if value is not None and any(float(item) != 1.0 for item in (value if isinstance(value, list) else [value])):
            parts.append(f"{label} ×{_vector(value, '', ',')}")
    inertia = (config.get("vehicle") or {}).get("rotor_inertia_kg_m2")
    if inertia:
        parts.append(f"gyro J_r {inertia:g}")
    seed = (config.get("simulation") or {}).get("seed")
    if seed:
        parts.append(f"seed {seed}")
    return "; ".join(parts) or "ideal (no disturbance, no model mismatch)"


def _settings_rows(config: dict) -> list[tuple[str, str]]:
    """Every physics setting of a schema-2 config, including the ones that are off."""
    if config.get("schema_version") != 2:
        return []
    effects = config.get("effects") or {}
    ou = effects.get("ou_wrench") or {}
    rows = []
    if ou.get("enabled"):
        end = ou.get("end_time_s")
        window = f"from {_g(ou.get('start_time_s') or 0.0)} s to " + ("the end" if end is None else f"{end:g} s")
        labels = (("force", "OU random force (world frame, at CoM)"), ("torque", "OU random torque (body frame)"))
        for kind, label in labels:
            values = _ou(ou, kind, ", ")
            if values is None:
                rows.append((label, "off"))
            else:
                mean, std, unit, tau = values
                rows.append((label, f"mean {mean} {unit}, SD {std} {unit}, τ {tau} s, {window}"))
    else:
        rows.append(("OU random force and torque", "off"))
    constants = [_constant(item, ", ") for item in effects.get("disturbances") or [] if item.get("enabled", True)]
    rows.append(("Constant disturbances", "; ".join(constants) or "none"))
    wind = effects.get("wind") or {}
    turbulence = (wind.get("turbulence") or {}).get("enabled")
    rows.append(("Wind", ("on" + (", turbulence" if turbulence else "")) if wind.get("enabled") else "off"))
    rows.append(("Aerodynamic drag", "on" if (effects.get("aerodynamic") or {}).get("enabled") else "off"))
    mismatch = config.get("model_mismatch") or {}
    ratios = [f"{label} ×{_vector(mismatch[key])}" for key, label in _MISMATCH if mismatch.get(key) is not None]
    rows.append(("Model mismatch (plant / controller)", ", ".join(ratios) or "none"))
    inertia = (config.get("vehicle") or {}).get("rotor_inertia_kg_m2")
    rows.append(("Rotor gyroscopic torque", f"on, J_r {inertia:g} kg·m²" if inertia else "off"))
    rows.append(("Seed", str((config.get("simulation") or {}).get("seed", "?"))))
    return rows


def _config_name(config: dict, metadata: dict) -> str:
    """``logging.run_name`` (the YAML file name), else ``--config`` from the command line."""
    name = (config.get("logging") or {}).get("run_name")
    if name:
        return name
    argv = (metadata.get("provenance") or {}).get("argv") or []
    for index, argument in enumerate(argv):
        if argument == "--config" and index + 1 < len(argv):
            return Path(argv[index + 1]).stem
        if argument.startswith("--config="):
            return Path(argument.split("=", 1)[1]).stem
    return DEFAULT_CONFIG.stem if argv else "?"


def _result(end: dict) -> str:
    if end.get("event") == "finished":
        return "ok" if end.get("success") else "failed"
    return "aborted" if end.get("event") == "aborted" else "incomplete"


def _video(config: dict, end: dict) -> str:
    summary = end.get("video") or {}
    if not summary:
        return "?" if (config.get("recording") or {}).get("enabled") else "-"
    if summary.get("failed"):
        return "failed"
    cameras = summary.get("camera_order") or (config.get("recording") or {}).get("cameras") or []
    return f"{len(cameras)} cams, {summary.get('frames', '?')} frames"


def describe_run(run_dir) -> dict:
    """One row of the ``runs`` table (all values are display strings except ``path`` and ``started_utc``)."""
    run = Path(run_dir)
    config, metadata = _read_json(run / "config.json"), _read_json(run / "metadata.json")
    end = _final_event(run)
    started = _started(run, metadata)
    rms = (_read_json(run / "metrics.json").get("overall") or {}).get("position_error_rms_m")
    return {
        "path": run,
        "name": run.name,
        "started_utc": started,
        "started": f"{run_clock(started):%m-%d %H:%M}",
        "config": _config_name(config, metadata),
        "backend": BACKEND_LABELS.get(metadata.get("backend_kind"), metadata.get("backend_kind") or "?"),
        "video": _video(config, end),
        "settings": settings_line(config) or "?",
        "result": _result(end),
        "rms": _number(rms, 3),
    }


def list_runs(runs_dir) -> list[dict]:
    """Every run directly under ``runs_dir``, oldest first (sweep/compare folders and videos/ are skipped)."""
    root = Path(runs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"No runs directory {root}")
    rows = [describe_run(path) for path in root.iterdir() if (path / "metadata.json").is_file()]
    return sorted(rows, key=lambda row: (row["started_utc"], row["name"]))


def format_table(rows) -> str:
    """Plain-text table with aligned columns, one run per line."""
    lines = [[title for _, title in _TABLE], *([str(row[key]) for key, _ in _TABLE] for row in rows)]
    widths = [max(len(line[column]) for line in lines) for column in range(len(_TABLE))]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in lines)


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _command(argv) -> str:
    if not argv:
        return "?"
    if Path(argv[0]).name == "__main__.py":
        return f"`python -m isaac_drone {shlex.join(argv[1:])}`"
    return f"`{shlex.join(argv)}`"


def _code(provenance: dict) -> str:
    commit = provenance.get("git_commit")
    if not commit:
        return "?"
    return f"`{commit[:7]}`" + (" with uncommitted changes" if provenance.get("git_dirty") else "")


def _result_detail(end: dict, info: dict) -> str:
    details = []
    simulated = end.get("simulated_time_s", end.get("time_s"))
    if simulated is not None:
        details.append(f"{simulated:g} s simulated")
    if end.get("physics_steps") is not None and end["physics_steps"] < end.get("requested_steps", 0):
        details.append("stopped before the configured duration")
    if info.get("completion_time_s") is not None:
        details.append(f"mission complete at {info['completion_time_s']:.2f} s")
    elif end.get("mission") is not None and not end["mission"].get("achieved"):
        details.append("mission completion criterion not met")
    if end.get("error"):
        details.append(f"error: {end['error']}")
    return _result(end) + (f" ({'; '.join(details)})" if details else "")


def _video_detail(run: Path, config: dict, end: dict) -> str:
    summary = end.get("video") or {}
    if not summary:
        return "requested, but the run left no video summary" if (config.get("recording") or {}).get("enabled") else "-"
    if summary.get("failed"):
        return "recording failed: " + ("; ".join(summary.get("errors") or []) or "unknown error")
    cameras = summary.get("camera_order") or (config.get("recording") or {}).get("cameras") or []
    views = ", ".join([*cameras, "combined"])
    text = f"{views}: {summary.get('frames', '?')} frames at {_g(summary.get('fps'))} fps"
    files = [video.get("filename") for video in (summary.get("videos") or {}).values()]
    links = [f"[{name}]({name})" for name in files if name and (run / name).is_file()]
    if links:
        text += "; files: " + ", ".join(links)
    copy = run.parent / VIDEO_DIR / f"{run.name}.mp4"
    if copy.is_file():
        text += f"; copy: [{VIDEO_DIR}/{copy.name}](../{VIDEO_DIR}/{copy.name})"
    return text


def _metrics_section(metrics: dict) -> list[str]:
    overall = metrics.get("overall")
    if not overall:
        return []
    columns = [("overall", overall), *(metrics.get("phases") or {}).items()]
    lines = ["", "## Metrics", "", "| | " + " | ".join(name for name, _ in columns) + " |"]
    lines.append("|---" * (len(columns) + 1) + "|")
    for key, label, digits in _CARD_METRICS:
        lines.append(f"| {label} | " + " | ".join(_number(values.get(key), digits) for _, values in columns) + " |")
    return lines


def _figures(run: Path) -> list[Path]:
    order = {name: index for index, name in enumerate(_FIGURES)}
    return sorted((run / "plots").glob("*.png"), key=lambda path: (order.get(path.stem, len(order)), path.name))


def run_card(run_dir) -> str:
    """README.md text for one run: what was run, the physics settings, video, result, metrics and figures."""
    run = Path(run_dir)
    config, metadata = _read_json(run / "config.json"), _read_json(run / "metadata.json")
    metrics, end = _read_json(run / "metrics.json"), _final_event(run)
    provenance = metadata.get("provenance") or {}
    started = run_clock(_started(run, metadata))
    controller = (config.get("controller") or {}).get("kind", "?")
    trajectory = (config.get("trajectory") or {}).get("kind", "?")
    facts = (
        ("Started", f"{started:%Y-%m-%d %H:%M} {started.tzname()}"),
        ("Config", _config_name(config, metadata)),
        ("Command", _command(provenance.get("argv"))),
        ("Backend", metadata.get("backend_kind") or "?"),
        ("Controller / trajectory", f"{controller} / {trajectory}"),
        ("Code", _code(provenance)),
        ("Result", _result_detail(end, metrics.get("run") or {})),
        ("Video", _video_detail(run, config, end)),
    )
    lines = [f"# {run.name}", "", "| | |", "|---|---|", *(f"| {key} | {_cell(value)} |" for key, value in facts)]
    settings = _settings_rows(config)
    if settings:
        lines += ["", "## Physics settings", "", "| Setting | Value |", "|---|---|"]
        lines += [f"| {key} | {_cell(value)} |" for key, value in settings]
    elif config:
        lines += ["", "## Physics settings", "", f"Config schema {config.get('schema_version')}: see config.json."]
    lines += _metrics_section(metrics)
    figures = _figures(run)
    if figures:
        lines += ["", "## Figures"]
        for path in figures:
            lines += ["", f"![{path.stem}](plots/{path.name})"]
    return "\n".join(lines) + "\n"


def write_run_card(run_dir) -> Path:
    path = Path(run_dir) / CARD_FILE
    path.write_text(run_card(run_dir), encoding="utf-8")
    return path


__all__ = ["CARD_FILE", "describe_run", "format_table", "list_runs", "run_card", "settings_line", "write_run_card"]
