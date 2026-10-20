"""English PNG figures for one finished run, drawn from its basic.csv.

basic.csv and telemetry.jsonl stay the data of record; figures are a derived
view and never write back to them. Each tracked quantity gets one figure: a
row per component with the ideal (target or command) and the actual value,
then a row with the ideal-minus-actual error. Disturbance figures (wind, wind
on the rotors, OU force/torque) are drawn only for runs that recorded that
model; their bottom row shows the random part instead of an error.
plot_comparison draws the same set with several runs overlaid. matplotlib is
optional and imported only when drawing. The object API (no pyplot) needs no
GUI backend, so this also runs headless and inside the Isaac Sim process.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

_AXES = ("x", "y", "z")
_AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
_RPY = ("roll", "pitch", "yaw")
_ROTORS = ("BL", "BR", "FL", "FR")
_ROTOR_LABELS = ("BL (back left)", "BR (back right)", "FL (front left)", "FR (front right)")
_ROTOR_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
_ACTUAL = {"color": "tab:blue", "linewidth": 1.2}
_IDEAL = {"color": "black", "linestyle": "--", "linewidth": 1.0}
_SOLVER = {"color": "tab:orange", "linewidth": 0.8, "alpha": 0.8}
# One colour per overlaid run, in this fixed order (blue, orange, aqua, yellow).
RUN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_LEGACY_PHASE_NAMES = ("delay", "takeoff", "helix", "hold")
_DEG = 180.0 / math.pi


def load_basic_csv(path) -> dict[str, np.ndarray]:
    """Return numeric basic.csv columns as float arrays; blank cells become NaN.

    Text provenance columns are skipped. A blank stays NaN, never zero, so an
    unavailable quantity leaves a gap instead of a false line at zero.
    """
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        rows = list(reader)
    if header is None or not rows:
        raise ValueError(f"{path} has no data rows")
    columns = {}
    for index, name in enumerate(header):
        try:
            columns[name] = np.array([float(row[index]) if row[index] else np.nan for row in rows])
        except ValueError:
            continue
    return columns


def phase_starts(config, metadata=None) -> list[tuple[float, str]]:
    """(start time [s], name) of each nonempty mission phase; empty when unknown.

    Runs record ``mission_phases`` in metadata.json. Runs recorded before that
    field existed stored helix ``phase_boundaries_s`` (delay, takeoff, helix,
    hold) in the feasibility report, or only configured durations.
    """
    metadata = metadata or {}
    if metadata.get("mission_phases"):
        phases = [(float(start), str(name)) for start, name in metadata["mission_phases"]]
    else:
        phases = _legacy_phase_starts(config, metadata)
    ends = [start for start, _ in phases[1:]] + [math.inf]
    return [(start, name) for (start, name), end in zip(phases, ends) if end > start]


def _legacy_phase_starts(config, metadata):
    trajectory = config.get("trajectory", {})
    if trajectory.get("kind") not in ("helix", "spiral"):
        return []
    feasibility = metadata.get("trajectory_feasibility")
    starts = feasibility.get("phase_boundaries_s") if isinstance(feasibility, dict) else None
    if starts is None:
        spiral = trajectory.get("spiral", {})
        durations = [spiral.get(key) for key in ("start_delay_s", "takeoff_duration_s", "spiral_duration_s")]
        if not all(isinstance(value, (int, float)) for value in durations):
            return []
        starts = np.cumsum([0.0] + durations)
    starts = [float(start) for start in starts]
    if len(starts) != len(_LEGACY_PHASE_NAMES):
        return []
    return list(zip(starts, _LEGACY_PHASE_NAMES))


def latest_run(runs_dir) -> Path:
    """The run directory whose basic.csv was written most recently."""
    candidates = list(Path(runs_dir).glob("*/basic.csv"))
    if not candidates:
        raise FileNotFoundError(f"No run with basic.csv under {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


# (file name, positional and keyword arguments of _tracking) for each time-series figure.
_TRACKING_FIGURES = (
    (
        "position.png",
        (
            "Position (world frame)",
            "m",
            0.1,
            _AXES,
            "position_w_{c}_m",
            "target_position_w_{c}_m",
            "position_error_w_{c}_m",
        ),
        {"norm": "position_error_norm_m"},
    ),
    (
        "velocity.png",
        (
            "Velocity (world frame)",
            "m/s",
            0.1,
            _AXES,
            "velocity_w_{c}_m_s",
            "target_velocity_w_{c}_m_s",
            "velocity_error_w_{c}_m_s",
        ),
        {"norm": "velocity_error_norm_m_s"},
    ),
    (
        "acceleration.png",
        (
            "Acceleration (world frame)",
            "m/s²",
            0.5,
            _AXES,
            "acceleration_w_{c}_m_s2",
            "target_acceleration_w_{c}_m_s2",
            "acceleration_error_w_{c}_m_s2",
        ),
        {
            "norm": "acceleration_error_norm_m_s2",
            "actual_label": "Actual (velocity difference)",
            "extra": ("PhysX solver", "acceleration_native_w_{c}_m_s2", _SOLVER),
        },
    ),
    (
        "attitude.png",
        (
            "Attitude (Z-Y-X Euler angles)",
            "deg",
            2.0,
            _RPY,
            "attitude_{c}_rad",
            "target_attitude_{c}_rad",
            "attitude_error_{c}_rad",
        ),
        {"norm": "attitude_error_angle_rad", "norm_label": "rotation angle", "euclidean": False, "scale": _DEG},
    ),
    (
        "angular_velocity.png",
        (
            "Angular velocity (body frame)",
            "deg/s",
            5.0,
            _AXES,
            "angular_velocity_b_{c}_rad_s",
            "target_angular_velocity_b_{c}_rad_s",
            "angular_velocity_error_b_{c}_rad_s",
        ),
        {"norm": "angular_velocity_error_norm_rad_s", "scale": _DEG},
    ),
    (
        "motor_speed.png",
        (
            "Motor speed (native motor state, not encoder data)",
            "rpm",
            100.0,
            _ROTORS,
            "motor_speed_{c}_rpm",
            "motor_speed_command_{c}_rpm",
            "motor_speed_error_{c}_rpm",
        ),
        {
            "ideal_label": "Command",
            "labels": _ROTOR_LABELS,
            "colors": _ROTOR_COLORS,
            "extra": ("With wind (equivalent)", "rotor_speed_effective_{c}_rpm", _SOLVER),
        },
    ),
    (
        "rotor_thrust.png",
        (
            "Rotor thrust",
            "N",
            0.5,
            _ROTORS,
            "motor_thrust_applied_{c}_n",
            "motor_thrust_command_{c}_n",
            "motor_thrust_error_{c}_n",
        ),
        {"ideal_label": "Command", "labels": _ROTOR_LABELS, "colors": _ROTOR_COLORS},
    ),
    (
        "force.png",
        (
            "Force on vehicle (body frame)",
            "N",
            0.5,
            _AXES,
            "force_motor_b_{c}_n",
            "force_command_b_{c}_n",
            "force_error_b_{c}_n",
        ),
        {"norm": "force_error_norm_n", "actual_label": "Motor output", "ideal_label": "Command"},
    ),
    (
        "torque.png",
        (
            "Torque about CoM (body frame)",
            "N·m",
            0.02,
            _AXES,
            "torque_motor_b_{c}_nm",
            "torque_command_b_{c}_nm",
            "torque_error_b_{c}_nm",
        ),
        {"norm": "torque_error_norm_nm", "actual_label": "Motor output", "ideal_label": "Command"},
    ),
)


# (file name, gate column, positional and keyword arguments of _tracking) for each disturbance figure.
# A figure is drawn only when some run has a nonzero value in its gate column: blank means the model was off.
_DISTURBANCE_FIGURES = (
    (
        "wind.png",
        "wind_w_{c}_m_s",
        (
            "Wind velocity at CoM (world frame)",
            "m/s",
            0.5,
            _AXES,
            "wind_w_{c}_m_s",
            "wind_mean_w_{c}_m_s",
            "wind_turbulence_w_{c}_m_s",
        ),
        {
            "actual_label": "Wind",
            "ideal_label": "Mean + gusts",
            "error_label": "OU part",
            "norm_label": "|OU part|",
            "derived_norm": True,
        },
    ),
    (
        "rotor_wind.png",
        "rotor_speed_offset_{c}_rpm",
        (
            "Rotor speed with wind on the rotors (equivalent speed, axial channel)",
            "rpm",
            100.0,
            _ROTORS,
            "rotor_speed_effective_{c}_rpm",
            "motor_speed_{c}_rpm",
            "rotor_speed_offset_{c}_rpm",
        ),
        {
            "actual_label": "With wind (equivalent)",
            "ideal_label": "Motor speed",
            "error_label": "Wind offset",
            "labels": _ROTOR_LABELS,
            "colors": _ROTOR_COLORS,
        },
    ),
    (
        "wind_drag.png",
        "force_wind_drag_b_{c}_n",
        ("Rotor-disk in-plane wind drag at CoM (body frame)", "N", 0.2, _AXES, "force_wind_drag_b_{c}_n", None, None),
        {"actual_label": "In-plane drag"},
    ),
    (
        "ou_force.png",
        "force_ou_w_{c}_n",
        (
            "OU random force at CoM (world frame)",
            "N",
            0.5,
            _AXES,
            "force_ou_w_{c}_n",
            "force_ou_mean_w_{c}_n",
            "force_ou_part_w_{c}_n",
        ),
        {
            "actual_label": "OU force",
            "ideal_label": "Mean",
            "error_label": "OU part",
            "norm_label": "|OU part|",
            "derived_norm": True,
        },
    ),
    (
        "ou_torque.png",
        "torque_ou_b_{c}_nm",
        (
            "OU random torque (body frame)",
            "N·m",
            0.02,
            _AXES,
            "torque_ou_b_{c}_nm",
            "torque_ou_mean_b_{c}_nm",
            "torque_ou_part_b_{c}_nm",
        ),
        {
            "actual_label": "OU torque",
            "ideal_label": "Mean",
            "error_label": "OU part",
            "norm_label": "|OU part|",
            "derived_norm": True,
        },
    ),
)


def _load_run(run_dir):
    """basic.csv columns (with attitude angles) and the mission phase starts of one run."""
    run = Path(run_dir)
    data = load_basic_csv(run / "basic.csv")
    _add_attitude_angles(data)
    _add_ou_parts(data)
    config, metadata = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        for path in (run / "config.json", run / "metadata.json")
    )
    return data, phase_starts(config, metadata)


def plot_run(run_dir, output_dir=None) -> list[Path]:
    """Write the PNG set for run_dir into output_dir (default run_dir/plots).

    The subtitle is the run directory name and, below it, the physics settings
    (disturbances, model mismatch, ...), so a figure on its own says which run it shows.
    """
    from .catalog import settings_line

    run = Path(run_dir)
    data, phases = _load_run(run)
    output = run / "plots" if output_dir is None else Path(output_dir)
    config_path = run / "config.json"
    settings = settings_line(json.loads(config_path.read_text(encoding="utf-8"))) if config_path.is_file() else None
    settings = None if settings is None else wrap_settings(settings)
    subtitle = run.name if settings is None else f"{run.name}\n{settings}"
    return _write_figures([(run.name, data)], subtitle, phases, output)


def wrap_settings(settings: str, width: int = 110) -> str:
    """Break a physics-settings line after '; ' so that a long one stays inside the figure title."""
    lines = [""]
    for part in settings.split("; "):
        if lines[-1] and len(lines[-1]) + 2 + len(part) > width:
            lines[-1] += ";"
            lines.append(part)
        else:
            lines[-1] = f"{lines[-1]}; {part}" if lines[-1] else part
    return "\n".join(lines)


def plot_comparison(run_dirs, labels, output_dir) -> list[Path]:
    """The same PNG set with up to four runs overlaid, one figure per quantity.

    Each run's actual values get one colour. A target identical in every run
    (the trajectory) is drawn once in black; one that differs (a controller
    command) is drawn per run, dashed in that run's colour. Each run then gets
    its own error row, all error rows on one y scale. The PhysX solver
    acceleration is left out, and phase markers come from the first run.
    """
    if not 1 <= len(run_dirs) <= len(RUN_COLORS) or len(labels) != len(run_dirs):
        raise ValueError(f"Give 1 to {len(RUN_COLORS)} runs and one label per run")
    loaded = [_load_run(run) for run in run_dirs]
    runs = [(label, data) for label, (data, _) in zip(labels, loaded)]
    subtitle = "   vs   ".join(f"{label}: {Path(run).name}" for label, run in zip(labels, run_dirs))
    return _write_figures(runs, subtitle, loaded[0][1], Path(output_dir))


def _write_figures(runs, subtitle, phases, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "trajectory_3d.png"]
    _trajectory(runs, subtitle).savefig(paths[0], dpi=150)
    for name, arguments, options in _TRACKING_FIGURES:
        paths.append(output / name)
        _tracking(runs, subtitle, phases, *arguments, **options).savefig(paths[-1], dpi=150)
    for name, gate, arguments, options in _DISTURBANCE_FIGURES:
        if _recorded(runs, gate, arguments[3]):
            paths.append(output / name)
            _tracking(runs, subtitle, phases, *arguments, **options).savefig(paths[-1], dpi=150)
    return paths


def _recorded(runs, pattern, components):
    """True when some run has a nonzero value in one of these columns; blank or all-zero columns are skipped."""
    for _, data in runs:
        for component in components:
            column = data.get(pattern.format(c=component))
            if column is not None and np.any(np.isfinite(column) & (column != 0)):
                return True
    return False


def _add_ou_parts(data):
    """Random part (applied minus mean) of the OU force and torque, for their figures."""
    for kind, frame, unit in (("force", "w", "n"), ("torque", "b", "nm")):
        for axis in _AXES:
            applied = data.get(f"{kind}_ou_{frame}_{axis}_{unit}")
            mean = data.get(f"{kind}_ou_mean_{frame}_{axis}_{unit}")
            if applied is not None and mean is not None:
                data[f"{kind}_ou_part_{frame}_{axis}_{unit}"] = applied - mean


def _add_attitude_angles(data):
    """Roll/pitch/yaw from the quaternion, for runs recorded before these columns existed."""
    if "attitude_roll_rad" in data or "quaternion_wxyz_w" not in data:
        return
    w, x, y, z = (data[f"quaternion_wxyz_{c}"] for c in "wxyz")
    data["attitude_roll_rad"] = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    data["attitude_pitch_rad"] = np.arcsin(np.clip(2 * (w * y - x * z), -1.0, 1.0))
    data["attitude_yaw_rad"] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _time_figure(rows, title, subtitle):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 1.2 + 1.8 * rows), layout="constrained")
    axes = figure.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
    figure.suptitle(f"{title}\n{subtitle}", fontsize=11)
    axes[-1].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    return figure, axes


def _plot(ax, t, column, **style):
    """Plot a column only if it exists and has any finite sample."""
    if column is None or not np.isfinite(column).any():
        return False
    ax.plot(t, column, **style)
    return True


def _display(column, scale, component):
    """Scale for display; unwrap yaw so a heading through ±180° is not drawn as a jump."""
    if column is None:
        return None
    if component == "yaw" and np.isfinite(column).all():
        column = np.unwrap(column)
    return column * scale


def _tracking(
    runs,
    subtitle,
    phases,
    title,
    unit,
    min_span,
    components,
    actual,
    ideal,
    error,
    *,
    norm=None,
    norm_label="|error|",
    euclidean=True,
    scale=1.0,
    labels=None,
    colors=_AXIS_COLORS,
    actual_label="Actual",
    ideal_label="Target",
    extra=None,
    error_label="Error",
    derived_norm=False,
):
    """One row per component (actual and ideal), then one row of ideal-minus-actual errors per run.

    ``ideal`` or ``error`` None leaves out that line or the error rows. ``error_label``
    names the bottom rows when they hold another quantity (the random part of a
    disturbance); ``derived_norm`` adds the magnitude of those rows' components.
    """
    single = len(runs) == 1
    labels = labels or components
    figure, axes = _time_figure(len(components) + (len(runs) if error else 0), title, subtitle)
    ideals = [
        [data["sample_time_s"]] + [data.get(ideal.format(c=c)) if ideal else None for c in components]
        for _, data in runs
    ]
    shared = all(_same(ideals[0], other) for other in ideals[1:])
    for row, (ax, component, label) in enumerate(zip(axes, components, labels)):
        for index, (name, data) in enumerate(runs):
            t = data["sample_time_s"]
            style = _ACTUAL if single else {**_ACTUAL, "color": RUN_COLORS[index]}
            values = _display(data.get(actual.format(c=component)), scale, component)
            _plot(ax, t, values, label=actual_label if single else f"{name}: {actual_label}", **style)
            if extra is not None and single:
                _plot(
                    ax,
                    t,
                    _display(data.get(extra[1].format(c=component)), scale, component),
                    label=extra[0],
                    **extra[2],
                )
        for index, (name, _) in enumerate(runs[:1] if shared else runs):
            style = _IDEAL if shared else {**_IDEAL, "color": RUN_COLORS[index], "linewidth": 0.8, "alpha": 0.8}
            t, values = ideals[index][0], _display(ideals[index][row + 1], scale, component)
            _plot(ax, t, values, label=ideal_label if shared else f"{name}: {ideal_label}", **style)
        ax.set_ylabel(f"{label} [{unit}]")
    error_axes = axes[len(components) :]
    for ax, (name, data) in zip(error_axes, runs):
        _error_row(
            ax,
            data,
            components,
            labels,
            colors,
            actual,
            ideal,
            error,
            norm,
            norm_label,
            euclidean,
            scale,
            unit,
            derived_norm,
        )
        ax.set_ylabel(f"{error_label} [{unit}]" if single else f"{name} {error_label.lower()} [{unit}]")
    if not single:
        lows, highs = zip(*(ax.get_ylim() for ax in error_axes))
        for ax in error_axes:
            ax.set_ylim(min(lows), max(highs))
    end = max(np.nanmax(data["sample_time_s"]) for _, data in runs)
    _finish(axes, end, phases, min_span, error_axes)
    return figure


def _same(first, second):
    """True when two runs hold the same values in every column (or both lack it)."""
    return all(
        (a is None and b is None)
        or (a is not None and b is not None and a.shape == b.shape and np.allclose(a, b, rtol=0.0, equal_nan=True))
        for a, b in zip(first, second)
    )


def _error_row(
    ax, data, components, labels, colors, actual, ideal, error, norm, norm_label, euclidean, scale, unit, derived_norm
):
    """One run's ideal-minus-actual error per component, plus its magnitude with max and RMS."""
    t = data["sample_time_s"]
    errors = []
    for component, label, color in zip(components, labels, colors):
        difference = data.get(error.format(c=component))
        actual_values = data.get(actual.format(c=component))
        ideal_values = data.get(ideal.format(c=component)) if ideal else None
        if difference is None and actual_values is not None and ideal_values is not None:
            difference = ideal_values - actual_values  # runs recorded before error columns existed
        errors.append(difference)
        _plot(
            ax,
            t,
            None if difference is None else difference * scale,
            label=label.split(" ")[0],
            color=color,
            linewidth=1.0,
        )
    magnitude = data.get(norm) if norm else None
    if magnitude is None and (norm or derived_norm) and euclidean and all(e is not None for e in errors):
        magnitude = np.sqrt(sum(e**2 for e in errors))
    if magnitude is not None and np.isfinite(magnitude).any():
        finite = magnitude[np.isfinite(magnitude)] * scale
        _plot(
            ax,
            t,
            magnitude * scale,
            color="black",
            linewidth=0.9,
            label=f"{norm_label}  max {finite.max():.3g}, RMS {math.sqrt(np.mean(finite**2)):.3g} {unit}",
        )


def _finish(axes, end, phases, min_span, error_axes):
    """Phase markers, legends, and a minimum y span so numerical noise is not magnified."""
    for ax in axes:
        low, high = ax.get_ylim()
        if high - low < min_span:
            middle = (low + high) / 2
            ax.set_ylim(middle - min_span / 2, middle + min_span / 2)
        for start, _ in phases:
            if 0 < start < end:
                ax.axvline(start, color="0.5", linestyle=":", linewidth=0.8)
    # Phase names above the top row, centred in each phase, clear of the data.
    bounds = [start for start, _ in phases] + [end]
    for (start, name), stop in zip(phases, bounds[1:]):
        if start < end:
            axes[0].text(
                (start + min(stop, end)) / 2,
                1.01,
                name,
                transform=axes[0].get_xaxis_transform(),
                fontsize=8,
                color="0.35",
                ha="center",
                va="bottom",
            )
    for ax in (axes[0], *error_axes):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8, ncol=1 if ax is axes[0] else 2)


def _xyz(data, prefix):
    size = len(data["sample_time_s"])
    points = np.column_stack([data.get(f"{prefix}_{axis}_m", np.full(size, np.nan)) for axis in _AXES])
    return points[np.isfinite(points).all(axis=1)]


def _trajectory(runs, subtitle):
    from matplotlib.figure import Figure

    single = len(runs) == 1
    actuals = [_xyz(data, "position_w") for _, data in runs]
    targets = [_xyz(data, "target_position_w") for _, data in runs]
    shared = all(_same([targets[0]], [target]) for target in targets[1:])
    figure = Figure(figsize=(12, 5.5), layout="constrained")
    figure.suptitle(f"Centre-of-mass trajectory\n{subtitle}", fontsize=11)
    view = figure.add_subplot(1, 2, 1, projection="3d")
    top = figure.add_subplot(1, 2, 2)
    for ax, columns in ((view, slice(0, 3)), (top, slice(0, 2))):
        for index, target in enumerate(targets[:1] if shared else targets):
            if len(target):
                style = _IDEAL if shared else {**_IDEAL, "color": RUN_COLORS[index], "linewidth": 0.8, "alpha": 0.8}
                ax.plot(*target[:, columns].T, label="Target" if shared else f"{runs[index][0]}: Target", **style)
        for index, ((name, _), actual) in enumerate(zip(runs, actuals)):
            style = _ACTUAL if single else {**_ACTUAL, "color": RUN_COLORS[index]}
            ax.plot(*actual[:, columns].T, label="Actual" if single else name, **style)
    view.scatter(*actuals[0][0, :, None], color="tab:green", s=25, label="Start")
    for index, ((name, _), actual) in enumerate(zip(runs, actuals)):
        if single:
            view.scatter(*actual[-1, :, None], color="tab:red", s=25, label="End")
        else:
            view.scatter(*actual[-1, :, None], color=RUN_COLORS[index], edgecolors="black", s=30, label=f"{name}: End")
    # Equal scale on all three axes, so the helix shape is not distorted.
    points = np.vstack(actuals + targets)
    low, high = points.min(axis=0), points.max(axis=0)
    centre, half = (low + high) / 2, max((high - low).max() / 2, 0.5)
    for setter, middle in zip((view.set_xlim, view.set_ylim, view.set_zlim), centre):
        setter(middle - half, middle + half)
    view.set_box_aspect((1, 1, 1))
    view.set_xlabel("x [m]")
    view.set_ylabel("y [m]")
    view.set_zlabel("z [m]")
    view.legend(loc="upper left", fontsize=8)
    top.set_aspect("equal", adjustable="datalim")
    top.set_title("Top view (XY)", fontsize=10)
    top.set_xlabel("x [m]")
    top.set_ylabel("y [m]")
    top.grid(True, alpha=0.3)
    top.legend(loc="best", fontsize=8)
    return figure
