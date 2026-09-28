"""English PNG figures for one finished run, drawn from its basic.csv.

basic.csv and telemetry.jsonl stay the data of record; figures are a derived
view and never write back to them. Each tracked quantity gets one figure: a
row per component with the ideal (target or command) and the actual value,
then a row with the ideal-minus-actual error. matplotlib is optional and
imported only when drawing. The object API (no pyplot) needs no GUI backend,
so this also runs headless and inside the Isaac Sim process.
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


def plot_run(run_dir, output_dir=None) -> list[Path]:
    """Write the PNG set for run_dir into output_dir (default run_dir/plots)."""
    run = Path(run_dir)
    data = load_basic_csv(run / "basic.csv")
    _add_attitude_angles(data)
    config, metadata = (json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                        for path in (run / "config.json", run / "metadata.json"))
    context = (data, data["sample_time_s"], run.name, phase_starts(config, metadata))
    output = run / "plots" if output_dir is None else Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures = (
        ("trajectory_3d.png", lambda: _trajectory(data, run.name)),
        ("position.png", lambda: _tracking(
            *context, "Position (world frame)", "m", 0.1, _AXES,
            "position_w_{c}_m", "target_position_w_{c}_m", "position_error_w_{c}_m", norm="position_error_norm_m")),
        ("velocity.png", lambda: _tracking(
            *context, "Velocity (world frame)", "m/s", 0.1, _AXES,
            "velocity_w_{c}_m_s", "target_velocity_w_{c}_m_s", "velocity_error_w_{c}_m_s",
            norm="velocity_error_norm_m_s")),
        ("acceleration.png", lambda: _tracking(
            *context, "Acceleration (world frame)", "m/s²", 0.5, _AXES,
            "acceleration_w_{c}_m_s2", "target_acceleration_w_{c}_m_s2", "acceleration_error_w_{c}_m_s2",
            norm="acceleration_error_norm_m_s2", actual_label="Actual (velocity difference)",
            extra=("PhysX solver", "acceleration_native_w_{c}_m_s2", _SOLVER))),
        ("attitude.png", lambda: _tracking(
            *context, "Attitude (Z-Y-X Euler angles)", "deg", 2.0, _RPY,
            "attitude_{c}_rad", "target_attitude_{c}_rad", "attitude_error_{c}_rad",
            norm="attitude_error_angle_rad", norm_label="rotation angle", euclidean=False, scale=_DEG)),
        ("angular_velocity.png", lambda: _tracking(
            *context, "Angular velocity (body frame)", "deg/s", 5.0, _AXES,
            "angular_velocity_b_{c}_rad_s", "target_angular_velocity_b_{c}_rad_s",
            "angular_velocity_error_b_{c}_rad_s", norm="angular_velocity_error_norm_rad_s", scale=_DEG)),
        ("motor_speed.png", lambda: _tracking(
            *context, "Motor speed (native motor state, not encoder data)", "rpm", 100.0, _ROTORS,
            "motor_speed_{c}_rpm", "motor_speed_command_{c}_rpm", "motor_speed_error_{c}_rpm",
            ideal_label="Command", labels=_ROTOR_LABELS, colors=_ROTOR_COLORS)),
        ("rotor_thrust.png", lambda: _tracking(
            *context, "Rotor thrust", "N", 0.5, _ROTORS,
            "motor_thrust_applied_{c}_n", "motor_thrust_command_{c}_n", "motor_thrust_error_{c}_n",
            ideal_label="Command", labels=_ROTOR_LABELS, colors=_ROTOR_COLORS)),
        ("force.png", lambda: _tracking(
            *context, "Force on vehicle (body frame)", "N", 0.5, _AXES,
            "force_motor_b_{c}_n", "force_command_b_{c}_n", "force_error_b_{c}_n",
            norm="force_error_norm_n", actual_label="Motor output", ideal_label="Command")),
        ("torque.png", lambda: _tracking(
            *context, "Torque about CoM (body frame)", "N·m", 0.02, _AXES,
            "torque_motor_b_{c}_nm", "torque_command_b_{c}_nm", "torque_error_b_{c}_nm",
            norm="torque_error_norm_nm", actual_label="Motor output", ideal_label="Command")),
    )
    paths = []
    for name, draw in figures:
        path = output / name
        draw().savefig(path, dpi=150)
        paths.append(path)
    return paths


def _add_attitude_angles(data):
    """Roll/pitch/yaw from the quaternion, for runs recorded before these columns existed."""
    if "attitude_roll_rad" in data or "quaternion_wxyz_w" not in data:
        return
    w, x, y, z = (data[f"quaternion_wxyz_{c}"] for c in "wxyz")
    data["attitude_roll_rad"] = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    data["attitude_pitch_rad"] = np.arcsin(np.clip(2*(w*y - x*z), -1.0, 1.0))
    data["attitude_yaw_rad"] = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def _time_figure(rows, title, run_name):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 1.2 + 1.8*rows), layout="constrained")
    axes = figure.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
    figure.suptitle(f"{title}\n{run_name}", fontsize=11)
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


def _tracking(data, t, run_name, phases, title, unit, min_span, components, actual, ideal, error, *,
              norm=None, norm_label="|error|", euclidean=True, scale=1.0, labels=None, colors=_AXIS_COLORS,
              actual_label="Actual", ideal_label="Target", extra=None):
    """One row per component (actual and ideal), then one row of ideal-minus-actual errors."""
    figure, axes = _time_figure(len(components) + 1, title, run_name)
    errors = []
    for ax, component, label, color in zip(axes, components, labels or components, colors):
        actual_values = data.get(actual.format(c=component))
        ideal_values = data.get(ideal.format(c=component))
        _plot(ax, t, _display(actual_values, scale, component), label=actual_label, **_ACTUAL)
        if extra is not None:
            _plot(ax, t, _display(data.get(extra[1].format(c=component)), scale, component),
                  label=extra[0], **extra[2])
        _plot(ax, t, _display(ideal_values, scale, component), label=ideal_label, **_IDEAL)
        ax.set_ylabel(f"{label} [{unit}]")
        difference = data.get(error.format(c=component))
        if difference is None and actual_values is not None and ideal_values is not None:
            difference = ideal_values - actual_values  # runs recorded before error columns existed
        errors.append(difference)
        _plot(axes[-1], t, None if difference is None else difference*scale,
              label=label.split(" ")[0], color=color, linewidth=1.0)
    magnitude = data.get(norm) if norm else None
    if magnitude is None and norm and euclidean and all(e is not None for e in errors):
        magnitude = np.sqrt(sum(e**2 for e in errors))
    if magnitude is not None and np.isfinite(magnitude).any():
        finite = magnitude[np.isfinite(magnitude)] * scale
        _plot(axes[-1], t, magnitude*scale, color="black", linewidth=0.9,
              label=f"{norm_label}  max {finite.max():.3g}, RMS {math.sqrt(np.mean(finite**2)):.3g} {unit}")
    axes[-1].set_ylabel(f"Error [{unit}]")
    _finish(axes, t, phases, min_span)
    return figure


def _finish(axes, t, phases, min_span):
    """Phase markers, legends, and a minimum y span so numerical noise is not magnified."""
    end = np.nanmax(t)
    for ax in axes:
        low, high = ax.get_ylim()
        if high - low < min_span:
            middle = (low + high) / 2
            ax.set_ylim(middle - min_span/2, middle + min_span/2)
        for start, _ in phases:
            if 0 < start < end:
                ax.axvline(start, color="0.5", linestyle=":", linewidth=0.8)
    # Phase names above the top row, centred in each phase, clear of the data.
    bounds = [start for start, _ in phases] + [end]
    for (start, name), stop in zip(phases, bounds[1:]):
        if start < end:
            axes[0].text((start + min(stop, end)) / 2, 1.01, name, transform=axes[0].get_xaxis_transform(),
                         fontsize=8, color="0.35", ha="center", va="bottom")
    for ax in (axes[0], axes[-1]):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8, ncol=2 if ax is axes[-1] else 1)


def _xyz(data, prefix):
    size = len(data["sample_time_s"])
    points = np.column_stack([data.get(f"{prefix}_{axis}_m", np.full(size, np.nan)) for axis in _AXES])
    return points[np.isfinite(points).all(axis=1)]


def _trajectory(data, run_name):
    from matplotlib.figure import Figure

    actual, target = _xyz(data, "position_w"), _xyz(data, "target_position_w")
    figure = Figure(figsize=(12, 5.5), layout="constrained")
    figure.suptitle(f"Centre-of-mass trajectory\n{run_name}", fontsize=11)
    view = figure.add_subplot(1, 2, 1, projection="3d")
    top = figure.add_subplot(1, 2, 2)
    for ax, columns in ((view, slice(0, 3)), (top, slice(0, 2))):
        if len(target):
            ax.plot(*target[:, columns].T, label="Target", **_IDEAL)
        ax.plot(*actual[:, columns].T, label="Actual", **_ACTUAL)
    view.scatter(*actual[0, :, None], color="tab:green", s=25, label="Start")
    view.scatter(*actual[-1, :, None], color="tab:red", s=25, label="End")
    # Equal scale on all three axes, so the helix shape is not distorted.
    points = np.vstack([actual, target])
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
