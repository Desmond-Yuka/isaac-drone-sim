"""Run metrics, multi-run comparison and synthetic parameter sweeps on short real runs."""

import json

import pytest

from isaac_drone.analysis.compare import compare_runs, markdown_table
from isaac_drone.analysis.metrics import METRIC_KEYS, compute_metrics, flatten
from isaac_drone.analysis.summary import summarize_run
from isaac_drone.analysis.sweep import grid_points, parse_grid, run_sweep
from isaac_drone.config import HELIX_CONFIG, ConfigurationError, load_config
from isaac_drone.sim.synthetic import run_simulation


def fly(tmp_path, *overrides, config=None):
    cfg = load_config(config or load_config.__defaults__[0], overrides=["simulation.duration_s=0.3", *overrides])
    return run_simulation(cfg, figures=False, log=lambda *_: None, log_root=tmp_path)


def test_every_run_writes_metrics_with_all_keys_per_phase(tmp_path):
    result = fly(tmp_path)
    stored = json.loads((result.run_dir / "metrics.json").read_text())
    assert stored == compute_metrics(result.run_dir)
    assert stored["run"]["backend"] == "synthetic" and stored["run"]["controller"] == "geometric"
    assert set(stored["phases"]) == {"hold"}
    for values in (stored["overall"], stored["phases"]["hold"]):
        assert tuple(values) == METRIC_KEYS
    assert stored["overall"]["duration_s"] == pytest.approx(0.3)
    row = flatten(stored)
    assert row["hold.position_error_rms_m"] == stored["phases"]["hold"]["position_error_rms_m"]
    assert "hold" in summarize_run(result.run_dir)


def test_ground_launch_metrics_split_spin_up_from_the_trajectory(tmp_path):
    # A run shorter than the mission must drop the completion criterion explicitly.
    result = fly(tmp_path, "simulation.duration_s=2.2", "completion=null", config=HELIX_CONFIG)
    metrics = compute_metrics(result.run_dir)
    assert list(metrics["phases"]) == ["spin_up", "takeoff"]
    assert metrics["phases"]["spin_up"]["duration_s"] == pytest.approx(2.0)
    assert result.success and metrics["run"]["completion_time_s"] is None


def test_compare_writes_table_csv_and_rejects_unknown_phase(tmp_path):
    first = fly(tmp_path / "a").run_dir
    second = fly(tmp_path / "b", "controller.position_kp=[2.0, 2.0, 2.0]").run_dir
    report = compare_runs([first, second], tmp_path / "out", labels=["base", "soft"], figures=False)
    assert [row["label"] for row in report["rows"]] == ["base", "soft"]
    assert "| base |" in report["table"] and (tmp_path / "out" / "comparison.csv").is_file()
    assert markdown_table(report["rows"]).count("\n") == 3
    with pytest.raises(ValueError, match="no phase"):
        compare_runs([first], tmp_path / "out2", phase="helix", figures=False)
    with pytest.raises(ValueError, match="one label per run"):
        compare_runs([first, second], tmp_path / "out3", labels=["only"], figures=False)


def test_compare_figures(tmp_path):
    pytest.importorskip("matplotlib")
    run = fly(tmp_path).run_dir
    files = compare_runs([run, run], tmp_path / "out", labels=["x", "y"])["files"]
    assert all(path.is_file() for path in files) and any(path.suffix == ".png" for path in files)


def test_grid_parsing_and_cartesian_product():
    grid = parse_grid(["controller.position_kp=[[1,1,1],[2,2,2]]", "trajectory.yaw_rad=[0.0, 0.5, null]"])
    points = grid_points(grid)
    assert len(points) == 6 and points[0] == {"controller.position_kp": [1, 1, 1], "trajectory.yaw_rad": 0.0}
    for bad in (["x=1"], ["x=[]"], ["novalue"], ["a=[1]", "a=[2]"]):
        with pytest.raises(ConfigurationError):
            parse_grid(bad)


def test_sweep_validates_every_point_before_flying(tmp_path):
    with pytest.raises(ConfigurationError):
        run_sweep(
            load_config.__defaults__[0],
            parse_grid(["controller.position_kp=[[1,1,1],[1,1]]"]),
            output_dir=tmp_path / "bad",
            log=lambda *_: None,
        )
    assert not (tmp_path / "bad").exists()


def test_sweep_runs_points_in_workers_and_ranks_them(tmp_path):
    grid = parse_grid(["controller.position_kp=[[1.0,1.0,1.0],[4.0,4.0,2.5]]"])
    report = run_sweep(
        load_config.__defaults__[0],
        grid,
        base_overrides=["simulation.duration_s=0.3"],
        output_dir=tmp_path / "sweep",
        workers=2,
        log=lambda *_: None,
    )
    rows = report["rows"]
    assert len(rows) == 2 and all(row["error"] is None and row["success"] for row in rows)
    assert rows[0]["position_error_rms_m"] <= rows[1]["position_error_rms_m"]
    for name in ("summary.csv", "summary.json", "summary.md"):
        assert (tmp_path / "sweep" / name).is_file()
    assert len(list((tmp_path / "sweep" / "points").glob("*/metrics.json"))) == 2
