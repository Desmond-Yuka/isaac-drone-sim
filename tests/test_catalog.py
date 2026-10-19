"""Run cards (README.md), the ``runs`` table and the figure subtitle, all read back from run files."""

import json
import shutil

import pytest

from isaac_drone.analysis import plots
from isaac_drone.analysis.catalog import CARD_FILE, describe_run, format_table, list_runs, run_card, settings_line
from isaac_drone.config import DEFAULT_CONFIG, HELIX_CONFIG, load_config
from isaac_drone.sim.synthetic import run_simulation

IDEAL = ["effects.ou_wrench.enabled=false", "model_mismatch.mass_ratio=1.0"]
PUSH = (
    "effects.disturbances=[{enabled: true, model: constant_wrench, frame: world, force_n: [1.5, 0, 0], "
    "torque_nm: [0, 0, 0], start_time_s: 10.0, end_time_s: null, application_point_b_m: null}]"
)


def fly(root, *overrides, config=DEFAULT_CONFIG, **options):
    cfg = load_config(config, overrides=["simulation.duration_s=0.05", "completion=null", *overrides])
    return run_simulation(cfg, figures=False, log=lambda *_: None, log_root=root, **options).run_dir


def test_settings_line_lists_only_departures_from_an_ideal_plant():
    assert settings_line(load_config(DEFAULT_CONFIG)) == "OU force (1,0,0)±0.5 N τ 1 s; mass ×1.2"
    assert settings_line(load_config(DEFAULT_CONFIG, overrides=IDEAL)) == "ideal (no disturbance, no model mismatch)"
    config = load_config(
        DEFAULT_CONFIG,
        overrides=[
            "effects.ou_wrench.force_mean_w_n=[0, 0, 0]",
            "effects.ou_wrench.force_std_w_n=[1, 1, 0]",
            "effects.ou_wrench.torque_std_b_nm=[0.01, 0.01, 0.01]",
            PUSH,
            "model_mismatch.mass_ratio=1.0",
            "model_mismatch.inertia_ratio=[1.1, 1.0, 1.0]",
            "model_mismatch.thrust_coefficient_ratio=0.9",
            "vehicle.rotor_inertia_kg_m2=3.5e-5",
            "simulation.seed=3",
        ],
    )
    assert settings_line(config) == (
        "OU force 0±(1,1,0) N τ 1 s; OU torque 0±0.01 N·m τ 1 s; const force (1.5,0,0) N from 10 s; "
        "inertia ×(1.1,1,1); thrust coeff ×0.9; gyro J_r 3.5e-05; seed 3"
    )
    assert settings_line({"schema_version": 1, "effects": {}}) is None


def test_every_run_writes_a_card_with_command_settings_and_metrics(tmp_path):
    run = fly(tmp_path, config=HELIX_CONFIG)
    card = (run / CARD_FILE).read_text(encoding="utf-8")
    assert card.startswith(f"# {run.name}\n") and card == run_card(run)
    assert "| Config | helix |" in card and "| Backend | synthetic |" in card
    assert "| Controller / trajectory | geometric / helix |" in card
    assert "| Video | - |" in card
    assert "| OU random force (world frame, at CoM) | mean (1, 0, 0) N, SD 0.5 N, τ 1 s, from 2 s to the end |" in card
    mismatch = "mass ×1.2, inertia ×1, thrust coeff ×1, reaction torque ×1"
    assert f"| Model mismatch (plant / controller) | {mismatch} |" in card
    assert "| Rotor gyroscopic torque | off |" in card and "| Seed | 0 |" in card
    assert "## Metrics" in card and "| Position error RMS [m] |" in card
    assert "## Figures" not in card  # flown without figures


def test_card_links_the_figures_and_both_video_locations(tmp_path):
    run = fly(tmp_path)
    (run / "plots").mkdir()
    for name in ("velocity.png", "trajectory_3d.png", "position.png"):
        (run / "plots" / name).write_bytes(b"png")
    events = (run / "telemetry.jsonl").read_text().splitlines()
    end = json.loads(events[-1])
    end["video"] = {
        "failed": False,
        "frames": 60,
        "fps": 60.0,
        "camera_order": ["overview", "top"],
        "videos": {name: {"filename": f"{name}.mp4"} for name in ("overview", "top", "combined")},
    }
    (run / "telemetry.jsonl").write_text("\n".join([*events[:-1], json.dumps(end)]) + "\n")
    (run / "combined.mp4").write_bytes(b"mp4")
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / f"{run.name}.mp4").write_bytes(b"mp4")
    card = run_card(run)
    figures = [line for line in card.splitlines() if line.startswith("![")]
    assert figures == [
        "![trajectory_3d](plots/trajectory_3d.png)",
        "![position](plots/position.png)",
        "![velocity](plots/velocity.png)",
    ]
    assert "overview, top, combined: 60 frames at 60 fps; files: [combined.mp4](combined.mp4)" in card
    assert f"copy: [videos/{run.name}.mp4](../videos/{run.name}.mp4)" in card
    assert describe_run(run)["video"] == "2 cams, 60 frames"


def old_style_run(tmp_path, source):
    """A copy of ``source`` as the pre-2026-10-19 runs looked: <UTC>Z_<hex> name, no start time or run_name."""
    old = tmp_path / "20261015T000727Z_9c7f0de9"
    shutil.copytree(source, old)
    metadata = json.loads((old / "metadata.json").read_text())
    del metadata["provenance"]["started_utc"]
    metadata["provenance"]["argv"] = ["/repo/isaac_drone/__main__.py", "run", "--config", "configs/helix_moderate.yaml"]
    (old / "metadata.json").write_text(json.dumps(metadata))
    config = json.loads((old / "config.json").read_text())
    del config["logging"]["run_name"]
    (old / "config.json").write_text(json.dumps(config))
    return old


def test_runs_table_describes_new_and_old_runs_oldest_first(tmp_path):
    ideal = fly(tmp_path, *IDEAL)
    old = old_style_run(tmp_path, ideal)
    (tmp_path / "videos").mkdir()
    (tmp_path / "sweep_1014-2021").mkdir()
    rows = list_runs(tmp_path)
    assert [row["name"] for row in rows] == [old.name, ideal.name]
    assert rows[0]["started"] == "10-14 20:07" and rows[0]["config"] == "helix_moderate"
    assert rows[1] == {
        **rows[1],
        "config": "arl_robot_1",
        "backend": "synth",
        "video": "-",
        "settings": "ideal (no disturbance, no model mismatch)",
        "result": "ok",
    }
    lines = format_table(rows).splitlines()
    assert lines[0].split()[:4] == ["Started", "Config", "Backend", "Video"] and lines[0].endswith("Directory")
    assert len(lines) == 3 and lines[2].endswith(ideal.name)


def test_runs_without_a_final_event_are_incomplete(tmp_path):
    run = fly(tmp_path)
    lines = (run / "telemetry.jsonl").read_text().splitlines()
    (run / "telemetry.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    assert describe_run(run)["result"] == "incomplete"


def test_cli_lists_filters_and_writes_cards(tmp_path, capsys):
    from isaac_drone.cli import main

    first = fly(tmp_path)
    second = fly(tmp_path, config=HELIX_CONFIG)
    (first / CARD_FILE).unlink()
    assert main(["runs", "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3 and out[1].endswith(first.name) and out[2].endswith(second.name)
    assert main(["runs", "--runs-dir", str(tmp_path), "--config", "configs/helix.yaml"]) == 0
    filtered = capsys.readouterr().out.splitlines()
    assert len(filtered) == 2 and filtered[1].endswith(second.name)
    assert main(["runs", "--runs-dir", str(tmp_path), "--last", "1", "--write-cards"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2 and not (first / CARD_FILE).exists()
    assert main(["runs", "--runs-dir", str(tmp_path), "--write-cards"]) == 0
    assert (first / CARD_FILE).is_file()
    capsys.readouterr()
    assert main(["runs", "--runs-dir", str(tmp_path), "--video"]) == 0
    assert capsys.readouterr().out == "No matching runs\n"
    with pytest.raises(SystemExit):
        main(["runs", "--runs-dir", str(tmp_path), "--last", "0"])


def test_figure_subtitle_names_the_run_and_its_physics_settings(tmp_path, monkeypatch):
    run = fly(tmp_path, config=HELIX_CONFIG)
    subtitles = []
    monkeypatch.setattr(plots, "_write_figures", lambda runs, subtitle, phases, output: subtitles.append(subtitle))
    plots.plot_run(run)
    (run / "config.json").unlink()
    plots.plot_run(run)
    assert subtitles == [f"{run.name}\nOU force (1,0,0)±0.5 N τ 1 s; mass ×1.2", run.name]
