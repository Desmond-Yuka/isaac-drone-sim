"""Backend-independent experiment runner: step the loop, record, and notify hooks.

One run = ``duration_s / dt`` physics steps of ``MotionControlLoop`` against a
``SimulationDriver``. Every run writes the same run directory (config, metadata,
telemetry.jsonl, basic.csv, schema) whatever the backend. Display, video, plots
and metrics attach as hooks, so the stepping logic is written exactly once.
"""
from __future__ import annotations

import math
import platform
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from isaac_drone.telemetry.recorder import RunRecorder

REPO_ROOT = Path(__file__).resolve().parents[2]


def physics_step_count(duration_s: float, dt_s: float) -> int:
    """Steps covering ``duration_s``; exact multiples never gain an extra step from round-off."""
    count = duration_s / dt_s
    nearest = round(count)
    if math.isclose(count, nearest, rel_tol=1e-12, abs_tol=1e-12):
        return max(1, nearest)
    return max(1, math.ceil(count))


def provenance() -> dict:
    """Code version and interpreter of this run; unknown fields stay None."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                                  timeout=5, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
            "python": sys.version.split()[0], "platform": platform.platform(), "argv": list(sys.argv)}


def resolve_log_root(config: dict) -> Path:
    directory = Path(config["logging"]["directory"]).expanduser()
    return directory if directory.is_absolute() else REPO_ROOT / directory


class RunHook:
    """Base class for run observers. ``name`` (if set) keys the hook's summary in end events."""

    name: str | None = None

    def start(self, run: RunContext) -> None:
        """Called inside the run directory's lifetime, before the first physics step."""

    def after_step(self, run: RunContext, record: dict) -> None:
        """Called after every completed physics step with the full step record."""

    def paused(self, run: RunContext) -> None:
        """Called for every idle iteration while the simulation is paused."""

    def summary(self) -> dict | None:
        return None

    def close(self) -> None:
        """Release resources; called once (reverse start order), even after errors."""

    def finalize(self, result: RunResult) -> None:
        """Called after the run directory is closed (e.g. figures); must not raise."""


@dataclass
class RunContext:
    loop: object
    driver: object
    config: dict
    run_dir: Path
    steps: int

    @property
    def end_time_s(self) -> float:
        return self.steps * self.loop.dt_s


@dataclass
class RunResult:
    success: bool
    run_dir: Path
    physics_steps: int
    requested_steps: int
    simulated_time_s: float
    mission: dict | None
    summaries: dict = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 0 if self.success else 2


def _summaries(hooks) -> dict:
    result = {}
    for hook in hooks:
        if hook.name:
            try:
                result[hook.name] = hook.summary()
            except Exception as error:  # a failing summary must not mask the run outcome
                result[hook.name] = {"summary_error": str(error)}
    return result


def run_experiment(loop, driver, config: dict, *, metadata: dict, hooks=(), log_root: Path | None = None,
                   log=print) -> RunResult:
    """Fly one reset ``loop`` for the configured duration and record it.

    Success requires every requested step and, if the trajectory defines one,
    an achieved completion criterion. Records are written every
    ``logging.every_n_steps`` steps and always for the final step. An exception
    (including Ctrl+C) writes an ``aborted`` event, closes hooks and propagates.
    """
    hooks = list(hooks)
    steps = physics_step_count(config["simulation"]["duration_s"], loop.dt_s)
    every = config["logging"]["every_n_steps"]
    metadata = {**metadata, "provenance": provenance()}
    with RunRecorder(log_root or resolve_log_root(config), config, metadata) as recorder:
        log(f"Run records: {recorder.path}")
        context = RunContext(loop, driver, config, recorder.path, steps)
        try:
            with ExitStack() as resources:
                for hook in hooks:
                    hook.start(context)
                    resources.callback(hook.close)
                while driver.is_running() and loop.step_index < steps:
                    if driver.is_paused():
                        driver.idle()
                        for hook in hooks:
                            hook.paused(context)
                        continue
                    loop.prepare_step()
                    driver.step()
                    record = loop.finish_step()
                    if record["step"] % every == 0 or loop.step_index == steps:
                        recorder.write(record)
                    for hook in hooks:
                        hook.after_step(context, record)
            mission = loop.mission_status
            success = loop.step_index == steps and (mission is None or bool(mission["achieved"]))
            summaries = _summaries(hooks)
            recorder.write({"event": "finished", "physics_steps": loop.step_index,
                            "simulated_time_s": loop.time_s, "requested_steps": steps,
                            "success": success, "mission": mission, **summaries})
        except BaseException as error:
            recorder.write({"event": "aborted", "time_s": loop.time_s, "error": str(error), **_summaries(hooks)})
            raise
    result = RunResult(success, recorder.path, loop.step_index, steps, loop.time_s, mission, summaries)
    for hook in hooks:
        hook.finalize(result)
    return result
