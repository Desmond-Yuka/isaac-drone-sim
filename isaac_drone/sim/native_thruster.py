"""NumPy transcription of Isaac Lab's native ``Thruster`` motor integrators.

Upstream: ``IsaacLab/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py``
(``motor_model_rate``, ``rk4_integration``, ``discrete_mixing_factor``,
``continuous_mixing_factor``). The synthetic backend drives these through the
same ``NativeRpsActuator`` adapter as the Isaac Lab backend, so both share one
RPS integration path. Contract tests compare this transcription with the
upstream source whenever an Isaac Lab checkout is available.

The functions are written as unbound methods: ``self`` must provide
``cfg.dt`` [s] and ``max_rate`` (elementwise rate clamp).
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import types

import numpy as np

UPSTREAM_RELATIVE_PATH = Path("source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py")
METHOD_NAMES = ("motor_model_rate", "rk4_integration", "discrete_mixing_factor", "continuous_mixing_factor")


def motor_model_rate(self, error, mixing):
    return np.clip(mixing * error, -self.max_rate, self.max_rate)


def rk4_integration(self, error, mixing):
    dt = self.cfg.dt
    k1 = self.motor_model_rate(error, mixing)
    k2 = self.motor_model_rate(error - 0.5 * dt * k1, mixing)
    k3 = self.motor_model_rate(error - 0.5 * dt * k2, mixing)
    k4 = self.motor_model_rate(error - dt * k3, mixing)
    return dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def discrete_mixing_factor(self, time_constant):
    return 1.0 / (self.cfg.dt + time_constant)


def continuous_mixing_factor(self, time_constant):
    return 1.0 / time_constant


TRANSCRIBED_METHODS = {name: globals()[name] for name in METHOD_NAMES}


def find_upstream_source() -> Path | None:
    """Locate thruster.py via $ISAACLAB_PATH or an ``IsaacLab`` checkout beside/inside the repo."""
    repo = Path(__file__).resolve().parents[2]
    roots = [Path(os.environ["ISAACLAB_PATH"])] if os.environ.get("ISAACLAB_PATH") else []
    roots += [repo / "IsaacLab", repo.parent / "IsaacLab", Path.home() / "IsaacLab"]
    for root in roots:
        candidate = root.expanduser() / UPSTREAM_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    return None


def load_upstream_methods(source: Path) -> dict:
    """Compile only the four integration methods from the upstream source with a NumPy ``torch`` shim."""
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Thruster")
    body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in METHOD_NAMES]
    namespace = {"torch": types.SimpleNamespace(Tensor=np.ndarray, clamp=np.clip)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    missing = set(METHOD_NAMES) - set(namespace)
    if missing:
        raise ValueError(f"Upstream Thruster lacks {sorted(missing)}")
    return {name: namespace[name] for name in METHOD_NAMES}
