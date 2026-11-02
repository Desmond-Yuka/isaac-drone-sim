"""State estimation for the control loop: modes, the estimator contract and the state-provider factory.

Only the ``passthrough`` provider exists so far; the filters (attitude, altitude,
15-state error-state EKF) are the next stage and will register here.
"""

from __future__ import annotations

from collections.abc import Mapping

from isaac_drone.core.validation import ConfigurationError

from .base import ESTIMATION_MODES, ESTIMATOR_KINDS, Estimator
from .errors import attitude_errors, control_errors, estimation_errors
from .passthrough import TruthPassthrough

DEFAULT_ESTIMATION = {"mode": "truth", "kind": "passthrough"}


def build_state_provider(section: Mapping, backend):
    """The ``StateProvider`` for ``estimation.kind``; the backend is needed by ``passthrough`` only."""
    kind = section["kind"]
    if kind == "passthrough":
        return TruthPassthrough(backend)
    raise ConfigurationError(f"estimation.kind {kind!r} is not implemented; available: {list(ESTIMATOR_KINDS)}")


__all__ = [
    "DEFAULT_ESTIMATION",
    "ESTIMATION_MODES",
    "ESTIMATOR_KINDS",
    "Estimator",
    "TruthPassthrough",
    "attitude_errors",
    "build_state_provider",
    "control_errors",
    "estimation_errors",
]
