"""The ``passthrough`` kind: a state provider that returns the backend truth.

It is not an estimator. It exists so that every mode and recording path can be
exercised and checked against the truth-mode flight before a real filter is
plugged in: ``closed_loop`` with ``passthrough`` must reproduce truth mode exactly.
"""

from __future__ import annotations

from collections.abc import Mapping

from isaac_drone.core.types import VehicleState


class TruthPassthrough:
    kind = "passthrough"

    def __init__(self, backend):
        self.backend = backend
        self.updates = 0
        self.last_readings = None

    def reset(self) -> None:
        self.updates = 0
        self.last_readings = None

    def update(self, readings: Mapping | None) -> None:
        self.updates += 1
        self.last_readings = readings

    def read_state(self, time_s: float) -> VehicleState:
        return self.backend.read_state(time_s)

    def diagnostics(self) -> dict:
        return {"covariance_diag": None, "nis": None}
