"""State-estimation boundary of the control loop: modes, estimator kinds and the estimator contract.

``estimation.mode`` selects what the controller sees:

- ``truth``: the backend truth, as before this package existed (default);
- ``shadow``: the estimator runs on the sensor readings and is recorded, the
  controller still flies on truth (debugging);
- ``closed_loop``: the controller flies on the estimate.

The estimator algorithms themselves are deliberately left open. The planned
stages (attitude from IMU + magnetometer with Mahony then a multiplicative EKF;
altitude with the barometer in a linear Kalman filter; a 15-state error-state
EKF with GPS, error convention R = R̂·Exp(δθ), delayed GPS through a state
buffer) will register further ``ESTIMATOR_KINDS`` implementing ``Estimator``.
Until then the only kind is ``passthrough``, which returns the truth and
verifies that the plumbing (sensor sampling, update/read ordering, recording)
does not alter a flight.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from isaac_drone.core.types import VehicleState

ESTIMATION_MODES = ("truth", "shadow", "closed_loop")
ESTIMATOR_KINDS = ("passthrough",)


class Estimator(Protocol):
    """A filter fed once per completed physics step; queried for the state at the loop time.

    ``update`` receives the ``SensorSuite`` readings of the step just finished
    (each entry a reading mapping or None) and must only use them. ``state``
    is query-only and must return the estimate at exactly ``time_s``; the
    controller requires strictly increasing time stamps. ``diagnostics`` returns
    JSON-ready values for the record: at least ``covariance_diag`` (list or None)
    and ``nis`` (mapping or None).
    """

    kind: str

    def reset(self) -> None: ...

    def update(self, readings: Mapping | None) -> None: ...

    def state(self, time_s: float) -> VehicleState: ...

    def diagnostics(self) -> dict: ...
