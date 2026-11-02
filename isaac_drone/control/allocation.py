"""Bounded four-rotor wrench allocation, with no SciPy dependency."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from isaac_drone.core.types import Wrench
from isaac_drone.core.validation import finite_array, finite_scalar


@dataclass(frozen=True)
class AllocationResult:
    """``residual = requested - achieved`` in SI units about the body COM.

    ``saturated`` means a bound is active, including a failed motor fixed at
    zero. Residual distinguishes exact commands on a bound from lost authority.
    """

    thrusts_n: np.ndarray
    achieved_wrench: Wrench
    residual: np.ndarray
    saturated: bool


class BoundedAllocator:
    """Exact box-constrained weighted least squares for four rotors.

    ``matrix`` maps rotor thrusts [N] to COM body wrench [N, N m]. Minimize
    ``||diag(weights) (matrix @ thrust - desired)||² + regularization *
    ||thrust - previous||²``. Weights multiply residuals, so their squares are
    cost coefficients; use them for unit normalization and priority weighting.
    No symmetry, identical geometry, or full controllability is presumed.

    At most 3**4 active sets are enumerated, avoiding pseudoinverse-then-clip
    artifacts. Equal zero bounds handle failed rotors. The regularization
    reference defaults to zero when ``previous_n`` is omitted.
    """

    def __init__(self, matrix: np.ndarray, weights: np.ndarray, regularization: float):
        self.matrix = finite_array(matrix, (6, 4), "allocation matrix")
        self.weights = finite_array(weights, (6,), "allocation weights")
        self.regularization = finite_scalar(regularization, "allocation regularization")
        if np.any(self.weights < 0) or not np.any(self.weights > 0):
            raise ValueError("allocation weights must be nonnegative with at least one positive entry")
        if self.regularization < 0:
            raise ValueError("allocation regularization must be nonnegative")
        self.matrix.setflags(write=False)
        self.weights.setflags(write=False)

    def allocate(
        self,
        wrench: Wrench,
        lower_n: np.ndarray,
        upper_n: np.ndarray,
        previous_n: np.ndarray | None = None,
    ) -> AllocationResult:
        desired = finite_array(wrench.vector, (6,), "desired wrench")
        lower = finite_array(lower_n, (4,), "lower_n")
        upper = finite_array(upper_n, (4,), "upper_n")
        if np.any(lower < 0) or np.any(lower > upper):
            raise ValueError("thrust bounds must satisfy 0 <= lower_n <= upper_n")
        previous = np.zeros(4) if previous_n is None else finite_array(previous_n, (4,), "previous_n")
        if np.any(previous < 0):
            raise ValueError("previous_n must be nonnegative")
        system = self.weights[:, None] * self.matrix
        target = self.weights * desired
        if self.regularization > 0:
            root_regularization = np.sqrt(self.regularization)
            system = np.vstack((system, root_regularization * np.eye(4)))
            target = np.concatenate((target, root_regularization * previous))
        tolerance = 1e-10 * max(1.0, float(np.max(upper)))
        best: np.ndarray | None = None
        best_cost = np.inf
        choices = [(-1,) if lower[i] == upper[i] else (-1, 0, 1) for i in range(4)]
        for statuses in product(*choices):
            status = np.asarray(statuses)
            free = status == 0
            candidate = np.where(status == 1, upper, lower)
            if np.any(free):
                rhs = target - system[:, ~free] @ candidate[~free]
                candidate[free] = np.linalg.lstsq(system[:, free], rhs, rcond=None)[0]
            if not np.all(np.isfinite(candidate)):
                continue
            if np.any(candidate < lower - tolerance) or np.any(candidate > upper + tolerance):
                continue
            # Only floating-point feasibility tolerance is removed here.
            candidate = np.minimum(np.maximum(candidate, lower), upper)
            error = system @ candidate - target
            cost = float(error @ error)
            if cost < best_cost:
                best, best_cost = candidate, cost
        if best is None:
            raise FloatingPointError("allocation failed: no finite feasible active-set solution")
        achieved = self.matrix @ best
        residual = desired - achieved
        saturated = bool(np.any(best <= lower + tolerance) or np.any(best >= upper - tolerance))
        return AllocationResult(best, Wrench(achieved[:3], achieved[3:]), residual, saturated)
