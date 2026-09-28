"""Numeric validation shared by every module: reject rather than repair bad values."""
from __future__ import annotations

from numbers import Real

import numpy as np


def finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return an owned finite float64 array with exactly ``shape``."""
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array of shape {shape}") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}; got {result.shape}")
    return result


def finite_scalar(value: object, name: str) -> float:
    """Validate a real scalar; booleans and strings are not numerical configuration."""
    if isinstance(value, (bool, np.bool_, str, bytes)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def numeric_array(value, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    """Like finite_array, but never coerce booleans or strings to numbers; ``shape=None`` accepts any."""
    def numeric(item):
        if isinstance(item, np.ndarray):
            return item.dtype.kind in "fiu"
        if isinstance(item, (list, tuple)):
            return all(numeric(child) for child in item)
        return isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
    if not numeric(value):
        raise ValueError(f"{name} must contain real numbers, not booleans/strings")
    if shape is not None:
        return finite_array(value, shape, name)
    result = np.array(value, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def positive_dt(dt_s: float) -> float:
    value = finite_scalar(dt_s, "dt_s")
    if value <= 0:
        raise ValueError("dt_s must be finite and positive")
    return value
