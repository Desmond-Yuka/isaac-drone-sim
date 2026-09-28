"""Motion references independent of low-level control and physics.

Importing this package registers the built-in trajectories in ``TRAJECTORIES``.
"""

from .base import HOLD_PHASE, TRAJECTORIES, Segment, SegmentedTrajectory
from .completion import CompletionMonitor, CompletionParams
from .helix import HelixParams, HelixTrajectory
from .hold import HoldParams, HoldTrajectory


def build_trajectory(section) -> SegmentedTrajectory:
    """Construct the trajectory selected by a ``trajectory: {kind: ..., ...}`` config section."""
    return TRAJECTORIES.build(section, "trajectory")


__all__ = [
    "HOLD_PHASE",
    "TRAJECTORIES",
    "CompletionMonitor",
    "CompletionParams",
    "HelixParams",
    "HelixTrajectory",
    "HoldParams",
    "HoldTrajectory",
    "Segment",
    "SegmentedTrajectory",
    "build_trajectory",
]
