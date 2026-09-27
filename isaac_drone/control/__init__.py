"""Simulation-independent motion control and constrained thrust allocation."""

from .allocation import AllocationResult, BoundedAllocator
from .geometric import GeometricController

__all__ = ["AllocationResult", "BoundedAllocator", "GeometricController"]
