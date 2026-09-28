"""Simulation-independent motion control and constrained thrust allocation.

Importing this package registers the built-in controllers in ``CONTROLLERS``.
"""

from .allocation import AllocationResult, BoundedAllocator
from .base import CONTROLLERS, Controller, ControllerDiagnostics, FlightLimits, VehicleModel, build_controller
from .geometric import GeometricController, GeometricGains

__all__ = ["CONTROLLERS", "AllocationResult", "BoundedAllocator", "Controller", "ControllerDiagnostics",
           "FlightLimits", "GeometricController", "GeometricGains", "VehicleModel", "build_controller"]
