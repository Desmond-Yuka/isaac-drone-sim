"""ARL-Robot-1 motion control with explicit SI and CoM contracts.

Pluggable trajectories (``isaac_drone.trajectories``), controllers
(``isaac_drone.control``) and physics backends (``isaac_drone.sim``) run in one
control loop (``isaac_drone.runtime``); ``python -m isaac_drone`` is the entry
point. Importing this package does not start Isaac Sim or import Torch/Warp.
See docs/configuration.md and docs/architecture.md.
"""

__version__ = "0.2.0"
