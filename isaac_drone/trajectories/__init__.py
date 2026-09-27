"""Motion objectives independent of low-level control and physics."""
from .hold import HoldTrajectory
from .spiral import SpiralTrajectory

HelixTrajectory = SpiralTrajectory


def build_trajectory(config):
    if config["kind"] == "hold":
        return HoldTrajectory(**config["hold"])
    if config["kind"] in ("spiral", "helix"):
        return SpiralTrajectory(config["spiral"])
    raise ValueError(f"Unknown trajectory kind: {config['kind']}")


__all__ = ["HoldTrajectory", "SpiralTrajectory", "HelixTrajectory", "build_trajectory"]
