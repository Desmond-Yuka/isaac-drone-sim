"""``python -m isaac_drone``: see isaac_drone.cli."""
from isaac_drone.cli import main

if __name__ == "__main__":  # worker processes (spawn) re-import this module as __mp_main__
    raise SystemExit(main())
