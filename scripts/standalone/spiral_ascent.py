#!/usr/bin/env python3
"""Backward-compatible entry point for the three-dimensional helix mission."""
import sys
from run_arl import main
from isaac_drone.config import HELIX_CONFIG


def helix_main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--config" or arg.startswith("--config=") for arg in args):
        args[:0] = ["--config", str(HELIX_CONFIG)]
    return main([*args, "--trajectory", "helix"])


if __name__ == "__main__":
    raise SystemExit(helix_main())
