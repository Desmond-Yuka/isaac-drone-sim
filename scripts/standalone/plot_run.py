#!/usr/bin/env python3
"""Draw English PNG figures for one run (default: the newest run under runs/).

Reads basic.csv and config.json and writes <run>/plots/*.png. Needs NumPy and
matplotlib only, not Isaac Sim, so it also works on a run directory copied to
another machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from isaac_drone.plots import latest_run, plot_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, help="Run directory (default: newest under runs/)")
    parser.add_argument("--output", type=Path, help="Output directory (default: <run>/plots)")
    args = parser.parse_args(argv)
    for path in plot_run(args.run or latest_run(ROOT / "runs"), args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
