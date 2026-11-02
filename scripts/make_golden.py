"""Regenerate tests/data/golden_synthetic.npz after a deliberate change of flight behaviour.

Run from the repository root: ``python scripts/make_golden.py``. It flies the same
two missions as tests/test_golden_regression.py (whose ``fly`` it reuses) and
overwrites the file; say so in the commit.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_golden_regression import fly  # noqa: E402

MISSIONS = {"helix": (4000, 10), "hold": (1000, 5)}  # physics steps, row stride


def main():
    arrays = {}
    for name, (steps, stride) in MISSIONS.items():
        rows, errors, record = fly(name, steps, stride)
        arrays.update({f"{name}_{key}": value for key, value in rows.items()})
        arrays[f"{name}_stride"] = np.array(stride)
        arrays[f"{name}_max_error"] = np.array(errors.max())
        arrays[f"{name}_rms_error"] = np.array(np.sqrt(np.mean(errors**2)))
        mission = record["mission"]
        print(f"{name}: max error {errors.max():.4f} m, rms {arrays[f'{name}_rms_error']:.4f} m, mission {mission}")
    path = ROOT / "tests" / "data" / "golden_synthetic.npz"
    np.savez(path, **arrays)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
