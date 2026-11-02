"""Run directory names: start time, experiment name, backend, and a video marker.

``1014-2007_helix_isaac_video`` = started 10-14 20:07 on the run clock, from
``configs/helix.yaml`` (``logging.run_name``), Isaac Lab backend, video
recorded. The run clock is one fixed time zone wherever the run happens (the
GPU server's own clock is UTC+8); the exact UTC start time is kept in
``metadata.json`` under ``provenance.started_utc``. Physics settings are not
part of the name; ``python -m isaac_drone runs`` and each run's README.md
list them.
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

RUN_TIMEZONE = "America/New_York"
# Copies of the composite videos of all runs, one file per run: <runs>/videos/<run name>.mp4
VIDEO_DIR = "videos"
BACKEND_LABELS = {"isaaclab": "isaac", "synthetic": "synth"}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def run_clock(moment: datetime | None = None) -> datetime:
    """``moment`` (default: now) in RUN_TIMEZONE; in UTC if the time zone database is missing."""
    moment = moment or datetime.now(timezone.utc)
    try:
        return moment.astimezone(ZoneInfo(RUN_TIMEZONE))
    except ZoneInfoNotFoundError:
        return moment.astimezone(timezone.utc)


def stamp(moment: datetime | None = None) -> str:
    """``MMDD-HHMM`` on the run clock, e.g. ``1014-2007`` (a ``Z`` suffix marks a UTC fallback)."""
    local = run_clock(moment)
    return local.strftime("%m%d-%H%M") + ("Z" if local.tzinfo is timezone.utc else "")


def safe_name(text) -> str:
    """Letters, digits, ``.``, ``_`` and ``-`` only, so the name works unquoted in a shell."""
    return _UNSAFE.sub("-", str(text)).strip("-.") or "run"


def run_directory_name(config: dict, metadata: dict, moment: datetime | None = None) -> str:
    """``<MMDD-HHMM>_<experiment>[_<isaac|synth>][_video]`` for a run starting at ``moment``."""
    experiment = (config.get("logging") or {}).get("run_name") or (config.get("trajectory") or {}).get("kind")
    parts = [stamp(moment), safe_name(experiment or "run")]
    backend = BACKEND_LABELS.get(metadata.get("backend_kind"))
    if backend:
        parts.append(backend)
    if (config.get("recording") or {}).get("enabled"):
        parts.append("video")
    return "_".join(parts)


def _candidates(root: Path, name: str):
    yield root / name
    for index in itertools.count(2):
        yield root / f"{name}_{index}"


def available_path(root, name: str) -> Path:
    """The first of ``root/name``, ``root/name_2``, ... that does not exist yet (nothing is created)."""
    return next(path for path in _candidates(Path(root).expanduser(), name) if not path.exists())


def create_unique_directory(root, name: str) -> Path:
    """Create the first free ``root/name``, ``root/name_2``, ...; exclusive creation is safe across processes."""
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for path in _candidates(root, name):
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path


__all__ = [
    "BACKEND_LABELS",
    "RUN_TIMEZONE",
    "VIDEO_DIR",
    "available_path",
    "create_unique_directory",
    "run_clock",
    "run_directory_name",
    "safe_name",
    "stamp",
]
