"""CPU temperature and free disk space. A missing sensor gives None (an empty cell)."""
from __future__ import annotations

import shutil
from pathlib import Path


def cpu_temp_c(path: Path = Path("/sys/class/thermal/thermal_zone0/temp")) -> float | None:
    try:
        return int(path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def disk_free_gb(folder: Path) -> float | None:
    folder = Path(folder)
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    try:
        return shutil.disk_usage(folder).free / 1e9
    except OSError:
        return None
