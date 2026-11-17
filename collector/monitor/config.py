"""Paths, fixed settings and site.json loading/saving for the collector."""
from __future__ import annotations

import json
import os
from pathlib import Path

from common.site import Site, SiteError, load_site, validate_site  # noqa: F401  (re-exported)

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SITE_PATH = REPO_DIR / "collector" / "site.json"
DEFAULT_DATA_DIR = REPO_DIR / "data"
WEB_PORT = 8080

LOST_TRACK_S = 2.0          # a track not seen for this long is finished (ByteTrack lost_track_buffer)
PRESENT_AT_START_S = 2.0    # a track first seen this soon after a camera starts was already there
TRACK_TICK_S = 0.5          # tracks file: one row per vehicle per 0.5 s
TRACK_TICK_WINDOW_S = 0.15  # ...using the nearest frame within 0.15 s of the tick
DETECTOR_HOLD_S = 1.0       # a vehicle missed by the detector for less than this stays "in the zone"
GREEN_MIN_GAP_S = 20.0      # at most one inferred green per approach per 20 s
WEB_IDLE_S = 600            # no sleep while the web UI or SSH was used in the last 10 minutes
STAY_AWAKE_S = 3600


def site_dict(site: Site) -> dict:
    return site.model_dump(mode="json")


def save_site(path: Path, data: dict) -> Site:
    """Check and then save site.json safely (write a temporary file, then rename)."""
    path = Path(path)
    site = validate_site(data, source=str(path))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return site
