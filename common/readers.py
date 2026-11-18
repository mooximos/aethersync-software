"""Find and read the data files (plain CSV or .csv.gz) without any extra packages."""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
from pathlib import Path
from typing import Iterator

from common.timeutil import DAY_PATTERN

_HOURLY_NAME = re.compile(r"^(vehicles|tracks|signal_events|system)_(\d{2})\.csv(\.gz)?$")
HOURLY_KINDS = ("vehicles", "tracks", "signal_events", "system")


def kind_and_hour(path: Path) -> tuple[str | None, int | None]:
    """'tracks_07.csv.gz' -> ('tracks', 7); 'clock_sync.csv' -> ('clock_sync', None)."""
    name = Path(path).name
    if name == "clock_sync.csv":
        return "clock_sync", None
    match = _HOURLY_NAME.match(name)
    if not match or int(match.group(2)) > 23:
        return None, None
    return match.group(1), int(match.group(2))


def open_text(path: Path) -> io.TextIOBase:
    """Open a data file for reading as text, whether or not it is gzipped."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def read_header_and_rows(path: Path) -> tuple[list[str], list[dict]]:
    """Read a whole CSV file. Every value stays text; missing values are ''."""
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_rows(path: Path) -> list[dict]:
    return read_header_and_rows(path)[1]


def iter_rows(path: Path) -> Iterator[dict]:
    """Read a CSV file one row at a time (for the big tracks files)."""
    with open_text(path) as handle:
        yield from csv.DictReader(handle)


def _inner(day_dir: Path) -> Path:
    """Windows 'Extract All' of 2026-12-05.zip makes 2026-12-05/2026-12-05/...: use the inner folder."""
    inner = day_dir / day_dir.name
    if not (day_dir / "site.json").exists() and inner.is_dir():
        return inner
    return day_dir


def list_day_folders(data_dir: Path) -> list[Path]:
    """All folders named YYYY-MM-DD inside data_dir (or data_dir itself if it is one), oldest first."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} is not a folder")
    if DAY_PATTERN.match(data_dir.name):
        return [_inner(data_dir)]
    return sorted((_inner(p) for p in data_dir.iterdir() if p.is_dir() and DAY_PATTERN.match(p.name)),
                  key=lambda p: p.name)


def zips_not_unzipped(data_dir: Path) -> list[str]:
    """Names of downloaded day zips in data_dir that have no unzipped day folder next to them."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    return sorted(p.name for p in data_dir.glob("*.zip")
                  if DAY_PATTERN.match(p.stem) and not (data_dir / p.stem).is_dir())


def hour_files(day_dir: Path, kind: str) -> list[tuple[int, Path]]:
    """All files of one kind in a day folder as (hour, path), in hour order.

    If both tracks_07.csv and tracks_07.csv.gz exist, the plain file wins
    (it is the one still being written)."""
    found: dict[int, Path] = {}
    for path in sorted(Path(day_dir).iterdir()):
        file_kind, hour = kind_and_hour(path)
        if file_kind == kind and hour is not None:
            if hour not in found or path.suffix != ".gz":
                found[hour] = path
    return sorted(found.items())


def read_day(day_dir: Path, kind: str) -> list[dict]:
    """All rows of one kind for a whole day, in hour order."""
    rows: list[dict] = []
    for _hour, path in hour_files(day_dir, kind):
        rows.extend(read_rows(path))
    return rows


def read_site_dict(day_dir: Path) -> dict:
    """The raw site.json of a day folder (not validated; see common.site.load_site)."""
    with open(Path(day_dir) / "site.json", "r", encoding="utf-8") as handle:
        return json.load(handle)
