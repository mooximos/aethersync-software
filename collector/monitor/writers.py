"""Writing the data files (data format section 2).

- One file per kind per local clock hour, in data/YYYY-MM-DD/.
- Every row is flushed at once; os.fsync runs every 10 s and when a file closes,
  so a power cut loses at most about 10 s.
- An hour's files close 30 s after the hour ends (vehicle rows are written up to
  a few seconds after their t_left). Closed tracks files are gzipped in the background.
- On (re)opening a file, a cut-off last line from a power cut is removed first.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, IO
from zoneinfo import ZoneInfo

from common import schema
from common.timeutil import format_epoch, local_day_and_hour

GRACE_S = 30.0


def repair_csv(path: Path) -> bool:
    """Cut a half-written last line (no newline at the end). Returns True if it changed the file."""
    with open(path, "rb+") as handle:
        data = handle.read()
        if not data or data.endswith(b"\n"):
            return False
        handle.truncate(data.rfind(b"\n") + 1)
        handle.flush()
        os.fsync(handle.fileno())
    return True


def gzip_file(plain: Path) -> Path:
    """tracks_HH.csv -> tracks_HH.csv.gz, safely: the plain file is removed only
    after the complete .gz is on disk."""
    packed = plain.with_name(plain.name + ".gz")
    tmp = plain.with_name(plain.name + ".gz.tmp")
    with open(plain, "rb") as source, open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as target:
            shutil.copyfileobj(source, target, 1 << 20)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(tmp, packed)
    plain.unlink()
    return packed


def gunzip_file(packed: Path) -> Path:
    """Undo gzip_file (when an hour's tracks file must be appended to again)."""
    plain = packed.with_name(packed.name[:-3])
    with gzip.open(packed, "rb") as source, open(plain, "wb") as target:
        shutil.copyfileobj(source, target, 1 << 20)
        target.flush()
        os.fsync(target.fileno())
    packed.unlink()
    return plain


@dataclass
class _Open:
    handle: IO[str]
    writer: "csv._writer"
    path: Path


class DataWriter:
    def __init__(self, data_dir: Path, zone: ZoneInfo, site_json: Callable[[], dict],
                 fsync_every_s: float = 10.0, grace_s: float = GRACE_S,
                 on_log: Callable[[str], None] | None = None) -> None:
        self.data_dir, self.zone, self.site_json = Path(data_dir), zone, site_json
        self.fsync_every_s, self.grace_s = fsync_every_s, grace_s
        self.on_log = on_log or (lambda line: None)
        self._open: dict[tuple[str, str, int], _Open] = {}
        self._days: set[str] = set()
        self._lock = threading.RLock()
        self._last_fsync = time.monotonic()
        self._gzipping: dict[Path, threading.Thread] = {}

    # ---- day folders -------------------------------------------------------
    def day_dir(self, epoch_s: float) -> Path:
        """The day folder for this time; created (with a copy of site.json) on first use."""
        day, _hour = local_day_and_hour(epoch_s, self.zone)
        folder = self.data_dir / day
        with self._lock:
            if day not in self._days:
                folder.mkdir(parents=True, exist_ok=True)
                if not (folder / "site.json").exists():
                    self._write_json(folder / "site.json", self.site_json())
                self._days.add(day)
        return folder

    def site_changed(self, epoch_s: float) -> Path:
        """Settings changed during the day: save a copy as site_HHMMSS.json."""
        stamp = datetime.fromtimestamp(epoch_s, self.zone).strftime("%H%M%S")
        path = self.day_dir(epoch_s) / f"site_{stamp}.json"
        self._write_json(path, self.site_json())
        return path

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def log_event(self, epoch_s: float, text: str) -> None:
        """One line in events.log: startup, shutdown, error or mode switch."""
        line = f"{format_epoch(epoch_s, self.zone)} {text}"
        with self._lock:
            with open(self.day_dir(epoch_s) / "events.log", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        self.on_log(line)

    def clock_sync_row(self, epoch_s: float, values: list) -> None:
        path = self.day_dir(epoch_s) / "clock_sync.csv"
        with self._lock:
            new = not path.exists() or path.stat().st_size == 0
            with open(path, "a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                if new:
                    writer.writerow(schema.CLOCK_SYNC_COLUMNS)
                writer.writerow(values)
                handle.flush()
                os.fsync(handle.fileno())

    # ---- hourly files ------------------------------------------------------
    def write(self, kind: str, epoch_s: float, values: list) -> None:
        """Write one row to the hour file of its main time (epoch_s)."""
        day, hour = local_day_and_hour(epoch_s, self.zone)
        with self._lock:
            entry = self._open.get((kind, day, hour)) or self._open_file(kind, epoch_s, day, hour)
            entry.writer.writerow(values)
            entry.handle.flush()
            if time.monotonic() - self._last_fsync >= self.fsync_every_s:
                self.fsync_all()

    def _open_file(self, kind: str, epoch_s: float, day: str, hour: int) -> _Open:
        path = self.day_dir(epoch_s) / f"{kind}_{hour:02d}.csv"
        if path in self._gzipping:   # a late row for an hour that is being gzipped right now
            self._gzipping[path].join()
        packed = path.with_name(path.name + ".gz")
        if packed.exists():
            if path.exists():
                packed.unlink()       # a crash between gzip and delete: the plain file is complete
            else:
                gunzip_file(packed)   # restarted within the same hour: keep appending to one file
        if path.exists() and repair_csv(path):
            self.on_log(f"repaired a cut-off last line in {path}")
        new = not path.exists() or path.stat().st_size == 0
        handle = open(path, "a", encoding="utf-8", newline="")
        writer = csv.writer(handle, lineterminator="\n")
        if new:
            writer.writerow(schema.COLUMNS[kind])
        entry = _Open(handle, writer, path)
        self._open[(kind, day, hour)] = entry
        return entry

    def fsync_all(self) -> None:
        with self._lock:
            for entry in self._open.values():
                entry.handle.flush()
                os.fsync(entry.handle.fileno())
            self._last_fsync = time.monotonic()

    def tick(self, now_epoch: float) -> None:
        """Call about once a second: regular fsync, and close hours that are over."""
        current = local_day_and_hour(now_epoch - self.grace_s, self.zone)
        with self._lock:
            for key in [k for k in self._open if (k[1], k[2]) != current]:
                self._close(key)
            if time.monotonic() - self._last_fsync >= self.fsync_every_s:
                self.fsync_all()

    def _close(self, key: tuple[str, str, int]) -> None:
        entry = self._open.pop(key)
        entry.handle.flush()
        os.fsync(entry.handle.fileno())
        entry.handle.close()
        if key[0] == "tracks":
            thread = threading.Thread(target=self._gzip_quietly, args=(entry.path,), daemon=False)
            self._gzipping[entry.path] = thread
            thread.start()

    def _gzip_quietly(self, path: Path) -> None:
        try:
            gzip_file(path)
        except OSError as error:
            self.on_log(f"could not gzip {path}: {error}")
        finally:
            self._gzipping.pop(path, None)

    def recover(self, now_epoch: float) -> list[str]:
        """At startup: fix what a power cut may have left in the two newest day folders."""
        notes = []
        if not self.data_dir.exists():
            return notes
        current = local_day_and_hour(now_epoch, self.zone)
        days = sorted(p for p in self.data_dir.iterdir() if p.is_dir() and len(p.name) == 10)[-2:]
        for folder in days:
            for tmp in folder.glob("*.gz.tmp"):
                tmp.unlink()
                notes.append(f"removed half-written {tmp.name}")
            for path in sorted(folder.glob("*.csv")):
                if repair_csv(path):
                    notes.append(f"repaired a cut-off last line in {folder.name}/{path.name}")
            for path in sorted(folder.glob("tracks_??.csv")):
                if (folder.name, int(path.name[7:9])) != current:
                    packed = path.with_name(path.name + ".gz")
                    if packed.exists():
                        packed.unlink()
                    gzip_file(path)
                    notes.append(f"gzipped {folder.name}/{path.name} left from last time")
        return notes

    def close(self) -> None:
        """Flush, fsync and close everything; gzip the tracks files; wait for that to finish."""
        with self._lock:
            for key in list(self._open):
                self._close(key)
        for thread in list(self._gzipping.values()):
            thread.join()
