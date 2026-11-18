"""Read the day folders into tables (pandas), checking every file against common/schema.

All times become epoch seconds (float, columns ending in _s) for arithmetic; the
original text stays in the original columns, so every number can be traced back
to a raw row. vehicle_id restarts every day, so rows are keyed by (day, vehicle_id).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.api.types import union_categoricals

from common import readers, schema
from common.site import SiteError, validate_site

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
VEHICLE_TIMES = ["t_entered", "t_stopped", "t_started", "t_crossed", "t_left"]


@dataclass
class Dataset:
    vehicles: pd.DataFrame
    tracks: pd.DataFrame
    signals: pd.DataFrame
    system: pd.DataFrame
    clock_sync: pd.DataFrame
    sites: dict[str, dict]
    zone: str
    problems: list[str] = field(default_factory=list)
    study_days: list[str] | None = None    # set by study.prepare from options.days and skip_dates

    @property
    def days(self) -> list[str]:
        return sorted(self.sites)

    @property
    def used_days(self) -> list[str]:
        return self.days if self.study_days is None else self.study_days


def to_epoch(text: pd.Series) -> pd.Series:
    """'2026-11-16T07:14:52.3-08:00' -> epoch seconds (NaN where empty)."""
    parsed = pd.to_datetime(text.where(text != ""), format="%Y-%m-%dT%H:%M:%S.%f%z", utc=True)
    seconds = (parsed - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    return seconds.round(1)


def local_times(epoch: pd.Series, zone: str) -> pd.Series:
    return pd.to_datetime(epoch, unit="s", utc=True).dt.tz_convert(zone)


def _read_kind(day_dir: Path, kind: str, problems: list[str]) -> pd.DataFrame:
    frames = []
    for _hour, path in readers.hour_files(day_dir, kind):
        found = schema.validate_file(path)
        if found:
            problems += [f"{day_dir.name}/{p}" for p in found]
        try:
            header, rows = readers.read_header_and_rows(path)
        except (OSError, EOFError, UnicodeDecodeError) as error:
            problems.append(f"{day_dir.name}/{path.name}: skipped, cannot be read ({error})")
            continue
        if schema.check_header(kind, header):
            problems.append(f"{day_dir.name}/{path.name}: skipped, wrong columns")
            continue
        if found:   # keep only the rows that pass the checks
            check = schema.ROW_CHECKS[kind]
            good = [r for r in rows if None not in r.values() and not check(r)]
            problems.append(f"{day_dir.name}/{path.name}: kept {len(good)} of {len(rows)} rows")
            rows = good
        frame = pd.DataFrame(rows, columns=schema.COLUMNS[kind], dtype=str)
        frame["file"] = f"{day_dir.name}/{path.name}"
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=schema.COLUMNS[kind] + ["file"], dtype=str)
    return pd.concat(frames, ignore_index=True)


def _numbers(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column].where(frame[column] != ""), errors="coerce")


def _compact_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    """tracks rows as numbers: key (day/vehicle_id), time_s, x_m, y_m, speed_mps, file, day."""
    out = pd.DataFrame({"key": (frame["day"] + "/" + frame["vehicle_id"]).astype("category"),
                        "time_s": to_epoch(frame["time"])})
    for column in ("x_m", "y_m", "speed_mps"):
        out[column] = pd.to_numeric(frame[column].where(frame[column] != ""), errors="coerce").astype("float32")
    out["file"] = frame["file"].astype("category")
    out["day"] = frame["day"].astype("category")
    return out


def _join_tracks(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _compact_tracks(pd.DataFrame(columns=schema.COLUMNS["tracks"] + ["file", "day"], dtype=str))
    categorical = ("key", "file", "day")
    out = pd.concat([f.drop(columns=list(categorical)) for f in frames], ignore_index=True)
    for column in categorical:   # joined without ever turning back into millions of strings
        out[column] = union_categoricals([f[column] for f in frames])
    return out


def load(data_dirs: list[Path], with_tracks: bool = True) -> Dataset:
    """Every day folder in data_dirs (a day folder itself, or a folder of day folders)."""
    problems: list[str] = []
    found: dict[str, Path] = {}
    for data_dir in data_dirs:
        for name in readers.zips_not_unzipped(Path(data_dir)):
            problems.append(f"{name} in {data_dir} is still zipped: unzip it there to use that day")
        for day_dir in readers.list_day_folders(Path(data_dir)):
            if day_dir.name in found and found[day_dir.name].resolve() != day_dir.resolve():
                problems.append(f"day {day_dir.name} is in two places ({found[day_dir.name]} and {day_dir}); "
                                f"only the first is used, so no vehicle is counted twice")
                continue
            found.setdefault(day_dir.name, day_dir)
    day_dirs = [found[name] for name in sorted(found)]
    if not day_dirs:
        hint = "; ".join(problems) or "each day must be a folder named YYYY-MM-DD"
        raise FileNotFoundError(f"no day folders (YYYY-MM-DD) found in {', '.join(map(str, data_dirs))} ({hint})")
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in ("vehicles", "tracks", "signal_events", "system")}
    sites, syncs = {}, []
    for day_dir in day_dirs:
        try:
            sites[day_dir.name] = validate_site(json.loads((day_dir / "site.json").read_text()), str(day_dir / "site.json")).model_dump(mode="json")
        except (OSError, json.JSONDecodeError, SiteError) as error:
            problems.append(f"{day_dir.name}: skipped the whole day, site.json problem: {error}")
            continue
        for kind in parts:
            if kind == "tracks" and not with_tracks:
                continue
            frame = _read_kind(day_dir, kind, problems)
            frame["day"] = day_dir.name
            if kind == "tracks":   # millions of rows: keep numbers only, so weeks of data fit in memory
                frame = _compact_tracks(frame)
            parts[kind].append(frame)
        if (day_dir / "clock_sync.csv").exists():
            frame = pd.DataFrame(readers.read_rows(day_dir / "clock_sync.csv"), dtype=str)
            frame["day"] = day_dir.name
            syncs.append(frame)
    if not sites:
        raise FileNotFoundError("no usable day folders: " + "; ".join(problems))
    zones = {site["time_zone"] for site in sites.values()}
    if len(zones) > 1:
        problems.append(f"different time zones in site.json files: {sorted(zones)}; using the first")
    zone = sorted(zones)[0]

    def joined(kind: str) -> pd.DataFrame:
        frames = parts[kind]
        if not frames:
            return pd.DataFrame(columns=schema.COLUMNS[kind] + ["file", "day"])
        return pd.concat(frames, ignore_index=True)

    vehicles = joined("vehicles")
    for column in VEHICLE_TIMES:
        vehicles[column + "_s"] = to_epoch(vehicles[column])
    _numbers(vehicles, ["lane", "conf", "wait_s", "stops", "queue_pos"])
    vehicles["lane"] = vehicles["lane"].astype("Int64")
    vehicles["key"] = vehicles["day"] + "/" + vehicles["vehicle_id"]
    vehicles["weekday"] = vehicles["day"].map(lambda d: DAY_NAMES[date.fromisoformat(d).weekday()])
    flags = vehicles["flags"].fillna("")
    for flag in schema.FLAGS:
        vehicles["flag_" + flag] = flags.str.split(";").map(lambda items, f=flag: f in items)

    tracks = _join_tracks(parts["tracks"])

    signals = joined("signal_events")
    signals["time_s"] = to_epoch(signals["time"])
    _numbers(signals, ["event_code", "lane"])
    signals["lane"] = signals["lane"].astype("Int64")
    signals["event_code"] = signals["event_code"].astype("Int64")

    system = joined("system")
    system["time_s"] = to_epoch(system["time"])
    _numbers(system, schema.SYSTEM_COLUMNS[1:])

    clock_sync = pd.concat(syncs, ignore_index=True) if syncs else pd.DataFrame(columns=schema.CLOCK_SYNC_COLUMNS + ["day"])
    return Dataset(vehicles, tracks, signals, system, clock_sync, sites, zone, problems)


def camera_of(site: dict, approach: str) -> str | None:
    for name, cam in site["cameras"].items():
        if cam["approach"] == approach:
            return name
    return None
