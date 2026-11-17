"""The data-file contract: column lists and validators for every CSV file.

Both programs import this file, so the collector cannot write something the
analysis does not expect. The validators never raise; they return a list of
plain-English problems (an empty list means the row or file is fine).
"""
from __future__ import annotations

import re
from pathlib import Path

from common import readers
from common.timeutil import DAY_PATTERN, TIME_PATTERN, parse_time

VEHICLE_COLUMNS = [
    "vehicle_id", "camera", "approach", "lane", "turn", "type", "conf",
    "t_entered", "t_stopped", "t_started", "t_crossed", "t_left",
    "wait_s", "stops", "queue_pos", "light", "flags",
]
TRACK_COLUMNS = ["time", "vehicle_id", "x_m", "y_m", "speed_mps"]
SIGNAL_COLUMNS = ["time", "event_code", "event", "approach", "lane", "detail"]
SYSTEM_COLUMNS = [
    "time", "battery_v", "current_a", "power_w", "cpu_temp_c", "hailo_temp_c",
    "fps_A", "fps_B", "fps_C", "fps_D", "dropped_frames", "disk_free_gb",
]
CLOCK_SYNC_COLUMNS = ["time_before", "time_after", "offset_s", "source"]

COLUMNS = {
    "vehicles": VEHICLE_COLUMNS,
    "tracks": TRACK_COLUMNS,
    "signal_events": SIGNAL_COLUMNS,
    "system": SYSTEM_COLUMNS,
    "clock_sync": CLOCK_SYNC_COLUMNS,
}
# The column whose time decides which hour file a row goes in.
MAIN_TIME_COLUMN = {"vehicles": "t_left", "tracks": "time", "signal_events": "time", "system": "time"}

CAMERAS = ("A", "B", "C", "D")
APPROACHES = ("N", "S", "E", "W")
TURNS = ("L", "T", "R", "U")
VEHICLE_TYPES = ("car", "motorcycle", "bus", "truck")
LIGHTS = ("day", "twilight", "night")
FLAGS = (
    "low_conf", "started_in_view", "queue_past_view", "no_cross", "short_track",
    "night_mode", "uncalibrated", "cut_at_shutdown", "id_switch_suspect",
)
EVENT_CODES = {82: "detector_on", 81: "detector_off", 1: "green_start_inferred", 10: "red_inferred_by"}
LANE_EVENT_CODES = (82, 81)
CLOCK_SOURCES = ("phone", "laptop")

VEHICLE_ID_PATTERN = re.compile(r"^([ABCD])-\d{4,}$")
_INT = re.compile(r"^\d+$")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_ONE_DECIMAL = re.compile(r"^-?\d+\.\d$")
_TWO_DECIMALS = re.compile(r"^\d\.\d{2}$")


def _choice(row: dict, column: str, allowed: tuple, problems: list[str]) -> None:
    if row[column] not in allowed:
        problems.append(f"{column} {row[column]!r} is not one of {', '.join(allowed)}")


def _time(row: dict, column: str, problems: list[str], required: bool = True) -> None:
    value = row[column]
    if value == "":
        if required:
            problems.append(f"{column} is empty")
    elif not TIME_PATTERN.match(value):
        problems.append(f"{column} {value!r} is not a time like 2026-11-16T07:14:52.3-08:00")


def _matches(row: dict, column: str, pattern: re.Pattern, what: str, problems: list[str],
             required: bool = True) -> None:
    value = row[column]
    if value == "":
        if required:
            problems.append(f"{column} is empty")
    elif not pattern.match(value):
        problems.append(f"{column} {value!r} is not {what}")


def check_vehicle_row(row: dict) -> list[str]:
    problems: list[str] = []
    flags = [f for f in row["flags"].split(";") if f]
    for flag in flags:
        if flag not in FLAGS:
            problems.append(f"flags has unknown flag {flag!r}")
    uncalibrated = "uncalibrated" in flags

    id_match = VEHICLE_ID_PATTERN.match(row["vehicle_id"])
    if not id_match:
        problems.append(f"vehicle_id {row['vehicle_id']!r} is not like A-0412")
    elif id_match.group(1) != row["camera"]:
        problems.append(f"vehicle_id {row['vehicle_id']!r} does not start with camera {row['camera']!r}")
    _choice(row, "camera", CAMERAS, problems)
    _choice(row, "approach", APPROACHES, problems)
    _choice(row, "turn", TURNS, problems)
    _choice(row, "type", VEHICLE_TYPES, problems)
    _choice(row, "light", LIGHTS, problems)
    _matches(row, "lane", _INT, "a whole number", problems, required=False)
    _matches(row, "conf", _TWO_DECIMALS, "a number from 0.00 to 1.00 with 2 decimals", problems)
    _matches(row, "wait_s", _ONE_DECIMAL, "seconds with 1 decimal", problems, required=not uncalibrated)
    _matches(row, "stops", _INT, "a whole number", problems, required=not uncalibrated)
    _matches(row, "queue_pos", _INT, "a whole number", problems, required=False)
    _time(row, "t_entered", problems, required=not uncalibrated)
    for column in ("t_stopped", "t_started", "t_crossed"):
        _time(row, column, problems, required=False)
    _time(row, "t_left", problems)
    if problems:
        return problems  # the cross-checks below need well-formed values

    stopped = row["t_stopped"] != ""
    if stopped != (row["t_started"] != ""):
        problems.append("t_stopped and t_started must both be filled or both be empty")
    if row["stops"] != "":
        if (int(row["stops"]) > 0) != stopped:
            problems.append("stops and t_stopped disagree about whether the vehicle stopped")
        if not stopped and float(row["wait_s"]) != 0.0:
            problems.append("wait_s must be 0.0 when the vehicle never stopped")
    if row["queue_pos"] != "" and (not stopped or int(row["queue_pos"]) < 1):
        problems.append("queue_pos must be empty for a vehicle that never stopped, else 1 or more")
    if ("no_cross" in flags) != (row["t_crossed"] == "") and not uncalibrated:
        problems.append("flag no_cross must be set exactly when t_crossed is empty")
    if float(row["conf"]) > 1.0:
        problems.append("conf is above 1.00")

    times = {c: parse_time(row[c]) for c in ("t_entered", "t_stopped", "t_started", "t_crossed", "t_left")
             if row[c] != ""}
    for earlier, later in (("t_entered", "t_left"), ("t_stopped", "t_started"),
                           ("t_started", "t_crossed"), ("t_crossed", "t_left"), ("t_started", "t_left")):
        if earlier in times and later in times and times[earlier] > times[later]:
            problems.append(f"{earlier} is after {later}")
    return problems


def check_track_row(row: dict) -> list[str]:
    problems: list[str] = []
    _time(row, "time", problems)
    _matches(row, "vehicle_id", VEHICLE_ID_PATTERN, "like A-0412", problems)
    _matches(row, "x_m", _NUMBER, "a number", problems, required=False)
    _matches(row, "y_m", _NUMBER, "a number", problems, required=False)
    _matches(row, "speed_mps", _ONE_DECIMAL, "m/s with 1 decimal", problems, required=False)
    if not problems and row["speed_mps"] != "" and not 0.0 <= float(row["speed_mps"]) <= 30.0:
        problems.append(f"speed_mps {row['speed_mps']} is outside 0-30")
    return problems


def check_signal_row(row: dict) -> list[str]:
    problems: list[str] = []
    _time(row, "time", problems)
    _choice(row, "approach", APPROACHES, problems)
    if not _INT.match(row["event_code"]) or int(row["event_code"]) not in EVENT_CODES:
        problems.append(f"event_code {row['event_code']!r} is not one of 82, 81, 1, 10")
        return problems
    code = int(row["event_code"])
    if row["event"] != EVENT_CODES[code]:
        problems.append(f"event {row['event']!r} does not match event_code {code} ({EVENT_CODES[code]})")
    if code in LANE_EVENT_CODES:
        _matches(row, "lane", _INT, "a whole number", problems)
        _matches(row, "detail", VEHICLE_ID_PATTERN, "a vehicle_id like A-0412", problems)
    else:
        if row["lane"] != "":
            problems.append("lane must be empty for an approach-level event")
        if row["detail"] == "":
            problems.append("detail is empty")
    return problems


def check_system_row(row: dict) -> list[str]:
    problems: list[str] = []
    _time(row, "time", problems)
    if not problems and not row["time"][17:21] == "00.0":
        problems.append(f"time {row['time']!r} is not on the minute")
    for column in SYSTEM_COLUMNS[1:]:
        pattern, what = (_INT, "a whole number") if column == "dropped_frames" else (_NUMBER, "a number")
        _matches(row, column, pattern, what, problems, required=False)
    return problems


def check_clock_sync_row(row: dict) -> list[str]:
    problems: list[str] = []
    _time(row, "time_before", problems)
    _time(row, "time_after", problems)
    _matches(row, "offset_s", _NUMBER, "a number", problems)
    _choice(row, "source", CLOCK_SOURCES, problems)
    return problems


ROW_CHECKS = {
    "vehicles": check_vehicle_row,
    "tracks": check_track_row,
    "signal_events": check_signal_row,
    "system": check_system_row,
    "clock_sync": check_clock_sync_row,
}


def check_header(kind: str, header: list[str]) -> list[str]:
    expected = COLUMNS[kind]
    if header == expected:
        return []
    if sorted(header) == sorted(expected):
        return [f"columns are in the wrong order: got {header}, want {expected}"]
    return [f"wrong columns: got {header}, want {expected}"]


def validate_file(path: Path, max_problems: int = 20) -> list[str]:
    """Check one CSV data file (plain or .gz). Returns a list of problems."""
    path = Path(path)
    kind, hour = readers.kind_and_hour(path)
    if kind is None:
        return [f"{path.name}: not a known data file name"]
    try:
        header, rows = readers.read_header_and_rows(path)
    except (OSError, EOFError, UnicodeDecodeError) as error:
        return [f"{path.name}: cannot be read ({error})"]
    problems = [f"{path.name}: {p}" for p in check_header(kind, header)]
    if problems:
        return problems
    day = path.parent.name if DAY_PATTERN.match(path.parent.name) else None
    check, main_column = ROW_CHECKS[kind], MAIN_TIME_COLUMN.get(kind)
    for line_number, row in enumerate(rows, start=2):
        if None in row or None in row.values():
            row_problems = ["wrong number of cells"]
        else:
            row_problems = check(row)
            if not row_problems and main_column:
                row_problems = _check_place(row[main_column], main_column, day, hour)
        problems += [f"{path.name} line {line_number}: {p}" for p in row_problems]
        if len(problems) >= max_problems:
            problems.append(f"{path.name}: stopped after {max_problems} problems")
            break
    return problems


def _check_place(time_text: str, column: str, day: str | None, hour: int | None) -> list[str]:
    problems = []
    if day is not None and time_text[:10] != day:
        problems.append(f"{column} {time_text} does not belong in day folder {day}")
    if hour is not None and int(time_text[11:13]) != hour:
        problems.append(f"{column} {time_text} does not belong in the hour-{hour:02d} file")
    return problems


def validate_day_folder(day_dir: Path) -> dict[str, list[str]]:
    """Check every data file in one day folder. Returns {file name: problems}."""
    from common.site import SiteError, load_site  # imported here to avoid a circular import

    results: dict[str, list[str]] = {}
    day_dir = Path(day_dir)
    for path in sorted(day_dir.iterdir()):
        if path.name.startswith("site") and path.suffix == ".json":
            try:
                load_site(path)
                results[path.name] = []
            except SiteError as error:
                results[path.name] = [str(error)]
        elif readers.kind_and_hour(path)[0] is not None:
            results[path.name] = validate_file(path)
    if "site.json" not in results:
        results["site.json"] = ["site.json is missing from the day folder"]
    return results
