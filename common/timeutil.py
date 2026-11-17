"""Time helpers shared by the collector and the analysis.

Every time in the data files looks like 2026-11-16T07:14:52.3-08:00:
local clock time, one decimal of seconds, and the UTC offset.
Inside the programs we use plain "epoch seconds" (seconds since 1970 UTC),
because they never jump when daylight-saving time starts or ends.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

DEFAULT_TIME_ZONE = "America/Los_Angeles"

TIME_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d)([+-])(\d{2}):(\d{2})$"
)
DAY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def get_zone(name: str = DEFAULT_TIME_ZONE) -> ZoneInfo:
    return ZoneInfo(name)


def to_tenths(epoch_s: float) -> int:
    """Round epoch seconds to whole tenths of a second (an exact integer)."""
    return round(epoch_s * 10)


def format_epoch(epoch_s: float, zone: ZoneInfo) -> str:
    """Epoch seconds -> '2026-11-16T07:14:52.3-08:00' in the given time zone."""
    whole_s, tenth = divmod(to_tenths(epoch_s), 10)
    local = datetime.fromtimestamp(whole_s, zone)
    offset_min = int(local.utcoffset().total_seconds() // 60)
    sign = "+" if offset_min >= 0 else "-"
    hours, minutes = divmod(abs(offset_min), 60)
    return f"{local:%Y-%m-%dT%H:%M:%S}.{tenth}{sign}{hours:02d}:{minutes:02d}"


def format_time(moment: datetime) -> str:
    """Timezone-aware datetime -> text, keeping the datetime's own UTC offset."""
    if moment.tzinfo is None:
        raise ValueError("format_time needs a timezone-aware datetime")
    fixed = timezone(moment.utcoffset())
    return format_epoch(moment.timestamp(), fixed)  # type: ignore[arg-type]


def parse_time(text: str) -> datetime:
    """Text in the data-file format -> timezone-aware datetime. Strict."""
    match = TIME_PATTERN.match(text)
    if not match:
        raise ValueError(f"not a valid time (want 2026-11-16T07:14:52.3-08:00): {text!r}")
    year, month, day, hour, minute, second, tenth = (int(g) for g in match.groups()[:7])
    sign = 1 if match.group(8) == "+" else -1
    offset = timedelta(hours=int(match.group(9)), minutes=int(match.group(10))) * sign
    return datetime(year, month, day, hour, minute, second, tenth * 100_000, timezone(offset))


def parse_epoch(text: str) -> float:
    """Text in the data-file format -> epoch seconds."""
    return to_tenths(parse_time(text).timestamp()) / 10


def is_valid_time(text: str) -> bool:
    try:
        parse_time(text)
    except ValueError:
        return False
    return True


def local_day_and_hour(epoch_s: float, zone: ZoneInfo) -> tuple[str, int]:
    """Which day folder ('2026-11-16') and hour file (7) a time belongs in."""
    whole_s = to_tenths(epoch_s) // 10
    local = datetime.fromtimestamp(whole_s, zone)
    return f"{local:%Y-%m-%d}", local.hour


def local_epoch(day: str, hour: int, minute: int, zone: ZoneInfo, second: float = 0.0) -> float:
    """Local clock time on a given day -> epoch seconds."""
    year, month, dom = (int(part) for part in day.split("-"))
    return datetime(year, month, dom, hour, minute, tzinfo=zone).timestamp() + second


def clock_text(epoch_s: float, zone: ZoneInfo) -> str:
    """Just the clock part, 'HH:MM:SS.s' (used in signal event details)."""
    return format_epoch(epoch_s, zone)[11:21]


def hour_suffix(hour: int) -> str:
    return f"{hour:02d}"
