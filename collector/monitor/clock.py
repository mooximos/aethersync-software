"""Clocks. Inside the collector every time is time.monotonic() seconds, which never jumps.

A wall-clock time for the data files is monotonic + offset. The offset is
measured at startup and again only after a clock sync, so durations never jump.
"""
from __future__ import annotations

import time
from typing import Callable


class Clock:
    def __init__(self, time_fn: Callable[[], float] = time.time,
                 mono_fn: Callable[[], float] = time.monotonic, offset: float | None = None) -> None:
        self._time, self._mono = time_fn, mono_fn
        self.offset = self.measure() if offset is None else offset

    def measure(self) -> float:
        """time.time() - time.monotonic(), from the tightest of 5 tries."""
        best = None
        for _ in range(5):
            before = self._mono()
            wall = self._time()
            after = self._mono()
            if best is None or after - before < best[0]:
                best = (after - before, wall - (before + after) / 2)
        return best[1]

    def resync(self) -> float:
        """Measure the offset again (after the system time was set). Returns the change."""
        old = self.offset
        self.offset = self.measure()
        return self.offset - old

    def mono(self) -> float:
        return self._mono()

    def wall(self, mono: float) -> float:
        return mono + self.offset

    def now(self) -> float:
        return self.wall(self._mono())


RTC_SINCE_EPOCH = "/sys/class/rtc/rtc0/since_epoch"
TOLERANCE_S = 120


def newest_file_time(data_dir) -> float | None:
    """Modification time of the newest file in the data folder (written with the clock of that day)."""
    from pathlib import Path
    newest = None
    for path in Path(data_dir).glob("*/*"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        newest = mtime if newest is None else max(newest, mtime)
    return newest


def clock_problem(now: float, data_dir, rtc_path: str = RTC_SINCE_EPOCH) -> str | None:
    """Why the system clock cannot be trusted right now, or None if nothing looks wrong.

    No fixed dates are used: the clock is wrong if it is behind files it wrote itself, or if it
    disagrees with the battery-backed RTC (a flat RTC battery resets the RTC when power is lost)."""
    newest = newest_file_time(data_dir)
    if newest is not None and now < newest - TOLERANCE_S:
        return (f"the clock is {(newest - now) / 3600:.1f} hours behind the newest data file; "
                "it was probably reset when the power went off")
    try:
        rtc = float(open(rtc_path).read().strip())
    except (OSError, ValueError):
        return None   # no RTC to compare with; check_hardware.py --rtc reports that separately
    if abs(rtc - now) > TOLERANCE_S:
        return (f"the system clock and the RTC differ by {abs(rtc - now) / 60:.0f} minutes; "
                "the RTC battery may be flat or missing")
    return None
