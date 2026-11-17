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
