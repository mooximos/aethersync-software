"""Inferring when the light turned green or red from how queues start and stop.

code 1  green_start_inferred: the first car stopped in the stop-line zone starts
        moving; the green began one reaction time earlier.
code 10 red_inferred_by: the first car that stops in the zone after the approach
        has had a crossing since the last green. The light was red by then.

Lanes that allow a right turn are left out, because a car turning right on red
looks just like a car leaving on green (unless the approach has no other lane).
"""
from __future__ import annotations

from collections import deque


def signal_lanes(lanes: dict[int, tuple[str, ...]]) -> set[int]:
    """Lanes used to infer the light."""
    without_right = {lane for lane, moves in lanes.items() if "R" not in moves}
    return without_right or set(lanes)


class SignalInference:
    def __init__(self, lanes: set[int], zone_m: float, reaction_s: float, min_gap_s: float) -> None:
        self.lanes, self.zone_m, self.reaction_s, self.min_gap_s = lanes, zone_m, reaction_s, min_gap_s
        self.last_green_move: float | None = None   # when the car that marked the last green moved
        self.red_done = True
        self.crossings: deque[float] = deque(maxlen=500)

    def _counts(self, y: float, lane: int | None) -> bool:
        return lane in self.lanes and 0.0 <= y <= self.zone_m

    def on_cross(self, t: float) -> None:
        self.crossings.append(t)

    def on_stop_end(self, t: float, y: float, lane: int | None) -> tuple[float, int, float] | None:
        """A stop episode ended at time t. Returns (event time, 1, the moment the car moved) or None."""
        if not self._counts(y, lane):
            return None
        if self.last_green_move is not None and t - self.last_green_move < self.min_gap_s:
            return None
        self.last_green_move, self.red_done = t, False
        return t - self.reaction_s, 1, t

    def on_stop_start(self, t: float, y: float, lane: int | None) -> tuple[float, int, float] | None:
        """A stop episode began at time t. Returns (t, 10, t) or None."""
        if self.red_done or self.last_green_move is None or not self._counts(y, lane):
            return None
        if any(self.last_green_move < crossed <= t for crossed in self.crossings):
            self.red_done = True
            return t, 10, t
        return None
