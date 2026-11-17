"""Turn simulated trajectories into data-file rows, using the section 2 rules.

This is a second, independent implementation of the rules (the collector has
its own, working frame by frame), so the two can be checked against each other.
Everything here works in whole simulation steps of 0.1 s.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from sample_sim import DT, LANE_MOVEMENTS, LANE_WIDTH, VIEW_LENGTH, Vehicle, path_xy

STOPPED_SPEED = 0.5
MOVING_SPEED = 1.0
MIN_STOP_STEPS = round(1.0 / DT)
MIN_MOVE_STEPS = round(0.5 / DT)
ZONE_M = 6.0
REACTION_S = 1.5
GREEN_GAP_STEPS = round(20.0 / DT)
TRACK_EVERY_STEPS = round(0.5 / DT)
LOW_CONF_BELOW = 0.65
LANE_X_M = {n: ((n - 1) * LANE_WIDTH, n * LANE_WIDTH) for n in LANE_MOVEMENTS}


@dataclass
class Seen:
    """One vehicle as the collector saw it between its start and its shutdown."""
    veh: Vehicle
    camera: str
    first_step: int               # simulation step of the first sample seen
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    present_at_start: bool
    cut_at_shutdown: bool
    conf: float
    vehicle_id: str = ""
    lane: int | None = None
    cross: float | None = None    # step (with fraction) when y crossed 0
    episodes: list[tuple[int, float, bool]] = field(default_factory=list)  # (start step, end step, really ended)

    def y_at(self, step: int) -> float:
        return float(self.y[step - self.first_step])

    @property
    def last_step(self) -> int:
        return self.first_step + len(self.y) - 1


def stop_episodes(speed: np.ndarray) -> list[tuple[int, int, bool]]:
    """Stop episodes as (start index, end index, ended).

    Starts when speed has been below 0.5 m/s for 1.0 s (the start is the first
    slow sample). Ends at the first sample of a run above 1.0 m/s lasting 0.5 s.
    `ended` is False when the track finished while still stopped."""
    episodes, n, i = [], len(speed), 0
    while i < n:
        if speed[i] >= STOPPED_SPEED:
            i += 1
            continue
        j = i
        while j + 1 < n and speed[j + 1] < STOPPED_SPEED:
            j += 1
        if j - i < MIN_STOP_STEPS:
            i = j + 1
            continue
        end, k = None, j + 1
        while k < n and end is None:
            if speed[k] <= MOVING_SPEED:
                k += 1
                continue
            m = k
            while m + 1 < n and speed[m + 1] > MOVING_SPEED:
                m += 1
            if m - k >= MIN_MOVE_STEPS or m == n - 1:
                end = k
            k = m + 1
        episodes.append((i, n - 1, False) if end is None else (i, end, True))
        i = (n if end is None else end) + 1
    return episodes


def lane_of(x_m: float) -> int | None:
    for lane, (x0, x1) in LANE_X_M.items():
        if x0 <= x_m < x1:
            return lane
    return None


def observe(veh: Vehicle, camera: str, start_step: int, end_step: int, conf: float) -> Seen | None:
    """Cut a simulated vehicle down to what was visible in [start_step, end_step]."""
    lo = max(start_step - veh.step0, 0)
    hi = min(end_step - veh.step0, len(veh.s_hist) - 1)
    if veh.step0 < 0 or hi < lo:
        return None
    x, y = path_xy(veh.turn, veh.x_lane, np.array(veh.s_hist[lo:hi + 1]))
    if not 0.0 <= y[0] <= VIEW_LENGTH:
        return None   # first seen past the stop line: never in the view zone, so not a vehicle row
    seen = Seen(veh, camera, veh.step0 + lo, x, y, np.array(veh.v_hist[lo:hi + 1]),
                present_at_start=lo > 0, cut_at_shutdown=hi < len(veh.s_hist) - 1, conf=conf)
    past = np.nonzero(y <= 0.0)[0]
    n_before = len(y)
    if len(past) and past[0] > 0:
        i = int(past[0])
        seen.cross = seen.first_step + (i - 1) + y[i - 1] / (y[i - 1] - y[i])
        n_before = i
    recent = x[max(n_before - round(3.0 / DT), 0):n_before]
    seen.lane = lane_of(float(np.median(recent)))
    for i0, i1, ended in stop_episodes(seen.speed):
        if i0 < n_before:   # only stops before the stop line count
            end = seen.first_step + i1
            seen.episodes.append((seen.first_step + i0, min(end, seen.cross) if seen.cross is not None else end, ended))
    return seen


def turn_of(seen: Seen) -> str:
    movements = LANE_MOVEMENTS.get(seen.lane, ())
    if len(movements) == 1:
        return movements[0]
    if seen.cross is None or seen.y.min() > -8.0:
        return "U"
    x_at_line = seen.x[int(np.ceil(seen.cross)) - seen.first_step]
    sideways = seen.x[-1] - x_at_line
    return "R" if sideways < -3.0 else "L" if sideways > 3.0 else "T"


def queue_positions(group: list[Seen]) -> dict[str, int]:
    """queue_pos for every stopped vehicle of one lane: 1 + stopped vehicles ahead of it."""
    spans = sorted((start, end, seen) for seen in group for start, end, _ in seen.episodes)
    starts = [span[0] for span in spans]
    result = {}
    for seen in group:
        if not seen.episodes:
            continue
        step = seen.episodes[0][0]
        here = seen.y_at(step)
        ahead = {other.vehicle_id for start, end, other in spans[:bisect.bisect_right(starts, step)]
                 if end > step and other is not seen and other.y_at(step) < here}
        result[seen.vehicle_id] = 1 + len(ahead)
    return result


def vehicle_flags(seen: Seen, night: bool) -> list[str]:
    flags = []
    if seen.conf < LOW_CONF_BELOW:
        flags.append("low_conf")
    if seen.present_at_start or seen.y[0] < VIEW_LENGTH - 15.0:
        flags.append("started_in_view")
    if any(seen.y_at(start) >= VIEW_LENGTH - 5.0 for start, _, _ in seen.episodes):
        flags.append("queue_past_view")
    if seen.cross is None:
        flags.append("no_cross")
    if len(seen.y) < 5 or (len(seen.y) - 1) * DT < 1.0:
        flags.append("short_track")
    if night:
        flags.append("night_mode")
    if seen.cut_at_shutdown:
        flags.append("cut_at_shutdown")
    return flags


def detector_events(group: list[Seen], end_step: int) -> list[tuple[float, int, str]]:
    """Virtual stop-line detector of one lane: (step, code, vehicle_id) for on (82) and off (81)."""
    visits = []
    for seen in group:
        inside = np.nonzero((seen.y >= 0.0) & (seen.y <= ZONE_M))[0]
        if len(inside):
            visits.append((seen.first_step + int(inside[0]), seen.first_step + int(inside[-1]), seen.vehicle_id))
    events: list[tuple[float, int, str]] = []
    busy_until, last_out = None, ""
    for enter, leave, vehicle_id in sorted(visits):
        if busy_until is None or enter > busy_until + 1:
            if busy_until is not None:
                events.append((min(busy_until + 1, end_step), 81, last_out))
            events.append((enter, 82, vehicle_id))
            busy_until, last_out = leave, vehicle_id
        elif leave > busy_until:
            busy_until, last_out = leave, vehicle_id
    if busy_until is not None:
        events.append((min(busy_until + 1, end_step), 81, last_out))   # still occupied at shutdown: closed then
    return events


def light_events(group: list[Seen]) -> list[tuple[float, int]]:
    """Inferred green starts (code 1) and 'red by' times (code 10) for one approach, as (step, code).

    For code 1 the step is when the first queued car moved; the caller subtracts the reaction time."""
    moments = []
    for seen in group:
        if seen.cross is not None:
            moments.append((seen.cross, 1, "cross"))
        for start, end, ended in seen.episodes:
            if 0.0 <= seen.y_at(start) <= ZONE_M:
                moments.append((start, 2, "stopped"))
            if ended and 0.0 <= seen.y_at(int(end)) <= ZONE_M:
                moments.append((end, 0, "moved"))
    events: list[tuple[float, int]] = []
    last_green, crossed, red_done = None, False, True
    for step, _order, what in sorted(moments):
        if what == "moved" and (last_green is None or step - last_green >= GREEN_GAP_STEPS):
            events.append((step, 1))
            last_green, crossed, red_done = step, False, False
        elif what == "cross" and last_green is not None:
            crossed = True
        elif what == "stopped" and crossed and not red_done:
            events.append((step, 10))
            red_done = True
    return events
