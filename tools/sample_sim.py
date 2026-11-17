"""A small traffic simulation used only to make sample data.

One 4-way intersection with a fixed-time signal. Each approach has 3 lanes
(1 = right + through, 2 = through, 3 = left). Vehicles arrive at random
(Poisson), enter 120 m before the stop line at 13.4 m/s, brake at 3 m/s2,
accelerate at 2 m/s2 and follow the car ahead with a safe gap.

Simplifications (on purpose): no lane changes, no right turn on red, and left
turns go with their approach's green without waiting for oncoming traffic.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DT = 0.1                 # simulation step, seconds
FREE_SPEED = 13.4        # m/s (30 mph)
ACCEL = 2.0              # m/s2
DECEL = 3.0              # m/s2
VIEW_LENGTH = 120.0      # vehicles appear this far before the stop line, meters
STANDSTILL_GAP = 2.0     # bumper-to-bumper gap in a stopped queue, meters
LANE_WIDTH = 3.5
REACTION_RANGE = (1.0, 2.3)   # seconds from green to the first car moving
# A follower crosses the line (its delay) + (time to cover one stopped-car spacing)
# after its leader. This constant is that second part at queue-discharge speed.
FOLLOW_SPACING_S = 0.5

LANE_MOVEMENTS = {1: ("R", "T"), 2: ("T",), 3: ("L",)}
QUARTER_FACTOR = (0.85, 1.0, 1.15, 1.0)   # share of the hourly volume in each quarter hour
TYPE_SHARE = {"car": 0.93, "truck": 0.04, "motorcycle": 0.02, "bus": 0.01}
LENGTH_M = {"car": 4.5, "motorcycle": 2.2, "truck": 8.0, "bus": 12.0}
SAT_HEADWAY_S = {"car": 2.0, "motorcycle": 2.0, "truck": 3.5, "bus": 3.5}
TURN_SPEED = {"T": FREE_SPEED, "R": 6.0, "L": 8.0}   # speed while turning, m/s
# Where each path leaves the camera view (meters of path past the stop line),
# where the turn begins, and the turn radius.
EXIT_PATH_M = {"T": 15.0, "R": 18.0, "L": 24.0}
TURN_START_M = {"R": 1.0, "L": 2.0}
TURN_RADIUS_M = {"R": 8.0, "L": 15.0}
DIVERGE_M = 6.0          # a turning leader this far past the line no longer blocks a follower

SIGNAL_GROUP = {"N": "NS", "S": "NS", "E": "EW", "W": "EW"}


@dataclass
class SignalPlan:
    """Fixed-time signal. Greens are (start, end) seconds within the cycle."""
    cycle: float = 120.0
    green: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"NS": (0.0, 50.0), "EW": (56.0, 110.0)})
    yellow: float = 4.0
    all_red: float = 2.0

    def check(self) -> None:
        (a0, a1), (b0, b1) = sorted(self.green.values())
        change = self.yellow + self.all_red
        if not (0 <= a0 < a1 and a1 + change <= b0 < b1 and b1 + change <= self.cycle + a0):
            raise ValueError("splits must leave room for yellow + all-red after each green, inside the cycle")

    def is_green(self, approach: str, clock_s: float, reaction_s: float = 0.0) -> bool:
        start, end = self.green[SIGNAL_GROUP[approach]]
        return start + reaction_s <= clock_s % self.cycle < end


@dataclass
class Vehicle:
    approach: str
    lane: int
    turn: str
    vtype: str
    arrival_s: float
    length: float
    headway: float        # saturation headway behind the vehicle ahead, seconds
    stop_at: float        # where the front bumper stops for a red light, meters before the line
    x_lane: float         # lateral position while in the lane, meters from the curb
    step0: int = -1       # simulation step of the first recorded sample
    s: float = VIEW_LENGTH   # path distance before the stop line (negative = past it)
    v: float = FREE_SPEED
    decision: str = ""       # "", "stop" or "go" for the current red
    leader: "Vehicle | None" = None   # the vehicle ahead in the lane when this one entered
    s_hist: list[float] = field(default_factory=list)
    v_hist: list[float] = field(default_factory=list)

    @property
    def delay_steps(self) -> int:
        return round((self.headway - FOLLOW_SPACING_S) / DT)


def virtual_leader(veh: Vehicle, step: int) -> tuple[float, float] | None:
    """Where this vehicle may be at most, and how fast that limit is moving.

    Car following (Newell's model): a follower repeats the path of the vehicle
    ahead, `delay` seconds later and one stopped-car spacing behind it.
    Returns (position limit, speed of the limit), or None if nothing is ahead."""
    lead = veh.leader
    while lead is not None:
        index = step - veh.delay_steps - lead.step0
        if 0 <= index < len(lead.s_hist):
            lead_s = lead.s_hist[index]
            turned_away = lead.turn != veh.turn and lead_s < -DIVERGE_M
            if not turned_away:
                next_index = min(index + 1, len(lead.v_hist) - 1)   # its speed one step later: brake in time
                return lead_s + lead.length + STANDSTILL_GAP, lead.v_hist[next_index]
        lead = lead.leader   # that one is gone or turned off; look one further ahead
    return None


def brake_speed(gap: float, target_speed: float = 0.0) -> float:
    """Fastest speed now that still lets us slow to target_speed within `gap` meters
    at 3 m/s2 (written for 0.1 s steps, so the stop is smooth right to the end)."""
    half = DECEL * DT / 2
    return -half + math.sqrt(half * half + target_speed * target_speed + 2 * DECEL * max(gap, 0.0))


def path_xy(turn: str, x_lane: float, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground position (x_m, y_m) for path distance s. Turns curve away after the stop line."""
    s = np.asarray(s, dtype=float)
    x = np.full_like(s, x_lane)
    y = s.copy()
    if turn == "T":
        return x, y
    start, radius = TURN_START_M[turn], TURN_RADIUS_M[turn]
    side = -1.0 if turn == "R" else 1.0          # right = toward the curb = smaller x
    along = np.clip(-start - s, 0.0, None)       # meters travelled since the turn began
    angle = np.minimum(along / radius, math.pi / 2)
    beyond = np.clip(along - radius * math.pi / 2, 0.0, None)
    turning = along > 0
    x = np.where(turning, x_lane + side * (radius * (1 - np.cos(angle)) + beyond), x)
    y = np.where(turning, -start - radius * np.sin(angle), y)
    return x, y


def _make_arrivals(volumes: dict[str, float], duration_s: float, clock0_s: float,
                   rng: np.random.Generator) -> list[tuple]:
    """Poisson arrivals for one approach: sorted (time, turn, type).

    Traffic builds and eases inside each clock hour (QUARTER_FACTOR)."""
    arrivals = []
    types, shares = list(TYPE_SHARE), list(TYPE_SHARE.values())
    first_quarter = int(clock0_s // 900)
    last_quarter = int((clock0_s + duration_s) // 900)
    for quarter in range(first_quarter, last_quarter + 1):
        start = max(quarter * 900 - clock0_s, 0.0)
        end = min((quarter + 1) * 900 - clock0_s, duration_s)
        for turn in ("L", "T", "R"):
            rate = volumes.get(turn, 0.0) / 3600.0 * QUARTER_FACTOR[quarter % 4]
            for time_s in rng.uniform(start, end, rng.poisson(rate * (end - start))):
                arrivals.append((float(time_s), turn, str(rng.choice(types, p=shares))))
    return sorted(arrivals)


class _Lane:
    def __init__(self, number: int) -> None:
        self.number = number
        self.vehicles: list[Vehicle] = []      # front of the queue first
        self.pending: deque[Vehicle] = deque()  # arrived upstream, not yet in view
        self.last_entry_s = -1e9
        self.last_entered: Vehicle | None = None


def _speed_limit(veh: Vehicle) -> float:
    if veh.turn == "T":
        return FREE_SPEED
    turn_speed = TURN_SPEED[veh.turn]
    if veh.s <= 0:
        return turn_speed
    return min(FREE_SPEED, math.sqrt(turn_speed * turn_speed + 2 * ACCEL * veh.s))


def _move(veh: Vehicle, step: int, green: bool) -> None:
    speed = min(veh.v + ACCEL * DT, _speed_limit(veh))
    limit = virtual_leader(veh, step)
    if limit is not None:
        speed = min(speed, brake_speed(veh.s - limit[0], limit[1]))
    if green:
        veh.decision = ""
    elif veh.s > 0:
        room = veh.s - veh.stop_at
        if veh.decision == "":
            can_stop = room >= veh.v * veh.v / (2 * DECEL)
            veh.decision = "stop" if can_stop else "go"   # "go" = too close, clears on yellow
        if veh.decision == "stop":
            speed = min(speed, brake_speed(room))
    veh.v = max(speed, 0.0)
    veh.s -= veh.v * DT
    veh.s_hist.append(veh.s)
    veh.v_hist.append(veh.v)


def _try_enter(lane: _Lane, step: int, time_s: float) -> None:
    if not lane.pending:
        return
    veh = lane.pending[0]
    if time_s < veh.arrival_s or time_s < lane.last_entry_s + veh.headway:
        return
    veh.leader = lane.last_entered
    speed = FREE_SPEED
    limit = virtual_leader(veh, step)
    if limit is not None:
        gap = VIEW_LENGTH - limit[0]
        if gap < 0.5:
            return   # the queue reaches past the view; wait outside
        speed = min(speed, brake_speed(gap, limit[1]))
    lane.pending.popleft()
    lane.last_entry_s = time_s
    lane.last_entered = veh
    veh.step0, veh.s, veh.v = step, VIEW_LENGTH, speed
    veh.s_hist.append(veh.s)
    veh.v_hist.append(veh.v)
    lane.vehicles.append(veh)


def _pick_lane(lanes: dict[int, _Lane], turn: str) -> _Lane:
    """Shortest lane that allows this movement; ties go to the lane nearer the center."""
    options = [lane for number, lane in lanes.items() if turn in LANE_MOVEMENTS[number]]
    return min(options, key=lambda lane: (len(lane.vehicles) + len(lane.pending), -lane.number))


def simulate_approach(approach: str, plan: SignalPlan, volumes: dict[str, float], n_steps: int,
                      clock0_s: float, rng: np.random.Generator) -> list[Vehicle]:
    """Simulate one approach. clock0_s = seconds after midnight at step 0."""
    lanes = {number: _Lane(number) for number in LANE_MOVEMENTS}
    arrivals = deque(_make_arrivals(volumes, n_steps * DT, clock0_s, rng))
    n_cycles = int((clock0_s + n_steps * DT) // plan.cycle) + 2
    reaction = {number: rng.uniform(*REACTION_RANGE, n_cycles) for number in lanes}
    finished: list[Vehicle] = []

    for step in range(n_steps):
        time_s = step * DT
        clock_s = clock0_s + time_s
        cycle_index = int(clock_s // plan.cycle)
        while arrivals and arrivals[0][0] <= time_s:
            arrival_s, turn, vtype = arrivals.popleft()
            lane = _pick_lane(lanes, turn)
            center = (lane.number - 0.5) * LANE_WIDTH
            lane.pending.append(Vehicle(
                approach=approach, lane=lane.number, turn=turn, vtype=vtype, arrival_s=arrival_s,
                length=LENGTH_M[vtype], headway=SAT_HEADWAY_S[vtype],
                stop_at=float(rng.uniform(0.5, 1.5)), x_lane=center + float(rng.uniform(-0.3, 0.3))))
        for lane in lanes.values():
            green = plan.is_green(approach, clock_s, reaction[lane.number][cycle_index])
            for veh in lane.vehicles:   # front first, so each leader has already moved
                _move(veh, step, green)
            if any(veh.s < -EXIT_PATH_M[veh.turn] for veh in lane.vehicles):
                finished += [veh for veh in lane.vehicles if veh.s < -EXIT_PATH_M[veh.turn]]
                lane.vehicles = [veh for veh in lane.vehicles if veh.s >= -EXIT_PATH_M[veh.turn]]
            _try_enter(lane, step, time_s)
    for lane in lanes.values():
        finished += lane.vehicles   # still in view when the simulation ends
    return finished
