"""HCM 6th edition delay model for an isolated pretimed intersection, and the ITE safety timings."""
from __future__ import annotations

import math
from dataclasses import dataclass

K_PRETIMED = 0.5     # incremental delay calibration (pretimed)
I_ISOLATED = 1.0     # upstream filtering (isolated)
T_HOURS = 0.25       # analysis period
FT_PER_S_PER_MPH = 5280 / 3600


@dataclass
class GroupInput:
    """One lane group's inputs to the model."""
    approach: str
    name: str
    lanes: int
    v: float          # demand flow rate, veh/h
    s: float          # saturation flow of the whole group, veh/h
    phases: list[int]
    rp: float = 1.0   # platoon ratio (measured), clipped to 0.5-2.0
    factor: float = 1.0   # calibration factor for its approach (1 = none)


def uniform_delay(g: float, cycle: float, x: float) -> float:
    ratio = g / cycle
    return 0.5 * cycle * (1 - ratio) ** 2 / (1 - min(1.0, x) * ratio)


def incremental_delay(x: float, capacity: float, t: float = T_HOURS) -> float:
    return 900 * t * ((x - 1) + math.sqrt((x - 1) ** 2 + 8 * K_PRETIMED * I_ISOLATED * x / (capacity * t)))


def progression_factor(rp: float, g: float, cycle: float) -> float:
    ratio = g / cycle
    p = min(1.0, min(max(rp, 0.5), 2.0) * ratio)
    return (1 - p) / (1 - ratio)


def lane_group_delay(v: float, s: float, g: float, cycle: float, rp: float = 1.0) -> dict:
    """Control delay d = d1 * PF + d2 (no initial queue). Returns d, X, capacity and the parts."""
    capacity = s * g / cycle
    x = v / capacity
    d1 = uniform_delay(g, cycle, x)
    d2 = incremental_delay(x, capacity) if v > 0 else 0.0
    pf = progression_factor(rp, g, cycle)
    return {"d": d1 * pf + d2, "X": x, "capacity": capacity, "d1": d1, "d2": d2, "PF": pf}


def evaluate(groups: list[GroupInput], greens: dict[int, float], cycle: float) -> dict:
    """Delay for every lane group, each approach and the whole intersection (volume-weighted)."""
    rows, approach_sum, approach_v = [], {}, {}
    for group in groups:
        g = sum(greens[p] for p in group.phases)
        result = lane_group_delay(group.v, group.s, g, cycle, group.rp)
        result["d"] *= group.factor
        rows.append({"approach": group.approach, "group": group.name, "v": group.v, "s": group.s, "g": g, **result})
        approach_sum[group.approach] = approach_sum.get(group.approach, 0.0) + group.v * result["d"]
        approach_v[group.approach] = approach_v.get(group.approach, 0.0) + group.v
    total_v = sum(approach_v.values())
    by_approach = {a: approach_sum[a] / approach_v[a] if approach_v[a] else 0.0 for a in approach_sum}
    average = sum(approach_sum.values()) / total_v if total_v else 0.0
    return {"groups": rows, "by_approach": by_approach, "avg_delay_s": average,
            "veh_hours_per_hour": sum(approach_sum.values()) / 3600,
            "max_X": max((r["X"] for r in rows), default=0.0)}


def round_up(value: float, step: float = 0.1) -> float:
    return math.ceil(round(value / step, 6)) * step


def yellow_s(speed_mph: float, grade_pct: float, reaction_s: float = 1.0, decel_ftps2: float = 10.0) -> float:
    """ITE: t + V / (2a + 2Gg), at least 3.0 s."""
    speed = speed_mph * FT_PER_S_PER_MPH
    return round(max(3.0, round_up(reaction_s + speed / (2 * decel_ftps2 + 2 * (grade_pct / 100) * 32.2))), 1)


def all_red_s(width_ft: float, speed_mph: float, vehicle_ft: float = 20.0) -> float:
    """ITE: (W + L) / V, at least 1.0 s."""
    return round(max(1.0, round_up((width_ft + vehicle_ft) / (speed_mph * FT_PER_S_PER_MPH))), 1)


def ped_min_green_s(walk_s: float, crossing_ft: float, walk_speed_ftps: float) -> float:
    """Walk + crossing time, so the flashing don't walk ends before the yellow."""
    return round(round_up(walk_s + crossing_ft / walk_speed_ftps), 1)
