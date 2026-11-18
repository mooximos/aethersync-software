"""The timing algorithm: Webster's plan, then the fixed-time plan with the lowest total delay.

Decision variables: the green time of each phase. The cycle is tried at every whole second
in the allowed range; for each, scipy SLSQP finds the best split (from Webster's split).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from tlopt import hcm
from tlopt.intersection import PERPENDICULAR, Intersection

Z95 = 1.645


@dataclass
class Plan:
    label: str
    period: str
    cycle_s: float
    greens: dict[int, float]
    yellow: dict[int, float]
    all_red: dict[int, float]
    min_green: dict[int, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class Problem:
    inter: Intersection
    groups: list[hcm.GroupInput]
    yellow: dict[int, float]
    all_red: dict[int, float]
    min_green: dict[int, float]
    period: str = ""

    def __post_init__(self) -> None:
        self.phases = self.inter.phase_numbers()
        self.sides = self.inter.ring_sides()
        self.lost = {p: self.yellow[p] + self.all_red[p] for p in self.phases}
        self.v = np.array([g.v for g in self.groups])
        self.s = np.array([g.s for g in self.groups])
        self.rp = np.array([g.rp for g in self.groups])
        self.factor = np.array([g.factor for g in self.groups])
        self.lanes = np.array([g.lanes for g in self.groups])
        self.use = np.array([[1.0 if p in g.phases else 0.0 for p in self.phases] for g in self.groups])
        self.approach = np.array([g.approach for g in self.groups])

    # ---- the model, vectorized ------------------------------------------------
    def delays(self, greens: np.ndarray, cycle: float) -> tuple[np.ndarray, np.ndarray]:
        g = self.use @ greens
        ratio = np.clip(g / cycle, 1e-3, 0.999)   # the solver may try a green of 0 or the whole cycle
        capacity = self.s * ratio
        x = self.v / capacity
        d1 = 0.5 * cycle * (1 - ratio) ** 2 / (1 - np.minimum(1.0, x) * ratio)
        d2 = 900 * hcm.T_HOURS * ((x - 1) + np.sqrt((x - 1) ** 2 + 8 * hcm.K_PRETIMED * hcm.I_ISOLATED * x / (capacity * hcm.T_HOURS)))
        p = np.minimum(1.0, np.clip(self.rp, 0.5, 2.0) * ratio)
        return (d1 * (1 - p) / (1 - ratio) + d2) * self.factor, x

    def total_delay(self, greens: np.ndarray, cycle: float) -> float:
        """Vehicle-hours of delay per hour."""
        d, _x = self.delays(greens, cycle)
        return float(self.v @ d) / 3600

    def max_wait(self, greens: np.ndarray, cycle: float) -> dict[str, float]:
        """Red time + clearing the 95th-percentile queue that builds during red."""
        g = self.use @ greens
        red = cycle - g
        per_lane = self.v / self.lanes / 3600 * red
        queue = per_lane + Z95 * np.sqrt(per_lane)
        wait = red + queue * 3600 / (self.s / self.lanes)
        return {str(a): float(wait[self.approach == a].max()) for a in sorted(set(self.approach))}

    def side_lengths(self, greens: dict[int, float]) -> list[float]:
        return [max(sum(greens[p] + self.lost[p] for p in ring) for ring in rings if ring) for rings in self.sides]

    def min_cycle(self) -> float:
        return sum(max(sum(self.min_green[p] + self.lost[p] for p in ring) for ring in rings if ring)
                   for rings in self.sides)


def safety(inter: Intersection) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, str]]:
    """ITE yellow and all-red, and the minimum green (7 s through, 5 s protected left, raised to
    the pedestrian minimum for through phases), per phase. Also why each minimum is what it is."""
    yellow, all_red, min_green, why = {}, {}, {}, {}
    ped = inter.pedestrians
    for number in inter.phase_numbers():
        phase = inter.phases[str(number)]
        geo = inter.approaches[phase.approach]
        yellow[number] = hcm.yellow_s(geo.speed_limit_mph, geo.grade_pct)
        all_red[number] = hcm.all_red_s(geo.clear_width_ft, geo.speed_limit_mph)
        if "T" in phase.movements:
            crossing = max(inter.approaches[leg].ped_crossing_ft for leg in PERPENDICULAR[phase.approach]
                           if leg in inter.approaches)
            walk = hcm.ped_min_green_s(ped.walk_s, crossing, ped.walk_speed_ftps)
            min_green[number] = max(7.0, walk)
            why[number] = f"pedestrians: {ped.walk_s:g} s walk + {crossing:g} ft at {ped.walk_speed_ftps:g} ft/s" \
                if walk > 7 else "7 s minimum for a through phase"
        else:
            min_green[number], why[number] = 5.0, "5 s minimum for a protected left"
    return yellow, all_red, min_green, why


def _share(total: float, weights: dict[int, float], mins: dict[int, float]) -> dict[int, float]:
    """Split `total` seconds of green in proportion to weights, but never below each minimum."""
    fixed: dict[int, float] = {}
    while True:
        free = {p: w for p, w in weights.items() if p not in fixed}
        left = total - sum(fixed.values())
        weight = sum(free.values())
        out = {p: (left * w / weight if weight > 0 else left / len(free)) for p, w in free.items()}
        low = {p: mins[p] for p, g in out.items() if g < mins[p]}
        if not low or len(low) == len(free):
            out.update(low)
            return {**fixed, **out}
        fixed.update(low)


def split_for_cycle(problem: Problem, cycle: float, y: dict[int, float]) -> dict[int, float]:
    """Webster-style greens for a given cycle: the green time of each barrier side in proportion to
    its critical flow ratio, then each ring's phases in proportion to theirs; never below minimums."""
    crit, lost, side_min = {}, {}, {}
    for side, rings in enumerate(problem.sides):
        full = [r for r in rings if r]
        crit[side] = max(sum(y[p] for p in ring) for ring in full)
        lost[side] = max(sum(problem.lost[p] for p in ring) for ring in full)
        side_min[side] = max(sum(problem.min_green[p] + problem.lost[p] for p in ring) for ring in full) - lost[side]
    weights = crit if sum(crit.values()) > 0 else {s: 1.0 for s in crit}
    side_green = _share(cycle - sum(lost.values()), weights, side_min)
    greens: dict[int, float] = {}
    for side, rings in enumerate(problem.sides):
        side_len = side_green[side] + lost[side]
        for ring in rings:
            if ring:
                available = side_len - sum(problem.lost[p] for p in ring)
                greens.update(_share(available, {p: y[p] for p in ring}, problem.min_green))
    return greens


def flow_ratios(problem: Problem) -> dict[int, float]:
    ratios = problem.v / problem.s
    return {p: float(max([ratios[i] for i in range(len(problem.groups)) if problem.use[i, j]] or [0.0]))
            for j, p in enumerate(problem.phases)}


def _round_ring(greens: dict[int, float], mins: dict[int, float], target: float | None = None) -> dict[int, float]:
    """Whole seconds whose total is `target` (default: the rounded total); the phases with the
    largest remainders get the extra seconds; never below a phase's minimum."""
    target = round(sum(greens.values())) if target is None else target
    out = {p: float(max(np.floor(g), np.ceil(mins[p] - 1e-9))) for p, g in greens.items()}
    order = sorted(greens, key=lambda p: greens[p] - np.floor(greens[p]), reverse=True)
    for p in order:
        if sum(out.values()) >= target:
            break
        out[p] += 1.0
    return out


def finish(problem: Problem, greens: dict[int, float], label: str, notes: list[str]) -> Plan:
    """Round greens to whole seconds. Ring 1's phases (the cycle's path) are rounded together so
    the cycle keeps its length; ring 2 is then fitted to ring 1 on each side of the barrier. If a
    ring cannot match exactly, its last phase gets the difference (a fraction of a second)."""
    first = [next(r for r in rings if r) for rings in problem.sides]
    rounded = _round_ring({p: greens[p] for ring in first for p in ring}, problem.min_green)
    for rings, lead in zip(problem.sides, first):
        side_len = sum(rounded[p] + problem.lost[p] for p in lead)
        for ring in rings:
            if ring and ring is not lead:
                room = side_len - sum(problem.lost[p] for p in ring)
                rounded.update(_round_ring({p: greens[p] for p in ring}, problem.min_green, np.floor(room + 1e-9)))
    for rings in problem.sides:
        totals = [sum(rounded[p] + problem.lost[p] for p in ring) if ring else None for ring in rings]
        target = max(t for t in totals if t is not None)
        for ring, total in zip(rings, totals):
            if ring and total < target:
                rounded[ring[-1]] = round(rounded[ring[-1]] + target - total, 1)
    cycle = round(sum(problem.side_lengths(rounded)), 1)
    return Plan(label, problem.period, cycle, rounded, dict(problem.yellow), dict(problem.all_red),
                dict(problem.min_green), list(notes))


def webster(problem: Problem) -> Plan:
    y = flow_ratios(problem)
    crit_y = sum(max(sum(y[p] for p in ring) for ring in rings if ring) for rings in problem.sides)
    lost = sum(max(sum(problem.lost[p] for p in ring) for ring in rings if ring) for rings in problem.sides)
    low, high = problem.inter.options.cycle_bounds_s
    notes = [f"Y = {crit_y:.3f}, L = {lost:.1f} s"]
    if crit_y >= 1:
        cycle = high
        notes.append("critical flow ratio Y >= 1: Webster's formula has no answer; used the longest allowed cycle")
    else:
        c0 = (1.5 * lost + 5) / (1 - crit_y)
        cycle = min(max(round(c0), low, np.ceil(problem.min_cycle())), high)
        notes.append(f"C0 = (1.5 L + 5) / (1 - Y) = {c0:.1f} s, used {cycle:g} s")
    plan = finish(problem, split_for_cycle(problem, cycle, y), "Webster", notes)
    if abs(plan.cycle_s - cycle) > 0.05:
        plan.notes.append(f"rounding the greens to whole seconds (never below a minimum) made the cycle {plan.cycle_s:g} s")
    return plan


def _solve(problem: Problem, cycle: float, start: dict[int, float], x_max: float | None):
    phases = problem.phases
    idx = {p: i for i, p in enumerate(phases)}
    cons = []
    for rings in problem.sides:
        full = [r for r in rings if r]
        for other in full[1:]:
            a = np.zeros(len(phases))
            for p in full[0]:
                a[idx[p]] += 1
            for p in other:
                a[idx[p]] -= 1
            const = sum(problem.lost[p] for p in full[0]) - sum(problem.lost[p] for p in other)
            cons.append({"type": "eq", "fun": lambda g, a=a, c=const: a @ g + c})
    first = np.zeros(len(phases))
    for rings in problem.sides:
        for p in next(r for r in rings if r):
            first[idx[p]] = 1
    lost_first = sum(problem.lost[p] for rings in problem.sides for p in next(r for r in rings if r))
    cons.append({"type": "eq", "fun": lambda g: first @ g + lost_first - cycle})
    if x_max is not None:   # X <= x_max  <=>  s * g / C * x_max - v >= 0 (linear in g)
        cons.append({"type": "ineq", "fun": lambda g: problem.s * (problem.use @ g) / cycle * x_max - problem.v})
    cap = problem.inter.options.max_wait_cap_s
    if cap:
        cons.append({"type": "ineq", "fun": lambda g: cap - np.array(list(problem.max_wait(g, cycle).values()))})
    bounds = [(problem.min_green[p], cycle) for p in phases]
    x0 = np.array([start[p] for p in phases])
    result = minimize(lambda g: problem.total_delay(g, cycle), x0, method="SLSQP", bounds=bounds,
                      constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    feasible = result.success and all(np.all(np.asarray(c["fun"](result.x)) >= -1e-6) if c["type"] == "ineq"
                                      else abs(c["fun"](result.x)) < 1e-4 for c in cons)
    return result, feasible


def best_plan(problem: Problem, label: str = "Optimized") -> tuple[Plan, list[dict]]:
    """Try every whole-second cycle; keep the lowest total delay. Returns the plan and the curve."""
    low, high = problem.inter.options.cycle_bounds_s
    y = flow_ratios(problem)
    x_max = problem.inter.options.max_degree_of_saturation
    notes: list[str] = []
    cycles = np.arange(max(np.ceil(low), np.ceil(problem.min_cycle())), np.floor(high) + 1)
    for attempt in range(2):
        curve, best = [], None
        for cycle in cycles:
            result, ok = _solve(problem, float(cycle), split_for_cycle(problem, cycle, y), x_max)
            curve.append({"cycle_s": float(cycle), "veh_hours_per_hour": float(result.fun) if ok else None})
            if ok and (best is None or result.fun < best[0]):
                best = (float(result.fun), float(cycle), result.x)
        if best is not None or attempt:
            break
        lowest = min(max(problem.delays(np.array([split_for_cycle(problem, c, y)[p] for p in problem.phases]), c)[1])
                     for c in cycles)
        notes.append(f"X <= {x_max} is not possible for every lane group; the lowest possible largest X is about "
                     f"{lowest:.2f}, so that limit was used instead")
        x_max = lowest + 0.01
    if best is None:
        raise ValueError("no timing plan meets the constraints (check cycle_bounds_s and max_wait_cap_s)")
    # Rounding to whole seconds changes the delay a little, so compare the rounded versions of the
    # best few cycles, and Webster's plan, and keep the best one that breaks no constraint.
    ranked = sorted((c for c in curve if c["veh_hours_per_hour"] is not None), key=lambda c: c["veh_hours_per_hour"])
    candidates = []
    for entry in ranked[:5]:
        cycle = entry["cycle_s"]
        result, _ok = _solve(problem, cycle, split_for_cycle(problem, cycle, y), x_max)
        candidates.append(finish(problem, dict(zip(problem.phases, result.x)), label, notes))
    candidates.append(finish(problem, webster(problem).greens, label, notes + ["(Webster's split was the best after rounding)"]))
    def score(plan: Plan) -> tuple[int, float]:
        greens = np.array([plan.greens[p] for p in problem.phases])
        return (len(check(problem, plan, x_max)), problem.total_delay(greens, plan.cycle_s))
    plan = min(candidates, key=score)
    plan.notes += check(problem, plan, x_max)
    return plan, curve


def check(problem: Problem, plan: Plan, x_max: float | None = None) -> list[str]:
    """Every constraint, checked on the final (rounded) plan. Returns the ones it breaks."""
    issues = []
    for p in problem.phases:
        if plan.greens[p] < problem.min_green[p] - 1e-6:
            issues.append(f"phase {p}: green {plan.greens[p]:g} s is below its minimum {problem.min_green[p]:g} s")
        for name, have, need in (("yellow", plan.yellow[p], problem.yellow[p]), ("all-red", plan.all_red[p], problem.all_red[p])):
            if have < need - 1e-6:
                issues.append(f"phase {p}: {name} {have:g} s is shorter than the ITE value {need:g} s")
    low, high = problem.inter.options.cycle_bounds_s
    if not low - 1e-6 <= plan.cycle_s <= high + 1e-6:
        issues.append(f"cycle {plan.cycle_s:g} s is outside {low:g}-{high:g} s")
    greens = np.array([plan.greens[p] for p in problem.phases])
    _d, x = problem.delays(greens, plan.cycle_s)
    limit = x_max if x_max is not None else problem.inter.options.max_degree_of_saturation
    for group, value in zip(problem.groups, x):
        if value > limit + 1e-3:
            issues.append(f"{group.approach} {group.name}: X = {value:.2f} is above {limit:.2f}")
    cap = problem.inter.options.max_wait_cap_s
    if cap:
        issues += [f"{a}: estimated max wait {w:.0f} s is above the cap {cap:g} s"
                   for a, w in problem.max_wait(greens, plan.cycle_s).items() if w > cap]
    return issues


def predicted(problem: Problem, plan: Plan) -> dict:
    result = hcm.evaluate(problem.groups, plan.greens, plan.cycle_s)
    greens = np.array([plan.greens[p] for p in problem.phases])
    return {"avg_delay_s": round(result["avg_delay_s"], 1),
            "by_approach": {a: round(d, 1) for a, d in result["by_approach"].items()},
            "max_X": round(result["max_X"], 3), "LOS": _los(result["avg_delay_s"]),
            "veh_hours_per_hour": round(result["veh_hours_per_hour"], 2),
            "est_max_wait_s": {a: round(w, 0) for a, w in problem.max_wait(greens, plan.cycle_s).items()}}


def _los(delay: float) -> str:
    from tlopt.measures import los
    return los(delay)


def plan_json(problem: Problem, plan: Plan) -> dict:
    phases = {}
    for p in problem.phases:
        phase = problem.inter.phases[str(p)]
        phases[str(p)] = {"movements": f"{phase.approach}: {' '.join(phase.movements)}"
                                       + (f" (left {phase.left})" if phase.left != "none" else ""),
                          "green_s": plan.greens[p], "yellow_s": plan.yellow[p], "all_red_s": plan.all_red[p],
                          "min_green_s": plan.min_green[p]}
    return {"label": plan.label, "period": plan.period, "cycle_s": plan.cycle_s, "phases": phases,
            "predicted": predicted(problem, plan), "notes": plan.notes}
