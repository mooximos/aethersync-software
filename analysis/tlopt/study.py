"""The steps of a study, per period: measure -> current timing -> model check -> plans.

Every table is saved under results/ so each number in the report can be traced back."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from tlopt import hcm, measures, optimize, quality, timing, validate
from tlopt.intersection import Intersection
from tlopt.load import Dataset, local_times

SCENARIOS = {"volumes +10%": ("v", 1.1), "volumes -10%": ("v", 0.9),
             "saturation flow +10%": ("s", 1.1), "saturation flow -10%": ("s", 0.9)}


def prepare(ds: Dataset, inter: Intersection) -> pd.DataFrame:
    """Exclusions, per-vehicle delays and lane groups. Returns the exclusion counts."""
    excluded = quality.mark_exclusions(ds, inter.options.daylight_only)
    measures.add_vehicle_delays(ds)
    ds.vehicles["group"] = measures.group_of(ds.vehicles, measures.lane_groups(inter))
    return excluded


@dataclass
class Period:
    name: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    timing: dict[str, timing.ApproachTiming | None] = field(default_factory=dict)
    current: dict = field(default_factory=dict)
    groups: list[hcm.GroupInput] = field(default_factory=list)
    model_check: dict = field(default_factory=dict)
    plans: dict[str, optimize.Plan] = field(default_factory=dict)
    problem: optimize.Problem | None = None


def measure(ds: Dataset, inter: Intersection, name: str) -> Period:
    v, zone = ds.vehicles, ds.zone
    served = v[measures.period_mask(v["t_crossed_s"], zone, inter, name) & v["counted"]]
    arriving = v[measures.period_mask(v["t_entered_s"], zone, inter, name) & ~v["excluded_light"]]
    signals = ds.signals[measures.period_mask(ds.signals["time_s"], zone, inter, name)]
    if not len(served):
        raise ValueError(f"no vehicles in period {name} ({inter.periods[name]})")
    start, end = inter.period_minutes(name)
    p = Period(name)
    groups = measures.lane_groups(inter)
    t = p.tables
    t["counts_served"] = measures.demand(served, zone, "served")
    t["counts_arriving"] = measures.demand(arriving, zone, "arriving")
    t["flows"] = measures.flow_summary(t["counts_arriving"], (end - start) / 60)
    t["delay_by_approach"] = measures.delay_stats(served, ["approach"])
    t["delay_by_movement"] = measures.delay_stats(served, ["approach", "turn"])
    t["delay_by_day"] = measures.delay_stats(served, ["day", "approach"])
    t["saturation_flow"] = measures.saturation_flow(served, groups)
    t["cycle_failures"] = measures.cycle_failures(served, signals)
    t["queues"] = measures.queues(served, ds.tracks, signals)
    sat = t["saturation_flow"]
    for approach in inter.approaches:
        main = sat[(sat["approach"] == approach) & ~sat["group"].isin(["L", "R"])]
        headway = 3600 / (main["sat_flow_vph"].iloc[0] / main["lanes"].iloc[0]) if len(main) else 2.0
        p.timing[approach] = timing.infer(signals, served, approach, headway)
    t["timing"] = pd.DataFrame([vars(x) for x in p.timing.values() if x is not None])
    greens = {a: x.green_s for a, x in p.timing.items() if x is not None}
    cycles = {a: x.cycle_s for a, x in p.timing.items() if x is not None}
    t["arrivals"] = measures.arrival_pattern(arriving, signals, greens, cycles, groups)
    yellow, all_red, min_green, _why = optimize.safety(inter)
    p.current = timing.current_plan(inter, p.timing, {q: (yellow[q], all_red[q]) for q in yellow})
    p.groups = _group_inputs(arriving, zone, inter, groups, sat, t["arrivals"], ds.days)
    return p


def _group_inputs(arriving: pd.DataFrame, zone: str, inter: Intersection, groups, sat: pd.DataFrame,
                  arrivals: pd.DataFrame, days: list[str]) -> list[hcm.GroupInput]:
    """Design flow = the lane group's arrivals in the day's busiest 15 minutes x 4 (mean over days)."""
    a = arriving.assign(interval=local_times(arriving["t_entered_s"], zone).dt.floor("15min"))
    totals = a.groupby(["day", "interval"]).size()
    out = []
    for g in groups:
        flows = []
        for day in days:
            if day not in totals.index.get_level_values(0):
                continue
            peak = totals[day].idxmax()
            flows.append(4 * int(((a["day"] == day) & (a["interval"] == peak) & (a["approach"] == g.approach)
                                  & (a["group"] == g.name)).sum()))
        s_row = sat[(sat["approach"] == g.approach) & (sat["group"] == g.name)]
        rp_row = arrivals[(arrivals["approach"] == g.approach) & (arrivals["group"] == g.name)] if len(arrivals) else arrivals
        phases = [int(k) for k, ph in inter.phases.items() if ph.approach == g.approach and set(g.movements) & set(ph.movements)]
        out.append(hcm.GroupInput(g.approach, g.name, len(g.lanes), float(np.mean(flows)) if flows else 0.0,
                                  float(s_row["sat_flow_vph"].iloc[0]), phases,
                                  float(rp_row["Rp"].iloc[0]) if len(rp_row) else 1.0))
    return out


def model(p: Period, ds: Dataset, inter: Intersection) -> None:
    """Model check against measured delays, then the current, Webster and optimized plans."""
    v = ds.vehicles
    arriving = v[measures.period_mask(v["t_entered_s"], ds.zone, inter, p.name) & ~v["excluded_light"]]
    current_greens = {q: spec[0] for q, spec in p.current["phases"].items()}
    table = validate.intervals(arriving, ds.zone, p.groups, current_greens, p.current["cycle_s"])
    p.tables["model_check"] = table
    p.model_check = validate.calibrate(table) if len(table) else {"raw": None, "used": False, "factors": {}}
    if p.model_check.get("raw"):
        p.model_check["verdict"] = validate.verdict(p.model_check)
    if p.model_check.get("used"):
        p.groups = [replace(g, factor=p.model_check["factors"].get(g.approach, 1.0)) for g in p.groups]
    yellow, all_red, min_green, why = optimize.safety(inter)
    p.problem = optimize.Problem(inter, p.groups, yellow, all_red, min_green, p.name)
    p.problem.min_green_why = why   # type: ignore[attr-defined]
    cur = p.current["phases"]
    current = optimize.Plan(f"Current ({p.current['source']})", p.name, p.current["cycle_s"],
                            {q: s[0] for q, s in cur.items()}, {q: s[1] for q, s in cur.items()},
                            {q: s[2] for q, s in cur.items()}, min_green)
    current.notes = optimize.check(p.problem, current)
    p.plans["Current"] = current
    p.plans["Webster"] = optimize.webster(p.problem)
    p.plans["Webster"].notes += optimize.check(p.problem, p.plans["Webster"])
    p.plans["Optimized"], curve = optimize.best_plan(p.problem)
    p.tables["cycle_curve"] = pd.DataFrame(curve)
    p.tables["robustness"] = robustness(p.problem, p.plans["Optimized"])


def robustness(problem: optimize.Problem, plan: optimize.Plan) -> pd.DataFrame:
    greens = np.array([plan.greens[q] for q in problem.phases])
    base = problem.total_delay(greens, plan.cycle_s) * 3600 / problem.v.sum()
    rows = []
    for name, (what, factor) in SCENARIOS.items():
        groups = [replace(g, v=g.v * factor) if what == "v" else replace(g, s=g.s * factor) for g in problem.groups]
        scenario = optimize.Problem(problem.inter, groups, problem.yellow, problem.all_red, problem.min_green, problem.period)
        own, _curve = optimize.best_plan(scenario, "scenario")
        mine = scenario.total_delay(greens, plan.cycle_s) * 3600 / scenario.v.sum()
        best = scenario.total_delay(np.array([own.greens[q] for q in scenario.phases]), own.cycle_s) * 3600 / scenario.v.sum()
        rows.append({"scenario": name, "recommended_plan_delay_s": round(mine, 1),
                     "change_vs_base_s": round(mine - base, 1), "scenario_best_cycle_s": own.cycle_s,
                     "scenario_best_delay_s": round(best, 1), "within_5pct_of_best": bool(mine <= 1.05 * best + 1e-9)})
    return pd.DataFrame(rows)


def time_of_day(periods: dict[str, Period]) -> dict:
    """Separate plans if two periods' optimized plans differ by > 10 s of cycle or > 10% of any split."""
    names = list(periods)
    if len(names) < 2:
        return {"separate": True, "reason": "only one period was studied"}
    reasons = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = periods[a].plans["Optimized"], periods[b].plans["Optimized"]
            if abs(pa.cycle_s - pb.cycle_s) > 10:
                reasons.append(f"{a} and {b} cycles differ by {abs(pa.cycle_s - pb.cycle_s):g} s")
            for q in pa.greens:
                sa, sb = pa.greens[q] / pa.cycle_s, pb.greens[q] / pb.cycle_s
                if abs(sa - sb) > 0.10 * max(sa, sb):
                    reasons.append(f"phase {q} gets {100 * sa:.0f}% of the cycle in {a} but {100 * sb:.0f}% in {b}")
    if reasons:
        return {"separate": True, "reason": "; ".join(reasons)}
    totals = {}
    for cand in names:
        plan = periods[cand].plans["Optimized"]
        totals[cand] = sum(p.problem.total_delay(np.array([plan.greens[q] for q in p.problem.phases]), plan.cycle_s)
                           for p in periods.values())
    best = min(totals, key=totals.get)
    return {"separate": False, "one_plan_from": best,
            "reason": f"the optimized plans are within 10 s of cycle and 10% of split; the {best} plan gives "
                      f"the lowest total delay over all periods",
            "veh_hours_per_hour_total": {k: round(v, 2) for k, v in totals.items()}}


def sumo_demand(p: Period) -> pd.DataFrame:
    """Mean arrivals per 15 min x approach x movement x type over the days. Unknown turns (U) are
    shared out over that approach's known movements in the same interval."""
    counts = p.tables["counts_arriving"]
    mean = counts.groupby(["interval", "approach", "turn", "type"])["vehicles"].sum() / counts["day"].nunique()
    mean = mean.reset_index()
    rows = []
    for (interval, approach), group in mean.groupby(["interval", "approach"]):
        known = group[group["turn"] != "U"]
        unknown = group.loc[group["turn"] == "U", "vehicles"].sum()
        total = known["vehicles"].sum()
        for row in known.itertuples():
            rows.append({"interval": interval, "approach": approach, "movement": row.turn, "type": row.type,
                         "vehicles": row.vehicles * (1 + unknown / total if total else 1)})
    return pd.DataFrame(rows)


def save(p: Period, folder: Path) -> None:
    (folder / "measures").mkdir(parents=True, exist_ok=True)
    (folder / "plans").mkdir(parents=True, exist_ok=True)
    for key, table in p.tables.items():
        sub = "plans" if key in ("model_check", "cycle_curve", "robustness") else "measures"
        table.to_csv(folder / sub / f"{p.name}_{key}.csv", index=False)
    sumo_demand(p).to_csv(folder / "measures" / f"{p.name}_sumo_demand.csv", index=False)
    if p.problem is not None:
        for label, plan in p.plans.items():
            data = optimize.plan_json(p.problem, plan)
            (folder / "plans" / f"{p.name}_{label.lower()}.json").write_text(json.dumps(data, indent=2))
        (folder / "plans" / f"{p.name}_model_check.json").write_text(json.dumps(p.model_check, indent=2, default=float))
        inputs = pd.DataFrame([vars(g) for g in p.groups])
        inputs.to_csv(folder / "plans" / f"{p.name}_model_inputs.csv", index=False)
