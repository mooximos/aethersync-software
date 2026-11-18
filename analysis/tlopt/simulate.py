"""The SUMO check of the plans: reads results/plans and results/measures, writes results/sumo."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tlopt import sumo_model as sm
from tlopt import sumo_run
from tlopt.intersection import Intersection

PLANS = ["current", "webster", "optimized"]


def target_sat_flow(results: Path, periods: list[str]) -> tuple[float, str]:
    """Measured saturation flow per through lane (weighted by the number of headways)."""
    values, weights = [], []
    for period in periods:
        table = pd.read_csv(results / "measures" / f"{period}_saturation_flow.csv")
        main = table[~table["group"].isin(["L", "R"]) & (table["source"] == "measured")]
        values += list(main["sat_flow_vph"] / main["lanes"])
        weights += list(main["headways"])
    if not values:
        return 1800.0, "no measured saturation flow; used 1800 veh/h/lane"
    return float(np.average(values, weights=weights)), "measured (through lane groups)"


def demand_dict(results: Path, period: str) -> tuple[dict, float]:
    table = pd.read_csv(results / "measures" / f"{period}_sumo_demand.csv")
    intervals = sorted(table["interval"].unique())
    demand = {}
    for row in table.itertuples():
        index = intervals.index(row.interval)
        demand[(index, row.approach, row.movement, row.type)] = row.vehicles
        if index == 0:   # warm-up: the first interval's traffic, 15 min before the period
            demand[(-1, row.approach, row.movement, row.type)] = row.vehicles
    return demand, 900.0 * len(intervals)


def simulate(inter: Intersection, results: Path, seeds: int = 10, log=print, processes: int | None = None) -> dict:
    started = time.time()
    out = results / "sumo"
    out.mkdir(parents=True, exist_ok=True)
    studied = json.loads((results / "measures" / "summary.json").read_text())["periods"] \
        if (results / "measures" / "summary.json").exists() else inter.periods
    periods = [p for p in inter.periods if p in studied and (results / "plans" / f"{p}_optimized.json").exists()
               and (results / "measures" / f"{p}_sumo_demand.csv").exists()]
    for p in inter.periods:
        if p in studied and p not in periods:
            log(f"SUMO: period {p} skipped, its plans or demand are missing (run `optimize` again)")
    if not periods:
        raise FileNotFoundError(f"no plans in {results / 'plans'}: run `optimize` first")
    folder = sm.work_dir()
    try:
        net = sm.build_network(inter, folder)
        links = sm.signal_links(net)
        plans = {(p, name): json.loads((results / "plans" / f"{p}_{name}.json").read_text())
                 for p in periods for name in PLANS}
        target, source = target_sat_flow(results, periods)
        log(f"SUMO: calibrating the driver model to {target:.0f} veh/h per through lane ({source})")
        tau, history = sumo_run.calibrate_tau(inter, net, links, plans[(periods[0], "current")], target,
                                              folder / "calibration", log)
        calibration = {"tau_s": round(tau, 3), "target_sat_flow_per_lane": round(target), "target_source": source,
                       "steps": history, "within_5pct": abs(history[-1]["sat_flow_per_lane"] - target) <= 0.05 * target}
        (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
        tasks = []
        for period in periods:
            demand, period_s = demand_dict(results, period)
            for name in PLANS:
                for seed in range(1, seeds + 1):
                    tasks.append({"folder": str(folder / f"{period}_{name}_{seed}"), "inter": inter, "plan": name,
                                  "plan_json": plans[(period, name)], "links": links, "net": str(net),
                                  "demand": demand, "seed": seed, "tau": tau, "period": period, "period_s": period_s})
        log(f"SUMO: {len(tasks)} runs ({len(periods)} periods x 3 plans x {seeds} seeds, the same seeds for each plan)")
        results_list = sumo_run.run_all(tasks, processes)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    per_seed = sumo_run.per_seed(results_list)
    per_seed["plan"] = per_seed["plan"].str.capitalize()
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary = sumo_run.summarize(per_seed)
    summary.to_csv(out / "summary.csv", index=False)
    check = sim_check(results, per_seed, periods)
    check.to_csv(out / "sim_check.csv", index=False)
    log(f"SUMO: done in {time.time() - started:.0f} s")
    return {"calibration": calibration, "summary": summary, "check": check}


def sim_check(results: Path, per_seed: pd.DataFrame, periods: list[str]) -> pd.DataFrame:
    """Does SUMO reproduce today's traffic? The current plan in SUMO vs the measurements, per approach."""
    rows = []
    for period in periods:
        measured = pd.read_csv(results / "measures" / f"{period}_delay_by_approach.csv").set_index("approach")
        queues = pd.read_csv(results / "measures" / f"{period}_queues.csv").set_index("approach")
        sim = per_seed[(per_seed["period"] == period) & (per_seed["plan"] == "Current")]
        for approach in measured.index:
            delay = sim.get(f"delay_{approach}_s")
            queue = sim.get(f"queue_{approach}_m")
            row = {"period": period, "approach": approach,
                   "measured_delay_s": measured.loc[approach, "mean_delay_s"],
                   "sumo_delay_s": round(float(delay.mean()), 1) if delay is not None else np.nan,
                   "measured_mean_queue_m": queues.loc[approach, "mean_queue_m"] if approach in queues.index else np.nan,
                   "sumo_mean_queue_m": round(float(queue.mean()), 1) if queue is not None else np.nan}
            row["delay_error_s"] = round(row["sumo_delay_s"] - row["measured_delay_s"], 1)
            row["queue_error_m"] = round(row["sumo_mean_queue_m"] - row["measured_mean_queue_m"], 1)
            rows.append(row)
    return pd.DataFrame(rows)
