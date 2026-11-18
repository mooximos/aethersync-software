"""Running SUMO: calibrate the driver model, run every plan with the same seeds, compare."""
from __future__ import annotations

import multiprocessing
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tlopt import sumo_model as sm
from tlopt.intersection import Intersection

WARMUP_S = 900.0
METRICS = {"time_loss_s": "timeLoss", "waiting_s": "waitingTime", "stops": "waitingCount"}
EMISSIONS = {"co2_g": "CO2_abs", "nox_g": "NOx_abs", "fuel_g": "fuel_abs"}   # SUMO gives mg


def _run_sumo(folder: Path, net: Path, routes: Path, additional: Path, seed: int, outputs: list[str]) -> None:
    command = [sm.binary("sumo"), "-n", str(net), "-r", routes.name, "-a", additional.name, "--seed", str(seed),
               "--no-step-log", "true", "--duration-log.disable", "true", "--time-to-teleport", "300",
               "--no-warnings", "true", *outputs]
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True, env=sm.sumo_env())
    if result.returncode != 0:
        raise RuntimeError(f"SUMO failed in {folder}: {result.stderr[-2000:]}")


# ---- calibration --------------------------------------------------------------------
def simulated_sat_flow(inter: Intersection, net: Path, links, plan: dict, tau: float, folder: Path) -> float:
    """Saturation flow per through lane in SUMO: heavy through traffic only, headways at the stop
    line from the 5th queued car of each green on (the same field method as the measurement)."""
    folder.mkdir(parents=True, exist_ok=True)
    demand = {(i, a, "T", "car"): 400.0 for i in range(2) for a in inter.approaches}
    routes = sm.write_routes(inter, demand, folder, 7, 0.0, tau, "calib.rou.xml")
    root = ET.Element("additional")
    root.append(sm.tls_program(inter, plan, links, "calib"))
    lanes = [(a, lane) for a, geo in inter.approaches.items() for lane, moves in geo.lanes.items() if moves == ["T"]]
    for approach, lane in lanes:
        ET.SubElement(root, "instantInductionLoop", id=f"s_{approach}_{lane}", lane=f"{approach}_in_{lane}",
                      pos="-0.5", friendlyPos="true", file="line.xml")
    (folder / "calib.add.xml").write_bytes(ET.tostring(root))
    _run_sumo(folder, net, routes, folder / "calib.add.xml", 7, ["--end", "1800"])
    crossings: dict[str, list[float]] = {}
    for _event, item in ET.iterparse(folder / "line.xml"):
        if item.tag == "instantOut" and item.get("state") == "enter":
            crossings.setdefault(item.get("id"), []).append(float(item.get("time")))
    headways = []
    for times in crossings.values():
        times = np.sort(times)
        runs = np.split(times, np.where(np.diff(times) > 6.0)[0] + 1)   # one run = one green's discharge
        for run in runs:
            if len(run) > 6:
                headways += list(np.diff(run)[4:])
    return 3600.0 / float(np.median(headways)) if headways else float("nan")


def calibrate_tau(inter: Intersection, net: Path, links, plan: dict, target: float, folder: Path,
                  log=print) -> tuple[float, list[dict]]:
    """Adjust tau until SUMO's saturation flow is within 5% of the measured one (bisection)."""
    low, high, history = 0.2, 3.0, []
    tau = 1.0
    for step in range(10):
        flow = simulated_sat_flow(inter, net, links, plan, tau, folder / f"c{step}")
        history.append({"tau": round(tau, 3), "sat_flow_per_lane": round(flow), "target": round(target)})
        log(f"  calibration: tau {tau:.2f} s -> {flow:.0f} veh/h/lane (target {target:.0f})")
        if abs(flow - target) / target <= 0.05:
            break
        if flow > target:
            low = tau
        else:
            high = tau
        tau = (low + high) / 2
    return tau, history


# ---- the comparison runs --------------------------------------------------------------
def _one_run(task: dict) -> dict:
    folder = Path(task["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    inter, plan_name = task["inter"], task["plan"]
    routes = sm.write_routes(inter, task["demand"], folder, task["seed"], WARMUP_S, task["tau"])
    additional = sm.write_additional(inter, {plan_name: task["plan_json"]}, task["links"], task["net"], folder,
                                     task["plan_json"]["cycle_s"])
    _run_sumo(folder, Path(task["net"]), routes, additional, task["seed"],
              ["--tripinfo-output", "trips.xml", "--device.emissions.probability", "1"])
    end = WARMUP_S + task["period_s"]
    rows = []
    for _event, item in ET.iterparse(folder / "trips.xml"):
        if item.tag != "tripinfo":
            continue
        depart = float(item.get("depart"))
        if WARMUP_S <= depart < end:
            row = {"approach": item.get("id")[0], "movement": item.get("id")[1], "type": item.get("vType")}
            row.update({k: float(item.get(v)) for k, v in METRICS.items()})
            emissions = item.find("emissions")
            row.update({k: float(emissions.get(v)) / 1000 for k, v in EMISSIONS.items()})
            rows.append(row)
        item.clear()
    queues = []
    for _event, item in ET.iterparse(folder / "queues.xml"):
        if item.tag == "interval" and WARMUP_S <= float(item.get("begin")) < end:
            queues.append({"approach": item.get("id").split("_")[1], "begin": float(item.get("begin")),
                           "jam_m": float(item.get("maxJamLengthInMeters"))})
    if not task.get("keep"):
        shutil.rmtree(folder, ignore_errors=True)
    trips = add_control_delay(pd.DataFrame(rows))
    q = pd.DataFrame(queues)
    queue = q.groupby(["approach", "begin"])["jam_m"].max().groupby("approach").mean() if len(q) else pd.Series(dtype=float)
    return {"period": task["period"], "plan": plan_name, "seed": task["seed"], "trips": trips, "queue_m": queue.to_dict()}


def add_control_delay(trips: pd.DataFrame) -> pd.DataFrame:
    """SUMO's timeLoss also counts free-flow losses (driver randomness, slowing to turn). Like the
    measured control delay, subtract what vehicles of the same approach, movement and type lose
    when they never stop (median, in the same run)."""
    if not len(trips):
        return trips.assign(delay_s=[])
    free = trips[trips["stops"] == 0]
    keys = ["approach", "movement", "type"]
    base = free.groupby(keys)["time_loss_s"].median().rename("base_a")
    base_move = free.groupby(["approach", "movement"])["time_loss_s"].median().rename("base_b")
    trips = trips.join(base, on=keys).join(base_move, on=["approach", "movement"])
    baseline = trips["base_a"].fillna(trips["base_b"]).fillna(free["time_loss_s"].median() if len(free) else 0.0)
    trips["delay_s"] = (trips["time_loss_s"] - baseline).clip(lower=0)
    return trips.drop(columns=["base_a", "base_b"])


def run_all(tasks: list[dict], processes: int | None = None) -> list[dict]:
    processes = processes or max(1, min(multiprocessing.cpu_count(), len(tasks)))
    with multiprocessing.get_context("spawn").Pool(processes) as pool:
        return pool.map(_one_run, tasks)


def per_seed(results: list[dict]) -> pd.DataFrame:
    """One row per period x plan x seed: per-vehicle means (and the max wait) of that run."""
    rows = []
    for r in results:
        trips = r["trips"]
        if not len(trips):
            continue
        row = {"period": r["period"], "plan": r["plan"], "seed": r["seed"], "vehicles": len(trips)}
        for column in ["delay_s"] + list(METRICS) + list(EMISSIONS):
            row[column] = float(trips[column].mean())
        row["max_wait_s"] = float(trips["waiting_s"].max())
        row["p95_wait_s"] = float(trips["waiting_s"].quantile(0.95))
        for approach, delay in trips.groupby("approach")["delay_s"].mean().items():
            row[f"delay_{approach}_s"] = float(delay)
        for approach, queue in r["queue_m"].items():
            row[f"queue_{approach}_m"] = float(queue)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(seeds: pd.DataFrame, baseline: str = "Current") -> pd.DataFrame:
    """Mean and 95% CI per plan; paired t-test of each plan against the baseline (same seeds)."""
    metrics = ["delay_s", "time_loss_s", "waiting_s", "p95_wait_s", "max_wait_s", "stops", "co2_g", "nox_g", "fuel_g"]
    rows = []
    for (period, plan), group in seeds.groupby(["period", "plan"]):
        base = seeds[(seeds["period"] == period) & (seeds["plan"] == baseline)].set_index("seed")
        mine = group.set_index("seed")
        for metric in metrics:
            values = mine[metric].to_numpy()
            mean = float(values.mean())
            half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
            row = {"period": period, "plan": plan, "metric": metric, "mean": mean, "ci95_low": mean - half,
                   "ci95_high": mean + half, "seeds": len(values)}
            if plan != baseline and len(base):
                paired = base.join(mine[[metric]], rsuffix="_plan", how="inner")
                change = 100 * (paired[metric + "_plan"].mean() - paired[metric].mean()) / paired[metric].mean()
                test = stats.ttest_rel(paired[metric + "_plan"], paired[metric]) if len(paired) > 1 else None
                row.update({"change_pct_vs_current": float(change),
                            "p_value": float(test.pvalue) if test is not None else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)
