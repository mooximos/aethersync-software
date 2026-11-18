"""Traffic measures for one period: demand, delay, queues, saturation flow, arrival pattern."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tlopt.intersection import Intersection
from tlopt.load import Dataset, local_times

MIN_HEADWAYS = 30
DEFAULT_SAT_FLOW = 1800.0   # veh/h/lane, before the heavy-vehicle adjustment
FFS_MIN_VEHICLES = 20


@dataclass
class LaneGroup:
    approach: str
    name: str
    lanes: list[int]
    movements: list[str]


def lane_groups(inter: Intersection) -> list[LaneGroup]:
    """HCM lane groups: exclusive left lanes, exclusive right lanes, and the rest (with the through lanes)."""
    groups = []
    for approach, geo in inter.approaches.items():
        buckets: dict[str, list[int]] = {}
        for lane, moves in sorted(geo.lanes.items(), key=lambda kv: int(kv[0])):
            key = "L" if moves == ["L"] else "R" if moves == ["R"] else "main"
            buckets.setdefault(key, []).append(int(lane))
        for key, lanes in buckets.items():
            moves = sorted({m for lane in lanes for m in geo.lanes[str(lane)]}, key="LTR".index)
            groups.append(LaneGroup(approach, "".join(moves) if key == "main" else key, lanes, moves))
    return groups


def group_of(v: pd.DataFrame, groups: list[LaneGroup]) -> pd.Series:
    """Each vehicle's lane group: by its lane, or by its turn if the lane is unknown."""
    by_lane = {(g.approach, lane): g.name for g in groups for lane in g.lanes}
    main = {g.approach: g.name for g in groups if g.name not in ("L", "R")}
    by_turn = {(g.approach, g.name): g.name for g in groups if g.name in ("L", "R")}

    def pick(row) -> str:
        if pd.notna(row.lane) and (row.approach, int(row.lane)) in by_lane:
            return by_lane[(row.approach, int(row.lane))]
        return by_turn.get((row.approach, row.turn), main.get(row.approach, "?"))
    return v.apply(pick, axis=1)


def los(delay_s: float) -> str:
    """HCM level of service for signalized intersections, from mean control delay."""
    for limit, grade in ((10, "A"), (20, "B"), (35, "C"), (55, "D"), (80, "E")):
        if delay_s <= limit:
            return grade
    return "F"


def add_vehicle_delays(ds: Dataset) -> None:
    """Per vehicle, from its track: path length, free-flow speed, control delay, arrival time at the line.

    control delay = (last - first sample time) - path length / free-flow speed, where the free-flow
    speed is the 85th percentile speed of vehicles that never stopped (per approach and movement when
    there are at least 20 of them, else per approach)."""
    v = ds.vehicles
    t = ds.tracks.dropna(subset=["x_m", "y_m"]).sort_values(["key", "time_s"])
    step = np.hypot(t.groupby("key")["x_m"].diff(), t.groupby("key")["y_m"].diff()).fillna(0)
    per = t.assign(step=step).groupby("key").agg(first_s=("time_s", "first"), last_s=("time_s", "last"),
                                                 path_m=("step", "sum"), entry_y=("y_m", "first"))
    per["mean_speed"] = per["path_m"] / (per["last_s"] - per["first_s"]).replace(0, np.nan)
    v.drop(columns=[c for c in per.columns if c in v.columns], inplace=True)
    merged = v.join(per, on="key")
    for column in per.columns:
        v[column] = merged[column]
    free = v[(v["stops"] == 0) & v["delay_ok"] & v["mean_speed"].notna()]
    by_move = free.groupby(["approach", "turn"])["mean_speed"].agg(["size", lambda s: s.quantile(0.85)])
    by_approach = free.groupby("approach")["mean_speed"].quantile(0.85)

    def ffs(row) -> float:
        key = (row.approach, row.turn)
        if key in by_move.index and by_move.loc[key, "size"] >= FFS_MIN_VEHICLES:
            return float(by_move.loc[key].iloc[1])
        return float(by_approach.get(row.approach, np.nan))
    v["ffs_mps"] = v.apply(ffs, axis=1)
    v["control_delay_s"] = ((v["last_s"] - v["first_s"]) - v["path_m"] / v["ffs_mps"]).clip(lower=0)
    v["arrive_line_s"] = v["t_entered_s"] + v["entry_y"] / v["ffs_mps"]


def period_mask(epoch: pd.Series, zone: str, inter: Intersection, period: str) -> pd.Series:
    start, end = inter.period_minutes(period)
    local = local_times(epoch, zone)
    minute = local.dt.hour * 60 + local.dt.minute
    return epoch.notna() & (minute >= start) & (minute < end)


def demand(v: pd.DataFrame, zone: str, by: str) -> pd.DataFrame:
    """15-min counts per day x approach x movement x type, by t_crossed ('served') or t_entered ('arriving')."""
    column = {"served": "t_crossed_s", "arriving": "t_entered_s"}[by]
    rows = v[v[column].notna()].copy()
    local = local_times(rows[column], zone)
    rows["interval"] = local.dt.floor("15min").dt.strftime("%H:%M")
    return (rows.groupby(["day", "interval", "approach", "turn", "type"]).size()
            .rename("vehicles").reset_index())


def flow_summary(counts: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Per approach x movement: flow rate (veh/h, mean over days), peak hour factor, heavy-vehicle %."""
    total = counts.groupby(["day", "approach", "turn"])["vehicles"].sum()
    heavy = counts[counts["type"].isin(["truck", "bus"])].groupby(["day", "approach", "turn"])["vehicles"].sum()
    peak = counts.groupby(["day", "approach", "turn", "interval"])["vehicles"].sum().groupby(level=[0, 1, 2]).max()
    table = pd.DataFrame({"total": total, "heavy": heavy, "peak15": peak}).fillna(0)
    table["flow_vph"] = table["total"] / hours
    table["phf"] = (table["flow_vph"] / (4 * table["peak15"]).replace(0, np.nan)).clip(upper=1)
    out = table.groupby(level=[1, 2]).agg(flow_vph=("flow_vph", "mean"), flow_vph_min=("flow_vph", "min"),
                                          flow_vph_max=("flow_vph", "max"), phf=("phf", "mean"),
                                          heavy=("heavy", "sum"), total=("total", "sum")).reset_index()
    out["heavy_pct"] = (100 * out["heavy"] / out["total"].clip(lower=1)).round(1)
    return out.drop(columns=["heavy", "total"]).round({"flow_vph": 0, "flow_vph_min": 0, "flow_vph_max": 0, "phf": 2})


def delay_stats(v: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    ok = v[v["delay_ok"] & v["control_delay_s"].notna()]
    table = ok.groupby(by).agg(vehicles=("control_delay_s", "size"), mean_delay_s=("control_delay_s", "mean"),
                               median_delay_s=("control_delay_s", "median"),
                               p95_delay_s=("control_delay_s", lambda s: s.quantile(0.95)),
                               max_delay_s=("control_delay_s", "max"), mean_wait_s=("wait_s", "mean"),
                               max_wait_s=("wait_s", "max")).reset_index()
    table["LOS"] = table["mean_delay_s"].map(los)
    return table.round(1)


def saturation_flow(v: pd.DataFrame, groups: list[LaneGroup]) -> pd.DataFrame:
    """HCM field method: headways between consecutive queued vehicles crossing the stop line in the
    same lane, from the 5th queued vehicle on. s = 3600 / median headway (per lane), summed per group."""
    rows = []
    for group in groups:
        sub = v[(v["approach"] == group.approach) & v["t_crossed_s"].notna()]
        heavy = float(sub[sub["lane"].isin(group.lanes)]["type"].isin(["truck", "bus"]).mean() or 0)
        total, used_default, headways_all = 0.0, False, 0
        for lane in group.lanes:
            lane_rows = sub[sub["lane"] == lane].sort_values("t_crossed_s")
            headways = []
            for _day, day_rows in lane_rows.groupby("day"):
                times = day_rows["t_crossed_s"].to_numpy()
                pos = day_rows["queue_pos"].to_numpy()
                for k in range(1, len(times)):
                    if pos[k] >= 5 and pos[k - 1] == pos[k] - 1 and times[k] - times[k - 1] < 10:
                        headways.append(times[k] - times[k - 1])
            headways_all += len(headways)
            if len(headways) >= MIN_HEADWAYS:
                total += 3600 / float(np.median(headways))
            else:
                total += DEFAULT_SAT_FLOW / (1 + heavy)
                used_default = True
        rows.append({"approach": group.approach, "group": group.name, "lanes": len(group.lanes),
                     "headways": headways_all, "sat_flow_vph": round(total),
                     "source": "default" if used_default else "measured", "heavy_share": round(heavy, 3)})
    return pd.DataFrame(rows)


def cycle_failures(v: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Vehicles that stayed stopped through two inferred green starts."""
    rows = []
    for approach, group in v[v["t_stopped_s"].notna()].groupby("approach"):
        greens = np.sort(signals.loc[(signals["approach"] == approach) & (signals["event_code"] == 1), "time_s"].to_numpy())
        failed = [int(((greens > a) & (greens < b)).sum() >= 2) for a, b in zip(group["t_stopped_s"], group["t_started_s"])]
        rows.append({"approach": approach, "stopped_vehicles": len(group), "cycle_failures": int(sum(failed))})
    return pd.DataFrame(rows)


def queues(v: pd.DataFrame, tracks: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Queue per cycle (between inferred green starts): the most queued vehicles in any lane, and the
    distance from the stop line to the front of the last one. 'at least' if it reached the view edge."""
    stopped = v[v["t_stopped_s"].notna()].copy()
    if len(tracks):
        t = tracks[["key", "time_s", "y_m"]].dropna().sort_values("time_s")
        stopped = pd.merge_asof(stopped.sort_values("t_stopped_s"), t, left_on="t_stopped_s", right_on="time_s",
                                by="key", direction="nearest", tolerance=1.0)
    else:
        stopped["y_m"] = np.nan
    rows = []
    for (day, approach), group in stopped.groupby(["day", "approach"]):
        greens = np.sort(signals.loc[(signals["day"] == day) & (signals["approach"] == approach)
                                     & (signals["event_code"] == 1), "time_s"].to_numpy())
        group = group.assign(cycle=np.searchsorted(greens, group["t_stopped_s"].to_numpy()))
        for _cycle, c in group.groupby("cycle"):
            rows.append({"day": day, "approach": approach, "queue_vehicles": int(c["queue_pos"].max()),
                         "queue_m": round(float(c["y_m"].max()), 1) if c["y_m"].notna().any() else np.nan,
                         "at_least": bool(c["flag_queue_past_view"].any())})
    cycles = pd.DataFrame(rows, columns=["day", "approach", "queue_vehicles", "queue_m", "at_least"])
    return cycles.groupby("approach").agg(cycles=("queue_vehicles", "size"), mean_queue_veh=("queue_vehicles", "mean"),
                                          max_queue_veh=("queue_vehicles", "max"), mean_queue_m=("queue_m", "mean"),
                                          max_queue_m=("queue_m", "max"),
                                          cycles_at_least=("at_least", "sum")).reset_index().round(1)


def arrival_pattern(v: pd.DataFrame, signals: pd.DataFrame, greens: dict[str, float], cycles: dict[str, float],
                    groups: list[LaneGroup]) -> pd.DataFrame:
    """P = share of vehicles reaching the stop line (free flow) during green; Rp = P / (g/C)."""
    rows = []
    for group in groups:
        green, cycle = greens.get(group.approach), cycles.get(group.approach)
        sub = v[(v["approach"] == group.approach) & (v["group"] == group.name) & v["arrive_line_s"].notna()]
        if green is None or cycle is None or not len(sub):
            continue
        inside = []
        for day, rows_day in sub.groupby("day"):
            starts = np.sort(signals.loc[(signals["day"] == day) & (signals["approach"] == group.approach)
                                         & (signals["event_code"] == 1), "time_s"].to_numpy())
            if not len(starts):
                continue
            arrive = rows_day["arrive_line_s"].to_numpy()
            index = np.searchsorted(starts, arrive, side="right") - 1
            ok = index >= 0
            since = np.where(ok, arrive - starts[np.clip(index, 0, None)], np.inf)
            inside += list(since[ok] <= green)
        if inside:
            share = float(np.mean(inside))
            rows.append({"approach": group.approach, "group": group.name, "vehicles": len(inside),
                         "P": round(share, 3), "g_over_C": round(green / cycle, 3), "Rp": round(share / (green / cycle), 2)})
    return pd.DataFrame(rows)
