"""Data quality: per-hour table, consistency checks (no hand counting), and exclusions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tlopt.load import Dataset, camera_of, local_times

# Left out of delay statistics (still counted as traffic if they crossed).
DELAY_FLAGS = ["no_cross", "short_track", "id_switch_suspect", "started_in_view"]
OUTAGE_S = 120


def mark_exclusions(ds: Dataset, daylight_only: bool, days: list[str] | None = None,
                    skip_dates: list[str] | None = None) -> pd.DataFrame:
    """Adds columns: excluded (left out of everything), counted, delay_ok and a short exclusion reason.
    Returns the counts per rule."""
    v = ds.vehicles
    days = days or ["Mon", "Tue", "Wed", "Thu", "Fri"]
    skipped = v["day"].isin([str(d) for d in (skip_dates or [])])
    other_day = ~v["weekday"].isin(days) & ~skipped
    dark = ((v["light"] != "day") if daylight_only else pd.Series(False, index=v.index)) & ~skipped & ~other_day
    v["excluded"] = skipped | other_day | dark
    reason = pd.Series("", index=v.index)
    for flag in DELAY_FLAGS:
        reason = reason.where(reason != "", np.where(v["flag_" + flag], flag, ""))
    reason = reason.where(~dark, "not daylight (" + v["light"] + ")")
    reason = reason.where(~other_day, "not a study day (" + v["weekday"] + ")")
    reason = reason.where(~skipped, "date in skip_dates")
    v["exclusion"] = reason
    v["counted"] = ~v["excluded"] & v["t_crossed_s"].notna()
    v["delay_ok"] = v["exclusion"] == ""
    rows = [{"rule": "date listed in skip_dates", "vehicles": int(skipped.sum()), "effect": "left out of everything"},
            {"rule": f"day of week not in days ({', '.join(days)})", "vehicles": int(other_day.sum()),
             "effect": "left out of everything"},
            {"rule": "not in daylight (daylight_only)", "vehicles": int(dark.sum()),
             "effect": "left out of everything"}]
    for flag in DELAY_FLAGS:
        n = int((v["flag_" + flag] & ~v["excluded"]).sum())
        rows.append({"rule": f"flag {flag}", "vehicles": n,
                     "effect": "left out of delay statistics; counted as traffic if it crossed"})
    kept = {"queue_past_view": "kept; its wait is a lower bound (the queue went past the view)",
            "low_conf": "kept", "cut_at_shutdown": "kept"}
    for flag, effect in kept.items():
        rows.append({"rule": f"flag {flag}", "vehicles": int(v["flag_" + flag].sum()), "effect": effect})
    rows.append({"rule": "turn U (unknown)", "vehicles": int((v["turn"] == "U").sum()),
                 "effect": "kept; counted in its lane group"})
    return pd.DataFrame(rows)


def quality_table(ds: Dataset) -> pd.DataFrame:
    v = ds.vehicles.copy()
    v["hour"] = local_times(v["t_left_s"], ds.zone).dt.hour
    pct = lambda s: round(100 * float(s.mean()), 1) if len(s) else np.nan  # noqa: E731
    table = v.groupby(["day", "approach", "hour"]).agg(
        rows=("vehicle_id", "size"),
        pct_low_conf=("flag_low_conf", pct), pct_no_cross=("flag_no_cross", pct),
        pct_started_in_view=("flag_started_in_view", pct),
        pct_U_turn=("turn", lambda s: pct(s == "U"))).reset_index()
    sysrows = ds.system.copy()
    if len(sysrows):
        sysrows["hour"] = local_times(sysrows["time_s"], ds.zone).dt.hour
        sysrows = sysrows.sort_values("time_s")
        sysrows["gap"] = sysrows.groupby("day")["time_s"].diff()
        extra = []
        for (day, approach, hour) in table[["day", "approach", "hour"]].itertuples(index=False):
            rows = sysrows[(sysrows.day == day) & (sysrows.hour == hour)]
            camera = camera_of(ds.sites[day], approach)
            fps = rows.get(f"fps_{camera}") if camera else None
            gaps = rows[rows["gap"] > OUTAGE_S]
            outages = sum(_in_window(ds.sites[day], ds.zone, end - gap, end) for end, gap in zip(gaps["time_s"], gaps["gap"]))
            extra.append({"min_fps": None if fps is None or fps.isna().all() else float(fps.min()),
                          "dropped_frames": int(rows["dropped_frames"].sum()) if len(rows) else None,
                          "min_battery_v": None if rows["battery_v"].isna().all() else float(rows["battery_v"].min()),
                          "outages": int(outages)})
        table = pd.concat([table, pd.DataFrame(extra)], axis=1)
    return table


def _in_window(site: dict, zone: str, start_s: float, end_s: float) -> bool:
    """True if the gap (start_s, end_s) covers some recording-window time: then it is an outage,
    not the device sleeping between windows."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(zone)
    first = datetime.fromtimestamp(start_s, tz)
    for window in site["schedule"]["windows"]:
        if first.strftime("%a") not in window["days"]:
            continue
        h0, m0 = map(int, window["start"].split(":"))
        h1, m1 = map(int, window["end"].split(":"))
        w0 = first.replace(hour=h0, minute=m0, second=0, microsecond=0).timestamp()
        w1 = first.replace(hour=h1, minute=m1, second=0, microsecond=0).timestamp()
        if start_s + OUTAGE_S < w1 and end_s > w0 + OUTAGE_S and min(end_s, w1) - max(start_s, w0) > OUTAGE_S:
            return True
    return False


def detector_vs_crossings(ds: Dataset) -> pd.DataFrame:
    """Check 1: vehicles with t_crossed vs detector_on events, per day and lane (should agree within 3%)."""
    crossed = ds.vehicles[ds.vehicles["t_crossed_s"].notna()].groupby(["day", "approach", "lane"]).size()
    ons = ds.signals[ds.signals["event_code"] == 82].groupby(["day", "approach", "lane"]).size()
    table = pd.DataFrame({"crossed": crossed, "detector_on": ons}).fillna(0).astype(int).reset_index()
    table["difference_pct"] = (100 * (table["detector_on"] - table["crossed"]) / table["crossed"].clip(lower=1)).round(1)
    table["within_3pct"] = table["difference_pct"].abs() <= 3.0
    return table


def entering_vs_crossing(ds: Dataset) -> pd.DataFrame:
    """Check 2: vehicles entering vs crossing per approach and hour; the difference = lost by tracking."""
    v = ds.vehicles.copy()
    v["hour"] = local_times(v["t_entered_s"].fillna(v["t_left_s"]), ds.zone).dt.hour
    table = v.groupby(["day", "approach", "hour"]).agg(
        entered=("t_entered_s", lambda s: int(s.notna().sum())),
        crossed=("t_crossed_s", lambda s: int(s.notna().sum()))).reset_index()
    table["lost"] = table["entered"] - table["crossed"]
    table["lost_pct"] = (100 * table["lost"] / table["entered"].clip(lower=1)).round(1)
    return table


def cycle_stability(ds: Dataset) -> pd.DataFrame:
    """Check 3: cycle lengths from green_start_inferred, per day and approach."""
    rows = []
    greens = ds.signals[ds.signals["event_code"] == 1].sort_values("time_s")
    for (day, approach), group in greens.groupby(["day", "approach"]):
        gaps = np.diff(group["time_s"].to_numpy())
        if len(gaps) < 3:
            continue
        q1, median, q3 = np.percentile(gaps, [25, 50, 75])
        rows.append({"day": day, "approach": approach, "greens": len(group), "median_cycle_s": round(median, 1),
                     "iqr_s": round(q3 - q1, 1), "share_within_2s_of_median": round(float(np.mean(abs(gaps - median) <= 2)), 2)})
    return pd.DataFrame(rows)


MIN_COVERAGE = 0.9   # a camera must have worked for 90% of a period's minutes for that day to count


def coverage(ds: Dataset, inter, period: str) -> pd.DataFrame:
    """Per study day x approach: did that approach's camera really record the whole period?

    A day where a camera was unplugged, or the Pi was off for part of the period, would otherwise
    look like a day with little traffic and pull the averages down. Uses the per-minute system rows
    (frames per second of the camera) and whether any vehicle was seen from that approach."""
    from tlopt.measures import period_mask
    start, end = inter.period_minutes(period)
    minutes = end - start
    v = ds.vehicles
    seen = v[period_mask(v["t_entered_s"], ds.zone, inter, period)].groupby(["day", "approach"]).size()
    system = ds.system[period_mask(ds.system["time_s"], ds.zone, inter, period)] if len(ds.system) else ds.system
    rows = []
    for day in ds.used_days:
        sys_day = system[system["day"] == day] if len(system) else system
        for approach in inter.approaches:
            camera = camera_of(ds.sites[day], approach)
            vehicles = int(seen.get((day, approach), 0))
            if len(sys_day) and camera and f"fps_{camera}" in sys_day:
                working = int((pd.to_numeric(sys_day[f"fps_{camera}"], errors="coerce") > 0).sum())
                share = working / minutes
                why = "" if share >= MIN_COVERAGE else f"camera worked {working} of {minutes} minutes"
            else:   # no system rows for that day: judge by the vehicles alone
                working, why = None, ""
            if camera is None:
                why = "no camera for this approach in site.json"
            elif vehicles == 0:
                why = why or "no vehicles seen from this approach"
            rows.append({"day": day, "approach": approach, "camera": camera or "", "minutes_working": working,
                         "vehicles": vehicles, "used": why == "", "why_not": why})
    return pd.DataFrame(rows, columns=["day", "approach", "camera", "minutes_working", "vehicles", "used", "why_not"])
