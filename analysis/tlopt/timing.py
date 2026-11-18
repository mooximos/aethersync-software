"""The light's current timing, inferred from green_start_inferred / red_inferred_by and crossings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tlopt.intersection import Intersection

ACTUATED_IQR_S = 10.0


@dataclass
class ApproachTiming:
    approach: str
    cycle_s: float
    cycle_iqr_s: float
    green_s: float           # effective green, median over cycles
    green_iqr_s: float
    cycles: int
    note: str = ""


def green_intervals(signals: pd.DataFrame, vehicles: pd.DataFrame, approach: str,
                    sat_headway_s: float) -> pd.DataFrame:
    """One row per cycle: green start (inferred) and green end =
    last t_crossed before the next red_inferred_by + half a saturation headway."""
    rows = []
    for day, sig in signals[signals["approach"] == approach].groupby("day"):
        greens = np.sort(sig.loc[sig["event_code"] == 1, "time_s"].to_numpy())
        reds = np.sort(sig.loc[sig["event_code"] == 10, "time_s"].to_numpy())
        crossed = np.sort(vehicles.loc[(vehicles["approach"] == approach) & (vehicles["day"] == day),
                                       "t_crossed_s"].dropna().to_numpy())
        for i, start in enumerate(greens):
            nxt = greens[i + 1] if i + 1 < len(greens) else np.inf
            red = reds[(reds > start) & (reds < nxt)]
            if not len(red):
                continue
            last = crossed[(crossed > start) & (crossed < red[0])]
            if not len(last):
                continue
            rows.append({"day": day, "green_start_s": start, "green_end_s": last[-1] + sat_headway_s / 2,
                         "red_by_s": red[0], "next_green_s": nxt})
    out = pd.DataFrame(rows, columns=["day", "green_start_s", "green_end_s", "red_by_s", "next_green_s"])
    out["green_s"] = out["green_end_s"] - out["green_start_s"]
    return out


def infer(signals: pd.DataFrame, vehicles: pd.DataFrame, approach: str, sat_headway_s: float) -> ApproachTiming | None:
    sig = signals[signals["approach"] == approach]
    gaps = []
    for _day, group in sig[sig["event_code"] == 1].groupby("day"):
        gaps += list(np.diff(np.sort(group["time_s"].to_numpy())))
    cycles = green_intervals(signals, vehicles, approach, sat_headway_s)
    if len(gaps) < 3 or len(cycles) < 3:
        return None
    c_q1, cycle, c_q3 = np.percentile(gaps, [25, 50, 75])
    g_q1, green, g_q3 = np.percentile(cycles["green_s"], [25, 50, 75])
    note = ""
    if c_q3 - c_q1 > ACTUATED_IQR_S:
        note = ("cycle length varies a lot (IQR over 10 s): the signal is probably actuated or "
                "coordinated; use the city's timing sheet")
    return ApproachTiming(approach, round(cycle, 1), round(c_q3 - c_q1, 1), round(green, 1),
                          round(g_q3 - g_q1, 1), len(cycles), note)


def current_plan(inter: Intersection, inferred: dict[str, ApproachTiming], safety: dict[int, tuple[float, float]]) -> dict:
    """The plan the model uses for 'current': the timing sheet if there is one, else the inferred timing
    (yellow and all-red then come from the ITE formulas, because the cameras cannot see them)."""
    phases = {}
    if inter.timing_sheet is not None:
        for key, sheet in inter.timing_sheet.phases.items():
            phases[int(key)] = (sheet.green_s, sheet.yellow_s, sheet.all_red_s)
        return {"source": "timing sheet", "cycle_s": inter.timing_sheet.cycle_s, "phases": phases}
    cycles = [t.cycle_s for t in inferred.values() if t is not None]
    if not cycles:
        raise ValueError("no timing sheet and the timing could not be inferred from the data")
    for number in inter.phase_numbers():
        phase = inter.phases[str(number)]
        timing = inferred.get(phase.approach)
        yellow, all_red = safety[number]
        green = timing.green_s if timing is not None else 7.0
        phases[number] = (green, yellow, all_red)
    return {"source": "inferred from the data", "cycle_s": float(np.median(cycles)), "phases": phases}
