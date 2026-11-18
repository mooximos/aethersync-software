"""Does the HCM model match reality? Predicted vs measured control delay per approach per 15 min."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tlopt import hcm
from tlopt.load import local_times

MIN_VEHICLES = 5


def intervals(v: pd.DataFrame, zone: str, groups: list[hcm.GroupInput], greens: dict[int, float],
              cycle: float) -> pd.DataFrame:
    """One row per day x approach x 15-min interval (by arrival): measured mean control delay of the
    vehicles arriving then, and the model's delay for that interval's own flow rates."""
    arrived = v[v["t_entered_s"].notna()].copy()
    arrived["interval"] = local_times(arrived["t_entered_s"], zone).dt.floor("15min").dt.strftime("%H:%M")
    counts = arrived.groupby(["day", "interval", "approach", "group"]).size()
    measured = arrived[arrived["delay_ok"] & arrived["control_delay_s"].notna()].groupby(
        ["day", "interval", "approach"])["control_delay_s"].agg(["mean", "size"])
    rows = []
    for (day, interval, approach), m in measured.iterrows():
        if m["size"] < MIN_VEHICLES:
            continue
        mine = [g for g in groups if g.approach == approach]
        flows = [4.0 * counts.get((day, interval, approach, g.name), 0) for g in mine]
        total = sum(flows)
        if total == 0:
            continue
        delay = sum(f * hcm.lane_group_delay(f, g.s, sum(greens[p] for p in g.phases), cycle, g.rp)["d"]
                    for f, g in zip(flows, mine) if f > 0) / total
        rows.append({"day": day, "interval": interval, "approach": approach, "vehicles": int(m["size"]),
                     "measured_s": float(m["mean"]), "predicted_s": delay})
    return pd.DataFrame(rows, columns=["day", "interval", "approach", "vehicles", "measured_s", "predicted_s"])


def scores(measured: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - measured
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((measured - measured.mean()) ** 2)) or np.nan
    return {"n": int(len(measured)), "MAE_s": round(float(np.mean(np.abs(error))), 2),
            "MAPE_pct": round(float(np.mean(np.abs(error) / measured) * 100), 1),
            "R2": round(1 - ss_res / ss_tot, 3), "bias_s": round(float(np.mean(error)), 2)}


def calibrate(table: pd.DataFrame) -> dict:
    """One multiplicative correction per approach (least squares), kept only if it lowers the
    leave-one-day-out error. Returns raw and calibrated scores and the factors used."""
    days = sorted(table["day"].unique())
    raw = scores(table["measured_s"].to_numpy(), table["predicted_s"].to_numpy())
    fit = lambda rows: float((rows["measured_s"] * rows["predicted_s"]).sum() / (rows["predicted_s"] ** 2).sum())  # noqa: E731
    factors = {a: fit(rows) for a, rows in table.groupby("approach")}
    out = {"raw": raw, "factors": {a: round(f, 3) for a, f in factors.items()}, "used": False}
    if len(days) < 2:
        out["reason"] = "only one day of data: leave-one-day-out is not possible, so no correction is used"
        return out
    held_raw, held_cal = [], []
    for day in days:
        train, test = table[table["day"] != day], table[table["day"] == day]
        for approach, rows in test.groupby("approach"):
            fit_rows = train[train["approach"] == approach]
            factor = fit(fit_rows) if len(fit_rows) else 1.0
            held_raw += list(np.abs(rows["predicted_s"] - rows["measured_s"]))
            held_cal += list(np.abs(factor * rows["predicted_s"] - rows["measured_s"]))
    out["lodo_MAE_raw_s"] = round(float(np.mean(held_raw)), 2)
    out["lodo_MAE_calibrated_s"] = round(float(np.mean(held_cal)), 2)
    out["used"] = out["lodo_MAE_calibrated_s"] < out["lodo_MAE_raw_s"]
    calibrated = table["predicted_s"] * table["approach"].map(factors)
    out["calibrated"] = scores(table["measured_s"].to_numpy(), calibrated.to_numpy())
    out["reason"] = ("the correction lowers the leave-one-day-out error, so it is used" if out["used"] else
                     "the correction does not lower the leave-one-day-out error, so it is not used")
    return out


def verdict(result: dict) -> str:
    best = result["calibrated"] if result.get("used") else result["raw"]
    mape = best["MAPE_pct"]
    if mape <= 15:
        quality = "matches the measured delays well"
    elif mape <= 30:
        quality = "matches the measured delays only roughly"
    else:
        quality = "does NOT match the measured delays well; treat its predictions with caution"
    return (f"The model {quality}: average error {best['MAE_s']} s ({mape}% of the measured delay, "
            f"bias {best['bias_s']:+} s) over {best['n']} 15-minute intervals. R2 = {best['R2']} "
            "(R2 is low when the delays hardly change from one 15-minute interval to the next).")
