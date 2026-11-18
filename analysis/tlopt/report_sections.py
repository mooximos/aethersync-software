"""The nine report sections. Each reads the saved result files, so every number is traceable."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tlopt import figures
from tlopt.report import NAMES, _read


def measured(doc, inter, summary, measures: Path) -> None:
    doc.h("1. What was measured")
    days = summary["days"]
    weekdays = ", ".join(summary.get("weekdays", []))
    doc.p(f"Cameras recorded every vehicle on all four approaches on {len(days)} day(s) "
          f"({', '.join(days)}; days of the week: {weekdays}). Study periods: "
          + "; ".join(f"{k} {v}" for k, v in summary["periods"].items()) + f" ({summary['time_zone']} local time). "
          f"{summary['vehicles']} vehicles were tracked; {summary['counted']} crossed the stop line and are counted "
          f"as traffic; {summary['in_delay_stats']} have a complete track and are used for delay statistics. "
          "Nobody counted anything by hand: every number comes from the timestamped rows.")
    if summary.get("days_left_out"):
        doc.p("Days in the data folder that were left out on purpose (options.days and skip_dates in "
              "intersection.yaml): " + ", ".join(summary["days_left_out"]) + ".")
    for name, reason in summary.get("periods_skipped", {}).items():
        doc.p(f"Period {name} was not studied: {reason}.")
    if summary["problems"]:
        doc.p(f"The data check found {len(summary['problems'])} problem(s) in the files; rows that failed the "
              "check were left out. First ones: " + "; ".join(summary["problems"][:5]))
    else:
        doc.p("Every data file passed the format check.")
    exclusions = _read(measures / "exclusions.csv")
    if exclusions is not None:
        doc.table(exclusions, "Vehicles left out, and why")
    quality = _read(measures / "quality.csv")
    if quality is not None:
        doc.table(quality, "Data quality per day, approach and hour (fps = processed camera frames per second; "
                           "outages = gaps of over 2 minutes in recording during a scheduled window)")
    checks = []
    det = _read(measures / "detector_check.csv")
    if det is not None:
        checks.append(f"Vehicles crossing vs the virtual stop-line detector: {int(det['within_3pct'].sum())} of "
                      f"{len(det)} lanes agree within 3% (largest difference {det['difference_pct'].abs().max():.1f}%).")
    lost = _read(measures / "entering_vs_crossing.csv")
    if lost is not None:
        share = 100 * lost["lost"].sum() / max(lost["entered"].sum(), 1)
        checks.append(f"Vehicles that entered the view but were not seen crossing (lost by tracking, or still "
                      f"waiting when recording stopped): {share:.1f}% overall.")
    cyc = _read(measures / "cycle_stability.csv")
    if cyc is not None and len(cyc):
        checks.append(f"Cycle lengths from the inferred green starts: median {cyc['median_cycle_s'].median():g} s, "
                      f"spread (IQR) at most {cyc['iqr_s'].max():g} s on any approach and day.")
    doc.bullets(checks)


def traffic(doc, periods, measures: Path, figs: Path) -> None:
    doc.h("2. Current traffic")
    rows = []
    for period in periods:
        flows = _read(measures / f"{period}_flows.csv")
        doc.table(flows, f"{period}: vehicles per hour by approach (where traffic comes from) and movement "
                         "(L left, T through, R right, U unknown); mean over days, with the lowest and highest day")
        for approach, value in flows.groupby("approach")["flow_vph"].sum().items():
            rows.append({"approach": approach, "period": period, "vehicles per hour": value})
        counts = _read(measures / f"{period}_counts_arriving.csv")
        mix = counts.groupby("type")["vehicles"].sum()
        doc.p(f"{period} vehicle mix: " + ", ".join(f"{t} {100 * n / mix.sum():.1f}%" for t, n in mix.items())
              + ". (The detector's 'truck' class is mostly pickups and box trucks.)")
    chart = figures.grouped_bars(pd.DataFrame(rows), "approach", "period", "vehicles per hour",
                                 dict(zip(periods, figures.COLORS)), "Traffic by approach",
                                 "approach (direction traffic comes from)", "vehicles per hour",
                                 figs / "traffic_by_approach.png")
    doc.image(chart, "Vehicles per hour by approach and period")


def timing(doc, inter, periods, measures: Path, plans_dir: Path) -> None:
    doc.h("3. Current timing")
    doc.p("The green starts are inferred from when the first queued car moves (minus a 1.5 s reaction time); "
          "the green ends from the last car crossing before cars start stopping again.")
    for period in periods:
        table = _read(measures / f"{period}_timing.csv")
        if table is None or not len(table):
            doc.p(f"{period}: the timing could not be inferred (too few queues).")
            continue
        table = table.rename(columns={"cycle_s": "cycle (s)", "cycle_iqr_s": "cycle spread IQR (s)",
                                      "green_s": "inferred green (s)", "green_iqr_s": "green spread IQR (s)"})
        if inter.timing_sheet is not None:
            sheet = {p.approach: inter.timing_sheet.phases[k].green_s for k, p in inter.phases.items() if "T" in p.movements}
            table["timing sheet green (s)"] = table["approach"].map(sheet)
            table["difference (s)"] = (table["inferred green (s)"] - table["timing sheet green (s)"]).round(1)
            table["sheet cycle (s)"] = inter.timing_sheet.cycle_s
        doc.table(table, f"{period}: inferred timing per approach")
        notes = [n for n in table.get("note", pd.Series(dtype=str)).fillna("") if n]
        if notes:
            doc.p(f"Note: {notes[0]}")
    current = json.loads((plans_dir / f"{periods[0]}_current.json").read_text())
    source = "the city's timing sheet" if inter.timing_sheet is not None else "the inferred timing"
    doc.p(f"The model uses {source} as the current plan.")
    if current["notes"]:
        doc.bullets([f"Safety check of the current plan: {n}" for n in current["notes"]])


def waits(doc, periods, measures: Path, figs: Path) -> None:
    doc.h("4. How long people wait now")
    doc.p("Control delay is the extra time a vehicle needs because of the signal: its time in view minus the "
          "time it would need at free-flow speed. Time stopped is the part spent standing still. Level of "
          "service (LOS) is the HCM grade from A (under 10 s) to F (over 80 s).")
    rows = []
    for period in periods:
        table = _read(measures / f"{period}_delay_by_approach.csv")
        doc.table(table, f"{period}: delay per approach (s per vehicle)")
        rows += [{"approach": r.approach, "period": period, "delay": r.mean_delay_s} for r in table.itertuples()]
        doc.table(_read(measures / f"{period}_delay_by_movement.csv"), f"{period}: delay per approach and movement")
        doc.table(_read(measures / f"{period}_delay_by_day.csv"), f"{period}: the same per day, to show day-to-day variation")
        queues = _read(measures / f"{period}_queues.csv")
        failures = _read(measures / f"{period}_cycle_failures.csv")
        doc.table(queues.merge(failures, on="approach", how="left"),
                  f"{period}: queue per cycle (vehicles in the longest lane; meters to the front of the last "
                  "queued car) and cycle failures (vehicles that waited through two greens); cycles_at_least = "
                  "cycles whose queue reached the edge of the camera view, so the true queue was longer")
    chart = figures.grouped_bars(pd.DataFrame(rows), "approach", "period", "delay", dict(zip(periods, figures.COLORS)),
                                 "Average delay per vehicle now", "approach", "control delay (s per vehicle)",
                                 figs / "delay_now.png")
    doc.image(chart, "Average control delay per approach now")


def model(doc, periods, plans_dir: Path, figs: Path) -> None:
    doc.h("5. Does the model match reality?")
    doc.p("The HCM 6th edition delay model was run with the measured traffic, the measured saturation flows "
          "and the current timing, for every 15 minutes of every day, and compared with the delay measured "
          "for the vehicles that arrived in those 15 minutes.")
    for period in periods:
        check = json.loads((plans_dir / f"{period}_model_check.json").read_text())
        if not check.get("raw"):
            doc.p(f"{period}: not enough data to check the model.")
            continue
        doc.p(f"{period}: {check['verdict']}")
        rows = [{"version": "raw model", **check["raw"]}]
        if check.get("calibrated"):
            rows.append({"version": "with one correction factor per approach", **check["calibrated"]})
        doc.table(pd.DataFrame(rows), f"{period}: model vs measurement (MAE = mean absolute error)")
        doc.p(f"Correction factors {check['factors']}: {check.get('reason', '')}.")
        table = _read(plans_dir / f"{period}_model_check.csv")
        doc.image(figures.model_scatter(table, f"{period}: predicted vs measured delay",
                                        figs / f"model_check_{period}.png"),
                  f"{period}: predicted vs measured delay per 15 minutes")


def plans(doc, inter, periods, plans_dir: Path, figs: Path) -> None:
    doc.h("6. The recommended plan")
    tod_path = plans_dir / "time_of_day.json"
    if tod_path.exists():
        tod = json.loads(tod_path.read_text())
        doc.p(("Separate morning and evening plans are recommended: " if tod["separate"] else
               f"One plan for all periods is enough (the {tod.get('one_plan_from')} plan): ") + tod["reason"] + ".")
    for period in periods:
        data = {label: json.loads((plans_dir / f"{period}_{label.lower()}.json").read_text())
                for label in ("Current", "Webster", "Optimized")}
        rows = []
        for phase, spec in data["Optimized"]["phases"].items():
            rows.append({"phase": phase, "serves": spec["movements"],
                         "current green (s)": data["Current"]["phases"][phase]["green_s"],
                         "Webster green (s)": data["Webster"]["phases"][phase]["green_s"],
                         "recommended green (s)": spec["green_s"], "yellow (s)": spec["yellow_s"],
                         "all-red (s)": spec["all_red_s"], "minimum green (s)": spec["min_green_s"]})
        rows.append({"phase": "cycle", "serves": "", "current green (s)": data["Current"]["cycle_s"],
                     "Webster green (s)": data["Webster"]["cycle_s"], "recommended green (s)": data["Optimized"]["cycle_s"]})
        doc.table(pd.DataFrame(rows), f"{period}: timing plans (yellow and all-red from the ITE formulas; minimum "
                                      "green includes the pedestrian crossing time)")
        notes = data["Optimized"]["notes"] + [f"Webster: {n}" for n in data["Webster"]["notes"]]
        if notes:
            doc.bullets(notes)
        curve = _read(plans_dir / f"{period}_cycle_curve.csv")
        if curve is not None and len(curve):
            doc.image(figures.cycle_curve(curve, data["Optimized"]["cycle_s"], f"{period}: best delay for each cycle length",
                                          figs / f"cycle_curve_{period}.png"), "Delay against cycle length")


def change(doc, periods, plans_dir: Path, sumo: Path, figs: Path) -> None:
    doc.h("7. Predicted and simulated change")
    rows = []
    for period in periods:
        base = None
        for label in ("Current", "Webster", "Optimized"):
            pred = json.loads((plans_dir / f"{period}_{label.lower()}.json").read_text())["predicted"]
            base = base or pred["avg_delay_s"]
            rows.append({"period": period, "plan": label, "model delay (s/veh)": pred["avg_delay_s"],
                         "change vs current": f"{100 * (pred['avg_delay_s'] - base) / base:+.0f}%", "LOS": pred["LOS"],
                         "largest X": pred["max_X"], "estimated longest wait (s)": max(pred["est_max_wait_s"].values())})
    doc.table(pd.DataFrame(rows), "The HCM model's prediction for each plan (X = demand / capacity of the busiest lane group)")
    summary = _read(sumo / "summary.csv")
    if summary is None:
        doc.p("The SUMO simulation has not been run yet (python -m tlopt simulate).")
        return
    calibration = json.loads((sumo / "calibration.json").read_text())
    doc.p(f"An independent traffic simulation (SUMO) ran each plan with the measured traffic, "
          f"{int(summary['seeds'].max())} times with different random arrivals (the same ones for every plan), after "
          f"a 15-minute warm-up. Drivers were calibrated to the measured saturation flow "
          f"({calibration['target_sat_flow_per_lane']} veh/h per lane; tau = {calibration['tau_s']} s"
          + ("" if calibration["within_5pct"] else "; NOT within 5% after 10 tries") + ").")
    show = summary.copy()
    show["measure"] = show["metric"].map(NAMES)
    show["mean [95% CI]"] = show.apply(lambda r: f"{r['mean']:.1f} [{r['ci95_low']:.1f}, {r['ci95_high']:.1f}]", axis=1)
    show["change vs current"] = show["change_pct_vs_current"].map(lambda x: "" if pd.isna(x) else f"{x:+.1f}%")
    show["p-value"] = show["p_value"].map(lambda x: "" if pd.isna(x) else ("< 0.001" if x < 0.001 else f"{x:.3f}"))
    for period in periods:
        part = show[show["period"] == period]
        doc.table(part[["plan", "measure", "mean [95% CI]", "change vs current", "p-value"]],
                  f"{period}: SUMO results per vehicle (paired t-test against the current plan)")
    for metric, title in (("delay_s", "Simulated delay"), ("co2_g", "Simulated CO2")):
        part = summary[summary["metric"] == metric]
        doc.image(figures.grouped_bars(part, "period", "plan", "mean", figures.PLAN_COLORS, title + " (95% CI)",
                                       "period", NAMES[metric], figs / f"sumo_{metric}.png", ("ci95_low", "ci95_high")),
                  title)
    check = _read(sumo / "sim_check.csv")
    if check is not None:
        doc.table(check, "Does the simulation reproduce today? The current plan in SUMO vs the measurements")


def robust(doc, periods, plans_dir: Path) -> None:
    doc.h("8. Robustness")
    doc.p("Traffic changes from week to week, and the measured saturation flows have some error. The plan was "
          "re-optimized with 10% more and less traffic and 10% higher and lower saturation flow, to see how much "
          "the recommended plan's delay changes and whether it stays close (within 5%) to each case's own best plan.")
    for period in periods:
        doc.table(_read(plans_dir / f"{period}_robustness.csv"), f"{period}: robustness")


def limits(doc, inter, summary, measures: Path, plans_dir: Path, sumo: Path) -> None:
    doc.h("9. Limitations")
    items = [
        "The HCM model and the SUMO simulation are simplified pictures of real traffic; they are used to compare "
        "plans with the same assumptions, not to promise exact seconds.",
        "Only daylight data is used" + (" (daylight_only is on); night and twilight vehicles are left out."
                                        if inter.options.daylight_only else " is NOT required; night data is included."),
        "The detector's 'truck' class is mostly pickups and box trucks; SUMO models them as light rigid trucks.",
        "Queues longer than the camera view are reported as 'at least'; the delay of a car that first stopped "
        "beyond the view is a lower bound.",
        "Green times are inferred from when cars move and stop. They depend on a fixed 1.5 s reaction time, and on "
        "some cars still crossing at the end of green; the timing sheet, when given, is used instead.",
        "Pedestrians are not measured; their volumes and walking speed come from intersection.yaml.",
        "SUMO's left turners yield to oncoming traffic; on approaches with heavy oncoming traffic this makes "
        "simulated delays higher than measured (see the simulation check table).",
        "Recommended yellow and all-red times follow the ITE formulas; a traffic engineer must confirm them.",
    ]
    for period in summary["periods"]:
        sat = _read(measures / f"{period}_saturation_flow.csv")
        if sat is not None and (sat["source"] == "default").any():
            groups = ", ".join(f"{r.approach} {r.group}" for r in sat[sat["source"] == "default"].itertuples())
            items.append(f"{period}: fewer than 30 measured headways for {groups}; the textbook saturation flow "
                         "(1800 veh/h per lane, adjusted for heavy vehicles) was used there.")
    doc.bullets(items)
