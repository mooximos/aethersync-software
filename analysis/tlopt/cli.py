"""python -m tlopt <command> ...

    check    DATA_DIR...   data quality only
    measure  DATA_DIR...   measures + current timing      -> results/measures/
    optimize DATA_DIR...   model check + plans            -> results/plans/
    simulate               SUMO check of the plans        -> results/sumo/
    report                 results/REPORT.html and REPORT.md
    all      DATA_DIR...   everything in order
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tlopt.intersection import IntersectionError, load_intersection


def find_intersection(given: Path | None) -> Path:
    if given:
        return given
    for candidate in (Path("intersection.yaml"), Path("analysis/intersection.yaml")):
        if candidate.exists():
            return candidate
    raise SystemExit("No intersection.yaml found. Copy analysis/intersection.example.yaml to "
                     "analysis/intersection.yaml and edit it, or pass --intersection PATH.")


def _load(data: list[Path], with_tracks: bool = True):
    from tlopt.load import load
    ds = load(data, with_tracks=with_tracks)
    for problem in ds.problems[:30]:
        print(f"  data problem: {problem}")
    if len(ds.problems) > 30:
        print(f"  ... and {len(ds.problems) - 30} more data problems")
    return ds


def cmd_check(args, inter) -> None:
    from tlopt import quality
    ds = _load(args.data, with_tracks=False)
    out = args.results / "measures"
    out.mkdir(parents=True, exist_ok=True)
    tables = {"quality": quality.quality_table(ds), "detector_check": quality.detector_vs_crossings(ds),
              "entering_vs_crossing": quality.entering_vs_crossing(ds), "cycle_stability": quality.cycle_stability(ds)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    print(tables["quality"].to_string(index=False))
    bad = tables["detector_check"][~tables["detector_check"]["within_3pct"]]
    print(f"\nDetector vs crossings: {len(bad)} of {len(tables['detector_check'])} lanes differ by more than 3%.")
    print(f"Tables saved in {out}")
    return ds


def cmd_measure(args, inter, keep_going: bool = False):
    from tlopt import quality, study
    ds = _load(args.data)
    out = args.results / "measures"
    out.mkdir(parents=True, exist_ok=True)
    excluded = study.prepare(ds, inter)
    excluded.to_csv(out / "exclusions.csv", index=False)
    for name, table in {"quality": quality.quality_table(ds), "detector_check": quality.detector_vs_crossings(ds),
                        "entering_vs_crossing": quality.entering_vs_crossing(ds),
                        "cycle_stability": quality.cycle_stability(ds)}.items():
        table.to_csv(out / f"{name}.csv", index=False)
    keep = ["key", "day", "vehicle_id", "approach", "lane", "group", "turn", "type", "t_entered", "t_crossed",
            "wait_s", "stops", "queue_pos", "control_delay_s", "ffs_mps", "exclusion", "file"]
    ds.vehicles[keep].to_csv(out / "vehicle_delays.csv", index=False)
    periods = {}
    for name in inter.periods:
        try:
            periods[name] = study.measure(ds, inter, name)
        except ValueError as error:
            print(f"  period {name}: skipped ({error})")
    summary = {"days": ds.days, "weekdays": sorted(set(ds.vehicles["weekday"])), "vehicles": int(len(ds.vehicles)),
               "counted": int(ds.vehicles["counted"].sum()), "in_delay_stats": int(ds.vehicles["delay_ok"].sum()),
               "periods": {k: inter.periods[k] for k in periods}, "problems": ds.problems,
               "intersection": inter.name, "time_zone": ds.zone}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if not keep_going:
        for p in periods.values():
            study.save(p, args.results)
        print(f"Measured {summary['vehicles']} vehicles on {len(ds.days)} days; tables saved in {out}")
    return ds, periods


def cmd_optimize(args, inter) -> None:
    from tlopt import study
    ds, periods = cmd_measure(args, inter, keep_going=True)
    for p in periods.values():
        study.model(p, ds, inter)
        study.save(p, args.results)
        print(f"\nPeriod {p.name} ({inter.periods[p.name]}):")
        if p.model_check.get("verdict"):
            print(f"  {p.model_check['verdict']}")
        for label, plan in p.plans.items():
            greens = ", ".join(f"phase {q}: {g:g} s" for q, g in sorted(plan.greens.items()))
            print(f"  {label:9s} cycle {plan.cycle_s:g} s; {greens}")
    tod = study.time_of_day(periods)
    (args.results / "plans" / "time_of_day.json").write_text(json.dumps(tod, indent=2))
    print(f"\nTime of day: {'separate plans' if tod['separate'] else 'one plan'} - {tod['reason']}")


def cmd_simulate(args, inter) -> None:
    from tlopt import simulate
    simulate.simulate(inter, args.results, seeds=args.seeds)


def cmd_report(args, inter) -> None:
    from tlopt import report
    html, md = report.build(inter, args.results)
    print(f"Report: {html}\n        {md}")


def main(argv: list[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--intersection", type=Path, help="intersection.yaml (default: ./intersection.yaml or "
                                                           "analysis/intersection.yaml)")
    common.add_argument("--results", type=Path, default=Path("results"), help="output folder (default results)")
    common.add_argument("--seeds", type=int, default=10, help="SUMO runs per plan (default 10)")
    parser = argparse.ArgumentParser(prog="python -m tlopt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "measure", "optimize", "all"):
        sp = sub.add_parser(name, parents=[common])
        sp.add_argument("data", type=Path, nargs="+", help="data folder(s) with YYYY-MM-DD day folders")
    for name in ("simulate", "report"):
        sub.add_parser(name, parents=[common])
    args = parser.parse_args(argv)
    try:
        inter = load_intersection(find_intersection(args.intersection))
    except IntersectionError as error:
        raise SystemExit(f"Cannot start: {error}")
    args.results.mkdir(parents=True, exist_ok=True)
    steps = {"check": [cmd_check], "measure": [cmd_measure], "optimize": [cmd_optimize],
             "simulate": [cmd_simulate], "report": [cmd_report],
             "all": [cmd_optimize, cmd_simulate, cmd_report]}[args.command]
    try:
        for step in steps:
            step(args, inter)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
