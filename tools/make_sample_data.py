"""Make synthetic sample data in the exact data format (docs/DATA_FORMAT.md).

    python tools/make_sample_data.py --out sample_data --days 2 --hours 7 17

Hours next to each other (like 7 8) are simulated as one recording; each
separate block of hours is one recording that starts on the hour and is shut
down at the end of its last hour, just like a real collector session.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import SOFTWARE_VERSION, schema  # noqa: E402
from common.site import validate_site  # noqa: E402
from common.timeutil import clock_text, format_epoch, get_zone, local_day_and_hour, local_epoch  # noqa: E402
import sample_events as ev  # noqa: E402
from sample_sim import DT, SignalPlan, simulate_approach  # noqa: E402
from sample_site import CAMERA_APPROACH, build_site  # noqa: E402

WARMUP_S = 240   # simulated before the recording starts, so queues already exist
# Default traffic, vehicles per hour as (left, through, right). The morning is
# heavy from the north, the evening heavy from the south.
AM_VOLUMES = {"N": (90, 700, 90), "S": (70, 500, 80), "E": (60, 340, 70), "W": (80, 420, 60)}
PM_VOLUMES = {"N": (80, 540, 90), "S": (100, 720, 100), "E": (90, 500, 80), "W": (70, 460, 90)}


def parse_volumes(text: str) -> dict[str, tuple[float, ...]]:
    """'N=90/700/90,S=70/500/80,E=60/340/70,W=80/420/60' -> {approach: (L, T, R)}."""
    volumes = {}
    for part in text.split(","):
        approach, numbers = part.split("=")
        left, through, right = (float(n) for n in numbers.split("/"))
        volumes[approach.strip().upper()] = (left, through, right)
    if set(volumes) != set("NSEW"):
        raise argparse.ArgumentTypeError("volumes must name N, S, E and W, like N=90/700/90,S=...")
    return volumes


def parse_splits(text: str, cycle: float) -> SignalPlan:
    """'0-50,56-110' -> N/S green 0-50 s and E/W green 56-110 s of the cycle."""
    (ns0, ns1), (ew0, ew1) = (tuple(float(n) for n in part.split("-")) for part in text.split(","))
    plan = SignalPlan(cycle=cycle, green={"NS": (ns0, ns1), "EW": (ew0, ew1)})
    plan.check()
    return plan


def weekdays_from(start: str, count: int, weekends: bool = False) -> list[str]:
    day, days = date.fromisoformat(start), []
    while len(days) < count:
        if weekends or day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    return days


def hour_blocks(hours: list[int]) -> list[list[int]]:
    """[7, 8, 17] -> [[7, 8], [17]]: hours in a row belong to one recording."""
    blocks: list[list[int]] = []
    for hour in sorted(set(hours)):
        if blocks and blocks[-1][-1] == hour - 1:
            blocks[-1].append(hour)
        else:
            blocks.append([hour])
    return blocks


class DayFiles:
    """Collects the rows of one day, sorted into hour files, then writes them."""

    def __init__(self, zone) -> None:
        self.zone = zone
        self.rows: dict[tuple[str, int], list[tuple[float, list]]] = defaultdict(list)

    def add(self, kind: str, epoch_s: float, row: list) -> None:
        _day, hour = local_day_and_hour(epoch_s, self.zone)
        self.rows[(kind, hour)].append((epoch_s, row))

    def write(self, day_dir: Path) -> None:
        for (kind, hour), rows in sorted(self.rows.items()):
            text = io.StringIO()
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(schema.COLUMNS[kind])
            writer.writerows(row for _epoch, row in sorted(rows, key=lambda item: item[0]))
            data = text.getvalue().encode("utf-8")
            name = f"{kind}_{hour:02d}.csv"
            if kind == "tracks":   # closed hours of tracks are gzipped
                with open(day_dir / (name + ".gz"), "wb") as raw:
                    with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as packed:
                        packed.write(data)
            else:
                (day_dir / name).write_bytes(data)


def record_block(out: DayFiles, day: str, hours: list[int], plan: SignalPlan, volumes: dict,
                 night_hours: set[int], counters: dict[str, int], rng: np.random.Generator) -> None:
    """Simulate one recording (a block of hours) and add all of its rows."""
    zone = out.zone
    start_epoch = local_epoch(day, hours[0], 0, zone)
    sim_epoch0 = start_epoch - WARMUP_S
    first_step = round(WARMUP_S / DT)
    n_steps = first_step + round(len(hours) * 3600 / DT)
    last_step = n_steps - 1

    def when(step: float) -> float:
        return sim_epoch0 + round(step) * DT

    def stamp(step: float) -> str:
        return format_epoch(when(step), zone)

    for camera, approach in CAMERA_APPROACH.items():
        left, through, right = volumes[approach]
        vehicles = simulate_approach(approach, plan, {"L": left, "T": through, "R": right}, n_steps,
                                     hours[0] * 3600 - WARMUP_S, rng)
        seen_list = []
        for veh in sorted(vehicles, key=lambda v: (v.step0, v.lane)):
            entry_hour = local_day_and_hour(when(max(veh.step0, first_step)), zone)[1]
            night = entry_hour in night_hours
            typical = (0.66, 0.10) if night else (0.72, 0.08) if veh.vtype == "motorcycle" else (0.84, 0.07)
            conf = round(float(np.clip(rng.normal(*typical), 0.40, 0.97)), 2)
            seen = ev.observe(veh, camera, first_step, last_step, conf)
            if seen is None:
                continue
            counters[camera] += 1
            seen.vehicle_id = f"{camera}-{counters[camera]:04d}"
            seen_list.append((seen, night))

        by_lane = defaultdict(list)
        for seen, _night in seen_list:
            by_lane[seen.lane].append(seen)
        queue_pos = {}
        for lane, group in by_lane.items():
            queue_pos.update(ev.queue_positions(group))
            for step, code, vehicle_id in ev.detector_events(group, last_step):
                out.add("signal_events", when(step),
                        [stamp(step), code, schema.EVENT_CODES[code], approach, lane, vehicle_id])
        for step, code in ev.light_events([seen for seen, _ in seen_list]):
            if code == 1:
                moved = when(step)
                detail = f"first queued car moved at {clock_text(moved, zone)}; minus {ev.REACTION_S} s reaction"
                time_s = moved - ev.REACTION_S
            else:
                detail, time_s = "first car stopped at the line", when(step)
            out.add("signal_events", time_s,
                    [format_epoch(time_s, zone), code, schema.EVENT_CODES[code], approach, "", detail])

        for seen, night in seen_list:
            episodes = seen.episodes
            wait_s = sum(end - start for start, end, _ in episodes) * DT
            out.add("vehicles", when(seen.last_step), [
                seen.vehicle_id, camera, approach, seen.lane if seen.lane else "", ev.turn_of(seen),
                seen.veh.vtype, f"{seen.conf:.2f}", stamp(seen.first_step),
                stamp(episodes[0][0]) if episodes else "", stamp(episodes[-1][1]) if episodes else "",
                stamp(seen.cross) if seen.cross is not None else "", stamp(seen.last_step),
                f"{wait_s:.1f}", len(episodes), queue_pos.get(seen.vehicle_id, ""),
                "night" if night else "day", ";".join(ev.vehicle_flags(seen, night)),
            ])
            for index in range(len(seen.y)):
                step = seen.first_step + index
                if step % ev.TRACK_EVERY_STEPS == 0:
                    out.add("tracks", when(step), [stamp(step), seen.vehicle_id, f"{seen.x[index]:.2f}",
                                                   f"{seen.y[index]:.2f}", f"{seen.speed[index]:.1f}"])


def system_rows(out: DayFiles, day: str, hours: list[int], day_index: int, rng: np.random.Generator) -> None:
    for hour in hours:
        for minute in range(60):
            epoch_s = local_epoch(day, hour, minute, out.zone)
            battery = 13.30 - 0.02 * day_index - 0.008 * (hour - 6 + minute / 60) + rng.normal(0, 0.004)
            power = 11.6 + rng.normal(0, 0.25)
            warm = 4.0 if hour >= 12 else 0.0
            out.add("system", epoch_s, [
                format_epoch(epoch_s, out.zone), f"{battery:.2f}", f"{power / battery:.2f}", f"{power:.1f}",
                f"{58 + warm + rng.normal(0, 0.8):.1f}", f"{51 + warm + rng.normal(0, 0.6):.1f}",
                *[f"{10.0 - abs(rng.normal(0, 0.08)):.1f}" for _ in "ABCD"],
                int(rng.poisson(1.5)), f"{24.80 - 0.05 * day_index - 0.004 * (hour + minute / 60):.2f}",
            ])


def make_day(out_dir: Path, day: str, day_index: int, args: argparse.Namespace) -> Path:
    zone = get_zone("America/Los_Angeles")
    plan = parse_splits(args.splits, args.cycle)
    rng = np.random.default_rng([args.seed, day_index])
    day_dir = out_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    blocks = hour_blocks(args.hours)
    sync_before = local_epoch(day, blocks[0][0], 0, zone) - 600 - float(rng.uniform(0, 120))
    offset_s = round(float(rng.normal(0, 1.5)), 1)
    sync_after = sync_before + offset_s
    with open(day_dir / "clock_sync.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(schema.CLOCK_SYNC_COLUMNS)
        writer.writerow([format_epoch(sync_before, zone), format_epoch(sync_after, zone), f"{offset_s:.1f}", "phone"])

    site = build_site(format_epoch(sync_after, zone), offset_s)
    validate_site(site)
    (day_dir / "site.json").write_text(json.dumps(site, indent=2) + "\n", encoding="utf-8")

    out = DayFiles(zone)
    counters = {camera: 0 for camera in CAMERA_APPROACH}
    log = [f"{format_epoch(sync_after, zone)} clock synced from phone, offset {offset_s:+.1f} s"]
    day_factor = float(rng.normal(1.0, 0.04))   # some days are a little busier
    for hours in blocks:
        base = args.volumes or (AM_VOLUMES if hours[0] < 12 else PM_VOLUMES)
        volumes = {approach: tuple(v * day_factor for v in flows) for approach, flows in base.items()}
        record_block(out, day, hours, plan, volumes, set(args.night_hours), counters, rng)
        system_rows(out, day, hours, day_index, rng)
        start = local_epoch(day, hours[0], 0, zone)
        end = start + len(hours) * 3600 - DT
        log.append(f"{format_epoch(start, zone)} startup: collector {SOFTWARE_VERSION} (sample data generator)")
        log.append(f"{format_epoch(start, zone)} recording started")
        if hours[0] in args.night_hours:
            log.append(f"{format_epoch(start, zone)} mode switch: night mode on")
        log.append(f"{format_epoch(end, zone)} shutdown: recording stopped, files closed")
    out.write(day_dir)
    (day_dir / "events.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    return day_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Make synthetic intersection data in the exact data format.")
    parser.add_argument("--out", type=Path, default=Path("sample_data"), help="folder to write the day folders into")
    parser.add_argument("--start-date", default="2026-05-12", help="first day, YYYY-MM-DD (weekends are skipped unless --weekends)")
    parser.add_argument("--days", type=int, default=2, help="number of days to make")
    parser.add_argument("--weekends", action="store_true", help="make Saturdays and Sundays too")
    parser.add_argument("--hours", type=int, nargs="+", default=[7, 17], help="clock hours to make, e.g. 7 8 17")
    parser.add_argument("--seed", type=int, default=1, help="same seed = same data")
    parser.add_argument("--volumes", type=parse_volumes, default=None,
                        help="vehicles/hour as left/through/right, e.g. N=90/700/90,S=70/500/80,E=60/340/70,W=80/420/60")
    parser.add_argument("--cycle", type=float, default=120.0, help="signal cycle length, seconds")
    parser.add_argument("--splits", default="0-50,56-110", help="N/S green and E/W green, seconds within the cycle")
    parser.add_argument("--night-hours", type=int, nargs="*", default=[], help="hours to label as night (night_mode)")
    args = parser.parse_args(argv)
    if any(not 0 <= hour <= 23 for hour in args.hours):
        parser.error("--hours must be between 0 and 23")
    for index, day in enumerate(weekdays_from(args.start_date, args.days, args.weekends)):
        day_dir = make_day(args.out, day, index, args)
        print(f"wrote {day_dir}")


if __name__ == "__main__":
    main()
