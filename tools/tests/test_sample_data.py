"""The committed sample_data must be consistent with itself:
a car that waits in vehicles is stopped in tracks and sits on the detector."""
from collections import defaultdict
from pathlib import Path
from statistics import median

import pytest

from common import readers
from common.timeutil import parse_epoch

SAMPLE = Path(__file__).resolve().parents[2] / "sample_data"
DAYS = ["2026-05-12", "2026-05-13"]
CYCLE_S = 120.0
GREEN_START = {"N": 0.0, "S": 0.0, "E": 56.0, "W": 56.0}


@pytest.fixture(scope="module", params=DAYS)
def day(request):
    """All rows of one sample day, with times turned into epoch seconds."""
    day_dir = SAMPLE / request.param
    vehicles = readers.read_day(day_dir, "vehicles")
    tracks = defaultdict(list)
    for row in readers.read_day(day_dir, "tracks"):
        tracks[row["vehicle_id"]].append((parse_epoch(row["time"]), float(row["y_m"]), float(row["speed_mps"])))
    signals = readers.read_day(day_dir, "signal_events")
    for row in signals:
        row["epoch"] = parse_epoch(row["time"])
    return {"vehicles": vehicles, "tracks": tracks, "signals": signals}


def stopped_time_from_track(samples, t_crossed, t_left):
    """Stopped seconds seen in the 0.5 s track samples, as a (low, high) range.

    A stop needs 1.0 s below 0.5 m/s. With samples every 0.5 s, a stop with only
    two slow samples lasted somewhere between 0.5 and 1.5 s, so we cannot tell if
    it counts: `low` leaves those out and `high` includes them."""
    low = high = 0.0
    start, slow_samples = None, 0
    for time_s, _y, speed in samples:
        if t_crossed is not None and time_s >= t_crossed:
            break
        if start is None:
            if speed < 0.5:
                start, slow_samples = time_s, 1
        elif speed < 0.5:
            slow_samples += 1
        elif speed > 1.0:
            high += time_s - start
            low += (time_s - start) if slow_samples >= 3 else 0.0
            start = None
    if start is not None:   # still stopped at the stop line or when the track ended
        end = t_crossed if t_crossed is not None else t_left
        high += end - start
        low += (end - start) if slow_samples >= 3 else 0.0
    return low, high


def test_wait_matches_the_stopped_time_in_the_track_within_0_6_s(day):
    assert len(day["vehicles"]) > 5000
    worst = 0.0
    for row in day["vehicles"]:
        samples = day["tracks"].get(row["vehicle_id"], [])
        crossed = parse_epoch(row["t_crossed"]) if row["t_crossed"] else None
        low, high = stopped_time_from_track(samples, crossed, parse_epoch(row["t_left"]))
        wait = float(row["wait_s"])
        worst = max(worst, low - wait, wait - high)
        assert low - 0.6 <= wait <= high + 0.6, (row["vehicle_id"], wait, low, high)
    assert worst <= 0.6


def test_every_vehicle_has_a_track_and_every_track_has_a_vehicle(day):
    ids = [row["vehicle_id"] for row in day["vehicles"]]
    assert len(ids) == len(set(ids))
    assert set(day["tracks"]) <= set(ids)
    for row in day["vehicles"]:
        samples = day["tracks"].get(row["vehicle_id"])
        if samples is None:   # seen for less than half a second, between two track ticks
            assert "short_track" in row["flags"], row["vehicle_id"]
            continue
        assert parse_epoch(row["t_entered"]) <= samples[0][0] < parse_epoch(row["t_entered"]) + 0.5
        assert parse_epoch(row["t_left"]) - 0.5 < samples[-1][0] <= parse_epoch(row["t_left"])
        times = [s[0] for s in samples]
        assert all(round(b - a, 1) == 0.5 for a, b in zip(times, times[1:]))


def test_every_detector_on_has_a_matching_detector_off(day):
    state = {}
    counts = defaultdict(int)
    for row in day["signals"]:
        if row["event_code"] not in ("82", "81"):
            continue
        key = (row["approach"], row["lane"])
        if row["event_code"] == "82":
            assert key not in state, f"detector {key} turned on twice at {row['time']}"
            state[key] = row["epoch"]
        else:
            assert key in state, f"detector {key} turned off while off at {row['time']}"
            assert row["epoch"] >= state.pop(key)
        counts[row["event"]] += 1
    assert state == {}, "a detector was left on at the end"
    assert counts["detector_on"] == counts["detector_off"] > 4000


def test_a_waiting_car_at_the_line_sits_on_the_detector(day):
    on_spans = defaultdict(list)
    open_since = {}
    for row in day["signals"]:
        key = (row["approach"], row["lane"])
        if row["event_code"] == "82":
            open_since[key] = row["epoch"]
        elif row["event_code"] == "81":
            on_spans[key].append((open_since.pop(key), row["epoch"]))
    checked = 0
    for row in day["vehicles"]:
        if row["queue_pos"] != "1" or float(row["wait_s"]) < 3.0:
            continue
        stopped, started = parse_epoch(row["t_stopped"]), parse_epoch(row["t_started"])
        spans = on_spans[(row["approach"], row["lane"])]
        assert any(on <= stopped and started <= off for on, off in spans), row["vehicle_id"]
        checked += 1
    assert checked > 300


def test_detector_count_agrees_with_crossings_within_3_percent(day):
    crossings = defaultdict(int)
    for row in day["vehicles"]:
        if row["t_crossed"]:
            crossings[(row["approach"], row["lane"])] += 1
    detections = defaultdict(int)
    for row in day["signals"]:
        if row["event_code"] == "82":
            detections[(row["approach"], row["lane"])] += 1
    assert len(crossings) == 12
    for key, crossed in crossings.items():
        assert abs(detections[key] - crossed) <= 0.03 * crossed, (key, detections[key], crossed)


def test_inferred_greens_follow_the_generator_signal(day):
    for approach, true_start in GREEN_START.items():
        greens = [r["epoch"] for r in day["signals"] if r["event_code"] == "1" and r["approach"] == approach]
        assert len(greens) >= 55   # 2 hours x 30 cycles, a few may have no queue
        offsets = [(g - true_start) % CYCLE_S for g in greens]
        assert all(-0.01 <= off <= 2.0 for off in offsets), (approach, max(offsets))
        gaps = [b - a for a, b in zip(greens, greens[1:]) if b - a < 1.5 * CYCLE_S]
        assert abs(median(gaps) - CYCLE_S) <= 0.5
        reds = [r["epoch"] for r in day["signals"] if r["event_code"] == "10" and r["approach"] == approach]
        assert len(reds) >= 55
        # the first car stops at the line shortly after the green has ended (50 or 54 s long)
        green_len = 50.0 if approach in "NS" else 54.0
        assert all(green_len <= (r - true_start) % CYCLE_S < CYCLE_S for r in reds)


def test_waits_look_like_a_fixed_time_signal(day):
    waits = [float(r["wait_s"]) for r in day["vehicles"]]
    assert 10.0 < sum(waits) / len(waits) < 35.0
    assert max(waits) < 80.0   # nobody waits longer than one red
    stopped = [r for r in day["vehicles"] if r["stops"] != "0"]
    assert all(r["t_stopped"] and r["t_started"] and r["queue_pos"] for r in stopped)
    never = [r for r in day["vehicles"] if r["stops"] == "0"]
    assert all(r["wait_s"] == "0.0" and not r["t_stopped"] and not r["queue_pos"] for r in never)
    assert len(never) > 0.2 * len(day["vehicles"])


def test_through_lane_saturation_headway_is_about_2_seconds(day):
    """Headways between queued vehicles crossing the line, from the 5th in each queue on."""
    by_lane = defaultdict(list)
    for row in day["vehicles"]:
        if row["lane"] == "2" and row["t_crossed"]:
            by_lane[row["approach"]].append((parse_epoch(row["t_crossed"]), row["queue_pos"]))
    for approach, crossings in by_lane.items():
        crossings.sort()
        headways = [b[0] - a[0] for a, b in zip(crossings, crossings[1:])
                    if a[1] and b[1] and int(b[1]) >= 5 and int(a[1]) == int(b[1]) - 1]
        assert len(headways) >= 30
        assert 1.9 <= median(headways) <= 2.2, (approach, median(headways))


def test_lanes_turns_and_flags_follow_the_rules(day):
    for row in day["vehicles"]:
        flags = row["flags"].split(";") if row["flags"] else []
        if row["lane"] == "2":
            assert row["turn"] == "T"
        if row["lane"] == "3":
            assert row["turn"] == "L"
        if row["lane"] == "1":
            assert row["turn"] in ("R", "T", "U")
            assert (row["turn"] == "U") == (not row["t_crossed"] or "cut_at_shutdown" in flags)
        assert ("low_conf" in flags) == (float(row["conf"]) < 0.65)
        assert row["light"] == "day" and "night_mode" not in flags
    types = defaultdict(int)
    for row in day["vehicles"]:
        types[row["type"]] += 1
    assert set(types) == {"car", "motorcycle", "bus", "truck"}
    assert types["car"] > 0.85 * len(day["vehicles"])
