"""The whole collector pipeline on boxes made from sample_data tracks.

The fake detector gets one box per vehicle per frame (10 fps), made by projecting
each sample vehicle through the sample camera's known homography, with 0.5 px of
jitter and 3% missed detections. The pipeline (ByteTrack, Kalman, events, writers)
must give back what the generator put in."""
from collections import Counter

import numpy as np
import pytest

from common import readers, schema
from common.timeutil import local_epoch, parse_epoch
from helpers import SAMPLE_DAY, ZONE, sample_site
from monitor.detector import FakeDetector, Frame
from monitor.replay import run_offline
import synthetic

MINUTES = 10
START = local_epoch("2026-05-12", 7, 10, ZONE)
END = START + MINUTES * 60
MARGIN = 120   # compare only vehicles crossing well inside the window


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e")
    rows, offset = synthetic.fake_boxes(SAMPLE_DAY, 7, START, END, 1000.0, 10.0, np.random.default_rng(3))
    times = np.round(np.arange(1000.0, 1000.0 + MINUTES * 60, 0.1), 3)
    frames = (Frame(camera, float(t), None) for t in times for camera in "ABCD")
    counts = run_offline(sample_site(), frames, FakeDetector(rows=rows), out, offset, label="end-to-end test")
    day = out / "2026-05-12"
    return {"counts": counts, "day": day, "got": readers.read_rows(day / "vehicles_07.csv"),
            "want": readers.read_rows(SAMPLE_DAY / "vehicles_07.csv")}


def crossed_inside(rows):
    return [r for r in rows if r["t_crossed"] and START + MARGIN <= parse_epoch(r["t_crossed"]) < END - MARGIN]


def pairs(result):
    got = crossed_inside(result["got"])
    out = []
    for want in crossed_inside(result["want"]):
        t = parse_epoch(want["t_crossed"])
        same = [g for g in got if g["approach"] == want["approach"] and g["lane"] == want["lane"]]
        best = min(same, key=lambda g: abs(parse_epoch(g["t_crossed"]) - t), default=None)
        out.append((want, best if best and abs(parse_epoch(best["t_crossed"]) - t) <= 1.0 else None))
    return out


def test_every_frame_was_processed_and_the_files_are_valid(result):
    assert result["counts"] == {c: MINUTES * 600 for c in "ABCD"}
    assert {name: p for name, p in schema.validate_day_folder(result["day"]).items() if p} == {}


def test_exact_counts_per_approach(result):
    got = Counter(r["approach"] for r in crossed_inside(result["got"]))
    want = Counter(r["approach"] for r in crossed_inside(result["want"]))
    assert sum(want.values()) > 100
    assert got == want


def test_every_vehicle_is_found_in_its_lane_with_its_turn(result):
    matched = pairs(result)
    assert all(got is not None for _want, got in matched)
    assert all(got["turn"] == want["turn"] and got["type"] == want["type"] for want, got in matched)
    assert sum(got["stops"] == want["stops"] for want, got in matched) >= 0.95 * len(matched)


def test_wait_within_1_s_for_95_percent(result):
    diffs = np.array([abs(float(got["wait_s"]) - float(want["wait_s"])) for want, got in pairs(result)])
    assert (diffs <= 1.0).mean() >= 0.95
    assert np.median(diffs) <= 0.5


def test_crossing_times_within_0_3_s(result):
    diffs = [abs(parse_epoch(got["t_crossed"]) - parse_epoch(want["t_crossed"])) for want, got in pairs(result)]
    assert max(diffs) <= 0.3


def test_queue_positions_mostly_agree(result):
    stopped = [(w, g) for w, g in pairs(result) if w["queue_pos"]]
    assert sum(w["queue_pos"] == g["queue_pos"] for w, g in stopped) >= 0.9 * len(stopped)


def test_signal_events_are_inferred(result):
    events = readers.read_rows(result["day"] / "signal_events_07.csv")
    kinds = Counter(r["event"] for r in events)
    assert kinds["detector_on"] == kinds["detector_off"] > 100
    for approach, green_start in {"N": 0.0, "S": 0.0, "E": 56.0, "W": 56.0}.items():
        greens = [parse_epoch(r["time"]) for r in events if r["event_code"] == "1" and r["approach"] == approach]
        assert len(greens) >= MINUTES / 2 - 1
        # generator: drivers react 1.0-2.3 s after green; minus 1.5 s; plus smoothing lag
        assert all(-1.0 <= (g - green_start) % 120 <= 3.0 for g in greens), approach
