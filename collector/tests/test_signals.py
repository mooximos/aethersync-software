"""Inferred green/red and the virtual detectors on a synthetic queue."""
import pytest

from common import readers
from common.timeutil import local_epoch, parse_epoch
from helpers import ZONE, PipelineRig, drive, stop_at
from monitor.signals import SignalInference, signal_lanes
from monitor.zones import VirtualDetector

T0 = local_epoch("2026-05-12", 7, 0, ZONE)


def queue_and_release(green_at: float, n: int = 3, lane_x: float = 5.25, first_arrival: float = 0.0):
    """n cars stop in one lane before green_at; the first moves 1.0 s after green_at, then one every 2 s."""
    cars = []
    for k in range(n):
        arrive = first_arrival + 3.0 * k
        stopped = stop_at(120.0, 13.4, 1.0 + 6.5 * k, wait_s=0.0, after_s=0.0)
        braking_done = stopped[-1][0]
        wait = green_at + 1.0 + 2.0 * k - (arrive + braking_done)
        car = stop_at(120.0, 13.4, 1.0 + 6.5 * k, wait_s=wait, after_s=12.0)
        cars.append([(round(t + arrive, 1), lane_x, y, v) for t, _x, y, v in car])
    return cars


def events(tmp_path):
    rows = readers.read_rows(tmp_path / "2026-05-12" / "signal_events_07.csv")
    for row in rows:
        row["t"] = parse_epoch(row["time"]) - T0
    return rows


def test_green_red_and_detectors_on_a_synthetic_queue(tmp_path):
    rig = PipelineRig(tmp_path)
    cars = queue_and_release(green_at=40.0)
    late = stop_at(120.0, 13.4, 1.0, wait_s=10.0, after_s=10.0)            # arrives after the queue left
    cars.append([(round(t + 47.0, 1), 5.25, y, v) for t, _x, y, v in late])
    rig.run(cars, until=90.0)
    rig.close()
    rows = events(tmp_path)
    greens = [r for r in rows if r["event"] == "green_start_inferred"]
    reds = [r for r in rows if r["event"] == "red_inferred_by"]
    assert len(greens) == 2 and len(reds) == 1
    first_moved = 41.0 + 1.0 / 2.0          # above 1.0 m/s 0.5 s after it starts at 2 m/s2
    # the smoothed speed reacts late (under 1 s) when a car starts moving
    assert first_moved - 1.5 - 0.2 <= greens[0]["t"] <= first_moved - 1.5 + 1.0
    assert greens[0]["approach"] == "N" and greens[0]["lane"] == ""
    assert greens[0]["detail"].startswith("first queued car moved at 07:00:4") and "minus 1.5 s reaction" in greens[0]["detail"]
    assert reds[0]["detail"] == "first car stopped at the line"
    assert greens[0]["t"] < reds[0]["t"] < greens[1]["t"]
    detector = [r for r in rows if r["event_code"] in ("82", "81")]
    # Cars queue 6.5 m apart and the zone is 6 m long, so each car turns it on and off.
    assert [r["event_code"] for r in detector] == ["82", "81"] * 4
    assert {r["lane"] for r in detector} == {"2"}
    assert [r["detail"] for r in detector[::2]] == ["A-0001", "A-0002", "A-0003", "A-0004"]
    assert all(on["t"] <= off["t"] and off["detail"] == on["detail"]
               for on, off in zip(detector[::2], detector[1::2]))
    assert detector[0]["t"] < greens[0]["t"] and detector[1]["t"] > first_moved


def test_right_turn_lane_does_not_infer_green(tmp_path):
    rig = PipelineRig(tmp_path)
    rig.run(queue_and_release(green_at=30.0, n=2, lane_x=1.8), until=60.0)   # lane 1 = R + T
    rig.close()
    rows = events(tmp_path)
    assert not [r for r in rows if r["event"] == "green_start_inferred"]
    ons = [r for r in rows if r["event_code"] == "82"]
    assert ons and {r["lane"] for r in ons} == {"1"}   # the detector still works in that lane


def test_signal_lanes_leave_out_right_turns_unless_nothing_is_left():
    assert signal_lanes({1: ("R", "T"), 2: ("T",), 3: ("L",)}) == {2, 3}
    assert signal_lanes({1: ("R", "T", "L")}) == {1}


def test_at_most_one_green_per_20_s_and_red_needs_a_crossing():
    s = SignalInference({2}, 6.0, 1.5, 20.0)
    assert s.on_stop_start(5.0, 1.0, 2) is None          # no green seen yet
    assert s.on_stop_end(10.0, 1.0, 2) == (8.5, 1, 10.0)
    assert s.on_stop_end(25.0, 1.0, 2) is None           # within 20 s of the last one
    assert s.on_stop_start(40.0, 1.0, 2) is None         # nobody crossed since the green
    s.on_cross(41.0)
    assert s.on_stop_start(45.0, 9.0, 2) is None         # outside the stop-line zone
    assert s.on_stop_start(46.0, 1.0, 2) == (46.0, 10, 46.0)
    assert s.on_stop_start(47.0, 2.0, 2) is None         # only the first one
    assert s.on_stop_end(31.0, 1.0, 3) is None           # lane not used


def test_virtual_detector_holds_through_a_missed_frame_and_pairs_on_off():
    d = VirtualDetector(2, 6.0, hold_s=1.0)
    assert d.update(0.0, {"A-0001": (5.0, 5.0, 2)}) == [(0.0, 82, "A-0001")]
    assert d.update(0.1, {}) == []                                      # missed for one frame
    assert d.update(0.2, {"A-0001": (5.0, 4.0, 2), "A-0002": (5.0, 7.0, 2)}) == []
    assert d.update(0.3, {"A-0001": (5.0, -0.5, 2), "A-0002": (5.0, 5.9, 2)}) == []
    assert d.update(0.4, {"A-0002": (5.0, 5.8, 1)}) == [(0.4, 81, "A-0002")]   # changed lane: zone empty
    assert d.update(0.5, {"A-0003": (5.0, 1.0, 2)}) == [(0.5, 82, "A-0003")]
    assert d.update(2.0, {}) == [(0.5, 81, "A-0003")]                   # vanished: off at its last sighting
    assert d.update(2.1, {"A-0004": (5.0, 1.0, 2)}) == [(2.1, 82, "A-0004")]
    assert d.close(3.0) == [(3.0, 81, "A-0004")] and d.close(3.0) == []
