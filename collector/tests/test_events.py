"""Vehicle events from synthetic trajectories (data format section 2.2)."""
import pytest

from common import readers
from common.timeutil import parse_epoch
from helpers import PipelineRig, drive, sample_site, stop_at
from monitor.events import Rules, VehicleTrack


def run_track(samples, rules=None, cut=False, present=False, scores=None):
    rules = rules or Rules.from_site(sample_site(), "A")
    track = VehicleTrack(1, rules, calibrated=True, present_at_start=present)
    notes = []
    for i, (t, x, y, speed) in enumerate(samples):
        score = scores[i] if scores else 0.9
        notes += track.add(t, score, "car", (x, y, speed), False, False)
    return track.finish(cut_at_shutdown=cut), notes


def test_never_stopped():
    s, notes = run_track(drive(120.0, 13.4, [(12.0, 0.0)]))
    assert s.stops == 0 and s.wait_s == 0 and s.t_stopped is None and s.t_started is None
    assert s.t_entered == 0.0
    assert s.t_crossed == pytest.approx(120.0 / 13.4, abs=0.01)   # interpolated between samples
    assert s.lane == 2 and s.turn == "T" and s.flags == []
    assert [n[0] for n in notes] == ["entered", "cross"]


def test_one_stop_wait_is_the_stopped_time():
    s, notes = run_track(stop_at(120.0, 13.4, 1.0, wait_s=20.0))
    assert s.stops == 1
    braking = 13.4 ** 2 / (2 * 119.0)
    # speed < 0.5 m/s starts 0.5 / braking s before the car is fully stopped;
    # speed > 1.0 m/s starts 0.5 s after it drives off at 2 m/s2.
    assert s.wait_s == pytest.approx(20.0 + 0.5 / braking + 0.5, abs=0.15)
    assert s.t_started - s.t_stopped == pytest.approx(s.wait_s)
    assert s.t_crossed > s.t_started
    assert [n[0] for n in notes] == ["entered", "stop_start", "stop_end", "cross"]


def test_short_slowdown_is_not_a_stop():
    samples = drive(60.0, 5.0, [(2.0, -2.4), (0.5, 0.0), (5.0, 2.0)])   # below 0.5 m/s for about 0.8 s
    s, _ = run_track(samples)
    assert s.stops == 0 and s.wait_s == 0.0


def test_creeping_queue_two_stops():
    samples = drive(40.0, 6.0, [(2.0, -3.0), (5.0, 0.0), (1.5, 2.0), (1.0, -3.0), (6.0, 0.0), (10.0, 2.0)])
    s, notes = run_track(samples)
    assert s.stops == 2
    assert [n[0] for n in notes].count("stop_start") == 2
    assert s.t_started - s.t_stopped > s.wait_s   # the creep between the two stops is not waiting


def test_creep_below_moving_speed_does_not_end_the_stop():
    # 0.3 s at 1.5 m/s is not 0.5 s above 1.0 m/s, so this stays one stop (hysteresis)
    samples = [(round(i * 0.1, 1), 5.0, 10.0, 0.0) for i in range(30)]
    samples += [(round(3.0 + i * 0.1, 1), 5.0, 10.0 - 0.15 * i, 1.5) for i in range(3)]
    samples += [(round(3.3 + i * 0.1, 1), 5.0, 9.6, 0.0) for i in range(20)]
    samples += [(round(5.3 + i * 0.1, 1), 5.0, 9.6 - 0.5 * i, 5.0) for i in range(30)]
    s, _ = run_track(samples)
    assert s.stops == 1 and s.wait_s == pytest.approx(5.3, abs=0.01)


def test_lost_before_the_line():
    samples = [(t, 1.8, y, v) for t, _x, y, v in stop_at(80.0, 13.4, 20.0, wait_s=5.0, after_s=0.0)]
    s, _ = run_track(samples)
    assert s.t_crossed is None and "no_cross" in s.flags
    assert s.lane == 1 and s.turn == "U"   # lane 1 allows R and T, and it never reached the line
    assert s.stops == 1 and s.t_started is not None   # an open stop ends where the track ends


def test_short_track_and_started_in_view():
    s, _ = run_track(drive(50.0, 13.4, [(0.6, 0.0)]), present=True)
    assert {"short_track", "started_in_view"} <= set(s.flags)


def test_low_conf_and_type_vote():
    rules = Rules.from_site(sample_site(), "A")
    track = VehicleTrack(1, rules, True, False)
    for i, (t, x, y, v) in enumerate(drive(100.0, 13.4, [(8.0, 0.0)])):
        track.add(t, 0.5, "truck" if i % 3 == 0 else "car", (x, y, v), False, False)
    s = track.finish(False)
    assert s.vtype == "car" and "low_conf" in s.flags and s.conf == pytest.approx(0.5)


def test_queue_past_view_and_cut_at_shutdown():
    samples = drive(118.0, 2.0, [(1.0, -2.0), (5.0, 0.0)])
    s, _ = run_track(samples, cut=True)
    assert {"queue_past_view", "cut_at_shutdown", "no_cross"} <= set(s.flags)


@pytest.mark.parametrize("x,lane", [(1.0, 1), (5.0, 2), (9.0, 3), (12.0, None)])
def test_lane_from_median_x_of_last_3_s(x, lane):
    s, _ = run_track(drive(60.0, 13.4, [(6.0, 0.0)], x=x))
    assert s.lane == lane


def test_lane_uses_only_the_last_3_s_before_the_line():
    early = drive(100.0, 13.4, [(5.0, 0.0)], x=5.0)                      # lane 2 far away
    late = [(round(t + 5.1, 1), 1.5, y, v) for t, _x, y, v in drive(early[-1][2] - 1.34, 13.4, [(5.0, 0.0)])]
    s, _ = run_track(early + late)
    assert s.lane == 1


def _turning(x_after_line: float):
    samples = drive(30.0, 8.0, [(3.0, 0.0)], x=1.8)
    t, y = samples[-1][0], samples[-1][2]
    for i in range(1, 25):
        samples.append((round(t + i * 0.1, 1), 1.8 + x_after_line * i / 24, y - 0.8 * i, 8.0))
    return samples


@pytest.mark.parametrize("dx,turn", [(-6.0, "R"), (0.5, "T"), (6.0, "L")])
def test_turn_rule_from_sideways_movement_after_the_line(dx, turn):
    s, _ = run_track(_turning(dx))
    assert s.lane == 1 and s.turn == turn


def test_turn_unknown_if_not_seen_8_m_past_the_line():
    samples = drive(20.0, 8.0, [(3.0, 0.0)], x=1.8)   # ends 4 m past the line
    s, _ = run_track(samples)
    assert s.lane == 1 and s.turn == "U"


def test_single_movement_lane_gives_its_turn():
    s, _ = run_track(drive(20.0, 8.0, [(2.0, 0.0)], x=9.0))
    assert s.lane == 3 and s.turn == "L"


def test_queue_positions_and_stop_across_the_hour_boundary(tmp_path):
    """Three cars queue in lane 2 at 07:59:xx and leave after 08:00; files follow t_left."""
    rig = PipelineRig(tmp_path, start_hour=7, start_minute=59)
    cars = [stop_at(120.0, 13.4, 1.0 + 6.5 * k, wait_s=45.0 - 2 * k) for k in range(3)]
    cars = [[(round(t + 4.0 * k, 1), x, y, v) for t, x, y, v in car] for k, car in enumerate(cars)]
    rig.run(cars, until=90.0)
    rig.close()
    day = tmp_path / "2026-05-12"
    rows = readers.read_rows(day / "vehicles_08.csv")
    assert len(rows) == 3 and not (day / "vehicles_07.csv").exists()
    rows.sort(key=lambda r: r["t_crossed"])
    assert [r["queue_pos"] for r in rows] == ["1", "2", "3"]
    for row in rows:
        assert row["t_stopped"].startswith("2026-05-12T07:59") and row["t_started"].startswith("2026-05-12T08:00")
        assert row["lane"] == "2" and row["turn"] == "T" and row["stops"] == "1"
        assert parse_epoch(row["t_started"]) - parse_epoch(row["t_stopped"]) == pytest.approx(float(row["wait_s"]), abs=0.11)
    assert (day / "tracks_07.csv.gz").exists() and (day / "tracks_08.csv.gz").exists()
