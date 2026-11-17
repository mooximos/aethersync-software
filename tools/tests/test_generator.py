"""Tests of the generator's own code (the simulation and the rules)."""
import math
from pathlib import Path

import numpy as np
import pytest

import make_sample_data as gen
import sample_events as ev
import sample_sim as sim
from common import readers, schema

SAMPLE = Path(__file__).resolve().parents[2] / "sample_data"


def speeds(*parts):
    """speeds((0.0, 3.0), (5.0, 1.0)) -> 3.0 s at 0 m/s then 1.0 s at 5 m/s, in 0.1 s steps."""
    return np.concatenate([np.full(round(seconds / sim.DT), value) for value, seconds in parts])


def test_stop_episode_needs_one_second_below_half_a_meter_per_second():
    assert ev.stop_episodes(speeds((5.0, 1.0), (0.2, 0.9), (5.0, 2.0))) == []
    assert ev.stop_episodes(speeds((5.0, 1.0), (0.2, 1.1), (5.0, 2.0))) == [(10, 21, True)]


def test_stop_episode_survives_a_short_creep_forward():
    # Creeping above 1.0 m/s for only 0.3 s does not end the stop (hysteresis).
    track = speeds((5.0, 1.0), (0.0, 3.0), (1.5, 0.3), (0.0, 2.0), (4.0, 1.0))
    assert ev.stop_episodes(track) == [(10, 63, True)]


def test_two_stops_are_two_episodes():
    track = speeds((5.0, 1.0), (0.0, 3.0), (3.0, 1.0), (0.0, 2.0), (4.0, 1.0))
    assert ev.stop_episodes(track) == [(10, 40, True), (50, 70, True)]


def test_stop_episode_still_open_when_the_track_ends():
    assert ev.stop_episodes(speeds((5.0, 1.0), (0.0, 2.0))) == [(10, 29, False)]


def test_signal_plan_default_matches_the_spec():
    plan = sim.SignalPlan()
    plan.check()
    assert plan.is_green("N", 0.0) and plan.is_green("S", 49.9) and not plan.is_green("N", 50.0)
    assert plan.is_green("E", 56.0) and plan.is_green("W", 109.9) and not plan.is_green("E", 110.0)
    assert plan.is_green("N", 7 * 3600 + 10.0)             # the cycle repeats every 120 s
    assert not plan.is_green("N", 1.0, reaction_s=1.5)     # drivers react late to the green


def test_bad_splits_are_refused():
    with pytest.raises(ValueError):
        gen.parse_splits("0-50,52-110", 120.0)    # only 2 s between greens: no room for yellow + all-red
    with pytest.raises(ValueError):
        gen.parse_splits("0-50,56-118", 120.0)
    assert gen.parse_splits("0-30,36-54", 60.0).cycle == 60.0


def test_command_line_helpers():
    assert gen.hour_blocks([17, 7, 8]) == [[7, 8], [17]]
    assert gen.weekdays_from("2026-05-15", 3) == ["2026-05-15", "2026-05-18", "2026-05-19"]   # skips the weekend
    assert gen.parse_volumes("N=1/2/3,S=4/5/6,E=7/8/9,W=10/11/12")["W"] == (10.0, 11.0, 12.0)


def test_brake_speed_stops_exactly_at_3_mps2():
    position, speed, room = 0.0, 13.4, 13.4 ** 2 / (2 * sim.DECEL) + 5.0
    previous = speed
    while True:
        speed = min(speed, sim.brake_speed(room - position))
        assert previous - speed <= sim.DECEL * sim.DT + 1e-9
        if speed <= 0:
            break
        position += speed * sim.DT
        previous = speed
    assert abs(position - room) < 0.05


def test_turn_paths_end_on_the_correct_side():
    s = np.array([10.0, 0.0, -30.0])
    x, y = sim.path_xy("T", 5.0, s)
    assert list(x) == [5.0, 5.0, 5.0] and list(y) == [10.0, 0.0, -30.0]
    for turn, exit_m in sim.EXIT_PATH_M.items():
        if turn == "T":
            continue
        x, y = sim.path_xy(turn, 5.0, np.array([0.0, -exit_m]))
        assert y[1] <= -8.0                              # far enough past the line for the turn rule
        assert (x[1] - x[0] < -3.0) if turn == "R" else (x[1] - x[0] > 3.0)


@pytest.fixture(scope="module")
def approach_run():
    rng = np.random.default_rng(7)
    return sim.simulate_approach("N", sim.SignalPlan(), {"L": 90, "T": 700, "R": 90}, 6000, 7 * 3600 - 240, rng)


def test_simulated_vehicles_obey_the_physics(approach_run):
    assert len(approach_run) > 100
    for veh in approach_run:
        speed = np.array(veh.v_hist)
        change = np.diff(speed) / sim.DT
        assert speed.min() >= 0 and speed.max() <= sim.FREE_SPEED + 1e-9
        assert change.max(initial=0) <= sim.ACCEL + 1e-6
        assert change.min(initial=0) >= -sim.DECEL - 1e-6
        assert np.all(np.diff(np.array(veh.s_hist)) <= 1e-9)   # never rolls backward


def test_simulated_vehicles_never_touch(approach_run):
    by_lane = {}
    for veh in sorted(approach_run, key=lambda v: v.step0):
        by_lane.setdefault(veh.lane, []).append(veh)
    for lane in by_lane.values():
        for ahead, behind in zip(lane, lane[1:]):
            offset = behind.step0 - ahead.step0
            n = min(len(ahead.s_hist) - offset, len(behind.s_hist))
            if n <= 0:
                continue
            front = np.array(ahead.s_hist[offset:offset + n])
            same_path = (front > -sim.DIVERGE_M) | (ahead.turn == behind.turn)
            gap = np.array(behind.s_hist[:n]) - front - ahead.length
            assert gap[same_path].min(initial=99.0) >= sim.STANDSTILL_GAP - 0.1


def test_nobody_runs_the_red(approach_run):
    plan = sim.SignalPlan()
    for veh in approach_run:
        s = np.array(veh.s_hist)
        past = np.nonzero(s <= 0)[0]
        if len(past) and past[0] > 0:
            clock = (7 * 3600 - 240 + (veh.step0 + past[0]) * sim.DT) % plan.cycle
            assert clock < 50.0 + plan.yellow, f"crossed {clock:.1f} s into the cycle"


def test_sample_homography_puts_the_road_in_the_picture():
    import sample_site
    to_ground = np.array(sample_site.sample_homography())
    site = sample_site.build_site()
    for point in site["cameras"]["A"]["calib_points"]:
        u, v = point["px"]
        assert 0 <= u <= 1920 and 0 <= v <= 1080
        x, y, w = to_ground @ np.array([u, v, 1.0])
        assert math.isclose(x / w, point["m"][0], abs_tol=0.05) and math.isclose(y / w, point["m"][1], abs_tol=0.3)


def test_generator_is_repeatable_and_night_hours_are_labeled(tmp_path):
    """Same seed -> the same vehicles as the committed sample; --night-hours labels rows."""
    gen.main(["--out", str(tmp_path), "--days", "1", "--hours", "7", "--night-hours", "7"])
    day_dir = tmp_path / "2026-05-12"
    assert {name: p for name, p in schema.validate_day_folder(day_dir).items() if p} == {}
    new = readers.read_rows(day_dir / "vehicles_07.csv")
    old = readers.read_rows(SAMPLE / "2026-05-12" / "vehicles_07.csv")
    assert len(new) == len(old)
    for new_row, old_row in zip(new, old):
        for column in ("vehicle_id", "lane", "turn", "type", "stops", "queue_pos", "t_left"):
            assert new_row[column] == old_row[column]
        assert abs(float(new_row["wait_s"]) - float(old_row["wait_s"])) <= 0.11
        assert new_row["light"] == "night" and "night_mode" in new_row["flags"]
    assert "night mode on" in (day_dir / "events.log").read_text()
