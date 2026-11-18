"""HCM delay, ITE timings and Webster, checked against numbers worked out by hand."""
import math

import pytest

from tlopt import hcm, optimize
from tlopt_helpers import example_dict, intersection, light_problem


def test_lane_group_delay_hand_worked():
    # v = 450 veh/h, s = 1800 veh/h, g = 30 s, C = 90 s:
    # c = 1800 * 30 / 90 = 600, X = 0.75
    # d1 = 0.5 * 90 * (1 - 1/3)^2 / (1 - 0.75 / 3) = 20.0 / 0.75 = 26.667
    # d2 = 900 * 0.25 * (-0.25 + sqrt(0.0625 + 8*0.5*1*0.75 / (600*0.25))) = 225 * (sqrt(0.0825) - 0.25) = 8.376
    r = hcm.lane_group_delay(450, 1800, 30, 90, rp=1.0)
    assert r["capacity"] == pytest.approx(600) and r["X"] == pytest.approx(0.75)
    assert r["d1"] == pytest.approx(26.667, abs=0.001)
    assert r["d2"] == pytest.approx(8.376, abs=0.001)
    assert r["PF"] == pytest.approx(1.0)                      # random arrivals: P = g/C
    assert r["d"] == pytest.approx(35.043, abs=0.001)


def test_progression_factor_hand_worked():
    # Rp = 1.5: P = min(1, 1.5 * 1/3) = 0.5, PF = (1 - 0.5) / (1 - 1/3) = 0.75, d = 20.0 + 8.376
    r = hcm.lane_group_delay(450, 1800, 30, 90, rp=1.5)
    assert r["PF"] == pytest.approx(0.75) and r["d"] == pytest.approx(28.376, abs=0.001)
    assert hcm.progression_factor(5.0, 30, 90) == hcm.progression_factor(2.0, 30, 90)   # Rp clipped to 2.0
    assert hcm.progression_factor(0.1, 30, 90) == hcm.progression_factor(0.5, 30, 90)   # and to 0.5


def test_oversaturated_group_uses_x_capped_at_1_in_d1():
    # v = 900, c = 600, X = 1.5: d1 = 0.5*90*(2/3)^2 / (1 - 1/3) = 30.0
    # d2 = 225 * (0.5 + sqrt(0.25 + 6/150)) = 225 * (0.5 + 0.538516) = 233.666
    r = hcm.lane_group_delay(900, 1800, 30, 90)
    assert r["d1"] == pytest.approx(30.0) and r["d2"] == pytest.approx(233.666, abs=0.001)


def test_evaluate_weights_by_volume():
    groups = [hcm.GroupInput("N", "TR", 2, 450, 1800, [2]), hcm.GroupInput("N", "L", 1, 50, 1800, [2])]
    result = hcm.evaluate(groups, {2: 30.0}, 90.0)
    d_main = hcm.lane_group_delay(450, 1800, 30, 90)["d"]
    d_left = hcm.lane_group_delay(50, 1800, 30, 90)["d"]
    assert result["by_approach"]["N"] == pytest.approx((450 * d_main + 50 * d_left) / 500)
    assert result["veh_hours_per_hour"] == pytest.approx((450 * d_main + 50 * d_left) / 3600)


@pytest.mark.parametrize("mph,grade,expected", [
    (30, 0, 3.2),     # 1 + 44 / 20 = 3.2
    (30, -3, 3.5),    # 1 + 44 / (20 - 2*0.03*32.2) = 3.435 -> rounded up to 3.5
    (45, 0, 4.3),     # 1 + 66 / 20 = 4.3
    (10, 0, 3.0),     # 1.73 -> minimum 3.0
])
def test_ite_yellow(mph, grade, expected):
    assert hcm.yellow_s(mph, grade) == expected


@pytest.mark.parametrize("width,mph,expected", [(80, 30, 2.3), (80, 45, 1.6), (10, 60, 1.0)])
def test_ite_all_red(width, mph, expected):
    # (W + 20 ft) / V, rounded up to 0.1 s, at least 1.0 s: 100/44 = 2.27, 100/66 = 1.52, 30/88 = 0.34
    assert hcm.all_red_s(width, mph) == expected


def test_pedestrian_minimum_green():
    assert hcm.ped_min_green_s(7, 64, 3.5) == 25.3        # 7 + 18.29
    assert math.isclose(hcm.ped_min_green_s(7, 35, 3.5), 17.0)


def test_safety_uses_the_perpendicular_crosswalk():
    data = example_dict()
    data["approaches"]["E"]["ped_crossing_ft"] = 100       # crossing the east leg runs beside N/S traffic
    inter = intersection(approaches=data["approaches"])
    _y, _ar, min_green, why = optimize.safety(inter)
    assert min_green[2] == min_green[6] == hcm.ped_min_green_s(7, 100, 3.5)
    assert min_green[4] == 25.3 and "pedestrians" in why[2]


def test_webster_hand_worked():
    # y = 540/1800 = 0.3 (N/S) and 360/1800 = 0.2 (E/W); Y = 0.5; L = 2 x (3.2 + 2.3) = 11 s
    # C0 = (1.5 * 11 + 5) / (1 - 0.5) = 43 s; green 32 s split 0.6 / 0.4 = 19.2 / 12.8 -> 19 / 13
    plan = optimize.webster(light_problem(540, 360))
    assert plan.cycle_s == 43.0
    assert plan.greens == {2: 19.0, 6: 19.0, 4: 13.0, 8: 13.0}
    assert "C0 = (1.5 L + 5) / (1 - Y) = 43.0 s" in plan.notes[1]


def test_webster_cycle_is_kept_inside_the_bounds():
    assert optimize.webster(light_problem(540, 360, bounds=(60, 180))).cycle_s == 60.0
    assert optimize.webster(light_problem(900, 900)).cycle_s == 180.0   # Y = 1: no Webster answer
