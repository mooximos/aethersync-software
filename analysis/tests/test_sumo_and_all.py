"""The SUMO model, and the whole `all` command on sample_data."""
import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from tlopt import sumo_model as sm
from tlopt.cli import main
from tlopt_helpers import EXAMPLE, SAMPLE, intersection

PLAN = {"cycle_s": 120, "phases": {"2": {"green_s": 50, "yellow_s": 4, "all_red_s": 2},
                                   "6": {"green_s": 50, "yellow_s": 4, "all_red_s": 2},
                                   "4": {"green_s": 54, "yellow_s": 4, "all_red_s": 6},
                                   "8": {"green_s": 54, "yellow_s": 4, "all_red_s": 6}}}


@pytest.fixture(scope="module")
def network(tmp_path_factory):
    inter = intersection()
    net = sm.build_network(inter, tmp_path_factory.mktemp("net"))
    return inter, net, sm.signal_links(net)


def test_network_has_every_movement_and_crosswalk(network):
    _inter, _net, links = network
    vehicle = {(l.approach, l.movement) for l in links if l.approach}
    assert vehicle == {(a, m) for a in "NSEW" for m in "LTR"}
    assert sorted(l.crossing_leg for l in links if l.crossing_leg) == ["E", "N", "S", "W"]


def test_signal_program_follows_the_plan(network):
    inter, _net, links = network
    logic = sm.tls_program(inter, PLAN, links, "current")
    phases = [(float(p.get("duration")), p.get("state")) for p in logic]
    assert sum(d for d, _ in phases) == pytest.approx(120)
    index = {(l.approach, l.movement): l.index for l in links if l.approach}
    walk = {l.crossing_leg: l.index for l in links if l.crossing_leg}

    def time_in(state_char, key):
        return sum(d for d, s in phases if s[index[key]] in state_char)
    assert time_in("Gg", ("N", "T")) == pytest.approx(50) and time_in("y", ("N", "T")) == pytest.approx(4)
    assert time_in("Gg", ("E", "T")) == pytest.approx(54)
    assert all(s[index[("N", "L")]] in "gry" for _, s in phases)       # permissive left yields
    assert sum(d for d, s in phases if s[walk["E"]] == "G") == pytest.approx(7)   # WALK beside N/S traffic
    assert not any(s[index[("N", "T")]] in "Gg" and s[index[("E", "T")]] in "Gg" for _, s in phases)


def test_routes_match_the_demand(tmp_path):
    inter = intersection()
    demand = {(0, "N", "T", "car"): 50.0, (0, "E", "L", "truck"): 8.0, (-1, "N", "T", "car"): 50.0}
    path = sm.write_routes(inter, demand, tmp_path, seed=3, warmup_s=900, tau=1.2)
    root = ET.parse(path).getroot()
    cars = [v for v in root.iter("vehicle")]
    departs = [float(v.get("depart")) for v in cars]
    assert departs == sorted(departs) and 60 < len(cars) < 150
    assert {v.find("route").get("edges") for v in cars} == {"N_in S_out", "E_in S_out"}
    assert root.find("vType[@id='car']").get("emissionClass") == "HBEFA4/PC_petrol_Euro-6ab"
    assert root.find("vType[@id='truck']").get("emissionClass").startswith("HBEFA4/RT_le7.5t")


def test_all_on_sample_data(tmp_path):
    results = tmp_path / "results"
    main(["all", str(SAMPLE), "--intersection", str(EXAMPLE), "--results", str(results), "--seeds", "3"])
    for period in ("am", "pm"):
        current = json.loads((results / "plans" / f"{period}_current.json").read_text())
        best = json.loads((results / "plans" / f"{period}_optimized.json").read_text())
        webster = json.loads((results / "plans" / f"{period}_webster.json").read_text())
        assert best["predicted"]["avg_delay_s"] <= current["predicted"]["avg_delay_s"]
        assert best["predicted"]["avg_delay_s"] <= webster["predicted"]["avg_delay_s"]
        assert best["notes"] == [] or all("Webster" in n for n in best["notes"])
    summary = pd.read_csv(results / "sumo" / "summary.csv")
    delay = summary[summary["metric"] == "delay_s"].set_index(["period", "plan"])["mean"]
    assert delay[("am", "Optimized")] < delay[("am", "Current")]
    calibration = json.loads((results / "sumo" / "calibration.json").read_text())
    assert calibration["within_5pct"]
    html = (results / "REPORT.html").read_text()
    assert "data:image/png;base64," in html and "9. Limitations" in html and "src='figures" not in html
    md = (results / "REPORT.md").read_text()
    for heading in ("1. What was measured", "5. Does the model match reality?", "7. Predicted and simulated change"):
        assert heading in md
    assert (results / "measures" / "vehicle_delays.csv").exists()
