"""Loading, quality checks, measures and the inferred timing, on sample_data."""
import shutil

import numpy as np
import pytest

from tlopt import measures, quality, study, timing, validate
from tlopt.intersection import IntersectionError, load_intersection
from tlopt.load import load
from tlopt_helpers import EXAMPLE, SAMPLE, example_dict, intersection

TRUE_GREEN = {"N": 50.0, "S": 50.0, "E": 54.0, "W": 54.0}   # the generator's plan


@pytest.fixture(scope="module")
def prepared():
    inter = load_intersection(EXAMPLE)
    ds = load([SAMPLE])
    excluded = study.prepare(ds, inter)
    return inter, ds, excluded


def test_load_reads_every_file_with_no_problems(prepared):
    _inter, ds, _excluded = prepared
    assert ds.days == ["2026-05-12", "2026-05-13"] and ds.problems == []
    assert len(ds.vehicles) > 10000 and len(ds.tracks) > 700000
    assert set(ds.vehicles["weekday"]) == {"Tue", "Wed"}
    assert ds.vehicles["key"].is_unique


def test_bad_rows_and_files_are_reported_not_fatal(tmp_path):
    day = tmp_path / "2026-05-12"
    shutil.copytree(SAMPLE / "2026-05-12", day)
    lines = (day / "vehicles_07.csv").read_text().splitlines()
    lines[5] = lines[5].replace(",day,", ",dusk,")
    (day / "vehicles_07.csv").write_text("\n".join(lines) + "\n")
    (day / "system_17.csv").write_text("not,the,right,header\n1,2,3,4\n")
    ds = load([tmp_path], with_tracks=False)
    assert any("vehicles_07.csv line 6" in p for p in ds.problems)
    assert any("system_17.csv: skipped" in p for p in ds.problems)
    assert len(ds.vehicles) == sum(1 for _ in open(SAMPLE / "2026-05-12" / "vehicles_07.csv")) - 2 + \
        sum(1 for _ in open(SAMPLE / "2026-05-12" / "vehicles_17.csv")) - 1


def test_exclusions_are_counted(prepared):
    _inter, ds, excluded = prepared
    table = excluded.set_index("rule")["vehicles"]
    assert table["flag no_cross"] == int(ds.vehicles["flag_no_cross"].sum())
    assert not ds.vehicles.loc[ds.vehicles["flag_no_cross"], "delay_ok"].any()
    assert ds.vehicles["delay_ok"].sum() > 0.9 * len(ds.vehicles)


def test_consistency_checks(prepared):
    _inter, ds, _ = prepared
    assert quality.detector_vs_crossings(ds)["within_3pct"].all()
    lost = quality.entering_vs_crossing(ds)
    assert (lost["lost_pct"] < 5).all()
    assert (quality.cycle_stability(ds)["median_cycle_s"] - 120).abs().max() <= 0.5
    assert (quality.quality_table(ds)["outages"] <= 1).all()


def test_control_delay_is_close_to_the_wait(prepared):
    _inter, ds, _ = prepared
    v = ds.vehicles[ds.vehicles["delay_ok"]]
    never = v[v["stops"] == 0]
    assert never["control_delay_s"].median() < 1.0      # free-flowing cars have almost no delay
    stopped = v[v["stops"] > 0]
    gap = stopped["control_delay_s"] - stopped["wait_s"]
    assert 0 < gap.median() < 8                         # plus the time lost slowing down and speeding up


@pytest.mark.parametrize("period", ["am", "pm"])
def test_timing_inference_recovers_the_generator_plan(prepared, period):
    inter, ds, _ = prepared
    p = study.measure(ds, inter, period)
    for approach, inferred in p.timing.items():
        assert abs(inferred.cycle_s - 120.0) <= 2.0
        assert abs(inferred.green_s - TRUE_GREEN[approach]) <= 3.0, (approach, inferred.green_s)


def test_saturation_flow_from_the_5th_queued_vehicle(prepared):
    inter, ds, _ = prepared
    p = study.measure(ds, inter, "am")
    sat = p.tables["saturation_flow"].set_index(["approach", "group"])
    n_tr = sat.loc[("N", "TR")]
    assert n_tr["source"] == "measured" and n_tr["headways"] >= 30
    assert 1550 <= n_tr["sat_flow_vph"] / 2 <= 1850        # generator: 2.0 s headway, slower right turners
    assert (sat.loc[sat["source"] == "default", "sat_flow_vph"] < 1800 * 3).all()


def test_measures_tables(prepared):
    inter, ds, _ = prepared
    p = study.measure(ds, inter, "pm")
    flows = p.tables["flows"].set_index(["approach", "turn"])
    assert flows.loc[("S", "T"), "flow_vph"] > flows.loc[("E", "T"), "flow_vph"]   # pm is heavy from the south
    busy = p.tables["flows"][p.tables["flows"]["flow_vph"] >= 50]   # a PHF of 1-2 vehicles means little
    assert (busy["phf"].between(0.5, 1.0)).all()
    assert set(p.tables["delay_by_approach"]["LOS"]) <= set("ABCDEF")
    assert (p.tables["cycle_failures"]["cycle_failures"] == 0).all()
    assert (p.tables["arrivals"]["Rp"].between(0.7, 1.3)).all()    # random arrivals in the generator


def test_model_check_numbers(prepared):
    inter, ds, _ = prepared
    p = study.measure(ds, inter, "am")
    study.model(p, ds, inter)
    raw = p.model_check["raw"]
    assert raw["n"] == 32 and raw["MAPE_pct"] < 15
    assert "leave-one-day-out" in p.model_check["reason"]


def test_validate_scores_and_calibration_rule():
    import pandas as pd
    table = pd.DataFrame({"day": ["d1", "d1", "d2", "d2"], "interval": ["07:00"] * 4, "approach": ["N", "S"] * 2,
                          "vehicles": [10] * 4, "measured_s": [20.0, 30.0, 22.0, 33.0],
                          "predicted_s": [10.0, 15.0, 11.0, 16.0]})
    result = validate.calibrate(table)
    assert result["raw"]["MAE_s"] == pytest.approx(13.25)
    assert result["used"] and result["factors"]["N"] == pytest.approx(2.0, abs=0.01)


def test_intersection_errors_name_the_key(tmp_path):
    data = example_dict()
    data["phases"]["2"]["approach"] = "Q"
    import yaml
    (tmp_path / "x.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(IntersectionError, match="phases.2.approach"):
        load_intersection(tmp_path / "x.yaml")
    data = example_dict()
    del data["phases"]["8"]
    data["timing_sheet"]["phases"].pop("8")
    (tmp_path / "y.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(IntersectionError, match="no phase serves movement"):
        load_intersection(tmp_path / "y.yaml")


def test_lane_groups_follow_the_lanes():
    groups = measures.lane_groups(intersection())
    assert {(g.approach, g.name, tuple(g.lanes)) for g in groups if g.approach == "N"} == {("N", "TR", (1, 2)), ("N", "L", (3,))}
