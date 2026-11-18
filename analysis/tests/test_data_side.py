"""Loading, quality checks, measures and the inferred timing, on sample_data."""
import shutil

import numpy as np
import pytest
import yaml

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


def test_study_days_and_skip_dates_leave_whole_days_out(prepared):
    default, ds, _ = prepared
    try:
        inter = intersection(options={"days": ["Tue"]})        # 2026-05-12 is a Tuesday, 05-13 a Wednesday
        table = study.prepare(ds, inter).set_index("rule")["vehicles"]
        wednesday = int((ds.vehicles["day"] == "2026-05-13").sum())
        assert ds.study_days == ["2026-05-12"] and table["day of week not in days (Tue)"] == wednesday
        assert not ds.vehicles.loc[ds.vehicles["day"] == "2026-05-13", "counted"].any()
        inter = intersection(options={"skip_dates": ["2026-05-12"]})
        table = study.prepare(ds, inter).set_index("rule")["vehicles"]
        assert ds.study_days == ["2026-05-13"]
        assert table["date listed in skip_dates"] == int((ds.vehicles["day"] == "2026-05-12").sum())
    finally:
        study.prepare(ds, default)


def test_an_empty_period_says_why(prepared, tmp_path):
    default, ds, _ = prepared
    try:
        inter = intersection(periods={"noon": "12:00-13:00"})
        study.prepare(ds, inter)
        with pytest.raises(ValueError, match="Hours in the data: 07:00, 17:00"):
            study.measure(ds, inter, "noon")
        inter = intersection(options={"days": ["Sat", "Sun"]})
        study.prepare(ds, inter)
        with pytest.raises(ValueError, match="is a study day"):
            study.measure(ds, inter, "am")
    finally:
        study.prepare(ds, default)


def test_a_bad_skip_date_is_refused(tmp_path):
    data = example_dict()
    data["options"] = {"skip_dates": ["Dec 24"]}
    path = tmp_path / "intersection.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(IntersectionError, match="skip_dates"):
        load_intersection(path)


def test_a_day_in_two_places_is_used_once(tmp_path):
    shutil.copytree(SAMPLE / "2026-05-12", tmp_path / "copy" / "2026-05-12")
    ds = load([SAMPLE, tmp_path / "copy"], with_tracks=False)
    assert ds.vehicles["key"].is_unique and ds.days == ["2026-05-12", "2026-05-13"]
    assert any("is in two places" in p for p in ds.problems)


def _unplug_camera(day_dir, camera: str, hour: int) -> None:
    """Make one hour look as if a camera was unplugged: its rows gone, its frame rate 0."""
    import csv
    import gzip
    import io
    for name in (f"vehicles_{hour:02d}.csv", f"tracks_{hour:02d}.csv.gz", f"signal_events_{hour:02d}.csv"):
        path = day_dir / name
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        id_col = rows[0].index("vehicle_id") if "vehicle_id" in rows[0] else rows[0].index("detail")
        keep = [rows[0]] + [r for r in rows[1:] if not r[id_col].startswith(f"{camera}-")]
        if "approach" in rows[0] and name.startswith("signal"):   # approach-level events of that camera too
            keep = [keep[0]] + [r for r in keep[1:] if r[rows[0].index("approach")] != "E"]
        out = io.StringIO()
        csv.writer(out, lineterminator="\n").writerows(keep)
        with opener(path, "wt", encoding="utf-8", newline="") as f:
            f.write(out.getvalue())
    path = day_dir / f"system_{hour:02d}.csv"
    rows = list(csv.reader(open(path, encoding="utf-8")))
    col = rows[0].index(f"fps_{camera}")
    for r in rows[1:]:
        r[col] = "0.0"
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


def test_a_day_with_a_dead_camera_is_not_counted_as_no_traffic(prepared, tmp_path):
    inter, _ds, _ = prepared
    for day in ("2026-05-12", "2026-05-13"):
        shutil.copytree(SAMPLE / day, tmp_path / day)
    _unplug_camera(tmp_path / "2026-05-13", "B", 7)          # camera B looks at approach E
    broken = load([tmp_path])
    study.prepare(broken, inter)
    p = study.measure(broken, inter, "am")
    cov = p.tables["coverage"].set_index(["day", "approach"])
    assert not cov.loc[("2026-05-13", "E"), "used"] and "0 of 60 minutes" in cov.loc[("2026-05-13", "E"), "why_not"]
    assert cov["used"].sum() == 7
    one_day = load([tmp_path / "2026-05-12"])
    study.prepare(one_day, inter)
    reference = {(g.approach, g.name): g.v for g in study.measure(one_day, inter, "am").groups}
    for g in p.groups:
        if g.approach == "E":   # the E design flow comes from the good day only, not halved
            assert g.v == reference[("E", g.name)]


def test_damaged_files_are_reported_and_the_rest_is_used(tmp_path):
    day = tmp_path / "2026-05-12"
    shutil.copytree(SAMPLE / "2026-05-12", day)
    packed = (day / "tracks_17.csv.gz").read_bytes()
    (day / "tracks_17.csv.gz").write_bytes(packed[: len(packed) // 2])        # a download cut off halfway
    (day / "tracks_07.csv.gz.tmp").write_bytes(b"half-written by a power cut")  # left over, ignored
    text = (day / "vehicles_07.csv").read_text(encoding="utf-8")
    (day / "vehicles_07.csv").write_text("﻿" + text.replace("\n", "\r\n"), encoding="utf-8")  # BOM + CRLF
    ds = load([tmp_path])
    assert any("tracks_17.csv.gz" in p and "cannot be read" in p for p in ds.problems)
    assert not any("vehicles_07" in p for p in ds.problems)                     # BOM and CRLF are fine
    assert ds.tracks["file"].astype(str).str.contains("tracks_07").all()
    assert len(ds.vehicles) == sum(1 for _ in open(SAMPLE / "2026-05-12" / "vehicles_07.csv")) - 1 + \
        sum(1 for _ in open(SAMPLE / "2026-05-12" / "vehicles_17.csv")) - 1
