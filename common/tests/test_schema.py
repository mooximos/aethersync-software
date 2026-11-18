"""The validators must accept the sample data and reject broken files."""
import copy
import gzip
import json
import shutil
from pathlib import Path

import pytest

from common import readers, schema
from common.site import SiteError, load_site, validate_site

SAMPLE = Path(__file__).resolve().parents[2] / "sample_data"
DAY = SAMPLE / "2026-05-12"


def test_sample_data_has_two_weekdays_with_hours_07_and_17():
    days = readers.list_day_folders(SAMPLE)
    assert [d.name for d in days] == ["2026-05-12", "2026-05-13"]
    for day in days:
        for kind in readers.HOURLY_KINDS:
            assert [hour for hour, _ in readers.hour_files(day, kind)] == [7, 17], (day, kind)
        assert (day / "tracks_07.csv.gz").exists() and not (day / "tracks_07.csv").exists()
        assert (day / "clock_sync.csv").exists() and (day / "events.log").exists()


@pytest.mark.parametrize("day", ["2026-05-12", "2026-05-13"])
def test_every_sample_file_is_valid(day):
    results = schema.validate_day_folder(SAMPLE / day)
    assert len(results) == 10   # 4 kinds x 2 hours + clock_sync.csv + site.json
    assert {name: problems for name, problems in results.items() if problems} == {}


def _copy(tmp_path: Path, name: str) -> Path:
    day_dir = tmp_path / DAY.name
    day_dir.mkdir(exist_ok=True)
    return Path(shutil.copy(DAY / name, day_dir / name))


def _edit_line(path: Path, line_index: int, change) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    lines[line_index] = change(lines[line_index])
    path.write_text("\n".join(lines), encoding="utf-8")


def _set_cell(column: str, value: str, columns: list[str]):
    def change(line: str) -> str:
        cells = line.split(",")
        cells[columns.index(column)] = value
        return ",".join(cells)
    return change


def test_wrong_column_order_is_rejected(tmp_path):
    path = _copy(tmp_path, "vehicles_07.csv")
    _edit_line(path, 0, lambda header: header.replace("camera,approach", "approach,camera"))
    problems = schema.validate_file(path)
    assert len(problems) == 1 and "wrong order" in problems[0]


def test_missing_column_is_rejected(tmp_path):
    path = _copy(tmp_path, "system_07.csv")
    _edit_line(path, 0, lambda header: header.replace(",disk_free_gb", ""))
    assert "wrong columns" in schema.validate_file(path)[0]


def test_bad_time_format_is_rejected(tmp_path):
    path = _copy(tmp_path, "vehicles_07.csv")
    _edit_line(path, 5, _set_cell("t_left", "2026-05-12 07:00:05", schema.VEHICLE_COLUMNS))
    problems = schema.validate_file(path)
    assert len(problems) == 1 and "line 6" in problems[0] and "t_left" in problems[0]


@pytest.mark.parametrize("column,value,expect", [
    ("turn", "X", "turn 'X'"),
    ("type", "bicycle", "type 'bicycle'"),
    ("approach", "NE", "approach 'NE'"),
    ("light", "dusk", "light 'dusk'"),
    ("flags", "made_up_flag", "unknown flag 'made_up_flag'"),
    ("vehicle_id", "A-12", "vehicle_id 'A-12'"),
    ("conf", "0.9", "conf '0.9'"),
    ("wait_s", "3", "wait_s '3'"),
])
def test_unknown_values_are_rejected(tmp_path, column, value, expect):
    path = _copy(tmp_path, "vehicles_07.csv")
    _edit_line(path, 3, _set_cell(column, value, schema.VEHICLE_COLUMNS))
    problems = schema.validate_file(path)
    assert any("line 4" in p and expect in p for p in problems), problems


def test_row_in_the_wrong_hour_file_is_rejected(tmp_path):
    path = _copy(tmp_path, "system_07.csv")
    _edit_line(path, 2, _set_cell("time", "2026-05-12T08:01:00.0-07:00", schema.SYSTEM_COLUMNS))
    assert "hour-07 file" in schema.validate_file(path)[0]


def test_signal_event_code_and_name_must_match(tmp_path):
    path = _copy(tmp_path, "signal_events_07.csv")
    _edit_line(path, 1, _set_cell("event", "detector_off", schema.SIGNAL_COLUMNS))
    assert "does not match event_code 82" in schema.validate_file(path)[0]
    _edit_line(path, 1, _set_cell("event_code", "99", schema.SIGNAL_COLUMNS))
    assert "event_code '99'" in schema.validate_file(path)[0]


def test_gzipped_tracks_are_checked_too(tmp_path):
    day_dir = tmp_path / DAY.name
    day_dir.mkdir()
    with gzip.open(DAY / "tracks_07.csv.gz", "rt", encoding="utf-8") as handle:
        lines = [next(handle) for _ in range(50)]
    lines[10] = lines[10].replace(",A-", ",Z-")
    with gzip.open(day_dir / "tracks_07.csv.gz", "wt", encoding="utf-8") as handle:
        handle.writelines(lines)
    problems = schema.validate_file(day_dir / "tracks_07.csv.gz")
    assert len(problems) == 1 and "line 11" in problems[0] and "vehicle_id" in problems[0]


def test_row_with_too_few_cells_is_reported_not_crashed(tmp_path):
    path = _copy(tmp_path, "system_07.csv")
    _edit_line(path, 4, lambda line: line.rsplit(",", 3)[0])
    assert "wrong number of cells" in schema.validate_file(path)[0]


def test_unknown_file_name_is_reported(tmp_path):
    path = tmp_path / "vehicles_7.csv"
    path.write_text("x\n")
    assert "not a known data file name" in schema.validate_file(path)[0]


def test_site_json_of_the_sample_loads():
    site = load_site(DAY / "site.json")
    assert site.time_zone == "America/Los_Angeles"
    assert sorted(site.cameras) == ["A", "B", "C", "D"]
    assert site.camera_for_approach("N") == "A"
    assert site.cameras["A"].lanes["1"].movements == ["R", "T"]
    assert site.privacy == "no images or video are stored"


@pytest.mark.parametrize("path,value,key_in_message", [
    (("cameras", "A", "approach"), "X", "cameras.A.approach"),
    (("cameras", "B", "lane_x_m", "2"), [7.0, 3.5], "lane_x_m.2"),
    (("cameras", "C", "homography"), [[1, 0], [0, 1]], "cameras.C"),
    (("detection", "min_confidence"), 1.5, "detection.min_confidence"),
    (("detection", "classes"), ["car", "bicycle"], "detection.classes.1"),
    (("schedule", "windows", 0, "start"), "6:30am", "schedule.windows.0"),
    (("schedule", "windows", 1, "days"), ["Mon", "Funday"], "schedule.windows.1.days.1"),
    (("time_zone",), "Mars/Olympus", "time_zone"),
    (("event_codes",), {"82": "detector_on"}, "event_codes"),
    (("latitude",), 123.0, "latitude"),
    (("longitude",), 122.0, "minus sign"),          # the sample site is at -121.87 in America/Los_Angeles
])
def test_bad_site_values_name_the_bad_key(path, value, key_in_message):
    data = json.loads((DAY / "site.json").read_text())
    target = data
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = copy.deepcopy(value)
    with pytest.raises(SiteError) as error:
        validate_site(data)
    assert key_in_message in str(error.value)


def test_site_still_at_zero_zero_is_rejected():
    data = json.loads((DAY / "site.json").read_text())
    data["latitude"] = data["longitude"] = 0.0
    with pytest.raises(SiteError, match="still 0.0, 0.0"):
        validate_site(data)


def test_site_with_misspelled_or_missing_key_is_rejected():
    data = json.loads((DAY / "site.json").read_text())
    data["stoped_speed_mps"] = 0.5
    del data["schedule"]
    with pytest.raises(SiteError) as error:
        validate_site(data)
    assert "stoped_speed_mps" in str(error.value) and "schedule" in str(error.value)


def test_site_before_calibration_is_allowed():
    data = json.loads((DAY / "site.json").read_text())
    data["cameras"]["A"]["homography"] = None
    data["cameras"]["A"]["calib_points"] = []
    assert validate_site(data).cameras["A"].homography is None


def test_missing_site_file_gives_a_clear_message(tmp_path):
    with pytest.raises(SiteError, match="does not exist"):
        load_site(tmp_path / "site.json")
    (tmp_path / "site.json").write_text("{not json")
    with pytest.raises(SiteError, match="not valid JSON"):
        load_site(tmp_path / "site.json")


def test_windows_extract_all_nesting_is_handled(tmp_path):
    outer = tmp_path / DAY.name
    shutil.copytree(DAY, outer / DAY.name)     # data/D/D/files, as Windows "Extract All" makes it
    assert readers.list_day_folders(tmp_path) == [outer / DAY.name]
    assert readers.list_day_folders(outer) == [outer / DAY.name]
    (tmp_path / "2026-05-13.zip").write_bytes(b"")
    assert readers.zips_not_unzipped(tmp_path) == ["2026-05-13.zip"]
