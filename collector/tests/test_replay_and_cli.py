"""--replay on a video file, and refusing to start with a bad site.json."""
import json

import cv2
import numpy as np
import pytest

from common import readers, schema
from helpers import SAMPLE_DAY
from monitor import main as cli


def test_replay_a_video_file_with_the_fake_detector(tmp_path):
    """A 6 s test video (plain gray frames made here, not a camera picture) and boxes for one car."""
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 180))
    for _ in range(180):
        writer.write(np.full((180, 320, 3), 100, np.uint8))
    writer.release()
    site = json.loads((SAMPLE_DAY / "site.json").read_text())
    (tmp_path / "site.json").write_text(json.dumps(site))
    import synthetic
    from monitor.ground import GroundMapper
    to_image = GroundMapper(site["cameras"]["A"]["homography"]).inverse
    lines = ["t,camera,x0,y0,x1,y1,score,cls"]
    for i in range(60):               # 10 fps from the 30 fps video
        t = round(i * 0.1, 3)
        box = synthetic.box_for(to_image, 5.0, 60.0 - 13.4 * t, "car")
        if box:
            lines.append(f"{t},A,{box[0]},{box[1]},{box[2]},{box[3]},0.9,car")
    (tmp_path / "boxes.csv").write_text("\n".join(lines) + "\n")
    out = tmp_path / "data_test"
    cli.main(["--replay", str(video), "--camera", "A", "--detector", "fake", "--fake-boxes", str(tmp_path / "boxes.csv"),
              "--site", str(tmp_path / "site.json"), "--out", str(out), "--start", "2026-05-12T07:30:00"])
    day = out / "2026-05-12"
    assert {name: p for name, p in schema.validate_day_folder(day).items() if p} == {}
    rows = readers.read_rows(day / "vehicles_07.csv")
    assert len(rows) == 1 and rows[0]["t_crossed"].startswith("2026-05-12T07:30:04") and rows[0]["lane"] == "2"
    assert "startup: replay of clip.avi" in (day / "events.log").read_text()


def test_bad_site_json_is_refused_with_the_key_named(tmp_path, capsys):
    site = json.loads((SAMPLE_DAY / "site.json").read_text())
    site["cameras"]["B"]["approach"] = "Q"
    (tmp_path / "site.json").write_text(json.dumps(site))
    with pytest.raises(SystemExit) as stop:
        cli.main(["--site", str(tmp_path / "site.json"), "--data", str(tmp_path)])
    assert stop.value.code == 2
    assert "cameras.B.approach" in capsys.readouterr().err


def test_example_site_json_is_valid():
    from monitor.config import REPO_DIR, load_site
    site = load_site(REPO_DIR / "collector" / "site.example.json")
    assert all(cam.homography is None for cam in site.cameras.values())
