"""The real collector main loop (Service.run) with fake cameras and a fake detector.

Checks the whole chain in real time: a recording window starts recording, a car is tracked
and written as a valid vehicles row, the files are closed on shutdown, nothing is powered off,
and not a single image is written to disk (rule 1)."""
import json
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from common import readers, schema
from common.site import validate_site
from helpers import SAMPLE_DAY
from monitor import service as service_module
from monitor.detector import Detection, Detector
from monitor.ground import GroundMapper
from monitor.webstate import WebState
from synthetic import box_for

SPEED_MPS, START_Y, END_Y = 15.0, 45.0, -15.0
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"\x00\x00\x00\x18ftyp", b"\x1aE\xdf\xa3", b"\x93NUMPY")


class FakeCapture:
    """Stands in for CameraCapture: hands out a gray 1920x1080 frame every 0.1 s."""
    def __init__(self, name, device, fps, log, *args, **kwargs):
        self.name, self.started_at, self.last = name, None, -1.0
        self.image = np.full((1080, 1920, 3), 90, np.uint8)
        self.dropped, self.connected = 0, True

    def start(self):
        self.started_at = time.monotonic()

    def set_rate(self, fps):
        pass

    def take(self):
        now = time.monotonic()
        if now - self.last < 0.1:
            return None
        self.last = now
        return self.image, now

    def stop(self):
        pass


class OneCarDetector(Detector):
    """Camera A sees one car drive from 45 m before the stop line to 15 m past it at 15 m/s."""
    name = "fake"

    def __init__(self, site):
        self.to_image = GroundMapper(site.cameras["A"].homography).inverse
        self.t0 = None

    def detect_frames(self, frames, tiles):
        out = []
        for f in frames:
            boxes = []
            if f.camera == "A":
                self.t0 = f.t if self.t0 is None else self.t0
                y = START_Y - SPEED_MPS * (f.t - self.t0)
                box = box_for(self.to_image, 5.25, y, "car") if y > END_Y else None
                if box:
                    boxes.append(Detection(*box, 0.9, "car"))
            out.append(boxes)
        return out


def test_main_loop_records_a_car_and_never_writes_an_image(tmp_path, monkeypatch):
    if datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%H:%M") >= "23:57":
        pytest.skip("the all-day test window ends at 23:59")
    data = json.loads((SAMPLE_DAY / "site.json").read_text())
    data["time_zone"] = "America/Los_Angeles"
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    data["schedule"]["windows"] = [{"days": days, "start": "00:00", "end": "23:59"}]
    site = validate_site(data)
    (tmp_path / "site.json").write_text(json.dumps(data))
    powered_off = []
    monkeypatch.setattr(service_module, "CameraCapture", FakeCapture)
    monkeypatch.setattr(service_module, "make_detector", lambda kind, s, *a: OneCarDetector(s))
    monkeypatch.setattr(service_module.nightmode, "list_controls", lambda device: "")
    monkeypatch.setattr(service_module.schedule, "power_off", lambda: powered_off.append(True))
    monkeypatch.setattr(service_module.signal, "signal", lambda *a: None)   # not the main thread
    monkeypatch.setattr(service_module, "clock_problem", lambda now, folder: None)

    out = tmp_path / "data"
    service = service_module.Service(site, tmp_path / "site.json", out, "fake")
    runner = threading.Thread(target=service.run)
    runner.start()
    deadline = time.time() + 20
    picture = None
    try:
        while time.time() < deadline:
            service.watching["A"] = time.monotonic()      # someone has the live view open
            picture = picture or WebState(service).picture("A")
            rows = [r for d in out.glob("*/") for _h, p in readers.hour_files(d, "vehicles")
                    for r in readers.read_rows(p)]
            if rows:
                break
            time.sleep(0.2)
    finally:
        service.stop_event.set()
        runner.join(timeout=10)
    assert not runner.is_alive() and not powered_off
    assert picture and picture.startswith(b"\xff\xd8")   # the live view made a JPEG, in memory only

    day = next(out.glob("*/"))
    problems = {name: found for name, found in schema.validate_day_folder(day).items() if found}
    assert problems == {}
    rows = [r for _h, p in readers.hour_files(day, "vehicles") for r in readers.read_rows(p)]
    assert len(rows) == 1
    car = rows[0]
    assert car["camera"] == "A" and car["approach"] == "N" and car["lane"] == "2" and car["type"] == "car"
    assert car["t_crossed"] and car["wait_s"] == "0.0" and car["stops"] == "0"
    log = (day / "events.log").read_text()
    for text in ("startup", "recording started", "recording stopped", "shutdown"):
        assert text in log
    assert list(readers.hour_files(day, "system"))   # the first minute's system row

    for path in tmp_path.rglob("*"):   # rule 1: no picture or video anywhere
        if path.is_file():
            assert path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mkv", ".npy"}
            head = path.read_bytes()[:16]
            assert not any(head.startswith(magic) for magic in IMAGE_MAGIC), path
