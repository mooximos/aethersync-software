"""The web UI routes, with a stand-in for the running collector."""
import base64
import io
import zipfile

import pytest

from monitor.webstate import clock_source, set_system_time
from monitor.webui import create_app


class FakeState:
    def __init__(self, tmp_path, password=None):
        self.password, self.data_dir, self.touched, self.awake = password, tmp_path, 0, False
        day = tmp_path / "2026-05-12"
        day.mkdir()
        (day / "vehicles_07.csv").write_text("vehicle_id\nA-0001\n")
        (day / "tracks_07.csv.gz").write_bytes(b"\x1f\x8b fake")
        (day / ".downloaded").write_text('{"time": "x"}')
        self.marked = []

    def cameras(self):
        return ["A", "B"]

    def touch(self):
        self.touched += 1

    def status(self):
        return {"recording": True, "fps": {"A": "10.0"}}

    def stay_awake(self):
        self.awake = True

    def picture(self, camera, overlays=True):
        return b"\xff\xd8jpeg" if camera == "A" else None

    def calibrate(self, camera, payload):
        return {"ok": True, "camera": camera, "n": len(payload["points"])}

    def days(self):
        return [{"name": "2026-05-12", "size_mb": 0.1, "downloaded": ""}]

    def day_folder(self, name):
        folder = self.data_dir / name
        return folder if folder.is_dir() else None

    def mark_downloaded(self, name):
        self.marked.append(name)

    def clock_sync(self, phone_ms, user_agent):
        return {"changed": True, "phone_ms": phone_ms, "source": clock_source(user_agent)}


@pytest.fixture
def client(tmp_path):
    state = FakeState(tmp_path)
    app = create_app(state)
    return app.test_client(), state


def test_pages_load_without_internet(client):
    c, state = client
    for path in ("/", "/live", "/calibrate", "/download", "/clock"):
        page = c.get(path)
        assert page.status_code == 200 and b"http" not in page.data.replace(b"http-equiv", b"")
    assert state.touched == 5


def test_status_and_stay_awake(client):
    c, state = client
    assert c.get("/api/status").get_json()["recording"] is True
    c.post("/api/stay_awake")
    assert state.awake


def test_snapshot_from_ram_or_503(client):
    c, _ = client
    assert c.get("/snapshot/A.jpg").data == b"\xff\xd8jpeg"
    assert c.get("/snapshot/B.jpg").status_code == 503
    assert c.get("/snapshot/Z.jpg").status_code == 404


def test_calibrate_and_clock_api(client):
    c, _ = client
    assert c.post("/api/calibrate/A", json={"points": [1, 2, 3, 4]}).get_json() == {"ok": True, "camera": "A", "n": 4}
    answer = c.post("/api/clock", json={"phone_ms": 123}, headers={"User-Agent": "Mozilla (iPhone)"}).get_json()
    assert answer["source"] == "phone" and answer["phone_ms"] == 123


def test_download_is_a_zip_built_on_the_fly(client):
    c, state = client
    response = c.get("/download/2026-05-12.zip")
    archive = zipfile.ZipFile(io.BytesIO(response.data))
    assert sorted(archive.namelist()) == ["2026-05-12/tracks_07.csv.gz", "2026-05-12/vehicles_07.csv"]
    assert archive.read("2026-05-12/vehicles_07.csv") == b"vehicle_id\nA-0001\n"
    assert state.marked == ["2026-05-12"]
    assert c.get("/download/../etc.zip").status_code == 404


def test_password(tmp_path):
    c = create_app(FakeState(tmp_path, password="tomato")).test_client()
    assert c.get("/").status_code == 401
    good = base64.b64encode(b"any:tomato").decode()
    assert c.get("/", headers={"Authorization": f"Basic {good}"}).status_code == 200


def test_set_system_time_uses_sudo_date():
    calls = []

    class Done:
        returncode, stderr = 0, ""
    problems = set_system_time(1_800_000_000.25, run=lambda cmd, **k: (calls.append(cmd), Done())[1])
    assert calls[0] == ["sudo", "-n", "date", "-s", "@1800000000.250"] and problems == []
