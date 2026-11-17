"""Writers: hour rotation, day folder, fsync, gzip, and surviving a kill mid-write."""
import gzip
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import readers, schema
from common.timeutil import format_epoch, local_epoch
from helpers import REPO, ZONE, sample_site
from monitor import config
from monitor.writers import DataWriter, repair_csv

SITE = config.site_dict(sample_site())


def system_row(epoch):
    return [format_epoch(epoch, ZONE), "13.20", "0.85", "11.2", "55.0", "50.0", "10.0", "10.0", "9.9", "10.0", 0, "20.00"]


def track_row(epoch, vid="A-0001"):
    return [format_epoch(epoch, ZONE), vid, "1.00", "50.00", "13.4"]


def test_day_folder_site_copy_and_hour_rotation(tmp_path):
    writer = DataWriter(tmp_path, ZONE, lambda: SITE, grace_s=30.0)
    start = local_epoch("2026-05-12", 7, 59, ZONE)
    writer.write("system", start, system_row(start))
    writer.write("tracks", start + 50, track_row(start + 50))
    day = tmp_path / "2026-05-12"
    assert (day / "site.json").exists() and (day / "system_07.csv").exists()
    writer.write("tracks", start + 70, track_row(start + 70))             # 08:00:10 -> hour 08 file
    writer.tick(start + 80)                                                # 08:00:20: within the grace time
    assert (day / "tracks_07.csv").exists()
    writer.write("tracks", start + 59.9, track_row(start + 59.9))          # a late row still goes in hour 07
    writer.tick(start + 95)                                                # 08:00:35: hour 07 closes
    for name in list(writer._gzipping.values()):
        name.join()
    assert (day / "tracks_07.csv.gz").exists() and not (day / "tracks_07.csv").exists()
    assert (day / "system_07.csv").exists()                                # only tracks are gzipped
    writer.close()
    assert (day / "tracks_08.csv.gz").exists()
    assert {name: p for name, p in schema.validate_day_folder(day).items() if p} == {}
    assert len(readers.read_rows(day / "tracks_07.csv.gz")) == 2


def test_rows_are_flushed_at_once_and_fsynced_every_10_s(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    writer = DataWriter(tmp_path, ZONE, lambda: SITE, fsync_every_s=10.0)
    epoch = local_epoch("2026-05-12", 7, 0, ZONE)
    writer.day_dir(epoch)                      # the day folder and its site.json copy are fsynced themselves
    calls.clear()
    writer._last_fsync = time.monotonic()
    writer.write("system", epoch, system_row(epoch))
    assert calls == []                                                     # not yet: under 10 s
    assert (tmp_path / "2026-05-12" / "system_07.csv").read_text().count("\n") == 2   # but already flushed
    writer._last_fsync -= 11
    writer.write("system", epoch + 60, system_row(epoch + 60))
    assert len(calls) == 1
    writer.close()


def test_settings_change_copy(tmp_path):
    writer = DataWriter(tmp_path, ZONE, lambda: SITE)
    path = writer.site_changed(local_epoch("2026-05-12", 7, 12, ZONE, second=34))
    assert path.name == "site_071234.json" and (path.parent / "site.json").exists()


def test_cut_off_last_line_is_repaired(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"a,b\n1,2\n3,")
    assert repair_csv(path) and path.read_bytes() == b"a,b\n1,2\n"
    assert not repair_csv(path)


WRITER_SCRIPT = """
import sys, time
sys.path[:0] = [{repo!r}, {repo!r} + "/collector"]
from common.timeutil import format_epoch, get_zone, local_epoch
from monitor.writers import DataWriter
zone = get_zone()
w = DataWriter({out!r}, zone, lambda: {{"site": "test"}})
t = local_epoch("2026-05-12", 7, 0, zone)
i = 0
while True:
    i += 1
    w.write("tracks", t + i * 0.001, [format_epoch(t + i * 0.001, zone), "A-%04d" % (i % 9000 + 1), "1.00", "%.2f" % (i % 100), "5.0"])
    if i == 2000:
        print("ready", flush=True)
"""


def test_killed_mid_write_leaves_valid_csv_after_restart(tmp_path):
    out = tmp_path / "data"
    proc = subprocess.Popen([sys.executable, "-c", WRITER_SCRIPT.format(repo=str(REPO), out=str(out))],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    time.sleep(0.2)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    path = out / "2026-05-12" / "tracks_07.csv"
    with open(path, "ab") as handle:
        handle.write(b"2026-05-12T07:00:09.9-07:00,A-00")    # what a power cut can leave behind
    writer = DataWriter(out, ZONE, lambda: SITE)
    assert any("repaired" in note for note in writer.recover(local_epoch("2026-05-12", 7, 30, ZONE)))
    epoch = local_epoch("2026-05-12", 7, 30, ZONE)
    writer.write("tracks", epoch, track_row(epoch, "A-9999"))
    writer.close()
    packed = Path(str(path) + ".gz")
    assert packed.exists() and not path.exists()
    assert schema.validate_file(packed) == []
    rows = readers.read_rows(packed)
    assert len(rows) > 2000 and rows[-1]["vehicle_id"] == "A-9999"


def test_restart_in_the_same_hour_appends_to_the_gzipped_file(tmp_path):
    epoch = local_epoch("2026-05-12", 7, 10, ZONE)
    for i in range(2):
        writer = DataWriter(tmp_path, ZONE, lambda: SITE)
        writer.write("tracks", epoch + i, track_row(epoch + i, f"A-000{i + 1}"))
        writer.close()
    packed = tmp_path / "2026-05-12" / "tracks_07.csv.gz"
    with gzip.open(packed, "rt") as handle:
        lines = handle.read().splitlines()
    assert lines[0] == ",".join(schema.TRACK_COLUMNS) and len(lines) == 3
