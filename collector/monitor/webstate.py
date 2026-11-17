"""What the web UI can see and do, on top of the running Service."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from common.timeutil import format_epoch
from monitor import config, draw
from monitor.ground import fit_homography
from monitor.pipeline import CameraPipeline

MAX_MEAN_ERROR_M = 0.5
DAY_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def clock_source(user_agent: str) -> str:
    return "phone" if re.search(r"Mobile|Android|iPhone|iPad", user_agent or "") else "laptop"


def set_system_time(epoch_s: float, run=subprocess.run) -> list[str]:
    """Set the Pi's clock (and the RTC, if hwclock is installed). Returns problems, if any."""
    problems = []
    result = run(["sudo", "-n", "date", "-s", f"@{epoch_s:.3f}"], capture_output=True, text=True)
    if result.returncode != 0:
        problems.append(f"date failed: {result.stderr.strip()}")
    if shutil.which("hwclock") and Path("/dev/rtc0").exists():
        result = run(["sudo", "-n", "hwclock", "-w"], capture_output=True, text=True)
        if result.returncode != 0:
            problems.append(f"hwclock failed: {result.stderr.strip()}")
    return problems


class WebState:
    def __init__(self, service) -> None:
        self.s = service

    # ---- basics ---------------------------------------------------------
    @property
    def password(self) -> str | None:
        return self.s.site.web_password

    @property
    def data_dir(self) -> Path:
        return self.s.data_dir

    def cameras(self) -> list[str]:
        return list(self.s.site.cameras)

    def touch(self) -> None:
        self.s.last_activity = time.time()

    def stay_awake(self) -> None:
        self.s.stay_awake_until = time.time() + config.STAY_AWAKE_S
        self.s.log("stay awake for 1 hour (web UI)")

    # ---- status ----------------------------------------------------------
    def status(self) -> dict:
        s, now = self.s, time.time()
        upcoming = s.schedule.next_start(now)
        volts, amps = s.battery
        system = s.last_system
        return {
            "recording": s.recording,
            "next_window": format_epoch(upcoming, s.zone)[:16].replace("T", " ") if upcoming else "none",
            "battery_v": volts, "power_w": None if volts is None or amps is None else round(volts * amps, 1),
            "cpu_temp_c": system.get("cpu_temp_c", ""), "hailo_temp_c": system.get("hailo_temp_c", ""),
            "fps": {name: system.get(f"fps_{name}", "") for name in self.cameras()},
            "vehicles_15min": s.recorder.count_recent(now),
            "disk_free_gb": system.get("disk_free_gb", ""),
            "night_mode": s.night, "fps_target": s.fps,
            "stay_awake_until": format_epoch(s.stay_awake_until, s.zone)[11:19] if s.stay_awake_until > now else "",
            "pi_time_ms": now * 1000,
        }

    # ---- pictures (RAM only) ----------------------------------------------
    def picture(self, camera: str, overlays: bool = True) -> bytes | None:
        """The newest frame as JPEG bytes, made in memory and never saved."""
        self.s.watching[camera] = time.monotonic()
        item = self.s.latest.get(camera)
        if item is None:
            return None
        image, boxes = item
        cam = self.s.site.cameras[camera]
        return draw.jpeg(draw.overlay(image, cam, boxes) if overlays else image, 75 if overlays else 85)

    # ---- calibration ---------------------------------------------------------
    def calibrate(self, camera: str, payload: dict) -> dict:
        points = payload.get("points", [])
        try:
            homography, errors = fit_homography(points)
        except ValueError as error:
            return {"ok": False, "message": str(error)}
        mean = sum(errors) / len(errors)
        result = {"ok": True, "errors_m": [round(e, 2) for e in errors], "mean_error_m": round(mean, 2),
                  "saved": False, "message": ""}
        if len(points) < 6:
            result["message"] = "Only 4-5 points: the fit is exact, so the error says little. Use 6 or more."
        if not payload.get("save"):
            return result
        if mean >= MAX_MEAN_ERROR_M:
            result["message"] = f"Not saved: mean error {mean:.2f} m (must be below {MAX_MEAN_ERROR_M} m)."
            return result
        data = config.site_dict(self.s.site)
        cam = data["cameras"][camera]
        cam["homography"], cam["calib_points"] = homography, points
        for key in ("lane_x_m", "view_length_m", "tiles"):
            if key in payload and payload[key] not in (None, ""):
                cam[key] = payload[key]
        try:
            site = config.save_site(self.s.site_path, data)
        except config.SiteError as error:
            result["message"] = f"Not saved: {error}"
            return result
        self.apply_site(site, camera)
        result["saved"], result["message"] = True, "Saved to site.json."
        return result

    def apply_site(self, site, camera: str) -> None:
        s = self.s
        with s.lock:
            s.site = s.recorder.site = site
            now = time.time()
            if (s.data_dir / format_epoch(now, s.zone)[:10]).exists():
                s.writer.site_changed(now)
            if camera in s.pipelines:
                mono = time.monotonic()
                s.pipelines[camera].close(mono)   # tracks measured with the old calibration end here
                s.pipelines[camera] = CameraPipeline(camera, site, s.recorder, s.fps, mono)
        s.log(f"settings changed: camera {camera} calibration saved")

    # ---- clock ---------------------------------------------------------------
    def clock_sync(self, phone_ms: float, user_agent: str, run=subprocess.run) -> dict:
        s = self.s
        before = time.time()
        difference = phone_ms / 1000 - before
        if abs(difference) <= 0.5:
            return {"changed": False, "difference_s": round(difference, 2)}
        problems = set_system_time(phone_ms / 1000 + (time.time() - before), run)
        after = time.time()
        source = clock_source(user_agent)
        s.writer.clock_sync_row(after, [format_epoch(before, s.zone), format_epoch(after, s.zone),
                                        f"{after - before:.1f}", source])
        change = s.clock.resync()
        s.log(f"clock synced from {source}: offset {after - before:+.1f} s; monotonic offset re-measured "
              f"({change:+.1f} s)" + (f"; problems: {'; '.join(problems)}" if problems else ""))
        return {"changed": True, "difference_s": round(difference, 2), "problems": problems}

    # ---- downloads -----------------------------------------------------------
    def days(self) -> list[dict]:
        if not self.data_dir.exists():
            return []
        out = []
        for folder in sorted((p for p in self.data_dir.iterdir() if p.is_dir() and DAY_NAME.match(p.name)),
                             reverse=True):
            size = sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
            marker = folder / ".downloaded"
            out.append({"name": folder.name, "size_mb": round(size / 1e6, 1),
                        "downloaded": json.loads(marker.read_text())["time"] if marker.exists() else ""})
        return out

    def day_folder(self, name: str) -> Path | None:
        folder = self.data_dir / name
        return folder if DAY_NAME.match(name) and folder.is_dir() else None

    def mark_downloaded(self, name: str) -> None:
        folder = self.day_folder(name)
        if folder is not None:
            (folder / ".downloaded").write_text(json.dumps({"time": format_epoch(time.time(), self.s.zone)}))
