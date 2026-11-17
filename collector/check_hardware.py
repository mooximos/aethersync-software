#!/usr/bin/env python3
"""Checks the real hardware. Prints PASS or FAIL for each item, and the fix for each FAIL.

    .venv/bin/python collector/check_hardware.py --all
    .venv/bin/python collector/check_hardware.py --cameras --hailo --ina226 --rtc --site
    .venv/bin/python collector/check_hardware.py --jitter     (while cars wait at a red light)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

from monitor import config, nightmode  # noqa: E402
from monitor.camera import mjpeg_rates  # noqa: E402

results: list[bool] = []


def report(ok: bool, item: str, detail: str = "", fix: str = "") -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {item}" + (f": {detail}" if detail else ""))
    if not ok and fix:
        for line in fix.splitlines():
            print(f"      fix: {line}")


def run(command: list[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return done.returncode, (done.stdout + done.stderr).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return 1, str(error)


def check_site(site_path: Path):
    try:
        site = config.load_site(site_path)
    except config.SiteError as error:
        report(False, "site.json", str(error), f"edit {site_path}")
        return None
    left = [line.strip() for line in site_path.read_text().splitlines() if "REPLACE" in line]
    report(not left, "site.json filled in", f"{len(left)} REPLACE placeholders left" if left else "",
           "replace every REPLACE value (camera usb_path names are printed by --cameras)")
    report(site.latitude != 0 or site.longitude != 0, "site.json latitude/longitude",
           f"{site.latitude}, {site.longitude}", "set the intersection's latitude and longitude (for sunrise/sunset)")
    return site


def check_cameras(site) -> None:
    found = sorted(glob.glob("/dev/v4l/by-path/*video-index0"))
    print("      cameras plugged in now:" + "".join(f"\n        {p}" for p in found) if found else
          "      no cameras found in /dev/v4l/by-path/")
    if site is None:
        return
    import cv2
    for name, cam in site.cameras.items():
        if not Path(cam.usb_path).exists():
            report(False, f"camera {name} ({cam.approach})", f"{cam.usb_path} does not exist",
                   "plug it in (A = top USB 3, B = bottom USB 3, C = top USB 2, D = bottom USB 2),\n"
                   "then copy its name from the list above into site.json cameras." + name + ".usb_path")
            continue
        rates = mjpeg_rates(cam.usb_path)
        report(bool(rates), f"camera {name} MJPEG 1920x1080", f"frame rates {rates}",
               "sudo apt install v4l-utils; check the camera supports MJPEG 1920x1080")
        controls = nightmode.ExposureControls.find(nightmode.parse_controls(nightmode.list_controls(cam.usb_path)))
        report(controls.auto is not None and controls.exposure is not None, f"camera {name} exposure controls",
               f"auto={controls.auto}, exposure={controls.exposure}, gain={controls.gain}",
               "night mode cannot set the exposure on this camera; check v4l2-ctl --list-ctrls")
        capture = cv2.VideoCapture(cam.usb_path, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ok, image = capture.read()   # kept in RAM only, then dropped
        kernel = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        capture.release()
        method = "V4L2 buffer timestamp" if kernel > 0 and abs(kernel - time.monotonic()) < 1.0 else "time.monotonic()"
        report(ok and image is not None and image.shape[1] == 1920, f"camera {name} grabs a frame",
               f"{image.shape[1]}x{image.shape[0]}, frame times from {method}" if ok else "no frame",
               "another program may be using the camera (stop the service: sudo systemctl stop intersection-monitor)")


def describe(value, indent: str = "      ") -> str:
    """A short picture of the raw model output's structure."""
    import numpy as np
    if isinstance(value, dict):
        return "\n".join(f"{indent}{key!r}:\n{describe(item, indent + '  ')}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        sizes = [np.asarray(v).shape if not isinstance(v, (list, tuple, dict)) else f"list[{len(v)}]" for v in value[:8]]
        return f"{indent}{type(value).__name__} of {len(value)}; first items: {sizes}"
    array = np.asarray(value)
    return f"{indent}array shape {array.shape} dtype {array.dtype}"


def to_json(value):
    import numpy as np
    if isinstance(value, dict):
        return {key: to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    return np.asarray(value, dtype=float).tolist()


def check_hailo(site) -> None:
    code, text = run(["hailortcli", "fw-control", "identify"])
    report(code == 0, "Hailo chip answers", text.splitlines()[-1] if text else "",
           "sudo apt install hailo-all, then reboot; check the AI HAT+ ribbon cable")
    if site is None:
        return
    hef = Path(site.detection.hef_path)
    report(hef.exists(), "YOLOv8s HEF file", str(hef),
           "ls /usr/share/hailo-models/  and put the yolov8s_h8l.hef path in site.json detection.hef_path")
    if code != 0 or not hef.exists():
        return
    import numpy as np
    from monitor.detector import HailoDetector, parse_hailo_output
    detector = HailoDetector(str(hef), site.detection.min_confidence, site.detection.input_px)
    image = grab_one(site) if site else None
    image = image if image is not None else np.full((1080, 1920, 3), 114, np.uint8)
    raws, boxes = detector.raw_outputs([image])
    print(f"      API used: {detector.api}\n      raw output structure:\n{describe(raws[0], '        ')}")
    try:
        found = parse_hailo_output(raws[0], boxes[0], site.detection.min_confidence)
        report(True, "Hailo output parsed", f"{len(found)} vehicles found in the test picture")
    except Exception as error:
        report(False, "Hailo output parsed", str(error),
               "the output layout differs: send the printed structure and the saved sample to whoever maintains detector.py")
    sample = HERE / "tests" / "fixtures" / "hailo_output_recorded.json"
    lb = boxes[0]
    sample.write_text(json.dumps({"api": detector.api, "min_conf": site.detection.min_confidence,
                                  "letterbox": {"scale": lb.scale, "pad_x": lb.pad_x, "pad_y": lb.pad_y, "size": lb.size},
                                  "raw": to_json(raws[0])}))
    print(f"      saved the raw numbers (no picture) for the unit test: {sample}")
    print(f"      chip temperature: {detector.temperature()}")
    detector.close()


def grab_one(site):
    import cv2
    for cam in site.cameras.values():
        if Path(cam.usb_path).exists():
            capture = cv2.VideoCapture(cam.usb_path, cv2.CAP_V4L2)
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            ok, image = capture.read()
            capture.release()
            if ok:
                return image
    return None


def check_jitter(site) -> None:
    """How much do the boxes of parked or queued cars jitter? Speed (and so wait time)
    is only reliable where this is small: on sample data, 0.75 px gave 97% of waits
    within 1 s and 1.0 px gave 94%."""
    if site is None:
        return
    import numpy as np
    from monitor.detector import make_detector
    from monitor.tiling import iou
    detector = make_detector("hailo", site)
    for name, cam in site.cameras.items():
        if not Path(cam.usb_path).exists():
            continue
        import cv2
        capture = cv2.VideoCapture(cam.usb_path, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        frames = []
        for _ in range(40):            # about 4 s; frames stay in RAM
            ok, image = capture.read()
            if ok:
                frames.append(detector.detect(image))
            time.sleep(0.07)
        capture.release()
        chains = [[d] for d in frames[0]] if frames else []
        for found in frames[1:]:
            for chain in chains:
                best = max(found, key=lambda d: iou(d, chain[-1]), default=None)
                if best is not None and iou(best, chain[-1]) > 0.8:
                    chain.append(best)
        still = [np.std([d.y1 for d in c]) for c in chains
                 if len(c) >= 30 and max(d.y1 for d in c) - min(d.y1 for d in c) < 6]
        if not still:
            report(False, f"camera {name} box jitter", "no parked or queued vehicle seen for 4 s",
                   "run this again while cars wait at a red light")
            continue
        jitter = float(np.median(still))
        report(jitter <= 0.75, f"camera {name} box jitter", f"{jitter:.2f} px (from {len(still)} still vehicles)",
               "speeds far from the camera will be noisy: shorten view_length_m in the Calibrate page,\n"
               "or add a tile over the far part of the queue; then retest with a phone video (--replay)")
    detector.close()


def check_ina226() -> None:
    fix = ("sudo raspi-config nonint do_i2c 0 and reboot; then  i2cdetect -y 1  should show 40;\n"
           "check the SDA/SCL/3.3V/GND wires and that A0/A1 are tied to GND (address 0x40)")
    try:
        from monitor.power import INA226
        sensor = INA226()
        manufacturer, die = sensor.ids()
    except Exception as error:
        report(False, "INA226 on I2C bus 1 address 0x40", str(error), fix)
        return
    report((manufacturer, die) == (0x5449, 0x2260), "INA226 identity",
           f"manufacturer 0x{manufacturer:04X}, die 0x{die:04X}", "this is not an INA226 at 0x40")
    volts, amps = sensor.read()
    report(volts is not None and 10.0 <= volts <= 15.0, "battery voltage", f"{volts} V, {amps} A",
           "should be 12-14.6 V for the LiFePO4 battery: check the VBUS wire to the battery + side")
    report(amps is not None and 0.05 <= amps <= 3.0, "current", f"{amps} A",
           "should be about 0.3-1.5 A: check the shunt is in the + wire, and its direction (negative = reversed)")


def check_rtc() -> None:
    alarm = Path("/sys/class/rtc/rtc0/wakealarm")
    report(alarm.exists(), "RTC wake alarm file", str(alarm), "this Pi has no RTC at rtc0 (a Pi 5 should)")
    if alarm.exists():
        report(os.access(alarm, os.W_OK), "collector may set the wake alarm", "",
               "run collector/install.sh again (it adds a udev rule), then reboot")
    code, text = run(["rpi-eeprom-config"])
    has_setting = any(line.strip() == "POWER_OFF_ON_HALT=1" for line in text.splitlines())
    report(code == 0 and has_setting, "EEPROM POWER_OFF_ON_HALT=1", "" if has_setting else "not set",
           "sudo -E rpi-eeprom-config --edit   add the line POWER_OFF_ON_HALT=1, save, then reboot\n"
           "(without it the Pi stays half on after shutdown and the wake alarm may not start it)")
    since_epoch = Path("/sys/class/rtc/rtc0/since_epoch")
    if since_epoch.exists():
        drift = int(since_epoch.read_text()) - time.time()
        report(abs(drift) < 5, "RTC time matches the system time", f"{drift:+.0f} s",
               "do a clock sync from the web UI (it also writes the RTC)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("site", "cameras", "hailo", "jitter", "ina226", "rtc", "all"):
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--site-file", type=Path, default=config.DEFAULT_SITE_PATH)
    args = parser.parse_args()
    if not any([args.site, args.cameras, args.hailo, args.jitter, args.ina226, args.rtc, args.all]):
        args.all = True
    site = check_site(args.site_file) if (args.all or args.site or args.cameras or args.hailo or args.jitter) else None
    if args.all or args.cameras:
        check_cameras(site)
    if args.all or args.hailo:
        check_hailo(site)
    if args.jitter:   # not part of --all: it needs cars waiting in view
        check_jitter(site)
    if args.all or args.ina226:
        check_ina226()
    if args.all or args.rtc:
        check_rtc()
    failed = results.count(False)
    print(f"\n{len(results) - failed} PASS, {failed} FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
