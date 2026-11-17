"""Build the site.json that goes with the sample data (a made-up but valid site)."""
from __future__ import annotations

import math

import numpy as np

from common import SOFTWARE_VERSION
from common.schema import EVENT_CODES
from common.site import PRIVACY_TEXT
from sample_events import LANE_X_M, LOW_CONF_BELOW, MOVING_SPEED, REACTION_S, STOPPED_SPEED, ZONE_M
from sample_sim import LANE_MOVEMENTS, VIEW_LENGTH

CAMERA_APPROACH = {"A": "N", "B": "E", "C": "S", "D": "W"}
IMAGE_SIZE = (1920, 1080)
LENS_WIDTH_DEG = 96.0
# Pretend camera: on the far-side pole, 30 m past the stop line, 5.8 m up,
# above the middle lane, tilted 15 degrees down, looking at the oncoming cars.
CAMERA_X_M, CAMERA_Y_M, CAMERA_HEIGHT_M, CAMERA_TILT_DEG = 5.25, -30.0, 5.8, 15.0
CALIB_GROUND_POINTS = [(0.0, 0.0), (10.5, 0.0), (0.0, 20.0), (10.5, 20.0), (0.0, 60.0), (10.5, 60.0)]


def ground_to_image_matrix() -> np.ndarray:
    """3x3 matrix taking ground (x_m, y_m, 1) to image pixels, for the pretend camera."""
    width, height = IMAGE_SIZE
    focal = (width / 2) / math.tan(math.radians(LENS_WIDTH_DEG / 2))
    tilt = math.radians(CAMERA_TILT_DEG)
    lens = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    pose = np.array([
        [1, 0, -CAMERA_X_M],
        [0, -math.sin(tilt), math.sin(tilt) * CAMERA_Y_M + CAMERA_HEIGHT_M * math.cos(tilt)],
        [0, math.cos(tilt), -math.cos(tilt) * CAMERA_Y_M + CAMERA_HEIGHT_M * math.sin(tilt)],
    ])
    return lens @ pose


def sample_homography() -> list[list[float]]:
    """Image pixels -> ground meters (what site.json stores)."""
    matrix = np.linalg.inv(ground_to_image_matrix())
    matrix = matrix / matrix[2, 2]
    return [[float(f"{value:.9g}") for value in row] for row in matrix]


def _calib_points() -> list[dict]:
    to_image = ground_to_image_matrix()
    points = []
    for x_m, y_m in CALIB_GROUND_POINTS:
        u, v, w = to_image @ np.array([x_m, y_m, 1.0])
        points.append({"px": [round(u / w, 1), round(v / w, 1)], "m": [x_m, y_m]})
    return points


def build_site(clock_last_sync: str | None = None, clock_offset_s: float | None = None) -> dict:
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    cameras = {}
    for name, approach in CAMERA_APPROACH.items():
        cameras[name] = {
            "approach": approach,
            "usb_path": f"/dev/v4l/by-path/SAMPLE-DATA-camera-{name}-(no-real-camera)",
            "lanes": {str(n): {"movements": list(moves)} for n, moves in LANE_MOVEMENTS.items()},
            "lane_x_m": {str(n): list(x_range) for n, x_range in LANE_X_M.items()},
            "homography": sample_homography(),
            "calib_points": _calib_points(),
            "tiles": [[640, 290, 640, 360]],
            "stop_line_zone_m": ZONE_M,
            "view_length_m": VIEW_LENGTH,
        }
    return {
        "site_name": "Sample Intersection (synthetic data from tools/make_sample_data.py)",
        "latitude": 37.6688,
        "longitude": -121.8747,
        "time_zone": "America/Los_Angeles",
        "researchers": ["Sample Student One", "Sample Student Two"],
        "software_version": SOFTWARE_VERSION,
        "privacy": PRIVACY_TEXT,
        "device": {"computer": "Raspberry Pi 5 (2 GB)", "ai_chip": "Hailo-8L (AI HAT+ 13 TOPS)",
                   "mount_height_m": CAMERA_HEIGHT_M},
        "cameras": cameras,
        "coordinates": {
            "x_m": "meters from the curbside edge of the approach toward the center line",
            "y_m": "meters upstream of the stop line; 0 = stop line, negative = past it",
        },
        "detection": {"model": "yolov8s", "hef_path": "/usr/share/hailo-models/yolov8s_h8l.hef",
                      "input_px": 640, "frames_per_second": 10, "min_confidence": 0.35,
                      "low_conf_flag_below": LOW_CONF_BELOW,
                      "classes": ["car", "motorcycle", "bus", "truck"]},
        "stopped_speed_mps": STOPPED_SPEED,
        "moving_speed_mps": MOVING_SPEED,
        "reaction_time_s": REACTION_S,
        "track_sample_rate_hz": 2,
        "schedule": {"windows": [{"days": weekdays, "start": "06:30", "end": "09:30"},
                                 {"days": weekdays, "start": "15:30", "end": "18:30"}],
                     "boot_grace_min": 15, "wake_early_min": 10},
        "power": {"shutdown_below_v": 12.0, "shutdown_after_s": 60},
        "night": {"switch_sun_elevation_deg": -2, "night_exposure": 40, "night_gain": None},
        "event_codes": {str(code): name for code, name in EVENT_CODES.items()},
        "clock": {"last_sync": clock_last_sync, "offset_found_s": clock_offset_s},
    }
