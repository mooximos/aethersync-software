"""Helpers for the tests: boxes a detector would see, made from sample_data tracks."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import readers
from common.timeutil import parse_epoch
from monitor.ground import GroundMapper

WIDTH, HEIGHT = 1920, 1080
SIZE_M = {"car": (1.8, 4.5, 1.5), "motorcycle": (0.8, 2.2, 1.4), "truck": (2.2, 8.0, 3.0), "bus": (2.6, 12.0, 3.2)}


def _project(to_image: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    u, v, w = to_image @ np.array([x, y, 1.0])
    return None if w <= 0 else (u / w, v / w)


def box_for(to_image: np.ndarray, x: float, y: float, vtype: str) -> tuple[float, float, float, float] | None:
    """Image box of a vehicle whose front-center is at ground (x, y). For a camera
    looking at oncoming traffic the box's bottom edge is the vehicle's front."""
    width, length, height = SIZE_M[vtype]
    front = _project(to_image, x, y)
    if front is None or not (0 <= front[0] <= WIDTH and 0 <= front[1] <= HEIGHT):
        return None
    corners = [_project(to_image, x + dx, y + dy) for dx in (-width / 2, width / 2) for dy in (0.0, length)]
    if any(c is None for c in corners):
        return None
    rear = _project(to_image, x, y + length)
    side = _project(to_image, x + 1.0, y + length)
    pixels_per_m = math.hypot(side[0] - rear[0], side[1] - rear[1])
    top = min(c[1] for c in corners) - height * pixels_per_m
    left, right = min(c[0] for c in corners), max(c[0] for c in corners)
    if right - left < 2:
        return None
    return max(left, 0.0), max(top, 0.0), min(right, float(WIDTH)), front[1]


def fake_boxes(day_dir: Path, hour: int, start_s: float, end_s: float, base_t: float, fps: float,
               rng: np.random.Generator, noise_px: float = 0.5, miss: float = 0.03) -> tuple[list[dict], float]:
    """Fake detector rows for [start_s, end_s) epoch seconds at `fps`, all 4 cameras.
    Returns (rows, clock offset) where frame time t = epoch - offset."""
    site = readers.read_site_dict(day_dir)
    vehicles = {r["vehicle_id"]: r for r in readers.read_rows(day_dir / f"vehicles_{hour:02d}.csv")}
    tracks = defaultdict(list)
    for row in readers.iter_rows(day_dir / f"tracks_{hour:02d}.csv.gz"):
        tracks[row["vehicle_id"]].append((parse_epoch(row["time"]), float(row["x_m"]), float(row["y_m"])))
    to_image = {name: GroundMapper(cam["homography"]).inverse for name, cam in site["cameras"].items()}
    offset = start_s - base_t
    step = 1.0 / fps
    frame_times = np.round(np.arange(start_s, end_s, step), 3)
    rows = []
    for vehicle_id, samples in tracks.items():
        vehicle = vehicles.get(vehicle_id)
        if vehicle is None:
            continue
        camera = vehicle["camera"]
        times = np.array([s[0] for s in samples])
        xs, ys = np.array([s[1] for s in samples]), np.array([s[2] for s in samples])
        inside = frame_times[(frame_times >= times[0]) & (frame_times <= times[-1])]
        for t in inside:
            if rng.random() < miss:
                continue
            box = box_for(to_image[camera], float(np.interp(t, times, xs)), float(np.interp(t, times, ys)),
                          vehicle["type"])
            if box is None:
                continue
            jitter = rng.normal(0, noise_px, 4)
            rows.append({"t": round(t - offset, 3), "camera": camera,
                         "x0": box[0] + jitter[0], "y0": box[1] + jitter[1], "x1": box[2] + jitter[2],
                         "y1": box[3] + jitter[3], "score": float(rng.uniform(0.6, 0.95)), "cls": vehicle["type"]})
    return rows, offset
