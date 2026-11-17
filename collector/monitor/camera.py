"""USB cameras (one capture thread each) and video files for --replay.

Frames stay in RAM: each camera keeps only its newest frame in a 1-slot queue.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

RETRY_S = 5.0
WIDTH, HEIGHT = 1920, 1080


def mjpeg_rates(device: str) -> list[float]:
    """Frame rates the camera offers for MJPEG 1920x1080 (from v4l2-ctl)."""
    try:
        text = subprocess.run(["v4l2-ctl", "-d", device, "--list-formats-ext"], capture_output=True,
                              text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_mjpeg_rates(text)


def parse_mjpeg_rates(text: str) -> list[float]:
    rates, in_mjpeg, in_size = [], False, False
    for line in text.splitlines():
        if "'" in line and "]" in line:   # a format line like  [0]: 'MJPG' (Motion-JPEG, compressed)
            in_mjpeg, in_size = "'MJPG'" in line, False
        elif "Size:" in line:
            in_size = in_mjpeg and f"{WIDTH}x{HEIGHT}" in line
        elif in_size and "fps" in line:
            match = re.search(r"\(([\d.]+) fps\)", line)
            if match:
                rates.append(float(match.group(1)))
    return sorted(set(rates))


def choose_rate(offered: list[float], target: float) -> float:
    """The lowest offered rate that is at least the target (else the fastest one)."""
    good = [r for r in offered if r >= target]
    return min(good) if good else (max(offered) if offered else max(target, 15.0))


class CameraCapture(threading.Thread):
    """Grabs every frame from one camera (to keep the driver queue empty) but
    decodes only as many as needed for the processing rate."""

    def __init__(self, name: str, device: str, target_fps: float, log: Callable[[str], None],
                 mono: Callable[[], float] = time.monotonic) -> None:
        super().__init__(name=f"camera-{name}", daemon=True)
        self.camera, self.device, self.target_fps, self.log, self.mono = name, device, target_fps, log, mono
        self._slot: tuple[np.ndarray, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.connected = False
        self.dropped = 0          # decoded but replaced before the main loop took it
        self.started_at: float | None = None   # monotonic time of the first frame since (re)connecting
        self.timestamp_method = ""

    def set_rate(self, fps: float) -> None:
        self.target_fps = fps

    def take(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            item, self._slot = self._slot, None
        return item

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            capture = self._open()
            if capture is None:
                self._stop.wait(RETRY_S)
                continue
            try:
                self._read_loop(capture)
            finally:
                capture.release()
                if self.connected:
                    self.log(f"camera {self.camera} disconnected; retrying every {RETRY_S:g} s")
                self.connected = False
                self._stop.wait(0 if self._stop.is_set() else RETRY_S)

    def _open(self) -> cv2.VideoCapture | None:
        if not Path(self.device).exists():
            return None
        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        rate = choose_rate(mjpeg_rates(self.device), self.target_fps)
        capture.set(cv2.CAP_PROP_FPS, rate)
        if not capture.isOpened():
            capture.release()
            return None
        self.log(f"camera {self.camera} opened {self.device} at {rate:g} fps MJPEG {WIDTH}x{HEIGHT}")
        return capture

    def _read_loop(self, capture: cv2.VideoCapture) -> None:
        last_kernel, use_kernel, next_due, failures = 0.0, None, 0.0, 0
        while not self._stop.is_set():
            if not capture.grab():
                failures += 1
                if failures >= 20:
                    return
                time.sleep(0.05)
                continue
            failures = 0
            now = self.mono()
            kernel = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if use_kernel is None:   # decide once per connection which clock to trust
                use_kernel = kernel > 0 and abs(kernel - now) < 1.0
                self.timestamp_method = ("V4L2 buffer timestamp" if use_kernel
                                         else "time.monotonic() right after grab")
                self.log(f"camera {self.camera} frame times: {self.timestamp_method}")
                self.connected, self.started_at = True, now
            stamp = kernel if use_kernel and kernel > last_kernel else now
            last_kernel = max(last_kernel, kernel)
            if stamp < next_due:
                continue
            next_due = max(next_due + 1.0 / self.target_fps, stamp + 0.5 / self.target_fps)
            ok, image = capture.retrieve()
            if not ok:
                continue
            with self._lock:
                if self._slot is not None:
                    self.dropped += 1
                self._slot = (image, stamp)


def video_frames(path: Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Frames of a video file at its own timing, thinned to about fps. Yields (seconds, image)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    file_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    index, next_due = 0, 0.0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                return
            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0   # the file's own frame time
            t = position if index > 0 and position > 0 else index / file_fps
            index += 1
            if t + 1e-6 >= next_due:
                next_due += 1.0 / fps
                yield t, image
    finally:
        capture.release()
