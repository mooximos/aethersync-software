"""Drawing lanes, stop line, zones and tracks on a live-view frame (in RAM only)."""
from __future__ import annotations

import cv2
import numpy as np

from monitor.ground import GroundMapper

LIVE_WIDTH = 960


def _line(image, mapper: GroundMapper, a: tuple, b: tuple, color, width: int = 2) -> None:
    pa, pb = mapper.to_image(*a), mapper.to_image(*b)
    if pa is not None and pb is not None:
        cv2.line(image, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, width, cv2.LINE_AA)


def overlay(image: np.ndarray, camera, boxes: list[tuple]) -> np.ndarray:
    """A copy of the frame with the calibration and current tracks drawn on it,
    shrunk to LIVE_WIDTH pixels wide."""
    out = image.copy()
    mapper = GroundMapper(camera.homography)
    if mapper.calibrated:
        x_min = min(x0 for x0, _ in camera.lane_x_m.values())
        x_max = max(x1 for _, x1 in camera.lane_x_m.values())
        for x0, x1 in camera.lane_x_m.values():
            for x in (x0, x1):
                _line(out, mapper, (x, 0.0), (x, camera.view_length_m), (255, 255, 0), 1)
        _line(out, mapper, (x_min, 0.0), (x_max, 0.0), (0, 0, 255), 3)                      # stop line
        _line(out, mapper, (x_min, camera.stop_line_zone_m), (x_max, camera.stop_line_zone_m), (0, 165, 255))
        _line(out, mapper, (x_min, camera.view_length_m), (x_max, camera.view_length_m), (200, 200, 200))
    for x, y, w, h in camera.tiles:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 1)
    for label, (x0, y0, x1, y1), ground in boxes:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
        text = label + ("" if ground is None else f" {ground[2] * 2.237:.0f} mph")
        cv2.putText(out, text, (int(x0), max(int(y0) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    scale = LIVE_WIDTH / out.shape[1]
    return cv2.resize(out, (LIVE_WIDTH, round(out.shape[0] * scale))) if scale < 1 else out


def jpeg(image: np.ndarray, quality: int = 75) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else b""
