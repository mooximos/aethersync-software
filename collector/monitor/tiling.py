"""Run the detector on the full frame plus extra full-resolution crops (tiles),
then merge everything back into full-frame boxes.

Far-away cars are tiny in a 1920x1080 frame shrunk to 640x640. A tile is cut
from the original frame, so those cars are 3x bigger when the tile is shrunk.
"""
from __future__ import annotations

import numpy as np

from monitor.detector import Detection

EDGE_PX = 3.0          # a box this close to a tile edge was probably cut off by it
COVERED_SHARE = 0.5    # ...and is dropped if a full-frame box covers this much of it
NMS_IOU = 0.5


def crops(image: np.ndarray, tiles: list) -> list[np.ndarray]:
    """[full frame, tile 1, tile 2, ...]. Tiles are [x, y, w, h] in pixels (clipped to the frame)."""
    height, width = image.shape[:2]
    out = [image]
    for x, y, w, h in tiles:
        x0, y0 = max(int(x), 0), max(int(y), 0)
        out.append(image[y0:min(int(y + h), height), x0:min(int(x + w), width)])
    return out


def _area(d: Detection) -> float:
    return max(d.x1 - d.x0, 0.0) * max(d.y1 - d.y0, 0.0)


def _intersection(a: Detection, b: Detection) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: Detection, b: Detection) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], threshold: float = NMS_IOU) -> list[Detection]:
    """Class-agnostic non-maximum suppression: keep the best box of each overlapping group."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(iou(det, other) <= threshold for other in kept):
            kept.append(det)
    return kept


def _cut_by_tile(d: Detection, tile: list, frame_w: int, frame_h: int) -> bool:
    x, y, w, h = tile
    edges = [(d.x0 - x, x > 0), (x + w - d.x1, x + w < frame_w),
             (d.y0 - y, y > 0), (y + h - d.y1, y + h < frame_h)]
    return any(gap <= EDGE_PX and inside_frame for gap, inside_frame in edges)


def merge(full: list[Detection], per_tile: list[list[Detection]], tiles: list,
          frame_w: int, frame_h: int) -> list[Detection]:
    """Tile boxes are in tile pixels; shift them to frame pixels, drop the ones a
    tile edge cut in half that the full frame already found, then NMS everything."""
    combined = list(full)
    for tile, found in zip(tiles, per_tile):
        x, y = max(int(tile[0]), 0), max(int(tile[1]), 0)
        for d in found:
            shifted = Detection(d.x0 + x, d.y0 + y, d.x1 + x, d.y1 + y, d.score, d.cls)
            if _cut_by_tile(shifted, tile, frame_w, frame_h) and any(
                    _intersection(shifted, f) >= COVERED_SHARE * _area(shifted) for f in full):
                continue
            combined.append(shifted)
    return nms(combined)
