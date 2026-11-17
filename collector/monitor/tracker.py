"""ByteTrack (supervision package) - one tracker per camera."""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
import supervision as sv

from monitor.config import LOST_TRACK_S
from monitor.detector import Detection

CLASS_IDS = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}
CONFIRM_HITS = 3


@dataclass
class TrackedBox:
    track_id: int
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    cls: str


class Tracker:
    def __init__(self, frame_rate: float) -> None:
        self.frame_rate = frame_rate
        # supervision keeps a lost track for int(frame_rate / 30 * lost_track_buffer)
        # frames, so lost_track_buffer = 30 * LOST_TRACK_S gives LOST_TRACK_S seconds.
        with warnings.catch_warnings():   # supervision 0.30 warns that ByteTrack moves to another package
            warnings.simplefilter("ignore", FutureWarning)
            self.bytetrack = sv.ByteTrack(track_activation_threshold=0.35,
                                          lost_track_buffer=round(30 * LOST_TRACK_S),
                                          minimum_matching_threshold=0.8, frame_rate=frame_rate,
                                          minimum_consecutive_frames=CONFIRM_HITS)

    def set_rate(self, frame_rate: float) -> None:
        self.frame_rate = frame_rate
        self.bytetrack.max_time_lost = int(frame_rate * LOST_TRACK_S)

    def update(self, detections: list[Detection]) -> list[TrackedBox]:
        if detections:
            boxes = sv.Detections(
                xyxy=np.array([[d.x0, d.y0, d.x1, d.y1] for d in detections], dtype=np.float32),
                confidence=np.array([d.score for d in detections], dtype=np.float32),
                class_id=np.array([CLASS_IDS[d.cls] for d in detections], dtype=int))
        else:
            boxes = sv.Detections.empty()
        tracked = self.bytetrack.update_with_detections(boxes)
        return [TrackedBox(int(tid), *(float(v) for v in box), float(score), CLASS_NAMES[int(cid)])
                for box, score, cid, tid in zip(tracked.xyxy, tracked.confidence,
                                                tracked.class_id, tracked.tracker_id)]
