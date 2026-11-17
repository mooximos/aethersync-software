"""Vehicle detection. Detector.detect(frame_bgr) -> list of Detection in full-frame pixels.

Backends:
- hailo:       the Raspberry Pi AI HAT+ (Hailo-8L) with the YOLOv8s HEF
- ultralytics: yolov8s.pt on the laptop CPU (for --replay only)
- fake:        boxes read from a file (tests)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

COCO_TYPES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass(frozen=True)
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    cls: str


@dataclass
class Frame:
    camera: str
    t: float                     # monotonic seconds
    image: np.ndarray | None     # BGR, in RAM only


@dataclass
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    size: int


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, Letterbox]:
    """Shrink to fit size x size keeping the shape, pad the rest with gray."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(image, (new_w, new_h),
                                                                interpolation=cv2.INTER_LINEAR)
    return canvas, Letterbox(scale, pad_x, pad_y, size)


def from_letterbox(x0: float, y0: float, x1: float, y1: float, lb: Letterbox) -> tuple[float, ...]:
    """Box in letterboxed pixels -> box in the original image's pixels."""
    return ((x0 - lb.pad_x) / lb.scale, (y0 - lb.pad_y) / lb.scale,
            (x1 - lb.pad_x) / lb.scale, (y1 - lb.pad_y) / lb.scale)


def parse_hailo_output(raw, lb: Letterbox, min_conf: float) -> list[Detection]:
    """The one place that knows the Hailo YOLOv8 NMS output layout.

    Layout (checked on the Pi with `check_hardware.py --hailo`): one list per COCO
    class (80), each an array of rows [y0, x0, y1, x1, score], coordinates 0-1 of
    the 640x640 model input. It may be wrapped in a dict {output name: ...} and/or
    a batch list of length 1."""
    if isinstance(raw, dict):
        if len(raw) != 1:
            raise ValueError(f"expected one model output, got {list(raw)}")
        raw = next(iter(raw.values()))
    while len(raw) == 1 and isinstance(raw[0], (list, tuple)):
        raw = raw[0]   # batch of one
    if len(raw) < max(COCO_TYPES) + 1:
        raise ValueError(f"expected one entry per COCO class (80), got {len(raw)}")
    found = []
    for class_id, vtype in COCO_TYPES.items():
        rows = np.asarray(raw[class_id], dtype=np.float64).reshape(-1, 5)
        for y0, x0, y1, x1, score in rows:
            if score >= min_conf:
                box = from_letterbox(x0 * lb.size, y0 * lb.size, x1 * lb.size, y1 * lb.size, lb)
                found.append(Detection(*box, float(score), vtype))
    return found


class Detector:
    name = "base"
    supports_tiles = True

    def detect(self, image: np.ndarray) -> list[Detection]:
        return self.detect_many([image])[0]

    def detect_many(self, images: list[np.ndarray]) -> list[list[Detection]]:
        raise NotImplementedError

    def detect_frames(self, frames: list[Frame], tiles: dict[str, list]) -> list[list[Detection]]:
        """All cameras' frames (and their tiles) in one batch."""
        from monitor import tiling
        images, spans = [], []
        for frame in frames:
            parts = tiling.crops(frame.image, tiles.get(frame.camera, []))
            spans.append((len(images), len(parts)))
            images += parts
        results = self.detect_many(images)
        merged = []
        for frame, (start, count) in zip(frames, spans):
            height, width = frame.image.shape[:2]
            merged.append(tiling.merge(results[start], results[start + 1:start + count],
                                       tiles.get(frame.camera, []), width, height))
        return merged

    def temperature(self) -> float | None:
        return None

    def close(self) -> None:
        pass


class FakeDetector(Detector):
    """Boxes from a CSV file with columns t, camera, x0, y0, x1, y1, score, cls
    (t = the frame's monotonic time). Used by the tests."""
    name = "fake"
    supports_tiles = False

    def __init__(self, path: Path | None = None, rows: list[dict] | None = None) -> None:
        self.boxes: dict[tuple[str, int], list[Detection]] = defaultdict(list)
        if path is not None:
            with open(path, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        for row in rows or []:
            key = (row["camera"], round(float(row["t"]) * 1000))
            self.boxes[key].append(Detection(float(row["x0"]), float(row["y0"]), float(row["x1"]),
                                             float(row["y1"]), float(row["score"]), row["cls"]))

    def detect_frames(self, frames: list[Frame], tiles: dict[str, list]) -> list[list[Detection]]:
        return [list(self.boxes.get((f.camera, round(f.t * 1000)), [])) for f in frames]


class UltralyticsDetector(Detector):
    name = "ultralytics"

    def __init__(self, min_conf: float, weights: str = "yolov8s.pt") -> None:
        from ultralytics import YOLO   # only needed for --replay on the laptop
        self.model, self.min_conf = YOLO(weights), min_conf

    def detect_many(self, images: list[np.ndarray]) -> list[list[Detection]]:
        results = self.model.predict(images, imgsz=640, conf=self.min_conf, classes=list(COCO_TYPES),
                                     device="cpu", verbose=False)
        out = []
        for result in results:
            boxes = result.boxes
            out.append([Detection(*map(float, xyxy), float(conf), COCO_TYPES[int(cls)])
                        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())])
        return out


class HailoDetector(Detector):
    """Tries picamera2.devices.Hailo first (simplest), else hailo_platform directly."""
    name = "hailo"

    def __init__(self, hef_path: str, min_conf: float, input_px: int = 640) -> None:
        if not Path(hef_path).exists():
            raise FileNotFoundError(f"HEF model not found: {hef_path} (see detection.hef_path in site.json)")
        self.min_conf, self.size = min_conf, input_px
        self._stack = ExitStack()
        try:
            from picamera2.devices import Hailo
            self.hailo = self._stack.enter_context(Hailo(hef_path))
            self.api, self.target = "picamera2.devices.Hailo", getattr(self.hailo, "target", None)
        except ImportError:
            self._open_hailo_platform(hef_path)

    def _open_hailo_platform(self, hef_path: str) -> None:
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams, VDevice)
        hef = HEF(hef_path)
        self.target = VDevice()
        group = self.target.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        self.input_name = hef.get_input_vstream_infos()[0].name
        self.pipe = self._stack.enter_context(InferVStreams(
            group, InputVStreamParams.make(group, format_type=FormatType.UINT8),
            OutputVStreamParams.make(group, format_type=FormatType.FLOAT32)))
        self._stack.enter_context(group.activate(group.create_params()))
        self.hailo, self.api = None, "hailo_platform (InferVStreams, batched)"

    def raw_outputs(self, images: list[np.ndarray]) -> tuple[list, list[Letterbox]]:
        """Run the model; returns the raw outputs (one per image) and the letterbox used."""
        inputs, boxes = [], []
        for image in images:
            canvas, lb = letterbox(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.size)
            inputs.append(canvas)
            boxes.append(lb)
        if self.hailo is not None:
            if hasattr(self.hailo, "run_async"):   # queue them all, so the chip stays busy
                futures = [self.hailo.run_async(canvas) for canvas in inputs]
                return [future.result() for future in futures], boxes
            return [self.hailo.run(canvas) for canvas in inputs], boxes
        result = self.pipe.infer({self.input_name: np.stack(inputs)})
        name = next(iter(result))
        return [{name: result[name][i]} for i in range(len(inputs))], boxes

    def detect_many(self, images: list[np.ndarray]) -> list[list[Detection]]:
        raws, boxes = self.raw_outputs(images)
        return [parse_hailo_output(raw, lb, self.min_conf) for raw, lb in zip(raws, boxes)]

    def temperature(self) -> float | None:
        try:
            return float(self.target.get_physical_devices()[0].control.get_chip_temperature().ts0_temperature)
        except Exception:   # not every HailoRT version exposes it; the cell stays empty
            return None

    def close(self) -> None:
        self._stack.close()


def make_detector(kind: str, site, fake_boxes: Path | None = None) -> Detector:
    if kind == "hailo":
        return HailoDetector(site.detection.hef_path, site.detection.min_confidence, site.detection.input_px)
    if kind == "ultralytics":
        return UltralyticsDetector(site.detection.min_confidence)
    if kind == "fake":
        return FakeDetector(fake_boxes)
    raise ValueError(f"unknown detector {kind!r} (hailo, ultralytics or fake)")
