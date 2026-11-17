"""Letterboxing, the Hailo output adapter, tiling and the fake detector."""
import json
from pathlib import Path

import numpy as np
import pytest

from monitor import tiling
from monitor.detector import Detection, Detector, FakeDetector, Frame, Letterbox, letterbox, parse_hailo_output

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("hailo_*.json"))


def test_letterbox_keeps_the_shape_and_maps_boxes_back():
    image = np.zeros((1080, 1920, 3), np.uint8)
    canvas, lb = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3) and lb.scale == pytest.approx(1 / 3) and (lb.pad_x, lb.pad_y) == (0, 140)
    assert canvas[0, 0, 0] == 114 and canvas[320, 320, 0] == 0
    from monitor.detector import from_letterbox
    assert from_letterbox(160, 256, 240, 320, lb) == pytest.approx((480, 348, 720, 540))


@pytest.mark.parametrize("path", FIXTURES, ids=[p.name for p in FIXTURES])
def test_hailo_output_samples_parse(path):
    sample = json.loads(path.read_text())
    lb = Letterbox(**sample["letterbox"])
    found = parse_hailo_output(sample["raw"], lb, sample["min_conf"])
    assert all(d.cls in ("car", "motorcycle", "bus", "truck") and d.score >= sample["min_conf"] for d in found)
    if "expected" in sample:
        got = [[round(d.x0, 1), round(d.y0, 1), round(d.x1, 1), round(d.y1, 1), round(d.score, 2), d.cls] for d in found]
        assert got == sample["expected"]


def test_hailo_adapter_unwraps_a_dict_and_a_batch_of_one():
    sample = json.loads(FIXTURES[0].read_text())
    lb = Letterbox(**sample["letterbox"])
    plain = parse_hailo_output(sample["raw"], lb, 0.35)
    assert parse_hailo_output({"yolov8s/nms": [sample["raw"]]}, lb, 0.35) == plain
    with pytest.raises(ValueError):
        parse_hailo_output([[0.1, 0.2]] * 3, lb, 0.35)


def d(x0, y0, x1, y1, score=0.9):
    return Detection(x0, y0, x1, y1, score, "car")


def test_nms_is_class_agnostic():
    boxes = [d(0, 0, 10, 10, 0.9), Detection(1, 1, 10, 10, 0.8, "truck"), d(50, 50, 60, 60, 0.7)]
    assert tiling.nms(boxes) == [boxes[0], boxes[2]]


def test_merge_shifts_tile_boxes_and_drops_ones_cut_by_a_tile_edge():
    tile = [640, 290, 640, 360]
    full = [d(600, 400, 700, 460)]                      # a car across the tile's left edge
    in_tile = [d(0, 110, 60, 170, 0.8),                 # the same car, cut by the tile edge
               d(300, 50, 320, 62, 0.6)]                 # a small far car only the tile sees
    merged = tiling.merge(full, [in_tile], [tile], 1920, 1080)
    assert full[0] in merged and len(merged) == 2
    assert d(940, 340, 960, 352, 0.6) in merged


def test_crops_are_clipped_to_the_frame():
    image = np.zeros((1080, 1920, 3), np.uint8)
    parts = tiling.crops(image, [[1800, 1000, 640, 360]])
    assert parts[1].shape == (80, 120, 3)


def test_detect_frames_batches_all_images_and_merges_per_frame():
    class Stub(Detector):
        def __init__(self):
            self.calls = []

        def detect_many(self, images):
            self.calls.append([im.shape for im in images])
            return [[d(10, 10, 20, 20)] if im.shape[1] == 1920 else [] for im in images]
    stub = Stub()
    frames = [Frame("A", 0.0, np.zeros((1080, 1920, 3), np.uint8)), Frame("B", 0.0, np.zeros((1080, 1920, 3), np.uint8))]
    out = stub.detect_frames(frames, {"A": [[0, 0, 640, 360]], "B": []})
    assert stub.calls == [[(1080, 1920, 3), (360, 640, 3), (1080, 1920, 3)]]   # one batch
    assert out == [[d(10, 10, 20, 20)], [d(10, 10, 20, 20)]]


def test_fake_detector_reads_boxes_by_camera_and_time(tmp_path):
    path = tmp_path / "boxes.csv"
    path.write_text("t,camera,x0,y0,x1,y1,score,cls\n1.0,A,1,2,3,4,0.9,car\n1.1,B,5,6,7,8,0.5,bus\n")
    fake = FakeDetector(path)
    out = fake.detect_frames([Frame("A", 1.0, None), Frame("B", 1.0, None), Frame("B", 1.1, None)], {})
    assert out == [[Detection(1, 2, 3, 4, 0.9, "car")], [], [Detection(5, 6, 7, 8, 0.5, "bus")]]
