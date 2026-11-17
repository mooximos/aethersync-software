"""Per camera: tracker -> ground position and speed -> vehicle events, virtual
detectors and inferred signal events -> rows (through the Recorder)."""
from __future__ import annotations

from monitor import config
from monitor.detector import Detection
from monitor.events import Point, Rules, TickSampler, VehicleTrack, JUMP_M
from monitor.ground import GroundMapper, Kalman
from monitor.recorder import Recorder
from monitor.signals import SignalInference, signal_lanes
from monitor.tracker import Tracker
from monitor.zones import VirtualDetector, lane_of


class CameraPipeline:
    def __init__(self, name: str, site, recorder: Recorder, frame_rate: float, start_mono: float) -> None:
        cam = site.cameras[name]
        self.name, self.approach, self.recorder = name, cam.approach, recorder
        self.rules = Rules.from_site(site, name)
        self.mapper = GroundMapper(cam.homography)
        self.tracker = Tracker(frame_rate)
        self.detectors = {lane: VirtualDetector(lane, cam.stop_line_zone_m, config.DETECTOR_HOLD_S)
                          for lane in self.rules.lanes}
        self.signals = SignalInference(signal_lanes(self.rules.lanes), cam.stop_line_zone_m,
                                       site.reaction_time_s, config.GREEN_MIN_GAP_S)
        self.start_mono = start_mono
        self.tracks: dict[int, VehicleTrack] = {}
        self.filters: dict[int, Kalman] = {}
        self.samplers: dict[int, TickSampler] = {}
        self.queue_pos: dict[int, int] = {}
        self.last_boxes: list[tuple] = []   # for the live view: (label, box, ground point)

    def restart(self, now: float) -> None:
        """The camera reconnected: finish the old tracks and start a fresh tracker."""
        self.close(now, cut=False)
        self.tracker = Tracker(self.tracker.frame_rate)
        self.start_mono = now

    def process(self, t: float, detections: list[Detection]) -> None:
        positions: dict[str, tuple[float, float, int | None]] = {}
        boxes = []
        for box in self.tracker.update(detections):
            track = self.tracks.get(box.track_id)
            if track is None:
                track = VehicleTrack(box.track_id, self.rules, self.mapper.calibrated,
                                     present_at_start=t - self.start_mono < config.PRESENT_AT_START_S)
                self.tracks[box.track_id] = track
            ground, jumped = None, False
            if self.mapper.calibrated:
                u, v = (box.x0 + box.x1) / 2, box.y1
                x, y = self.mapper.to_ground(u, v)
                noise = self.mapper.noise(u, v)
                kalman = self.filters.get(box.track_id)
                if kalman is None:
                    self.filters[box.track_id] = Kalman(t, x, y, noise)
                    ground = (x, y, 0.0)
                else:
                    ground = kalman.update(t, x, y, noise)
                    jumped = kalman.jump_m > JUMP_M
            for note in track.add(t, box.score, box.cls, ground, jumped, self.recorder.night):
                self._handle(track, note)
            if track.vehicle_id is not None:
                point = track.points[-1] if track.points and track.points[-1].t == t else None
                sampler = self.samplers.setdefault(box.track_id, TickSampler(config.TRACK_TICK_S,
                                                                             config.TRACK_TICK_WINDOW_S))
                for tick, item in sampler.offer(t, point):
                    self.recorder.track_row(tick, track.vehicle_id, item)
                if point is not None:
                    positions[track.vehicle_id] = (point.x, point.y, lane_of(point.x, self.rules.lane_x_m))
            boxes.append((track.vehicle_id or "", (box.x0, box.y0, box.x1, box.y1), ground))
        self.last_boxes = boxes
        if self.mapper.calibrated:
            for lane, detector in self.detectors.items():
                for when, code, vehicle_id in detector.update(t, positions):
                    self.recorder.signal_row(when, code, self.approach, lane, vehicle_id)
        self.finish_lost(t)

    def _handle(self, track: VehicleTrack, note: tuple) -> None:
        kind = note[0]
        if kind == "entered":
            track.vehicle_id = self.recorder.new_vehicle_id(self.name, track.t_last)
        elif kind == "cross":
            self.signals.on_cross(note[1])
        elif kind == "stop_start":
            start: Point = note[1]
            lane = lane_of(start.x, self.rules.lane_x_m)
            if track.track_id not in self.queue_pos:
                ahead = 0
                for other in self.tracks.values():
                    if other is track or other.vehicle_id is None:
                        continue
                    where = other.stopped_at(start.t)
                    if where is not None and where.y < start.y and lane_of(where.x, self.rules.lane_x_m) == lane:
                        ahead += 1
                self.queue_pos[track.track_id] = ahead + 1
            event = self.signals.on_stop_start(start.t, start.y, lane)
            if event:
                self.recorder.signal_row(event[0], 10, self.approach, None, "first car stopped at the line")
        elif kind == "stop_end":
            end: Point = note[1]
            event = self.signals.on_stop_end(end.t, end.y, lane_of(end.x, self.rules.lane_x_m))
            if event:
                self.recorder.signal_row(event[0], 1, self.approach, None, self.recorder.green_detail(event[2]))

    def finish_lost(self, now: float) -> None:
        for track_id in [i for i, tr in self.tracks.items() if now - tr.t_last > config.LOST_TRACK_S]:
            self._finish(track_id, cut=False)

    def _finish(self, track_id: int, cut: bool) -> None:
        track = self.tracks.pop(track_id)
        self.filters.pop(track_id, None)
        sampler = self.samplers.pop(track_id, None)
        queue_pos = self.queue_pos.pop(track_id, None)
        if track.vehicle_id is None:
            return   # never entered the view zone (e.g. cross traffic): not a vehicle of this approach
        if sampler is not None:
            for tick, item in sampler.flush():
                self.recorder.track_row(tick, track.vehicle_id, item)
        summary = track.finish(cut_at_shutdown=cut)
        self.recorder.vehicle_row(self.name, self.approach, track.vehicle_id, summary,
                                  queue_pos if summary.stops else None)

    def close(self, now: float, cut: bool = True) -> None:
        """Shutdown: write every open track (flag cut_at_shutdown) and switch detectors off."""
        for track_id in list(self.tracks):
            self._finish(track_id, cut=cut)
        for lane, detector in self.detectors.items():
            for when, code, vehicle_id in detector.close(now):
                self.recorder.signal_row(when, code, self.approach, lane, vehicle_id)
