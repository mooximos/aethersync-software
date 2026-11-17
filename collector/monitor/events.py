"""One vehicle's track -> the facts for its row in vehicles_HH.csv (data format section 2.2).

The track is fed one processed frame at a time. All times are monotonic seconds.
"""
from __future__ import annotations

import bisect
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from monitor.zones import lane_of

MIN_STOP_S = 1.0       # below stopped_speed_mps this long = a stop episode
MIN_MOVE_S = 0.5       # above moving_speed_mps this long = the stop is over
LANE_WINDOW_S = 3.0    # lane = median x over the last 3 s before the stop line
TURN_PAST_M = 8.0      # the turn rule needs the track to go this far past the line
TURN_SIDE_M = 3.0      # sideways movement after the line that makes it a turn
JUMP_M = 4.0           # a bigger jump suggests the tracker swapped two vehicles


@dataclass
class Rules:
    stopped_mps: float
    moving_mps: float
    view_length_m: float
    lanes: dict[int, tuple[str, ...]]
    lane_x_m: dict[int, tuple[float, float]]
    low_conf_below: float

    @classmethod
    def from_site(cls, site, camera: str) -> "Rules":
        cam = site.cameras[camera]
        return cls(site.stopped_speed_mps, site.moving_speed_mps, cam.view_length_m,
                   {int(k): tuple(v.movements) for k, v in cam.lanes.items()},
                   {int(k): tuple(v) for k, v in cam.lane_x_m.items()},
                   site.detection.low_conf_flag_below)


@dataclass
class Point:
    t: float
    x: float
    y: float
    speed: float


@dataclass
class Summary:
    """Everything for one vehicles row except the formatting (done by the recorder)."""
    lane: int | None
    turn: str
    vtype: str
    conf: float
    t_entered: float | None
    t_stopped: float | None
    t_started: float | None
    t_crossed: float | None
    t_left: float
    t_first: float
    wait_s: float | None
    stops: int | None
    flags: list[str] = field(default_factory=list)


class VehicleTrack:
    def __init__(self, track_id: int, rules: Rules, calibrated: bool, present_at_start: bool) -> None:
        self.track_id, self.rules, self.calibrated = track_id, rules, calibrated
        self.present_at_start = present_at_start
        self.vehicle_id: str | None = None
        self.t_first: float | None = None
        self.t_last = 0.0
        self.scores: list[float] = []
        self.class_weight: dict[str, float] = defaultdict(float)
        self.points: list[Point] = []      # smoothed, from entering the view zone on
        self.entered: Point | None = None
        self.crossed: tuple[float, float] | None = None   # (time, x)
        self.episodes: list[tuple[Point, Point]] = []    # finished stop episodes (start, end)
        self.episode_start: Point | None = None          # open stop episode
        self.slow_since: Point | None = None
        self.move_since: Point | None = None
        self.jumped = False
        self.night = False

    @property
    def in_episode(self) -> bool:
        return self.episode_start is not None

    def lane_now(self) -> int | None:
        return lane_of(self.points[-1].x, self.rules.lane_x_m) if self.points else None

    def add(self, t: float, score: float, cls: str, ground: tuple[float, float, float] | None,
            jumped: bool, night: bool) -> list[tuple]:
        """Feed one frame. ground = smoothed (x, y, speed) or None if uncalibrated.

        Returns notes for the camera pipeline: ("entered",), ("stop_start", Point),
        ("stop_end", Point), ("cross", t)."""
        if self.t_first is None:
            self.t_first = t
        self.t_last = t
        self.scores.append(score)
        self.class_weight[cls] += score
        self.night = self.night or night
        if ground is None:
            if self.vehicle_id is None:
                return [("entered",)]
            return []
        point = Point(t, *ground)
        notes: list[tuple] = []
        if self.entered is None:
            if not 0.0 <= point.y <= self.rules.view_length_m:
                return notes
            self.entered = point
            notes.append(("entered",))
        else:
            self.jumped = self.jumped or jumped
            previous = self.points[-1]
            if self.crossed is None and previous.y > 0.0 >= point.y:
                share = previous.y / (previous.y - point.y)
                t_cross = previous.t + share * (t - previous.t)
                self.crossed = (t_cross, previous.x + share * (point.x - previous.x))
                notes.append(("cross", t_cross))
                if self.episode_start is not None:   # still creeping when it crossed: cut at the line
                    end = Point(t_cross, self.crossed[1], 0.0, point.speed)
                    self.episodes.append((self.episode_start, end))
                    self.episode_start = None
                    notes.append(("stop_end", end))
        self.points.append(point)
        if self.crossed is None:
            notes += self._stop_rules(point)
        return notes

    def _stop_rules(self, p: Point) -> list[tuple]:
        """Stop episodes with hysteresis: start after 1.0 s below stopped_mps, end after
        0.5 s above moving_mps. The episode's times are where those runs began."""
        if self.episode_start is None:
            if p.speed >= self.rules.stopped_mps:
                self.slow_since = None
            elif self.slow_since is None:
                self.slow_since = p
            elif p.t - self.slow_since.t >= MIN_STOP_S - 1e-6:
                self.episode_start, self.slow_since, self.move_since = self.slow_since, None, None
                return [("stop_start", self.episode_start)]
            return []
        if p.speed <= self.rules.moving_mps:
            self.move_since = None
        elif self.move_since is None:
            self.move_since = p
        elif p.t - self.move_since.t >= MIN_MOVE_S - 1e-6:
            end = self.move_since
            self.episodes.append((self.episode_start, end))
            self.episode_start = self.move_since = None
            return [("stop_end", end)]
        return []

    def stopped_at(self, t: float) -> Point | None:
        """Where the vehicle was, if it was in a stop episode at time t."""
        spans = list(self.episodes)
        if self.episode_start is not None:
            spans.append((self.episode_start, None))
        for start, end in spans:
            if start.t <= t and (end is None or end.t > t):
                return self.position_at(t)
        return None

    def position_at(self, t: float) -> Point:
        index = bisect.bisect_right([p.t for p in self.points], t) - 1
        return self.points[max(index, 0)]

    def _lane(self) -> int | None:
        if not self.points:
            return None
        until = self.crossed[0] if self.crossed else self.points[-1].t
        recent = [p.x for p in self.points if until - LANE_WINDOW_S <= p.t <= until]
        return lane_of(statistics.median(recent or [self.points[-1].x]), self.rules.lane_x_m)

    def _turn(self, lane: int | None) -> str:
        moves = self.rules.lanes.get(lane, ())
        if len(moves) == 1:
            return moves[0]
        if self.crossed is None or min(p.y for p in self.points) > -TURN_PAST_M:
            return "U"
        sideways = self.points[-1].x - self.crossed[1]   # + = toward the center line
        return "R" if sideways < -TURN_SIDE_M else "L" if sideways > TURN_SIDE_M else "T"

    def finish(self, cut_at_shutdown: bool) -> Summary:
        if self.episode_start is not None:   # the track ended while stopped
            end = self.move_since or self.points[-1]
            self.episodes.append((self.episode_start, end))
            self.episode_start = None
        conf = sum(self.scores) / len(self.scores)
        vtype = max(self.class_weight, key=self.class_weight.get)
        flags = []
        if conf < self.rules.low_conf_below:
            flags.append("low_conf")
        if not self.calibrated:
            checks = [("started_in_view", self.present_at_start),
                      ("short_track", self.t_last - self.t_first < 1.0 or len(self.scores) < 5),
                      ("night_mode", self.night), ("uncalibrated", True), ("cut_at_shutdown", cut_at_shutdown)]
            flags += [name for name, on in checks if on]
            return Summary(None, "U", vtype, conf, None, None, None, None, self.t_last, self.t_first,
                           None, None, flags)
        if self.present_at_start or self.entered.y < self.rules.view_length_m - 15.0:
            flags.append("started_in_view")
        if any(start.y >= self.rules.view_length_m - 5.0 for start, _ in self.episodes):
            flags.append("queue_past_view")
        if self.crossed is None:
            flags.append("no_cross")
        if self.points[-1].t - self.entered.t < 1.0 or len(self.points) < 5:
            flags.append("short_track")
        if self.night:
            flags.append("night_mode")
        if cut_at_shutdown:
            flags.append("cut_at_shutdown")
        if self.jumped:
            flags.append("id_switch_suspect")
        lane = self._lane()
        wait = sum(end.t - start.t for start, end in self.episodes)
        return Summary(
            lane, self._turn(lane), vtype, conf, self.entered.t,
            self.episodes[0][0].t if self.episodes else None,
            self.episodes[-1][1].t if self.episodes else None,
            self.crossed[0] if self.crossed else None, self.t_last, self.t_first,
            wait, len(self.episodes), flags)


class TickSampler:
    """Picks one point per 0.5 s tick of the monotonic clock for the tracks file:
    the frame nearest the tick within 0.15 s. A tick with no such frame is skipped."""

    def __init__(self, every_s: float, window_s: float) -> None:
        self.every_s, self.window_s = every_s, window_s
        self.tick: float | None = None
        self.best: tuple[float, object] | None = None

    def offer(self, t: float, item: object) -> list[tuple[float, object]]:
        if self.tick is None:
            self.tick = -(-round((t - self.window_s) * 1000) // round(self.every_s * 1000)) * self.every_s
        done = []
        while t > self.tick + self.window_s + 1e-9:
            if self.best is not None:
                done.append((self.tick, self.best[1]))
            self.best = None
            self.tick += self.every_s
        distance = abs(t - self.tick)
        if distance <= self.window_s + 1e-9 and (self.best is None or distance < self.best[0]):
            self.best = (distance, item)
        return done

    def flush(self) -> list[tuple[float, object]]:
        done = [(self.tick, self.best[1])] if self.best is not None else []
        self.best = None
        return done
