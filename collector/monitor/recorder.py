"""Turns pipeline results into formatted rows (times, IDs, light) for the writers."""
from __future__ import annotations

from collections import deque

from common import readers
from common.schema import EVENT_CODES, VEHICLE_ID_PATTERN
from common.timeutil import clock_text, format_epoch, local_day_and_hour
from monitor.clock import Clock
from monitor.events import Point, Summary
from monitor.sun import light_label, sun_elevation
from monitor.writers import DataWriter


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


class Recorder:
    def __init__(self, site, writer: DataWriter, clock: Clock) -> None:
        self.site, self.writer, self.clock = site, writer, clock
        self.zone = site.zone()
        self.night = False
        self.counters: dict[str, dict[str, int]] = {}
        self.recent: deque[tuple[float, str]] = deque(maxlen=20000)   # (epoch, approach), for the status page

    def epoch(self, mono: float) -> float:
        return self.clock.wall(mono)

    def text(self, mono: float | None) -> str:
        return "" if mono is None else format_epoch(self.clock.wall(mono), self.zone)

    def new_vehicle_id(self, camera: str, mono: float) -> str:
        """<camera>-<counter>; the counter restarts each day and continues after a restart."""
        day, _hour = local_day_and_hour(self.epoch(mono), self.zone)
        if day not in self.counters:
            self.counters[day] = self._highest_ids(day)
        self.counters[day][camera] = self.counters[day].get(camera, 0) + 1
        return f"{camera}-{self.counters[day][camera]:04d}"

    def _highest_ids(self, day: str) -> dict[str, int]:
        folder = self.writer.data_dir / day
        highest: dict[str, int] = {}
        if not folder.exists():
            return highest
        self.writer.fsync_all()
        for kind, column in (("vehicles", "vehicle_id"), ("tracks", "vehicle_id"), ("signal_events", "detail")):
            for _hour, path in readers.hour_files(folder, kind):
                try:
                    for row in readers.iter_rows(path):
                        match = VEHICLE_ID_PATTERN.match(row.get(column) or "")
                        if match:
                            number = int(row[column].split("-")[1])
                            highest[match.group(1)] = max(highest.get(match.group(1), 0), number)
                except (OSError, EOFError):
                    continue
        return highest

    def light(self, mono: float) -> str:
        return light_label(sun_elevation(self.site.latitude, self.site.longitude, self.epoch(mono)))

    def vehicle_row(self, camera: str, approach: str, vehicle_id: str, s: Summary, queue_pos: int | None) -> None:
        row = [vehicle_id, camera, approach, "" if s.lane is None else s.lane, s.turn, s.vtype, f"{s.conf:.2f}",
               self.text(s.t_entered), self.text(s.t_stopped), self.text(s.t_started), self.text(s.t_crossed),
               self.text(s.t_left), _fmt(s.wait_s, 1), "" if s.stops is None else s.stops,
               "" if queue_pos is None else queue_pos,
               self.light(s.t_entered if s.t_entered is not None else s.t_first), ";".join(s.flags)]
        self.writer.write("vehicles", self.epoch(s.t_left), row)
        self.recent.append((self.epoch(s.t_left), approach))

    def track_row(self, tick: float, vehicle_id: str, point: Point | None) -> None:
        values = ["", "", ""] if point is None else [f"{point.x:.2f}", f"{point.y:.2f}", f"{point.speed:.1f}"]
        self.writer.write("tracks", self.epoch(tick), [self.text(tick), vehicle_id, *values])

    def signal_row(self, mono: float, code: int, approach: str, lane: int | None, detail: str) -> None:
        self.writer.write("signal_events", self.epoch(mono),
                          [self.text(mono), code, EVENT_CODES[code], approach, "" if lane is None else lane, detail])

    def green_detail(self, moved_mono: float) -> str:
        return (f"first queued car moved at {clock_text(self.epoch(moved_mono), self.zone)}; "
                f"minus {self.site.reaction_time_s:g} s reaction")

    def count_recent(self, now_epoch: float, minutes: float = 15) -> dict[str, int]:
        counts = {a: 0 for a in "NSEW"}
        for epoch, approach in self.recent:
            if now_epoch - epoch <= minutes * 60:
                counts[approach] += 1
        return counts
