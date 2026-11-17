"""Lanes and the virtual stop-line detectors (event codes 82 detector_on / 81 detector_off)."""
from __future__ import annotations


def lane_of(x_m: float, lane_x_m: dict[int, tuple[float, float]]) -> int | None:
    for lane, (x0, x1) in lane_x_m.items():
        if x0 <= x_m < x1:
            return lane
    return None


class VirtualDetector:
    """One lane's detector: 0 <= y <= stop_line_zone_m across the lane.

    ON when a vehicle enters the empty zone. OFF when the zone becomes empty.
    A vehicle the camera misses for less than hold_s still counts as inside."""

    def __init__(self, lane: int, zone_m: float, hold_s: float) -> None:
        self.lane, self.zone_m, self.hold_s = lane, zone_m, hold_s
        self.inside: dict[str, float] = {}   # vehicle_id -> last time seen inside
        self.last_out = ""

    @property
    def on(self) -> bool:
        return bool(self.inside)

    def update(self, t: float, positions: dict[str, tuple[float, float, int | None]]) -> list[tuple[float, int, str]]:
        """positions: vehicle_id -> (x_m, y_m, lane) for vehicles seen in this frame.
        Returns events as (time, code, vehicle_id). The zone is judged frame by frame,
        so one car leaving while another arrives in the same frame keeps it ON."""
        was_on = bool(self.inside)
        entered, left = [], []
        for vehicle_id, (_x, y, lane) in positions.items():
            if lane == self.lane and 0.0 <= y <= self.zone_m:
                if vehicle_id not in self.inside:
                    entered.append(vehicle_id)
                self.inside[vehicle_id] = t
            elif vehicle_id in self.inside:
                del self.inside[vehicle_id]
                left.append((t, vehicle_id))
        for vehicle_id, seen in list(self.inside.items()):
            if vehicle_id not in positions and t - seen > self.hold_s:
                del self.inside[vehicle_id]
                left.append((seen, vehicle_id))   # vanished: it left when last seen
        if left:
            self.last_out = max(left)[1]
        if not was_on and self.inside:
            return [(t, 82, entered[0])]
        if was_on and not self.inside:
            return [(max(left)[0], 81, self.last_out)]
        return []

    def close(self, t: float) -> list[tuple[float, int, str]]:
        """At shutdown an occupied detector turns off, so every ON has an OFF."""
        if not self.inside:
            return []
        self.last_out = max(self.inside, key=self.inside.get)
        self.inside.clear()
        return [(t, 81, self.last_out)]
