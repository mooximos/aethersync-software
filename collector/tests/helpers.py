"""Shared test helpers: synthetic trajectories and a camera pipeline writing to a temp folder."""
from __future__ import annotations

from pathlib import Path

from common.site import load_site
from common.timeutil import get_zone, local_epoch
from monitor.clock import Clock
from monitor.detector import Detection
from monitor.ground import GroundMapper
from monitor.pipeline import CameraPipeline
from monitor.recorder import Recorder
from monitor.writers import DataWriter
from monitor import config
import synthetic

REPO = Path(__file__).resolve().parents[2]
SAMPLE_DAY = REPO / "sample_data" / "2026-05-12"
ZONE = get_zone()
DT = 0.1

def sample_site():
    return load_site(SAMPLE_DAY / "site.json")

def drive(y0: float, v0: float, plan: list[tuple[float, float]], x: float = 5.25,
          t0: float = 0.0, dt: float = DT) -> list[tuple[float, float, float, float]]:
    """Kinematics: start at y0 (m before the stop line) at v0 m/s; plan = [(seconds, accel m/s2)].
    Returns samples (t, x, y, speed) every dt."""
    out, t, y, v = [], t0, y0, v0
    out.append((round(t, 3), x, y, v))
    for duration, accel in plan:
        for _ in range(round(duration / dt)):
            v = min(max(v + accel * dt, 0.0), 13.4)
            y -= v * dt
            t += dt
            out.append((round(t, 3), x, y, v))
    return out

def stop_at(y0: float, v0: float, stop_y: float, wait_s: float, after_s: float = 8.0) -> list[tuple[float, ...]]:
    """Drive at v0, brake at a steady rate to stop at stop_y, wait, then drive off at 2 m/s2."""
    braking = v0 * v0 / (2 * (y0 - stop_y))   # constant deceleration that stops exactly at stop_y
    return drive(y0, v0, [(v0 / braking, -braking), (wait_s, 0.0), (after_s, 2.0)])

class PipelineRig:
    """One camera pipeline (sample camera A) writing real files into a temp folder.
    Time t (monotonic) = seconds after 07:00:00 local on 2026-05-12."""

    def __init__(self, out_dir: Path, start_hour: int = 7, start_minute: int = 0) -> None:
        self.site = sample_site()
        self.t_box = {"t": 0.0}
        self.offset = local_epoch("2026-05-12", start_hour, start_minute, ZONE)
        self.clock = Clock(mono_fn=lambda: self.t_box["t"], offset=self.offset)
        site_json = config.site_dict(self.site)
        self.writer = DataWriter(out_dir, ZONE, lambda: site_json)
        self.recorder = Recorder(self.site, self.writer, self.clock)
        self.pipe = CameraPipeline("A", self.site, self.recorder, 10.0, start_mono=-100.0)
        self.to_image = GroundMapper(self.site.cameras["A"].homography).inverse

    def run(self, vehicles: list[list[tuple]], until: float | None = None, vtype: str = "car") -> None:
        """vehicles: lists of (t, x, y, speed) samples on the 0.1 s grid."""
        by_time: dict[float, list[Detection]] = {}
        for samples in vehicles:
            for t, x, y, _speed in samples:
                box = synthetic.box_for(self.to_image, x, y, vtype)
                if box is not None:
                    by_time.setdefault(round(t, 1), []).append(Detection(*box, 0.9, vtype))
        last = until if until is not None else max(by_time) + 3.0
        for step in range(0, round(last / DT) + 1):
            t = round(step * DT, 1)
            self.t_box["t"] = t
            self.pipe.process(t, by_time.get(t, []))

    def close(self) -> None:
        self.pipe.close(self.t_box["t"])
        self.writer.close()
