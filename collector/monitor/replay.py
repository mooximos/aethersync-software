"""Run the whole pipeline on recorded frames instead of live cameras.

Used by `python -m monitor --replay video.mp4 ...` (a phone video, before the Pi
is mounted) and by the end-to-end test (fake detector). Writes real data files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from monitor import config
from monitor.clock import Clock
from monitor.detector import Detector, Frame
from monitor.pipeline import CameraPipeline
from monitor.recorder import Recorder
from monitor.writers import DataWriter


def run_offline(site, frames: Iterable[Frame], detector: Detector, out_dir: Path,
                clock_offset: float, label: str = "replay") -> dict[str, int]:
    """frames must come in time order (t = seconds). Wall time = t + clock_offset.
    Returns the number of frames processed per camera."""
    now = {"t": 0.0}
    clock = Clock(mono_fn=lambda: now["t"], offset=clock_offset)
    site_json = config.site_dict(site)
    writer = DataWriter(Path(out_dir), site.zone(), lambda: site_json)
    recorder = Recorder(site, writer, clock)
    tiles = {name: [list(t) for t in cam.tiles] for name, cam in site.cameras.items()}
    pipelines: dict[str, CameraPipeline] = {}
    counts: dict[str, int] = {}
    last_tick = None
    for frame in frames:
        now["t"] = frame.t
        if not pipelines:
            writer.log_event(clock.wall(frame.t), f"startup: {label} with detector {detector.name}")
        pipe = pipelines.get(frame.camera)
        if pipe is None:
            pipe = CameraPipeline(frame.camera, site, recorder, site.detection.frames_per_second, frame.t)
            pipelines[frame.camera] = pipe
        pipe.process(frame.t, detector.detect_frames([frame], tiles)[0])
        counts[frame.camera] = counts.get(frame.camera, 0) + 1
        if last_tick is None or frame.t - last_tick >= 1.0:
            last_tick = frame.t
            for other in pipelines.values():
                other.finish_lost(frame.t)
            writer.tick(clock.wall(frame.t))
    for pipe in pipelines.values():
        pipe.close(now["t"])
    if pipelines:
        writer.log_event(clock.wall(now["t"]), f"shutdown: end of {label}, files closed")
    writer.close()
    return counts
