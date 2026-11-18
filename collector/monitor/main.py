"""Command line.

On the Pi (normally started by systemd, see monitor.service):
    python -m monitor --site collector/site.json --data data

On a laptop, test the whole pipeline with a phone video (writes real data files):
    python -m monitor --replay video.mp4 --camera A --detector ultralytics --site site.json --out data_test/
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from monitor import config
from monitor.camera import video_frames
from monitor.detector import Frame, make_detector
from monitor.replay import run_offline


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m monitor", description="Intersection Monitor collector")
    parser.add_argument("--site", type=Path, default=config.DEFAULT_SITE_PATH, help="site.json to use")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATA_DIR, help="folder for the day folders")
    parser.add_argument("--detector", choices=["hailo", "ultralytics", "fake"], default=None,
                        help="hailo on the Pi (default), ultralytics for --replay on a laptop")
    parser.add_argument("--fake-boxes", type=Path, help="boxes file for --detector fake")
    parser.add_argument("--port", type=int, default=config.WEB_PORT, help="web UI port")
    parser.add_argument("--replay", type=Path, help="run on this video file instead of the cameras")
    parser.add_argument("--camera", default="A", help="with --replay: which camera's settings the video uses")
    parser.add_argument("--out", type=Path, help="with --replay: output folder (default data_test)")
    parser.add_argument("--start", help="with --replay: local start time of the video, written "
                                        "YYYY-MM-DDTHH:MM:SS (default: now)")
    return parser.parse_args(argv)


def load_or_exit(path: Path):
    try:
        return config.load_site(path)
    except config.SiteError as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        raise SystemExit(2)


def replay(args: argparse.Namespace, site) -> None:
    if args.camera not in site.cameras:
        raise SystemExit(f"camera {args.camera} is not in {args.site}")
    detector = make_detector(args.detector or "ultralytics", site, args.fake_boxes)
    if args.start:
        try:
            start = datetime.fromisoformat(args.start).replace(tzinfo=site.zone()).timestamp()
        except ValueError:
            raise SystemExit(f"--start {args.start!r}: write it as YYYY-MM-DDTHH:MM:SS, local time")
    else:
        start = time.time()
    only_one = site.model_copy(update={"cameras": {args.camera: site.cameras[args.camera]}})
    frames = (Frame(args.camera, t, image) for t, image in video_frames(args.replay, site.detection.frames_per_second))
    out = args.out or Path("data_test")
    counts = run_offline(only_one, frames, detector, out, clock_offset=start, label=f"replay of {args.replay.name}")
    print(f"processed {counts.get(args.camera, 0)} frames; data files are in {out}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    site = load_or_exit(args.site)
    if args.replay:
        replay(args, site)
        return
    from monitor.service import Service
    from monitor.webstate import WebState
    from monitor.webui import serve
    service = Service(site, args.site, args.data, args.detector or "hailo")
    serve(WebState(service), args.port)
    service.run()
