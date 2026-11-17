"""Phone-friendly web UI on port 8080 (Flask, no internet or CDN needed).

Pictures are made in RAM from the newest frame and streamed only while a page is
open; nothing is ever written to disk."""
from __future__ import annotations

import hmac
import io
import threading
import time
import zipfile
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template_string, request
from werkzeug.serving import make_server

from monitor import webui_pages as pages

LIVE_FPS = 5


class _Chunks(io.RawIOBase):
    """A write-only stream for zipfile that hands out what was written so far."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        self.parts.append(bytes(data))
        return len(data)

    def take(self) -> bytes:
        data, self.parts = b"".join(self.parts), []
        return data


def zip_stream(folder: Path):
    """Zip a day folder on the fly, chunk by chunk (no temporary file)."""
    out = _Chunks()
    with zipfile.ZipFile(out, "w") as archive:
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            info = zipfile.ZipInfo(f"{folder.name}/{path.name}", time.localtime(path.stat().st_mtime)[:6])
            info.compress_type = zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED
            with open(path, "rb") as source, archive.open(info, "w", force_zip64=True) as target:
                while chunk := source.read(1 << 16):
                    target.write(chunk)
                    yield out.take()
    yield out.take()


def create_app(state) -> Flask:
    app = Flask(__name__)

    @app.before_request
    def check_password():
        password = state.password
        if password:
            auth = request.authorization
            if auth is None or not hmac.compare_digest(auth.password or "", password):
                return Response("Password needed", 401, {"WWW-Authenticate": 'Basic realm="Intersection Monitor"'})
        state.touch()

    def page(name: str, body: str, **values):
        return render_template_string(pages.BASE, title=name, body=render_template_string(body, **values))

    @app.get("/")
    def status_page():
        return page("Status", pages.STATUS, cameras=state.cameras())

    @app.get("/api/status")
    def status_api():
        return jsonify(state.status())

    @app.post("/api/stay_awake")
    def stay_awake():
        state.stay_awake()
        return jsonify(state.status())

    @app.get("/live")
    def live_page():
        return page("Live view", pages.LIVE, cameras=state.cameras())

    @app.get("/stream/<camera>")
    def stream(camera: str):
        if camera not in state.cameras():
            abort(404)

        def frames():
            while True:
                started = time.monotonic()
                data = state.picture(camera, overlays=True)
                if data:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                time.sleep(max(1.0 / LIVE_FPS - (time.monotonic() - started), 0.05))
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/snapshot/<camera>.jpg")
    def snapshot(camera: str):
        if camera not in state.cameras():
            abort(404)
        data = state.picture(camera, overlays=request.args.get("overlay") == "1")
        if not data:
            return Response("No picture yet: recording must be on (or wait a second and reload).", 503)
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/calibrate")
    def calibrate_page():
        return page("Calibrate", pages.CALIBRATE, cameras=state.cameras())

    @app.post("/api/calibrate/<camera>")
    def calibrate(camera: str):
        if camera not in state.cameras():
            abort(404)
        return jsonify(state.calibrate(camera, request.get_json(force=True)))

    @app.get("/download")
    def download_page():
        return page("Download", pages.DOWNLOAD, days=state.days())

    @app.get("/download/<name>.zip")
    def download(name: str):
        folder = state.day_folder(name)
        if folder is None:
            abort(404)
        state.mark_downloaded(name)
        return Response(zip_stream(folder), mimetype="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})

    @app.get("/clock")
    def clock_page():
        return page("Clock sync", pages.CLOCK)

    @app.post("/api/clock")
    def clock_sync():
        phone_ms = float(request.get_json(force=True)["phone_ms"])
        return jsonify(state.clock_sync(phone_ms, request.headers.get("User-Agent", "")))

    return app


def serve(state, port: int) -> threading.Thread:
    server = make_server("0.0.0.0", port, create_app(state), threaded=True)
    thread = threading.Thread(target=server.serve_forever, name="webui", daemon=True)
    thread.start()
    return thread
