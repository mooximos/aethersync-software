"""The collector as it runs on the Pi: schedule, cameras, detector, data files,
power, night mode, sleep, and the state the web UI shows."""
from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

from common.timeutil import format_epoch
from monitor import config, nightmode, schedule, sysinfo
from monitor.camera import CameraCapture
from monitor.clock import Clock, clock_problem
from monitor.detector import Frame, make_detector
from monitor.pipeline import CameraPipeline
from monitor.power import INA226, LowBatteryWatch
from monitor.recorder import Recorder
from monitor.sun import sun_elevation
from monitor.writers import DataWriter

MIN_FPS = 4


class Service:
    def __init__(self, site, site_path: Path, data_dir: Path, detector_kind: str) -> None:
        self.site, self.site_path, self.data_dir = site, Path(site_path), Path(data_dir)
        self.detector_kind = detector_kind
        self.zone = site.zone()
        self.clock = Clock()
        self.writer = DataWriter(self.data_dir, self.zone, lambda: config.site_dict(self.site), on_log=print)
        self.recorder = Recorder(site, self.writer, self.clock)
        self.schedule = schedule.Schedule(site.schedule.windows, self.zone)
        self.stop_event = threading.Event()
        self.lock = threading.RLock()            # the web UI thread and the main loop share state
        self.cameras: dict[str, CameraCapture] = {}
        self.pipelines: dict[str, CameraPipeline] = {}
        self.camera_starts: dict[str, float | None] = {}
        self.detector = None
        self.recording = False
        self.fps = site.detection.frames_per_second
        self.processed = {name: 0 for name in "ABCD"}
        self.dropped_seen = 0
        self.minute_start = time.monotonic()
        self.busy_s, self.busy_window_start, self.calm_since = 0.0, time.monotonic(), None
        self.battery: tuple[float | None, float | None] = (None, None)
        self.low_battery = LowBatteryWatch(site.power.shutdown_below_v, site.power.shutdown_after_s)
        self.power_sensor = self._open_power_sensor()
        self.night = False
        self.exposure_controls: dict[str, nightmode.ExposureControls] = {}
        self.last_activity = time.time()
        self.stay_awake_until = 0.0
        self.watching: dict[str, float] = {}      # camera -> last time someone asked for its picture
        self.latest: dict[str, tuple] = {}        # camera -> (image, boxes) while someone watches
        self.last_system: dict = {}
        self.clock_warning: str | None = None     # set when the clock looks wrong; recording waits for a sync

    # ---- helpers -----------------------------------------------------------
    def log(self, text: str) -> None:
        self.writer.log_event(self.clock.now(), text)

    def _open_power_sensor(self):
        try:
            return INA226()
        except Exception as error:   # no sensor, no smbus2, or I2C disabled: cells stay empty
            print(f"INA226 not available: {error}")
            return None

    # ---- main loop ---------------------------------------------------------
    def run(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: self.stop_event.set())
        self.log(f"startup: collector, detector {self.detector_kind}, site {self.site.site_name}")
        self.check_clock()
        for note in self.writer.recover(self.clock.now()):
            self.log(f"startup repair: {note}")
        next_second = next_minute = 0.0
        reason = "service stopped"
        while not self.stop_event.is_set():
            now = time.time()
            wanted = self.schedule.window_at(now) is not None and self.clock_warning is None
            if wanted and not self.recording:
                self.start_recording()
            elif not wanted and self.recording:
                self.stop_recording("end of recording window")
            try:
                if not (self.recording and self.step()):
                    self.stop_event.wait(0.005 if self.recording else 0.2)
                if now >= next_second:
                    next_second = now + 1.0
                    reason = self.every_second(now) or reason
                if now >= next_minute:
                    minute = now - now % 60
                    next_minute = minute + 60
                    self.every_minute(minute)
            except Exception as error:   # log it and keep going; systemd restarts us if we die anyway
                self.log(f"error: {type(error).__name__}: {error}")
                self.stop_event.wait(1.0)
        self.stop_recording(reason)
        self.log(f"shutdown: {reason}, files closed")
        self.writer.close()

    def check_clock(self) -> None:
        """Recording waits while the clock looks wrong: rows with wrong times would land in the wrong day
        and hour files, and the schedule itself depends on the clock. A clock sync clears this."""
        problem = clock_problem(time.time(), self.data_dir)
        if problem and problem != self.clock_warning:
            self.log(f"error: clock looks wrong ({problem}); not recording until a clock sync from a phone "
                     "or laptop (web page, Clock sync)")
        elif self.clock_warning and not problem:
            self.log("clock looks right again; recording allowed")
        self.clock_warning = problem

    def step(self) -> bool:
        """Process the newest frame of every camera that has one. False if none had one."""
        frames = []
        for name, camera in self.cameras.items():
            if camera.started_at != self.camera_starts.get(name) and camera.started_at is not None:
                self.camera_starts[name] = camera.started_at
                self.pipelines[name].restart(camera.started_at)
            item = camera.take()
            if item is not None:
                frames.append(Frame(name, item[1], item[0]))
        if not frames:
            return False
        started = time.monotonic()
        tiles = {name: [list(t) for t in cam.tiles] for name, cam in self.site.cameras.items()}
        results = self.detector.detect_frames(frames, tiles)
        with self.lock:
            for frame, detections in zip(frames, results):
                pipe = self.pipelines[frame.camera]
                pipe.process(frame.t, detections)
                self.processed[frame.camera] += 1
                if time.monotonic() - self.watching.get(frame.camera, -1e9) < 10:
                    self.latest[frame.camera] = (frame.image, list(pipe.last_boxes))
        self.busy_s += time.monotonic() - started
        return True

    def every_second(self, now: float) -> str | None:
        """Housekeeping. Returns a reason if the device must shut down now."""
        with self.lock:
            mono = time.monotonic()
            for pipe in self.pipelines.values():
                pipe.finish_lost(mono)
            self.writer.tick(self.clock.now())
        if self.recording:
            self.adjust_rate()
        if self.power_sensor is not None:
            self.battery = self.power_sensor.read()
        if self.low_battery.update(now, self.battery[0]):
            self.log(f"battery below {self.site.power.shutdown_below_v:g} V for "
                     f"{self.site.power.shutdown_after_s:g} s: shutting down with no wake alarm")
            self._power_off(None)
            return "low battery"
        if not self.recording and int(now) % 30 == 0 and self.maybe_sleep(now):
            return "sleeping until the next window"
        return None

    def adjust_rate(self) -> None:
        """If processing falls behind, lower every camera's rate evenly; raise it back when calm."""
        elapsed = time.monotonic() - self.busy_window_start
        if elapsed < 10:
            return
        busy = self.busy_s / elapsed
        self.busy_s, self.busy_window_start = 0.0, time.monotonic()
        target = self.site.detection.frames_per_second
        if busy > 0.9 and self.fps > MIN_FPS:
            self._set_rate(self.fps - 1, f"loop busy {busy:.0%}: lowering every camera to")
            self.calm_since = None
        elif busy < 0.6 and self.fps < target:
            self.calm_since = self.calm_since or time.monotonic()
            if time.monotonic() - self.calm_since > 60:
                self._set_rate(self.fps + 1, f"loop busy only {busy:.0%}: raising every camera to")
                self.calm_since = None
        else:
            self.calm_since = None

    def _set_rate(self, fps: float, why: str) -> None:
        self.fps = fps
        for name, camera in self.cameras.items():
            camera.set_rate(fps)
            self.pipelines[name].tracker.set_rate(fps)
        self.log(f"{why} {fps:g} processed frames per second")

    def every_minute(self, minute_epoch: float) -> None:
        elapsed = max(time.monotonic() - self.minute_start, 1.0)
        self.minute_start = time.monotonic()
        fps = {name: self.processed.get(name, 0) / elapsed for name in "ABCD"}
        self.processed = {name: 0 for name in "ABCD"}
        dropped_total = sum(c.dropped for c in self.cameras.values())
        dropped, self.dropped_seen = max(dropped_total - self.dropped_seen, 0), dropped_total
        volts, amps = self.battery
        hailo = self.detector.temperature() if self.detector else None

        def cell(value, digits):
            return "" if value is None else f"{value:.{digits}f}"
        row = [format_epoch(minute_epoch, self.zone), cell(volts, 2), cell(amps, 2),
               cell(volts * amps if volts is not None and amps is not None else None, 1),
               cell(sysinfo.cpu_temp_c(), 1), cell(hailo, 1), *(cell(fps[n], 1) for n in "ABCD"),
               dropped, cell(sysinfo.disk_free_gb(self.data_dir), 2)]
        self.writer.write("system", minute_epoch, row)
        self.last_system = dict(zip(["time", "battery_v", "current_a", "power_w", "cpu_temp_c", "hailo_temp_c",
                                     "fps_A", "fps_B", "fps_C", "fps_D", "dropped_frames", "disk_free_gb"], row))
        self.update_night_mode(minute_epoch)

    # ---- recording on/off ----------------------------------------------------
    def start_recording(self) -> None:
        try:
            self.detector = self.detector or make_detector(self.detector_kind, self.site)
        except Exception as error:
            self.log(f"error: detector {self.detector_kind} could not start: {error}")
            self.stop_event.wait(30)
            return
        self.fps = self.site.detection.frames_per_second
        now = time.monotonic()
        for name, cam in self.site.cameras.items():
            capture = CameraCapture(name, cam.usb_path, self.fps, self.log)
            capture.start()
            self.cameras[name] = capture
            self.camera_starts[name] = None
            self.pipelines[name] = CameraPipeline(name, self.site, self.recorder, self.fps, now)
            self.exposure_controls[name] = nightmode.ExposureControls.find(
                nightmode.parse_controls(nightmode.list_controls(cam.usb_path)))
        self.recording = True
        self.log(f"recording started ({len(self.cameras)} cameras, detector {self.detector.name})")
        self.update_night_mode(self.clock.now(), force=True)

    def stop_recording(self, why: str) -> None:
        if not self.recording:
            return
        with self.lock:
            mono = time.monotonic()
            for pipe in self.pipelines.values():
                pipe.close(mono)   # open tracks get cut_at_shutdown
            for camera in self.cameras.values():
                camera.stop()
            self.cameras.clear()
            self.pipelines.clear()
            self.latest.clear()
            self.recording = False
            self.writer.fsync_all()
        self.log(f"recording stopped: {why}")

    def update_night_mode(self, epoch: float, force: bool = False) -> None:
        elevation = sun_elevation(self.site.latitude, self.site.longitude, epoch)
        night = elevation < self.site.night.switch_sun_elevation_deg
        if night == self.night and not force:
            return
        self.night = self.recorder.night = night
        if not self.recording:
            return
        applied = [nightmode.apply(self.site.cameras[name].usb_path, controls, night,
                                   self.site.night.night_exposure, self.site.night.night_gain)
                   for name, controls in self.exposure_controls.items()]
        self.log(f"mode switch: night mode {'on' if night else 'off'} (sun {elevation:.1f} deg): "
                 + "; ".join(applied))

    # ---- sleeping --------------------------------------------------------------
    def maybe_sleep(self, now: float) -> bool:
        if schedule.ssh_session_open():
            self.last_activity = now
        ok, why = schedule.should_sleep(now, self.schedule, schedule.uptime_s(),
                                        self.site.schedule.boot_grace_min, self.last_activity,
                                        self.stay_awake_until, config.WEB_IDLE_S)
        if not ok:
            return False
        wake = schedule.wake_time(now, self.schedule, self.site.schedule.wake_early_min)
        when = "no wake alarm (no window)" if wake is None else f"wake alarm {format_epoch(wake, self.zone)}"
        self.log(f"going to sleep: {why}; {when}")
        self._power_off(wake)
        return True

    def _power_off(self, wake_epoch: float | None) -> None:
        self.stop_recording("power off")
        self.writer.fsync_all()
        try:
            schedule.set_wake_alarm(wake_epoch)
        except OSError as error:
            self.log(f"error: could not set the RTC wake alarm: {error} (run check_hardware.py --rtc)")
        schedule.power_off()
        self.stop_event.set()
