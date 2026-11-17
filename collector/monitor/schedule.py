"""Recording windows, when to sleep, and the RTC wake alarm."""
from __future__ import annotations

import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WAKEALARM = Path("/sys/class/rtc/rtc0/wakealarm")
MIN_SLEEP_S = 600   # not worth shutting down if the next wake-up is sooner than this


def local_epoch(day: date, clock: str, zone: ZoneInfo) -> float:
    hour, minute = (int(part) for part in clock.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone).timestamp()


class Schedule:
    def __init__(self, windows: list, zone: ZoneInfo) -> None:
        self.windows = windows   # items with .days, .start, .end (site.json schedule.windows)
        self.zone = zone

    def _windows_on(self, day: date) -> list[tuple[float, float]]:
        name = DAY_NAMES[day.weekday()]
        return sorted((local_epoch(day, w.start, self.zone), local_epoch(day, w.end, self.zone))
                      for w in self.windows if name in w.days)

    def window_at(self, epoch_s: float) -> tuple[float, float] | None:
        """(start, end) of the window running now, or None."""
        today = datetime.fromtimestamp(epoch_s, self.zone).date()
        for start, end in self._windows_on(today):
            if start <= epoch_s < end:
                return start, end
        return None

    def next_start(self, epoch_s: float) -> float | None:
        """Start of the next window that begins after epoch_s (looks up to 8 days ahead)."""
        today = datetime.fromtimestamp(epoch_s, self.zone).date()
        for offset in range(9):
            for start, _end in self._windows_on(today + timedelta(days=offset)):
                if start > epoch_s:
                    return start
        return None


def should_sleep(now_s: float, schedule: Schedule, uptime_s: float, boot_grace_min: float,
                 last_activity_s: float, stay_awake_until_s: float, idle_needed_s: float) -> tuple[bool, str]:
    """All of the sleep rules in one place. Returns (sleep?, reason)."""
    if schedule.window_at(now_s):
        return False, "inside a recording window"
    if uptime_s < boot_grace_min * 60:
        return False, f"less than {boot_grace_min:g} min since boot"
    if now_s - last_activity_s < idle_needed_s:
        return False, "web UI or SSH used recently"
    if stay_awake_until_s > now_s:
        return False, "Stay awake is on"
    upcoming = schedule.next_start(now_s)
    if upcoming is not None and upcoming - now_s < MIN_SLEEP_S:
        return False, "next window starts soon"
    return True, "outside every window and nobody is using the device"


def wake_time(now_s: float, schedule: Schedule, wake_early_min: float) -> float | None:
    upcoming = schedule.next_start(now_s)
    return None if upcoming is None else upcoming - wake_early_min * 60


def set_wake_alarm(epoch_s: float | None, path: Path = WAKEALARM) -> None:
    """Clear the alarm (write 0), then set it, as the kernel requires. None = no alarm."""
    path.write_text("0\n")
    if epoch_s is not None:
        path.write_text(f"{int(epoch_s)}\n")


def ssh_session_open() -> bool:
    """True if someone is logged in over SSH (a pts session in `who`)."""
    try:
        output = subprocess.run(["who"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any("pts/" in line for line in output.splitlines())


def uptime_s() -> float:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        import time
        return time.monotonic()


def power_off() -> None:
    subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
