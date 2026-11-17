"""Night mode: manual exposure after sunset, automatic exposure in daylight (v4l2-ctl).

Control names differ between camera drivers (auto_exposure or exposure_auto,
exposure_time_absolute or exposure_absolute), so we read what each camera has."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

_CONTROL = re.compile(r"^\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)$")
UVC_MANUAL, UVC_APERTURE_PRIORITY = 1, 3   # auto_exposure menu values


def parse_controls(text: str) -> dict[str, dict[str, int]]:
    """`v4l2-ctl --list-ctrls` output -> {name: {"min": .., "max": .., "value": ..}}."""
    controls = {}
    for line in text.splitlines():
        match = _CONTROL.match(line)
        if match:
            values = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", match.group(3))}
            controls[match.group(1)] = values
    return controls


@dataclass
class ExposureControls:
    auto: str | None
    exposure: str | None
    gain: str | None
    limits: dict[str, dict[str, int]]

    @classmethod
    def find(cls, controls: dict[str, dict[str, int]]) -> "ExposureControls":
        def first(*names: str) -> str | None:
            return next((n for n in names if n in controls), None)
        return cls(first("auto_exposure", "exposure_auto"),
                   first("exposure_time_absolute", "exposure_absolute"), first("gain"), controls)

    def _clamp(self, name: str, value: int) -> int:
        limit = self.limits.get(name, {})
        return max(limit.get("min", value), min(limit.get("max", value), value))

    def settings(self, night: bool, exposure: int, gain: int | None) -> list[tuple[str, int]]:
        if self.auto is None:
            return []
        if not night:
            return [(self.auto, UVC_APERTURE_PRIORITY)]
        out = [(self.auto, UVC_MANUAL)]
        if self.exposure:
            out.append((self.exposure, self._clamp(self.exposure, exposure)))
        if gain is not None and self.gain:
            out.append((self.gain, self._clamp(self.gain, gain)))
        return out


def list_controls(device: str) -> str:
    try:
        return subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls"], capture_output=True,
                              text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def apply(device: str, controls: ExposureControls, night: bool, exposure: int, gain: int | None,
          run=subprocess.run) -> str:
    """Set the camera; returns a plain description for events.log."""
    settings = controls.settings(night, exposure, gain)
    if not settings:
        return f"{device}: no exposure control found, nothing changed"
    command = ["v4l2-ctl", "-d", device] + [arg for name, value in settings for arg in ("-c", f"{name}={value}")]
    try:
        result = run(command, capture_output=True, text=True, timeout=10)
        ok = result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    applied = ", ".join(f"{name}={value}" for name, value in settings)
    return f"{device}: {applied}" + ("" if ok else " (v4l2-ctl FAILED)")
