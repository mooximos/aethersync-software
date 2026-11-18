"""site.json: the settings and calibration file (section 2.1 of the data format).

load_site() reads and checks it. If anything is wrong it raises SiteError with a
message that names the bad key, e.g. "cameras.A.approach: Input should be 'N', 'S', 'E' or 'W'".
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from common.schema import EVENT_CODES

Approach = Literal["N", "S", "E", "W"]
Movement = Literal["L", "T", "R"]
VehicleType = Literal["car", "motorcycle", "bus", "truck"]
Day = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PRIVACY_TEXT = "no images or video are stored"
_CLOCK = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class SiteError(ValueError):
    """site.json is missing or wrong. The message says which key."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelled key is an error, not silently ignored


class Device(Strict):
    computer: str
    ai_chip: str
    mount_height_m: float = Field(gt=0)


class Lane(Strict):
    movements: list[Movement] = Field(min_length=1)


class CalibPoint(Strict):
    px: tuple[float, float]
    m: tuple[float, float]


class Camera(Strict):
    approach: Approach
    usb_path: str
    lanes: dict[str, Lane] = Field(min_length=1)
    lane_x_m: dict[str, tuple[float, float]]
    homography: Optional[list[list[float]]] = None
    calib_points: list[CalibPoint] = []
    tiles: list[tuple[int, int, int, int]] = []
    stop_line_zone_m: float = Field(default=6.0, gt=0)
    view_length_m: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> "Camera":
        if any(not key.isdigit() or int(key) < 1 for key in self.lanes):
            raise ValueError("lanes: lane names must be whole numbers, 1 = curbside")
        if set(self.lanes) != set(self.lane_x_m):
            raise ValueError("lane_x_m must have exactly the same lane numbers as lanes")
        for lane, (x0, x1) in self.lane_x_m.items():
            if not x0 < x1:
                raise ValueError(f"lane_x_m.{lane}: the first number must be smaller than the second")
        if self.homography is not None:
            if len(self.homography) != 3 or any(len(r) != 3 for r in self.homography):
                raise ValueError("homography must be a 3x3 matrix (or null until calibrated)")
        return self


class Coordinates(Strict):
    x_m: str
    y_m: str


class Detection(Strict):
    model: str = "yolov8s"
    hef_path: str
    input_px: int = Field(default=640, gt=0)
    frames_per_second: float = Field(default=10, gt=0, le=30)
    min_confidence: float = Field(default=0.35, ge=0, le=1)
    low_conf_flag_below: float = Field(default=0.65, ge=0, le=1)
    classes: list[VehicleType] = Field(min_length=1)


class Window(Strict):
    days: list[Day] = Field(min_length=1)
    start: str
    end: str

    @model_validator(mode="after")
    def _check(self) -> "Window":
        for name in ("start", "end"):
            if not _CLOCK.match(getattr(self, name)):
                raise ValueError(f"{name}: must be a 24-hour clock time like 06:30")
        if not self.start < self.end:
            raise ValueError("start must be earlier than end (a window cannot cross midnight)")
        return self


class Schedule(Strict):
    windows: list[Window]
    boot_grace_min: float = Field(default=15, ge=0)
    wake_early_min: float = Field(default=10, ge=0)


class Power(Strict):
    shutdown_below_v: float = Field(default=12.0, gt=0)
    shutdown_after_s: float = Field(default=60, ge=0)


class Night(Strict):
    switch_sun_elevation_deg: float = -2
    night_exposure: int = Field(default=40, gt=0)
    night_gain: Optional[int] = None


class Clock(Strict):
    last_sync: Optional[str] = None
    offset_found_s: Optional[float] = None


class Site(Strict):
    site_name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    time_zone: str
    researchers: list[str]
    software_version: str
    privacy: str = PRIVACY_TEXT
    web_password: Optional[str] = None
    device: Device
    cameras: dict[Literal["A", "B", "C", "D"], Camera] = Field(min_length=1)
    coordinates: Coordinates
    detection: Detection
    stopped_speed_mps: float = Field(default=0.5, gt=0)
    moving_speed_mps: float = Field(default=1.0, gt=0)
    reaction_time_s: float = Field(default=1.5, ge=0)
    track_sample_rate_hz: float = Field(default=2, gt=0)
    schedule: Schedule
    power: Power
    night: Night
    event_codes: dict[str, str]
    clock: Clock = Clock()

    @field_validator("time_zone")
    @classmethod
    def _zone_exists(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("not a known time zone name (example: America/Los_Angeles)")
        return value

    @field_validator("event_codes")
    @classmethod
    def _codes_match(cls, value: dict[str, str]) -> dict[str, str]:
        expected = {str(code): name for code, name in EVENT_CODES.items()}
        if value != expected:
            raise ValueError(f"must be exactly {expected}")
        return value

    @model_validator(mode="after")
    def _check(self) -> "Site":
        if not self.stopped_speed_mps < self.moving_speed_mps:
            raise ValueError("stopped_speed_mps must be smaller than moving_speed_mps")
        if self.latitude == 0 and self.longitude == 0:
            raise ValueError("latitude, longitude: still 0.0, 0.0; type the intersection's position from an "
                             "online map (west of Greenwich the longitude is negative, e.g. -122.3). Day and "
                             "night are worked out from it")
        zone = ZoneInfo(self.time_zone)   # its name was checked above
        standard_h = min(datetime(2001, month, 1, 12, tzinfo=zone).utcoffset().total_seconds() / 3600
                         for month in (1, 7))       # the offset without daylight saving time
        if abs(standard_h - self.longitude / 15) > 4:
            raise ValueError(f"longitude: {self.longitude} does not fit time_zone {self.time_zone} "
                             "(did the minus sign get lost? West of Greenwich the longitude is negative)")
        approaches = [camera.approach for camera in self.cameras.values()]
        if len(set(approaches)) != len(approaches):
            raise ValueError("cameras: two cameras have the same approach")
        return self

    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.time_zone)

    def camera_for_approach(self, approach: str) -> str | None:
        for name, camera in self.cameras.items():
            if camera.approach == approach:
                return name
        return None


def validate_site(data: dict, source: str = "site.json") -> Site:
    """Check a site dictionary. Raises SiteError naming each bad key."""
    try:
        return Site.model_validate(data)
    except ValidationError as error:
        lines = []
        for item in error.errors():
            key = ".".join(str(part) for part in item["loc"]) or "(whole file)"
            message = item["msg"].removeprefix("Value error, ")
            lines.append(f"  {key}: {message}")
        raise SiteError(f"{source} is not valid:\n" + "\n".join(lines)) from None


def load_site(path: Path) -> Site:
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise SiteError(f"{path} does not exist") from None
    except json.JSONDecodeError as error:
        raise SiteError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(data, dict):
        raise SiteError(f"{path} must contain a JSON object")
    return validate_site(data, source=str(path))
