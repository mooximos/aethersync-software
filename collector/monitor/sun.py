"""Sun elevation (astral package) and the day/twilight/night label."""
from __future__ import annotations

from datetime import datetime, timezone

from astral import Observer
from astral.sun import elevation


def sun_elevation(latitude: float, longitude: float, epoch_s: float) -> float:
    """Degrees above the horizon (negative = below), including refraction."""
    moment = datetime.fromtimestamp(epoch_s, timezone.utc)
    return float(elevation(Observer(latitude, longitude), moment))


def light_label(elevation_deg: float) -> str:
    if elevation_deg > 0:
        return "day"
    if elevation_deg >= -6:
        return "twilight"
    return "night"
