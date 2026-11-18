"""intersection.yaml: what the cameras cannot see (geometry, phasing, timing sheet, periods)."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Approach = Literal["N", "S", "E", "W"]
Movement = Literal["L", "T", "R"]
RING1 = [[1, 2], [3, 4]]   # NEMA ring-barrier: [phases before the barrier], [phases after it]
RING2 = [[5, 6], [7, 8]]
PERPENDICULAR = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}
_PERIOD = re.compile(r"^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-4]):[0-5]\d$")


class IntersectionError(ValueError):
    pass


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApproachGeometry(Strict):
    lanes: dict[str, list[Movement]] = Field(min_length=1)
    speed_limit_mph: float = Field(gt=0)
    grade_pct: float = 0.0
    ped_crossing_ft: float = Field(gt=0)
    clear_width_ft: float = Field(gt=0)


class Phase(Strict):
    approach: Approach
    movements: list[Movement] = Field(min_length=1)
    left: Literal["protected", "permissive", "none"] = "none"


class SheetPhase(Strict):
    green_s: float = Field(gt=0)
    yellow_s: float = Field(gt=0)
    all_red_s: float = Field(ge=0)


class TimingSheet(Strict):
    cycle_s: float = Field(gt=0)
    phases: dict[str, SheetPhase]
    offset_s: float = 0.0
    note: str = ""


class Pedestrians(Strict):
    walk_s: float = 7.0
    walk_speed_ftps: float = 3.5
    peds_per_hour: dict[Approach, float] = {"N": 30, "S": 30, "E": 30, "W": 30}


class Options(Strict):
    daylight_only: bool = True
    days: list[Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]] = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    skip_dates: list[str] = []     # e.g. holidays or a day with a crash; YYYY-MM-DD
    cycle_bounds_s: tuple[float, float] = (60, 180)
    max_degree_of_saturation: float = Field(default=0.92, gt=0, le=1.5)
    max_wait_cap_s: Optional[float] = None


class Intersection(Strict):
    name: str
    approaches: dict[Approach, ApproachGeometry]
    phases: dict[str, Phase]
    timing_sheet: Optional[TimingSheet] = None
    pedestrians: Pedestrians = Pedestrians()
    periods: dict[str, str]
    options: Options = Options()

    @model_validator(mode="after")
    def _check(self) -> "Intersection":
        for key in self.phases:
            if key not in {str(n) for n in range(1, 9)}:
                raise ValueError(f"phases: {key!r} is not a NEMA phase 1-8")
        for key, phase in self.phases.items():
            lanes = self.approaches[phase.approach].lanes if phase.approach in self.approaches else {}
            served = {m for moves in lanes.values() for m in moves}
            missing = set(phase.movements) - served
            if missing:
                raise ValueError(f"phases.{key}: approach {phase.approach} has no lane for {sorted(missing)}")
        for approach, geo in self.approaches.items():
            for move in {m for moves in geo.lanes.values() for m in moves}:
                if not any(p.approach == approach and move in p.movements for p in self.phases.values()):
                    raise ValueError(f"approaches.{approach}: no phase serves movement {move}")
        if self.timing_sheet and set(self.timing_sheet.phases) != set(self.phases):
            raise ValueError("timing_sheet.phases must list exactly the same phases as phases")
        for name, text in self.periods.items():
            if not _PERIOD.match(text):
                raise ValueError(f"periods.{name}: write it like 07:00-09:00")
        for text in self.options.skip_dates:
            try:
                date.fromisoformat(str(text))
            except ValueError:
                raise ValueError(f"options.skip_dates: {text!r} is not a date written YYYY-MM-DD")
        if not self.options.days:
            raise ValueError("options.days: list at least one day, e.g. [Mon, Tue, Wed, Thu, Fri]")
        low, high = self.options.cycle_bounds_s
        if not 0 < low < high:
            raise ValueError("options.cycle_bounds_s: must be [smaller, larger]")
        return self

    # ---- helpers ------------------------------------------------------------
    def phase_numbers(self) -> list[int]:
        return sorted(int(k) for k in self.phases)

    def ring_sides(self) -> list[list[list[int]]]:
        """[[ring1 phases on side A, ring2 phases on side A], [... side B]], only phases that exist."""
        present = set(self.phase_numbers())
        sides = []
        for side in range(2):
            rings = [[p for p in ring[side] if p in present] for ring in (RING1, RING2)]
            if any(rings):
                sides.append(rings)
        return sides

    def period_minutes(self, name: str) -> tuple[int, int]:
        start, end = self.periods[name].split("-")
        to_min = lambda hm: int(hm[:2]) * 60 + int(hm[3:])  # noqa: E731
        return to_min(start), to_min(end)


def load_intersection(path: Path) -> Intersection:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise IntersectionError(f"{path} does not exist") from None
    except yaml.YAMLError as error:
        raise IntersectionError(f"{path} is not valid YAML: {error}") from None
    try:
        return Intersection.model_validate(data)
    except ValidationError as error:
        lines = [f"  {'.'.join(str(p) for p in e['loc']) or '(whole file)'}: {e['msg'].removeprefix('Value error, ')}"
                 for e in error.errors()]
        raise IntersectionError(f"{path} is not valid:\n" + "\n".join(lines)) from None
