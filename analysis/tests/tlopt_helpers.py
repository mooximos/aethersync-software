"""Shared helpers for the analysis tests."""
from __future__ import annotations

from pathlib import Path

import yaml

from tlopt import hcm, optimize
from tlopt.intersection import Intersection

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "analysis" / "intersection.example.yaml"
SAMPLE = REPO / "sample_data"


def example_dict() -> dict:
    return yaml.safe_load(EXAMPLE.read_text())


def intersection(**changes) -> Intersection:
    data = example_dict()
    for key, value in changes.items():
        data[key] = value
    return Intersection.model_validate(data)


def light_problem(v_ns: float, v_ew: float, bounds=(30, 180)):
    """Two-phase intersection with tiny crosswalks, so only the 7 s minimum applies."""
    data = example_dict()
    for leg in data["approaches"].values():
        leg["ped_crossing_ft"] = 0.01
    inter = intersection(approaches=data["approaches"], options={"cycle_bounds_s": list(bounds)})
    groups = [hcm.GroupInput("S", "TR", 2, v_ns, 1800, [2]), hcm.GroupInput("N", "TR", 2, v_ns, 1800, [6]),
              hcm.GroupInput("W", "TR", 2, v_ew, 1800, [4]), hcm.GroupInput("E", "TR", 2, v_ew, 1800, [8])]
    yellow, all_red, min_green, _why = optimize.safety(inter)
    return optimize.Problem(inter, groups, yellow, all_red, min_green, "test")
