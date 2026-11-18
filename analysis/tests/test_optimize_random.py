"""Randomized checks: whatever the traffic, every plan keeps the safety rules.

Uses the example intersection's real phases, crosswalks and clearance times, with random
volumes (unequal on the two sides of each phase pair, so the dual ring matters)."""
import numpy as np
import pytest

from tlopt import hcm, optimize
from tlopt_helpers import intersection

PHASE_OF = {"S": 2, "N": 6, "W": 4, "E": 8}


def random_problem(rng, scale: float):
    inter = intersection()
    groups = []
    for approach, phase in PHASE_OF.items():
        groups.append(hcm.GroupInput(approach, "TR", 2, float(rng.uniform(20, 900) * scale),
                                     float(rng.uniform(1500, 1900)), [phase]))
        groups.append(hcm.GroupInput(approach, "L", 1, float(rng.uniform(0, 200) * scale),
                                     float(rng.uniform(600, 1500)), [phase]))
    yellow, all_red, min_green, _why = optimize.safety(inter)
    return optimize.Problem(inter, groups, yellow, all_red, min_green, "random")


def ring_totals_equal(problem, plan):
    for rings in problem.sides:
        totals = [sum(plan.greens[p] + problem.lost[p] for p in ring) for ring in rings if ring]
        if max(totals) - min(totals) > 1e-6:
            return False
    return True


@pytest.mark.parametrize("seed", range(30))
def test_random_traffic_always_gives_a_safe_plan(seed):
    rng = np.random.default_rng(seed)
    problem = random_problem(rng, scale=1.0 if seed < 20 else 1.8)   # the last 10 are overloaded
    plan, _curve = optimize.best_plan(problem)
    low, high = problem.inter.options.cycle_bounds_s
    assert low <= plan.cycle_s <= high and float(plan.cycle_s).is_integer()
    assert all(float(g).is_integer() and g >= problem.min_green[p] for p, g in plan.greens.items())
    assert all(plan.yellow[p] >= problem.yellow[p] and plan.all_red[p] >= problem.all_red[p] for p in plan.greens)
    assert ring_totals_equal(problem, plan)
    sides = problem.side_lengths(plan.greens)
    assert abs(sum(sides) - plan.cycle_s) < 1e-6          # the two sides of the barrier fill the cycle
    problems = optimize.check(problem, plan)
    if problems:   # only allowed when the X limit cannot be met, and then the plan must say so
        assert all("X =" in issue for issue in problems)
        assert any("not possible" in note for note in plan.notes)
    greens = np.array([plan.greens[p] for p in problem.phases])
    webster = optimize.webster(problem)
    w_greens = np.array([webster.greens[p] for p in problem.phases])
    if not optimize.check(problem, webster) and not problems:   # Webster is allowed too: never worse than it
        assert problem.total_delay(greens, plan.cycle_s) <= problem.total_delay(w_greens, webster.cycle_s) + 1e-6
    assert np.isfinite(problem.total_delay(greens, plan.cycle_s))


def test_an_approach_with_no_traffic_still_gets_its_minimum_green():
    rng = np.random.default_rng(99)
    problem = random_problem(rng, 1.0)
    for group in problem.groups:
        if group.approach == "E":
            group.v = 0.0
    problem = optimize.Problem(problem.inter, problem.groups, problem.yellow, problem.all_red,
                               problem.min_green, "no E traffic")
    plan, _ = optimize.best_plan(problem)
    assert plan.greens[8] >= problem.min_green[8]
    assert optimize.check(problem, plan) == [] or any("not possible" in n for n in plan.notes)
