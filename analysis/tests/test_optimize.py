"""The optimizer keeps every constraint and is never worse than Webster."""
import numpy as np
import pytest

from tlopt import hcm, optimize
from tlopt_helpers import light_problem


def delay(problem, plan):
    return problem.total_delay(np.array([plan.greens[p] for p in problem.phases]), plan.cycle_s)


def ring_totals_equal(problem, plan):
    for rings in problem.sides:
        totals = [sum(plan.greens[p] + problem.lost[p] for p in ring) for ring in rings if ring]
        if max(totals) - min(totals) > 1e-6:
            return False
    return True


@pytest.mark.parametrize("v_ns,v_ew", [(540, 360), (900, 300), (1000, 500), (200, 150)])
def test_plan_meets_every_constraint_and_beats_webster(v_ns, v_ew):
    problem = light_problem(v_ns, v_ew)
    plan, curve = optimize.best_plan(problem)
    webster = optimize.webster(problem)
    assert optimize.check(problem, plan) == []
    assert ring_totals_equal(problem, plan)
    assert all(float(g).is_integer() for g in plan.greens.values())
    low, high = problem.inter.options.cycle_bounds_s
    assert low <= plan.cycle_s <= high
    assert delay(problem, plan) <= delay(problem, webster) + 1e-9
    assert len(curve) == high - max(low, np.ceil(problem.min_cycle())) + 1   # every whole-second cycle tried


def test_heavier_direction_gets_more_green():
    plan, _ = optimize.best_plan(light_problem(1000, 300))
    assert plan.greens[2] > plan.greens[4]


def test_impossible_x_limit_is_reported_and_relaxed():
    problem = light_problem(1500, 1400, bounds=(30, 120))   # Y = 1.61: X <= 0.92 cannot be met
    plan, _ = optimize.best_plan(problem)
    assert any("not possible" in note for note in plan.notes)
    assert optimize.check(problem, plan, 9.9) == []


def test_max_wait_cap_is_respected():
    problem = light_problem(540, 360)
    free, _ = optimize.best_plan(problem)
    worst = max(problem.max_wait(np.array([free.greens[p] for p in problem.phases]), free.cycle_s).values())
    capped_inter = problem.inter.model_copy(update={"options": problem.inter.options.model_copy(
        update={"max_wait_cap_s": worst - 5})})
    capped = optimize.Problem(capped_inter, problem.groups, problem.yellow, problem.all_red, problem.min_green)
    plan, _ = optimize.best_plan(capped)
    assert max(capped.max_wait(np.array([plan.greens[p] for p in capped.phases]), plan.cycle_s).values()) <= worst - 5


def test_vectorized_model_matches_the_plain_formulas():
    problem = light_problem(700, 400)
    greens = {2: 30.0, 6: 30.0, 4: 20.0, 8: 20.0}
    vector = problem.total_delay(np.array([greens[p] for p in problem.phases]), 72.0)
    plain = hcm.evaluate(problem.groups, greens, 72.0)["veh_hours_per_hour"]
    assert vector == pytest.approx(plain)
