"""Recording windows across midnight, weekends and both daylight-saving changes."""
from datetime import datetime, timezone

import pytest

from common.timeutil import get_zone, local_epoch
from helpers import sample_site
from monitor import schedule

ZONE = get_zone()
SITE = sample_site()
PLAN = schedule.Schedule(SITE.schedule.windows, ZONE)


def at(day, hour, minute=0):
    return local_epoch(day, hour, minute, ZONE)


def test_inside_and_outside_windows():
    assert PLAN.window_at(at("2026-11-16", 7, 0)) == (at("2026-11-16", 6, 30), at("2026-11-16", 9, 30))
    assert PLAN.window_at(at("2026-11-16", 9, 30)) is None            # the end is not inside
    assert PLAN.window_at(at("2026-11-16", 16, 0)) is not None
    assert PLAN.window_at(at("2026-11-21", 7, 0)) is None             # Saturday


def test_next_window_same_day_and_across_midnight():
    assert PLAN.next_start(at("2026-11-16", 10, 0)) == at("2026-11-16", 15, 30)
    assert PLAN.next_start(at("2026-11-16", 23, 0)) == at("2026-11-17", 6, 30)
    assert PLAN.next_start(at("2026-11-16", 7, 0)) == at("2026-11-16", 15, 30)   # the next one, not the current


def test_friday_evening_to_monday_morning():
    assert PLAN.next_start(at("2026-11-20", 19, 0)) == at("2026-11-23", 6, 30)


def test_spring_forward_weekend():
    # Sat 2026-03-07 20:00 PST -> Mon 2026-03-09 06:30 PDT is 33.5 real hours (the night lost one)
    start, nxt = at("2026-03-07", 20, 0), PLAN.next_start(at("2026-03-07", 20, 0))
    assert nxt == at("2026-03-09", 6, 30)
    assert nxt - start == pytest.approx(33.5 * 3600)
    assert datetime.fromtimestamp(nxt, timezone.utc).hour == 13           # 06:30 PDT = 13:30 UTC


def test_fall_back_weekend():
    # Sat 2026-10-31 20:00 PDT -> Mon 2026-11-02 06:30 PST is 35.5 real hours (one hour repeated)
    start, nxt = at("2026-10-31", 20, 0), PLAN.next_start(at("2026-10-31", 20, 0))
    assert nxt == at("2026-11-02", 6, 30)
    assert nxt - start == pytest.approx(35.5 * 3600)
    assert datetime.fromtimestamp(nxt, timezone.utc).hour == 14


def test_wake_time_is_10_minutes_early():
    assert schedule.wake_time(at("2026-11-20", 19, 0), PLAN, 10) == at("2026-11-23", 6, 20)


@pytest.mark.parametrize("now,uptime,last_use,awake,expected", [
    ((10, 0), 3600, -3600, 0, True),
    ((7, 0), 3600, -3600, 0, False),     # inside a window
    ((10, 0), 600, -3600, 0, False),     # boot grace (15 min)
    ((10, 0), 3600, -300, 0, False),     # web UI used 5 min ago
    ((10, 0), 3600, -3600, 1800, False), # Stay awake on
    ((15, 25), 3600, -3600, 0, False),   # next window in 5 minutes
])
def test_should_sleep(now, uptime, last_use, awake, expected):
    t = at("2026-11-16", *now)
    ok, why = schedule.should_sleep(t, PLAN, uptime, 15, t + last_use, t + awake, 600)
    assert ok is expected, why


def test_wake_alarm_is_cleared_then_set(tmp_path):
    path = tmp_path / "wakealarm"
    writes = []

    class Recorder(type(path)):
        pass
    path.write_text("")
    original = type(path).write_text
    try:
        type(path).write_text = lambda self, text, *a, **k: (writes.append(text), original(self, text, *a, **k))[1]
        schedule.set_wake_alarm(1_800_000_000.7, path)
        schedule.set_wake_alarm(None, path)
    finally:
        type(path).write_text = original
    assert writes == ["0\n", "1800000000\n", "0\n"]
