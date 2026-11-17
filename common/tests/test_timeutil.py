from datetime import datetime, timedelta, timezone

import pytest

from common import timeutil

ZONE = timeutil.get_zone("America/Los_Angeles")


def test_format_matches_the_example_in_the_data_format():
    moment = datetime(2026, 11, 16, 7, 14, 52, 300_000, tzinfo=ZONE)
    assert timeutil.format_epoch(moment.timestamp(), ZONE) == "2026-11-16T07:14:52.3-08:00"
    assert timeutil.format_time(moment) == "2026-11-16T07:14:52.3-08:00"


def test_summer_time_uses_the_other_offset():
    moment = datetime(2026, 7, 1, 17, 0, 0, tzinfo=ZONE)
    assert timeutil.format_epoch(moment.timestamp(), ZONE) == "2026-07-01T17:00:00.0-07:00"


def test_parse_and_format_round_trip():
    text = "2026-11-16T07:14:52.3-08:00"
    assert timeutil.format_epoch(timeutil.parse_epoch(text), ZONE) == text
    assert timeutil.parse_time(text).utcoffset() == timedelta(hours=-8)


def test_rounding_carries_into_the_next_second():
    base = datetime(2026, 11, 16, 7, 59, 59, tzinfo=ZONE).timestamp()
    assert timeutil.format_epoch(base + 0.96, ZONE) == "2026-11-16T08:00:00.0-08:00"
    assert timeutil.format_epoch(base + 0.94, ZONE) == "2026-11-16T07:59:59.9-08:00"


def test_spring_forward_day_skips_an_hour_but_durations_stay_right():
    # 2026-03-08: clocks jump from 02:00 PST to 03:00 PDT.
    before = datetime(2026, 3, 8, 9, 59, 59, tzinfo=timezone.utc).timestamp()   # 01:59:59 PST
    assert timeutil.format_epoch(before, ZONE) == "2026-03-08T01:59:59.0-08:00"
    assert timeutil.format_epoch(before + 1, ZONE) == "2026-03-08T03:00:00.0-07:00"
    assert timeutil.parse_epoch("2026-03-08T03:00:00.0-07:00") - timeutil.parse_epoch("2026-03-08T01:59:59.0-08:00") == 1.0


def test_fall_back_day_repeats_an_hour_and_the_offset_tells_them_apart():
    # 2026-11-01: clocks go from 02:00 PDT back to 01:00 PST.
    first = datetime(2026, 11, 1, 8, 30, tzinfo=timezone.utc).timestamp()    # 01:30 PDT
    assert timeutil.format_epoch(first, ZONE) == "2026-11-01T01:30:00.0-07:00"
    assert timeutil.format_epoch(first + 3600, ZONE) == "2026-11-01T01:30:00.0-08:00"
    assert timeutil.local_day_and_hour(first, ZONE) == ("2026-11-01", 1)
    assert timeutil.local_day_and_hour(first + 3600, ZONE) == ("2026-11-01", 1)


@pytest.mark.parametrize("bad", [
    "2026-11-16 07:14:52.3-08:00",     # space instead of T
    "2026-11-16T07:14:52-08:00",       # no tenths
    "2026-11-16T07:14:52.34-08:00",    # two decimals
    "2026-11-16T07:14:52.3",           # no offset
    "2026-11-16T07:14:52.3Z",          # Z instead of an offset
    "2026-13-16T07:14:52.3-08:00",     # month 13
    "",
])
def test_bad_times_are_rejected(bad):
    assert not timeutil.is_valid_time(bad)
    with pytest.raises(ValueError):
        timeutil.parse_time(bad)


def test_local_epoch_and_day_hour_agree():
    epoch = timeutil.local_epoch("2026-05-12", 17, 30, ZONE, second=12.5)
    assert timeutil.format_epoch(epoch, ZONE) == "2026-05-12T17:30:12.5-07:00"
    assert timeutil.local_day_and_hour(epoch, ZONE) == ("2026-05-12", 17)
    assert timeutil.clock_text(epoch, ZONE) == "17:30:12.5"
