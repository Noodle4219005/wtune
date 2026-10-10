#!/usr/bin/env python3
"""Tests for wtune/runlength.py -- the single-sourced WRF run-length resolver.

Run: /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q test_wtune_runlength.py

Pure unit tests on resolve_run_seconds() itself (already-extracted scalars in, seconds out).
For the same real-namelist values (lafe/isc21) exercised THROUGH each of the four call sites,
see tools/test_temporal_validation.py (read_run_length_hours, pre-existing/unchanged),
test_wtune_predict.py (get_run_seconds), test_wtune.py (extract_case) and
tools/test_metdata.py (the WPS GRIB-time count).
"""
import datetime

import pytest

from wtune import runlength as runlen


def test_keys_win_over_dates():
    """isc21 shape: run_minutes=30 alone is nonzero -> 1800s, even though start->end spans a
    DIFFERENT 15 hours (2016-08-29 09:00 -> 2016-08-30 00:00) that must NOT win."""
    start_dt = datetime.datetime(2016, 8, 29, 9, 0, 0)
    end_dt = datetime.datetime(2016, 8, 30, 0, 0, 0)
    assert runlen.resolve_run_seconds(0, 0, 30, 0, start_dt, end_dt) == 1800.0


def test_dates_used_when_all_keys_zero():
    """lafe shape: every run_* key is 0 -> falls back to the start->end span (2017-08-23 06:00
    -> 2017-08-24 06:00 = 24h = 86400s), not 0."""
    start_dt = datetime.datetime(2017, 8, 23, 6, 0, 0)
    end_dt = datetime.datetime(2017, 8, 24, 6, 0, 0)
    assert runlen.resolve_run_seconds(0, 0, 0, 0, start_dt, end_dt) == 86400.0


def test_days_hours_minutes_seconds_all_combine():
    assert runlen.resolve_run_seconds(1, 2, 3, 4, None, None) == 1 * 86400 + 2 * 3600 + 3 * 60 + 4


def test_raises_when_no_keys_and_no_dates():
    with pytest.raises(ValueError):
        runlen.resolve_run_seconds(0, 0, 0, 0, None, None)


def test_raises_when_no_keys_and_dates_are_equal():
    """A zero-length date span is not a positive span either -- must still raise, never return 0."""
    same = datetime.datetime(2020, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError):
        runlen.resolve_run_seconds(0, 0, 0, 0, same, same)


def test_raises_when_no_keys_and_end_before_start():
    start_dt = datetime.datetime(2020, 1, 2, 0, 0, 0)
    end_dt = datetime.datetime(2020, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError):
        runlen.resolve_run_seconds(0, 0, 0, 0, start_dt, end_dt)


def test_raises_when_no_keys_and_only_one_date_given():
    with pytest.raises(ValueError):
        runlen.resolve_run_seconds(0, 0, 0, 0, datetime.datetime(2020, 1, 1), None)


def test_never_returns_zero_or_negative():
    """Every branch (keys, dates, raise) must be exercised without ever producing <= 0."""
    assert runlen.resolve_run_seconds(0, 0, 1, 0, None, None) > 0
    assert runlen.resolve_run_seconds(
        0, 0, 0, 0, datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)) > 0
    with pytest.raises(ValueError):
        runlen.resolve_run_seconds(0, 0, 0, 0, None, None)
