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


def test_raises_when_no_keys_and_end_before_start():
    start_dt = datetime.datetime(2020, 1, 2, 0, 0, 0)
    end_dt = datetime.datetime(2020, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError):
        runlen.resolve_run_seconds(0, 0, 0, 0, start_dt, end_dt)
