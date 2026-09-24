#!/usr/bin/env python3
"""Tests for wtune.py's own core functions (parse_namelist/extract_case) that are not already
covered indirectly through wtune_case.py/wtune_predict.py's own test files.

Run: /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q test_wtune.py

Focus here: extract_case's 'run_total_hours' field, which used to be `run_hours + run_days * 24`
-- silently dropping run_minutes/run_seconds and having no start_date/end_date fallback, so it
returned 0 for BOTH real cluster namelists this module is regression-tested against below (see
wtune_runlength.py's module docstring for the shared resolver both this function and
tools/temporal_validation.py:read_run_length_hours / wtune_predict.py:get_run_seconds now go
through). Fixtures are transcribed verbatim from the real templates (leading-zero months,
per-domain comma lists, irregular spacing all preserved -- see
tools/test_temporal_validation.py's NML_LAFE_SHAPE/NML_ISC21_SHAPE and
test_wtune_predict.py's LAFE_TIME_CONTROL/ISC21_TIME_CONTROL for the same two cases exercised
through the other call sites).
"""
import os

import wtune as wt

LAFE_NML = """\
 &time_control
 run_days                            = 0,
 run_hours                           = 0,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2017, 2017, 2017,
 start_month                         = 08,   08,   08,
 start_day                           = 23,   23,   23,
 start_hour                          = 06,   06,   06,
 start_minute                        = 00,   00,   00,
 start_second                        = 00,   00,   00,
 end_year                            = 2017, 2017, 2017,
 end_month                           = 08,   08,   08,
 end_day                             = 24,   24,   24,
 end_hour                            = 06,   06,   06,
 end_minute                          = 00,   00,   00,
 end_second                          = 00,   00,   00,
 interval_seconds                    = 21600,
 /
 &domains
 time_step                           = 10,
 max_dom                             = 3,
 e_we                                = 1000,  1001,  1201,
 e_sn                                = 1000,  1001,  1001,
 e_vert                              = 100,    100,   100,
 dx                                  = 2500,  500, 100,
 dy                                  = 2500,  500, 100,
 /
 &physics
 /
"""

ISC21_NML = """\
&time_control
   run_days = 0,
   run_hours = 0,
   run_minutes = 30,
   run_seconds = 00,
   start_year =  2016, 2016, 2016
   start_month = 08,   08,   08,
   start_day =   29,   29,   29,
   start_hour =  09,   09,   09,
   end_year =    2016, 2016, 2016,
   end_month =   08,   08,   08,
   end_day =     30,   30,   30,
   end_hour =    00,   00,   00,
   interval_seconds = 10800,
 /
 &domains
   time_step = 15,
   max_dom = 3,
   e_we = 793, 805, 1001,
   e_sn = 853, 805, 1001,
   e_vert = 51, 51, 51,
   dx = 3000,
   dy = 3000,
 /
 &physics
 /
"""


def _write(tmp_path, name, text):
    p = os.path.join(str(tmp_path), name)
    with open(p, "w") as fh:
        fh.write(text)
    return p


def test_extract_case_lafe_shape_run_total_hours_is_24(tmp_path):
    """Every run_* key is 0 -> extract_case must fall back to start_date (2017-08-23 06:00) ->
    end_date (2017-08-24 06:00) = 24.0h, not 0 (the old `run_hours + run_days*24` formula's
    answer for this exact shape, since it never looked at start_date/end_date at all)."""
    nl = wt.parse_namelist(_write(tmp_path, "namelist.input", LAFE_NML))
    case = wt.extract_case(nl)
    assert case["run_total_hours"] == 24.0


def test_extract_case_isc21_shape_run_total_hours_is_half_hour(tmp_path):
    """run_minutes=30 alone is nonzero -> 0.5h exactly, even though start->end spans a DIFFERENT
    15h (2016-08-29 09:00 -> 2016-08-30 00:00) that must NOT win, and even though the old
    `run_hours + run_days*24` formula dropped run_minutes entirely and returned 0."""
    nl = wt.parse_namelist(_write(tmp_path, "namelist.input", ISC21_NML))
    case = wt.extract_case(nl)
    assert case["run_total_hours"] == 0.5


def test_extract_case_run_minutes_and_seconds_no_longer_dropped(tmp_path):
    """A case whose length is expressed only in run_minutes/run_seconds (no run_days/run_hours)
    must not come out silently as 0 -- the old formula only ever read run_hours/run_days."""
    text = """\
&time_control
 run_days = 0,
 run_hours = 0,
 run_minutes = 45,
 run_seconds = 30,
 start_year = 2021,
 start_month = 1,
 start_day = 1,
 /
&domains
 e_we = 101,
 e_sn = 101,
 /
&physics
 /
"""
    nl = wt.parse_namelist(_write(tmp_path, "namelist.input", text))
    case = wt.extract_case(nl)
    assert case["run_total_hours"] == 45 / 60.0 + 30 / 3600.0


def test_extract_case_no_time_info_falls_back_to_zero_not_raise(tmp_path):
    """extract_case must never raise merely because run length can't be resolved (it is called
    from contexts, e.g. pure domain-geometry queries, that never need it) -- falls back to the
    same 0.0 'unset' sentinel it always used, rather than propagating wtune_runlength's
    ValueError."""
    text = """\
&domains
 e_we = 101,
 e_sn = 101,
 /
&physics
 /
"""
    nl = wt.parse_namelist(_write(tmp_path, "namelist.input", text))
    case = wt.extract_case(nl)
    assert case["run_total_hours"] == 0.0
