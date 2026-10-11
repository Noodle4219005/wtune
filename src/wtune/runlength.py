#!/usr/bin/env python3
"""wtune/runlength.py -- the ONE place that knows how WRF decides a case's run length.

Before this module existed, "how long does this WRF case run" was implemented FOUR times
(tools/temporal_validation.py:read_run_length_hours, wtune/predict.py:get_run_seconds,
wtune/cli.py:extract_case, and tools/metdata.py/tools/grib_state.py inheriting extract_case's
answer), and only the first of those read WRF's real semantics. Verified against two real
cluster namelists (see tools/test_temporal_validation.py's NML_LAFE_SHAPE/NML_ISC21_SHAPE,
transcribed from the actual templates): lafe (all four run_* keys 0, start=2017-08-23 06:00 ->
end=2017-08-24 06:00) runs 24.0h; isc21 (run_minutes=30, every other run_* key 0, start/end date
span a DIFFERENT 15h that must NOT win) runs 0.5h. wtune.cli's extract_case and wtune/predict.py's
get_run_seconds both used to return/raise on 0 for lafe (dropping the date-span case entirely),
and the run_minutes/run_seconds-dropping variant in extract_case separately turned isc21's
30-minute run into a nonsense 0.

WRF's own rule (&time_control, confirmed against campaign/configs.py's edit_namelist, which
writes run_days/run_hours/run_minutes/run_seconds verbatim and nothing else): if ANY of
run_days/run_hours/run_minutes/run_seconds is nonzero, THAT is the run length, full stop --
start_date/end_date are ignored even if they disagree. Only when ALL FOUR are zero (or absent)
does WRF fall back to computing the run length from the first domain's start_date -> end_date
span. WRF has no notion of a genuinely zero-length run, so a namelist that gives neither a
positive run_* sum nor a positive date span is not a runnable case -- callers must not treat 0
as "unset" silently; this module raises instead.

This module is intentionally dependency-free (stdlib only) and takes ALREADY-EXTRACTED scalars,
never a namelist object: the three callers each own a different in-memory namelist
representation (temporal_validation.py parses raw text field-by-field via campaign/configs.py's
_find_namelist_key; wtune.cli's parse_namelist returns {group: {key: raw_string}}; wtune/predict.py's
parse_namelist returns {group: {key: [scalar, ...]}} and is list/per-domain aware) and unifying
those representations is out of scope here -- only the resulting six scalars are.
"""
from __future__ import annotations


def resolve_run_seconds(days, hours, minutes, seconds, start_dt, end_dt):
    """The run length WRF will ACTUALLY use, in seconds. NEVER returns 0.

    days*86400 + hours*3600 + minutes*60 + seconds when that sum is > 0 (any run_* key absent
    counts as 0, matching WRF's own default) -- start_dt/end_dt are not even consulted in this
    case, so a start/end span that disagrees (isc21's 15h span vs. its real 30-minute
    run_minutes) never wins.

    Otherwise, when start_dt and end_dt are both given and end_dt is strictly after start_dt,
    (end_dt - start_dt) in seconds -- WRF's own documented fallback once every run_days/
    run_hours/run_minutes/run_seconds is 0 (this is the lafe shape: 24h expressed only via
    start_date/end_date).

    Raises ValueError naming both exhausted sources when NEITHER a positive run_*-key sum NOR a
    positive start_dt/end_dt span is available -- every caller here already treats a namelist
    with no run length WRF could use as an error, not as a genuine zero-length run (WRF has no
    such thing).
    """
    total = (days or 0) * 86400 + (hours or 0) * 3600 + (minutes or 0) * 60 + (seconds or 0)
    if total > 0:
        return float(total)
    if start_dt is not None and end_dt is not None:
        span = (end_dt - start_dt).total_seconds()
        if span > 0:
            return float(span)
    raise ValueError(
        "resolve_run_seconds: run_days/run_hours/run_minutes/run_seconds are all 0 (or absent) "
        "AND start_dt/end_dt do not establish a positive span -- neither source yields a run "
        "length WRF could actually use"
    )
