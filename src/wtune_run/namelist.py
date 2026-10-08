"""Segment namelist (pure): the case's namelist.input text with the segment's time window,
layout, restart keys and I/O streams edited in; every other key is left as the case has it.
tools/coldstart_sample and campaign/configs do the text editing."""
from __future__ import annotations

import datetime as dt
import difflib
import re
from typing import Optional

from ._deps import coldstart, sweep
from .common import ADIOS2_IO_FORM, RunError, _TIME_PARTS
from .layout import Config


def _time_values(t: dt.datetime) -> tuple:
    return (t.year, "%02d" % t.month, "%02d" % t.day, "%02d" % t.hour, "%02d" % t.minute,
            "%02d" % t.second)


def _set_one(txt: str, key: str, value, group: str = "time_control") -> str:
    """One-value key: campaign/configs._set_scalar in place when the key is assigned exactly once
    (keeps the line layout -- coldstart set_list joins a comma-less 'key = v' line with the next
    line), else coldstart set_list (absent -> inserted into &group; packed/odd lines)."""
    raw = len(re.findall(r"(?<![A-Za-z0-9_])%s\s*=" % re.escape(key), txt))
    if raw == 1 and coldstart.count_key(txt, key) == 1:
        return sweep.campaign_configs._set_scalar(txt, key, value)
    return coldstart.set_list(txt, key, value, 1, new_key_group=group)


def _segment_seconds(seg_start: dt.datetime, seg_end: dt.datetime) -> int:
    length_s = (seg_end - seg_start).total_seconds()
    if length_s <= 0 or length_s != int(length_s):
        raise RunError("segment length %r s is not a positive whole number of seconds" % length_s)
    return int(length_s)


def _set_window_and_layout(text: str, seg_start: dt.datetime, seg_end: dt.datetime,
                           config: Config) -> str:
    """start_*/end_* of EVERY domain (yagi truth 1038: a nest with a stale end_* stopped
    early), run_* = the segment length, nproc_x/nproc_y/numtiles of the layout."""
    ndom = int(coldstart.get_first(text, "max_dom", "1"))
    length_s = _segment_seconds(seg_start, seg_end)
    txt = text
    for prefix, t in (("start_", seg_start), ("end_", seg_end)):
        for part, v in zip(_TIME_PARTS, _time_values(t)):
            txt = coldstart.set_list(txt, prefix + part, v, ndom)
    for key, v in (("run_days", 0), ("run_hours", 0), ("run_minutes", 0),
                   ("run_seconds", length_s)):
        txt = _set_one(txt, key, v)
    for key, v in (("nproc_x", config.nx), ("nproc_y", config.ny), ("numtiles", config.numtiles)):
        txt = _set_one(txt, key, v, "domains")
    return txt


def _set_restart_keys(text: str, txt: str, restart_interval_min: Optional[int],
                      restart_io_form: int, restarted: bool) -> str:
    """restart_interval (None: the case's own keys stay), io_form_restart, restart, and
    override_restart_timers forced .false. (never .true.: see plan.py)."""
    if restart_interval_min is not None:
        rst_min = int(restart_interval_min)
        txt = _set_one(txt, "restart_interval", rst_min)
        for suffix, v in (("_d", 0), ("_h", 0), ("_s", 0), ("_m", rst_min)):
            if coldstart.count_key(text, "restart_interval" + suffix):
                txt = _set_one(txt, "restart_interval" + suffix, v)
    txt = _set_one(txt, "io_form_restart", int(restart_io_form))
    if restarted:
        txt = _set_one(txt, "restart", ".true.")
    elif coldstart.count_key(text, "restart"):
        txt = _set_one(txt, "restart", ".false.")
    if coldstart.count_key(text, "override_restart_timers"):
        txt = _set_one(txt, "override_restart_timers", ".false.")
    return txt


def _set_io_keys(text: str, txt: str, config: Config, io_streams: Optional[list],
                 per_record_edits: Optional[dict]) -> tuple[str, list[str]]:
    """ADIOS2 streams (io_form 14, .bp innames), the per-record edits over them, and
    adios2_numaggregators = the rank count -> (text, the keys that must now be present)."""
    cc = sweep.campaign_configs
    io_keys = []
    if io_streams is not None:
        io_keys = ["io_form_history", "adios2_numaggregators"]
        txt = _set_one(txt, "io_form_history", ADIOS2_IO_FORM)
        for st in io_streams:
            for key, v, group in cc.adios2_stream_edits(st):
                txt = _set_one(txt, key, v, group)
                io_keys.append(key)
    for group, kv in sorted((per_record_edits or {}).items()):
        for key, v in sorted(kv.items()):
            # the whole value list is replaced (_set_one would keep a stale per-domain tail)
            # and the group is created when absent
            txt = cc._set_namelist_value(txt, key, v, group)
            io_keys.append(key)
    if io_streams is not None or coldstart.count_key(text, "adios2_numaggregators"):
        txt = _set_one(txt, "adios2_numaggregators", config.ranks, "namelist_adios2")
    return txt, io_keys


def render_segment_namelist(text: str, seg_start: dt.datetime, seg_end: dt.datetime, k: int,
                            config: Config, restart_interval_min: Optional[int],
                            restart_io_form: int = 14, time_step: Optional[int] = None,
                            io_streams: Optional[list] = None,
                            per_record_edits: Optional[dict] = None,
                            resumed: bool = False) -> str:
    """The case's namelist.input text -> segment k's. restart_interval_min None keeps the
    case's restart keys; time_step (s) is the CFL-rescue step; resumed: segment 0 restarted
    mid-way. io_streams (configs.adios2_input_streams entries; None = --io netcdf) switch
    those streams to ADIOS2; per_record_edits (wtune_bp.per_record_namelist_edits) go last."""
    config = Config.coerce(config)
    restarted = k > 0 or resumed
    txt = _set_window_and_layout(text, seg_start, seg_end, config)
    txt = _set_restart_keys(text, txt, restart_interval_min, restart_io_form, restarted)
    txt, io_keys = _set_io_keys(text, txt, config, io_streams, per_record_edits)
    required = (["%s_%s" % (p, q) for p in ("start", "end") for q in _TIME_PARTS]
                + ["run_days", "run_hours", "run_minutes", "run_seconds", "nproc_x", "nproc_y",
                   "numtiles", "io_form_restart"] + io_keys
                + (["restart_interval"] if restart_interval_min is not None else [])
                + (["restart"] if restarted else []))
    if time_step is not None:
        txt = _set_one(txt, "time_step", int(time_step), "domains")
        if coldstart.count_key(text, "time_step_fract_num"):
            txt = _set_one(txt, "time_step_fract_num", 0, "domains")
        required.append("time_step")
    coldstart.assert_no_key_drift(text, txt, "segment %d namelist" % k, required)
    return txt


def namelist_diff_summary(old: str, new: str) -> list[str]:
    """Changed lines only ('- old' / '+ new')."""
    return [ln for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
            if not ln.startswith(("---", "+++", "@@"))]
