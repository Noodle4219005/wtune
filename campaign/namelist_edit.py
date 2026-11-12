#!/usr/bin/env python3
"""The campaign's edit of a case namelist for one run: run length, history policy,
decomposition, case overrides, ADIOS2 I/O, restart hand-off, physics template.

Pure text edits, apart from the WTUNE_* environment the ADIOS2 step reads (namelist_adios2).
Stdlib only, Python 3.6.
"""
import datetime
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from namelist_adios2 import _apply_adios2
from namelist_text import (_delete_key_lines, _force_gsfcgce_radar_ref, _insert_after_domains,
                           _insert_after_section_header, _replace_list_line,
                           _set_or_insert_time_control_scalar, _set_scalar)
from physics_templates import Template, apply_physics_template

HISTORY_NEVER_MINUTES = 1000000  # ~694 days: longer than any campaign run, fits int32 as seconds

# Diagnostics that make WRF abort on a domain whose history_interval is 0
# (module_check_a_mundo.F:2105, module_diagnostics_driver.F:264; jobs 1078290, 1078305).
HISTORY_REQUIRING_DIAGNOSTICS = ("nwp_diagnostics", "hailcast_opt", "afwa_diag_opt")

_TILE_KEYS = ("numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy")
_START_FIELDS = ("year", "month", "day", "hour", "minute", "second")


# --------------------------------------------------------------------------- history
def _history_key_and_value(text: str, minutes: int) -> Tuple[str, str, int]:
    """(interval key, begin key, `minutes` in that key's unit) for the variant this namelist
    uses: history_interval / _m (minutes) or _s (seconds), registry.io_boilerplate:57-62."""
    if re.search(r"(?m)^[ \t]*history_interval_m[ \t]*=", text):
        return "history_interval_m", "history_begin_m", minutes
    if re.search(r"(?m)^[ \t]*history_interval_s[ \t]*=", text):
        return "history_interval_s", "history_begin_s", minutes * 60
    return "history_interval", "history_begin", minutes


def _history_requiring_diagnostics_on(text: str) -> bool:
    """True when any HISTORY_REQUIRING_DIAGNOSTICS key is nonzero on any domain."""
    for key in HISTORY_REQUIRING_DIAGNOSTICS:
        for m in re.finditer(r"(?mi)^(?![ \t]*!)[^\n]*?(?<![A-Za-z0-9_])%s[ \t]*=([^\n!]*)" % key, text):
            if any(int(v) != 0 for v in re.findall(r"-?\d+", m.group(1))):
                return True
    return False


def _set_history_interval(text: str, history_minutes: int, max_dom: int,
                          nest_history_on: bool = False) -> str:
    """Set the namelist's history-interval variant: d01 = history_minutes, nests the same
    only with nest_history_on (nest history dominates wall time; diffwrf compares d01).

    One domain: edited in place, keeping the template's spacing. A nest that must not be 0
    (HISTORY_REQUIRING_DIAGNOSTICS) gets an interval that never elapses instead.
    """
    key, _begin_key, minutes = _history_key_and_value(text, history_minutes)
    if max_dom <= 1:
        return _set_scalar(text, key, minutes)
    if nest_history_on:
        values = [minutes] * max_dom
    else:
        nest_value = (_history_key_and_value(text, HISTORY_NEVER_MINUTES)[2]
                      if _history_requiring_diagnostics_on(text) else 0)
        values = [minutes] + [nest_value] * (max_dom - 1)
    return _replace_list_line(text, key, values)


def _begin_values(value: int, max_dom: int, nest_history_on: bool) -> List[int]:
    """history_begin per domain: d01 `value`, the nests too only with nest history on."""
    if max_dom > 1:
        return [value] * max_dom if nest_history_on else [value] + [0] * (max_dom - 1)
    return [value]


def _set_history_begin(text: str, minutes: int, max_dom: int, nest_history_on: bool) -> str:
    _interval_key, begin_key, value = _history_key_and_value(text, minutes)
    return _replace_list_line(text, begin_key, _begin_values(value, max_dom, nest_history_on))


def _history_off(text: str, run_minutes: int, max_dom: int, nest_history_on: bool,
                 history_minutes: Optional[int]) -> str:
    """No frame at all (timing runs): interval 0, or -- with diagnostics that forbid 0 --
    interval == begin == never, which also skips the t=0 frame."""
    if not _history_requiring_diagnostics_on(text):
        return _set_history_interval(text, 0, max_dom, nest_history_on=False)
    text = _set_history_interval(text, HISTORY_NEVER_MINUTES, max_dom, nest_history_on=True)
    _interval_key, begin_key, value = _history_key_and_value(text, HISTORY_NEVER_MINUTES)
    return _replace_list_line(text, begin_key, [value] * max_dom)


def _history_interval(text: str, run_minutes: int, max_dom: int, nest_history_on: bool,
                      history_minutes: Optional[int]) -> str:
    """d01 frames every history_minutes from t=0 (begin 0 on every domain)."""
    if history_minutes is None:
        raise ValueError("history_minutes is required when history_mode='interval'")
    text = _set_history_interval(text, history_minutes, max_dom, nest_history_on)
    return _set_history_begin(text, 0, max_dom, True)


def _history_end_d01(text: str, run_minutes: int, max_dom: int, nest_history_on: bool,
                     history_minutes: Optional[int]) -> str:
    """Exactly one d01 frame, at the end: begin == interval == run_minutes, so the only
    boundary that elapses is the last step (set_timekeeping.f90:2745-2757)."""
    text = _set_history_interval(text, run_minutes, max_dom, nest_history_on)
    return _set_history_begin(text, run_minutes, max_dom, nest_history_on)


def _history_mid_end_d01(text: str, run_minutes: int, max_dom: int, nest_history_on: bool,
                         history_minutes: Optional[int]) -> str:
    """Two d01 frames, at run_minutes//2 and at the end: the smallest set fidelity_gate can
    label, since a RESTARTED run writes no frame at its own start (scc job 2194)."""
    first = int(run_minutes) // 2
    if first < 1 or int(run_minutes) - first < 1:
        raise ValueError("mid_end_d01 needs run_minutes >= 2, got %r" % (run_minutes,))
    text = _set_history_interval(text, int(run_minutes) - first, max_dom, nest_history_on)
    return _set_history_begin(text, first, max_dom, nest_history_on)


# history_mode -> policy(text, run_minutes, max_dom, nest_history_on, history_minutes)
_HISTORY_MODES: Dict[str, Callable[[str, int, int, bool, Optional[int]], str]] = {
    "off": _history_off,
    "interval": _history_interval,
    "end_d01": _history_end_d01,
    "mid_end_d01": _history_mid_end_d01,
}


def _set_history_mode(text: str, history_mode: str, run_minutes: int, max_dom: int,
                      nest_history_on: bool = False, history_minutes: Optional[int] = None) -> str:
    """Apply a history policy: 'off', 'interval', 'end_d01' or 'mid_end_d01' (see _HISTORY_MODES)."""
    policy = _HISTORY_MODES.get(history_mode)
    if policy is None:
        raise ValueError("history_mode must be 'off', 'interval', 'end_d01' or 'mid_end_d01', got %r"
                         % (history_mode,))
    return _aux_streams_off(policy(text, run_minutes, max_dom, nest_history_on, history_minutes),
                            max_dom)


_AUX_INTERVAL = re.compile(r"(?mi)^[ \t]*auxhist(\d+)_interval(_[dhms])?[ \t]*=([^\n!]*)")
_AUX_NEVER = {"": HISTORY_NEVER_MINUTES, "_m": HISTORY_NEVER_MINUTES,
              "_s": HISTORY_NEVER_MINUTES * 60, "_h": HISTORY_NEVER_MINUTES // 60,
              "_d": HISTORY_NEVER_MINUTES // 1440}


def _aux_streams_off(text: str, max_dom: int) -> str:
    """Every auxhistN stream with a nonzero interval begins only after HISTORY_NEVER_MINUTES.
    A case's own aux output is not timed I/O: isc21's auxhist24 cost ~160 s per launch."""
    for n, unit, values in _AUX_INTERVAL.findall(text):
        if any(int(v) for v in re.findall(r"-?\d+", values)):
            unit = unit.lower()
            text = _replace_list_line(text, "auxhist%s_begin%s" % (n, unit),
                                      [_AUX_NEVER[unit]] * max_dom)
    return text


# --------------------------------------------------------------------------- the other steps
def _set_run_length(text: str, run_minutes: int, run_seconds: Optional[int]) -> str:
    """run_days = run_hours = 0, run_minutes, and run_seconds = 0 or the given tail.

    WRF ADDS the four run_* keys: a template's own run_seconds would lengthen the run
    (decision 0055); a given tail makes the window STEPS*dt exactly (POINT_FUSED).
    """
    text = _set_scalar(text, "run_days", 0)
    text = _set_scalar(text, "run_hours", 0)
    text = _set_scalar(text, "run_minutes", run_minutes)
    text = _set_scalar(text, "run_seconds", 0)
    if run_seconds:
        text = _set_or_insert_time_control_scalar(text, "run_seconds", int(run_seconds))
    return text


def _set_decomposition(text: str, nproc_x: int, nproc_y: int, numtiles: int, max_dom: int,
                       tile_extra: List[Tuple[str, int]]) -> str:
    """One fresh nproc_x / nproc_y / numtiles [/ tile geometry] line under &domains."""
    for key in ("nproc_x", "nproc_y", "numtiles"):
        text = _delete_key_lines(text, key)
    for key, _value in tile_extra:
        text = _delete_key_lines(text, key)
    return _insert_after_domains(text, nproc_x, nproc_y, numtiles, max_dom, extra=tile_extra)


def _apply_case_overrides(text: str, delete_keys: Optional[Iterable[str]],
                          overrides: Optional[Mapping[str, Any]]) -> str:
    """Delete the case's delete_keys (any case, any group), then force its scalar overrides."""
    if delete_keys:
        for key in delete_keys:
            text = _delete_key_lines(text, key, ignore_case=True)
    if overrides:
        for key, value in overrides.items():
            text = _set_scalar(text, key, value)
    return text


def _apply_restart_from(text: str, restart_time: datetime.datetime, max_dom: int) -> str:
    """Start EVERY active domain at restart_time with restart = .true. (a ranking probe).

    override_restart_timers = .true.: else WRF keeps the restart interval stored in the
    restart file and writes a restart inside the measured window (jobs 1371/1372).
    """
    for field in _START_FIELDS:
        text = _replace_list_line(text, "start_" + field, [getattr(restart_time, field)] * max_dom)
    text = _set_scalar(text, "restart", ".true.")
    if re.search(r"(?im)^\s*override_restart_timers\s*=", text):
        return _set_scalar(text, "override_restart_timers", ".true.")
    return _insert_after_section_header(text, "&time_control",
                                        " override_restart_timers = .true.,\n")


def _apply_restart_options(text: str, restart_from: Optional[datetime.datetime],
                           write_restart_interval: Optional[int],
                           restart_io_form: Optional[int], max_dom: int) -> str:
    """The restart hand-off of a ranking probe, applied after ADIOS2 so its defaults lose."""
    if restart_from is not None:
        text = _apply_restart_from(text, restart_from, max_dom)
    if write_restart_interval is not None:
        text = _set_or_insert_time_control_scalar(text, "restart_interval", write_restart_interval)
    if restart_io_form is not None:
        text = _set_or_insert_time_control_scalar(text, "io_form_restart", restart_io_form)
    return text


def _apply_named_template(text: str, template_id: str, templates: Mapping[str, Template],
                          max_dom: int, case_id: Optional[str]) -> str:
    template = templates.get(template_id)
    if template is None:
        raise ValueError("unknown physics template %r" % (template_id,))
    text, _summary = apply_physics_template(text, template, max_dom, case_id=case_id)
    return text


def edit_namelist(text: str, nproc_x: int, nproc_y: int, numtiles: int, run_minutes: int,
                  max_dom: int = 1, overrides: Optional[Mapping[str, Any]] = None,
                  delete_keys: Optional[Iterable[str]] = None, nest_history_on: bool = False,
                  history_mode: str = "off", adios2: bool = True, ranks: Optional[int] = None,
                  history_minutes: Optional[int] = None, physics_template: Optional[str] = None,
                  physics_case: Optional[str] = None,
                  numtiles_x: Optional[int] = None, numtiles_y: Optional[int] = None,
                  tile_sz_x: Optional[int] = None, tile_sz_y: Optional[int] = None,
                  tile_strategy: Optional[int] = None,
                  restart_from: Optional[datetime.datetime] = None,
                  restart_io_form: Optional[int] = None,
                  write_restart_interval: Optional[int] = None,
                  run_seconds: Optional[int] = None,
                  physics_templates: Optional[Mapping[str, Template]] = None) -> str:
    """Apply every campaign edit, in the order below; an option left at None is not written.

    ADIOS2 (ranks required) runs after the overrides so a stale io_form override cannot
    revert it; the restart options and the physics template (an id of physics_templates)
    come last. ValueError when an edit cannot be applied.
    """
    tile_values = (numtiles_x, numtiles_y, tile_sz_x, tile_sz_y, tile_strategy)
    tile_extra = [(k, v) for k, v in zip(_TILE_KEYS, tile_values) if v is not None]
    text = _set_run_length(text, run_minutes, run_seconds)
    text = _set_history_mode(text, history_mode, run_minutes, max_dom, nest_history_on,
                             history_minutes=history_minutes)
    if not adios2:
        text = _set_scalar(text, "io_form_history", 2)     # legacy: nothing else about I/O
    text = _set_decomposition(text, nproc_x, nproc_y, numtiles, max_dom, tile_extra)
    text = _apply_case_overrides(text, delete_keys, overrides)
    if adios2:
        # start: per-record BP inputs need a record at the time the run really starts
        text = _apply_adios2(text, ranks, max_dom, start=restart_from)
    text = _apply_restart_options(text, restart_from, write_restart_interval, restart_io_form,
                                  max_dom)
    if physics_template:
        text = _apply_named_template(text, physics_template, physics_templates or {}, max_dom,
                                     physics_case)
    return _force_gsfcgce_radar_ref(text)
