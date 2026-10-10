#!/usr/bin/env python3
"""wtune/resume.py -- which failed segment jobs `wtune run --resume` may resume, and from where.

A restart time T is valid when every domain's wrfrst_d<NN>_<T> is complete on disk and the rsl
shows its write finished; `wtune run --sidecar` records the same list while the job runs."""
from __future__ import annotations

import datetime as dt
import glob
import os
import re
from typing import Callable, Iterable, Optional

import wtune.watch

WRF_FMT = "%Y-%m-%d_%H:%M:%S"
INFRA_STATES = ("NODE_FAIL", "TIMEOUT", "PREEMPTED", "BOOT_FAIL")
# a dead wrf.exe still lets the batch script exit 0, so sacct says COMPLETED, not FAILED
QUIET_STATES = ("FAILED", "COMPLETED")
MODEL_NEEDLES = ("cfl", "segmentation fault", "forrtl", "fatal called")
_RST_RE = re.compile(r"^wrfrst_d(\d\d)_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)(\.bp)?$")


def rsl_mentions(rundir: str, needles: Iterable[str]) -> bool:
    """True when any rsl.* line of any rank contains one of `needles` (lower case)."""
    for path in glob.glob(os.path.join(rundir, "rsl.*")):
        try:
            with open(path, "r", errors="replace") as f:
                for ln in f:
                    low = ln.lower()
                    if any(n in low for n in needles):
                        return True
        except OSError:
            continue
    return False


def classify_failure(state: str, model_error: bool) -> str:
    """Failure class of a segment job that did not finish: 'infrastructure' (resumable),
    'model', 'cancelled' or 'other'."""
    if state.startswith("CANCELLED"):
        return "cancelled"
    if state in INFRA_STATES:
        return "infrastructure"
    if model_error:
        return "model"
    return "infrastructure" if state in QUIET_STATES else "other"


def restart_complete(path: str, is_bp: Callable[[str], bool]) -> bool:
    """A written restart: a BP directory, or a non-empty netCDF file (io_form_restart 2)."""
    return is_bp(path) or (os.path.isfile(path) and os.path.getsize(path) > 0)


def restart_sets_on_disk(seg_dir: str, max_dom: int, is_bp: Callable[[str], bool]) -> list[str]:
    """Sorted times whose wrfrst_d<NN>_<T> is complete for every domain 1..max_dom."""
    try:
        names = os.listdir(seg_dir)
    except OSError:
        return []
    doms = {}
    for name in names:
        m = _RST_RE.match(name)
        path = os.path.join(seg_dir, name)
        # a link is the restart this attempt READS (the previous attempt's), never its own
        if m and not os.path.islink(path) and restart_complete(path, is_bp):
            doms.setdefault(m.group(2), set()).add(int(m.group(1)))
    want = set(range(1, max_dom + 1))
    return sorted(t for t, got in doms.items() if want <= got)


def restart_writes_finished(rsl_text: str, max_dom: int) -> list[dt.datetime]:
    """Times whose restart write finished for every domain: WRF prints one 'Timing for Writing
    restart for domain N' per domain after T's d01 Timing line and before the next one."""
    steps, trailing = wtune.watch.parse_d01_steps(rsl_text)
    want = set(range(1, max_dom + 1))
    out, prev = [], None
    for st in steps + [{"t": None, "writes": trailing}]:
        doms = set(dom for name, dom, _s in st["writes"] if name == "restart")
        if prev is not None and want <= doms:
            out.append(prev)
        prev = st["t"]
    return out


def valid_restart_times(seg_dir: str, max_dom: int, is_bp: Callable[[str], bool],
                        skip: Iterable[str] = ()) -> list[str]:
    """WRF_FMT times of seg_dir's valid restarts (module docstring), those in `skip` left out.
    The rsl is the rank-0 file wtune_watch.read_rsl picks (either can be truncated)."""
    on_disk = [t for t in restart_sets_on_disk(seg_dir, max_dom, is_bp) if t not in skip]
    chosen = wtune.watch.read_rsl(seg_dir)[0] if on_disk else None
    if chosen is None:
        return []
    done = set(t.strftime(WRF_FMT) for t in restart_writes_finished(chosen["text"], max_dom))
    return [t for t in on_disk if t in done]


def last_d01_time(seg_dir: str) -> Optional[dt.datetime]:
    """The last d01 'Timing for main' time in seg_dir's rsl, or None."""
    chosen = wtune.watch.read_rsl(seg_dir)[0]
    times = [t for dom, t, _s in (chosen["timing"] if chosen else []) if dom == 1 and t]
    return times[-1] if times else None


def resume_point(steps: list, a: int, b: int, unit: int,
                 last_step) -> tuple[Optional[int], Optional[str]]:
    """(step, None): the latest valid restart step usable to resume an attempt over (a, b] --
    after its start, before its end, on the `unit` grid (input record times) -- or (None, why).
    `steps` are the valid restarts' d01 step indices, None for one off the step grid."""
    if not steps:
        return None, "no valid restart (complete for every domain and finished in the rsl)"
    # the nests' history frame at T is written after the restart: the run must have got past T
    ok = [n for n in steps if n is not None and a < n < b and n % unit == 0
          and last_step is not None and n < last_step]
    if not ok:
        return None, ("none of the %d valid restart(s) is after the attempt start, on an input "
                      "record time and behind the last d01 step" % len(steps))
    return max(ok), None
