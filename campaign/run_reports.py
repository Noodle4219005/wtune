#!/usr/bin/env python3
"""Pure parsers of what a campaign run leaves behind: per-step timings (rsl.out), the disk
quota (mmlsquota) and fidelity_gate.py's JSON. Their outputs go into strict_campaign.sbatch's
`read` and the ledger, so the formats are fixed. Stdlib only, Python 3.6.
"""
import datetime
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

FIDELITY_TOKEN_NA = "NA"

_TIMING_D01_RE = re.compile(
    r"Timing for main: time (\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)\s+on domain\s+1:\s+([0-9.]+) elapsed")


# --------------------------------------------------------------------------- step timings
def step_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Statistics of one run's d01 step wall times (s); values[0] (init + radiation) is dropped.

    median: of the steps <= 3x the plain median (radiation steps excluded); mean_all: of ALL
    steps 2..n (the ledger's s_per_step_mean); n_measured / n_excluded / n_all are the counts.
    median and mean_all are None without a step 2.
    """
    n_excluded = 1 if values else 0
    rest = values[1:]
    n_all = len(rest)
    if not rest:
        return dict(median=None, mean_all=None, n_measured=0,
                    n_excluded=n_excluded, n_all=n_all)
    median0 = statistics.median(rest)
    kept = [v for v in rest if v <= 3 * median0]
    n_excluded += len(rest) - len(kept)
    mean_all = statistics.mean(rest)
    median = statistics.median(kept) if kept else None
    return dict(median=median, mean_all=mean_all, n_measured=len(kept),
                n_excluded=n_excluded, n_all=n_all)


def post_spinup_steps(rsl_text: str, spin_seconds: float, dt: float) -> List[float]:
    """POINT_FUSED: the d01 step times after the first spin_seconds of model time, in order.

    rsl.out stamps each step with the model time reached, so line j has run
    (stamp_j - stamp_1) + dt of model time. Feed the result to step_stats() unchanged.
    """
    fmt = "%Y-%m-%d_%H:%M:%S"
    first = None
    out = []
    for m in _TIMING_D01_RE.finditer(rsl_text):
        ts = datetime.datetime.strptime(m.group(1), fmt)
        if first is None:
            first = ts
        elapsed = (ts - first).total_seconds() + float(dt)
        if elapsed > float(spin_seconds):
            out.append(float(m.group(2)))
    return out


# --------------------------------------------------------------------------- disk quota
def quota_free_gib(mmlsquota_text: str, filesystem: str = "home1") -> Optional[float]:
    """Free GiB below the SOFT quota in `mmlsquota -u $USER <filesystem>` output, or None.

    None: no '<filesystem> USR <used_kb> <quota_kb> <limit_kb>' row, or quota 0 (not enforced
    at user level: wiki mmlsquota-u-says-no-limit-while-the-fileset-is-99-percent-full).
    """
    pattern = re.compile(r"^\s*(\S+)\s+USR\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+(\d+))?")
    for line in mmlsquota_text.splitlines():
        m = pattern.match(line)
        if m is None:
            continue
        fs, used_kb, quota_kb, _limit_kb, in_doubt_kb = m.groups()
        if fs != filesystem:
            continue
        quota_kb = int(quota_kb)
        if quota_kb <= 0:
            return None
        used_kb = int(used_kb) + int(in_doubt_kb or 0)        # open 16 MiB-block files sit in in_doubt
        return (quota_kb - used_kb) / (1024.0 * 1024.0)    # KB = 1024-byte GPFS blocks
    return None


# --------------------------------------------------------------------------- fidelity JSON
def fidelity_max_over_fields(payload: Dict[str, Any], key: str) -> str:
    """The MEASURED max of `key` over per_field_summary_at_last_time's fields, '%.6g';
    FIDELITY_TOKEN_NA when no field has a number (a bool such as floor_zero is not one)."""
    summary = payload.get("per_field_summary_at_last_time") or {}
    values = []
    for field in summary.values():
        if not isinstance(field, dict):
            continue
        v = field.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        values.append(v)
    return ("%.6g" % max(values)) if values else FIDELITY_TOKEN_NA


def fidelity_notes_tokens(payload: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """(label, nd_max, floor_ratio_max, deciding): four shell words for the sbatch's
    `read -r FID_LABEL FID_ND FID_FLOOR FID_DECIDING` (NA when absent, spaces -> '_')."""
    label = str(payload.get("label") or FIDELITY_TOKEN_NA)
    deciding = str(payload.get("deciding") or FIDELITY_TOKEN_NA).replace(" ", "_")
    return (label, fidelity_max_over_fields(payload, "nd"),
            fidelity_max_over_fields(payload, "floor_ratio"), deciding)


def describe_field(value: Any) -> str:
    """A TSV field for bash `IFS=$'\\t' read`, which collapses empty fields: '-' for empty
    (it shifted every ledger column left in campaigns 1077168-1077173)."""
    s = "" if value is None else str(value)
    return s if s != "" else "-"
