"""Small pure helpers shared by every layer: time formatting, run-id parsing, slot counts and
the NOISE-clustered ranking."""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Callable

from ._deps import pred_targets

ANCHOR_END_ID = "ANCHOREND"      # the ANCHOR re-measured last in a ranking job (drift probe)


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()            # a Record field that is absent from its JSON form


def is_set(value: Any) -> bool:
    return value is not UNSET


def if_set(value: Any, default: Any = None) -> Any:
    """`value`, or `default` when it is UNSET (dict.get semantics for an optional field)."""
    return default if value is UNSET else value


def _plain(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


class Record:
    """Base of the report dataclasses: to_dict() gives the JSON form in field order, without the
    UNSET fields and with nested records converted."""

    def to_dict(self) -> dict:
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)
                if getattr(self, f.name) is not UNSET}

# RUN_ID="${CASE}_${cid}_r${rep}_${J}" (strict_campaign.sbatch:962); duplicated from
# predictor/decomp_search.py so parsing a run id does not import numpy/sklearn/catboost
_RUN_ID_RE = re.compile(r"^(?P<case>[^_]+)_(?P<cid>.+)_r(?P<rep>\d+)_(?P<job>\d+)$")


def _fmt_hms(seconds: float | None) -> str:
    if seconds is None or seconds == float("inf"):
        return "none"
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, s) if h else "%d:%02d" % (m, s)


def _fmt_sbatch_time(seconds: float) -> str:
    m = int(math.ceil(seconds / 60.0))
    return "%02d:%02d:00" % (m // 60, m % 60)


def parse_run_id(run_id: str | None) -> dict | None:
    """{case, cid, rep, job} of a ledger run_id, or None when it does not parse."""
    m = _RUN_ID_RE.match(run_id or "")
    return m.groupdict() if m else None


def compute_slots(max_nodes: int, nodes: int) -> int:
    """Parallel jobs of `nodes` nodes that fit in `max_nodes`: floor(max_nodes/N), at least 1."""
    return max(1, int(max_nodes) // int(nodes))


def would_exceed_budget(elapsed_s: float, estimated_job_s: float, budget_s: float) -> bool:
    return (elapsed_s + estimated_job_s) > budget_s


def chunk_round_robin(items: list, n: int) -> list[list]:
    """`items` dealt round-robin into up to `n` lists, empty ones dropped; item 0 (the anchor)
    always lands in chunk 0."""
    n = max(1, int(n))
    chunks: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return [c for c in chunks if c]


def _log_key(value: float) -> float:
    """log(value); -inf for a non-positive value so sorting never raises."""
    return math.log(value) if value > 0 else float("-inf")


def _cluster_rank(entries: list[dict], log_key: Callable[[dict], float], anchor_cid: Any,
                  order_index: dict) -> list[dict]:
    """`entries` by log_key ascending, grouped into clusters whose spread from the cluster head
    is < predictor/targets.py NOISE (0.5%); inside a cluster the anchor first, then order_index.
    Comparing against the head keeps the order transitive and input-order independent (the old
    pairwise comparator was not)."""
    ordered = sorted(entries, key=lambda e: (log_key(e), order_index.get(e["cid"], 0)))
    out = []
    i, n = 0, len(ordered)
    while i < n:
        head_key = log_key(ordered[i])
        j = i + 1
        while j < n and (log_key(ordered[j]) - head_key) < pred_targets.NOISE:
            j += 1
        cluster = ordered[i:j]
        cluster.sort(key=lambda e: (e["cid"] != anchor_cid, order_index.get(e["cid"], 0)))
        out.extend(cluster)
        i = j
    return out


def _parse_slurm_elapsed(s: str | None) -> float | None:
    """Seconds of a Slurm Elapsed value ('[D-][HH:]MM:SS'); None when empty or unparsable."""
    if not s:
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    if len(parts) != 3:
        return None
    h, m, sec = parts
    return float(days * 86400 + h * 3600 + m * 60 + sec)


def _notes_kv(notes: str | None) -> dict:
    """{key: value} of a ledger row's space-separated key=value notes tokens."""
    out = {}
    for tok in (notes or "").split():
        k, sep, v = tok.partition("=")
        if sep:
            out[k] = v
    return out
