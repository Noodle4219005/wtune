#!/usr/bin/env python3
"""STRICT-configuration campaign generator (predictor phase B).

Generates, for each runnable WRF case in cases.json, the list of STRICT (compliance
model, decision 0050 -- see wtune_compliance.py) run configurations from the fixed
template table below, ranks a case's legal MPI decompositions to resolve each
template's abstract shape ('best'/'alt1'/'alt2') into an actual (nproc_x, nproc_y),
dedupes configs that collapse to the same physical run, and edits namelists for the
sbatch runner. Stdlib only, Python 3.6 compatible (no f-strings avoided is not
required -- f-strings are fine on 3.6 -- but no dataclasses, no walrus, no
`from __future__ import annotations`).

IMPORTANT ENVIRONMENT FACT (verified 2026-09-14): wtune_decomp.py, the repo's "own
decomposition scorer" this generator is asked to prefer, opens with
`from __future__ import annotations`, a Python >=3.7 feature. This host's `python3`
is 3.6.8, and this campaign is required to run under python3 (3.6) on both the
login node and (via the same module environment) the compute nodes. So
`import wtune_decomp` raises SyntaxError at import time here -- not a missing file,
a language-version mismatch. `_load_decomp()` below tries the import, catches the
failure, and every shape lookup on this host silently and correctly falls through
to the documented fallback scorer:

    score(nx, ny) = |log((e_we/nx) / (e_sn/ny))| + 0.1 * (nx > ny)

lower is better; legality is `ceil(e_we/nx) >= 10 and ceil(e_sn/ny) >= 10` (the
`MIN_PATCH` used by wtune_decomp.py too, but see the docstring of `legal_shapes`
for a documented difference in which patch dimension the two check). If this code
is ever run under Python >=3.7, `_load_decomp` will succeed and the generator will
prefer wtune_decomp.enumerate_configs's own `score` field instead, per-group, with
no code change needed here.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys

CAMPAIGN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CAMPAIGN_DIR)
# CASES_JSON env var (optional) points every consumer at an alternate case file with the same
# {"cases": {...}} schema, e.g. a not-yet-merged fragment for a smoke run; unset = cases.json.
CASES_JSON_PATH = os.environ.get("CASES_JSON") or os.path.join(CAMPAIGN_DIR, "cases.json")
PHYSICS_TEMPLATES_JSON_PATH = os.path.join(CAMPAIGN_DIR, "physics_templates.json")
DEFAULT_SCREEN_DIR = "/home/u5727520/wrf-bench/wscore/physics"

MIN_PATCH = 10
TOTAL_CORES = 448  # 4 nodes x 112 cores, ct448

# Deterministic case order (matches the task table); used by --all and cmd_order so
# output does not depend on JSON/dict iteration order.
# 2026-09-14: deep_conv_ot, arctic_sbl, edouard_landfall and lowell_cat4 were removed from the campaign at
# the user's request (their inputs stay on disk). The old restart-only "maria" (5 km) case is removed
# (disk data untouched); the testcase-A cases CP-1 isc21 (3-domain), DOI-4 lafe (d01+d02) and OB-3
# maria1km (2-domain cold start) are added now that their inputs are prepared.
CASE_ORDER = [
    "karina", "conus25", "conus12", "isc21", "maria1km", "lafe",
    "severe2023", "matthew", "c34", "zhengzhou", "elevconv", "marshall", "bigchill",
]

# Fixed config templates. ranks*omp must equal TOTAL_CORES for every row (asserted
# by test_configs.py). blocktime is the literal KMP_BLOCKTIME value to export, or
# the string 'default' meaning: do not export KMP_BLOCKTIME at all.
TEMPLATES = [
    dict(id="C01", ranks=448, omp=1, shape="best", numtiles=1, dist="cyclic", blocktime="0"),
    dict(id="C02", ranks=448, omp=1, shape="best", numtiles=8, dist="cyclic", blocktime="0"),
    dict(id="C03", ranks=448, omp=1, shape="alt1", numtiles=8, dist="cyclic", blocktime="0"),
    dict(id="C04", ranks=448, omp=1, shape="best", numtiles=8, dist="block", blocktime="0"),
    dict(id="C05", ranks=224, omp=2, shape="best", numtiles=13, dist="cyclic", blocktime="0"),
    dict(id="C06", ranks=224, omp=2, shape="best", numtiles=12, dist="cyclic", blocktime="0"),
    dict(id="C07", ranks=224, omp=2, shape="best", numtiles=14, dist="cyclic", blocktime="0"),
    dict(id="C08", ranks=224, omp=2, shape="best", numtiles=2, dist="cyclic", blocktime="0"),
    dict(id="C09", ranks=224, omp=2, shape="alt1", numtiles=12, dist="cyclic", blocktime="0"),
    dict(id="C10", ranks=224, omp=2, shape="alt2", numtiles=12, dist="cyclic", blocktime="0"),
    dict(id="C11", ranks=224, omp=2, shape="best", numtiles=12, dist="block", blocktime="0"),
    dict(id="C12", ranks=224, omp=2, shape="best", numtiles=12, dist="cyclic", blocktime="default"),
    dict(id="C13", ranks=112, omp=4, shape="best", numtiles=16, dist="cyclic", blocktime="0"),
    dict(id="C14", ranks=112, omp=4, shape="best", numtiles=12, dist="cyclic", blocktime="0"),
    dict(id="C15", ranks=112, omp=4, shape="alt1", numtiles=16, dist="cyclic", blocktime="0"),
    dict(id="C16", ranks=112, omp=4, shape="best", numtiles=13, dist="cyclic", blocktime="0"),
]

_SHAPE_INDEX = {"best": 0, "alt1": 1, "alt2": 2}

# Cost-estimate guesses (all labelled GUESS in output), keyed by (e_we, e_sn, e_vert) of
# the case's d01 (its scalar e_we/e_sn/e_vert fields, not any per-domain e_we_list).
# s_per_step_guess (s), init_io_guess (s). From the task spec, not measured, except:
# - maria1km's ORIGINAL s_per_step was scaled up from the task's 2.0 s starting point --
#   see cases.json's maria1km "notes" for the reasoning (d02 workload vs. karina). Now
#   superseded by an actual smoke measurement (see below).
# - isc21 and maria1km s_per_step/init_io below are measured on a single smoke run each
#   (jobs 1074515-1074517, 2026-09-14), WITH nest history still ON (the smoke jobs predate
#   this campaign's nest-history-off default) -- init_io is relabelled here for what it
#   would be with nest IO off (per-run wall minus steps*s_per_step, nested wrfout/aux-stream
#   writes excluded); s_per_step uses the ALL-d01-steps mean (radiation steps included),
#   matching the campaign ledger's new s_per_step_mean semantics (see step_stats()) since
#   that is what actually drives total per-run wall time. Still GUESS-labelled in cmd_all's
#   output because the lead has not yet re-measured with nest IO actually off.
COST_GUESS = {
    (901, 901, 45): (0.30, 70),    # karina, edouard_landfall, lowell_cat4
    (901, 901, 80): (0.50, 110),   # deep_conv_ot, arctic_sbl
    (1501, 1201, 50): (0.60, 120),  # conus25
    (425, 300, 50): (0.08, 30),    # conus12
    (793, 853, 51): (15.0, 230),   # isc21: measured, all-steps mean ~15 s/step (~6.4 s/step
                                    # excluding radiation steps), init+IO ~230 s w/ nest IO off
    (900, 900, 50): (5.8, 250),    # maria1km: measured, ~5.8 s/step, init+IO ~250 s w/ nest IO off
    (1000, 1000, 100): (4.0, 150),  # lafe (d01+d02, 500 m nest @ ratio 5) -- GUESS, unchanged
}


def load_cases(path=CASES_JSON_PATH):
    with open(path) as f:
        data = json.load(f)
    return data["cases"]


CASES = load_cases()


def load_physics_templates(path=PHYSICS_TEMPLATES_JSON_PATH):
    with open(path) as f:
        data = json.load(f)
    return data["templates"]


PHYSICS_TEMPLATES = load_physics_templates()
# "P00" < "P01" < ... sorts in numeric order lexicographically as long as ids stay
# zero-padded to 2 digits (physics_templates.json's own convention).
PHYSICS_TEMPLATE_ORDER = sorted(PHYSICS_TEMPLATES)


# ---------------------------------------------------------------------------
# Decomposition legality and shape ranking
# ---------------------------------------------------------------------------

def factor_pairs(n):
    return [(i, n // i) for i in range(1, n + 1) if n % i == 0]


def patch_dims(extent, nproc):
    """ceil(extent / nproc), per the task's own legality formula. NOTE: this is
    NOT the same quantity wtune_decomp.py's own patch_dims uses for its legality
    check (it uses the floor/worst-case patch, wmin); see the module docstring
    and the campaign report for the documented discrepancy."""
    q, rem = divmod(extent, nproc)
    return q + 1 if rem else q


def wrf_patch_legal(extent, nproc, min_patch=MIN_PATCH):
    """WRF's startup patch-size gate: integer division extent/nproc >= min_patch
    (share/module_check_a_mundo.F:3014-3029)."""
    return extent // nproc >= min_patch


def legal_shapes(e_we, e_sn, ranks, min_patch=MIN_PATCH):
    """All (nx, ny, patch_x, patch_y) with nx*ny == ranks passing the MIN_PATCH gate.

    e_we/e_sn may each be a single domain's extent (int) or a list of per-domain
    extents (nested case). When lists are given, a shape is legal only if EVERY
    domain d (paired by index, e_we[d] with e_sn[d]) passes ceil(extent/n) >=
    min_patch -- a shape can be legal on d01 alone yet illegal on a smaller
    nested domain, and this must reject that shape, not just score d01.
    patch_x/patch_y in the returned tuples are always d01's (index 0) patch
    dims, since ranking/reporting is d01-based regardless of domain count.

    2026-09-15: the gate itself now uses WRF's own rule, Fortran integer division
    e_we(i)/nproc_x >= 10 (share/module_check_a_mundo.F:3014-3029), via wrf_patch_legal.
    The earlier ceil-based gate was looser: e.g. conus12 e_sn=300 with nproc_y=32 gives
    ceil 10 (accepted) but floor 9, which WRF aborts on. None of the measured C01-C16
    shapes was affected (all pass the floor rule, predictor/decomp_space.py check)."""
    we_list = list(e_we) if isinstance(e_we, (list, tuple)) else [e_we]
    sn_list = list(e_sn) if isinstance(e_sn, (list, tuple)) else [e_sn]
    out = []
    for nx, ny in factor_pairs(ranks):
        px = patch_dims(we_list[0], nx)
        py = patch_dims(sn_list[0], ny)
        if all(wrf_patch_legal(we_d, nx, min_patch) and wrf_patch_legal(sn_d, ny, min_patch)
               for we_d, sn_d in zip(we_list, sn_list)):
            out.append((nx, ny, px, py))
    return out


def _fallback_score(e_we, e_sn, nx, ny):
    wx = e_we / float(nx)
    wy = e_sn / float(ny)
    return abs(math.log(wx / wy)) + (0.1 if nx > ny else 0.0)


_DECOMP = None
_DECOMP_TRIED = False


def _load_decomp():
    """Try to import wtune_decomp from the repo root. Returns the module or None.
    Caches the result (including failure) for the life of the process."""
    global _DECOMP, _DECOMP_TRIED
    if _DECOMP_TRIED:
        return _DECOMP
    _DECOMP_TRIED = True
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        import wtune_decomp
        _DECOMP = wtune_decomp
    except Exception as e:
        sys.stderr.write(
            "NOTE: wtune_decomp unavailable (%s: %s); using fallback shape scorer\n"
            % (type(e).__name__, e)
        )
        _DECOMP = None
    return _DECOMP


def ranked_shapes(e_we, e_sn, e_vert, ranks, omp, nodes=4):
    """Distinct legal (nx, ny) pairs for `ranks` MPI ranks at `omp` threads/rank,
    best first. Returns (shapes, method) where method is 'wtune_decomp' or 'fallback'."""
    decomp = _load_decomp()
    if decomp is not None:
        try:
            cores = ranks * omp
            cfgs = decomp.enumerate_configs(e_we, e_sn, e_vert, cores, nodes)
            group = [c for c in cfgs if c["nranks"] == ranks and c["threads"] == omp]
            if group:
                group.sort(key=lambda c: c["score"])
                seen = []
                for c in group:
                    pair = (c["nx"], c["ny"])
                    if pair not in seen:
                        seen.append(pair)
                return seen, "wtune_decomp"
        except Exception as e:
            sys.stderr.write(
                "NOTE: wtune_decomp.enumerate_configs failed (%s: %s); using fallback\n"
                % (type(e).__name__, e)
            )
    shapes = legal_shapes(e_we, e_sn, ranks)
    shapes.sort(key=lambda s: _fallback_score(e_we, e_sn, s[0], s[1]))
    return [(nx, ny) for nx, ny, px, py in shapes], "fallback"


def _active_domain_extents(case):
    """(e_we_list, e_sn_list) for the case's active domains only (index 0..max_dom-1).
    Single-domain cases (no e_we_list/e_sn_list in cases.json) fall back to the
    scalar d01 fields, unchanged from the pre-multi-domain behaviour."""
    max_dom = case.get("max_dom", 1)
    we_list = case.get("e_we_list", [case["e_we"]])[:max_dom]
    sn_list = case.get("e_sn_list", [case["e_sn"]])[:max_dom]
    return we_list, sn_list


def resolve_config(case, template):
    """Resolve one template against one case. Returns (config_dict, None) or
    (None, reason_str) when the template has no legal shape for this case.

    Shapes are ranked using d01 (case["e_we"]/case["e_sn"]), via wtune_decomp
    when available or the fallback scorer otherwise -- neither of which knows
    about nesting. The ranked list is then filtered to shapes that are ALSO
    legal (ceil(extent/n) >= MIN_PATCH) on every active domain, so a shape
    that only wtune_decomp/d01 would have accepted but a smaller nested
    domain would reject is dropped here, uniformly for both scoring methods."""
    ranks, omp = template["ranks"], template["omp"]
    if ranks * omp != TOTAL_CORES:
        return None, "ranks(%d)*omp(%d) != %d" % (ranks, omp, TOTAL_CORES)
    shapes, method = ranked_shapes(case["e_we"], case["e_sn"], case["e_vert"], ranks, omp)
    we_list, sn_list = _active_domain_extents(case)
    if len(we_list) > 1:
        legal_pairs = set((nx, ny) for nx, ny, px, py in legal_shapes(we_list, sn_list, ranks))
        shapes = [s for s in shapes if s in legal_pairs]
    idx = _SHAPE_INDEX[template["shape"]]
    if idx >= len(shapes):
        return None, (
            "no legal '%s' shape (index %d) at ranks=%d: only %d legal shape(s) "
            "found via %s (max_dom=%d)" % (template["shape"], idx, ranks, len(shapes),
                                            method, case.get("max_dom", 1))
        )
    nx, ny = shapes[idx]
    px = patch_dims(case["e_we"], nx)
    py = patch_dims(case["e_sn"], ny)
    return dict(
        id=template["id"], ranks=ranks, omp=omp, nx=nx, ny=ny,
        numtiles=template["numtiles"], dist=template["dist"], blocktime=template["blocktime"],
        patch_x=px, patch_y=py, shape=template["shape"], shape_method=method,
    ), None


def dedupe(resolved):
    """Collapse configs identical in (ranks, omp, nx, ny, numtiles, dist, blocktime),
    keeping the first template id encountered. Returns (kept_list, aliases) where
    aliases maps a kept id -> list of dropped ids that were identical to it."""
    seen = {}
    kept = []
    aliases = {}
    for cfg in resolved:
        key = (cfg["ranks"], cfg["omp"], cfg["nx"], cfg["ny"], cfg["numtiles"],
               cfg["dist"], cfg["blocktime"])
        if key in seen:
            aliases.setdefault(seen[key], []).append(cfg["id"])
        else:
            seen[key] = cfg["id"]
            kept.append(cfg)
    return kept, aliases


def resolve_case(case):
    """Resolve + dedupe every template for one case. Returns (kept, skipped, aliases)."""
    resolved = []
    skipped = []
    for t in TEMPLATES:
        cfg, err = resolve_config(case, t)
        if err:
            skipped.append((t["id"], err))
        else:
            resolved.append(cfg)
    kept, aliases = dedupe(resolved)
    return kept, skipped, aliases


def build_order(config_ids, reps):
    """Palindromic run order. reps=2 -> forward then reversed (a palindrome).
    reps>2 -> alternate forward/reversed so drift is still split evenly."""
    reps = int(reps)
    if reps <= 0:
        return []
    forward = list(config_ids)
    backward = list(reversed(config_ids))
    order = []
    for i in range(reps):
        order.extend(forward if i % 2 == 0 else backward)
    return order


# ---------------------------------------------------------------------------
# Namelist editing
# ---------------------------------------------------------------------------

def _set_scalar(text, key, value):
    """Set namelist scalar `key` to `value`, preserving the existing '='
    spacing style. No-op (returns text unchanged) if the key is not found."""
    pattern = re.compile(r"\b%s\b([ \t]*=[ \t]*)[^,\n]*" % re.escape(key))
    new_text, n = pattern.subn(lambda m: "%s%s%s" % (key, m.group(1), value), text, count=1)
    return new_text


def _delete_key_lines(text, key, ignore_case=False):
    """Delete every whole line whose first token (after leading whitespace) is
    `key =`. Used for nproc_x/nproc_y/numtiles so a stale single- or
    multi-domain assignment is fully removed before re-inserting a fresh one,
    and (with ignore_case=True) for the campaign's `delete_keys` feature that
    strips namelist keys this WRF build no longer has (e.g. lafe's
    bl_mynn_tkebudget/opt_run -- see cases.json's lafe notes)."""
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(r"(?m)^[ \t]*%s[ \t]*=.*\n?" % re.escape(key), flags)
    return pattern.sub("", text)


def _insert_after_domains(text, nproc_x, nproc_y, numtiles, max_dom=1, extra=None):
    """numtiles, nproc_x and nproc_y are single namelist-global values (Registry dimension 1), so they are
    always written once, whatever max_dom is. Writing numtiles as a per-domain list ("numtiles=13, 13, 13,")
    makes the Fortran namelist read fail: smoke jobs 1074506-1074508 all aborted with
    "ERROR reading namelist domains". max_dom is accepted for interface compatibility and ignored here.

    `extra` is an optional ordered list of (key, value) pairs (WTune D1 decomposition-space work:
    numtiles_x/numtiles_y/tile_sz_x/tile_sz_y/tile_strategy, Registry.EM_COMMON, all dimension 1
    like numtiles/nproc_x/nproc_y) appended onto the SAME &domains line, once each -- callers
    (edit_namelist) only pass a key here when its CLI/param value was actually given, so a default
    (extra=None or []) call produces byte-identical output to before this parameter existed."""
    idx = text.find("&domains")
    if idx == -1:
        raise ValueError("no &domains section found in namelist")
    eol = text.find("\n", idx)
    if eol == -1:
        eol = len(text)
    parts = ["nproc_x=%d" % nproc_x, "nproc_y=%d" % nproc_y, "numtiles=%d" % numtiles]
    for key, value in (extra or []):
        parts.append("%s=%d" % (key, value))
    new_line = " " + ", ".join(parts) + ",\n"
    return text[: eol + 1] + new_line + text[eol + 1 :]


def _replace_list_line(text, key, values):
    """Delete every whole-line assignment of `key` and insert a single fresh
    one, as a comma list, right after the &time_control header. Used for the
    history-interval family, which IS a per-domain (dimension max_domains)
    namelist var in the Registry, unlike numtiles/nproc_x/nproc_y above."""
    text = _delete_key_lines(text, key)
    idx = text.find("&time_control")
    if idx == -1:
        raise ValueError("no &time_control section found in namelist")
    eol = text.find("\n", idx)
    if eol == -1:
        eol = len(text)
    value_str = ", ".join(str(v) for v in values)
    new_line = " %s = %s,\n" % (key, value_str)
    return text[: eol + 1] + new_line + text[eol + 1 :]


def _history_key_and_value(text, minutes):
    """Pick whichever history-interval/history-begin variant pair this
    namelist actually uses, and convert `minutes` to that variant's unit.
    Three variants exist in this campaign's real templates: history_interval
    (MINUTES, karina/conus/maria1km/lafe), history_interval_m (MINUTES,
    conus25/conus12) and history_interval_s (SECONDS, isc21). history_begin
    pairs with history_interval (both minutes -- set_timekeeping.f90:2751
    folds history_begin into history_begin_m when history_begin_m is 0),
    history_begin_m pairs with history_interval_m (minutes), history_begin_s
    pairs with history_interval_s (seconds); the whole family is declared in
    registry.io_boilerplate:57-62.

    Returns (interval_key, begin_key, value)."""
    if re.search(r"(?m)^[ \t]*history_interval_m[ \t]*=", text):
        return "history_interval_m", "history_begin_m", minutes
    if re.search(r"(?m)^[ \t]*history_interval_s[ \t]*=", text):
        return "history_interval_s", "history_begin_s", minutes * 60
    return "history_interval", "history_begin", minutes


def _set_history_interval(text, history_minutes, max_dom, nest_history_on=False):
    """Set whichever history-interval variant this namelist actually uses to
    `history_minutes` for every active domain (see _history_key_and_value).

    max_dom==1 (the pre-multi-domain default) keeps using in-place _set_scalar
    so a single-domain namelist's own "key=value" spacing style is preserved
    byte-for-byte, exactly as before this campaign supported nesting. Only
    max_dom>1 switches to the delete+reinsert _replace_list_line, which always
    writes " key = v1, v2, ...,\n" regardless of the template's original style,
    because there is no single existing line whose spacing to preserve once
    the value list's length itself must change.

    nest_history_on controls d02..dN: False (default) writes 0 for every
    nested domain -- diffwrf only ever compares wrfout_d01_* frames, and
    nested-domain history/IO dominates per-run wall time (isc21/maria1km
    smoke jobs), so nest history is off by default for the STRICT compute
    campaign. nest_history_on=True (cases.json "nest_history": true, or CLI
    --nest-history-on) restores the old behaviour of writing the same value
    to every domain."""
    key, _begin_key, minutes = _history_key_and_value(text, history_minutes)
    if max_dom > 1:
        if nest_history_on:
            values = [minutes] * max_dom
        else:
            values = [minutes] + [0] * (max_dom - 1)
        return _replace_list_line(text, key, values)
    return _set_scalar(text, key, minutes)


HISTORY_NEVER_MINUTES = 1000000  # ~694 days: longer than any campaign run, fits int32 as seconds


# Diagnostics that make WRF abort when a domain's history_interval is 0:
# nwp_diagnostics (share/module_check_a_mundo.F:2105-2111, bigchill smoke 1078290) and
# hailcast_opt (phys/module_diagnostics_driver.F:264, debug job 1078305); afwa_diag_opt uses the
# same history-alarm path.
HISTORY_REQUIRING_DIAGNOSTICS = ("nwp_diagnostics", "hailcast_opt", "afwa_diag_opt")


def _history_requiring_diagnostics_on(text):
    """True when a non-comment assignment of any HISTORY_REQUIRING_DIAGNOSTICS key has a nonzero
    value for any domain (e.g. `hailcast_opt = 1, 1,`)."""
    for key in HISTORY_REQUIRING_DIAGNOSTICS:
        for m in re.finditer(r"(?mi)^(?![ \t]*!)[^\n]*?(?<![A-Za-z0-9_])%s[ \t]*=([^\n!]*)" % key, text):
            if any(int(v) != 0 for v in re.findall(r"-?\d+", m.group(1))):
                return True
    return False


def _set_history_mode(text, history_mode, run_minutes, max_dom, nest_history_on=False,
                       history_minutes=None):
    """Apply one of the STRICT/PHYSICS-GRAY campaign history policies (2026-09-14
    user decision: timing runs write no history at all; correctness is checked
    in a separate small batch; 2026-09-14 PHYSICS-GRAY data campaign decision
    0051 adds 'interval' for fidelity replicates).

    'off' (timing runs): every history-interval variant is 0 for every domain
    -- no wrfout_d0* frame is ever written, so a timing run's wall time never
    includes history I/O.

    'interval' (PHYSICS-GRAY fidelity replicate 1): d01 writes a frame every
    history_minutes, starting at t=0 (history_begin*=0 for every domain --
    begin=0 always triggers the very first interval boundary, so the t=0
    frame is included, unlike 'end_d01' below which deliberately skips it).
    Nested domains still get 0 (off), same nest_history_on-gated rule as
    'off'/'end_d01' -- diffwrf/fidelity_gate.py only ever compare
    wrfout_d01_* frames. Requires history_minutes.

    'end_d01' (correctness runs): d01 writes exactly one frame, at the end of
    the run, via history_begin==history_interval==run_minutes (nested domains
    always get 0 -- diffwrf only ever compares wrfout_d01_* frames). This
    relies on history_begin/_m/_s, which DO exist in this build
    (registry.io_boilerplate:57-62, confirmed 2026-09-14 -- NOT in
    Registry.EM_COMMON, where the task asked to look first) and fold into a
    WRFU_TimeIntervalSet(history_begin) delay before the first write
    (share/set_timekeeping.f90:2745-2757): with begin==interval==run_minutes
    the only interval boundary that ever elapses is the run's last step, so
    the usual t=0 frame is skipped for free -- no post-run frame deletion
    needed, unlike the history_begin-absent fallback the task also described.
    nest_history_on (default False) still governs nested domains the same
    way it always has, in case a future case opts back into nest history."""
    if history_mode == "off":
        if _history_requiring_diagnostics_on(text):
            # WRF aborts when these diagnostics are on and a domain's history_interval is 0 (see
            # HISTORY_REQUIRING_DIAGNOSTICS). Keep the diagnostics (not ours to change) and still
            # write nothing: every active domain gets an interval that never elapses with
            # history_begin == interval, which also skips the t=0 frame (same begin mechanism as
            # 'end_d01'). Namelists without these diagnostics are unchanged.
            text = _set_history_interval(text, HISTORY_NEVER_MINUTES, max_dom, nest_history_on=True)
            _interval_key, begin_key, value = _history_key_and_value(text, HISTORY_NEVER_MINUTES)
            return _replace_list_line(text, begin_key, [value] * max_dom)
        return _set_history_interval(text, 0, max_dom, nest_history_on=False)
    if history_mode == "interval":
        if history_minutes is None:
            raise ValueError("history_minutes is required when history_mode='interval'")
        text = _set_history_interval(text, history_minutes, max_dom, nest_history_on)
        _interval_key, begin_key, _value = _history_key_and_value(text, history_minutes)
        begin_values = [0] * max_dom if max_dom > 1 else [0]
        return _replace_list_line(text, begin_key, begin_values)
    if history_mode == "end_d01":
        text = _set_history_interval(text, run_minutes, max_dom, nest_history_on)
        _interval_key, begin_key, value = _history_key_and_value(text, run_minutes)
        if max_dom > 1:
            begin_values = [value] * max_dom if nest_history_on else [value] + [0] * (max_dom - 1)
        else:
            begin_values = [value]
        return _replace_list_line(text, begin_key, begin_values)
    raise ValueError("history_mode must be 'off', 'interval' or 'end_d01', got %r" % (history_mode,))


def _set_or_insert_time_control_scalar(text, key, value):
    """Like _set_scalar, but INSERTS a fresh line right after &time_control's
    header when this namelist has no existing line for `key` at all, instead
    of silently no-op'ing. Needed because real WRF namelist.input templates
    are not consistent about which io_form_* keys they bother to write when
    the value would just be WRF's own default: isc21's real template
    (/home/u5727520/wrf-bench/atlas/isc21/extract/WRF_ISC21_3_DOMAIN/namelist.input)
    has io_form_history but no io_form_restart/io_form_input/io_form_boundary
    lines at all -- plain _set_scalar would leave those three at WRF's
    default while the campaign feeds isc21 .bp (ADIOS2) input files, which is
    a silent readback failure waiting to happen, not a harmless no-op."""
    # The key may sit mid-line next to other keys (karina: "io_form_history=2, io_form_restart=2,
    # io_form_input=2, io_form_boundary=2," on ONE line). A start-of-line-only check missed it, inserted a
    # second line with 14, and WRF kept the later "=2" (smoke job 1074653: "error opening wrfinput_d01.bp").
    # So detect the key anywhere outside comments and rewrite every occurrence.
    # One match per line (a key appears at most once per line), so a plain sub rewrites every occurrence.
    occ = re.compile(r"(?m)^(?![ \t]*!)(.*?)(?<![A-Za-z0-9_])%s([ \t]*=[ \t]*)[^,\n]*" % re.escape(key))
    if occ.search(text):
        return occ.sub(lambda m: "%s%s%s%s" % (m.group(1), key, m.group(2), value), text)
    idx = text.find("&time_control")
    if idx == -1:
        raise ValueError("no &time_control section found in namelist")
    eol = text.find("\n", idx)
    if eol == -1:
        eol = len(text)
    return text[: eol + 1] + " %s=%s,\n" % (key, value) + text[eol + 1 :]


def _apply_adios2(text, ranks, max_dom):
    """Force ADIOS2 I/O for input/boundary/history/restart (2026-09-14 user
    decision: all WRF I/O in this campaign uses ADIOS2, io_form 14) and
    disable restart writes (restart_interval set far beyond any campaign run
    length so nothing is ever written -- no campaign run uses WRF restart).
    Recipe proven in regress_bp.sbatch / job 1072441
    (/home/u5727520/wrf-stack/wtune/plov/regress_bp.sbatch:47-50).

    adios2_numaggregators MUST equal the MPI rank count: WRF's default (0 =
    one aggregator per node) combined with --distribution=cyclic produced BP
    output that cannot be read back ("Read past end of file ... data.3").

    Called LAST in edit_namelist, after `overrides`, so a case override that
    sets an io_form (e.g. lafe's old io_form_input=2 for pnetcdf) can never
    revert ADIOS2 -- see cases.json's lafe namelist_overrides notes.

    Uses _set_or_insert_time_control_scalar (not plain _set_scalar) for the
    io_form_* keys and restart_interval so a template that omits one of them
    still gets ADIOS2 forced, rather than silently keeping WRF's default."""
    if ranks is None:
        raise ValueError("ranks is required when adios2=True (for adios2_numaggregators)")
    for key in ("io_form_input", "io_form_boundary", "io_form_history", "io_form_restart"):
        text = _set_or_insert_time_control_scalar(text, key, 14)
    text = _set_or_insert_time_control_scalar(text, "restart_interval", 1000000)
    text = _delete_key_lines(text, "input_inname")
    text = _delete_key_lines(text, "bdy_inname")
    idx = text.find("&time_control")
    if idx == -1:
        raise ValueError("no &time_control section found in namelist")
    eol = text.find("\n", idx)
    if eol == -1:
        eol = len(text)
    new_line = ' input_inname="wrfinput_d<domain>.bp", bdy_inname="wrfbdy_d<domain>.bp",\n'
    text = text[: eol + 1] + new_line + text[eol + 1 :]
    text = re.sub(r"(?ms)^&namelist_adios2\b.*?^/[ \t]*\n?", "", text)
    text = text.rstrip("\n") + "\n&namelist_adios2\n adios2_numaggregators=%d,\n/\n" % ranks
    # Quilting (WRF I/O server ranks) is always off: with nio_tasks_per_group=32, nio_groups=4 (the LAFE
    # template) 128 of 224 ranks become I/O servers, nproc_x*nproc_y no longer matches the compute ranks,
    # and WRF hangs after "Quilting with 4 groups of 32 I/O tasks" (smoke job 1074562, 1500 s timeout).
    # ADIOS2 output goes through WRF's own writers, so the whole quilt group is replaced.
    text = re.sub(r"(?msi)^[ \t]*&namelist_quilt\b.*?^[ \t]*/[ \t]*\n?", "", text)
    text = text.rstrip("\n") + "\n&namelist_quilt\n nio_tasks_per_group=0,\n nio_groups=1,\n/\n"
    return text


# ---------------------------------------------------------------------------
# PHYSICS-GRAY physics-template editing (decision 0051, physics_templates.json)
# ---------------------------------------------------------------------------

def _find_namelist_key(text, key):
    """Locate the first (case-sensitive, non-comment) namelist assignment of
    `key`, ANYWHERE on its line -- not just when it is the first token. Real
    templates in this campaign jam several physics keys onto one line (e.g.
    karina's prep_dir namelist.input: " mp_physics=8, ra_lw_physics=4,
    ra_sw_physics=4, radt=15,", or its &domains "time_step=15, max_dom=1,
    e_we=901, ..."), so anchoring the key to column 1 would silently miss a
    key that is genuinely present. Uses the same lazy-prefix-plus-negative-
    lookbehind recipe as _set_or_insert_time_control_scalar (proven against
    exactly this shape by TestAdios2MultiKeyLine) rather than a line-start
    anchor. Returns None if `key` has no assignment anywhere, else a dict:
      value_start, value_end -- span in `text` covering EXACTLY this key's own
        comma-separated value list (safe to replace with a fresh value list;
        text[:value_start] is "...key = " and text[value_end:] is whatever
        follows -- a same-line trailing key's own assignment jammed onto the
        same line, an inline '!' comment, or the rest of the file starting
        with '\\n' -- all preserved untouched)
      values -- list of this key's own value tokens (stripped; a trailing
        empty token from a trailing comma is dropped)
    Stops consuming comma-separated tokens as soon as one contains '=' (that
    token belongs to a DIFFERENT key's assignment on the same line, not a
    value of this one) -- this is only safe for keys whose own values are
    never quoted strings (a quoted string could itself contain ',' or '=');
    true of every physics_templates.json key used so far (mp_physics,
    radt, swint_opt, time_step -- all numeric)."""
    pattern = re.compile(
        r"(?m)^(?![ \t]*!)(?:.*?)(?<![A-Za-z0-9_])(%s[ \t]*=[ \t]*)" % re.escape(key)
    )
    m = pattern.search(text)
    if not m:
        return None
    key_start = m.start(1)
    value_start = m.end()
    line_end = text.find("\n", value_start)
    if line_end == -1:
        line_end = len(text)
    rest = text[value_start:line_end]
    bang = rest.find("!")
    scan = rest if bang == -1 else rest[:bang]
    values = []
    consumed = 0
    pos = 0
    n = len(scan)
    while True:
        comma = scan.find(",", pos)
        tok = scan[pos:comma] if comma != -1 else scan[pos:]
        if "=" in tok:
            break
        stripped = tok.strip()
        if stripped:
            values.append(stripped)
        consumed = (comma + 1) if comma != -1 else n
        if comma == -1:
            break
        pos = comma + 1
    return dict(key_start=key_start, value_start=value_start,
                value_end=value_start + consumed, values=values)


def _insert_after_section_header(text, section, line):
    """Insert `line` (expected to already end with '\\n') right after the
    first line containing `section` (e.g. '&physics'). Raises ValueError if
    the section is not present at all."""
    idx = text.find(section)
    if idx == -1:
        raise ValueError("no %s section found in namelist" % (section,))
    eol = text.find("\n", idx)
    if eol == -1:
        eol = len(text)
    return text[: eol + 1] + line + text[eol + 1 :]


def apply_physics_edit(text, edit, max_dom):
    """Apply one physics_templates.json 'edits' entry (a {'op':...} dict) to
    `text`. Returns (new_text, old_active, new_active): old_active/new_active
    are the active-domain value lists before/after for 'set'/'mul' (None for
    'delete'). Raises ValueError (with a message identifying the offending
    key/op) when the edit cannot be applied -- see physics_templates.json's
    top-level '_comment' for the exact semantics of each op, the -1
    physics_suite 'mul' restriction, and the single-value insert-when-absent
    convention for 'set'."""
    op = edit["op"]
    key = edit["key"]
    section = edit.get("section", "&physics")
    if op == "delete":
        found = _find_namelist_key(text, key)
        if found is None:
            raise ValueError("physics template: key %r not found for 'delete'" % (key,))
        # Span removal (not _delete_key_lines' whole-line removal): safe when this
        # key is jammed on a line with others (karina-style), since it only removes
        # this key's own "key=value," text, never a neighbour's assignment. If that
        # leaves the line containing nothing but whitespace, drop the now-empty
        # line too (cosmetic only -- a namelist blank line is harmless either way).
        new_text = text[:found["key_start"]] + text[found["value_end"]:]
        line_start = new_text.rfind("\n", 0, found["key_start"]) + 1
        line_end = new_text.find("\n", found["key_start"])
        if line_end == -1:
            line_end = len(new_text)
        if new_text[line_start:line_end].strip() == "":
            new_text = new_text[:line_start] + new_text[line_end + 1:]
        return new_text, None, None
    found = _find_namelist_key(text, key)
    if found is None:
        if op != "set":
            raise ValueError("physics template: key %r not found for op %r" % (key, op))
        new_active = [str(edit["value"])]
        line = " %s = %s,\n" % (key, ", ".join(new_active))
        return _insert_after_section_header(text, section, line), [], new_active
    values = found["values"]
    n_active = min(len(values), max_dom) if max_dom else len(values)
    if n_active == 0:
        raise ValueError("physics template: key %r has no value(s) to edit" % (key,))
    active_old = values[:n_active]
    trailing = values[n_active:]
    if op == "set":
        active_new = [str(edit["value"])] * n_active
    elif op == "mul":
        factor = edit["factor"]
        round_int = edit.get("round") == "int"
        mn, mx = edit.get("min"), edit.get("max")
        active_new = []
        for v in active_old:
            try:
                native = float(v)
            except ValueError:
                raise ValueError("physics template: key %r value %r is not numeric" % (key, v))
            if native == -1:
                raise ValueError(
                    "physics template: cannot 'mul' key %r whose native value is -1 "
                    "(physics_suite default) -- use 'set' to override a suite key "
                    "instead" % (key,)
                )
            new_v = native * factor
            new_v = int(round(new_v)) if round_int else new_v
            if mn is not None:
                new_v = max(new_v, mn)
            if mx is not None:
                new_v = min(new_v, mx)
            active_new.append(str(new_v))
    else:
        raise ValueError("unknown physics edit op %r" % (op,))
    new_values_str = ", ".join(active_new + trailing) + ","
    new_text = text[:found["value_start"]] + new_values_str + text[found["value_end"]:]
    return new_text, active_old, active_new


def apply_physics_template(text, template, max_dom, case_id=None):
    """Apply every edit in template['edits'] to `text`, in order, followed by
    template['extra_edits'][case_id] when case_id is given (per-case paired
    edits, e.g. lafe's aer_opt=0 next to a scheme swap). Returns
    (new_text, edits_summary) where edits_summary is a list of
    {'key','op','old','new'} dicts (old/new are None for 'delete'), in the
    same order as the edits were applied. Raises ValueError
    (propagated from apply_physics_edit) if any edit cannot be applied --
    callers must not use a partially-applied `text` from a raised call."""
    summary = []
    edits = list(template.get("edits", []))
    if case_id is not None:
        edits += list((template.get("extra_edits") or {}).get(case_id, []))
    for edit in edits:
        text, old, new = apply_physics_edit(text, edit, max_dom)
        summary.append(dict(key=edit["key"], op=edit["op"], old=old, new=new))
    return text, summary


def physics_template_applicability(case_id, template):
    """(applicable: bool, reason: str or None) for one template on one case,
    from the template's optional 'applies_to' allowlist and 'skip_cases' map
    only -- does NOT attempt the edits themselves (see resolve_physics_config
    for that; a template can be 'applicable' here and still fail to resolve,
    e.g. a 'mul' on a physics_suite -1 value)."""
    applies_to = template.get("applies_to")
    if applies_to is not None and case_id not in applies_to:
        return False, "not in applies_to %s" % (applies_to,)
    skip_reason = (template.get("skip_cases") or {}).get(case_id)
    if skip_reason is not None:
        return False, skip_reason
    return True, None


def _case_effective_max_dom(case):
    """max_dom as edit_namelist will actually see it for this case: a
    namelist_overrides.max_dom (e.g. lafe trims 3 -> 1) wins over the case's
    own top-level max_dom field (e.g. maria1km's 2, no override)."""
    overrides = case.get("namelist_overrides") or {}
    if "max_dom" in overrides:
        try:
            return int(overrides["max_dom"])
        except (TypeError, ValueError):
            pass
    return case.get("max_dom", 1)


def _case_namelist_path(case):
    return case.get("namelist_template") or os.path.join(case["prep_dir"], "namelist.input")


def resolve_physics_config(case_id, template_id, cases=None, templates=None):
    """Resolve one physics template against one case by reading the case's
    OWN real namelist.input (namelist_template, or prep_dir/namelist.input),
    the same file edit-namelist itself would edit -- so the reported
    old/new values are the case's actual native physics, not cases.json's
    d01-scalar-for-ledger convention. Returns (edits_summary, None) on
    success (edits_summary is [] for a template with no edits, i.e. P00) or
    (None, reason) when the template does not apply to this case (see
    physics_template_applicability) or an edit op errors for it (e.g. 'mul'
    on a native -1 physics_suite value)."""
    cases = CASES if cases is None else cases
    templates = PHYSICS_TEMPLATES if templates is None else templates
    case = cases.get(case_id)
    if case is None:
        return None, "unknown case %r" % (case_id,)
    template = templates.get(template_id)
    if template is None:
        return None, "unknown physics template %r" % (template_id,)
    ok, reason = physics_template_applicability(case_id, template)
    if not ok:
        return None, reason
    path = _case_namelist_path(case)
    try:
        with open(path) as f:
            text = f.read()
    except IOError as e:
        return None, "cannot read namelist template %s: %s" % (path, e)
    max_dom = _case_effective_max_dom(case)
    try:
        _new_text, summary = apply_physics_template(text, template, max_dom, case_id=case_id)
    except ValueError as e:
        return None, str(e)
    return summary, None


def format_physics_edits(summary):
    """'edits=' notes-field value for a resolve_physics_config() summary:
    ';'-joined 'key=newvalue' tokens (no spaces -- ledger notes are
    space-separated key=value tokens), multi-domain new values '/'-joined
    within one key. Empty string for a no-edit template (P00)."""
    parts = []
    for e in summary:
        if e["op"] == "delete":
            parts.append("%s=DELETED" % (e["key"],))
        else:
            parts.append("%s=%s" % (e["key"], "/".join(e["new"])))
    return ";".join(parts)


_PROBE_VERDICT_RE = re.compile(
    r"(?m)^[ \t]*(?P<knob>\S+)[ \t]+\[(?:PHYSICS-GRAY|STRICT)[ \t]*\][ \t]+(?P<verdict>\S+)"
)


def parse_probe_verdicts(screen_text):
    """{knob: verdict} for every '<knob> [PHYSICS-GRAY ] VERDICT ...' or
    '<knob> [STRICT ] VERDICT ...' line in a wtune screen/physics funnel
    report's STAGE 2 SpeedyWeather-probe section (e.g.
    /home/u5727520/wrf-bench/wscore/physics/screen_<case>.txt, job 1076060).
    VERDICT is a single token (PASS->WRF, CONFIRM, REJECT, NEED-PROBE, ...).
    A later line for the same knob overwrites an earlier one (there is
    normally exactly one line per knob)."""
    out = {}
    for m in _PROBE_VERDICT_RE.finditer(screen_text):
        out[m.group("knob")] = m.group("verdict")
    return out


def probe_verdicts_for_case(case_id, keys, screen_dir=DEFAULT_SCREEN_DIR):
    """'probe=<knob>:<verdict>,...' notes-field value for the given edited
    knob names (order preserved, duplicates kept as given), reading
    screen_<case_id>.txt from screen_dir. 'probe=none' when there are no
    keys to look up (e.g. P00) or the screen file does not exist. A key with
    no matching verdict line in the screen file maps to 'NOTFOUND'."""
    if not keys:
        return "probe=none"
    path = os.path.join(screen_dir, "screen_%s.txt" % (case_id,))
    if not os.path.isfile(path):
        return "probe=none"
    with open(path) as f:
        verdicts = parse_probe_verdicts(f.read())
    parts = ["%s:%s" % (k, verdicts.get(k, "NOTFOUND")) for k in keys]
    return "probe=" + ",".join(parts)


def physics_tag_template(tag):
    """(template_id, role) for one physics-campaign run-set tag, as produced
    by cmd_physics_order/build_order: 'P00_ref' -> ('P00','reference'),
    'P00_twin' -> ('P00','twin'), any other tag (a Pxx id) -> (tag,
    'candidate')."""
    if tag == "P00_ref":
        return "P00", "reference"
    if tag == "P00_twin":
        return "P00", "twin"
    return tag, "candidate"


def edit_namelist(text, nproc_x, nproc_y, numtiles, run_minutes,
                   max_dom=1, overrides=None, delete_keys=None, nest_history_on=False,
                   history_mode="off", adios2=True, ranks=None,
                   history_minutes=None, physics_template=None, physics_case=None,
                   numtiles_x=None, numtiles_y=None, tile_sz_x=None, tile_sz_y=None,
                   tile_strategy=None):
    """Apply all campaign namelist edits and return the new text. Physics and
    everything else not listed below is left untouched, UNLESS physics_template
    names a physics_templates.json template id (see below).

    max_dom controls how many per-domain values are written for numtiles and
    the history-interval variant (see _insert_after_domains/_set_history_mode).
    history_mode is 'off' (timing runs: no domain writes history), 'interval'
    (PHYSICS-GRAY fidelity replicates, decision 0051: d01 writes a frame every
    history_minutes starting at t=0; requires history_minutes) or 'end_d01'
    (correctness runs: d01 writes only its final frame; see
    _set_history_mode's docstring). For 'off'/'end_d01', run_minutes alone
    drives the history-interval value the campaign always kept equal to it,
    so there is no separate history_interval knob to pass in for those modes.
    nest_history_on is forwarded to _set_history_mode/_set_history_interval:
    default False zeroes nested-domain history (diffwrf only compares d01;
    nested history/IO dominates wall time per the isc21/maria1km smoke jobs).
    adios2 (default True, 2026-09-14 user decision: all WRF I/O uses ADIOS2)
    forces io_form_input/boundary/history/restart=14, disables restart writes,
    sets input_inname/bdy_inname and a &namelist_adios2 group -- see
    _apply_adios2. ranks is required whenever adios2 is True (for
    adios2_numaggregators, which MUST equal the MPI rank count). Pass
    adios2=False (CLI --no-adios2) to keep the legacy behaviour instead:
    io_form_history forced to 2, nothing else about I/O touched, ranks unused.
    overrides is an optional {namelist_key: value} dict of additional scalar
    (dimension-1) knobs to force to an exact value after every other edit
    except adios2 -- e.g. {"max_dom": 2, "restart": ".false."}. A key with no
    existing line in the template is silently skipped (matches _set_scalar's
    own no-op-if-absent behaviour). An override that names an io_form key is
    NOT the last word when adios2 is True: _apply_adios2 runs after overrides
    specifically so ADIOS2 can never be reverted by a stale override (see
    cases.json's lafe namelist_overrides notes).
    delete_keys is an optional list of namelist keys to delete entirely
    (case-insensitive, any group) -- for namelist templates that assign a key
    this WRF build no longer has under that name (e.g. lafe's
    bl_mynn_tkebudget/opt_run, see cases.json's lafe notes for the Registry
    successors and why deleting them changes nothing else).
    physics_template (PHYSICS-GRAY data campaign, decision 0051) names a
    physics_templates.json template id (e.g. 'P02') to apply ON TOP OF every
    other edit above, including ADIOS2 -- see apply_physics_template. Raises
    ValueError for an unknown template id or (propagated from
    apply_physics_edit) when the template's edits do not resolve for this
    namelist (e.g. 'mul' on a physics_suite -1 value); callers that need a
    resolve-without-raising check should call resolve_physics_config first.

    numtiles_x/numtiles_y/tile_sz_x/tile_sz_y/tile_strategy (WTune D1
    decomposition-space work, predictor/decomp_space.py) are the tile-geometry
    &domains keys frame/module_tiles.F actually reads (Registry.EM_COMMON,
    dimension 1, same "written once" convention as numtiles/nproc_x/nproc_y --
    see _insert_after_domains). Each is None by default and, when None, is
    NEVER TOUCHED: no line is deleted, nothing is written, and the namelist
    text comes out byte-identical to a call that predates these parameters.
    Passing a value (including 0, WRF's own default for all five) deletes any
    stale line for that key and writes a fresh one on the same &domains line
    as nproc_x/nproc_y/numtiles."""
    text = _set_scalar(text, "run_days", 0)
    text = _set_scalar(text, "run_hours", 0)
    text = _set_scalar(text, "run_minutes", run_minutes)
    text = _set_history_mode(text, history_mode, run_minutes, max_dom, nest_history_on,
                              history_minutes=history_minutes)
    if not adios2:
        text = _set_scalar(text, "io_form_history", 2)
    for key in ("nproc_x", "nproc_y", "numtiles"):
        text = _delete_key_lines(text, key)
    tile_extra = []
    for key, value in (
        ("numtiles_x", numtiles_x), ("numtiles_y", numtiles_y),
        ("tile_sz_x", tile_sz_x), ("tile_sz_y", tile_sz_y),
        ("tile_strategy", tile_strategy),
    ):
        if value is not None:
            tile_extra.append((key, value))
    for key, _value in tile_extra:
        text = _delete_key_lines(text, key)
    text = _insert_after_domains(text, nproc_x, nproc_y, numtiles, max_dom, extra=tile_extra)
    if delete_keys:
        for key in delete_keys:
            text = _delete_key_lines(text, key, ignore_case=True)
    if overrides:
        for key, value in overrides.items():
            text = _set_scalar(text, key, value)
    if adios2:
        text = _apply_adios2(text, ranks, max_dom)
    if physics_template:
        template = PHYSICS_TEMPLATES.get(physics_template)
        if template is None:
            raise ValueError("unknown physics template %r" % (physics_template,))
        text, _summary = apply_physics_template(text, template, max_dom, case_id=physics_case)
    return text


# ---------------------------------------------------------------------------
# Step statistics
# ---------------------------------------------------------------------------

def step_stats(values):
    """Statistics for one run's per-d01-step wall times (seconds), `values[0]`
    being step 1 (init + first radiation call). Returns a dict:
      median     -- median of steps 2..n after excluding any step > 3x that
                     median (radiation-heavy steps), as computed inline in
                     strict_campaign.sbatch before this function existed
      mean_all   -- mean of ALL steps 2..n, INCLUDING the excluded/radiation
                     steps (this is what the ledger's s_per_step_mean now
                     records -- s_per_step_median stays the radiation-excluded
                     median)
      n_measured -- count of steps used for `median`
      n_excluded -- step 1 plus any step excluded from `median`
      n_all      -- len(values) - 1 (all steps after step 1)
    `median`/`mean_all` are None when there are no steps 2..n to compute them
    from (empty input, or a single-step input)."""
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


# ---------------------------------------------------------------------------
# Disk quota guard (problem 1, 2026-09-15: PHYSICS-GRAY smoke jobs 1076238-1076340 kept
# every failed/incomplete run's BP history and netCDF and filled home1's soft quota; this
# is the parser half of the guard strict_campaign.sbatch runs before every WRF srun in
# both the physics loop and the STRICT loop -- see quota-free-gib in main()/_cmd_quota_free_gib)
# ---------------------------------------------------------------------------

def quota_free_gib(mmlsquota_text, filesystem="home1"):
    """Parse `mmlsquota -u $USER <filesystem>` output and return the free space (GiB)
    below the SOFT quota (the 'quota' column, not the hard 'limit') for `filesystem`'s USR
    block-usage row, or None if no such row is found -- e.g. mmlsquota is unavailable, its
    output is empty/garbled, or the row's quota column is 0 (mmlsquota's "no limit enforced
    at the user level here" for a fileset only quota'd at the fileset level -- see
    [[mmlsquota-u-says-no-limit-while-the-fileset-is-99-percent-full]] -- not a signal to
    abort a run on).

    Robust to noise around the real row: the 'Block Limits'/'File Limits' banner line, the
    'Filesystem type KB quota limit ...' header, a stray login-shell warning line, or a
    no-quota row for another filesystem with no numeric columns at all (observed:
    'work2      USR            none'). Only a line shaped like
    '<filesystem>  USR  <used_kb>  <quota_kb>  <limit_kb> ...' matches.

    KB columns are raw GPFS blocks (1024-byte units), so GiB = KB / 1024 / 1024 -- verified
    against home1's real quota row (1048576000 KB quota == 1000.0 GiB, see
    [[f1-home1-soft-quota-raised-to-1000gb]])."""
    pattern = re.compile(r"^\s*(\S+)\s+USR\s+(\d+)\s+(\d+)\s+(\d+)")
    for line in mmlsquota_text.splitlines():
        m = pattern.match(line)
        if m is None:
            continue
        fs, used_kb, quota_kb, _limit_kb = m.groups()
        if fs != filesystem:
            continue
        quota_kb = int(quota_kb)
        if quota_kb <= 0:
            return None
        return (quota_kb - int(used_kb)) / (1024.0 * 1024.0)
    return None


# ---------------------------------------------------------------------------
# STRICT compliance assertion
# ---------------------------------------------------------------------------

_COMPLIANCE_KNOBS = (
    "mpi_ranks", "OMP_NUM_THREADS", "nproc_x", "nproc_y",
    "numtiles", "srun_distribution", "KMP_BLOCKTIME",
)


def assert_strict_compliance():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        import wtune_compliance
    except Exception as e:
        sys.stderr.write(
            "WARNING: could not import wtune_compliance (%s: %s); skipping the "
            "STRICT-compliance assertion for this campaign's varied knobs\n"
            % (type(e).__name__, e)
        )
        return
    bad = [(n, wtune_compliance.classify(n)) for n in _COMPLIANCE_KNOBS]
    bad = [(n, c) for n, c in bad if c != wtune_compliance.STRICT]
    assert not bad, "campaign varies non-STRICT knob(s): %s" % (bad,)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_case(case_id):
    case = CASES.get(case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % case_id)
        return 2
    if not case.get("enabled", True):
        sys.stderr.write("case %s is disabled: %s\n" % (case_id, case.get("reason", "(no reason recorded)")))
        return 1
    kept, skipped, aliases = resolve_case(case)
    print("config_id\tranks\tomp\tnproc_x\tnproc_y\tnumtiles\tdist\tblocktime\tpatch_x\tpatch_y")
    for cfg in kept:
        print("\t".join(str(x) for x in (
            cfg["id"], cfg["ranks"], cfg["omp"], cfg["nx"], cfg["ny"], cfg["numtiles"],
            cfg["dist"], cfg["blocktime"], cfg["patch_x"], cfg["patch_y"],
        )))
    for tid, reason in skipped:
        sys.stderr.write("SKIP %s: %s\n" % (tid, reason))
    for kid, dropped in aliases.items():
        sys.stderr.write("ALIAS %s -> %s\n" % (kid, ",".join(dropped)))
    return 0


# ---------------------------------------------------------------------------
# CONFIG_FILE: run a predictor/decomp_space.py-generated config set instead of
# the fixed C01-C16 STRICT templates (WTune D1 decomposition-space work).
# Entirely opt-in via the CONFIG_FILE env var -- every function above (TEMPLATES,
# resolve_case, cmd_case, the default cmd_order path) is untouched and byte-identical
# when CONFIG_FILE is unset; see edit_namelist's own numtiles_x/y etc. docstring for
# the matching guarantee on the namelist-editing side.
# ---------------------------------------------------------------------------

_GENERATED_TILE_FIELDS = ("numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy")


def load_generated_configs(path):
    """Config dicts from a predictor/decomp_space.py `to_campaign_configs()` JSON file: either a
    bare JSON list, or {"configs": [...]} (decomp_space.py's `sample` CLI output shape) -- both
    accepted so a file can be fed either straight from `to_campaign_configs` or from `sample`."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("configs", data)
    return data


def cmd_case_generated(case_id, config_file):
    """CONFIG_FILE-driven equivalent of cmd_case: prints the SAME first 10 TSV columns
    (config_id..patch_y) plus 5 tile-geometry columns and a config_source column, for every
    config in `config_file`, in file order (no scoring/dedup -- decomp_space.py's sample_design
    already produced a deduplicated, deliberately-ordered list)."""
    case = CASES.get(case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % case_id)
        return 2
    if not case.get("enabled", True):
        sys.stderr.write("case %s is disabled: %s\n" % (case_id, case.get("reason", "(no reason recorded)")))
        return 1
    configs_list = load_generated_configs(config_file)
    header = ["config_id", "ranks", "omp", "nproc_x", "nproc_y", "numtiles", "dist", "blocktime",
              "patch_x", "patch_y"] + list(_GENERATED_TILE_FIELDS) + ["config_source"]
    print("\t".join(header))
    for cfg in configs_list:
        row = [cfg.get("id"), cfg.get("ranks"), cfg.get("omp"), cfg.get("nx"), cfg.get("ny"),
               cfg.get("numtiles"), cfg.get("dist"), cfg.get("blocktime"),
               cfg.get("patch_x"), cfg.get("patch_y")]
        row += [cfg.get(k, 0) for k in _GENERATED_TILE_FIELDS]
        row.append(cfg.get("config_source", "generated"))
        print("\t".join("" if v is None else str(v) for v in row))
    return 0


def cmd_order(case_id, reps, config_file=None):
    case = CASES.get(case_id)
    if case is None or not case.get("enabled", True):
        sys.stderr.write("case %s unknown or disabled\n" % case_id)
        return 2
    if config_file:
        ids = [cfg["id"] for cfg in load_generated_configs(config_file)]
    else:
        kept, _skipped, _aliases = resolve_case(case)
        ids = [c["id"] for c in kept]
    for cid in build_order(ids, reps):
        print(cid)
    return 0


def cmd_all():
    print("case\tn_configs\tn_skipped\treps\truns\test_wall_h\tnode_hours\tcost")
    grand_runs = 0
    grand_node_h = 0.0
    for cid in CASE_ORDER:
        case = CASES.get(cid)
        if case is None:
            sys.stderr.write("%s: missing from cases.json\n" % cid)
            continue
        if not case.get("enabled", True):
            sys.stderr.write("%s: DISABLED (%s)\n" % (cid, case.get("reason", "no reason recorded")))
            continue
        kept, skipped, _aliases = resolve_case(case)
        reps = 2
        runs = len(kept) * reps
        key = (case["e_we"], case["e_sn"], case["e_vert"])
        if key not in COST_GUESS:
            sys.stderr.write("%s: no cost guess for grid %s\n" % (cid, key))
            continue
        s_per_step, init_io = COST_GUESS[key]
        per_run_s = init_io + case["steps"] * s_per_step
        wall_s = runs * (per_run_s + 15.0)
        wall_h = wall_s / 3600.0
        node_h = wall_h * 4
        cost = node_h * 22.4
        grand_runs += runs
        grand_node_h += node_h
        print("%s\t%d\t%d\t%d\t%d\t%.3f\t%.3f\t%.2f" % (
            cid, len(kept), len(skipped), reps, runs, wall_h, node_h, cost))
    grand_wall_h = grand_node_h / 4.0
    print("TOTAL\truns=%d\test_wall_h=%.3f\tnode_hours=%.3f\tcost=%.2f  (GUESS: s_per_step/init_io "
          "are task-spec estimates, not measurements)"
          % (grand_runs, grand_wall_h, grand_node_h, grand_node_h * 22.4))
    return 0


def _cmd_edit_namelist(argv):
    ap = argparse.ArgumentParser(prog="configs.py edit-namelist")
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--nproc-x", type=int, required=True)
    ap.add_argument("--nproc-y", type=int, required=True)
    ap.add_argument("--numtiles", type=int, required=True)
    ap.add_argument("--run-minutes", type=int, required=True)
    ap.add_argument("--max-dom", type=int, default=1,
                     help="number of active domains; controls how many per-domain "
                          "values are written for numtiles and history_interval*")
    ap.add_argument("--history-mode", choices=["off", "interval", "end_d01"], default="off",
                     help="'off' (default, timing runs): no domain writes history. "
                          "'interval' (PHYSICS-GRAY fidelity replicates): d01 writes a "
                          "frame every --history-minutes, starting at t=0; requires "
                          "--history-minutes. 'end_d01': d01 writes only its final "
                          "frame (correctness runs); nested domains always get 0.")
    ap.add_argument("--history-minutes", type=int, default=None,
                     help="required when --history-mode interval is given")
    ap.add_argument("--physics-template", default=None, metavar="Pxx",
                     help="apply a physics_templates.json template id (decision "
                          "0051 PHYSICS-GRAY data campaign) on top of every other "
                          "edit, including ADIOS2, e.g. --physics-template P02")
    ap.add_argument("--physics-case", default=None, metavar="CASE",
                     help="cases.json case id whose extra_edits (per-case paired "
                          "edits) are applied with --physics-template")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="repeatable; forces namelist scalar KEY to VALUE after "
                          "every other edit except ADIOS2, e.g. --set max_dom=2 "
                          "--set restart=.false.")
    ap.add_argument("--delete-key", action="append", default=[], dest="delete_keys",
                     metavar="KEY",
                     help="repeatable; delete every line assigning this namelist key "
                          "(case-insensitive, any group), e.g. --delete-key opt_run")
    ap.add_argument("--nest-history-on", action="store_true",
                     help="keep history-interval writes for nested domains (default: "
                          "nested domains get 0, d01 gets the run length)")
    ap.add_argument("--no-adios2", dest="adios2", action="store_false", default=True,
                     help="disable ADIOS2 I/O settings (default: on). Legacy behaviour: "
                          "io_form_history forced to 2, nothing else about I/O touched.")
    ap.add_argument("--ranks", type=int, default=None,
                     help="MPI rank count; required unless --no-adios2 is given "
                          "(adios2_numaggregators MUST equal the rank count)")
    ap.add_argument("--numtiles-x", type=int, default=None, metavar="N",
                     help="&domains numtiles_x (WTune D1 decomposition-space work); "
                          "not written at all when omitted")
    ap.add_argument("--numtiles-y", type=int, default=None, metavar="N",
                     help="&domains numtiles_y; not written at all when omitted")
    ap.add_argument("--tile-sz-x", type=int, default=None, metavar="N",
                     help="&domains tile_sz_x (overrides numtiles entirely, "
                          "frame/module_tiles.F:353); not written at all when omitted")
    ap.add_argument("--tile-sz-y", type=int, default=None, metavar="N",
                     help="&domains tile_sz_y; not written at all when omitted")
    ap.add_argument("--tile-strategy", type=int, default=None, metavar="N",
                     help="&domains tile_strategy (TILE_NONE=0 TILE_X=1 TILE_Y=2 TILE_XY=3, "
                          "module_machine.F:11); not written at all when omitted")
    a = ap.parse_args(argv)
    if a.adios2 and a.ranks is None:
        ap.error("--ranks is required unless --no-adios2 is given")
    if a.history_mode == "interval" and a.history_minutes is None:
        ap.error("--history-minutes is required when --history-mode interval is given")
    overrides = {}
    for kv in a.set:
        key, sep, value = kv.partition("=")
        if not sep:
            ap.error("--set expects KEY=VALUE, got %r" % kv)
        overrides[key] = value
    with open(a.infile) as f:
        text = f.read()
    new_text = edit_namelist(text, a.nproc_x, a.nproc_y, a.numtiles, a.run_minutes,
                              max_dom=a.max_dom, overrides=overrides,
                              delete_keys=a.delete_keys, nest_history_on=a.nest_history_on,
                              history_mode=a.history_mode, adios2=a.adios2, ranks=a.ranks,
                              history_minutes=a.history_minutes,
                              physics_template=a.physics_template,
                              physics_case=a.physics_case,
                              numtiles_x=a.numtiles_x, numtiles_y=a.numtiles_y,
                              tile_sz_x=a.tile_sz_x, tile_sz_y=a.tile_sz_y,
                              tile_strategy=a.tile_strategy)
    with open(a.outfile, "w") as f:
        f.write(new_text)
    return 0


def cmd_physics_list(argv):
    ap = argparse.ArgumentParser(prog="configs.py physics-list")
    ap.add_argument("case_id")
    ap.add_argument("--json", action="store_true",
                     help="print applicable non-P00 template ids as a JSON list "
                          "(PHYSICS_TEMPLATE_ORDER order) instead of the "
                          "human-readable per-template report")
    a = ap.parse_args(argv)
    if a.case_id not in CASES:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    if a.json:
        applicable = []
        for tid in PHYSICS_TEMPLATE_ORDER:
            if tid == "P00":
                continue
            summary, _reason = resolve_physics_config(a.case_id, tid)
            if summary is not None:
                applicable.append(tid)
        print(json.dumps(applicable))
        return 0
    for tid in PHYSICS_TEMPLATE_ORDER:
        template = PHYSICS_TEMPLATES[tid]
        family = template.get("family", "-")
        summary, reason = resolve_physics_config(a.case_id, tid)
        if summary is None:
            print("%s (%s) SKIP: %s" % (tid, family, reason))
            continue
        if not summary:
            print("%s (%s) APPLICABLE: (no edits -- reference template)" % (tid, family))
            continue
        edit_strs = []
        for e in summary:
            if e["op"] == "delete":
                edit_strs.append("%s: DELETED" % (e["key"],))
            else:
                edit_strs.append("%s: %s -> %s" % (
                    e["key"], ",".join(e["old"]), ",".join(e["new"])))
        print("%s (%s) APPLICABLE: %s" % (tid, family, "; ".join(edit_strs)))
    return 0


def cmd_physics_order(argv):
    ap = argparse.ArgumentParser(prog="configs.py physics-order")
    ap.add_argument("case_id")
    ap.add_argument("--pconfigs", default=None,
                     help="'+'-separated physics template ids (default: every "
                          "template applicable to this case, in "
                          "PHYSICS_TEMPLATE_ORDER, excluding P00)")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args(argv)
    if a.case_id not in CASES:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    if a.pconfigs:
        pconfigs = [p for p in a.pconfigs.split("+") if p]
    else:
        pconfigs = []
        for tid in PHYSICS_TEMPLATE_ORDER:
            if tid == "P00":
                continue
            summary, _reason = resolve_physics_config(a.case_id, tid)
            if summary is not None:
                pconfigs.append(tid)
    entries = ["P00_ref", "P00_twin"] + pconfigs
    for tag in build_order(entries, a.reps):
        print(tag)
    return 0


# ledger columns a physics template edit can actually change (time_step is its own
# ledger column, distinct from the physics_* columns, but goes through the same
# native-value-then-override logic as the rest -- see physics_ledger_columns).
_LEDGER_PHYSICS_KEYS = (
    "time_step", "mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
    "sf_surface_physics", "ra_lw_physics", "ra_sw_physics", "radt", "cudt",
)


def physics_ledger_columns(case, summary):
    """{ledger column name: value string} for a resolved physics template
    summary (resolve_physics_config's non-None first element): starts from
    the case's own native d01-scalar-for-ledger values (case['dt'] for
    time_step, case['physics'][key] for the rest -- the same convention
    cases.json/strict_campaign.sbatch already use for STRICT ledger rows),
    then overrides any key the template actually edited with its new D01
    (first active-domain) value -- so a P02/P03/P04 row's edited column(s)
    show the edited value while every other physics column keeps showing the
    case's native one, exactly like an unedited STRICT row would."""
    cols = {"time_step": case["dt"]}
    phys = case.get("physics", {})
    for k in _LEDGER_PHYSICS_KEYS:
        if k == "time_step":
            continue
        cols[k] = phys.get(k, "")
    for e in summary:
        key = e["key"]
        if key not in _LEDGER_PHYSICS_KEYS or e["op"] == "delete":
            continue
        cols[key] = e["new"][0]
    return cols


def describe_field(value):
    """physics-describe TSV field: '-' for empty, since bash `IFS=$'\\t' read` collapses empty
    tab-separated fields (strict_campaign.sbatch maps '-' back to empty)."""
    s = "" if value is None else str(value)
    return s if s != "" else "-"


def cmd_physics_describe(argv):
    """physics-describe CASE_ID TEMPLATE_ID: one TSV line summarizing
    everything strict_campaign.sbatch's MODE=physics needs for one (case,
    template) pair's ledger row --
      family edits probe time_step mp_physics cu_physics bl_pbl_physics \
      sf_sfclay_physics sf_surface_physics ra_lw_physics ra_sw_physics radt cudt
    or, when the template does not apply to this case:
      SKIP <reason>
    (2 fields only -- callers must check field 1 == 'SKIP' first)."""
    ap = argparse.ArgumentParser(prog="configs.py physics-describe")
    ap.add_argument("case_id")
    ap.add_argument("template_id")
    ap.add_argument("--screen-dir", default=DEFAULT_SCREEN_DIR)
    a = ap.parse_args(argv)
    case = CASES.get(a.case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    template = PHYSICS_TEMPLATES.get(a.template_id)
    if template is None:
        sys.stderr.write("unknown physics template %r\n" % (a.template_id,))
        return 2
    summary, reason = resolve_physics_config(a.case_id, a.template_id)
    if summary is None:
        print("SKIP\t%s" % (reason,))
        return 0
    family = template.get("family", "-")
    edits_str = format_physics_edits(summary)
    edited_keys = [e["key"] for e in summary if e["op"] != "delete"]
    probe_str = probe_verdicts_for_case(a.case_id, edited_keys, screen_dir=a.screen_dir)
    cols = physics_ledger_columns(case, summary)
    # Empty fields are written as "-": bash `IFS=$'\t' read` treats tab as whitespace and
    # collapses consecutive tabs, so an empty edits field (P00) shifted every later column left
    # by one and put mp_physics into time_step in campaign ledgers 1077168-1077173.
    print("\t".join(describe_field(x) for x in [
        family, edits_str, probe_str,
        cols["time_step"], cols["mp_physics"], cols["cu_physics"],
        cols["bl_pbl_physics"], cols["sf_sfclay_physics"], cols["sf_surface_physics"],
        cols["ra_lw_physics"], cols["ra_sw_physics"], cols["radt"], cols["cudt"],
    ]))
    return 0


def _cmd_step_stats(argv):
    ap = argparse.ArgumentParser(prog="configs.py step-stats")
    ap.add_argument("steps_file")
    a = ap.parse_args(argv)
    try:
        with open(a.steps_file) as f:
            values = [float(x) for x in f if x.strip()]
    except IOError:
        values = []
    stats = step_stats(values)

    def fmt(v):
        return "%.5f" % v if v is not None else ""

    print("med=%s mean=%s n=%d excl=%d nall=%d" % (
        fmt(stats["median"]), fmt(stats["mean_all"]), stats["n_measured"],
        stats["n_excluded"], stats["n_all"]))
    return 0


def _cmd_quota_free_gib(argv):
    ap = argparse.ArgumentParser(prog="configs.py quota-free-gib")
    ap.add_argument("--filesystem", default="home1")
    a = ap.parse_args(argv)
    text = sys.stdin.read()
    free_gib = quota_free_gib(text, filesystem=a.filesystem)
    if free_gib is None:
        print("NA")
        return 1
    print("%.3f" % free_gib)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.stderr.write(
            "usage: configs.py CASE_ID | --all | order CASE_ID [--reps N] | "
            "edit-namelist IN OUT --nproc-x X --nproc-y Y --numtiles N "
            "--run-minutes M [--max-dom N] [--history-mode off|interval|end_d01] "
            "[--history-minutes N] [--set KEY=VALUE ...] [--delete-key KEY ...] "
            "[--nest-history-on] [--ranks N] [--no-adios2] "
            "[--numtiles-x N] [--numtiles-y N] [--tile-sz-x N] [--tile-sz-y N] "
            "[--tile-strategy N] [--physics-template Pxx] | "
            "step-stats STEPS_FILE | "
            "physics-list CASE_ID [--json] | "
            "physics-order CASE_ID [--pconfigs P01+P02+...] [--reps N] | "
            "physics-describe CASE_ID TEMPLATE_ID | "
            "quota-free-gib [--filesystem home1] (reads mmlsquota text on stdin)\n"
            "CONFIG_FILE=path.json env (predictor/decomp_space.py to_campaign_configs/sample "
            "output): CASE_ID and order CASE_ID measure that generated config set instead of "
            "the fixed C01-C16 STRICT templates.\n"
        )
        return 2
    if argv[0] == "edit-namelist":
        return _cmd_edit_namelist(argv[1:])
    if argv[0] == "step-stats":
        return _cmd_step_stats(argv[1:])
    if argv[0] == "quota-free-gib":
        return _cmd_quota_free_gib(argv[1:])
    if argv[0] == "physics-list":
        return cmd_physics_list(argv[1:])
    if argv[0] == "physics-order":
        return cmd_physics_order(argv[1:])
    if argv[0] == "physics-describe":
        return cmd_physics_describe(argv[1:])
    assert_strict_compliance()
    if argv[0] == "--all":
        return cmd_all()
    config_file = os.environ.get("CONFIG_FILE")
    if argv[0] == "order":
        ap = argparse.ArgumentParser(prog="configs.py order")
        ap.add_argument("case_id")
        ap.add_argument("--reps", type=int, default=2)
        a = ap.parse_args(argv[1:])
        return cmd_order(a.case_id, a.reps, config_file=config_file)
    if config_file:
        return cmd_case_generated(argv[0], config_file)
    return cmd_case(argv[0])


if __name__ == "__main__":
    raise SystemExit(main())
