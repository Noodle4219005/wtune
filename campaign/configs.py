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

IMPORTANT ENVIRONMENT FACT (verified 2026-09-14, f1-SPECIFIC -- see 2026-09-24 update):
wtune_decomp.py, the repo's "own decomposition scorer" this generator is asked to
prefer, opens with `from __future__ import annotations`, a Python >=3.7 feature. On
f1, where this campaign is required to run under the bare `python3` on both the login
node and (via the same module environment) the compute nodes, `python3` is 3.6.8, so
`import wtune_decomp` raises SyntaxError at import time -- not a missing file, a
language-version mismatch. `_load_decomp()` below tries the import, catches the
failure, and every shape lookup on f1 silently and correctly falls through to the
documented fallback scorer:

    score(nx, ny) = |log((e_we/nx) / (e_sn/ny))| + 0.1 * (nx > ny)

lower is better; legality is `ceil(e_we/nx) >= 10 and ceil(e_sn/ny) >= 10` (the
`MIN_PATCH` used by wtune_decomp.py too, but see the docstring of `legal_shapes`
for a documented difference in which patch dimension the two check). If this code
is ever run under Python >=3.7, `_load_decomp` will succeed and the generator will
prefer wtune_decomp.enumerate_configs's own `score` field instead, per-group, with
no code change needed here.

UPDATE 2026-09-24: the paragraph above describes f1's interpreter, not every host
this repo runs on. On scc, the bare `python3` is 3.14 (>=3.7), so `import
wtune_decomp` succeeds there and `_load_decomp` takes the real-scorer branch, not
the fallback, with no code change needed. Across the whole WTune toolset (this
generator, the tuner, and anything under predictor/ needing numpy/netCDF4/catboost/
scikit-learn, none of which scc's bare `python3` has), the supported interpreter is
now the envs/wtune pixi environment, launched via `bin/wtune` -- see that script and
envs/wtune/pixi.toml.
"""
import argparse
import datetime
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
# SITE env var (optional, decision 0050 step 8 minimal hardware abstraction): unset/"f1" leaves
# every consumer of CASES/TOTAL_CORES below byte-identical to before SITE existed. Any other
# value additionally overlays campaign/cases.<site>.json onto cases.json in load_cases() below.
SITE = os.environ.get("SITE") or "f1"

MIN_PATCH = 10
# 4 nodes x 112 cores, ct448 -- f1's own C01-C16 TEMPLATES table (below) is anchored at this
# exact value (ranks*omp == TOTAL_CORES, asserted by test_configs.py) and is NOT rescaled by
# WTUNE_TOTAL_CORES; a different total (e.g. a non-448 node count on f1, or any node count on
# scc, whose 72 cores/node never divides 448/224/112 evenly) simply makes every template fail
# resolve_config's ranks*omp==TOTAL_CORES check and skip -- by design, see the "generated space"
# fallback in predictor/experiment.py and the module docstring of predictor/decomp_space.py.
TOTAL_CORES = int(os.environ.get("WTUNE_TOTAL_CORES", 448))

# ADIOS2 compression, written explicitly into every run's &namelist_adios2 block
# (see _apply_adios2) rather than inherited from the WRF Registry defaults in
# Registry/registry.io_boilerplate:149-150 (.true. / "lz4"). A run's own namelist
# must record what it used: the codec is a measured choice, not a default.
#
# zstd over lz4: established by tier-a-runtime-battery (2026-08-22, jobs
# 1037956-64, 4N/448) -- zstd steady write 0.468 s vs lz4 0.496 s vs none
# 1.014 s -- and confirmed on size by conus12 job 1082176, which had been the
# missing half of that comparison: frame 1 stored 166,682,400 B with zstd
# against 181,090,208 B with lz4 and 595,875,392 B uncompressed, i.e. zstd
# 3.57x, lz4 3.29x, zstd 8.0% smaller than lz4.
#
# KNOWN LIMITATION of the WTune build (/home/u5727520/wrf-stack/wrf/
# intel-adios2-dmsm/WRF), measured by job 1082176 and NOT fixed by these keys:
# only the FIRST ADIOS2 write stream of a run is actually compressed.
# wrf_io.F90:1283-1285 assigns DH%blosc_compressor only on the branch that
# DEFINES the operator; a later stream's adios2_inquire_operator succeeds, so
# %valid is true but its DH%blosc_compressor is blank, and the codec chain at
# wrf_io.F90:2497-2508 matches nothing, so adios2_add_operation is never
# called. Measured: frames 2 and 3 of a 3-frame conus12 run stored exactly the
# uncompressed 595,875,392 B in BOTH the zstd and the lz4 arm. This is a WRF
# source bug, not a namelist one. It does not affect MODE=timing (no history)
# or --history-mode end_d01 (exactly one frame, which is the first stream and
# does compress); it DOES affect --history-mode interval. tools/
# bp_compression_check.py is how a campaign detects it instead of assuming.
ADIOS2_COMPRESSION_ENABLE = ".true."
ADIOS2_BLOSC_COMPRESSOR = "zstd"

# Deterministic case order (matches the task table); used by --all and cmd_order so
# output does not depend on JSON/dict iteration order.
# 2026-09-14: deep_conv_ot, arctic_sbl, edouard_landfall and lowell_cat4 were removed from the campaign at
# the user's request (their inputs stay on disk). The old restart-only "maria" (5 km) case is removed
# (disk data untouched); the testcase-A cases CP-1 isc21 (3-domain), DOI-4 lafe (d01+d02) and OB-3
# maria1km (2-domain cold start) are added now that their inputs are prepared.
CASE_ORDER = [
    "karina", "conus25", "conus12", "isc21", "maria1km", "lafe",
    "sc26p1", "sc26p2", "sc26p2narr",
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


def load_cases(path=CASES_JSON_PATH, site=None):
    """cases.json, optionally overlaid by campaign/cases.<site>.json (decision 0050 step 8):
    {"cases": {id: {overriding keys}}}, merged key-by-key OVER the matching cases.json entry (a
    case absent from the overlay is untouched; a key absent from an override entry keeps its
    cases.json value). site=None (the default) reads the module-level SITE env var; site="f1"
    (or SITE unset/"f1") never overlays anything, so CASES below is byte-identical to before
    SITE existed. A str override value may use $VAR/~ (e.g. "$HOME/..." -- a site's real $HOME
    is not known at authoring time) and is expanded here, same rule as
    strict_campaign.sbatch's own CASES_JSON reader."""
    with open(path) as f:
        data = json.load(f)
    cases = data["cases"]
    site = SITE if site is None else site
    if site and site != "f1":
        overlay_path = os.path.join(CAMPAIGN_DIR, "cases.%s.json" % site)
        if os.path.exists(overlay_path):
            with open(overlay_path) as f:
                overlay = json.load(f)["cases"]
            merged = dict(cases)
            for cid, over in overlay.items():
                base = dict(merged.get(cid, {}))
                for k, v in over.items():
                    if isinstance(v, str):
                        v = os.path.expanduser(os.path.expandvars(v))
                    base[k] = v
                merged[cid] = base
            cases = merged
    return cases


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
    """Try to import wtune_decomp from src/. Returns the module or None.
    Caches the result (including failure) for the life of the process."""
    global _DECOMP, _DECOMP_TRIED
    if _DECOMP_TRIED:
        return _DECOMP
    _DECOMP_TRIED = True
    _src = os.path.join(REPO_ROOT, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
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
    domain would reject is dropped here, uniformly for both scoring methods.

    case["max_omp"] (optional, absent = no cap): a per-case hard ceiling on OMP threads
    per rank, e.g. c34/marshall's max_omp=1 (WRF-Fire crashes rc=137/139 at the first d01
    step with any omp>1, see cases.json's notes for those cases). A template whose omp
    exceeds it is skipped here with the same reason-string convention as every other
    illegal-shape rejection below, so it never reaches resolve_case()'s kept list."""
    ranks, omp = template["ranks"], template["omp"]
    if ranks * omp != TOTAL_CORES:
        return None, "ranks(%d)*omp(%d) != %d" % (ranks, omp, TOTAL_CORES)
    max_omp = case.get("max_omp")
    if max_omp is not None and omp > max_omp:
        return None, "omp(%d) exceeds case max_omp=%d" % (omp, max_omp)
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
            # A nest normally gets 0 (no history at all). But hailcast_opt aborts at init on any
            # domain where it is ON and that domain's history_interval is 0 ("HAILCAST Error : No
            # 'history_interval' defined in namelist", phys/module_diagnostics_driver.F; bigchill
            # and elevconv reference runs, jobs 1081089/1081090). A never-elapsing interval
            # satisfies the check and still writes nothing.
            nest_value = (_history_key_and_value(text, HISTORY_NEVER_MINUTES)[2]
                          if _history_requiring_diagnostics_on(text) else 0)
            values = [minutes] + [nest_value] * (max_dom - 1)
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
    if history_mode == "mid_end_d01":
        # two d01 frames: at run_minutes//2 and at run_minutes (begin=first, interval=run-first).
        # A RESTARTED run never writes a frame at its own start time (measured, scc job 2194:
        # history_begin=0 gave only the end frame), and fidelity_gate.run_gate needs a matched
        # time after the reference's first valid time, so the pair mid+end is the smallest set
        # of frames that gate can label.
        first = int(run_minutes) // 2
        if first < 1 or int(run_minutes) - first < 1:
            raise ValueError("mid_end_d01 needs run_minutes >= 2, got %r" % (run_minutes,))
        text = _set_history_interval(text, int(run_minutes) - first, max_dom, nest_history_on)
        _interval_key, begin_key, value = _history_key_and_value(text, first)
        if max_dom > 1:
            begin_values = [value] * max_dom if nest_history_on else [value] + [0] * (max_dom - 1)
        else:
            begin_values = [value]
        return _replace_list_line(text, begin_key, begin_values)
    raise ValueError("history_mode must be 'off', 'interval', 'end_d01' or 'mid_end_d01', got %r"
                     % (history_mode,))


def _set_or_insert_time_control_scalar(text, key, value, group="time_control"):
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
    # `group` (default &time_control): the &fdda stream keys (io_form_gfdda, gfdda_inname) are
    # inserted into &fdda instead -- see adios2_input_streams.
    m = re.search(r"(?im)^[ \t]*&%s\b" % re.escape(group), text)
    if m is None:
        raise ValueError("no &%s section found in namelist" % group)
    eol = text.find("\n", m.start())
    if eol == -1:
        eol = len(text)
    return text[: eol + 1] + " %s=%s,\n" % (key, value) + text[eol + 1 :]


_NML_NEXT_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_%]*[ \t]*=")


def _set_namelist_value(text, key, literal, group):
    """`key` = `literal` (a Fortran literal, possibly a per-domain list "1, 1") -- the WHOLE
    value list of every non-comment assignment is replaced (a packed line keeps its other keys;
    _set_scalar would leave a stale per-domain tail behind: "1, 999999,"). When `key` is not
    assigned anywhere it is inserted right after the &`group` header; when that group is absent
    too (e.g. no &bdy_control), a new &`group` holding just this key is appended."""
    rhs = literal.strip().rstrip(",").rstrip() + ","
    key_re = re.compile(r"(?<![A-Za-z0-9_])%s[ \t]*=[ \t]*" % re.escape(key), re.I)
    lines = text.split("\n")
    hit = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("!"):
            continue
        m = key_re.search(ln)
        if m is None:
            continue
        rest, end, quote, j = ln[m.end():], None, None, 0
        while j < len(rest):                  # the value list ends at the next `key =`, '!', '/'
            c = rest[j]
            if quote:
                quote = None if c == quote else quote
            elif c in "'\"":
                quote = c
            elif c in "!/":
                end = j
                break
            elif ((c.isalpha() or c == "_") and (j == 0 or rest[j - 1] in " \t,")
                  and _NML_NEXT_KEY_RE.match(rest, j)):
                end = j
                break
            j += 1
        tail = rest[end:] if end is not None else ""
        lines[i] = ln[:m.end()] + rhs + ((" " + tail.lstrip()) if tail.strip() else "")
        hit = True
    if hit:
        return "\n".join(lines)
    m = re.search(r"(?im)^[ \t]*&%s\b[^\n]*\n" % re.escape(group), text)
    if m is not None:
        return text[:m.end()] + " %s = %s\n" % (key, rhs) + text[m.end():]
    return text.rstrip("\n") + "\n&%s\n %s = %s\n/\n" % (group, key, rhs)


def _get_scalar_first(text, key, default=None):
    """First value of namelist key `key` (a comma-list's first entry, or a bare scalar) --
    read-only counterpart to _set_scalar. Used by case_start_datetime for the RANKING-restart
    spin-up (2026-09-26, end-to-end wiring): needs the case's own t0 to compute the shared
    restart's absolute time (t0 + write_restart_interval minutes)."""
    m = re.search(r"\b%s\b[ \t]*=[ \t]*([^,\n]*)" % re.escape(key), text)
    if not m:
        return default
    return m.group(1).strip()


def case_start_datetime(text):
    """(datetime) from &time_control's start_year/month/day/hour/minute/second -- domain 1's
    values (every domain in this campaign shares one simulated t0). minute/second default to 0
    when absent, same convention as tools/coldstart_sample.py's own start-time readers."""
    year = int(_get_scalar_first(text, "start_year"))
    month = int(_get_scalar_first(text, "start_month"))
    day = int(_get_scalar_first(text, "start_day"))
    hour = int(_get_scalar_first(text, "start_hour", "0") or 0)
    minute = int(_get_scalar_first(text, "start_minute", "0") or 0)
    second = int(_get_scalar_first(text, "start_second", "0") or 0)
    return datetime.datetime(year, month, day, hour, minute, second)


def _apply_restart_from(text, restart_time, max_dom):
    """Overrides EVERY active domain's start_year/month/day/hour/minute/second to
    `restart_time` (a datetime) and forces restart='.true.' -- the per-domain rewrite
    edit_namelist's own `overrides`/--set cannot express (see edit_namelist's restart_from
    docstring). Uses _replace_list_line (delete + reinsert), same as _set_history_interval's own
    max_dom>1 branch, so every domain gets the SAME restart_time (a ranking probe measures one
    config from one shared point, not per-domain analog times). Second is written even when the
    template omits it (same reasoning as coldstart_sample.build_wrf_window_namelist_text: a
    missing end_second/start_second defaults to 0, which would silently start every domain 0-59s
    off restart_time otherwise)."""
    for key in ("start_year", "start_month", "start_day", "start_hour", "start_minute",
                "start_second"):
        value = {"start_year": restart_time.year, "start_month": restart_time.month,
                 "start_day": restart_time.day, "start_hour": restart_time.hour,
                 "start_minute": restart_time.minute, "start_second": restart_time.second}[key]
        text = _replace_list_line(text, key, [value] * max_dom)
    text = _set_scalar(text, "restart", ".true.")
    # override_restart_timers (decision 0055 G0, jobs 1371/1372): WRF's default .false. keeps
    # the restart/history intervals STORED IN the restart file (the spin-up's own
    # write_restart_interval), ignoring this namelist's restart_interval=1000000 -- every
    # candidate then wrote a restart M minutes after resuming, inside its measured window
    # (netCDF: 299 s landing on one radiation step, r 121 instead of ~5).
    if re.search(r"(?im)^\s*override_restart_timers\s*=", text):
        return _set_scalar(text, "override_restart_timers", ".true.")
    return _insert_after_section_header(text, "&time_control",
                                        " override_restart_timers = .true.,\n")


def _apply_adios2(text, ranks, max_dom, start=None):
    """Force ADIOS2 I/O for input/boundary/history/restart (2026-09-14 user
    decision: all WRF I/O in this campaign uses ADIOS2, io_form 14) and
    disable restart writes (restart_interval set far beyond any campaign run
    length so nothing is ever written -- no campaign run uses WRF restart).
    Recipe proven in regress_bp.sbatch / job 1072441
    (/home/u5727520/wrf-stack/wtune/plov/regress_bp.sbatch:47-50).

    adios2_numaggregators MUST equal the MPI rank count: WRF's default (0 =
    one aggregator per node) combined with --distribution=cyclic produced BP
    output that cannot be read back ("Read past end of file ... data.3").

    adios2_compression_enable/adios2_blosc_compressor are written explicitly
    (ADIOS2_COMPRESSION_ENABLE/ADIOS2_BLOSC_COMPRESSOR at the top of this file,
    where the evidence for the codec and the WRF first-stream-only limitation
    is recorded) so a run's namelist states its own I/O settings instead of
    silently inheriting the Registry defaults.

    Called LAST in edit_namelist, after `overrides`, so a case override that
    sets an io_form (e.g. lafe's old io_form_input=2 for pnetcdf) can never
    revert ADIOS2 -- see cases.json's lafe namelist_overrides notes.

    Uses _set_or_insert_time_control_scalar (not plain _set_scalar) for the
    io_form_* keys and restart_interval so a template that omits one of them
    still gets ADIOS2 forced, rather than silently keeping WRF's default.

    Per-record inputs (2026-10-03): WRF 4.8's ADIOS2 reader never selects a BP step, so a
    multi-record wrfbdy/wrflowinp/wrffdda BP always yields record 1; stage-bp-inputs therefore
    converts them one BP per record (wtune_bp.convert_inputs) and exports their names in
    WTUNE_PER_RECORD_BP (and the cache in WTUNE_BP_DIR). For those streams this applies
    wtune_bp.per_record_namelist_edits (multi_bdy_files + dated innames, frames_per_auxinputN=1).
    WRF names each file by the CLOCK TIME of the read, so a run starting at `start` (the
    restart_from time for a restarted probe; default: the namelist's own start) needs
    <name>_<start>.bp. A stream with any name lacking that record (a shared restart taken off a
    boundary/SST record time) falls back to its netCDF file for this run -- io_form 2 and the
    default inname (the netCDF wrfbdy_d01 / wrflowinp_d0N are linked from prep_dir) -- because
    no BP layout reads correctly there: a single multi-record BP gives record 1 everywhere. The
    fallback overwrites whatever a previous pass set (the RANKING restart pass edits the first
    pass's output), so it is idempotent. A stderr line names each fallback."""
    if ranks is None:
        raise ValueError("ranks is required when adios2=True (for adios2_numaggregators)")
    # WTUNE_INPUT_FORMAT=netcdf is the LEGACY escape only (an explicit export at submission;
    # strict_campaign.sbatch warns): inputs stay netCDF (io_form 2, default file names);
    # history/restart are still forced to ADIOS2. Since 2026-10-03 an empty bp_dir no longer
    # means netCDF: strict_campaign converts prep_dir's inputs into a BP cache first.
    netcdf_inputs = os.environ.get("WTUNE_INPUT_FORMAT", "adios2") == "netcdf"
    for key in ("io_form_input", "io_form_boundary", "io_form_history", "io_form_restart"):
        value = 2 if netcdf_inputs and key in ("io_form_input", "io_form_boundary") else 14
        text = _set_or_insert_time_control_scalar(text, key, value)
    text = _set_or_insert_time_control_scalar(text, "restart_interval", 1000000)
    text = _delete_key_lines(text, "input_inname")
    text = _delete_key_lines(text, "bdy_inname")
    idx = text.find("&time_control")
    if idx == -1:
        raise ValueError("no &time_control section found in namelist")
    eol = text.find("\n", idx)
    if eol == -1:
        eol = len(text)
    if not netcdf_inputs:
        # aux input streams (2026-10-03 user decision: ADIOS2 for every input stream):
        # WTUNE_AUX_BP_STREAMS (set by strict_campaign.sbatch from stage-bp-inputs) lists the aux
        # streams whose files ARE available as <name>.bp; only those are switched, so a bp_dir
        # without e.g. wrflowinp_d01.bp keeps that stream on the case's own netCDF settings.
        aux_ok = set(x for x in os.environ.get("WTUNE_AUX_BP_STREAMS", "").split(",") if x)
        pr_ok, pr_fallback = per_record_streams(
            os.environ.get("WTUNE_PER_RECORD_BP", ""), start, text,
            os.environ.get("WTUNE_BP_DIR") or None, aux_ok=aux_ok)
        new_line = ' input_inname="wrfinput_d<domain>.bp",'
        if "boundary" not in pr_ok and "boundary" not in pr_fallback:
            new_line += ' bdy_inname="wrfbdy_d<domain>.bp",'
        text = text[: eol + 1] + new_line + "\n" + text[eol + 1 :]
        for st in adios2_input_streams(text):
            if st["stream"] in aux_ok and st["stream"] not in ("input", "boundary"):
                if st["stream"] in pr_ok or st["stream"] in pr_fallback:
                    continue                    # per-record: handled below
                for key, value, group in adios2_stream_edits(st):
                    text = _set_or_insert_time_control_scalar(text, key, value, group)
        if pr_ok:
            bp = _import_wtune_bp()
            names = [n for s in pr_ok.values() for n in s]
            for group, kv in bp.per_record_namelist_edits(names, max_dom).items():
                for key, value in kv.items():
                    text = _set_namelist_value(text, key, value, group)
            for stream in pr_ok:
                io_key, group = PER_RECORD_STREAMS[stream][1:3]
                text = _set_namelist_value(text, io_key, str(ADIOS2_IO_FORM), group)
        for stream, missing in sorted(pr_fallback.items()):
            text = _per_record_netcdf_fallback(text, stream, max_dom)
            sys.stderr.write("NOTE: ADIOS2 per-record %s input has no record at the run start "
                             "(%s) -- this run reads that stream from netCDF instead\n"
                             % (stream, ", ".join(missing)))
    text = re.sub(r"(?ms)^&namelist_adios2\b.*?^/[ \t]*\n?", "", text)
    text = text.rstrip("\n") + (
        "\n&namelist_adios2\n"
        " adios2_numaggregators=%d,\n"
        " adios2_compression_enable=%s,\n"
        ' adios2_blosc_compressor="%s",\n'
        "/\n" % (ranks, ADIOS2_COMPRESSION_ENABLE, ADIOS2_BLOSC_COMPRESSOR))
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


def _set_values_for_domains(edit, n, max_dom):
    """The n per-domain strings a 'set' edit writes, from either its scalar 'value' (the default,
    unchanged: one value repeated on every active domain) or its optional 'values_by_domain'.

    'values_by_domain' is a LIST (d01 first) or a MAPPING from 1-BASED domain index to value; JSON
    object keys arrive as strings, so "1"/1 are both accepted. It exists because a nest does not
    always want its parent's value: SC26-P1's d01 carries 3x its d02's acoustic margin at the same
    dt (parent_time_step_ratio=1 -- see predictor/stability.py), so the tuned time_step_sound there
    is a per-domain pair, not one number written twice.

    Validated against the case's own domain count: `max_dom` when the case has one (which is what
    the caller always passes for a real case), else the number of active entries `n` found in the
    namelist. A length/index mismatch RAISES -- silently padding or truncating a per-domain list is
    how a nest ends up at a value nobody chose."""
    vbd = edit.get("values_by_domain")
    if vbd is None:
        return [str(edit["value"])] * n
    key = edit["key"]
    expected = int(max_dom) if max_dom else n
    if isinstance(vbd, dict):
        index = {}
        for k, v in vbd.items():
            try:
                index[int(k)] = v
            except (TypeError, ValueError):
                raise ValueError(
                    "physics template: key %r values_by_domain has a non-integer domain index %r "
                    "(keys are 1-based domain numbers)" % (key, k))
        outside = sorted(d for d in index if d < 1 or d > expected)
        if outside:
            raise ValueError(
                "physics template: key %r values_by_domain names domain(s) %s outside this case's "
                "%d active domain(s)" % (key, outside, expected))
        missing = [d for d in range(1, expected + 1) if d not in index]
        if missing:
            raise ValueError(
                "physics template: key %r values_by_domain is missing domain(s) %s of this case's "
                "%d active domain(s)" % (key, missing, expected))
        ordered = [index[d] for d in range(1, expected + 1)]
    else:
        ordered = list(vbd)
        if len(ordered) != expected:
            raise ValueError(
                "physics template: key %r values_by_domain has %d value(s) for this case's %d "
                "active domain(s)" % (key, len(ordered), expected))
    if len(ordered) != n:
        raise ValueError(
            "physics template: key %r values_by_domain has %d value(s) but only %d namelist "
            "entr(ies) are active for this key -- add the missing per-domain entries to the "
            "namelist first" % (key, len(ordered), n))
    return [str(v) for v in ordered]


def case_frozen_keys(case):
    """The namelist keys this case's TASK MANDATES (cases.json 'frozen_keys'), lowercased. A
    competition problem statement can fix a parameter outright -- SC26-P1 mandates time_step=10 and
    debug_level=50 -- and a recommendation that changes one is a RULES VIOLATION, not a tuning
    trade-off, so no physics template may touch it. Empty list when the case declares none."""
    return [str(k).strip().lower() for k in (case or {}).get("frozen_keys") or []]


def template_frozen_key_conflicts(template, frozen_keys, case_id=None):
    """The frozen keys `template` would write, in edit order, duplicates removed. Looks at
    template['edits'] plus template['extra_edits'][case_id] -- the same edit list
    apply_physics_template itself applies. Empty list when nothing conflicts."""
    frozen = {str(k).strip().lower() for k in frozen_keys or ()}
    if not frozen:
        return []
    edits = list((template or {}).get("edits") or [])
    if case_id is not None:
        edits += list(((template or {}).get("extra_edits") or {}).get(case_id, []))
    hit = []
    for e in edits:
        k = str(e.get("key", "")).lower()
        if k in frozen and k not in hit:
            hit.append(k)
    return hit


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
        # per_domain: a max_domains key that is ABSENT (every domain at the Registry default, e.g.
        # time_step_sound=0 -> auto) must be written for every active domain, or the nests keep the
        # default while only d01 changes (2026-09-16, D01/D03 dynamics templates).
        # values_by_domain is per-domain by construction, so it implies per_domain on insert.
        per_domain = edit.get("per_domain") or edit.get("values_by_domain") is not None
        n_insert = max_dom if (per_domain and max_dom) else 1
        new_active = _set_values_for_domains(edit, n_insert, max_dom if per_domain else None)
        line = " %s = %s,\n" % (key, ", ".join(new_active))
        return _insert_after_section_header(text, section, line), [], new_active
    values = found["values"]
    n_active = min(len(values), max_dom) if max_dom else len(values)
    if n_active == 0:
        raise ValueError("physics template: key %r has no value(s) to edit" % (key,))
    active_old = values[:n_active]
    trailing = values[n_active:]
    if op == "set":
        active_new = _set_values_for_domains(edit, n_active, max_dom)
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


def apply_physics_template(text, template, max_dom, case_id=None, frozen_keys=None):
    """Apply every edit in template['edits'] to `text`, in order, followed by
    template['extra_edits'][case_id] when case_id is given (per-case paired
    edits, e.g. lafe's aer_opt=0 next to a scheme swap). Returns
    (new_text, edits_summary) where edits_summary is a list of
    {'key','op','old','new'} dicts (old/new are None for 'delete'), in the
    same order as the edits were applied. Raises ValueError
    (propagated from apply_physics_edit) if any edit cannot be applied --
    callers must not use a partially-applied `text` from a raised call.

    `frozen_keys` (cases.json 'frozen_keys', see case_frozen_keys) are keys the
    TASK mandates for this case: the whole template is REFUSED before a single
    edit is applied when it would write one of them, because changing a mandated
    parameter is a rules violation, not a tuning trade-off."""
    summary = []
    edits = list(template.get("edits", []))
    if case_id is not None:
        edits += list((template.get("extra_edits") or {}).get(case_id, []))
    conflicts = template_frozen_key_conflicts(template, frozen_keys, case_id=case_id)
    if conflicts:
        raise ValueError(
            "physics template: refusing to edit task-mandated frozen key(s) %s for case %s -- "
            "cases.json lists them in 'frozen_keys', i.e. the task fixes their value and changing "
            "one is a rules violation, not a tuning trade-off"
            % (", ".join(conflicts), case_id if case_id is not None else "<unnamed>"))
    for edit in edits:
        text, old, new = apply_physics_edit(text, edit, max_dom)
        summary.append(dict(key=edit["key"], op=edit["op"], old=old, new=new))
    return text, summary


def physics_template_applicability(case_id, template, frozen_keys=None):
    """(applicable: bool, reason: str or None) for one template on one case,
    from the template's optional 'applies_to' allowlist, its 'skip_cases' map,
    and the case's task-mandated `frozen_keys` -- does NOT attempt the edits
    themselves (see resolve_physics_config for that; a template can be
    'applicable' here and still fail to resolve, e.g. a 'mul' on a
    physics_suite -1 value)."""
    applies_to = template.get("applies_to")
    if applies_to is not None and case_id not in applies_to:
        return False, "not in applies_to %s" % (applies_to,)
    skip_reason = (template.get("skip_cases") or {}).get(case_id)
    if skip_reason is not None:
        return False, skip_reason
    conflicts = template_frozen_key_conflicts(template, frozen_keys, case_id=case_id)
    if conflicts:
        return False, ("edits task-mandated frozen key(s) %s for case %s: the task FIXES their "
                       "value, so changing one is a rules violation, not a tuning trade-off"
                       % (", ".join(conflicts), case_id))
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
    frozen = case_frozen_keys(case)
    ok, reason = physics_template_applicability(case_id, template, frozen_keys=frozen)
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
        _new_text, summary = apply_physics_template(text, template, max_dom, case_id=case_id,
                                                    frozen_keys=frozen)
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
                   tile_strategy=None, restart_from=None, restart_io_form=None,
                   write_restart_interval=None, run_seconds=None):
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
    as nproc_x/nproc_y/numtiles.

    restart_from/restart_io_form/write_restart_interval (WTune probe multi-point redesign,
    2026-09-26 -- see wtune_probe_plan.py's module docstring and this task's report for why):
    all three default to None, which reproduces byte-identical output to before they existed.
    restart_from (a datetime) makes this a RANKING-PROBE run that starts from an existing WRF
    restart file instead of from the case's own t0: every active domain's start_year/month/day/
    hour/minute/second is overridden to restart_from (see _apply_restart_from -- the per-domain
    list rewrite `overrides`/--set cannot do, since _set_scalar only ever touches the FIRST
    comma-separated value) and restart is forced to '.true.'. write_restart_interval (minutes)
    is the OPPOSITE direction: the ONE anchor spin-up run that WRITES the shared restart file at
    t0+spinup -- restart_interval is set to exactly this value (restart itself is left untouched,
    i.e. still '.false.': this run starts at t0 like any other). restart_io_form (e.g. 2 for
    netCDF -- decomposition-independent, unlike ADIOS2's io_form 14) overrides io_form_restart on
    EITHER kind of run. All three are applied AFTER `overrides` and AFTER _apply_adios2 (last),
    the same "never let a stale default win" ordering _apply_adios2's own docstring already uses
    for `overrides` -- adios2=True's own io_form_restart=14/restart_interval=1000000 would
    otherwise silently undo exactly what these three parameters are for."""
    text = _set_scalar(text, "run_days", 0)
    text = _set_scalar(text, "run_hours", 0)
    text = _set_scalar(text, "run_minutes", run_minutes)
    # run_seconds too (decision 0055): WRF ADDS all four run_* keys, so a template carrying a
    # non-zero run_seconds (every coldstart/rebuild real/wrf-window namelist sets it -- see
    # tools/coldstart_sample.build_real_namelist_text) would silently lengthen the run past
    # run_minutes. _set_scalar is a no-op when the key is absent, so a template without it (or
    # with run_seconds=0 already) is byte-identical to before.
    text = _set_scalar(text, "run_seconds", 0)
    # run_seconds (opt-in, default None = unchanged): an exact sub-minute tail added to run_minutes
    # (WRF ADDS the run_* keys). Used by POINT_FUSED / WINDOW_EXACT so the window is STEPS*dt
    # exactly instead of rounded UP to a whole minute. Only sound with history_mode 'off' (the
    # history interval keys follow run_minutes, not the seconds tail).
    if run_seconds:
        text = _set_or_insert_time_control_scalar(text, "run_seconds", int(run_seconds))
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
        # start: per-record BP inputs need a record at the time the run really starts
        text = _apply_adios2(text, ranks, max_dom, start=restart_from)
    if restart_from is not None:
        text = _apply_restart_from(text, restart_from, max_dom)
    if write_restart_interval is not None:
        text = _set_or_insert_time_control_scalar(text, "restart_interval", write_restart_interval)
    if restart_io_form is not None:
        text = _set_or_insert_time_control_scalar(text, "io_form_restart", restart_io_form)
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


_TIMING_D01_RE = re.compile(
    r"Timing for main: time (\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)\s+on domain\s+1:\s+([0-9.]+) elapsed")


def post_spinup_steps(rsl_text, spin_seconds, dt):
    """POINT_FUSED: the d01 "Timing for main" values of the steps AFTER the first `spin_seconds` of
    model time, in order. One wrf.exe runs spin-up + measurement window from t0; its rsl.out
    stamps each line with the model time reached (step k of a t0 start prints t0 + k*dt), so
    elapsed model time of line j is (stamp_j - stamp_1) + dt and a step belongs to the window iff
    that exceeds spin_seconds. Feed the result to step_stats() unchanged: it drops values[0] and
    excludes >3x-median (radiation/history) steps exactly as for a restart run. Returns [] when
    no line is past the spin-up."""
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


def resolve_point_plan(order, point_fused=False, anchor_end=None, window_frame=False):
    """Decide a prediction-point job's run list from its ORDER (config ids, ANCHOR first and
    ANCHOREND last when the orchestrator wrapped it).
      anchor_end None -> default: ON, except OFF in fused mode; True/False explicit.
      ANCHOREND entries are removed when anchor_end is off ("drift gate off").
      Fused (POINT_FUSED=1) applies only when, after removing ANCHOR/ANCHOREND, exactly ONE run
      remains (a winner-only plan; with several configs or REPS>1 the shared-restart comparison is
      the point, so fusing would silently change what is measured) and no window frame is wanted.
    Returns dict(order, fused, cid, anchor_end, note). In fused mode ANCHOR is dropped too (its only
    job, writing the restart, is what fusing removes) and an explicit anchor_end=True is refused
    (the drift gate needs the restart hand-off): anchor_end comes back False with a note."""
    order = list(order)
    note = ""
    cands = [c for c in order if c not in ("ANCHOR", "ANCHOREND")]
    if not cands and order:
        cands = [c for c in order if c == "ANCHOR"]
    fused = bool(point_fused) and len(cands) == 1 and not window_frame
    if point_fused and not fused:
        note = "POINT_FUSED ignored: %s" % (
            "window_frame requested" if window_frame and len(cands) == 1
            else "%d runs after dropping anchors (need exactly 1)" % len(cands))
    if fused:
        if anchor_end is True:
            note = "ANCHOR_END=1 ignored in fused mode (no restart to bracket)"
        return dict(order=cands, fused=True, cid=cands[0], anchor_end=False, note=note)
    if anchor_end is None:
        anchor_end = True
    if not anchor_end:
        order = [c for c in order if c != "ANCHOREND"]
    return dict(order=order, fused=False, cid="", anchor_end=bool(anchor_end), note=note)


def _cmd_fused_steps(argv):
    ap = argparse.ArgumentParser(prog="configs.py fused-steps")
    ap.add_argument("rsl_file")
    ap.add_argument("--spin-seconds", type=float, required=True)
    ap.add_argument("--dt", type=float, required=True)
    a = ap.parse_args(argv)
    try:
        with open(a.rsl_file, "rb") as f:
            text = f.read().decode("ascii", "ignore")
    except IOError:
        text = ""
    for v in post_spinup_steps(text, a.spin_seconds, a.dt):
        print("%s" % v)
    return 0


def _cmd_point_plan(argv):
    """configs.py point-plan --fused 0|1 --anchor-end ''|0|1 [--window-frame 0|1] ID...
    prints shell assignments POINT_ORDER=(...) POINT_FUSED_ON=0|1 POINT_ANCHOR_END=0|1 POINT_NOTE='..'"""
    ap = argparse.ArgumentParser(prog="configs.py point-plan")
    ap.add_argument("--fused", default="0")
    ap.add_argument("--anchor-end", default="")
    ap.add_argument("--window-frame", default="0")
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args(argv)
    ae = None if a.anchor_end == "" else (a.anchor_end != "0")
    r = resolve_point_plan(a.ids, a.fused == "1", ae, a.window_frame == "1")
    print("POINT_ORDER=(%s)" % " ".join("'%s'" % c for c in r["order"]))
    print("POINT_FUSED_ON=%d" % (1 if r["fused"] else 0))
    print("POINT_ANCHOR_END=%d" % (1 if r["anchor_end"] else 0))
    print("POINT_NOTE='%s'" % r["note"].replace("'", ""))
    return 0


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
    _src = os.path.join(REPO_ROOT, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
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


def cmd_restart_time(case_id, spinup_minutes):
    """Prints (t0 + spinup_minutes), 'YYYY-MM-DD_HH:MM:SS' -- the RANKING-restart's absolute
    time (2026-09-26, end-to-end wiring): campaign/strict_campaign.sbatch's RANKING_START=spinup
    block calls this once per job (not per candidate) to get the --restart-from value every
    candidate's second edit-namelist pass uses. t0 is read from the case's own namelist template
    (case_start_datetime), same file edit_namelist itself edits -- so the reported time always
    matches whatever this case's namelist actually starts at, even if cases.json's own metadata
    (not read here) disagreed."""
    case = CASES.get(case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % case_id)
        return 2
    namelist_template = case.get("namelist_template") or (case["prep_dir"] + "/namelist.input")
    with open(namelist_template) as f:
        text = f.read()
    start = case_start_datetime(text)
    restart_time = start + datetime.timedelta(minutes=float(spinup_minutes))
    print(restart_time.strftime("%Y-%m-%d_%H:%M:%S"))
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


def config_extras_rows(configs_list, case_id):
    """(config_id, physics_namelist, window_frame) per config, for strict_campaign.sbatch's
    CONFIG_FILE loop. Two optional per-config JSON fields the decomposition-only TSV above cannot
    carry (decision 0059, `wtune tune --pipeline full`):
      physics_namelists  {point case id: namelist path}. When it names `case_id`, that run uses
                         the file as its namelist template (physics edits already rendered by
                         wtune_physics_advice.render_candidate_namelist /
                         wtune_run.render_segment_namelist) instead of the case's own template;
                         decomposition/history/restart are still applied on top per config.
                         Empty string = no override for this case.
      window_frame       true / "end": this config's run writes exactly one d01 history frame at
                         the window end; "mid_end": two frames, window midpoint and window end
                         (fidelity_gate.run_gate returns INVALID on files holding only ONE valid
                         time -- it needs a matched time after the reference's first -- so a gate
                         input is the sorted list of the run's mid+end frames; a restarted run never writes a
                         frame at its own start time). Frames are
                         converted to NetCDF and kept in the run directory.
    Returns window_frame as the level '0' (off) / '1' (end) / '2' (mid+end)."""
    rows = []
    for cfg in configs_list:
        nls = cfg.get("physics_namelists") or {}
        rows.append((cfg.get("id"), str(nls.get(case_id) or ""),
                     str(window_frame_level(cfg.get("window_frame")))))
    return rows


def window_frame_level(value):
    """0 off, 1 end frame only, 2 mid+end frames. Accepts bools, 0/1/2, 'end', 'mid_end'."""
    if value in (None, False, 0, "0", "", "off", "false", "False"):
        return 0
    if value in (2, "2", "mid_end"):
        return 2
    return 1


def cmd_config_extras(case_id, config_file):
    """`configs.py config-extras CASE_ID`: TSV config_id, physics_namelist, window_frame for every
    config in CONFIG_FILE, in file order (see config_extras_rows)."""
    for row in config_extras_rows(load_generated_configs(config_file), case_id):
        print("\t".join("" if v is None else str(v) for v in row))
    return 0


# ---------------------------------------------------------------------------
# MODE=physics decomposition pair (2026-09-17): the reference decomposition and
# the chaos-floor twin, resolved from ONE rule set whatever they are named in.
#
# Until now strict_campaign.sbatch's MODE=physics could only name fixed STRICT
# template ids (DECOMP=C02 TWIN_DECOMP=C06), and sc26p1 resolves NONE of C01-C16:
# its d02 has 100 cells, WRF needs floor(e/nproc) >= MIN_PATCH on every active
# domain, so BOTH domains are capped at 10x10 = 100 ranks through the shared
# decomposition while every template is anchored at 448/224/112 ranks --
# `configs.py sc26p1` prints 16 SKIP lines and an empty table. PHYSICS-GRAY was
# therefore unreachable for sc26p1, and for any case whose legal rank cap is
# below 112.
#
# The fix reuses the CONFIG_FILE mechanism that already exists for the STRICT
# loop rather than inventing a second one: with
# CONFIG_FILE=<predictor/decomp_space.py sample|seed output>, DECOMP and
# TWIN_DECOMP name ids from that file (G0001, S0003, R01D0007, ...) instead of
# C01-C16, and cmd_case_generated already prints those configs in the TSV shape
# the sbatch reads. Nothing here changes what a STRICT-template pair resolves to:
# with CONFIG_FILE unset, resolve_decomp_pair looks the ids up in exactly the
# resolve_case() output cmd_case prints, and the sbatch's own pre-existing
# "not a resolved STRICT config" guards are left in place ahead of it.
#
# The twin is NOT a replicate. It is the same physics (P00) on a DIFFERENT
# decomposition, and the reference-vs-twin spread is the chaos floor that
# fidelity_gate.py compares every candidate against. An identical twin would
# measure a floor of ~0 and relabel every real candidate LARGE-DRIFT, so an
# identical pair is refused here instead of being measured.
# ---------------------------------------------------------------------------

#: What makes two decompositions genuinely different RUNS. patch_x/patch_y are derived
#: from nx/ny, and id/config_source/tile_shape are labels, so none of them is compared.
#: STRICT-resolved configs carry no tile-geometry keys at all and normalise to 0 here,
#: which is exactly what the sbatch passes for them (TILE_ARGS default 0).
DECOMP_IDENTITY_FIELDS = ("ranks", "omp", "nx", "ny", "numtiles", "dist", "blocktime",
                          "numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy")


def _decomp_identity(cfg):
    out = []
    for k in DECOMP_IDENTITY_FIELDS:
        v = cfg.get(k)
        out.append("0" if v is None or v == "" else str(v))
    return tuple(out)


def describe_decomp(cfg):
    """One-line human form of a decomposition, for error messages and the job log."""
    return ("ranks=%s omp=%s nproc_x=%s nproc_y=%s numtiles=%s dist=%s blocktime=%s"
            % (cfg.get("ranks"), cfg.get("omp"), cfg.get("nx"), cfg.get("ny"),
               cfg.get("numtiles"), cfg.get("dist"), cfg.get("blocktime")))


def available_decomps(case_id, config_file=None, cases=None):
    """(ordered_ids, {id: config}, None) or (None, None, reason).

    config_file set -> the generated set, in file order, no scoring/dedup (same source
    cmd_case_generated prints). Unset -> resolve_case()'s kept STRICT configs, i.e.
    exactly the ids cmd_case prints for this case."""
    cases = CASES if cases is None else cases
    case = cases.get(case_id)
    if case is None:
        return None, None, "unknown case %r" % (case_id,)
    if not case.get("enabled", True):
        return None, None, ("case %s is disabled: %s"
                            % (case_id, case.get("reason", "(no reason recorded)")))
    if config_file:
        ids, by_id = [], {}
        for c in load_generated_configs(config_file) or ():
            if c.get("id") is None:
                continue
            cid = str(c["id"])
            if cid not in by_id:
                ids.append(cid)
            by_id[cid] = c
        if not ids:
            return None, None, ("CONFIG_FILE %s carries no configs with an 'id' field "
                                "(expected predictor/decomp_space.py sample|seed output: a bare "
                                "JSON list or {\"configs\": [...]})" % (config_file,))
        return ids, by_id, None
    kept, skipped, _aliases = resolve_case(case)
    if not kept:
        first = skipped[0][1] if skipped else "no reason recorded"
        return None, None, (
            "case %s resolves ZERO of the %d STRICT templates C01-C16 (first reason: %s), so no "
            "template id can name a decomposition for it. Generate a decomposition set instead -- "
            "predictor/decomp_space.py sample %s --n N --seed S --total-cores C --nodes N "
            "--out FILE.json -- and run with CONFIG_FILE=FILE.json plus DECOMP/TWIN_DECOMP ids "
            "taken from that file" % (case_id, len(TEMPLATES), first, case_id))
    return [c["id"] for c in kept], dict((c["id"], c) for c in kept), None


def resolve_decomp_pair(case_id, decomp_id, twin_id, config_file=None, cases=None):
    """(reference_cfg, twin_cfg, None) or (None, None, reason).

    The single resolution+validation rule set behind MODE=physics's DECOMP/TWIN_DECOMP,
    for both STRICT template ids (config_file=None) and a generated config set."""
    ids, by_id, reason = available_decomps(case_id, config_file, cases)
    if reason:
        return None, None, reason
    source = ("CONFIG_FILE=%s" % config_file) if config_file else "the STRICT templates C01-C16"
    picked = []
    for role, cid in (("DECOMP", decomp_id), ("TWIN_DECOMP", twin_id)):
        cid = "" if cid is None else str(cid)
        if not cid:
            return None, None, "%s is empty: name one of %s" % (role, ", ".join(ids))
        if "_" in cid or cid.split() != [cid]:
            return None, None, (
                "%s=%r contains an underscore or whitespace. A config id is embedded verbatim in "
                "the ledger run_id (case_cid_rN_job, parsed by predictor/decomp_search.py "
                "parse_run_id) and in the space-separated key=value notes field, so neither "
                "character can appear in it" % (role, cid))
        cfg = by_id.get(cid)
        if cfg is None:
            return None, None, (
                "%s=%r is not a resolved config for case %s from %s -- available ids: %s"
                % (role, cid, case_id, source, ", ".join(ids)))
        picked.append(cfg)
    ref, twin = picked
    if _decomp_identity(ref) == _decomp_identity(twin):
        return None, None, (
            "TWIN_DECOMP=%s is the SAME decomposition as DECOMP=%s (%s). The twin is not a "
            "replicate: it is the same physics on a DIFFERENT decomposition, and the "
            "reference-vs-twin spread is the chaos floor fidelity_gate.py measures every "
            "candidate against. An identical twin reports a floor of ~0 and would relabel every "
            "real candidate LARGE-DRIFT, so pick a twin differing in at least one of: %s"
            % (twin_id, decomp_id, describe_decomp(ref), ", ".join(DECOMP_IDENTITY_FIELDS)))
    return ref, twin, None


#: role + the same 15 value columns cmd_case_generated prints (minus patch_x/patch_y,
#: which nothing downstream of the pair consumes).
_DECOMP_PAIR_COLUMNS = ("role", "config_id", "ranks", "omp", "nproc_x", "nproc_y", "numtiles",
                        "dist", "blocktime") + _GENERATED_TILE_FIELDS + ("config_source",)


def cmd_decomp_pair(argv):
    """Validate (and print) the MODE=physics reference/twin decomposition pair.

    rc=0 + a 2-row TSV on stdout, or rc=2 + one 'DECOMP-PAIR ERROR: <reason>' line on
    stderr. strict_campaign.sbatch calls this as a fail-fast gate before the first WRF
    srun, so an unusable pair costs a few seconds instead of a whole allocation."""
    ap = argparse.ArgumentParser(prog="configs.py decomp-pair")
    ap.add_argument("case_id")
    ap.add_argument("decomp")
    ap.add_argument("twin_decomp")
    ap.add_argument("--config-file", default=None,
                    help="generated config set (predictor/decomp_space.py sample|seed output); "
                         "default: the CONFIG_FILE env var, and unset means the STRICT templates")
    a = ap.parse_args(argv)
    config_file = a.config_file or os.environ.get("CONFIG_FILE")
    ref, twin, reason = resolve_decomp_pair(a.case_id, a.decomp, a.twin_decomp, config_file)
    if reason:
        sys.stderr.write("DECOMP-PAIR ERROR: %s\n" % reason)
        return 2
    print("\t".join(_DECOMP_PAIR_COLUMNS))
    for role, cfg in (("reference", ref), ("twin", twin)):
        row = [role, cfg.get("id"), cfg.get("ranks"), cfg.get("omp"), cfg.get("nx"), cfg.get("ny"),
               cfg.get("numtiles"), cfg.get("dist"), cfg.get("blocktime")]
        row += [cfg.get(k, 0) for k in _GENERATED_TILE_FIELDS]
        row.append(cfg.get("config_source", "strict_template"))
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
    ap.add_argument("--history-mode", choices=["off", "interval", "end_d01", "mid_end_d01"], default="off",
                     help="'off' (default, timing runs): no domain writes history. "
                          "'interval' (PHYSICS-GRAY fidelity replicates): d01 writes a "
                          "frame every --history-minutes, starting at t=0; requires "
                          "--history-minutes. 'end_d01': d01 writes only its final "
                          "frame (correctness runs); nested domains always get 0. "
                          "'mid_end_d01': d01 writes two frames, at run_minutes//2 and at the "
                          "end (window-frame fidelity-gate input).")
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
    ap.add_argument("--restart-from", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
                     help="WTune probe multi-point (2026-09-26): this is a RANKING-PROBE run "
                          "that starts from an existing WRF restart file at this time instead "
                          "of the case's own t0 -- every domain's start_* is overridden to it "
                          "and restart is forced to .true. (see edit_namelist's docstring). "
                          "Not written at all when omitted (default: unchanged behaviour).")
    ap.add_argument("--restart-io-form", type=int, default=None, metavar="N",
                     help="overrides io_form_restart on EITHER a --restart-from run or a "
                          "--write-restart-interval run, applied after --no-adios2/ADIOS2's own "
                          "default (e.g. 2 for netCDF, decomposition-independent -- unlike "
                          "ADIOS2's io_form 14). Not written at all when omitted.")
    ap.add_argument("--write-restart-interval", type=int, default=None, metavar="MINUTES",
                     help="the ANCHOR spin-up run that WRITES the shared restart file: sets "
                          "restart_interval to exactly this value (restart itself is left "
                          "'.false.' -- this run still starts at t0). Not written at all when "
                          "omitted.")
    ap.add_argument("--run-seconds", type=int, default=0, metavar="S",
                     help="exact sub-minute tail added to --run-minutes (POINT_FUSED / "
                          "WINDOW_EXACT: window = STEPS*dt, not rounded up to whole minutes). "
                          "Only with --history-mode off. Not written when 0 (default).")
    a = ap.parse_args(argv)
    if a.run_seconds and a.history_mode != "off":
        ap.error("--run-seconds requires --history-mode off")
    if a.adios2 and a.ranks is None:
        ap.error("--ranks is required unless --no-adios2 is given")
    if a.history_mode == "interval" and a.history_minutes is None:
        ap.error("--history-minutes is required when --history-mode interval is given")
    restart_from_dt = None
    if a.restart_from is not None:
        try:
            restart_from_dt = datetime.datetime.strptime(a.restart_from, "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            ap.error("--restart-from must be YYYY-MM-DD_HH:MM:SS, got %r" % a.restart_from)
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
                              tile_strategy=a.tile_strategy, restart_from=restart_from_dt,
                              restart_io_form=a.restart_io_form,
                              write_restart_interval=a.write_restart_interval,
                              run_seconds=a.run_seconds)
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


# ---------------------------------------------------------------------------
# fidelity_gate.py JSON -> the four fields strict_campaign.sbatch MODE=physics puts in the ledger
# `notes` column. The 48-column LEDGER_HEADER is depended on by many tests and by every existing
# campaign ledger CSV on disk, so the MEASURED numeric error is carried as extra `key=value`
# tokens inside `notes` (which already carries mode=/template=/edits=/decomp=/... tokens) rather
# than as new columns -- predictor/ledger.py's parse_fidelity_notes() is the reader.
#
# This lived twice as an inline heredoc inside strict_campaign.sbatch (the twin-sanity arm and the
# candidate arm, byte-identical copies). It is a pure function here so it can be unit-tested
# without running the sbatch; the output line format is unchanged, so the notes format the
# existing campaign ledgers were written with still parses.
# ---------------------------------------------------------------------------
FIDELITY_TOKEN_NA = "NA"


def fidelity_max_over_fields(payload, key):
    """MEASURED max of `key` over every field in fidelity_gate.py's
    per_field_summary_at_last_time (i.e. at the last common time, not a threshold), formatted
    '%.6g'; FIDELITY_TOKEN_NA when no field carries a numeric value for it. Booleans are
    rejected explicitly: `floor_zero` is a bool and `isinstance(True, int)` is True in Python,
    so a bare numeric check would let a flag through as 1/0 if a key were ever mistyped."""
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


def fidelity_notes_tokens(payload):
    """(label, nd_max, floor_ratio_max, deciding) as 4 whitespace-free strings, for the
    `read -r FID_LABEL FID_ND FID_FLOOR FID_DECIDING` in strict_campaign.sbatch MODE=physics.
    Every field falls back to FIDELITY_TOKEN_NA, and `deciding` has its spaces replaced by '_'
    so the whole record stays four shell words and each ledger note token stays one token."""
    label = str(payload.get("label") or FIDELITY_TOKEN_NA)
    deciding = str(payload.get("deciding") or FIDELITY_TOKEN_NA).replace(" ", "_")
    return (label, fidelity_max_over_fields(payload, "nd"),
            fidelity_max_over_fields(payload, "floor_ratio"), deciding)


def cmd_fidelity_tokens(argv):
    """fidelity-tokens FIDELITY_JSON: print the one line strict_campaign.sbatch reads into
    FID_LABEL/FID_ND/FID_FLOOR/FID_DECIDING. Exits 2 without printing anything when the JSON is
    missing or unreadable, so the caller's `read` leaves those variables empty -- which is exactly
    the no-fidelity-JSON behaviour (notes carry `fidelity_nd_max=` with an empty value)."""
    ap = argparse.ArgumentParser(prog="configs.py fidelity-tokens")
    ap.add_argument("fidelity_json")
    a = ap.parse_args(argv)
    try:
        with open(a.fidelity_json) as f:
            payload = json.load(f)
    except (IOError, OSError, ValueError) as e:
        sys.stderr.write("cannot read fidelity JSON %s: %s\n" % (a.fidelity_json, e))
        return 2
    print(" ".join(fidelity_notes_tokens(payload)))
    return 0


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


# ---------------------------------------------------------------------------
# files the namelist names (staging into a run dir)
# ---------------------------------------------------------------------------
# A run dir used to be staged from the case's input_files plus wrfinput/wrfbdy only; files the
# NAMELIST references were never staged. climate-ea50 (2026-10-02, scc probe jobs 2541-2545): the
# case sets iofields_filename = "iofields_d01.txt", the probe dirs lacked it, and WRF 4.8 hung
# forever after "med_initialdata_input: calling input_input" (ignore_iofields_warning=.true. does
# not help) -- every probe ran into its 1260 s timeout with 0 steps. WRF expands "<domain>" to the
# 2-digit domain id; a name still carrying "<date>" (or any other <...>) is globbed.
_NML_REF_SKIP_AUXINPUT = (1, 4)   # 1 = met_em (real.exe only); 4 = wrflowinp, via sst_update


def _namelist_string_values(text, key):
    """Value tokens of the LAST non-comment assignment of `key` (case-insensitive), quotes
    stripped; None when the key is not assigned. Stops at an unquoted '!' or at a token that
    holds '=' (the next key on a packed line)."""
    pat = re.compile(r"(?<![A-Za-z0-9_])%s[ \t]*=[ \t]*" % re.escape(key), re.I)
    found = None
    for line in text.splitlines():
        if line.lstrip().startswith("!"):
            continue
        m = None
        for m in pat.finditer(line):
            pass
        if m is None:
            continue
        rest = line[m.end():]
        vals, i, n = [], 0, len(rest)
        while i < n:
            c = rest[i]
            if c in " \t,":
                i += 1
            elif c in "'\"":
                j = rest.find(c, i + 1)
                j = n if j == -1 else j
                vals.append(rest[i + 1:j])
                i = j + 1
            elif c == "!":
                break
            else:
                j = i
                while j < n and rest[j] not in " \t,!":
                    j += 1
                tok = rest[i:j]
                if "=" in tok:
                    break
                vals.append(tok)
                i = j
        found = vals
    return found


def _namelist_ints(text, key):
    out = []
    for v in _namelist_string_values(text, key) or []:
        try:
            out.append(int(v.split("*")[-1]))
        except ValueError:
            out.append(0)
    return out


def namelist_file_refs(text):
    """[(key, name, required), ...] files wrf.exe opens because the namelist names them
    ('<domain>' resolved per active domain; '<date>' left in -- a glob pattern):
      iofields_filename (per domain; not NONE_SPECIFIED)              required
      auxinput4_inname (default wrflowinp_d<domain>) when sst_update=1 required
      gfdda_inname (wrffdda_d<domain>) for domains with grid_fdda>0    required
      sgfdda_inname (wrfsfdda_d<domain>) for domains with grid_sfdda>0 required
      auxinput<N>_inname set in the namelist (N != 1, 4)               staged when present
      OBS_DOMAIN<d>* for domains with obs_nudge_opt=1                  staged when present
      tslist (wrf.exe reads it when present)                           staged when present"""
    max_dom = (_namelist_ints(text, "max_dom") or [1])[0] or 1
    doms = range(1, max_dom + 1)

    def per_dom(values, d):          # WRF's per-domain arrays: an absent entry keeps the default
        return values[d - 1] if values and d - 1 < len(values) else None

    def expand(name):
        name = name.strip()
        return ([name.replace("<domain>", "%02d" % d) for d in doms] if "<domain>" in name
                else [name])

    refs = []

    def add(key, names, required):
        for nm in names:
            if nm and (key, nm, required) not in refs:
                refs.append((key, nm, required))

    io = _namelist_string_values(text, "iofields_filename") or []
    for d in doms:
        v = per_dom(io, d)
        if v and v.strip().upper() != "NONE_SPECIFIED":
            add("iofields_filename", [v.strip().replace("<domain>", "%02d" % d)], True)
    if (_namelist_ints(text, "sst_update") or [0])[0] == 1:
        nm = (_namelist_string_values(text, "auxinput4_inname") or ["wrflowinp_d<domain>"])[0]
        add("auxinput4_inname", expand(nm), True)
    for flag, key, default in (("grid_fdda", "gfdda_inname", "wrffdda_d<domain>"),
                               ("grid_sfdda", "sgfdda_inname", "wrfsfdda_d<domain>")):
        on = _namelist_ints(text, flag)
        nm = (_namelist_string_values(text, key) or [default])[0].strip()
        for d in doms:
            if (per_dom(on, d) or 0) > 0:
                add(key, [nm.replace("<domain>", "%02d" % d)], True)
    streams = re.findall(r"(?i)(?<![A-Za-z0-9_])auxinput(\d+)_inname", text)
    for m in sorted(set(int(x) for x in streams)):
        if m in _NML_REF_SKIP_AUXINPUT:
            continue
        v = _namelist_string_values(text, "auxinput%d_inname" % m)
        if v:
            add("auxinput%d_inname" % m, expand(v[0]), False)
    obs = _namelist_ints(text, "obs_nudge_opt")
    for d in doms:
        if per_dom(obs, d) == 1:
            add("obs_nudge_opt", ["OBS_DOMAIN%d*" % d], False)
    add("tslist", ["tslist"], False)
    return refs


def stage_namelist_files(text, run_dir, search_dirs):
    """Symlink every relative file namelist_file_refs(text) names that is absent from run_dir,
    from the first of search_dirs holding it ('sub/name' gets its parent dir created; an existing
    entry is never replaced; a '..' or absolute name is only checked, never staged). Returns
    {"linked": [(key, name, src)], "present": [(key, name)], "missing": [(key, name)]} --
    missing = required and found nowhere, left absent for the caller to report."""
    import glob as _glob
    out = {"linked": [], "present": [], "missing": []}
    # a name without .bp is a netCDF/text file: a directory of that name is a BP input or a
    # per-record container (BP cache: wrflowinp_d01/ next to the netCDF fallback's own
    # wrflowinp_d01 in prep_dir) and must not be staged in its place
    def usable(path, name):
        return name.endswith(".bp") or not os.path.isdir(path)
    dirs = []
    for s in search_dirs:
        if s and os.path.isdir(s) and os.path.abspath(s) not in dirs:
            dirs.append(os.path.abspath(s))
    for key, name, required in namelist_file_refs(text):
        pattern = re.sub(r"<[^>]*>", "*", name)
        rel = os.path.normpath(pattern)
        if os.path.isabs(rel) or rel.split(os.sep)[0] == "..":
            if _glob.glob(rel if os.path.isabs(rel) else os.path.join(run_dir, rel)):
                out["present"].append((key, name))
            elif required:
                out["missing"].append((key, name))
            continue
        if _glob.glob(os.path.join(run_dir, rel)):
            out["present"].append((key, name))
            continue
        srcs = []
        for d in dirs:
            srcs = sorted(p for p in _glob.glob(os.path.join(d, rel)) if usable(p, name))
            if srcs:
                break
        if not srcs:
            if required:
                out["missing"].append((key, name))
            continue
        for src in srcs:
            dst = os.path.join(run_dir, os.path.relpath(src, d))
            if os.path.lexists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
            out["linked"].append((key, os.path.relpath(src, d), src))
    return out


def format_namelist_stage(result, search_dirs):
    """Log lines for stage_namelist_files' result (missing ones as WARNING lines)."""
    lines = ["namelist file: linked %s (%s) from %s" % (n, k, s) for k, n, s in result["linked"]]
    where = " / ".join(d for d in search_dirs if d) or "(no search dir)"
    for k, n in result["missing"]:
        lines.append("WARNING: the namelist's %s names %s, which exists neither in the run dir "
                     "nor in %s -- left absent%s"
                     % (k, n, where, " (WRF 4.8 hangs at input_input when the iofields file is "
                                     "missing, even with ignore_iofields_warning=.true.)"
                        if k == "iofields_filename" else ""))
    return lines


# ---------------------------------------------------------------------------
# ADIOS2 input streams (2026-10-03 user decision: netCDF is no longer used for WRF I/O; every
# input stream -- wrfinput, wrfbdy, aux inputs -- is read as ADIOS2 BP, netCDF inputs are
# converted first). Shared by strict_campaign.sbatch (stage-bp-inputs) and wtune_run.
# ---------------------------------------------------------------------------
ADIOS2_IO_FORM = 14
# WRF's io_adios2 opens the file name exactly as the stream's *_inname template expands
# (external/io_adios2/wrf_io.F90 ext_adios2_open_for_read_begin: adios2_declare_io/adios2_open
# with FileName; no suffix is added), so every BP input is linked as "<netCDF name>.bp" and its
# inname is rewritten to "<template>.bp" -- the convention _apply_adios2 has always used for
# wrfinput/wrfbdy (input_inname="wrfinput_d<domain>.bp", bdy_inname="wrfbdy_d<domain>.bp").


def adios2_input_streams(text):
    """WRF input streams this namelist reads, in order:
      input     io_form_input,  input_inname (wrfinput_d<domain>)    every active domain (d01
                required; a nest without input_from_file has no file)
      boundary  io_form_boundary, bdy_inname (wrfbdy_d<domain>)      d01 only
      auxinput4 io_form_auxinput4, auxinput4_inname (wrflowinp_d<domain>)  when sst_update=1
      gfdda     io_form_gfdda (&fdda), gfdda_inname (wrffdda_d<domain>)    domains grid_fdda>0
      sgfdda    io_form_sgfdda (&fdda), sgfdda_inname (wrfsfdda_d<domain>) domains grid_sfdda>0
      auxinputN io_form_auxinputN, auxinputN_inname (as set)  N != 1, 4, inname set and the
                namelist's io_form_auxinputN > 0
    Defaults follow namelist_file_refs. Each: {stream, label, io_key, group, inname_key,
    template (no .bp), inname ("<basename(template)>.bp"), names ([expanded per domain; may still
    hold <date>]), required ([names that must exist])}."""
    max_dom = (_namelist_ints(text, "max_dom") or [1])[0] or 1
    doms = list(range(1, max_dom + 1))

    def tmpl(key, default):
        v = _namelist_string_values(text, key)
        t = (v[0].strip() if v and v[0].strip() else default)
        return t[:-3] if t.endswith(".bp") else t

    def expand(t, ds):
        return [t.replace("<domain>", "%02d" % d) for d in ds] if "<domain>" in t else [t]

    out = []

    def add(stream, label, io_key, group, inname_key, template, ds, required):
        names = expand(template, ds)
        out.append({"stream": stream, "label": label, "io_key": io_key, "group": group,
                    "inname_key": inname_key, "template": template,
                    "inname": os.path.basename(template) + ".bp", "names": names,
                    "required": names if required is None else required(names)})

    add("input", "input", "io_form_input", "time_control", "input_inname",
        tmpl("input_inname", "wrfinput_d<domain>"), doms, lambda n: n[:1])
    add("boundary", "boundary", "io_form_boundary", "time_control", "bdy_inname",
        tmpl("bdy_inname", "wrfbdy_d<domain>"), [1], None)
    if (_namelist_ints(text, "sst_update") or [0])[0] == 1:
        add("auxinput4", "aux4", "io_form_auxinput4", "time_control", "auxinput4_inname",
            tmpl("auxinput4_inname", "wrflowinp_d<domain>"), doms, None)
    for flag, stream, default in (("grid_fdda", "gfdda", "wrffdda_d<domain>"),
                                  ("grid_sfdda", "sgfdda", "wrfsfdda_d<domain>")):
        on = _namelist_ints(text, flag)
        ds = [d for d in doms if d - 1 < len(on) and on[d - 1] > 0]
        if ds:
            add(stream, stream, "io_form_" + stream, "fdda", stream + "_inname",
                tmpl(stream + "_inname", default), ds, None)
    for n in sorted(set(int(x) for x in re.findall(
            r"(?i)(?<![A-Za-z0-9_])auxinput(\d+)_inname", text))):
        if n in _NML_REF_SKIP_AUXINPUT or not (_namelist_ints(text, "io_form_auxinput%d" % n)
                                               or [0])[0]:
            continue
        t = tmpl("auxinput%d_inname" % n, "")
        if t:
            add("auxinput%d" % n, "aux%d" % n, "io_form_auxinput%d" % n, "time_control",
                "auxinput%d_inname" % n, t, doms, None)
    return out


# Multi-record input streams converted one BP per record (wtune_bp.PER_RECORD_NAME_RE): stream ->
# (file kind, io_form key, group, inname key, netCDF default inname, frames_per key or None).
PER_RECORD_STREAMS = {
    "boundary": ("wrfbdy", "io_form_boundary", "time_control", "bdy_inname",
                 "wrfbdy_d<domain>", None),
    "auxinput4": ("wrflowinp", "io_form_auxinput4", "time_control", "auxinput4_inname",
                  "wrflowinp_d<domain>", "frames_per_auxinput4"),
    "gfdda": ("wrffdda", "io_form_gfdda", "fdda", "gfdda_inname", "wrffdda_d<domain>",
              "frames_per_auxinput10"),
}
_PER_RECORD_NAME_RE = re.compile(r"^(wrfbdy|wrflowinp|wrffdda)_d\d\d$")
WRF_DATE_FMT = "%Y-%m-%d_%H:%M:%S"


def per_record_has(bp_dir, name, when):
    """True when bp_dir holds the record file <name>_<when>.bp WRF opens at `when` (WRF date
    string): inside the container <bp_dir>/<name>[.bp]/ or as a flat link <bp_dir>/<rec>."""
    rec = "%s_%s.bp" % (name, when)
    if not bp_dir:
        return False
    return any(os.path.isdir(os.path.join(bp_dir, *parts))
               for parts in ((name, rec), (name + ".bp", rec), (rec,)))


def per_record_streams(names, start, text, bp_dir, aux_ok=None):
    """Split the per-record input names (WTUNE_PER_RECORD_BP: 'wrfbdy_d01 wrflowinp_d01 ...')
    by stream into ({stream: [names]} readable per record from the run start,
    {stream: [missing record files]}) -- a stream is readable only when EVERY one of its names
    has <name>_<start>.bp (one inname template serves all domains). start: datetime (default
    the namelist's own start). Aux streams not in aux_ok (when given) are dropped: those keep
    the case's own settings anyway."""
    if isinstance(names, str):
        names = [n for n in re.split(r"[\s,]+", names) if n]
    kind_stream = {v[0]: k for k, v in PER_RECORD_STREAMS.items()}
    by_stream = {}
    for n in names or []:
        m = _PER_RECORD_NAME_RE.match(n)
        if m:
            by_stream.setdefault(kind_stream[m.group(1)], []).append(n)
    if aux_ok is not None:
        by_stream = {s: v for s, v in by_stream.items() if s == "boundary" or s in aux_ok}
    if not by_stream:
        return {}, {}
    when = (start or case_start_datetime(text)).strftime(WRF_DATE_FMT)
    ok, fallback = {}, {}
    for stream, ns in by_stream.items():
        missing = ["%s_%s.bp" % (n, when) for n in ns if not per_record_has(bp_dir, n, when)]
        if missing:
            fallback[stream] = missing
        else:
            ok[stream] = ns
    return ok, fallback


def _has_key(text, key):
    return any(re.search(r"(?<![A-Za-z0-9_])%s[ \t]*=" % re.escape(key), ln, re.I)
               for ln in text.splitlines() if not ln.lstrip().startswith("!"))


def _per_record_netcdf_fallback(text, stream, max_dom):
    """Make `stream` read its netCDF file: io_form 2, the default inname (boundary: none, i.e.
    WRF's wrfbdy_d<domain>), and undo a previous pass's per-record keys (multi_bdy_files
    .false., frames_per_auxinputN back to the Registry default 999999) when present."""
    _kind, io_key, group, inname_key, default, frames_key = PER_RECORD_STREAMS[stream]
    text = _set_namelist_value(text, io_key, "2", group)
    if stream == "boundary":
        text = _delete_key_lines(text, inname_key)
        if _has_key(text, "multi_bdy_files"):
            text = _set_namelist_value(text, "multi_bdy_files", ".false.", "bdy_control")
    else:
        text = _set_namelist_value(text, inname_key, '"%s"' % default, group)
        if _has_key(text, frames_key):
            text = _set_namelist_value(text, frames_key, ", ".join(["999999"] * max_dom),
                                       "time_control")
    return text


def adios2_stream_edits(stream):
    """[(key, value, group)] that make WRF read `stream` as ADIOS2: io_form = 14 and the inname
    with the .bp suffix (a quoted string)."""
    return [(stream["io_key"], ADIOS2_IO_FORM, stream["group"]),
            (stream["inname_key"], '"%s"' % stream["inname"], stream["group"])]


def bp_cache_dir(case_id, src_dir, root=None):
    """Default BP input cache of one case's netCDF source dir: ${WTUNE_BP_CACHE:-
    $HOME/wtune-runs/bp-cache}/<case_id>/<sha1(realpath(src_dir))[:12]>. Keyed by the source dir
    too, so two prep dirs of one case id (a rebuilt point re-made by another tune run) never
    share -- a job reading one cache is never reconverted under by a job of the other."""
    import hashlib
    root = root or os.environ.get("WTUNE_BP_CACHE") or os.path.join(
        os.path.expanduser("~"), "wtune-runs", "bp-cache")
    key = hashlib.sha1(os.path.realpath(src_dir).encode()).hexdigest()[:12]
    return os.path.join(root, case_id, key)


def _import_wtune_bp():
    """src/wtune_bp (the converter module), imported lazily: only staging needs it. It runs
    in-process on the stdlib and calls the adios2 converters in envs/adios2's interpreter
    ($WTUNE_ADIOS2_PYTHON overrides). Looked up in $WTUNE_SRC, then <this checkout>/src."""
    for src in (os.environ.get("WTUNE_SRC"), os.path.join(os.path.dirname(CAMPAIGN_DIR), "src")):
        if src and os.path.isfile(os.path.join(src, "wtune_bp.py")):
            for p in (os.path.join(src, "tools"), src):
                if p not in sys.path:
                    sys.path.insert(0, p)
            break
    import wtune_bp
    return wtune_bp


def _src_path(src_dir, name):
    return name if os.path.isabs(name) else os.path.join(src_dir, name)


def resolve_bp_inputs(text, src_dir, cache_dir, convert=True, bp=None, log=None):
    """Every input-stream file of `text` (adios2_input_streams) as a BP directory, in this order
    per name: <src>/<name>.bp (a directory: the case already has BP inputs) -> <src>/<name> when
    it is itself a directory (BP written without a suffix) -> the netCDF <src>/<name> converted
    by wtune_bp.convert_inputs(<src dir>, cache_dir, names, engine=wtune_bp.DEFAULT_ENGINE, BP4) into cache_dir/<name>
    (cached with a manifest, reconverted only when the source changed; convert_inputs itself
    serializes on a lockf lock in cache_dir, so concurrent callers convert once). A '<date>'
    name is globbed in src_dir. convert=False plans only (nothing converted, nothing written).
    Per-record inputs (wrfbdy/wrflowinp/wrffdda_d0N, wtune_bp.PER_RECORD_NAME_RE): convert_inputs
    writes them as a directory of one-step BPs <name>_<date>.bp, because WRF 4.8's ADIOS2 reader
    returns record 1 for every read of a multi-record BP. Such a target's links are one per
    record file (wtune_bp.bp_input_links: <name>_<date>.bp -> <container>/<name>_<date>.bp) and
    it is listed in "per_record" {name: container} (planned: cache_dir/<name>). An existing
    SINGLE BP of a per-record name is that wrong multi-record layout: it is converted from the
    netCDF next to it instead when there is one, else used with an entry in "single_multirecord"
    (callers warn).
    Returns {"links": {link name: target}, "convert": [netCDF paths to convert / converted],
    "existing": [BP targets used as-is], "missing": [(stream, name) required but found nowhere],
    "ready": [streams whose every required name resolved], "needs_convert": [streams with at
    least one netCDF-only file], "per_record": {name: container}, "single_multirecord":
    [single-BP targets of per-record names], "streams": [...]}."""
    import glob as _glob
    streams = adios2_input_streams(text)
    res = {"links": {}, "convert": [], "existing": [], "missing": [], "ready": [],
           "needs_convert": [], "per_record": {}, "single_multirecord": [], "streams": streams}
    todo = {}                                   # src dir -> [(basename, link name)]
    status = {}

    def use_existing(target, base, link):
        nonlocal bp
        target = os.path.abspath(target)
        res["existing"].append(target)
        if _PER_RECORD_NAME_RE.match(base):
            bp = bp or _import_wtune_bp()
            if bp.is_per_record_dir(target):
                res["per_record"][base] = target
                res["links"].update(bp.bp_input_links(base, target))
                return
            res["single_multirecord"].append(target)
        res["links"][link] = target

    for st in streams:
        ok = True
        for name in st["names"]:
            if "<" in name:
                pat = _src_path(src_dir, re.sub(r"<[^>]*>", "*", name))
                hits = sorted(set(p[:-3] if p.endswith(".bp") else p
                                  for p in _glob.glob(pat) + _glob.glob(pat + ".bp")))
                concrete = [os.path.relpath(p, src_dir) if not os.path.isabs(name) else p
                            for p in hits]
            else:
                concrete = [name]
            found_any = False
            for c in concrete:
                path = _src_path(src_dir, c)
                base = os.path.basename(c)
                link = base + ".bp"
                existing = (path + ".bp" if os.path.isdir(path + ".bp")
                            else path if os.path.isdir(path) else None)
                if existing is not None and _PER_RECORD_NAME_RE.match(base) \
                        and os.path.isfile(path):
                    bp = bp or _import_wtune_bp()
                    if not bp.is_per_record_dir(existing):
                        existing = None         # a single multi-record BP: convert per record
                if existing is not None:
                    use_existing(existing, base, link)
                elif os.path.isfile(path):
                    todo.setdefault(os.path.dirname(os.path.abspath(path)), []).append(
                        (base, link))
                    res["convert"].append(os.path.abspath(path))
                    if st["stream"] not in res["needs_convert"]:
                        res["needs_convert"].append(st["stream"])
                    if _PER_RECORD_NAME_RE.match(base):
                        res["per_record"][base] = os.path.abspath(os.path.join(cache_dir, base))
                else:
                    continue
                found_any = True
            if not found_any and name in st["required"]:
                res["missing"].append((st["stream"], name))
                ok = False
        status[st["stream"]] = ok
    res["ready"] = [st["stream"] for st in streams if status[st["stream"]]]
    if not convert or not todo:
        return res
    bp = bp or _import_wtune_bp()
    bp.require_adios2_python()
    for d in sorted(todo):
        names = [b for b, _ in todo[d]]
        if log:
            log("ADIOS2 inputs: converting %s from %s into %s (BP4, cached)"
                % (" ".join(names), d, cache_dir))
        got = bp.convert_inputs(d, cache_dir, names=names)   # engine: wtune_bp.DEFAULT_ENGINE (BP4)
        for base, link in todo[d]:
            target = (got or {}).get(base)
            if not target or not os.path.isdir(target):
                raise RuntimeError("convert_inputs gave no BP directory for %s (got %r)"
                                   % (os.path.join(d, base), target))
            target = os.path.abspath(target)
            if bp.is_per_record_dir(target):
                res["per_record"][base] = target
                res["links"].update(bp.bp_input_links(base, target))
            else:
                res["per_record"].pop(base, None)
                res["links"][link] = target
    return res


def _cmd_stage_bp_inputs(argv):
    """configs.py stage-bp-inputs CASE_ID NAMELIST SRC_DIR [--cache-dir DIR] [--check]
    (strict_campaign.sbatch, on the compute node, python from the wtune env): resolve_bp_inputs
    into DIR (default bp_cache_dir(CASE_ID, SRC_DIR)) and make DIR/<name>.bp point at each BP
    input, so link_wrf_inputs/link_namelist_files find them with BP_DIR=DIR. --check: no
    conversion and no writes (a bp_dir that already holds the .bp inputs). stdout: shell
    assignments BP_DIR=..., WTUNE_AUX_BP_STREAMS=<aux streams available as BP>,
    WTUNE_PER_RECORD_BP=<input names held one BP per record in DIR/<name>/<name>_<date>.bp:
    wrfbdy/wrflowinp/wrffdda, see resolve_bp_inputs and _apply_adios2> and
    BP_CONVERTED=<SRC_DIR netCDF names now held as BP in DIR>; log on stderr.
    Exit 1 when wrfinput_d01/wrfbdy_d01 resolve nowhere or conversion fails."""
    ap = argparse.ArgumentParser(prog="configs.py stage-bp-inputs")
    ap.add_argument("case_id")
    ap.add_argument("namelist")
    ap.add_argument("src_dir")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    with open(a.namelist) as f:
        text = f.read()
    cache = os.path.abspath(a.cache_dir or bp_cache_dir(a.case_id, a.src_dir))
    err = lambda m: sys.stderr.write(m + "\n")
    lockf = None
    try:
        if not a.check:
            # several probe jobs of one case start together: one converts and links, the others
            # block here (lockf: NFS/GPFS-safe across nodes), then find the manifest current
            import fcntl
            os.makedirs(cache, exist_ok=True)
            lockf = open(os.path.join(cache, ".stage.lock"), "a")
            fcntl.lockf(lockf, fcntl.LOCK_EX)
        return _stage_bp_inputs_locked(a, text, cache, err)
    except Exception as e:                     # ImportError/RuntimeError/OSError from the converter
        err("FATAL: ADIOS2 input staging failed for %s: %s: %s" % (a.case_id, type(e).__name__, e))
        return 1
    finally:
        if lockf is not None:
            lockf.close()                      # closing the fd releases the lockf lock


def _link_idempotent(dst, target):
    """dst -> target symlink; an existing identical link is kept, a stale link replaced, a real
    file/dir is refused (never deleted). dst == target (a cache entry itself) is a no-op."""
    if os.path.abspath(dst) == os.path.abspath(target):
        return
    if os.path.islink(dst):
        if os.readlink(dst) == target:
            return
        os.remove(dst)
    elif os.path.lexists(dst):
        raise RuntimeError("%s exists and is not a link -- refusing to replace it" % dst)
    os.symlink(target, dst)


def _stage_bp_inputs_locked(a, text, cache, err):
    res = resolve_bp_inputs(text, a.src_dir, cache, convert=not a.check, log=err)
    for stream, name in res["missing"]:
        err("WARNING: %s input %s found nowhere in %s (neither netCDF nor .bp)"
            % (stream, name, a.src_dir))
    if not {"input", "boundary"} <= set(res["ready"]):
        err("FATAL: wrfinput/wrfbdy not available as ADIOS2 inputs for %s" % a.case_id)
        return 1
    if a.check:
        for p in res["convert"]:
            err("WARNING: %s is netCDF only (no .bp next to it) -- its stream stays netCDF" % p)
        if {"input", "boundary"} & set(res["needs_convert"]):
            err("FATAL: --check: wrfinput/wrfbdy are not all BP in %s" % a.src_dir)
            return 1
        aux = [s for s in res["ready"]
               if s not in ("input", "boundary") and s not in res["needs_convert"]]
    else:
        containers = set(res["per_record"].values())
        for link, target in sorted(res["links"].items()):
            if os.path.dirname(target) in containers:
                continue                       # a record file: reached via its container below
            _link_idempotent(os.path.join(cache, link), target)
            err("BP input %s -> %s" % (link, target))
        for name, target in sorted(res["per_record"].items()):
            # cache/<name> IS the container when converted here (no-op); an existing per-record
            # dir elsewhere gets a cache/<name> link, so link_wrf_inputs finds every container
            # as $BP_DIR/<name>
            _link_idempotent(os.path.join(cache, name), target)
            err("BP input %s -> %s (per record, %d files)"
                % (name, target, sum(1 for t in res["links"].values()
                                     if os.path.dirname(t) == target)))
        aux = [s for s in res["ready"] if s not in ("input", "boundary")]
    for target in res["single_multirecord"]:
        err("WARNING: %s is a single multi-record BP: WRF 4.8's ADIOS2 reader returns its FIRST "
            "record for every boundary/SST/FDDA read -- convert the netCDF per record instead "
            "(wtune convert inputs)" % target)
    per_record = sorted(n for n, t in res["per_record"].items() if os.path.isdir(t))
    src = os.path.realpath(a.src_dir)
    print("BP_DIR=%s" % _shq(cache))
    print("WTUNE_AUX_BP_STREAMS=%s" % _shq(",".join(aux)))
    print("WTUNE_PER_RECORD_BP=%s" % _shq(" ".join(per_record)))
    print("BP_CONVERTED=%s" % _shq(" ".join(
        os.path.basename(p) for p in res["convert"]
        if os.path.realpath(os.path.dirname(p)) == src and not a.check)))
    return 0


def _shq(v):
    return "'%s'" % str(v).replace("'", "'\\''")


def _cmd_link_namelist_files(argv):
    """configs.py link-namelist-files NAMELIST RUN_DIR SEARCH_DIR... (from a staged run dir, once
    namelist.input is final): stage_namelist_files + its log lines; exits 0 (a missing file is
    reported, not fatal -- `wtune check` is the gate)."""
    if len(argv) < 2:
        sys.stderr.write("usage: configs.py link-namelist-files NAMELIST RUN_DIR SEARCH_DIR...\n")
        return 2
    with open(argv[0]) as f:
        text = f.read()
    res = stage_namelist_files(text, argv[1], argv[2:])
    for line in format_namelist_stage(res, argv[2:]):
        (sys.stderr if line.startswith("WARNING") else sys.stdout).write(line + "\n")
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
            "decomp-pair CASE_ID DECOMP TWIN_DECOMP [--config-file FILE.json] | "
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
            "fidelity-tokens FIDELITY_JSON | "
            "quota-free-gib [--filesystem home1] (reads mmlsquota text on stdin) | "
            "link-namelist-files NAMELIST RUN_DIR SEARCH_DIR... | "
            "stage-bp-inputs CASE_ID NAMELIST SRC_DIR [--cache-dir DIR] [--check]\n"
            "CONFIG_FILE=path.json env (predictor/decomp_space.py to_campaign_configs/sample "
            "output): CASE_ID and order CASE_ID measure that generated config set instead of "
            "the fixed C01-C16 STRICT templates, and decomp-pair resolves DECOMP/TWIN_DECOMP "
            "from it -- which is how MODE=physics reaches a case that resolves no STRICT "
            "template at all (sc26p1).\n"
        )
        return 2
    if argv[0] == "edit-namelist":
        return _cmd_edit_namelist(argv[1:])
    if argv[0] == "step-stats":
        return _cmd_step_stats(argv[1:])
    if argv[0] == "fused-steps":
        return _cmd_fused_steps(argv[1:])
    if argv[0] == "point-plan":
        return _cmd_point_plan(argv[1:])
    if argv[0] == "quota-free-gib":
        return _cmd_quota_free_gib(argv[1:])
    if argv[0] == "link-namelist-files":
        return _cmd_link_namelist_files(argv[1:])
    if argv[0] == "stage-bp-inputs":
        return _cmd_stage_bp_inputs(argv[1:])
    if argv[0] == "physics-list":
        return cmd_physics_list(argv[1:])
    if argv[0] == "physics-order":
        return cmd_physics_order(argv[1:])
    if argv[0] == "physics-describe":
        return cmd_physics_describe(argv[1:])
    if argv[0] == "fidelity-tokens":
        return cmd_fidelity_tokens(argv[1:])
    if argv[0] == "restart-time":
        if len(argv) != 3:
            sys.stderr.write("usage: configs.py restart-time CASE_ID SPINUP_MINUTES\n")
            return 2
        return cmd_restart_time(argv[1], argv[2])
    if argv[0] == "config-extras":
        config_file = os.environ.get("CONFIG_FILE")
        if len(argv) != 2 or not config_file:
            sys.stderr.write("usage: CONFIG_FILE=path.json configs.py config-extras CASE_ID\n")
            return 2
        return cmd_config_extras(argv[1], config_file)
    assert_strict_compliance()
    if argv[0] == "decomp-pair":
        return cmd_decomp_pair(argv[1:])
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
