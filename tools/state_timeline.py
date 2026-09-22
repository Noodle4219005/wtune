#!/usr/bin/env python3
"""tools/state_timeline.py -- training-data extraction for a per-step WRF cost model driven by
the atmosphere's physical state (this task's own framing: "predict WRF's per-step cost from the
atmosphere's physical state").

Companion to tools/temporal_validation.py's new TIMELINE mode (submit-timeline / TV_MODE=timeline
in tools/temporal_run.sbatch): that mode runs a COMPLETE case (netCDF, cold start, full length or
--hours) at up to 3 decompositions (anchor + <=2 alternatives) and additionally writes a small
2-D auxiliary history stream every K simulated minutes. This module turns one such run directory
into training rows:

  `extract --run-dir D --case C [--config JSON]`: reads the run's auxhist netCDF frames (STATE)
     and its rsl.out.0000 per-step timing series (TARGET), aggregates both to the same K-minute
     interval grid, and writes <out>/<case>__<config>__timeline.csv + a JSON header.

  `plan-training`: lists campaign/cases.json (+ SITE overlay) cases with their run length and a
     rough full-run time estimate on N nodes (from existing scc-sweep ledgers when one exists,
     else "unknown"), cheapest first, plus ready-to-paste `temporal_validation.py submit-timeline`
     commands.

netCDF4 is imported LAZILY, only inside the actual frame-reading path (`_lazy_netcdf4`,
`read_2d_field`) -- never at module import time and never by any function under test. Every
feature-math and namelist/rsl-parsing function below is pure (plain python lists/dicts/floats, no
numpy, no netCDF4), specifically so tools/test_state_timeline.py can exercise the whole feature
pipeline on a dev machine with neither installed (this repo's own convention: see
tools/temporal_validation.py's module docstring on pixi-vs-bare-python3, and CAMPAIGN's own
synthetic-fixture test style)."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
_PREDICTOR_DIR = os.path.join(_REPO_ROOT, "predictor")
for _p in (_CAMPAIGN_DIR, _PREDICTOR_DIR, _REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Same module-level SITE default as tools/temporal_validation.py -- must precede the first
# `import configs` anywhere in the process (campaign/configs.py reads SITE at import time).
os.environ.setdefault("SITE", "scc")

import configs as campaign_configs          # campaign/configs.py: CASES/CASE_ORDER, namelist helpers
import decomp_space                         # predictor/decomp_space.py: rsl_patch_width_for_rank
import scc_sweep                            # tools/scc_sweep.py: s/step ledger estimate, _expand
import temporal_validation as tv            # this task's sibling module: case metadata helpers

DEFAULT_INTERVAL_MINUTES = 30
MIN_INTERVAL_DT_MULTIPLE = 20              # task: K >= 20*dt

# The task's own field list (2-D diagnostics added to the auxiliary stream) plus the
# time-invariant static fields (written every frame like everything else on that stream --
# WRF's iofields mechanism has no "first frame only" toggle -- but only ever READ from frame 0
# by extract(), since they never change during one run).
DEFAULT_TIMELINE_FIELDS = ("RAINNC", "RAINC", "SWDOWN", "OLR", "PBLH", "U10", "V10", "HFX", "LH",
                            "COSZEN")
DEFAULT_STATIC_FIELDS = ("XLAT", "XLONG", "LANDMASK")

PRECIP_THRESHOLD_MM_H = 1.0
CLOUD_OLR_THRESHOLD_WM2 = 200.0
SUNLIT_SWDOWN_THRESHOLD_WM2 = 1.0

AUX_STREAM_PREFERRED = 23
AUX_STREAM_MIN = 1
AUX_STREAM_MAX = 24


# ===========================================================================
# 1. rsl timing parsing, multi-domain, radiation/normal step separation
# ===========================================================================
_TIMING_RE = re.compile(r"Timing for main: time (\S+) on domain\s*(\d+):\s*([0-9.]+) elapsed")


def parse_step_series(rsl_text, max_dom):
    """{domain: [wall_seconds, ...]} for domain in 1..max_dom, in file order, from WRF's own
    'Timing for main: time <sim_time> on domain <N>: <X> elapsed seconds' rsl.out.0000 lines
    (same line family tools/temporal_validation.py's parse_domain1_step_series and
    campaign/phase_ranking.py's PATTERN already consume) -- extended here to keep EVERY active
    domain's own series, not just domain 1: a nested domain's line is its own sub-stepped
    integration, not double-counted inside domain 1's line (confirmed against isc21 job 1074515,
    same citation temporal_validation.py's own docstring uses), so per-domain cost is only
    visible by keeping domain>1 lines too. A domain number outside 1..max_dom is ignored (stray
    line, or a nest this run never activated)."""
    by_domain = {d: [] for d in range(1, max_dom + 1)}
    for line in rsl_text.splitlines():
        if "Timing for main" not in line:
            continue
        m = _TIMING_RE.search(line)
        if not m:
            continue
        dom = int(m.group(2))
        if dom < 1 or dom > max_dom:
            continue
        by_domain[dom].append(float(m.group(3)))
    return by_domain


def steps_per_radiation_call(radt_minutes, dt_seconds):
    """STEPRA = round(radt_minutes*60/dt_seconds), WRF's own radiation-call cadence (radiation is
    invoked when itimestep==1 .or. mod(itimestep, STEPRA)==0 -- phys/module_radiation_driver.F via
    share/module_first_rk_step_part1.F; the same radt/STEPRA relationship campaign/cases.json's
    sc26p1 note already uses: "radt=9 at dt=10 => STEPRA=54" is round(9*60/10)==54). Floored at 1
    (a radt so small relative to dt that it would fire every step)."""
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be > 0")
    if radt_minutes <= 0:
        raise ValueError("radt_minutes must be > 0")
    return max(1, int(round(radt_minutes * 60.0 / dt_seconds)))


def classify_steps(n_steps, stepra):
    """1-indexed itimestep -> is_radiation bool, for itimestep in 1..n_steps: True at
    itimestep==1 (the initial radiation call every WRF run makes before its first dynamics step)
    or wherever itimestep % stepra == 0 -- see steps_per_radiation_call. Unlike
    temporal_validation.median_s_per_step's convention (step 1 always excluded from the median),
    this module classifies step 1 as a real radiation-cost sample rather than discarding it: a
    K-minute training bucket that happens to start at t=0 would otherwise silently lose its one
    and only radiation-step observation."""
    if n_steps <= 0:
        return []
    return [(i == 1) or (i % stepra == 0) for i in range(1, n_steps + 1)]


def step_target_features(step_wall_times, dt_seconds, radt_minutes, interval_minutes):
    """{bucket_index: {s_step_normal_median, s_step_rad_median, n_steps}} -- one domain's TARGET
    columns per K=interval_minutes-minute simulated-time bucket. bucket_index = floor(sim_seconds
    at the START of the step / (interval_minutes*60)), sim_seconds for 1-indexed step i is
    (i-1)*dt_seconds -- same "bucket by simulated time at step start" convention
    temporal_validation.drift_curve already uses for its 1-hour buckets, just parameterised by K
    instead of a fixed hour. median() over an empty list is None, never 0 or NaN, so a bucket that
    happened to contain zero radiation (or zero normal) steps is visibly "no data" downstream, not
    a fabricated zero cost."""
    stepra = steps_per_radiation_call(radt_minutes, dt_seconds)
    flags = classify_steps(len(step_wall_times), stepra)
    interval_seconds = interval_minutes * 60.0
    buckets = {}
    for i, (w, is_rad) in enumerate(zip(step_wall_times, flags), start=1):
        sim_s = (i - 1) * dt_seconds
        b = int(sim_s // interval_seconds)
        d = buckets.setdefault(b, {"normal": [], "rad": []})
        d["rad" if is_rad else "normal"].append(w)
    out = {}
    for b, d in buckets.items():
        out[b] = dict(
            s_step_normal_median=(statistics.median(d["normal"]) if d["normal"] else None),
            s_step_rad_median=(statistics.median(d["rad"]) if d["rad"] else None),
            n_steps=len(d["normal"]) + len(d["rad"]),
        )
    return out


# ===========================================================================
# 2. namelist array readers (radt, parent_time_step_ratio) -- per-domain dt for STEPRA
# ===========================================================================
def read_namelist_int_list(text, key, max_dom, default=None):
    """Up to max_dom int values of `key` (e.g. radt, parent_time_step_ratio), via
    campaign/configs.py's own _find_namelist_key (never re-derived -- that function already
    handles keys jammed onto one line with others, e.g. "mp_physics=8, ra_lw_physics=4, radt=15,",
    the exact shape cases.json's own case notes cite for karina's template). Fewer values than
    max_dom (a template that only bothered to write one value, meaning every domain uses it) pads
    by repeating the LAST given value -- a pragmatic fallback, not a claim about strict Fortran
    namelist-read semantics (which requires an exact count or an explicit r*value repeat).
    Returns [default]*max_dom when the key has no assignment in `text` at all and default is not
    None; raises ValueError when it is missing and default is None."""
    found = campaign_configs._find_namelist_key(text, key)
    if not found or not found["values"]:
        if default is None:
            raise ValueError("namelist key %r not found and no default given" % (key,))
        return [default] * max_dom
    values = [int(float(v)) for v in found["values"]]
    if len(values) >= max_dom:
        return values[:max_dom]
    return values + [values[-1]] * (max_dom - len(values))


def domain_dt_seconds(dt0_seconds, parent_time_step_ratio_list):
    """Per-domain integration timestep (seconds), domain 1 = dt0_seconds, domain d>1 =
    dt0_seconds / prod(parent_time_step_ratio_list[1:d]) -- WRF's own nested-timestep convention
    (share/module_domain.F: each child steps parent_time_step_ratio times per parent step, so its
    own dt is the parent's dt divided by that ratio; the ratio is applied once per nesting level,
    domain 1's own ratio entry is unused/1). parent_time_step_ratio_list is 1-indexed by domain in
    the namelist (index 0 == domain 1, conventionally 1 or unset)."""
    out = [float(dt0_seconds)]
    for d in range(1, len(parent_time_step_ratio_list)):
        ratio = parent_time_step_ratio_list[d] or 1
        out.append(out[-1] / ratio)
    return out


# ===========================================================================
# 3. auxiliary-stream namelist editing (TIMELINE mode's own state stream)
# ===========================================================================
def _aux_interval_key(n):
    return "auxhist%d_interval" % n


def _aux_ioform_key(n):
    return "io_form_auxhist%d" % n


def _aux_frames_key(n):
    return "frames_per_auxhist%d" % n


def used_aux_streams(text):
    """Every aux-stream number this namelist ALREADY assigns (auxhistN_interval or
    io_form_auxhistN, any N) -- the "already uses it" detection the task asks for before
    defaulting to stream 23."""
    used = set()
    for m in re.finditer(r"(?m)^(?![ \t]*!)[^\n]*?\b(?:auxhist|io_form_auxhist)(\d+)(?:_interval)?\b",
                          text):
        used.add(int(m.group(1)))
    return used


def pick_free_aux_stream(text, preferred=AUX_STREAM_PREFERRED, lo=AUX_STREAM_MIN, hi=AUX_STREAM_MAX):
    """`preferred` (default 23, the task's own default) if this namelist does not already use it,
    else the lowest free stream number in [lo, hi]. Raises ValueError when every stream in range
    is already spoken for (24 aux streams is WRF's own ceiling; no case in this campaign is
    expected to use anywhere near that many, but the check is explicit rather than silently
    reusing/overwriting an in-use stream)."""
    used = used_aux_streams(text)
    if preferred not in used:
        return preferred
    for n in range(lo, hi + 1):
        if n not in used:
            return n
    raise ValueError("no free aux stream in [%d, %d]: all already used by this namelist" % (lo, hi))


def validate_aux_interval_minutes(interval_minutes, dt_seconds, min_multiple=MIN_INTERVAL_DT_MULTIPLE):
    """Raises ValueError when interval_minutes*60 < min_multiple*dt_seconds (task: "K >= 20xdt").
    dt_seconds is always domain 1's own timestep (case["dt"]) -- the largest dt among domains (a
    nest's dt is smaller, see domain_dt_seconds), so satisfying this bound for d01 satisfies it
    for every nest too."""
    floor_s = min_multiple * dt_seconds
    if interval_minutes * 60.0 < floor_s:
        raise ValueError("aux interval %g min (%g s) is below the required %dxdt floor (%g s, dt=%gs)"
                          % (interval_minutes, interval_minutes * 60.0, min_multiple, floor_s, dt_seconds))


def build_iofields_content(fields, stream):
    """iofields_filename file content adding exactly `fields` to history stream `stream`, WRF's
    own '+:h:N:field1,field2,...' add-field syntax (Registry/registry.io_boilerplate and the
    iofields_filename mechanism it documents: a leading '+' ADDS to whatever the stream already
    carries by default, ':h:' scopes it to history-type output, ':N:' to stream N only -- so the
    main history stream, which this campaign always turns off for STRICT/temporal runs, is
    unaffected). One line, trailing newline; WRF reads iofields_filename line-by-line and this
    field count (<=13) is far under any documented per-line limit."""
    return "+:h:%d:%s\n" % (stream, ",".join(fields))


def filter_fields_by_registry(fields, registry_text):
    """(kept, dropped) -- `fields` split by whether a non-comment Registry 'state' line
    (Registry/registry.* convention: "state <type> <NAME> <dims> ... ") mentions the field name as
    a whole token anywhere on the line. Registry lines are whitespace/comma separated, case as
    written (WRF Registry variable names are conventionally upper-case, e.g. RAINNC) -- matched
    case-insensitively here since the task only needs existence, not the Registry's own casing
    convention re-derived. Order of `fields` is preserved in both outputs."""
    kept, dropped = [], []
    state_lines = [ln for ln in registry_text.splitlines()
                   if re.match(r"^\s*state\b", ln, re.IGNORECASE)]
    blob = "\n".join(state_lines)
    for f in fields:
        pat = re.compile(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(f), re.IGNORECASE)
        (kept if pat.search(blob) else dropped).append(f)
    return kept, dropped


def read_registry_dir(wrf_tree):
    """Concatenated text of every file directly under $wrf_tree/Registry/ (WRF ships its Registry
    as many Registry.* fragments, #include-d from a top-level Registry.EM/Registry.NMM -- rather
    than resolve the #include graph, this reads every fragment, which only widens the search, it
    cannot spuriously report a field as present). Returns None when the directory does not exist
    (e.g. this dev checkout, or a wrf_tree that is not actually a WRF source/build tree) -- callers
    must treat that as "not validated here", not as "no fields exist"."""
    reg_dir = os.path.join(wrf_tree, "Registry")
    if not os.path.isdir(reg_dir):
        return None
    chunks = []
    for name in sorted(os.listdir(reg_dir)):
        path = os.path.join(reg_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, errors="replace") as f:
                chunks.append(f.read())
        except OSError:
            continue
    return "\n".join(chunks)


def edit_namelist_timeline(text, nproc_x, nproc_y, numtiles, ranks, max_dom, run_minutes,
                            restart_every_minutes, aux_stream, aux_interval_minutes,
                            iofields_filename, frames_per_auxhist=2000, dist="block",
                            blocktime="0"):
    """TIMELINE namelist: exactly tools/temporal_validation.edit_namelist_long's shape (cold
    start, history off, netCDF restart every restart_every_minutes) PLUS one auxiliary output
    stream: auxhist<N>_interval=aux_interval_minutes on d01 (0/never on any nest -- same
    "diffwrf/analysis only ever look at d01" convention campaign/configs.py's own
    _set_history_interval nest handling already documents), io_form_auxhist<N>=2 (netCDF, no
    ADIOS2 aux writer path exists here), frames_per_auxhist<N>=frames_per_auxhist on every domain,
    and iofields_filename=<iofields_filename> on every domain (WRF reads it once, applies its
    '+:h:N:...' lines regardless of which domain triggered the read)."""
    text = tv.edit_namelist_long(text, nproc_x, nproc_y, numtiles, ranks, max_dom, run_minutes,
                                  restart_every_minutes, dist=dist, blocktime=blocktime)
    interval_values = ([aux_interval_minutes] + [0] * (max_dom - 1)) if max_dom > 1 \
        else [aux_interval_minutes]
    text = campaign_configs._replace_list_line(text, _aux_interval_key(aux_stream), interval_values)
    text = campaign_configs._replace_list_line(
        text, _aux_frames_key(aux_stream), [frames_per_auxhist] * max_dom)
    text = campaign_configs._set_or_insert_time_control_scalar(
        text, _aux_ioform_key(aux_stream), 2)
    quoted = '"%s"' % iofields_filename
    text = campaign_configs._replace_list_line(text, "iofields_filename", [quoted] * max_dom)
    return text


# ===========================================================================
# 4. MPI patch mapping (WRF's exact RSL-lite remainder-at-both-ends rule) + imbalance features
# ===========================================================================
def patch_bounds(extent, nproc):
    """[(start, end), ...] length nproc, 0-indexed half-open column ranges along ONE dimension of
    ONE domain split `nproc` ways -- built by summing predictor/decomp_space.rsl_patch_width_for_rank
    (external/RSL_LITE/task_for_point.c's remainder-at-both-ends rule, the same ground truth
    rsl_patch_extents' envelope is derived from) rank by rank, rather than re-deriving the
    front/middle/back split here. sum(end-start for every rank) == extent."""
    bounds = []
    start = 0
    for r in range(nproc):
        w = decomp_space.rsl_patch_width_for_rank(extent, nproc, r)
        bounds.append((start, start + w))
        start += w
    return bounds


def patch_of_index(bounds, idx):
    """0-indexed rank owning column `idx`, from patch_bounds() output. Linear scan: nproc is at
    most a few hundred here and this runs once per analysis pass, not per grid column in a hot
    loop (patch_aggregate below calls it once per column, which for a ~1000x1000 domain is still
    <=1e6 calls total per field per frame -- fine for an offline extraction script)."""
    for r, (lo, hi) in enumerate(bounds):
        if lo <= idx < hi:
            return r
    raise ValueError("index %d out of range for bounds up to %d" % (idx, bounds[-1][1] if bounds else 0))


def patch_aggregate(field_2d, we, sn, nproc_x, nproc_y):
    """{(rank_x, rank_y): mean(field_2d over that patch's columns)} -- field_2d is a list of `sn`
    rows, each a list of `we` values (row-major [south_north][west_east], matching a WRF history
    variable's own axis order). Used for the imbalance features: run patch_aggregate on a 0/1
    per-column indicator (precip-over-threshold, sunlit) to get each MPI patch's own fraction,
    then max_mean_over_patches for the max/mean-over-patches ratio the task asks for."""
    x_bounds = patch_bounds(we, nproc_x)
    y_bounds = patch_bounds(sn, nproc_y)
    sums, counts = {}, {}
    for j in range(sn):
        row = field_2d[j]
        ry = patch_of_index(y_bounds, j)
        for i in range(we):
            rx = patch_of_index(x_bounds, i)
            key = (rx, ry)
            sums[key] = sums.get(key, 0.0) + row[i]
            counts[key] = counts.get(key, 0) + 1
    return {k: (sums[k] / counts[k] if counts[k] else 0.0) for k in sums}


def max_mean_over_patches(patch_values):
    """(max, mean) over patch_values.values() -- None, None for an empty map."""
    vals = list(patch_values.values())
    if not vals:
        return None, None
    return max(vals), statistics.mean(vals)


# ===========================================================================
# 5. state features (pure, numpy-free, over flat per-column lists)
# ===========================================================================
def precip_rate_mm_per_h(rainnc_prev, rainc_prev, rainnc_cur, rainc_cur, interval_hours):
    """One column's precip rate (mm/h) from the RAINNC+RAINC bucket difference between two
    frames, divided by the elapsed simulated time (hours) between them. RAINNC/RAINC are
    monotonically non-decreasing accumulators that only reset on a WRF restart
    (share/module_physics_init.F) -- this TIMELINE run never restarts mid-run (restart=.false.,
    restart writes are snapshots, never reads, within one run) -- so a negative diff should never
    occur; clipped to >=0 defensively rather than propagating a bogus negative rate into the
    training data."""
    rate = ((rainnc_cur + rainc_cur) - (rainnc_prev + rainc_prev)) / interval_hours
    return max(0.0, rate)


def precip_frac_and_mean(rates, threshold_mm_h=PRECIP_THRESHOLD_MM_H):
    """(precip_frac, precip_mean) over a flat list of per-column precip rates (mm/h): precip_frac
    is the fraction of columns with rate > threshold_mm_h; precip_mean is the plain mean over ALL
    columns. (None, None) for an empty list."""
    if not rates:
        return None, None
    n_over = sum(1 for r in rates if r > threshold_mm_h)
    return n_over / len(rates), statistics.mean(rates)


def precip_indicator(rates, threshold_mm_h=PRECIP_THRESHOLD_MM_H):
    """Per-column 0.0/1.0 precip-over-threshold indicator, for patch_aggregate."""
    return [1.0 if r > threshold_mm_h else 0.0 for r in rates]


def cloud_frac(olr_values, threshold=CLOUD_OLR_THRESHOLD_WM2):
    """Fraction of columns with OLR < threshold W/m^2 (deep convective cloud proxy, task's own
    definition). None for an empty list."""
    if not olr_values:
        return None
    return sum(1 for v in olr_values if v < threshold) / len(olr_values)


def sunlit_frac(swdown_values, coszen_values=None, swdown_threshold=SUNLIT_SWDOWN_THRESHOLD_WM2):
    """Fraction of columns sunlit: SWDOWN > swdown_threshold, OR (when coszen_values is given, same
    length) COSZEN > 0. coszen_values=None is the expected path when COSZEN is not in this WRF
    build's Registry (task point 1: drop it, fall back to SWDOWN alone) -- callers should call
    sunlit_indicator with the same coszen_values so the domain-aggregate and per-patch numbers use
    the identical rule."""
    n = len(swdown_values)
    if n == 0:
        return None
    if coszen_values is not None and len(coszen_values) == n:
        return sum(1 for s, c in zip(swdown_values, coszen_values)
                    if s > swdown_threshold or c > 0) / n
    return sum(1 for s in swdown_values if s > swdown_threshold) / n


def sunlit_indicator(swdown_values, coszen_values=None, swdown_threshold=SUNLIT_SWDOWN_THRESHOLD_WM2):
    """Per-column 0.0/1.0 sunlit indicator, same rule as sunlit_frac, for patch_aggregate."""
    if coszen_values is not None and len(coszen_values) == len(swdown_values):
        return [1.0 if (s > swdown_threshold or c > 0) else 0.0
                for s, c in zip(swdown_values, coszen_values)]
    return [1.0 if s > swdown_threshold else 0.0 for s in swdown_values]


def pblh_stats(pblh_values):
    """(mean, p90) over a flat PBLH list. p90 is nearest-rank on the sorted list (index =
    ceil(0.9*n)-1, clamped into range) -- no numpy/statistics.quantiles dependency, kept
    deliberately simple/bare-python since this module's pure functions are the part required to
    import and run with neither numpy nor netCDF4 installed. (None, None) for an empty list."""
    if not pblh_values:
        return None, None
    s = sorted(pblh_values)
    n = len(s)
    idx = min(n - 1, max(0, math.ceil(0.9 * n) - 1))
    return statistics.mean(pblh_values), s[idx]


def wind10_max(u10_values, v10_values):
    """max over columns of hypot(u10, v10). None for an empty list."""
    if not u10_values:
        return None
    return max(math.hypot(u, v) for u, v in zip(u10_values, v10_values))


def mean_or_none(values):
    return statistics.mean(values) if values else None


# ===========================================================================
# 6. row assembly + CSV/JSON I/O
# ===========================================================================
def domain_target_columns(max_dom):
    """Ordered list of every per-domain TARGET column name, d01..d<max_dom>."""
    cols = []
    for d in range(1, max_dom + 1):
        df = "d%02d" % d
        cols += ["s_step_normal_median_%s" % df, "s_step_rad_median_%s" % df, "n_steps_%s" % df]
    return cols


STATE_COLUMNS = [
    "precip_frac", "precip_mean", "cloud_frac", "sunlit_frac", "pblh_mean", "pblh_p90",
    "wind10_max", "hfx_mean",
    "precip_frac_patch_max", "precip_frac_patch_mean", "precip_frac_imbalance",
    "sunlit_frac_patch_max", "sunlit_frac_patch_mean", "sunlit_frac_imbalance",
]


def imbalance_ratio(patch_max, patch_mean):
    """max-over-patches / mean-over-patches -- None when either input is None/0 (no data, or a
    domain-wide-zero field where the ratio is undefined rather than meaningfully 0 or inf)."""
    if patch_max is None or patch_mean is None or patch_mean == 0:
        return None
    return patch_max / patch_mean


def build_state_row(precip_rates, olr, swdown, coszen, pblh, u10, v10, hfx, we, sn, nproc_x,
                     nproc_y):
    """One interval's full STATE_COLUMNS dict from flat per-column lists (length we*sn, row-major)
    plus the run's decomposition (nproc_x, nproc_y) for the imbalance features. Any field list
    that is empty/None is skipped (its dependent columns come out None) -- callers pass [] for a
    field this build's Registry did not have (task point 1: COSZEN is the expected case)."""
    precip_frac, precip_mean = precip_frac_and_mean(precip_rates)
    row = dict(
        precip_frac=precip_frac, precip_mean=precip_mean,
        cloud_frac=cloud_frac(olr) if olr else None,
        sunlit_frac=sunlit_frac(swdown, coszen) if swdown else None,
        wind10_max=wind10_max(u10, v10) if u10 and v10 else None,
        hfx_mean=mean_or_none(hfx),
    )
    pblh_mean, pblh_p90 = pblh_stats(pblh) if pblh else (None, None)
    row["pblh_mean"] = pblh_mean
    row["pblh_p90"] = pblh_p90

    def _grid(flat):
        return [flat[j * we:(j + 1) * we] for j in range(sn)]

    have_decomp = bool(nproc_x) and bool(nproc_y)
    if precip_rates and have_decomp:
        ind = _grid(precip_indicator(precip_rates))
        pmax, pmean = max_mean_over_patches(patch_aggregate(ind, we, sn, nproc_x, nproc_y))
        row["precip_frac_patch_max"] = pmax
        row["precip_frac_patch_mean"] = pmean
        row["precip_frac_imbalance"] = imbalance_ratio(pmax, pmean)
    else:
        row["precip_frac_patch_max"] = row["precip_frac_patch_mean"] = row["precip_frac_imbalance"] = None
    if swdown and have_decomp:
        ind = _grid(sunlit_indicator(swdown, coszen))
        smax, smean = max_mean_over_patches(patch_aggregate(ind, we, sn, nproc_x, nproc_y))
        row["sunlit_frac_patch_max"] = smax
        row["sunlit_frac_patch_mean"] = smean
        row["sunlit_frac_imbalance"] = imbalance_ratio(smax, smean)
    else:
        row["sunlit_frac_patch_max"] = row["sunlit_frac_patch_mean"] = row["sunlit_frac_imbalance"] = None
    return row


def merge_interval_rows(state_by_bucket, target_by_domain_bucket, max_dom, interval_minutes):
    """One row per bucket index present in EITHER state_by_bucket or any domain's target buckets
    -- state_by_bucket: {bucket: STATE_COLUMNS dict}; target_by_domain_bucket: {domain:
    {bucket: {s_step_normal_median, s_step_rad_median, n_steps}}}. Missing columns come out None,
    never dropped (visibility over silent loss). Rows sorted by bucket index; t_sim_hours =
    bucket * interval_minutes / 60."""
    buckets = set(state_by_bucket)
    for d in range(1, max_dom + 1):
        buckets |= set(target_by_domain_bucket.get(d, {}))
    rows = []
    for b in sorted(buckets):
        row = dict(interval_index=b, t_sim_hours=b * interval_minutes / 60.0)
        state = state_by_bucket.get(b, {})
        for col in STATE_COLUMNS:
            row[col] = state.get(col)
        for d in range(1, max_dom + 1):
            df = "d%02d" % d
            t = target_by_domain_bucket.get(d, {}).get(b, {})
            row["s_step_normal_median_%s" % df] = t.get("s_step_normal_median")
            row["s_step_rad_median_%s" % df] = t.get("s_step_rad_median")
            row["n_steps_%s" % df] = t.get("n_steps")
        rows.append(row)
    return rows


def timeline_csv_fieldnames(max_dom):
    return ["interval_index", "t_sim_hours"] + STATE_COLUMNS + domain_target_columns(max_dom)


def write_timeline_csv(path, rows, max_dom):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = timeline_csv_fieldnames(max_dom)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_timeline_header_json(path, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(header, f, indent=2, sort_keys=True)
        f.write("\n")


# ===========================================================================
# 7. netCDF glue -- LAZY import, never exercised by tests
# ===========================================================================
def _lazy_netcdf4():
    import netCDF4  # noqa: local import -- only available inside the pixi campaign env on the
                     # cluster ($HOME/wtune-envs/campaign); never imported at module load time.
    return netCDF4


def find_auxhist_frames(run_dir, stream):
    """Sorted auxhist<stream>_d01_* file paths in run_dir -- d01 (mother domain) only, matching
    every other "diffwrf/analysis only ever looks at d01" convention already established in this
    campaign (campaign/configs.py's own _set_history_mode docstring)."""
    return sorted(glob.glob(os.path.join(run_dir, "auxhist%d_d01_*" % stream)))


def read_2d_field(ds, name):
    """First (Time=0) frame's [south_north, west_east] 2-D slice of variable `name`, as a
    list-of-lists of python floats. Raises KeyError(name) when this file has no such variable --
    the task's own "validate against the first output frame when analyzing" requirement; callers
    catch this per-field and drop it rather than aborting the whole extraction."""
    if name not in ds.variables:
        raise KeyError(name)
    arr = ds.variables[name][0]
    return [[float(v) for v in row] for row in arr]


def extract_frame(path, fields):
    """(fields_dict, missing_list) for one auxhist frame file: fields_dict maps a found field name
    to its 2-D list-of-lists; missing_list holds every requested field this specific file does not
    carry (validated against the actual output, never assumed from the Registry)."""
    netCDF4 = _lazy_netcdf4()
    found, missing = {}, []
    with netCDF4.Dataset(path) as ds:
        for name in fields:
            try:
                found[name] = read_2d_field(ds, name)
            except KeyError:
                missing.append(name)
    return found, missing


# ===========================================================================
# 8. CLI: extract
# ===========================================================================
def _flatten(grid):
    return [v for row in grid for v in row]


def cmd_extract(args):
    cfg = json.loads(args.config) if args.config else {}
    max_dom = args.max_dom or campaign_configs._case_effective_max_dom(tv.case_json(args.case))
    interval_minutes = args.interval_minutes
    dt0 = args.dt or tv.case_json(args.case)["dt"]

    rsl_path = os.path.join(args.run_dir, "rsl.out.0000")
    with open(rsl_path, errors="replace") as f:
        rsl_text = f.read()
    by_domain_steps = parse_step_series(rsl_text, max_dom)

    tmpl_text, _path = tv.read_case_template(tv.case_json(args.case))
    if tmpl_text is not None:
        radt_list = read_namelist_int_list(tmpl_text, "radt", max_dom, default=None)
        ratio_list = read_namelist_int_list(tmpl_text, "parent_time_step_ratio", max_dom, default=1)
    else:
        radt_list = [args.radt] * max_dom if args.radt else None
        ratio_list = [1] * max_dom
    if radt_list is None:
        raise SystemExit("extract: could not read radt from the namelist template and no --radt "
                          "fallback was given")
    dts = domain_dt_seconds(dt0, ratio_list)

    target_by_domain_bucket = {}
    for d in range(1, max_dom + 1):
        target_by_domain_bucket[d] = step_target_features(
            by_domain_steps.get(d, []), dts[d - 1], radt_list[d - 1], interval_minutes)

    fields = list(args.fields.split(",")) if args.fields else list(DEFAULT_TIMELINE_FIELDS)
    frames = find_auxhist_frames(args.run_dir, args.stream)
    state_by_bucket = {}
    fields_missing_overall = set()
    nproc_x, nproc_y = cfg.get("nx"), cfg.get("ny")
    prev = None
    for i, path in enumerate(frames):
        found, missing = extract_frame(path, fields)
        fields_missing_overall |= set(missing)
        cur = found
        if prev is not None and "RAINNC" in prev and "RAINC" in prev and \
                "RAINNC" in cur and "RAINC" in cur:
            rn0, rc0 = _flatten(prev["RAINNC"]), _flatten(prev["RAINC"])
            rn1, rc1 = _flatten(cur["RAINNC"]), _flatten(cur["RAINC"])
            rates = [precip_rate_mm_per_h(a, b, c, d, interval_minutes / 60.0)
                     for a, b, c, d in zip(rn0, rc0, rn1, rc1)]
            we = len(cur["RAINNC"][0]) if cur["RAINNC"] else 0
            sn = len(cur["RAINNC"])
            state_by_bucket[i - 1] = build_state_row(
                rates,
                _flatten(cur.get("OLR", [])), _flatten(cur.get("SWDOWN", [])),
                _flatten(cur.get("COSZEN", [])) if "COSZEN" in cur else None,
                _flatten(cur.get("PBLH", [])), _flatten(cur.get("U10", [])),
                _flatten(cur.get("V10", [])), _flatten(cur.get("HFX", [])),
                we, sn, nproc_x, nproc_y)
        prev = cur

    rows = merge_interval_rows(state_by_bucket, target_by_domain_bucket, max_dom, interval_minutes)
    config_id = cfg.get("config_id", "unknown")
    out_csv = os.path.join(args.out, "%s__%s__timeline.csv" % (args.case, config_id))
    out_json = os.path.join(args.out, "%s__%s__timeline.json" % (args.case, config_id))
    write_timeline_csv(out_csv, rows, max_dom)

    static_present = []
    if frames:
        # Task point 1: XLAT/XLONG/LANDMASK are static grid metadata, only needed from the FIRST
        # frame -- written to a separate companion file (not the per-interval CSV: these are
        # 2-D fields, not scalars, and never change across a run). UNVERIFIED at full domain
        # resolution (see this task's own report): for a large domain this file could be tens of
        # MB; not measured without a real run, and a future pass may want to down-sample instead
        # of writing the full array.
        static_found, static_missing = extract_frame(frames[0], DEFAULT_STATIC_FIELDS)
        static_present = sorted(static_found)
        if static_found:
            out_grid = os.path.join(args.out, "%s__%s__grid.json" % (args.case, config_id))
            write_timeline_header_json(out_grid, static_found)
        if static_missing:
            print("WARNING: static fields absent from frame 0: %s" % ",".join(static_missing),
                  file=sys.stderr)
    write_timeline_header_json(out_json, dict(
        case=args.case, config=cfg, dt=dt0, dt_by_domain=dts, radt_by_domain=radt_list,
        max_dom=max_dom, interval_minutes=interval_minutes, aux_stream=args.stream,
        fields_requested=fields, fields_missing_in_output=sorted(fields_missing_overall),
        static_fields_present=static_present, n_frames=len(frames), n_rows=len(rows),
    ))
    print("wrote %s (%d rows), %s" % (out_csv, len(rows), out_json))
    if fields_missing_overall:
        print("WARNING: fields absent from every auxhist frame read: %s"
              % ",".join(sorted(fields_missing_overall)), file=sys.stderr)
    return 0


# ===========================================================================
# 9. CLI plumbing for tools/temporal_run.sbatch's TIMELINE mode (TV_MODE=timeline) -- mirrors
# tools/temporal_validation.py's own "internal plumbing" section (case-metadata,
# edit-namelist-long, ...), just for the aux-stream/iofields concerns this module owns.
# ===========================================================================
def cmd_default_fields(args):
    print(",".join(DEFAULT_TIMELINE_FIELDS))
    return 0


def cmd_pick_aux_stream(args):
    with open(args.namelist_file, errors="replace") as f:
        text = f.read()
    print(pick_free_aux_stream(text, preferred=args.preferred))
    return 0


def cmd_check_iofields(args):
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    registry_text = read_registry_dir(args.wrf_tree)
    if registry_text is None:
        sys.stderr.write("check-iofields: no Registry dir under %s -- fields not validated, "
                          "passing all through unchanged\n" % args.wrf_tree)
        print(",".join(fields))
        return 0
    kept, dropped = filter_fields_by_registry(fields, registry_text)
    for f in dropped:
        sys.stderr.write("check-iofields: DROPPED %s (no Registry 'state' line found under %s)\n"
                          % (f, args.wrf_tree))
    print(",".join(kept))
    return 0


def cmd_write_iofields(args):
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    with open(args.outfile, "w") as f:
        f.write(build_iofields_content(fields, args.stream))
    return 0


def cmd_edit_namelist_timeline(args):
    with open(args.infile) as f:
        text = f.read()
    dt0 = args.dt if args.dt is not None else tv.case_json(args.case)["dt"]
    validate_aux_interval_minutes(args.aux_interval, dt0)
    out = edit_namelist_timeline(
        text, args.nproc_x, args.nproc_y, args.numtiles, args.ranks, args.max_dom,
        args.hours * 60, args.every * 60, args.aux_stream, args.aux_interval,
        args.iofields_filename, frames_per_auxhist=args.frames_per_auxhist)
    with open(args.outfile, "w") as f:
        f.write(out)
    return 0


# ===========================================================================
# 10. CLI: plan-training
# ===========================================================================
def cmd_plan_training(args):
    cases = args.cases if args.cases else tv.auto_select_cases(0)
    rows = []
    for case in cases:
        cj = tv.case_json(case)
        text, path = tv.read_case_template(cj)
        full_hours = tv.read_run_length_hours(text) if text is not None else None
        hours = (min(int(full_hours), args.hours) if (full_hours is not None and args.hours)
                 else (int(full_hours) if full_hours is not None else args.hours))
        s_est, s_source, _n = scc_sweep.estimate_s_per_step(
            case, args.nodes, args.ledger_dir, args.hardware_id, fallback=None)
        est_seconds = None
        time_str = "unknown"
        if s_est is not None and hours:
            steps_est = max(1, int(hours * 3600 / cj["dt"]))
            time_est = scc_sweep.estimate_time(1, 1, steps_est, s_est)
            est_seconds = time_est["total_run_s"]
            time_str = time_est["time_limit"]
        rows.append(dict(case=case, hours=hours, s_est=s_est, s_source=s_source,
                          est_seconds=est_seconds, time_str=time_str))
    rows.sort(key=lambda r: (r["est_seconds"] is None, r["est_seconds"] or 0.0))
    for r in rows:
        s_est_str = ("%.3fs/step[%s]" % (r["s_est"], r["s_source"])) if r["s_est"] is not None \
            else "s/step unknown (no ledger row for %s on %s)" % (r["case"], args.hardware_id)
        print("%-14s hours=%-4s %s est_time=%s" % (r["case"], r["hours"], s_est_str, r["time_str"]))
        cmd = ("tools/temporal_validation.py submit-timeline --case %s --hours %s "
               "--nodes %d --k %d" % (r["case"], r["hours"], args.nodes, args.k))
        print("  %s" % cmd)
    return 0


# ===========================================================================
# CLI wiring
# ===========================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(prog="state_timeline.py")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("extract")
    sp.add_argument("--run-dir", required=True, dest="run_dir")
    sp.add_argument("--case", required=True)
    sp.add_argument("--config", default=None, help="JSON dict: config_id, nx, ny, ranks, omp, ...")
    sp.add_argument("--out", default=".")
    sp.add_argument("--stream", type=int, default=AUX_STREAM_PREFERRED)
    sp.add_argument("--interval-minutes", type=int, default=DEFAULT_INTERVAL_MINUTES,
                     dest="interval_minutes")
    sp.add_argument("--fields", default=None, help="comma-separated; default: task's own list")
    sp.add_argument("--max-dom", type=int, default=None, dest="max_dom")
    sp.add_argument("--dt", type=float, default=None)
    sp.add_argument("--radt", type=float, default=None,
                     help="fallback radt (minutes) when the namelist template is unreadable here")

    sp = sub.add_parser("default-fields")

    sp = sub.add_parser("pick-aux-stream")
    sp.add_argument("namelist_file")
    sp.add_argument("--preferred", type=int, default=AUX_STREAM_PREFERRED)

    sp = sub.add_parser("check-iofields")
    sp.add_argument("--wrf-tree", required=True, dest="wrf_tree")
    sp.add_argument("--fields", required=True)

    sp = sub.add_parser("write-iofields")
    sp.add_argument("outfile")
    sp.add_argument("--stream", type=int, required=True)
    sp.add_argument("--fields", required=True)

    sp = sub.add_parser("edit-namelist-timeline")
    sp.add_argument("infile")
    sp.add_argument("outfile")
    sp.add_argument("--case", required=True)
    sp.add_argument("--hours", type=int, required=True)
    sp.add_argument("--every", type=int, required=True)
    sp.add_argument("--nproc-x", type=int, required=True, dest="nproc_x")
    sp.add_argument("--nproc-y", type=int, required=True, dest="nproc_y")
    sp.add_argument("--numtiles", type=int, required=True)
    sp.add_argument("--ranks", type=int, required=True)
    sp.add_argument("--max-dom", type=int, default=1, dest="max_dom")
    sp.add_argument("--aux-stream", type=int, required=True, dest="aux_stream")
    sp.add_argument("--aux-interval", type=int, required=True, dest="aux_interval")
    sp.add_argument("--frames-per-auxhist", type=int, default=2000, dest="frames_per_auxhist")
    sp.add_argument("--iofields-filename", required=True, dest="iofields_filename")
    sp.add_argument("--dt", type=float, default=None)

    sp = sub.add_parser("plan-training")
    sp.add_argument("--cases", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                     default=None)
    sp.add_argument("--hours", type=int, default=None)
    sp.add_argument("--nodes", type=int, default=tv.DEFAULT_NODES)
    sp.add_argument("--k", type=int, default=2)
    sp.add_argument("--hardware-id", default=tv.DEFAULT_HARDWARE_ID, dest="hardware_id")
    sp.add_argument("--ledger-dir", default="$HOME/wtune-runs/campaign", dest="ledger_dir")

    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    handlers = {
        "extract": cmd_extract, "plan-training": cmd_plan_training,
        "default-fields": cmd_default_fields, "pick-aux-stream": cmd_pick_aux_stream,
        "check-iofields": cmd_check_iofields, "write-iofields": cmd_write_iofields,
        "edit-namelist-timeline": cmd_edit_namelist_timeline,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
