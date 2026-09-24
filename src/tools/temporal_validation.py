#!/usr/bin/env python3
"""tools/temporal_validation.py -- offline temporal validation (decision 0053 step 2).

Question this answers: is a config ranking measured in the first minutes of a forecast still
right 3-12 simulated hours later, and how does s/step drift over simulated time?

Two kinds of WRF run, both netCDF restart (io_form_restart=2) -- NEVER ADIOS2: ADIOS2 restart
corrupts the heap when rank placement changes across the sampled decompositions below, so the
LONG run and every SAMPLES run must agree on plain netCDF restart I/O.

  A. LONG RUN (one per case): a single WRF run of the case's full length (or up to --hours,
     default 12) at a fixed ANCHOR decomposition, restart files written every --every hours
     (default 3), history off. The full per-step timing series from this run IS the s/step
     drift curve (see drift_curve()).
  B. SAMPLES (several per case): for each restart time T in {t0, +every, +2*every, ..., +hours}
     the SAME set of N configs (anchor + K diverse, predictor/decomp_space.py sample_design,
     fixed seed) run a short window (--steps, default 30) starting from T (t0 is a normal cold
     start; every other T reads the LONG run's wrfrst_d0?_<T> files).

Analysis (see build_rankings/summarize/drift_curve): per T, the ranking of configs by median
s/step; Kendall tau between t0 and T; whether t0's best config is still within 2% of T's own
best at T (pre-registered: tau >= 0.7 and within-2% at the final T, e.g. +12h); the drift curve
normalised to t0. Writes <root>/<case>/{rankings,drift,summary}.csv and prints a table.

CLI: {plan, submit-long, submit-samples, analyze} are the operator-facing subcommands.
{case-metadata, restart-timestamp, edit-namelist-long, edit-namelist-sample, parse-timing} are
internal plumbing, invoked by tools/temporal_run.sbatch (the sbatch wrapper this tool submits --
see that file's own header for why it is a separate script rather than a new
campaign/strict_campaign.sbatch mode).

Reuses, by import (never re-derived): campaign/configs.py's namelist-editing primitives
(_replace_list_line, _set_or_insert_time_control_scalar, _find_namelist_key, edit_namelist,
_case_namelist_path, _case_effective_max_dom, HISTORY_NEVER_MINUTES, CASES/CASE_ORDER) and
tools/scc_sweep.py's decomposition-sweep config generator (build_anchor/build_configs,
write_configs_json) and its sbatch-command helpers (_expand, default_sbatch_runner, submit_job,
estimate_s_per_step, estimate_time). Like scc_sweep.py (which this module mirrors), this file
targets the pixi-provided Python (>=3.7, `from __future__ import annotations` used below), not
the bare python3 (3.6) campaign/configs.py itself is constrained to.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)  # == src/ (tools/ moved one level deeper under src/)
_TRUE_ROOT = os.path.dirname(_REPO_ROOT)  # actual repo root; campaign/ did not move
_CAMPAIGN_DIR = os.path.join(_TRUE_ROOT, "campaign")
_PREDICTOR_DIR = os.path.join(_REPO_ROOT, "predictor")
for _p in (_CAMPAIGN_DIR, _PREDICTOR_DIR, _REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Must precede the first `import configs` anywhere in the process (campaign/configs.py reads
# this env var at MODULE level) -- setdefault so an operator's own SITE=xxx export, or a test's
# monkeypatch before import, is not clobbered. Decision 0053 step 2 targets scc.
os.environ.setdefault("SITE", "scc")

import configs as campaign_configs          # campaign/configs.py, stdlib only
import scc_sweep                            # tools/scc_sweep.py: config generation + sbatch helpers
                                             # (imports predictor/decomp_space.py itself)
import wtune_runlength                      # repo root: resolve_run_seconds, the single-sourced
                                             # run_*-keys-vs-date-span resolver (dependency-free)

DEFAULT_HOURS = 12
DEFAULT_EVERY = 3
DEFAULT_STEPS = 30
DEFAULT_K = 7                       # + anchor = 8 configs, matching the task's "default 8"
DEFAULT_SEED = 1
DEFAULT_NODES = 4
DEFAULT_CORES_PER_NODE = 72
DEFAULT_HARDWARE_ID = "scc-gh200"
DEFAULT_TAU_THRESHOLD = 0.7
DEFAULT_PCT_THRESHOLD = 0.02
DEFAULT_ROOT = "$HOME/wtune-runs/temporal"
DEFAULT_TIMEOUT_S = 1200
DEFAULT_TIMELINE_K = 2               # anchor + up to 2 alternatives = up to 3 configs (task spec)
DEFAULT_LONG_S_PER_STEP_FALLBACK = 5.0     # task spec: "steps x s_est ... else 5 s/step"
DEFAULT_ENV_SOURCE = "$HOME/wtune-scc/env.sh"
DEFAULT_SBATCH_SCRIPT = os.path.join(_HERE, "temporal_run.sbatch")

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"


# ---------------------------------------------------------------------------
# restart time offsets
# ---------------------------------------------------------------------------
def restart_offsets_hours(hours, every):
    """[0, every, 2*every, ...] up to and including `hours` -- default hours=12, every=3 gives
    [0, 3, 6, 9, 12] (the task's own {t0, +3h, +6h, +9h, +12h} example). If `hours` is not an
    exact multiple of `every`, `hours` itself is still appended as the final point: the
    pre-registered criteria (tau>=0.7, within-2%) are evaluated at the FINAL T, so it must always
    be in the list even when it does not land on an `every`-aligned boundary."""
    if hours <= 0 or every <= 0:
        return [0]
    offs = list(range(0, hours + 1, every))
    if offs[-1] != hours:
        offs.append(hours)
    return offs


def offset_datetime(start_dt, hours_offset):
    return start_dt + datetime.timedelta(hours=hours_offset)


def fmt_wrf_time(dt):
    return dt.strftime(WRF_TIME_FMT)


def steps_to_minutes(steps, dt_seconds):
    """Ceil(steps*dt_seconds / 60) -- same rounding convention campaign/strict_campaign.sbatch's
    own MIN=$(( (STEPS*DT+59)/60 )) uses, so a SAMPLES run's namelist run length is computed the
    same way the rest of the campaign already computes it."""
    total_s = int(round(steps * dt_seconds))
    return (total_s + 59) // 60


def resolve_run_minutes(full_hours, hours_cap):
    """Effective LONG/TIMELINE run length in whole MINUTES -- min(full_hours, hours_cap) when the
    template's own length is known, else hours_cap alone, rounded to the nearest minute (never
    floored to whole hours first).

    Jobs 587 (lafe) and 588 (isc21) were killed at their 20-minute SLURM wall limit from TWO
    separate, compounding defects, neither of which is "HOURS=0 correctly means the full case" --
    that claim was checked against the real cluster namelists and was FALSE for both cases, for
    two different reasons (see read_run_length_hours's own docstring for the full mechanism):

    isc21 (run_minutes=30, every other run_* key 0, start/end date span 15h): a genuine 30-minute
    case. The OLD `hours = min(int(full_hours), args.hours)` computed int(0.5) == 0, and that 0
    used to be exported as HOURS and fed straight into the namelist edit -- which does NOT mean
    "run 30 minutes"; because campaign/configs.py's edit_namelist always writes literal
    run_days/run_hours/run_minutes values (never skips the write), HOURS=0 there means
    run_minutes=0, which is WRF's OWN trigger for "derive the run length from start_date ->
    end_date instead" -- i.e. it silently turned a 30-minute run into a 15-HOUR one. This function
    fixes that: resolve_run_minutes(0.5, 12) == 30, and cmd_submit_timeline now plumbs this exact
    minute count to the namelist edit via RUN_MINUTES (see build_timeline_sbatch_command), never
    through HOURS*60.

    lafe (all four run_* keys 0, start/end date span 24h): read_run_length_hours(text) used to
    return 0.0 here NOT because the case is short, but because its length is expressed only via
    start_date/end_date, which the old implementation never read at all. That 0.0 is what this
    function used to receive as `full_hours` and floor to 1 minute (max(1, ...)) -- sizing a
    24-hour job for 1 minute. read_run_length_hours now resolves this case itself (computing the
    real 24h span when every run_* key is 0), so `full_hours` reaching this function is already
    correct; this function's own min(full_hours, hours_cap)*60 logic did not need to change."""
    if full_hours is not None:
        hours = min(full_hours, hours_cap) if hours_cap else full_hours
    else:
        hours = hours_cap or DEFAULT_HOURS
    return max(1, int(round(hours * 60)))


# ---------------------------------------------------------------------------
# case metadata
# ---------------------------------------------------------------------------
def case_json(case_id):
    c = campaign_configs.CASES.get(case_id)
    if c is None:
        raise SystemExit("unknown case %r (not in campaign/cases.json)" % (case_id,))
    return c


def case_namelist_path(case):
    return campaign_configs._case_namelist_path(case)


def read_case_template(case):
    """(text, path) of the case's real namelist template. text is None when the file is not
    reachable from this machine (e.g. prep_dir lives only on the cluster) -- callers fall back to
    an explicit --hours/--cases from the operator in that case, per this tool's own CLI help."""
    path = case_namelist_path(case)
    try:
        with open(path, errors="replace") as f:
            return f.read(), path
    except OSError:
        return None, path


def _first_namelist_int(text, key, default=None):
    found = campaign_configs._find_namelist_key(text, key)
    if not found or not found["values"]:
        return default
    return int(float(found["values"][0]))


def read_case_start_datetime(text):
    """d01's start_year/month/day/hour/minute/second (its first value -- every active domain of
    a temporal-validation restart is aligned to the same wall time). minute/second default to 0
    since many templates omit start_second (and occasionally start_minute) when it is 0. Returns
    None when start_year/month/day cannot all be found (not a real WRF namelist)."""
    y = _first_namelist_int(text, "start_year")
    mo = _first_namelist_int(text, "start_month")
    d = _first_namelist_int(text, "start_day")
    if y is None or mo is None or d is None:
        return None
    h = _first_namelist_int(text, "start_hour", 0)
    mi = _first_namelist_int(text, "start_minute", 0)
    s = _first_namelist_int(text, "start_second", 0)
    return datetime.datetime(y, mo, d, h, mi, s)


def read_case_end_datetime(text):
    """d01's end_year/month/day/hour/minute/second (its first value) -- same convention as
    read_case_start_datetime (minute/second default to 0, only y/mo/d are mandatory). Returns
    None when end_year/month/day cannot all be found."""
    y = _first_namelist_int(text, "end_year")
    mo = _first_namelist_int(text, "end_month")
    d = _first_namelist_int(text, "end_day")
    if y is None or mo is None or d is None:
        return None
    h = _first_namelist_int(text, "end_hour", 0)
    mi = _first_namelist_int(text, "end_minute", 0)
    s = _first_namelist_int(text, "end_second", 0)
    return datetime.datetime(y, mo, d, h, mi, s)


def read_run_length_hours(text, path=None):
    """The run length WRF will ACTUALLY use, in hours -- not merely what the run_days/run_hours/
    run_minutes/run_seconds &time_control keys literally sum to.

    When at least one of those four keys is nonzero, they mean exactly what they say:
    days*24 + hours + minutes/60 + seconds/3600 (0 for any key that is absent, matching WRF's own
    default) -- this is the isc21 shape (run_minutes=30, every other run_* key 0): a genuine
    30-minute run, returned as 0.5.

    When ALL FOUR are zero (or absent), WRF does NOT run a zero-length case -- it falls back to
    computing the run length from &time_control's start_date -> end_date instead (WRF's own
    documented behaviour once every run_days/run_hours/run_minutes/run_seconds is 0; verified
    against campaign/configs.py's _set_scalar/edit_namelist, which writes exactly these four keys
    verbatim and nothing else). This is the lafe shape (all four run_* keys 0, start=2017-08-23
    06:00 -> end=2017-08-24 06:00): a genuine 24-hour run, NOT a 0-hour one. Returning 0 here for
    that case previously made every downstream consumer (auto_select_cases, cmd_plan,
    cmd_submit_long, cmd_submit_timeline) treat a real 24-hour case as if it had no run length at
    all -- job 587/588's second, independent DEFECT, distinct from isc21's sub-hour truncation.

    Raises ValueError (citing `path` when given) if NEITHER source yields a positive length: 0 is
    never returned for what claims to be a runnable case, since every caller here already treats
    0 as "unset", not as a genuine zero-length run (WRF has no such thing).

    Thin adapter over wtune_runlength.resolve_run_seconds (the single-sourced resolver every
    other "how long does this case run" call site now also goes through) -- this function still
    owns its OWN raw-text field reads (_first_namelist_int / read_case_start_datetime /
    read_case_end_datetime; a different in-memory representation from wtune.py's/
    wtune_predict.py's own namelist parsers, see wtune_runlength.py's module docstring) and just
    hands the six already-extracted scalars to the shared resolver, converting its result to
    HOURS and re-wrapping its generic ValueError with this module's own path-citing message so
    this function's behaviour (and this module's own tests) are unchanged."""
    days = _first_namelist_int(text, "run_days", 0)
    hrs = _first_namelist_int(text, "run_hours", 0)
    mins = _first_namelist_int(text, "run_minutes", 0)
    secs = _first_namelist_int(text, "run_seconds", 0)
    start_dt = read_case_start_datetime(text)
    end_dt = read_case_end_datetime(text)
    try:
        run_s = wtune_runlength.resolve_run_seconds(days, hrs, mins, secs, start_dt, end_dt)
    except ValueError:
        raise ValueError(
            "read_run_length_hours: run_days/run_hours/run_minutes/run_seconds are all 0 AND "
            "start_date/end_date do not establish a positive span%s -- this namelist has no "
            "run length WRF could actually use"
            % (" (%s)" % path if path else ""))
    return run_s / 3600.0


def auto_select_cases(min_hours=DEFAULT_HOURS):
    """Every enabled campaign/cases.json (SITE-overlaid) case, in CASE_ORDER, whose namelist
    template is readable from THIS machine and whose own run length is >= min_hours. Cases whose
    template cannot be read here (the normal case on a dev checkout -- prep_dir lives on the
    cluster) are silently skipped, not counted as short: the caller must pass --cases explicitly
    to include them (see this module's own docstring / plan --help)."""
    out = []
    for case_id in campaign_configs.CASE_ORDER:
        c = campaign_configs.CASES.get(case_id)
        if not c or not c.get("enabled", True):
            continue
        text, path = read_case_template(c)
        if text is None:
            continue
        if read_run_length_hours(text, path) >= min_hours:
            out.append(case_id)
    return out


# ---------------------------------------------------------------------------
# namelist edits -- long run (cold start, netCDF restart writer) and samples (netCDF restart
# reader, starting at T)
# ---------------------------------------------------------------------------
def _force_netcdf_restart_io(text, restart_flag, restart_interval_minutes, max_dom):
    """io_form_input/boundary/restart -> 2 (netCDF, NEVER ADIOS2 -- see module docstring),
    restart on/off, restart_interval (minutes), and quilting off.

    Quilting-off is the same 2-line recipe campaign/configs.py's _apply_adios2 uses (duplicated
    here, not imported, since that helper is inseparable from ADIOS2/&namelist_adios2, which this
    tool must never write) -- an active I/O quilt group can hang WRF outright (that function's own
    comment, job history cited there), and this tool has no ADIOS2 writer path to replace it the
    way _apply_adios2 does."""
    for key in ("io_form_input", "io_form_boundary", "io_form_restart"):
        text = campaign_configs._set_or_insert_time_control_scalar(text, key, 2)
    text = campaign_configs._set_or_insert_time_control_scalar(
        text, "restart", ".true." if restart_flag else ".false.")
    text = campaign_configs._set_or_insert_time_control_scalar(
        text, "restart_interval", int(restart_interval_minutes))
    text = re.sub(r"(?msi)^[ \t]*&namelist_quilt\b.*?^[ \t]*/[ \t]*\n?", "", text)
    text = text.rstrip("\n") + "\n&namelist_quilt\n nio_tasks_per_group=0,\n nio_groups=1,\n/\n"
    return text


def _set_start_datetime(text, dt, max_dom):
    """Write start_year/month/day/hour/minute/second as a fresh max_dom-length comma list (every
    active domain restarts at the SAME wall time) via configs._replace_list_line -- the same
    delete+reinsert primitive campaign/configs.py already uses for history_interval. Writing
    exactly max_dom entries (not the template's original domain count, if different) is the same
    "harmless, WRF only reads the first max_dom entries" precedent already established for
    history_interval / cases.json's lafe max_dom-trim note."""
    for key, value in (
        ("start_year", dt.year), ("start_month", dt.month), ("start_day", dt.day),
        ("start_hour", dt.hour), ("start_minute", dt.minute), ("start_second", dt.second),
    ):
        text = campaign_configs._replace_list_line(text, key, [value] * max_dom)
    return text


def edit_namelist_long(text, nproc_x, nproc_y, numtiles, ranks, max_dom, run_minutes,
                        restart_every_minutes, dist="block", blocktime="0"):
    """LONG RUN namelist: cold start (restart=.false., template's own native start_* untouched),
    history off, netCDF restart written every restart_every_minutes minutes, run_minutes total.
    `dist`/`blocktime` are not namelist keys (the runner env-exports them for site_launch /
    KMP_BLOCKTIME) -- accepted here only so callers can pass one config dict uniformly with
    edit_namelist_sample; unused."""
    text = campaign_configs.edit_namelist(
        text, nproc_x, nproc_y, numtiles, run_minutes, max_dom=max_dom,
        history_mode="off", adios2=False, ranks=ranks)
    text = _force_netcdf_restart_io(text, restart_flag=False,
                                     restart_interval_minutes=restart_every_minutes, max_dom=max_dom)
    return text


def edit_namelist_sample(text, nproc_x, nproc_y, numtiles, ranks, max_dom, run_minutes,
                          restart_dt=None, dist="cyclic", blocktime="0"):
    """SAMPLES namelist, short run_minutes window. restart_dt=None means T=0: "t0 uses the
    normal cold start" (task spec) -- restart=.false., the template's own native start_* left
    untouched. Any other restart_dt means restart=.true. from that wall time (every active
    domain set to it). Either way: history off, netCDF restart IO (matching the long run's
    io_form_restart=2 so its wrfrst_* files are readable), and no restart write during this short
    window (HISTORY_NEVER_MINUTES, reused from campaign/configs.py: this run is far too short to
    ever reach it, same trick that module already uses to mean "never")."""
    text = campaign_configs.edit_namelist(
        text, nproc_x, nproc_y, numtiles, run_minutes, max_dom=max_dom,
        history_mode="off", adios2=False, ranks=ranks)
    text = _force_netcdf_restart_io(
        text, restart_flag=(restart_dt is not None),
        restart_interval_minutes=campaign_configs.HISTORY_NEVER_MINUTES, max_dom=max_dom)
    if restart_dt is not None:
        text = _set_start_datetime(text, restart_dt, max_dom)
    return text


def wrfrst_filenames(max_dom, restart_dt):
    """wrfrst_d01_<ts>, wrfrst_d02_<ts>, ... for every active domain -- WRF's own restart
    filename convention (share/module_io_wrf.F writes wrfrst_d<NN>_<valid time>), NN zero-padded
    to 2 digits, matching campaign/strict_campaign.sbatch's own `printf "d%02d"` convention."""
    ts = fmt_wrf_time(restart_dt)
    return ["wrfrst_d%02d_%s" % (d, ts) for d in range(1, max_dom + 1)]


# ---------------------------------------------------------------------------
# rsl timing parsing (per-step series -> drift curve)
# ---------------------------------------------------------------------------
_TIMING_RE = re.compile(r"on domain +(\d+): +([0-9.]+) elapsed")


def parse_domain1_step_series(rsl_text):
    """Full per-step s/step series for domain 1 (the drift curve's raw data), in order, from
    'Timing for main: time ... on domain N: X elapsed' rsl lines -- same source
    campaign/configs.py's step_stats() consumes for its median, but here the WHOLE series is
    kept. A nested domain's own line (N != 1) is skipped: domain 1's line already includes the
    nested integration (campaign/strict_campaign.sbatch's own comment on this, verified against
    isc21 job 1074515), so adding domain>1 lines would double-count."""
    out = []
    for line in rsl_text.splitlines():
        if "Timing for main" not in line:
            continue
        m = _TIMING_RE.search(line)
        if not m or int(m.group(1)) != 1:
            continue
        out.append(float(m.group(2)))
    return out


def median_s_per_step(step_values):
    """Median over step_values[1:] -- step 1 (init + first radiation call) excluded, same
    convention as campaign/configs.py's step_stats(). None when there are <1 remaining values."""
    rest = step_values[1:]
    return statistics.median(rest) if rest else None


def drift_curve(step_series, dt_seconds):
    """Median s/step per simulated hour (bucket = floor(cumulative simulated seconds at the
    START of the step / 3600)), normalised to hour 0's median. Step 1 (index 0) is excluded from
    every bucket, same convention as median_s_per_step/configs.step_stats. Returns a list of
    dicts sorted by sim_hour: sim_hour, median_s_per_step, n_steps, normalized (None if hour 0
    itself has no measured steps, e.g. a 1-step series)."""
    buckets = {}
    sim_time = 0.0
    for i, s in enumerate(step_series):
        if i > 0:
            hour = int(sim_time // 3600)
            buckets.setdefault(hour, []).append(s)
        sim_time += dt_seconds
    hour0 = buckets.get(0, [])
    hour0_median = statistics.median(hour0) if hour0 else None
    out = []
    for hour in sorted(buckets):
        vals = buckets[hour]
        med = statistics.median(vals)
        norm = (med / hour0_median) if hour0_median else None
        out.append(dict(sim_hour=hour, median_s_per_step=med, n_steps=len(vals), normalized=norm))
    return out


# ---------------------------------------------------------------------------
# Kendall tau, rankings, within-2%
# ---------------------------------------------------------------------------
def kendall_tau(a, b):
    """Kendall's tau-b for two same-length, index-paired sequences -- pairwise concordant/
    discordant count, O(n^2) (fine here: n <= ~10 configs). Returns float('nan') when either
    sequence is constant (denominator 0, no defined correlation)."""
    n = len(a)
    if n != len(b):
        raise ValueError("kendall_tau: sequences must be the same length")
    conc = disc = tie_a = tie_b = tie_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                tie_both += 1
            elif da == 0:
                tie_a += 1
            elif db == 0:
                tie_b += 1
            elif (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tie_a - tie_both) * (n0 - tie_b - tie_both))
    if denom == 0:
        return float("nan")
    return (conc - disc) / denom


def build_rankings(samples_rows):
    """samples_rows: iterable of dicts with restart_hours(int), config_id(str), ok(0/1 or
    truthy), steps(list[float]). Returns a list of {"restart_hours", "config_id",
    "median_s_per_step", "rank"} rows, one per (restart_hours, config_id), sorted by
    (restart_hours, rank). A row with ok!=1 (or steps too short to measure) gets
    median_s_per_step=None/rank=None and sorts after every ranked row at that T -- kept for
    visibility in rankings.csv, excluded from tau/within-2% (see summarize())."""
    by_t = {}
    for r in samples_rows:
        by_t.setdefault(r["restart_hours"], []).append(r)
    out = []
    for t in sorted(by_t):
        scored = []
        for r in by_t[t]:
            ok = str(r.get("ok", "")) in ("1", "True", "true")
            med = median_s_per_step(r["steps"]) if ok else None
            scored.append((r["config_id"], med))
        ranked = sorted(scored, key=lambda cm: (cm[1] is None, cm[1] if cm[1] is not None else 0.0))
        rank = 0
        for cid, med in ranked:
            rank = rank + 1 if med is not None else None
            out.append(dict(restart_hours=t, config_id=cid, median_s_per_step=med, rank=rank))
    return out


def summarize(rankings, tau_threshold=DEFAULT_TAU_THRESHOLD, pct_threshold=DEFAULT_PCT_THRESHOLD):
    """rankings: build_rankings() output. Returns (summary_rows, final_hours):

    summary_rows: one dict per restart_hours T, sorted by T:
      restart_hours, kendall_tau_vs_t0 (None at T==0, or when <2 configs are measured at both
        t0 and T), best_config (T's own argmin), best_config_t0 (t0's argmin),
      pct_over_using_t0_choice: (s of best_config_t0 measured AT T - s of best_config at T) /
        s of best_config at T -- i.e. the cost, in fraction, of having stuck with t0's pick
        instead of switching to T's actual best. None if either config has no usable measurement.
      within_pct_threshold: pct_over_using_t0_choice <= pct_threshold (None if not computable).
      pass_tau: kendall_tau_vs_t0 is not None and >= tau_threshold (always None at T==0).
      is_final: True for the row at the largest T (the pre-registered tau/within-2% criteria are
        evaluated there, e.g. +12h for the default hours=12/every=3 sweep).

    final_hours: the T value marked is_final, for convenience.
    Raises ValueError if rankings has no restart_hours==0 (t0) rows at all."""
    by_t = {}
    for row in rankings:
        by_t.setdefault(row["restart_hours"], {})[row["config_id"]] = row["median_s_per_step"]
    if 0 not in by_t:
        raise ValueError("summarize(): rankings has no restart_hours=0 (t0) baseline")
    t0_vals = by_t[0]
    t0_usable = [c for c, v in t0_vals.items() if v is not None]
    best_t0_cfg = min(t0_usable, key=lambda c: t0_vals[c]) if t0_usable else None

    final_hours = max(by_t)
    out = []
    for t in sorted(by_t):
        vals = by_t[t]
        usable = [c for c in vals if vals[c] is not None]
        best_t_cfg = min(usable, key=lambda c: vals[c]) if usable else None

        tau = None
        if t != 0:
            shared = [c for c in vals if c in t0_vals and t0_vals[c] is not None and vals[c] is not None]
            if len(shared) >= 2:
                tau = kendall_tau([t0_vals[c] for c in shared], [vals[c] for c in shared])

        pct_over = within2 = None
        if best_t0_cfg is not None and best_t_cfg is not None and vals.get(best_t0_cfg) is not None:
            s_t0cfg_at_t = vals[best_t0_cfg]
            s_best_t = vals[best_t_cfg]
            if s_best_t:
                pct_over = (s_t0cfg_at_t - s_best_t) / s_best_t
                within2 = pct_over <= pct_threshold

        pass_tau = (tau is not None) and not math.isnan(tau) and (tau >= tau_threshold)
        out.append(dict(
            restart_hours=t, kendall_tau_vs_t0=tau, best_config=best_t_cfg,
            best_config_t0=best_t0_cfg, pct_over_using_t0_choice=pct_over,
            within_pct_threshold=within2, pass_tau=(pass_tau if t != 0 else None),
            is_final=(t == final_hours),
        ))
    return out, final_hours


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def write_rankings_csv(path, rankings):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["restart_hours", "config_id", "median_s_per_step", "rank"])
        for r in rankings:
            w.writerow([r["restart_hours"], r["config_id"], r["median_s_per_step"], r["rank"]])


def write_summary_csv(path, summary_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["restart_hours", "kendall_tau_vs_t0", "best_config", "best_config_t0",
              "pct_over_using_t0_choice", "within_pct_threshold", "pass_tau", "is_final"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in summary_rows:
            w.writerow([r[k] for k in fields])


def write_drift_csv(path, drift_rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_hour", "median_s_per_step", "n_steps", "normalized"])
        for r in drift_rows:
            w.writerow([r["sim_hour"], r["median_s_per_step"], r["n_steps"], r["normalized"]])


def read_samples_ledger(path):
    """One row per (restart_hours, config) run, as written by tools/temporal_run.sbatch's
    SAMPLES loop: restart_hours,config_id,ranks,omp,nx,ny,numtiles,dist,rc,ok,steps (steps is
    ';'-joined floats, one per measured step of that short run)."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            steps_field = row.get("steps") or ""
            steps = [float(v) for v in steps_field.split(";") if v]
            out.append(dict(
                restart_hours=int(float(row["restart_hours"])), config_id=row["config_id"],
                ranks=row.get("ranks"), omp=row.get("omp"), nx=row.get("nx"), ny=row.get("ny"),
                numtiles=row.get("numtiles"), dist=row.get("dist"),
                rc=row.get("rc"), ok=row.get("ok"), steps=steps,
            ))
    return out


def read_long_timing_csv(path):
    """The 'step_index,s_per_step' CSV `parse-timing` (no --flatten) writes -- returns just the
    ordered s_per_step column."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(float(row["s_per_step"]))
    return out


def print_summary_table(case_id, summary_rows):
    print("=== temporal validation: %s ===" % case_id)
    header = "%-6s %8s %-10s %-10s %10s %8s %8s" % (
        "T(h)", "tau_t0", "best@T", "best@t0", "pct_over", "<=2%%?", "final")
    print(header)
    for r in summary_rows:
        tau = "%.3f" % r["kendall_tau_vs_t0"] if r["kendall_tau_vs_t0"] is not None else "-"
        pct = ("%.2f%%" % (r["pct_over_using_t0_choice"] * 100)
               if r["pct_over_using_t0_choice"] is not None else "-")
        w2 = ("yes" if r["within_pct_threshold"] else "no") if r["within_pct_threshold"] is not None else "-"
        print("%-6s %8s %-10s %-10s %10s %8s %8s" % (
            r["restart_hours"], tau, r["best_config"] or "-", r["best_config_t0"] or "-",
            pct, w2, "yes" if r["is_final"] else ""))
    final = next((r for r in summary_rows if r["is_final"]), None)
    if final is not None:
        verdict = "PASS" if (final["pass_tau"] and final["within_pct_threshold"]) else "FAIL"
        print("FINAL T=%sh: tau=%s within_2pct=%s -> %s (criteria: tau>=%.2f AND within-2%%)"
              % (final["restart_hours"], final["kendall_tau_vs_t0"], final["within_pct_threshold"],
                 verdict, DEFAULT_TAU_THRESHOLD))


# ---------------------------------------------------------------------------
# config generation (anchor + K sampled) -- delegates entirely to tools/scc_sweep.py
# ---------------------------------------------------------------------------
def build_sample_configs(case, case_json_, nodes, cores_per_node, hardware_id, k, seed):
    """anchor + up to k sampled, deduped, anchor wins ties -- exactly
    tools/scc_sweep.py's own build_configs(), reused verbatim (not re-derived) so the SAMPLES set
    always includes the same decomposition the LONG run itself was anchored at."""
    return scc_sweep.build_configs(case, case_json_, nodes, k, seed, hardware_id, cores_per_node)


def anchor_config(case_json_, nodes, cores_per_node, hardware_id):
    total_cores = nodes * cores_per_node
    return scc_sweep.build_anchor(case_json_, total_cores, nodes, hardware_id)


def sample_configs_path(root, case):
    return os.path.join(scc_sweep._expand(root), case, "samples_configs.json")


def ensure_samples_configs(root, case, nodes, cores_per_node, hardware_id, k, seed):
    """Loads <root>/<case>/samples_configs.json if it already exists (so every submit-samples
    call across all T's reuses the IDENTICAL set written once, by plan or by the first
    submit-samples call), else generates + writes it via build_sample_configs. Returns
    (configs_list, path)."""
    path = sample_configs_path(root, case)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["configs"], path
    cj = case_json(case)
    built = build_sample_configs(case, cj, nodes, cores_per_node, hardware_id, k, seed)
    if built["skip_reason"]:
        raise SystemExit("temporal_validation: %s" % built["skip_reason"])
    scc_sweep.write_configs_json(path, built["configs"])
    return built["configs"], path


def timeline_configs_path(root, case):
    return os.path.join(scc_sweep._expand(root), case, "timeline_configs.json")


def ensure_timeline_configs(root, case, nodes, cores_per_node, hardware_id, k, seed):
    """Same load-or-build-once convention as ensure_samples_configs, but capped at
    DEFAULT_TIMELINE_K alternatives (anchor + up to 2 = up to 3 configs total, this task's own
    "config x state interaction" spec) and written to its own timeline_configs.json path so a
    TIMELINE run's config set is never confused with (or overwritten by) a SAMPLES run's
    samples_configs.json for the same case/root."""
    k = min(k, DEFAULT_TIMELINE_K)
    path = timeline_configs_path(root, case)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["configs"], path
    cj = case_json(case)
    built = build_sample_configs(case, cj, nodes, cores_per_node, hardware_id, k, seed)
    if built["skip_reason"]:
        raise SystemExit("temporal_validation: %s" % built["skip_reason"])
    scc_sweep.write_configs_json(path, built["configs"])
    return built["configs"], path


# ---------------------------------------------------------------------------
# sbatch command builders (reusing scc_sweep._expand / default_sbatch_runner / submit_job)
# ---------------------------------------------------------------------------
def build_long_sbatch_command(case, nodes, cores_per_node, hours, every, time_limit, out_dir,
                               env_source=DEFAULT_ENV_SOURCE, sbatch_script=DEFAULT_SBATCH_SCRIPT,
                               site="scc"):
    export = "ALL,SITE=%s,CASE=%s,TV_MODE=long,HOURS=%d,EVERY=%d" % (site, case, hours, every)
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive --exclude=head "
              "--time=%s --job-name=tv-long-%s --output=%s/temporal-long-%%j.out --export=%s %s"
              % (nodes, cores_per_node, time_limit, case, out_dir, export,
                 scc_sweep._expand(sbatch_script)))
    return "source %s; %s" % (scc_sweep._expand(env_source), sbatch)


def build_samples_sbatch_command(case, restart_hours, long_run_dir, configs_path, steps,
                                  nodes, cores_per_node, time_limit, timeout_s, out_dir,
                                  env_source=DEFAULT_ENV_SOURCE, sbatch_script=DEFAULT_SBATCH_SCRIPT,
                                  site="scc"):
    export = ("ALL,SITE=%s,CASE=%s,TV_MODE=samples,RESTART_HOURS=%d,LONG_RUN_DIR=%s,"
              "CONFIG_FILE=%s,STEPS=%d,TIMEOUT_S=%d"
              % (site, case, restart_hours, long_run_dir, configs_path, steps, timeout_s))
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive --exclude=head "
              "--time=%s --job-name=tv-smp-%s-h%d --output=%s/temporal-samples-%%j.out --export=%s %s"
              % (nodes, cores_per_node, time_limit, case, restart_hours, out_dir, export,
                 scc_sweep._expand(sbatch_script)))
    return "source %s; %s" % (scc_sweep._expand(env_source), sbatch)


def build_timeline_sbatch_command(case, nodes, cores_per_node, hours, every, run_minutes,
                                   configs_path, time_limit, out_dir, aux_interval_minutes,
                                   aux_stream=None, frames_per_auxhist=2000, fields=None,
                                   env_source=DEFAULT_ENV_SOURCE, sbatch_script=DEFAULT_SBATCH_SCRIPT,
                                   site="scc"):
    """TV_MODE=timeline: same LONG-run shape (full length or --hours, cold start, netCDF restart
    every `every` hours) as build_long_sbatch_command, but looped over `configs_path` (anchor +
    up to DEFAULT_TIMELINE_K alternatives, see ensure_timeline_configs) inside ONE sbatch job --
    same CONFIG_FILE-driven loop tools/temporal_run.sbatch's own SAMPLES branch already uses (see
    that file's header). aux_stream=None lets the sbatch script auto-detect a free stream against
    the case's own namelist template (tools/state_timeline.py pick-aux-stream) rather than fixing
    one here at plan time, since only the runner has the exact template text at hand.

    RUN_MINUTES (job 587/588 postmortem, correction 5) is exported ALONGSIDE HOURS, not instead of
    it: HOURS is still whatever cmd_submit_timeline's own `hours` (int, possibly 0) is, kept for
    any other consumer of that var, but the sbatch script's TV_MODE=timeline branch now passes
    RUN_MINUTES -- not HOURS -- to tools/state_timeline.py's edit-namelist-timeline --run-minutes,
    so the namelist's actual run length is whatever resolve_run_minutes() computed (e.g. isc21's
    genuine 30 minutes), never HOURS*60 (which would silently re-truncate a sub-hour length to 0,
    or an already-capped length to a coarser multiple of 60 minutes)."""
    export = ("ALL,SITE=%s,CASE=%s,TV_MODE=timeline,HOURS=%d,EVERY=%d,RUN_MINUTES=%d,CONFIG_FILE=%s,"
              "AUX_INTERVAL_MIN=%d,FRAMES_PER_AUXHIST=%d"
              % (site, case, hours, every, run_minutes, configs_path, aux_interval_minutes,
                 frames_per_auxhist))
    if aux_stream is not None:
        export += ",AUX_STREAM=%d" % aux_stream
    if fields:
        export += ",AUX_FIELDS=%s" % ",".join(fields)
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive --exclude=head "
              "--time=%s --job-name=tv-tl-%s --output=%s/temporal-timeline-%%j.out --export=%s %s"
              % (nodes, cores_per_node, time_limit, case, out_dir, export,
                 scc_sweep._expand(sbatch_script)))
    return "source %s; %s" % (scc_sweep._expand(env_source), sbatch)


# ---------------------------------------------------------------------------
# CLI: plan
# ---------------------------------------------------------------------------
def cmd_plan(args):
    cases = args.cases if args.cases else auto_select_cases(args.hours)
    if not cases:
        raise SystemExit(
            "plan: no case both enabled and >= --hours=%d h was found via a LOCALLY readable "
            "namelist template -- pass --cases explicitly (and, if the template is also "
            "unreadable here, --hours is then taken as given, not measured)" % args.hours)
    for case in cases:
        cj = case_json(case)
        text, path = read_case_template(cj)
        full_hours = read_run_length_hours(text) if text is not None else None
        if full_hours is not None:
            hours = min(int(full_hours), args.hours) if args.hours else int(full_hours)
            hours_source = "measured(%s)" % path
        else:
            hours = args.hours or DEFAULT_HOURS
            hours_source = "unmeasured (template unreadable at %s), using --hours" % path
        offsets = restart_offsets_hours(hours, args.every)
        built = build_sample_configs(case, cj, args.nodes, args.cores_per_node,
                                      args.hardware_id, args.k, args.seed)
        print("%-14s hours=%-3d (%s) every=%-2d restart_times=%s configs=%d (%s)"
              % (case, hours, hours_source, args.every, offsets,
                 len(built["configs"]), built["skip_reason"] or "ok"))
        if not built["skip_reason"] and not args.dry_run:
            path_out = sample_configs_path(args.root, case)
            scc_sweep.write_configs_json(path_out, built["configs"])
            print("  wrote %s" % path_out)
    return 0


# ---------------------------------------------------------------------------
# CLI: submit-long / submit-samples
# ---------------------------------------------------------------------------
def cmd_submit_long(args):
    cj = case_json(args.case)
    text, path = read_case_template(cj)
    full_hours = read_run_length_hours(text) if text is not None else None
    hours = min(int(full_hours), args.hours) if full_hours is not None else args.hours
    anchor = anchor_config(cj, args.nodes, args.cores_per_node, args.hardware_id)
    if anchor is None:
        raise SystemExit("submit-long: no legal decomposition for case %r at %d cores"
                          % (args.case, args.nodes * args.cores_per_node))
    s_est, s_source, _n = scc_sweep.estimate_s_per_step(
        args.case, args.nodes, args.ledger_dir, args.hardware_id,
        fallback=DEFAULT_LONG_S_PER_STEP_FALLBACK)
    steps_est = max(1, int(hours * 3600 / cj["dt"]))
    time_est = scc_sweep.estimate_time(1, 1, steps_est, s_est)
    out_dir = scc_sweep._expand(args.out_dir)
    cmd_str = build_long_sbatch_command(
        args.case, args.nodes, args.cores_per_node, hours, args.every, time_est["time_limit"],
        out_dir, env_source=args.env_source, sbatch_script=args.sbatch_script, site=args.site)
    print("LONG case=%s hours=%d every=%d anchor=(ranks=%d,omp=%d,nx=%d,ny=%d) "
          "s_est=%.3fs/step[%s] time=%s" % (
              args.case, hours, args.every, anchor["ranks"], anchor["omp"], anchor["nx"],
              anchor["ny"], s_est, s_source, time_est["time_limit"]))
    if args.dry_run:
        print("DRY-RUN: %s" % cmd_str)
        return 0
    job_id = scc_sweep.submit_job(cmd_str, sbatch_runner=args.sbatch_runner)
    print("submitted job %s (output under %s)" % (job_id, out_dir))
    return 0


def cmd_submit_samples(args):
    cj = case_json(args.case)
    configs_list, configs_path = ensure_samples_configs(
        args.root, args.case, args.nodes, args.cores_per_node, args.hardware_id, args.k, args.seed)
    restart_hours_list = (args.restart_times if args.restart_times is not None
                           else restart_offsets_hours(args.hours, args.every))
    out_dir = scc_sweep._expand(args.out_dir)
    time_est = scc_sweep.estimate_time(len(configs_list), 1, args.steps,
                                        scc_sweep.CONSERVATIVE_S_PER_STEP)
    for t in restart_hours_list:
        cmd_str = build_samples_sbatch_command(
            args.case, t, args.long_run_dir, os.path.abspath(configs_path), args.steps,
            args.nodes, args.cores_per_node, time_est["time_limit"], time_est["timeout_s"],
            out_dir, env_source=args.env_source, sbatch_script=args.sbatch_script, site=args.site)
        print("SAMPLES case=%s T=+%dh configs=%d steps=%d time=%s" % (
            args.case, t, len(configs_list), args.steps, time_est["time_limit"]))
        if args.dry_run:
            print("DRY-RUN: %s" % cmd_str)
            continue
        job_id = scc_sweep.submit_job(cmd_str, sbatch_runner=args.sbatch_runner)
        print("  submitted job %s" % job_id)
    return 0


def cmd_submit_timeline(args):
    """TIMELINE mode (decision 0053 step 2's own training-data extension, tools/state_timeline.py):
    one sbatch job runs the case's full length (or --hours) at anchor + up to
    DEFAULT_TIMELINE_K alternative decompositions, each writing the auxiliary state stream
    tools/temporal_run.sbatch's TV_MODE=timeline branch configures. Config set generation and
    sbatch-command shape otherwise mirror cmd_submit_long/cmd_submit_samples exactly (same
    hours-resolution, same s/step-ledger time estimate, same --dry-run contract).

    Two DEFECT fixes from the job 587 (lafe)/588 (isc21) postmortem, both pre-registered
    invariants, not negotiable:

    DEFECT A (both the wall limit AND the actual namelist run length were wrong for a run length
    below one hour, or one expressed only via start_date/end_date): steps_est is now derived from
    resolve_run_minutes()'s MINUTE-precision effective run length (fed by read_run_length_hours,
    itself now correct for both the isc21 shape -- genuine sub-hour run_minutes -- and the lafe
    shape -- all run_* keys 0, length defined by the date span; see both functions' own
    docstrings), never from int(hours). The SAME run_minutes value is now also exported as
    RUN_MINUTES and plumbed all the way to tools/state_timeline.py's edit-namelist-timeline
    --run-minutes (see build_timeline_sbatch_command), so the namelist's actual run length always
    matches what steps_est assumed -- HOURS/HOURS*60 no longer drives the TIMELINE namelist edit
    at all.

    DEFECT B (wall limit sized off the unmeasured 5s/step constant): when
    scc_sweep.estimate_s_per_step falls back to DEFAULT_LONG_S_PER_STEP_FALLBACK (source
    'fallback_10s', meaning: no ledger row for this case/node-count/hardware combination), that
    estimate is routinely off by close to an order of magnitude (isc21 measured 4.94s/step at 8
    nodes vs. the 5.0s/step *1-node* fallback used for job 588 -- roughly 8x low once node-count
    scaling is accounted for) and WILL undersize --time. Refuse to submit in that case (still
    printing the full plan first, even under --dry-run, so an operator sees exactly what would be
    refused) unless --s-per-step overrides it with a measured value (source 'override')."""
    cj = case_json(args.case)
    text, path = read_case_template(cj)
    full_hours = read_run_length_hours(text, path) if text is not None else None
    hours = min(int(full_hours), args.hours) if full_hours is not None else args.hours
    run_minutes = resolve_run_minutes(full_hours, args.hours)
    configs_list, configs_path = ensure_timeline_configs(
        args.root, args.case, args.nodes, args.cores_per_node, args.hardware_id, args.k, args.seed)
    if args.s_per_step is not None:
        s_est, s_source = args.s_per_step, "override"
    else:
        s_est, s_source, _n = scc_sweep.estimate_s_per_step(
            args.case, args.nodes, args.ledger_dir, args.hardware_id,
            fallback=DEFAULT_LONG_S_PER_STEP_FALLBACK)
    steps_est = max(1, int(run_minutes * 60 / cj["dt"]))
    # One sbatch job runs every config in configs_list sequentially (temporal_run.sbatch's
    # CONFIG_FILE loop) -- n_configs multiplies the single-run estimate, same
    # n_configs*reps*per_run shape scc_sweep.estimate_time already uses for SAMPLES-style loops.
    time_est = scc_sweep.estimate_time(len(configs_list), 1, steps_est, s_est)
    out_dir = scc_sweep._expand(args.out_dir)
    cmd_str = build_timeline_sbatch_command(
        args.case, args.nodes, args.cores_per_node, hours, args.every, run_minutes,
        os.path.abspath(configs_path), time_est["time_limit"], out_dir, args.aux_interval_minutes,
        aux_stream=args.aux_stream, frames_per_auxhist=args.frames_per_auxhist,
        fields=args.fields, env_source=args.env_source, sbatch_script=args.sbatch_script,
        site=args.site)
    print("TIMELINE case=%s hours=%d every=%d configs=%d aux_interval=%dmin run_min=%d "
          "s_est=%.3fs/step[%s] time=%s" % (
              args.case, hours, args.every, len(configs_list), args.aux_interval_minutes,
              run_minutes, s_est, s_source, time_est["time_limit"]))
    if args.dry_run:
        print("DRY-RUN: %s" % cmd_str)
    if s_source == "fallback_10s":
        raise SystemExit(
            "submit-timeline: refusing to submit -- s/step estimate for case=%s at %d node(s) on "
            "hardware=%s has no ledger row (source=fallback_10s, %.1fs/step constant), which is "
            "typically off by close to an order of magnitude and undersizes --time (see jobs 587/"
            "588, both killed at their 20-minute wall limit). Produce a ledger row first:\n"
            "  tools/scc_sweep.py submit --cases %s --nodes %d --hardware-id %s\n"
            "then re-run this submit-timeline command, or pass --s-per-step <measured s/step> to "
            "override." % (args.case, args.nodes, args.hardware_id,
                            DEFAULT_LONG_S_PER_STEP_FALLBACK, args.case, args.nodes,
                            args.hardware_id))
    if args.dry_run:
        return 0
    job_id = scc_sweep.submit_job(cmd_str, sbatch_runner=args.sbatch_runner)
    print("submitted job %s (output under %s)" % (job_id, out_dir))
    return 0


# ---------------------------------------------------------------------------
# CLI: analyze
# ---------------------------------------------------------------------------
def cmd_analyze(args):
    samples_paths = args.samples_csv or sorted(glob.glob(
        os.path.join(scc_sweep._expand(args.raw_dir), "temporal_%s_samples_*.csv" % args.case)))
    if not samples_paths:
        raise SystemExit("analyze: no samples ledger CSV found (pass --samples-csv or "
                          "--raw-dir); expected temporal_%s_samples_*.csv" % args.case)
    rows = []
    for p in samples_paths:
        rows.extend(read_samples_ledger(p))
    rankings = build_rankings(rows)
    summary_rows, _final = summarize(rankings, args.tau_threshold, args.pct_threshold)

    case_root = os.path.join(scc_sweep._expand(args.root), args.case)
    write_rankings_csv(os.path.join(case_root, "rankings.csv"), rankings)
    write_summary_csv(os.path.join(case_root, "summary.csv"), summary_rows)
    print_summary_table(args.case, summary_rows)

    long_csv = args.long_timing_csv
    if not long_csv:
        cand = sorted(glob.glob(os.path.join(
            scc_sweep._expand(args.raw_dir), "temporal_%s_long_*_timing.csv" % args.case)))
        long_csv = cand[-1] if cand else None
    if long_csv:
        dt_seconds = args.dt or case_json(args.case)["dt"]
        series = read_long_timing_csv(long_csv)
        drift_rows = drift_curve(series, dt_seconds)
        write_drift_csv(os.path.join(case_root, "drift.csv"), drift_rows)
        print("drift curve: %d simulated-hour buckets from %s (%s)"
              % (len(drift_rows), long_csv, case_root))
    else:
        print("WARNING: no long-run timing CSV found -- drift.csv not written "
              "(pass --long-timing-csv explicitly)")
    print("wrote %s/{rankings,summary%s}.csv" % (case_root, ",drift" if long_csv else ""))
    return 0


# ---------------------------------------------------------------------------
# CLI: internal plumbing for tools/temporal_run.sbatch
# ---------------------------------------------------------------------------
def _shq(v):
    return "'%s'" % str(v).replace("'", "'\\''")


def cmd_case_metadata(args):
    cj = case_json(args.case)
    max_dom = campaign_configs._case_effective_max_dom(cj)
    print("PREP_DIR=%s" % _shq(cj["prep_dir"]))
    print("DT=%s" % _shq(cj["dt"]))
    print("MAX_DOM=%s" % _shq(max_dom))
    print("INPUT_FILES=%s" % _shq(" ".join(cj["input_files"])))
    print("NAMELIST_TEMPLATE=%s" % _shq(case_namelist_path(cj)))
    print("CASE_FAMILY=%s" % _shq(cj.get("case_family", "")))
    return 0


def cmd_anchor_config(args):
    cj = case_json(args.case)
    a = anchor_config(cj, args.nodes, args.cores_per_node, args.hardware_id)
    if a is None:
        raise SystemExit("anchor-config: no legal decomposition for case %r at %d cores"
                          % (args.case, args.nodes * args.cores_per_node))
    print("RANKS=%s" % _shq(a["ranks"]))
    print("OMP=%s" % _shq(a["omp"]))
    print("NX=%s" % _shq(a["nx"]))
    print("NY=%s" % _shq(a["ny"]))
    print("NUMTILES=%s" % _shq(a["numtiles"]))
    print("DIST=%s" % _shq(a["dist"]))
    print("BLOCKTIME=%s" % _shq(a["blocktime"]))
    return 0


def cmd_restart_timestamp(args):
    cj = case_json(args.case)
    text, path = read_case_template(cj)
    if text is None:
        raise SystemExit("restart-timestamp: cannot read namelist template at %s" % path)
    start_dt = read_case_start_datetime(text)
    if start_dt is None:
        raise SystemExit("restart-timestamp: no start_year/month/day in %s" % path)
    print(fmt_wrf_time(offset_datetime(start_dt, args.restart_hours)))
    return 0


def cmd_edit_namelist_long(args):
    with open(args.infile) as f:
        text = f.read()
    every_minutes = args.every * 60
    run_minutes = args.hours * 60
    out = edit_namelist_long(text, args.nproc_x, args.nproc_y, args.numtiles, args.ranks,
                              args.max_dom, run_minutes, every_minutes)
    with open(args.outfile, "w") as f:
        f.write(out)
    return 0


def cmd_edit_namelist_sample(args):
    with open(args.infile) as f:
        text = f.read()
    cj = case_json(args.case)
    restart_dt = None
    if args.restart_hours != 0:
        tmpl_text, path = read_case_template(cj)
        if tmpl_text is None:
            raise SystemExit("edit-namelist-sample: cannot read namelist template at %s" % path)
        start_dt = read_case_start_datetime(tmpl_text)
        if start_dt is None:
            raise SystemExit("edit-namelist-sample: no start_year/month/day in %s" % path)
        restart_dt = offset_datetime(start_dt, args.restart_hours)
    run_minutes = steps_to_minutes(args.steps, cj["dt"])
    out = edit_namelist_sample(text, args.nproc_x, args.nproc_y, args.numtiles, args.ranks,
                                args.max_dom, run_minutes, restart_dt)
    with open(args.outfile, "w") as f:
        f.write(out)
    return 0


def cmd_parse_timing(args):
    with open(args.rsl_file, errors="replace") as f:
        text = f.read()
    series = parse_domain1_step_series(text)
    if args.flatten:
        print(";".join("%.6f" % v for v in series))
        return 0
    w = csv.writer(sys.stdout)
    w.writerow(["step_index", "s_per_step"])
    for i, v in enumerate(series, start=1):
        w.writerow([i, v])
    return 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------
def _add_generation_args(sp):
    sp.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    sp.add_argument("--cores-per-node", type=int, default=DEFAULT_CORES_PER_NODE, dest="cores_per_node")
    sp.add_argument("--hardware-id", default=DEFAULT_HARDWARE_ID, dest="hardware_id")
    sp.add_argument("--k", type=int, default=DEFAULT_K)
    sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sp.add_argument("--root", default=DEFAULT_ROOT)


def _add_submit_args(sp):
    sp.add_argument("--out-dir", default=None, dest="out_dir")
    sp.add_argument("--env-source", default=DEFAULT_ENV_SOURCE, dest="env_source")
    sp.add_argument("--sbatch-script", default=DEFAULT_SBATCH_SCRIPT, dest="sbatch_script")
    sp.add_argument("--site", default="scc")
    sp.add_argument("--ledger-dir", default="$HOME/wtune-runs/campaign", dest="ledger_dir")
    sp.add_argument("--dry-run", action="store_true", dest="dry_run")


def build_arg_parser():
    p = argparse.ArgumentParser(prog="temporal_validation.py")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("plan")
    sp.add_argument("--cases", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                     default=None)
    sp.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    sp.add_argument("--every", type=int, default=DEFAULT_EVERY)
    sp.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    sp.add_argument("--dry-run", action="store_true", dest="dry_run")
    _add_generation_args(sp)

    sp = sub.add_parser("submit-long")
    sp.add_argument("--case", required=True)
    sp.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    sp.add_argument("--every", type=int, default=DEFAULT_EVERY)
    _add_generation_args(sp)
    _add_submit_args(sp)

    sp = sub.add_parser("submit-samples")
    sp.add_argument("--case", required=True)
    sp.add_argument("--long-run-dir", required=True, dest="long_run_dir")
    sp.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    sp.add_argument("--every", type=int, default=DEFAULT_EVERY)
    sp.add_argument("--restart-times", type=lambda s: [int(x) for x in s.split(",") if x.strip()],
                     default=None, dest="restart_times")
    sp.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    _add_generation_args(sp)
    _add_submit_args(sp)

    sp = sub.add_parser("submit-timeline")
    sp.add_argument("--case", required=True)
    sp.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    sp.add_argument("--every", type=int, default=DEFAULT_EVERY)
    sp.add_argument("--aux-interval-minutes", type=int, default=30, dest="aux_interval_minutes")
    sp.add_argument("--aux-stream", type=int, default=None, dest="aux_stream",
                     help="fix the aux stream number instead of auto-detecting on the runner")
    sp.add_argument("--frames-per-auxhist", type=int, default=2000, dest="frames_per_auxhist")
    sp.add_argument("--fields", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                     default=None, help="override the default 2-D field list (comma-separated)")
    sp.add_argument("--s-per-step", type=float, default=None, dest="s_per_step",
                     help="measured s/step, bypassing the no-ledger-row refusal (source 'override')")
    _add_generation_args(sp)
    sp.set_defaults(k=DEFAULT_TIMELINE_K)
    _add_submit_args(sp)

    sp = sub.add_parser("analyze")
    sp.add_argument("--case", required=True)
    sp.add_argument("--root", default=DEFAULT_ROOT)
    sp.add_argument("--raw-dir", default="$HOME/wtune-runs/campaign", dest="raw_dir")
    sp.add_argument("--samples-csv", nargs="*", default=None, dest="samples_csv")
    sp.add_argument("--long-timing-csv", default=None, dest="long_timing_csv")
    sp.add_argument("--dt", type=float, default=None)
    sp.add_argument("--tau-threshold", type=float, default=DEFAULT_TAU_THRESHOLD, dest="tau_threshold")
    sp.add_argument("--pct-threshold", type=float, default=DEFAULT_PCT_THRESHOLD, dest="pct_threshold")

    sp = sub.add_parser("case-metadata")
    sp.add_argument("case")

    sp = sub.add_parser("anchor-config")
    sp.add_argument("case")
    _add_generation_args(sp)

    sp = sub.add_parser("restart-timestamp")
    sp.add_argument("case")
    sp.add_argument("restart_hours", type=int)

    sp = sub.add_parser("edit-namelist-long")
    sp.add_argument("infile")
    sp.add_argument("outfile")
    sp.add_argument("--hours", type=int, required=True)
    sp.add_argument("--every", type=int, required=True)
    sp.add_argument("--nproc-x", type=int, required=True, dest="nproc_x")
    sp.add_argument("--nproc-y", type=int, required=True, dest="nproc_y")
    sp.add_argument("--numtiles", type=int, required=True)
    sp.add_argument("--ranks", type=int, required=True)
    sp.add_argument("--max-dom", type=int, default=1, dest="max_dom")

    sp = sub.add_parser("edit-namelist-sample")
    sp.add_argument("infile")
    sp.add_argument("outfile")
    sp.add_argument("--case", required=True)
    sp.add_argument("--restart-hours", type=int, required=True, dest="restart_hours")
    sp.add_argument("--steps", type=int, required=True)
    sp.add_argument("--nproc-x", type=int, required=True, dest="nproc_x")
    sp.add_argument("--nproc-y", type=int, required=True, dest="nproc_y")
    sp.add_argument("--numtiles", type=int, required=True)
    sp.add_argument("--ranks", type=int, required=True)
    sp.add_argument("--max-dom", type=int, default=1, dest="max_dom")

    sp = sub.add_parser("parse-timing")
    sp.add_argument("rsl_file")
    sp.add_argument("--dt", type=float, default=None)  # accepted, unused (kept for symmetry with the sbatch caller)
    sp.add_argument("--flatten", action="store_true")

    p.set_defaults(sbatch_runner=scc_sweep.default_sbatch_runner)
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    handlers = {
        "plan": cmd_plan,
        "submit-long": cmd_submit_long,
        "submit-samples": cmd_submit_samples,
        "submit-timeline": cmd_submit_timeline,
        "analyze": cmd_analyze,
        "case-metadata": cmd_case_metadata,
        "anchor-config": cmd_anchor_config,
        "restart-timestamp": cmd_restart_timestamp,
        "edit-namelist-long": cmd_edit_namelist_long,
        "edit-namelist-sample": cmd_edit_namelist_sample,
        "parse-timing": cmd_parse_timing,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
