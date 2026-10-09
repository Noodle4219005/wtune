#!/usr/bin/env python3
"""tools/temporal_validation.py -- offline temporal validation (decision 0053 step 2): does a config
ranking measured in a forecast's first minutes still hold 3-12 simulated hours later?

LONG run (per case): full length or --hours (12) at the anchor decomposition, restart every
--every hours (3), history off; its step series is the drift curve. SAMPLES: at each restart
time T the same N configs (anchor + K diverse, fixed seed) run --steps (30) from T. Pass
(pre-registered): Kendall tau >= 0.7 and t0's best within 2 % of T's best at the final T.
Restart I/O is netCDF only: ADIOS2 restart corrupts the heap when rank placement changes.
CLI: plan, submit-long, submit-samples, analyze (others are plumbing for temporal_run.sbatch).
Reuses campaign/configs.py namelist edits and tools/scc_sweep.py; needs Python >= 3.7."""
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
from typing import Any

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

DEFAULT_HOURS = 12           # simulated h
DEFAULT_EVERY = 3            # h between restart samples
DEFAULT_STEPS = 30           # d01 steps per sample window
DEFAULT_K = 7                       # + anchor = 8 configs, matching the task's "default 8"
DEFAULT_SEED = 1
DEFAULT_NODES = 4
DEFAULT_CORES_PER_NODE = 72  # scc GH200
DEFAULT_HARDWARE_ID = "scc-gh200"
DEFAULT_TAU_THRESHOLD = 0.7  # Kendall tau, pre-registered
DEFAULT_PCT_THRESHOLD = 0.02  # fraction: t0's pick within 2 % of T's best
DEFAULT_ROOT = "$HOME/wtune-runs/temporal"
DEFAULT_TIMEOUT_S = 1200     # s per sample window
DEFAULT_TIMELINE_K = 2               # anchor + up to 2 alternatives = up to 3 configs (task spec)
DEFAULT_LONG_S_PER_STEP_FALLBACK = 5.0     # task spec: "steps x s_est ... else 5 s/step"
DEFAULT_ENV_SOURCE = "$HOME/wtune-scc/env.sh"
DEFAULT_SBATCH_SCRIPT = os.path.join(_HERE, "temporal_run.sbatch")

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"


# ---------------------------------------------------------------------------
# restart time offsets
# ---------------------------------------------------------------------------
def restart_offsets_hours(hours: float, every: float) -> list[float]:
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


def offset_datetime(start_dt: datetime.datetime, hours_offset: float) -> datetime.datetime:
    return start_dt + datetime.timedelta(hours=hours_offset)


def fmt_wrf_time(dt: datetime.datetime) -> str:
    return dt.strftime(WRF_TIME_FMT)


def steps_to_minutes(steps: int, dt_seconds: float) -> int:
    """Ceil(steps*dt_seconds / 60) -- same rounding convention campaign/strict_campaign.sbatch's
    own MIN=$(( (STEPS*DT+59)/60 )) uses, so a SAMPLES run's namelist run length is computed the
    same way the rest of the campaign already computes it."""
    total_s = int(round(steps * dt_seconds))
    return (total_s + 59) // 60


def resolve_run_minutes(full_hours: float | None, hours_cap: float) -> int:
    """Effective LONG/TIMELINE run length in whole minutes: min(full_hours, hours_cap), or
    hours_cap when the length is unknown, rounded (never floored to hours). int(0.5 h) == 0 once
    turned isc21's 30-minute run into a 15-hour one via run_minutes=0 (jobs 587/588)."""
    if full_hours is not None:
        hours = min(full_hours, hours_cap) if hours_cap else full_hours
    else:
        hours = hours_cap or DEFAULT_HOURS
    return max(1, int(round(hours * 60)))


# ---------------------------------------------------------------------------
# case metadata
# ---------------------------------------------------------------------------
def case_json(case_id: str) -> dict:
    c = campaign_configs.CASES.get(case_id)
    if c is None:
        raise SystemExit("unknown case %r (not in campaign/cases.json)" % (case_id,))
    return c


def case_namelist_path(case: dict) -> str:
    return campaign_configs._case_namelist_path(case)


def read_case_template(case: dict) -> tuple[str | None, str]:
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


def read_case_start_datetime(text: str) -> datetime.datetime | None:
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


def read_case_end_datetime(text: str) -> datetime.datetime | None:
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


def read_run_length_hours(text: str, path: str | None = None) -> float:
    """Run length WRF will actually use (h): the run_days/hours/minutes/seconds sum when any is
    nonzero (isc21: 0.5), else the start_date -> end_date span (lafe: 24). ValueError (citing
    path) when neither is positive -- 0 is never returned. Adapter over
    wtune_runlength.resolve_run_seconds with this module's own raw-text field reads."""
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


def auto_select_cases(min_hours: float = DEFAULT_HOURS) -> list:
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
    """io_form_input/boundary/restart = 2 (netCDF, never ADIOS2), restart on/off,
    restart_interval (min), quilting off (an active quilt group can hang WRF; same recipe as
    configs._apply_adios2, duplicated because that helper also writes ADIOS2 settings)."""
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


def edit_namelist_long(text: str, nproc_x: int, nproc_y: int, numtiles: int, ranks: int,
                       max_dom: int, run_minutes: int, restart_every_minutes: int,
                       dist: str = "block", blocktime: str = "0") -> str:
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


def edit_namelist_sample(text: str, nproc_x: int, nproc_y: int, numtiles: int, ranks: int,
                         max_dom: int, run_minutes: int, restart_dt: datetime.datetime | None = None,
                         dist: str = "cyclic", blocktime: str = "0") -> str:
    """SAMPLES namelist for a short window: restart_dt None = t0 cold start (start_* untouched),
    else restart=.true. from restart_dt on every domain. History off, netCDF restart reads, no
    restart write in the window (HISTORY_NEVER_MINUTES)."""
    text = campaign_configs.edit_namelist(
        text, nproc_x, nproc_y, numtiles, run_minutes, max_dom=max_dom,
        history_mode="off", adios2=False, ranks=ranks)
    text = _force_netcdf_restart_io(
        text, restart_flag=(restart_dt is not None),
        restart_interval_minutes=campaign_configs.HISTORY_NEVER_MINUTES, max_dom=max_dom)
    if restart_dt is not None:
        text = _set_start_datetime(text, restart_dt, max_dom)
    return text


def wrfrst_filenames(max_dom: int, restart_dt: datetime.datetime) -> list[str]:
    """wrfrst_d01_<ts>, wrfrst_d02_<ts>, ... for every active domain -- WRF's own restart
    filename convention (share/module_io_wrf.F writes wrfrst_d<NN>_<valid time>), NN zero-padded
    to 2 digits, matching campaign/strict_campaign.sbatch's own `printf "d%02d"` convention."""
    ts = fmt_wrf_time(restart_dt)
    return ["wrfrst_d%02d_%s" % (d, ts) for d in range(1, max_dom + 1)]


# ---------------------------------------------------------------------------
# rsl timing parsing (per-step series -> drift curve)
# ---------------------------------------------------------------------------
_TIMING_RE = re.compile(r"on domain +(\d+): +([0-9.]+) elapsed")


def parse_domain1_step_series(rsl_text: str) -> list:
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


def median_s_per_step(step_values: list[float]) -> float | None:
    """Median over step_values[1:] -- step 1 (init + first radiation call) excluded, same
    convention as campaign/configs.py's step_stats(). None when there are <1 remaining values."""
    rest = step_values[1:]
    return statistics.median(rest) if rest else None


def drift_curve(step_series: list, dt_seconds: float) -> list[dict]:
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
def kendall_tau(a: list[float], b: list[float]) -> float:
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


def build_rankings(samples_rows: list[dict]) -> list[dict]:
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


def summarize(rankings: list[dict], tau_threshold: float = DEFAULT_TAU_THRESHOLD,
              pct_threshold: float = DEFAULT_PCT_THRESHOLD) -> tuple[list[dict], float]:
    """build_rankings() rows -> (summary_rows sorted by T, final_hours). Per T: restart_hours,
    kendall_tau_vs_t0, best_config, best_config_t0, pct_over_using_t0_choice (cost of keeping t0's
    pick at T), within_pct_threshold, pass_tau, is_final (largest T, where the pre-registered
    criteria apply). None where not computable. ValueError without t0 rows."""
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
def write_rankings_csv(path: str, rankings: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["restart_hours", "config_id", "median_s_per_step", "rank"])
        for r in rankings:
            w.writerow([r["restart_hours"], r["config_id"], r["median_s_per_step"], r["rank"]])


def write_summary_csv(path: str, summary_rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["restart_hours", "kendall_tau_vs_t0", "best_config", "best_config_t0",
              "pct_over_using_t0_choice", "within_pct_threshold", "pass_tau", "is_final"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in summary_rows:
            w.writerow([r[k] for k in fields])


def write_drift_csv(path: str, drift_rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_hour", "median_s_per_step", "n_steps", "normalized"])
        for r in drift_rows:
            w.writerow([r["sim_hour"], r["median_s_per_step"], r["n_steps"], r["normalized"]])


def read_samples_ledger(path: str) -> list[dict]:
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


def read_long_timing_csv(path: str) -> list:
    """The 'step_index,s_per_step' CSV `parse-timing` (no --flatten) writes -- returns just the
    ordered s_per_step column."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(float(row["s_per_step"]))
    return out


def print_summary_table(case_id: str, summary_rows: list[dict]) -> None:
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
def build_sample_configs(case: str, case_json_: dict, nodes: int, cores_per_node: int,
                         hardware_id: str, k: int, seed: int) -> Any:
    """anchor + up to k sampled, deduped, anchor wins ties -- exactly
    tools/scc_sweep.py's own build_configs(), reused verbatim (not re-derived) so the SAMPLES set
    always includes the same decomposition the LONG run itself was anchored at."""
    return scc_sweep.build_configs(case, case_json_, nodes, k, seed, hardware_id, cores_per_node)


def anchor_config(case_json_: dict, nodes: int, cores_per_node: int, hardware_id: str) -> Any:
    total_cores = nodes * cores_per_node
    return scc_sweep.build_anchor(case_json_, total_cores, nodes, hardware_id)


def sample_configs_path(root: str, case: str) -> str:
    return os.path.join(scc_sweep._expand(root), case, "samples_configs.json")


def ensure_samples_configs(root: str, case: str, nodes: int, cores_per_node: int,
                           hardware_id: str, k: int, seed: int) -> tuple[list[dict], str]:
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


def timeline_configs_path(root: str, case: str) -> str:
    return os.path.join(scc_sweep._expand(root), case, "timeline_configs.json")


def ensure_timeline_configs(root: str, case: str, nodes: int, cores_per_node: int,
                            hardware_id: str, k: int, seed: int) -> tuple[list[dict], str]:
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
def build_long_sbatch_command(case: str, nodes: int, cores_per_node: int, hours: int, every: float,
                              time_limit: str, out_dir: str, env_source: str = DEFAULT_ENV_SOURCE,
                              sbatch_script: str = DEFAULT_SBATCH_SCRIPT, site: str = "scc") -> str:
    export = "ALL,SITE=%s,CASE=%s,TV_MODE=long,HOURS=%d,EVERY=%d" % (site, case, hours, every)
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive --exclude=head "
              "--time=%s --job-name=tv-long-%s --output=%s/temporal-long-%%j.out --export=%s %s"
              % (nodes, cores_per_node, time_limit, case, out_dir, export,
                 scc_sweep._expand(sbatch_script)))
    return "source %s; %s" % (scc_sweep._expand(env_source), sbatch)


def build_samples_sbatch_command(case: str, restart_hours: float, long_run_dir: str,
                                 configs_path: str, steps: int, nodes: int, cores_per_node: int,
                                 time_limit: str, timeout_s: int, out_dir: str,
                                 env_source: str = DEFAULT_ENV_SOURCE,
                                 sbatch_script: str = DEFAULT_SBATCH_SCRIPT, site: str = "scc") -> str:
    export = ("ALL,SITE=%s,CASE=%s,TV_MODE=samples,RESTART_HOURS=%d,LONG_RUN_DIR=%s,"
              "CONFIG_FILE=%s,STEPS=%d,TIMEOUT_S=%d"
              % (site, case, restart_hours, long_run_dir, configs_path, steps, timeout_s))
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive --exclude=head "
              "--time=%s --job-name=tv-smp-%s-h%d --output=%s/temporal-samples-%%j.out --export=%s %s"
              % (nodes, cores_per_node, time_limit, case, restart_hours, out_dir, export,
                 scc_sweep._expand(sbatch_script)))
    return "source %s; %s" % (scc_sweep._expand(env_source), sbatch)


def build_timeline_sbatch_command(case: str, nodes: int, cores_per_node: int, hours: int,
                                  every: float, run_minutes: int, configs_path: str,
                                  time_limit: str, out_dir: str, aux_interval_minutes: float,
                                  aux_stream: int | None = None, frames_per_auxhist: int = 2000,
                                  fields: str | None = None, env_source: str = DEFAULT_ENV_SOURCE,
                                  sbatch_script: str = DEFAULT_SBATCH_SCRIPT,
                                  site: str = "scc") -> str:
    """TV_MODE=timeline sbatch command: the LONG-run shape looped over configs_path (anchor + up to
    DEFAULT_TIMELINE_K) in one job. aux_stream None = the runner picks a free stream. RUN_MINUTES
    (resolve_run_minutes) drives the namelist run length, not HOURS*60 (jobs 587/588)."""
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
def cmd_plan(args: argparse.Namespace) -> int:
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
def cmd_submit_long(args: argparse.Namespace) -> int:
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


def cmd_submit_samples(args: argparse.Namespace) -> int:
    case_json(args.case)  # result unused; kept so an unknown case still fails here, before any submission
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


def cmd_submit_timeline(args: argparse.Namespace) -> int:
    """TIMELINE mode: one job runs the full length (or --hours) at anchor + up to
    DEFAULT_TIMELINE_K alternatives, writing the auxiliary state stream; otherwise as
    cmd_submit_long/samples (--dry-run contract included). steps_est and RUN_MINUTES both come from
    resolve_run_minutes (jobs 587/588). Refuses to submit when s/step is the unmeasured fallback
    ('fallback_10s', ~8x low for isc21) unless --s-per-step gives a measured value; the plan is
    printed either way."""
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
def cmd_analyze(args: argparse.Namespace) -> int:
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


def cmd_case_metadata(args: argparse.Namespace) -> int:
    cj = case_json(args.case)
    max_dom = campaign_configs._case_effective_max_dom(cj)
    print("PREP_DIR=%s" % _shq(cj["prep_dir"]))
    print("DT=%s" % _shq(cj["dt"]))
    print("MAX_DOM=%s" % _shq(max_dom))
    print("INPUT_FILES=%s" % _shq(" ".join(cj["input_files"])))
    print("NAMELIST_TEMPLATE=%s" % _shq(case_namelist_path(cj)))
    print("CASE_FAMILY=%s" % _shq(cj.get("case_family", "")))
    return 0


def cmd_anchor_config(args: argparse.Namespace) -> int:
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


def cmd_restart_timestamp(args: argparse.Namespace) -> int:
    cj = case_json(args.case)
    text, path = read_case_template(cj)
    if text is None:
        raise SystemExit("restart-timestamp: cannot read namelist template at %s" % path)
    start_dt = read_case_start_datetime(text)
    if start_dt is None:
        raise SystemExit("restart-timestamp: no start_year/month/day in %s" % path)
    print(fmt_wrf_time(offset_datetime(start_dt, args.restart_hours)))
    return 0


def cmd_edit_namelist_long(args: argparse.Namespace) -> int:
    with open(args.infile) as f:
        text = f.read()
    every_minutes = args.every * 60
    run_minutes = args.hours * 60
    out = edit_namelist_long(text, args.nproc_x, args.nproc_y, args.numtiles, args.ranks,
                              args.max_dom, run_minutes, every_minutes)
    with open(args.outfile, "w") as f:
        f.write(out)
    return 0


def cmd_edit_namelist_sample(args: argparse.Namespace) -> int:
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


def cmd_parse_timing(args: argparse.Namespace) -> int:
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


def build_arg_parser() -> argparse.ArgumentParser:
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


def main(argv: list[str] | None = None) -> int:
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
