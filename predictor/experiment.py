#!/usr/bin/env python3
"""predictor/experiment.py -- WTune decision-0050 step 5: minimal experiment engine.

Turns a candidate set into ONE pre-registered, cost-stated Slurm job for the existing
campaign/strict_campaign.sbatch runner, then reconciles cost against sacct and scores the
result. First use: leave-one-case-out (LOCO) validation of the STRICT recommender
(predictor/recommend.py recommend()) against each registered campaign/cases.json case --
does the model's top recommendation actually beat (or roughly match) the best config anyone
has ever measured for that case, when the model is never allowed to train on that case?

Pipeline (decision-0050's step 8 table, step 5 row):
    candidate set -> pre-registered pass/fail criterion -> cost estimate -> ONE sbatch
    command (never submitted here) -> `record` stores the job id -> `reconcile` reads sacct
    + the resulting campaign ledger -> verdict.

Four subcommands, one pipeline stage each:
    plan       --case C --ledger L [L...] --namelist N --hardware-id H [--nodes K]
               [--top-k 3] [--loco] [--configs explicit.json] [--reps 2] --out DIR
    record     --plan DIR --job-id N
    reconcile  --plan DIR [--ledger-dir D] [--sacct-cmd sacct]
    summarize  --plans DIR [DIR...]

Dependency boundary (load-bearing): everything in this file is plain-stdlib-importable
(campaign/configs.py and campaign/decomp_summary.py are both stdlib-only, like this file) so
`plan --configs ...`, `record`, `reconcile` and `summarize` run with a bare python3 -- no
scikit-learn/catboost/lightgbm needed. The ONE exception is `plan` without --configs, which
calls predictor/recommend.py's recommend() to build the candidate shortlist; that import is
deferred to the call site and raises a clear, actionable error if the predictor's ML stack
(scikit-learn at minimum; catboost/lightgbm optional with their own fallback chain inside
recommend.py) is not importable -- see _call_recommend().

Generated space (--space generated, or auto-selected -- see `plan`'s --space below): for a case
whose grid is too small for every C01-C16 template (every template is anchored at 448/224/112
ranks; WRF needs floor(extent/nproc) >= 10 on every active domain, share/module_check_a_mundo.F:
3014-3029), resolve_case() resolves NO C01 and the template path above has no baseline to plan
against. predictor/decomp_space.py (enumerate_space/legal_shapes_exact/rsl_patch_extents) is
ALSO plain-stdlib-importable (no numpy/sklearn -- verified: it imports campaign/configs.py and
predictor/features.py only) and is imported directly by _generated_baseline(), same as
campaign_configs/dsum above -- the generated-space BASELINE is always buildable with no ML stack
at all. Only the generated space's RECOMMENDED candidates need predictor/decomp_search.py (via
_call_decomp_search_propose()), which -- like recommend.py -- needs scikit-learn/catboost
(predictor/models.py) and is therefore ALSO deferred to its call site with the same clear-error
contract as _call_recommend().

Deliberately NOT imported: predictor/ledger.py (imports repo-root wtune_compliance.py) and
predictor/benchmark.py (imports predictor/models.py, i.e. sklearn) -- this file reads/writes
campaign ledger CSVs itself via the stdlib csv module using the canonical LEDGER_HEADER
below (verbatim copy of campaign/strict_campaign.sbatch's own LEDGER_HEADER, kept in sync by
hand, not by import) instead. This is also why wtune_compliance.py/test_wtune_compliance.py
being edited concurrently by another agent cannot break this module even transiently.

CONFIG_FILE vs CONFIGS (architectural stop-rule note, campaign/strict_campaign.sbatch:224-248
and campaign/configs.py:1376-1402): a case's STRICT candidates from recommend() (--space
templates, the only mode this file calls) are ALWAYS resolved campaign/configs.py TEMPLATES
ids (C01-C16) -- recommend.build_candidates() calls the exact same campaign_configs.resolve_
case() this file calls directly. CONFIG_FILE and CONFIGS are mutually exclusive selectors for
the runner's default (non-MODE=physics) loop: available_decomps() reads either the CONFIG_
FILE-named generated set OR resolve_case()'s STRICT templates, never both at once. So: when
every candidate in a plan is a template id, the sbatch command selects them via
CONFIGS=Cxx+Cyy (the already-tested resolve_case()/TEMPLATES path -- the smallest faithful
option). configs.json is still always written in the CONFIG_FILE JSON shape (a documented,
round-trippable record of exactly what was measured, and the mechanism used for every
non-template candidate: a predictor/decomp_search.py proposal, a predictor/decomp_space.py
generated decomposition -- see "Generated space" above -- or a future kind not yet invented)
-- when a plan ever contains a non-template candidate (every candidate in a --space generated
plan always is, since GBASE/GREC* ids never appear in campaign_configs.resolve_case()'s
TEMPLATES table), the sbatch command switches to CONFIG_FILE=configs.json instead (which must
then carry EVERY candidate, template ids included, since CONFIG_FILE replaces the whole config
set for that job).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import shlex
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
for _p in (_HERE, _REPO_ROOT, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as campaign_configs     # campaign/configs.py: CASES, resolve_case, COST_GUESS,
                                        # load_generated_configs -- stdlib only
import decomp_summary as dsum          # campaign/decomp_summary.py: summarize()/row_identity()/
                                        # config_id() -- stdlib only, reused verbatim for
                                        # best_known and reconcile's per-config measured stats

# ---------------------------------------------------------------------------
# Canonical ledger schema -- verbatim copy of campaign/strict_campaign.sbatch:94's
# LEDGER_HEADER (== predictor/ledger.py LEDGER_FIELDS, not imported -- see module docstring).
# ---------------------------------------------------------------------------
LEDGER_HEADER = [
    "run_id", "source_dataset", "job_id", "rep_index", "case_raw", "case_base", "case_family",
    "hardware_id", "binary_kind", "compliance_class",
    "e_we", "e_sn", "e_vert", "dx", "dy", "time_step",
    "mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics",
    "ra_lw_physics", "ra_sw_physics", "radt", "cudt",
    "history_interval", "io_form_history",
    "n_nodes", "n_ranks", "omp_threads", "nproc_x", "nproc_y", "numtiles", "distribution",
    "kmp_blocktime",
    "s_per_step_median", "s_per_step_mean", "n_steps_measured",
    "diffwrf_differing_fields", "fidelity_label",
    "profile_micro_share", "profile_radiation_share", "profile_dynamics_share",
    "profile_mpi_share", "profile_io_share", "profile_rank_imbalance", "profile_thread_imbalance",
    "notes",
]

DEFAULT_NODES = 4
DEFAULT_REPS = 2
DEFAULT_TOP_K = 3
DEFAULT_STARTUP_S = 60.0
CORES_PER_NODE = 112
SU_PER_CORE_HOUR = 0.2                  # AGENTS.md cost model: 0.2 per core-hour
PASS_THRESHOLD_MULTIPLIER = 1.02        # pre-registered PASS criterion (decision-0050 step 5)
FIXED_JOB_OVERHEAD_MIN = 5.0            # module load / setup overhead, once per job
TIME_SAFETY_FACTOR = 1.5
TIME_PAD_MIN = 10
PARTITION_FULL = "cf448"
PARTITION_SMALL = "cf112"
SBATCH_SCRIPT_REL = "campaign/strict_campaign.sbatch"   # relative to repo root, matches
                                                          # predictor/decomp_search.py's own
                                                          # sbatch_command() convention
BASELINE_ID = "C01"
GENERATED_BASELINE_ID = "GBASE"          # generated-space C01 analogue, see GENERATED_BASELINE_RULE
GENERATED_REC_PREFIX = "GREC"            # generated-space recommended:i candidate ids, GREC1..GRECk
GENERATED_BASELINE_RULE = (
    "generated-space baseline (C01 analogue, _generated_baseline()): omp=1, ranks = the MAXIMUM "
    "legal rank count <= total_cores (scanned descending via predictor/decomp_space.py's "
    "legal_shapes_exact, decomp_space.py:143), numtiles=1, dist=cyclic, blocktime='0'; among "
    "legal (nx,ny) shapes at that rank count, pick the least aspect of d01's RSL min-patch "
    "(max(patch_x,patch_y)/min(patch_x,patch_y) via decomp_space.rsl_patch_extents, "
    "decomp_space.py:107 -- the same metric decomp_space.seed_design()'s own aspect() helper "
    "uses, decomp_space.py:636-638, for the identical 'cold-start, no C01' scenario), tie-broken "
    "by legal_shapes_exact's own ascending-nx factor_pairs order (decomp_space.py:102,150)."
)
DEFAULT_LEDGER_DIR = "/home/u5727520/wrf-bench/wscore/campaign"   # strict_campaign.sbatch's
                                                                    # own $OUT; only a fallback,
                                                                    # always override with
                                                                    # --ledger-dir off-cluster
DEFAULT_RESULTS_DIR = os.path.join(_REPO_ROOT, "results", "derived-data")

# PART C: feasibility screen. A cheap MODE=timing/REPS=1/STEPS=SCREEN_STEPS probe of top_k +
# --screen-spares candidates, run BEFORE committing to the full REPS/full-steps timing job, so
# an infeasible candidate (WRF-Fire omp>1 crash, elevconv/bigchill numtiles>=12 timeout -- see
# PART A/B above) is caught even under --loco, where PART B's same-case history is deliberately
# blind to the target case. TIMEOUT_S is a PER-RUN watchdog (strict_campaign.sbatch's own
# `timeout "$TIMEOUT_S" srun ...`, rc=124 on expiry) -- independent of --time (the whole job's
# Slurm wall-clock, via _cost_estimate/TIME_SAFETY_FACTOR).
SCREEN_STEPS = 5
SCREEN_TIMEOUT_MULTIPLIER = 3.0
SCREEN_TIMEOUT_MIN_S = 120

CRITERION_TEXT = (
    "PASS iff min over recommended configs of measured median s/step <= %.2fx best_known "
    "measured median in the same job; also report speedup of best recommended vs %s."
    % (PASS_THRESHOLD_MULTIPLIER, BASELINE_ID)
)


# ---------------------------------------------------------------------------
# small stdlib helpers
# ---------------------------------------------------------------------------
def _utcnow_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _float_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def _blocktime_str(v):
    """Recommend.py's ranked-candidate kmp_blocktime is an int (0) or None ('default');
    campaign/configs.py TEMPLATES/resolve_case instead use the string '0' or 'default' --
    normalise to the latter so cfg dicts built from either source compare equal."""
    if v is None or v == "default":
        return "default"
    return str(int(v)) if str(v).replace(".", "", 1).isdigit() else str(v)


def _git_head(repo_root=_REPO_ROOT):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                              capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def _format_hms(total_minutes):
    total_minutes = int(math.ceil(total_minutes))
    hh, mm = divmod(total_minutes, 60)
    return "%02d:%02d:00" % (hh, mm)


# ---------------------------------------------------------------------------
# ledger IO (stdlib csv only -- see module docstring for why predictor/ledger.py /
# predictor/benchmark.py are not imported)
# ---------------------------------------------------------------------------
def _read_ledger_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_ledgers(paths):
    rows = []
    for p in paths:
        rows.extend(_read_ledger_csv(p))
    return rows


def _write_merged_ledger(rows, dest_path):
    """Write `rows` (from one or more source campaign ledgers, arbitrary/mixed fieldnames) as
    ONE canonical-header CSV -- predictor/recommend.py's recommend() takes a single ledger_path
    (predictor/benchmark.py load_ledger() opens exactly one file), so multiple --ledger inputs
    must be merged before being handed to it."""
    with open(dest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_HEADER, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LEDGER_HEADER})


def _startup_s(rows):
    """startup_s from a ledger column if one exists (none does in the current LEDGER_HEADER --
    see strict_campaign.sbatch:94), else DEFAULT_STARTUP_S with startup_assumed=True."""
    if rows and "startup_s" in rows[0]:
        vals = [v for v in (_float_or_none(r.get("startup_s")) for r in rows) if v is not None]
        if vals:
            return statistics.median(vals), False
    return DEFAULT_STARTUP_S, True


# ---------------------------------------------------------------------------
# case / template resolution (campaign/configs.py -- the runner's own source of truth)
# ---------------------------------------------------------------------------
def _case_json(case):
    case_json = campaign_configs.CASES.get(case)
    if case_json is None:
        raise SystemExit("unknown case %r (not in campaign/cases.json)" % case)
    return case_json


def _resolve_case_configs(case):
    """id -> resolved cfg dict (ranks/omp/nx/ny/numtiles/dist/blocktime/patch_x/patch_y) for
    every campaign/configs.py TEMPLATES entry legal for this case -- exactly what
    `configs.py order/case CASE` (CONFIG_FILE unset) resolves for the runner, and exactly what
    recommend.build_candidates() calls internally (predictor/recommend.py:253)."""
    kept, skipped, aliases = campaign_configs.resolve_case(_case_json(case))
    return {cfg["id"]: cfg for cfg in kept}, skipped, aliases


def _rows_on_nodes(rows, nodes):
    """Only rows measured on the same node count as the planned job: a best_known or s/step
    taken from a 2- or 8-node run is neither runnable nor comparable on `nodes` nodes."""
    return [r for r in rows if _float_or_none(r.get("n_nodes")) == nodes]


def _best_known(full_rows, case):
    """(config_id, cfg_dict, measured_median_s_per_step, n) for the lowest-median-s/step
    STRICT config ever measured for `case` in `full_rows` (successful runs only), or None.
    Reuses campaign/decomp_summary.summarize() verbatim: it already groups by what was
    actually run (not by id), extracts config= from notes, and sorts ascending by median."""
    ranked = dsum.summarize(full_rows, case)
    if not ranked:
        return None
    best = ranked[0]
    ids = sorted(i for i in best["ids"] if i)
    if not ids:
        return None
    ranks, omp, nx, ny, numtiles, dist, blocktime = best["identity"][:7]
    ntx, nty, tsx, tsy, tstrat = best["identity"][7:]
    cfg = dict(ranks=ranks, omp=omp, nx=nx, ny=ny, numtiles=numtiles, dist=dist,
               blocktime=blocktime, numtiles_x=ntx, numtiles_y=nty, tile_sz_x=tsx,
               tile_sz_y=tsy, tile_strategy=tstrat, patch_x=None, patch_y=None)
    return ids[0], cfg, best["s_median"], best["n"]


def _s_per_step_estimates(full_rows, case, candidate_ids, case_json):
    """Per candidate: (s_per_step_est, source) where source is 'config_median' (this exact
    config measured for this case before), 'case_median' (this case measured under some other
    config), or 'cost_guess' (campaign/configs.py COST_GUESS, keyed by grid -- last resort,
    used only when this case has no measured STRICT row at all)."""
    ranked = dsum.summarize(full_rows, case)
    by_id = {}
    for g in ranked:
        for cid in g["ids"]:
            by_id.setdefault(cid, g["s_median"])

    case_values = [v for v in (
        _float_or_none(r.get("s_per_step_median")) for r in full_rows
        if r.get("case_base") == case and r.get("compliance_class") == "STRICT"
    ) if v is not None and v > 0]
    case_median = statistics.median(case_values) if case_values else None

    guess = campaign_configs.COST_GUESS.get(
        (case_json.get("e_we"), case_json.get("e_sn"), case_json.get("e_vert")))

    est, source = {}, {}
    for cid in candidate_ids:
        if cid in by_id:
            est[cid], source[cid] = by_id[cid], "config_median"
        elif case_median is not None:
            est[cid], source[cid] = case_median, "case_median"
        elif guess is not None:
            est[cid], source[cid] = guess[0], "cost_guess"
        else:
            raise SystemExit(
                "no s/step estimate available for case=%r config=%r: no ledger rows for this "
                "case and no campaign/configs.py COST_GUESS entry for its grid -- supply a "
                "--ledger with at least one measured row for this case" % (case, cid))
    return est, source


# ---------------------------------------------------------------------------
# PART A: feasibility ledger -- results/derived-data/2026-09-22-atlas-loco-phase1-verdicts.csv:
# 3 of 10 LOCO cases FAILED-infeasible because the recommender proposed configs that cannot
# actually run at all (WRF-Fire omp>1 crash rc=137/139; elevconv/bigchill 224x2 numtiles>=12
# always time out rc=124, 0/19 historically). Root cause: every candidate-building path above
# (_s_per_step_estimates, recommend.recommend(), decomp_search.propose_core) trains/ranks only on
# rows with a NUMERIC s_per_step_median -- a FAILED row (empty median, non-zero rc in notes) is
# silently dropped, so nothing ever learns a decomposition is unrunnable. This index reads that
# dropped half of the ledger directly.
# ---------------------------------------------------------------------------
def _is_physics_row(row):
    return dsum._notes_fields(row.get("notes")).get("mode") == "physics"


def row_is_decomposition_observation(row):
    """True if `row` should count toward _failure_index's per-(case,nodes,identity) ok/fail
    tally: STRICT compliance, and either not a MODE=physics row at all (an ordinary STRICT
    decomposition-only run), or a physics row whose OWN template is P00 (the physics reference --
    same decomposition as whatever DECOMP_ID was selected, no physics edit applied, notes'
    edits= empty -- campaign/strict_campaign.sbatch:621) -- ANY OTHER Pxx template's failure is a
    physics-EDIT failure (a namelist knob crashed WRF, not the decomposition), and must not
    pollute a feasibility index that answers "can this decomposition run at all"."""
    if row.get("compliance_class") != "STRICT":
        return False
    if not _is_physics_row(row):
        return True
    notes = dsum._notes_fields(row.get("notes"))
    return notes.get("template") == "P00" and not notes.get("edits")


def _row_rc(row):
    """int(notes' rc=<int> token), or None if absent/unparseable. strict_campaign.sbatch only
    ever writes rc= when the run's own exit code was non-zero (lines 628/906: `[ "$RC" != "0" ]
    && NOTES="$NOTES rc=$RC"`) -- a successful row (rc=0) never carries the token at all."""
    rc = dsum._notes_fields(row.get("notes")).get("rc")
    if rc is None:
        return None
    try:
        return int(rc)
    except ValueError:
        return None


def _failure_index(rows):
    """{(case_base, n_nodes, decomposition_identity): {'fail': n, 'ok': n, 'rcs': [...],
    'kinds': {'timeout'|'crash'}}} -- `decomposition_identity` is
    campaign/decomp_summary.row_identity(row) (ranks/omp/nx/ny/numtiles/dist/blocktime/tile
    geometry), the SAME physical-run grouping key decomp_summary.summarize() uses for
    best_known, so a candidate can be looked up here with the identical key
    _identity_from_cfg() builds from its own resolved cfg fields.

    Only rows passing row_is_decomposition_observation() are counted. 'ok' = a numeric
    s_per_step_median (the run measured normally). 'fail' = an empty median whose notes carry a
    non-zero rc= token (seen so far: 124 timeout, 137/139/1/38/174 crash -- see cases.json's
    c34/marshall/elevconv/bigchill notes); rc=124 is `timeout(1)`'s own SIGTERM exit code
    (strict_campaign.sbatch's `timeout "$TIMEOUT_S" srun ...`), every other non-zero rc is
    bucketed 'crash'. A row with neither a numeric median nor an rc= token (e.g. still running,
    or a row shape this index does not recognise) is silently skipped -- absence of evidence, not
    evidence of failure."""
    index = {}
    for r in rows:
        if not row_is_decomposition_observation(r):
            continue
        n_nodes = _float_or_none(r.get("n_nodes"))
        if n_nodes is None:
            continue
        key = (r.get("case_base"), int(n_nodes), dsum.row_identity(r))
        entry = index.setdefault(key, dict(fail=0, ok=0, rcs=[], kinds=set()))
        median = _float_or_none(r.get("s_per_step_median"))
        if median is not None:
            entry["ok"] += 1
            continue
        rc = _row_rc(r)
        if not rc:            # None or 0 -- no failure evidence either way
            continue
        entry["fail"] += 1
        entry["rcs"].append(rc)
        entry["kinds"].add("timeout" if rc == 124 else "crash")
    return index


def _identity_from_cfg(cfg):
    """decomp_summary.row_identity()-shaped 12-tuple built from a CANDIDATE's own resolved cfg
    dict (ranks/omp/nx/ny/numtiles/dist/blocktime[/numtiles_x/numtiles_y/tile_sz_x/tile_sz_y/
    tile_strategy]) instead of a measured ledger row, so an unmeasured candidate can be looked up
    in _failure_index's ledger-derived keys. blocktime is normalised via _blocktime_str first: a
    candidate's blocktime is an int/None (recommend.py shortlist entries) or a string
    (campaign_configs templates / decomp_search chosen dicts) depending on source, while
    row_identity's own bucket is always exactly '0' or 'default'."""
    blocktime = "0" if _blocktime_str(cfg.get("blocktime")) == "0" else "default"
    return (int(cfg["ranks"]), int(cfg["omp"]), int(cfg["nx"]), int(cfg["ny"]),
            int(cfg.get("numtiles") or 1), cfg.get("dist") or "", blocktime,
            int(cfg.get("numtiles_x") or 0), int(cfg.get("numtiles_y") or 0),
            int(cfg.get("tile_sz_x") or 0), int(cfg.get("tile_sz_y") or 0),
            int(cfg.get("tile_strategy") or 0))


def _exclude_known_infeasible(entries, get_cfg, get_label, case, nodes, failure_index):
    """PART B: split `entries` (recommend()'s shortlist dicts, or decomp_search's chosen dicts --
    NOT baseline/best_known, which are added separately by the caller after this filter runs)
    into (kept, excluded) against `failure_index` (see _failure_index above) on (case, nodes,
    identity): fail>=1 and ok==0 -> excluded, reason 'known_infeasible_on_case', with the rcs and
    kinds observed. fail>=1 and ok>=1 -> KEPT (mixed evidence -- flaky, not reliably infeasible)
    but tagged entry['flaky']=True in place, same convention _filter_max_omp's `excluded` dicts
    use for its own reason string.

    `failure_index` must be built from the SAME rows the caller's own --loco filtering already
    produced (train_rows, not full_rows) -- under --loco this makes the exclusion rule blind to
    the target case's own history too (no leakage), exactly like the recommender's training rows
    themselves; under a normal (non-LOCO) plan train_rows IS full_rows, so nothing changes there.
    """
    kept, excluded = [], []
    for e in entries:
        ident = _identity_from_cfg(get_cfg(e))
        info = failure_index.get((case, nodes, ident))
        if info and info["fail"] >= 1 and info["ok"] == 0:
            excluded.append(dict(label=get_label(e), reason="known_infeasible_on_case",
                                  rcs=list(info["rcs"]), kinds=sorted(info["kinds"])))
            continue
        if info and info["fail"] >= 1 and info["ok"] >= 1:
            e["flaky"] = True
        kept.append(e)
    return kept, excluded


# ---------------------------------------------------------------------------
# candidate assembly
# ---------------------------------------------------------------------------
def _load_explicit_ids(path):
    """--configs FILE: a bare JSON list of template ids (['C05','C06','C07']), a list of
    {'id': 'C05'} dicts, or {'configs': [...]} of either -- accepted so a hand-written
    reproduction file (e.g. karina numtiles 12/13/14) and a machine-generated one both work."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("configs", data.get("ids", data))
    ids = []
    for item in data:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and "id" in item:
            ids.append(str(item["id"]))
        else:
            raise ValueError("--configs entries must be a string id or {'id': ...} dict, got %r"
                              % (item,))
    return ids


def _call_recommend(ledger_path, namelist_path, case, hardware_id, nodes, top_k):
    try:
        import recommend as recommend_mod
    except ImportError as e:
        raise SystemExit(
            "predictor/recommend.py could not be imported (%s: %s) -- its dependency chain "
            "needs scikit-learn at minimum (catboost/lightgbm are optional, recommend.py "
            "already falls back when they are missing); install/activate the predictor pixi "
            "env, or pass --configs to bypass the recommender entirely and use plain stdlib "
            "template resolution instead" % (type(e).__name__, e))
    # --loco is applied by the CALLER (_build_candidates below drops case_base==case rows from
    # `full_rows` before this function ever sees them -- "the recommender never sees the target
    # case"), not via recommend()'s own --holdout-case: passing holdout_case=case here IN
    # ADDITION to a pre-filtered ledger would make recommend()'s own n_dropped bookkeeping
    # report "matched ZERO training rows -- nothing was actually held out" (predictor/
    # recommend.py ~line 606-608), which is actively misleading (something certainly was held
    # out, just earlier). One filter, one place.
    return recommend_mod.recommend(
        ledger_path=ledger_path, namelist_path=namelist_path, case_name=case,
        hardware_id=hardware_id, nodes=nodes, top_k=top_k)


class _Candidates(object):
    """Ordered dict of config_id -> candidate fields, with role accumulation + dedupe.
    A config_id can carry several roles (recommended rank k, best_known, baseline) --
    the FIRST call to add() for a given id sets its resolved decomposition fields (roles
    processed in add-order: explicit/recommended first, then baseline, then best_known, so
    the runner-facing resolution -- recommend()'s own, or campaign_configs.resolve_case()'s --
    wins over best_known's ledger-row-derived identity when both apply to the same id)."""

    def __init__(self):
        self._by_id = {}

    def add(self, config_id, cfg, role, **extra):
        entry = self._by_id.get(config_id)
        if entry is None:
            entry = dict(
                config_id=config_id, roles=[],
                ranks=cfg["ranks"], omp=cfg["omp"], nx=cfg["nx"], ny=cfg["ny"],
                numtiles=cfg["numtiles"], dist=cfg["dist"], blocktime=cfg["blocktime"],
                numtiles_x=cfg.get("numtiles_x", 0), numtiles_y=cfg.get("numtiles_y", 0),
                tile_sz_x=cfg.get("tile_sz_x", 0), tile_sz_y=cfg.get("tile_sz_y", 0),
                tile_strategy=cfg.get("tile_strategy", 0),
                patch_x=cfg.get("patch_x"), patch_y=cfg.get("patch_y"),
                predicted_gain=None, gain_lower_95=None, label=None,
                best_known_measured_median=None, best_known_measured_n=None,
                flaky=False,   # PART B: mixed ok/fail evidence on (case, nodes, identity) --
                                # see _exclude_known_infeasible; stays False for every entry
                                # whose add() call never passes flaky=True explicitly.
            )
            self._by_id[config_id] = entry
        if role not in entry["roles"]:
            entry["roles"].append(role)
        entry.update(extra)
        return entry

    def ids(self):
        return sorted(self._by_id)

    def as_dict(self):
        return self._by_id


def _filter_max_omp(entries, max_omp, get_omp, get_label):
    """Split `entries` into (kept, excluded) against case_json.get("max_omp") (None = no cap,
    every entry kept -- unchanged behaviour for every case without the key, e.g. c34/marshall's
    WRF-Fire omp>1 crash is the only case that sets it today). `get_omp`/`get_label` pull the omp
    thread count / a human-readable id out of one entry. `excluded` entries are plain dicts
    (label, omp, reason) -- the reason string plan.json records for the drop, matching the
    convention campaign/configs.py resolve_config uses for its own skip reasons. This is a
    defense-in-depth filter: campaign/configs.py resolve_case() already keeps omp>max_omp
    templates out of the template-path candidate pool, and decomp_search.propose_for_case's own
    max_omp filtering (once experiment.py passes the cap through) already keeps the generated
    space from proposing them -- but a candidate reaching here has already cleared both, so this
    is the last gate before it is written into plan.json/configs.json."""
    if max_omp is None:
        return list(entries), []
    kept, excluded = [], []
    for e in entries:
        omp = get_omp(e)
        if omp > max_omp:
            excluded.append(dict(label=get_label(e), omp=omp,
                                  reason="omp(%d) exceeds case max_omp=%d" % (omp, max_omp)))
        else:
            kept.append(e)
    return kept, excluded


def _build_candidates(args, case_json, id_to_cfg, full_rows):
    cands = _Candidates()
    recommend_info = None

    if args.configs:
        explicit_ids = _load_explicit_ids(args.configs)
        for i, cid in enumerate(explicit_ids, start=1):
            if cid not in id_to_cfg:
                raise SystemExit(
                    "--configs id %r is not a legal resolved template for case %r -- legal "
                    "ids: %s" % (cid, args.case, sorted(id_to_cfg)))
            cands.add(cid, id_to_cfg[cid], "explicit:%d" % i)
    else:
        # --loco: drop every row whose case_base == this case BEFORE recommend() is called at
        # all, so the recommender never sees the target case. best_known below always reads
        # `full_rows` (unfiltered), never `train_rows` -- LOCO blinds only the model.
        train_rows = full_rows
        if args.loco:
            train_rows = [r for r in full_rows if r.get("case_base") != args.case]
        merged_path = os.path.join(args.out, "merged_ledger.csv")
        _write_merged_ledger(train_rows, merged_path)
        result = _call_recommend(merged_path, args.namelist, args.case, args.hardware_id,
                                  args.nodes, args.top_k)
        recommend_info = dict(
            model_used=result.get("model_used"),
            magnitude_model_used=result.get("magnitude_model_used"),
            warnings=result.get("warnings"),
            loco=bool(args.loco),
            n_ledger_rows_total=len(full_rows), n_ledger_rows_used=len(train_rows),
            merged_ledger=merged_path,
        )
        shortlist, excluded_omp = _filter_max_omp(
            result.get("shortlist", []), case_json.get("max_omp"),
            lambda e: e["omp_threads"], lambda e: e["config_id"])
        # PART B: same-case exclusion -- built from train_rows (LOCO-filtered above), never
        # full_rows, so under --loco this rule is exactly as blind to the target case's own
        # history as the recommender itself (no leakage); under a normal plan train_rows ==
        # full_rows, so nothing changes there. See _exclude_known_infeasible's docstring.
        shortlist, excluded_infeasible = _exclude_known_infeasible(
            shortlist,
            lambda e: dict(ranks=e["n_ranks"], omp=e["omp_threads"], nx=e["nproc_x"],
                            ny=e["nproc_y"], numtiles=e["numtiles"], dist=e["distribution"],
                            blocktime=e.get("kmp_blocktime")),
            lambda e: e["config_id"], args.case, args.nodes or DEFAULT_NODES,
            _failure_index(train_rows))
        recommend_info["excluded"] = excluded_omp + excluded_infeasible
        for entry in shortlist:
            cid = entry["config_id"]
            cfg = dict(ranks=entry["n_ranks"], omp=entry["omp_threads"], nx=entry["nproc_x"],
                       ny=entry["nproc_y"], numtiles=entry["numtiles"], dist=entry["distribution"],
                       blocktime=_blocktime_str(entry.get("kmp_blocktime")))
            cands.add(cid, cfg, "recommended:%d" % entry["rank"],
                      predicted_gain=entry.get("predicted_gain"),
                      gain_lower_95=entry.get("gain_lower_95"),
                      label=entry.get("label"), flaky=bool(entry.get("flaky")))

    if BASELINE_ID not in id_to_cfg:
        raise SystemExit(
            "case %r resolves no %s baseline template -- cannot plan an experiment without a "
            "baseline (skipped templates: see campaign/configs.py resolve_case)"
            % (args.case, BASELINE_ID))
    cands.add(BASELINE_ID, id_to_cfg[BASELINE_ID], "baseline")

    bk = _best_known(_rows_on_nodes(full_rows, args.nodes or DEFAULT_NODES), args.case)
    if bk is not None:
        bk_id, bk_cfg, bk_median, bk_n = bk
        cands.add(bk_id, bk_cfg, "best_known",
                   best_known_measured_median=bk_median, best_known_measured_n=bk_n)

    return cands, recommend_info, bk


# ---------------------------------------------------------------------------
# generated-space candidate assembly (predictor/decomp_space.py -- a case that resolves no C01
# STRICT template, see the module docstring's "Generated space" paragraph)
# ---------------------------------------------------------------------------
def _import_decomp_space():
    try:
        import decomp_space
    except ImportError as e:
        raise SystemExit(
            "predictor/decomp_space.py could not be imported (%s: %s) -- it should need only "
            "the stdlib plus campaign/configs.py and predictor/features.py; check those modules "
            "if this is unexpected" % (type(e).__name__, e))
    return decomp_space


def _max_legal_ranks(case_json, max_ranks):
    """Largest ranks in 1..max_ranks with >=1 legal WRF shape for `case_json`'s active domains
    (predictor/decomp_space.legal_shapes_exact -- the exact floor-based share/module_check_a_
    mundo.F gate), scanning downward from max_ranks: (ranks, shapes), or (None, []) if no rank
    count in that range has any legal shape at all. This is GENERATED_BASELINE_RULE's own
    downward scan (_generated_baseline below), factored out so PART D's
    _call_decomp_search_propose fix (a case whose max_omp caps every candidate to a SINGLE omp
    value whose exact-fit ranks=total_cores//omp has no legal shape -- see that function's
    docstring) can reuse the identical rule instead of a second implementation."""
    decomp_space = _import_decomp_space()
    we_list, sn_list = campaign_configs._active_domain_extents(case_json)
    for ranks in range(max_ranks, 0, -1):
        shapes = decomp_space.legal_shapes_exact(we_list, sn_list, ranks)
        if shapes:
            return ranks, shapes
    return None, []


def _generated_baseline(case_json, total_cores, nodes, hardware_id):
    """The generated-space C01 analogue -- see GENERATED_BASELINE_RULE. Always resolvable from
    predictor/decomp_space.enumerate_space's exact legal-shape gate: scoped to
    total_cores=<the chosen max-legal rank count>, not the caller's total_cores, because the
    caller's own total_cores can itself be illegal at omp=1 -- enumerate_space(total_cores=
    nodes*112, omp_options=[1]) would enumerate ZERO omp=1 candidates whenever ranks=total_cores
    itself has no legal (nx,ny) split (e.g. matthew 91x100 at total_cores=112: legal_shapes_exact
    is empty there; the true max legal rank count is 90, shape 9x10)."""
    decomp_space = _import_decomp_space()
    we_list, sn_list = campaign_configs._active_domain_extents(case_json)
    ranks, shapes = _max_legal_ranks(case_json, total_cores)
    if ranks is None:
        raise SystemExit(
            "no legal MPI decomposition exists at omp=1 for any rank count 1..%d -- cannot "
            "build a generated-space baseline" % total_cores)

    we0, sn0 = we_list[0], sn_list[0]

    def _aspect(shape):
        nx, ny = shape
        px, _ = decomp_space.rsl_patch_extents(we0, nx)
        py, _ = decomp_space.rsl_patch_extents(sn0, ny)
        return max(px, py) / float(min(px, py))

    shape = min(shapes, key=lambda s: (_aspect(s), s))

    base_candidates, _axis = decomp_space.enumerate_space(
        case_json, total_cores=ranks, nodes=nodes, hardware_id=hardware_id, omp_options=[1],
        tile_options=dict(numtiles_multiples=(), max_numtiles=1, include_2d=False))
    base = next(c for c in base_candidates
                if (c["nx"], c["ny"]) == shape and c["dist"] == "cyclic" and c["blocktime"] == "0")
    return dict(
        ranks=base["ranks"], omp=1, nx=base["nx"], ny=base["ny"], numtiles=1,
        dist="cyclic", blocktime="0",
        numtiles_x=base.get("numtiles_x", 0), numtiles_y=base.get("numtiles_y", 0),
        tile_sz_x=base.get("tile_sz_x", 0), tile_sz_y=base.get("tile_sz_y", 0),
        tile_strategy=base.get("tile_strategy", 0),
        patch_x=base.get("patch_x"), patch_y=base.get("patch_y"),
    )


def _call_decomp_search_propose(ledger_paths, case, case_json, hardware_id, nodes, total_cores,
                                 top_k, loco):
    """predictor/decomp_search.py's dependency chain needs scikit-learn/catboost (via
    predictor/models.py), exactly like predictor/recommend.py's -- see _call_recommend above.
    Calls decomp_search.propose_for_case (decomp_search.py:505), that module's own documented
    low-level reuse entry point ("lets a caller with its OWN namelist-derived case ... reuse this
    loop"), which itself calls propose_core (decomp_search.py:426) as a PURE-EXPLOITATION
    recommender: kappa=0.0 makes the LCB acquisition mu - kappa*sigma degenerate to mu (rank
    candidates by predicted mean only, no exploration bonus for uncertain ones -- see the
    acq = mu - kappa*sigma line, decomp_search.py:463). `state` is a FRESH, never-measured
    decomp_search state (the same shape decomp_search.load_state(None) returns for a path that
    does not exist) -- nothing is read from or written to disk; decomp_search's own round-
    tracking state is never persisted here, only experiment.py's plan.json records what was
    chosen."""
    try:
        import benchmark as benchmark_mod
        import decomp_search
    except ImportError as e:
        raise SystemExit(
            "predictor/decomp_search.py could not be imported (%s: %s) -- its dependency chain "
            "needs scikit-learn/catboost at minimum (predictor/models.py); install/activate the "
            "predictor pixi env to use the generated-space recommender, or pass --space template "
            "for a case that resolves %s instead" % (type(e).__name__, e, BASELINE_ID))

    prior_rows = []
    for p in ledger_paths:
        prior_rows.extend(benchmark_mod.load_ledger(p))
    prior_rows = [r for r in prior_rows if r.get("compliance_class") != "RESEARCH"]
    n_total = len(prior_rows)
    # --loco: drop every row whose case_base == this case BEFORE the surrogate ever sees them --
    # same "one filter, one place" convention _build_candidates uses ahead of _call_recommend.
    if loco:
        prior_rows = [r for r in prior_rows if r.get("case_base") != case]
    n_used = len(prior_rows)

    case_fields = decomp_search.case_row_fields(case_json)
    state = dict(configs={}, measured={}, rounds=[])
    # case_json["max_omp"] (optional, e.g. c34/marshall's WRF-Fire omp>1 crash, see
    # campaign/cases.json): propose_for_case already implements this cap (decomp_search.py:521-524,
    # "failed runs are never ingested, so D3 cannot learn that a thread count crashes") -- it is
    # only ever exercised when a caller actually passes it, which experiment.py did not do before.
    max_omp = case_json.get("max_omp")
    # PART D fix (diagnosed against c34, campaign/cases.json e_we=e_sn=118, max_omp=1):
    # decomp_search.propose_for_case (decomp_search.py:520) calls decomp_space.enumerate_space
    # at the EXACT ranks=total_cores//omp for every omp in default_omp_options, then filters
    # omp<=max_omp AFTER the fact (decomp_search.py:521-524) -- it never scans downward the way
    # _generated_baseline does. When max_omp caps the survivors to ONE omp value whose own
    # exact-fit ranks has NO legal shape at all (c34 at omp=1, total_cores=112: every factor pair
    # of 112 needs one side >11, but floor(118/x)>=10 requires x<=11 -- confirmed empty via
    # decomp_space.legal_shapes_exact([118,118],[118,118],112) == []), every candidate at that
    # omp is filtered out and NOTHING survives -- job 1090690's "after fix propose returns 0
    # recs". decomp_search.py is NOT owned by this change (see PART D's file-ownership note), so
    # the fix cannot live in propose_for_case's own filtering order; instead, shrink the
    # total_cores THIS FUNCTION passes down to the largest under-subscribed rank count that IS
    # legal at max_omp -- the SAME downward-scan rule _generated_baseline uses for GBASE
    # (_max_legal_ranks above) -- so propose_for_case's own unmodified exact-fit division
    # (ranks = effective_total_cores // max_omp) lands on a legal shape instead. c34: ranks=110
    # (10x11), matching cases.json's own "largest legal pure-MPI counts are 110 (10x11)" note.
    # Deliberately scoped to `max_omp is not None`: a case with no max_omp (e.g. matthew) is
    # completely unaffected -- enumerate_space keeps enumerating ZERO omp=1 shapes for it exactly
    # as before (predictor/tests/test_decomp_space.py's own shapes_count[1]==0 assertion for a
    # tiny case is untouched), because matthew still has OTHER legal omp values (2, 7, 14) to
    # supply real recommended candidates from, so no downscale is needed or applied there.
    effective_total_cores = total_cores
    if max_omp is not None:
        ranks, _shapes = _max_legal_ranks(case_json, total_cores // max_omp)
        if ranks is not None:
            effective_total_cores = ranks * max_omp
    result, _configs_out = decomp_search.propose_for_case(
        case, case_json, case_fields, prior_rows, state, top_k,
        total_cores=effective_total_cores, nodes=nodes, hardware_id=hardware_id, kappa=0.0,
        max_omp=max_omp)
    result = dict(result)
    result["n_ledger_rows_total"] = n_total
    result["n_ledger_rows_used"] = n_used
    return result


def _build_generated_candidates(args, case_json, full_rows, nodes, total_cores):
    """Generated-space equivalent of _build_candidates: GBASE (the C01 analogue, always added)
    plus up to --top-k GREC1..GRECk from decomp_search.propose_core used as a pure-exploitation
    recommender, plus best_known (unchanged logic, same as the template path)."""
    cands = _Candidates()

    cands.add(GENERATED_BASELINE_ID,
              _generated_baseline(case_json, total_cores, nodes, args.hardware_id), "baseline")

    result = _call_decomp_search_propose(args.ledger, args.case, case_json, args.hardware_id,
                                          nodes, total_cores, args.top_k, bool(args.loco))
    recommend_info = dict(
        space="generated", loco=bool(args.loco), kappa=0.0, top_k=args.top_k,
        n_ledger_rows_total=result.get("n_ledger_rows_total"),
        n_ledger_rows_used=result.get("n_ledger_rows_used"),
        stopped=result.get("stopped"), reason=result.get("reason"), round=result.get("round"),
        n_training_rows=result.get("n_training_rows"), ensemble_size=result.get("ensemble_size"),
    )
    chosen, excluded_omp = _filter_max_omp(
        result.get("chosen") or [], case_json.get("max_omp"),
        lambda c: c["omp"], lambda c: "%dx%d(omp%d)" % (c["nx"], c["ny"], c["omp"]))
    # PART B (generated-space equivalent of _build_candidates' own exclusion above): train_rows
    # is `full_rows` (this function's own parameter, loaded once by cmd_plan) LOCO-filtered the
    # same way -- deliberately a SEPARATE read from _call_decomp_search_propose's own internal
    # benchmark_mod.load_ledger(args.ledger) + RESEARCH-row-drop + loco-filter (that one feeds
    # decomp_search's trained surrogate; this one only ever counts ok/fail rc= evidence, and
    # reusing the already-loaded full_rows avoids a second ledger read).
    train_rows = full_rows
    if args.loco:
        train_rows = [r for r in full_rows if r.get("case_base") != args.case]
    chosen, excluded_infeasible = _exclude_known_infeasible(
        chosen, lambda c: c, lambda c: "%dx%d(omp%d)" % (c["nx"], c["ny"], c["omp"]),
        args.case, nodes, _failure_index(train_rows))
    recommend_info["excluded"] = excluded_omp + excluded_infeasible
    for i, c in enumerate(chosen, start=1):
        cfg = dict(ranks=c["ranks"], omp=c["omp"], nx=c["nx"], ny=c["ny"], numtiles=c["numtiles"],
                   dist=c["dist"], blocktime=c["blocktime"],
                   numtiles_x=c.get("numtiles_x", 0), numtiles_y=c.get("numtiles_y", 0),
                   tile_sz_x=c.get("tile_sz_x", 0), tile_sz_y=c.get("tile_sz_y", 0),
                   tile_strategy=c.get("tile_strategy", 0),
                   patch_x=c.get("patch_x"), patch_y=c.get("patch_y"))
        cands.add("%s%d" % (GENERATED_REC_PREFIX, i), cfg, "recommended:%d" % i,
                  flaky=bool(c.get("flaky")))

    bk = _best_known(_rows_on_nodes(full_rows, nodes), args.case)
    if bk is not None:
        bk_id, bk_cfg, bk_median, bk_n = bk
        cands.add(bk_id, bk_cfg, "best_known",
                   best_known_measured_median=bk_median, best_known_measured_n=bk_n)

    return cands, recommend_info, bk


# ---------------------------------------------------------------------------
# CONFIG_FILE writer (campaign/configs.py load_generated_configs()/cmd_case_generated() shape)
# ---------------------------------------------------------------------------
def _write_configs_json(path, candidate_ids, candidates, case_json):
    out = []
    for cid in candidate_ids:
        c = candidates[cid]
        patch_x, patch_y = c.get("patch_x"), c.get("patch_y")
        if patch_x is None and c["nx"]:
            patch_x = -(-case_json["e_we"] // c["nx"])   # ceil division, campaign_configs.patch_dims
        if patch_y is None and c["ny"]:
            patch_y = -(-case_json["e_sn"] // c["ny"])
        out.append(dict(
            id=cid, ranks=c["ranks"], omp=c["omp"], nx=c["nx"], ny=c["ny"],
            numtiles=c["numtiles"], dist=c["dist"], blocktime=c["blocktime"],
            patch_x=patch_x, patch_y=patch_y,
            numtiles_x=c.get("numtiles_x", 0), numtiles_y=c.get("numtiles_y", 0),
            tile_sz_x=c.get("tile_sz_x", 0), tile_sz_y=c.get("tile_sz_y", 0),
            tile_strategy=c.get("tile_strategy", 0),
            config_source="experiment:%s" % "+".join(c["roles"]),
        ))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    return out


# ---------------------------------------------------------------------------
# cost estimate
# ---------------------------------------------------------------------------
def _cost_estimate(case_json, candidate_ids, s_est, startup_s, reps, nodes):
    steps = case_json["steps"]
    total_seconds = sum(reps * (steps * s_est[cid] + startup_s) for cid in candidate_ids)
    job_hours = total_seconds / 3600.0 + FIXED_JOB_OVERHEAD_MIN / 60.0
    node_hours = nodes * job_hours
    su = node_hours * CORES_PER_NODE * SU_PER_CORE_HOUR
    cost_line = ("1 jobs x %d nodes x %.2f h = %.2f node-hours = %.2f SU, wall-clock ~%.2f h"
                 % (nodes, job_hours, node_hours, su, job_hours))
    time_str = _format_hms(TIME_SAFETY_FACTOR * job_hours * 60.0 + TIME_PAD_MIN)
    return dict(steps=steps, n_configs=len(candidate_ids), reps=reps,
                s_per_step_est=s_est, startup_s=startup_s,
                job_hours=job_hours, node_hours=node_hours, su=su,
                cost_line=cost_line, time_limit=time_str)


# ---------------------------------------------------------------------------
# PART C helpers: configs.json + cost estimate + sbatch command for ONE job. cmd_plan's own
# non-screen (--screen-spares 0, the default/old-behaviour) path is left untouched below, byte-
# for-byte, to keep zero regression risk on its many existing tests; these two helpers are used
# only by the NEW screen-job path (cmd_plan when --screen-spares>0) and by cmd_screen's
# screen->timing rewrite, which both need the identical configs.json/cost/sbatch construction
# with different (reps, steps, extra --export=) parameters.
# ---------------------------------------------------------------------------
def _job_configs_and_cost(filename, out_dir, case, case_json, id_to_cfg, candidates,
                           candidate_ids, full_rows, nodes, reps, steps=None):
    s_est, s_source = _s_per_step_estimates(_rows_on_nodes(full_rows, nodes), case,
                                             candidate_ids, case_json)
    startup_s, startup_assumed = _startup_s(full_rows)
    job_case_json = case_json if steps is None else dict(case_json, steps=steps)
    cost = _cost_estimate(job_case_json, candidate_ids, s_est, startup_s, reps, nodes)
    cost["s_per_step_source"] = s_source
    cost["startup_assumed"] = startup_assumed

    configs_json_path = os.path.join(out_dir, filename)
    _write_configs_json(configs_json_path, candidate_ids, candidates, case_json)

    template_ids = [c for c in candidate_ids if c in id_to_cfg]
    nontemplate_ids = [c for c in candidate_ids if c not in id_to_cfg]
    return cost, os.path.abspath(configs_json_path), template_ids, nontemplate_ids


def _sbatch_command(case, mode, reps, nodes, partition, time_limit, nontemplate_ids,
                     candidate_ids, configs_json_path, extra_export=()):
    export_parts = ["ALL", "CASE=%s" % case, "MODE=%s" % mode, "REPS=%d" % reps]
    for k, v in extra_export:
        export_parts.append("%s=%s" % (k, v))
    if nontemplate_ids:
        export_parts.append("CONFIG_FILE=%s" % configs_json_path)
    else:
        export_parts.append("CONFIGS=%s" % "+".join(candidate_ids))
    return ("sbatch -p %s -N %d --time %s --export=%s %s"
            % (partition, nodes, time_limit, ",".join(export_parts), SBATCH_SCRIPT_REL))


def _screen_timeout_s(startup_s, s_est):
    s_est_max = max(s_est.values()) if s_est else 0.0
    return max(SCREEN_TIMEOUT_MIN_S,
               int(math.ceil(SCREEN_TIMEOUT_MULTIPLIER * (startup_s + SCREEN_STEPS * s_est_max))))


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def cmd_plan(args):
    os.makedirs(args.out, exist_ok=True)
    case_json = _case_json(args.case)
    id_to_cfg, skipped, aliases = _resolve_case_configs(args.case)
    full_rows = _read_ledgers(args.ledger)
    nodes = args.nodes or DEFAULT_NODES
    total_cores = nodes * CORES_PER_NODE

    # PART C: --screen-spares S (default 0 -- see cmd_plan's else-branch below, left byte-for-
    # byte unchanged from before PART C) asks the recommender/proposer for top_k+S candidates
    # instead of top_k, via a shallow args copy so _build_candidates/_build_generated_candidates
    # (which both read args.top_k internally) need no change of their own.
    screen_spares = getattr(args, "screen_spares", 0) or 0
    build_args = args
    if screen_spares > 0:
        build_args = argparse.Namespace(**vars(args))
        build_args.top_k = args.top_k + screen_spares

    space = args.space
    if space == "auto":
        space = "template" if BASELINE_ID in id_to_cfg else "generated"

    if space == "template":
        cands, recommend_info, bk = _build_candidates(build_args, case_json, id_to_cfg, full_rows)
        baseline_rule = None
    else:
        if args.configs:
            raise SystemExit(
                "--configs (a template-id list resolved against campaign/configs.py TEMPLATES) "
                "is not supported with --space generated -- the generated space has no fixed "
                "template-id table to resolve ids against; drop --configs, or use --space "
                "template/auto with a case that resolves %s" % BASELINE_ID)
        cands, recommend_info, bk = _build_generated_candidates(build_args, case_json, full_rows,
                                                                  nodes, total_cores)
        baseline_rule = GENERATED_BASELINE_RULE

    candidate_ids = cands.ids()
    candidates = cands.as_dict()

    partition = PARTITION_SMALL if nodes == 1 else PARTITION_FULL

    screen_block = None
    if screen_spares > 0:
        # SCREEN job: REPS=1/STEPS=SCREEN_STEPS/TIMEOUT_S=<computed>, written to DIR/screen/
        # configs.json (kept as a permanent record) -- the TOP-LEVEL configs.json/sbatch/cost
        # below describe THIS job (the only one that can be submitted right now); `screen`
        # (cmd_screen) later rewrites those same top-level fields for the real timing job once
        # this job's ledger says which candidates are actually feasible.
        screen_dir = os.path.join(args.out, "screen")
        os.makedirs(screen_dir, exist_ok=True)
        cost, configs_json_path, template_ids, nontemplate_ids = _job_configs_and_cost(
            "configs.json", screen_dir, args.case, case_json, id_to_cfg, candidates,
            candidate_ids, full_rows, nodes, reps=1, steps=SCREEN_STEPS)
        timeout_s = _screen_timeout_s(cost["startup_s"], cost["s_per_step_est"])
        sbatch_command = _sbatch_command(
            args.case, "timing", 1, nodes, partition, cost["time_limit"], nontemplate_ids,
            candidate_ids, configs_json_path,
            extra_export=[("STEPS", SCREEN_STEPS), ("TIMEOUT_S", timeout_s)])
        print(cost["cost_line"])
        print(sbatch_command)
        screen_block = dict(
            spares=screen_spares, top_k=args.top_k, steps=SCREEN_STEPS, timeout_s=timeout_s,
            candidate_ids=candidate_ids, configs_json_path=configs_json_path,
            cost=cost, sbatch=dict(
                command=sbatch_command,
                config_file=(configs_json_path if nontemplate_ids else None),
                configs_env=("+".join(candidate_ids) if not nontemplate_ids else None)),
            job_id=None, state="draft",
        )
    else:
        s_est, s_source = _s_per_step_estimates(_rows_on_nodes(full_rows, nodes), args.case,
                                                candidate_ids, case_json)
        startup_s, startup_assumed = _startup_s(full_rows)
        cost = _cost_estimate(case_json, candidate_ids, s_est, startup_s, args.reps, nodes)
        cost["s_per_step_source"] = s_source
        cost["startup_assumed"] = startup_assumed

        configs_json_path = os.path.join(args.out, "configs.json")
        _write_configs_json(configs_json_path, candidate_ids, candidates, case_json)

        template_ids = [c for c in candidate_ids if c in id_to_cfg]
        nontemplate_ids = [c for c in candidate_ids if c not in id_to_cfg]
        export_parts = ["ALL", "CASE=%s" % args.case, "MODE=timing", "REPS=%d" % args.reps]
        if nontemplate_ids:
            export_parts.append("CONFIG_FILE=%s" % os.path.abspath(configs_json_path))
        else:
            export_parts.append("CONFIGS=%s" % "+".join(candidate_ids))
        sbatch_command = ("sbatch -p %s -N %d --time %s --export=%s %s"
                           % (partition, nodes, cost["time_limit"], ",".join(export_parts),
                              SBATCH_SCRIPT_REL))
        print(cost["cost_line"])
        print(sbatch_command)
        configs_json_path = os.path.abspath(configs_json_path)

    plan = dict(
        case=args.case, hardware_id=args.hardware_id, nodes=nodes, partition=partition,
        reps=args.reps, top_k=args.top_k, loco=bool(args.loco),
        space=space, baseline_rule=baseline_rule,
        namelist=os.path.abspath(args.namelist) if args.namelist else None,
        ledgers=[os.path.abspath(p) for p in args.ledger],
        explicit_configs=os.path.abspath(args.configs) if args.configs else None,
        candidates=[candidates[cid] for cid in candidate_ids],
        template_ids=template_ids, nontemplate_ids=nontemplate_ids,
        skipped_templates=[dict(id=tid, reason=reason) for tid, reason in skipped],
        best_known=(dict(config_id=bk[0], measured_median_s_per_step=bk[2], n=bk[3])
                    if bk is not None else None),
        criterion=dict(text=CRITERION_TEXT, threshold_multiplier=PASS_THRESHOLD_MULTIPLIER),
        cost=cost,
        configs_json_path=configs_json_path,
        sbatch=dict(command=sbatch_command,
                    config_file=(configs_json_path if nontemplate_ids else None),
                    configs_env=("+".join(candidate_ids) if not nontemplate_ids else None)),
        recommend=recommend_info,
        screen_spares=screen_spares,
        screen=screen_block,
        created=_utcnow_iso(), git_head=_git_head(),
        state="draft",
    )
    _write_json(os.path.join(args.out, "plan.json"), plan)
    return plan


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
def cmd_record(args):
    plan_path = os.path.join(args.plan, "plan.json")
    plan = _load_json(plan_path)
    stage = getattr(args, "stage", None) or "timing"
    if stage == "screen":
        if not plan.get("screen"):
            raise SystemExit(
                "plan.json has no screen block -- run `plan --screen-spares S` (S>0) first")
        plan["screen"]["job_id"] = args.job_id
        plan["screen"]["submitted_at"] = _utcnow_iso()
        plan["screen"]["state"] = "submitted"
        plan["state"] = "screen_submitted"
        _write_json(plan_path, plan)
        print("recorded screen job_id=%s for plan=%s" % (args.job_id, args.plan))
    else:
        plan["job_id"] = args.job_id
        plan["submitted_at"] = _utcnow_iso()
        plan["state"] = "submitted"
        _write_json(plan_path, plan)
        print("recorded job_id=%s for plan=%s" % (args.job_id, args.plan))
    return plan


# ---------------------------------------------------------------------------
# PART C: screen -- reads the screen job's own ledger (recorded via `record --stage screen`),
# marks each screen candidate feasible (a numeric median came back) or infeasible (it did not),
# keeps the first top_k feasible RECOMMENDED candidates in rank order plus baseline/best_known
# (always kept regardless of their own screen outcome), and rewrites the plan's TOP-LEVEL
# configs.json/cost/sbatch for the real timing job -- exactly the shape cmd_plan's own non-screen
# path already produces, so `record`/`reconcile` need no --stage afterward (backward compatible).
# ---------------------------------------------------------------------------
def _recommended_rank(entry):
    for role in entry.get("roles", ()):
        if role.startswith("recommended:"):
            return int(role.split(":", 1)[1])
    return None


def screen(plan_dir, ledger_dir=None):
    """Core, directly-testable screen logic (the `screen` CLI subcommand is a thin argv wrapper,
    same convention as reconcile())."""
    plan_path = os.path.join(plan_dir, "plan.json")
    plan = _load_json(plan_path)
    sblock = plan.get("screen")
    if not sblock:
        raise SystemExit("plan.json has no screen block -- run `plan --screen-spares S` first")
    job_id = sblock.get("job_id")
    if not job_id:
        raise SystemExit("screen block has no job_id -- run `record --plan %s --job-id N "
                          "--stage screen` first" % plan_dir)

    case = plan["case"]
    ledger_dir = ledger_dir or DEFAULT_LEDGER_DIR
    screen_ledger_path = os.path.join(ledger_dir, "ledger_%s_%s.csv" % (case, job_id))
    screen_rows = (_read_ledger_csv(screen_ledger_path)
                   if os.path.exists(screen_ledger_path) else [])

    ranked = dsum.summarize(screen_rows, case)
    by_config_id_measured = {}
    for g in ranked:
        for cid in g["ids"]:
            by_config_id_measured[cid] = g
    notes_rows_by_cid = {}
    for r in screen_rows:
        cid = dsum.config_id(r)
        if cid:
            notes_rows_by_cid.setdefault(cid, []).append(r)

    by_config_id = {c["config_id"]: c for c in plan["candidates"]}
    results = {}
    for cid in sblock["candidate_ids"]:
        g = by_config_id_measured.get(cid)
        feasible = g is not None
        rcs = [] if feasible else [rc for rc in (_row_rc(r) for r in notes_rows_by_cid.get(cid, ()))
                                    if rc]
        results[cid] = dict(feasible=feasible, s_median=(g["s_median"] if g else None), rcs=rcs)

    recommended_ids = sorted(
        (cid for cid in sblock["candidate_ids"] if _recommended_rank(by_config_id.get(cid, {})) is not None),
        key=lambda cid: _recommended_rank(by_config_id[cid]))
    kept_recommended = [cid for cid in recommended_ids if results[cid]["feasible"]][: sblock["top_k"]]
    dropped_ids = [cid for cid in recommended_ids if cid not in kept_recommended]
    underfilled = len(kept_recommended) < sblock["top_k"]

    # "+ baseline + best_known, always" -- keep an id carrying either role even when it is ALSO
    # a recommended candidate that the feasibility filter above would otherwise have dropped
    # (test_dedupe_collapses_multiple_roles_onto_one_candidate's same "one id, several roles"
    # shape, e.g. the historical best_known happening to also be this round's rank-1 pick).
    always_ids = [cid for cid in sblock["candidate_ids"]
                  if set(by_config_id.get(cid, {}).get("roles", ())) & {"baseline", "best_known"}]
    final_ids = sorted(set(always_ids) | set(kept_recommended))

    case_json = _case_json(case)
    id_to_cfg, _skipped, _aliases = _resolve_case_configs(case)
    full_rows = _read_ledgers(plan["ledgers"])
    nodes = plan["nodes"]
    cost, configs_json_path, template_ids, nontemplate_ids = _job_configs_and_cost(
        "configs.json", plan_dir, case, case_json, id_to_cfg, by_config_id, final_ids,
        full_rows, nodes, reps=plan["reps"])
    sbatch_command = _sbatch_command(case, "timing", plan["reps"], nodes, plan["partition"],
                                      cost["time_limit"], nontemplate_ids, final_ids,
                                      configs_json_path)

    plan["cost"] = cost
    plan["configs_json_path"] = configs_json_path
    plan["template_ids"] = template_ids
    plan["nontemplate_ids"] = nontemplate_ids
    plan["candidates"] = [by_config_id[cid] for cid in final_ids]
    plan["sbatch"] = dict(command=sbatch_command,
                           config_file=(configs_json_path if nontemplate_ids else None),
                           configs_env=("+".join(final_ids) if not nontemplate_ids else None))
    plan["screen"]["results"] = results
    plan["screen"]["kept_recommended_ids"] = kept_recommended
    plan["screen"]["dropped_ids"] = dropped_ids
    plan["screen"]["underfilled"] = underfilled
    plan["screen"]["screened_at"] = _utcnow_iso()
    plan["state"] = "draft"
    _write_json(plan_path, plan)
    return plan


def cmd_screen(args):
    plan = screen(args.plan, ledger_dir=args.ledger_dir)
    print(plan["cost"]["cost_line"])
    print(plan["sbatch"]["command"])
    if plan["screen"]["underfilled"]:
        print("WARNING: only %d/%d recommended candidates were feasible in the screen job"
              % (len(plan["screen"]["kept_recommended_ids"]), plan["top_k"]))
    return plan


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------
def _parse_elapsed(s):
    if not s:
        return 0.0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return float(days * 86400 + h * 3600 + m * 60 + sec)


def _parse_sacct(stdout):
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    fields = lines[0].split("|")
    keys = ["JobID", "JobName", "State", "Elapsed", "NNodes", "NCPUS"]
    row = dict(zip(keys, fields))
    row["elapsed_s"] = _parse_elapsed(row.get("Elapsed", ""))
    for k in ("NNodes", "NCPUS"):
        try:
            row[k] = int(row[k])
        except (KeyError, ValueError):
            row[k] = None
    return row


def _run_sacct(job_id, sacct_cmd):
    cmd = list(sacct_cmd) + ["-j", str(job_id), "-X", "-n", "-P",
                              "--format=JobID,JobName,State,Elapsed,NNodes,NCPUS"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return _parse_sacct(proc.stdout)


def _write_reconcile_csv(path, per_config, plan):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["case", "job_id", "config_id", "roles", "ranks", "omp", "nx", "ny", "numtiles",
              "dist", "blocktime", "n", "s_median", "s_min", "s_max", "success",
              "predicted_gain", "gain_lower_95", "label", "verdict", "regret",
              "speedup_vs_c01"]
    by_id = {c["config_id"]: c for c in plan["candidates"]}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        rec = plan.get("reconcile", {})
        for e in per_config:
            c = by_id.get(e["config_id"], {})
            w.writerow(dict(
                case=plan["case"], job_id=plan.get("job_id"), config_id=e["config_id"],
                roles=";".join(e["roles"]), ranks=c.get("ranks"), omp=c.get("omp"),
                nx=c.get("nx"), ny=c.get("ny"), numtiles=c.get("numtiles"), dist=c.get("dist"),
                blocktime=c.get("blocktime"), n=e["n"], s_median=e["s_median"],
                s_min=e["s_min"], s_max=e["s_max"], success=e["success"],
                predicted_gain=c.get("predicted_gain"), gain_lower_95=c.get("gain_lower_95"),
                label=c.get("label"), verdict=rec.get("verdict"), regret=rec.get("regret"),
                speedup_vs_c01=rec.get("speedup_vs_c01"),
            ))


def reconcile(plan_dir, ledger_dir=None, sacct_cmd=None, results_dir=None):
    """Core, directly-testable reconcile logic (the `reconcile` CLI subcommand is a thin
    argv wrapper around this)."""
    plan_path = os.path.join(plan_dir, "plan.json")
    plan = _load_json(plan_path)
    job_id = plan.get("job_id")
    if not job_id:
        raise SystemExit("plan.json has no job_id -- run `record --plan %s --job-id N` first"
                          % plan_dir)
    case = plan["case"]
    reps = plan["reps"]

    sacct_row = _run_sacct(job_id, sacct_cmd or ["sacct"])
    ledger_dir = ledger_dir or DEFAULT_LEDGER_DIR
    ledger_path = os.path.join(ledger_dir, "ledger_%s_%s.csv" % (case, job_id))
    job_rows = _read_ledger_csv(ledger_path) if os.path.exists(ledger_path) else []

    ranked = dsum.summarize(job_rows, case)
    by_id = {}
    for g in ranked:
        for cid in g["ids"]:
            by_id[cid] = g

    per_config = []
    controls_ok = True
    underpowered = []
    for c in plan["candidates"]:
        cid = c["config_id"]
        g = by_id.get(cid)
        n = g["n"] if g else 0
        entry = dict(config_id=cid, roles=c["roles"], n=n,
                     s_median=(g["s_median"] if g else None),
                     s_min=(g["s_min"] if g else None), s_max=(g["s_max"] if g else None),
                     success=(n >= reps))
        per_config.append(entry)
        if n < reps:
            underpowered.append(cid)
        if ("baseline" in c["roles"] or "best_known" in c["roles"]) and g is None:
            controls_ok = False

    state = (sacct_row or {}).get("State", "")
    completed = (state == "COMPLETED")

    verdict = None
    regret = None
    speedup_vs_c01 = None
    best_rec = None
    best_known_entry = next((e for e in per_config if "best_known" in e["roles"]), None)
    baseline_entry = next((e for e in per_config if "baseline" in e["roles"]), None)

    if not completed or not controls_ok or underpowered:
        verdict = "INCONCLUSIVE"
    else:
        rec_entries = [e for e in per_config
                        if any(r.startswith("recommended:") or r.startswith("explicit:")
                               for r in e["roles"])]
        if not rec_entries or best_known_entry is None or best_known_entry["s_median"] is None:
            verdict = "INCONCLUSIVE"
        else:
            best_rec = min(rec_entries, key=lambda e: e["s_median"])
            regret = best_rec["s_median"] / best_known_entry["s_median"] - 1.0
            threshold = plan["criterion"]["threshold_multiplier"]
            verdict = "PASS" if best_rec["s_median"] <= threshold * best_known_entry["s_median"] \
                else "FAIL"
            if baseline_entry and baseline_entry["s_median"]:
                speedup_vs_c01 = 1.0 - best_rec["s_median"] / baseline_entry["s_median"]

    overlap = None
    if best_rec and best_known_entry and best_rec["config_id"] != best_known_entry["config_id"]:
        overlap = not (best_rec["s_max"] < best_known_entry["s_min"]
                        or best_known_entry["s_max"] < best_rec["s_min"])

    elapsed_h = (sacct_row["elapsed_s"] / 3600.0) if sacct_row else None
    nnodes_actual = sacct_row.get("NNodes") if sacct_row else None
    ncpus_actual = sacct_row.get("NCPUS") if sacct_row else None
    su_actual = (ncpus_actual * elapsed_h * SU_PER_CORE_HOUR
                 if (elapsed_h is not None and ncpus_actual) else None)
    node_hours_actual = (nnodes_actual * elapsed_h
                          if (elapsed_h is not None and nnodes_actual) else None)

    est = plan["cost"]
    reconcile_block = dict(
        job_id=job_id, sacct=sacct_row, ledger_path=ledger_path, verdict=verdict,
        regret=regret, speedup_vs_c01=speedup_vs_c01,
        best_recommended=best_rec, best_known=best_known_entry, baseline=baseline_entry,
        overlap_with_best_known=overlap, underpowered=underpowered,
        controls_present=controls_ok, per_config=per_config,
        est_su=est["su"], actual_su=su_actual,
        est_node_hours=est["node_hours"], actual_node_hours=node_hours_actual,
        reconciled_at=_utcnow_iso(),
    )
    plan["reconcile"] = reconcile_block
    plan["state"] = {"PASS": "verified", "FAIL": "failed"}.get(verdict, "inconclusive")

    results_dir = results_dir or DEFAULT_RESULTS_DIR
    csv_path = os.path.join(
        results_dir, "%s-experiment-%s-%s.csv" % (datetime.date.today().isoformat(), case, job_id))
    _write_reconcile_csv(csv_path, per_config, plan)
    reconcile_block["csv_path"] = csv_path
    _write_json(plan_path, plan)
    return reconcile_block


def cmd_reconcile(args):
    sacct_cmd = shlex.split(args.sacct_cmd) if args.sacct_cmd else None
    block = reconcile(args.plan, ledger_dir=args.ledger_dir, sacct_cmd=sacct_cmd)
    print("verdict=%s regret=%s speedup_vs_c01=%s est_su=%.2f actual_su=%s" % (
        block["verdict"], block["regret"], block["speedup_vs_c01"], block["est_su"],
        block["actual_su"]))
    return block


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def cmd_summarize(args):
    header = ["case", "job_id", "verdict", "regret", "speedup_vs_c01", "est_su", "actual_su"]
    rows = []
    total_est = total_actual = 0.0
    for d in args.plans:
        plan = _load_json(os.path.join(d, "plan.json"))
        rec = plan.get("reconcile") or {}
        est_su = (plan.get("cost") or {}).get("su") or 0.0
        act_su = rec.get("actual_su") or 0.0
        total_est += est_su
        total_actual += act_su
        rows.append(dict(case=plan.get("case"), job_id=plan.get("job_id"),
                          verdict=rec.get("verdict", "NOT_RECONCILED"),
                          regret=rec.get("regret"), speedup_vs_c01=rec.get("speedup_vs_c01"),
                          est_su=est_su, actual_su=rec.get("actual_su")))
    print("\t".join(header))
    for r in rows:
        print("\t".join("" if r.get(h) is None else str(r.get(h)) for h in header))
    print("TOTAL\test_su=%.2f\tactual_su=%.2f" % (total_est, total_actual))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="WTune decision-0050 step 5 experiment engine: candidate set -> "
                    "pre-registered cost-stated Slurm job -> sacct/ledger reconciliation.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="build a candidate set + cost estimate + sbatch command")
    p_plan.add_argument("--case", required=True)
    p_plan.add_argument("--ledger", nargs="+", required=True)
    p_plan.add_argument("--namelist", required=True)
    p_plan.add_argument("--hardware-id", required=True, dest="hardware_id")
    p_plan.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    p_plan.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    p_plan.add_argument("--loco", action="store_true")
    p_plan.add_argument("--space", choices=("auto", "template", "generated"), default="auto",
                         help="'template' forces campaign/configs.py's C01-C16 STRICT templates "
                              "(fails if the case resolves no C01 baseline); 'generated' forces "
                              "predictor/decomp_space.py's exact-legality decomposition space "
                              "(GBASE baseline + decomp_search.propose_core top-k); 'auto' "
                              "(default) picks 'template' when the case resolves C01, else "
                              "'generated'")
    p_plan.add_argument("--configs", default=None)
    p_plan.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p_plan.add_argument("--screen-spares", type=int, default=0, dest="screen_spares",
                         help="PART C: 0 (default) is the old behaviour -- one direct timing "
                              "plan. >0 asks for top_k+S candidates and writes a cheap REPS=1/"
                              "STEPS=%d feasibility-screen job (DIR/screen/configs.json) instead; "
                              "run it, `record --stage screen`, then `screen --plan DIR` to "
                              "rewrite this plan for the real timing job with only the feasible "
                              "candidates." % SCREEN_STEPS)
    p_plan.add_argument("--out", required=True)
    p_plan.set_defaults(func=cmd_plan)

    p_record = sub.add_parser("record", help="store a submitted job id into plan.json")
    p_record.add_argument("--plan", required=True)
    p_record.add_argument("--job-id", required=True, dest="job_id")
    p_record.add_argument("--stage", choices=("screen", "timing"), default=None,
                           help="PART C: 'screen' records the feasibility-screen job's id under "
                                "plan.json's screen block; omitted (default) or 'timing' records "
                                "the real timing job's id, exactly as before PART C existed.")
    p_record.set_defaults(func=cmd_record)

    p_screen = sub.add_parser(
        "screen", help="PART C: reconcile the feasibility-screen job and rewrite the plan for "
                        "the real timing job")
    p_screen.add_argument("--plan", required=True)
    p_screen.add_argument("--ledger-dir", default=None, dest="ledger_dir")
    p_screen.set_defaults(func=cmd_screen)

    p_rec = sub.add_parser("reconcile", help="reconcile sacct + ledger against the plan")
    p_rec.add_argument("--plan", required=True)
    p_rec.add_argument("--ledger-dir", default=None, dest="ledger_dir")
    p_rec.add_argument("--sacct-cmd", default=None, dest="sacct_cmd")
    p_rec.set_defaults(func=cmd_reconcile)

    p_sum = sub.add_parser("summarize", help="one row per reconciled plan, plus totals")
    p_sum.add_argument("--plans", nargs="+", required=True)
    p_sum.set_defaults(func=cmd_summarize)

    return ap


def main(argv=None):
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
