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
round-trippable record of exactly what was measured, and the only mechanism available at all
for a future non-template candidate, e.g. a predictor/decomp_search.py proposal or a
predictor/decomp_space.py generated decomposition) -- when a plan ever contains a
non-template candidate, the sbatch command switches to CONFIG_FILE=configs.json instead
(which must then carry EVERY candidate, template ids included, since CONFIG_FILE replaces
the whole config set for that job).
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
DEFAULT_LEDGER_DIR = "/home/u5727520/wrf-bench/wscore/campaign"   # strict_campaign.sbatch's
                                                                    # own $OUT; only a fallback,
                                                                    # always override with
                                                                    # --ledger-dir off-cluster
DEFAULT_RESULTS_DIR = os.path.join(_REPO_ROOT, "results", "derived-data")

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
        for entry in result.get("shortlist", []):
            cid = entry["config_id"]
            cfg = dict(ranks=entry["n_ranks"], omp=entry["omp_threads"], nx=entry["nproc_x"],
                       ny=entry["nproc_y"], numtiles=entry["numtiles"], dist=entry["distribution"],
                       blocktime=_blocktime_str(entry.get("kmp_blocktime")))
            cands.add(cid, cfg, "recommended:%d" % entry["rank"],
                      predicted_gain=entry.get("predicted_gain"),
                      gain_lower_95=entry.get("gain_lower_95"),
                      label=entry.get("label"))

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
# plan
# ---------------------------------------------------------------------------
def cmd_plan(args):
    os.makedirs(args.out, exist_ok=True)
    case_json = _case_json(args.case)
    id_to_cfg, skipped, aliases = _resolve_case_configs(args.case)
    full_rows = _read_ledgers(args.ledger)

    cands, recommend_info, bk = _build_candidates(args, case_json, id_to_cfg, full_rows)
    candidate_ids = cands.ids()
    candidates = cands.as_dict()

    nodes = args.nodes or DEFAULT_NODES
    partition = PARTITION_SMALL if nodes == 1 else PARTITION_FULL
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

    plan = dict(
        case=args.case, hardware_id=args.hardware_id, nodes=nodes, partition=partition,
        reps=args.reps, top_k=args.top_k, loco=bool(args.loco),
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
        configs_json_path=os.path.abspath(configs_json_path),
        sbatch=dict(command=sbatch_command,
                    config_file=(os.path.abspath(configs_json_path) if nontemplate_ids else None),
                    configs_env=("+".join(candidate_ids) if not nontemplate_ids else None)),
        recommend=recommend_info,
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
    plan["job_id"] = args.job_id
    plan["submitted_at"] = _utcnow_iso()
    plan["state"] = "submitted"
    _write_json(plan_path, plan)
    print("recorded job_id=%s for plan=%s" % (args.job_id, args.plan))
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
    p_plan.add_argument("--configs", default=None)
    p_plan.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p_plan.add_argument("--out", required=True)
    p_plan.set_defaults(func=cmd_plan)

    p_record = sub.add_parser("record", help="store a submitted job id into plan.json")
    p_record.add_argument("--plan", required=True)
    p_record.add_argument("--job-id", required=True, dest="job_id")
    p_record.set_defaults(func=cmd_record)

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
