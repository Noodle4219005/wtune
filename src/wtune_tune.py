#!/usr/bin/env python3
"""wtune_tune.py -- user-facing autotuner: finds the best STRICT decomposition/tiling
configuration for a WRF case on scc within a wall-clock budget, with NO agent involvement.

    python3 wtune_tune.py <case> --site scc --minutes 30 [--nodes N|auto] [--max-nodes 13] [--out DIR]

Runs as an ordinary controller process on the scc login node (may run for the whole budget,
polling Slurm) -- not a Slurm job itself. Stages:

  1. Profile   wtune_case.build_profile() on the case's own namelist/wrfinput (case dir resolved
               from campaign/configs.py CASES with SITE=scc overlay) -- a PRIOR, not a measurement.
  2. Round 0   ANCHOR (tools/scc_sweep.py build_anchor -- predictor/experiment.py
               _generated_baseline) + a decomp_space.py-sampled diverse set, split across
               `floor(max_nodes/N)` parallel Slurm jobs (cold start, no prior measurements needed).
  3. Adaptive  predictor/decomp_search.py propose_for_case (kappa small, light exploration),
     rounds    one job per round (sequential -- each round needs the previous round's measurements),
               seeded with this case's own measured rows. predictor/decomp_search.py needs
               numpy+scikit-learn+catboost, which this controller does NOT require to import --
               the round loop calls an injectable `proposer(state, batch_size)`, and the real one is
               imported lazily on first use (see `_default_proposer`).
  4. Confirm   top-3 by median + the anchor, one job, REPS=2 (the runner's own palindromic order),
               same N.
  5. Report    ranked table + best config's parameter table (namelist lines, mpirun line, env) +
               job ids, printed to stdout and written to <out>/tune_report.json. Calls
               wtune_predict.build_prediction()/render_text() for the full-run timeline when that
               module is importable (owned by another agent, edited concurrently -- best-effort,
               never touched here).

--rank-by predicted (decision 0055, opt-in; default t0 is everything above, unchanged): plan
ERA5-rebuildable points, rebuild each once (coldstart real_only), rank round0/adaptive candidates
from P*'s shared spin-up restart (ANCHOR first + ANCHOREND last in every job), then measure the
top-3 + anchor at EVERY point in parallel and pick the minimum predicted whole-run compute+init
(wtune_probe_plan.combine) -- see run_predicted.

Budget discipline: never submits a job whose estimated (safety-padded) wall time would finish
after the --minutes deadline; at the deadline, cancels only Slurm job ids THIS run itself
submitted. Everything that touches the outside world (sbatch/sacct/scancel, the wall clock, the
ledger reader, the adaptive-round proposer) is a parameter with a real default, injected by tests.

Reuses (does not reimplement): tools/scc_sweep.py (sbatch command builder, manifest-free
submit_job/sacct_states runners, build_anchor/build_configs, estimate_time/estimate_s_per_step --
also re-exports campaign/configs.py, predictor/decomp_space.py and predictor/experiment.py as
`sweep.campaign_configs`/`sweep.decomp_space`/`sweep.campaign_experiment`), wtune_case.py
(case-understanding profile), predictor/decomp_search.py (active search loop, lazy import),
wtune_predict.py (full-run prediction + parameter-table shape, best-effort import).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import datetime
import glob
import io
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.join(_HERE, "tools")
_PREDICTOR_DIR = os.path.join(_HERE, "predictor")
for _p in (_TOOLS_DIR, _PREDICTOR_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scc_sweep as sweep      # tools/scc_sweep.py -- see module docstring for what is reused
import wtune_case              # wtune_case.py: build_profile (PRIOR, pre-measurement)
import wtune_compliance as wc  # decision 0056: assert_strict_only guard on the final report
import wtune_extensions
import wtune_nodes         # wtune_nodes.py: pure node-count choice from measured s/step
import wtune_config            # wtune_config.py: --config YAML (search restrictions, budget/probe/
                                # prediction overrides) -- stdlib only (PyYAML is imported lazily,
                                # only when --config is actually given), safe to import unconditionally
import targets as pred_targets # predictor/targets.py -- stdlib-only (math + predictor/features.py,
                                # itself stdlib-only), safe to import unconditionally like wtune_config
                                # above; used only for its NOISE constant (run-to-run reproducibility
                                # floor, log(1.005) ~ 0.5% relative) to keep the predicted-wall-time
                                # ranking's tie-break stable across noise-level differences.

try:
    import wtune_predict        # wtune_predict.py -- owned by another agent; best-effort only
except ImportError:
    wtune_predict = None

try:
    import wtune_probe_plan     # wtune_probe_plan.py -- multi-point probe planning/combine
                                 # (2026-09-26 redesign); best-effort only, same convention as
                                 # wtune_predict above (it depends on wtune_predict + coldstart_
                                 # sample, so anything that makes either unimportable also makes
                                 # this unimportable).
except ImportError:
    wtune_probe_plan = None

try:
    import wtune_io_model       # run-length I/O terms of the pipeline's whole-run prediction
except ImportError:             # (depends on wtune_predict + coldstart_sample, same convention)
    wtune_io_model = None

try:
    import coldstart_sample as coldstart   # tools/coldstart_sample.py -- REBUILD-source
                                            # execution (fetch/stage/score_windows_from_rsl);
                                            # same best-effort convention as the two imports above.
except ImportError:
    coldstart = None

campaign_configs = sweep.campaign_configs
decomp_space = sweep.decomp_space

DEFAULT_MAX_NODES = 13                 # scc compute pool: work[1-13], head excluded (scc_sweep.py)
DEFAULT_STEPS = 20
DEFAULT_SEED = 1
DEFAULT_KAPPA = 0.5                    # light exploration (spec: "kappa small, e.g. 0.5")
DEFAULT_HARDWARE_ID = sweep.DEFAULT_HARDWARE_ID
CONFIRM_REPS = 2
ROUND0_BUDGET_FRACTION = 0.35          # share of the total budget round 0's parallel slots may use
ADAPTIVE_STOP_FRACTION = 0.75          # spec: "stop proposing when projected finish > 75% of budget"
ADAPTIVE_MAX_ROUNDS = 6
KAPPA_ESCALATION_FACTOR = 2.0          # retry a round that stopped early with more exploration
KAPPA_MAX_ESCALATIONS = 2              # pressure before falling back to raw decomp_space diversity
DEFAULT_POLL_INTERVAL_S = 20.0
PROGRESS_BAR_WIDTH = 24

_RUN_ID_RE = re.compile(r"^(?P<case>[^_]+)_(?P<cid>.+)_r(?P<rep>\d+)_(?P<job>\d+)$")
# mirrors campaign/strict_campaign.sbatch:962 RUN_ID="${CASE}_${cid}_r${rep}_${J}" and
# predictor/decomp_search.py's own _RUN_ID_RE -- duplicated (not imported) because decomp_search
# pulls in numpy/sklearn/catboost this controller must not require just to parse a run id.


# ---------------------------------------------------------------------------
# clock / scancel: the two runners scc_sweep.py does not already provide
# ---------------------------------------------------------------------------
class QueueCapExceeded(RuntimeError):
    """--max-queue-hours: the accumulated queue-only wait reached the cap; in-flight jobs were
    cancelled and the run stops."""


class RealClock:
    def now(self):
        return time.monotonic()

    def sleep(self, seconds):
        if seconds > 0:
            time.sleep(seconds)


def default_scancel_runner(job_ids):
    if not job_ids:
        return ""
    out = subprocess.run(["scancel"] + [str(j) for j in job_ids],
                          capture_output=True, text=True, check=False)
    return out.stdout


def default_ledger_reader(ledger_dir, case, job_ids):
    """[row, ...] -- every ledger row for these job ids (ledger_<case>_<job>.csv, one file per
    Slurm job -- campaign/strict_campaign.sbatch's own naming, matches scc_sweep.py cmd_status)."""
    rows = []
    for jid in job_ids:
        path = os.path.join(ledger_dir, "ledger_%s_%s.csv" % (case, jid))
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


# ---------------------------------------------------------------------------
# small pure helpers (each independently unit-testable)
# ---------------------------------------------------------------------------
def _fmt_hms(seconds):
    if seconds is None or seconds == float("inf"):
        return "none"
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, s) if h else "%d:%02d" % (m, s)


def parse_run_id(run_id):
    m = _RUN_ID_RE.match(run_id or "")
    return m.groupdict() if m else None


def compute_slots(max_nodes, nodes):
    """Parallel Slurm jobs that fit within `max_nodes` at `nodes` each -- floor(max_nodes/N),
    never less than 1 (a single job still runs even if N alone exceeds max_nodes)."""
    return max(1, int(max_nodes) // int(nodes))


def _legal_rows(profile, max_nodes):
    table = profile.get("core_count_priors", {}).get("efficiency_table", [])
    return sorted((r for r in table if r.get("legal") and r["nodes"] <= max_nodes),
                  key=lambda r: r["nodes"])


def choose_nodes_min(profile, max_nodes):
    """policy=min: the smallest legal profiled node count -- the conservative choice when the
    efficiency proxy is not trusted at all."""
    rows = _legal_rows(profile, max_nodes)
    if not rows:
        return 1, "policy=min: no legal node count in the profile table; defaulting to nodes=1"
    return rows[0]["nodes"], "policy=min: smallest legal profiled node count (cap=%d)" % max_nodes


def choose_nodes_cap(profile, max_nodes):
    """policy=cap: the largest legal profiled node count within --max-nodes, ignoring the
    efficiency proxy entirely -- maximum parallelism, no scaling-model trust required."""
    rows = _legal_rows(profile, max_nodes)
    if not rows:
        return max_nodes, ("policy=cap: no legal node count in the profile table; defaulting to "
                            "--max-nodes=%d" % max_nodes)
    return rows[-1]["nodes"], "policy=cap: largest legal profiled node count within cap=%d" % max_nodes


def choose_nodes_knee(profile, max_nodes):
    """policy=knee (default): the node count where predicted marginal speedup of the next
    doubling step (wtune_case's own NODE_COUNTS_TO_CHECK table: 1,2,4,8,13) drops below 1.5x.
    actual_speedup(n) = n*E(n) (E = wtune_case's efficiency proxy), so the wall-time speedup of
    going from n_i to n_j is (n_j*E(n_j)) / (n_i*E(n_i)); stops at the first n_i whose next step
    does not clear 1.5x, or at the largest capped node count if every step still does."""
    rows = _legal_rows(profile, max_nodes)
    if not rows:
        return 1, "policy=knee: no legal node count in the profile table; defaulting to nodes=1"
    for i in range(len(rows) - 1):
        n_i, e_i = rows[i]["nodes"], rows[i].get("efficiency_E") or 0.0
        n_j, e_j = rows[i + 1]["nodes"], rows[i + 1].get("efficiency_E") or 0.0
        speedup_i, speedup_j = n_i * e_i, n_j * e_j
        marginal = (speedup_j / speedup_i) if speedup_i > 0 else float("inf")
        if marginal < 1.5:
            return n_i, ("policy=knee: doubling nodes=%d->%d has predicted marginal speedup "
                          "%.2fx < 1.5x; stopping at nodes=%d" % (n_i, n_j, marginal, n_i))
    n = rows[-1]["nodes"]
    if n < max_nodes:
        # Every doubling among the LEGAL profiled node counts cleared 1.5x, but nothing above n
        # is legal for this case's grid (or was not among wtune_case's fixed checkpoints) -- say
        # so explicitly, so "using nodes=%d" is never read as if it had verified doubling all the
        # way to --max-nodes and then inexplicably stopped short.
        return n, ("policy=knee: every doubling among the LEGAL profiled node counts (<=%d) "
                   "clears 1.5x; nodes=%d is the largest one wtune_case profiled as legal "
                   "(--max-nodes cap is %d, not reached)" % (n, n, max_nodes))
    return n, ("policy=knee: every profiled doubling step up to --max-nodes cap=%d clears 1.5x; "
               "using nodes=%d" % (max_nodes, n))


NODES_POLICIES = {"min": choose_nodes_min, "cap": choose_nodes_cap, "knee": choose_nodes_knee}


def choose_nodes(profile, max_nodes, policy):
    """(nodes, note, warning). The efficiency proxy every policy but `cap` is built on
    (wtune_case.py's E(px,py), halo constant c=1.0) is UNCALIBRATED -- never fit against a real
    strong-scaling run -- so this never silently commits to a threshold crossing (the old
    behaviour: falling back to nodes=1 with no comment when nothing cleared an 0.8 bar). The
    policy is always explicit and the choice always carries a printable warning."""
    fn = NODES_POLICIES.get(policy)
    if fn is None:
        raise SystemExit("unknown --nodes-policy %r (choices: %s)"
                          % (policy, ", ".join(sorted(NODES_POLICIES))))
    nodes, note = fn(profile, max_nodes)
    warning = ("scaling model is UNCALIBRATED (wtune_case.py's halo-efficiency proxy, c=1.0, "
               "never fit to a real measurement) -- nodes=%d (policy=%s) is a starting point, "
               "not a validated optimum" % (nodes, policy))
    return nodes, note, warning


def would_exceed_budget(elapsed_s, estimated_job_s, budget_s):
    return (elapsed_s + estimated_job_s) > budget_s


def chunk_round_robin(items, n):
    """Split `items` into up to `n` lists, round-robin (so a mostly-empty tail doesn't collapse
    every candidate into one slot) -- empty chunks are dropped. Item 0 (the anchor, by
    scc_sweep.build_configs's own ordering) always lands in chunk 0."""
    n = max(1, int(n))
    chunks = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return [c for c in chunks if c]


def _apply_search_filter(ctx, configs):
    """The ONE shared config-search filter (wtune_config.candidate_allowed/filter_candidates),
    applied at every place this controller generates decomposition candidates: round0's own
    ANCHOR + sampled configs, round0's top-up wave, the adaptive-round proposer's own output, and
    the diverse-sampling fallback (sample_more_diverse_configs, below -- shared by both the top-up
    wave and the adaptive-round fallback). With no --config given, ctx.search_cfg is
    wtune_config.DEFAULTS['search'] (every axis None) and ctx.exclude_rules is [], so this is a
    no-op and behaviour is unchanged from before --config existed.

    The anchor (scc_sweep.build_anchor's own id="ANCHOR"/config_source="anchor") is exempt from
    the STRICT search restrictions (omp/numtiles/rank_distribution/aspect_max) but still must obey
    `exclude` rules -- an anchor dropped by `exclude` is simply omitted, not replaced."""
    out = []
    omp_cap = getattr(ctx, "omp_cap", None)
    for c in configs:
        exempt = c.get("config_source") == "anchor" or c.get("id") == "ANCHOR"
        chem_e_sn = getattr(ctx, "chem_e_sn", None)
        if chem_e_sn and c.get("ny"):     # PROVISIONAL WRF-Chem narrow-j-patch hang guard
            why = wtune_extensions.chem_j_patch_reason(chem_e_sn, int(c["ny"]), c.get("nx"))
            if why:
                print("config: %s %s" % (c.get("id"), why))
                continue
        if omp_cap and int(c.get("omp", 1) or 1) > omp_cap:   # dmpar-only extension build
            if exempt:
                print("config: ANCHOR omp=%s dropped: OpenMP not supported by this extension build"
                      % c.get("omp"))
            continue
        if wtune_config.candidate_allowed(c, ctx.search_cfg, ctx.exclude_rules, ctx.is_fire,
                                           exempt_search=exempt):
            out.append(c)
        elif exempt:
            print("config: ANCHOR excluded by an `exclude` rule")
    return out


def sample_more_diverse_configs(ctx, exclude_configs, k, seed):
    """Up to `k` NOT-yet-measured decomp_space candidates -- the same enumerate_space/sample_design
    pool round0 itself draws from (predictor/decomp_space.py), covering omp levels, numtiles and
    aspect ratios round0/adaptive rounds have not tried yet. Used both to top up round0 when it
    turns out cheaper than assumed and, in run_adaptive_rounds, as the fallback once
    predictor/decomp_search.py's own Bayesian proposer has nothing left to suggest -- reused
    (never re-derived): `exclude_configs`' identity is scc_sweep.py's own build_configs dedup key
    (_identity: ranks/omp/nx/ny/numtiles/dist/blocktime). Returns [] when the pool is exhausted.

    Filtered through `_apply_search_filter` BEFORE `sample_design` draws from it, so a --config
    search restriction actually shrinks the pool sample_design samples from, not just what
    survives afterward."""
    total_cores = ctx.nodes * cores_used(ctx)
    candidates, _axis = decomp_space.enumerate_space(ctx.case_json, total_cores, ctx.nodes,
                                                       ctx.hardware_id)
    candidates = _apply_search_filter(ctx, candidates)
    seen = {sweep._identity(c) for c in exclude_configs if c}
    fresh = [c for c in candidates if sweep._identity(c) not in seen]
    if not fresh or k <= 0:
        return []
    chosen, _coverage = decomp_space.sample_design(fresh, min(k, len(fresh)), seed,
                                                     must_include=None)
    return decomp_space.to_campaign_configs(chosen, prefix="D")


def aggregate_by_cid(rows, configs_by_cid):
    """cid -> {cid, config, s_values, median, lo, hi, n, run_ids} -- one entry per config id that
    has at least one row with a real s_per_step_median (a failed run has an empty
    s_per_step_median and an rc= note -- see campaign/strict_campaign.sbatch -- so it is silently
    dropped here, not counted as a measurement). run_ids is each contributing row's own run_id
    string (parallel to s_values), kept so a caller can locate that exact run's own RUNDIR --
    campaign/strict_campaign.sbatch's RUNDIR="$OUT/${CASE}_${cid}_r${rep}_${J}" has run_id
    "${CASE}_${cid}_r${rep}_${J}" as its own basename verbatim -- see _rsl_paths_for_entry."""
    by_cid = {}
    for r in rows:
        parsed = parse_run_id(r.get("run_id"))
        if not parsed:
            continue
        s = sweep._to_float(r.get("s_per_step_median"))
        if s is None or s <= 0:
            continue
        by_cid.setdefault(parsed["cid"], []).append((s, r.get("run_id")))
    out = {}
    for cid, pairs in by_cid.items():
        svals = [p[0] for p in pairs]
        out[cid] = dict(cid=cid, config=configs_by_cid.get(cid), s_values=svals,
                         median=statistics.median(svals), lo=min(svals), hi=max(svals),
                         n=len(svals), run_ids=[p[1] for p in pairs])
    return out


def _log_key(value):
    """log(value), or -inf for a non-positive value (degenerate -- a real wall-time prediction or
    median s/step is always > 0, but this keeps sorting well-defined instead of raising)."""
    return math.log(value) if value > 0 else float("-inf")


def _cluster_rank(entries, log_key, anchor_cid, order_index):
    """Rank `entries` by log_key ascending, but a run-to-run difference smaller than
    predictor/targets.py NOISE (0.5% relative -- the reproducibility floor, see that module's own
    docstring) is not a reliable ordering. Fixes a real correctness bug in the previous
    functools.cmp_to_key pairwise comparator: pairwise "within NOISE of each OTHER" is not
    transitive (e.g. log-ratios 0/0.3%/0.6% -- first and third both differ from the middle one by
    less than NOISE, but not from each other), so sorted()'s result depended on the input order,
    meaning the winner was not deterministic. This is a proper two-pass algorithm instead:

      1. stable-sort every entry by (log_key, order_index) -- a single, transitive ordering.
      2. walk that sorted list left to right and greedily group into clusters: a cluster's HEAD is
         its first (fastest) member; a later entry joins the current cluster iff
         log_key(entry) - log_key(head) < NOISE, else it starts a new cluster. Comparing only
         against the head (not the previous element) keeps a cluster's own spread bounded to
         < NOISE, and clusters partition the sorted list, so the cluster order itself is exactly
         the log_key order -- no cross-cluster ambiguity, no input-order dependence.
      3. within one cluster, the anchor (anchor_cid) sorts first; every other member keeps its
         pre-sort order_index (the candidate set's original, pre-ranking order) -- the existing
         outcome-independent tie-break, now applied per cluster instead of pairwise."""
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


def rank_predicted_entries(predicted, anchor_cid):
    """Full ranking of predict_candidates() output, PREDICTED WHOLE-RUN WALL TIME ascending
    (2026-09-24 decision: the tuner picks by predicted total wall time, not raw s/step) --
    successfully predicted entries first (clustered/tie-broken by _cluster_rank on
    log(pred_total_s)), then entries whose prediction raised (pred_error set), clustered the same
    way on log(median) instead. order_index is by cid, sorted -- a canonical order that depends
    only on the candidate SET, not on `predicted`'s dict-construction order, so the within-cluster
    tie-break (and therefore the final ranking) is the same regardless of what order a caller
    happened to build the dict in -- see _cluster_rank's docstring for why raw dict-insertion
    order broke transitivity across a chain of near-ties."""
    order_index = {cid: i for i, cid in enumerate(sorted(predicted))}
    ok = [e for e in predicted.values() if e.get("pred_total_s") is not None]
    failed = [e for e in predicted.values() if e.get("pred_total_s") is None]
    ranked_ok = _cluster_rank(ok, lambda e: _log_key(e["pred_total_s"]), anchor_cid, order_index)
    ranked_failed = _cluster_rank(failed, lambda e: _log_key(e["median"]), anchor_cid, order_index)
    return ranked_ok + ranked_failed


def select_confirm_entries(ctx, nodes, all_results, anchor_cid):
    """Top-3 by PREDICTED whole-run wall time (predict_candidates + rank_predicted_entries; an
    entry whose prediction failed falls back to median s/step and ranks after every successfully
    predicted entry -- see rank_predicted_entries), plus the anchor's own result if it did not
    already make the top 3 (order preserved, deduped by cid) -- the confirm-stage candidate list,
    as full aggregate_by_cid entries (config + this run's own measured medians + pred_* fields)."""
    predicted = predict_candidates(ctx, nodes, all_results)
    ranked = rank_predicted_entries(predicted, anchor_cid)
    top = list(ranked[:3])
    cids = {r["cid"] for r in top}
    if anchor_cid in predicted and anchor_cid not in cids:
        top.append(predicted[anchor_cid])
    seen, entries = set(), []
    for r in top:
        if r["cid"] in seen or r["config"] is None:
            continue
        seen.add(r["cid"])
        entries.append(r)
    return entries


def select_confirm_configs(ctx, nodes, all_results, anchor_cid):
    """[config, ...] -- select_confirm_entries's configs only (kept for callers/tests that just
    want the candidate list, not each entry's own measured s_values/predictions)."""
    return [e["config"] for e in select_confirm_entries(ctx, nodes, all_results, anchor_cid)]


DEFAULT_STARTUP_S = sweep.STARTUP_S
# scc_sweep.py's own documented constant ("per-run startup overhead assumed for the time-limit
# estimate") -- the fallback until THIS run has real Slurm elapsed time of its own to measure a
# better number from (see measure_startup_s / Ctx.startup_s).


def estimate_run_time(n_or_pairs, reps, steps, s_est=None, startup_s=DEFAULT_STARTUP_S):
    """Wall-time estimate for one Slurm job running `reps` replicate(s) each. `n_or_pairs` is
    EITHER a plain config COUNT, sized off one blended `s_est` (round0/adaptive rounds, before
    anything in the batch is measured -- `s_est` is required in this form), OR a list of
    [(config, s_per_step), ...] PAIRS, sized off each config's OWN already-measured median
    (confirm, round0's top-up -- `s_est` is ignored in this form: "per-config" sizing).

    Returns two DIFFERENT numbers and callers must never confuse them -- this function exists
    because a prior version didn't, and every budget decision silently used the wrong one (2026-
    09-23 conus25 run: a raw 5.5-minute confirm job was rejected as if it were an 18-minute one):
      total_run_s   RAW/EXPECTED duration (reps * sum(startup_s + steps*s_per_step)), no safety
                    padding at all. The ONLY number would_exceed_budget/projected-finish-vs-75%
                    or any other budget/feasibility decision may ever use.
      job_hours/time_limit/timeout_s
                    the SAME padding formula as scc_sweep.estimate_time (TIME_SAFETY_FACTOR x1.5
                    + TIME_PAD_MIN 10min, capped at TIME_CAP_HOURS; TIMEOUT_MULTIPLIER/
                    TIMEOUT_MIN_S -- every constant reused from scc_sweep.py, not copied),
                    generalized to take `startup_s` as a parameter instead of scc_sweep.py's own
                    fixed STARTUP_S. For the sbatch --time/TIMEOUT_S request ONLY -- never for a
                    budget decision."""
    if isinstance(n_or_pairs, int):
        per_run_s_list = [startup_s + steps * s_est] * n_or_pairs
    else:
        per_run_s_list = [startup_s + steps * s for _cfg, s in n_or_pairs]
    total_run_s = reps * sum(per_run_s_list)
    job_hours = total_run_s / 3600.0
    padded_hours = job_hours * sweep.TIME_SAFETY_FACTOR + sweep.TIME_PAD_MIN / 60.0
    capped_hours = min(padded_hours, sweep.TIME_CAP_HOURS)
    max_per_run_s = max(per_run_s_list) if per_run_s_list else 0.0
    timeout_s = max(sweep.TIMEOUT_MIN_S, int(math.ceil(sweep.TIMEOUT_MULTIPLIER * max_per_run_s)))
    return dict(total_run_s=total_run_s, job_hours=capped_hours, uncapped_hours=padded_hours,
                time_limit=sweep.campaign_experiment._format_hms(capped_hours * 60.0),
                timeout_s=timeout_s)


def _parse_slurm_elapsed(s):
    """Seconds from a Slurm --format=Elapsed value ('[D-][HH:]MM:SS'), or None if empty/unparsable."""
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


def measure_startup_s(job_configs, states, wave_results, steps):
    """(startup_s, n_runs) measured from THIS wave's own Slurm elapsed time (sacct, via
    wait_for_jobs' return value) minus its own measured compute time (steps * each config's
    measured s/step, from aggregate_by_cid) -- the empirical alternative to
    DEFAULT_STARTUP_S/scc_sweep.py's documented constant, which cannot know what THIS case's real
    per-run overhead (Slurm dispatch, WRF init/read, ...) actually is. `job_configs`:
    {job_id: [config, ...]} -- exactly what that job ran, REPS=1 assumed (round0/adaptive only,
    never confirm's REPS=2, whose per-run duration this formula does not disentangle). Only
    COMPLETED jobs count: a cancelled or TIMEOUT_S-killed job's elapsed reflects the budget
    deadline or the watchdog, not real per-run cost. (None, 0) when nothing usable is available --
    the caller then keeps whatever startup_s it already had."""
    total_elapsed_s, total_compute_s, n_runs = 0.0, 0.0, 0
    for job_id, configs in job_configs.items():
        state, elapsed_str = states.get(job_id, ("UNKNOWN", ""))
        if state != "COMPLETED":
            continue
        elapsed_s = _parse_slurm_elapsed(elapsed_str)
        if elapsed_s is None:
            continue
        compute_s, counted = 0.0, 0
        for c in configs:
            r = wave_results.get(c["id"])
            if r is None:
                continue
            compute_s += steps * r["median"]
            counted += 1
        if counted == 0:
            continue
        total_elapsed_s += elapsed_s
        total_compute_s += compute_s
        n_runs += counted
    if n_runs == 0:
        return None, 0
    return max(0.0, (total_elapsed_s - total_compute_s) / n_runs), n_runs


def update_s_est(all_results, fallback):
    """This run's own s_est, refreshed from THIS run's successful measurements so far (median of
    each measured config's own median s/step) -- must win over the ledger-history/no-ledger
    `fallback` once real numbers exist, or every later budget decision keeps sizing jobs off a
    number this run has already disproven (see the run() call site for the bug this fixes)."""
    medians = [r["median"] for r in all_results.values() if r.get("median")]
    if not medians:
        return fallback
    return statistics.median(medians)


def local_param_table(config, nodes):
    """Fallback parameter table (namelist lines / mpirun line / env) built directly from
    campaign/sites/scc.sh's site_launch()/site_env() and campaign/configs.py's own
    _insert_after_domains() namelist-line format -- used whenever wtune_predict is not
    importable, or its own build_param_table raises."""
    ranks, omp = int(config["ranks"]), int(config["omp"])
    nx, ny, numtiles = int(config["nx"]), int(config["ny"]), int(config["numtiles"])
    dist = config.get("dist", "block")
    tpn = ranks // nodes if nodes else ranks
    launch = "mpirun -np %d --map-by ppr:%d:node:PE=%d --bind-to core" % (ranks, tpn, omp)
    if dist == "cyclic":
        launch += " --rank-by node"     # site_launch(): cyclic reproduces WRF's cyclic rank order
    launch += " ./wrf.exe"
    return dict(
        ranks=ranks, omp=omp, nodes=nodes, ranks_per_node=tpn, nproc_x=nx, nproc_y=ny,
        numtiles=numtiles, dist=dist, launch_line=launch,
        env=dict(OMP_NUM_THREADS=str(omp), OMP_STACKSIZE="1G"),
        namelist_lines=["nproc_x=%d," % nx, "nproc_y=%d," % ny, "numtiles=%d," % numtiles],
        ulimit="ulimit -s unlimited",
        note="KMP_BLOCKTIME: n/a on scc (nvhpc OpenMP runtime has no KMP_BLOCKTIME concept)",
    )


# ---------------------------------------------------------------------------
# run context
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Ctx:
    case: str
    case_json: dict
    namelist_path: str
    nodes: int
    cores_per_node: int
    hardware_id: str
    steps: int
    seed: int
    kappa: float
    budget_s: float
    max_nodes: int
    out_dir: str
    ledger_dir: str
    env_source: str
    sbatch_script: str
    sacct_cmd: str
    poll_interval: float
    sbatch_runner: object
    sacct_runner: object
    scancel_runner: object
    ledger_reader: object
    proposer: object
    clock: object
    start_time: float = 0.0
    owned_jobs: list = dataclasses.field(default_factory=list)
    cancelled_jobs: list = dataclasses.field(default_factory=list)
    # --config wiring (wtune_config.py): defaults reproduce pre---config behaviour exactly --
    # search_cfg with every axis None / exclude_rules=[] is a no-op filter, is_fire=False never
    # matches a `when: fire` exclude rule, confirm_reps==CONFIRM_REPS, prediction_cfg is
    # wtune_config.DEFAULTS['prediction'] (io_target="nfs", no drift override, r_rad not requested).
    search_cfg: dict = dataclasses.field(
        default_factory=lambda: copy.deepcopy(wtune_config.DEFAULTS["search"]))
    exclude_rules: list = dataclasses.field(default_factory=list)
    is_fire: bool = False
    chem_e_sn: object = None    # e_sn list of an active WRF-Chem case (j-patch guard), else None
    omp_cap: object = None      # 1 for WRF-Chem/Hydro/WRFDA (dmpar-only builds), else None
    confirm_reps: int = CONFIRM_REPS
    prediction_cfg: dict = dataclasses.field(
        default_factory=lambda: copy.deepcopy(wtune_config.DEFAULTS["prediction"]))
    # probe.* beyond steps/confirm_reps (spinup_min/ranking_start/points/max_points/budget_pct/
    # sources -- 2026-09-26 multi-point redesign): threaded through so build_full_run_prediction
    # can call wtune_probe_plan.plan_points for the winner. Default reproduces pre-redesign
    # behaviour's OWN steps/confirm_reps via the ctx.steps/ctx.confirm_reps fields above, which
    # this dict does not override.
    probe_cfg: dict = dataclasses.field(
        default_factory=lambda: copy.deepcopy(wtune_config.DEFAULTS["probe"]))
    ranking_start_note: str = None
    # Per-run startup overhead every estimate_run_time call sizes off -- starts at
    # DEFAULT_STARTUP_S (scc_sweep.py's documented constant) and is refreshed once round0's own
    # Slurm elapsed time gives a real measurement (measure_startup_s); startup_source records
    # which one is currently in effect, for the log line.
    startup_s: float = DEFAULT_STARTUP_S
    startup_source: str = "documented constant (scc_sweep.STARTUP_S)"
    startup_runs: int = None         # completed round0 runs behind a measured startup_s
    # decision 0055 (--rank-by predicted): rank_by is "t0" (pre-0055 behaviour) or "predicted".
    # `ranking` is the RANKING CONTEXT every ranking job is submitted with in predicted mode (see
    # ranking_context()); None -- the default and always in t0 mode -- leaves every submission,
    # chunk and ledger read exactly as before. ranking_gates collects each ranking job's
    # evaluate_job_gates() record (stage 1 warns, stage 2 raises).
    rank_by: str = "t0"
    restart_io_form: int = 14
    ranking: dict = None
    ranking_gates: list = dataclasses.field(default_factory=list)
    # stage-1 per-job record {job_id: {s_anchor_job, excluded, reason}} (_record_stage1_gates),
    # consumed by stage1_normalized_ranking; always {} in t0 mode.
    stage1_jobs: dict = dataclasses.field(default_factory=dict)
    # Budget clock (decision 0055 gap 3). budget_mode "wall" (default, t0 mode): elapsed() is
    # plain wall time, exactly as before. "running" (predicted mode): elapsed() excludes every
    # wait_for_jobs interval during which none of the waited jobs was RUNNING (queue_only_s) --
    # the budget pays for running + controller work, not for queueing -- and max_wall_s is the
    # hard wall-clock cap (default 3x budget) after which in-flight jobs are cancelled anyway.
    budget_mode: str = "wall"
    queue_only_s: float = 0.0
    max_wall_s: float = None
    # predicted-mode stage-2 reserve inputs (stage2_reserve_s) and the s/step prior used before
    # anything is measured; job_cids/not_measured record which candidates each submitted job
    # carried and which were cancelled unstarted at the stage-1 cutoff.
    reserve: dict = None
    s_prior: float = None
    restart_read_s: float = None           # None -> RESTART_READ_S_BY_IO[restart_io_form]
    stage1_max_running: int = 0            # max stage-1 ranking jobs seen RUNNING at once
    # predicted-mode STAGE-1 context {mode: "t0"|"pstar", anchor}: set for the whole of stage 1
    # in either mode (ctx.ranking is set only for pstar). None outside stage 1 and in t0 mode.
    stage1: dict = None
    stage1_mode: str = "t0"                # predicted mode's stage 1: "t0" (default) | "pstar"
    # stage1=t0 (2026-09-28): --minutes is the STAGE-1 search budget only; stage 2 is a
    # separate, announced cost. reserve_in_stage1=False there (no stage-2 reserve held back from
    # stage 1; pstar keeps True). uncharged_s: controller time that is not stage-1 work (the
    # rebuild-prep fetch/stage/submit), excluded from the running clock like queue_only_s.
    # stage2_budget_s: optional --stage2-minutes cap -- the only thing fit_stage2 trims to in
    # stage1=t0. max_wall_explicit: --max-wall-minutes was given (else it is derived).
    reserve_in_stage1: bool = True
    uncharged_s: float = 0.0
    stage2_budget_s: float = None
    max_wall_explicit: bool = False
    # stage1=t0: rebuild prep's estimate, held back from the stage-1 budget on top of the
    # stage-2 reserve (prep runs concurrently with stage 1); 0.0 otherwise
    stage1_extra_reserve_s: float = 0.0
    stage2_estimate_start_s: float = None
    prep_submit_s: float = 0.0
    # jobs of this run that run in the background of a stage-1 wait (rebuild prep in
    # stage1=t0): sampled for "is anything of ours RUNNING" (running budget), never waited on
    background_jobs: list = dataclasses.field(default_factory=list)
    stage2_overrun_s: float = None         # stage-2 running time beyond the budget
    stage1_trimmed: list = dataclasses.field(default_factory=list)
    job_cids: dict = dataclasses.field(default_factory=dict)
    not_measured: list = dataclasses.field(default_factory=list)
    # node search: the sweep's hybrid ANCHOR at ctx.nodes, used as round0's ANCHOR (else None)
    anchor_override: dict = None
    # node search: cores per node the decomposition search may use (None = all of cores_per_node)
    core_budget: int = None
    # full layout search (default level): enumerate every legal (ranks x omp, nx x ny, numtiles)
    # layout at the finalist instead of sampling; layout_notes is shared across ctx copies
    full_space: bool = False
    layout_notes: dict = None
    # node health pre-flight result (None when --no-node-preflight): see node_preflight
    node_health: dict = None

    # --pipeline full: ONE queue-wait accumulator shared by every dataclasses.replace() copy of the
    # ctx (the per-phase / per-finalist copies), {"s": queue-only seconds, "cap_hit": bool}; None
    # elsewhere (queue_only_s above is then the per-ctx accumulator). max_queue_s: stop (raise
    # QueueCapExceeded) once the accumulated queue-only time reaches it.
    shared_queue: dict = None
    max_queue_s: float = None

    @property
    def queue_wait_s(self):
        return self.shared_queue["s"] if self.shared_queue is not None else self.queue_only_s

    def add_queue_wait(self, seconds):
        if self.shared_queue is not None:
            self.shared_queue["s"] += seconds
        else:
            self.queue_only_s += seconds

    def add_running(self, seconds):
        """--pipeline full: charge `seconds` during which at least one of our jobs was RUNNING."""
        if self.shared_queue is not None and "run_s" in self.shared_queue:
            self.shared_queue["run_s"] += seconds

    def wall_elapsed(self):
        return self.clock.now() - self.start_time

    def elapsed(self):
        if self.budget_mode == "running":
            if self.shared_queue is not None and "run_s" in self.shared_queue:
                # --pipeline full: the budget is the time our jobs were RUNNING -- queue wait AND
                # controller work (report building, predictions, the fidelity gate, conversions,
                # submission) are excluded
                return self.shared_queue["run_s"]
            return (self.clock.now() - self.start_time - self.queue_wait_s - self.uncharged_s)
        return self.clock.now() - self.start_time


# ---------------------------------------------------------------------------
# case resolution / profile
# ---------------------------------------------------------------------------
def resolve_case(case_id, cores_per_node):
    case_json = sweep.resolve_case_json(case_id)
    prep_dir = sweep._expand(case_json["prep_dir"])
    namelist_path = sweep._expand(campaign_configs._case_namelist_path(case_json))
    _, wrfinput_path = wtune_case.find_case_inputs(prep_dir)
    profile = wtune_case.build_profile(namelist_path, wrfinput_path, cores_per_node)
    return case_json, namelist_path, profile


def resolve_auto_probe_steps(namelist_path):
    """probe.steps='auto' (2026-09-26 multi-point redesign, wtune_probe_plan.steps_for_stepra):
    max(30, stepra+10), so round0/adaptive/confirm's own probe always contains >= 1 radiation
    step -- the whole reason this redesign exists (see this task's report / wtune_probe_plan.py's
    module docstring for the conus25 evidence: a fixed 20-step t0 probe never reached radt=10min/
    dt=15s's own 40-step radiation cadence). Returns (steps, note_or_None). Never raises: falls
    back to DEFAULT_STEPS with an explanatory note when wtune_predict/wtune_probe_plan are not
    importable or the namelist's own dt/radt cannot be read -- probing with a slightly-wrong step
    count is better than refusing to run at all."""
    if wtune_predict is None or wtune_probe_plan is None:
        return DEFAULT_STEPS, ("probe.steps='auto' requested but wtune_predict/wtune_probe_plan "
                                "is not importable -- using DEFAULT_STEPS=%d instead" % DEFAULT_STEPS)
    try:
        nl = wtune_predict.parse_namelist(namelist_path)
        dt = wtune_predict.get_timestep_seconds(nl)
        radt = wtune_predict.g(nl, "physics", "radt", None)
        if radt is None:
            raise ValueError("no radt in &physics")
        stepra = wtune_predict.steps_per_radiation_call(float(radt), dt)
        steps = wtune_probe_plan.steps_for_stepra(stepra)
        return steps, ("probe.steps='auto' resolved to %d steps (dt=%.3fs, radt=%smin, stepra=%d)"
                        % (steps, dt, radt, stepra))
    except Exception as e:
        return DEFAULT_STEPS, ("probe.steps='auto' requested but could not be resolved from %s "
                                "(%s: %s) -- using DEFAULT_STEPS=%d instead"
                                % (namelist_path, type(e).__name__, e, DEFAULT_STEPS))


# ---------------------------------------------------------------------------
# progress / polling / cancellation
# ---------------------------------------------------------------------------
def print_progress(stage, elapsed_s, budget_s, note=""):
    frac = 0.0 if budget_s <= 0 else min(1.0, elapsed_s / budget_s)
    filled = int(round(frac * PROGRESS_BAR_WIDTH))
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    line = "[%-9s] [%s] %3d%%  elapsed=%s / budget=%s" % (
        stage, bar, int(frac * 100), _fmt_hms(elapsed_s), _fmt_hms(budget_s))
    if note:
        line += "  " + note
    print(line)


def cancel_jobs(ctx, job_ids):
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        return
    print("budget deadline reached: cancelling %d in-flight job(s) owned by this run: %s"
          % (len(job_ids), ",".join(job_ids)))
    ctx.scancel_runner(job_ids)
    ctx.cancelled_jobs.extend(job_ids)


def wait_for_jobs(ctx, job_ids, stage, note="", pending_cutoff=None, exempt_budget=False):
    """Polls sacct until every id in `job_ids` is out of PENDING/RUNNING, or the budget deadline
    passes (in which case only THOSE ids -- never any other job -- are scancel'd). Prints one
    progress line per poll. Returns the last sacct states dict seen ({job_id: (State, Elapsed)},
    {} when `job_ids` was empty) -- callers use the terminal Elapsed to measure real per-run
    startup cost (measure_startup_s). With ctx.budget_mode == "running" (predicted mode) the
    loop is _wait_running_budget instead; `pending_cutoff`/`exempt_budget` are only used there."""
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        return {}
    if ctx.budget_mode == "running":
        return _wait_running_budget(ctx, job_ids, stage, note, pending_cutoff, exempt_budget)
    while True:
        elapsed = ctx.elapsed()
        states = sweep.sacct_states(job_ids, sacct_runner=ctx.sacct_runner, sacct_cmd=ctx.sacct_cmd)
        in_flight = [j for j in job_ids
                     if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES]
        print_progress(stage, elapsed, ctx.budget_s, note)
        if not in_flight:
            return states
        if elapsed >= ctx.budget_s:
            cancel_jobs(ctx, in_flight)
            return states
        ctx.clock.sleep(min(ctx.poll_interval, max(0.0, ctx.budget_s - elapsed)))


def _wait_running_budget(ctx, job_ids, stage, note, pending_cutoff, exempt_budget=False):
    """wait_for_jobs under the RUNNING-time budget (decision 0055 gap 3; helper any mode can
    adopt by setting ctx.budget_mode="running"). Each poll interval is charged to the budget
    only when at least one waited job was RUNNING at the start of it; otherwise it is added to
    ctx.queue_only_s (pure queueing). Cancels every in-flight id (cancel_jobs) once the
    running-time budget or ctx.max_wall_s (hard wall-clock cap) is reached. `pending_cutoff`
    (callable -> bool, optional): once true, ids still PENDING are cancelled as never-started
    (cancel_unstarted -- their candidates are "not measured", not an error) while RUNNING ones
    are waited for. `exempt_budget` (stage 2 once started): only the wall cap cancels -- running
    time past the budget is reported as overrun instead. During predicted-mode stage 1 (either
    stage1 mode), every poll's RUNNING count updates ctx.stage1_max_running
    (stage2_concurrency). ctx.background_jobs (rebuild prep running beside stage 1) count as
    "ours RUNNING" for the budget clock but are never waited on or cancelled here."""
    while True:
        elapsed = ctx.elapsed()
        wall = ctx.wall_elapsed()
        bg = [j for j in ctx.background_jobs if j not in job_ids]
        states = sweep.sacct_states(job_ids + bg, sacct_runner=ctx.sacct_runner,
                                    sacct_cmd=ctx.sacct_cmd)
        bg_running = any(states.get(j, ("", ""))[0] == "RUNNING" for j in bg)
        states = {j: v for j, v in states.items() if j not in bg}
        in_flight = [j for j in job_ids
                     if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES
                     and j not in ctx.cancelled_jobs]
        print_progress(stage, elapsed, ctx.budget_s,
                       (note + " " if note else "") + "(running budget; wall %s, queue-only %s)"
                       % (_fmt_hms(wall), _fmt_hms(ctx.queue_wait_s)))
        if not in_flight:
            return states
        if pending_cutoff is not None and pending_cutoff():
            pend = [j for j in in_flight if states.get(j, ("", ""))[0] == "PENDING"]
            if pend:
                cancel_unstarted(ctx, pend, stage)
                in_flight = [j for j in in_flight if j not in pend]
                if not in_flight:
                    return states
        over_budget = elapsed >= ctx.budget_s and not exempt_budget
        if over_budget or (ctx.max_wall_s is not None and wall >= ctx.max_wall_s):
            if not over_budget:
                print("hard wall-clock cap reached (wall %s >= %s)"
                      % (_fmt_hms(wall), _fmt_hms(ctx.max_wall_s)))
            cancel_jobs(ctx, in_flight)
            return states
        n_running = sum(1 for j in in_flight if states.get(j, ("", ""))[0] == "RUNNING")
        if _stage1_active(ctx) and n_running > ctx.stage1_max_running:
            ctx.stage1_max_running = n_running
            print("%s: observed %d of this run's ranking job(s) RUNNING at once (stage-2 "
                  "concurrency now %d)" % (stage, n_running, stage2_concurrency(ctx)))
        running = n_running > 0 or bg_running
        step = min(ctx.poll_interval, max(0.0, ctx.budget_s - elapsed))
        if exempt_budget:
            step = ctx.poll_interval
        if ctx.max_wall_s is not None:
            step = min(step, max(0.0, ctx.max_wall_s - wall))
        if not running:
            # nothing is consuming budget: poll at the normal interval, bounded only by the
            # wall-clock cap (the running budget cannot run out while nothing runs)
            step = ctx.poll_interval if ctx.max_wall_s is None else \
                min(ctx.poll_interval, max(0.0, ctx.max_wall_s - wall))
        t0 = ctx.clock.now()
        ctx.clock.sleep(step)
        if running:
            ctx.add_running(ctx.clock.now() - t0)
        if not running:
            ctx.add_queue_wait(ctx.clock.now() - t0)
            if ctx.max_queue_s is not None and ctx.queue_wait_s >= ctx.max_queue_s:
                if ctx.shared_queue is not None:
                    ctx.shared_queue["cap_hit"] = True
                cancel_jobs(ctx, [j for j in in_flight if j not in ctx.cancelled_jobs])
                raise QueueCapExceeded(
                    "queue-only wait %s reached --max-queue-hours (%s); in-flight job(s) %s "
                    "cancelled" % (_fmt_hms(ctx.queue_wait_s), _fmt_hms(ctx.max_queue_s),
                                   ",".join(in_flight)))


def cancel_unstarted(ctx, job_ids, stage):
    """Stage-1 cutoff: scancel jobs that never started and record their candidates (from
    ctx.job_cids, minus ANCHOR/ANCHOREND) in ctx.not_measured -- reported, not an error."""
    print("%s: stage-1 cutoff reached -- cancelling %d job(s) that never started: %s "
          "(their candidates are not measured)" % (stage, len(job_ids), ",".join(job_ids)))
    ctx.scancel_runner(job_ids)
    ctx.cancelled_jobs.extend(job_ids)
    for j in job_ids:
        cids = [c for c in ctx.job_cids.get(j, []) if c not in ("ANCHOR", ANCHOR_END_ID)]
        ctx.not_measured.append(dict(job_id=j, cids=cids, stage=stage,
                                     reason="job still PENDING at the stage-1 cutoff"))


# ---------------------------------------------------------------------------
# submitting one CONFIG_FILE job (thin wrapper around scc_sweep's own builders)
# ---------------------------------------------------------------------------
def submit_configs_job(ctx, configs, reps, name_suffix, s_est, time_est=None, ranking=None):
    """One strict_campaign.sbatch CONFIG_FILE job. `ranking` (default: ctx.ranking; None in t0
    mode) is the decision-0055 ranking context (see ranking_context): the job then runs as the
    rebuilt point's own case id with CASES_JSON/RANKING_START=spinup/SPINUP_MINUTES/
    RANKING_STRICT_RESTART/SPINUP_TIMEOUT_S/RANKING_RESTART_IO_FORM exported, and its config list
    is bracketed ANCHOR ... ANCHOREND (wrap_with_anchors -- idempotent, so a caller that already
    wrapped for its own time estimate is unaffected). CAMPAIGN=<the sbatch script's own campaign/
    dir> is exported whenever that dir is a real campaign checkout (scc_sweep.campaign_dir_for)
    -- deliberate 2026-09-27 default change so the spooled script never pairs with a stale
    ~/wtune/campaign's helpers."""
    ranking = ctx.ranking if ranking is None else ranking
    if ranking:
        configs = wrap_with_anchors(configs, ranking["anchor"])
    elif _stage1_active(ctx):
        configs = _ranking_wrap(ctx, configs)        # stage1=t0: anchors in every job too
    if time_est is None:
        time_est = _job_time_est(ctx, len(configs), reps, s_est, ranking=ranking)
    job_out_dir = os.path.join(ctx.out_dir, name_suffix)
    os.makedirs(job_out_dir, exist_ok=True)
    configs_path = os.path.join(job_out_dir, "configs.json")
    sweep.write_configs_json(configs_path, configs)
    cmd_str = sweep.build_sbatch_command(
        ranking["case_id"] if ranking else ctx.case, ctx.nodes, ctx.cores_per_node,
        time_est["time_limit"], time_est["timeout_s"],
        reps, ranking["steps"] if ranking else ctx.steps, os.path.abspath(configs_path),
        job_out_dir, env_source=ctx.env_source, sbatch_script=ctx.sbatch_script,
        campaign_dir=sweep.campaign_dir_for(ctx.sbatch_script),
        extra_export=ranking_export(ranking, time_est) if ranking else None)
    job_id = sweep.submit_job(cmd_str, sbatch_runner=ctx.sbatch_runner)
    ctx.owned_jobs.append(job_id)
    ctx.job_cids[job_id] = [c["id"] for c in configs]
    return job_id, configs_path, time_est


# ---------------------------------------------------------------------------
# round 0: anchor + diverse sample, fanned out across parallel slots
# ---------------------------------------------------------------------------
def _submit_round0_wave(ctx, configs, slots, s_est, name_prefix):
    """Chunks `configs` round-robin across `slots` parallel jobs (each a fresh SLURM allocation of
    ctx.nodes nodes) sized at `s_est` s/step and ctx.startup_s per run, submits every chunk whose
    RAW expected duration (never the safety-padded one) still fits the budget, and waits for all
    of them. Shared by round0's first wave and its top-up wave. Returns (job_ids, configs_by_cid,
    rows, job_configs, states) -- job_configs ({job_id: [config,...]}) and states (wait_for_jobs'
    own sacct snapshot) are what measure_startup_s needs afterward. Predicted mode (ctx.ranking)
    goes through _submit_ranking_wave instead (shrink-to-fit, ANCHOR/ANCHOREND per slot)."""
    if _stage1_active(ctx):
        return _submit_ranking_wave(ctx, configs, slots, s_est, name_prefix)
    chunks = chunk_round_robin(configs, slots)
    job_ids, configs_by_cid, job_configs = [], {}, {}
    for i, chunk in enumerate(chunks):
        chunk = _ranking_wrap(ctx, chunk)     # predicted mode: ANCHOR ... ANCHOREND in EVERY chunk
        time_est = _job_time_est(ctx, len(chunk), 1, s_est)
        if would_exceed_budget(ctx.elapsed(), time_est["total_run_s"], ctx.budget_s):
            print("%s slot %d: SKIP -- estimated end would exceed budget" % (name_prefix, i))
            continue
        ok, why = _stage1_may_start(ctx, time_est["total_run_s"])
        if not ok:
            print("%s slot %d: SKIP -- %s" % (name_prefix, i, why))
            continue
        job_id, _path, _te = submit_configs_job(ctx, chunk, reps=1,
                                                  name_suffix="%s-s%d" % (name_prefix, i),
                                                  s_est=s_est, time_est=time_est)
        job_ids.append(job_id)
        job_configs[job_id] = chunk
        for c in chunk:
            configs_by_cid[c["id"]] = c
        print("%s slot %d: submitted job %s (%d configs)" % (name_prefix, i, job_id, len(chunk)))

    states = wait_for_jobs(ctx, job_ids, name_prefix, pending_cutoff=_stage1_cutoff(ctx))
    rows = ctx.ledger_reader(_ledger_case(ctx), job_ids)
    _record_stage1_gates(ctx, job_ids, rows, name_prefix)
    return job_ids, configs_by_cid, rows, job_configs, states


def fit_stage1_job(ctx, candidates, s_est):
    """Predicted mode: the largest PREFIX of `candidates` (ANCHOR/ANCHOREND stripped; the order
    is sampling priority, so the later-sampled ones are dropped first) whose job -- spin-up +
    ANCHOR + prefix + ANCHOREND -- still fits: elapsed_running + job + stage-2 reserve <= budget.
    Minimum is ANCHOR + 1 candidate + ANCHOREND. Returns (wrapped, dropped, time_est), or
    (None, candidates, time_est_of_the_minimum) when even the minimum does not fit."""
    cands = [c for c in candidates if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    est = None
    for n in range(len(cands), 0, -1):
        wrapped = wrap_with_anchors(cands[:n], _stage1_anchor(ctx))
        est = _job_time_est(ctx, len(wrapped), 1, s_est)
        ok, _why = _stage1_may_start(ctx, est["total_run_s"])
        if ok:
            return wrapped, cands[n:], est
    return None, cands, est


def _submit_ranking_wave(ctx, configs, slots, s_est, name_prefix):
    """Predicted-mode stage-1 wave: candidates (ANCHOR stripped -- every slot gets its own
    ANCHOR/ANCHOREND) are dealt round-robin over min(slots, #candidates) slots (slots =
    floor(max_nodes/nodes), what can run at once), each slot is SHRUNK to fit (fit_stage1_job)
    instead of skipped, the plan (slots, candidates per slot, expected minutes, cost model) is
    logged BEFORE anything is submitted, and a slot is skipped only when even ANCHOR + 1 +
    ANCHOREND does not fit. Trimmed candidates go to ctx.stage1_trimmed (reported)."""
    cands = [c for c in configs if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    n_slots = max(1, min(int(slots), len(cands)))
    plan = []
    for i, chunk in enumerate(chunk_round_robin(cands, n_slots)):
        wrapped, dropped, est = fit_stage1_job(ctx, chunk, s_est)
        plan.append((i, wrapped, dropped, est))
    print("%s plan: %d slot(s) (max %d concurrent = floor(max_nodes/nodes)), candidates per slot "
          "%s, expected %s per job; elapsed %s, %s"
          % (name_prefix, sum(1 for p in plan if p[1]), slots,
             [len(p[1]) - 2 if p[1] else 0 for p in plan],
             ", ".join(_fmt_hms(p[3]["total_run_s"]) if p[1] else "skip" for p in plan),
             _fmt_hms(ctx.elapsed()), describe_stage2_reserve(ctx)))
    if plan and plan[0][3] and plan[0][3].get("cost_model"):
        print("%s cost model: %s" % (name_prefix, describe_cost_model(plan[0][3]["cost_model"])))
    job_ids, configs_by_cid, job_configs = [], {}, {}
    for i, wrapped, dropped, est in plan:
        for c in dropped:
            ctx.stage1_trimmed.append(dict(cid=c["id"], stage=name_prefix,
                                           reason="trimmed to fit the budget + stage-2 reserve"))
        if wrapped is None:
            print("%s slot %d: SKIP -- even ANCHOR + 1 candidate + ANCHOREND (%s) does not fit: %s"
                  % (name_prefix, i, _fmt_hms(est["total_run_s"]) if est else "-",
                     _stage1_may_start(ctx, est["total_run_s"])[1] if est else "no candidate"))
            continue
        if dropped:
            print("%s slot %d: trimmed %d candidate(s) to fit: %s"
                  % (name_prefix, i, len(dropped), ", ".join(c["id"] for c in dropped)))
        job_id, _path, _te = submit_configs_job(ctx, wrapped, reps=1,
                                                  name_suffix="%s-s%d" % (name_prefix, i),
                                                  s_est=s_est, time_est=est)
        job_ids.append(job_id)
        job_configs[job_id] = wrapped
        for c in wrapped:
            configs_by_cid[c["id"]] = c
        print("%s slot %d: submitted job %s (%d configs, expected %s)"
              % (name_prefix, i, job_id, len(wrapped), _fmt_hms(est["total_run_s"])))
    states = wait_for_jobs(ctx, job_ids, name_prefix, pending_cutoff=_stage1_cutoff(ctx))
    rows = ctx.ledger_reader(_ledger_case(ctx), job_ids)
    _record_stage1_gates(ctx, job_ids, rows, name_prefix)
    return job_ids, configs_by_cid, rows, job_configs, states


def measure_ranking_overhead(ctx, job_configs, states, wave_results):
    """Predicted-mode analogue of measure_startup_s: a ranking job's Slurm elapsed also holds
    the shared spin-up and every run's restart read, so per-run overhead = (elapsed - compute) /
    (runs + 1), compute = sum(steps * median * rad_factor) + spin_steps * s_anchor *
    rad_factor(spin). The result replaces startup_s + restart read together. (None, 0) when no
    COMPLETED job can be used."""
    r = ctx.ranking
    spin_steps = r["spinup_minutes"] * 60.0 / float(r["dt"])
    f_run, f_spin = rad_factor(r["steps"], r.get("stepra")), rad_factor(spin_steps, r.get("stepra"))
    tot_e = tot_c = 0.0
    n = 0
    for jid, configs in job_configs.items():
        state, elapsed_str = states.get(jid, ("UNKNOWN", ""))
        el = _parse_slurm_elapsed(elapsed_str)
        a = wave_results.get("ANCHOR")
        if state != "COMPLETED" or el is None or not a:
            continue
        comp, cnt = spin_steps * a["median"] * f_spin, 0
        for c in configs:
            w = wave_results.get(c["id"])
            if w:
                comp += r["steps"] * w["median"] * f_run
                cnt += 1
        if cnt:
            tot_e += el
            tot_c += comp
            n += cnt + 1
    if not n:
        return None, 0
    return max(0.0, (tot_e - tot_c) / n), n


def _configs_for_budget(budget_s, per_run_s, slots):
    """How many configs fit `budget_s` at `per_run_s` each; with NO time cap (budget infinite) a
    fixed effort instead: EST_ROUND0_CONFIGS spread over the slots."""
    if not math.isfinite(budget_s):
        return max(1, int(math.ceil(EST_ROUND0_CONFIGS / float(max(1, slots)))))
    return max(1, int(budget_s // per_run_s))


FULL_SPACE_CONFIGS_PER_JOB = 24      # configs in one full-space job (~1 h at 20 steps)


def enumerate_full_space(ctx):
    """EVERY legal layout at ctx.nodes x cores_used(ctx): ranks x omp pairs, their nx x ny shapes,
    numtiles (decomp_space.enumerate_space, every active domain legal). The axes the user ranks
    highest; dist=block and 1-D tiles only (scc has no KMP_BLOCKTIME, so blocktime variants are the
    same run -- cyclic dist and 2-D tile shapes are left to the adaptive proposer / sampled
    levels). Returns (configs, info): configs = [campaign configs, ANCHOR override first],
    info = counts and the reasons candidates were dropped (extension OMP cap, WRF-Chem j-patch
    guard, --config excludes)."""
    total = ctx.nodes * cores_used(ctx)
    cands, _axis = decomp_space.enumerate_space(ctx.case_json, total, ctx.nodes, ctx.hardware_id)
    seen, uniq = set(), []
    for c in cands:
        if c["dist"] != "block" or c["blocktime"] != "0" or c["tile_shape"] != "1d_y":
            continue
        ident = sweep._identity(c)
        if ident not in seen:
            seen.add(ident)
            uniq.append(c)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kept = _apply_search_filter(ctx, uniq)
    reasons = sorted({l.strip() for l in buf.getvalue().splitlines() if l.strip()})
    info = dict(nodes=ctx.nodes, cores_per_node=cores_used(ctx), raw=len(cands), unique=len(uniq),
                legal=len(kept), excluded=len(uniq) - len(kept), excluded_reasons=reasons[:8])
    configs = decomp_space.to_campaign_configs(kept, prefix="D")
    ov = ctx.anchor_override
    if ov:
        configs = [ov] + [c for c in configs if sweep._identity(c) != sweep._identity(ov)]
    return configs, info


def _run_full_space_round0(ctx, s_est, slots):
    """round0 for the full layout search: every enumerated layout measured once (REPS=1), in
    waves of `slots` concurrent jobs of FULL_SPACE_CONFIGS_PER_JOB configs, nothing sampled.
    Anything that could not be measured is recorded in ctx.layout_notes["untried"] with its
    reason. Returns (job_ids, configs_by_cid, rows) like run_round0."""
    configs, info = enumerate_full_space(ctx)
    notes = ctx.layout_notes if ctx.layout_notes is not None else dict(untried=[], enumerated={})
    notes["enumerated"]["N=%d c=%d" % (ctx.nodes, cores_used(ctx))] = info
    print("round0 (full layout search): N=%d cores/node=%d: %d enumerated, %d unique, %d legal "
          "(%d excluded) -> %d waves of %d job(s)"
          % (ctx.nodes, cores_used(ctx), info["raw"], info["unique"], info["legal"],
             info["excluded"], math.ceil(len(configs) / float(slots * FULL_SPACE_CONFIGS_PER_JOB)),
             slots))
    if not configs:
        return [], {}, []
    job_ids, configs_by_cid, rows = [], {}, []
    wave_size = slots * FULL_SPACE_CONFIGS_PER_JOB
    for w, start in enumerate(range(0, len(configs), wave_size)):
        chunk = configs[start:start + wave_size]
        ids, by_cid, wrows, job_configs, states = _submit_round0_wave(
            ctx, chunk, slots, s_est, "round0-w%d" % w)
        job_ids += ids
        configs_by_cid.update(by_cid)
        rows += wrows
        if w == 0 and ids:               # the first wave measures this allocation's startup
            res = aggregate_by_cid(wrows, by_cid)
            measured, n_runs = measure_startup_s(job_configs, states, res, ctx.steps)
            if measured is not None:
                print("startup_s: measured %.1fs/run from %d run(s) (was %.1fs, %s)"
                      % (measured, n_runs, ctx.startup_s, ctx.startup_source))
                ctx.startup_s = measured
                ctx.startup_source = "measured from round0 (n=%d completed runs)" % n_runs
                ctx.startup_runs = n_runs
            s_est = update_s_est(res, s_est)
    measured_cids = {r["cid"] for r in aggregate_by_cid(rows, configs_by_cid).values()}
    for c in configs:
        if c["id"] not in configs_by_cid:
            reason = "not submitted: the job would exceed the time cap"
        elif c["id"] not in measured_cids:
            reason = "no measurement (the run failed or its job was cancelled)"
        else:
            continue
        notes["untried"].append(dict(nodes=ctx.nodes, cores_per_node=cores_used(ctx), id=c["id"],
                                     ranks=c["ranks"], omp=c["omp"], nx=c["nx"], ny=c["ny"],
                                     numtiles=c["numtiles"], reason=reason))
    return job_ids, configs_by_cid, rows


def run_round0(ctx):
    s_est, s_source, n_rows = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                          ctx.hardware_id)
    if ctx.full_space:
        return _run_full_space_round0(ctx, s_est, compute_slots(ctx.max_nodes, ctx.nodes))
    if _stage1_active(ctx) and ctx.s_prior and n_rows == 0:
        # predicted mode, no ledger history: the cell-step model beats the 10 s/step fallback,
        # which would also inflate the stage-2 reserve until nothing in stage 1 fits
        s_est, s_source = ctx.s_prior, "cell-step model"
    per_run_s = ctx.startup_s + ctx.steps * s_est
    slots = compute_slots(ctx.max_nodes, ctx.nodes)
    round0_budget_s = min(ctx.budget_s - ctx.elapsed() - stage1_reserve_s(ctx),
                          ctx.budget_s * ROUND0_BUDGET_FRACTION)
    configs_per_job = _configs_for_budget(round0_budget_s, per_run_s, slots)
    total_k = slots * configs_per_job
    if _stage1_active(ctx) and not ctx.ranking:
        # stage1=t0: the t0 tuner's own round0 sizing (above) exactly; +1 because
        # build_configs' ANCHOR is stripped again -- every slot carries its own ANCHOR/ANCHOREND
        # on top (two extra runs per job; fit_stage1_job trims a slot that then overruns)
        total_k = slots * configs_per_job + 1
    elif _stage1_active(ctx):
        # stage1=pstar: what one job can hold after its spin-up and two anchor runs, with the
        # stage-1 reserve (the stage-2 reserve) set aside; +1 as above
        est1 = _job_time_est(ctx, 1, 1, s_est)
        m = est1.get("cost_model")
        per_run = m["per_run_s"] if m else est1["total_run_s"]
        spin = m["spin_s"] if m else 0.0
        avail = ctx.budget_s - ctx.elapsed() - stage1_reserve_s(ctx) - spin - 2 * per_run
        configs_per_job = max(1, int(avail // per_run))
        total_k = slots * configs_per_job + 1

    built = sweep.build_configs(ctx.case, ctx.case_json, ctx.nodes, max(0, total_k - 1), ctx.seed,
                                 ctx.hardware_id, cores_used(ctx))
    if ctx.anchor_override:
        # node-search finalist: the sweep's hybrid ANCHOR (all cores used) replaces build_configs'
        # omp=1 downward-scanned one; the other candidates stay as sampled
        ov = ctx.anchor_override
        rest = [c for c in built["configs"]
                if c.get("id") != "ANCHOR" and sweep._identity(c) != sweep._identity(ov)]
        built = dict(built, configs=[ov] + rest, skip_reason=None)
    print("round0: s_est=%.3fs/step[%s,n=%d] startup_s=%.1fs[%s] slots=%d configs_per_job=%d "
          "total=%d (%s)"
          % (s_est, s_source, n_rows, ctx.startup_s, ctx.startup_source, slots, configs_per_job,
             len(built["configs"]), built["skip_reason"] or "ok"))
    if built["skip_reason"]:
        return [], {}, []

    configs0 = _apply_search_filter(ctx, built["configs"])
    if not configs0:
        print("round0: SKIP -- every candidate (including the anchor) was excluded by --config "
              "search restrictions/exclude rules")
        return [], {}, []

    job_ids, configs_by_cid, rows, job_configs, states = _submit_round0_wave(
        ctx, configs0, slots, s_est, "round0")

    if job_ids:
        wave0_results = aggregate_by_cid(rows, configs_by_cid)

        # Measured startup: refresh ctx.startup_s from THIS wave's own Slurm elapsed time before
        # sizing the top-up below, so the top-up (and every later stage) sizes off what this run
        # actually measured, not the documented fallback constant -- logged either way so a run's
        # own log always says which one is in effect (2026-09-23 conus25 ask).
        if ctx.ranking:
            measured_startup, n_runs = measure_ranking_overhead(ctx, job_configs, states,
                                                                wave0_results)
            if measured_startup is not None:
                ctx.restart_read_s = 0.0      # the measured overhead already includes it
        else:
            measured_startup, n_runs = measure_startup_s(job_configs, states, wave0_results,
                                                         ctx.steps)
        if measured_startup is not None:
            print("startup_s: measured %.1fs/run from %d completed run(s) in round0 (was %.1fs, %s)"
                  % (measured_startup, n_runs, ctx.startup_s, ctx.startup_source))
            ctx.startup_s = measured_startup
            ctx.startup_source = "measured from round0 (n=%d completed runs)" % n_runs
            ctx.startup_runs = n_runs
        else:
            print("startup_s: no COMPLETED round0 run to measure from; keeping %.1fs (%s)"
                  % (ctx.startup_s, ctx.startup_source))

        # Top-up wave: `s_est` above is the PRE-round0 estimate (ledger history, or the no-ledger
        # fallback CONSERVATIVE_S_PER_STEP=10s/step for a case never run on scc before), so wave 0
        # itself can be badly under-provisioned on a fast case -- 2026-09-22 sc26p2 run: real cost
        # ~0.03s/step, fallback assumed 10s/step (333x higher), so configs_per_job floored to 1
        # and only 6 configs (slots) got tried out of a budget that could fund far more. Once wave
        # 0's own measurements are in, refresh s_est and, if real cost came in below what wave 0
        # assumed and there is still budget, spend one more wave sampling MORE not-yet-measured
        # decomp_space candidates (other omp levels/numtiles/aspect ratios) sized off the now-
        # accurate estimate -- this is also how a single-slot case (slots==1, e.g. --max-nodes
        # close to --nodes) still gets more than one job's worth of exploration out of round0.
        refreshed_s_est = update_s_est(wave0_results, s_est)
        if refreshed_s_est < s_est and ctx.elapsed() < ctx.budget_s:
            topup_per_run_s = ctx.startup_s + ctx.steps * refreshed_s_est
            topup_budget_s = min(ctx.budget_s - ctx.elapsed(), ctx.budget_s * ROUND0_BUDGET_FRACTION)
            topup_configs_per_job = _configs_for_budget(topup_budget_s, topup_per_run_s, slots)
            topup_k = slots * topup_configs_per_job
            fresh = sample_more_diverse_configs(ctx, list(configs_by_cid.values()), topup_k,
                                                 ctx.seed + 1)
            print("round0 top-up: s_est=%.3fs/step (measured, was %.3fs/step) startup_s=%.1fs "
                  "configs_per_job=%d candidates=%d"
                  % (refreshed_s_est, s_est, ctx.startup_s, topup_configs_per_job, len(fresh)))
            if fresh:
                topup_ids, topup_by_cid, topup_rows, _jc, _st = _submit_round0_wave(
                    ctx, fresh, slots, refreshed_s_est, "round0-topup")
                job_ids = job_ids + topup_ids
                configs_by_cid.update(topup_by_cid)
                rows = rows + topup_rows

    return job_ids, configs_by_cid, rows


# ---------------------------------------------------------------------------
# adaptive rounds: predictor/decomp_search.py propose_for_case, lazily imported
# ---------------------------------------------------------------------------
def _default_proposer(ctx):
    """Builds a proposer(state, batch_size) -> (result, configs_out) closure over
    predictor/decomp_search.py, imported here (not at module load) so wtune_tune.py itself never
    requires numpy/scikit-learn/catboost -- only exercised when no test-injected proposer is
    supplied and an adaptive round actually runs."""
    import decomp_search   # predictor/decomp_search.py

    case_fields = decomp_search.case_row_fields(ctx.case_json)
    prior_rows = []
    for path in sorted(glob.glob(os.path.join(ctx.ledger_dir, "ledger_*.csv"))):
        if os.path.basename(path).startswith("ledger_%s_" % ctx.case):
            continue    # this case's own ledgers are already in `state` (case_measured_rows)
        if ctx.ranking and os.path.basename(path).startswith("ledger_%s_" % ctx.ranking["case_id"]):
            continue    # predicted mode: this run's own rebuilt-point ledgers, same reason
        try:
            prior_rows.extend(decomp_search.benchmark.load_ledger(path))
        except Exception:
            continue    # best-effort cross-case prior; a malformed sibling ledger is not fatal

    def proposer(state, batch_size, kappa=None):
        return decomp_search.propose_for_case(
            ctx.case, ctx.case_json, case_fields, prior_rows, state, batch_size,
            total_cores=ctx.nodes * cores_used(ctx), nodes=ctx.nodes,
            hardware_id=ctx.hardware_id, kappa=ctx.kappa if kappa is None else kappa)

    return proposer


def run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est):
    proposer = ctx.proposer
    if proposer is None:
        try:
            proposer = _default_proposer(ctx)
        except ImportError as e:
            print("adaptive rounds skipped: predictor/decomp_search.py not importable (%s)" % e)
            return

    state = dict(configs={}, measured={}, rounds=[])
    for cid, r in all_results.items():
        if r["config"] is None:
            continue
        state["configs"][cid] = r["config"]
        state["measured"][cid] = [dict(run_id="%s_%s_r%d_seed" % (ctx.case, cid, i),
                                        job_id=None, rep_index=i, s_per_step_median=s)
                                   for i, s in enumerate(r["s_values"])]

    for round_idx in range(1, ADAPTIVE_MAX_ROUNDS + 1):
        elapsed = ctx.elapsed()
        if not _stage1_active(ctx) and elapsed >= ADAPTIVE_STOP_FRACTION * ctx.budget_s:
            print("adaptive rounds: stopping before round %d -- elapsed already >= %.0f%% of budget"
                  % (round_idx, ADAPTIVE_STOP_FRACTION * 100))
            return

        remaining_s = ctx.budget_s - elapsed - stage1_reserve_s(ctx)
        if _stage1_active(ctx) and remaining_s <= 0:
            print("adaptive rounds: stopping before round %d -- no stage-1 time left (elapsed %s, "
                  "stage-1 reserve %s, budget %s)"
                  % (round_idx, _fmt_hms(elapsed), _fmt_hms(stage1_reserve_s(ctx)),
                     _fmt_hms(ctx.budget_s)))
            return
        per_run_s = ctx.startup_s + ctx.steps * s_est
        batch_size = _configs_for_budget(min(remaining_s, ctx.budget_s * ROUND0_BUDGET_FRACTION), per_run_s, 1)

        # `state["rounds"]` (propose_core's own round counter) must grow exactly once per LOGICAL
        # round_idx, so a kappa retry below re-reads the SAME internal round_idx from
        # propose_core's perspective -- appending on every attempt would desync round_idx==1's
        # C01-reference forcing and the round cap from what was actually measured.
        result, configs_out, kappa = None, [], ctx.kappa
        for attempt in range(KAPPA_MAX_ESCALATIONS + 1):
            result, configs_out = proposer(state, batch_size, kappa=kappa)
            if not result.get("stopped") and configs_out:
                break
            if attempt < KAPPA_MAX_ESCALATIONS:
                next_kappa = kappa * KAPPA_ESCALATION_FACTOR
                print("adaptive round %d: proposer stopped at kappa=%.3g (%s); retrying with more "
                      "exploration pressure (kappa=%.3g)"
                      % (round_idx, kappa, result.get("reason", ""), next_kappa))
                kappa = next_kappa

        # Constrain the Bayesian proposer's own output too -- predictor/decomp_search.py knows
        # nothing about --config search restrictions, so a proposal outside them must be dropped
        # here (never submitted). If that empties the batch, fall through to the already-filtered
        # diverse-sampling fallback below exactly as if the proposer itself had stopped.
        configs_out = _apply_search_filter(ctx, configs_out)

        if result.get("stopped") or not configs_out:
            # Bayesian exploitation is exhausted even after escalating kappa -- fall back to raw
            # decomp_space diversity sampling (other omp levels/numtiles/aspect ratios not yet
            # measured) before accepting there is genuinely nothing left worth trying. Keeps
            # proposing until the 75% budget line or the legal decomp_space pool is exhausted,
            # instead of stopping at round 1 with most of the budget still unused (2026-09-22
            # sc26p2 run: stopped with 90% of the budget left after only 6 measured configs).
            fresh = sample_more_diverse_configs(ctx, list(all_configs_by_cid.values()), batch_size,
                                                 ctx.seed + round_idx)
            if not fresh:
                print("adaptive rounds: stopped at round %d -- proposer exhausted even after "
                      "kappa escalation (%s) and the decomp_space pool has no unmeasured "
                      "candidate left" % (round_idx, result.get("reason", "no candidates proposed")))
                return
            print("adaptive round %d: proposer exhausted (%s); falling back to %d decomp_space-"
                  "diverse candidate(s) not yet measured"
                  % (round_idx, result.get("reason", ""), len(fresh)))
            configs_out = fresh

        state.setdefault("rounds", []).append(dict(round=round_idx, batch_size=batch_size,
                                                     kappa=kappa))

        time_est = _job_time_est(ctx, len(_ranking_wrap(ctx, configs_out)), 1, s_est)
        if _stage1_active(ctx):
            wrapped, dropped, fit_est = fit_stage1_job(ctx, configs_out, s_est)
            if wrapped is None:
                print("adaptive rounds: stopping before round %d -- even ANCHOR + 1 candidate + "
                      "ANCHOREND does not fit before the stage-2 reserve" % round_idx)
                return
            for c in dropped:
                ctx.stage1_trimmed.append(dict(cid=c["id"], stage="round%d" % round_idx,
                                               reason="trimmed to fit the budget + stage-2 "
                                                      "reserve"))
            configs_out = wrapped[1:-1]
            time_est = fit_est
            print("adaptive round %d plan: %d candidate(s), expected %s%s"
                  % (round_idx, len(configs_out), _fmt_hms(time_est["total_run_s"]),
                     (" (trimmed %d)" % len(dropped)) if dropped else ""))
        projected_end = elapsed + time_est["total_run_s"]        # RAW/EXPECTED -- never job_hours
        ok, why = _stage1_may_start(ctx, time_est["total_run_s"])
        if not ok:
            print("adaptive rounds: stopping before round %d -- %s" % (round_idx, why))
            return
        if not _stage1_active(ctx) and projected_end > ADAPTIVE_STOP_FRACTION * ctx.budget_s:
            print("adaptive rounds: stopping before round %d -- projected finish %s > %.0f%% "
                  "of budget" % (round_idx, _fmt_hms(projected_end), ADAPTIVE_STOP_FRACTION * 100))
            return

        job_id, _path, _te = submit_configs_job(ctx, configs_out, reps=1,
                                                  name_suffix="round%d" % round_idx, s_est=s_est,
                                                  time_est=time_est)
        print("adaptive round %d: submitted job %s (%d configs)"
              % (round_idx, job_id, len(configs_out)))
        for c in configs_out:
            all_configs_by_cid[c["id"]] = c
            state["configs"][c["id"]] = c

        wait_for_jobs(ctx, [job_id], "round%d" % round_idx, pending_cutoff=_stage1_cutoff(ctx))
        rows = ctx.ledger_reader(_ledger_case(ctx), [job_id])
        _record_stage1_gates(ctx, [job_id], rows, "round%d" % round_idx)
        new_results = aggregate_by_cid(rows, all_configs_by_cid)
        new_results.pop(ANCHOR_END_ID, None)       # drift check only, never a candidate
        for cid, r in new_results.items():
            all_results[cid] = r
            state["measured"][cid] = [dict(run_id="%s_%s_r%d_%s" % (ctx.case, cid, i, job_id),
                                            job_id=job_id, rep_index=i, s_per_step_median=s)
                                       for i, s in enumerate(r["s_values"])]
        s_est = update_s_est(all_results, s_est)   # keep sizing off this run's freshest numbers


# ---------------------------------------------------------------------------
# confirm stage
# ---------------------------------------------------------------------------
def run_confirm(ctx, all_configs_by_cid, all_results, anchor_cid, s_est):
    entries = select_confirm_entries(ctx, ctx.nodes, all_results, anchor_cid)
    if not entries:
        print("confirm: SKIP -- no measured configs to confirm")
        return None, None

    # Log every candidate's own measured median + per-run estimate up front, always -- the
    # 2026-09-22 sc26p2 run skipped confirm entirely at elapsed 2.7/30 min on a case measured at
    # 0.03-0.04 s/step with no visibility into WHY; this makes the actual numbers behind that
    # decision show up in the log every time, not just when something looks wrong.
    for e in entries:
        per_run_s = ctx.startup_s + ctx.steps * e["median"]
        print("confirm candidate %-8s median=%.4fs/step (n=%d) -> per_run~%.1fs (startup_s=%.1fs) "
              "x REPS=%d" % (e["cid"], e["median"], e["n"], per_run_s, ctx.startup_s,
                              ctx.confirm_reps))

    # Greedily keep the cheapest candidates that still fit the remaining budget, cheapest first,
    # instead of an all-or-nothing decision on the WHOLE top-3+anchor set: with round0 sampling
    # only a handful of configs (see run_round0's top-up fix), "top-3 by median" can be forced to
    # include a genuinely slow outlier just to fill 3 slots (e.g. one ~6.7s/step config next to an
    # anchor at 0.03s/step) -- that single candidate's own real cost dominated the combined
    # estimate and vetoed confirming the fast ones too, even though they trivially fit alone. Uses
    # total_run_s (RAW/EXPECTED), never job_hours (safety-padded x1.5 + 10min, sbatch --time only)
    # -- 2026-09-23 conus25 run: a real 5.5-minute confirm job was rejected as an 18-minute one.
    entries_sorted = sorted(entries, key=lambda e: e["median"])
    kept, dropped, time_est = [], [], None
    for e in entries_sorted:
        trial = kept + [e]
        trial_est = estimate_run_time([(x["config"], x["median"]) for x in trial],
                                       ctx.confirm_reps, ctx.steps, startup_s=ctx.startup_s)
        if would_exceed_budget(ctx.elapsed(), trial_est["total_run_s"], ctx.budget_s):
            dropped.append(e)
            continue
        kept.append(e)
        time_est = trial_est

    if not kept:
        solo_est = estimate_run_time(
            [(entries_sorted[0]["config"], entries_sorted[0]["median"])], ctx.confirm_reps,
            ctx.steps, startup_s=ctx.startup_s)
        print("confirm: SKIP -- even the cheapest candidate alone (%s, %.4fs/step) is estimated "
              "at %s (expected, REPS=%d), which would exceed the remaining budget "
              "(elapsed=%s / budget=%s)"
              % (entries_sorted[0]["cid"], entries_sorted[0]["median"],
                 _fmt_hms(solo_est["total_run_s"]), ctx.confirm_reps, _fmt_hms(ctx.elapsed()),
                 _fmt_hms(ctx.budget_s)))
        return None, None

    if dropped:
        print("confirm: dropping %d candidate(s) that would not fit the remaining budget: %s"
              % (len(dropped),
                 ", ".join("%s(%.4fs/step)" % (e["cid"], e["median"]) for e in dropped)))

    configs = [e["config"] for e in kept]
    print("confirm: estimated %s for %d config(s) (REPS=%d, expected -- not the padded --time "
          "request) -- elapsed=%s / budget=%s"
          % (_fmt_hms(time_est["total_run_s"]), len(configs), ctx.confirm_reps,
             _fmt_hms(ctx.elapsed()), _fmt_hms(ctx.budget_s)))

    job_id, _path, _te = submit_configs_job(ctx, configs, reps=ctx.confirm_reps, name_suffix="confirm",
                                              s_est=s_est, time_est=time_est)
    print("confirm: submitted job %s (%d configs, REPS=%d)" % (job_id, len(configs), ctx.confirm_reps))
    wait_for_jobs(ctx, [job_id], "confirm")
    rows = ctx.ledger_reader(ctx.case, [job_id])
    confirm_configs_by_cid = {c["id"]: c for c in configs}
    confirm_results = aggregate_by_cid(rows, confirm_configs_by_cid)
    return job_id, confirm_results


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
_PROBE_LEDGER_FIELDS = ["case_base", "n_ranks", "omp_threads", "nproc_x", "nproc_y", "numtiles",
                         "s_per_step_median"]


def write_probe_ledger(ctx, all_results, confirm_results):
    """Writes every config THIS run actually measured (confirm's own numbers win when a config
    was both round-measured and confirmed) as a minimal ledger-shaped CSV -- exactly the columns
    wtune_predict.resolve_probe()/load_ledger_rows() read (case_base, n_ranks, omp_threads,
    nproc_x, nproc_y, numtiles, s_per_step_median), one row per replicate so a REPS=2 confirm run
    still reports its measured spread. Passed back into wtune_predict only through its own public
    build_prediction(ledger_paths=...) -- wtune_predict.py itself is never edited. An entry whose
    config is missing one of the 5 decomposition fields (a degenerate/partial config, e.g. a
    hand-built test double that only carries "id") is skipped rather than raising -- this
    function no longer runs only on the final winner (predict_candidates now calls it once per
    round/confirm result set too), so a config this incomplete must not take down every other
    candidate's probe row with it. Returns the path, or None when there is nothing measured to
    write."""
    merged = dict(all_results)
    merged.update(confirm_results or {})
    entries = [r for r in merged.values() if r.get("config") and r.get("s_values")]
    if not entries:
        return None
    os.makedirs(ctx.out_dir, exist_ok=True)
    path = os.path.join(ctx.out_dir, "predict_probe.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROBE_LEDGER_FIELDS)
        w.writeheader()
        for r in entries:
            cfg = r["config"]
            if any(k not in cfg for k in ("ranks", "omp", "nx", "ny", "numtiles")):
                continue
            for s in r["s_values"]:
                w.writerow(dict(case_base=ctx.case, n_ranks=cfg["ranks"], omp_threads=cfg["omp"],
                                 nproc_x=cfg["nx"], nproc_y=cfg["ny"], numtiles=cfg["numtiles"],
                                 s_per_step_median=s))
    return path


def _rsl_paths_for_entry(ctx, entry):
    """[path, ...] of THIS entry's own kept rsl.error.0000 probe logs, or [] when it has no
    run_ids (e.g. a result built directly by a test, not via aggregate_by_cid) or none of its
    RUNDIRs still have an rsl.error.0000 on disk. RUNDIR = ctx.ledger_dir/<run_id> exactly --
    campaign/strict_campaign.sbatch's RUN_ID="${CASE}_${cid}_r${rep}_${J}" IS its own RUNDIR's
    basename (RUNDIR="$OUT/${CASE}_${cid}_r${rep}_${J}", and $OUT == ctx.ledger_dir -- see
    campaign/sites/scc.sh's OUT= and tools/scc_sweep.py's DEFAULT_LEDGER_DIR), and cleanup keeps
    rsl.out.0000/rsl.error.0000 (never removes RUNDIR itself) in every MODE -- see that script's
    'cleanup: keep rsl.out.0000, rsl.error.0000' comment -- so a plain path join is exact, no
    reparsing of run_id needed."""
    if not entry:
        return []
    return [p for rid in (entry.get("run_ids") or [])
            for p in [os.path.join(ctx.ledger_dir, rid, "rsl.error.0000")]
            if os.path.exists(p)]


def predict_candidates(ctx, nodes, results):
    """{cid: {...results[cid]'s own keys..., pred_total_s, pred_low_s, pred_high_s, pred_source,
    pred_error}} -- one wtune_predict.build_prediction() call per candidate's OWN config, so the
    tuner can rank by PREDICTED WHOLE-RUN WALL TIME (compute + init, no I/O) instead of raw
    s/step (2026-09-24 decision). Every call shares ONE write_probe_ledger() CSV across every
    candidate: wtune_predict.resolve_probe() matches a call's own ranks/omp/nx/ny/numtiles
    exactly, so a candidate's prediction only ever sees ITS OWN measured probe rows even though
    the shared file holds every candidate's rows. rsl_paths (this candidate's own kept
    rsl.error.0000 logs, see _rsl_paths_for_entry) are scoped per candidate the same way, never
    shared. NEVER passes include_io=True: history I/O is estimated but, by user decision
    2026-09-24, excluded from the ranking objective (write rate was off -42%..+156% across cases
    on the SC26 atlas -- see wtune_predict.build_prediction's own module comment). A candidate
    with no config, or whose build_prediction() call raises, gets pred_total_s=None and a
    pred_error string instead of raising out of this function."""
    out = {}
    if wtune_predict is None:
        for cid, entry in results.items():
            out[cid] = dict(entry, pred_total_s=None, pred_low_s=None, pred_high_s=None,
                             pred_source=None, pred_error="wtune_predict not importable")
        return out

    probe_path = write_probe_ledger(ctx, results, None)
    calibration_path = str(wtune_predict.DEFAULT_CALIBRATION)
    if ctx.prediction_cfg.get("drift_band_pct") is not None:
        calibration_path, _note = _calibration_with_drift_override(
            ctx, calibration_path, ctx.prediction_cfg["drift_band_pct"])

    for cid, entry in results.items():
        cfg = entry.get("config")
        if cfg is None:
            out[cid] = dict(entry, pred_total_s=None, pred_low_s=None, pred_high_s=None,
                             pred_source=None, pred_error="no config")
            continue
        rsl_paths = _rsl_paths_for_entry(ctx, entry)
        try:
            pred = wtune_predict.build_prediction(
                namelist_path=ctx.namelist_path, case_dir=ctx.case_json["prep_dir"], nodes=nodes,
                config=dict(ranks=cfg["ranks"], omp=cfg["omp"], nx=cfg["nx"], ny=cfg["ny"],
                            numtiles=cfg["numtiles"]),
                ledger_paths=[probe_path] if probe_path else None,
                calibration_path=calibration_path, io_form_override=None,
                io_target=ctx.prediction_cfg.get("io_target") or "nfs", case_name=ctx.case,
                rsl_paths=rsl_paths or None, include_io=False)
        except Exception as e:
            out[cid] = dict(entry, pred_total_s=None, pred_low_s=None, pred_high_s=None,
                             pred_source=None, pred_error="%s: %s" % (type(e).__name__, e))
            continue
        out[cid] = dict(entry, pred_total_s=pred["total_s"], pred_low_s=pred["range_low_s"],
                         pred_high_s=pred["range_high_s"],
                         pred_source=pred["compute"]["r_rad_source"], pred_error=None)
    return out


def _calibration_with_drift_override(ctx, calibration_path, drift_pct):
    """wtune_predict.build_prediction() has no drift_band_pct PARAMETER -- it always reads
    calibration.get("drift_band_pct", 5.0) from whatever calibration_path it is given (see
    wtune_predict.py's own build_prediction). A --config prediction.drift_band_pct override is
    therefore wired in by writing a copy of the calibration file with that one field overridden
    and pointing calibration_path at the copy, never by editing wtune_predict.py itself. Returns
    (calibration_path_to_use, note)."""
    try:
        with open(calibration_path) as f:
            calib = json.load(f)
    except (OSError, ValueError) as e:
        return calibration_path, ("NOTE: prediction.drift_band_pct=%s requested but the default "
                                   "calibration file %s could not be read (%s); using it unmodified"
                                   % (drift_pct, calibration_path, e))
    calib["drift_band_pct"] = drift_pct
    os.makedirs(ctx.out_dir, exist_ok=True)
    out_path = os.path.join(ctx.out_dir, "calibration_override.json")
    with open(out_path, "w") as f:
        json.dump(calib, f)
    return out_path, ("NOTE: prediction.drift_band_pct=%s applied via calibration override %s"
                       % (drift_pct, out_path))


def _cost_series_for_ctx(ctx, run_start, run_end, case_dir, grib_dir=None):
    """wtune_cost_drivers.CostSeries when probe.points_by == "cost-indicators", else None. Never
    raises (a provider failure falls back inside build_cost_series; a missing import or domain
    bounds gives None, i.e. the time-based selection)."""
    if (ctx.probe_cfg or {}).get("points_by") != "cost-indicators":
        return None
    try:
        import wtune_cost_drivers
        return wtune_cost_drivers.cost_series_for_case(
            case_dir, run_start, run_end, ctx.probe_cfg.get("indicator_sources"), grib_dir,
            namelist_path=ctx.namelist_path)
    except Exception:
        return None


def _probe_plan_preview(ctx, nodes, best_config):
    """(points, notes, run_start, run_end, dt, stepra) from wtune_probe_plan.plan_points, or None
    on ANY failure (namelist can't be parsed, wtune_probe_plan not importable, ...) -- this is a
    best-effort, purely informational PREVIEW of where the PREDICTION-layer would probe, not a
    requirement; a failure here must never break the (already-working) single-probe prediction
    build_full_run_prediction falls back to. See that function's own probe_measurements
    parameter for the actual multi-point combine() path, used only when points were measured."""
    if wtune_predict is None or wtune_probe_plan is None:
        return None
    try:
        nl = wtune_predict.parse_namelist(ctx.namelist_path)
        steps_n, run_s, dt, _note = wtune_predict.get_d01_steps(nl)
        run_start = wtune_predict.get_time_control_datetime(nl, "start")
        if run_start is None:
            return None
        run_end = run_start + datetime.timedelta(seconds=run_s)
        radt = wtune_predict.g(nl, "physics", "radt", None)
        if radt is None:
            return None
        stepra = wtune_predict.steps_per_radiation_call(float(radt), dt)
        spinup_s = float(ctx.probe_cfg.get("spinup_min") or 0) * 60.0
        points, notes = wtune_probe_plan.plan_points(
            nl, run_start, run_end, dt, stepra, spinup_s, ctx.probe_cfg,
            case_dir=ctx.case_json.get("prep_dir"),
            indicators=_cost_series_for_ctx(ctx, run_start, run_end, ctx.case_json.get("prep_dir")))
        return points, notes, run_start, run_end, dt, stepra
    except Exception:
        return None


# ---------------------------------------------------------------------------
# C3 execution: measuring the planned points for real (2026-09-26 end-to-end wiring, decision
# 0054 "one allocation where feasible"). trajectory/restart points share ONE strict_campaign.
# sbatch job (RANKING_START=spinup, one replicate per point -- see that script's own
# SPINUP_MINUTES docstring); rebuild points share ONE coldstart_sample plan/fetch/stage/
# run.sbatch job (build_plan's own multi-`times=` pin). Each source type is its own allocation
# (not literally "one" across the whole prediction): trajectory/restart uses the STRICT/wrf.exe-
# only loop, rebuild needs the full WPS+real.exe+wrf.exe chain -- fundamentally different
# execution machinery, so combining them into a single job would mean reimplementing one inside
# the other rather than reusing either, which this task's "reuse, don't reimplement" directive
# rules out.
# ---------------------------------------------------------------------------
_RANKING_RUNDIR_ROOT = os.path.expanduser("~/wtune-runs/campaign")   # campaign/sites/scc.sh's
                                                                       # own OUT (hardcoded there,
                                                                       # not overridable via env --
                                                                       # see that file's top-level
                                                                       # OUT= assignment)


def build_ranking_sbatch_command(case, nodes, cores_per_node, time_limit, timeout_s, reps, steps,
                                  configs_path, out_dir, spinup_minutes, env_source=None,
                                  sbatch_script=None, campaign_dir=None):
    """sbatch command for a RANKING_START=spinup trajectory/restart measurement job -- same shape
    as scc_sweep.build_sbatch_command (reused: -p compute -N/--ntasks-per-node/--exclusive/
    --exclude=head/--time/--job-name/--output/--export), plus RANKING_START=spinup,
    SPINUP_MINUTES=<':'-separated list; a ',' would split sbatch --export> (one target per probe point, see strict_campaign.
    sbatch's own docstring for the rep-index mapping), and CAMPAIGN= when given (points a
    non-default checkout, e.g. a deployed validation copy, without needing $CAMPAIGN exported in
    the submitting shell)."""
    env_source = env_source or sweep.DEFAULT_ENV_SOURCE
    sbatch_script = sbatch_script or sweep.DEFAULT_SBATCH_SCRIPT
    total_cores = nodes * cores_per_node
    export = ("ALL,SITE=scc,WTUNE_TOTAL_CORES=%d,CASE=%s,MODE=timing,REPS=%d,STEPS=%d,"
              "TIMEOUT_S=%d,CONFIG_FILE=%s,RANKING_START=spinup,SPINUP_MINUTES=%s"
              % (total_cores, case, reps, steps, timeout_s, configs_path, spinup_minutes))
    if "," in str(spinup_minutes):
        raise ValueError("SPINUP_MINUTES %r contains ',' which sbatch --export would split; "
                         "use ':'" % (spinup_minutes,))
    campaign_dir = campaign_dir or sweep.campaign_dir_for(sbatch_script)
    if campaign_dir:
        export += ",CAMPAIGN=%s" % campaign_dir
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive %s "
              "--time=%s --job-name=c3-traj-%s-n%d --output=%s/c3traj-%%j.out --export=%s %s"
              % (nodes, cores_per_node, sweep.exclude_arg(), time_limit, case, nodes, out_dir, export,
                 sweep._expand(sbatch_script)))
    return "source %s; %s" % (sweep._expand(env_source), sbatch)


def run_trajectory_points(ctx, points, best_config, nodes, run_start, stepra, out_dir,
                          sbatch_script=None, campaign_dir=None):
    """Measures every trajectory/restart-sourced ProbePoint of `points` in ONE strict_campaign.
    sbatch job (RANKING_START=spinup): one CONFIG_FILE entry (id=ANCHOR, best_config's own
    decomposition -- the winner IS the candidate being measured, so ranking the anchor against
    anything else is not needed here), REPS=len(points), SPINUP_MINUTES = each point's own
    t0-relative minute (rounded to the nearest whole minute -- WRF's run-minutes/restart_interval
    namelist keys are integers; sub-minute drift from rounding is <=30s out of an hours-long run).
    Waits for the job (wait_for_jobs), then parses each replicate's own steps.txt (rep r ->
    points[r-1], matching strict_campaign.sbatch's (rep-1) mod len(targets) rule with
    reps==len(targets), no wraparound) via measurement_from_steps_file. Returns
    [PointMeasurement, ...] in `points` order. Raises RuntimeError on any infrastructure failure
    (job not COMPLETED, a replicate's steps.txt missing/empty) -- never silently drops a point."""
    if not points:
        return []
    minute_list = [int(round((p.at - run_start).total_seconds() / 60.0)) for p in points]
    reps = len(points)
    cfg = dict(id="ANCHOR", ranks=best_config["ranks"], omp=best_config["omp"],
               nx=best_config["nx"], ny=best_config["ny"], numtiles=best_config["numtiles"],
               dist=best_config["dist"], blocktime=best_config.get("blocktime", 0),
               patch_x=None, patch_y=None, numtiles_x=0, numtiles_y=0, tile_sz_x=0,
               tile_sz_y=0, tile_strategy=0, config_source="anchor")
    os.makedirs(out_dir, exist_ok=True)
    configs_path = os.path.join(out_dir, "configs_trajectory.json")
    with open(configs_path, "w") as f:
        json.dump({"configs": [cfg]}, f)
    steps = points[0].steps
    # per-run watchdog: budget 20s of wall time per simulated minute of anchor spin-up (job 1211's
    # postmortem, this task's report: a 300-min anchor needed MORE than a 20x-per-minute budget
    # once TIMEOUT_S=1800 -- 1800/300=6s/min -- was not enough) plus a floor for the short probe
    # runs/overhead.
    timeout_s = max(1800, max(minute_list) * 20)
    time_min = max(45, timeout_s // 60 + 15)
    time_limit = "%02d:%02d:00" % (time_min // 60, time_min % 60)
    cmd_str = build_ranking_sbatch_command(
        ctx.case, nodes, ctx.cores_per_node, time_limit, timeout_s, reps, steps,
        os.path.abspath(configs_path), out_dir, ":".join(str(m) for m in minute_list),
        env_source=ctx.env_source, sbatch_script=sbatch_script, campaign_dir=campaign_dir)
    job_id = sweep.submit_job(cmd_str, sbatch_runner=ctx.sbatch_runner)
    ctx.owned_jobs.append(job_id)
    print("C3 trajectory job submitted: %s (targets=%s minutes)" % (job_id, minute_list))
    states = wait_for_jobs(ctx, [job_id], "c3-trajectory")
    state = states.get(job_id, ("UNKNOWN", ""))[0]
    if state != "COMPLETED":
        raise RuntimeError("C3 trajectory job %s ended in state %s (expected COMPLETED) -- "
                            "infrastructure failure, stop and diagnose read-only" % (job_id, state))
    _n, _run_s, dt, _note = wtune_predict.get_d01_steps(
        wtune_predict.parse_namelist(ctx.namelist_path))
    measurements = []
    for i, p in enumerate(points):
        rep = i + 1
        rundir = os.path.join(_RANKING_RUNDIR_ROOT, "%s_ANCHOR_r%d_%s" % (ctx.case, rep, job_id))
        steps_path = os.path.join(rundir, "steps.txt")
        if not os.path.exists(steps_path) or os.path.getsize(steps_path) == 0:
            raise RuntimeError("C3 trajectory job %s: %s missing/empty for point %s (rep %d)"
                                % (job_id, steps_path, p.at.isoformat(), rep))
        g0 = wtune_probe_plan.first_global_step_from_rsl(
            os.path.join(rundir, "rsl.error.0000"), run_start, dt)
        measurements.append(wtune_probe_plan.measurement_from_steps_file(
            steps_path, stepra, p.at, p.reason, p.source, first_global_step=g0))
    return measurements, job_id


def rebuild_segment_timeout_min(case_dir, spinup_min, ranks, omp=1, safety=3.0, floor_min=30):
    """wrf.exe watchdog for one rebuild segment (spin-up + coldstart's measure window), from the
    uncalibrated cell-step cost model x `safety`, never below `floor_min`. The coldstart default
    of 15 min killed every maria1km segment (106 d01 steps x ~9.6 s = ~17 min, 2026-09-27); the
    watchdog only guards against hangs, so it errs long."""
    try:
        nl = wtune_predict.parse_namelist(os.path.join(case_dir, "namelist.input"))
        _n, _run_s, dt, _note = wtune_predict.get_d01_steps(nl)
        stepra = wtune_predict.steps_per_radiation_call(
            float(wtune_predict.g(nl, "physics", "radt", None)), dt)
        max_dom = int(wtune_predict.g(nl, "domains", "max_dom", 1))
        cal = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
        s_step = (cal["fallback_s_per_cell_step_ns"] * 1e-9
                  * wtune_predict.nested_equivalent_cells(nl, max_dom) / max(ranks * omp, 1))
        steps = (spinup_min * 60.0) / dt + max(30, 2 * stepra)
        est_s = steps * s_step * 1.5 + 120.0          # radiation steps + init
        return max(floor_min, int(math.ceil(safety * est_s / 60.0)))
    except Exception:
        return 90


def run_rebuild_points(ctx, points, best_config, nodes, case_dir, out_dir, run_start,
                       spinup_min=60, grib_dir=None, date_shift_to=None, forcing=None,
                       measure_steps=None):
    """Measures every rebuild-sourced ProbePoint of `points` in ONE coldstart_sample job: plans
    (wtune_probe_plan.plan_rebuild_segments -- one build_plan(times=[...]) call covering every
    point, pinning each one's forcing-grid START T recovered from its already-resolved measured
    point M -- see that function's docstring), fetches forcing (coldstart_sample.resolve_and_
    fetch_forcing, real network), stages (coldstart_sample.stage_plan, writes run.sbatch),
    submits it (a plain `sbatch run.sbatch` -- the generated script is self-contained: #SBATCH
    --exclude=head/-N/--exclusive/--time are already in its own header, see render_run_sbatch),
    waits, then reads each window's own s_k/n_rad/r_local via coldstart_sample.score_windows_
    from_rsl (data-driven radiation-phase detection -- reliable here because coldstart's own
    window sizing, measure_steps=max(30,2*stepra), is long enough for it, unlike the RANKING
    probe's shorter window -- see measurement_from_steps_file's docstring).

    `spinup_min` MUST equal the spinup (in minutes) plan_points was called with -- plan_
    rebuild_segments uses it to recover each point's own T from its already-resolved M (see that
    function's docstring); a mismatch here would silently re-derive the WRONG start time.
    Each returned PointMeasurement's `.at` is score_windows_from_rsl's own measured-part MIDPOINT
    (recovered from its `anchors` return value, run_start-relative), NOT the window's raw 't'
    field (which is T, the pinned forcing-grid START -- using that directly would misplace every
    rebuild point by ~spinup_min minutes in combine()'s interpolation). Returns
    [PointMeasurement, ...] in `points` order. Raises RuntimeError/ProbePlanError on any
    infrastructure failure -- never silently drops a point."""
    if not points:
        return []
    missing_env = [v for v in ("WPS_DIR", "WRF_TREE") if not os.environ.get(v)]
    if missing_env:
        # job 1236: staged without these, the window dirs had no geogrid/wrf executables and the
        # job died in 3 s. Fail before submitting instead.
        raise RuntimeError("run_rebuild_points: %s not set -- source the site env (e.g. "
                           "~/wtune-scc/env.sh) before running the driver" % ", ".join(missing_env))
    config = dict(ranks=best_config["ranks"], omp=best_config["omp"], nx=best_config["nx"],
                  ny=best_config["ny"], numtiles=best_config.get("numtiles", 1), nodes=nodes)
    # One plan dir and one job PER POINT, all submitted before waiting: the segments are
    # independent, so the wall time is that of the slowest segment rather than their sum
    # (decision 0054). They cannot share one job's $PLAN/wps (ungrib/metgrid of different
    # windows would overwrite each other's FILE:/met_em files). Forcing is fetched once into a
    # shared cache; geogrid and the Thompson tables come from the run.sbatch caches.
    seg_timeout_min = rebuild_segment_timeout_min(case_dir, spinup_min, config["ranks"],
                                                  config.get("omp", 1))

    def _prepare_and_submit(i, p):
        pdir = os.path.join(out_dir, "pt_%d" % i)
        plan = wtune_probe_plan.plan_rebuild_segments(
            case_dir, [p], config, pdir, spinup_min=spinup_min, date_shift_to=date_shift_to,
            grib_dir=grib_dir, forcing=forcing, measure_steps=measure_steps)
        # The point's own PLAN/grib: run.sbatch links "$PLAN/grib/<manifest name>", so a cache
        # shared across point dirs left every GRIB link dangling (ungrib "unable to open
        # GRIBFILE.AAA", GFS-chain zhengzhou 2026-09-27; ERA5 has no GRIB and was unaffected).
        total_bytes, _forcing = coldstart.resolve_and_fetch_forcing(
            plan, pdir, os.path.join(pdir, "grib"), grib_dir=grib_dir)
        print("C3 rebuild point %d (%s): fetched %.1f MB forcing"
              % (i, p.at.isoformat(), total_bytes / 1e6), flush=True)
        sbatch_path = coldstart.stage_plan(pdir, site="scc", per_window_min=seg_timeout_min)
        job_id = sweep.submit_job("sbatch %s" % sbatch_path, sbatch_runner=ctx.sbatch_runner)
        print("C3 rebuild job submitted: %s (%s)" % (job_id, sbatch_path), flush=True)
        plan["_plan_dir"] = pdir
        return plan, job_id

    # Fetch/stage/submit every point concurrently: each fetch is network-bound (ERA5 ~3.5 min
    # per point), so doing them in sequence delayed the last submission by the sum of all fetches.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(points))) as pool:
        done = list(pool.map(lambda ip: _prepare_and_submit(*ip), enumerate(points)))
    plans = [d[0] for d in done]
    job_ids = [d[1] for d in done]
    ctx.owned_jobs.extend(job_ids)
    states = wait_for_jobs(ctx, job_ids, "c3-rebuild")
    bad = {j: states.get(j, ("UNKNOWN", ""))[0] for j in job_ids
           if states.get(j, ("UNKNOWN", ""))[0] != "COMPLETED"}
    if bad:
        raise RuntimeError("C3 rebuild job(s) %s not COMPLETED -- infrastructure failure, stop "
                           "and diagnose read-only" % bad)
    sample_points, _geom_src = wtune_predict.read_wrfinput_sample_points(case_dir)
    calibration = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
    measurements = []
    for plan in plans:
        windows_out, _pooled, anchors = coldstart.score_windows_from_rsl(plan, sample_points,
                                                                          calibration)
        for w, (anchor_s, _s_k) in zip(windows_out, anchors):
            at = run_start + datetime.timedelta(seconds=anchor_s)
            measurements.append(wtune_probe_plan.PointMeasurement(
                at=at, reason=w["reason"], source="rebuild", s_normal=w["s_k"],
                n_rad=w["n_rad"], r=w["r_local"]))
    return measurements, ",".join(job_ids)


def run_c3_prediction(ctx, best_config, nodes, out_dir, sbatch_script=None, campaign_dir=None,
                      grib_dir=None):
    """Full C3 measurement + prediction for `best_config` (the already-known winner): plan_points
    for its own timeline, split by source, measure trajectory/restart points (run_trajectory_
    points, one job) and rebuild points (run_rebuild_points, one job) -- decision 0054, one
    allocation PER SOURCE TYPE (trajectory/restart share the STRICT wrf.exe-only loop; rebuild
    needs the full WPS+real.exe+wrf.exe chain -- see the module comment above run_trajectory_
    points for why these are not merged into one job) -- then combine()s every measurement into
    one whole-run prediction. Returns (result_dict, points, notes, job_ids). Raises on any
    infrastructure failure from either run_* function; a point plan_points itself could not
    assign a source to is named in `notes` (never silently dropped, never fails the whole
    prediction by itself)."""
    nl = wtune_predict.parse_namelist(ctx.namelist_path)
    steps_n, run_s, dt, _note = wtune_predict.get_d01_steps(nl)
    run_start = wtune_predict.get_time_control_datetime(nl, "start")
    run_end = run_start + datetime.timedelta(seconds=run_s)
    radt = wtune_predict.g(nl, "physics", "radt", None)
    stepra = wtune_predict.steps_per_radiation_call(float(radt), dt)
    spinup_s = float(ctx.probe_cfg.get("spinup_min") or 0) * 60.0
    case_dir = ctx.case_json["prep_dir"]
    cost_series = _cost_series_for_ctx(ctx, run_start, run_end, case_dir, grib_dir)
    points, notes = wtune_probe_plan.plan_points(
        nl, run_start, run_end, dt, stepra, spinup_s, ctx.probe_cfg, case_dir=case_dir,
        indicators=cost_series)

    traj_points = [p for p in points if p.source in ("trajectory", "restart")]
    rebuild_points = [p for p in points if p.source == "rebuild"]

    job_ids = []
    measurements = []
    if traj_points:
        out_traj = os.path.join(out_dir, "c3_trajectory")
        m, jid = run_trajectory_points(ctx, traj_points, best_config, nodes, run_start, stepra,
                                        out_traj, sbatch_script=sbatch_script,
                                        campaign_dir=campaign_dir)
        measurements.extend(m)
        job_ids.append(jid)
    if rebuild_points:
        out_rebuild = os.path.join(out_dir, "c3_rebuild")
        spinup_min = ctx.probe_cfg.get("spinup_min") or 60
        m, jid = run_rebuild_points(ctx, rebuild_points, best_config, nodes, case_dir, out_rebuild,
                                     run_start, spinup_min=spinup_min, grib_dir=grib_dir)
        measurements.extend(m)
        job_ids.append(jid)

    calibration = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
    if ctx.prediction_cfg.get("drift_band_pct") is not None:
        calibration = dict(calibration, drift_band_pct=ctx.prediction_cfg["drift_band_pct"])
    sample_points, _geom_src = wtune_predict.read_wrfinput_sample_points(case_dir)
    result = wtune_probe_plan.combine(
        measurements, run_start, run_end, dt, stepra, calibration=calibration,
        sample_points=sample_points, case_dir=case_dir, nl=nl, ranks=best_config.get("ranks"),
        nodes=nodes, cost_series=cost_series)
    return result, points, notes, job_ids


def build_full_run_prediction(ctx, best_config, nodes, all_results, confirm_results,
                               probe_measurements=None):
    """probe_measurements (2026-09-26 multi-point redesign): [wtune_probe_plan.PointMeasurement,
    ...] for the winner, when the PREDICTION layer's planned points (see _probe_plan_preview)
    were actually measured -- combine()s them into the whole-run prediction instead of today's
    single t0-probe wtune_predict.build_prediction call. No caller populates this yet (measuring
    multiple points needs the same submission wiring Layer A's probe.ranking_start=spinup is
    waiting on -- see ctx.ranking_start_note / this task's report): the single-probe path below
    is what every real run takes today, and SAYS so via the appended probe-plan preview note."""
    if wtune_predict is None or best_config is None:
        return ("full-run prediction: run `python3 wtune.py predict %s --nodes %d --config "
                "'%s'`" % (ctx.case_json["prep_dir"], nodes,
                           json.dumps(dict(ranks=best_config["ranks"], omp=best_config["omp"],
                                            nx=best_config["nx"], ny=best_config["ny"],
                                            numtiles=best_config["numtiles"]))
                           if best_config else "..."))
    if probe_measurements and wtune_probe_plan is not None:
        try:
            nl = wtune_predict.parse_namelist(ctx.namelist_path)
            steps_n, run_s, dt, _note = wtune_predict.get_d01_steps(nl)
            run_start = wtune_predict.get_time_control_datetime(nl, "start")
            run_end = run_start + datetime.timedelta(seconds=run_s)
            radt = wtune_predict.g(nl, "physics", "radt", None)
            stepra = wtune_predict.steps_per_radiation_call(float(radt), dt)
            calibration = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
            if ctx.prediction_cfg.get("drift_band_pct") is not None:
                calibration = dict(calibration, drift_band_pct=ctx.prediction_cfg["drift_band_pct"])
            sample_points, _src = wtune_predict.read_wrfinput_sample_points(
                ctx.case_json["prep_dir"])
            result = wtune_probe_plan.combine(
                probe_measurements, run_start, run_end, dt, stepra, calibration=calibration,
                sample_points=sample_points, case_dir=ctx.case_json["prep_dir"], nl=nl,
                ranks=best_config.get("ranks"), nodes=nodes,
                cost_series=_cost_series_for_ctx(ctx, run_start, run_end,
                                                 ctx.case_json["prep_dir"]))
            lines = ["=== full-run prediction (MULTI-POINT, wtune_probe_plan.combine, "
                     "%d measured point(s)) ===" % len(probe_measurements),
                     "compute_total: %s   init: %.1fs   TOTAL: %s (range %s - %s, "
                     "+-%.1f%% drift)"
                     % (_fmt_hms(result["compute_total_s"]), result["init_s"],
                        _fmt_hms(result["total_s"]), _fmt_hms(result["range_low_s"]),
                        _fmt_hms(result["range_high_s"]), result["drift_pct"]),
                     "r_rad: night=%.3f day=%.3f source=%s"
                     % (result["r_night"], result["r_day"], result["r_source"])]
            if result.get("cost_indicators") and "error" not in result["cost_indicators"]:
                ci = result["cost_indicators"]
                lines.append("cost-indicator totals (compute+init): interpolated %s, stratified "
                             "%s, regression %s (R2 %s), spread %s%%; sources %s"
                             % (_fmt_hms(result["total_s"]),
                                _fmt_hms(result["stratified_total_s"])
                                if result.get("stratified_total_s") else "-",
                                _fmt_hms(result["regression_total_s"])
                                if result.get("regression_total_s") else "-",
                                ("%.3f" % result["regression_r2"])
                                if result.get("regression_r2") is not None else "-",
                                ("%.1f" % ci["spread_pct"]) if ci.get("spread_pct") is not None
                                else "-", ci.get("sources")))
            for p in result["points"]:
                lines.append("  point %s reason=%s source=%s s_normal=%.4f n_rad=%d r=%s"
                              % (p["at"].isoformat(), p["reason"], p["source"], p["s_normal"],
                                 p["n_rad"], ("%.3f" % p["r"]) if p["r"] is not None else "-"))
            if result.get("init_note"):
                lines.append("NOTE: " + result["init_note"])
            return "\n".join(lines)
        except Exception as e:
            return ("full-run prediction (multi-point) FAILED (%s: %s); falling back would "
                     "need a re-run -- run `python3 wtune.py predict ...` manually"
                     % (type(e).__name__, e))
    try:
        # Bug fix: this used to pass ledger_paths=None, so wtune_predict's own resolve_probe()
        # never saw this run's measurements and always fell back to source=no_probe_fallback
        # (+-50%) even right after measuring the winning config directly. Feed it back in through
        # wtune_predict's own public ledger_paths parameter instead.
        probe_path = write_probe_ledger(ctx, all_results, confirm_results)
        calibration_path = str(wtune_predict.DEFAULT_CALIBRATION)
        drift_note = None
        if ctx.prediction_cfg.get("drift_band_pct") is not None:
            calibration_path, drift_note = _calibration_with_drift_override(
                ctx, calibration_path, ctx.prediction_cfg["drift_band_pct"])
        entry = (confirm_results or {}).get(best_config.get("id")) or \
            (all_results or {}).get(best_config.get("id"))
        rsl_paths = _rsl_paths_for_entry(ctx, entry)
        pred = wtune_predict.build_prediction(
            namelist_path=ctx.namelist_path, case_dir=ctx.case_json["prep_dir"], nodes=nodes,
            config=dict(ranks=best_config["ranks"], omp=best_config["omp"], nx=best_config["nx"],
                        ny=best_config["ny"], numtiles=best_config["numtiles"]),
            ledger_paths=[probe_path] if probe_path else None,
            calibration_path=calibration_path, io_form_override=None,
            io_target=ctx.prediction_cfg.get("io_target") or "nfs", case_name=ctx.case,
            rsl_paths=rsl_paths or None, include_io=False)
        text = wtune_predict.render_text(pred)
        if drift_note:
            text += "\n" + drift_note
        if ctx.prediction_cfg.get("measure_r_rad"):
            # Honesty requirement (see wtune_config.py's schema doc): report whether r_rad was
            # actually derived from this run's own probe measurements (now possible when this
            # candidate's own rsl.error.0000 logs survived -- see _rsl_paths_for_entry) rather
            # than silently using the fixed calibration default with no indication either way.
            r_rad_source = pred["compute"]["r_rad_source"]
            if r_rad_source in ("fit_day_night", "ratio_from_night", "ratio_from_day"):
                text += ("\nNOTE: prediction.measure_r_rad requested -- derived from this run's "
                          "own probe rsl.error.0000 logs (source=%s)." % r_rad_source)
            else:
                text += ("\nNOTE: prediction.measure_r_rad requested (derive the radiation-step "
                          "cost multiplier r_rad from this run's own probe measurements) -- no "
                          "usable rsl.error.0000 probe log for this candidate's own runs "
                          "(source=%s); deriving r_rad from measurements is NOT YET IMPLEMENTED "
                          "without one -- used the fixed calibration default instead."
                          % r_rad_source)
        preview = _probe_plan_preview(ctx, nodes, best_config)
        if preview:
            preview_points, preview_notes, _rs, _re, _dt, _stepra = preview
            text += ("\nNOTE: multi-point probe planning (wtune_probe_plan.plan_points) would "
                     "place %d point(s) for this winner (%s) -- none measured for this run, so "
                     "the single t0-probe prediction above was used; see this task's report for "
                     "the Layer A/B wiring status."
                     % (len(preview_points),
                        ", ".join("%s@%s" % (p.reason, p.at.strftime("%m-%d_%H:%M"))
                                  for p in preview_points)))
            for n in preview_notes:
                text += "\nNOTE: " + n
        return text
    except Exception as e:
        return ("full-run prediction FAILED (%s: %s); run `python3 wtune.py predict ...` manually"
                 % (type(e).__name__, e))


def _assert_report_is_strict_only(ctx, param_table):
    """decision 0056 guard: param_table's namelist_lines are the final recommended
    configuration `wtune tune` hands to the user (build_report's `best`) -- verify every
    namelist key it changes from the case's own namelist classifies as STRICT
    (wtune_compliance.assert_strict_only). Best-effort like every other wtune_predict use
    in this module (see the top-of-file try/except import): skipped, not raised, when
    wtune_predict is not importable or this ctx's namelist_path cannot be read (e.g. a
    synthetic test Ctx with no real namelist on disk) -- there is nothing to diff then."""
    if wtune_predict is None:
        return
    try:
        orig_nested = wtune_predict.parse_namelist(ctx.namelist_path)
    except OSError:
        return
    orig_flat = {k: v for grp in orig_nested.values() for k, v in grp.items()}
    final_flat = dict(orig_flat)
    # every "key=value," line the report hands out, not just the keys we expect today
    for line in param_table.get('namelist_lines', []):
        key, _, val = line.partition('=')
        new = [v.strip() for v in val.strip().rstrip(',').split(',')]
        if val and [str(v).strip() for v in orig_flat.get(key.strip()) or []] != new:
            final_flat[key.strip()] = new
    wc.assert_strict_only(orig_flat, final_flat)
    # WRF extensions (chem/hydro/WRFDA/moving nest): group-level byte-for-byte guard on top of the
    # per-key classification (extension-owned groups such as &chem or &wrfvar* may not change at all)
    import wtune_extensions
    final_nested = {g: dict(v) for g, v in orig_nested.items()}
    for g, grp in final_nested.items():
        for k in grp:
            if k in final_flat and final_flat[k] != orig_flat.get(k):
                grp[k] = final_flat[k]
    bad = wtune_extensions.extension_keys_changed(orig_nested, final_nested)
    if bad:
        raise wc.ComplianceError(
            "extension parameter(s) changed in the final recommended configuration: "
            + ", ".join("&%s %s" % gk for gk in bad))


def build_report(ctx, nodes, nodes_note, all_results, confirm_results, anchor_cid,
                 full_prediction=True):
    ranked_source = confirm_results if confirm_results else all_results
    predicted = predict_candidates(ctx, nodes, ranked_source)
    ranked = rank_predicted_entries(predicted, anchor_cid)
    anchor_median = (all_results.get(anchor_cid) or {}).get("median")
    anchor_pred_total_s = (predicted.get(anchor_cid) or {}).get("pred_total_s")

    table = []
    for r in ranked:
        cfg = r["config"] or {}
        speedup = (anchor_median / r["median"]) if anchor_median else None
        pred_total_s = r.get("pred_total_s")
        speedup_pred = ((anchor_pred_total_s / pred_total_s)
                         if anchor_pred_total_s and pred_total_s else None)
        table.append(dict(
            id=r["cid"], ranks=cfg.get("ranks"), omp=cfg.get("omp"),
            nproc_x=cfg.get("nx"), nproc_y=cfg.get("ny"), numtiles=cfg.get("numtiles"),
            dist=cfg.get("dist"), median_s_per_step=r["median"], range_s_per_step=[r["lo"], r["hi"]],
            n_measurements=r["n"], speedup_vs_anchor=speedup,
            # confirmed only when ALL ctx.confirm_reps replicates of THIS config were measured: a
            # confirm job cancelled at the budget deadline can leave 1 rep (or none) per config
            confirmed=bool(confirm_results and r["cid"] in confirm_results
                           and confirm_results[r["cid"]].get("n", 0) >= ctx.confirm_reps),
            predicted_total_s=pred_total_s,
            predicted_total_hms=_fmt_hms(pred_total_s) if pred_total_s is not None else None,
            predicted_range_s=([r["pred_low_s"], r["pred_high_s"]]
                                if pred_total_s is not None else None),
            pred_source=r.get("pred_source"), pred_error=r.get("pred_error"),
            speedup_vs_anchor_predicted=speedup_pred,
        ))

    best = table[0] if table else None
    best_config = ranked[0]["config"] if ranked else None
    # Always our own local_param_table, not wtune_predict.build_param_table: that function (owned
    # by another agent) hardcodes dist="cyclic" and never adds --rank-by node, so it would report
    # a wrong launch line for a dist=block winner. wtune_predict is used only for the full-run
    # timeline hook below, where the exact launch line does not matter.
    param_table = local_param_table(best_config, nodes) if best_config is not None else None
    if param_table is not None:
        _assert_report_is_strict_only(ctx, param_table)

    return dict(
        case=ctx.case, nodes=nodes, nodes_note=nodes_note, cores_per_node=ctx.cores_per_node,
        budget_minutes=(ctx.budget_s / 60.0 if math.isfinite(ctx.budget_s) else None),
        elapsed_minutes=ctx.elapsed() / 60.0,
        job_ids=list(ctx.owned_jobs), cancelled_job_ids=list(ctx.cancelled_jobs),
        anchor_id=anchor_cid, ranking=table, best=best, param_table=param_table,
        full_run_prediction=(build_full_run_prediction(ctx, best_config, nodes, all_results,
                                                         confirm_results)
                             if full_prediction else None),
        ranking_start_note=ctx.ranking_start_note,
    )


def render_report_text(report):
    lines = []
    lines.append("=== wtune_tune: %s ===" % report["case"])
    lines.append("nodes=%d (%s)" % (report["nodes"], report["nodes_note"]))
    lines.append("budget=%s  elapsed=%.1f min  jobs=%s"
                  % (("%.1f min" % report["budget_minutes"]) if report["budget_minutes"] is not None
                     else "no cap", report["elapsed_minutes"],
                     ",".join(report["job_ids"]) or "-"))
    if report["cancelled_job_ids"]:
        lines.append("cancelled (budget deadline): %s" % ",".join(report["cancelled_job_ids"]))
    lines.append("")
    # predicted_total (whole-run wall time: compute + init, no I/O -- see predict_candidates) is
    # the ranking key and comes first; median_s/range/speedup are still the raw per-step
    # measurement this run made, kept for comparison.
    lines.append("%-10s %12s %6s %4s %5s %5s %5s %7s %10s %14s %7s %8s %4s %-11s %s"
                  % ("id", "pred_total", "ranks", "omp", "nx", "ny", "tiles", "dist", "median_s",
                     "range_s", "speedup", "pred_spd", "n", "pred_source", "confirmed"))
    for row in report["ranking"]:
        rng = "%.4f-%.4f" % tuple(row["range_s_per_step"]) if row["range_s_per_step"] else "-"
        spd = ("%.3fx" % row["speedup_vs_anchor"]) if row["speedup_vs_anchor"] else "-"
        pred_spd = ("%.3fx" % row["speedup_vs_anchor_predicted"]
                    if row["speedup_vs_anchor_predicted"] else "-")
        pred_hms = row["predicted_total_hms"] or "FAILED"
        lines.append("%-10s %12s %6s %4s %5s %5s %5s %7s %10.4f %14s %7s %8s %4s %-11s %s"
                      % (row["id"], pred_hms, row["ranks"], row["omp"], row["nproc_x"],
                         row["nproc_y"], row["numtiles"], row["dist"], row["median_s_per_step"],
                         rng, spd, pred_spd, row["n_measurements"], row["pred_source"] or "-",
                         row["confirmed"]))
    failed = [row for row in report["ranking"] if row.get("pred_error")]
    if failed:
        lines.append("")
        for row in failed:
            lines.append("PREDICTION FAILED for %s: %s -- ranked by median s/step instead"
                          % (row["id"], row["pred_error"]))
    if report["best"]:
        lines.append("")
        lines.append("BEST: %s" % report["best"]["id"])
        pt = report["param_table"]
        lines.append("  namelist: " + " ".join(pt["namelist_lines"]))
        lines.append("  launch:   " + pt["launch_line"])
        lines.append("  env:      " + " ".join("%s=%s" % kv for kv in pt["env"].items())
                      + "  " + pt.get("ulimit", "ulimit -s unlimited"))
        if pt.get("note"):
            lines.append("  NOTE:     " + pt["note"])
    lines.append("")
    lines.append(report["full_run_prediction"])
    if report.get("ranking_start_note"):
        lines.append("")
        lines.append("NOTE: " + report["ranking_start_note"])
    if report.get("node_selection"):
        lines.append(render_node_selection_text(report["node_selection"]))
    if report.get("node_health"):
        lines.append(render_node_health_text(report["node_health"]))
    if report.get("pipeline"):
        lines.append(render_pipeline_text(report["pipeline"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# decision 0055: --rank-by predicted. Candidates are ranked by PREDICTED WHOLE-RUN compute+init
# built from measurements at several timeline points; each point is rebuilt from ERA5 once
# (coldstart real_only: WPS + real.exe, no wrf.exe) and turned into its own strict_campaign case
# entry, spun up ONCE inside each ranking job (strict_campaign RANKING_START=spinup), and every
# candidate of that job starts from the same shared restart. ANCHOR is measured first and again
# last (ANCHOREND, same decomposition) in every such job so node drift inside the allocation is
# measured, not assumed. Nothing here runs unless ctx.rank_by == "predicted".
# ---------------------------------------------------------------------------
ANCHOR_END_ID = "ANCHOREND"
RANK_BY_CHOICES = ("t0", "predicted")
STAGE1_CHOICES = ("t0", "pstar")         # predicted mode's stage 1 (see run_predicted)
PREDICTED_FORCING = "era5_arco"         # wtune_rebuild_predict.ERA5_PIN (1 h cadence, with cloud)
PREDICTED_SPINUP_MIN = 20              # ERA5 spin-up validated in rebuild phase 3 (2026-09-27)
ANCHOR_DRIFT_MAX = 0.01                # |s(ANCHOREND)/s(ANCHOR) - 1| allowed inside one job
TOP_K_STAGE2 = 3


PREDICTED_MAX_POINTS = 3               # predicted-mode default (probe.max_points / --max-points win)
MAX_WALL_FACTOR = 3.0                  # predicted-mode hard wall-clock cap = 3x the budget
REBUILD_PREP_EST_S = 480.0             # ERA5 fetch + WPS + real.exe, 1-node jobs in parallel
                                        # (conus25 e2e attempt 2, 2026-09-27: 7 min)
RANKING_R_RAD = 3.93                   # radiation-step cost / normal step: calibration's day
                                        # level (r_rad_default_day) -- P* is the most daylit point
RESTART_READ_S_BY_IO = {14: 30.0, 2: 120.0}   # per-run restart read on top of startup_s:
                                        # ADIOS2 BP (parallel) vs serial netCDF (conus25 11.5 GB,
                                        # job 1231: 120 s). Zeroed once the per-run overhead is
                                        # MEASURED from a ranking job (it already includes it).


def rad_factor(steps, stepra, r_rad=RANKING_R_RAD):
    """Mean cost of `steps` consecutive steps relative to a normal step: ceil(steps/stepra)
    radiation steps cost r_rad each. 1.0 without a stepra. (Replaces the earlier blanket x1.5,
    which made a conus25 82-step run look ~1.36x its real cost.)"""
    if not stepra or steps <= 0:
        return 1.0
    return 1.0 + math.ceil(float(steps) / stepra) * (r_rad - 1.0) / float(steps)


def ranking_cost_model(startup_s, restart_read_s, steps, spin_steps, stepra, s_step):
    """The ONE running-time model every predicted-mode estimate uses (stage-1 jobs, the stage-2
    reserve, stage-2 timeouts, the feasibility pre-check):
      per_run_s = startup_s + restart_read_s + steps * s_step * rad_factor(steps)
      spin_s    = startup_s + spin_steps * s_step * rad_factor(spin_steps)   (reads wrfinput)
    Returned with its inputs so every log line can show what was assumed."""
    f_run, f_spin = rad_factor(steps, stepra), rad_factor(spin_steps, stepra)
    return dict(per_run_s=startup_s + restart_read_s + steps * s_step * f_run,
                spin_s=startup_s + spin_steps * s_step * f_spin, s_step=s_step, steps=steps,
                spin_steps=spin_steps, stepra=stepra, rad_factor_run=f_run,
                rad_factor_spin=f_spin, r_rad=RANKING_R_RAD, startup_s=startup_s,
                restart_read_s=restart_read_s)


def describe_cost_model(m):
    return ("per-run %.0fs = startup %.0fs + restart read %.0fs + %d steps x %.3fs x rad factor "
            "%.3f (r_rad %.2f every %s steps); spin-up %.0fs (%d steps, rad factor %.3f)"
            % (m["per_run_s"], m["startup_s"], m["restart_read_s"], m["steps"], m["s_step"],
               m["rad_factor_run"], m["r_rad"], m["stepra"] or "-", m["spin_s"], m["spin_steps"],
               m["rad_factor_spin"]))


def _restart_read_s(ctx):
    return ctx.restart_read_s if ctx.restart_read_s is not None else \
        RESTART_READ_S_BY_IO.get(int(ctx.restart_io_form), 120.0)


def estimate_stage2_s(n_points, n_configs, steps, spinup_min, dt, s_step, slots, startup_s,
                      restart_read_s=30.0, stepra=None):
    """Expected RUNNING time of stage 2: one job per point, each = shared spin-up + n_configs
    runs (ranking_cost_model). The point jobs run in parallel, but only `slots`
    (= floor(max_nodes/nodes)) allocations at once: ceil(n_points/slots) sequential waves."""
    m = ranking_cost_model(startup_s, restart_read_s, steps, spinup_min * 60.0 / dt, stepra,
                           s_step)
    waves = int(math.ceil(float(n_points) / max(1, int(slots))))
    return waves * (m["spin_s"] + n_configs * m["per_run_s"])


def _reserve_s_step(ctx):
    """s/step the stage-2 reserve is sized with: the median of stage 1's own per-job anchor
    normalizers once any exist (measured at P*, this allocation shape), else ctx.s_prior."""
    vals = [j["s_anchor_job"] for j in ctx.stage1_jobs.values() if j.get("s_anchor_job")]
    return statistics.median(vals) if vals else ctx.s_prior


def stage2_concurrency(ctx):
    """How many stage-2 point jobs the reserve assumes run at once (conus25-pred3: the reserve
    assumed floor(max_nodes/nodes)=3, the busy cluster delivered 1 and stage 2 overran): the
    MAXIMUM number of this run's stage-1 ranking jobs OBSERVED running at the same time
    (ctx.stage1_max_running, from _wait_running_budget's polls), at least 1 -- so 1, the
    conservative value, before any stage-1 job has run -- capped at floor(max_nodes/nodes)."""
    cap = compute_slots(ctx.max_nodes, ctx.nodes)
    return max(1, min(cap, int(ctx.stage1_max_running or 0) or 1))


def _stage2_estimate(ctx, n_configs, s_step):
    r = ctx.reserve
    return estimate_stage2_s(r["n_points"], n_configs, r["steps"], r["spinup_min"], r["dt"],
                             s_step, stage2_concurrency(ctx), ctx.startup_s, _restart_read_s(ctx),
                             stepra=r.get("stepra"))


def _reserve_k(ctx, s_step):
    """Candidates the stage-2 reserve holds room for. TOP_K_STAGE2 once any stage-1 job has
    reported (stage 1 then only continues if the full stage 2 still fits). Before that, the
    largest k in TOP_K_STAGE2..1 that still leaves room for ONE minimal stage-1 job (spin-up +
    ANCHOR + 1 + ANCHOREND) -- with the conservative concurrency of 1 the full top-3 reserve
    would otherwise block stage 1 entirely; fit_stage2 trims stage 2 to what is left later."""
    if ctx.stage1_jobs:
        return TOP_K_STAGE2
    # one minimal stage-1 job: pstar = spin-up + 3 ranking runs; t0 = 3 plain t0-STEPS runs
    min_job = _job_time_est(ctx, 3, 1, s_step)["total_run_s"]
    for k in range(TOP_K_STAGE2, 0, -1):
        if ctx.elapsed() + min_job + ctx.stage1_extra_reserve_s \
                + _stage2_estimate(ctx, k + 2, s_step) <= ctx.budget_s:
            return k
    return 1


def stage2_reserve_s(ctx):
    """Running time stage 1 must leave for stage 2 (predicted mode; 0.0 otherwise, so every t0
    budget expression is unchanged): estimate_stage2_s for _reserve_k candidates + ANCHOR/
    ANCHOREND at every point, ceil(points / stage2_concurrency) waves, at _reserve_s_step."""
    r = ctx.reserve
    if not _stage1_active(ctx) or not r:
        return 0.0
    s_step = _reserve_s_step(ctx)
    if not s_step:
        return 0.0
    return _stage2_estimate(ctx, _reserve_k(ctx, s_step) + 2, s_step)


def stage1_reserve_s(ctx):
    """What every stage-1 decision holds back from --minutes: the stage-2 reserve (+
    stage1_extra_reserve_s) when ctx.reserve_in_stage1 (stage1=pstar: --minutes covers both
    stages), 0.0 in stage1=t0 (--minutes is the stage-1 search budget only, stage 2 is a
    separate announced cost) and outside predicted-mode stage 1."""
    if not _stage1_active(ctx) or not ctx.reserve_in_stage1:
        return 0.0
    return stage2_reserve_s(ctx) + ctx.stage1_extra_reserve_s


def _stage1_active(ctx):
    """True during predicted-mode stage 1, in either stage1 mode."""
    return bool(ctx.ranking or ctx.stage1)


def _stage1_anchor(ctx):
    if ctx.ranking:
        return ctx.ranking["anchor"]
    return (ctx.stage1 or {}).get("anchor")


def describe_stage2_reserve(ctx):
    """One log line: how the stage-2 reserve was sized (so it can be judged against reality)."""
    r = ctx.reserve
    s_step = _reserve_s_step(ctx)
    if not _stage1_active(ctx) or not r or not s_step:
        return "stage-2 reserve: none"
    m = ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), r["steps"],
                           r["spinup_min"] * 60.0 / r["dt"], r.get("stepra"), s_step)
    conc = stage2_concurrency(ctx)
    waves = int(math.ceil(float(r["n_points"]) / conc))
    return ("stage-2 %s %s = %d wave(s) [%d point job(s), %d concurrent: observed max %d "
            "stage-1 job(s) running at once, cap floor(max_nodes/nodes)=%d] x (spin-up + %d "
            "runs); %s"
            % ("reserve" if ctx.reserve_in_stage1 else
               "estimate (NOT held back from --minutes, stage1=t0)",
               _fmt_hms(stage2_reserve_s(ctx)), waves, r["n_points"], conc,
               ctx.stage1_max_running, compute_slots(ctx.max_nodes, ctx.nodes),
               _reserve_k(ctx, s_step) + 2, describe_cost_model(m)))


def _stage1_may_start(ctx, job_run_s):
    """(ok, reason): a stage-1 job/round may start only if elapsed_running + its own expected
    running time + the stage-2 reserve still fits the budget. Always ok outside predicted mode
    (t0 keeps its own checks only)."""
    if not _stage1_active(ctx):
        return True, None
    reserve = stage1_reserve_s(ctx)
    if ctx.elapsed() + job_run_s + reserve > ctx.budget_s:
        return False, ("elapsed %s + this job %s + stage-2 reserve %s > budget %s"
                       % (_fmt_hms(ctx.elapsed()), _fmt_hms(job_run_s), _fmt_hms(reserve),
                          _fmt_hms(ctx.budget_s)))
    return True, None


def _stage1_cutoff(ctx):
    """pending_cutoff for stage-1 waits in predicted mode (None otherwise): true once
    elapsed_running + the stage-2 reserve reaches the budget."""
    if not _stage1_active(ctx):
        return None
    return lambda: ctx.elapsed() + stage1_reserve_s(ctx) >= ctx.budget_s


def _ledger_case(ctx):
    """Case id ranking-job ledgers/RUNDIRs are named after: the rebuilt point's own id in
    predicted mode (ledger_<id>_<job>.csv), ctx.case otherwise (unchanged)."""
    return ctx.ranking["case_id"] if ctx.ranking else ctx.case


def anchor_end_of(anchor):
    """ANCHOREND: the ANCHOR decomposition re-measured as the LAST run of a ranking job
    (config_source "anchor_end") -- a drift probe, excluded from winner selection."""
    return dict(anchor, id=ANCHOR_END_ID, config_source="anchor_end")


def wrap_with_anchors(configs, anchor):
    """[ANCHOR] + configs (minus any ANCHOR/ANCHOREND already present, order kept) +
    [ANCHOREND]. Idempotent. strict_campaign's RANKING_START=spinup needs the ANCHOR row (its
    decomposition writes the shared restart) and, with REPS=1, build_order keeps file order, so
    ANCHOR runs first and ANCHOREND last."""
    if anchor is None:
        raise ValueError("wrap_with_anchors: no ANCHOR config -- RANKING_START=spinup needs one")
    rest = [c for c in configs if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    return [dict(anchor, id="ANCHOR", config_source="anchor")] + rest + [anchor_end_of(anchor)]


def _ranking_wrap(ctx, configs):
    """wrap_with_anchors in predicted mode, `configs` unchanged (same list object) otherwise."""
    anchor = _stage1_anchor(ctx) if _stage1_active(ctx) else None
    return wrap_with_anchors(configs, anchor) if anchor else configs


def ranking_context(case_id, cases_json, spinup_min, steps, anchor, restart_io_form=14,
                    spinup_timeout_s=None, strict_restart=1):
    """The ranking context dict submit_configs_job/measure_candidates_at_points consume."""
    return dict(case_id=case_id, cases_json=cases_json, ranking_start="spinup",
                spinup_minutes=int(spinup_min), strict_restart=int(strict_restart),
                spinup_timeout_s=spinup_timeout_s, anchor=anchor, steps=int(steps),
                restart_io_form=int(restart_io_form))


def ranking_export(ranking, time_est):
    """[(KEY, VALUE), ...] for scc_sweep.build_sbatch_command's extra_export."""
    spin_to = ranking.get("spinup_timeout_s") or time_est.get("spinup_timeout_s") \
        or time_est["timeout_s"]
    return [("CASES_JSON", ranking["cases_json"]), ("RANKING_START", ranking["ranking_start"]),
            ("SPINUP_MINUTES", ranking["spinup_minutes"]),
            ("RANKING_STRICT_RESTART", ranking["strict_restart"]),
            ("SPINUP_TIMEOUT_S", int(spin_to)),
            ("RANKING_RESTART_IO_FORM", ranking["restart_io_form"])]


def _padded_limit(total_run_s):
    """(job_hours, time_limit) -- estimate_run_time's own padding (x1.5 + 10 min, capped)."""
    padded = (total_run_s / 3600.0) * sweep.TIME_SAFETY_FACTOR + sweep.TIME_PAD_MIN / 60.0
    capped = min(padded, sweep.TIME_CAP_HOURS)
    return capped, sweep.campaign_experiment._format_hms(capped * 60.0)


def _job_time_est(ctx, n_or_pairs, reps, s_est, ranking=None):
    """estimate_run_time, plus -- only in predicted mode -- the shared anchor spin-up every
    ranking job runs before its candidates (startup + spinup_steps * s_est), added to the RAW
    total (budget decisions) and the padded --time, with its own SPINUP_TIMEOUT_S. In t0 mode
    this IS estimate_run_time (identical dict)."""
    ranking = ctx.ranking if ranking is None else ranking
    if isinstance(n_or_pairs, int):
        est = estimate_run_time(n_or_pairs, reps, ctx.steps if not ranking else ranking["steps"],
                                s_est=s_est, startup_s=ctx.startup_s)
    else:
        est = estimate_run_time(n_or_pairs, reps, ctx.steps if not ranking else ranking["steps"],
                                startup_s=ctx.startup_s)
    if not ranking:
        return est
    # predicted mode: ranking_cost_model (restart read + radiation factor), plus the shared
    # anchor spin-up every ranking job runs first
    dt = float(ranking.get("dt") or 0) or None
    spin_steps = (ranking["spinup_minutes"] * 60.0 / dt) if dt else 0.0
    steps = ranking["steps"]
    pairs = ([(None, s_est)] * n_or_pairs if isinstance(n_or_pairs, int)
             else [(c, s) for c, s in n_or_pairs])
    runs = [ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), steps, spin_steps,
                               ranking.get("stepra"), s)["per_run_s"] for _c, s in pairs]
    m = ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), steps, spin_steps,
                           ranking.get("stepra"), s_est if s_est else (pairs[0][1] if pairs else 0))
    total = reps * sum(runs) + m["spin_s"]
    est = dict(est, total_run_s=total, cost_model=m,
               timeout_s=max(sweep.TIMEOUT_MIN_S,
                             int(math.ceil(sweep.TIMEOUT_MULTIPLIER * max(runs or [0])))),
               spinup_timeout_s=max(sweep.TIMEOUT_MIN_S,
                                    int(math.ceil(sweep.TIMEOUT_MULTIPLIER * m["spin_s"]))))
    est["job_hours"], est["time_limit"] = _padded_limit(est["total_run_s"])
    return est


def _notes_kv(notes):
    """{key: value} of a ledger row's space-separated key=value notes tokens."""
    out = {}
    for tok in (notes or "").split():
        k, sep, v = tok.partition("=")
        if sep:
            out[k] = v
    return out


def evaluate_job_gates(job_id, rows, s_by_cid, expected_cids, drift_max=ANCHOR_DRIFT_MAX,
                       require_restart=True, check_drift=True):
    """Validity gates for ONE ranking job (decision 0055). Returns {job_id, ok, failures,
    anchor_drift, s_anchor, s_anchor_end}; never raises (the caller decides: stage 1 warns,
    stage 2 raises). Fails when: any expected cid has no ledger row; any row's notes lack
    restart_used=1 (strict_campaign RANKING_STRICT_RESTART skips carry status=/reason=); any row
    carries foreign_load=1 or a node_dirty other than 'none'; ANCHOR or ANCHOREND has no s value;
    or |s(ANCHOREND)/s(ANCHOR) - 1| > drift_max. `s_by_cid`: {cid: s} -- stage 2 passes each
    run's steps.txt s_normal, stage 1 the ledger s_per_step_median. require_restart=False
    (stage1=t0: plain t0-start jobs) skips the restart_used check."""
    failures = []
    by_cid = {}
    for r in rows:
        parsed = parse_run_id(r.get("run_id"))
        if parsed and str(parsed["job"]) == str(job_id):
            by_cid[parsed["cid"]] = r
    for cid in expected_cids:
        row = by_cid.get(cid)
        if row is None:
            failures.append("%s: no ledger row" % cid)
            continue
        kv = _notes_kv(row.get("notes"))
        if require_restart and kv.get("restart_used") != "1":
            failures.append("%s: restart_used=%s (status=%s reason=%s) -- did not start from the "
                            "shared restart" % (cid, kv.get("restart_used", "missing"),
                                                kv.get("status", "-"), kv.get("reason", "-")))
        if kv.get("foreign_load") == "1":
            failures.append("%s: foreign_load=1 (%s)" % (cid, kv.get("foreign", "?")))
        if kv.get("node_dirty", "none") != "none":
            failures.append("%s: node_dirty=%s" % (cid, kv.get("node_dirty")))
    a, e = s_by_cid.get("ANCHOR"), s_by_cid.get(ANCHOR_END_ID)
    drift = None
    if not check_drift:                       # ANCHOR_END=0 / fused: no ANCHOREND, gate off
        pass
    elif not a or not e:
        failures.append("anchor drift: ANCHOR s=%s ANCHOREND s=%s (both required)" % (a, e))
    else:
        drift = e / a - 1.0
        if abs(drift) > drift_max:
            failures.append("anchor drift %+.2f%% exceeds +-%.1f%% (ANCHOR %.4fs, ANCHOREND %.4fs)"
                            % (100 * drift, 100 * drift_max, a, e))
    return dict(job_id=str(job_id), ok=not failures, failures=failures, anchor_drift=drift,
                s_anchor=a, s_anchor_end=e)


def _record_stage1_gates(ctx, job_ids, rows, label):
    """Stage 1 (round0/adaptive in predicted mode): evaluate every job's gates from its ledger
    s_per_step_median and RECORD them (report + a WARNING line) without raising; stage 2's gates
    are the ones that raise. Also records, per job, in ctx.stage1_jobs:
      s_anchor_job  mean(s ANCHOR, s ANCHOREND) of THAT job (ANCHOR alone if ANCHOREND has no
                    value) -- the per-allocation normalizer stage1_normalized_ranking divides
                    each candidate by (cross-node bias: different jobs land on different nodes).
      excluded      True when any row has foreign_load=1 or any row's restart_used != 1: that
                    job's candidates are dropped from the stage-1 ranking (reason recorded). Anchor
                    drift > ANCHOR_DRIFT_MAX alone only warns (normalization absorbs most of it).
    stage1=t0 jobs start at t0 (no shared restart), so restart_used is not required there.
    No-op outside predicted-mode stage 1."""
    if not _stage1_active(ctx):
        return
    require_restart = ctx.ranking is not None
    for jid in job_ids:
        mine = [r for r in rows if (parse_run_id(r.get("run_id")) or {}).get("job") == str(jid)]
        cids = [parse_run_id(r["run_id"])["cid"] for r in mine]
        s_by_cid = {}
        for r in mine:
            s = sweep._to_float(r.get("s_per_step_median"))
            if s:
                s_by_cid[parse_run_id(r["run_id"])["cid"]] = s
        rec = evaluate_job_gates(jid, mine, s_by_cid, cids, require_restart=require_restart)
        reasons = []
        for r in mine:
            kv = _notes_kv(r.get("notes"))
            cid = parse_run_id(r["run_id"])["cid"]
            if kv.get("foreign_load") == "1":
                reasons.append("%s foreign_load=1 (%s)" % (cid, kv.get("foreign", "?")))
            if require_restart and kv.get("restart_used") != "1":
                reasons.append("%s restart_used=%s" % (cid, kv.get("restart_used", "missing")))
        a, e = s_by_cid.get("ANCHOR"), s_by_cid.get(ANCHOR_END_ID)
        s_anchor_job = ((a + e) / 2.0) if (a and e) else a
        if not s_anchor_job:
            reasons.append("no ANCHOR measurement to normalize by")
        rec.update(stage="stage1", label=label, s_anchor_job=s_anchor_job,
                   excluded=bool(reasons), exclude_reason="; ".join(reasons) or None)
        ctx.ranking_gates.append(rec)
        ctx.stage1_jobs[str(jid)] = dict(s_anchor_job=s_anchor_job, excluded=bool(reasons),
                                         reason=rec["exclude_reason"], label=label)
        if reasons:
            print("WARNING: %s job %s EXCLUDED from the stage-1 ranking: %s"
                  % (label, jid, rec["exclude_reason"]))
        elif not rec["ok"]:
            print("WARNING: %s job %s failed ranking gates (kept, anchor-normalized): %s"
                  % (label, jid, "; ".join(rec["failures"])))


def stage1_normalized_ranking(all_results, stage1_jobs):
    """(ranked, excluded) for stage 1 (decision 0055 gap 1, cross-node bias): every candidate's
    run (run_ids -> job) is divided by that job's own s_anchor_job; key = median of those ratios
    over the candidate's runs in NON-excluded jobs. ranked: [{cid, config, median, norm_key,
    jobs}] ascending by (norm_key, cid) -- ANCHOR/ANCHOREND never ranked. excluded: [{cid,
    reason}] for candidates with no run left after excluding jobs."""
    ranked, excluded = [], []
    for cid, r in sorted(all_results.items()):
        if cid in ("ANCHOR", ANCHOR_END_ID) or r.get("config") is None:
            continue
        ratios, jobs, why = [], [], []
        for s, rid in zip(r["s_values"], r.get("run_ids") or []):
            job = str((parse_run_id(rid) or {}).get("job"))
            info = stage1_jobs.get(job)
            if info is None or not info.get("s_anchor_job"):
                why.append("job %s: no anchor normalizer" % job)
            elif info["excluded"]:
                why.append("job %s: %s" % (job, info["reason"]))
            else:
                ratios.append(s / info["s_anchor_job"])
                jobs.append(job)
        if ratios:
            ranked.append(dict(cid=cid, config=r["config"], median=r["median"],
                               norm_key=statistics.median(ratios), jobs=jobs))
        else:
            excluded.append(dict(cid=cid, median=r.get("median"),
                                 reason="; ".join(why) or "no run"))
    ranked.sort(key=lambda e: (e["norm_key"], e["cid"]))
    return ranked, excluded


def _fallback_s_per_step(nl, ranks, omp):
    """Uncalibrated cell-step cost model (same arithmetic as rebuild_segment_timeout_min)."""
    max_dom = int(wtune_predict.g(nl, "domains", "max_dom", 1))
    cal = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
    return (cal["fallback_s_per_cell_step_ns"] * 1e-9
            * wtune_predict.nested_equivalent_cells(nl, max_dom) / max(int(ranks) * int(omp), 1))


def ranking_job_timeouts(ctx, configs, steps, spinup_min, dt, s_by_cid=None, nl=None,
                         safety=3.0, restart_read_s=None, stepra=None):
    """(cand_timeout_s, spinup_timeout_s, time_limit, expected_s) for one stage-2 ranking job,
    from a cost estimate instead of a blanket: per config s/step = its stage-1 measured median
    when known, else the cell-step model (_fallback_s_per_step), else ctx's fallback; per-run
    expected = ranking_cost_model's per_run_s (restart read, radiation factor). TIMEOUT_S is one
    value per job (strict_campaign), so the slowest config sets it: max(TIMEOUT_MIN_S,
    safety*expected). Spin-up: ANCHOR's s over spinup_min*60/dt steps, same padding. --time is
    the larger of the padded expected total and spin-up timeout + every run's expected."""
    s_by_cid = s_by_cid or {}

    def s_of(c):
        if s_by_cid.get(c["id"]):
            return s_by_cid[c["id"]]
        if s_by_cid.get("ANCHOR") and c.get("config_source") in ("anchor", "anchor_end"):
            return s_by_cid["ANCHOR"]
        try:
            return _fallback_s_per_step(nl, c["ranks"], c.get("omp", 1))
        except Exception:
            return sweep.CONSERVATIVE_S_PER_STEP

    rr = _restart_read_s(ctx) if restart_read_s is None else restart_read_s
    spin_steps = spinup_min * 60.0 / dt
    per_run = [ranking_cost_model(ctx.startup_s, rr, steps, spin_steps, stepra,
                                  s_of(c))["per_run_s"] for c in configs]
    cand_timeout = max(sweep.TIMEOUT_MIN_S, int(math.ceil(safety * max(per_run))))
    anchor = next((c for c in configs if c.get("id") == "ANCHOR"), configs[0])
    spin = ranking_cost_model(ctx.startup_s, rr, steps, spin_steps, stepra,
                              s_of(anchor))["spin_s"]
    spin_timeout = max(sweep.TIMEOUT_MIN_S, int(math.ceil(safety * spin)))
    expected = spin + sum(per_run)
    limit_s = max(expected * sweep.TIME_SAFETY_FACTOR + sweep.TIME_PAD_MIN * 60.0,
                  spin_timeout + sum(per_run) + sweep.TIME_PAD_MIN * 60.0)
    limit_h = min(limit_s / 3600.0, sweep.TIME_CAP_HOURS)
    return (cand_timeout, spin_timeout, sweep.campaign_experiment._format_hms(limit_h * 60.0),
            expected)


def job_timing(ctx, job_ids):
    """{job_id: {queue_s, run_s}} from sacct Submit/Start/End (queue = Start-Submit, run =
    End-Start); a job sacct does not report (or a fake runner) gets None values."""
    ids = [str(j) for j in job_ids if j]
    out = {j: dict(queue_s=None, run_s=None) for j in ids}
    if not ids:
        return out
    try:
        text = ctx.sacct_runner([ctx.sacct_cmd, "-j", ",".join(ids), "-X", "-n", "-P",
                                 "--format=JobID,Submit,Start,End"])
    except Exception:
        return out
    fmt = "%Y-%m-%dT%H:%M:%S"
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        jid = parts[0].split(".")[0]
        if jid not in out:
            continue
        try:
            sub, st = (datetime.datetime.strptime(parts[1], fmt),
                       datetime.datetime.strptime(parts[2], fmt))
        except ValueError:
            continue
        out[jid]["queue_s"] = (st - sub).total_seconds()
        try:
            out[jid]["run_s"] = (datetime.datetime.strptime(parts[3], fmt) - st).total_seconds()
        except ValueError:
            pass
    return out


def choose_primary_point(points, sample_points):
    """P* (decision 0055): the planned point whose measured time M (`.at`) has the LARGEST
    sunlit fraction over the case's sample points (wtune_predict.sunlit_fraction); ties -> the
    earliest. Rationale: stage 1 ranks every candidate at ONE point, and the daylit state carries
    the SW-radiation + convective load that dominates the whole-run cost differences between
    decompositions, so a daylit point is the most discriminating single stage-1 point. None
    sample_points (no wrfinput geometry) -> every fraction 0 -> earliest point. Returns (index,
    point, fraction)."""
    if not points:
        raise ValueError("choose_primary_point: no planned points")
    best = None
    for i, p in enumerate(points):
        frac = (wtune_predict.sunlit_fraction(sample_points, p.at) if sample_points else None) or 0.0
        key = (-frac, p.at, i)
        if best is None or key < best[0]:
            best = (key, i, p, frac)
    return best[1], best[2], best[3]


def rebuilt_case_id(case, index):
    """Underscore-free case id for a rebuilt point (strict_campaign RUN_ID/_RUN_ID_RE split the
    case off at the first '_')."""
    return re.sub(r"[^A-Za-z0-9]", "", case) + "p%d" % index


def rebuilt_steps(stepra):
    """Candidate STEPS at a rebuilt point: max(30, 2*stepra) + 2 (>= 2 radiation steps plus the
    dropped first step and one spare)."""
    return max(30, 2 * int(stepra)) + 2


def rebuilt_measure_steps(steps, dt):
    """coldstart measure_steps so the rebuilt window W covers spin-up + the candidates' whole
    run: strict_campaign runs ceil(STEPS*dt/60) whole minutes (MIN), so W needs
    ceil(MIN*60/dt) >= STEPS steps after the spin-up."""
    run_min = int(math.ceil(steps * dt / 60.0))
    return int(math.ceil(run_min * 60.0 / dt))


def write_rebuilt_case_entry(case, case_entry, index, point, plan, plan_dir, spinup_min, stepra,
                             anchor, steps=None):
    """Turns one COMPLETED real_only coldstart plan into a strict_campaign case entry. Writes
    (a) <plan_dir>/namelist.template.input -- coldstart.build_wrf_window_namelist_text (the same
    renderer rebuild wrf windows use: start=T, decomposition, history off, restart off) over the
    window's own real.exe namelist, then run_seconds forced to 0 (strict_campaign's --run-minutes
    alone sets the length); and (b) symlinks of the original case's input_files into win_0/real
    (the new prep_dir) when missing there. Returns (case_id, entry). Only ever writes under
    plan_dir (never inside the user's case dir). Raises RuntimeError when real.exe output is
    missing."""
    real_dir = os.path.join(plan_dir, "win_0", "real")
    need = ["namelist.input", "wrfbdy_d01"] + ["wrfinput_d%02d" % d for d in
                                                range(1, int(plan.get("max_dom", 1)) + 1)]
    missing = [f for f in need if not os.path.exists(os.path.join(real_dir, f))]
    if missing:
        raise RuntimeError("rebuilt point %d (%s): real.exe output missing in %s: %s"
                           % (index, point.at.isoformat(), real_dir, ", ".join(missing)))
    with open(os.path.join(real_dir, "namelist.input")) as f:
        real_text = f.read()
    T = datetime.datetime.strptime(plan["windows"][0]["t"], "%Y-%m-%d_%H:%M:%S")
    dt = float(plan["dt_s"])
    win_steps = int(plan["measure_steps"]) + int(round(float(plan["spinup_min"]) * 60.0 / dt))
    txt = coldstart.build_wrf_window_namelist_text(real_text, T, win_steps, dt, anchor["nx"],
                                                   anchor["ny"], anchor.get("numtiles", 1))
    txt = coldstart.set_list(txt, "run_seconds", 0, 1)
    template = os.path.join(plan_dir, "namelist.template.input")
    with open(template, "w") as f:
        f.write(txt)
    orig_prep = sweep._expand(case_entry.get("prep_dir") or "")
    for name in case_entry.get("input_files") or []:
        dst = os.path.join(real_dir, name)
        if not os.path.lexists(dst) and orig_prep:
            os.symlink(os.path.join(orig_prep, name), dst)
    case_id = rebuilt_case_id(case, index)
    entry = copy.deepcopy(case_entry)
    # bp_dir="" (2026-10-03, ADIOS2 for every input stream): real.exe keeps writing netCDF
    # wrfinput/wrfbdy -- this function, wtune_predict.read_wrfinput_sample_points and the state
    # tools read them as netCDF, and a WRF-written BP5 file is not readable by WRF's own reader
    # (restart, job 1273), so real.exe ADIOS2 output would also need a BP4 adios2.xml in the real
    # dir. Instead strict_campaign.sbatch converts this prep_dir once into its BP cache
    # (configs.py stage-bp-inputs, keyed by prep_dir, lock-serialized across the point's jobs) and
    # every probe of the point runs all-ADIOS2.
    entry.update(prep_dir=os.path.abspath(real_dir), bp_dir="", namelist_template=template,
                 steps=int(steps if steps is not None else rebuilt_steps(stepra)), enabled=True,
                 rebuilt_from=case, rebuilt_point_at=point.at.isoformat(),
                 rebuilt_start=T.strftime("%Y-%m-%d_%H:%M:%S"), rebuilt_spinup_min=spinup_min)
    return case_id, entry


def prepare_rebuilt_points(ctx, points, anchor, case_dir, namelist_path, case_entry, out_dir,
                           spinup_min, stepra, dt, forcing=PREDICTED_FORCING, grib_dir=None,
                           date_shift_to=None):
    """submit_rebuilt_points + finish_rebuilt_points in one call (stage1=pstar needs the
    rebuilt points before stage 1)."""
    return finish_rebuilt_points(ctx, submit_rebuilt_points(
        ctx, points, anchor, case_dir, namelist_path, case_entry, out_dir, spinup_min, stepra,
        dt, forcing=forcing, grib_dir=grib_dir, date_shift_to=date_shift_to))


def submit_rebuilt_points(ctx, points, anchor, case_dir, namelist_path, case_entry, out_dir,
                          spinup_min, stepra, dt, forcing=PREDICTED_FORCING, grib_dir=None,
                          date_shift_to=None):
    """D1 (decision 0055): rebuild EVERY planned point from ERA5 once, in parallel -- one
    coldstart real_only plan dir + job per point (wtune_probe_plan.plan_rebuild_segments with
    real_only=True, T = M - spinup_min; fetch/stage/submit concurrently, same structure as
    run_rebuild_points), wait, then write_rebuilt_case_entry per point and ONE
    <out_dir>/rebuilt_cases.json holding every entry (strict_campaign CASES_JSON). measure_steps
    is rebuilt_measure_steps(rebuilt_steps(stepra)) so wrfbdy covers spin-up + the candidates'
    run. This half plans, fetches, stages and SUBMITS (does not wait) and returns a handle for
    finish_rebuilt_points; stage1=t0 runs stage 1 while these jobs run. Refuses an out_dir
    inside case_dir."""
    case_dir = os.path.abspath(sweep._expand(case_dir))
    out_dir = os.path.abspath(out_dir)
    if os.path.commonpath([out_dir, case_dir]) == case_dir:
        raise RuntimeError("prepare_rebuilt_points: out_dir %s is inside the case dir %s"
                           % (out_dir, case_dir))
    missing_env = [v for v in ("WPS_DIR", "WRF_TREE") if not os.environ.get(v)]
    if missing_env:
        raise RuntimeError("prepare_rebuilt_points: %s not set -- source the site env first"
                           % ", ".join(missing_env))
    steps = rebuilt_steps(stepra)
    measure_steps = rebuilt_measure_steps(steps, dt)
    # real_only jobs run only WPS + real.exe, whose output does not depend on the candidates'
    # decomposition, so each point needs ONE node, not the tuner's allocation: on a busy cluster
    # (conus25-pred1, 2026-09-27: 6 free nodes) 4-node prep jobs ran one at a time. Falls back to
    # the anchor's own shape when no legal 1-node decomposition exists.
    one = sweep.build_anchor(ctx.case_json, ctx.cores_per_node, 1, ctx.hardware_id)
    shape = one or anchor
    config = dict(ranks=shape["ranks"], omp=shape["omp"], nx=shape["nx"], ny=shape["ny"],
                  numtiles=shape.get("numtiles", 1), nodes=1 if one else ctx.nodes)
    seg_timeout_min = rebuild_segment_timeout_min(case_dir, spinup_min, config["ranks"],
                                                  config["omp"])
    campaign_dir = sweep.campaign_dir_for(ctx.sbatch_script)

    def _prepare_and_submit(i, p):
        pdir = os.path.join(out_dir, "pt_%d" % i)
        plan = wtune_probe_plan.plan_rebuild_segments(
            case_dir, [p], config, pdir, spinup_min=spinup_min, date_shift_to=date_shift_to,
            namelist=namelist_path, grib_dir=grib_dir, forcing=forcing,
            measure_steps=measure_steps, real_only=True)
        total_bytes, _forcing = coldstart.resolve_and_fetch_forcing(
            plan, pdir, os.path.join(pdir, "grib"), grib_dir=grib_dir)
        print("rebuilt point %d (%s): fetched %.1f MB forcing"
              % (i, p.at.isoformat(), total_bytes / 1e6), flush=True)
        sbatch_path = coldstart.stage_plan(pdir, site="scc", per_window_min=seg_timeout_min)
        cmd = ("sbatch --export=ALL,CAMPAIGN=%s %s" % (campaign_dir, sbatch_path)
               if campaign_dir else "sbatch %s" % sbatch_path)
        job_id = sweep.submit_job(cmd, sbatch_runner=ctx.sbatch_runner)
        print("rebuilt point %d: real_only job %s (%s)" % (i, job_id, sbatch_path), flush=True)
        return plan, pdir, job_id

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(points))) as pool:
        done = list(pool.map(lambda ip: _prepare_and_submit(*ip), enumerate(points)))
    job_ids = [d[2] for d in done]
    ctx.owned_jobs.extend(job_ids)
    return dict(points=points, done=done, job_ids=job_ids, anchor=anchor, case_entry=case_entry,
                out_dir=out_dir, spinup_min=spinup_min, stepra=stepra, dt=dt,
                steps=rebuilt_steps(stepra))


def finish_rebuilt_points(ctx, handle):
    """Waits for submit_rebuilt_points' real_only jobs, then write_rebuilt_case_entry per point
    and ONE <out_dir>/rebuilt_cases.json holding every entry (strict_campaign CASES_JSON).
    Returns [dict(index, point, case_id, T, cases_json, prep_dir, template, steps, spinup_min,
    stepra, dt, plan_dir, job_id)]. Raises on any job not COMPLETED or missing real.exe output
    -- never silently drops a point."""
    points, done, job_ids = handle["points"], handle["done"], handle["job_ids"]
    anchor, case_entry, out_dir = handle["anchor"], handle["case_entry"], handle["out_dir"]
    spinup_min, stepra, dt, steps = (handle["spinup_min"], handle["stepra"], handle["dt"],
                                     handle["steps"])
    states = wait_for_jobs(ctx, job_ids, "rebuild-prep")
    bad = {j: states.get(j, ("UNKNOWN", ""))[0] for j in job_ids
           if states.get(j, ("UNKNOWN", ""))[0] != "COMPLETED"}
    if bad:
        raise RuntimeError("rebuilt-point real_only job(s) %s not COMPLETED -- stop and diagnose "
                           "read-only" % bad)
    cases_json = os.path.join(out_dir, "rebuilt_cases.json")
    entries, out = {}, []
    for i, (p, (plan, pdir, jid)) in enumerate(zip(points, done)):
        case_id, entry = write_rebuilt_case_entry(ctx.case, case_entry, i, p, plan, pdir,
                                                  spinup_min, stepra, anchor, steps=steps)
        entries[case_id] = entry
        out.append(dict(index=i, point=p, case_id=case_id,
                        T=datetime.datetime.strptime(entry["rebuilt_start"], "%Y-%m-%d_%H:%M:%S"),
                        cases_json=cases_json, prep_dir=entry["prep_dir"],
                        template=entry["namelist_template"], steps=steps, spinup_min=spinup_min,
                        stepra=stepra, dt=dt, plan_dir=pdir, job_id=jid))
    with open(cases_json, "w") as f:
        json.dump({"cases": entries}, f, indent=2, sort_keys=True)
        f.write("\n")
    return out


POINT_MODES = ("default", "no_end", "fused")


def point_runs(wrapped, mode):
    """The configs of a point job that actually run. default: ANCHOR + candidates + ANCHOREND
    (drift gate on). no_end (ANCHOR_END=0): ANCHOREND is dropped, drift gate off. fused
    (POINT_FUSED=1, winner-only): ONE wrf.exe runs spin-up + window for the single candidate; the
    ANCHOR placeholder is not run and there is no ANCHOREND (strict_campaign.sbatch point-job run
    list, decisions 0055/0059)."""
    if mode == "fused":
        return [c for c in wrapped if c["id"] not in ("ANCHOR", ANCHOR_END_ID)]
    if mode == "no_end":
        return [c for c in wrapped if c["id"] != ANCHOR_END_ID]
    return list(wrapped)


def point_export(mode):
    """Extra sbatch --export entries for a point-job mode. A fused run additionally needs
    RANKING_START=spinup, MODE=timing, one SPINUP_MINUTES target and REPS=1 (all set by the point
    job already) and NO window frame; WINDOW_EXACT=1 makes the window exactly STEPS*dt."""
    if mode == "fused":
        return [("POINT_FUSED", 1), ("ANCHOR_END", 0), ("WINDOW_EXACT", 1)]
    if mode == "no_end":
        return [("ANCHOR_END", 0)]
    return []


def choose_point_mode(configs, anchor):
    """The default point-job mode for `configs` (candidates besides the ANCHOR placeholder):
    fused for exactly ONE candidate without a window frame (winner-only plan); no_end (drift gate
    off) when the whole job has <= 2 configs (ANCHOR + one more); default otherwise (phase-1
    multi-config waves never come through here and keep ANCHOREND)."""
    cands = [c for c in configs if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    framed = any(c.get("window_frame") for c in cands) or bool((anchor or {}).get("window_frame"))
    if len(cands) == 1 and not framed:
        return "fused"
    if len(cands) + 1 <= 2:
        return "no_end"
    return "default"


def _measure_run(ctx, rp, cid, job_id, dt, first_global_step=None):
    """PointMeasurement for one stage-2 run (RUNDIR = ledger_dir/<case_id>_<cid>_r1_<job>):
    first_global_step_from_rsl relative to the rebuilt start T, measurement_from_steps_file with
    the case stepra; `.at` = the point's measured-window midpoint on the CASE timeline (M +
    steps*dt/2). Returns (PointMeasurement, g0) or raises."""
    rundir = os.path.join(ctx.ledger_dir, "%s_%s_r1_%s" % (rp["case_id"], cid, job_id))
    steps_path = os.path.join(rundir, "steps.txt")
    if not os.path.exists(steps_path) or os.path.getsize(steps_path) == 0:
        raise RuntimeError("%s missing/empty" % steps_path)
    # fused rows keep only the post-spin-up steps (steps_all.txt has the raw list): the first one is
    # known (spin-up steps + 1), the rsl log would say 1
    g0 = (first_global_step if first_global_step is not None else
          wtune_probe_plan.first_global_step_from_rsl(os.path.join(rundir, "rsl.error.0000"),
                                                      rp["T"], dt))
    at = rp["point"].at + datetime.timedelta(seconds=rp["steps"] * dt / 2.0)
    pm = wtune_probe_plan.measurement_from_steps_file(
        steps_path, rp["stepra"], at, rp["point"].reason, "rebuild", first_global_step=g0)
    return pm, g0


def measure_candidates_at_points(ctx, configs, rebuilt_points, anchor, s_by_cid=None,
                                 restart_io_form=None, name_prefix="stage2", mode=None):
    """D3 (decision 0055): one strict_campaign job per rebuilt point, ALL submitted before any
    wait. Each: CONFIG_FILE = ANCHOR + configs + ANCHOREND, REPS=1, CASE=<point id>,
    CASES_JSON, RANKING_START=spinup, SPINUP_MINUTES=spinup_min, RANKING_STRICT_RESTART=1,
    TIMEOUT_S/SPINUP_TIMEOUT_S/--time from ranking_job_timeouts. Each run's steps.txt +
    rsl.error.0000 -> _measure_run. Gates per job (evaluate_job_gates on the steps.txt s_normal;
    also every run's first global step must be exactly spinup_steps+1, i.e. it resumed at M):
    any failure RAISES RuntimeError naming the job and every failure -- nothing is dropped.
    Returns (meas_by_cid {cid: [PointMeasurement, ...] in point order, ANCHOREND excluded},
    gate_records, job_ids)."""
    restart_io_form = restart_io_form or ctx.restart_io_form
    mode = mode or "default"
    if mode not in POINT_MODES:
        raise ValueError("unknown point-job mode %r" % (mode,))
    wrapped = wrap_with_anchors(configs, anchor)
    runs = point_runs(wrapped, mode)              # what the runner actually executes
    nl = None
    try:
        nl = wtune_predict.parse_namelist(ctx.namelist_path)
    except Exception:
        pass
    jobs = []
    for rp in rebuilt_points:
        dt = float(rp["dt"])
        cand_to, spin_to, time_limit, expected = ranking_job_timeouts(
            ctx, runs, rp["steps"], rp["spinup_min"], dt, s_by_cid=s_by_cid, nl=nl,
            stepra=rp.get("stepra"))
        job_out = os.path.join(ctx.out_dir, name_prefix, "pt_%d" % rp["index"])
        os.makedirs(job_out, exist_ok=True)
        configs_path = os.path.join(job_out, "configs.json")
        # the file keeps the ANCHOR placeholder (it names the spin-up decomposition) and drops
        # ANCHOREND whenever ANCHOR_END=0 (the runner would drop it anyway)
        sweep.write_configs_json(configs_path, [c for c in wrapped if mode == "default"
                                                or c["id"] != ANCHOR_END_ID])
        ranking = ranking_context(rp["case_id"], rp["cases_json"], rp["spinup_min"], rp["steps"],
                                  anchor, restart_io_form=restart_io_form,
                                  spinup_timeout_s=spin_to)
        cmd = sweep.build_sbatch_command(
            rp["case_id"], ctx.nodes, ctx.cores_per_node, time_limit, cand_to, 1, rp["steps"],
            os.path.abspath(configs_path), job_out, env_source=ctx.env_source,
            sbatch_script=ctx.sbatch_script, campaign_dir=sweep.campaign_dir_for(ctx.sbatch_script),
            extra_export=ranking_export(ranking, {"timeout_s": cand_to}) + point_export(mode))
        jid = sweep.submit_job(cmd, sbatch_runner=ctx.sbatch_runner)
        ctx.owned_jobs.append(jid)
        print("%s point %d (%s, case %s): job %s, %d runs, TIMEOUT_S=%d SPINUP_TIMEOUT_S=%d "
              "--time=%s (expected %s)" % (name_prefix, rp["index"], rp["point"].at.isoformat(),
                                           rp["case_id"], jid, len(runs), cand_to, spin_to,
                                           time_limit, _fmt_hms(expected)) + (
            " [%s]" % mode if mode != "default" else ""))
        jobs.append((rp, jid))
    job_ids = [j for _rp, j in jobs]
    # once started, stage 2 runs to completion: exempt from the running budget (overrun is
    # reported by run_predicted), still cancelled by the hard wall cap
    states = wait_for_jobs(ctx, job_ids, name_prefix, exempt_budget=True)
    bad = {j: states.get(j, ("UNKNOWN", ""))[0] for j in job_ids
           if states.get(j, ("UNKNOWN", ""))[0] != "COMPLETED"}
    if bad:
        raise RuntimeError("%s job(s) %s not COMPLETED -- stop and diagnose read-only"
                           % (name_prefix, bad))
    meas_by_cid, gates, errors = {}, [], []
    for rp, jid in jobs:
        dt = float(rp["dt"])
        rows = ctx.ledger_reader(rp["case_id"], [jid])
        s_by, extra = {}, []
        # a run resumed at the shared restart (T + spinup) has its first step at global step
        # spinup*60/dt + 1 exactly (predicted_spinup_min makes spinup*60 a multiple of dt); a t0
        # start is at 1.
        expect_g0 = rp["spinup_min"] * 60.0 / dt + 1.0
        for c in runs:
            try:
                pm, g0 = _measure_run(ctx, rp, c["id"], jid, dt,
                                      first_global_step=expect_g0 if mode == "fused" else None)
            except Exception as e:
                extra.append("%s: no measurement (%s: %s)" % (c["id"], type(e).__name__, e))
                continue
            if g0 != expect_g0:
                extra.append("%s: first global step %d, expected %g -- run did not resume at "
                             "the shared restart" % (c["id"], g0, expect_g0))
            s_by[c["id"]] = pm.s_normal
            if c["id"] != ANCHOR_END_ID:
                meas_by_cid.setdefault(c["id"], []).append(pm)
        # drift gate only when ANCHOREND ran; a fused/no_end row carries drift_gate=off and a
        # fused row's restart_used=1 is NOT evidence that a restart file was read
        rec = evaluate_job_gates(jid, rows, s_by, [c["id"] for c in runs],
                                 check_drift=(mode == "default"))
        rec["point_mode"] = mode
        rec["failures"].extend(extra)
        rec.update(ok=not rec["failures"], stage=name_prefix, point_index=rp["index"],
                   point_at=rp["point"].at.isoformat(), case_id=rp["case_id"])
        gates.append(rec)
        ctx.ranking_gates.append(rec)
        if not rec["ok"]:
            errors.append("job %s (point %d %s): %s" % (jid, rp["index"],
                                                        rp["point"].at.isoformat(),
                                                        "; ".join(rec["failures"])))
    if errors:
        raise RuntimeError("%s validity gates FAILED -- no ranking is reported from these "
                           "measurements:\n  " % name_prefix + "\n  ".join(errors))
    return meas_by_cid, gates, job_ids


def rank_by_predicted_total(meas_by_cid, run_start, run_end, dt, stepra, calibration,
                            sample_points, case_dir, nl, ranks_by_cid, nodes):
    """D4: wtune_probe_plan.combine() per cid (ANCHOREND never ranked) -> [{cid, pred_total_s,
    pred_low_s, pred_high_s, result}], ascending by pred_total_s with _cluster_rank's NOISE
    clustering (ANCHOR wins an in-noise tie, then cid order)."""
    entries = []
    for cid, pms in meas_by_cid.items():
        if cid == ANCHOR_END_ID or not pms:
            continue
        res = wtune_probe_plan.combine(
            sorted(pms, key=lambda m: m.at), run_start, run_end, dt, stepra,
            calibration=calibration, sample_points=sample_points, case_dir=case_dir, nl=nl,
            ranks=ranks_by_cid.get(cid), nodes=nodes)
        entries.append(dict(cid=cid, pred_total_s=res["total_s"], pred_low_s=res["range_low_s"],
                            pred_high_s=res["range_high_s"], result=res))
    order_index = {cid: i for i, cid in enumerate(sorted(e["cid"] for e in entries))}
    return _cluster_rank(entries, lambda e: _log_key(e["pred_total_s"]), "ANCHOR", order_index)


def predicted_spinup_min(dt, floor_min=PREDICTED_SPINUP_MIN, max_min=60):
    """Smallest whole minute m >= floor_min with (m*60) % dt == 0, so the shared restart falls
    exactly on a time step (WRF's restart alarm and configs.py restart-time agree, and a resumed
    run's first global step is exactly m*60/dt + 1). dt 36 -> 21 (35 steps); dt 15/10/25 -> 20.
    SystemExit when none exists up to max_min."""
    for m in range(int(floor_min), int(max_min) + 1):
        if abs((m * 60.0) / float(dt) - round((m * 60.0) / float(dt))) < 1e-9:
            return m
    raise SystemExit("--rank-by predicted: no whole-minute spin-up in [%d, %d] min is a multiple "
                     "of time_step=%gs -- the shared restart would not fall on a time step"
                     % (floor_min, max_min, dt))


def plan_predicted_points(ctx, spinup_min=PREDICTED_SPINUP_MIN):
    """Planned rebuild points for predicted mode: wtune_rebuild_predict.plan with the ERA5 pin,
    `spinup_min`, probe.max_points, on ctx.namelist_path (the case's own template)."""
    import wtune_rebuild_predict
    return wtune_rebuild_predict.plan(
        sweep._expand(ctx.case_json["prep_dir"]), PREDICTED_FORCING, spinup_min,
        ctx.probe_cfg.get("max_points"), namelist_path=ctx.namelist_path)


def predicted_min_budget_s(ctx, n_points, steps, spinup_min, dt, stepra, s_step, slots,
                           stage1_mode="pstar"):
    """(seconds, detail): the least RUNNING budget predicted mode can succeed with -- rebuild
    prep (REBUILD_PREP_EST_S) + a minimal stage 1 (one job with ANCHOR + 1 candidate +
    ANCHOREND: pstar = shared spin-up + 3 ranking runs at `steps`; t0 = 3 plain t0 runs at
    ctx.steps, estimate_run_time) + stage 2 for that 1 candidate + anchors at every point
    (estimate_stage2_s, ceil(points/slots) waves). Checked before anything is submitted."""
    m = ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), steps, spinup_min * 60.0 / dt,
                           stepra, s_step)
    if stage1_mode == "t0":
        stage1 = estimate_run_time(3, 1, ctx.steps, s_est=s_step,
                                   startup_s=ctx.startup_s)["total_run_s"]
        s1_note = "3 t0 runs x %d steps" % ctx.steps
    else:
        stage1 = m["spin_s"] + 3 * m["per_run_s"]
        s1_note = "spin-up + 3 runs"
    stage2 = estimate_stage2_s(n_points, 3, steps, spinup_min, dt, s_step, slots, ctx.startup_s,
                               _restart_read_s(ctx), stepra=stepra)
    total = REBUILD_PREP_EST_S + stage1 + stage2
    return total, ("minimum %s = rebuild prep %s + stage 1=%s %s (%s) + stage 2 %s "
                   "(%d point(s), %d concurrent -- nothing observed yet, conservative); %s"
                   % (_fmt_hms(total), _fmt_hms(REBUILD_PREP_EST_S), stage1_mode,
                      _fmt_hms(stage1), s1_note, _fmt_hms(stage2), n_points, slots,
                      describe_cost_model(m)))


def run_predicted(ctx, nodes, nodes_note, s_est):
    """D5 (decision 0055), rank_by == "predicted":
      0. plan points (ERA5 pin, spin-up predicted_spinup_min(dt)), P* = choose_primary_point, and
         rebuild EVERY point once (1-node real_only jobs in parallel).
      1. Stage 1, ctx.stage1_mode:
           t0 (default, conus25-pred4 G3: the P*-restart stage 1 was so expensive per candidate
              that the t0 tuner's winner was never in the pool) -- the ordinary broad t0 search
              (run_round0 + run_adaptive_rounds, no ranking context, t0 STEPS), ANCHOR/ANCHOREND
              in every job for per-job normalization. BUDGET SEMANTICS (2026-09-28,
              conus25-pred5): --minutes is the STAGE-1 SEARCH budget only (running time, same
              meaning as t0 mode) -- no stage-2 reserve is held back. Stage 2 is a separate cost,
              announced at start ("stage 2 will need about X min more"), always run after stage 1
              and trimmed only to an optional --stage2-minutes cap. Rebuild prep is SUBMITTED
              first (its controller fetch/submit is not charged), runs beside stage 1 (charged
              only while no stage-1 job runs) and is awaited only before stage 2.
           pstar -- rebuild prep first, then round0 + adaptive rounds with P*'s ranking context
              (every job spins up once at P*, every candidate starts from that restart). Budget
              semantics unchanged: --minutes covers stage 1 AND the stage-2 reserve (stage 2 is
              exempt once started; overrun reported).
      2. Stage 2 (replaces confirm): top-3 by stage-1 ANCHOR-NORMALIZED median (each run / its
         own job's mean(ANCHOR, ANCHOREND); stage1_normalized_ranking) + ANCHOR/ANCHOREND measured
         at ALL points (measure_candidates_at_points, all jobs in parallel; gates raise); winner
         = min predicted whole-run compute+init (rank_by_predicted_total).
    Returns the report dict."""
    if wtune_predict is None or wtune_probe_plan is None or coldstart is None:
        raise SystemExit("--rank-by predicted needs wtune_predict, wtune_probe_plan and "
                         "tools/coldstart_sample importable")
    case_dir = sweep._expand(ctx.case_json["prep_dir"])
    _n, _run_s, dt0, _note = wtune_predict.get_d01_steps(
        wtune_predict.parse_namelist(ctx.namelist_path))
    spinup_min = predicted_spinup_min(dt0)
    nl, run_start, run_end, dt, stepra, points, notes = plan_predicted_points(ctx, spinup_min)
    print("predicted mode: spin-up %d min (%d steps of %gs)" % (spinup_min,
                                                                 spinup_min * 60 // dt, dt))
    if not points:
        raise SystemExit("--rank-by predicted: no ERA5-rebuildable point planned (%s)"
                         % "; ".join(notes))
    sample_points, _src = wtune_predict.read_wrfinput_sample_points(case_dir)
    p_idx, p_star, p_frac = choose_primary_point(points, sample_points)
    print("predicted mode: %d point(s) %s; P* = point %d %s (sunlit fraction %.2f)"
          % (len(points), ", ".join(p.at.strftime("%m-%d_%H:%M") for p in points), p_idx,
             p_star.at.isoformat(), p_frac))
    anchor = sweep.build_anchor(ctx.case_json, ctx.nodes * ctx.cores_per_node, ctx.nodes,
                                ctx.hardware_id)
    if anchor is None:
        raise SystemExit("--rank-by predicted: no legal ANCHOR decomposition at %d node(s)"
                         % ctx.nodes)
    s_led, _s_src, n_led = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                      ctx.hardware_id)
    if n_led:
        ctx.s_prior = s_led
    else:
        try:
            ctx.s_prior = _fallback_s_per_step(nl, anchor["ranks"], anchor.get("omp", 1))
        except Exception:
            ctx.s_prior = s_led
    steps = rebuilt_steps(stepra)
    slots = compute_slots(ctx.max_nodes, ctx.nodes)
    ctx.reserve = dict(n_points=len(points), n_configs=TOP_K_STAGE2 + 2, steps=steps,
                       spinup_min=spinup_min, dt=dt, stepra=stepra, slots=slots)
    stage2_est0 = _stage2_estimate(ctx, TOP_K_STAGE2 + 2, ctx.s_prior)
    if ctx.stage1_mode == "t0":
        # --minutes is the stage-1 search budget only: refuse only when it cannot hold one
        # minimal stage-1 job (ANCHOR + 1 + ANCHOREND at t0 STEPS); stage 2 is announced
        need_s = estimate_run_time(3, 1, ctx.steps, s_est=ctx.s_prior,
                                   startup_s=ctx.startup_s)["total_run_s"]
        detail = ("stage-1 minimum %s = one t0 job, ANCHOR + 1 + ANCHOREND x %d steps at "
                  "%.3fs/step + startup %.0fs" % (_fmt_hms(need_s), ctx.steps, ctx.s_prior,
                                                  ctx.startup_s))
    else:
        need_s, detail = predicted_min_budget_s(ctx, len(points), steps, spinup_min, dt, stepra,
                                                ctx.s_prior, stage2_concurrency(ctx),
                                                stage1_mode=ctx.stage1_mode)
    print("predicted mode: feasibility -- %s" % detail)
    ctx.stage2_estimate_start_s = stage2_est0
    print("predicted mode: stage 2 will need about %.0f min %s (%d point(s) x (spin-up + %d "
          "runs), concurrency %d, estimated at %.3fs/step)"
          % (stage2_est0 / 60.0,
             "MORE, on top of --minutes" if ctx.stage1_mode == "t0" else "INSIDE --minutes",
             len(points), TOP_K_STAGE2 + 2, stage2_concurrency(ctx), ctx.s_prior))
    if ctx.stage1_mode == "t0" and not ctx.max_wall_explicit and ctx.max_wall_s is not None:
        ctx.max_wall_s = MAX_WALL_FACTOR * (ctx.budget_s + stage2_est0)
    if ctx.budget_s < need_s:
        raise SystemExit("predicted mode needs at least %.0f min of running time for %s at %d "
                         "node(s) (have %.0f): %s"
                         % (math.ceil(need_s / 60.0), ctx.case, ctx.nodes, ctx.budget_s / 60.0,
                            detail))
    rebuild_args = (ctx, points, anchor, case_dir, ctx.namelist_path, ctx.case_json,
                    os.path.join(ctx.out_dir, "rebuilt"), spinup_min, stepra, dt)
    rebuilt, prep_handle = None, None
    if ctx.stage1_mode == "pstar":
        rebuilt = prepare_rebuilt_points(*rebuild_args)
        rp_star = rebuilt[p_idx]
        ctx.steps = rp_star["steps"]
        ctx.stage1 = dict(mode="pstar", anchor=anchor)
        ctx.ranking = ranking_context(rp_star["case_id"], rp_star["cases_json"], spinup_min,
                                      rp_star["steps"], anchor,
                                      restart_io_form=ctx.restart_io_form)
        ctx.ranking["dt"] = dt
        ctx.ranking["stepra"] = stepra
        print("stage 1 (pstar): ranking at P* (case %s, start %s, STEPS=%d, "
              "RANKING_RESTART_IO_FORM=%d); s_prior %.4fs/step [%s]"
              % (rp_star["case_id"], rp_star["T"].isoformat(), ctx.steps, ctx.restart_io_form,
                 ctx.s_prior or 0.0, "ledger" if n_led else "cell-step model"))
    else:
        # stage 1 does not need the rebuilt points: submit prep now, await it before stage 2.
        # The controller's fetch/stage/submit is not stage-1 work: excluded from the clock.
        t_prep = ctx.clock.now()
        prep_handle = submit_rebuilt_points(*rebuild_args)
        ctx.prep_submit_s = ctx.clock.now() - t_prep
        ctx.uncharged_s += ctx.prep_submit_s
        ctx.background_jobs = list(prep_handle["job_ids"])
        ctx.stage1 = dict(mode="t0", anchor=anchor)
        ctx.reserve_in_stage1 = False
        ctx.stage1_extra_reserve_s = 0.0
        print("stage 1 (t0): broad t0 search on the whole --minutes (%s of running time), "
              "STEPS=%d, ANCHOR/ANCHOREND in every job; rebuild prep job(s) %s running beside "
              "it; s_prior %.4fs/step [%s]"
              % (_fmt_hms(ctx.budget_s), ctx.steps, ",".join(prep_handle["job_ids"]),
                 ctx.s_prior or 0.0, "ledger" if n_led else "cell-step model"))
    print(describe_stage2_reserve(ctx))

    job_ids0, configs_by_cid, rows0 = run_round0(ctx)
    configs_by_cid.pop(ANCHOR_END_ID, None)
    all_results = aggregate_by_cid(rows0, configs_by_cid)
    all_results.pop(ANCHOR_END_ID, None)
    s_est = update_s_est(all_results, s_est)
    if job_ids0:
        run_adaptive_rounds(ctx, configs_by_cid, all_results, s_est)
    all_results.pop(ANCHOR_END_ID, None)
    stage1, stage1_excluded = stage1_normalized_ranking(all_results, ctx.stage1_jobs)
    for x in stage1_excluded:
        print("WARNING: stage 1: %s excluded from ranking (%s)" % (x["cid"], x["reason"]))
    top = [r["config"] for r in stage1[:TOP_K_STAGE2]]
    if not top and "ANCHOR" not in all_results:
        raise SystemExit("--rank-by predicted: stage 1 measured nothing -- see gate warnings")
    s_by_cid = {cid: r["median"] for cid, r in all_results.items()}
    for x in ctx.not_measured:
        print("stage 1: job %s never started -- not measured: %s"
              % (x["job_id"], ", ".join(x["cids"]) or "-"))
    reserve_note = describe_stage2_reserve(ctx)
    stage1_running_s = ctx.elapsed()
    ctx.ranking = None                                   # stage 2 builds its own per-point contexts
    ctx.stage1 = None
    ctx.stage1_extra_reserve_s = 0.0
    if prep_handle is not None:
        ctx.background_jobs = []
        rebuilt = finish_rebuilt_points(ctx, prep_handle)
    if ctx.stage1_mode == "t0":
        top, stage2_est, skip_reason = fit_stage2(ctx, top, anchor, rebuilt, s_by_cid, nl,
                                                  left_s=ctx.stage2_budget_s)
    else:
        top, stage2_est, skip_reason = fit_stage2(ctx, top, anchor, rebuilt, s_by_cid, nl)
    if skip_reason:
        print("stage 2: SKIPPED -- %s" % skip_reason)
        meas_by_cid, gates, stage2_jobs = {}, [], []
    else:
        print("stage 2: %s + ANCHOR/ANCHOREND at all %d point(s), expected running %s "
              "(elapsed %s / budget %s)"
              % (", ".join("%s(%.4fs/step, %.4fx anchor)" % (r["cid"], r["median"], r["norm_key"])
                           for r in stage1 if r["cid"] in {c["id"] for c in top})
                 or "(no candidate besides ANCHOR)", len(rebuilt), _fmt_hms(stage2_est),
                 _fmt_hms(ctx.elapsed()), _fmt_hms(ctx.budget_s)))
        try:
            meas_by_cid, gates, stage2_jobs = measure_candidates_at_points(
                ctx, top, rebuilt, anchor, s_by_cid=s_by_cid)
        finally:
            if ctx.stage1_mode != "t0":   # stage1=t0: stage 2 is outside --minutes by design
                ctx.stage2_overrun_s = max(0.0, ctx.elapsed() - ctx.budget_s)
        if ctx.stage2_overrun_s:
            print("stage 2: ran %s of running time past the budget (exempt once started)"
                  % _fmt_hms(ctx.stage2_overrun_s))
    stage2_running_s = ctx.elapsed() - stage1_running_s
    calibration = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
    if ctx.prediction_cfg.get("drift_band_pct") is not None:
        calibration = dict(calibration, drift_band_pct=ctx.prediction_cfg["drift_band_pct"])
    cfg_by_cid = dict({c["id"]: c for c in top}, ANCHOR=anchor)
    ranked = rank_by_predicted_total(
        meas_by_cid, run_start, run_end, dt, stepra, calibration, sample_points, case_dir, nl,
        {cid: c["ranks"] for cid, c in cfg_by_cid.items()}, nodes)
    report = build_predicted_report(ctx, nodes, nodes_note, ranked, meas_by_cid, cfg_by_cid,
                                    all_results, rebuilt, p_idx, gates, stage2_jobs, notes)
    norm = {e["cid"]: e["norm_key"] for e in stage1}
    for row in report["ranking"]:
        row["stage1_norm_key"] = norm.get(row["id"])
    report.update(spinup_min=spinup_min,
                  stage1_ranking=[dict(id=e["cid"], median_s_per_step=e["median"],
                                       norm_key=e["norm_key"], jobs=e["jobs"],
                                       stage2=e["cid"] in {c["id"] for c in top})
                                  for e in stage1],
                  stage1_excluded=stage1_excluded, not_measured=list(ctx.not_measured),
                  stage1_trimmed=list(ctx.stage1_trimmed), stage1_mode=ctx.stage1_mode,
                  stage2_reserve_note=reserve_note,
                  stage2_overrun_s=ctx.stage2_overrun_s,
                  stage2_concurrency=stage2_concurrency(ctx),
                  stage1_max_running=ctx.stage1_max_running,
                  stage2_estimate_s=stage2_est, stage2_skipped_reason=skip_reason,
                  stage1_running_min=stage1_running_s / 60.0,
                  stage2_running_min=stage2_running_s / 60.0,
                  stage2_estimate_min=stage2_est0 / 60.0,
                  prep_submit_min=ctx.prep_submit_s / 60.0,
                  total_running_min=(stage1_running_s + stage2_running_s
                                     + ctx.prep_submit_s) / 60.0,
                  stage2_minutes_cap=(ctx.stage2_budget_s / 60.0
                                      if ctx.stage2_budget_s is not None else None),
                  budget_semantics=("--minutes = stage-1 search only; stage 2 separate"
                                    if ctx.stage1_mode == "t0" else
                                    "--minutes = stage 1 + stage-2 reserve (pstar)"))
    return report


_BUDGET_LEFT = object()


def fit_stage2(ctx, top, anchor, rebuilt, s_by_cid, nl=None, left_s=_BUDGET_LEFT, mode=None):
    """(top_fit, expected_running_s, skip_reason): stage 2 never starts with less running budget
    left than its own estimate. Expected = ceil(points/stage2_concurrency) waves x the slowest point job's
    ranking_job_timeouts `expected` (spin-up + ANCHOR + candidates + ANCHOREND, measured stage-1
    s/step where known). Drops the lowest-ranked candidate until it fits (fewer than 2
    candidates is fine -- run with what exists); if not even the best single candidate (or,
    with no candidate at all, ANCHOR alone) fits, skip stage 2 with the reason (reported, the
    tuner then has no predicted ranking rather than a cancelled one). `left_s`: the running
    time stage 2 may use -- default (pstar) what is left of --minutes; stage1=t0 passes the
    optional --stage2-minutes cap, and None there means no cap: nothing is trimmed."""
    waves = int(math.ceil(float(len(rebuilt)) / stage2_concurrency(ctx)))
    left = ctx.budget_s - ctx.elapsed() if left_s is _BUDGET_LEFT else left_s
    est = None
    for k in range(len(top), min(1, len(top)) - 1, -1):
        wrapped = point_runs(wrap_with_anchors(top[:k], anchor), mode or "default")
        per_job = max(ranking_job_timeouts(ctx, wrapped, rp["steps"], rp["spinup_min"],
                                           float(rp["dt"]), s_by_cid=s_by_cid, nl=nl,
                                           stepra=rp.get("stepra"))[3]
                      for rp in rebuilt)
        est = waves * per_job
        if left is None or est <= left:
            if k < len(top):
                print("stage 2: dropped %s to fit the budget (expected %s, left %s)"
                      % (", ".join(c["id"] for c in top[k:]), _fmt_hms(est), _fmt_hms(left)))
            return top[:k], est, None
    return [], est, ("even %s needs an expected %s of running time; only %s of the budget is "
                     "left (elapsed %s / budget %s)"
                     % ("the best candidate" if top else "ANCHOR alone", _fmt_hms(est or 0),
                        _fmt_hms(max(0.0, left)), _fmt_hms(ctx.elapsed()),
                        _fmt_hms(ctx.budget_s if left_s is _BUDGET_LEFT else left)))


def build_predicted_report(ctx, nodes, nodes_note, ranked, meas_by_cid, cfg_by_cid, stage1,
                           rebuilt, p_idx, gates, stage2_jobs, plan_notes):
    """tune_report.json for predicted mode: ranking by pred_total_s (+range), per cid x point
    s_normal/r/n_rad, stage-1 medians, every gate record (stage 1 warnings + stage 2), and per
    job queue time vs run time (sacct Start-Submit vs End-Start, separately)."""
    anchor_pred = next((e["pred_total_s"] for e in ranked if e["cid"] == "ANCHOR"), None)
    table = []
    for e in ranked:
        cfg = cfg_by_cid.get(e["cid"]) or {}
        table.append(dict(
            id=e["cid"], ranks=cfg.get("ranks"), omp=cfg.get("omp"), nproc_x=cfg.get("nx"),
            nproc_y=cfg.get("ny"), numtiles=cfg.get("numtiles"), dist=cfg.get("dist"),
            predicted_total_s=e["pred_total_s"], predicted_total_hms=_fmt_hms(e["pred_total_s"]),
            predicted_range_s=[e["pred_low_s"], e["pred_high_s"]],
            speedup_vs_anchor_predicted=(anchor_pred / e["pred_total_s"]
                                         if anchor_pred and e["pred_total_s"] else None),
            stage1_median_s_per_step=(stage1.get(e["cid"]) or {}).get("median"),
            points=[dict(at=pm.at.isoformat(), s_normal=pm.s_normal, r=pm.r, n_rad=pm.n_rad)
                    for pm in sorted(meas_by_cid.get(e["cid"], []), key=lambda m: m.at)]))
    best_cfg = cfg_by_cid.get(ranked[0]["cid"]) if ranked else None
    timing = job_timing(ctx, ctx.owned_jobs)
    return dict(
        case=ctx.case, rank_by="predicted", nodes=nodes, nodes_note=nodes_note,
        cores_per_node=ctx.cores_per_node, budget_minutes=ctx.budget_s / 60.0,
        elapsed_minutes=ctx.elapsed() / 60.0, job_ids=list(ctx.owned_jobs),
        cancelled_job_ids=list(ctx.cancelled_jobs), anchor_id="ANCHOR", ranking=table,
        best=table[0] if table else None,
        param_table=local_param_table(best_cfg, nodes) if best_cfg else None,
        points=[dict(index=rp["index"], at=rp["point"].at.isoformat(), reason=rp["point"].reason,
                     case_id=rp["case_id"], start=rp["T"].isoformat(),
                     rebuild_job=rp["job_id"], primary=(rp["index"] == p_idx))
                for rp in rebuilt],
        plan_notes=list(plan_notes), stage2_job_ids=list(stage2_jobs),
        gates=list(ctx.ranking_gates) if ctx.ranking_gates else list(gates),
        job_timing=timing,
        # budget accounting (running-time budget): elapsed_minutes above is RUNNING time
        running_elapsed_minutes=ctx.elapsed() / 60.0,
        wall_elapsed_minutes=ctx.wall_elapsed() / 60.0,
        queue_only_minutes=ctx.queue_wait_s / 60.0,
        max_wall_minutes=(ctx.max_wall_s / 60.0) if ctx.max_wall_s else None,
        total_queue_s=sum(t["queue_s"] for t in timing.values() if t["queue_s"] is not None),
        total_job_run_s=sum(t["run_s"] for t in timing.values() if t["run_s"] is not None),
        full_run_prediction=_predicted_winner_text(ranked[0]) if ranked else "no ranking",
        ranking_start_note=ctx.ranking_start_note)


def _predicted_winner_text(entry):
    res = entry["result"]
    lines = ["=== full-run prediction (rank_by=predicted, %s, %d rebuilt point(s)) ==="
             % (entry["cid"], len(res["points"])),
             "compute_total: %s   init: %.1fs   TOTAL: %s (range %s - %s, +-%.1f%% drift)"
             % (_fmt_hms(res["compute_total_s"]), res["init_s"], _fmt_hms(res["total_s"]),
                _fmt_hms(res["range_low_s"]), _fmt_hms(res["range_high_s"]), res["drift_pct"]),
             "r_rad: night=%.3f day=%.3f source=%s" % (res["r_night"], res["r_day"],
                                                        res["r_source"])]
    if res.get("init_note"):
        lines.append("NOTE: " + res["init_note"])
    return "\n".join(lines)


def render_predicted_report_text(report):
    lines = ["=== wtune_tune (rank_by=predicted): %s ===" % report["case"],
             "nodes=%d (%s)" % (report["nodes"], report["nodes_note"]),
             "budget=%.1f min (running time)  running=%.1f min  wall=%.1f min  queue-only=%.1f min"
             "  hard wall cap=%s  jobs=%s"
             % (report["budget_minutes"], report["running_elapsed_minutes"],
                report["wall_elapsed_minutes"], report["queue_only_minutes"],
                ("%.1f min" % report["max_wall_minutes"]) if report.get("max_wall_minutes")
                else "-", ",".join(report["job_ids"]) or "-")]
    for p in report["points"]:
        lines.append("point %d %s %-8s case=%s start=%s rebuild_job=%s%s"
                     % (p["index"], p["at"], p["reason"], p["case_id"], p["start"],
                        p["rebuild_job"], "  [P*]" if p["primary"] else ""))
    lines.append("")
    lines.append("%-10s %12s %21s %6s %4s %5s %5s %5s %9s %10s"
                 % ("id", "pred_total", "range", "ranks", "omp", "nx", "ny", "tiles",
                    "pred_spd", "stage1_s"))
    for row in report["ranking"]:
        lo, hi = row["predicted_range_s"]
        lines.append("%-10s %12s %21s %6s %4s %5s %5s %5s %9s %10s"
                     % (row["id"], row["predicted_total_hms"],
                        "%s-%s" % (_fmt_hms(lo), _fmt_hms(hi)), row["ranks"], row["omp"],
                        row["nproc_x"], row["nproc_y"], row["numtiles"],
                        ("%.3fx" % row["speedup_vs_anchor_predicted"])
                        if row["speedup_vs_anchor_predicted"] else "-",
                        ("%.4f" % row["stage1_median_s_per_step"])
                        if row["stage1_median_s_per_step"] else "-"))
        for pt in row["points"]:
            lines.append("    %s s_normal=%.4f r=%s n_rad=%d"
                         % (pt["at"], pt["s_normal"],
                            ("%.3f" % pt["r"]) if pt["r"] is not None else "-", pt["n_rad"]))
    if report.get("stage1_ranking") is not None:
        lines.append("")
        lines.append("stage 1=%s (spin-up %s min; key = median / own job's mean(ANCHOR, "
                     "ANCHOREND)):" % (report.get("stage1_mode", "-"), report.get("spinup_min", "-")))
        for e in report["stage1_ranking"]:
            lines.append("  %-10s norm=%.4f raw=%.4fs/step jobs=%s%s"
                         % (e["id"], e["norm_key"], e["median_s_per_step"], ",".join(e["jobs"]),
                            "  -> stage 2" if e["stage2"] else ""))
        for x in report.get("stage1_excluded") or []:
            lines.append("  %-10s EXCLUDED: %s" % (x["cid"], x["reason"]))
        for x in report.get("not_measured") or []:
            lines.append("  not measured (job %s never started before the stage-1 cutoff): %s"
                         % (x["job_id"], ", ".join(x["cids"]) or "-"))
    if report.get("stage1_running_min") is not None:
        lines.append("")
        lines.append("running time: stage 1 %.1f min + stage 2 %.1f min (estimated %.1f at start"
                     "%s) + rebuild-prep submit %.1f min = %.1f min  [%s]"
                     % (report["stage1_running_min"], report["stage2_running_min"],
                        report["stage2_estimate_min"],
                        (", cap %.1f" % report["stage2_minutes_cap"])
                        if report.get("stage2_minutes_cap") else "",
                        report.get("prep_submit_min") or 0.0, report["total_running_min"],
                        report.get("budget_semantics", "")))
    if report.get("stage2_overrun_s"):
        lines.append("")
        lines.append("STAGE 2 OVERRUN: %s of running time past the budget (stage 2 is exempt once "
                     "started; stage-2 concurrency assumed %s, stage-1 max observed running %s)"
                     % (_fmt_hms(report["stage2_overrun_s"]), report.get("stage2_concurrency"),
                        report.get("stage1_max_running")))
    if report.get("stage2_skipped_reason"):
        lines.append("")
        lines.append("STAGE 2 SKIPPED: %s" % report["stage2_skipped_reason"])
    lines.append("")
    for g in report["gates"]:
        if g.get("excluded"):
            lines.append("gate stage1   job %s: EXCLUDED  %s" % (g["job_id"], g["exclude_reason"]))
        drift = ("%+.2f%%" % (100 * g["anchor_drift"])) if g.get("anchor_drift") is not None else "-"
        lines.append("gate %-8s job %s: %s anchor_drift=%s%s"
                     % (g.get("stage", "-"), g["job_id"], "PASS" if g["ok"] else "FAIL", drift,
                        ("  " + "; ".join(g["failures"])) if g["failures"] else ""))
    lines.append("")
    lines.append("jobs: total queue %s, total running %s (per job below)"
                 % (_fmt_hms(report.get("total_queue_s") or 0),
                    _fmt_hms(report.get("total_job_run_s") or 0)))
    for jid, t in sorted(report["job_timing"].items()):
        lines.append("job %s: queue=%s run=%s" % (
            jid, _fmt_hms(t["queue_s"]) if t["queue_s"] is not None else "-",
            _fmt_hms(t["run_s"]) if t["run_s"] is not None else "-"))
    if report["best"]:
        pt = report["param_table"]
        lines += ["", "BEST: %s" % report["best"]["id"],
                  "  namelist: " + " ".join(pt["namelist_lines"]),
                  "  launch:   " + pt["launch_line"],
                  "  env:      " + " ".join("%s=%s" % kv for kv in pt["env"].items())
                  + "  " + pt.get("ulimit", "ulimit -s unlimited")]
    lines += ["", report["full_run_prediction"]]
    for n in report.get("plan_notes") or []:
        lines.append("NOTE: " + n)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# top-level run
# ---------------------------------------------------------------------------
def _cfg_key_explicit(config_path, section, key):
    """True when the --config YAML itself sets <section>.<key> (not just the built-in default)."""
    if not config_path:
        return False
    try:
        yaml = wtune_config._load_yaml_module()
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return key in (raw.get(section) or {})
    except Exception:
        return False


def _probe_key_explicit(config_path, key):
    """True when the --config YAML itself sets probe.<key> (not just the built-in default)."""
    return _cfg_key_explicit(config_path, "probe", key)


def _final_cancel_safety_net(ctx):
    """Nothing owned by this run should still be in flight once we report: past the budget
    deadline, cancel whatever of ctx.owned_jobs sacct still shows PENDING/RUNNING."""
    still_up = [j for j in ctx.owned_jobs if j not in ctx.cancelled_jobs]
    if still_up:
        states = sweep.sacct_states(still_up, sacct_runner=ctx.sacct_runner,
                                     sacct_cmd=ctx.sacct_cmd)
        in_flight = [j for j in still_up
                     if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES]
        if in_flight and ctx.elapsed() >= ctx.budget_s:
            cancel_jobs(ctx, in_flight)


# ---------------------------------------------------------------------------
# node search (--nodes search, default): MEASURE s/step at several node counts, choose N by an
# objective (wtune_nodes.py), then run the usual round0/adaptive/confirm search at each finalist N
# ---------------------------------------------------------------------------
NODE_SWEEP_REPS = 2
DEFAULT_SWEEP_BUDGET_FRAC = 0.4       # share of --minutes the node sweep may use


def _total_steps(namelist_path):
    """Total d01 steps of the full run (run length / time_step, wtune_predict.get_d01_steps), or
    None when unavailable (no wtune_predict, unreadable namelist)."""
    if wtune_predict is None:
        return None
    try:
        steps_n, _run_s, _dt, _note = wtune_predict.get_d01_steps(
            wtune_predict.parse_namelist(namelist_path))
        return int(steps_n) if steps_n and steps_n > 0 else None
    except Exception:
        return None


def cores_used(ctx):
    """Cores per node the decomposition search may use: ctx.core_budget when a node-search
    finalist runs undersubscribed, else the hardware's cores_per_node."""
    return ctx.core_budget or ctx.cores_per_node


def _sweep_omp_options(ctx, nodes, cores=None):
    """The allowed OpenMP settings at `nodes` x `cores` cores/node: decomp_space's own omp set
    (divisors of the NUMA width whose rank count total/omp packs a whole number of ranks per
    node), minus anything above the extension OMP cap (dmpar-only builds)."""
    opts = decomp_space.default_omp_options(nodes * (cores or ctx.cores_per_node), nodes,
                                            ctx.hardware_id)
    cap = getattr(ctx, "omp_cap", None)
    return [o for o in opts if not cap or o <= cap]


def _hybrid_anchor(ctx, nodes, omp, cores=None):
    """ANCHOR at `nodes` using `cores` (default: all) cores per node: ranks = nodes*cores/omp,
    numtiles = omp, the least-aspect legal (nx, ny), dist=block (same conventions as
    scc_sweep.build_anchor). None when no (nx, ny) is legal for that rank count
    (decomp_space.legal_shapes_exact, every active domain)."""
    total = nodes * (cores or ctx.cores_per_node)
    cands, _axis = decomp_space.enumerate_space(
        ctx.case_json, total_cores=total, nodes=nodes, hardware_id=ctx.hardware_id,
        omp_options=[omp],
        tile_options=dict(numtiles_multiples=(1,), max_numtiles=omp, include_2d=False))
    cands = [c for c in cands if c["numtiles"] == omp and c["dist"] == "cyclic"
             and c["blocktime"] == "0"]
    if not cands:
        return None
    we_list, sn_list = campaign_configs._active_domain_extents(ctx.case_json)

    def aspect(c):
        px, _ = decomp_space.rsl_patch_extents(we_list[0], c["nx"])
        py, _ = decomp_space.rsl_patch_extents(sn_list[0], c["ny"])
        return (max(px, py) / float(min(px, py)), c["nx"], c["ny"])

    base = min(cands, key=aspect)
    return dict(
        id="ANCHOR", ranks=base["ranks"], omp=omp, nx=base["nx"], ny=base["ny"],
        numtiles=base["numtiles"], dist="block", blocktime=base["blocktime"],
        patch_x=base.get("patch_x"), patch_y=base.get("patch_y"),
        numtiles_x=base.get("numtiles_x", 0), numtiles_y=base.get("numtiles_y", 0),
        tile_sz_x=base.get("tile_sz_x", 0), tile_sz_y=base.get("tile_sz_y", 0),
        tile_strategy=base.get("tile_strategy", 0), tile_shape="1d_y", config_source="anchor")


def _sweep_anchor(ctx, nodes, cores=None):
    """(anchor config | None, reason). `nodes` x `cores` is a candidate when SOME allowed omp (see
    _sweep_omp_options) makes ranks = nodes*cores/omp a legal decomposition -- pure MPI (omp=1)
    is not required, so small grids still scale out hybrid. The ANCHOR uses the SMALLEST such omp
    (that also passes the shared legality filter: WRF-Chem j-patch guard, --config exclude
    rules). The ranks-per-node x omp = cores product is what the launcher places
    (mpirun --map-by ppr:<ranks/node>:node:PE=<omp>); the other cores of the node stay idle.
    KNOWN LIMITATION: OMPI's ppr binding is packed -- an undersubscribed run (c < cores per node)
    occupies cores [0, c) and idles [c, C); it is not spread evenly across the node. Whether a
    spread binding (e.g. a pe-list) is faster is unmeasured; scc.sh's site_launch is unchanged."""
    cn = dataclasses.replace(ctx, nodes=nodes)
    omps = _sweep_omp_options(ctx, nodes, cores)
    msgs = []
    for omp in sorted(omps):
        anchor = _hybrid_anchor(ctx, nodes, omp, cores)
        if anchor is None:
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            kept = _apply_search_filter(cn, [anchor])
        if kept:
            return kept[0], None
        msgs.extend(l.strip() for l in buf.getvalue().splitlines() if l.strip())
    if msgs:
        return None, "ANCHOR rejected by the legality filter (%s)" % "; ".join(msgs)
    return None, "no legal decomposition at any omp in {%s}" % ", ".join(str(o) for o in omps)


_RSS_RE = re.compile(r"^(\d+(?:\.\d+)?)([KMGT]?)$")
_RSS_TO_KB = {"": 1.0, "K": 1.0, "M": 1024.0, "G": 1024.0 ** 2, "T": 1024.0 ** 3}
MAXRSS_NOTE = ("MaxRSS is whatever Slurm accounting reports for the job's steps; mpirun-launched "
               "ranks can bypass step accounting (f1: knowledge/hpc/2026-08-20-f1-reports-no-"
               "rapl-energy-and-no-useful-maxrss.md -- it reported the batch script's 6 MB), so "
               "a small value means 'not accounted', not 'small footprint'")


def _maxrss_mb_by_job(ctx, job_ids):
    """{job_id: MaxRSS in MB | None} from ONE sacct call over the job's steps (no -X). None when
    sacct returns no usable MaxRSS for that job (nothing is invented)."""
    job_ids = [str(j) for j in job_ids]
    best = {}
    try:
        out = ctx.sacct_runner([ctx.sacct_cmd, "-j", ",".join(job_ids), "-n", "-P",
                                "--format=JobID,MaxRSS"])
    except Exception:
        out = ""
    for line in (out or "").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        m = _RSS_RE.match(parts[1].strip())
        if not m:
            continue
        kb = float(m.group(1)) * _RSS_TO_KB[m.group(2)]
        jid = parts[0].strip().split(".")[0]
        if kb > 0:
            best[jid] = max(best.get(jid, 0.0), kb)
    return {j: (best[j] / 1024.0 if j in best else None) for j in job_ids}


def _pack_items(items, max_nodes):
    """Greedy first-fit ascending packing of jobs (items with ['nodes']) into concurrent waves
    whose node sums are <= max_nodes (same rule as wtune_nodes.sweep_waves)."""
    waves = []
    for it in sorted(items, key=lambda x: (x["nodes"], -x.get("cores", 0))):
        for w in waves:
            if sum(i["nodes"] for i in w) + it["nodes"] <= max_nodes:
                w.append(it)
                break
        else:
            waves.append([it])
    return waves


def _measure_anchor_wave(ctx, items, s_est_for, skipped):
    """Submit one REPS=2 job per item (dict: key, nodes, anchor, suffix, tag), skipping any whose
    estimated end would exceed the budget, wait for all, read the ledger. Returns {key: dict(agg,
    job, maxrss_mb)} for the items that were submitted AND produced a measurement."""
    jobs = {}
    for it in items:
        n = it["nodes"]
        cn = dataclasses.replace(ctx, nodes=n)
        s_est, src = s_est_for(n)
        est = _job_time_est(cn, 1, NODE_SWEEP_REPS, s_est)
        if would_exceed_budget(ctx.elapsed(), est["total_run_s"], ctx.budget_s):
            why = ("estimated end would exceed the budget (job ~%s at %.3fs/step[%s], elapsed %s "
                   "/ budget %s)" % (_fmt_hms(est["total_run_s"]), s_est, src,
                                     _fmt_hms(ctx.elapsed()), _fmt_hms(ctx.budget_s)))
            skipped.append(dict(it["tag"], reason=why))
            print("node sweep: %s skipped -- %s" % (it["suffix"], why))
            continue
        job_id, _p, _te = submit_configs_job(cn, [it["anchor"]], reps=NODE_SWEEP_REPS,
                                               name_suffix=it["suffix"], s_est=s_est, time_est=est)
        jobs[it["key"]] = job_id
        a = it["anchor"]
        print("node sweep: %s submitted job %s (ANCHOR ranks=%d omp=%d %dx%d, REPS=%d, ~%s)"
              % (it["suffix"], job_id, a["ranks"], a["omp"], a["nx"], a["ny"], NODE_SWEEP_REPS,
                 _fmt_hms(est["total_run_s"])))
    out = {}
    if not jobs:
        return out
    ids = list(jobs.values())
    states = wait_for_jobs(ctx, ids, "node-sweep")
    rows = ctx.ledger_reader(ctx.case, ids)
    rss = _maxrss_mb_by_job(ctx, ids)
    for it in items:
        j = jobs.get(it["key"])
        if j is None:
            continue
        mine = [r for r in rows if (parse_run_id(r.get("run_id")) or {}).get("job") == j]
        agg = aggregate_by_cid(mine, {"ANCHOR": it["anchor"]}).get("ANCHOR")
        if agg:
            out[it["key"]] = dict(agg=agg, job=j, maxrss_mb=rss.get(j))
        else:
            skipped.append(dict(it["tag"], reason="no successful measurement (job %s, state %s)"
                                % (j, states.get(j, ("UNKNOWN", ""))[0])))
    return out


def run_node_sweep(ctx, legal_nodes, budget_frac=DEFAULT_SWEEP_BUDGET_FRAC, chooser=None,
                   on_wave=None, refine=lambda: True):
    """Stage 'node sweep', coarse to fine. Candidates: every N in `legal_nodes` (<= max_nodes)
    that has a legal ANCHOR layout (_sweep_anchor). Seeds {1,2,4,8,max_nodes} (wtune_nodes.
    seed_nodes) are measured first, one REPS=2 ANCHOR job per N, in ascending waves with early
    stop on the upward side; then, around the current best N* (`chooser(rows)`, default fastest),
    the nearest legal untested N on each side inside the bracket of known N is measured, round
    after round, until the best's legal neighbours are measured. Everything not measured is
    reported in `skipped` with its reason. The `budget_frac` cap stops further submissions. Jobs
    are ordinary strict_campaign jobs: their rows land in the same ledger_<case>_<job>.csv files
    (n_nodes column) round-0 rows do. Returns dict(measurements, anchors, results, skipped, jobs,
    legal, maxrss_mb)."""
    chooser = chooser or (lambda rows: wtune_nodes.choose(rows, "time")[0])
    skipped, anchors = [], {}
    for n in wtune_nodes.candidate_node_counts(legal_nodes, ctx.max_nodes):
        anchor, why = _sweep_anchor(ctx, n)
        if anchor is None:
            skipped.append(dict(nodes=n, reason=why))
            print("node sweep: N=%d skipped -- %s" % (n, why))
        else:
            anchors[n] = anchor
    h = ctx.node_health or {}
    for n in range(ctx.max_nodes + 1, int(h.get("max_nodes_requested") or 0) + 1):
        skipped.append(dict(nodes=n, reason="skipped (only %d healthy nodes)" % h["healthy"]))
    legal = sorted(anchors)
    meas, results, jobs, rss = {}, {}, {}, {}
    attempted = set()
    seeds = wtune_nodes.seed_nodes(legal, ctx.max_nodes) if legal else []
    sweep_t0 = ctx.elapsed()
    cap_s = budget_frac * max(0.0, ctx.budget_s - sweep_t0)   # share of what is left
    state = dict(stopped=None)

    def s_est_for(n):
        if meas:
            near = min(meas, key=lambda m: (abs(m - n), m))
            return statistics.median(meas[near]), "median at N=%d" % near
        s_est, src, _k = sweep.estimate_s_per_step(ctx.case, n, ctx.ledger_dir, ctx.hardware_id)
        return s_est, src

    def cap_reason():
        if ctx.elapsed() - sweep_t0 >= cap_s:
            return ("sweep budget cap: sweep used %s >= %.0f%% of the budget (%s)"
                    % (_fmt_hms(ctx.elapsed() - sweep_t0), budget_frac * 100, _fmt_hms(cap_s)))
        return None

    def run_wave(ns):
        items = [dict(key=n, nodes=n, anchor=anchors[n], suffix="nodes-n%d" % n, tag=dict(nodes=n))
                 for n in ns]
        attempted.update(ns)
        for n, r in _measure_anchor_wave(ctx, items, s_est_for, skipped).items():
            meas[n], results[n], jobs[n], rss[n] = (r["agg"]["s_values"], r["agg"], r["job"],
                                                    r["maxrss_mb"])

    # phase 1: geometric seeds, early stop on the upward side
    for wave in wtune_nodes.sweep_waves(seeds, ctx.max_nodes):
        if not state["stopped"]:
            state["stopped"] = cap_reason()
            if state["stopped"]:
                print("node sweep: %s; larger N not submitted" % state["stopped"])
        if state["stopped"]:
            attempted.update(wave)
            skipped.extend(dict(nodes=n, reason=state["stopped"]) for n in wave)
            continue
        run_wave(wave)
        if on_wave is not None and not state.get("halt"):
            halt = on_wave(meas)
            if halt:                  # e.g. the short-run guard: no more waves, no refinement
                state["stopped"] = state["halt"] = halt
                continue
        srows = wtune_nodes.summarize(meas)
        if wtune_nodes.stop_early(srows):
            a, b = srows[-2], srows[-1]
            state["stopped"] = ("early stop: N=%d (%.4f s/step) did not beat N=%d (%.4f s/step) "
                                "by >%.0f%%" % (b["nodes"], b["median"], a["nodes"], a["median"],
                                                wtune_nodes.NOISE_FRAC * 100))
            print("node sweep: %s; larger N not submitted" % state["stopped"])

    # phase 2: refine around the best N
    cap_hit = state.get("halt")
    while meas and not state.get("halt") and refine():
        best = chooser(wtune_nodes.summarize(meas))
        todo = wtune_nodes.refine_targets(best, set(seeds) | set(meas), legal, attempted)
        if not todo:
            break
        cap_hit = cap_reason()
        if cap_hit:
            print("node sweep: %s; refinement stopped" % cap_hit)
            break
        print("node sweep: refining around best N=%d: measuring N=%s" % (best, todo))
        for wave in wtune_nodes.sweep_waves(todo, ctx.max_nodes):
            run_wave(wave)

    # everything legal that was never measured gets an explicit reason
    rows = wtune_nodes.summarize(meas)
    best = chooser(rows) if rows else None
    lo, hi = wtune_nodes.bracket(best, set(seeds) | set(meas)) if best else (0, None)
    already = {s["nodes"] for s in skipped}
    for n in legal:
        if n in meas or n in already:
            continue
        if cap_hit:
            why = cap_hit
        elif not refine():
            why = "not measured: refinement is off at this level (O1 measures only {1,2,4,8,max})"
        elif best is None:
            why = "not measured: no seed measurement succeeded"
        elif lo < n and (hi is None or n < hi):
            why = ("not measured: not adjacent to the best N=%d (inside the refinement bracket "
                   "%d..%s)" % (best, lo, hi))
        else:
            why = ("not measured: outside refinement bracket (%d..%s around the best N=%d)"
                   % (lo, hi, best))
        skipped.append(dict(nodes=n, reason=why))
    skipped.sort(key=lambda s: (s["nodes"], s.get("cores", 0)))
    return dict(measurements=meas, anchors=anchors, results=results, skipped=skipped, jobs=jobs,
                legal=legal, maxrss_mb=rss)


def _core_row(cores, nodes, anchor, agg, maxrss_mb, chosen=False):
    return dict(cores=cores, ranks_per_node=anchor["ranks"] // nodes, omp=anchor["omp"],
                ranks=anchor["ranks"], nx=anchor["nx"], ny=anchor["ny"], median=agg["median"],
                n=agg["n"], spread=((agg["hi"] - agg["lo"]) / agg["median"]
                                    if agg["n"] >= 2 and agg["median"] else 0.0),
                maxrss_mb=maxrss_mb, chosen=chosen)


def run_core_sweep(ctx, finals, sw, core_options, budget_frac=DEFAULT_SWEEP_BUDGET_FRAC / 2):
    """Cores-per-node dimension at each finalist N: the hybrid ANCHOR using c < C cores per node
    (ranks/node x omp = c; the rest idle) is measured as one REPS=2 job per (N, c), for every c in
    `core_options` with a legal layout; c=C reuses the node sweep's own measurement. c is chosen
    per N by wtune_nodes.choose_cores (fastest, ties within noise -> fuller c). Returns
    ({N: dict(anchors={c: anchor}, rows=[...], chosen=c, note=...)}, skipped_cores)."""
    C = ctx.cores_per_node
    info, skipped, items = {}, [], []
    for n in finals:
        info[n] = dict(anchors={C: sw["anchors"][n]}, results={C: sw["results"][n]},
                       rss={C: sw["maxrss_mb"].get(n)})
        for c in sorted((c for c in core_options if c < C), reverse=True):
            a, why = _sweep_anchor(ctx, n, cores=c)
            if a is None:
                skipped.append(dict(nodes=n, cores=c, reason=why))
                print("core sweep: N=%d c=%d skipped -- %s" % (n, c, why))
                continue
            info[n]["anchors"][c] = a
            items.append(dict(key=(n, c), nodes=n, cores=c, anchor=a,
                              suffix="cores-n%d-c%d" % (n, c), tag=dict(nodes=n, cores=c)))
    t0 = ctx.elapsed()
    cap_s = budget_frac * max(0.0, ctx.budget_s - t0)

    def s_est_for(n):
        return statistics.median(sw["measurements"][n]), "node-sweep median at N=%d" % n

    for wave in _pack_items(items, ctx.max_nodes):
        if ctx.elapsed() - t0 >= cap_s:
            why = "core sweep budget cap: used %s >= %.0f%% of the budget" % (
                _fmt_hms(ctx.elapsed() - t0), budget_frac * 100)
            skipped.extend(dict(it["tag"], reason=why) for it in wave)
            continue
        for key, r in _measure_anchor_wave(ctx, wave, s_est_for, skipped).items():
            info[key[0]]["results"][key[1]] = r["agg"]
            info[key[0]]["rss"][key[1]] = r["maxrss_mb"]
    for n in finals:
        rows = [_core_row(c, n, info[n]["anchors"][c], info[n]["results"][c], info[n]["rss"][c])
                for c in sorted(info[n]["results"], reverse=True)]
        chosen, note = wtune_nodes.choose_cores(rows)
        for r in rows:
            r["chosen"] = (r["cores"] == chosen)
        info[n].update(rows=rows, chosen=chosen, note=note)
        print("core sweep: N=%d chose cores/node=%d -- %s" % (n, chosen, note))
    skipped.sort(key=lambda s: (s["nodes"], -s["cores"]))
    return info, skipped


def node_choices(rows, *, deadline_s, total_steps, overhead_s, min_efficiency):
    """{objective: {nodes, note}} for all three objectives (cost needs total_steps)."""
    out = {}
    for obj in wtune_nodes.OBJECTIVES:
        try:
            n, note = wtune_nodes.choose(rows, obj, deadline_s=deadline_s, total_steps=total_steps,
                                         overhead_s=overhead_s, min_efficiency=min_efficiency)
            out[obj] = dict(nodes=n, note=note)
        except ValueError as e:
            out[obj] = dict(nodes=None, note="unavailable: %s" % e)
    return out


def run_node_search(ctx, profile, objective, deadline_s, min_efficiency, k_finalists,
                    total_steps, nodes_policy="knee", sweep_budget_frac=DEFAULT_SWEEP_BUDGET_FRAC,
                    core_options=None, on_sweep=None, collect=None, on_wave=None, plan=None):
    """Node sweep (coarse to fine over every legal N in 1..max_nodes) -> choose N -> cores-per-node
    sweep at the finalists -> round0/adaptive/confirm at each finalist (N, c) (remaining budget
    split evenly over the finalists still to run) -> final (N, c, config) by the same objective
    over the finalists' best tuned medians. Falls back to the proxy path when nothing was
    measured. `profile` is only used by that fallback."""
    lvl = (lambda: plan["level"]) if plan else (lambda: 3)    # effort level, read live
    overhead_s = ctx.startup_s
    kw = dict(deadline_s=deadline_s, total_steps=total_steps, overhead_s=overhead_s,
              min_efficiency=min_efficiency)
    sw = run_node_sweep(ctx, list(range(1, ctx.max_nodes + 1)), budget_frac=sweep_budget_frac,
                        chooser=lambda rows: wtune_nodes.choose(rows, objective, **kw)[0],
                        on_wave=on_wave, refine=lambda: lvl() >= 2)
    rows = wtune_nodes.summarize(sw["measurements"])
    if not rows:
        nodes, nodes_note, warning = choose_nodes(profile, ctx.max_nodes, nodes_policy)
        print("WARNING: node sweep produced no measurement; falling back to the proxy. " + warning)
        fctx = dataclasses.replace(ctx, nodes=nodes)
        res = run_tuning_at(fctx)
        report = build_report(ctx, nodes, "node sweep failed; " + nodes_note, *res)
        report["node_selection"] = dict(mode="search", objective=objective, chosen=nodes,
                                        table=[], choices={}, skipped=sw["skipped"], sweep_jobs=[],
                                        finalists=[nodes], note=nodes_note)
        return report

    choices = node_choices(rows, **kw)
    chosen, note = wtune_nodes.choose(rows, objective, **kw)
    finals = wtune_nodes.finalists(rows, objective, k=(1 if lvl() == 2 else k_finalists), **kw)
    tab = wtune_nodes.table(rows, total_steps=total_steps, overhead_s=overhead_s)
    for r in tab:
        r["maxrss_mb"] = sw["maxrss_mb"].get(r["nodes"])
    if on_sweep is not None:        # --pipeline full: re-estimate the budget from the measurement
        on_sweep(rows, sw)
    print("node sweep: chose N=%d -- %s; finalists %s" % (chosen, note, finals))
    for r in tab:
        print("node sweep: N=%-2d median=%.4f s/step n=%d spread=%.1f%% speedup=%.2f eff=%.2f"
              % (r["nodes"], r["median"], r["n"], r["spread"] * 100, r["speedup"],
                 r["efficiency"]))

    if core_options is None:
        core_options = wtune_nodes.default_core_options(ctx.cores_per_node)
    cores_info, skipped_cores = {}, []
    if lvl() <= 1:                       # O1: the t0 wave only -- no cores sweep, no finalists
        finals = []
    if lvl() >= 2 and any(c < ctx.cores_per_node for c in core_options):
        cores_info, skipped_cores = run_core_sweep(ctx, finals, sw, core_options)

    layout_notes = dict(untried=[], enumerated={})
    outcomes = {}                  # N -> dict(report, best_median, ...)
    for i, n in enumerate(finals):
        remaining = ctx.budget_s - ctx.elapsed()
        if remaining <= 0:
            print("node search: no budget left for finalist N=%d" % n)
            continue
        share = remaining / (len(finals) - i)
        c = cores_info[n]["chosen"] if n in cores_info else ctx.cores_per_node
        anchor = cores_info[n]["anchors"][c] if n in cores_info else sw["anchors"].get(n)
        fctx = dataclasses.replace(ctx, nodes=n, budget_s=ctx.elapsed() + share,
                                    out_dir=os.path.join(ctx.out_dir, "n%d" % n),
                                    anchor_override=anchor, core_budget=c,
                                    full_space=(plan is not None and lvl() >= 3), layout_notes=layout_notes)
        os.makedirs(fctx.out_dir, exist_ok=True)
        print("node search: finalist N=%d cores/node=%d, budget share %s"
              % (n, c, _fmt_hms(share)))
        all_results, confirm_results, anchor_cid = run_tuning_at(
            fctx, adaptive=(lvl() >= 3), confirm=(lvl() >= 3))      # O3: full space + confirm reps
        if not all_results:
            res = (cores_info[n]["results"][c] if n in cores_info else sw["results"].get(n))
            if res:
                print("node search: nothing measured at N=%d beyond the sweep; reporting the "
                      "sweep ANCHOR" % n)
                all_results, anchor_cid = {"ANCHOR": dict(res, config=anchor)}, "ANCHOR"
        if not all_results:
            continue
        rep = build_report(fctx, n, "node search finalist", all_results, confirm_results,
                           anchor_cid, full_prediction=False)   # built once, for the winner
        if rep["best"]:
            outcomes[n] = dict(report=rep, best_median=rep["best"]["median_s_per_step"],
                               ctx=fctx, all_results=all_results,
                               confirm_results=confirm_results, cores=c, anchor=anchor)
    if not outcomes:
        # budget gone before any finalist produced a result: report the sweep ANCHOR at `chosen`
        c = cores_info[chosen]["chosen"] if chosen in cores_info else ctx.cores_per_node
        anchor = cores_info[chosen]["anchors"][c] if chosen in cores_info else sw["anchors"][chosen]
        res = cores_info[chosen]["results"][c] if chosen in cores_info else sw["results"][chosen]
        fctx = dataclasses.replace(ctx, nodes=chosen, core_budget=c)
        rep = build_report(fctx, chosen, note, {"ANCHOR": dict(res, config=anchor)}, None,
                           "ANCHOR")
        outcomes[chosen] = dict(report=rep, best_median=res["median"], cores=c, anchor=anchor,
                                ctx=fctx, all_results={"ANCHOR": dict(res, config=anchor)},
                                confirm_results=None)
        final_n, final_note = chosen, note + (
            "; level O1: t0 sweep only -- ANCHOR config" if lvl() <= 1 else
            "; budget exhausted after the sweep -- ANCHOR config")
    else:
        by_sweep = {r["nodes"]: r for r in rows}
        frows = wtune_nodes.summarize({n: [o["best_median"]] for n, o in outcomes.items()})
        for r in frows:                     # scaling figures stay the sweep's (anchor-based)
            r["speedup"], r["efficiency"] = (by_sweep[r["nodes"]]["speedup"],
                                              by_sweep[r["nodes"]]["efficiency"])
        final_n, final_note = wtune_nodes.choose(frows, objective, **kw)
        final_note = "final over tuned finalists %s: %s" % (sorted(outcomes), final_note)
    if collect is not None:
        collect.update(outcomes=outcomes, final_n=final_n, sweep=sw, cores_info=cores_info)
    report = outcomes[final_n]["report"]
    final_c = outcomes[final_n]["cores"]
    if report.get("full_run_prediction") is None:       # winner of a finalist search
        o = outcomes[final_n]
        best_id = report["best"]["id"]
        entry = (o["confirm_results"] or {}).get(best_id) or o["all_results"].get(best_id)
        report["full_run_prediction"] = build_full_run_prediction(
            o["ctx"], entry["config"] if entry else None, final_n, o["all_results"],
            o["confirm_results"])
    report["nodes"], report["nodes_note"] = final_n, final_note
    report["cores_used_per_node"] = final_c
    report["budget_minutes"] = ctx.budget_s / 60.0 if math.isfinite(ctx.budget_s) else None
    report["elapsed_minutes"] = ctx.elapsed() / 60.0
    report["job_ids"] = list(ctx.owned_jobs)
    report["cancelled_job_ids"] = list(ctx.cancelled_jobs)
    layouts_tab = build_layout_table(sw, cores_info, outcomes, ctx)
    report["layout_table"] = layouts_tab
    report["node_selection"] = dict(
        mode="search", objective=objective, chosen=chosen, chosen_note=note, final=final_n,
        final_note=final_note, final_cores_per_node=final_c, table=tab, choices=choices,
        skipped=sw["skipped"], finalists=finals,
        finalist_best={str(n): o["best_median"] for n, o in outcomes.items()},
        core_tables={str(n): ci["rows"] for n, ci in cores_info.items()},
        core_choice={str(n): dict(cores=ci["chosen"], note=ci["note"])
                     for n, ci in cores_info.items()},
        skipped_cores=skipped_cores, core_options=list(core_options),
        total_steps=total_steps, deadline_s=deadline_s, min_efficiency=min_efficiency,
        overhead_s=overhead_s, sweep_jobs=list(sw["jobs"].values()), maxrss_note=MAXRSS_NOTE,
        layouts=layouts_tab["rows"], winner=layouts_tab["winner"],
        runner_up=layouts_tab["runner_up"], winner_vs_runner_up=layouts_tab["comparison"],
        untried=(sw["skipped"] + skipped_cores + layout_notes["untried"]),
        enumerated=layout_notes["enumerated"])
    return report


def _layout_row(source, nodes, cores, cfg, vals):
    med = statistics.median(vals)
    return dict(source=source, nodes=nodes, cores_per_node=cores, ranks=cfg.get("ranks"),
                omp=cfg.get("omp"), nx=cfg.get("nx"), ny=cfg.get("ny"),
                numtiles=cfg.get("numtiles"), median=med, reps=len(vals),
                spread=((max(vals) - min(vals)) / med if len(vals) >= 2 and med else 0.0),
                cid=cfg.get("id"))


def build_layout_table(sw, cores_info, outcomes, ctx):
    """Every MEASURED layout, best first: the t0 ANCHOR at each N, the anchor at each cores/node
    option, every enumerated layout measured at a finalist (confirm reps replace the single rep
    where they exist). Plus the winner (lowest median over the finalists' layouts), the runner-up
    (next best DIFFERENT layout) and whether they differ beyond the noise floor."""
    rows, seen = [], {}

    def add(row, key):
        old = seen.get(key)
        if old is None or row["reps"] > old["reps"] or (row["reps"] == old["reps"]
                                                       and row["median"] < old["median"]):
            seen[key] = row

    C = ctx.cores_per_node
    for n, agg in sw["results"].items():
        a = sw["anchors"][n]
        add(_layout_row("t0 sweep ANCHOR", n, C, a, agg["s_values"]),
            (n, C, sweep._identity(a)))
    for n, ci in cores_info.items():
        for c, agg in ci["results"].items():
            a = ci["anchors"][c]
            add(_layout_row("cores/node sweep ANCHOR", n, c, a, agg["s_values"]),
                (n, c, sweep._identity(a)))
    tuned = []
    for n, o in outcomes.items():
        merged = dict(o.get("all_results") or {})
        merged.update(o.get("confirm_results") or {})
        for cid, r in merged.items():
            cfg = r.get("config")
            if not cfg or not r.get("s_values"):
                continue
            row = _layout_row("finalist search", n, o["cores"], cfg, r["s_values"])
            add(row, (n, o["cores"], sweep._identity(cfg)))
            tuned.append((n, o["cores"], sweep._identity(cfg)))
    rows = sorted(seen.values(), key=lambda r: r["median"])
    pool = [seen[k] for k in dict.fromkeys(tuned)] or rows
    pool = sorted(pool, key=lambda r: r["median"])
    winner = pool[0] if pool else None
    runner = pool[1] if len(pool) > 1 else None
    comparison = None
    if winner and runner:
        gap = (runner["median"] - winner["median"]) / winner["median"]
        comparison = dict(gap=gap, noise=wtune_nodes.NOISE_FRAC,
                          beyond_noise=gap > wtune_nodes.NOISE_FRAC)
    return dict(rows=rows, winner=winner, runner_up=runner, comparison=comparison)


def render_node_selection_text(ns):
    def mb(v):
        return "%.0f" % v if v is not None else "-"

    lines = ["", "--- node selection (measured, ANCHOR config per node count) ---",
             "%5s %12s %5s %8s %8s %6s %10s %10s %9s"
             % ("nodes", "median_s", "reps", "spread", "speedup", "eff", "wall_s", "node_hours",
                "maxrss_MB")]
    for r in ns.get("table", []):
        lines.append("%5d %12.4f %5d %7.1f%% %7.2fx %6.2f %10s %10s %9s"
                     % (r["nodes"], r["median"], r["n"], r["spread"] * 100, r["speedup"],
                        r["efficiency"],
                        "%.0f" % r["wall_s"] if r.get("wall_s") is not None else "-",
                        "%.2f" % r["node_hours"] if r.get("node_hours") is not None else "-",
                        mb(r.get("maxrss_mb"))))
    for obj, c in (ns.get("choices") or {}).items():
        lines.append("choice %-5s%s N=%s  %s" % (obj, "*" if obj == ns.get("objective") else " ",
                                                 c["nodes"], c["note"]))
    lines.append("(* = objective used; finalists searched: %s; final N=%s cores/node=%s)"
                 % (ns.get("finalists"), ns.get("final", ns.get("chosen")),
                    ns.get("final_cores_per_node", "-")))
    for n, rows in sorted((ns.get("core_tables") or {}).items(), key=lambda kv: int(kv[0])):
        lines.append("")
        lines.append("--- cores per node at N=%s (ANCHOR; idle cores = hardware - c) ---" % n)
        lines.append("%7s %10s %4s %11s %6s %8s %9s"
                     % ("cores", "ranks/node", "omp", "median_s", "reps", "spread", "maxrss_MB"))
        for r in rows:
            lines.append("%7d %10d %4d %11.4f %6d %7.1f%% %9s %s"
                         % (r["cores"], r["ranks_per_node"], r["omp"], r["median"], r["n"],
                            r["spread"] * 100, mb(r.get("maxrss_mb")),
                            "<- chosen" if r.get("chosen") else ""))
    if ns.get("layouts"):
        rows = ns["layouts"]
        lines.append("")
        lines.append("--- measured layouts (%d; best first; the full table is in tune_report.json "
                     "node_selection.layouts) ---" % len(rows))
        lines.append("%4s %5s %6s %4s %9s %5s %11s %7s %5s  %s"
                     % ("N", "c/n", "ranks", "omp", "nx x ny", "tiles", "median_s", "spread",
                        "reps", "source"))
        for r in rows[:40]:
            lines.append("%4d %5d %6s %4s %9s %5s %11.4f %6.1f%% %5d  %s"
                         % (r["nodes"], r["cores_per_node"], r["ranks"], r["omp"],
                            "%sx%s" % (r["nx"], r["ny"]), r["numtiles"], r["median"],
                            r["spread"] * 100, r["reps"], r["source"]))
        if len(rows) > 40:
            lines.append("  ... %d more" % (len(rows) - 40))
        w, ru, cmp_ = ns.get("winner"), ns.get("runner_up"), ns.get("winner_vs_runner_up")
        fmt = lambda r: ("N=%d c/n=%d ranks=%s omp=%s %sx%s tiles=%s %.4f s/step"   # noqa: E731
                         % (r["nodes"], r["cores_per_node"], r["ranks"], r["omp"], r["nx"], r["ny"],
                            r["numtiles"], r["median"]))
        if w:
            lines.append("WINNER:    " + fmt(w))
        if ru:
            lines.append("RUNNER-UP: " + fmt(ru))
        if cmp_:
            lines.append("winner vs runner-up: %.1f%% apart -> %s (noise floor %.0f%%)"
                         % (cmp_["gap"] * 100,
                            "differ BEYOND noise" if cmp_["beyond_noise"]
                            else "NOT distinguishable from noise: either is fine",
                            cmp_["noise"] * 100))
    for k, v in sorted((ns.get("enumerated") or {}).items()):
        lines.append("enumerated %s: %d raw, %d unique (block, 1-D tiles), %d legal, %d excluded%s"
                     % (k, v["raw"], v["unique"], v["legal"], v["excluded"],
                        " (%s)" % "; ".join(v["excluded_reasons"][:2]) if v["excluded_reasons"]
                        else ""))
    untried = ns.get("untried") or []
    if untried:
        reasons = {}
        for u in untried:
            reasons.setdefault(u["reason"], []).append(u)
        lines.append("NOT TRIED: %d (reasons below; full list in node_selection.untried)"
                     % len(untried))
        for why, us in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
            lines.append("  %4d x %s" % (len(us), why))
    if ns.get("table") and any(r.get("maxrss_mb") is not None for r in ns["table"]):
        lines.append("NOTE: " + ns.get("maxrss_note", MAXRSS_NOTE))
    for s in ns.get("skipped", []) + ns.get("skipped_cores", []):
        lines.append("not run N=%d%s: %s" % (s["nodes"],
                                              " c=%d" % s["cores"] if "cores" in s else "",
                                              s["reason"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --pipeline full (decision 0059): one pass = layout search (node, cores/node, ranks, threads,
# decomposition, tiles) -> whole-run prediction from rebuilt points -> physics advice, under a hard
# wall-clock budget B = --minutes
# ---------------------------------------------------------------------------
try:
    import wtune_physics_advice     # src/wtune_physics_advice.py (plan_candidates, predicted_speedup,
                                    # fidelity_verdict, advice_table); best-effort, like wtune_predict
except ImportError:
    wtune_physics_advice = None

PIPELINE_FRACS = (("plan", 0.05), ("layout", 0.50), ("predict", 0.25), ("physics", 0.20))
PIPELINE_MAX_QUEUE_HOURS = 6.0        # --max-queue-hours: stop once queue-only wait reaches this
PIPELINE_DEFAULT_MINUTES = 30          # --pipeline full: B when neither CLI nor YAML gives one
PIPELINE_PHYSICS_MAX_CANDIDATES = 3
PIPELINE_PHYSICS_POINTS = 2            # phase 3 measures t0 and the costliest predicted point
PHYSICS_RUNNER_MARKER = "physics_namelists"   # strict_campaign.sbatch must read a per-config
                                          # {case id: namelist path} map for phase 3 to be runnable
NOT_RUN_BUDGET = "not run (budget)"


def physics_runner_supported(sbatch_script):
    """True when the campaign runner reads per-config physics edits (PHYSICS_RUNNER_MARKER appears
    in strict_campaign.sbatch / configs.py). Phase 3 needs it to run namelist-edited candidates
    from a shared restart; today's CONFIG_FILE path only carries decomposition fields, and
    MODE=physics starts from t0 with a twin run, so this is False until the runner grows it."""
    base = os.path.dirname(os.path.abspath(sweep._expand(sbatch_script)))
    for name in (os.path.basename(sbatch_script), "configs.py"):
        try:
            with open(os.path.join(base, name)) as f:
                if PHYSICS_RUNNER_MARKER in f.read():
                    return True
        except OSError:
            continue
    return False


def _namelist_dt(ctx):
    """d01 time step (s) of the case's namelist."""
    return wtune_predict.get_d01_steps(wtune_predict.parse_namelist(ctx.namelist_path))[2]


def _prediction_context(ctx):
    """(nl, run_start, run_end, dt, stepra, calibration, sample_points, case_dir) for
    combine()-based prediction (same derivations as build_full_run_prediction)."""
    nl = wtune_predict.parse_namelist(ctx.namelist_path)
    _steps_n, run_s, dt, _note = wtune_predict.get_d01_steps(nl)
    run_start = wtune_predict.get_time_control_datetime(nl, "start")
    run_end = run_start + datetime.timedelta(seconds=run_s)
    radt = wtune_predict.g(nl, "physics", "radt", None)
    stepra = wtune_predict.steps_per_radiation_call(float(radt), dt)
    calibration = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
    if ctx.prediction_cfg.get("drift_band_pct") is not None:
        calibration = dict(calibration, drift_band_pct=ctx.prediction_cfg["drift_band_pct"])
    case_dir = sweep._expand(ctx.case_json["prep_dir"])
    sample_points, _src = wtune_predict.read_wrfinput_sample_points(case_dir)
    return nl, run_start, run_end, dt, stepra, calibration, sample_points, case_dir


def pipeline_layouts(collect, k=2):
    """The top-k DISTINCT layouts of phase 1: [dict(nodes, cores, cfg, median, anchor)], the
    phase-1 final N's best first, then the best of every other finalist, then the next-best
    configs of the winner (identity = nodes, cores, decomposition)."""
    outs = collect["outcomes"]
    rows = []
    for n, o in outs.items():
        res = o.get("confirm_results") or o.get("all_results") or {}
        for cid, r in res.items():
            if r.get("config") and r.get("median"):
                rows.append(dict(nodes=n, cores=o["cores"], cfg=r["config"], median=r["median"],
                                 anchor=o.get("anchor"), cid=cid))
    final_n = collect["final_n"]
    rows.sort(key=lambda r: (r["nodes"] != final_n, r["median"]))
    firsts, rest, seen_n = [], [], set()
    for r in rows:
        (rest if r["nodes"] in seen_n else firsts).append(r)
        seen_n.add(r["nodes"])
    out, seen = [], set()
    for r in firsts + rest:
        ident = (r["nodes"], r["cores"], sweep._identity(r["cfg"]))
        if ident in seen:
            continue
        seen.add(ident)
        out.append(r)
        if len(out) >= k:
            break
    return out


def _mark(ctx):
    """(running clock, wall clock, accumulated queue-only wait) -- the budget runs on the first."""
    return dict(run=ctx.elapsed(), wall=ctx.wall_elapsed(), queue=ctx.queue_wait_s)


def _phase_record(name, m0, m1, limit_s, status, **extra):
    """start_s/end_s/limit_s are on the RUNNING clock (what the budget counts: time one of our jobs
    was RUNNING); wall_s is wall time, queue_wait_s the queue-only wait and controller_s the rest
    (controller-side work: submission, report building, predictions, fidelity gate, conversions)
    -- both excluded from the budget; running_s the budget the phase consumed."""
    running = max(0.0, m1["run"] - m0["run"])
    wall = max(0.0, m1["wall"] - m0["wall"])
    queue = max(0.0, m1["queue"] - m0["queue"])
    return dict(name=name, start_s=m0["run"], end_s=m1["run"], running_s=running, wall_s=wall,
                queue_wait_s=queue, controller_s=max(0.0, wall - running - queue),
                limit_s=limit_s, status=status, **extra)


def _attach_phase_timing(ctx, phases, job_slices):
    """Per-job queue vs run time per phase from sacct Submit/Start/End, for the jobs each phase
    SUBMITTED (the rebuild jobs belong to phase 0 although they run beside phase 1).
    queue_s_max/job_run_s_max are the longest single job; the *_sum fields add up concurrent jobs,
    so they can exceed the phase's wall time. None when sacct reports nothing."""
    all_ids = [j for ids in job_slices.values() for j in ids]
    timing = job_timing(ctx, all_ids)
    for ph in phases:
        ids = job_slices.get(ph["name"], [])
        q = [timing[j]["queue_s"] for j in ids if timing.get(j, {}).get("queue_s") is not None]
        r = [timing[j]["run_s"] for j in ids if timing.get(j, {}).get("run_s") is not None]
        ph.update(jobs=list(ids), queue_s_sum=sum(q) if q else None,
                  queue_s_max=max(q) if q else None, job_run_s_sum=sum(r) if r else None,
                  job_run_s_max=max(r) if r else None)


WINDOW_FRAME_LEVEL = "mid_end"        # per-config `window_frame`: mid + end d01 frames (the gate
                                      # needs a matched time after the reference's first; a
                                      # restarted run never writes a frame at its own start)
MIN_WINDOW_MIN = 2                    # window length (model minutes) mid+end frames need


def _run_dir(ctx, rp, cid, job_id):
    return os.path.join(ctx.ledger_dir, "%s_%s_r1_%s" % (rp["case_id"], cid, job_id))


def _window_frames(ctx, rp, cid, job_id):
    """The run's kept window frames (mid + end), sorted: the gate's input list."""
    return sorted(glob.glob(os.path.join(_run_dir(ctx, rp, cid, job_id), "wrfout_d01_*.nc")))


def _delete_window_frames(ctx, runs):
    """Delete the window-frame .nc files of `runs` [(rp, cid, job)]; returns how many."""
    n = 0
    for rp, cid, jid in runs:
        for f in _window_frames(ctx, rp, cid, jid):
            try:
                os.remove(f)
                n += 1
            except OSError:
                pass
    return n


def build_physics_window_namelists(ctx, cands, rsub, anchor):
    """Phase-3 inputs. For every point in `rsub` and the baseline + every candidate, write the
    full namelist of the point's window (from the shared restart at T + spin-up to the window
    end): candidate = wtune_physics_advice.render_candidate_namelist(point template, cand), then
    wtune_run.render_segment_namelist(text, restart time, window end, k=1, winner layout, no
    further restart) with d01 history_interval = the window length, io_form_history = 2 (NetCDF),
    so a frame is written exactly at the window end for baseline, twin and candidates alike.
    Files go under <out>/physics/pt_<i>/<id>/namelist.input; the winner layout's configs carry
    them as `physics_namelists` {point case id: path} (PHYSICS_RUNNER_MARKER). Returns
    (baseline anchor config, [candidate config, ...]). UNVERIFIED against the real runner (it does
    not read `physics_namelists` yet), unit-tested only through fakes."""
    import wtune_run
    api = wtune_physics_advice
    cfg = dict(ranks=anchor["ranks"], omp=anchor["omp"], nx=anchor["nx"], ny=anchor["ny"],
               numtiles=anchor["numtiles"])

    def write(rp, cid, text):
        d = os.path.join(ctx.out_dir, "physics", "pt_%d" % rp["index"], cid)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "namelist.input")
        with open(path, "w") as f:
            f.write(text)
        return path

    base_paths, cand_paths = {}, {c["id"]: {} for c in cands}
    for rp in rsub:
        with open(rp["template"]) as f:
            template = f.read()
        dt = float(rp["dt"])
        seg_start = rp["T"] + datetime.timedelta(minutes=rp["spinup_min"])
        wmin = int(math.ceil(rp["steps"] * dt / 60.0))
        seg_end = seg_start + datetime.timedelta(minutes=wmin)

        def window(text):
            txt = wtune_run.render_segment_namelist(text, seg_start, seg_end, 1, cfg,
                                                    wtune_run.RESTART_OFF_MIN,
                                                    restart_io_form=ctx.restart_io_form)
            txt = wtune_run._set_one(txt, "history_interval", wmin)
            return wtune_run._set_one(txt, "io_form_history", 2)

        base_paths[rp["case_id"]] = write(rp, "ANCHOR", window(template))
        for cd in cands:
            cand_paths[cd["id"]][rp["case_id"]] = write(
                rp, cd["id"], window(api.render_candidate_namelist(template, cd)))
    base = dict(anchor, physics_namelists=base_paths)
    return base, [dict(anchor, id=cd["id"], config_source="generated",
                       physics_namelists=cand_paths[cd["id"]]) for cd in cands]


def _merge_verdicts(verdicts):
    """One fidelity result per candidate from its per-point verdicts: the first failing point,
    else the one with the largest worst rel_rms; None when no point could be compared."""
    vs = [v for v in verdicts if v]
    if not vs:
        return None
    fails = [v for v in vs if not v.get("passed")]
    if fails:
        return fails[0]
    return max(vs, key=lambda v: (v["worst"][0]["rel_rms"] if v.get("worst") else 0.0))


# Effort levels (like compiler -O levels). They control ONLY the layout search effort (number of
# waves, refinement, decomposition/adaptive search) and whether physics advice runs. Which
# prediction points are run and the whole-run error band come from the case type
# (wtune_cost_drivers.plan_points_for_case), never from the level -- except O0, which runs no job
# and so carries the cold-start band.
LEVELS = {
    0: dict(name="O0", method="no jobs: cold-start predict + proxy layout"),
    1: dict(name="O1", method="one t0 wave over N in {1,2,4,8,max healthy}"),
    2: dict(name="O2", method="O1 + refinement, cores/node, sampled round0 at the winner"),
    3: dict(name="O3", method="DEFAULT, full layout search: every legal N (coarse-to-fine), "
                              "cores/node, ranks x omp, nx x ny, numtiles at the finalists + "
                              "confirm reps"),
}
DEFAULT_LEVEL = 3
PRECISION_DEFAULT_MIN = 15.0         # --precision: whole-run time needed to +-MIN minutes
# Whole-run wall-time error bands by prediction method -- the FALLBACK table used when
# wtune_cost_drivers is not importable. Conservative figures from this wiki's measurements:
# cold-start `wtune predict` held-out median error 5.8%, padded to +-25% for weather it cannot
# see; a t0-only probe underestimates ~9.6% on conus25 (spin-up makes hour 1 cheap) -> +-15%;
# missing the costliest point cost -13.6% on elevconv (+44% convective peak) -> t0 + costliest
# point +-10%; 3+ points through combine() <= ~7%.
COLD_START_BAND = 0.25
FALLBACK_BANDS = (                   # (rebuilt points, method, band)
    (0, "t0 anchor (phase-1 data)", 0.15),
    (1, "t0 + costliest point", 0.10),
    (3, "3+ points", 0.07),
)
TIME_STEP_S = 15 * 60.0              # --time is rounded UP to this granularity
# Pre-run effort model constants (documented guesses, replaced by measurements when they exist):
EST_REBUILD_S = 180.0                # ERA5 rebuild per point: 1:21-3:00 measured (sc26p1)
EST_REFINE_ROUNDS = 2                # coarse-to-fine refinement rounds around the best N
EST_ADAPTIVE_ROUNDS = 3              # round0 -> adaptive proposer rounds (O3 only)
EST_ROUND0_CONFIGS = 6               # decomposition candidates measured at the winner N
EST_SPINUP_STEPS = 8
EST_PHYSICS_RUNS = 5                 # baseline + 3 candidates + ANCHOREND, per point job
EST_PHYSICS_POINTS = 2

try:
    import wtune_cost_drivers       # plan_points_for_case(case, precision_min, ...): the points a
except ImportError:                  # case type needs for the requested precision + their band
    wtune_cost_drivers = None


def _band_for_points(points, measured):
    if points <= 0:
        return ("t0 anchor (phase-1 data)", 0.15) if measured else ("cold-start predict",
                                                                    COLD_START_BAND)
    return (FALLBACK_BANDS[1][1], FALLBACK_BANDS[1][2]) if points < 3 else (
        FALLBACK_BANDS[2][1], FALLBACK_BANDS[2][2])


def _fallback_point_plan(T_s, precision_s, measured, n_available):
    options = [(0,) + _band_for_points(0, measured), (1,) + _band_for_points(1, measured),
               (3,) + _band_for_points(3, measured)]
    for pts, method, band in options:
        if pts <= n_available and band * T_s <= precision_s:
            return pts, method, band
    pts = n_available
    method, band = _band_for_points(pts, measured)
    return pts, method, band


def _match_points(res_points, wrapped):
    """Map what plan_points_for_case returned (indices, points with .at, datetimes, or dicts) onto
    the planned points `wrapped` [{index, point}]."""
    out = []
    for item in res_points or []:
        at = item if isinstance(item, datetime.datetime) else None
        idx = item if isinstance(item, int) and not isinstance(item, bool) else None
        if isinstance(item, dict):
            idx, at = item.get("index"), item.get("at")
        elif hasattr(item, "at"):
            at = item.at
        for w in wrapped:
            if (idx is not None and w["index"] == idx) or (at is not None and w["point"].at == at):
                out.append(w)
                break
    return out


def _case_types_and_track(ctx):
    """(case type labels, best track or None) of ctx.case: wtune_advise.build_case ->
    wtune_cost_drivers.case_types_for (decision-0057 classifier, reused) and, for a TC,
    wtune_advise.best_track."""
    import wtune_advise
    case = wtune_advise.build_case(ctx.case)
    types = list(wtune_cost_drivers.case_types_for(case))
    track = wtune_advise.best_track(case) if "tc" in types else None
    return types, track


def _nearest_wrapped(wrapped, times):
    """The planned point nearest in time to each of `times` (each planned point used once)."""
    out, free = [], list(wrapped)
    for t in times:
        if not free:
            break
        best = min(free, key=lambda w: abs((w["point"].at - t).total_seconds()))
        free.remove(best)
        out.append(best)
    return sorted(out, key=lambda w: w["index"])


def plan_prediction_points(ctx, wrapped, precision_min, T_est_s, measured, sample_points=None,
                           run_start=None, run_end=None, window_s=0.0):
    """The prediction points + expected whole-run band for this case at `precision_min` minutes.
    wtune_cost_drivers.plan_points_for_case(case_types, run_start, run_end, precision_min,
    T_est_s, indicators=<CostSeries when --points-by cost-indicators built one>, best_track=<TC
    best track>, window_s=<measurement window>) -> {points: [(time, reason)], band_pct,
    needs_rebuild: [time], ...}; every time in `needs_rebuild` is mapped onto the nearest planned
    point (t0 is phase-1 data: no rebuild). On a real exception the fallback band table is used
    and the error is printed. Returns dict(selected=[wrapped...], band, method, source, ...)."""
    api = wtune_cost_drivers
    if api is not None and getattr(api, "plan_points_for_case", None) and run_start is not None:
        try:
            types, track = _case_types_and_track(ctx)
            indicators = _cost_series_for_ctx(ctx, run_start, run_end,
                                              sweep._expand(ctx.case_json["prep_dir"]))
            res = api.plan_points_for_case(
                types, run_start, run_end, precision_min, T_est_s, indicators=indicators,
                best_track=track, window_s=window_s)
            sel = _nearest_wrapped(wrapped, res.get("needs_rebuild") or [])
            why = (res.get("rationale") or [""])
            return dict(selected=sel, band=float(res["band_pct"]) / 100.0,
                        method="%s: %s" % ("/".join(res.get("types") or types), why[0]),
                        source="wtune_cost_drivers", types=list(res.get("types") or types),
                        meets_precision=res.get("meets_precision"),
                        confidence=res.get("confidence"), rationale=list(why),
                        points=[(t.isoformat(), r) for t, r in res.get("points", [])])
        except Exception as e:                              # noqa: BLE001
            print("pipeline: wtune_cost_drivers.plan_points_for_case FAILED (%s: %s) -- using the "
                  "fallback band table instead" % (type(e).__name__, e))
    pts, method, band = _fallback_point_plan(T_est_s or 0.0, precision_min * 60.0, measured,
                                             len(wrapped))
    return dict(selected=pick_precision_points(wrapped, pts, sample_points), band=band,
                method=method, source="fallback band table")


def time_suggestion_s(T_s, band, step_s=TIME_STEP_S):
    """sbatch --time for the real run: T rounded UP to the 15-minute step, plus one band
    (band x T), never below 1.2 x T (25:27 at +-15% -> 30 + 3.8 = 33.8 min)."""
    return max(math.ceil(T_s / step_s) * step_s + band * T_s, 1.2 * T_s)


def _fmt_sbatch_time(seconds):
    m = int(math.ceil(seconds / 60.0))
    return "%02d:%02d:00" % (m // 60, m % 60)


def pipeline_run_length_io(namelist_path):
    """wtune_io_model.run_length_io of the case namelist (history frames, lateral boundary
    updates, sst_update auxinput4 reads, restart writes over one unsegmented run); None, with the
    reason printed, when the model or the namelist is unavailable -- the prediction is then
    compute + init only, as before 0.8.2."""
    if wtune_io_model is None or wtune_predict is None or not namelist_path:
        return None
    try:
        return wtune_io_model.run_length_io(wtune_predict.parse_namelist(namelist_path))
    except Exception as e:                                  # noqa: BLE001
        print("pipeline: run-length I/O model unavailable (%s: %s) -- whole-run prediction is "
              "compute + init only" % (type(e).__name__, e))
        return None


def prediction_init(ctx, fallback_s, fallback_source):
    """(init_s, init_source) of a whole-run prediction: the startup round0 MEASURED on this
    allocation when there is one (scc: predict_init's 125.6 s for climate-ea50 vs ~50 s measured,
    it charges the whole 13-month wrfbdy as an init read), else `fallback_s` (predict_init, or
    the startup constant in effect)."""
    if str(getattr(ctx, "startup_source", "") or "").startswith("measured"):
        n = getattr(ctx, "startup_runs", None)
        return float(ctx.startup_s), ("measured round0 startup (%d runs)" % n if n
                                      else "measured round0 startup (%s)" % ctx.startup_source)
    return float(fallback_s), fallback_source


def add_run_length_io(pred, io, compute_s, init_s, band, init_source=None):
    """Whole-run prediction dict `pred` (predicted_total_s = compute_s + init_s) -> + run-length
    I/O, with init_s (possibly a measured startup) in place of the init `pred` was built with, in
    place: predicted_total_s/_hms; predicted_range_s and every candidates[] total shift by the
    same amount (new total - old total; I/O is layout-independent and the init swap is applied
    uniformly, so the candidate order is unchanged); time_suggestion and error_min are recomputed
    at `band`. Records compute_s, init_s + init_source (the startup `wtune run` pays once per
    segment), io, breakdown (the 'compute X +
    history Y + boundary Z + aux W + restart R + init I = TOTAL' line) and segment_note. io None:
    totals unchanged, breakdown says I/O is not included."""
    io_s = io["total_s"] if io else 0.0
    T = compute_s + init_s + io_s
    shift = T - (pred.get("predicted_total_s") or 0.0)
    pred["compute_s"], pred["init_s"], pred["io"] = compute_s, init_s, io
    pred["init_source"], pred["io_s"] = init_source, io_s
    pred["predicted_total_s"], pred["predicted_total_hms"] = T, _fmt_hms(T)
    if pred.get("predicted_range_s"):
        pred["predicted_range_s"] = [x + shift for x in pred["predicted_range_s"]]
    for c in pred.get("candidates") or []:
        if c.get("predicted_total_s") is not None:
            c["predicted_total_s"] += shift
    pred["time_suggestion"] = _fmt_sbatch_time(time_suggestion_s(T, band))
    pred["error_min"] = band * T / 60.0
    if io:
        pred["breakdown"] = wtune_io_model.breakdown_line(compute_s, io, init_s)
        pred["segment_note"] = wtune_io_model.segment_note(init_s)
    else:
        pred["breakdown"] = ("compute %.0fs + init %.0fs = %.0fs (run-length I/O NOT included)"
                             % (compute_s, init_s, T))
        pred["segment_note"] = None
    return pred


def cold_start_prediction(ctx, nodes, config, total_steps):
    """(whole-run seconds, s/step at `nodes`, source): wtune_predict's cold-start prediction (no
    probe) when it works, else this case's ledger estimate (10 s/step fallback). Crude by design:
    the +-25% O0 band says so."""
    if wtune_predict is not None and config:
        try:
            pred = wtune_predict.build_prediction(
                namelist_path=ctx.namelist_path, case_dir=ctx.case_json["prep_dir"], nodes=nodes,
                config=dict(ranks=config["ranks"], omp=config["omp"], nx=config["nx"],
                            ny=config["ny"], numtiles=config["numtiles"]),
                ledger_paths=None, calibration_path=str(wtune_predict.DEFAULT_CALIBRATION),
                io_form_override=None, io_target=ctx.prediction_cfg.get("io_target") or "nfs",
                case_name=ctx.case, rsl_paths=None, include_io=False)
            T = float(pred["total_s"])
            return T, max(1e-6, (T - ctx.startup_s) / max(1, total_steps)), "wtune_predict cold start"
        except Exception:                                   # noqa: BLE001
            pass
    s_est, src, _k = sweep.estimate_s_per_step(ctx.case, nodes, ctx.ledger_dir, ctx.hardware_id)
    return total_steps * s_est + ctx.startup_s, s_est, "ledger estimate (%s)" % src


def layout_estimates(*, s1, steps_probe, startup_s, max_nodes, n_full=0, finalists=2):
    """{level: expected RUNNING seconds of the LAYOUT search} for O0..O3, from the seconds/step at
    ONE node `s1` (ideal scaling s(N) = s1/N, optimistic at large N) and the startup per run.
    Counts jobs per wave given `max_nodes` healthy nodes: O1 = the t0 wave N in {1,2,4,8,max}
    packed into waves (wtune_nodes.sweep_waves) of 2-rep jobs; O2 adds refinement rounds, a
    cores/node wave and one sampled round0 wave at the winner N; O3 (the default) replaces the
    sample by the FULL layout space (`n_full` enumerated layouts, one 1-rep run each, spread over
    the concurrent slots) at each of `finalists` node counts, plus a confirm job."""
    S = float(startup_s)
    job = lambda n, reps=2: reps * (S + steps_probe * s1 / n)            # noqa: E731
    seeds = sorted({n for n in (1, 2, 4, 8, max_nodes) if 1 <= n <= max_nodes})
    t1 = sum(max(job(n) for n in wave) for wave in wtune_nodes.sweep_waves(seeds, max_nodes))
    nstar = max(1, min(4, max_nodes))
    wave_n = job(nstar)
    slots = max(1, max_nodes // nstar)
    t_round0 = math.ceil(EST_ROUND0_CONFIGS / slots) * (S + steps_probe * s1 / nstar)
    t2 = t1 + EST_REFINE_ROUNDS * wave_n + wave_n + t_round0
    full = finalists * math.ceil(max(n_full, EST_ROUND0_CONFIGS) / float(slots)) * (
        S + steps_probe * s1 / nstar)
    t3 = t2 - t_round0 + full + wave_n
    return {0: 0.0, 1: t1, 2: t2, 3: t3}


def points_estimate(*, s1, startup_s, max_nodes, n_points, layouts, layout_s, physics,
                    nstar=None, window_steps=32, restart_read_s=30.0):
    """Expected RUNNING seconds of the prediction part (case type decides n_points): the rebuild
    (ERA5 + WPS/real, one 1-node job per point, 1:21-3:00 measured) runs beside the layout
    search so only what sticks out counts; then one point job per point = spin-up + `layouts` x
    (ANCHOR) + ANCHOREND runs, ceil(points x N / max_nodes) waves; physics advice (O3) adds two
    point jobs of EST_PHYSICS_RUNS runs."""
    if n_points <= 0:
        return 0.0
    S = float(startup_s)
    nstar = nstar or max(1, min(4, max_nodes))
    spin = S + EST_SPINUP_STEPS * s1 / nstar
    per_run = S + restart_read_s + window_steps * s1 / nstar

    def point_jobs(points, runs):
        waves = math.ceil(max(1, points) * nstar / float(max_nodes))
        return waves * (spin + runs * per_run)

    exposed = max(0.0, EST_REBUILD_S - layout_s)
    total = exposed + point_jobs(n_points, layouts + 1)
    if physics:
        total += point_jobs(EST_PHYSICS_POINTS, EST_PHYSICS_RUNS)
    return total


def level_estimates(*, s1, steps_probe, startup_s, max_nodes, n_points, n_full=0, **_ignored):
    """{level: total expected RUNNING seconds}: layout_estimates(level) + points_estimate(case
    type's n_points): O1/O2 run the winner layout at the points, O3 the top-2 layouts plus the
    physics advice."""
    lay = layout_estimates(s1=s1, steps_probe=steps_probe, startup_s=startup_s,
                           max_nodes=max_nodes, n_full=n_full)
    out = {0: 0.0}
    for lv in (1, 2, 3):
        out[lv] = lay[lv] + points_estimate(
            s1=s1, startup_s=startup_s, max_nodes=max_nodes, n_points=n_points,
            layouts=1, layout_s=lay[lv], physics=False)
    return out


def highest_level_within(estimates, real_run_s, requested):
    """The highest level <= `requested` whose estimated tuning time is <= the real run (never
    below O1: a measurement wave is the floor)."""
    for lv in range(min(3, requested), 0, -1):
        if estimates[lv] <= real_run_s:
            return lv
    return 1


def format_level_estimates(estimates, chosen):
    return ("  ".join("O%d ≈ %.0f min%s" % (lv, max(estimates[lv], 0) / 60.0,
                                           " <-" if lv == chosen else "")
                       for lv in (1, 2, 3)) + "  (running time; queue wait extra)")


def pick_precision_points(wrapped, k, sample_points=None):
    """k prediction points: 1 -> the costliest alone (t0 is the free phase-1 data), 2 -> the t0
    nearest + the costliest, 3+ -> those plus the next costliest."""
    if k <= 0:
        return []
    if k == 1 and len(wrapped) > 1:
        return [wrapped[_costliest_point_index(wrapped, sample_points)]]
    return pick_points(wrapped, k, sample_points)


def _costliest_point_index(rebuilt, sample_points):
    """Index of the point expected to cost most per step (the radiation-heaviest = most sunlit,
    choose_primary_point) when nothing was measured yet; the last point if that fails."""
    try:
        return choose_primary_point([rp["point"] for rp in rebuilt], sample_points)[0]
    except Exception:                                       # noqa: BLE001
        return len(rebuilt) - 1


def pick_points(rebuilt, k, sample_points=None, s_normal=None):
    """k rebuilt points: ALWAYS the earliest (t0) plus the costliest one -- by measured s_normal
    (list parallel to `rebuilt`) when given, else by the sunlit-fraction proxy. Rebuilt order."""
    if len(rebuilt) <= k:
        return list(rebuilt)
    idx = list(range(len(rebuilt)))
    earliest = min(idx, key=lambda i: rebuilt[i]["point"].at)
    if s_normal:
        rest = sorted((i for i in idx if i != earliest),
                      key=lambda i: -(s_normal[i] if i < len(s_normal) else 0.0))
    else:
        c = _costliest_point_index(rebuilt, sample_points)
        rest = [c] if c != earliest else []
        rest += [i for i in reversed(idx) if i != earliest and i not in rest]
    return [rebuilt[i] for i in sorted([earliest] + rest[:k - 1])]


def phase2_plans(layouts, rebuilt, sample_points, winner_only=False):
    """The degradation ladder of phase 2, largest first: [(label, layouts, points)]. Layouts: both
    top-2 layouts (twin for the gate), then the winner only; points: all, then two (t0 + the
    costliest), then t0 alone."""
    r2 = pick_points(rebuilt, 2, sample_points)
    r1 = pick_points(rebuilt, 1)
    lay_sets = [("%d layout(s)" % len(layouts), layouts)] if len(layouts) > 1 and not winner_only \
        else []
    lay_sets.append(("winner only", layouts[:1]))
    plans, seen = [], set()
    ladder = []
    for ln, ls in lay_sets:
        ladder += [(ln, ls, rebuilt, "all %d points" % len(rebuilt)),
                   (ln, ls, r2, "2 points (t0 + costliest)")]
    ladder.append(("winner only", layouts[:1], r1, "1 point (t0)"))
    for ln, ls, pts, pn in ladder:
        key = (len(ls), tuple(rp["index"] for rp in pts))
        if key in seen:
            continue
        seen.add(key)
        plans.append(("%s x %s" % (ln, pn), ls, pts))
    return plans


def run_pipeline_full(ctx, profile, *, minutes=None, level=3, total_steps=None, objective="time",
                      deadline_s=None,
                      min_efficiency=None, k_finalists=2, nodes_policy="knee",
                      sweep_budget_frac=None, core_options=None, extension=False, physics=False,
                      physics_max_candidates=PIPELINE_PHYSICS_MAX_CANDIDATES,
                      max_queue_hours=PIPELINE_MAX_QUEUE_HOURS, keep_frames=False, recheck=None,
                      stop_if_short=False, precision_min=PRECISION_DEFAULT_MIN):
    """Decision 0059 at effort level `level` (1..3; 0 is run_level0): O1 = one t0 wave over
    N in {1,2,4,8,max healthy}; O2 = + refinement, cores/node, round0 at the winner and 2
    prediction points; O3 = + adaptive search, top-2 layouts at all points, physics advice.
    `minutes` is an OPTIONAL hard cap (None: no cap). When given, B = `minutes` of RUNNING time: the budget clock runs only while one of our
    jobs is RUNNING. Queue wait (every job PENDING) and controller-side work (submission, report
    building, predictions, fidelity gate, conversions) are reported per phase but never charged
    (--max-queue-hours caps the queue wait: past it the run stops). Phases take cumulative limits of 5/55/80/100% of B (plan 5, layout 50, prediction 25,
    physics 20; unused share rolls forward; with physics off its share goes to layout and
    prediction 2:1); every phase's jobs are cancelled at its limit, B is the hard deadline, and
    what does not fit is reported "not run (budget)":
      0 plan     K points (wtune_rebuild_predict.plan); submit the per-point ERA5 rebuild
                 (real_only) jobs NOW so they overlap phase 1
      1 layout   node search (coarse-to-fine N, cores/node) + round0/adaptive/confirm at the
                 finalists, t0; the first measured s/step gives an informational whole-run time
      2 predict  top-2 layouts from each point's shared restart (measure_candidates_at_points) ->
                 combine() (rank_by_predicted_total) -> winner by predicted whole-run time,
                 `--time` suggestion
      3 physics  wtune_physics_advice candidates (<= 3) on the winner layout at TWO points (t0
                 and the costliest predicted one): one job per point = baseline + candidates run
                 one after the other from that point's shared restart; advice table (skipped for
                 extension cases)
    Returns the phase-1 report (layout result, updated to phase 2's winner) with a `pipeline`
    key: phases (running/wall/queue per phase), B, prediction, physics advice, P1 check."""
    ctx.budget_mode = "running"
    ctx.shared_queue = {"s": 0.0, "cap_hit": False, "run_s": 0.0}
    ctx.max_queue_s = float(max_queue_hours) * 3600.0 if max_queue_hours else None
    ctx.background_jobs = []
    t_start = ctx.elapsed()
    level_req = int(level)
    plan = dict(level=level_req)                  # the EFFECTIVE level; the guard may lower it
    physics = bool(physics and level_req >= 3)       # physics advice is opt-in (--physics-advice)
    # the step-down guard is OFF unless asked for (--stop-if-short) or a time cap (--minutes) is set
    guard = bool(stop_if_short or minutes is not None)
    fracs = dict(PIPELINE_FRACS)
    if not physics:                     # phase 3's share rolls back to layout/prediction (2:1)
        extra = fracs.pop("physics")
        total = fracs["layout"] + fracs["predict"]
        fracs["layout"] += extra * fracs["layout"] / total
        fracs["predict"] += extra * fracs["predict"] / total
        fracs["physics"] = 0.0
    cum, acc = {}, 0.0
    for name, _f in PIPELINE_FRACS:
        acc += fracs[name]
        cum[name] = acc
    B = float(minutes) * 60.0 if minutes is not None else float("inf")
    pipe = dict(total_steps=total_steps, B_s=(B if B != float("inf") else None),
                budget_source=("--minutes hard cap (job running time; queue wait and controller "
                               "work excluded)" if minutes is not None else "no time cap"),
                T_updated_s=None, phases=[], rebuild=None, extension=bool(extension),
                max_queue_s=ctx.max_queue_s, queue_cap_hit=False,
                shares={n: fracs[n] for n, _f in PIPELINE_FRACS})
    slices, _last = {}, [len(ctx.owned_jobs)]

    def cut(name):
        slices[name] = list(ctx.owned_jobs[_last[0]:])
        _last[0] = len(ctx.owned_jobs)

    def limit(name):
        return t_start + cum[name] * B

    def cap_hit():
        return bool(ctx.shared_queue["cap_hit"])

    print("pipeline O%d: hard cap %s of RUNNING time (queue wait not charged; queue cap %s)%s"
          % (level_req, _fmt_hms(B), _fmt_hms(ctx.max_queue_s or 0),
             "; phase limits (cumulative): " + ", ".join(
                 "%s %s" % (n, _fmt_hms(cum[n] * B)) for n, _f in PIPELINE_FRACS)
             if minutes is not None else ""))

    # ---- pre-run estimate of the tuning time per level (printed BEFORE any job) ----
    # layout(level) + prediction points(case type): the points come from
    # wtune_cost_drivers.plan_points_for_case (or the fallback band table), never from the level
    est, T_cold, est_info, pre = None, None, {}, dict(ok=False)
    pipe_n_full = 0
    pplan0 = None
    try:                                  # planning only reads the namelist: no job is submitted
        spinup0 = predicted_spinup_min(_namelist_dt(ctx))
        _n, _rs, _re, dt_p, stepra_p, points_p, notes_p = plan_predicted_points(ctx, spinup0)
        pre = dict(ok=True, spinup_min=spinup0, dt=dt_p, stepra=stepra_p, points=points_p,
                   pkw=dict(run_start=_rs, run_end=_re, window_s=rebuilt_steps(stepra_p) * dt_p),
                   notes=list(notes_p),
                   wrapped=[dict(index=i, point=p) for i, p in enumerate(points_p)])
    except Exception as e:                                  # noqa: BLE001
        pre = dict(ok=False, error="%s: %s" % (type(e).__name__, e), points=[], wrapped=[])
    if total_steps:
        try:
            n0 = choose_nodes(profile, ctx.max_nodes, nodes_policy)[0]
            cfg0, _why = _sweep_anchor(ctx, n0)
            T_cold, s_n0, src = cold_start_prediction(ctx, n0, cfg0, total_steps)
            pplan0 = plan_prediction_points(ctx, pre["wrapped"], precision_min, T_cold, False,
                                            **pre.get("pkw", {}))
            try:
                n_full = len(enumerate_full_space(dataclasses.replace(
                    ctx, nodes=n0, anchor_override=None, core_budget=None))[0])
            except Exception:                               # noqa: BLE001
                n_full = 0
            pipe_n_full = n_full
            est = level_estimates(s1=s_n0 * n0, steps_probe=ctx.steps, startup_s=ctx.startup_s,
                                  max_nodes=ctx.max_nodes,
                                  n_points=len(pplan0["selected"]), n_full=n_full)
            est_info = dict(nodes=n0, s_per_step_n=s_n0, source=src, startup_s=ctx.startup_s,
                            prediction_points=len(pplan0["selected"]),
                            points_source=pplan0["source"], precision_min=precision_min)
        except Exception as e:                              # noqa: BLE001
            print("pipeline: pre-run estimate unavailable (%s: %s)" % (type(e).__name__, e))
    if est is not None and guard and plan["level"] >= 2:
        lv = highest_level_within(est, T_cold, plan["level"])
        if lv < plan["level"]:
            print("pipeline: 模擬本身只要約 %.0f 分鐘（冷啟動估計），比 O%d 的調校時間短；"
                  "自動降為 O%d（--stop-if-short / --minutes 啟用）" % (T_cold / 60.0, plan["level"], lv))
            plan["level"] = lv
    pipe["level"] = dict(requested=level_req, effective=plan["level"],
                         estimate_s={("O%d" % k): v for k, v in (est or {}).items()},
                         estimate_info=est_info, T_cold_s=T_cold, stepped_down=[])
    pipe["n_points_est"] = len(pplan0["selected"]) if pplan0 else 0
    pipe["n_full_est"] = pipe_n_full
    if plan["level"] < level_req:
        pipe["level"]["stepped_down"].append(dict(when="pre-run", to=plan["level"],
                                                  T_real_s=T_cold))
    if est is not None:
        print("pipeline: 預估調校時間 " + format_level_estimates(est, plan["level"])
              + "; whole run ~%s (cold start); prediction points for this case at ±%g min: %d "
                "(%s)" % (_fmt_hms(T_cold), precision_min, len(pplan0["selected"]),
                          pplan0["source"]))
    pipe["T_cold_s"] = T_cold
    pipe["T_cold_compute_s"] = (T_cold - ctx.startup_s) if T_cold else None

    # ---------------- phase 0: plan + submit the rebuild jobs ----------------
    m0 = _mark(ctx)
    prep_handle, plan_info = None, {}
    sub = {}                                     # what phase 0 needs again for a late submission
    try:
        if not pre["ok"]:
            raise RuntimeError("point planning failed: %s" % pre["error"])
        spinup_min, dt0, stepra0, points, notes = (pre["spinup_min"], pre["dt"], pre["stepra"],
                                                   pre["points"], pre["notes"])
        plan_info = dict(points=[p.at.isoformat() for p in points], spinup_min=spinup_min,
                         notes=list(notes))
        if not points:
            raise RuntimeError("no ERA5-rebuildable point planned (%s)" % "; ".join(notes))
        anchor0 = sweep.build_anchor(ctx.case_json, ctx.cores_per_node, 1, ctx.hardware_id)
        if anchor0 is None:
            for n in range(1, ctx.max_nodes + 1):
                anchor0, _why = _sweep_anchor(ctx, n)
                if anchor0:
                    break
        case_dir = sweep._expand(ctx.case_json["prep_dir"])
        try:
            sample_pts0, _src = wtune_predict.read_wrfinput_sample_points(case_dir)
        except Exception:                                   # noqa: BLE001
            sample_pts0 = None
        wrapped0 = [dict(index=i, point=p) for i, p in enumerate(points)]
        sub = dict(wrapped=wrapped0, anchor=anchor0, case_dir=case_dir, spinup_min=spinup_min,
                   stepra=stepra0, dt=dt0, sample_points=sample_pts0)
        # rebuild + spin-up ONLY for the points this CASE needs at the requested precision
        # (plan_prediction_points; the cold-start T is the worst case: nothing measured yet);
        # phase 1's measurement may need fewer -> the surplus jobs are cancelled below
        pp0 = pplan0 if pplan0 is not None else plan_prediction_points(
            ctx, wrapped0, precision_min, T_cold, False, sample_pts0, **pre.get("pkw", {}))
        chosen0 = pp0["selected"]
        plan_info.update(prediction_points=len(chosen0), points_source=pp0["source"])
        if chosen0:
            prep_handle = submit_rebuilt_points(
                ctx, [w["point"] for w in chosen0], anchor0, case_dir, ctx.namelist_path,
                ctx.case_json, os.path.join(ctx.out_dir, "rebuilt"), spinup_min, stepra0, dt0)
            ctx.background_jobs[:] = list(prep_handle["job_ids"])   # run beside phase 1: they
            plan_info.update(rebuild_jobs=list(prep_handle["job_ids"]))   # count as "ours RUNNING"
            p0_status = "ok"
        else:
            p0_status = "ok (no rebuild submitted: the case needs no prediction point)"
    except Exception as e:                                  # noqa: BLE001 -- reported, not hidden
        p0_status = "failed: %s: %s" % (type(e).__name__, e)
        print("pipeline: phase 0 rebuild submission FAILED -- %s" % p0_status)
    pipe["phases"].append(_phase_record("plan", m0, _mark(ctx), limit("plan"), p0_status,
                                        **plan_info))
    pipe["rebuild"] = dict(plan_info, status=p0_status)
    cut("plan")

    # ---------------- phase 1: layout search ----------------
    m1 = _mark(ctx)
    ctx1 = dataclasses.replace(ctx, budget_s=max(limit("layout"), m1["run"] + 1.0))

    def on_sweep(rows, _sw):
        """Informational only: the whole-run time implied by the best measured s/step (anchor
        config, so an upper bound). B is NOT changed."""
        if not total_steps:
            return
        s_best = min(r["median"] for r in rows)
        pipe["T_updated_s"] = total_steps * s_best + ctx.startup_s
        pipe["T_updated_compute_s"] = total_steps * s_best
        print("pipeline: best measured %.4f s/step -> whole run ~%s" % (s_best,
                                                                       _fmt_hms(pipe["T_updated_s"])))

    def on_wave(meas):
        """Short-run guard (default; --force-full disables): after a measurement wave, when the
        whole run (steps x best measured s/step + startup) is shorter than the estimated tuning
        time of the level still planned, step down to the highest level whose estimate fits the
        real run (never below O1). The run flags are read live: lowering to O1 stops refinement,
        the core sweep and the finalists."""
        if not guard or not total_steps or not meas or plan["level"] < 2:
            return None
        n_best, vals = min(((n, statistics.median(v)) for n, v in meas.items() if v),
                           key=lambda kv: kv[1])
        T = total_steps * vals + ctx.startup_s
        est2 = level_estimates(s1=vals * n_best, steps_probe=ctx.steps, startup_s=ctx.startup_s,
                               max_nodes=ctx.max_nodes, n_points=pipe.get("n_points_est", 0),
                               n_full=pipe.get("n_full_est", 0))
        lv = highest_level_within(est2, T, plan["level"])
        if lv >= plan["level"]:
            return None
        msg = ("模擬本身只要約 %.0f 分鐘，比 O%d 的調校時間（約 %.0f 分鐘）短；已自動降為 O%d，"
               "建議直接以目前最佳設定執行" % (T / 60.0, plan["level"],
                                          est2[plan["level"]] / 60.0, lv))
        pipe["level"]["stepped_down"].append(dict(when="after first wave", frm=plan["level"],
                                                  to=lv, T_real_s=T, message=msg))
        plan["level"] = lv
        pipe["level"]["effective"] = lv
        pipe["short_run"] = dict(message=msg, real_run_s=T)
        print("pipeline: %s" % msg)
        return None                      # the sweep itself goes on; later stages read plan

    collect, report = {}, None
    try:
        report = run_node_search(
            ctx1, profile, objective, deadline_s,
            min_efficiency if min_efficiency is not None else wtune_nodes.DEFAULT_MIN_EFFICIENCY,
            k_finalists, total_steps, nodes_policy=nodes_policy,
            sweep_budget_frac=(sweep_budget_frac if sweep_budget_frac is not None
                               else DEFAULT_SWEEP_BUDGET_FRAC),
            core_options=core_options, on_sweep=on_sweep, collect=collect, on_wave=on_wave,
            plan=plan)
        ph1 = "ok" if not pipe.get("short_run") else "ok (stepped down: short run)"
    except QueueCapExceeded as e:
        ph1 = "stopped: %s" % e
        print("pipeline: phase 1 STOPPED -- %s" % e)
    pipe["phases"].append(_phase_record(
        "layout", m1, _mark(ctx), limit("layout"), ph1,
        **(dict(nodes=report["nodes"], cores_per_node=report.get("cores_used_per_node"))
           if report else {})))
    cut("layout")
    if report is None:                         # aborted before any layout result existed
        report = dict(case=ctx.case, pipeline_aborted=True, nodes=None, job_ids=[],
                      cancelled_job_ids=[])
    layouts = pipeline_layouts(collect, 2) if collect.get("outcomes") else []
    # the startup overhead measured by phase 1's round0 lives on the finalist ctx copies: adopt it
    # (the default 120 s/run made phase 2 look 2-4x too expensive)
    meas_ctx = [o["ctx"] for o in collect.get("outcomes", {}).values()
                if o.get("ctx") is not None
                and str(getattr(o["ctx"], "startup_source", "")).startswith("measured")]
    meas_su = [c.startup_s for c in meas_ctx]
    if meas_su:
        ctx.startup_s = statistics.median(meas_su)
        ctx.startup_source = "measured in phase 1 (median of %d finalist(s))" % len(meas_su)
        ctx.startup_runs = sum(getattr(c, "startup_runs", None) or 0 for c in meas_ctx) or None

    # ---------------- phase 2: prediction from the shared restarts ----------------
    health_checks = [ctx.node_health] if ctx.node_health else []
    if recheck is not None and not cap_hit():
        try:                  # nodes can drop out during phase 1: probe again (controller time)
            h = recheck("before phase 2")
            health_checks.append(h)
            ctx.node_health = h
            ctx.max_nodes = h["max_nodes_effective"]
        except Exception as e:                              # noqa: BLE001
            print("pipeline: node re-check failed (%s: %s); keeping the earlier exclusions"
                  % (type(e).__name__, e))
    # re-plan the prediction points with phase 1's measurement (the t0 anchor gives the real
    # whole-run time): fewer points than submitted -> cancel the surplus rebuild jobs
    T_dec = pipe["T_updated_s"] or pipe.get("T_cold_s")
    lvl = plan["level"]
    pipe["level"]["effective"] = lvl
    pplan = (plan_prediction_points(ctx, sub["wrapped"], precision_min, T_dec,
                                    pipe["T_updated_s"] is not None, sub.get("sample_points"),
                                    **pre.get("pkw", {}))
             if (sub and T_dec) else pplan0)
    if pplan is not None:
        pipe["prediction_plan"] = dict(points=[w["index"] for w in pplan["selected"]],
                                       band=pplan["band"], method=pplan["method"],
                                       source=pplan["source"], T_est_s=T_dec,
                                       precision_min=precision_min,
                                       measured=pipe["T_updated_s"] is not None)
    have = len(prep_handle["points"]) if prep_handle else 0
    if prep_handle and pplan is not None:
        sel_at = {w["point"].at for w in pplan["selected"]}
        keep = [i for i, p in enumerate(prep_handle["points"]) if p.at in sel_at]
        if not keep and pplan["selected"]:           # the plan wants other points than submitted
            keep = list(range(have))
        if len(keep) < have:
            drop = [j for i, j in enumerate(prep_handle["job_ids"]) if i not in keep]
            print("pipeline: the case needs %d prediction point(s) at ±%g min: cancelling %d "
                  "surplus rebuild job(s) %s" % (len(keep), precision_min, len(drop),
                                                 ",".join(drop)))
            ctx.scancel_runner(drop)
            ctx.cancelled_jobs.extend(drop)
            pipe["prediction_plan"]["rebuild_cancelled"] = drop
            if keep:
                prep_handle = dict(prep_handle, points=[prep_handle["points"][i] for i in keep],
                                   done=[prep_handle["done"][i] for i in keep],
                                   job_ids=[prep_handle["job_ids"][i] for i in keep])
                ctx.background_jobs[:] = list(prep_handle["job_ids"])
            else:
                prep_handle = None
                ctx.background_jobs[:] = []
    m2 = _mark(ctx)
    rebuilt, winner, ranked_all, frame_runs = None, None, [], []
    nl = run_start = run_end = dt = stepra = calibration = sample_points = case_dir = None
    ph2 = None
    pc_ok = False
    if cap_hit():
        ph2 = "not run (max queue wait exceeded)"
    elif pplan is not None and not pplan["selected"] and layouts:
        ph2 = ("not run (the t0 anchor suffices for this case at ±%g min: %s, band ±%.0f%%)"
               % (precision_min, pplan["method"], pplan["band"] * 100))
    elif prep_handle is None or not layouts:
        ph2 = "not run (no rebuilt points)" if prep_handle is None else "not run (no layout)"
    else:
        # the rebuild jobs ran beside phase 1: collect them whatever phase 2 can afford, phase 3
        # needs the points too (own budget: the whole run, never cancelled by the phase-2 limit)
        c2 = dataclasses.replace(ctx, budget_s=limit("physics"))
        try:
            rebuilt = finish_rebuilt_points(c2, prep_handle)
        except QueueCapExceeded as e:
            ph2, rebuilt = "stopped: %s" % e, None
        except Exception as e:                              # noqa: BLE001
            ph2 = "failed: rebuild jobs: %s: %s" % (type(e).__name__, e)
            rebuilt = None
        ctx.background_jobs[:] = []
    if rebuilt is not None:
        try:
            (nl, run_start, run_end, dt, stepra, calibration, sample_points,
             case_dir) = _prediction_context(ctx)
            pc_ok = True
        except Exception as e:                              # noqa: BLE001
            ph2 = "failed: prediction context: %s: %s" % (type(e).__name__, e)
    if ph2 is None and ctx.elapsed() >= limit("predict"):
        ph2 = NOT_RUN_BUDGET + ": phase-2 limit reached"
    if ph2 is None:
        # phase 3 gates on window frames: phase-2 configs write them too (they are the twin,
        # same physics at another decomposition) -- only then, since frames cost I/O
        frames_wanted = bool(physics and not extension and wtune_physics_advice is not None
                             and physics_runner_supported(ctx.sbatch_script))
        left2 = limit("predict") - ctx.elapsed()
        short = None

        def build_groups(lays_sel):
            """[(n, c, anchor, others, s_by, lays)] for the layouts selected, one per (N, c)."""
            groups = {}
            for lay in lays_sel:
                groups.setdefault((lay["nodes"], lay["cores"]), []).append(lay)
            out = []
            for (n, c), lays in groups.items():
                wf = dict(window_frame=WINDOW_FRAME_LEVEL) if frames_wanted else {}
                first, anchor, others = lays[0], dict(lays[0]["cfg"], **wf), []
                for i, lay in enumerate(lays[1:], start=1):
                    lay["pid"] = "L%d" % i
                    others.append(dict(lay["cfg"], id=lay["pid"], config_source="generated", **wf))
                first["pid"] = "ANCHOR"
                if len(lays) == 1 and not frames_wanted:
                    # winner-only: ONE fused wrf.exe (spin-up + window) for the single layout; the
                    # ANCHOR placeholder only names the spin-up decomposition
                    first["pid"] = "W0"
                    others = [dict(first["cfg"], id="W0", config_source="generated")]
                s_by = {"ANCHOR": first["median"], first["pid"]: first["median"],
                        **{l["pid"]: l["median"] for l in lays[1:]}}
                out.append((n, c, anchor, others, s_by, lays))
            return out

        def gctx_for(n, c):
            return dataclasses.replace(ctx, nodes=n, core_budget=c, budget_s=limit("predict"),
                                       stage1_max_running=compute_slots(ctx.max_nodes, n))

        # degradation ladder: largest plan whose summed estimate fits what is left of phase 2
        # (groups run one after the other); short-run mode: winner only, and only if it is cheaper
        # than simply running the simulation
        tried, chosen_plan = [], None
        for label, lays_sel, pts in phase2_plans(layouts, rebuilt, sample_points,
                                                 winner_only=(lvl <= 2 or not physics)):
            est = 0.0
            for n, c, anchor, others, s_by, _l in build_groups(lays_sel):
                est += fit_stage2(gctx_for(n, c), others, anchor, pts, s_by, nl, left_s=None,
                                  mode=choose_point_mode(others, anchor))[1]
            tried.append(dict(plan=label, estimate_s=est))
            if est <= left2 and not (short and est >= short["real_run_s"]):
                chosen_plan = (label, lays_sel, pts, est)
                break
        pipe["predict_plan"] = dict(left_s=left2, tried=tried, startup_s=ctx.startup_s,
                                    chosen=chosen_plan[0] if chosen_plan else None,
                                    degraded=bool(chosen_plan and tried and len(tried) > 1))
        skipped2 = []
        if chosen_plan is None:
            why = ("the smallest plan (%s) needs ~%s of running time, %s left of the phase-2 "
                   "share" % (tried[-1]["plan"], _fmt_hms(tried[-1]["estimate_s"]),
                              _fmt_hms(max(0.0, left2))))
            if short and tried and tried[-1]["estimate_s"] >= short["real_run_s"]:
                why = ("short run: even the winner alone needs ~%s, more than the real run "
                       "(~%s)" % (_fmt_hms(tried[-1]["estimate_s"]),
                                  _fmt_hms(short["real_run_s"])))
            ph2 = NOT_RUN_BUDGET + ": " + why
            print("pipeline: phase 2 %s" % ph2)
        else:
            label, lays_sel, pts, est = chosen_plan
            if pipe["predict_plan"]["degraded"]:
                print("pipeline: phase 2 DEGRADED to %s (estimate %s of %s left): %s"
                      % (label, _fmt_hms(est), _fmt_hms(left2),
                         "; ".join("%s ~%s" % (t["plan"], _fmt_hms(t["estimate_s"]))
                                   for t in tried[:-1] if t["plan"] != label)))
            # One (N, c) group at a time: the running-time budget clock cannot be shared by
            # concurrent waits (the point jobs WITHIN a group do run in parallel).
            for n, c, anchor, others, s_by, lays in build_groups(lays_sel):
                gctx = gctx_for(n, c)
                pmode = choose_point_mode(others, anchor)
                top, _est, skip = fit_stage2(gctx, others, anchor, pts, s_by, nl,
                                             left_s=limit("predict") - ctx.elapsed(), mode=pmode)
                if skip:
                    skipped2.append("N=%d c=%d: %s" % (n, c, skip))
                    continue
                try:
                    meas, _gates, jobs = measure_candidates_at_points(gctx, top, pts, anchor,
                                                                      s_by_cid=s_by, mode=pmode)
                    cfgs = dict({o["id"]: o for o in top}, ANCHOR=anchor)
                    ranked = rank_by_predicted_total(
                        meas, run_start, run_end, dt, stepra, calibration, sample_points,
                        case_dir, nl, {cid: cf["ranks"] for cid, cf in cfgs.items()}, n)
                except QueueCapExceeded as e:
                    skipped2.append("N=%d c=%d: stopped: %s" % (n, c, e))
                    break
                except Exception as e:                      # noqa: BLE001
                    skipped2.append("N=%d c=%d: failed: %s: %s" % (n, c, type(e).__name__, e))
                    continue
                by_pid = {lay["pid"]: lay for lay in lays}
                frame_runs.extend((rp, cid, jid) for cid in ["ANCHOR", ANCHOR_END_ID] + [
                    o["id"] for o in top] for rp, jid in zip(pts, jobs))
                ranked_all.extend(
                    dict(layout=by_pid[e["cid"]], nodes=n, cores=c, cid=e["cid"],
                         pred_total_s=e["pred_total_s"], pred_low_s=e["pred_low_s"],
                         pred_high_s=e["pred_high_s"], result=e["result"],
                         meas=meas.get(e["cid"]), jobs=list(jobs),
                         points=[rp["index"] for rp in pts]) for e in ranked if e["cid"] in by_pid)
            if ranked_all:
                ranked_all.sort(key=lambda e: e["pred_total_s"])
                winner = ranked_all[0]
                ph2 = ("ok" if not skipped2 and not pipe["predict_plan"]["degraded"]
                       else "partial: " + "; ".join(
                           (["degraded to %s" % label] if pipe["predict_plan"]["degraded"] else [])
                           + skipped2))
            else:
                budget_msgs = ("exceed", "needs an expected", "left")
                ph2 = (NOT_RUN_BUDGET if skipped2 and all(any(m in x for m in budget_msgs)
                                                          for x in skipped2)
                       else "failed: " + "; ".join(skipped2))
    pipe["phases"].append(_phase_record("predict", m2, _mark(ctx), limit("predict"), ph2))
    cut("predict")

    pred = None
    band_eff = pplan["band"] if pplan else COLD_START_BAND
    # run-length I/O (history, lateral boundary, auxinput4, restart) is added to every whole-run
    # prediction below; combine()'s total is compute + init only
    io_run = (pipeline_run_length_io(getattr(ctx, "namelist_path", None))
              if (winner is not None or T_dec or pipe["T_updated_s"]) else None)
    if winner is not None:
        cfg = winner["layout"]["cfg"]
        pred_s = winner["pred_total_s"]
        pred = dict(layout_id=winner["layout"].get("cid"), nodes=winner["nodes"],
                    cores_per_node=winner["cores"],
                    config={k: cfg.get(k) for k in ("ranks", "omp", "nx", "ny", "numtiles",
                                                    "dist")},
                    predicted_total_s=pred_s, predicted_total_hms=_fmt_hms(pred_s),
                    predicted_range_s=[winner["pred_low_s"], winner["pred_high_s"]],
                    time_suggestion=_fmt_sbatch_time(
                        time_suggestion_s(pred_s, pplan["band"] if pplan else COLD_START_BAND)),
                    points_used=len(winner.get("points") or []),
                    degraded=bool(pipe.get("predict_plan", {}).get("degraded")),
                    candidates=[dict(nodes=e["nodes"], cores=e["cores"], pid=e["cid"],
                                     predicted_total_s=e["pred_total_s"],
                                     config={k: e["layout"]["cfg"].get(k)
                                             for k in ("ranks", "omp", "nx", "ny", "numtiles")})
                                for e in ranked_all])
        init_w = float((winner.get("result") or {}).get("init_s") or 0.0)
        init_s, init_src = prediction_init(ctx, init_w, "wtune_predict.predict_init")
        add_run_length_io(pred, io_run, pred_s - init_w, init_s, band_eff, init_src)
        pred_s = pred["predicted_total_s"]
        if (winner["nodes"], winner["cores"], sweep._identity(cfg)) != (
                report["nodes"], report.get("cores_used_per_node"),
                sweep._identity(layouts[0]["cfg"])):
            # phase 2 overturned phase 1's pick (rank flip across timeline points): report it
            pt = local_param_table(cfg, winner["nodes"])
            _assert_report_is_strict_only(ctx, pt)
            report["nodes"], report["nodes_note"] = (winner["nodes"],
                                                     "phase 2 predicted-whole-run winner")
            report["cores_used_per_node"] = winner["cores"]
            report["param_table"] = pt
            report["best"] = dict(id=winner["layout"].get("cid"), ranks=cfg.get("ranks"),
                                  omp=cfg.get("omp"), nproc_x=cfg.get("nx"),
                                  nproc_y=cfg.get("ny"), numtiles=cfg.get("numtiles"),
                                  dist=cfg.get("dist"), predicted_total_s=pred_s)
        report["full_run_prediction"] = "\n".join(
            [_predicted_winner_text(dict(cid=pred["layout_id"], result=winner["result"])),
             "whole run: " + pred["breakdown"]
             + ("  [init: %s]" % pred["init_source"] if pred.get("init_source") else "")]
            + (["NOTE: " + pred["segment_note"]] if pred.get("segment_note") else []))
    if pred is not None:
        pred["band"] = band_eff
        pred["error_min"] = band_eff * pred["predicted_total_s"] / 60.0
    elif pplan is not None and not pplan["selected"] and layouts and T_dec:
        lay0 = layouts[0]
        T = T_dec
        pred = dict(layout_id=lay0.get("cid"), nodes=lay0["nodes"], cores_per_node=lay0["cores"],
                    config={k: lay0["cfg"].get(k) for k in ("ranks", "omp", "nx", "ny", "numtiles",
                                                            "dist")},
                    predicted_total_s=T, predicted_total_hms=_fmt_hms(T),
                    predicted_range_s=[T * (1 - band_eff), T * (1 + band_eff)],
                    time_suggestion=_fmt_sbatch_time(time_suggestion_s(T, band_eff)),
                    points_used=0, degraded=False, band=band_eff, error_min=band_eff * T / 60.0,
                    source="phase-1 t0 anchor s/step x steps + startup (no rebuilt point)")
        comp = (pipe.get("T_updated_compute_s") if pipe["T_updated_s"]
                else pipe.get("T_cold_compute_s"))
        if comp is None:
            comp = T - ctx.startup_s
        init_s, init_src = prediction_init(ctx, ctx.startup_s, ctx.startup_source)
        add_run_length_io(pred, io_run, comp, init_s, band_eff, init_src)
        T = pred["predicted_total_s"]                     # this path's range is the band around T
        pred["predicted_range_s"] = [T * (1 - band_eff), T * (1 + band_eff)]
    pipe["prediction"] = pred

    # ---------------- phase 3: physics advice ----------------
    m3 = _mark(ctx)
    advice, ph3, cands = None, None, []
    if not physics:
        ph3 = ("not run (physics advice is opt-in: pass --physics-advice with -O3)"
               if plan["level"] >= 3 or level_req >= 3 and plan["level"] >= 3
               else "not run (physics advice needs -O3)" if level_req < 3
               else "not run (stepped down to O%d)" % plan["level"])
    elif extension:
        ph3 = "not run (extension case: physics parameters are not tunable)"
    elif cap_hit():
        ph3 = "not run (max queue wait exceeded)"
    elif rebuilt is None or not layouts:
        ph3 = "not run (no rebuilt points)" if rebuilt is None else "not run (no layout)"
    elif ctx.elapsed() >= limit("physics"):
        ph3 = NOT_RUN_BUDGET
    elif wtune_physics_advice is None:
        ph3 = "not run (wtune_physics_advice not importable)"
    elif not physics_runner_supported(ctx.sbatch_script):
        ph3 = ("not run (campaign runner has no per-config physics namelists: %s not found in "
               "strict_campaign.sbatch/configs.py)" % PHYSICS_RUNNER_MARKER)
    else:
        try:
            if min(rp["steps"] * float(rp["dt"]) / 60.0 for rp in rebuilt) < MIN_WINDOW_MIN:
                ph3 = "not run (window < %d min of model time: mid+end frames need more)" \
                    % MIN_WINDOW_MIN
            cands = wtune_physics_advice.plan_candidates(
                ctx.namelist_path, max_candidates=min(int(physics_max_candidates),
                                                      PIPELINE_PHYSICS_MAX_CANDIDATES),
                case_name=ctx.case)
            if not cands and ph3 is None:
                ph3 = "not run (no applicable physics candidate: %s)" % (
                    getattr(cands, "note", "") or "none planned")
        except Exception as e:                              # noqa: BLE001
            ph3 = "failed: plan_candidates: %s: %s" % (type(e).__name__, e)
    if ph3 is None:
        try:
            api = wtune_physics_advice
            # a skipped/failed phase 2 must not skip phase 3 (unused time rolls forward): fall
            # back to phase 1's best layout; without a phase-2 twin the gate is the provisional
            # config-gate, and the weights come from the phase-3 baseline itself
            w3 = winner or dict(layout=layouts[0], nodes=layouts[0]["nodes"],
                                cores=layouts[0]["cores"], meas=None, points=None)
            lay, (n, c) = w3["layout"], (w3["nodes"], w3["cores"])
            wm = (w3.get("meas") or []) if w3.get("points") in (None, [rp["index"] for rp in rebuilt]) \
                else []
            rsub = pick_points(rebuilt, PIPELINE_PHYSICS_POINTS, sample_points,
                               [pm.s_normal for pm in wm] if wm else None)
            # only restart-compatible candidates (radt, bldt, cudt, ...) are measured; scheme swaps
            # cannot start from the shared restart and stay estimate-only rows
            measurable = [cd for cd in cands if cd.get("restart_compatible", True)]
            base_cfg, cand_cfgs = (build_physics_window_namelists(ctx, measurable, rsub,
                                                                  dict(lay["cfg"]))
                                   if measurable else (dict(lay["cfg"]), []))
            base_cfg = dict(base_cfg, window_frame=WINDOW_FRAME_LEVEL)
            cand_cfgs = [dict(cf, window_frame=WINDOW_FRAME_LEVEL) for cf in cand_cfgs]
            gctx = dataclasses.replace(ctx, nodes=n, core_budget=c, budget_s=limit("physics"),
                                       stage1_max_running=compute_slots(ctx.max_nodes, n))
            s_by = {"ANCHOR": lay["median"], **{cf["id"]: lay["median"] for cf in cand_cfgs}}
            # ONE job per point: baseline (ANCHOR) + the candidates one after the other from the
            # point's shared restart (+ ANCHOREND as drift probe); the 2 point jobs run in parallel
            top, _est, skip = (fit_stage2(gctx, cand_cfgs, base_cfg, rsub, s_by, None,
                                          left_s=limit("physics") - ctx.elapsed())
                               if cand_cfgs else ([], 0.0, None))
            if skip:
                ph3 = NOT_RUN_BUDGET + ": " + skip
            elif not cand_cfgs:
                advice = api.advice_table(list(cands), {}, {})
                ph3 = "partial: no restart-compatible candidate -- estimates only"
            else:
                meas, _g, jobs = measure_candidates_at_points(gctx, top, rsub, base_cfg,
                                                              s_by_cid=s_by)
                key = lambda pm: pm.at.isoformat()                          # noqa: E731
                wsrc = wm or meas.get("ANCHOR", [])        # no phase-2 meas: the baseline's points
                try:
                    if not pc_ok:
                        raise RuntimeError("no prediction context")
                    weights = api.weights_from_combine(
                        sorted(wsrc, key=lambda m: m.at), run_start, run_end, dt, stepra,
                        calibration=calibration, sample_points=sample_points)
                except Exception as e:                      # noqa: BLE001
                    weights = {key(pm): 1.0 for pm in wsrc}
                    pipe["physics_weights"] = "equal (combine unavailable: %s)" % e
                base_by = {key(pm): pm.s_normal for pm in meas.get("ANCHOR", [])}
                twin_e = next((e for e in ranked_all if e is not winner), None)
                used_twin = False
                speed, fid = {}, {}
                for cd in cands:
                    cid = cd["id"]
                    if cid not in {t["id"] for t in top} or cid not in meas:
                        continue
                    speed[cid] = api.predicted_speedup(
                        base_by, {key(pm): pm.s_normal for pm in meas[cid]}, weights)
                    vs = []
                    for idx, (rp, jid) in enumerate(zip(rsub, jobs)):
                        b, cw = (_window_frames(gctx, rp, "ANCHOR", jid),
                                 _window_frames(gctx, rp, cid, jid))
                        if not (b and cw):
                            continue
                        twin = (_window_frames(gctx, rp, twin_e["cid"], twin_e["jobs"][idx])
                                if twin_e and idx < len(twin_e["jobs"]) else None) or None
                        gate_dir = os.path.join(ctx.out_dir, "physics", "gate")
                        os.makedirs(gate_dir, exist_ok=True)
                        used_twin = used_twin or bool(twin)
                        vs.append(api.fidelity_verdict(
                            b, cw, twin=twin, judged_fields=api.PROGNOSTIC_FIELDS,
                            out_prefix=os.path.join(gate_dir, "%s_pt%d" % (cid, rp["index"]))))
                    fid[cid] = _merge_verdicts(vs)
                frame_runs.extend((rp, cid, jid) for cid in ["ANCHOR", ANCHOR_END_ID]
                                  + [t["id"] for t in top] for rp, jid in zip(rsub, jobs))
                # measured candidates + the estimate-only scheme swaps (cost-model prior)
                advice = api.advice_table(list(cands), speed,
                                          {k: v for k, v in fid.items() if v})
                ph3 = "ok" if len(top) == len(cand_cfgs) else (
                    "partial: %d of %d candidates fit the budget" % (len(top), len(cand_cfgs)))
                pipe["physics_points"] = [rp["point"].at.isoformat() for rp in rsub]
                pipe["physics_fidelity_gate"] = (
                    "chaos-floor (phase-2 twin)" if used_twin else
                    "config-gate, PROVISIONAL (no phase-2 twin: uncalibrated tolerance)")
                pipe["physics_baseline"] = ("phase-2 winner" if winner else
                                            "phase-1 best layout (phase 2 not run)")
        except QueueCapExceeded as e:
            ph3 = "stopped: %s" % e
        except Exception as e:                              # noqa: BLE001
            ph3 = "failed: %s: %s" % (type(e).__name__, e)
    pipe["phases"].append(_phase_record("physics", m3, _mark(ctx), limit("physics"), ph3))
    cut("physics")
    pipe["physics_advice"] = advice
    # the gate's json/csv are kept (<out>/physics/gate); the window frames are bulky: delete them
    # unless --keep-frames
    pipe["frames_kept"] = bool(keep_frames)
    pipe["frames_deleted"] = 0 if keep_frames else _delete_window_frames(ctx, frame_runs)
    _attach_phase_timing(ctx, pipe["phases"], slices)

    # ---------------- totals ----------------
    running = ctx.elapsed() - t_start
    predicted_real = (pred["predicted_total_s"] if pred else
                      (pipe.get("T_updated_compute_s", pipe["T_updated_s"] - ctx.startup_s)
                       + ctx.startup_s + (io_run["total_s"] if io_run else 0.0))
                      if pipe["T_updated_s"] else None)
    pipe["time_suggestion"] = (_fmt_sbatch_time(time_suggestion_s(predicted_real,
                                                                  band_eff))
                               if predicted_real else None)
    pipe["prediction_band"] = dict(method=(pplan or {}).get("method", "cold-start predict"),
                                   band=band_eff,
                                   error_min=(band_eff * predicted_real / 60.0
                                              if predicted_real else None))
    pipe["level"]["actual_running_s"] = ctx.elapsed() - t_start
    pipe["level"]["effective"] = plan["level"]
    pipe.update(
        total_running_s=running, total_wall_s=ctx.wall_elapsed(),
        total_queue_wait_s=ctx.queue_wait_s, queue_cap_hit=cap_hit(),
        total_controller_s=max(0.0, ctx.wall_elapsed() - running - ctx.queue_wait_s),
        predicted_real_s=predicted_real,
        predicted_real_source=("phase 2 combine()" if pred else "phase-1 anchor s/step x steps")
        + (" + run-length I/O" if io_run else ""),
        hard_deadline_s=(B if B != float("inf") else None),
        over_deadline_s=(max(0.0, running - B) if B != float("inf") else 0.0))
    report["pipeline"] = pipe
    if ctx.node_health:
        report["node_health"] = dict(ctx.node_health, checks=[
            {k: v for k, v in c.items() if k != "nodes"} for c in health_checks])
    report["job_ids"] = list(ctx.owned_jobs)
    report["cancelled_job_ids"] = list(ctx.cancelled_jobs)
    report["elapsed_minutes"] = running / 60.0
    report["budget_minutes"] = (B / 60.0 if B != float("inf") else None)
    return report


def render_pipeline_text(pipe):
    lv = pipe.get("level") or {}
    lines = ["", "=== wtune tune -O%s (decision 0059) ===" % lv.get("effective", "?"),
             "level O%s%s; time cap: %s (%s)  whole run implied by the t0 anchor (info): %s"
             % (lv.get("effective", "?"),
                " (requested O%s)" % lv["requested"]
                if lv.get("requested") != lv.get("effective") else "",
                _fmt_hms(pipe["B_s"]), pipe["budget_source"],
                _fmt_hms(pipe["T_updated_s"]) if pipe["T_updated_s"] else "-")]
    if lv.get("estimate_s") and "O1" in lv["estimate_s"]:
        est = {int(k[1:]): v for k, v in lv["estimate_s"].items()}
        act = lv.get("actual_running_s")
        lines.append("tuning time, estimated at start: %s%s"
                     % (format_level_estimates(est, lv["effective"]),
                        "; ACTUAL running time %.1f min vs estimated O%d ≈ %.1f min"
                        % (act / 60.0, lv["effective"], est[lv["effective"]] / 60.0)
                        if act is not None and lv["effective"] in est else ""))
    for sd in lv.get("stepped_down") or []:
        lines.append("STEPPED DOWN (%s) -> O%s%s" % (sd.get("when"), sd.get("to"),
                                                     ": " + sd["message"] if sd.get("message")
                                                     else ""))
    for ph in pipe["phases"]:
        lines.append("phase %-8s running %-8s (limit %-8s) wall %-8s queue wait %-8s controller "
                     "%-8s  %s"
                     % (ph["name"], _fmt_hms(ph["running_s"]), _fmt_hms(ph["limit_s"]),
                        _fmt_hms(ph["wall_s"]), _fmt_hms(ph["queue_wait_s"]),
                        _fmt_hms(ph["controller_s"]), ph["status"]))
        if ph.get("queue_s_max") is not None or ph.get("job_run_s_max") is not None:
            lines.append("      jobs %s: longest queue %s, longest run %s (sacct)"
                         % (",".join(ph.get("jobs") or []) or "-",
                            _fmt_hms(ph["queue_s_max"]) if ph.get("queue_s_max") is not None
                            else "-",
                            _fmt_hms(ph["job_run_s_max"]) if ph.get("job_run_s_max") is not None
                            else "-"))
    pb = pipe.get("prediction_band")
    if pb and pb.get("error_min") is not None:
        lines.append("whole-run time ±%.0f%% = ±%.1f min (%s)"
                     % (pb["band"] * 100, pb["error_min"], pb.get("method")))
    if pipe.get("time_suggestion"):
        lines.append("suggested --time=%s (T rounded up to 15 min plus one band, >= 1.2x T)"
                     % pipe["time_suggestion"])
    if pipe.get("short_run"):
        lines.append("SHORT RUN: " + pipe["short_run"]["message"])
    pl = pipe.get("predict_plan")
    if pl:
        lines.append("phase 2 plan: %s (%s left; startup %.0fs/run); tried: %s"
                     % (pl["chosen"] or "none fits" , _fmt_hms(max(0.0, pl["left_s"])),
                        pl["startup_s"], "; ".join("%s ~%s" % (t["plan"], _fmt_hms(t["estimate_s"]))
                                                   for t in pl["tried"])))
    if pipe.get("physics_fidelity_gate"):
        lines.append("physics fidelity gate: %s; baseline: %s"
                     % (pipe["physics_fidelity_gate"], pipe.get("physics_baseline")))
    pr = pipe.get("prediction")
    if pr:
        lines.append("predicted whole run: %s (range %s - %s) at N=%d cores/node=%s %s; "
                     "suggested --time=%s"
                     % (pr["predicted_total_hms"], _fmt_hms(pr["predicted_range_s"][0]),
                        _fmt_hms(pr["predicted_range_s"][1]), pr["nodes"], pr["cores_per_node"],
                        pr["config"], pr["time_suggestion"]))
        if pr.get("breakdown"):
            lines.append("  " + pr["breakdown"] + ("  [init: %s]" % pr["init_source"]
                                                   if pr.get("init_source") else ""))
        if pr.get("segment_note"):
            lines.append("  NOTE: " + pr["segment_note"])
    adv = pipe.get("physics_advice")
    if adv:
        lines.append("physics advice (ADVISORY ONLY, not applied; points %s):"
                     % ", ".join(pipe.get("physics_points") or []))
        for row in adv:
            sv = row.get("predicted_saving_pct")
            lines.append("  %-10s saving %-7s fidelity %-8s -> %s  (%s)"
                         % (row.get("id"), ("%.1f%%" % sv) if sv is not None else "-",
                            row.get("fidelity", "-"), row.get("recommendation", "-"),
                            row.get("label", "")))
    lines.append("tune running time %s (wall %s = running + queue wait %s + controller %s); "
                 "predicted real run %s (%s)%s"
                 % (_fmt_hms(pipe.get("total_running_s", 0.0)), _fmt_hms(pipe.get("total_wall_s", 0.0)),
                    _fmt_hms(pipe.get("total_queue_wait_s", 0.0)),
                    _fmt_hms(pipe.get("total_controller_s", 0.0)),
                    _fmt_hms(pipe.get("predicted_real_s") or 0), pipe.get("predicted_real_source"),
                    "  [STOPPED: max queue wait exceeded]" if pipe.get("queue_cap_hit") else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# node health pre-flight (all modes): scc 2026-10-01 -- work2/work10 lost the /nfs/home mount while
# sinfo still said idle (jobs died in 0-1 s, a 4-node mpirun hung 600 s in the prted launch)
# ---------------------------------------------------------------------------
PREFLIGHT_TIMEOUT_S = 45.0
HEALTH_BAD_STATES = ("down", "drain", "fail", "maint", "unk", "no_respond", "inval", "powered")
DEFAULT_WRF_TREE = "$HOME/wrf-stack/wrf/nvhpc-fast-adios2-dmsm/WRF"     # campaign/sites/scc.sh's W


def default_preflight_runner(argv, timeout_s):
    """(returncode, stdout+stderr); raises subprocess.TimeoutExpired / OSError."""
    out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    return out.returncode, ((out.stdout or "") + (out.stderr or "")).strip()


def _probe_script(case_dir, src_dir, wrf_exe):
    q = shlex.quote
    return ('[ -d "$HOME" ] || { echo "no $HOME"; exit 3; }; '
            '[ -r %s ] || { echo "case dir unreadable: "%s; exit 3; }; '
            '[ -r %s ] || { echo "wtune-src unreadable: "%s; exit 3; }; '
            '[ -x %s ] || { echo "wrf.exe not executable: "%s; exit 3; }; echo ok'
            % (q(case_dir), q(case_dir), q(src_dir), q(src_dir), q(wrf_exe), q(wrf_exe)))


def node_preflight(runner, *, case_dir, src_dir, wrf_exe, requested, timeout_s=PREFLIGHT_TIMEOUT_S,
                   when="start"):
    """Which nodes can actually run a job. sinfo (`-h -N -o "%n %T"`) gives each node's state: head
    is never used, down/drain/fail/... states are unusable. Every other node gets ONE parallel
    probe `srun -w <n> -N1 -n1 -t 1 --exclude=head` that checks $HOME, the case dir, wtune-src and
    the site's wrf.exe; a failure or a timeout puts the node into the exclusion set (a timeout on a
    fully ALLOCATED node only means it is busy: kept, marked unprobed). The failures are published
    for this process's sbatch commands in WTUNE_PREFLIGHT_EXCLUDE (merged with the user's
    WTUNE_SBATCH_EXCLUDE by scc_sweep.excluded_nodes). Returns the `node_health` report dict;
    max_nodes_effective = min(requested, healthy nodes). When sinfo itself is unavailable the
    pre-flight is skipped (no exclusion, max_nodes unchanged), and says so."""
    t0 = time.monotonic()
    health = dict(when=when, skipped=None, nodes={}, failed=[], user_exclude=[], exclusions=[],
                  healthy=None, max_nodes_requested=int(requested),
                  max_nodes_effective=int(requested), probe_s=0.0)
    user = [n.strip() for n in os.environ.get("WTUNE_SBATCH_EXCLUDE", "").split(",") if n.strip()]
    try:
        rc, text = runner(["sinfo", "-h", "-N", "-o", "%n %T"], 30)
        if rc != 0:
            raise OSError("sinfo rc=%d: %s" % (rc, text))
    except (OSError, subprocess.SubprocessError) as e:
        health["skipped"] = "sinfo unavailable (%s: %s)" % (type(e).__name__, e)
        health["exclusions"] = sweep.excluded_nodes()
        return health
    states = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] != "head":
            prev = states.get(parts[0])
            bad = lambda st: any(b in st.lower() for b in HEALTH_BAD_STATES) or st.endswith("*")
            if prev is None or (bad(parts[1]) and not bad(prev)):
                states[parts[0]] = parts[1]
    to_probe = []
    for n, st in sorted(states.items()):
        low = st.lower()
        if n in user:
            health["nodes"][n] = dict(state=st, ok=False, probed=False,
                                      reason="excluded by WTUNE_SBATCH_EXCLUDE")
        elif any(b in low for b in HEALTH_BAD_STATES) or st.endswith("*"):
            health["nodes"][n] = dict(state=st, ok=False, probed=False,
                                      reason="sinfo state %s" % st)
        else:
            to_probe.append(n)
    script = _probe_script(case_dir, src_dir, wrf_exe)

    def probe(n):
        argv = ["srun", "-w", n, "-N1", "-n1", "-t", "1", "--exclude=head", "bash", "-c", script]
        st = states[n]
        try:
            rc, out = runner(argv, timeout_s)
        except subprocess.TimeoutExpired:
            if st.lower().startswith("alloc"):
                return n, dict(state=st, ok=True, probed=False,
                               reason="allocated by another job: not probed (timed out waiting)")
            return n, dict(state=st, ok=False, probed=True,
                           reason="probe timed out after %gs" % timeout_s)
        except (OSError, subprocess.SubprocessError) as e:
            return n, dict(state=st, ok=False, probed=True, reason="probe error: %s" % e)
        if rc == 0:
            return n, dict(state=st, ok=True, probed=True, reason="")
        return n, dict(state=st, ok=False, probed=True,
                       reason="probe failed rc=%d: %s" % (rc, (out or "").splitlines()[-1]
                                                           if out else "no output"))
    if to_probe:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_probe)) as pool:
            for n, res in pool.map(probe, to_probe):
                health["nodes"][n] = res
    health["failed"] = sorted(n for n, r in health["nodes"].items() if not r["ok"])
    healthy = sorted(n for n, r in health["nodes"].items() if r["ok"])
    health.update(user_exclude=user, healthy=len(healthy),
                  healthy_nodes=healthy)
    os.environ[sweep.PREFLIGHT_EXCLUDE_ENV] = ",".join(health["failed"])
    health["exclusions"] = sweep.excluded_nodes()
    if healthy:
        health["max_nodes_effective"] = min(int(requested), len(healthy))
    else:
        health["warning"] = "no healthy node found; max_nodes left unchanged"
    health["probe_s"] = time.monotonic() - t0
    print("node pre-flight (%s): %d healthy of %d node(s); excluded %s; max_nodes %d -> %d (%.1fs, "
          "controller time)" % (when, len(healthy), len(states), ",".join(health["failed"]) or "-",
                                 int(requested), health["max_nodes_effective"],
                                 health["probe_s"]))
    return health


def render_node_health_text(h):
    lines = ["", "--- node health pre-flight (latest check: %s) ---" % h.get("when", "start")]
    if h.get("skipped"):
        lines.append("skipped: %s" % h["skipped"])
        return "\n".join(lines)
    for n, r in sorted(h["nodes"].items()):
        lines.append("%-8s %-12s %-5s %s" % (n, r["state"], "ok" if r["ok"] else "FAIL",
                                              r["reason"]))
    lines.append("exclusions: %s; healthy %d; max_nodes %d -> %d; probe %.1fs (controller time, "
                 "not budget)" % (",".join(h["exclusions"]), h["healthy"], h["max_nodes_requested"],
                                  h["max_nodes_effective"], h["probe_s"]))
    for c in h.get("checks") or []:
        lines.append("  check %-14s healthy %s, failed %s, max_nodes -> %s, probe %.1fs"
                     % (c.get("when"), c.get("healthy"), ",".join(c.get("failed") or []) or "-",
                        c.get("max_nodes_effective"), c.get("probe_s") or 0.0))
    return "\n".join(lines)


def run_level0(ctx, profile, total_steps, nodes_policy="knee"):
    """-O0: NO jobs. The proxy's node count (choose_nodes), the all-cores hybrid anchor layout at
    it (no measurement) and wtune_predict's cold-start whole-run time (+-25%)."""
    n0, note, _warn = choose_nodes(profile, ctx.max_nodes, nodes_policy)
    cfg, why = None, None
    for n in [n0] + [m for m in range(1, ctx.max_nodes + 1) if m != n0]:
        cfg, why = _sweep_anchor(ctx, n)
        if cfg:
            n0 = n
            break
    if cfg is None:
        raise SystemExit("O0: no legal layout at any node count (%s)" % why)
    band = COLD_START_BAND
    T, s_n, src = (cold_start_prediction(ctx, n0, cfg, total_steps) if total_steps
                   else (None, None, "run length unknown"))
    io_run = pipeline_run_length_io(ctx.namelist_path) if T else None
    if T:
        T += io_run["total_s"] if io_run else 0.0
    pt = local_param_table(cfg, n0)
    _assert_report_is_strict_only(ctx, pt)
    pipe = dict(
        total_steps=total_steps, B_s=None, budget_source="no jobs", T_updated_s=None, phases=[],
        rebuild=None, extension=False, queue_cap_hit=False,
        level=dict(requested=0, effective=0, estimate_s={"O0": 0.0}, stepped_down=[],
                   actual_running_s=0.0),
        prediction=dict(layout_id="ANCHOR", nodes=n0, cores_per_node=ctx.cores_per_node,
                        config={k: cfg.get(k) for k in ("ranks", "omp", "nx", "ny", "numtiles",
                                                        "dist")},
                        predicted_total_s=T, predicted_total_hms=_fmt_hms(T) if T else None,
                        predicted_range_s=[T * (1 - band), T * (1 + band)] if T else None,
                        time_suggestion=(_fmt_sbatch_time(time_suggestion_s(T, band)) if T else None),
                        band=band, error_min=(band * T / 60.0 if T else None), points_used=0,
                        degraded=False, source=src + (" + run-length I/O" if io_run else "")),
        time_suggestion=(_fmt_sbatch_time(time_suggestion_s(T, band)) if T else None),
        prediction_band=dict(method="cold-start predict", band=band,
                             error_min=(band * T / 60.0 if T else None)),
        total_running_s=0.0, total_wall_s=ctx.wall_elapsed(), total_queue_wait_s=0.0,
        total_controller_s=ctx.wall_elapsed(), predicted_real_s=T, hard_deadline_s=None,
        over_deadline_s=0.0, physics_advice=None)
    if T:
        # same fields as the -O1..3 predictions; T already holds the I/O, ranges stay symmetric
        p0 = pipe["prediction"]
        io_s = io_run["total_s"] if io_run else 0.0
        add_run_length_io(p0, io_run, T - io_s - ctx.startup_s, ctx.startup_s, band,
                          ctx.startup_source)
        p0["predicted_range_s"] = [T * (1 - band), T * (1 + band)]
    print("O0: no jobs -- nodes=%d (%s), layout %s, cold-start whole run %s ±%.0f%% (%s)"
          % (n0, note, {k: cfg.get(k) for k in ("ranks", "omp", "nx", "ny")},
             _fmt_hms(T) if T else "unknown", band * 100, src))
    return dict(case=ctx.case, pipeline_only=True, nodes=n0,
                nodes_note="O0: proxy node count, no measurement", param_table=pt, pipeline=pipe,
                cores_per_node=ctx.cores_per_node, job_ids=[], cancelled_job_ids=[],
                best=dict(id="ANCHOR", **{k: cfg.get(k) for k in ("ranks", "omp", "numtiles",
                                                                   "dist")},
                          nproc_x=cfg.get("nx"), nproc_y=cfg.get("ny")))


def run(args, sbatch_runner=None, sacct_runner=None, scancel_runner=None, ledger_reader=None,
        proposer=None, clock=None, preflight_runner=None):
    sbatch_runner = sbatch_runner or sweep.default_sbatch_runner
    sacct_runner = sacct_runner or sweep.default_sacct_runner
    scancel_runner = scancel_runner or default_scancel_runner
    clock = clock or RealClock()
    ledger_dir = sweep._expand(args.ledger_dir)
    ledger_reader = ledger_reader or (lambda case, job_ids: default_ledger_reader(
        ledger_dir, case, job_ids))

    # --config resolution (wtune_config.py). Precedence: CLI flag > file > built-in defaults.
    # With no --config, `cfg` is resolve_config({})'s pure built-in-defaults result, so every
    # merge below reproduces this module's own pre---config hardcoded argparse defaults exactly
    # (DEFAULT_MAX_NODES=13, "auto", "knee", DEFAULT_STEPS=20, CONFIRM_REPS=2).
    config_path = getattr(args, "config", None)
    if config_path:
        try:
            cfg = wtune_config.load_config(config_path)
        except wtune_config.ConfigError as e:
            raise SystemExit("--config %s is invalid:\n%s"
                              % (config_path, "\n".join("  - %s" % m for m in e.errors)))
    else:
        cfg, _errs = wtune_config.resolve_config({})

    case = args.case if args.case is not None else cfg["case"]
    if case is None:
        if cfg.get("case_dir"):
            raise SystemExit(
                "config declares case_dir=%r, but wtune_tune.py cannot run a tune against an "
                "arbitrary case_dir yet (only `wtune config show`/`config init` preview it) -- "
                "pass a registered case name positionally, or set 'case' in --config instead"
                % cfg["case_dir"])
        raise SystemExit("case is required: pass it positionally or set 'case' in --config")

    minutes = args.minutes if args.minutes is not None else cfg["budget"]["minutes"]
    # effort level (-O0..-O3 / --level; --pipeline full = O3): selects the pipeline, where
    # --minutes is only an OPTIONAL hard cap
    level = getattr(args, "level", None)
    if level is None and getattr(args, "pipeline", None) == "full":
        level = 3
    if minutes is None and level is None:
        raise SystemExit("--minutes is required: pass it on the command line or set "
                          "budget.minutes in --config")

    nodes_raw = args.nodes if args.nodes is not None else cfg["budget"]["nodes"]
    if (args.nodes is None and nodes_raw == "auto"
            and not _cfg_key_explicit(config_path, "budget", "nodes")
            and (getattr(args, "rank_by", None) or cfg["probe"].get("rank_by") or "t0") == "t0"):
        nodes_raw = "search"      # default: MEASURE s/step across node counts (node sweep)
    if isinstance(nodes_raw, list):
        if len(nodes_raw) != 1:
            raise SystemExit(
                "budget.nodes as a list of multiple node counts (%r) is not yet run by `wtune "
                "tune` (its pipeline targets one node count per run) -- use `wtune config show` "
                "to preview candidate counts across node counts, or pass a single --nodes value"
                % (nodes_raw,))
        nodes_raw = nodes_raw[0]

    nodes_policy = args.nodes_policy if args.nodes_policy is not None else cfg["budget"]["nodes_policy"]
    max_nodes_requested = args.max_nodes if args.max_nodes is not None else cfg["budget"]["max_nodes"]
    steps_cfg = args.steps if args.steps is not None else cfg["probe"]["steps"]
    confirm_reps = args.confirm_reps if args.confirm_reps is not None else cfg["probe"]["confirm_reps"]
    rank_by = getattr(args, "rank_by", None) or cfg["probe"].get("rank_by") or "t0"
    stage1_mode = getattr(args, "stage1", None) or cfg["probe"].get("stage1") or "t0"
    stage2_minutes = getattr(args, "stage2_minutes", None)
    if stage2_minutes is None:
        stage2_minutes = cfg["probe"].get("stage2_minutes")
    if stage1_mode not in STAGE1_CHOICES:
        raise SystemExit("--stage1 %r not in %s" % (stage1_mode, list(STAGE1_CHOICES)))
    if getattr(args, "points_by", None) is not None:
        cfg["probe"]["points_by"] = args.points_by
    if getattr(args, "max_points", None) is not None:
        cfg["probe"]["max_points"] = args.max_points
    elif rank_by == "predicted" and not _probe_key_explicit(config_path, "max_points"):
        cfg["probe"]["max_points"] = PREDICTED_MAX_POINTS
    restart_io_form = getattr(args, "restart_io_form", None) or \
        cfg["probe"].get("restart_io_form") or 14
    if rank_by not in RANK_BY_CHOICES:
        raise SystemExit("--rank-by %r not in %s" % (rank_by, list(RANK_BY_CHOICES)))

    case_json, namelist_path, profile = resolve_case(case, args.cores_per_node)
    is_fire = wtune_config.is_fire_case(case_json, namelist_path)
    import wtune_extensions
    _ext_notes, _ext_refusal = wtune_extensions.tune_gate(namelist_path)
    if _ext_refusal:
        raise SystemExit(_ext_refusal)
    for _n in _ext_notes:
        print("EXTENSION: %s" % _n)

    if steps_cfg == "auto":
        steps, steps_note = resolve_auto_probe_steps(namelist_path)
    else:
        steps, steps_note = steps_cfg, None
    if steps_note:
        print("NOTE: %s" % steps_note)
    ranking_start_note = None
    if cfg["probe"].get("ranking_start") == "spinup":
        # 2026-09-26 multi-point redesign, Layer A (see this task's report): restart-based
        # ranking-probe start needs a shared-restart hand-off in campaign/strict_campaign.sbatch's
        # run loop that is not wired yet (campaign/configs.py's edit_namelist(restart_from=...,
        # write_restart_interval=..., restart_io_form=...) is ready -- see that function's
        # docstring -- but nothing yet submits the anchor's spin-up run or stages its wrfrst into
        # each candidate's RUNDIR). Every candidate is still probed from t0 until that wiring
        # lands; SAY so rather than silently ignoring probe.ranking_start=spinup, same honesty
        # convention as prediction.measure_r_rad below.
        ranking_start_note = (
            "probe.ranking_start=spinup requested but restart-based ranking-probe start is NOT "
            "YET WIRED to the campaign submission pipeline (campaign/configs.py's edit_namelist "
            "restart_from/write_restart_interval/restart_io_form params exist; the sbatch run-"
            "loop wiring that would use them does not yet) -- every candidate is still probed "
            "from t0, exactly as before probe.ranking_start existed.")
        print("NOTE: %s" % ranking_start_note)

    max_nodes = min(int(max_nodes_requested), DEFAULT_MAX_NODES)
    # node health pre-flight (before ctx.start_time: controller time, never budget)
    node_health, preflight = None, None
    if not getattr(args, "no_node_preflight", False) and level != 0:      # O0 runs no jobs
        _src_dir = os.path.dirname(sweep.campaign_dir_for(args.sbatch_script) or _HERE)
        _wrf = os.path.join(sweep._expand(os.environ.get("WRF_TREE") or DEFAULT_WRF_TREE),
                            "main", "wrf.exe")
        _requested = max_nodes

        def preflight(when="start"):
            return node_preflight(preflight_runner or default_preflight_runner,
                                  case_dir=sweep._expand(case_json["prep_dir"]), src_dir=_src_dir,
                                  wrf_exe=_wrf, requested=_requested, when=when)
        node_health = preflight("start")
        max_nodes = node_health["max_nodes_effective"]
    pipeline = "full" if level is not None else None
    if pipeline == "full":
        if args.nodes not in (None, "search") or rank_by == "predicted":
            raise SystemExit("-O/--pipeline runs its own node search: do not combine it with "
                              "--nodes N/auto or --rank-by predicted")
        nodes_raw = "search"
    node_search = (nodes_raw == "search")
    if node_search and rank_by == "predicted":
        raise SystemExit("--nodes search is not supported with --rank-by predicted; pass --nodes "
                          "N or --nodes auto")
    def _nopt(name, default):          # CLI flag > --config budget.<name> > built-in default
        v = getattr(args, name, None)
        if v is None:
            v = cfg["budget"].get(name)
        return default if v is None else v
    node_objective = _nopt("node_objective", "time")
    deadline_minutes = _nopt("deadline_minutes", None)
    min_efficiency = _nopt("min_efficiency", wtune_nodes.DEFAULT_MIN_EFFICIENCY)
    node_finalists = max(1, int(_nopt("node_finalists", 2)))
    sweep_budget_frac = float(_nopt("sweep_budget_frac", DEFAULT_SWEEP_BUDGET_FRAC))
    core_options_raw = getattr(args, "cores_per_node_options", None)
    core_options = None                    # None -> wtune_nodes.default_core_options(C)
    if core_options_raw:
        try:
            core_options = sorted({int(v) for v in core_options_raw.split(",") if v.strip()},
                                  reverse=True)
        except ValueError:
            raise SystemExit("--cores-per-node-options must be a comma list of integers, got %r"
                              % core_options_raw)
        if not core_options or any(v < 1 or v > args.cores_per_node for v in core_options):
            raise SystemExit("--cores-per-node-options values must be in 1..%d (--cores-per-node), "
                              "got %r" % (args.cores_per_node, core_options_raw))
    total_steps = _total_steps(namelist_path) if node_search else None
    if node_search and node_objective == "cost" and total_steps is None:
        raise SystemExit("--node-objective cost needs the case's total step count (run length / "
                          "time_step from the namelist), which could not be determined")
    if node_search:
        nodes, nodes_note = 1, "node search (measured)"
    elif nodes_raw in ("auto", None):
        nodes, nodes_note, nodes_warning = choose_nodes(profile, max_nodes, nodes_policy)
        print("WARNING: %s" % nodes_warning)
    else:
        nodes = int(nodes_raw)
        nodes_note = "user-specified (cap=%d)" % max_nodes
        if nodes > max_nodes:
            raise SystemExit("--nodes %d exceeds --max-nodes %d" % (nodes, max_nodes))

    if args.out:
        out_dir = sweep._expand(args.out)
    else:
        import wtune_project
        _project_dir = wtune_project.find_project(explicit=getattr(args, "project", None))
        out_dir = (wtune_project.default_out_dir(_project_dir, "tune") if _project_dir
                  else sweep._expand("$HOME/wtune-runs/tune/%s" % case))
    os.makedirs(out_dir, exist_ok=True)

    ctx = Ctx(case=case, case_json=case_json, namelist_path=namelist_path, nodes=nodes,
              cores_per_node=args.cores_per_node, hardware_id=args.hardware_id, steps=steps,
              seed=args.seed, kappa=args.kappa, budget_s=(minutes * 60.0 if minutes is not None else float("inf")),
              max_nodes=max_nodes,
              out_dir=out_dir, ledger_dir=ledger_dir, env_source=args.env_source,
              sbatch_script=args.sbatch_script, sacct_cmd=args.sacct_cmd,
              poll_interval=args.poll_interval, sbatch_runner=sbatch_runner,
              sacct_runner=sacct_runner, scancel_runner=scancel_runner,
              ledger_reader=ledger_reader, proposer=proposer, clock=clock,
              search_cfg=cfg["search"], exclude_rules=cfg["exclude"], is_fire=is_fire,
              omp_cap=wtune_extensions.omp_cap_for_namelist(namelist_path),
              chem_e_sn=wtune_extensions.chem_e_sn_for_namelist(namelist_path),
              confirm_reps=confirm_reps, prediction_cfg=cfg["prediction"],
              probe_cfg=cfg["probe"], ranking_start_note=ranking_start_note,
              rank_by=rank_by, restart_io_form=int(restart_io_form), stage1_mode=stage1_mode)
    ctx.node_health = node_health
    ctx.start_time = clock.now()

    print("case=%s nodes=%s (%s) budget=%s max_nodes=%d out=%s"
          % (case, "search" if node_search else nodes, nodes_note,
             ("%.1f min" % minutes) if minutes is not None else "none", max_nodes, out_dir))

    s_est, _src, _n = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                 ctx.hardware_id)

    if rank_by == "predicted":
        # decision 0055: stage 1 at P*'s shared restart, stage 2 at every rebuilt point, winner
        # by predicted whole-run compute+init -- see run_predicted. Replaces round0..confirm.
        # Budget = RUNNING time (queue-only intervals are not charged), hard wall cap on top.
        ctx.budget_mode = "running"
        ctx.stage2_budget_s = stage2_minutes * 60.0 if stage2_minutes is not None else None
        max_wall_min = getattr(args, "max_wall_minutes", None)
        ctx.max_wall_explicit = max_wall_min is not None
        ctx.max_wall_s = (max_wall_min if max_wall_min is not None
                          else MAX_WALL_FACTOR * minutes) * 60.0
        print("predicted mode: budget %.1f min of RUNNING time, hard wall-clock cap %.1f min"
              % (minutes, ctx.max_wall_s / 60.0))
        report = run_predicted(ctx, nodes, nodes_note, s_est)
        if node_health:
            report["node_health"] = node_health
        _final_cancel_safety_net(ctx)
        text = render_predicted_report_text(report) + _attach_advice(report)
        print("")
        print(text)
        report["winner"] = final_winner(report)
        with open(os.path.join(out_dir, "tune_report.json"), "w") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        return report

    if level == 0:
        report = run_level0(ctx, profile, total_steps, nodes_policy)
        _emit_report(report, out_dir, node_health)
        return report
    if pipeline == "full":
        report = run_pipeline_full(
            ctx, profile, minutes=minutes, level=level, total_steps=total_steps,
            objective=node_objective,
            deadline_s=(deadline_minutes * 60.0 if deadline_minutes is not None else None),
            min_efficiency=min_efficiency, k_finalists=node_finalists, nodes_policy=nodes_policy,
            sweep_budget_frac=sweep_budget_frac, core_options=core_options,
            extension=bool(_ext_notes or ctx.omp_cap or ctx.chem_e_sn),
            physics=bool(getattr(args, "physics_advice", False)),
            physics_max_candidates=int(getattr(args, "physics_max_candidates", None)
                                       or PIPELINE_PHYSICS_MAX_CANDIDATES),
            recheck=preflight, stop_if_short=bool(getattr(args, "stop_if_short", False)),
            precision_min=float(args.precision if getattr(args, "precision", None) is not None
                                else PRECISION_DEFAULT_MIN),
            max_queue_hours=(getattr(args, "max_queue_hours", None)
                             if getattr(args, "max_queue_hours", None) is not None
                             else PIPELINE_MAX_QUEUE_HOURS),
            keep_frames=bool(getattr(args, "keep_frames", False)))
        _final_cancel_safety_net(ctx)
        _emit_report(report, out_dir, node_health)
        return report
    if node_search:
        report = run_node_search(ctx, profile, objective=node_objective,
                                  deadline_s=(deadline_minutes * 60.0
                                              if deadline_minutes is not None else None),
                                  min_efficiency=min_efficiency, k_finalists=node_finalists,
                                  total_steps=total_steps, nodes_policy=nodes_policy,
                                  sweep_budget_frac=sweep_budget_frac,
                                  core_options=core_options)
        _final_cancel_safety_net(ctx)
        _emit_report(report, out_dir, node_health)
        return report

    all_results, confirm_results, anchor_cid = run_tuning_at(ctx, s_est)

    _final_cancel_safety_net(ctx)

    report = build_report(ctx, nodes, nodes_note, all_results, confirm_results, anchor_cid)
    _emit_report(report, out_dir, node_health)
    return report


def run_tuning_at(ctx, s_est=None, adaptive=True, confirm=True):
    """round0 -> adaptive rounds -> confirm at ctx.nodes. Returns (all_results, confirm_results,
    anchor_cid). Extracted unchanged from run() so a node-search finalist can reuse it."""
    if s_est is None:
        s_est, _src, _n = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                     ctx.hardware_id)
    job_ids0, configs_by_cid, rows0 = run_round0(ctx)
    all_results = aggregate_by_cid(rows0, configs_by_cid)
    anchor_cid = "ANCHOR" if "ANCHOR" in configs_by_cid else None

    # Bug fix: s_est above is the PRE-round0 estimate (ledger history, or the fallback when this
    # case has never run on scc before -- correct for SIZING round0 itself, since nothing of this
    # case's own is measured yet). Every stage after round0 must use what round0 itself just
    # measured instead, or every later job looks too slow against the stale fallback and nothing
    # after round0 ever runs even when real per-step cost is far below the fallback (2026-09-22
    # conus12 run: fallback stayed 10s/step all the way through, so neither the adaptive round nor
    # confirm ever submitted, leaving 13.6 of 30 minutes unused).
    s_est = update_s_est(all_results, s_est)

    if job_ids0 and adaptive and not ctx.full_space:       # the full space leaves nothing to propose
        run_adaptive_rounds(ctx, configs_by_cid, all_results, s_est)

    confirm_job_id, confirm_results = (None, None)
    if all_results and confirm:
        confirm_job_id, confirm_results = run_confirm(ctx, configs_by_cid, all_results, anchor_cid,
                                                        s_est)

    return all_results, confirm_results, anchor_cid


def _attach_advice(report):
    """Decision 0061: advice-only tail of the report (tune_report.json["advice"] + text). Never
    fails the tune: an error is recorded under report["advice_error"] instead."""
    try:
        import wtune_advise
        res = wtune_advise.advise_case_id(report["case"])
        report["advice"] = res["advice"]
        report["advice_unavailable"] = res["unavailable"]
        return "\n\n" + wtune_advise.render_text(res)
    except Exception as e:                                  # noqa: BLE001
        report["advice"] = []
        report["advice_error"] = "%s: %s" % (type(e).__name__, e)
        return ""


def final_winner(report):
    """Top-level report["winner"]: the FINAL pick (report["nodes"] + report["best"], i.e. after a
    phase-2 overturn too) as a node_selection.layouts-style row (cid, nodes, cores_per_node, ranks,
    omp, nx, ny, numtiles, median s/step, ...). The matching measured layouts row when there is
    one, else built from the best ranking row (cores/node: cores_used_per_node, else the core
    choice for that N, else cores_per_node). None for an unmeasured pick (O0, aborted)."""
    best, n = report.get("best"), report.get("nodes")
    if not best or n is None or report.get("pipeline_only") or report.get("pipeline_aborted"):
        return None
    ns = report.get("node_selection") or {}
    c = report.get("cores_used_per_node")
    if c is None:
        c = ((ns.get("core_choice") or {}).get(str(n)) or {}).get("cores")
    if c is None:
        c = report.get("cores_per_node")
    key = (best.get("ranks"), best.get("omp"), best.get("nproc_x"), best.get("nproc_y"),
           best.get("numtiles"))
    for row in ns.get("layouts") or []:
        if (row.get("nodes") == n and (c is None or row.get("cores_per_node") == c)
                and (row.get("ranks"), row.get("omp"), row.get("nx"), row.get("ny"),
                     row.get("numtiles")) == key):
            return dict(row)
    return dict(source="ranking", nodes=n, cores_per_node=c, ranks=best.get("ranks"),
                omp=best.get("omp"), nx=best.get("nproc_x"), ny=best.get("nproc_y"),
                numtiles=best.get("numtiles"), median=best.get("median_s_per_step"),
                reps=best.get("n_measurements"), cid=best.get("id"))


def _emit_report(report, out_dir, node_health=None):
    if node_health and "node_health" not in report:
        report["node_health"] = node_health
    report["winner"] = final_winner(report)
    only = report.get("pipeline_aborted") or report.get("pipeline_only")
    advice_text = "" if only else _attach_advice(report)
    print("")
    print(render_pipeline_text(report["pipeline"]) if only
          else render_report_text(report) + advice_text)
    with open(os.path.join(out_dir, "tune_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(prog="wtune_tune.py")
    p.add_argument("case", nargs="?", default=None,
                    help="registered campaign/cases.json case id; required here or via "
                         "--config's 'case' key")
    p.add_argument("--config", default=None,
                    help="wtune_config.py YAML file: search-space restrictions + budget/probe/"
                         "prediction defaults (wtune_config.py); CLI flags given explicitly still "
                         "override it, which overrides this module's own built-in defaults")
    p.add_argument("--site", default="scc")
    p.add_argument("--minutes", type=float, default=None,
                    help="required here or via --config's budget.minutes")
    p.add_argument("--nodes", default=None,
                    help="N = fixed node count; 'search' (DEFAULT) = measure the ANCHOR at each "
                         "legal node count, choose N by --node-objective, then tune at the "
                         "finalist N; 'auto' = uncalibrated efficiency proxy (--nodes-policy), "
                         "no measurement. (--rank-by predicted defaults to 'auto'; a --config "
                         "budget.nodes also overrides the default)")
    p.add_argument("--node-objective", choices=wtune_nodes.OBJECTIVES, default=None,
                    dest="node_objective",
                    help="--nodes search: time = fastest (smallest N within noise), cost = fewest "
                         "node-hours (within --deadline-minutes), knee = largest N with "
                         "efficiency >= --min-efficiency (default: time)")
    p.add_argument("--deadline-minutes", type=float, default=None, dest="deadline_minutes",
                    help="--node-objective cost: whole-run wall-time limit (minutes)")
    p.add_argument("--min-efficiency", type=float, default=None, dest="min_efficiency",
                    help="--node-objective knee: efficiency threshold (default %.2f)"
                         % wtune_nodes.DEFAULT_MIN_EFFICIENCY)
    p.add_argument("--sweep-budget-frac", type=float, default=None, dest="sweep_budget_frac",
                    help="--nodes search: share of --minutes the node sweep may use; once spent, "
                         "larger N are skipped ('sweep budget cap') so the finalists keep budget "
                         "(default %.1f, or --config's budget.sweep_budget_frac)"
                         % DEFAULT_SWEEP_BUDGET_FRAC)
    p.add_argument("--cores-per-node-options", default=None, dest="cores_per_node_options",
                    help="--nodes search: comma list of cores per node to measure at the finalist "
                         "node counts (the rest of the node idles). Default derived from the "
                         "hardware core count C: C, C-2, round(C*8/9), round(C*2/3), C/2 "
                         "(72 -> 72,70,64,48,36); layouts that are not legal are skipped. "
                         "'--cores-per-node-options 72' disables the dimension")
    p.add_argument("--pipeline", choices=("full",), default=None,
                    help="full = -O3 (kept for compatibility): one pass -- layout search (node, "
                         "cores/node, ranks, threads, decomposition, tiles), whole-run "
                         "prediction from rebuilt points, physics advice (decision 0059)")
    p.add_argument("--max-queue-hours", type=float, default=None, dest="max_queue_hours",
                    help="--pipeline full: safety cap on the accumulated queue-only wait (hours, "
                         "default %g); queue wait is not charged to --minutes, but past this the "
                         "run stops and reports" % PIPELINE_MAX_QUEUE_HOURS)
    p.add_argument("--no-node-preflight", action="store_true", dest="no_node_preflight",
                    help="skip the node health pre-flight (sinfo states + one 1-minute srun probe "
                         "per node checking $HOME, the case dir, wtune-src and wrf.exe); by "
                         "default failing nodes are excluded from every sbatch and max_nodes is "
                         "capped at the healthy node count")
    for lv in (0, 1, 2, 3):
        p.add_argument("-O%d" % lv, action="store_const", const=lv, dest="level",
                        help="effort level %d: %s (layout search effort + physics advice; the "
                             "prediction points come from the case type)" % (lv,
                                                                            LEVELS[lv]["method"]))
    p.add_argument("--precision", type=float, default=None, dest="precision",
                    help="whole-run wall time needed to +-MIN minutes (default %g): the case type "
                         "decides how many prediction points that takes "
                         "(wtune_cost_drivers.plan_points_for_case)" % PRECISION_DEFAULT_MIN)
    p.add_argument("--level", type=int, choices=(0, 1, 2, 3), default=None, dest="level",
                    help="same as -O<level>; selects the one-pass pipeline (default level when "
                         "only --pipeline full is given: 3). The estimated tuning time of every "
                         "level is printed before anything runs; --minutes is an optional hard cap")
    p.add_argument("--stop-if-short", action="store_true", dest="stop_if_short",
                    help="-O levels: when the whole run is shorter than the estimated tuning time, "
                         "step down to the highest level whose estimate fits (never below O1). "
                         "OFF by default: the layout search is not cut short; also active when "
                         "--minutes is given")
    p.add_argument("--physics-advice", action="store_true", dest="physics_advice",
                    help="-O3: also run phase 3, the physics advice (opt-in, not part of the "
                         "default run)")
    p.add_argument("--keep-frames", action="store_true", dest="keep_frames",
                    help="--pipeline full: keep the phase-2/3 window-frame .nc files after the "
                         "fidelity gate (default: delete them, keeping the gate json/csv)")
    p.add_argument("--physics-max-candidates", type=int, default=None,
                    dest="physics_max_candidates",
                    help="--pipeline full: physics candidates tried in phase 3 (default %d)"
                         % PIPELINE_PHYSICS_MAX_CANDIDATES)
    p.add_argument("--node-finalists", type=int, default=None, dest="node_finalists",
                    help="--nodes search: how many node counts get the full round0/adaptive/"
                         "confirm search (default 2)")
    p.add_argument("--nodes-policy", choices=sorted(NODES_POLICIES), default=None,
                    dest="nodes_policy",
                    help="how --nodes auto picks a node count from the (uncalibrated) profile: "
                         "min=smallest legal, cap=largest legal within --max-nodes, "
                         "knee=stop doubling once marginal speedup drops below 1.5x "
                         "(default: 'knee', or --config's budget.nodes_policy)")
    p.add_argument("--max-nodes", type=int, default=None, dest="max_nodes",
                    help="default: %d (or --config's budget.max_nodes)" % DEFAULT_MAX_NODES)
    p.add_argument("--out", default=None)
    p.add_argument("--project", default=None, help="project dir (decision 0058): when given "
                   "(or via $WTUNE_PROJECT / a project.yaml above cwd) and --out is not, "
                   "defaults --out to <project>/tune/<id>/ instead of $HOME/wtune-runs/tune/"
                   "<case>. Does not otherwise change this command's behaviour.")
    p.add_argument("--steps", type=int, default=None,
                    help="default: %d (or --config's probe.steps)" % DEFAULT_STEPS)
    p.add_argument("--confirm-reps", type=int, default=None, dest="confirm_reps",
                    help="default: %d (or --config's probe.confirm_reps)" % CONFIRM_REPS)
    p.add_argument("--rank-by", choices=RANK_BY_CHOICES, default=None, dest="rank_by",
                    help="t0 (default, or --config's probe.rank_by): rank by t0-probe s/step then "
                         "predicted wall time. predicted (decision 0055): rebuild ERA5 points, "
                         "spin each up once, rank every candidate from the shared restarts by "
                         "predicted whole-run compute+init")
    p.add_argument("--restart-io-form", type=int, choices=(14, 2), default=None,
                    dest="restart_io_form",
                    help="--rank-by predicted only: io_form of the shared ranking restart "
                         "(RANKING_RESTART_IO_FORM); default 14 (ADIOS2), or --config's "
                         "probe.restart_io_form")
    p.add_argument("--stage1", choices=STAGE1_CHOICES, default=None, dest="stage1",
                    help="--rank-by predicted only: stage-1 search. t0 (default, or --config's "
                         "probe.stage1): the broad t0 tuner search, rebuild prep running beside "
                         "it. pstar: every stage-1 job spins up at P* and starts candidates "
                         "from that restart (pre-2026-09-28 behaviour)")
    p.add_argument("--stage2-minutes", type=float, default=None, dest="stage2_minutes",
                    help="--rank-by predicted --stage1 t0 only: optional cap on stage 2's "
                         "running time (fit_stage2 trims the shortlist to it); default none -- "
                         "stage 2 always runs, its estimated cost is announced at start")
    p.add_argument("--max-points", type=int, default=None, dest="max_points",
                    help="probe.max_points override (--rank-by predicted default: %d)"
                         % PREDICTED_MAX_POINTS)
    p.add_argument("--points-by", choices=("time", "cost-indicators"), default=None,
                    dest="points_by",
                    help="probe.points_by override (decision 0060): place the prediction points by "
                         "sun/time (default) or by a weather cost proxy (wtune_cost_drivers; "
                         "sources from probe.indicator_sources)")
    p.add_argument("--max-wall-minutes", type=float, default=None, dest="max_wall_minutes",
                    help="--rank-by predicted only: hard wall-clock cap (default %gx --minutes); "
                         "the --minutes budget itself counts only RUNNING time there"
                         % MAX_WALL_FACTOR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    p.add_argument("--cores-per-node", type=int, default=sweep.DEFAULT_CORES_PER_NODE,
                    dest="cores_per_node")
    p.add_argument("--hardware-id", default=DEFAULT_HARDWARE_ID, dest="hardware_id")
    p.add_argument("--env-source", default=sweep.DEFAULT_ENV_SOURCE, dest="env_source")
    p.add_argument("--sbatch-script", default=sweep.DEFAULT_SBATCH_SCRIPT, dest="sbatch_script")
    p.add_argument("--ledger-dir", default=sweep.DEFAULT_LEDGER_DIR, dest="ledger_dir")
    p.add_argument("--sacct-cmd", default="sacct", dest="sacct_cmd")
    p.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S,
                    dest="poll_interval")
    return p


def main(argv=None):
    # Line-buffer stdout: under `nohup ... &` stdout is redirected to a file, which Python
    # block-buffers by default (not line-buffered like a tty), so the log stayed empty for the
    # whole run and only flushed at exit. This run can take the entire --minutes budget, so a
    # human watching the log needs each progress line as it happens, not at the end.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    args = build_arg_parser().parse_args(argv)
    if args.site != "scc":
        print("ERROR: only --site scc is supported", file=sys.stderr)
        return 2
    if (getattr(args, "level", None) is None and getattr(args, "pipeline", None) is None
            and args.nodes is None and getattr(args, "rank_by", None) is None):
        args.level = DEFAULT_LEVEL            # the default run: the O3 full layout search
    os.environ.setdefault("SITE", "scc")
    try:
        run(args)
    except SystemExit as e:
        if e.code not in (0, None):
            print("ERROR: %s" % e, file=sys.stderr)
            return 1
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
