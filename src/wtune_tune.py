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
    total_cores = ctx.nodes * ctx.cores_per_node
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

    def wall_elapsed(self):
        return self.clock.now() - self.start_time

    def elapsed(self):
        if self.budget_mode == "running":
            return self.clock.now() - self.start_time - self.queue_only_s - self.uncharged_s
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
                       % (_fmt_hms(wall), _fmt_hms(ctx.queue_only_s)))
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
        if not running:
            ctx.queue_only_s += ctx.clock.now() - t0


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


def run_round0(ctx):
    s_est, s_source, n_rows = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                          ctx.hardware_id)
    if _stage1_active(ctx) and ctx.s_prior and n_rows == 0:
        # predicted mode, no ledger history: the cell-step model beats the 10 s/step fallback,
        # which would also inflate the stage-2 reserve until nothing in stage 1 fits
        s_est, s_source = ctx.s_prior, "cell-step model"
    per_run_s = ctx.startup_s + ctx.steps * s_est
    slots = compute_slots(ctx.max_nodes, ctx.nodes)
    round0_budget_s = min(ctx.budget_s - ctx.elapsed() - stage1_reserve_s(ctx),
                          ctx.budget_s * ROUND0_BUDGET_FRACTION)
    configs_per_job = max(1, int(round0_budget_s // per_run_s))
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
                                 ctx.hardware_id, ctx.cores_per_node)
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
            topup_configs_per_job = max(1, int(topup_budget_s // topup_per_run_s))
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
            total_cores=ctx.nodes * ctx.cores_per_node, nodes=ctx.nodes,
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
        batch_size = max(1, int(min(remaining_s, ctx.budget_s * ROUND0_BUDGET_FRACTION) // per_run_s))

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
            case_dir=ctx.case_json.get("prep_dir"))
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
    points, notes = wtune_probe_plan.plan_points(
        nl, run_start, run_end, dt, stepra, spinup_s, ctx.probe_cfg, case_dir=case_dir)

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
        nodes=nodes)
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
                ranks=best_config.get("ranks"), nodes=nodes)
            lines = ["=== full-run prediction (MULTI-POINT, wtune_probe_plan.combine, "
                     "%d measured point(s)) ===" % len(probe_measurements),
                     "compute_total: %s   init: %.1fs   TOTAL: %s (range %s - %s, "
                     "+-%.1f%% drift)"
                     % (_fmt_hms(result["compute_total_s"]), result["init_s"],
                        _fmt_hms(result["total_s"]), _fmt_hms(result["range_low_s"]),
                        _fmt_hms(result["range_high_s"]), result["drift_pct"]),
                     "r_rad: night=%.3f day=%.3f source=%s"
                     % (result["r_night"], result["r_day"], result["r_source"])]
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


def build_report(ctx, nodes, nodes_note, all_results, confirm_results, anchor_cid):
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
            confirmed=bool(confirm_results and r["cid"] in confirm_results),
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
        budget_minutes=ctx.budget_s / 60.0, elapsed_minutes=ctx.elapsed() / 60.0,
        job_ids=list(ctx.owned_jobs), cancelled_job_ids=list(ctx.cancelled_jobs),
        anchor_id=anchor_cid, ranking=table, best=best, param_table=param_table,
        full_run_prediction=build_full_run_prediction(ctx, best_config, nodes, all_results,
                                                        confirm_results),
        ranking_start_note=ctx.ranking_start_note,
    )


def render_report_text(report):
    lines = []
    lines.append("=== wtune_tune: %s ===" % report["case"])
    lines.append("nodes=%d (%s)" % (report["nodes"], report["nodes_note"]))
    lines.append("budget=%.1f min  elapsed=%.1f min  jobs=%s"
                  % (report["budget_minutes"], report["elapsed_minutes"],
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
                       require_restart=True):
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
    if not a or not e:
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


def _measure_run(ctx, rp, cid, job_id, dt):
    """PointMeasurement for one stage-2 run (RUNDIR = ledger_dir/<case_id>_<cid>_r1_<job>):
    first_global_step_from_rsl relative to the rebuilt start T, measurement_from_steps_file with
    the case stepra; `.at` = the point's measured-window midpoint on the CASE timeline (M +
    steps*dt/2). Returns (PointMeasurement, g0) or raises."""
    rundir = os.path.join(ctx.ledger_dir, "%s_%s_r1_%s" % (rp["case_id"], cid, job_id))
    steps_path = os.path.join(rundir, "steps.txt")
    if not os.path.exists(steps_path) or os.path.getsize(steps_path) == 0:
        raise RuntimeError("%s missing/empty" % steps_path)
    g0 = wtune_probe_plan.first_global_step_from_rsl(os.path.join(rundir, "rsl.error.0000"),
                                                     rp["T"], dt)
    at = rp["point"].at + datetime.timedelta(seconds=rp["steps"] * dt / 2.0)
    pm = wtune_probe_plan.measurement_from_steps_file(
        steps_path, rp["stepra"], at, rp["point"].reason, "rebuild", first_global_step=g0)
    return pm, g0


def measure_candidates_at_points(ctx, configs, rebuilt_points, anchor, s_by_cid=None,
                                 restart_io_form=None, name_prefix="stage2"):
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
    wrapped = wrap_with_anchors(configs, anchor)
    nl = None
    try:
        nl = wtune_predict.parse_namelist(ctx.namelist_path)
    except Exception:
        pass
    jobs = []
    for rp in rebuilt_points:
        dt = float(rp["dt"])
        cand_to, spin_to, time_limit, expected = ranking_job_timeouts(
            ctx, wrapped, rp["steps"], rp["spinup_min"], dt, s_by_cid=s_by_cid, nl=nl,
            stepra=rp.get("stepra"))
        job_out = os.path.join(ctx.out_dir, name_prefix, "pt_%d" % rp["index"])
        os.makedirs(job_out, exist_ok=True)
        configs_path = os.path.join(job_out, "configs.json")
        sweep.write_configs_json(configs_path, wrapped)
        ranking = ranking_context(rp["case_id"], rp["cases_json"], rp["spinup_min"], rp["steps"],
                                  anchor, restart_io_form=restart_io_form,
                                  spinup_timeout_s=spin_to)
        cmd = sweep.build_sbatch_command(
            rp["case_id"], ctx.nodes, ctx.cores_per_node, time_limit, cand_to, 1, rp["steps"],
            os.path.abspath(configs_path), job_out, env_source=ctx.env_source,
            sbatch_script=ctx.sbatch_script, campaign_dir=sweep.campaign_dir_for(ctx.sbatch_script),
            extra_export=ranking_export(ranking, {"timeout_s": cand_to}))
        jid = sweep.submit_job(cmd, sbatch_runner=ctx.sbatch_runner)
        ctx.owned_jobs.append(jid)
        print("%s point %d (%s, case %s): job %s, %d runs, TIMEOUT_S=%d SPINUP_TIMEOUT_S=%d "
              "--time=%s (expected %s)" % (name_prefix, rp["index"], rp["point"].at.isoformat(),
                                           rp["case_id"], jid, len(wrapped), cand_to, spin_to,
                                           time_limit, _fmt_hms(expected)))
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
        for c in wrapped:
            try:
                pm, g0 = _measure_run(ctx, rp, c["id"], jid, dt)
            except Exception as e:
                extra.append("%s: no measurement (%s: %s)" % (c["id"], type(e).__name__, e))
                continue
            if g0 != expect_g0:
                extra.append("%s: first global step %d, expected %g -- run did not resume at "
                             "the shared restart" % (c["id"], g0, expect_g0))
            s_by[c["id"]] = pm.s_normal
            if c["id"] != ANCHOR_END_ID:
                meas_by_cid.setdefault(c["id"], []).append(pm)
        rec = evaluate_job_gates(jid, rows, s_by, [c["id"] for c in wrapped])
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


def fit_stage2(ctx, top, anchor, rebuilt, s_by_cid, nl=None, left_s=_BUDGET_LEFT):
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
        wrapped = wrap_with_anchors(top[:k], anchor)
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
        queue_only_minutes=ctx.queue_only_s / 60.0,
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


def _sweep_anchor(ctx, nodes):
    """(anchor config | None, reason) -- the same ANCHOR round 0 builds at `nodes`
    (sweep.build_configs), passed through the shared legality filter (_apply_search_filter: OMP
    cap, WRF-Chem j-patch guard, --config exclude rules)."""
    cn = dataclasses.replace(ctx, nodes=nodes)
    built = sweep.build_configs(ctx.case, ctx.case_json, nodes, 0, ctx.seed, ctx.hardware_id,
                                 ctx.cores_per_node)
    anchors = [c for c in built["configs"] if c.get("id") == "ANCHOR"]
    if not anchors:
        return None, built["skip_reason"] or "no legal ANCHOR decomposition"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kept = _apply_search_filter(cn, anchors)
    if not kept:
        why = "; ".join(l.strip() for l in buf.getvalue().splitlines() if l.strip())
        return None, "ANCHOR rejected by the legality filter (%s)" % (why or "excluded")
    return kept[0], None


def run_node_sweep(ctx, legal_nodes, budget_frac=DEFAULT_SWEEP_BUDGET_FRAC):
    """Stage 'node sweep': one Slurm job per candidate N measuring the ANCHOR at REPS=2, submitted
    in ascending waves (wtune_nodes.sweep_waves); after each wave stop_early() decides whether
    larger N are still worth measuring. Returns dict(measurements={N: [s/step]}, anchors={N:
    config}, results={N: aggregate entry}, skipped=[{nodes, reason}], jobs={N: job_id}). The jobs
    are ordinary strict_campaign jobs, so their rows land in the same ledger_<case>_<job>.csv
    files (n_nodes column) round-0 rows do."""
    cands = wtune_nodes.candidate_node_counts(legal_nodes, ctx.max_nodes)
    skipped, anchors, meas, results, jobs = [], {}, {}, {}, {}
    todo = []
    for n in cands:
        anchor, why = _sweep_anchor(ctx, n)
        if anchor is None:
            skipped.append(dict(nodes=n, reason=why))
            print("node sweep: N=%d skipped -- %s" % (n, why))
        else:
            anchors[n] = anchor
            todo.append(n)
    s_prev = None
    stopped = None
    cap_s = budget_frac * ctx.budget_s
    sweep_t0 = ctx.elapsed()
    for wave in wtune_nodes.sweep_waves(todo, ctx.max_nodes):
        if not stopped and ctx.elapsed() - sweep_t0 >= cap_s:
            stopped = ("sweep budget cap: sweep used %s >= %.0f%% of the budget (%s)"
                       % (_fmt_hms(ctx.elapsed() - sweep_t0), budget_frac * 100,
                          _fmt_hms(cap_s)))
            print("node sweep: %s; larger N not submitted" % stopped)
        if stopped:
            for n in wave:
                skipped.append(dict(nodes=n, reason=stopped))
            continue
        wave_jobs = {}
        for n in wave:
            cn = dataclasses.replace(ctx, nodes=n)
            if s_prev is not None:
                s_est, src = s_prev, "last measured median"
            else:
                s_est, src, _k = sweep.estimate_s_per_step(ctx.case, n, ctx.ledger_dir,
                                                            ctx.hardware_id)
            est = _job_time_est(cn, 1, NODE_SWEEP_REPS, s_est)
            if would_exceed_budget(ctx.elapsed(), est["total_run_s"], ctx.budget_s):
                why = ("estimated end would exceed the budget (job ~%s at %.3fs/step[%s], "
                       "elapsed %s / budget %s)" % (_fmt_hms(est["total_run_s"]), s_est, src,
                                                    _fmt_hms(ctx.elapsed()),
                                                    _fmt_hms(ctx.budget_s)))
                skipped.append(dict(nodes=n, reason=why))
                print("node sweep: N=%d skipped -- %s" % (n, why))
                continue
            job_id, _p, _te = submit_configs_job(cn, [anchors[n]], reps=NODE_SWEEP_REPS,
                                                   name_suffix="nodes-n%d" % n, s_est=s_est,
                                                   time_est=est)
            wave_jobs[n] = job_id
            jobs[n] = job_id
            print("node sweep: N=%d submitted job %s (ANCHOR, REPS=%d, ~%s)"
                  % (n, job_id, NODE_SWEEP_REPS, _fmt_hms(est["total_run_s"])))
        if not wave_jobs:
            continue
        states = wait_for_jobs(ctx, list(wave_jobs.values()), "node-sweep")
        rows = ctx.ledger_reader(ctx.case, list(wave_jobs.values()))
        for n, j in wave_jobs.items():
            mine = [r for r in rows if (parse_run_id(r.get("run_id")) or {}).get("job") == j]
            agg = aggregate_by_cid(mine, {"ANCHOR": anchors[n]}).get("ANCHOR")
            if agg:
                meas[n] = agg["s_values"]
                results[n] = agg
            else:
                skipped.append(dict(nodes=n, reason="no successful measurement (job %s, state %s)"
                                    % (j, states.get(j, ("UNKNOWN", ""))[0])))
        srows = wtune_nodes.summarize(meas)
        if srows:
            s_prev = srows[-1]["median"]
        if wtune_nodes.stop_early(srows):
            a, b = srows[-2], srows[-1]
            stopped = ("early stop: N=%d (%.4f s/step) did not beat N=%d (%.4f s/step) by >%.0f%%"
                       % (b["nodes"], b["median"], a["nodes"], a["median"],
                          wtune_nodes.NOISE_FRAC * 100))
            print("node sweep: %s; larger N not submitted" % stopped)
    return dict(measurements=meas, anchors=anchors, results=results, skipped=skipped, jobs=jobs)


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
                    total_steps, nodes_policy="knee", sweep_budget_frac=DEFAULT_SWEEP_BUDGET_FRAC):
    """Node sweep -> choose N -> round0/adaptive/confirm at each finalist N (remaining budget
    split evenly over the finalists still to run) -> final (N, config) by the same objective over
    the finalists' best tuned medians. Falls back to the proxy path when nothing was measured."""
    legal = [r["nodes"] for r in _legal_rows(profile, ctx.max_nodes)]
    if not legal:
        print("node sweep: profile lists no legal node count; sweeping N=1 only")
        legal = [1]
    sw = run_node_sweep(ctx, legal, budget_frac=sweep_budget_frac)
    overhead_s = ctx.startup_s
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

    kw = dict(deadline_s=deadline_s, total_steps=total_steps, overhead_s=overhead_s,
              min_efficiency=min_efficiency)
    choices = node_choices(rows, **kw)
    chosen, note = wtune_nodes.choose(rows, objective, **kw)
    finals = wtune_nodes.finalists(rows, objective, k=k_finalists, **kw)
    tab = wtune_nodes.table(rows, total_steps=total_steps, overhead_s=overhead_s)
    print("node sweep: chose N=%d -- %s; finalists %s" % (chosen, note, finals))
    for r in tab:
        print("node sweep: N=%-2d median=%.4f s/step n=%d spread=%.1f%% speedup=%.2f eff=%.2f"
              % (r["nodes"], r["median"], r["n"], r["spread"] * 100, r["speedup"],
                 r["efficiency"]))

    outcomes = {}                  # N -> dict(report, best_median)
    for i, n in enumerate(finals):
        remaining = ctx.budget_s - ctx.elapsed()
        if remaining <= 0:
            print("node search: no budget left for finalist N=%d" % n)
            continue
        share = remaining / (len(finals) - i)
        fctx = dataclasses.replace(ctx, nodes=n, budget_s=ctx.elapsed() + share,
                                    out_dir=os.path.join(ctx.out_dir, "n%d" % n))
        os.makedirs(fctx.out_dir, exist_ok=True)
        print("node search: finalist N=%d, budget share %s" % (n, _fmt_hms(share)))
        all_results, confirm_results, anchor_cid = run_tuning_at(fctx)
        if not all_results and n in sw["results"]:
            print("node search: nothing measured at N=%d beyond the sweep; reporting the sweep "
                  "ANCHOR" % n)
            all_results, anchor_cid = {"ANCHOR": sw["results"][n]}, "ANCHOR"
        if not all_results:
            continue
        rep = build_report(fctx, n, "node search finalist", all_results, confirm_results,
                           anchor_cid)
        if rep["best"]:
            outcomes[n] = dict(report=rep, best_median=rep["best"]["median_s_per_step"])
    if not outcomes:
        # budget gone before any finalist produced a result: report the sweep ANCHOR at `chosen`
        fctx = dataclasses.replace(ctx, nodes=chosen)
        res = sw["results"].get(chosen)
        rep = build_report(fctx, chosen, note, {"ANCHOR": res} if res else {}, None,
                           "ANCHOR" if res else None)
        outcomes[chosen] = dict(report=rep, best_median=res["median"] if res else None)
        final_n, final_note = chosen, note + "; budget exhausted after the sweep -- ANCHOR config"
    else:
        by_sweep = {r["nodes"]: r for r in rows}
        frows = wtune_nodes.summarize({n: [o["best_median"]] for n, o in outcomes.items()})
        for r in frows:                     # scaling figures stay the sweep's (anchor-based)
            r["speedup"], r["efficiency"] = (by_sweep[r["nodes"]]["speedup"],
                                              by_sweep[r["nodes"]]["efficiency"])
        final_n, final_note = wtune_nodes.choose(frows, objective, **kw)
        final_note = "final over tuned finalists %s: %s" % (sorted(outcomes), final_note)
    report = outcomes[final_n]["report"]
    report["nodes"], report["nodes_note"] = final_n, final_note
    report["budget_minutes"] = ctx.budget_s / 60.0
    report["elapsed_minutes"] = ctx.elapsed() / 60.0
    report["job_ids"] = list(ctx.owned_jobs)
    report["cancelled_job_ids"] = list(ctx.cancelled_jobs)
    report["node_selection"] = dict(
        mode="search", objective=objective, chosen=chosen, chosen_note=note, final=final_n,
        final_note=final_note, table=tab, choices=choices, skipped=sw["skipped"],
        finalists=finals,
        finalist_best={str(n): o["best_median"] for n, o in outcomes.items()},
        total_steps=total_steps, deadline_s=deadline_s, min_efficiency=min_efficiency,
        overhead_s=overhead_s, sweep_jobs=list(sw["jobs"].values()))
    return report


def render_node_selection_text(ns):
    lines = ["", "--- node selection (measured, ANCHOR config per node count) ---",
             "%5s %12s %5s %8s %8s %6s %10s %10s"
             % ("nodes", "median_s", "reps", "spread", "speedup", "eff", "wall_s", "node_hours")]
    for r in ns.get("table", []):
        lines.append("%5d %12.4f %5d %7.1f%% %7.2fx %6.2f %10s %10s"
                     % (r["nodes"], r["median"], r["n"], r["spread"] * 100, r["speedup"],
                        r["efficiency"],
                        "%.0f" % r["wall_s"] if r.get("wall_s") is not None else "-",
                        "%.2f" % r["node_hours"] if r.get("node_hours") is not None else "-"))
    for obj, c in (ns.get("choices") or {}).items():
        lines.append("choice %-5s%s N=%s  %s" % (obj, "*" if obj == ns.get("objective") else " ",
                                                 c["nodes"], c["note"]))
    lines.append("(* = objective used; finalists searched: %s; final N=%s)"
                 % (ns.get("finalists"), ns.get("final", ns.get("chosen"))))
    for s in ns.get("skipped", []):
        lines.append("skipped N=%d: %s" % (s["nodes"], s["reason"]))
    return "\n".join(lines)


def run(args, sbatch_runner=None, sacct_runner=None, scancel_runner=None, ledger_reader=None,
        proposer=None, clock=None):
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
    if minutes is None:
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
              seed=args.seed, kappa=args.kappa, budget_s=minutes * 60.0, max_nodes=max_nodes,
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
    ctx.start_time = clock.now()

    print("case=%s nodes=%s (%s) budget=%.1f min max_nodes=%d out=%s"
          % (case, "search" if node_search else nodes, nodes_note, minutes, max_nodes, out_dir))

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
        _final_cancel_safety_net(ctx)
        text = render_predicted_report_text(report)
        print("")
        print(text)
        with open(os.path.join(out_dir, "tune_report.json"), "w") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        return report

    if node_search:
        report = run_node_search(ctx, profile, objective=node_objective,
                                  deadline_s=(deadline_minutes * 60.0
                                              if deadline_minutes is not None else None),
                                  min_efficiency=min_efficiency, k_finalists=node_finalists,
                                  total_steps=total_steps, nodes_policy=nodes_policy,
                                  sweep_budget_frac=sweep_budget_frac)
        _final_cancel_safety_net(ctx)
        _emit_report(report, out_dir)
        return report

    all_results, confirm_results, anchor_cid = run_tuning_at(ctx, s_est)

    _final_cancel_safety_net(ctx)

    report = build_report(ctx, nodes, nodes_note, all_results, confirm_results, anchor_cid)
    _emit_report(report, out_dir)
    return report


def run_tuning_at(ctx, s_est=None):
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

    if job_ids0:
        run_adaptive_rounds(ctx, configs_by_cid, all_results, s_est)

    confirm_job_id, confirm_results = (None, None)
    if all_results:
        confirm_job_id, confirm_results = run_confirm(ctx, configs_by_cid, all_results, anchor_cid,
                                                        s_est)

    return all_results, confirm_results, anchor_cid


def _emit_report(report, out_dir):
    print("")
    print(render_report_text(report))
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
