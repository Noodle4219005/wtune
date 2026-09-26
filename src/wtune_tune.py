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
import copy
import csv
import dataclasses
import datetime
import glob
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
    for c in configs:
        exempt = c.get("config_source") == "anchor" or c.get("id") == "ANCHOR"
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

    def elapsed(self):
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


def wait_for_jobs(ctx, job_ids, stage, note=""):
    """Polls sacct until every id in `job_ids` is out of PENDING/RUNNING, or the budget deadline
    passes (in which case only THOSE ids -- never any other job -- are scancel'd). Prints one
    progress line per poll. Returns the last sacct states dict seen ({job_id: (State, Elapsed)},
    {} when `job_ids` was empty) -- callers use the terminal Elapsed to measure real per-run
    startup cost (measure_startup_s)."""
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        return {}
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


# ---------------------------------------------------------------------------
# submitting one CONFIG_FILE job (thin wrapper around scc_sweep's own builders)
# ---------------------------------------------------------------------------
def submit_configs_job(ctx, configs, reps, name_suffix, s_est, time_est=None):
    if time_est is None:
        time_est = estimate_run_time(len(configs), reps, ctx.steps, s_est=s_est,
                                      startup_s=ctx.startup_s)
    job_out_dir = os.path.join(ctx.out_dir, name_suffix)
    os.makedirs(job_out_dir, exist_ok=True)
    configs_path = os.path.join(job_out_dir, "configs.json")
    sweep.write_configs_json(configs_path, configs)
    cmd_str = sweep.build_sbatch_command(
        ctx.case, ctx.nodes, ctx.cores_per_node, time_est["time_limit"], time_est["timeout_s"],
        reps, ctx.steps, os.path.abspath(configs_path), job_out_dir,
        env_source=ctx.env_source, sbatch_script=ctx.sbatch_script)
    job_id = sweep.submit_job(cmd_str, sbatch_runner=ctx.sbatch_runner)
    ctx.owned_jobs.append(job_id)
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
    own sacct snapshot) are what measure_startup_s needs afterward."""
    chunks = chunk_round_robin(configs, slots)
    job_ids, configs_by_cid, job_configs = [], {}, {}
    for i, chunk in enumerate(chunks):
        time_est = estimate_run_time(len(chunk), 1, ctx.steps, s_est=s_est, startup_s=ctx.startup_s)
        if would_exceed_budget(ctx.elapsed(), time_est["total_run_s"], ctx.budget_s):
            print("%s slot %d: SKIP -- estimated end would exceed budget" % (name_prefix, i))
            continue
        job_id, _path, _te = submit_configs_job(ctx, chunk, reps=1,
                                                  name_suffix="%s-s%d" % (name_prefix, i),
                                                  s_est=s_est, time_est=time_est)
        job_ids.append(job_id)
        job_configs[job_id] = chunk
        for c in chunk:
            configs_by_cid[c["id"]] = c
        print("%s slot %d: submitted job %s (%d configs)" % (name_prefix, i, job_id, len(chunk)))

    states = wait_for_jobs(ctx, job_ids, name_prefix)
    rows = ctx.ledger_reader(ctx.case, job_ids)
    return job_ids, configs_by_cid, rows, job_configs, states


def run_round0(ctx):
    s_est, s_source, n_rows = sweep.estimate_s_per_step(ctx.case, ctx.nodes, ctx.ledger_dir,
                                                          ctx.hardware_id)
    per_run_s = ctx.startup_s + ctx.steps * s_est
    slots = compute_slots(ctx.max_nodes, ctx.nodes)
    round0_budget_s = min(ctx.budget_s - ctx.elapsed(), ctx.budget_s * ROUND0_BUDGET_FRACTION)
    configs_per_job = max(1, int(round0_budget_s // per_run_s))
    total_k = slots * configs_per_job

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
        measured_startup, n_runs = measure_startup_s(job_configs, states, wave0_results, ctx.steps)
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
        if elapsed >= ADAPTIVE_STOP_FRACTION * ctx.budget_s:
            print("adaptive rounds: stopping before round %d -- elapsed already >= %.0f%% of budget"
                  % (round_idx, ADAPTIVE_STOP_FRACTION * 100))
            return

        remaining_s = ctx.budget_s - elapsed
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

        time_est = estimate_run_time(len(configs_out), 1, ctx.steps, s_est=s_est,
                                      startup_s=ctx.startup_s)
        projected_end = elapsed + time_est["total_run_s"]        # RAW/EXPECTED -- never job_hours
        if projected_end > ADAPTIVE_STOP_FRACTION * ctx.budget_s:
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

        wait_for_jobs(ctx, [job_id], "round%d" % round_idx)
        rows = ctx.ledger_reader(ctx.case, [job_id])
        new_results = aggregate_by_cid(rows, all_configs_by_cid)
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
    if campaign_dir:
        export += ",CAMPAIGN=%s" % campaign_dir
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive --exclude=head "
              "--time=%s --job-name=c3-traj-%s-n%d --output=%s/c3traj-%%j.out --export=%s %s"
              % (nodes, cores_per_node, time_limit, case, nodes, out_dir, export,
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
    measurements = []
    for i, p in enumerate(points):
        rep = i + 1
        rundir = os.path.join(_RANKING_RUNDIR_ROOT, "%s_ANCHOR_r%d_%s" % (ctx.case, rep, job_id))
        steps_path = os.path.join(rundir, "steps.txt")
        if not os.path.exists(steps_path) or os.path.getsize(steps_path) == 0:
            raise RuntimeError("C3 trajectory job %s: %s missing/empty for point %s (rep %d)"
                                % (job_id, steps_path, p.at.isoformat(), rep))
        measurements.append(wtune_probe_plan.measurement_from_steps_file(
            steps_path, stepra, p.at, p.reason, p.source))
    return measurements, job_id


def run_rebuild_points(ctx, points, best_config, nodes, case_dir, out_dir, run_start,
                       spinup_min=60, grib_dir=None, date_shift_to=None):
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
    config = dict(ranks=best_config["ranks"], omp=best_config["omp"], nx=best_config["nx"],
                  ny=best_config["ny"], nodes=nodes)
    plan = wtune_probe_plan.plan_rebuild_segments(
        case_dir, points, config, out_dir, spinup_min=spinup_min, date_shift_to=date_shift_to,
        grib_dir=grib_dir)
    cache_dir = os.path.join(out_dir, "grib")
    total_bytes, forcing = coldstart.resolve_and_fetch_forcing(plan, out_dir, cache_dir,
                                                                grib_dir=grib_dir)
    print("C3 rebuild: fetched %.1f MB forcing for %d point(s)" % (total_bytes / 1e6, len(points)))
    sbatch_path = coldstart.stage_plan(out_dir, site="scc")
    job_id = sweep.submit_job("sbatch %s" % sbatch_path, sbatch_runner=ctx.sbatch_runner)
    ctx.owned_jobs.append(job_id)
    print("C3 rebuild job submitted: %s (%s)" % (job_id, sbatch_path))
    states = wait_for_jobs(ctx, [job_id], "c3-rebuild")
    state = states.get(job_id, ("UNKNOWN", ""))[0]
    if state != "COMPLETED":
        raise RuntimeError("C3 rebuild job %s ended in state %s (expected COMPLETED) -- "
                            "infrastructure failure, stop and diagnose read-only" % (job_id, state))
    plan["_plan_dir"] = out_dir
    sample_points, _geom_src = wtune_predict.read_wrfinput_sample_points(case_dir)
    calibration = wtune_predict.load_calibration(str(wtune_predict.DEFAULT_CALIBRATION))
    windows_out, _pooled, anchors = coldstart.score_windows_from_rsl(plan, sample_points,
                                                                      calibration)
    measurements = []
    for w, (anchor_s, _s_k) in zip(windows_out, anchors):
        at = run_start + datetime.timedelta(seconds=anchor_s)
        measurements.append(wtune_probe_plan.PointMeasurement(
            at=at, reason=w["reason"], source="rebuild", s_normal=w["s_k"], n_rad=w["n_rad"],
            r=w["r_local"]))
    return measurements, job_id


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
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# top-level run
# ---------------------------------------------------------------------------
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

    case_json, namelist_path, profile = resolve_case(case, args.cores_per_node)
    is_fire = wtune_config.is_fire_case(case_json, namelist_path)

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
    if nodes_raw in ("auto", None):
        nodes, nodes_note, nodes_warning = choose_nodes(profile, max_nodes, nodes_policy)
        print("WARNING: %s" % nodes_warning)
    else:
        nodes = int(nodes_raw)
        nodes_note = "user-specified (cap=%d)" % max_nodes
        if nodes > max_nodes:
            raise SystemExit("--nodes %d exceeds --max-nodes %d" % (nodes, max_nodes))

    out_dir = sweep._expand(args.out) if args.out else sweep._expand(
        "$HOME/wtune-runs/tune/%s" % case)
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
              confirm_reps=confirm_reps, prediction_cfg=cfg["prediction"],
              probe_cfg=cfg["probe"], ranking_start_note=ranking_start_note)
    ctx.start_time = clock.now()

    print("case=%s nodes=%d (%s) budget=%.1f min max_nodes=%d out=%s"
          % (case, nodes, nodes_note, minutes, max_nodes, out_dir))

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

    # final safety net: nothing owned by this run should still be in flight once we report.
    still_up = [j for j in ctx.owned_jobs if j not in ctx.cancelled_jobs]
    if still_up:
        states = sweep.sacct_states(still_up, sacct_runner=ctx.sacct_runner,
                                     sacct_cmd=ctx.sacct_cmd)
        in_flight = [j for j in still_up
                     if states.get(j, ("UNKNOWN", ""))[0] in sweep.IN_FLIGHT_STATES]
        if in_flight and ctx.elapsed() >= ctx.budget_s:
            cancel_jobs(ctx, in_flight)

    report = build_report(ctx, nodes, nodes_note, all_results, confirm_results, anchor_cid)
    text = render_report_text(report)
    print("")
    print(text)
    with open(os.path.join(out_dir, "tune_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return report


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
    p.add_argument("--nodes", default=None, help="default: 'auto' (or --config's budget.nodes)")
    p.add_argument("--nodes-policy", choices=sorted(NODES_POLICIES), default=None,
                    dest="nodes_policy",
                    help="how --nodes auto picks a node count from the (uncalibrated) profile: "
                         "min=smallest legal, cap=largest legal within --max-nodes, "
                         "knee=stop doubling once marginal speedup drops below 1.5x "
                         "(default: 'knee', or --config's budget.nodes_policy)")
    p.add_argument("--max-nodes", type=int, default=None, dest="max_nodes",
                    help="default: %d (or --config's budget.max_nodes)" % DEFAULT_MAX_NODES)
    p.add_argument("--out", default=None)
    p.add_argument("--steps", type=int, default=None,
                    help="default: %d (or --config's probe.steps)" % DEFAULT_STEPS)
    p.add_argument("--confirm-reps", type=int, default=None, dest="confirm_reps",
                    help="default: %d (or --config's probe.confirm_reps)" % CONFIRM_REPS)
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
