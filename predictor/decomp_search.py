#!/usr/bin/env python3
"""predictor/decomp_search.py -- active decomposition search for ONE case (WTune D3): find a
near-best stock-WRF decomposition/tiling with few real WRF runs, instead of measuring the whole
`decomp_space.enumerate_space` pool or even the 16 fixed STRICT templates exhaustively.

Loop (propose -> run on the cluster -> ingest -> repeat):
  1. `propose` fits a surrogate on every available ledger row (other cases + this case's own
     already-measured configs so far) and picks a small, diverse batch of NOT-yet-measured
     candidates from `decomp_space.enumerate_space`'s pool by lower-confidence-bound acquisition.
  2. The human runs the printed sbatch command (this tool never submits anything itself).
  3. `ingest` records the resulting campaign ledger's measurements back into the search state.
  4. Repeat until the stop rule fires or --max-rounds is reached.

Surrogate + uncertainty (documented choice, per the brief's "choose, document, and test"): a small
BOOTSTRAP ENSEMBLE of `models.CatBoostPointModel` (default 8 members), each fit on a
with-replacement resample of the labeled training rows. mean/std across the ensemble give
(mu, sigma) per candidate. This was chosen over validate.py's leave-one-case-out conformal
machinery for two reasons: (a) conformal_intervals_for_new_rows needs >=2 case_base groups in the
WHOLE pool just to fit a single fold, which fails outright the first time this tool is ever run
against a brand-new case with no sibling cases yet measured; (b) it would refit len(case_base)
models per acquisition round (expensive, and conflates "which case" uncertainty with "which
config within this case" uncertainty, which is not the question being asked here). The tradeoff,
documented honestly: `CatBoostPointModel.fit` hardcodes `random_seed=models.SEED` (0), so ensemble
diversity here comes ONLY from which rows each bootstrap resample happens to contain, not from
varying CatBoost's own internal randomness -- still a legitimate (if narrower) bagging-variance
estimate, and cheap enough to refit every round. See predictor/tests/test_decomp_search.py's
TestEnsembleUncertainty for the synthetic check that this actually produces usable sigma.

Acquisition: lower-confidence-bound, mu - kappa*sigma (lower = predicted faster, matching
targets.py's y=log(s/baseline) convention throughout this repo) with a batch diversity cap of at
most `max_per_rank_omp` picks per (ranks, omp) allocation. Round 1 always force-includes the
case's own C01-equivalent STRICT reference candidate (so the very first batch is anchored to a
config whose real cost is knowable independently, e.g. from campaign/cases.json's own COST_GUESS
or a prior campaign run); later rounds force-include the best still-unmeasured candidate that
SHARES the current incumbent's (ranks, omp) allocation ("same-allocation comparison": explore
tile/dist/blocktime variants around the incumbent's own resource allocation, not just whichever
allocation the raw LCB ranking happens to prefer that round).

Stop rule: stop when even the most optimistic remaining candidate (the one with the lowest LCB,
i.e. the best case for further exploration) does not beat the incumbent's own measured y by more
than the noise floor (targets.NOISE, ~0.5%) -- "no candidate's predicted improvement exceeds the
noise floor with reasonable confidence" -- or after `max_rounds` rounds, whichever comes first.

CLI:
    decomp_search.py propose --case CASE --ledger L1 [L2 ...] --state STATE.json --batch 8 \\
        --out proposal.json
    decomp_search.py ingest --state STATE.json --ledger NEW_LEDGER.csv
    decomp_search.py simulate --ledger L1 L2 ... --out DIR

Reuses (does not reimplement): decomp_space.py (candidate pool, to_campaign_configs schema),
targets.py (NOISE, compute_case_baselines/add_y), models.py (CatBoostPointModel), ledger.py
(adapt_campaign_ledger, to parse a real measured ledger back into this tool's state), benchmark.py
(load_ledger), campaign/configs.py (CASES, resolve_case -- to find the C01-equivalent reference
and, read-only, to validate the proposal JSON schema in tests via load_generated_configs).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
for _p in (_HERE, _REPO_ROOT, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark                       # predictor/benchmark.py: load_ledger()
import decomp_space as ds              # predictor/decomp_space.py
import features as feat                # predictor/features.py
import ledger as ledgermod             # predictor/ledger.py: adapt_campaign_ledger
import models as modelmod              # predictor/models.py: CatBoostPointModel, NotEnoughData
try:
    from catboost import CatBoostError
except ImportError:
    class CatBoostError(Exception):
        pass
import targets                         # predictor/targets.py: NOISE, compute_case_baselines, add_y
import configs as campaign_configs     # campaign/configs.py: CASES, resolve_case, CASE_ORDER

DEFAULT_KAPPA = 1.5
DEFAULT_MAX_ROUNDS = 6
DEFAULT_MAX_PER_RANK_OMP = 2
DEFAULT_ENSEMBLE_SIZE = 8
DEFAULT_BATCH = 8
DEFAULT_PARTITION = "cf448"
DEFAULT_REPS = 2

NOISE = targets.NOISE


# ---------------------------------------------------------------------------
# Candidate identity
# ---------------------------------------------------------------------------
def candidate_key(c):
    """Identity of "the same decomposition/tiling configuration" for dedup/measured-tracking
    purposes: every axis enumerate_space varies, nothing derived. Deliberately its own function
    (not decomp_space._candidate_key, which omits `ranks` and `numtiles_x/y`/`tile_sz_x/y`/
    `tile_strategy` -- fine for that module's own stratified-sample dedup, since tile_shape alone
    already distinguishes a "2d" from a "tile_sz" candidate there, but NOT precise enough here:
    two different tile_sz_variants would both carry tile_shape=="tile_sz" and collide)."""
    return (c["ranks"], c["omp"], c["nx"], c["ny"], c["numtiles"], c["tile_shape"],
            c.get("numtiles_x", 0) or 0, c.get("numtiles_y", 0) or 0,
            c.get("tile_sz_x", 0) or 0, c.get("tile_sz_y", 0) or 0,
            c.get("tile_strategy", 0) or 0, c["dist"], c["blocktime"])


# ---------------------------------------------------------------------------
# Case fields + candidate -> ledger-row-shaped dict (mirrors recommend.py's _candidate_row)
# ---------------------------------------------------------------------------
def case_row_fields(case_json, io_form_history=14):
    """Case-level ledger fields (e_we/e_sn/e_vert/dx/dy/time_step/physics...) from a
    campaign/cases.json case dict (campaign_configs.CASES[case_id]). io_form_history defaults to
    14 (ADIOS2) since every STRICT campaign run forces it (campaign/configs.py _apply_adios2) --
    matches the real campaign ledgers' own column so decision_group_key groups consistently."""
    physics = case_json.get("physics", {})
    return dict(
        e_we=case_json["e_we"], e_sn=case_json["e_sn"], e_vert=case_json["e_vert"],
        dx=case_json.get("dx"), dy=case_json.get("dy"), time_step=case_json.get("dt"),
        mp_physics=physics.get("mp_physics"), cu_physics=physics.get("cu_physics"),
        bl_pbl_physics=physics.get("bl_pbl_physics"),
        sf_sfclay_physics=physics.get("sf_sfclay_physics"),
        sf_surface_physics=physics.get("sf_surface_physics"),
        ra_lw_physics=physics.get("ra_lw_physics"), ra_sw_physics=physics.get("ra_sw_physics"),
        radt=physics.get("radt"), cudt=physics.get("cudt"),
        history_interval=None, io_form_history=io_form_history,
    )


def candidate_row(case_name, case_fields, hardware_id, nodes, c, run_id, s_per_step_median=None):
    """A ledger-row-shaped dict for one decomp_space candidate `c`, suitable for
    features.case_features()/config_features() and targets.compute_case_baselines/add_y -- same
    shape/contract as recommend.py's `_candidate_row`, extended with the tile-geometry fields
    (numtiles_x/y, tile_sz_x/y, tile_strategy) config_features() also reads."""
    blocktime = c["blocktime"]
    kmp_blocktime = 0 if str(blocktime) == "0" else None
    row = dict(
        run_id=run_id, case_base=case_name, case_family=None, hardware_id=hardware_id,
        compliance_class="STRICT", binary_kind="pristine", source_dataset="decomp_search",
        n_nodes=nodes, n_ranks=c["ranks"], omp_threads=c["omp"],
        nproc_x=c["nx"], nproc_y=c["ny"], numtiles=c["numtiles"],
        numtiles_x=c.get("numtiles_x", 0) or 0, numtiles_y=c.get("numtiles_y", 0) or 0,
        tile_sz_x=c.get("tile_sz_x", 0) or 0, tile_sz_y=c.get("tile_sz_y", 0) or 0,
        tile_strategy=c.get("tile_strategy", 0) or 0,
        distribution=c["dist"], kmp_blocktime=kmp_blocktime,
        s_per_step_median=s_per_step_median,
    )
    row.update(case_fields)
    row["_cfg"] = c
    return row


def reference_predicate_for_case(case_json):
    """The C01 template's (ranks,omp,nx,ny,numtiles,dist,blocktime) predicate for this case, via
    campaign/configs.py's OWN resolve_case (the same shape-resolution logic the real campaign
    used) -- None if C01 has no legal shape for this case (resolve_case skipped it)."""
    kept, _skipped, _aliases = campaign_configs.resolve_case(case_json)
    c01 = next((c for c in kept if c["id"] == "C01"), None)
    if c01 is None:
        return None
    return dict(ranks=c01["ranks"], omp=c01["omp"], nx=c01["nx"], ny=c01["ny"],
                numtiles=c01["numtiles"], dist=c01["dist"], blocktime=c01["blocktime"])


# ---------------------------------------------------------------------------
# Surrogate: bootstrap ensemble of CatBoostPointModel (see module docstring)
# ---------------------------------------------------------------------------
def fit_ensemble(labeled_rows, n_models=DEFAULT_ENSEMBLE_SIZE, seed=0):
    """[fitted CatBoostPointModel, ...] (length <= n_models -- a bootstrap resample that fails to
    fit is skipped, not fatal). Two distinct failure modes are both tolerated per-member: (a)
    models.NotEnoughData (fewer than 2 labeled rows in the resample -- cannot happen here since
    every resample has exactly len(labeled_rows) rows, kept for symmetry with CatBoostPointModel's
    own contract) and (b) catboost.CatBoostError "All features are either constant or ignored" --
    a real, observed failure mode with a VERY small labeled_rows (e.g. 2-3 rows, exactly the
    situation a case's first search round is in): a with-replacement resample of that few rows can
    degenerate to every draw being the same row, making every feature literally constant. Raises
    models.NotEnoughData when NO resample fits at all."""
    if len(labeled_rows) < 2:
        raise modelmod.NotEnoughData(
            "decomp_search ensemble needs >=2 labeled rows, got %d" % len(labeled_rows))
    rng = random.Random(seed)
    fitted = []
    n = len(labeled_rows)
    for _ in range(n_models):
        boot = [labeled_rows[rng.randrange(n)] for _ in range(n)]
        m = modelmod.CatBoostPointModel()
        try:
            m.fit(boot)
        except (modelmod.NotEnoughData, CatBoostError):
            continue
        fitted.append(m)
    if not fitted:
        raise modelmod.NotEnoughData(
            "decomp_search ensemble: no bootstrap resample fit (labeled_rows=%d, tried %d "
            "resamples -- likely too few distinct rows to ever avoid a degenerate/constant-"
            "feature resample)" % (n, n_models))
    return fitted


def ensemble_predict(models_, rows):
    """(mu, sigma) numpy arrays aligned with `rows`. sigma is the sample std across ensemble
    members (0 when only one member fit)."""
    if not rows:
        return np.array([]), np.array([])
    preds = np.array([m.score(rows) for m in models_], dtype=float)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0, ddof=1) if preds.shape[0] > 1 else np.zeros(preds.shape[1])
    return mu, sigma


# ---------------------------------------------------------------------------
# Acquisition + batch diversity
# ---------------------------------------------------------------------------
DEFAULT_MAX_PATCH_ASPECT = 20.0     # skip strip shapes (e.g. 64x1 on 901x901, aspect 64: 15x slower, smoke 1078036)
DEFAULT_EXPLOIT_FRACTION = 0.5      # share of each batch picked by lowest predicted mean, not LCB
DEFAULT_MAX_EXPLORE_SLOWDOWN = 0.40  # LCB picks must have a predicted mean at most log(1.5) slower


def _patch_aspect(c):
    px, py = c.get("patch_x") or 0, c.get("patch_y") or 0
    if not px or not py:
        return 1.0
    return max(px, py) / float(min(px, py))


def _shape_key(c):
    """Candidates differing only in distribution/KMP_BLOCKTIME share a shape; one per shape per
    batch so a round is not spent on blocktime-only duplicates (round-1 proposals 2026-09-15)."""
    return (c["ranks"], c["omp"], c.get("nx"), c.get("ny"), c.get("numtiles"),
            c.get("numtiles_x"), c.get("numtiles_y"), c.get("tile_sz_x"), c.get("tile_sz_y"))


def select_batch(scored, batch_size, max_per_rank_omp, forced=None,
                 exploit_fraction=DEFAULT_EXPLOIT_FRACTION,
                 max_patch_aspect=DEFAULT_MAX_PATCH_ASPECT,
                 max_explore_slowdown=DEFAULT_MAX_EXPLORE_SLOWDOWN):
    """`scored`: [(acq, candidate_dict, mu, sigma), ...], already excludes measured candidates.
    Returns up to `batch_size` candidate dicts: `forced` first, then about `exploit_fraction` of the
    batch by lowest predicted mean (exploitation), then the rest by lowest LCB (exploration) among
    candidates whose predicted mean is not hopeless (mu <= max_explore_slowdown). Strip shapes with
    patch aspect > `max_patch_aspect` are never proposed (pure LCB picked 64x1 / 56x1 / 32x1 strips
    because their ensemble sigma is huge). Caps: `max_per_rank_omp` per (ranks, omp) and one
    candidate per shape (ignoring distribution / KMP_BLOCKTIME)."""
    pool = [t for t in scored if _patch_aspect(t[1]) <= max_patch_aspect]
    by_mean = sorted(pool, key=lambda t: t[2])
    by_acq = sorted((t for t in pool if t[2] <= max_explore_slowdown), key=lambda t: t[0])
    chosen, chosen_keys, chosen_shapes = [], set(), set()
    per_group = defaultdict(int)

    def _try_add(c):
        k = candidate_key(c)
        s = _shape_key(c)
        if k in chosen_keys or s in chosen_shapes:
            return False
        grp = (c["ranks"], c["omp"])
        if per_group[grp] >= max_per_rank_omp:
            return False
        chosen.append(c)
        chosen_keys.add(k)
        chosen_shapes.add(s)
        per_group[grp] += 1
        return True

    for c in (forced or []):
        if len(chosen) >= batch_size:
            break
        _try_add(c)
    n_exploit_target = len(chosen) + int(round(exploit_fraction * max(batch_size - len(chosen), 0)))
    for _acq, c, _mu, _sigma in by_mean:
        if len(chosen) >= n_exploit_target:
            break
        _try_add(c)
    for _acq, c, _mu, _sigma in by_acq:
        if len(chosen) >= batch_size:
            break
        _try_add(c)
    for _acq, c, _mu, _sigma in by_mean:   # fill if the exploration pool ran dry
        if len(chosen) >= batch_size:
            break
        _try_add(c)
    return chosen


def incumbent_from_measured(case_measured_rows):
    """(cfg, y) of the measured row with the lowest y (fastest), or (None, None) when
    `case_measured_rows` carries no row with y set yet."""
    labeled = [r for r in case_measured_rows if r.get("y") is not None]
    if not labeled:
        return None, None
    best = min(labeled, key=lambda r: r["y"])
    return best["_cfg"], best["y"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def measured_candidate_keys(state):
    keys = set()
    for cid in state.get("measured", {}):
        cfg = state.get("configs", {}).get(cid)
        if cfg is not None:
            keys.add(candidate_key(cfg))
    return keys


def build_case_measured_rows(state, case_name, case_fields, hardware_id, nodes):
    """Ledger-row-shaped dicts (WITHOUT y attached -- see propose()/simulate(), which attach y
    differently) for every replicate recorded in state['measured'], one row per replicate so
    targets.aggregate_by_config's own mean-of-log-across-replicates convention applies unchanged."""
    rows = []
    for cid, reps in state.get("measured", {}).items():
        cfg = state["configs"][cid]
        for i, rep in enumerate(reps):
            rows.append(candidate_row(
                case_name, case_fields, hardware_id, nodes, cfg,
                run_id="decompsearch:%s:%s:%d" % (case_name, cid, i),
                s_per_step_median=rep.get("s_per_step_median")))
    return rows


def load_state(path):
    if not path or not os.path.exists(path):
        return dict(configs={}, measured={}, rounds=[])
    with open(path) as f:
        return json.load(f)


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Core loop (pure function: does not touch state, files, or the ledger pool directly -- callers
# assemble `candidates`/`labeled_rows`/`incumbent_*` and apply the result to `state` themselves;
# this is what makes both `propose()` (real ledgers) and `simulate()` (offline ground-truth pool)
# share one implementation).
# ---------------------------------------------------------------------------
def propose_core(candidates, labeled_rows, case_name, case_fields, hardware_id, nodes, state,
                  batch_size, incumbent_cfg, incumbent_y, kappa=DEFAULT_KAPPA,
                  max_per_rank_omp=DEFAULT_MAX_PER_RANK_OMP, ensemble_size=DEFAULT_ENSEMBLE_SIZE,
                  ensemble_seed=0, max_rounds=DEFAULT_MAX_ROUNDS, reference_predicate=None,
                  noise=NOISE):
    """Returns a dict: stopped (bool), reason (str|None), round (int), chosen (list of candidate
    dicts, empty when stopped), and (when not stopped) n_training_rows/ensemble_size actually used.
    Never mutates `state` or `candidates`/`labeled_rows`."""
    round_idx = len(state.get("rounds", [])) + 1
    if round_idx > max_rounds:
        return dict(stopped=True, round=round_idx, chosen=[],
                    reason="max_rounds=%d reached" % max_rounds)

    measured_keys = measured_candidate_keys(state)
    unmeasured = [c for c in candidates if candidate_key(c) not in measured_keys]
    if not unmeasured:
        return dict(stopped=True, round=round_idx, chosen=[],
                    reason="every candidate in the pool has already been measured")

    try:
        models_ = fit_ensemble(labeled_rows, n_models=ensemble_size, seed=ensemble_seed)
    except modelmod.NotEnoughData as e:
        return dict(stopped=True, round=round_idx, chosen=[],
                    reason="surrogate cannot be fit yet: %s" % e)

    score_rows = [candidate_row(case_name, case_fields, hardware_id, nodes, c,
                                 run_id="candidate:%d" % i) for i, c in enumerate(unmeasured)]
    mu, sigma = ensemble_predict(models_, score_rows)
    acq = mu - kappa * sigma
    scored = list(zip(acq.tolist(), unmeasured, mu.tolist(), sigma.tolist()))

    if incumbent_y is not None:
        best_acq = min(t[0] for t in scored)
        if incumbent_y - best_acq <= noise:
            return dict(
                stopped=True, round=round_idx, chosen=[], incumbent=incumbent_cfg,
                incumbent_y=incumbent_y,
                reason=("best remaining candidate's LCB (%.5f) does not beat the incumbent's "
                        "measured y (%.5f) by more than the noise floor (%.5f, ~%.2f%%)"
                        % (best_acq, incumbent_y, noise, 100 * (np.exp(noise) - 1))))

    forced = []
    if round_idx == 1 and reference_predicate is not None:
        ref = next((c for c in unmeasured if ds._matches_predicate(c, reference_predicate)), None)
        if ref is not None:
            forced.append(ref)
    elif incumbent_cfg is not None:
        grp = (incumbent_cfg["ranks"], incumbent_cfg["omp"])
        same_group = sorted((t for t in scored if (t[1]["ranks"], t[1]["omp"]) == grp),
                             key=lambda t: t[0])
        if same_group:
            forced.append(same_group[0][1])

    chosen = select_batch(scored, batch_size, max_per_rank_omp, forced=forced)
    return dict(stopped=False, round=round_idx, chosen=chosen, incumbent=incumbent_cfg,
                incumbent_y=incumbent_y, n_training_rows=len(labeled_rows),
                ensemble_size=len(models_))


# ---------------------------------------------------------------------------
# propose: real CLI path, against decomp_space.enumerate_space's full pool
# ---------------------------------------------------------------------------
_RUN_ID_RE = re.compile(r"^(?P<case>[^_]+)_(?P<cid>.+)_r(?P<rep>\d+)_(?P<job>\d+)$")


def parse_run_id(run_id):
    m = _RUN_ID_RE.match(run_id or "")
    return m.groupdict() if m else None


def propose_for_case(case_name, case_json, case_fields, prior_rows, state, batch_size,
                      total_cores=448, nodes=4, hardware_id=ds.DEFAULT_HARDWARE,
                      kappa=DEFAULT_KAPPA, max_per_rank_omp=DEFAULT_MAX_PER_RANK_OMP,
                      ensemble_size=DEFAULT_ENSEMBLE_SIZE, ensemble_seed=0,
                      max_rounds=DEFAULT_MAX_ROUNDS):
    """Lower-level entry point: `case_json` (decomp_space.enumerate_space-shaped: e_we/e_sn/
    e_vert/max_dom[/e_we_list/e_sn_list]) and `case_fields` (ledger-row-shaped case fields, see
    case_row_fields) are supplied directly rather than looked up from campaign/cases.json -- lets
    a caller with its OWN namelist-derived case (recommend.py's --propose-batch, for a case that
    may not be registered in cases.json at all) reuse this loop. `propose()` below is the
    cases.json-backed convenience wrapper the CLI uses. Returns (result_dict_from_propose_core,
    configs_out) where configs_out is decomp_space.to_campaign_configs(chosen, ...) -- [] when
    stopped. Does not mutate `state`; the caller applies bookkeeping once it has written the
    proposal file."""
    candidates, _axis = ds.enumerate_space(case_json, total_cores, nodes, hardware_id)

    case_measured_rows = build_case_measured_rows(state, case_name, case_fields, hardware_id, nodes)
    combined = list(prior_rows) + case_measured_rows
    baselines = targets.compute_case_baselines(combined)
    targets.add_y(combined, baselines)
    labeled_rows = [r for r in combined if r.get("y") is not None]
    incumbent_cfg, incumbent_y = incumbent_from_measured(case_measured_rows)

    reference_predicate = reference_predicate_for_case(case_json)
    result = propose_core(
        candidates, labeled_rows, case_name, case_fields, hardware_id, nodes, state, batch_size,
        incumbent_cfg, incumbent_y, kappa=kappa, max_per_rank_omp=max_per_rank_omp,
        ensemble_size=ensemble_size, ensemble_seed=ensemble_seed, max_rounds=max_rounds,
        reference_predicate=reference_predicate)

    configs_out = []
    if not result["stopped"]:
        prefix = "R%02dD" % result["round"]
        configs_out = ds.to_campaign_configs(result["chosen"], prefix=prefix)
    return result, configs_out


def propose(case_name, ledger_paths, state, batch_size, total_cores=448, nodes=4,
            hardware_id=ds.DEFAULT_HARDWARE, kappa=DEFAULT_KAPPA,
            max_per_rank_omp=DEFAULT_MAX_PER_RANK_OMP, ensemble_size=DEFAULT_ENSEMBLE_SIZE,
            ensemble_seed=0, max_rounds=DEFAULT_MAX_ROUNDS):
    """CLI-facing convenience wrapper: `case_name` must be a campaign/cases.json case id. See
    propose_for_case for the reusable core."""
    case_json = campaign_configs.CASES.get(case_name)
    if case_json is None:
        raise ValueError("unknown case %r (not in campaign/cases.json)" % case_name)
    case_fields = case_row_fields(case_json)

    prior_rows = []
    for p in ledger_paths:
        prior_rows.extend(benchmark.load_ledger(p))
    prior_rows = [r for r in prior_rows if r.get("compliance_class") != "RESEARCH"]

    return propose_for_case(
        case_name, case_json, case_fields, prior_rows, state, batch_size,
        total_cores=total_cores, nodes=nodes, hardware_id=hardware_id, kappa=kappa,
        max_per_rank_omp=max_per_rank_omp, ensemble_size=ensemble_size,
        ensemble_seed=ensemble_seed, max_rounds=max_rounds)


def sbatch_command(case_name, config_file, partition=DEFAULT_PARTITION, reps=DEFAULT_REPS):
    return ("sbatch -p %s --export=ALL,CASE=%s,MODE=timing,REPS=%d,CONFIG_FILE=%s "
            "campaign/strict_campaign.sbatch" % (partition, case_name, reps, config_file))


# ---------------------------------------------------------------------------
# ingest: apply a real measured ledger's results back into state
# ---------------------------------------------------------------------------
def ingest(state, ledger_path):
    """Returns (n_ingested, n_skipped, skip_reasons). Mutates `state` in place (caller saves)."""
    rows = ledgermod.adapt_campaign_ledger(ledger_path)
    n_ingested, n_skipped = 0, 0
    skip_reasons = []
    for row in rows:
        parsed = parse_run_id(row.get("run_id"))
        if parsed is None:
            n_skipped += 1
            skip_reasons.append("run_id %r does not match this tool's format" % row.get("run_id"))
            continue
        cid = parsed["cid"]
        if cid not in state.get("configs", {}):
            n_skipped += 1
            skip_reasons.append("config id %r (run_id %r) is not one of this search's own "
                                 "proposed configs" % (cid, row.get("run_id")))
            continue
        s = row.get("s_per_step_median")
        if s is None:
            n_skipped += 1
            skip_reasons.append("run_id %r has no s_per_step_median (failed run)" % row.get("run_id"))
            continue
        state.setdefault("measured", {}).setdefault(cid, []).append(dict(
            run_id=row.get("run_id"), job_id=row.get("job_id"),
            rep_index=int(parsed["rep"]), s_per_step_median=s))
        n_ingested += 1
    return n_ingested, n_skipped, skip_reasons


# ---------------------------------------------------------------------------
# simulate: offline evaluation over the 6 measured STRICT campaign ledgers (16 templates each)
# ---------------------------------------------------------------------------
def _true_y_candidates_for_case(case_rows):
    """[(cfg, true_y), ...] -- one entry per distinct measured config (feat.config_key) in
    `case_rows`, cfg shaped like a decomp_space candidate dict (tile_shape='1d_y': every STRICT
    campaign template used the namelist-default tile resolution, no numtiles_x/y/tile_sz
    override), true_y = mean of that config's own replicate y values (already attached by the
    caller via targets.add_y against the FULL unhidden ledger -- the one and only ground truth)."""
    by_ck = defaultdict(list)
    for r in case_rows:
        if r.get("y") is None:
            continue
        by_ck[feat.config_key(r)].append(r)
    out = []
    for _ck, rs in by_ck.items():
        r0 = rs[0]
        kmp = r0.get("kmp_blocktime")
        cfg = dict(
            ranks=int(r0["n_ranks"]), omp=int(r0["omp_threads"]),
            nx=int(r0["nproc_x"]), ny=int(r0["nproc_y"]), numtiles=int(r0["numtiles"]),
            tile_shape="1d_y", numtiles_x=0, numtiles_y=0, tile_sz_x=0, tile_sz_y=0,
            tile_strategy=0, dist=r0["distribution"],
            blocktime=("0" if kmp == 0 else "default"),
        )
        true_y = sum(r["y"] for r in rs) / len(rs)
        out.append((cfg, true_y))
    return out


def _run_adaptive_loop(candidates_with_y, prior_rows, case_name, case_fields, hardware_id, nodes,
                        reference_predicate, kappa, max_per_rank_omp, ensemble_size, seed):
    """Sequential (batch=1) D3 loop over a KNOWN, closed candidate pool (offline evaluation only
    -- see simulate()). Returns [(n_revealed, regret), ...] for n=1..len(candidates_with_y) (or
    fewer, if the stop rule fires first -- the last regret value is then carried forward by the
    caller for any remaining n)."""
    candidates = [c for c, _y in candidates_with_y]
    true_y_by_key = {candidate_key(c): y for c, y in candidates_with_y}
    true_y_best = min(true_y_by_key.values())

    state = dict(configs={}, measured={}, rounds=[])
    id_by_key = {}
    for i, c in enumerate(candidates, start=1):
        cid = "T%02d" % i
        state["configs"][cid] = c
        id_by_key[candidate_key(c)] = cid

    case_measured_rows = []
    out = []
    for round_idx in range(1, len(candidates) + 1):
        combined = prior_rows + case_measured_rows
        incumbent_cfg, incumbent_y = incumbent_from_measured(case_measured_rows)
        result = propose_core(
            candidates, combined, case_name, case_fields, hardware_id, nodes, state,
            batch_size=1, incumbent_cfg=incumbent_cfg, incumbent_y=incumbent_y, kappa=kappa,
            max_per_rank_omp=max_per_rank_omp, ensemble_size=ensemble_size, ensemble_seed=seed,
            max_rounds=len(candidates), reference_predicate=reference_predicate)
        if result["stopped"] or not result["chosen"]:
            break
        for c in result["chosen"]:
            k = candidate_key(c)
            cid = id_by_key[k]
            state["measured"].setdefault(cid, []).append(dict(rep_index=0))
            row = candidate_row(case_name, case_fields, hardware_id, nodes, c,
                                 run_id="reveal:%s:%d" % (cid, round_idx))
            row["y"] = true_y_by_key[k]
            case_measured_rows.append(row)
        state["rounds"].append(dict(round=round_idx))
        incumbent_y_now = min(true_y_by_key[candidate_key(r["_cfg"])] for r in case_measured_rows)
        regret = float(np.exp(incumbent_y_now - true_y_best) - 1)
        out.append((len(case_measured_rows), regret))
    # carry the last regret forward for any rounds the stop rule cut short, so every method's
    # regret series covers the full n=1..len(candidates) range for a fair table.
    while out and len(out) < len(candidates):
        out.append((out[-1][0] + 1, out[-1][1]))
    return out


def _random_baseline(candidates_with_y, n_seeds, seed0=0):
    """Mean regret(n) over `n_seeds` uniformly random reveal orders, n=1..len(candidates_with_y)."""
    n = len(candidates_with_y)
    true_y_best = min(y for _c, y in candidates_with_y)
    totals = [0.0] * n
    for s in range(n_seeds):
        rng = random.Random(seed0 + s)
        order = list(candidates_with_y)
        rng.shuffle(order)
        best_so_far = None
        for i, (_c, y) in enumerate(order):
            best_so_far = y if best_so_far is None else min(best_so_far, y)
            totals[i] += float(np.exp(best_so_far - true_y_best) - 1)
    return [totals[i] / n_seeds for i in range(n)]


def _predictor_top_k_baseline(candidates_with_y, prior_rows, case_name, case_fields, hardware_id,
                               nodes, ensemble_size, seed):
    """Rank every candidate ONCE by a surrogate trained on `prior_rows` ONLY (no revealed data
    from this case at all -- a genuinely non-adaptive "measure the predicted top-k in one batch"
    baseline), then regret(n) = best TRUE y among the first n candidates in that fixed order."""
    candidates = [c for c, _y in candidates_with_y]
    true_y_by_key = {candidate_key(c): y for c, y in candidates_with_y}
    true_y_best = min(true_y_by_key.values())
    labeled_prior = [r for r in prior_rows if r.get("y") is not None]
    models_ = fit_ensemble(labeled_prior, n_models=ensemble_size, seed=seed)
    score_rows = [candidate_row(case_name, case_fields, hardware_id, nodes, c,
                                 run_id="topk:%d" % i) for i, c in enumerate(candidates)]
    mu, _sigma = ensemble_predict(models_, score_rows)
    order = [c for _mu, c in sorted(zip(mu.tolist(), candidates), key=lambda t: t[0])]
    out = []
    best_so_far = None
    for c in order:
        y = true_y_by_key[candidate_key(c)]
        best_so_far = y if best_so_far is None else min(best_so_far, y)
        out.append(float(np.exp(best_so_far - true_y_best) - 1))
    return out


def simulate(ledger_paths, out_dir, n_seeds=50, seed=0, kappa=DEFAULT_KAPPA,
             max_per_rank_omp=DEFAULT_MAX_PER_RANK_OMP, ensemble_size=DEFAULT_ENSEMBLE_SIZE):
    """Offline evaluation: for each of the campaign cases present, hide its own measurements, run
    the D3 loop with the OTHER cases as prior against its 16 measured templates as the (only fully
    measured) ground-truth candidate pool, and compare regret-vs-n-runs against a random-search
    baseline and a non-adaptive "predictor top-k measured in one batch" baseline. Writes
    <out_dir>/simulate_regret.csv and <out_dir>/simulate_summary.md; returns the same table as a
    dict for callers that want it in-process (e.g. tests)."""
    all_rows = []
    for p in ledger_paths:
        all_rows.extend(benchmark.load_ledger(p))
    all_rows = [r for r in all_rows if r.get("compliance_class") != "RESEARCH"]
    baselines = targets.compute_case_baselines(all_rows)
    targets.add_y(all_rows, baselines)

    present_cases = [c for c in campaign_configs.CASE_ORDER
                      if any(r.get("case_base") == c for r in all_rows)]
    if len(present_cases) < 2:
        raise ValueError(
            "simulate needs >=2 campaign cases in --ledger (found %d: %s) -- the 'other cases as "
            "prior' design needs at least one sibling case" % (len(present_cases), present_cases))

    table = defaultdict(dict)  # {case: {"d3": [...], "random": [...], "topk": [...]}}
    for case_name in present_cases:
        case_rows = [r for r in all_rows if r.get("case_base") == case_name]
        prior_rows = [r for r in all_rows if r.get("case_base") != case_name]
        candidates_with_y = _true_y_candidates_for_case(case_rows)
        if len(candidates_with_y) < 2:
            continue
        case_json = campaign_configs.CASES.get(case_name)
        case_fields = case_row_fields(case_json) if case_json else {}
        hardware_id = case_rows[0].get("hardware_id")
        nodes = int(case_rows[0].get("n_nodes") or 4)
        reference_predicate = reference_predicate_for_case(case_json) if case_json else None

        d3 = _run_adaptive_loop(candidates_with_y, prior_rows, case_name, case_fields,
                                 hardware_id, nodes, reference_predicate, kappa,
                                 max_per_rank_omp, ensemble_size, seed)
        rnd = _random_baseline(candidates_with_y, n_seeds, seed0=seed)
        topk = _predictor_top_k_baseline(candidates_with_y, prior_rows, case_name, case_fields,
                                          hardware_id, nodes, ensemble_size, seed)
        n_pool = len(candidates_with_y)
        d3_by_n = dict(d3)
        table[case_name] = dict(
            n_pool=n_pool,
            d3=[d3_by_n.get(n, d3_by_n.get(max(d3_by_n), 0.0) if d3_by_n else 0.0)
                for n in range(1, n_pool + 1)],
            random=rnd, topk=topk,
        )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "simulate_regret.csv")
    with open(csv_path, "w") as f:
        f.write("case,n_pool,n,method,regret\n")
        for case_name, d in table.items():
            for n in range(1, d["n_pool"] + 1):
                for method in ("d3", "random", "topk"):
                    f.write("%s,%d,%d,%s,%.6f\n" % (case_name, d["n_pool"], n, method,
                                                      d[method][n - 1]))

    md_path = os.path.join(out_dir, "simulate_summary.md")
    with open(md_path, "w") as f:
        f.write("# decomp_search simulate: regret vs. number of runs\n\n")
        f.write("Pool = each case's 16 measured STRICT templates only (the only fully measured "
                "pool available) -- this measures SEARCH EFFICIENCY over a small, fixed pool, "
                "NOT search quality over the full decomp_space.enumerate_space space.\n\n")
        for case_name, d in sorted(table.items()):
            f.write("## %s (pool=%d)\n\n" % (case_name, d["n_pool"]))
            f.write("| n | D3 (LCB) | random (mean of %d seeds) | predictor top-k |\n"
                    "|---|---|---|---|\n" % n_seeds)
            for n in range(1, d["n_pool"] + 1):
                f.write("| %d | %.4f | %.4f | %.4f |\n"
                        % (n, d["d3"][n - 1], d["random"][n - 1], d["topk"][n - 1]))
            f.write("\n")
        f.write("## Median across cases\n\n")
        max_n = max((d["n_pool"] for d in table.values()), default=0)
        f.write("| n | D3 (median) | random (median) | predictor top-k (median) |\n"
                "|---|---|---|---|\n")
        for n in range(1, max_n + 1):
            row = {"d3": [], "random": [], "topk": []}
            for d in table.values():
                if n <= d["n_pool"]:
                    for method in row:
                        row[method].append(d[method][n - 1])
            if not row["d3"]:
                continue
            f.write("| %d | %.4f | %.4f | %.4f |\n" % (
                n, float(np.median(row["d3"])), float(np.median(row["random"])),
                float(np.median(row["topk"]))))

    return dict(table=table, csv_path=csv_path, md_path=md_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cmd_propose(argv):
    ap = argparse.ArgumentParser(prog="decomp_search.py propose")
    ap.add_argument("--case", required=True)
    ap.add_argument("--ledger", nargs="+", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-cores", type=int, default=448)
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--hardware-id", default=ds.DEFAULT_HARDWARE)
    ap.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    ap.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    ap.add_argument("--max-per-rank-omp", type=int, default=DEFAULT_MAX_PER_RANK_OMP)
    ap.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    ap.add_argument("--ensemble-seed", type=int, default=0)
    ap.add_argument("--partition", default=DEFAULT_PARTITION)
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    a = ap.parse_args(argv)

    state = load_state(a.state)
    result, configs_out = propose(
        a.case, a.ledger, state, a.batch, total_cores=a.total_cores, nodes=a.nodes,
        hardware_id=a.hardware_id, kappa=a.kappa, max_per_rank_omp=a.max_per_rank_omp,
        ensemble_size=a.ensemble_size, ensemble_seed=a.ensemble_seed, max_rounds=a.max_rounds)

    if result["stopped"]:
        print("STOP (round %d): %s" % (result["round"], result["reason"]))
        return 0

    payload = dict(case_id=a.case, round=result["round"], kappa=a.kappa,
                   n_training_rows=result["n_training_rows"],
                   ensemble_size=result["ensemble_size"], configs=configs_out)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    state["case"] = a.case
    state["hardware_id"] = a.hardware_id
    state["nodes"] = a.nodes
    state["total_cores"] = a.total_cores
    state.setdefault("configs", {}).update({c["id"]: c for c in configs_out})
    state.setdefault("rounds", []).append(dict(
        round=result["round"], proposed_ids=[c["id"] for c in configs_out],
        out_file=os.path.abspath(a.out),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z"))
    save_state(a.state, state)

    incumbent_desc = None
    if result.get("incumbent") is not None:
        ic = result["incumbent"]
        incumbent_desc = "%dx%d numtiles=%s tile_shape=%s dist=%s blocktime=%s" % (
            ic["ranks"], ic["omp"], ic["numtiles"], ic["tile_shape"], ic["dist"], ic["blocktime"])
    print("round %d: proposed %d config(s) -> %s" % (result["round"], len(configs_out), a.out))
    print("  n_training_rows=%d ensemble_size=%d incumbent=%s incumbent_y=%s"
          % (result["n_training_rows"], result["ensemble_size"], incumbent_desc,
             result.get("incumbent_y")))
    print(sbatch_command(a.case, os.path.abspath(a.out), partition=a.partition, reps=a.reps))
    return 0


def _cmd_ingest(argv):
    ap = argparse.ArgumentParser(prog="decomp_search.py ingest")
    ap.add_argument("--state", required=True)
    ap.add_argument("--ledger", nargs="+", required=True)
    a = ap.parse_args(argv)

    state = load_state(a.state)
    total_in, total_skip = 0, 0
    for p in a.ledger:
        n_in, n_skip, reasons = ingest(state, p)
        total_in += n_in
        total_skip += n_skip
        for r in reasons:
            print("SKIP: %s" % r)
    save_state(a.state, state)
    print("ingested %d measurement(s), skipped %d row(s), across %d config id(s) measured so far"
          % (total_in, total_skip, len(state.get("measured", {}))))
    return 0


def _cmd_simulate(argv):
    ap = argparse.ArgumentParser(prog="decomp_search.py simulate")
    ap.add_argument("--ledger", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    ap.add_argument("--max-per-rank-omp", type=int, default=DEFAULT_MAX_PER_RANK_OMP)
    ap.add_argument("--ensemble-size", type=int, default=DEFAULT_ENSEMBLE_SIZE)
    a = ap.parse_args(argv)

    result = simulate(a.ledger, a.out, n_seeds=a.n_seeds, seed=a.seed, kappa=a.kappa,
                       max_per_rank_omp=a.max_per_rank_omp, ensemble_size=a.ensemble_size)
    print("wrote %s and %s" % (result["csv_path"], result["md_path"]))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.stderr.write(
            "usage: decomp_search.py propose --case CASE --ledger L1 [L2 ...] --state S.json "
            "--batch N --out P.json | decomp_search.py ingest --state S.json --ledger L.csv | "
            "decomp_search.py simulate --ledger L1 L2 ... --out DIR\n")
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "propose":
        return _cmd_propose(rest)
    if cmd == "ingest":
        return _cmd_ingest(rest)
    if cmd == "simulate":
        return _cmd_simulate(rest)
    sys.stderr.write("unknown subcommand %r\n" % (cmd,))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
