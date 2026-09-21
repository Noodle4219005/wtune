#!/usr/bin/env python3
"""predictor/recommend.py -- narrow a new/lightly-run WRF case to a STRICT config shortlist.

This is a "manager narrows the search, human decides" tool. It NEVER auto-applies a config: it
prints (and can dump as JSON) a ranked shortlist, an explicit uncertainty label per candidate, and
the exact commands a human would run to verify the top candidates -- nothing more.

CLI:
    python3 predictor/recommend.py --ledger predictor/ledger/runs.csv \\
        --namelist /path/to/namelist.input --case-name mycase --hardware-id f1-spr-icpnq

Reuses (does not reimplement):
  - campaign/configs.py     TEMPLATES / resolve_case / dedupe / build_order -- the 16 STRICT
                             configs and their legality/decomposition-resolution logic.
  - wtune.py (repo root)    parse_namelist()/first_value() -- the namelist reader. NOT
                             wtune.extract_case(), which fills every unread field with an
                             analytical-model sentinel default (e.g. e_vert=50, mp_physics=-1) --
                             exactly the silent-guessing this tool's brief forbids.
  - predictor/targets.py    _reference_order (the same outcome-independent baseline tie-break
                             compute_case_baselines uses) and NOISE (the measured run-to-run
                             reproducibility floor).
  - predictor/validate.py   conformal_intervals_for_new_rows (added alongside this file) and
                             recommendation_label for the SAFE/EXPERIMENTAL uncertainty label.
  - predictor/models.py     the five model classes as-is; see the catboost_pairwise design note
                             below and in the final report for why its score() is handled
                             differently from the other four.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
for _p in (_HERE, _REPO_ROOT, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark                       # predictor/benchmark.py: load_ledger()
import decomp_space as decompspace     # predictor/decomp_space.py: enumerate_space (WTune D4)
import decomp_search as decompsearch   # predictor/decomp_search.py: propose_for_case (WTune D4)
import features as feat                # predictor/features.py
import ledger as ledgermod             # predictor/ledger.py: CORES_PER_NODE
import models as modelmod              # predictor/models.py
import physics as physicsmod           # predictor/physics.py: PHYSICS-GRAY models (decision 0051)
import targets                         # predictor/targets.py
import validate                        # predictor/validate.py
import verify_obs                      # predictor/verify_obs.py: score_case() vs REAL observations
import configs as campaign_configs     # campaign/configs.py
import wtune as wtune_core             # repo-root wtune.py: parse_namelist()/first_value()


class UnsupportedComplianceError(Exception):
    """Raised when --compliance names a scope this tool does not (yet) implement."""


_SUPPORTED_COMPLIANCE = ("STRICT",)

# catboost_pairwise falls back to catboost_point, then ridge (both always importable: ridge is
# pure sklearn, never "unavailable"). catboost_point itself falls back only to ridge.
_MODEL_FALLBACK_CHAIN = {
    "catboost_pairwise": ("catboost_pairwise", "catboost_point", "ridge"),
    "catboost_point": ("catboost_point", "ridge"),
}

_KNOWN_FAILURE_MODE = (
    "a tiny 'conus12'-style case at ~0.055 s/step where every model degrades to ~= naive "
    "(predictor/README.md: 'conus12 (smallest grid) ~= naive for every model')"
)


# ---------------------------------------------------------------------------
# Namelist parsing. Reuses wtune.py's parse_namelist()/first_value() (the group->raw-string
# Fortran namelist reader already in this repo) rather than writing a second one. Deliberately
# does NOT reuse wtune.extract_case(), which fills every field it cannot read with an
# analytical-model default (e_we=101, e_vert=50, mp_physics=-1, io_form_history=2, ...) -- the
# opposite of this tool's "never invent/guess a default value silently" requirement.
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def _all_numbers(raw):
    """Every numeric token in a raw comma-list namelist value string, in order. Reuses
    wtune.first_value's own numeric-token regex convention, applied to every match instead of
    just the first, so per-domain (nested) values are not silently truncated to d01's."""
    if raw is None:
        return []
    out = []
    for m in _NUM_RE.finditer(raw):
        s = m.group(0)
        out.append(float(s) if ("." in s or "e" in s.lower()) else int(s))
    return out


def _scalar(nl, group, key):
    raw = nl.get(group, {}).get(key)
    if raw is None:
        return None
    return wtune_core.first_value(raw)


def _int_scalar(nl, group, key):
    v = _scalar(nl, group, key)
    return int(v) if v is not None else None


def _string_scalar(nl, group, key):
    """A quoted-string namelist scalar (e.g. physics_suite = 'conus') -- wtune_core.first_value's
    own regex only extracts numbers, so a string-valued key needs its own reader: first
    comma-separated token, quotes/whitespace stripped."""
    raw = nl.get(group, {}).get(key)
    if raw is None:
        return None
    tok = raw.split(",", 1)[0].strip().strip("'\"")
    return tok or None


def _domain_list(nl, group, key, max_dom):
    """The first `max_dom` numeric values of a per-domain namelist var, or None if the key is
    entirely absent. Returns fewer than `max_dom` entries (never padded/guessed) when the
    namelist itself lists fewer -- callers must treat that as "legality unknown" for the domains
    beyond what was actually written."""
    vals = _all_numbers(nl.get(group, {}).get(key))
    if not vals:
        return None
    return [int(v) for v in vals[:max_dom]]


def _history_interval_minutes(nl):
    """Whichever history-interval variant this namelist uses, in MINUTES. Same three-variant
    fact campaign/configs.py's `_history_key_and_value` documents (history_interval/_m in
    minutes, _s in seconds) -- reimplemented here against the parsed-dict form wtune.parse_namelist
    returns, since `_history_key_and_value` itself operates on raw namelist TEXT via re.search,
    for in-place editing, not on an already-parsed group dict."""
    tc = nl.get("time_control", {})
    if "history_interval_m" in tc:
        v = wtune_core.first_value(tc["history_interval_m"])
        return float(v) if v is not None else None
    if "history_interval_s" in tc:
        v = wtune_core.first_value(tc["history_interval_s"])
        return float(v) / 60.0 if v is not None else None
    if "history_interval" in tc:
        v = wtune_core.first_value(tc["history_interval"])
        return float(v) if v is not None else None
    return None


def _run_minutes(nl):
    """Total run length in minutes from run_days/run_hours/run_minutes, or None if none of the
    three keys is present at all (never defaulted to 0 -- a real namelist always has at least
    one of them, so an all-absent namelist is worth flagging, not silently zeroing)."""
    tc = nl.get("time_control", {})
    days = wtune_core.first_value(tc.get("run_days"))
    hours = wtune_core.first_value(tc.get("run_hours"))
    minutes = wtune_core.first_value(tc.get("run_minutes"))
    if days is None and hours is None and minutes is None:
        return None
    return int(days or 0) * 1440 + int(hours or 0) * 60 + int(minutes or 0)


def parse_case_from_namelist(namelist_path):
    """Parse the case fields recommend.py needs from a namelist.input. Any field this namelist
    does not carry stays None -- never guessed/defaulted."""
    nl = wtune_core.parse_namelist(namelist_path)
    max_dom = _int_scalar(nl, "domains", "max_dom") or 1
    e_we_list = _domain_list(nl, "domains", "e_we", max_dom)
    e_sn_list = _domain_list(nl, "domains", "e_sn", max_dom)
    e_vert_list = _domain_list(nl, "domains", "e_vert", max_dom)
    return dict(
        max_dom=max_dom,
        e_we=(e_we_list[0] if e_we_list else None),
        e_sn=(e_sn_list[0] if e_sn_list else None),
        e_vert=(e_vert_list[0] if e_vert_list else None),
        e_we_list=e_we_list, e_sn_list=e_sn_list, e_vert_list=e_vert_list,
        dx=_scalar(nl, "domains", "dx"), dy=_scalar(nl, "domains", "dy"),
        time_step=_scalar(nl, "domains", "time_step"),
        mp_physics=_int_scalar(nl, "physics", "mp_physics"),
        cu_physics=_int_scalar(nl, "physics", "cu_physics"),
        bl_pbl_physics=_int_scalar(nl, "physics", "bl_pbl_physics"),
        sf_sfclay_physics=_int_scalar(nl, "physics", "sf_sfclay_physics"),
        sf_surface_physics=_int_scalar(nl, "physics", "sf_surface_physics"),
        ra_lw_physics=_int_scalar(nl, "physics", "ra_lw_physics"),
        ra_sw_physics=_int_scalar(nl, "physics", "ra_sw_physics"),
        radt=_scalar(nl, "physics", "radt"),
        cudt=_scalar(nl, "physics", "cudt"),
        physics_suite=_string_scalar(nl, "physics", "physics_suite"),
        history_interval=_history_interval_minutes(nl),
        io_form_history=_int_scalar(nl, "time_control", "io_form_history"),
        run_minutes=_run_minutes(nl),
    )


# ---------------------------------------------------------------------------
# max_dom resolution (coordinator review item 2): the namelist's own max_dom is what
# parse_case_from_namelist read off disk, but a human may deliberately want to run/verify with
# FEWER active domains than the raw template has (e.g. lafe's real template is max_dom=3, but
# this build cannot run its d02 LES nest -- see campaign/cases.json's lafe notes -- so the
# STRICT campaign itself only ever ran it at max_dom=1). --max-dom lets the caller say which
# domain count the legality check and the verify-plan's edit-namelist command should actually
# target; it can only ever narrow, never invent domains the namelist does not have.
# ---------------------------------------------------------------------------
def resolve_max_dom(requested_max_dom, case_fields):
    namelist_max_dom = case_fields.get("max_dom")
    if requested_max_dom is None:
        return namelist_max_dom
    if namelist_max_dom is not None and requested_max_dom > namelist_max_dom:
        raise ValueError(
            "--max-dom=%d exceeds this namelist's own max_dom=%d -- cannot check decomposition "
            "legality for domains that are not actually present in the namelist"
            % (requested_max_dom, namelist_max_dom))
    return requested_max_dom


# ---------------------------------------------------------------------------
# Candidate generation: campaign/configs.py's 16 STRICT templates, resolved against this case's
# grid. resolve_case() (not reimplemented here) already (a) rejects a template with no legal
# shape at the requested rank count and (b) checks legality against EVERY active nested domain,
# not just d01, and (c) rejects a template whose omp exceeds case_dict["max_omp"] (e.g. c34/
# marshall: WRF-Fire crashes rc=137/139 at the first d01 step with any omp>1). `max_dom` is the
# EFFECTIVE domain count (see resolve_max_dom above), not necessarily case_fields["max_dom"]
# (the namelist's own raw value). case_dict is otherwise built ENTIRELY from case_fields (this
# namelist's own parsed values) -- max_omp is the one exception: it is not derivable from a
# namelist, so it is looked up from campaign/cases.json by case_name instead (same
# CASES.get(case_name) pattern verify_run_minutes and case_frozen_keys already use elsewhere in
# this file); a case_name recommend() does not recognise (not in cases.json, or None) leaves
# case_dict without "max_omp", i.e. unchanged pre-existing behaviour (no cap).
# ---------------------------------------------------------------------------
def build_candidates(case_fields, nodes, max_dom, case_name=None):
    if case_fields["e_we"] is None or case_fields["e_sn"] is None:
        raise ValueError(
            "cannot generate candidates: e_we/e_sn not found in the namelist's &domains section "
            "(required for every decomposition legality check) -- refusing to guess a grid size")

    case_dict = dict(e_we=case_fields["e_we"], e_sn=case_fields["e_sn"],
                      e_vert=case_fields["e_vert"], max_dom=max_dom)
    known_case = campaign_configs.CASES.get(case_name)
    if known_case is not None and known_case.get("max_omp") is not None:
        case_dict["max_omp"] = known_case["max_omp"]
    domain_note = None
    we_list, sn_list = case_fields["e_we_list"], case_fields["e_sn_list"]
    if we_list and sn_list:
        case_dict["e_we_list"] = we_list
        case_dict["e_sn_list"] = sn_list
        n_known = min(len(we_list), len(sn_list))
        if n_known < max_dom:
            domain_note = (
                "namelist listed e_we/e_sn for only %d of %d active domain(s); legality was "
                "only checked against the domain(s) actually present in the namelist"
                % (n_known, max_dom))
    if case_fields.get("max_dom") is not None and max_dom != case_fields["max_dom"]:
        domain_note = (domain_note + "; " if domain_note else "") + (
            "--max-dom=%d overrides this namelist's own max_dom=%d for legality checking and "
            "the verify-plan's edit-namelist command" % (max_dom, case_fields["max_dom"]))

    kept, skipped, aliases = campaign_configs.resolve_case(case_dict)

    node_note = None
    if nodes != 4:
        node_note = (
            "campaign/configs.py's 16 STRICT templates are hardcoded to a 4-node/448-core "
            "allocation (module constant TOTAL_CORES=448; resolve_case() takes no node-count "
            "argument at all). --nodes=%d was requested, but every candidate below is actually "
            "a 4-node/448-core run regardless -- there is no N-node variant of these templates "
            "yet, so treat n_nodes=4 in the table below as the real allocation, not --nodes"
            % nodes
        )
    return kept, skipped, aliases, domain_note, node_note


def _candidate_row(cfg, case_fields, hardware_id):
    """A ledger-row-shaped dict for one resolved candidate config, built only from the parsed
    namelist's case fields + this candidate's own parallel-config knobs -- suitable as input to
    features.case_features()/config_features() and any fitted model's .score()."""
    blocktime = cfg["blocktime"]
    kmp_blocktime = int(blocktime) if str(blocktime).isdigit() else None
    row = dict(
        run_id="candidate:%s" % cfg["id"], case_base=None, hardware_id=hardware_id,
        compliance_class="STRICT", binary_kind="pristine",
        e_we=case_fields["e_we"], e_sn=case_fields["e_sn"], e_vert=case_fields["e_vert"],
        dx=case_fields["dx"], dy=case_fields["dy"], time_step=case_fields["time_step"],
        mp_physics=case_fields["mp_physics"], cu_physics=case_fields["cu_physics"],
        bl_pbl_physics=case_fields["bl_pbl_physics"],
        sf_sfclay_physics=case_fields["sf_sfclay_physics"],
        sf_surface_physics=case_fields["sf_surface_physics"],
        ra_lw_physics=case_fields["ra_lw_physics"], ra_sw_physics=case_fields["ra_sw_physics"],
        radt=case_fields["radt"], cudt=case_fields["cudt"],
        history_interval=case_fields["history_interval"],
        io_form_history=case_fields["io_form_history"],
        n_nodes=(cfg["ranks"] * cfg["omp"]) / float(ledgermod.CORES_PER_NODE),
        n_ranks=cfg["ranks"], omp_threads=cfg["omp"],
        nproc_x=cfg["nx"], nproc_y=cfg["ny"], numtiles=cfg["numtiles"],
        distribution=cfg["dist"], kmp_blocktime=kmp_blocktime,
    )
    row["_cfg"] = cfg
    return row


# ---------------------------------------------------------------------------
# Training data + model fitting (with fallback chain)
# ---------------------------------------------------------------------------
def load_training_rows(ledger_path):
    """Same load-then-baseline-then-y pipeline benchmark.py's main() uses, reused verbatim:
    RESEARCH rows are dropped before baselines are even computed (they are never a candidate
    baseline or evaluated), then y=log(s/s_baseline) is attached per decision group."""
    rows = benchmark.load_ledger(ledger_path)
    rows = [r for r in rows if r.get("compliance_class") != "RESEARCH"]
    baselines = targets.compute_case_baselines(rows)
    targets.add_y(rows, baselines)
    return rows


def _fit_with_fallback(requested_model, train_rows):
    """Try `requested_model`, then its fallback chain, in order; returns
    (model_name_used, fitted_model, attempts) where `attempts` documents every model tried and
    why it was skipped (unavailable import, or NotEnoughData at fit time)."""
    chain = _MODEL_FALLBACK_CHAIN.get(requested_model, (requested_model,))
    avail = modelmod.available_models()
    attempts = []
    for name in chain:
        ok, reason = avail.get(name, (False, "unknown model %r" % name))
        if not ok:
            attempts.append(dict(model=name, used=False, reason=reason or "unavailable"))
            continue
        model = validate.build_model(name)
        try:
            model.fit(train_rows)
        except modelmod.NotEnoughData as e:
            attempts.append(dict(model=name, used=False, reason=str(e)))
            continue
        attempts.append(dict(model=name, used=True, reason=""))
        return name, model, attempts
    raise RuntimeError(
        "no usable model in the fallback chain for %r: %s" % (requested_model, attempts))


# ---------------------------------------------------------------------------
# Scoring: convert each model's score() output to a percent-change-vs-reference for the four
# y-scale models, or leave catboost_pairwise's output as an opaque ranking score. See the module
# docstring and the final report's printed explanation for why these are NOT interchangeable.
# ---------------------------------------------------------------------------
def _score_candidates(model_name, model, candidate_rows):
    preds = model.score(candidate_rows)
    is_pairwise = (model_name == "catboost_pairwise")
    ref_idx = min(range(len(candidate_rows)),
                  key=lambda i: targets._reference_order(feat.config_key(candidate_rows[i])))
    ref_pred = preds[ref_idx]
    scored = []
    for i in range(len(candidate_rows)):
        if is_pairwise:
            scored.append(dict(kind="ranking_score", value=float(preds[i])))
        else:
            scored.append(dict(kind="pct_change", value=math.exp(preds[i] - ref_pred) - 1.0))
    return preds, ref_idx, scored


def _compute_intervals(training_rows, model_name, candidate_rows, compliance, no_intervals):
    """None-and-a-reason when intervals cannot/should not be computed:
      - catboost_pairwise: its score() is not on the y=log(s/baseline) scale the residual pool
        is measured in (see the module docstring) -- an interval built from it would not mean
        anything, so it is skipped by DESIGN, not as a fallback.
      - --no-intervals: skipped by the caller's request.
      - NotEnoughGroups/NotEnoughData: skipped because the scoped training data is too thin.
    """
    if model_name == "catboost_pairwise":
        return None, None  # explained separately, once, in recommend()'s own warning
    if no_intervals:
        return None, "intervals skipped (--no-intervals)"
    try:
        intervals = validate.conformal_intervals_for_new_rows(
            training_rows, candidate_rows, model_name, alphas=(0.10, 0.05), scope=compliance)
    except (validate.NotEnoughGroups, modelmod.NotEnoughData) as e:
        return None, "intervals unavailable: %s" % e
    return intervals, None


def _label_for(y_pred, intervals_for_row):
    if intervals_for_row is None or 0.05 not in intervals_for_row:
        return "EXPERIMENTAL", None
    lo, hi = intervals_for_row[0.05]
    rec = validate.recommendation_label(y_pred, hi)
    return rec["label"], rec


# ---------------------------------------------------------------------------
# Design hook for the next phase (PHYSICS-GRAY + a real fidelity-risk model, predictor/validate.py
# fidelity_model_status/constrained_recommend). Every candidate already carries its own
# compliance_class (see _candidate_row); this is the one pluggable function that decides what to
# say about FIDELITY risk for a candidate, so a future phase only needs to change this one
# function -- never scattered through recommend()/format_report(). STRICT candidates have no
# physics deviation from the case's own physics, so there is nothing to assess: "N/A (STRICT)" is
# not a placeholder for a low-confidence answer, it means the question does not apply yet.
# --compliance=PHYSICS-GRAY is refused earlier in recommend() (UnsupportedComplianceError), so
# the branch below is unreachable today but documents the intended hook for when it lands.
# ---------------------------------------------------------------------------
def _fidelity_status(candidate_row):
    if candidate_row.get("compliance_class") == "STRICT":
        return "N/A (STRICT)"
    return "UNKNOWN (no fidelity model wired in yet)"


# ---------------------------------------------------------------------------
# Evidence warnings (point 5 of the brief): computed against the SAME scope_rows the model was
# actually trained on, so what is reported here is exactly the evidence behind the shown model.
# ---------------------------------------------------------------------------
def _hardware_match(scope_rows, hardware_id):
    same = [r for r in scope_rows if r.get("hardware_id") == hardware_id]
    groups = {feat.decision_group_key(r) for r in same if feat.decision_group_key(r) is not None}
    cases = {r.get("case_base") for r in same if r.get("case_base") is not None}
    return dict(n_rows=len(same), n_decision_groups=len(groups), n_cases=len(cases))


def _grid_neighbors(scope_rows, query_grid_size, hardware_id, top_n=3):
    """Nearest training cases by grid size, same-hardware neighbours listed first (coordinator
    review item 5: pooling e.g. f1-spr-icpnq 'conus' rows with an f1-spr-icpnp query silently
    mixed hardware in the old output). Returns ([(case_base, grid_size, hardware_id), ...],
    out_of_range); out_of_range is still computed across every hardware in scope (case_features
    does feed the model hardware-derived columns, but the model itself pools all hardware for
    training -- see recommend()'s own training-evidence note for the hardware-specific row count)."""
    per_case = {}
    for r in scope_rows:
        cb = r.get("case_base")
        if cb is None or cb in per_case:
            continue
        e_we, e_sn, e_vert = r.get("e_we"), r.get("e_sn"), r.get("e_vert")
        if e_we is None or e_sn is None or e_vert is None:
            continue
        per_case[cb] = (float(e_we) * float(e_sn) * float(e_vert), r.get("hardware_id"))
    if not per_case or query_grid_size is None:
        return [], None
    ordered = sorted(
        per_case.items(),
        key=lambda kv: (kv[1][1] != hardware_id, abs(kv[1][0] - query_grid_size)))
    out_of_range = (query_grid_size < min(gs for gs, _hw in per_case.values())
                    or query_grid_size > max(gs for gs, _hw in per_case.values()))
    neighbors = [(cb, gs, hw) for cb, (gs, hw) in ordered[:top_n]]
    return neighbors, out_of_range


# ---------------------------------------------------------------------------
# Verify-run measurement length (coordinator review item 1): a namelist's own run_minutes is
# frequently a campaign-time placeholder (e.g. lafe's real template carries run_days/hours/
# minutes=0/0/0 -- the STRICT campaign's own runner always overwrites it via edit-namelist before
# ever launching WRF), so using it verbatim produced a useless "run 0 minutes" verify command.
# This derives an actual measurement length instead, and always states how, in that order:
#   1. campaign/cases.json's own steps x dt for a case this tool recognises by name -- the exact
#      length the STRICT campaign itself measures that case with.
#   2. a documented default of 30 d01 steps x this namelist's own time_step, rounded up to whole
#      minutes, for a case not in cases.json.
#   3. (None, reason) when neither is computable (time_step unknown and case not in cases.json).
# ---------------------------------------------------------------------------
_DEFAULT_VERIFY_STEPS = 30


def verify_run_minutes(case_name, case_fields):
    case = campaign_configs.CASES.get(case_name)
    if case is not None and case.get("steps") and case.get("dt"):
        seconds = case["steps"] * case["dt"]
        minutes = int(math.ceil(seconds / 60.0))
        return minutes, (
            "campaign/cases.json[%r]: steps=%d x dt=%ss = %ds -> %d min, rounded up (same "
            "length the STRICT campaign measures this case with)"
            % (case_name, case["steps"], case["dt"], seconds, minutes))
    time_step = case_fields.get("time_step")
    if time_step:
        seconds = _DEFAULT_VERIFY_STEPS * float(time_step)
        minutes = int(math.ceil(seconds / 60.0))
        return minutes, (
            "%r not found in campaign/cases.json (or missing steps/dt); documented default of "
            "%d d01 steps x this namelist's time_step=%ss = %.0fs -> %d min, rounded up"
            % (case_name, _DEFAULT_VERIFY_STEPS, time_step, seconds, minutes))
    return None, (
        "cannot derive a measurement length: %r not in campaign/cases.json and this namelist's "
        "own time_step is unknown" % case_name)


# ---------------------------------------------------------------------------
# "What to run to verify" -- reuses campaign/configs.py's own CLI (edit-namelist) and
# build_order(), and the exact srun invocation shape from campaign/strict_campaign.sbatch
# (ntasks_per_node = ranks / 4, OMP_NUM_THREADS/KMP_BLOCKTIME export, --distribution=), instead
# of hand-inventing a new one. `max_dom` is the EFFECTIVE domain count (resolve_max_dom), so the
# verify command targets the domains the user will actually run, not necessarily every domain
# the raw namelist happens to carry (coordinator review item 2).
# ---------------------------------------------------------------------------
def _build_verify_plan(reference_entry, shortlist, case_fields, nodes, max_dom, case_name):
    run_minutes, run_minutes_source = verify_run_minutes(case_name, case_fields)
    run_minutes_arg = (str(run_minutes) if run_minutes is not None
                        else "<RUN_MINUTES: set to your desired run length -- %s>"
                             % run_minutes_source)

    entries, ids_seen = [], []
    for e in list(shortlist[:3]) + [reference_entry]:
        if e["config_id"] in ids_seen:
            continue
        ids_seen.append(e["config_id"])
        ranks, omp = e["n_ranks"], e["omp_threads"]
        ntasks_per_node = ranks // 4  # campaign templates are always 4 nodes; see
                                       # campaign/strict_campaign.sbatch:144 `ranks / 4`
        namelist_cmd = (
            "python3 campaign/configs.py edit-namelist namelist.input.orig namelist.input "
            "--nproc-x %d --nproc-y %d --numtiles %d --run-minutes %s --max-dom %d "
            "--history-mode off --ranks %d"
            % (e["nproc_x"], e["nproc_y"], e["numtiles"], run_minutes_arg, max_dom, ranks)
        )
        run_lines = [
            "export OMP_NUM_THREADS=%d" % omp,
            ("unset KMP_BLOCKTIME  # template blocktime=default"
             if e["kmp_blocktime"] is None else "export KMP_BLOCKTIME=%d" % e["kmp_blocktime"]),
            "srun --mpi=pmi2 --distribution=%s -n %d --ntasks-per-node=%d -c %d ./wrf.exe"
            % (e["distribution"], ranks, ntasks_per_node, omp),
        ]
        entries.append(dict(
            config_id=e["config_id"],
            namelist_keys=dict(nproc_x=e["nproc_x"], nproc_y=e["nproc_y"], numtiles=e["numtiles"]),
            namelist_edit_command=namelist_cmd,
            run_command="\n".join(run_lines),
        ))
    order = campaign_configs.build_order(ids_seen, reps=2)
    note = ("measure the top-3 shortlisted configs PLUS the reference config, in this "
            "palindromic order, on the SAME node allocation, to control for run-to-run drift "
            "(order built by campaign/configs.py build_order)")
    if nodes != 4:
        note += (". NOTE: these are 4-node/448-core configs (campaign/configs.py TOTAL_CORES); "
                 "--nodes=%d does not change that -- see the warnings section" % nodes)
    return dict(configs=entries, palindromic_run_order=order, note=note,
                run_minutes=run_minutes, run_minutes_source=run_minutes_source)


# ---------------------------------------------------------------------------
# In-sample check (coordinator review item 3): a case already present in the training ledger
# (by case_base name, or by an identical grid under a different name) makes its own shortlist
# in-sample, not an honest out-of-sample prediction -- this must be loud and unmissable, not
# buried. Checked against whatever rows actually end up in the fitted model's training set (i.e.
# AFTER --holdout-case has already been applied by the caller), so passing the matching
# --holdout-case correctly makes this warning go away for that case_base (while still catching a
# DIFFERENT case_base that happens to share the exact same grid).
# ---------------------------------------------------------------------------
def _in_sample_check(scope_rows, case_name, case_fields):
    case_bases = {r.get("case_base") for r in scope_rows}
    by_name = case_name in case_bases

    grid = None
    if (case_fields["e_we"] is not None and case_fields["e_sn"] is not None
            and case_fields["e_vert"] is not None):
        grid = (float(case_fields["e_we"]), float(case_fields["e_sn"]), float(case_fields["e_vert"]))
    grid_match_cases = set()
    if grid is not None:
        for r in scope_rows:
            if (r.get("e_we") is not None and r.get("e_sn") is not None
                    and r.get("e_vert") is not None
                    and (float(r["e_we"]), float(r["e_sn"]), float(r["e_vert"])) == grid):
                grid_match_cases.add(r.get("case_base"))

    return by_name, grid_match_cases


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def recommend(ledger_path, namelist_path, case_name, hardware_id, nodes=4,
              compliance="STRICT", model="catboost_pairwise", top_k=5, no_intervals=False,
              max_dom=None, holdout_case=None):
    """Build a STRICT config shortlist for one case. Returns a JSON-serializable dict. Never
    auto-applies anything -- see result["auto_apply_disclaimer"].

    max_dom: override the number of active domains checked for legality and used in the
        verify-plan (default: this namelist's own max_dom; see resolve_max_dom).
    holdout_case: drop this case_base's rows from training before fitting anything, for an
        honest out-of-sample shortlist when `case_name` (or its grid) is itself in the ledger."""
    if compliance not in _SUPPORTED_COMPLIANCE:
        raise UnsupportedComplianceError(
            "--compliance=%s recommendation needs the fidelity-risk model (predictor/validate.py "
            "fidelity_model_status/constrained_recommend) and is not implemented yet -- only "
            "STRICT is supported today; refusing to proceed rather than silently treating it as "
            "STRICT" % compliance)

    case_fields = parse_case_from_namelist(namelist_path)
    effective_max_dom = resolve_max_dom(max_dom, case_fields)
    kept, skipped, aliases, domain_note, node_note = build_candidates(
        case_fields, nodes, effective_max_dom, case_name=case_name)

    warnings = []
    if domain_note:
        warnings.append(domain_note)
    if node_note:
        warnings.append(node_note)

    # Always-present evidence note (brief item 5): the run-to-run noise floor, regardless of
    # whether any two shown candidates happen to land inside it -- see the per-pair check below
    # for the sharper, shortlist-specific version of this same fact.
    warnings.append(
        "run-to-run noise floor: predicted differences smaller than ~%.2f%% (targets.NOISE, a "
        "log-ratio of %.4f, measured on 40-step karina runs) are not a reliable ordering -- do "
        "not read a rank gap below this as a real effect"
        % (100 * (math.exp(targets.NOISE) - 1), targets.NOISE))

    candidate_rows = [_candidate_row(cfg, case_fields, hardware_id) for cfg in kept]

    training_rows = load_training_rows(ledger_path)
    if holdout_case:
        n_before = len(training_rows)
        training_rows = [r for r in training_rows if r.get("case_base") != holdout_case]
        n_dropped = n_before - len(training_rows)
        if n_dropped:
            warnings.append(
                "--holdout-case=%r: dropped %d training row(s) with that case_base before "
                "fitting (honest out-of-sample evaluation)" % (holdout_case, n_dropped))
        else:
            warnings.append(
                "--holdout-case=%r matched ZERO training rows -- check the spelling against the "
                "ledger's case_base values; nothing was actually held out" % holdout_case)

    scope_rows = [r for r in training_rows if r.get("compliance_class") == compliance]

    by_name, grid_match_cases = _in_sample_check(scope_rows, case_name, case_fields)
    if by_name or grid_match_cases:
        reasons = []
        if by_name:
            reasons.append("case_base=%r appears in the training rows used to fit this model"
                            % case_name)
        other_grid_matches = sorted(grid_match_cases - ({case_name} if by_name else set()))
        if other_grid_matches:
            reasons.append("an identical grid (e_we*e_sn*e_vert) also matches training case(s) %s"
                            % other_grid_matches)
        warnings.append(
            "IN-SAMPLE: this case is in the training data; the shortlist is not an out-of-sample "
            "prediction (%s)%s"
            % ("; ".join(reasons),
               "" if holdout_case == case_name else
               " -- pass --holdout-case %r to see an honest out-of-sample shortlist" % case_name))

    hw_match = _hardware_match(scope_rows, hardware_id)
    warnings.append(
        "training evidence on hardware_id=%r (scope=%s): %d row(s), %d decision group(s), %d "
        "case(s)" % (hardware_id, compliance, hw_match["n_rows"], hw_match["n_decision_groups"],
                      hw_match["n_cases"]))
    if hw_match["n_rows"] == 0:
        warnings.append(
            "zero training rows share hardware_id=%r: cross-hardware prediction was measured at "
            "leave-one-hardware-out and found near-chance (pairwise accuracy ~0.37-0.54, i.e. "
            "~0.5 -- see predictor/README.md); every candidate label below is forced EXPERIMENTAL "
            "regardless of what its own interval would otherwise say" % hardware_id)

    query_grid_size = None
    if (case_fields["e_we"] is not None and case_fields["e_sn"] is not None
            and case_fields["e_vert"] is not None):
        query_grid_size = (float(case_fields["e_we"]) * float(case_fields["e_sn"])
                            * float(case_fields["e_vert"]))
    neighbors, out_of_range = _grid_neighbors(scope_rows, query_grid_size, hardware_id)
    if query_grid_size is None:
        warnings.append("e_vert unknown: cannot compare this case's grid size to training cases")
    elif not neighbors:
        warnings.append(
            "no training case in scope=%s has a known e_we*e_sn*e_vert to compare against"
            % compliance)
    else:
        warnings.append(
            "nearest training case(s) by grid size (e_we*e_sn*e_vert; this case=%.4g on "
            "hardware_id=%r; same-hardware neighbours listed first): %s"
            % (query_grid_size, hardware_id,
               ", ".join("%s [%s]%s (%.4g)" % (cb, hw, "" if hw == hardware_id else " CROSS-HW", gs)
                         for cb, gs, hw in neighbors)))
        if out_of_range:
            warnings.append(
                "this case's grid (e_we*e_sn*e_vert=%.4g) is outside the range spanned by every "
                "scope=%s training case (any hardware) -- out-of-range extrapolation. Known "
                "failure mode: %s" % (query_grid_size, compliance, _KNOWN_FAILURE_MODE))

    result = dict(
        case_name=case_name, hardware_id=hardware_id, nodes=nodes, compliance=compliance,
        max_dom=effective_max_dom, holdout_case=holdout_case, model_requested=model,
        auto_apply_disclaimer=(
            "This tool only narrows the search to a shortlist. A human decides and applies the "
            "final config -- nothing here is auto-applied."),
        hardware_match=hw_match,
        grid_neighbors=[dict(case_base=cb, grid_size=gs, hardware_id=hw,
                              same_hardware=(hw == hardware_id)) for cb, gs, hw in neighbors],
        grid_out_of_range=out_of_range,
        skipped=[dict(config_id=tid, reason=reason) for tid, reason in skipped],
        aliases=dict(aliases),
        warnings=warnings,
    )

    if not candidate_rows:
        result.update(model_used=None, shortlist=[], reference=None, all_ranked=[], verify_plan=None)
        result["warnings"].append(
            "no legal candidates for this case/--nodes combination -- see 'skipped' for why "
            "every one of the 16 STRICT templates was rejected")
        return result

    model_name, fitted_model, attempts = _fit_with_fallback(model, scope_rows)
    result["model_used"] = model_name
    result["model_fallback_attempts"] = attempts
    if model_name != model:
        warnings.append(
            "requested model %r unavailable or had insufficient training data; fell back to "
            "%r (see model_fallback_attempts for why each tier was skipped)" % (model, model_name))

    preds, ref_idx, scored = _score_candidates(model_name, fitted_model, candidate_rows)
    is_pairwise = (model_name == "catboost_pairwise")
    if is_pairwise:
        warnings.append(
            "catboost_pairwise's score() (predictor/models.py PairwiseModel.score) returns "
            "-CatBoostRanker/-LGBMRanker.predict(): a PairLogit/lambdarank RELATIVE ranking "
            "score, not a value on the y=log(s/baseline) scale the other four models share. "
            "Candidates below are ranked by that opaque score (lower=faster, same convention as "
            "the other models); it is used for RANKING/ORDER only. The percent-change estimate, "
            "interval and SAFE/EXPERIMENTAL label instead come from a separate magnitude "
            "companion model fit on the same data (see 'magnitude' below), not from this score.")

    intervals, interval_note = _compute_intervals(
        training_rows, model_name, candidate_rows, compliance, no_intervals)
    if interval_note:
        warnings.append(interval_note)

    # Magnitude companion (coordinator review item 4): always fit catboost_point (fallback ridge)
    # regardless of the requested/primary ranking model, so every recommendation carries an
    # honest, calibrated percent-change estimate + SAFE/EXPERIMENTAL label even when --model is
    # catboost_pairwise (which cannot produce either on its own -- see the warning above).
    companion_name, companion_model, companion_attempts = _fit_with_fallback(
        "catboost_point", scope_rows)
    result["magnitude_model_used"] = companion_name
    result["magnitude_model_fallback_attempts"] = companion_attempts
    if companion_name != "catboost_point":
        warnings.append(
            "magnitude companion requested catboost_point; unavailable or had insufficient "
            "training data, fell back to %r (see magnitude_model_fallback_attempts)"
            % companion_name)

    preds_c, ref_idx_c, scored_c = _score_candidates(companion_name, companion_model, candidate_rows)
    # ref_idx is the reference candidate's identity (targets._reference_order), which depends only
    # on candidate config knobs, never on which model scored it -- both calls must agree.
    assert ref_idx_c == ref_idx, "reference candidate must be model-independent"
    intervals_c, interval_note_c = _compute_intervals(
        training_rows, companion_name, candidate_rows, compliance, no_intervals)
    if interval_note_c:
        warnings.append("magnitude companion: %s" % interval_note_c)

    order = sorted(range(len(candidate_rows)), key=lambda i: preds[i])
    force_experimental = (hw_match["n_rows"] == 0)
    ref_pred = preds[ref_idx]
    ref_pred_c = preds_c[ref_idx_c]

    ranked = []
    for rank, i in enumerate(order, start=1):
        cfg = candidate_rows[i]["_cfg"]

        # Primary ranking model's own scale -- informational only (opaque for catboost_pairwise;
        # a genuine pct_change + interval for a y-scale primary). SAFE/EXPERIMENTAL never comes
        # from this any more; see the magnitude companion below.
        iv = None
        if not is_pairwise and intervals is not None and i in intervals:
            iv = {alpha: (lo - ref_pred, hi - ref_pred) for alpha, (lo, hi) in intervals[i].items()}

        # Magnitude companion: the one source of the percent-change estimate and the
        # SAFE/EXPERIMENTAL label, regardless of what the primary ranking model is. Same
        # interval-shift design as before: shift by the companion's OWN reference prediction so
        # the interval brackets "value vs reference", not an undefined absolute baseline -- a
        # documented simplification that does not propagate the reference's own uncertainty.
        iv_c, label, rec = None, "EXPERIMENTAL", None
        if not force_experimental and intervals_c is not None and i in intervals_c:
            iv_c = {alpha: (lo - ref_pred_c, hi - ref_pred_c)
                    for alpha, (lo, hi) in intervals_c[i].items()}
            label, rec = _label_for(preds_c[i] - ref_pred_c, iv_c)

        ranked.append(dict(
            rank=rank, config_id=cfg["id"], n_ranks=cfg["ranks"], omp_threads=cfg["omp"],
            nproc_x=cfg["nx"], nproc_y=cfg["ny"], numtiles=cfg["numtiles"],
            distribution=cfg["dist"], kmp_blocktime=candidate_rows[i]["kmp_blocktime"],
            score_kind=scored[i]["kind"], value=scored[i]["value"],
            predicted_y=float(preds[i]),
            interval_95=(list(iv[0.05]) if iv and 0.05 in iv else None),
            interval_90=(list(iv[0.10]) if iv and 0.10 in iv else None),
            magnitude_model=companion_name,
            magnitude_value=scored_c[i]["value"],
            magnitude_predicted_y=float(preds_c[i]),
            magnitude_interval_95=(list(iv_c[0.05]) if iv_c and 0.05 in iv_c else None),
            magnitude_interval_90=(list(iv_c[0.10]) if iv_c and 0.10 in iv_c else None),
            predicted_gain=(rec["predicted_gain"] if rec else None),
            gain_lower_95=(rec["gain_lower_95"] if rec else None),
            label=label,
            is_reference=(i == ref_idx),
            compliance_class=candidate_rows[i]["compliance_class"],
            fidelity_status=_fidelity_status(candidate_rows[i]),
        ))

    shortlist = ranked[:top_k]
    reference_entry = dict(next(e for e in ranked if e["is_reference"]))
    reference_entry["in_shortlist"] = reference_entry["rank"] <= top_k

    # Noise-floor check now always uses the magnitude companion's y (always on the
    # y=log(s/baseline) scale, regardless of what the primary ranking model is).
    y_by_id = {e["config_id"]: e["magnitude_predicted_y"] for e in ranked}
    show_ids = [e["config_id"] for e in shortlist]
    if reference_entry["config_id"] not in show_ids:
        show_ids = show_ids + [reference_entry["config_id"]]
    noise_pairs = [(a, b) for a, b in zip(show_ids, show_ids[1:])
                   if abs(y_by_id[a] - y_by_id[b]) < targets.NOISE]
    if noise_pairs:
        warnings.append(
            "predicted change (magnitude companion) is within the run-to-run noise floor "
            "(targets.NOISE, ~%.1f%%) between: %s -- not a reliable ordering, do not read the "
            "rank gap as real"
            % (100 * (math.exp(targets.NOISE) - 1), ", ".join("%s~%s" % p for p in noise_pairs))
        )

    result.update(
        shortlist=shortlist, reference=reference_entry, all_ranked=ranked,
        verify_plan=_build_verify_plan(
            reference_entry, shortlist, case_fields, nodes, effective_max_dom, case_name),
    )
    return result


# ---------------------------------------------------------------------------
# --space full (WTune D4): score the WHOLE legal decomp_space.enumerate_space pool for this case
# instead of just the 16 fixed STRICT templates. Still STRICT-only (decomp_space's pool is plain
# WRF decomposition/tiling knobs, no physics deviation) and still "narrows the search, human
# decides" -- see recommend()'s own module docstring, unchanged for --space templates (the
# default).
# ---------------------------------------------------------------------------
def _decomp_space_case_dict(case_fields, max_dom):
    """The subset of decomp_space.enumerate_space's case-shaped input this tool can read off a
    namelist (see build_candidates, which builds the same dict inline for the STRICT-templates
    path -- kept separate here rather than refactored into a shared helper both call, so neither
    path risks a regression from touching code the other one depends on)."""
    case_dict = dict(e_we=case_fields["e_we"], e_sn=case_fields["e_sn"],
                      e_vert=case_fields["e_vert"], max_dom=max_dom)
    we_list, sn_list = case_fields.get("e_we_list"), case_fields.get("e_sn_list")
    if we_list and sn_list:
        case_dict["e_we_list"] = we_list
        case_dict["e_sn_list"] = sn_list
    return case_dict


# campaign/strict_campaign.sbatch's own notes convention for a CONFIG_FILE-generated run (see
# ledger.py's adapt_campaign_ledger docstring): "config_source=generated numtiles_x=.. ...". Not
# parsed into a ledger column by ledger.py/benchmark.py (LEDGER_FIELDS has no such columns), so
# it must be read back out of `notes` here to know a training row's ACTUAL tile geometry.
_NOTES_KV_RE = re.compile(r"(\w+)=(\S*)")


def _parse_notes_tile_geometry(notes):
    """(numtiles_x, numtiles_y, tile_sz_x, tile_sz_y, tile_strategy) parsed from a ledger row's
    `notes` field -- all 0 when notes carries none of these keys, which is every STRICT template
    C01-C16 run: WRF's own default 1D-Y tile resolution, no explicit override (see
    campaign/configs.py TEMPLATES: only `numtiles` is ever set)."""
    if not notes:
        return 0, 0, 0, 0, 0
    kv = dict(_NOTES_KV_RE.findall(notes))

    def _i(k):
        try:
            return int(kv.get(k))
        except (TypeError, ValueError):
            return 0
    return _i("numtiles_x"), _i("numtiles_y"), _i("tile_sz_x"), _i("tile_sz_y"), _i("tile_strategy")


def _tile_shape_label(numtiles_x, numtiles_y, tile_sz_x, tile_sz_y):
    """Same three-way label decomp_space.py's own candidates carry as `tile_shape`."""
    if tile_sz_x or tile_sz_y:
        return "tile_sz"
    if numtiles_x or numtiles_y:
        return "2d"
    return "1d_y"


def _hardware_seen_sets(scope_rows, hardware_id):
    """What has actually been MEASURED on `hardware_id` among `scope_rows` (already scoped to
    compliance=STRICT by the caller): {(ranks,omp) seen}, {tile_shape seen}, and the [min,max]
    range of feat.config_features' own (approximate, ceil-based) patch_x/patch_y among those rows
    -- the three axes --space full's extrapolation warning checks a candidate against."""
    same_hw = [r for r in scope_rows if r.get("hardware_id") == hardware_id]
    seen_ranks_omp = set()
    seen_tile_shapes = set()
    patch_xs, patch_ys = [], []
    for r in same_hw:
        n_ranks, omp = r.get("n_ranks"), r.get("omp_threads")
        if n_ranks and omp:
            seen_ranks_omp.add((int(n_ranks), int(omp)))
        ntx, nty, tsx, tsy, _tstrat = _parse_notes_tile_geometry(r.get("notes"))
        seen_tile_shapes.add(_tile_shape_label(ntx, nty, tsx, tsy))
        cf = feat.config_features(r)
        if cf.get("patch_x") is not None:
            patch_xs.append(cf["patch_x"])
        if cf.get("patch_y") is not None:
            patch_ys.append(cf["patch_y"])
    patch_x_range = (min(patch_xs), max(patch_xs)) if patch_xs else None
    patch_y_range = (min(patch_ys), max(patch_ys)) if patch_ys else None
    return dict(n_rows=len(same_hw), ranks_omp=seen_ranks_omp, tile_shapes=seen_tile_shapes,
                patch_x_range=patch_x_range, patch_y_range=patch_y_range)


def _candidate_extrapolation_reasons(cfg, cfg_row, seen):
    """[reason, ...] -- empty iff this candidate's ranks x omp, tile shape, AND approximate patch
    size all fall within what `seen` (see _hardware_seen_sets) already measured on this hardware.
    Any non-empty result forces the candidate's label to EXPERIMENTAL regardless of what its own
    conformal interval would otherwise say (see recommend_full_space)."""
    reasons = []
    ro = (cfg["ranks"], cfg["omp"])
    if ro not in seen["ranks_omp"]:
        reasons.append("ranks x omp = %dx%d never measured on this hardware" % ro)
    tshape = cfg.get("tile_shape")
    if tshape not in seen["tile_shapes"]:
        reasons.append("tile shape %r never measured on this hardware" % (tshape,))
    cf = feat.config_features(cfg_row)
    px, py = cf.get("patch_x"), cf.get("patch_y")
    pxr, pyr = seen["patch_x_range"], seen["patch_y_range"]
    if pxr is None or px is None or not (pxr[0] <= px <= pxr[1]):
        reasons.append("patch_x=%s outside measured range %s" % (px, pxr))
    if pyr is None or py is None or not (pyr[0] <= py <= pyr[1]):
        reasons.append("patch_y=%s outside measured range %s" % (py, pyr))
    return reasons


DEFAULT_FULL_SPACE_TOP_K = 10


def recommend_full_space(ledger_path, namelist_path, case_name, hardware_id, nodes=4,
                          model="catboost_pairwise", top_k=DEFAULT_FULL_SPACE_TOP_K,
                          no_intervals=False, max_dom=None, holdout_case=None,
                          tile_options=None, omp_options=None):
    """Like recommend() (STRICT only, same "narrows the search, human decides" contract), but
    scores decomp_space.enumerate_space's WHOLE legal pool for this case/resource allocation
    instead of just the 16 fixed STRICT templates -- the pool can run into the thousands (see
    decomp_space.py's own module docstring), so conformal intervals (validate.py scores ONE
    candidate row at a time per fold -- see conformal_intervals_for_new_rows) are computed only
    for the already-ranked top_k, not the whole pool; this is the one deliberate accuracy/cost
    tradeoff vs recommend()'s STRICT-templates path, where 16 candidates make scoring everything
    cheap enough not to matter."""
    case_fields = parse_case_from_namelist(namelist_path)
    effective_max_dom = resolve_max_dom(max_dom, case_fields)
    if case_fields["e_we"] is None or case_fields["e_sn"] is None or case_fields["e_vert"] is None:
        raise ValueError(
            "cannot enumerate the full decomposition space: e_we/e_sn/e_vert not found in the "
            "namelist's &domains section -- refusing to guess a grid size (e_vert is required "
            "here, unlike the STRICT-templates path, because decomp_space.py's tile-geometry "
            "cells_per_tile fields multiply by it directly)")
    case_dict = _decomp_space_case_dict(case_fields, effective_max_dom)
    total_cores = nodes * ledgermod.CORES_PER_NODE
    candidates, axis = decompspace.enumerate_space(
        case_dict, total_cores=total_cores, nodes=nodes, hardware_id=hardware_id,
        omp_options=omp_options, tile_options=tile_options)

    warnings = [
        "--space full: enumerated %d legal candidate(s) (omp options=%s; total_cores=%d, "
        "nodes=%d) via decomp_space.enumerate_space -- conformal intervals below are computed "
        "only for the top_k shown, not the whole pool (see recommend_full_space's docstring)"
        % (axis["total"], axis["omp_options"], total_cores, nodes),
        "run-to-run noise floor: predicted differences smaller than ~%.2f%% (targets.NOISE) are "
        "not a reliable ordering -- do not read a rank gap below this as a real effect"
        % (100 * (math.exp(targets.NOISE) - 1)),
    ]
    if not candidates:
        return dict(case_name=case_name, hardware_id=hardware_id, nodes=nodes, space="full",
                    max_dom=effective_max_dom, total_candidates=0, shortlist=[], warnings=warnings,
                    auto_apply_disclaimer=(
                        "This tool only narrows the search to a shortlist. A human decides and "
                        "applies the final config -- nothing here is auto-applied."))

    training_rows = load_training_rows(ledger_path)
    if holdout_case:
        n_before = len(training_rows)
        training_rows = [r for r in training_rows if r.get("case_base") != holdout_case]
        if n_before != len(training_rows):
            warnings.append("--holdout-case=%r: dropped %d training row(s) before fitting"
                             % (holdout_case, n_before - len(training_rows)))
    scope_rows = [r for r in training_rows if r.get("compliance_class") == "STRICT"]
    seen = _hardware_seen_sets(scope_rows, hardware_id)
    warnings.append(
        "training evidence on hardware_id=%r (scope=STRICT): %d row(s); ranks x omp combos "
        "measured: %d; tile shapes measured: %s"
        % (hardware_id, seen["n_rows"], len(seen["ranks_omp"]), sorted(seen["tile_shapes"])))
    force_experimental_all = (seen["n_rows"] == 0)
    if force_experimental_all:
        warnings.append(
            "zero training rows share hardware_id=%r: every candidate below is forced "
            "EXPERIMENTAL regardless of predicted performance" % hardware_id)

    candidate_rows = [decompsearch.candidate_row(None, case_fields, hardware_id, nodes, c,
                                                  run_id="full:%d" % i)
                      for i, c in enumerate(candidates)]

    model_name, fitted_model, attempts = _fit_with_fallback(model, scope_rows)
    if model_name != model:
        warnings.append(
            "requested model %r unavailable or had insufficient training data; fell back to %r"
            % (model, model_name))
    preds, ref_idx, scored = _score_candidates(model_name, fitted_model, candidate_rows)

    companion_name, companion_model, _companion_attempts = _fit_with_fallback(
        "catboost_point", scope_rows)
    preds_c, ref_idx_c, scored_c = _score_candidates(companion_name, companion_model, candidate_rows)
    assert ref_idx_c == ref_idx, "reference candidate must be model-independent"
    ref_pred_c = preds_c[ref_idx_c]

    order = sorted(range(len(candidate_rows)), key=lambda i: preds[i])[:top_k]

    intervals_c, interval_note_c = _compute_intervals(
        training_rows, companion_name, [candidate_rows[i] for i in order], "STRICT", no_intervals)
    if interval_note_c:
        warnings.append("magnitude companion: %s" % interval_note_c)

    shortlist = []
    for rank, i in enumerate(order, start=1):
        cfg = candidates[i]
        row = candidate_rows[i]
        reasons = _candidate_extrapolation_reasons(cfg, row, seen)
        is_pairwise = (model_name == "catboost_pairwise")

        iv_c, label, rec = None, "EXPERIMENTAL", None
        local_idx = rank - 1  # position within the top_k-only intervals_c mapping (order[rank-1]==i)
        if not force_experimental_all and intervals_c is not None and local_idx in intervals_c:
            iv_c = {alpha: (lo - ref_pred_c, hi - ref_pred_c)
                    for alpha, (lo, hi) in intervals_c[local_idx].items()}
            label, rec = _label_for(preds_c[i] - ref_pred_c, iv_c)
        if reasons:
            label = "EXPERIMENTAL"

        shortlist.append(dict(
            rank=rank, n_ranks=cfg["ranks"], omp_threads=cfg["omp"],
            nproc_x=cfg["nx"], nproc_y=cfg["ny"],
            patch_x=cfg.get("patch_x"), patch_x_max=cfg.get("patch_x_max"),
            patch_y=cfg.get("patch_y"), patch_y_max=cfg.get("patch_y_max"),
            numtiles=cfg["numtiles"], tile_shape=cfg["tile_shape"],
            numtiles_x=cfg.get("numtiles_x", 0), numtiles_y=cfg.get("numtiles_y", 0),
            tile_sz_x=cfg.get("tile_sz_x", 0), tile_sz_y=cfg.get("tile_sz_y", 0),
            tile_strategy=cfg.get("tile_strategy", 0),
            cells_per_tile_min=cfg.get("cells_per_tile_min"),
            cells_per_tile_max=cfg.get("cells_per_tile_max"),
            distribution=cfg["dist"], kmp_blocktime=row["kmp_blocktime"],
            score_kind=scored[i]["kind"], value=scored[i]["value"],
            magnitude_value=scored_c[i]["value"],
            magnitude_interval_95=(list(iv_c[0.05]) if iv_c and 0.05 in iv_c else None),
            magnitude_interval_90=(list(iv_c[0.10]) if iv_c and 0.10 in iv_c else None),
            predicted_gain=(rec["predicted_gain"] if rec else None),
            gain_lower_95=(rec["gain_lower_95"] if rec else None),
            label=label, extrapolated=bool(reasons), extrapolation_reasons=reasons,
            is_reference=(i == ref_idx), is_pairwise_ranking=is_pairwise,
            namelist_edit_command=_full_space_edit_command(cfg, effective_max_dom),
            run_command=_full_space_run_command(cfg, nodes),
        ))

    return dict(
        case_name=case_name, hardware_id=hardware_id, nodes=nodes, space="full",
        max_dom=effective_max_dom, total_candidates=axis["total"],
        model_requested=model, model_used=model_name, magnitude_model_used=companion_name,
        shortlist=shortlist, warnings=warnings,
        auto_apply_disclaimer=(
            "This tool only narrows the search to a shortlist. A human decides and applies the "
            "final config -- nothing here is auto-applied."),
    )


def _full_space_edit_command(cfg, max_dom):
    parts = ["python3 campaign/configs.py edit-namelist namelist.input.orig namelist.input",
             "--nproc-x %d --nproc-y %d --numtiles %d" % (cfg["nx"], cfg["ny"], cfg["numtiles"]),
             "--run-minutes <RUN_MINUTES> --max-dom %d --history-mode off --ranks %d"
             % (max_dom, cfg["ranks"])]
    if cfg.get("numtiles_x"):
        parts.append("--numtiles-x %d" % cfg["numtiles_x"])
    if cfg.get("numtiles_y"):
        parts.append("--numtiles-y %d" % cfg["numtiles_y"])
    if cfg.get("tile_sz_x"):
        parts.append("--tile-sz-x %d" % cfg["tile_sz_x"])
    if cfg.get("tile_sz_y"):
        parts.append("--tile-sz-y %d" % cfg["tile_sz_y"])
    if cfg.get("tile_strategy"):
        parts.append("--tile-strategy %d" % cfg["tile_strategy"])
    return " ".join(parts)


def _full_space_run_command(cfg, nodes):
    ranks, omp = cfg["ranks"], cfg["omp"]
    ntasks_per_node = ranks // nodes
    kmp_line = ("unset KMP_BLOCKTIME  # blocktime=default (not overridden)"
                if cfg["blocktime"] != "0" else "export KMP_BLOCKTIME=0")
    return "\n".join([
        "export OMP_NUM_THREADS=%d" % omp,
        kmp_line,
        "srun --mpi=pmi2 --distribution=%s -n %d --ntasks-per-node=%d -c %d ./wrf.exe"
        % (cfg["dist"], ranks, ntasks_per_node, omp),
    ])


def format_full_space_report(result):
    lines = []
    lines.append("=" * 78)
    lines.append("WTune STRICT full-space shortlist -- case=%s hardware=%s nodes=%s max_dom=%s"
                  % (result["case_name"], result["hardware_id"], result["nodes"],
                     result.get("max_dom")))
    lines.append(result["auto_apply_disclaimer"])
    lines.append("=" * 78)
    if result["warnings"]:
        lines.append("\n-- Evidence warnings --")
        for w in result["warnings"]:
            lines.append("  * %s" % w)
    if not result.get("shortlist"):
        lines.append("\nNo shortlist: the enumerated space was empty for this case/resource "
                      "allocation.")
        return "\n".join(lines)

    lines.append("\n-- Model: requested=%s, used=%s%s; magnitude companion=%s --" % (
        result["model_requested"], result["model_used"],
        "" if result["model_requested"] == result["model_used"] else " (FELL BACK)",
        result["magnitude_model_used"]))
    lines.append("\n-- Shortlist (top %d of %d legal candidate(s)) --"
                  % (len(result["shortlist"]), result["total_candidates"]))
    header = ("%-4s %-9s %-9s %-16s %-16s %-8s %-8s %-16s %-11s %-20s %-12s %s"
              % ("rank", "ranksxomp", "nprocxXy", "patch_x min-max", "patch_y min-max",
                 "numtiles", "tileshape", "tile cells min-max", "dist/KMPBT", "magnitude 95% CI",
                 "label", "extrapolated"))
    lines.append(header)
    for e in result["shortlist"]:
        patchx = "%s-%s" % (e.get("patch_x"), e.get("patch_x_max"))
        patchy = "%s-%s" % (e.get("patch_y"), e.get("patch_y_max"))
        cells = "%s-%s" % (e.get("cells_per_tile_min"), e.get("cells_per_tile_max"))
        distbt = "%s/%s" % (e["distribution"], "default" if e["kmp_blocktime"] is None
                             else e["kmp_blocktime"])
        iv = e.get("magnitude_interval_95")
        ivstr = "n/a" if iv is None else "[%+.2f%%, %+.2f%%]" % (
            (math.exp(iv[0]) - 1) * 100.0, (math.exp(iv[1]) - 1) * 100.0)
        magval = "%+.2f%%" % (e["magnitude_value"] * 100.0)
        lines.append("%-4d %-9s %-9s %-16s %-16s %-8d %-9s %-16s %-11s %-20s %-12s %s%s" % (
            e["rank"], "%dx%d" % (e["n_ranks"], e["omp_threads"]),
            "%dx%d" % (e["nproc_x"], e["nproc_y"]), patchx, patchy, e["numtiles"],
            e["tile_shape"], cells, distbt, "%s %s" % (magval, ivstr), e["label"],
            "YES: %s" % "; ".join(e["extrapolation_reasons"]) if e["extrapolated"] else "no",
            "  <- reference" if e["is_reference"] else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --propose-batch (WTune D4 hook into D3): a stateless, always-round-1 call into
# decomp_search.propose_for_case, so --case-name need not be registered in campaign/cases.json at
# all (unlike decomp_search.py's own CLI `propose`, which looks the case up there). For a real
# multi-round search, seed decomp_search.py's own state file with this call's output and continue
# with its `propose`/`ingest` CLI directly -- recommend.py itself keeps no persistent state.
# ---------------------------------------------------------------------------
def run_propose_batch(ledger_path, namelist_path, case_name, hardware_id, nodes, max_dom,
                       batch_size):
    """Returns (result_dict_from_propose_core, configs_out) -- see decomp_search.propose_core."""
    case_fields = parse_case_from_namelist(namelist_path)
    effective_max_dom = resolve_max_dom(max_dom, case_fields)
    if case_fields["e_we"] is None or case_fields["e_sn"] is None or case_fields["e_vert"] is None:
        raise ValueError(
            "cannot run --propose-batch: e_we/e_sn/e_vert not found in the namelist's &domains "
            "section -- refusing to guess a grid size")
    case_json = _decomp_space_case_dict(case_fields, effective_max_dom)
    total_cores = nodes * ledgermod.CORES_PER_NODE
    prior_rows = benchmark.load_ledger(ledger_path)
    prior_rows = [r for r in prior_rows if r.get("compliance_class") != "RESEARCH"]
    state = dict(configs={}, measured={}, rounds=[])
    return decompsearch.propose_for_case(
        case_name, case_json, case_fields, prior_rows, state, batch_size,
        total_cores=total_cores, nodes=nodes, hardware_id=hardware_id)


def format_propose_batch_report(case_name, result, configs_out, out_path):
    lines = ["\n-- WTune D3 propose-batch (round %d) --" % result["round"]]
    if result["stopped"]:
        lines.append("  STOP: %s" % result["reason"])
        return "\n".join(lines)
    lines.append("  proposed %d config(s), n_training_rows=%d, ensemble_size=%d"
                  % (len(configs_out), result["n_training_rows"], result["ensemble_size"]))
    lines.append("  wrote CONFIG_FILE JSON: %s" % out_path)
    lines.append("  %s" % decompsearch.sbatch_command(case_name, os.path.abspath(out_path)))
    lines.append("  the human decides whether to run this -- nothing here is auto-submitted")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PHYSICS-GRAY (decision 0051): a separate top-level function, not a branch inside recommend()
# above -- recommend() stays STRICT-only (its own docstring, and every existing STRICT test,
# already say so), and calling recommend(compliance="PHYSICS-GRAY") directly still raises
# UnsupportedComplianceError exactly as before. main() dispatches PHYSICS-GRAY here instead.
# ---------------------------------------------------------------------------
DEFAULT_PHYSICS_ALLOWED_LABELS = ("WITHIN-FLOOR", "SMALL-DRIFT")
_PROBE_MIN_DX_M = 1000.0  # decision 0051: the SpeedyWeather probe is not valid below dx=1 km
_DEFAULT_PHYSICS_DECOMP = "C02"          # campaign/strict_campaign.sbatch's own DECOMP default
_DEFAULT_PHYSICS_TWIN_DECOMP = "C06"     # ...and its own TWIN_DECOMP default
_TWIN_DECOMP_FALLBACK = "C13"            # used only when DECOMP itself resolves to C06


def _probe_validity(dx):
    """(valid, note): valid is True / False / None (unknown); note is a warning string or None.
    The SpeedyWeather probe's own analytic model is not valid below dx=1 km (decision 0051) -- a
    case that fine must never trust its probe verdicts, however clean they look."""
    if dx is None:
        return None, ("dx unknown: cannot confirm the SpeedyWeather probe is valid for this grid "
                       "-- probe verdicts below are used, but their applicability is unconfirmed")
    if dx < _PROBE_MIN_DX_M:
        return False, ("dx=%.0f m < 1000 m: the SpeedyWeather probe is NOT valid below 1 km -- "
                        "every probe verdict for this case is DISREGARDED below (treated as no "
                        "probe data), never trusted" % dx)
    return True, None


def run_probe_screen(namelist_path, latlon=None, wps=None, wrfinput=None, python_exe=None,
                      wtune_path=None, timeout=None):
    """Run `wtune.py screen --namelist NAMELIST --json [--latlon|--wps|--wrfinput]` as a
    subprocess, using the SAME python interpreter this process runs under unless overridden, and
    return its captured stdout (the probe verdict lines plus the JSON evidence manifest -- the
    same text a saved --probe-screen file would contain). Kept as its own thin wrapper so tests
    can monkeypatch it instead of actually invoking the Julia SpeedyWeather probe. Raises
    RuntimeError (with the subprocess's own stderr) on a non-zero exit."""
    python_exe = python_exe or sys.executable
    wtune_path = wtune_path or os.path.join(_REPO_ROOT, "wtune.py")
    cmd = [python_exe, wtune_path, "screen", "--namelist", namelist_path, "--json"]
    if latlon:
        cmd += ["--latlon", latlon]
    elif wps:
        cmd += ["--wps", wps]
    elif wrfinput:
        cmd += ["--wrfinput", wrfinput]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError("wtune.py screen failed (rc=%d): %s"
                            % (proc.returncode, proc.stderr[-4000:]))
    return proc.stdout


def _resolve_probe_text(namelist_path, probe_screen, latlon, wps, wrfinput):
    """(probe_text, source_description). --probe-screen wins outright. Otherwise (the "neither
    given" default, or an explicit --run-probe) actually runs the probe -- and refuses outright,
    rather than silently producing an un-vetted shortlist, when there is no lat/lon source at all
    (decision 0051: physics candidates must always be probe-vetted)."""
    if probe_screen:
        with open(probe_screen) as fh:
            return fh.read(), "--probe-screen %s" % probe_screen
    if not (latlon or wps or wrfinput):
        raise ValueError(
            "cannot run the SpeedyWeather probe for --compliance PHYSICS-GRAY: no "
            "--latlon/--wps/--wrfinput given and no --probe-screen provided -- refusing to "
            "produce an un-vetted physics shortlist (decision 0051: physics candidates must "
            "always be probe-vetted)")
    text = run_probe_screen(namelist_path, latlon=latlon, wps=wps, wrfinput=wrfinput)
    which = ("--latlon %s" % latlon if latlon else
             "--wps %s" % wps if wps else "--wrfinput %s" % wrfinput)
    return text, "auto --run-probe (wtune.py screen --namelist %s %s)" % (namelist_path, which)


def _default_decomp(ledger_path, namelist_path, case_name, hardware_id, nodes, max_dom, holdout_case):
    """(decomp, source): the STRICT top-1 shortlist entry of this same run when one is available,
    falling back to campaign/strict_campaign.sbatch's own DECOMP default otherwise -- never raises
    (a STRICT-side failure here must not block the PHYSICS-GRAY shortlist itself)."""
    try:
        strict_result = recommend(
            ledger_path=ledger_path, namelist_path=namelist_path, case_name=case_name,
            hardware_id=hardware_id, nodes=nodes, compliance="STRICT", model="catboost_pairwise",
            top_k=1, no_intervals=True, max_dom=max_dom, holdout_case=holdout_case)
    except Exception as e:
        return _DEFAULT_PHYSICS_DECOMP, "fallback default (STRICT shortlist unavailable: %s)" % e
    if strict_result.get("shortlist"):
        return strict_result["shortlist"][0]["config_id"], "this run's own STRICT top-1 shortlist entry"
    return _DEFAULT_PHYSICS_DECOMP, "fallback default (STRICT shortlist was empty)"


# Defect 5 (real physics-campaign data, ledger jobs 1077042/1077171 and their predecessors):
# isc21's DECOMP=C02/TWIN_DECOMP=C06 pair was found BIT-IDENTICAL (rms diff 0 on every compared
# field -- fidelity_gate.py's explicit "floor_zero" case) the one time it was tried; every real
# isc21 physics-campaign run since has used TWIN_DECOMP=C13 instead. There is no cheap way to
# predict a bit-identical twin ahead of a real measurement (it depends on WRF's own reduction
# order, not on anything resolve_case()'s dedupe key -- ranks/omp/nx/ny/numtiles/dist/blocktime --
# can see, since DECOMP and TWIN_DECOMP always differ in ranks*omp by construction), so this table
# only records CASES ALREADY OBSERVED to hit it; see the general floor_zero hint printed
# unconditionally below for any case not in this table.
_KNOWN_FLOOR_ZERO_TWIN = {
    ("isc21", _DEFAULT_PHYSICS_DECOMP): _TWIN_DECOMP_FALLBACK,
}


def _default_twin_decomp(decomp, case_name=None):
    """(twin_decomp, note). `note` is non-None only when a known-bad (DECOMP, TWIN_DECOMP) pair
    for this case_name was avoided (see _KNOWN_FLOOR_ZERO_TWIN) -- the general floor_zero hint that
    applies to every case regardless is printed separately by the caller, always, not just here."""
    override = _KNOWN_FLOOR_ZERO_TWIN.get((case_name, decomp))
    if override is not None:
        return override, (
            "case_name=%r at DECOMP=%s: TWIN_DECOMP=%s is a KNOWN bit-identical (floor_zero) pair "
            "for this case (measured campaign data) -- using TWIN_DECOMP=%s instead"
            % (case_name, decomp, _DEFAULT_PHYSICS_TWIN_DECOMP, override))
    twin = _TWIN_DECOMP_FALLBACK if decomp == _DEFAULT_PHYSICS_TWIN_DECOMP else _DEFAULT_PHYSICS_TWIN_DECOMP
    return twin, None


def _native_physics_from_case_fields(case_fields):
    return dict(
        mp_physics=case_fields.get("mp_physics"), cu_physics=case_fields.get("cu_physics"),
        bl_pbl_physics=case_fields.get("bl_pbl_physics"),
        sf_sfclay_physics=case_fields.get("sf_sfclay_physics"),
        sf_surface_physics=case_fields.get("sf_surface_physics"),
        ra_lw_physics=case_fields.get("ra_lw_physics"), ra_sw_physics=case_fields.get("ra_sw_physics"),
        radt=case_fields.get("radt"), time_step=case_fields.get("time_step"),
        physics_suite=case_fields.get("physics_suite"),
    )


# ---------------------------------------------------------------------------
# Cost-share-aware template filtering (2026-09-17)
#
# The ledger has carried profile_{radiation,micro,dynamics,mpi,io}_share since phase B and nothing
# in this file ever read them. A template can only ever save a fraction of the bucket it acts on,
# so the bucket share is a hard UPPER BOUND on its benefit -- and the buckets differ by more than
# an order of magnitude across cases: SC26-P1's radiation is 1.9% of step cost while SC26-P2's is
# 47% (~/SC26-WRF docs/testcase-atlas/cases/sc26-p{1,2}-*.md). Proposing a radt template on P1
# cannot pay, and radt x2 (P02) already bought nothing on six low-radiation-share cases.
#
# What this is NOT: a benefit predictor. The same atlas records radiation share failing to predict
# a compiler-flag gain (P2 at 47% share gained 5.3%, LAFE at 40% gained 11.2%). An upper bound is
# the only honest thing a share supports, so a low share DEMOTES a candidate -- it is still listed,
# with the number and the reason -- and a high share promotes nothing.
# ---------------------------------------------------------------------------
FAMILY_PROFILE_BUCKET = {
    "radiation": "profile_radiation_share",
    "microphysics": "profile_micro_share",
    "dynamics": "profile_dynamics_share",
    # A nest's cost lands in the dynamics core, and there is no profile_pbl_share COLUMN in the
    # ledger (see predictor/ledger.py LEDGER_FIELDS) -- pbl_surface therefore shares the dynamics
    # bucket rather than inventing one. Change this the day a pbl bucket is actually measured.
    "nesting": "profile_dynamics_share",
    "pbl_surface": "profile_dynamics_share",
    # A diagnostics/cadence template (D04: debug_level, history cadence) does not act on a physics
    # bucket at all, so no share can bound it and this rule must never demote one.
    "diagnostics": None,
    "none": None,
}
PROFILE_BUCKET_FIELDS = ("profile_radiation_share", "profile_micro_share",
                         "profile_dynamics_share", "profile_mpi_share", "profile_io_share")
DEFAULT_MIN_BUCKET_SHARE = 0.05


def _as_share_fraction(value):
    """A profile share as a FRACTION of step cost. The ledger column has no writer in this repo
    yet, so both conventions are accepted and normalised here: <= 1 is already a fraction, > 1 is
    read as a percent (the atlas notes quote 1.9% / 47%) and divided by 100. None passes through."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0:            # NaN or nonsense
        return None
    return v / 100.0 if v > 1.0 else v


def _median(values):
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def case_profile_shares(rows, case_name):
    """{profile field: fraction of step cost} measured for `case_name`, the median over every
    ledger row of that case_base carrying the field. {} when this case has no profile row at all --
    which is the current state of the whole production ledger, and the reason the filter below is a
    no-op until a profiled run lands."""
    out = {}
    case_rows = [r for r in rows if r.get("case_base") == case_name]
    for field in PROFILE_BUCKET_FIELDS:
        vals = [f for f in (_as_share_fraction(r.get(field)) for r in case_rows) if f is not None]
        if vals:
            out[field] = _median(vals)
    return out


def template_bucket(family):
    """The profile bucket a template family acts on, or None when no bucket bounds it."""
    return FAMILY_PROFILE_BUCKET.get(family)


def annotate_bucket_shares(shortlist, shares, min_bucket_share=DEFAULT_MIN_BUCKET_SHARE):
    """(shortlist, notes). With measured `shares`, annotate every shortlist entry with
    `bucket` / `bucket_share` / `upper_bound_saving` (= bucket_share: a template cannot save more
    than the bucket it acts on) / `bucket_demoted`, force a below-threshold candidate to
    UNCONFIRMED with an explicit reason, and re-sort demoted candidates last so the verify plan
    does not spend runs on them. Candidates are never hidden.

    With no measured share for this case (`shares` empty) NOTHING changes: the same list object
    comes back, unannotated, plus a note saying so."""
    threshold = _as_share_fraction(min_bucket_share) or 0.0
    if not shares:
        return shortlist, [
            "cost-share filter: this case has no profile row in the ledger "
            "(profile_{radiation,micro,dynamics,mpi,io}_share all absent) -- no candidate was "
            "annotated or demoted. Profile one run of this case to enable it."]
    notes = ["cost-share filter: measured profile shares for this case: %s; a template whose "
             "bucket is below --min-bucket-share=%.1f%% is DEMOTED (still listed) because its "
             "saving cannot exceed that bucket"
             % (", ".join("%s=%.1f%%" % (k.replace("profile_", "").replace("_share", ""), 100 * v)
                          for k, v in sorted(shares.items())), 100 * threshold)]
    for entry in shortlist:
        family = entry.get("family")
        bucket = template_bucket(family)
        entry["bucket"] = bucket
        share = shares.get(bucket) if bucket else None
        entry["bucket_share"] = share
        entry["upper_bound_saving"] = share
        entry["bucket_demoted"] = False
        if bucket is None:
            entry.setdefault("reasons", []).append(
                "no profile bucket bounds the %s family: cost share says nothing about this "
                "template either way" % (family or "unknown",))
            continue
        if share is None:
            entry.setdefault("reasons", []).append(
                "this case has no measured %s: cost share cannot bound this template" % bucket)
            continue
        if share < threshold:
            entry["bucket_demoted"] = True
            entry["status"] = "UNCONFIRMED"
            entry.setdefault("reasons", []).append(
                "DEMOTED on cost share: the %s family acts on %s=%.1f%% of step cost, below "
                "--min-bucket-share=%.1f%% -- even a perfect template cannot save more than "
                "%.1f%% here (measured contrast: SC26-P1 radiation 1.9%% vs SC26-P2 47%%)"
                % (family, bucket, 100 * share, 100 * threshold, 100 * share))
    demoted = [e["template"] for e in shortlist if e.get("bucket_demoted")]
    if demoted:
        notes.append("cost-share DEMOTED (ranked last, not hidden): %s" % ", ".join(demoted))
    shortlist = sorted(shortlist, key=lambda d: (
        1 if d.get("bucket_demoted") else 0,
        0 if d.get("status") == "RECOMMENDABLE" else 1,
        d["predicted_y"] if d.get("predicted_y") is not None else float("inf")))
    return shortlist, notes


def exclude_frozen_key_templates(candidates, frozen_keys, case_name, templates=None):
    """(candidates, notes): mark every candidate whose template would write a task-mandated frozen
    key inapplicable, with the reason naming the key and the case. A competition problem statement
    can FIX a parameter (SC26-P1 mandates time_step=10 and debug_level=50); a recommendation that
    changes one is a rules violation, not a tuning trade-off, so it is excluded rather than ranked
    low. Checks the template's declared edits AND the changes actually resolved for this case, so a
    generic per-case pairing rule cannot smuggle a frozen key back in."""
    frozen = [str(k).strip().lower() for k in frozen_keys or ()]
    if not frozen:
        return candidates, []
    templates = campaign_configs.PHYSICS_TEMPLATES if templates is None else templates
    frozen_set = set(frozen)
    notes = ["task-mandated frozen key(s) for case %s: %s -- any template editing one is EXCLUDED "
             "(changing a mandated parameter is a rules violation, not a tuning trade-off)"
             % (case_name, ", ".join(frozen))]
    excluded = []
    for cand in candidates:
        spec = templates.get(cand["template"]) or {}
        hit = campaign_configs.template_frozen_key_conflicts(spec, frozen, case_id=case_name)
        for key in cand.get("changes") or {}:
            if key.lower() in frozen_set and key.lower() not in hit:
                hit.append(key.lower())
        if not hit:
            continue
        cand["frozen_key_conflict"] = hit
        excluded.append(cand["template"])
        if cand.get("applicable", True):
            cand["applicable"] = False
            cand["skip_reason"] = (
                "edits task-mandated frozen key(s) %s for case %s: the task FIXES their value, so "
                "changing one is a rules violation, not a tuning trade-off"
                % (", ".join(hit), case_name))
    if excluded:
        notes.append("excluded on frozen keys: %s" % ", ".join(sorted(set(excluded))))
    return candidates, notes


# ---------------------------------------------------------------------------
# Accuracy-first mode (--objective accuracy, 2026-09-17)
#
# The PHYSICS-GRAY shortlist above answers "which physics template is FASTEST while staying within
# an allowed fidelity label". Accuracy-first answers the opposite question -- "which is CLOSEST to
# the reference solution, and what does that cost" -- over the same candidate set plus the
# physics_templates.json A-series, whose direction is "accuracy" (more work, less discretization
# error) rather than the P/D series' "cost".
#
# Three facts shape this code, and all three are easy to get wrong:
#
# 1. THE CHAOS FLOOR IS THE RESOLUTION LIMIT. fidelity_gate.py measures a candidate's drift from
#    the reference against the drift the SAME physics shows when merely re-decomposed (the twin).
#    floor_ratio <= 1 means the candidate moved the solution no more than changing nproc_x/nproc_y
#    does. Reporting such a candidate as an accuracy improvement is meaningless -- there is
#    nothing there to improve. Measured example from this campaign: pure RK2 (D02) on conus12 at
#    12 h had floor_ratio 1.593, i.e. dropping a whole RK stage was already within 2x of simply
#    re-decomposing the run. CHAOS_FLOOR_RATIO below is that hard rule, and it applies in BOTH
#    directions: a cost lever below the floor is a free speedup, an accuracy lever below the floor
#    is pure wasted work because the reference was already converged in that knob.
#
# 2. THE DECOMPOSITION IS A FREE AXIS. STRICT knobs (nproc_x/nproc_y/numtiles/OMP/distribution)
#    change only how the same arithmetic is distributed, never the physics. They set the chaos
#    floor but they are not a fidelity lever, so accuracy mode keeps recommending the FASTEST
#    legal decomposition exactly as perf mode does -- there is no accuracy to buy by slowing it
#    down.
#
# 3. nd IS SIGNED BY DIRECTION, NOT BY MAGNITUDE. nd is the distance from the campaign REFERENCE
#    (P00, the case as it actually runs). For a cost lever that distance IS the error it
#    introduced, so lower is better. For an A-series lever the same number is the CORRECTION the
#    extra work applies, and small means "the reference was already converged here", i.e. the
#    template is not worth its cost. Ranking A-templates by ascending nd alongside the cost levers
#    would perversely reward exactly the ones that do nothing, so they are split into their own
#    block. See _accuracy_verify_plan: the way out is to re-run fidelity_gate.py with the surviving
#    A-template's own wrfout as --reference, which turns every other candidate's nd into a genuine
#    error against a better-converged solution. That two-stage plan is what accuracy mode emits.
# ---------------------------------------------------------------------------
OBJECTIVES = ("perf", "accuracy")
CHAOS_FLOOR_RATIO = 1.0
ACCURACY_DIRECTION = "accuracy"
COST_DIRECTION = "cost"
# Which judge produced an entry's fidelity_error. OBSERVED comes from predictor/verify_obs.py
# (distance from REALITY, lower = more correct); MODEL is nd against a reference RUN (lower =
# more like the reference). Never rank the two in one list -- see _accuracy_entry.
OBSERVED_ERROR = "observation"
MODEL_ERROR = "model-vs-model"
BELOW_FLOOR_NOTE = ("indistinguishable from changing the decomposition (floor_ratio %.3f <= %.2f): "
                    "its drift is at or below the chaos floor, so this is NOT an accuracy "
                    "improvement")


def template_direction(template_id, templates=None):
    """'accuracy' for a physics_templates.json template that buys fidelity with extra work
    (the A-series), else 'cost'. An unknown template id is 'cost': the P/D series predate the
    'direction' field entirely and every one of them is a cost lever."""
    templates = campaign_configs.PHYSICS_TEMPLATES if templates is None else templates
    tpl = templates.get(template_id) or {}
    return ACCURACY_DIRECTION if tpl.get("direction") == ACCURACY_DIRECTION else COST_DIRECTION


def _row_fidelity_value(row, key):
    """Numeric fidelity measurement `key` for one ledger row, from the parsed column when the row
    came through ledger.adapt_campaign_ledger (which now writes fidelity_nd_max /
    fidelity_floor_ratio_max as row fields) and from the raw `notes` tokens otherwise --
    benchmark.load_ledger, which the --physics-ledger path uses, does no notes parsing at all."""
    v = row.get(key)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    notes = row.get("_phys_notes")
    if not isinstance(notes, dict):
        notes = physicsmod.parse_notes(row.get("notes"))
    raw = notes.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def measured_fidelity_by_template(train_rows):
    """{template: {"nd": entry, "floor_ratio": entry}} where entry is
    {"median", "n", "cases"} over the MEASURED physics rows for that template -- a model-free
    track record in the same spirit as physics.template_large_drift_rate, not a prediction.
    floor_ratio has no regression model at all (nothing predicts a chaos floor for a case that has
    not been run), so this is the only source accuracy mode has for the below-floor rule."""
    buckets = defaultdict(lambda: {"nd": [], "floor_ratio": []})
    cases = defaultdict(lambda: {"nd": set(), "floor_ratio": set()})
    for row in physicsmod.physics_rows(train_rows):
        template = physicsmod._template_of(row)
        if not template:
            continue
        for key, note_key in (("nd", "fidelity_nd_max"),
                               ("floor_ratio", "fidelity_floor_ratio_max")):
            value = _row_fidelity_value(row, note_key)
            if value is None:
                continue
            buckets[template][key].append(value)
            if row.get("case_base"):
                cases[template][key].add(row["case_base"])
    out = {}
    for template, per_key in buckets.items():
        out[template] = {
            key: dict(median=(statistics.median(values) if values else None), n=len(values),
                      cases=sorted(cases[template][key]))
            for key, values in per_key.items()}
    return out


def _parse_obs_wrfout(specs):
    """--obs-wrfout TEMPLATE=GLOB [...] -> {template: glob}. None/empty -> None."""
    if not specs:
        return None
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--obs-wrfout wants TEMPLATE=GLOB (e.g. P01='run_P01/wrfout_d01_*'),"
                             " got %r" % (spec,))
        template, glob_pattern = spec.split("=", 1)
        out[template.strip()] = glob_pattern
    return out


def case_observations(case_name, cases=None):
    """The case's declared `observations` block from cases.json, or None when it declares none.

    None is the normal state: only the two SC26 practice problems have real observation ground
    truth, and every other case is judged model-vs-model exactly as before. A MALFORMED block
    raises (verify_obs.observations_spec validates it) rather than being read as "no
    observations" -- a typo'd path must not quietly downgrade the judge."""
    cases = campaign_configs.CASES if cases is None else cases
    entry = cases.get(case_name)
    return None if not entry else verify_obs.observations_spec(entry)


def observation_scores_for_case(case_name, wrfout_by_template, cases=None):
    """{template_id: (error, source)} scored against the case's REAL observations.

    `wrfout_by_template` is {template_id: wrfout path or glob(s)} for the candidates that have
    ACTUALLY BEEN RUN. It is an input, not something this function goes and produces: there is
    no observation error for a template whose run does not exist, and inventing one from a
    model would defeat the point of scoring against observations at all.

    Returns {} when the case declares no observations or nothing has been run, so the caller
    falls back to model-vs-model fidelity with no extra branching."""
    spec = case_observations(case_name, cases)
    if spec is None or not wrfout_by_template:
        return {}
    scores = {}
    for template, wrfout in wrfout_by_template.items():
        result = verify_obs.verify(spec, wrfout)
        scores[template] = (verify_obs.accuracy_error(result),
                            verify_obs.accuracy_error_source(result))
    return scores


def _accuracy_entry(cand, measured, templates=None, obs_scores=None):
    """One accuracy-mode row for one shortlist candidate. fidelity_error prefers, in order:

      1. a MEASURED ERROR AGAINST REAL OBSERVATIONS for this template on this case
         (obs_scores, from predictor/verify_obs.py) -- only the SC26 practice cases have
         observation ground truth, but where it exists it outranks everything below,
         because it answers the question the other two only stand in for;
      2. the per-case REGRESSION prediction (predicted_fidelity_nd) -- model vs model, but
         at least about THIS case;
      3. the template's measured median nd across training cases;
      4. None when none of those exists.

    `fidelity_error_kind` records which, because 1 and 2/3 are not commensurable: nd is a
    drift away from a REFERENCE RUN (lower = more like the reference), an observation error
    is a distance from REALITY (lower = more correct). Ranking them in one list would be a
    category error, so build_accuracy_view keeps them apart.

    The chaos floor is NOT applied to an observation-scored entry: floor_ratio is a
    model-vs-model concept (how far a same-physics twin drifts by decomposition alone), and
    a real skill improvement that happens to be small has nothing to do with it. Marking
    such an entry below_floor would silently drop a genuine accuracy gain off the Pareto
    front (see pareto_front's exclusion rule)."""
    template = cand["template"]
    record = measured.get(template) or {}
    nd_rec, floor_rec = record.get("nd") or {}, record.get("floor_ratio") or {}
    observed = (obs_scores or {}).get(template)

    error, error_source, error_kind = None, None, None
    if observed is not None:
        error, error_source = observed[0], observed[1]
        error_kind = OBSERVED_ERROR
    elif cand.get("predicted_fidelity_nd") is not None:
        error, error_kind = cand["predicted_fidelity_nd"], MODEL_ERROR
        error_source = "predicted (fidelity regression, this case)"
    elif nd_rec.get("median") is not None:
        error, error_kind = nd_rec["median"], MODEL_ERROR
        error_source = ("measured median of %d row(s) over %d training case(s): %s"
                        % (nd_rec["n"], len(nd_rec["cases"]), ", ".join(nd_rec["cases"]) or "-"))

    floor_ratio, floor_source = floor_rec.get("median"), None
    if floor_ratio is not None:
        floor_source = ("measured median of %d row(s) over %d training case(s): %s"
                        % (floor_rec["n"], len(floor_rec["cases"]),
                           ", ".join(floor_rec["cases"]) or "-"))
    below_floor = (error_kind != OBSERVED_ERROR
                   and floor_ratio is not None and floor_ratio <= CHAOS_FLOOR_RATIO)
    return dict(
        template=template, family=cand.get("family"),
        direction=template_direction(template, templates),
        status=cand.get("status"),
        predicted_time_to_solution_pct=cand.get("predicted_time_to_solution_pct"),
        time_to_solution_interval_pct=cand.get("time_to_solution_interval_pct"),
        predicted_fidelity_label=cand.get("predicted_fidelity_label"),
        fidelity_error=error, fidelity_error_source=error_source,
        fidelity_error_kind=error_kind,
        floor_ratio=floor_ratio, floor_ratio_source=floor_source,
        below_floor=below_floor,
        below_floor_note=(BELOW_FLOOR_NOTE % (floor_ratio, CHAOS_FLOOR_RATIO)) if below_floor else None,
        on_pareto=False, pareto_excluded_reason=None)


def pareto_front(entries):
    """Mark the (predicted cost, fidelity error) Pareto front in place and return it. Both
    coordinates are minimised: cost is predicted_time_to_solution_pct (negative = faster), error
    is fidelity_error (lower = closer to the reference). perf-first and acc-first are the two ends
    of this front, so a user can pick any operating point between them instead of being handed one
    answer. An entry is on the front when no OTHER entry is at least as good on both coordinates
    and strictly better on one.

    Excluded, each with a recorded reason rather than silently: an entry missing either
    coordinate (nothing to place it with) and a below-the-chaos-floor entry (rule 1 above -- its
    "error" is numerical noise, so letting it anchor the accuracy end of the front would hand the
    user an operating point that does not exist)."""
    eligible = []
    for e in entries:
        if e["below_floor"]:
            e["pareto_excluded_reason"] = "below the chaos floor"
        elif e["predicted_time_to_solution_pct"] is None:
            e["pareto_excluded_reason"] = "no predicted time-to-solution"
        elif e["fidelity_error"] is None:
            e["pareto_excluded_reason"] = "no fidelity error (predicted or measured)"
        else:
            eligible.append(e)
    front = []
    for e in eligible:
        cost, err = e["predicted_time_to_solution_pct"], e["fidelity_error"]
        dominated = any(
            o is not e and o["predicted_time_to_solution_pct"] <= cost and o["fidelity_error"] <= err
            and (o["predicted_time_to_solution_pct"] < cost or o["fidelity_error"] < err)
            for o in eligible)
        e["on_pareto"] = not dominated
        if not dominated:
            front.append(e)
    front.sort(key=lambda e: (e["predicted_time_to_solution_pct"], e["fidelity_error"]))
    return front


def build_accuracy_view(shortlist, train_rows, templates=None, obs_scores=None,
                        observations=None):
    """Accuracy-first view over an existing perf-mode shortlist: the same candidates, re-ranked by
    fidelity error instead of predicted speed, split by template direction, with the chaos-floor
    rule applied and the (cost, error) Pareto front marked. Never mutates `shortlist`.

    `obs_scores` is {template_id: (error, source)} measured against REAL OBSERVATIONS by
    predictor/verify_obs.py -- see observation_scores_for_case(). Entries scored that way are
    ranked in their own block, ALWAYS ascending (a smaller distance from reality is better,
    whatever the template's direction), because a template that trades work for fidelity is
    judged by whether the forecast got closer to the observations, not by how far it moved the
    solution. `observations` is the case's declared observations block, carried into the view so
    a report can say which file the numbers were scored against. Both default to None, which is
    exactly the pre-existing model-vs-model behaviour -- a case that declares no observations
    falls back to it untouched."""
    measured = measured_fidelity_by_template(train_rows)
    entries = [_accuracy_entry(c, measured, templates, obs_scores) for c in shortlist]

    # An UNKNOWN error sorts last in BOTH directions -- never first. A plain reverse=True on a
    # float("inf") sentinel would put every un-measured template at the top of the accuracy-lever
    # block, i.e. rank the candidates we know nothing about above the ones with a measured
    # correction. The leading `is None` term keeps "no evidence" at the bottom either way.
    def _error_key_asc(e):
        v = e["fidelity_error"]
        return (v is None, v if v is not None else 0.0)

    def _error_key_desc(e):
        v = e["fidelity_error"]
        return (v is None, -v if v is not None else 0.0)

    # Observation-scored entries leave the nd blocks entirely: their error is a distance from
    # reality, not a drift from a reference run, and the descending rule below would invert it.
    observed = [e for e in entries if e["fidelity_error_kind"] == OBSERVED_ERROR]
    modelled = [e for e in entries if e["fidelity_error_kind"] != OBSERVED_ERROR]
    ranked_observed = sorted(observed, key=_error_key_asc)

    cost_levers = [e for e in modelled if e["direction"] == COST_DIRECTION]
    accuracy_levers = [e for e in modelled if e["direction"] == ACCURACY_DIRECTION]
    # Cost levers: ascending error -- least departure from the reference first.
    ranked = sorted([e for e in cost_levers if not e["below_floor"]], key=_error_key_asc)
    # Accuracy levers: DESCENDING error -- the largest measurable correction is the one most worth
    # having, and a small one means the reference was already converged (rule 3 above).
    ranked_accuracy = sorted([e for e in accuracy_levers if not e["below_floor"]],
                             key=_error_key_desc)
    for block in (ranked, ranked_accuracy, ranked_observed):
        for i, e in enumerate(block, start=1):
            e["accuracy_rank"] = i
    for e in entries:
        e.setdefault("accuracy_rank", None)
    front = pareto_front(entries)
    return dict(
        entries=entries, ranked_cost_levers=ranked, ranked_accuracy_levers=ranked_accuracy,
        ranked_observed=ranked_observed,
        below_floor=[e for e in entries if e["below_floor"]], pareto_front=front,
        chaos_floor_ratio=CHAOS_FLOOR_RATIO,
        observations=observations,
        judged_against=(OBSERVED_ERROR if observed else MODEL_ERROR))


def _accuracy_verify_plan(case_name, accuracy_view, candidates, decomp, twin_decomp, perf_plan):
    """The two-stage campaign accuracy mode needs, which is NOT the perf plan with a different
    sort order (see rule 3 above): stage 1 measures every applicable A-series template against
    P00 to find out whether P00 is converged at all, and stage 2 re-runs fidelity_gate.py with the
    surviving A-template's own wrfout as --reference, so every cost lever's nd becomes an error
    against a better-converged solution instead of a distance from an arbitrary operating point."""
    applicable = {c["template"] for c in candidates if c.get("applicable", True)}
    # Sorted, not in shortlist order: this string goes into a PCONFIGS= sbatch line a human reads
    # and re-runs, so it must be stable across runs rather than following whatever order the
    # perf ranking happened to produce.
    accuracy_templates = sorted(e["template"] for e in accuracy_view["entries"]
                                if e["direction"] == ACCURACY_DIRECTION
                                and e["template"] in applicable)
    if not accuracy_templates:
        return dict(stage1=None, stage2=None, accuracy_templates=[],
                    note=("no applicable accuracy-direction (A-series) template for this case -- "
                          "accuracy mode can only re-rank the existing cost levers by their drift "
                          "from P00; there is nothing here that can establish a better-converged "
                          "reference"))
    stage1 = (
        "sbatch --export=ALL,CASE=%s,MODE=physics,PCONFIGS=%s,DECOMP=%s,TWIN_DECOMP=%s "
        "campaign/strict_campaign.sbatch" % (case_name, "+".join(accuracy_templates), decomp,
                                             twin_decomp))
    stage2 = [
        "python3 fidelity_gate.py --reference <A-TEMPLATE wrfout_d01_*> --twin <TWIN wrfout_d01_*> "
        "--candidate <P00 REFERENCE wrfout_d01_*> --family all --out-prefix "
        "<out_dir>/fidelity_%s_P00_vs_converged" % case_name,
    ]
    if perf_plan:
        stage2.append(
            "then re-run each cost lever's gate the same way, swapping --reference for the "
            "A-template's wrfout: %s" % "; ".join(perf_plan["pconfigs"]))
    return dict(
        stage1=stage1, stage2=stage2, accuracy_templates=accuracy_templates,
        note=("STAGE 1 -- run the A-series at DECOMP alongside the usual P00 reference and "
              "TWIN_DECOMP twin, and gate each A-template against P00. An A-template whose "
              "floor_ratio <= %.2f means P00 was ALREADY converged in that knob: the extra work "
              "bought nothing measurable and that template is pure added cost, never an accuracy "
              "improvement. STAGE 2 -- for whichever A-template survives with the LARGEST drift "
              "(the biggest real correction), re-run fidelity_gate.py using its own wrfout as "
              "--reference. Only then is every other candidate's nd an error against a "
              "better-converged solution rather than a distance from P00, which is what the "
              "accuracy ranking above actually wants to measure. Until stage 2 exists, treat the "
              "cost-lever ranking as 'least departure from the current operating point', not as "
              "'most accurate'. The user makes the final decision -- this tool only narrows the "
              "search." % CHAOS_FLOOR_RATIO))


def recommend_physics_gray(ledger_path, namelist_path, case_name, hardware_id, nodes=4, top_k=5,
                            max_dom=None, holdout_case=None, decomp=None, physics_ledgers=(),
                            allowed_labels=DEFAULT_PHYSICS_ALLOWED_LABELS, probe_screen=None,
                            run_probe=True, latlon=None, wps=None, wrfinput=None,
                            max_infeasible_rate=0.0, max_dangerous_miss=0.0,
                            max_template_large_rate=physicsmod.DEFAULT_MAX_TEMPLATE_LARGE_RATE,
                            min_speedup_pct=physicsmod.DEFAULT_MIN_SPEEDUP_PCT,
                            objective="perf", min_bucket_share=DEFAULT_MIN_BUCKET_SHARE,
                            profile_shares=None, frozen_keys=None, obs_wrfout=None):
    """Build a PHYSICS-GRAY physics-template shortlist for one case (decision 0051). Same
    "narrows the search, human decides" contract as recommend(): NEVER auto-applies anything.
    Every applicable candidate template is always listed with its changes and SpeedyWeather probe
    verdict (decision 0051: never leave a physics candidate un-vetted); it is only ranked/labelled
    RECOMMENDABLE by physics.predict_physics_shortlist when --physics-ledger training data
    actually supports that -- none exists in production yet, so every candidate is UNCONFIRMED
    with reason "no PHYSICS-GRAY campaign data" until the physics campaign lands.

    objective="perf" (the default) is exactly the behaviour above and returns EXACTLY the keys it
    always has -- no accuracy key is added, so an existing --json consumer sees an unchanged
    document. objective="accuracy" ADDS "objective" and "accuracy_view" (build_accuracy_view) and
    leaves every perf key untouched, so the two modes are strictly additive, not alternative.

    min_bucket_share / profile_shares: the cost-share filter (annotate_bucket_shares). Shares are
    read from this case's own ledger rows unless `profile_shares` is passed explicitly; with no
    profile row for the case nothing is annotated and nothing is demoted.
    frozen_keys: task-mandated namelist keys no template may edit; defaults to this case's
    cases.json 'frozen_keys' entry (empty for every registered case today)."""
    if objective not in OBJECTIVES:
        raise ValueError("unknown objective %r (choose from %s)"
                         % (objective, ", ".join(OBJECTIVES)))
    if isinstance(allowed_labels, str):
        allowed_labels = tuple(s.strip() for s in allowed_labels.split(",") if s.strip())

    case_fields = parse_case_from_namelist(namelist_path)
    effective_max_dom = resolve_max_dom(max_dom, case_fields)
    warnings = []

    probe_text, probe_source = _resolve_probe_text(namelist_path, probe_screen, latlon, wps, wrfinput)
    raw_verdicts = campaign_configs.parse_probe_verdicts(probe_text)
    dx = case_fields.get("dx")
    probe_valid, probe_note = _probe_validity(dx)
    if probe_note:
        warnings.append(probe_note)
    probe_verdicts_used = {} if probe_valid is False else raw_verdicts
    warnings.append(
        "probe: %s; dx=%s; %d knob verdict(s) parsed%s"
        % (probe_source, dx, len(raw_verdicts),
           " (DISREGARDED below: dx < 1000 m)" if probe_valid is False else ""))

    with open(namelist_path) as fh:
        namelist_text = fh.read()
    native_physics = _native_physics_from_case_fields(case_fields)
    native_resolved = physicsmod.resolve_physics_suite(native_physics, native_physics.get("physics_suite"))
    candidates = physicsmod.physics_candidates_from_namelist(
        namelist_text, case_name, effective_max_dom, campaign_configs.PHYSICS_TEMPLATES_JSON_PATH,
        native_physics_resolved=native_resolved)
    # P00 is the reference (no edits) -- never a "candidate to recommend switching to"; the same
    # skip-P00 convention configs.py's own cmd_physics_list/cmd_physics_order already use at their
    # call sites (physics_candidates_from_namelist itself stays general-purpose and still reports
    # P00, same as physics_candidates_for_case always has).
    candidates = [c for c in candidates if c["template"] != "P00"]
    # Task-mandated frozen keys come BEFORE anything is ranked: a template that would violate the
    # problem statement is not a low-ranked option, it is not an option.
    if frozen_keys is None:
        frozen_keys = campaign_configs.case_frozen_keys(campaign_configs.CASES.get(case_name))
    candidates, frozen_notes = exclude_frozen_key_templates(candidates, frozen_keys, case_name)
    warnings.extend(frozen_notes)
    # A per-candidate "generic_rule_used" flag fires for MOST templates on MOST cases (P01/P02/P03
    # never have a physics_templates.json per-case entry at all, and P04/P06 usually don't need
    # one either) -- that alone does not mean this case is unknown to the campaign, so the
    # case-level warning checks cases.json membership directly instead of inferring it from any
    # one candidate's flag (which would wrongly flag lafe/isc21 just because, say, P01 or P04
    # happened to take the generic no-op path for them).
    if case_name not in campaign_configs.CASES:
        warnings.append(
            "this case is not one of the six PHYSICS-GRAY campaign cases physics_templates.json "
            "already knows about -- any template needing per-case pairing (P04/P05/P06) used the "
            "generic rule (physics.generic_extra_edits) instead of a hand-written entry")

    warnings.append("allowed fidelity labels: %s" % (", ".join(allowed_labels),))
    warnings.append(
        "fidelity labels describe the MODEL's OWN drift vs a same-physics twin at a different "
        "STRICT decomposition (the chaos floor), not observed forecast skill against any "
        "observation/reanalysis -- see decision 0051")
    # ... unless this case has REAL observations, in which case saying "not observed forecast
    # skill" next to numbers that ARE observed forecast skill would flatly contradict them.
    _obs_spec = case_observations(case_name)
    if _obs_spec is not None:
        warnings.append(
            "this case DOES have observation ground truth (%s, %s): with --objective accuracy "
            "and wrfout for a candidate, predictor/verify_obs.py scores it against that file and "
            "the score appears in accuracy_view['ranked_observed'], where lower is better and the "
            "chaos floor does not apply. The fidelity-label caveat above still governs every "
            "candidate that has NOT been scored that way. A FASTER config can legitimately be "
            "MORE accurate here -- the problem statement's own settings were never optimised."
            % (_obs_spec["kind"], _obs_spec["path"]))

    train_rows = []
    for p in (physics_ledgers or []):
        train_rows.extend(benchmark.load_ledger(p))

    # Defect 1 (real physics-campaign smoke, job 1077893): --holdout-case was accepted but never
    # actually applied to --physics-ledger rows -- a holdout run for a case already IN the given
    # ledger(s) (e.g. lafe, isc21) silently trained on that case's own rows and reported an
    # in-sample shortlist as if it were honest out-of-sample evidence. Apply it here, the same way
    # recommend()'s STRICT path already does for --ledger.
    if holdout_case:
        n_before = len(train_rows)
        train_rows = [r for r in train_rows if r.get("case_base") != holdout_case]
        n_dropped = n_before - len(train_rows)
        if n_dropped:
            warnings.append(
                "--holdout-case=%r: dropped %d PHYSICS-GRAY training row(s) with that case_base "
                "before fitting (honest out-of-sample evaluation)" % (holdout_case, n_dropped))
        else:
            warnings.append(
                "--holdout-case=%r matched ZERO PHYSICS-GRAY training rows -- check the spelling "
                "against the ledger's case_base values; nothing was actually held out"
                % holdout_case)

    physicsmod.attach_change_context(train_rows)
    physicsmod.add_perf_targets(train_rows)
    physicsmod.add_fidelity_targets(train_rows)

    all_cases_present = sorted({r.get("case_base") for r in physicsmod.physics_rows(train_rows)
                                 if r.get("case_base") is not None})
    if case_name in all_cases_present:
        warnings.append(
            "IN-SAMPLE: case_base=%r is still present in the PHYSICS-GRAY training rows used to "
            "fit this model; the shortlist is not an out-of-sample prediction%s"
            % (case_name, "" if holdout_case == case_name else
               " -- pass --holdout-case %r to see an honest out-of-sample shortlist" % case_name))

    if not physicsmod.physics_rows(train_rows):
        warnings.append(
            "no PHYSICS-GRAY campaign data (%s): every candidate below is UNCONFIRMED"
            % ("--physics-ledger not given" if not physics_ledgers
               else "the given --physics-ledger file(s) have zero mode=physics rows (after "
                    "--holdout-case, if any)"))
    else:
        same_hw = [r for r in physicsmod.physics_rows(train_rows) if r.get("hardware_id") == hardware_id]
        cases_hw = sorted({r.get("case_base") for r in same_hw if r.get("case_base") is not None})
        warnings.append(
            "PHYSICS-GRAY training evidence on hardware_id=%r: %d row(s), %d case(s) (%s)"
            % (hardware_id, len(same_hw), len(cases_hw), ", ".join(cases_hw) or "none"))
        if not same_hw:
            warnings.append(
                "zero PHYSICS-GRAY training rows share hardware_id=%r: every candidate below is "
                "UNCONFIRMED regardless of predicted performance/fidelity" % hardware_id)

    case_features_row = dict(
        hardware_id=hardware_id, case_base=None,
        e_we=case_fields["e_we"], e_sn=case_fields["e_sn"], e_vert=case_fields["e_vert"],
        dx=case_fields["dx"], dy=case_fields["dy"], time_step=case_fields["time_step"],
        mp_physics=native_resolved.get("mp_physics"), cu_physics=case_fields.get("cu_physics"),
        bl_pbl_physics=native_resolved.get("bl_pbl_physics"),
        sf_sfclay_physics=native_resolved.get("sf_sfclay_physics"),
        sf_surface_physics=native_resolved.get("sf_surface_physics"),
        ra_lw_physics=native_resolved.get("ra_lw_physics"), ra_sw_physics=native_resolved.get("ra_sw_physics"),
        radt=case_fields.get("radt"), cudt=case_fields.get("cudt"),
    )
    shortlist = physicsmod.predict_physics_shortlist(
        train_rows, case_features_row, candidates, allowed_labels=allowed_labels,
        probe_verdicts=probe_verdicts_used, max_infeasible_rate=max_infeasible_rate,
        max_dangerous_miss=max_dangerous_miss, max_template_large_rate=max_template_large_rate,
        min_speedup_pct=min_speedup_pct)

    # Defect 4: the time-to-solution interval's "not feasible" reason (if any) is the SAME string
    # on every shortlist row (performance_conformal_setup runs once) -- surface it in warnings a
    # single time instead of repeating it per row.
    tts_note = next((c["time_to_solution_interval_note"] for c in shortlist
                      if c.get("time_to_solution_interval_note")), None)
    if tts_note:
        warnings.append(tts_note)

    # Cost-share filter: applied BEFORE top3/verify_plan, so a demoted template does not consume a
    # 4-node verification slot it cannot pay back (see annotate_bucket_shares).
    if profile_shares is None:
        profile_rows = list(train_rows)
        try:
            profile_rows += benchmark.load_ledger(ledger_path)
        except Exception:
            # An unreadable STRICT ledger must never break the PHYSICS-GRAY shortlist -- the shares
            # are an optional annotation, and train_rows alone may already carry them.
            pass
        profile_shares = case_profile_shares(profile_rows, case_name)
    shortlist, share_notes = annotate_bucket_shares(shortlist, profile_shares, min_bucket_share)
    warnings.extend(share_notes)

    decomp_source = "--decomp"
    if decomp is None:
        decomp, decomp_source = _default_decomp(
            ledger_path, namelist_path, case_name, hardware_id, nodes, effective_max_dom, holdout_case)
    twin_decomp, twin_note = _default_twin_decomp(decomp, case_name)
    if twin_note:
        warnings.append(twin_note)
    warnings.append(
        "floor_zero hint: if DECOMP=%s and TWIN_DECOMP=%s happen to produce BIT-IDENTICAL output "
        "(rms diff 0 on every compared field), the twin's chaos floor is floor_zero -- a real "
        "degenerate case fidelity_gate.py handles explicitly, but a candidate's drift cannot be "
        "judged against a zero floor. If fidelity_gate.py reports floor_zero for this pair, rerun "
        "the twin at a different TWIN_DECOMP (known example: isc21's C02/C06 pair was "
        "bit-identical; C13 was used instead)." % (decomp, twin_decomp))

    top3 = [c["template"] for c in shortlist[:3]]
    verify_plan = None
    if top3:
        sbatch_line = (
            "sbatch --export=ALL,CASE=%s,MODE=physics,PCONFIGS=%s,DECOMP=%s,TWIN_DECOMP=%s "
            "campaign/strict_campaign.sbatch" % (case_name, "+".join(top3), decomp, twin_decomp))
        fidelity_cmds = []
        families_by_template = {c["template"]: (c.get("family") or "all") for c in candidates}
        for tid in top3:
            fidelity_cmds.append(
                "python3 fidelity_gate.py --reference <REFERENCE wrfout_d01_*> --twin <TWIN "
                "wrfout_d01_*> --candidate <%s CANDIDATE wrfout_d01_*> --family %s --out-prefix "
                "<out_dir>/fidelity_%s_%s" % (tid, families_by_template.get(tid, "all"), case_name, tid))
        verify_plan = dict(
            decomp=decomp, decomp_source=decomp_source, twin_decomp=twin_decomp, pconfigs=top3,
            sbatch_command=sbatch_line, fidelity_gate_commands=fidelity_cmds,
            note=("run P00 (reference at DECOMP + twin at TWIN_DECOMP -- same physics, different "
                  "STRICT decomposition -- for the chaos floor) plus the top-%d physics "
                  "candidate(s) above, all at DECOMP, via strict_campaign.sbatch MODE=physics; "
                  "then run fidelity_gate.py per candidate. The user makes the final "
                  "accept/reject decision from the measured labels -- this tool only narrows "
                  "the search" % len(top3)))
    else:
        warnings.append("no applicable candidate template for this case -- no verify plan")

    accuracy_extra = {}
    if objective == ACCURACY_DIRECTION:
        obs_scores = observation_scores_for_case(case_name, obs_wrfout)
        accuracy_view = build_accuracy_view(
            shortlist, train_rows, obs_scores=obs_scores,
            observations=case_observations(case_name))
        accuracy_view["verify_plan"] = _accuracy_verify_plan(
            case_name, accuracy_view, candidates, decomp, twin_decomp, verify_plan)
        accuracy_view["decomposition_note"] = (
            "DECOMP=%s (%s) is unchanged from perf mode: STRICT decomposition knobs "
            "(nproc_x/nproc_y/numtiles/OMP/distribution) redistribute the same arithmetic and "
            "never change the physics, so they are a FREE AXIS in accuracy mode -- the fastest "
            "legal decomposition is also the most accurate one. They do set the chaos floor every "
            "fidelity number below is judged against, which is the opposite of being a fidelity "
            "lever: a decomposition change is the NOISE, not the signal."
            % (decomp, decomp_source))
        accuracy_extra = dict(objective=objective, accuracy_view=accuracy_view)

    return dict(
        case_name=case_name, hardware_id=hardware_id, compliance="PHYSICS-GRAY", nodes=nodes,
        max_dom=effective_max_dom, **accuracy_extra,
        auto_apply_disclaimer=(
            "This tool only narrows the search to a shortlist. A human decides and applies the "
            "final physics template -- nothing here is auto-applied."),
        probe_source=probe_source, probe_valid=probe_valid, probe_verdicts=raw_verdicts,
        allowed_labels=list(allowed_labels), candidates=candidates, shortlist=shortlist,
        warnings=warnings, verify_plan=verify_plan,
        profile_shares=profile_shares, min_bucket_share=min_bucket_share,
        frozen_keys=list(frozen_keys),
    )


def format_physics_report(result):
    lines = []
    lines.append("=" * 78)
    lines.append("WTune PHYSICS-GRAY template shortlist -- case=%s hardware=%s max_dom=%s"
                  % (result["case_name"], result["hardware_id"], result["max_dom"]))
    lines.append(result["auto_apply_disclaimer"])
    lines.append("=" * 78)

    if result["warnings"]:
        lines.append("\n-- Evidence warnings --")
        for w in result["warnings"]:
            lines.append("  * %s" % w)

    changes_by_template = {c["template"]: c.get("changes") or {} for c in result["candidates"]}
    lines.append("\n-- Shortlist (%d template(s)) --" % len(result["shortlist"]))
    lines.append("  time-to-solution vs P00: NEGATIVE = FASTER (cost per simulated second); "
                  "[lo%,hi%] is a 95% leave-one-case-out jackknife+ interval when available, else "
                  "n/a with the reason in the warnings above (printed once, not per row).")
    lines.append("  infeasible: <failed>/<tried> training cases (every replicate FAILED); "
                  "OOF-miss: leave-one-case-out out-of-fold dangerous-miss <n>/<total> "
                  "(predicted allowed, measured LARGE-DRIFT), or insuff(n=<n>) below the "
                  "%d-prediction floor." % physicsmod.MIN_OOF_DANGEROUS_MISS)
    lines.append("  bucket: the profile bucket the template's family acts on and this case's "
                  "measured share of step cost; upper-bound saving = that share (a template cannot "
                  "save more than its bucket). '-' = no bucket bounds this family, or the case has "
                  "no profile row. A DEMOTED row is ranked last, never hidden.")
    lines.append("  train LARGE: MEASURED (not predicted) LARGE-DRIFT <k>/<labeled training "
                  "cases> -- a model-free track record that forces UNCONFIRMED on its own once "
                  "the rate reaches --max-template-large-rate, regardless of the fidelity "
                  "regression's own predicted label.")
    lines.append("%-4s %-9s %-42s %-20s %-20s %-9s %-9s %-11s %-9s %-13s %-13s %s"
                  % ("rank", "template", "changes (key old->new)", "time-to-sol vs P00",
                     "fidelity", "infeasible", "OOF-miss", "train LARGE", "probe", "bucket",
                     "status", "reasons"))
    for i, c in enumerate(result["shortlist"], start=1):
        changes = changes_by_template.get(c["template"], {})
        changes_str = ", ".join("%s %s->%s" % (k, v["from"], v["to"]) for k, v in changes.items()) or "-"
        tts, iv = c.get("predicted_time_to_solution_pct"), c.get("time_to_solution_interval_pct")
        if tts is None:
            perf_str = "n/a"
        elif iv is not None:
            perf_str = "%+.2f%% [%+.1f,%+.1f]" % (tts, iv[0], iv[1])
        else:
            perf_str = "%+.2f%% [n/a]" % tts
        label, nd = c.get("predicted_fidelity_label"), c.get("predicted_fidelity_nd")
        if label and nd is not None:
            fid_str = "%s (nd~%.3f)" % (label, nd)
        elif label:
            fid_str = label
        elif nd is not None:
            fid_str = "nd~%.3f" % nd
        else:
            fid_str = "n/a"
        infeas = c.get("infeasibility")
        infeasible_str = ("%d/%d" % (infeas["n_cases_failed"], infeas["n_cases_tried"])
                           if infeas is not None else "n/a")
        dang = c.get("dangerous_miss")
        if dang is None:
            oof_str = "insuff(n=0)"
        elif dang["insufficient"]:
            oof_str = "insuff(n=%d)" % dang["n"]
        else:
            oof_str = "%d/%d" % (dang["dangerous_miss_n"], dang["n"])
        track = c.get("template_track_record")
        track_str = ("LARGE %d/%d" % (track["n_cases_large"], track["n_cases_labeled"])
                     if track is not None else "n/a")
        share = c.get("bucket_share")
        bucket_str = ("-" if share is None else
                       "%s %.1f%%%s" % ((c.get("bucket") or "").replace("profile_", "")
                                         .replace("_share", ""), 100 * share,
                                         " DEMOTED" if c.get("bucket_demoted") else ""))
        lines.append("%-4d %-9s %-42s %-20s %-20s %-9s %-9s %-11s %-9s %-13s %-13s %s" % (
            i, c["template"], changes_str[:42], perf_str, fid_str, infeasible_str, oof_str,
            track_str, c.get("probe_verdict") or "none", bucket_str, c["status"],
            "; ".join(c.get("reasons") or [])))

    av = result.get("accuracy_view")
    if av is not None:
        lines.extend(_format_accuracy_sections(av))

    vp = result.get("verify_plan")
    if vp and av is None:
        lines.append("\n-- What to run to verify --")
        lines.append("  %s" % vp["note"])
        lines.append("  %s" % vp["sbatch_command"])
        for cmd in vp["fidelity_gate_commands"]:
            lines.append("  %s" % cmd)

    return "\n".join(lines)


_ACC_HEADER = "%-4s %-9s %-9s %-22s %-22s %-20s %-6s %s"


def _fmt_error(entry):
    v = entry["fidelity_error"]
    return "n/a" if v is None else "%.6g" % v


def _fmt_floor(entry):
    v = entry["floor_ratio"]
    return "n/a" if v is None else "%.3f" % v


def _fmt_tts(entry):
    tts, iv = entry["predicted_time_to_solution_pct"], entry.get("time_to_solution_interval_pct")
    if tts is None:
        return "n/a"
    return "%+.2f%% [%+.1f,%+.1f]" % (tts, iv[0], iv[1]) if iv is not None else "%+.2f%% [n/a]" % tts


def _acc_row(i, e):
    return _ACC_HEADER % (i, e["template"], e["direction"], _fmt_error(e), _fmt_floor(e),
                          _fmt_tts(e), "yes" if e["on_pareto"] else "no", e["status"])


def _format_accuracy_sections(av):
    """The --objective accuracy blocks appended to the standard PHYSICS-GRAY report. The perf
    shortlist above is left exactly as it prints in perf mode -- accuracy mode re-ranks the SAME
    candidates rather than replacing them, so both orderings stay visible side by side."""
    lines = ["\n" + "=" * 78,
             "ACCURACY-FIRST view (--objective accuracy) -- same candidates, ranked by fidelity "
             "error",
             "=" * 78,
             "  fidelity error = nd, the normalized difference from the campaign REFERENCE at the "
             "last common time.",
             "  floor_ratio = that drift divided by the chaos floor (the SAME physics re-run at "
             "TWIN_DECOMP).",
             "  HARD RULE: floor_ratio <= %.2f means the candidate moved the solution no more "
             "than merely re-decomposing the run does. Such a candidate is reported as "
             "indistinguishable from changing the decomposition and is NEVER presented as an "
             "accuracy improvement -- chasing error below the chaos floor is meaningless. "
             "Measured example from this campaign: pure RK2 (D02) on conus12 at 12 h had "
             "floor_ratio 1.593, i.e. dropping an entire RK stage was already within 2x of just "
             "re-decomposing." % av["chaos_floor_ratio"],
             "  Predicted speed is still reported for every candidate: accuracy mode changes the "
             "RANKING, not the evidence."]

    # Observation-scored candidates first, when this case has real ground truth: they answer the
    # question the nd blocks below only stand in for, so burying them under model-vs-model
    # rankings would be the wrong order to read the report in.
    observed = av.get("ranked_observed") or []
    if observed:
        spec = av.get("observations") or {}
        lines.append("\n-- Scored against REAL OBSERVATIONS (%d) --" % len(observed))
        lines.append("  Ground truth: %s (%s). Metric: %s, LOWER IS BETTER; this is a distance "
                     "from REALITY, not a drift from a reference run, so the chaos floor does "
                     "NOT apply to it and a FASTER candidate can legitimately rank HIGHER -- the "
                     "problem statement's own settings were never optimised."
                     % (spec.get("path", "?"), spec.get("kind", "?"),
                        verify_obs.PRIMARY_METRIC.get(spec.get("kind"), "?")))
        lines.append(_ACC_HEADER % ("rank", "template", "direction", "observation error",
                                    "floor_ratio (n/a)", "time-to-sol vs P00", "pareto", "status"))
        for i, e in enumerate(observed, start=1):
            lines.append(_acc_row(i, e))

    lines.append("\n-- Cost levers, ranked by LEAST departure from the reference (%d) --"
                 % len(av["ranked_cost_levers"]))
    lines.append("  These are the P/D-series templates: they buy speed and pay in drift, so a "
                 "SMALLER error is strictly better and rank 1 is the most conservative choice.")
    lines.append(_ACC_HEADER % ("rank", "template", "direction", "fidelity error (nd)",
                                "floor_ratio", "time-to-sol vs P00", "pareto", "status"))
    for i, e in enumerate(av["ranked_cost_levers"], start=1):
        lines.append(_acc_row(i, e))
    if not av["ranked_cost_levers"]:
        lines.append("  (none)")

    lines.append("\n-- Accuracy levers (A-series), ranked by LARGEST measurable correction (%d) --"
                 % len(av["ranked_accuracy_levers"]))
    lines.append("  Ranked the OTHER WAY ROUND on purpose. For an A-series template the same nd "
                 "is not an error, it is the correction the extra work applies to the reference; "
                 "a small one means the reference was already converged in that knob and the "
                 "template is pure added cost. Ranking these ascending alongside the cost levers "
                 "would reward exactly the ones that do nothing.")
    lines.append(_ACC_HEADER % ("rank", "template", "direction", "correction (nd)", "floor_ratio",
                                "time-to-sol vs P00", "pareto", "status"))
    for i, e in enumerate(av["ranked_accuracy_levers"], start=1):
        lines.append(_acc_row(i, e))
    if not av["ranked_accuracy_levers"]:
        lines.append("  (none -- no applicable A-series template for this case)")

    lines.append("\n-- Below the chaos floor: NOT accuracy improvements (%d) --"
                 % len(av["below_floor"]))
    if not av["below_floor"]:
        lines.append("  (none)")
    for e in av["below_floor"]:
        lines.append("  %-9s %-9s %s" % (e["template"], e["direction"], e["below_floor_note"]))
        if e["floor_ratio_source"]:
            lines.append("            floor_ratio source: %s" % e["floor_ratio_source"])

    lines.append("\n-- Pareto front over (predicted cost, fidelity error) (%d) --"
                 % len(av["pareto_front"]))
    lines.append("  Every operating point that is not beaten on BOTH axes at once. perf-first is "
                 "the cheapest end of this front and acc-first the lowest-error end; anything "
                 "between them is a legitimate trade the user can pick. A candidate off the front "
                 "is strictly worse than something on it.")
    if not av["pareto_front"]:
        lines.append("  (empty -- every candidate is missing a coordinate or sits below the chaos "
                     "floor; see the exclusions below)")
    for e in av["pareto_front"]:
        lines.append("  %-9s %-9s error=%-14s cost=%-22s %s"
                     % (e["template"], e["direction"], _fmt_error(e), _fmt_tts(e),
                        e["fidelity_error_source"] or ""))
    excluded = [e for e in av["entries"] if e.get("pareto_excluded_reason")]
    if excluded:
        lines.append("  not placed on the front: " + "; ".join(
            "%s (%s)" % (e["template"], e["pareto_excluded_reason"]) for e in excluded))

    lines.append("\n-- Decomposition: a free axis in accuracy mode --")
    lines.append("  %s" % av["decomposition_note"])

    vp = av.get("verify_plan")
    if vp:
        lines.append("\n-- What to run to verify (accuracy campaign) --")
        lines.append("  %s" % vp["note"])
        if vp.get("stage1"):
            lines.append("  STAGE 1: %s" % vp["stage1"])
        for cmd in vp.get("stage2") or []:
            lines.append("  STAGE 2: %s" % cmd)
    return lines


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------
def _fmt_value(entry):
    if entry["score_kind"] == "pct_change":
        return "%+.2f%%" % (entry["value"] * 100.0)
    return "%.4f (ranking score)" % entry["value"]


def _fmt_interval(entry):
    iv = entry.get("interval_95")
    if iv is None:
        return "n/a"
    lo, hi = iv
    return "[%+.2f%%, %+.2f%%]" % ((math.exp(lo) - 1) * 100.0, (math.exp(hi) - 1) * 100.0)


def _fmt_magnitude_value(entry):
    return "%+.2f%%" % (entry["magnitude_value"] * 100.0)


def _fmt_magnitude_interval(entry):
    iv = entry.get("magnitude_interval_95")
    if iv is None:
        return "n/a"
    lo, hi = iv
    return "[%+.2f%%, %+.2f%%]" % ((math.exp(lo) - 1) * 100.0, (math.exp(hi) - 1) * 100.0)


def format_report(result):
    lines = []
    lines.append("=" * 78)
    lines.append("WTune STRICT config shortlist -- case=%s hardware=%s nodes=%s compliance=%s "
                  "max_dom=%s%s"
                  % (result["case_name"], result["hardware_id"], result["nodes"],
                     result["compliance"], result.get("max_dom"),
                     (" holdout_case=%r" % result["holdout_case"]) if result.get("holdout_case")
                     else ""))
    lines.append(result["auto_apply_disclaimer"])
    lines.append("=" * 78)

    if result["warnings"]:
        lines.append("\n-- Evidence warnings --")
        for w in result["warnings"]:
            lines.append("  * %s" % w)

    if result["skipped"]:
        lines.append("\n-- Skipped templates (illegal for this grid) --")
        for s in result["skipped"]:
            lines.append("  skipped: %s -- %s" % (s["config_id"], s["reason"]))

    if result.get("aliases"):
        lines.append("\n-- Duplicate templates (collapsed to one physical run) --")
        for kid, dropped in result["aliases"].items():
            lines.append("  %s == %s" % (kid, ", ".join(dropped)))

    if not result.get("shortlist"):
        lines.append("\nNo shortlist: no legal candidates for this case/--nodes combination.")
        return "\n".join(lines)

    lines.append("\n-- Model: requested=%s, used=%s%s; magnitude companion=%s%s --" % (
        result["model_requested"], result["model_used"],
        "" if result["model_requested"] == result["model_used"] else " (FELL BACK)",
        result["magnitude_model_used"],
        "" if result["magnitude_model_used"] == "catboost_point" else " (FELL BACK from catboost_point)"))

    magnitude_header = "magnitude (%s)" % result["magnitude_model_used"]
    header = ("%-4s %-4s %-9s %-9s %-8s %-11s %-6s %-20s %-22s %-20s %-12s"
              % ("rank", "id", "ranksxomp", "nproc_xxy", "numtiles", "distribution",
                 "KMPBT", "ranking value", magnitude_header, "magnitude 95% CI", "label"))
    lines.append("\n-- Shortlist (top %d of %d legal candidates) --" %
                  (len(result["shortlist"]), len(result["all_ranked"])))
    lines.append("   ranking = %s (search order); %s = the magnitude companion's own percent "
                  "change vs reference + calibrated interval -- SAFE/EXPERIMENTAL is derived "
                  "from the magnitude column, never from the ranking column"
                  % (result["model_used"], magnitude_header))
    lines.append(header)
    for e in result["shortlist"]:
        lines.append("%-4d %-4s %-9s %-9s %-8d %-11s %-6s %-20s %-22s %-20s %-12s%s" % (
            e["rank"], e["config_id"], "%dx%d" % (e["n_ranks"], e["omp_threads"]),
            "%dx%d" % (e["nproc_x"], e["nproc_y"]), e["numtiles"], e["distribution"],
            "default" if e["kmp_blocktime"] is None else str(e["kmp_blocktime"]),
            _fmt_value(e), _fmt_magnitude_value(e), _fmt_magnitude_interval(e), e["label"],
            "  <- reference" if e["is_reference"] else ""))

    ref = result["reference"]
    if not ref["in_shortlist"]:
        lines.append("\nReference config (targets._reference_order, not in the top-%d shown "
                      "above): rank %d, %s -- ranking %s, magnitude %s %s"
                      % (len(result["shortlist"]), ref["rank"], ref["config_id"],
                         _fmt_value(ref), _fmt_magnitude_value(ref), ref["label"]))

    vp = result.get("verify_plan")
    if vp:
        lines.append("\n-- What to run to verify --")
        lines.append("  %s" % vp["note"])
        lines.append("  run length: %s min (%s)" % (vp["run_minutes"], vp["run_minutes_source"]))
        lines.append("  run order: %s" % " ".join(vp["palindromic_run_order"]))
        for c in vp["configs"]:
            lines.append("\n  config %s: namelist keys %s" % (c["config_id"], c["namelist_keys"]))
            lines.append("    %s" % c["namelist_edit_command"])
            for ln in c["run_command"].splitlines():
                lines.append("    %s" % ln)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Shortlist STRICT WRF configs for a case from the WTune predictor ledger. "
                    "Narrows the search; a human decides and applies the final config.")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--namelist", required=True)
    ap.add_argument("--case-name", required=True)
    ap.add_argument("--hardware-id", required=True)
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--compliance", default="STRICT")
    ap.add_argument("--model", default="catboost_pairwise")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--no-intervals", action="store_true")
    ap.add_argument("--max-dom", type=int, default=None, dest="max_dom",
                     help="override the number of active domains checked for decomposition "
                          "legality and used in the verify-plan's edit-namelist command "
                          "(default: this namelist's own max_dom; can only narrow, never exceed "
                          "the namelist's own domain count)")
    ap.add_argument("--holdout-case", default=None, dest="holdout_case",
                     help="drop this case_base's rows from training before fitting anything -- "
                          "use this to get an honest out-of-sample shortlist when --case-name "
                          "(or its grid) is itself already in the training ledger")

    ap.add_argument("--space", choices=["templates", "full"], default="templates",
                     help="STRICT only (WTune D4). 'templates' (default, unchanged): the 16 "
                          "fixed campaign/configs.py STRICT templates. 'full': enumerate every "
                          "legal decomp_space.py candidate for this namelist's grid/--nodes and "
                          "score the whole pool; can run into the thousands, so conformal "
                          "intervals are computed only for the ranked --top-k, and candidates "
                          "whose ranks x omp / tile shape / patch size were never measured on "
                          "--hardware-id are flagged EXTRAPOLATED and forced EXPERIMENTAL")
    ap.add_argument("--propose-batch", type=int, default=None, dest="propose_batch",
                     help="in addition to the shortlist above, also emit a WTune D3 round-1 "
                          "active-search proposal for this case (predictor/decomp_search.py): a "
                          "CONFIG_FILE JSON of this many candidates plus the exact sbatch command "
                          "to measure them. Requires --case-name to be a campaign/cases.json case "
                          "id UNLESS the namelist alone is enough (this tool builds the case from "
                          "the parsed namelist either way, so an unregistered case works too). "
                          "The tool only proposes -- the human decides whether to run it")

    ap.add_argument("--obs-wrfout", action="append", default=[], dest="obs_wrfout",
                     metavar="TEMPLATE=GLOB",
                     help="PHYSICS-GRAY + --objective accuracy only. Repeatable. Score an "
                          "ALREADY-RUN candidate against this case's REAL OBSERVATIONS instead "
                          "of against another model run, e.g. "
                          "--obs-wrfout \"P01=runs/P01/wrfout_d01_*\". Only works for a case "
                          "whose campaign/cases.json entry declares an 'observations' block "
                          "(today: the SC26 practice problems); every other case, and every "
                          "template with no run to point at, keeps the existing model-vs-model "
                          "fidelity. predictor/verify_obs.py does the scoring and the result "
                          "appears in accuracy_view['ranked_observed'], lower-is-better, with "
                          "the chaos floor NOT applied -- a faster config can legitimately be "
                          "more accurate, because the problem statement's own settings were "
                          "never optimised")
    ap.add_argument("--decomp", default=None,
                     help="PHYSICS-GRAY only: reference STRICT decomposition id (e.g. C02) to "
                          "run P00 and the physics candidates at; default: this run's own STRICT "
                          "top-1 shortlist entry, falling back to C02")
    ap.add_argument("--physics-ledger", action="append", default=[], dest="physics_ledger",
                     metavar="PATH",
                     help="PHYSICS-GRAY only: repeatable; a physics-campaign ledger CSV "
                          "(source_dataset=strict_campaign mode=physics rows). None exist in "
                          "production yet -- omit to get an honestly UNCONFIRMED shortlist")
    ap.add_argument("--allowed-labels", default="WITHIN-FLOOR,SMALL-DRIFT", dest="allowed_labels",
                     help="PHYSICS-GRAY only: comma-separated fidelity labels a candidate must "
                          "be predicted to have to ever be RECOMMENDABLE")
    ap.add_argument("--max-infeasible-rate", type=float, default=0.0, dest="max_infeasible_rate",
                     help="PHYSICS-GRAY only: a template that FAILED every replicate in more than "
                          "this fraction of the training cases it was tried in is forced "
                          "UNCONFIRMED (still listed); default 0.0 = any observed training "
                          "failure at all")
    ap.add_argument("--max-dangerous-miss", type=float, default=0.0, dest="max_dangerous_miss",
                     help="PHYSICS-GRAY only: a template whose leave-one-case-out out-of-fold "
                          "fidelity dangerous-miss rate (predicted allowed, measured LARGE-DRIFT) "
                          "exceeds this is forced UNCONFIRMED; default 0.0. Fewer than "
                          "physics.MIN_OOF_DANGEROUS_MISS out-of-fold predictions for a template "
                          "is ALWAYS insufficient evidence -> UNCONFIRMED regardless of this flag")
    ap.add_argument("--max-template-large-rate", type=float,
                     default=physicsmod.DEFAULT_MAX_TEMPLATE_LARGE_RATE,
                     dest="max_template_large_rate",
                     help="PHYSICS-GRAY only: a template MEASURED LARGE-DRIFT in at least this "
                          "fraction of its own labeled training cases is forced UNCONFIRMED, "
                          "regardless of the regression's own predicted label for the new case "
                          "(model-free track record; default 0.5)")
    ap.add_argument("--min-speedup-pct", type=float, default=physicsmod.DEFAULT_MIN_SPEEDUP_PCT,
                     dest="min_speedup_pct",
                     help="PHYSICS-GRAY only: a candidate is only RECOMMENDABLE if its predicted "
                          "time-to-solution change is at most this (negative=faster) and, when a "
                          "conformal interval exists, its upper bound is also < 0; default -1.0 "
                          "(at least 1%% faster)")
    ap.add_argument("--min-bucket-share", type=float, default=DEFAULT_MIN_BUCKET_SHARE,
                     dest="min_bucket_share",
                     help="PHYSICS-GRAY only: a template whose profile bucket (radiation -> "
                          "profile_radiation_share, microphysics -> micro, dynamics/nesting/"
                          "pbl_surface -> dynamics) is below this fraction of measured step cost "
                          "is DEMOTED to the bottom of the shortlist with the reason shown -- it "
                          "cannot save more than the bucket it acts on. Default 0.05 (5%%); a "
                          "value > 1 is read as a percent. Requires a profiled ledger row for the "
                          "case: with none, nothing is annotated or demoted. Measured contrast: "
                          "SC26-P1 radiation 1.9%% vs SC26-P2 47%%")
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="perf",
                     help="PHYSICS-GRAY only. 'perf' (default, unchanged in every byte): rank "
                          "candidates by predicted speed, gated on an allowed fidelity label. "
                          "'accuracy': keep the same candidates and the same predicted speeds but "
                          "rank them by the MEASURED fidelity error instead, add the A-series "
                          "accuracy templates' own block, print the Pareto front over (predicted "
                          "cost, fidelity error) so any operating point between perf-first and "
                          "acc-first can be picked, and refuse to call any candidate whose "
                          "floor_ratio <= %.2f an accuracy improvement -- at or below the chaos "
                          "floor it is indistinguishable from changing the decomposition. The "
                          "decomposition itself stays a free axis: it does not change the physics, "
                          "so the fastest legal one is still recommended" % CHAOS_FLOOR_RATIO)
    ap.add_argument("--probe-screen", default=None, dest="probe_screen",
                     help="PHYSICS-GRAY only: path to a saved 'wtune.py screen ... --json' "
                          "stdout capture, used for the SpeedyWeather probe verdicts instead of "
                          "running the probe now")
    ap.add_argument("--run-probe", action="store_true", dest="run_probe",
                     help="PHYSICS-GRAY only: run the SpeedyWeather probe now via 'wtune.py "
                          "screen' (this is also the default when --probe-screen is not given)")
    ap.add_argument("--latlon", default=None,
                     help="PHYSICS-GRAY --run-probe: domain-centre 'LAT,LON' for the probe bbox")
    ap.add_argument("--wps", default=None,
                     help="PHYSICS-GRAY --run-probe: namelist.wps path for the domain centre")
    ap.add_argument("--wrfinput", default=None,
                     help="PHYSICS-GRAY --run-probe: wrfinput/wrfout path for lat/lon/date")
    return ap


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.compliance == "PHYSICS-GRAY":
            result = recommend_physics_gray(
                ledger_path=args.ledger, namelist_path=args.namelist, case_name=args.case_name,
                hardware_id=args.hardware_id, nodes=args.nodes, top_k=args.top_k,
                max_dom=args.max_dom, holdout_case=args.holdout_case, decomp=args.decomp,
                physics_ledgers=args.physics_ledger, allowed_labels=args.allowed_labels,
                probe_screen=args.probe_screen, run_probe=args.run_probe, latlon=args.latlon,
                wps=args.wps, wrfinput=args.wrfinput,
                max_infeasible_rate=args.max_infeasible_rate,
                max_dangerous_miss=args.max_dangerous_miss,
                max_template_large_rate=args.max_template_large_rate,
                min_speedup_pct=args.min_speedup_pct, objective=args.objective,
                min_bucket_share=args.min_bucket_share,
                obs_wrfout=_parse_obs_wrfout(args.obs_wrfout))
            report = format_physics_report(result)
        elif args.objective != "perf":
            raise ValueError(
                "--objective %s requires --compliance PHYSICS-GRAY. STRICT knobs "
                "(nproc_x/nproc_y/numtiles/OMP/distribution) redistribute the same arithmetic and "
                "never change the physics, so there is no accuracy to buy on the STRICT axis -- "
                "the fastest legal decomposition is also the most accurate one, which is what the "
                "default --objective perf already recommends." % args.objective)
        elif args.space == "full":
            result = recommend_full_space(
                ledger_path=args.ledger, namelist_path=args.namelist, case_name=args.case_name,
                hardware_id=args.hardware_id, nodes=args.nodes, model=args.model,
                top_k=args.top_k, no_intervals=args.no_intervals, max_dom=args.max_dom,
                holdout_case=args.holdout_case)
            report = format_full_space_report(result)
        else:
            result = recommend(
                ledger_path=args.ledger, namelist_path=args.namelist, case_name=args.case_name,
                hardware_id=args.hardware_id, nodes=args.nodes, compliance=args.compliance,
                model=args.model, top_k=args.top_k, no_intervals=args.no_intervals,
                max_dom=args.max_dom, holdout_case=args.holdout_case)
            report = format_report(result)

        if args.propose_batch and args.compliance != "PHYSICS-GRAY":
            pb_result, pb_configs = run_propose_batch(
                args.ledger, args.namelist, args.case_name, args.hardware_id, args.nodes,
                args.max_dom, args.propose_batch)
            pb_out = (os.path.splitext(args.json_out)[0] + ".propose_batch.json"
                      if args.json_out else "%s_propose_batch.json" % args.case_name)
            if not pb_result["stopped"]:
                with open(pb_out, "w") as f:
                    json.dump(dict(case_id=args.case_name, round=pb_result["round"],
                                    configs=pb_configs), f, indent=2, sort_keys=True)
            report += "\n" + format_propose_batch_report(args.case_name, pb_result, pb_configs, pb_out)
    except UnsupportedComplianceError as e:
        sys.stderr.write("UNDERPOWERED: %s\n" % e)
        return 2
    except (ValueError, RuntimeError) as e:
        sys.stderr.write("ERROR: %s\n" % e)
        return 1

    print(report)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
