"""Phase 3 of `wtune tune --pipeline full` (decision 0059): PHYSICS-GRAY advice. PURE LOGIC.

Policy (decisions 0051 / 0056): a physics or time-step change alters the solution, so WTune only
ADVISES -- every row of the advice table says "advice only -- not applied". This module never
launches anything and never writes a namelist to disk; the orchestrator (wtune/tune/) runs the
candidates and feeds the measurements back in.

  plan_candidates(namelist_path)            -> case-relevant PHYSICS-GRAY candidates, best prior first
  weights_from_combine(points, ...)         -> per-point weights out of wtune.probe_plan.combine()
  predicted_speedup(base, cand, weights)    -> per-point ratio + whole-run saving + spread
  fidelity_verdict(ref_wrfout, cand_wrfout) -> wtune.config_gate field-by-field verdict
  advice_table(cands, speed, fidelity)      -> the report table
  render_candidate_namelist(text, cand)     -> candidate namelist TEXT (no file written)

Reused, not reimplemented: predictor/physics.py (physics_candidates_from_namelist, generic_extra_edits,
_template_case_entry), campaign/configs.py (apply_physics_template / apply_physics_edit /
template_frozen_key_conflicts, physics_templates.json), wtune.case (COST_WEIGHT, DYN_WEIGHT),
wtune.physics (cost_weight, resolve_suite, scheme_name), wtune.extensions.detect,
wtune.config_gate.compare, wtune.probe_plan.combine.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import tempfile

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from wtune import fidelity_gate as chaos_gate      # decision 0051 chaos-floor gate: run_gate() -> label
from wtune import config_gate as gate        # compare() + _check_results(): field-by-field gate
from wtune import extensions as wext         # detect(): chem / hydro / wrfda / movenest
from wtune import physics as phys            # cost_weight, resolve_suite, scheme_name
from wtune import predict as wpredict        # parse_namelist, g_dom (per-domain broadcast-last)
from wtune import probe_plan as probe_plan   # combine(), PointMeasurement
from wtune.case import COST_WEIGHT, DYN_WEIGHT   # the existing prior cost weights

CLASS = "PHYSICS-GRAY"
# Edits to any of these keys change the physics SIGNATURE: the runner skips such a config under
# RANKING_STRICT_RESTART=1 (the restart arrays of the shared spin-up do not fit another scheme), so
# the candidate cannot be measured from the shared restart -- it needs a cold start.
RESTART_SIGNATURE_KEYS = ("mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
                          "sf_surface_physics", "ra_lw_physics", "ra_sw_physics")
ESTIMATE_ONLY = "estimate only — needs a cold start to measure"
# Prognostic state fields the advice verdict is judged on. OLR/TSK (and other radiation/skin
# diagnostics) are snapshots sampled at an instant one run may have refreshed radiation and the
# other not: a sampling artefact of a short window, reported separately, never judged.
PROGNOSTIC_FIELDS = ("T", "U", "V", "QVAPOR", "PSFC", "T2", "Q2", "U10", "V10")
ADVICE_ONLY = "advice only — not applied"
MIN_SAVING_PCT = 5.0            # below this a candidate is "not worth it" (decision 0059)
MIN_PRIOR_GAIN_PCT = 0.5        # candidates the cost weights say save less than this are not offered
CU_OFF_MAX_DX_M = 4000.0        # cumulus parameterisation is switched off only where dx <= 4 km
# Tolerance used when the caller supplies none: PROVISIONAL, NOT calibrated (no accepted pair has
# been measured on a physics change yet). Pass tol_rel= or calibrate with calibrate_tolerance().
DEFAULT_TOL_REL = 0.05
# W is excluded by default: its rel_rms is O(1) for ANY solution change, so it fails every candidate.
GATE_DEFAULT_FIELDS = ("T2", "Q2", "U10", "V10", "PSFC", "T", "U", "V", "QVAPOR")

# moment class of mp_physics (wtune.physics._MP has cost labels, not moment counts)
_MP_SINGLE_MOMENT = (2, 3, 4, 6)    # Lin, WSM3, WSM5, WSM6 (Kessler=1 is warm-rain only: not a substitute)
_MP_DOUBLE_MOMENT = (8, 10, 16, 18, 28, 38, 40)
_COST_CATS = ("mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics")
_RA_CATS = ("ra_lw_physics", "ra_sw_physics")


class CandidateList(list):
    """A plain list of candidate dicts that also carries `.note` (why it is empty / what was
    filtered) and `.skipped` ([(id, reason)])."""
    note = ""
    skipped: list = []


def _deps():
    """Lazy: predictor.physics pulls in sklearn/catboost -- only plan_candidates needs it."""
    from wtune.predictor import physics as predictor_physics
    return predictor_physics, predictor_physics.campaign_configs


# --------------------------------------------------------------------------------------------
# namelist access
# --------------------------------------------------------------------------------------------
def _dv(nl, overlay, group, key, d, default=None):
    """Per-domain value (WRF broadcast-last), `overlay` ({key: [values]}) taking precedence."""
    if key in overlay:
        vals = overlay[key]
        return (vals[d] if d < len(vals) else vals[-1]) if vals else default
    return wpredict.g_dom(nl, group, key, d, default)


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _dx_list(nl, max_dom):
    dx = [_num(wpredict.g_dom(nl, "domains", "dx", d)) for d in range(max_dom)]
    if all(v is not None for v in dx) and len((nl.get("domains", {}).get("dx")) or []) >= max_dom:
        return dx
    out = [_num(wpredict.g(nl, "domains", "dx"))]
    for d in range(1, max_dom):
        pgr = _num(wpredict.g_dom(nl, "domains", "parent_grid_ratio", d), 1.0) or 1.0
        out.append(out[-1] / pgr if out[-1] is not None else None)
    return out


def _dt_list(nl, overlay, max_dom):
    dt = _num(overlay["time_step"][0]) if "time_step" in overlay else _num(
        wpredict.g(nl, "domains", "time_step"))
    if dt is None or dt <= 0:
        dt = 60.0                    # adaptive/auto: same fallback as wtune.cli.radiation_spacing
    out, cum = [], 1.0
    for d in range(max_dom):
        if d > 0:
            r = wpredict.g_dom(nl, "domains", "parent_time_step_ratio", d)
            if r is None:
                r = wpredict.g_dom(nl, "domains", "parent_grid_ratio", d, 1)
            cum *= (_num(r, 1.0) or 1.0)
        out.append(dt / cum)
    return out


def _resolved_phys(nl, overlay, d):
    suite = wpredict.g(nl, "physics", "physics_suite")
    case = {"physics_suite": suite.strip("'\" ").lower() if isinstance(suite, str) else suite}
    for k in _COST_CATS + _RA_CATS:
        v = _dv(nl, overlay, "physics", k, d)
        case[k] = int(v) if v is not None else -1
    resolved, _ = phys.resolve_suite(case)
    return resolved


def _cost_rate(nl, overlay, max_dom):
    """Relative cost per SIMULATED second (prior, from the existing cost weights): per domain,
    (dynamics + mp + cu + pbl/sfclay (amortised by bldt) + lsm + radiation (amortised by radt))
    x cells / dt_domain. Same ingredients as wtune.case._build_cost_structure, plus the bldt and
    time-step amortisation the PHYSICS-GRAY templates act on. A ranking prior, never a wall time."""
    dts = _dt_list(nl, overlay, max_dom)
    total = 0.0
    for d in range(max_dom):
        r = _resolved_phys(nl, overlay, d)
        w = {c: phys.cost_weight(c, r[c], COST_WEIGHT) for c in _COST_CATS + _RA_CATS}
        radt = _num(_dv(nl, overlay, "physics", "radt", d), 0.0) or 0.0
        spacing = max(1, int(round(radt * 60.0 / dts[d]))) if radt > 0 else 1
        bldt = _num(_dv(nl, overlay, "physics", "bldt", d), 0.0) or 0.0
        bl_n = max(1, int(round(bldt * 60.0 / dts[d]))) if bldt > 0 else 1
        step = (DYN_WEIGHT + w["mp_physics"] + w["cu_physics"] + w["sf_surface_physics"]
                + (w["bl_pbl_physics"] + w["sf_sfclay_physics"]) / bl_n
                + (w["ra_lw_physics"] + w["ra_sw_physics"]) / spacing)
        cells = 1.0
        for k in ("e_we", "e_sn", "e_vert"):
            cells *= _num(wpredict.g_dom(nl, "domains", k, d), 1.0) or 1.0
        total += step * cells / dts[d]
    return total


# --------------------------------------------------------------------------------------------
# candidate construction
# --------------------------------------------------------------------------------------------
def _template_ops(tid, spec, case_name, resolved, pp):
    """The edit ops physics_candidates_from_namelist applies for template `tid` on this case
    (JSON per-case entry wins, else generic_extra_edits) -- same rule, see predictor/physics.py."""
    has_entry, json_extra, _skip = pp._template_case_entry(spec, case_name)
    extra = json_extra if has_entry else pp.generic_extra_edits(tid, resolved)[0]
    return list(spec.get("edits") or []) + list(extra or [])


def _own_ops(nl, resolved, max_dom, dxs):
    """Candidates the templates do not provide: [(id, ops, family)]."""
    out = []
    cu_vals = list(nl.get("physics", {}).get("cu_physics") or [])
    if len(cu_vals) >= max_dom and max_dom > 0:
        new = [0 if (dxs[d] is not None and dxs[d] <= CU_OFF_MAX_DX_M and _num(cu_vals[d], 0) != 0)
               else cu_vals[d] for d in range(max_dom)]
        if new != cu_vals[:max_dom]:
            out.append(("CU-OFF", [dict(op="set", key="cu_physics", values_by_domain=new,
                                        section="&physics")], "cumulus"))
    mp = resolved.get("mp_physics")
    cls = (_MP_SINGLE_MOMENT if mp in _MP_SINGLE_MOMENT else
           _MP_DOUBLE_MOMENT if mp in _MP_DOUBLE_MOMENT else ())
    w0 = phys.cost_weight("mp_physics", mp, COST_WEIGHT) if cls else 0.0
    for tgt in cls:
        if tgt != mp and phys.cost_weight("mp_physics", tgt, COST_WEIGHT) < w0:
            out.append(("MP-%d" % tgt, [dict(op="set", key="mp_physics", value=tgt,
                                             section="&physics")], "microphysics"))
    return out


def _label_rationale(changes, family, gain_pct, tid):
    def fmt(k, vals):       # per-domain list, collapsed when every domain agrees
        names = [phys.scheme_name(k, v) if k in phys._KB else ("default" if v is None else str(v))
                 for v in vals] or ["default"]
        return names[0] if len(set(names)) == 1 else "/".join(names)
    label = "; ".join("%s %s->%s" % (k, fmt(k, c["from"]), fmt(k, c["to"]))
                      for k, c in changes.items())
    cls_note = ""
    if "mp_physics" in changes:
        a, b = changes["mp_physics"]["from"][:1], changes["mp_physics"]["to"][:1]
        if a and b and (a[0] in _MP_DOUBLE_MOMENT) != (b[0] in _MP_DOUBLE_MOMENT):
            cls_note = "; changes moment class (double->single)"
    rationale = ("%s family: prior cost weights give ~%.1f%% less work per simulated second "
                 "(prior, not measured)%s; results-affecting" % (family or "physics", gain_pct, cls_note))
    return label, rationale


def render_candidate_namelist(namelist_text, cand):
    """Candidate namelist TEXT = `namelist_text` with cand['ops'] applied through
    campaign/configs.py apply_physics_template (the single engine every PHYSICS-GRAY run uses).
    Raises ValueError if an edit cannot be applied."""
    _, cc = _deps()
    new_text, _summary = cc.apply_physics_template(namelist_text, {"edits": cand["ops"]},
                                                   cand["max_dom"])
    return new_text


def _edits_and_overlay(text, ops, max_dom, cc):
    _t, summary = cc.apply_physics_template(text, {"edits": ops}, max_dom)
    edits, overlay, changes = {}, {}, {}
    for e in summary:
        if e["op"] == "delete":
            edits[e["key"]] = None
            continue
        edits[e["key"]] = ", ".join(e["new"])
        overlay[e["key"]] = [wpredict._parse_scalar(v) for v in e["new"]]
        changes[e["key"]] = {"from": [wpredict._parse_scalar(v) for v in e["old"] or []],
                             "to": [wpredict._parse_scalar(v) for v in e["new"] or []]}
    return edits, overlay, changes


def restart_compatible(edits):
    """True when the candidate's edits leave every physics-signature key alone (radt, bldt, cudt,
    time_step... -- measurable from the shared restart)."""
    return not ({str(k).lower() for k in (edits or {})} & set(RESTART_SIGNATURE_KEYS))


def plan_candidates(namelist_path, *, max_candidates=6, case_name=None, frozen_keys=None,
                    max_estimate_only=3):
    """Case-relevant PHYSICS-GRAY candidates for this namelist, best prior saving first.

    Each dict: id, label, edits ({key: value string as written, per-domain comma list}),
    rationale (one line), class='PHYSICS-GRAY', plus ops (the physics_templates.json edit ops,
    the form render_candidate_namelist consumes), max_dom, family, template (id or None),
    prior_gain_pct (cost-weight prior; ordering only).

    Returns [] (a CandidateList whose .note says why) for chem / hydro / WRFDA / moving-nest cases.
    `case_name` selects physics_templates.json per-case pairings and cases.json frozen_keys;
    `frozen_keys` adds task-mandated keys (e.g. SC26-P1 time_step) -- candidates touching one are
    dropped, never ranked low. Deterministic: order by (-prior_gain_pct, id)."""
    pp, cc = _deps()
    out = CandidateList()
    out.skipped = []
    nl = wpredict.parse_namelist(namelist_path)
    exts = [e for e in wext.detect(nl, run_dir=os.path.dirname(os.path.abspath(namelist_path)))
            if e.get("active")]
    if exts:
        out.note = ("no physics advice: extension case (%s) -- extension/physics parameters are "
                    "not touched (decision 0056/0059)" % ", ".join(e["label"] for e in exts))
        return out

    max_dom = wpredict.get_max_dom(nl)
    dxs = _dx_list(nl, max_dom)
    resolved = _resolved_phys(nl, {}, 0)
    with open(namelist_path) as fh:
        text = fh.read()
    frozen = {str(k).strip().lower() for k in (frozen_keys or ())}
    frozen |= set(cc.case_frozen_keys(cc.CASES.get(case_name)))
    base_rate = _cost_rate(nl, {}, max_dom)

    pool = []   # (id, template, family, ops)
    for c in pp.physics_candidates_from_namelist(text, case_name or "", max_dom,
                                                 cc.PHYSICS_TEMPLATES_JSON_PATH,
                                                 native_physics_resolved=resolved):
        tid = c["template"]
        spec = cc.PHYSICS_TEMPLATES[tid]
        if tid == "P00" or spec.get("direction") in ("accuracy",):
            continue
        if not c["applicable"]:
            out.skipped.append((tid, "not applicable: %s" % c["skip_reason"]))
            continue
        pool.append((tid, tid, c.get("family"), _template_ops(tid, spec, case_name or "", resolved, pp)))
    for cid, ops, fam in _own_ops(nl, resolved, max_dom, dxs):
        pool.append((cid, None, fam, ops))

    cands = []
    for cid, tid, fam, ops in pool:
        hit = cc.template_frozen_key_conflicts({"edits": ops}, frozen) if frozen else []
        if hit:
            out.skipped.append((cid, "touches task-mandated frozen key(s) %s" % ",".join(hit)))
            continue
        try:
            edits, overlay, changes = _edits_and_overlay(text, ops, max_dom, cc)
        except ValueError as e:
            out.skipped.append((cid, "edit not applicable: %s" % e))
            continue
        gain = 100.0 * (1.0 - _cost_rate(nl, overlay, max_dom) / base_rate)
        if gain < MIN_PRIOR_GAIN_PCT:
            out.skipped.append((cid, "cost-weight prior gain %.2f%% < %.1f%% (cheaper scheme not "
                                     "available or lever not modelled)" % (gain, MIN_PRIOR_GAIN_PCT)))
            continue
        label, rationale = _label_rationale(changes, fam, gain, tid)
        cands.append({"id": cid, "label": label, "edits": edits, "rationale": rationale,
                      "class": CLASS, "ops": ops, "max_dom": max_dom, "family": fam,
                      "template": tid, "prior_gain_pct": round(gain, 3)})
    for c in cands:
        c["restart_compatible"] = restart_compatible(c["edits"])
    cands.sort(key=lambda c: (-c["prior_gain_pct"], c["id"]))
    # the `max_candidates` measured slots go to restart-compatible candidates first; scheme swaps
    # (CU-OFF, MP-*, PBL ...) cannot start from the shared restart and are listed, up to
    # `max_estimate_only` of them, as estimate-only (cost-model prior; never measured)
    compat = [c for c in cands if c["restart_compatible"]]
    other = [c for c in cands if not c["restart_compatible"]]
    out.extend(compat[:max_candidates] + other[:max_estimate_only])
    out.skipped.extend((c["id"], "beyond max_candidates=%d" % max_candidates)
                       for c in compat[max_candidates:])
    out.skipped.extend((c["id"], "beyond max_estimate_only=%d" % max_estimate_only)
                       for c in other[max_estimate_only:])
    out.note = "%d candidate(s) (%d measurable from the restart, %d estimate-only); %d skipped" % (
        len(out), len(compat[:max_candidates]), len(other[:max_estimate_only]), len(out.skipped))
    return out


# --------------------------------------------------------------------------------------------
# speed
# --------------------------------------------------------------------------------------------
def weights_from_combine(points_measurements, run_start, run_end, dt, stepra, calibration=None,
                         sample_points=None, eps=1e-3):
    """{point index/at-key: weight}: effective number of whole-run steps each measured point
    stands for, = d(combine().compute_total_s)/d(s_normal_p) by forward difference (combine is
    piecewise-linear in the anchors, so this is exact up to rounding). Keys are
    pm.at.isoformat() (what the orchestrator uses as point id). Compute only: init/IO excluded."""
    base = probe_plan.combine(points_measurements, run_start, run_end, dt, stepra,
                              calibration=calibration, sample_points=sample_points)["compute_total_s"]
    weights = {}
    for i, pm in enumerate(points_measurements):
        bumped = list(points_measurements)
        bumped[i] = dataclasses.replace(pm, s_normal=pm.s_normal + eps)
        tot = probe_plan.combine(bumped, run_start, run_end, dt, stepra, calibration=calibration,
                                 sample_points=sample_points)["compute_total_s"]
        weights[pm.at.isoformat()] = (tot - base) / eps
    return weights


def predicted_speedup(baseline_s_by_point, cand_s_by_point, point_weights):
    """Per-point seconds/step dicts (same point ids) + whole-run weights (weights_from_combine or
    any mapping; normalised here). Returns:
      per_point_ratio       {id: cand/base}  (<1 = faster)
      per_point_saving_pct  {id: 100*(1-ratio)}
      ratio_min, ratio_max  over points
      saving_pct            whole-run, sum_p w_p*(base_p-cand_p) / sum_p w_p*base_p * 100
      speedup               whole-run base/cand
      spread_pct            max-min of per-point saving_pct (point-to-point disagreement)
      n_points, dropped     points used / ids missing from an input or with weight<=0
    Raises ValueError when no point is usable or a time is non-positive."""
    ids, dropped = [], []
    for k in baseline_s_by_point:
        if k in cand_s_by_point and (point_weights or {}).get(k, 0.0) > 0.0:
            ids.append(k)
        else:
            dropped.append(k)
    dropped += [k for k in cand_s_by_point if k not in baseline_s_by_point]
    if not ids:
        raise ValueError("predicted_speedup: no point has baseline, candidate and weight > 0")
    for k in ids:
        if not (baseline_s_by_point[k] > 0 and cand_s_by_point[k] > 0):
            raise ValueError("predicted_speedup: non-positive s/step at point %r" % (k,))
    wsum = sum(point_weights[k] for k in ids)
    base = sum(point_weights[k] * baseline_s_by_point[k] for k in ids) / wsum
    cand = sum(point_weights[k] * cand_s_by_point[k] for k in ids) / wsum
    ratios = {k: cand_s_by_point[k] / baseline_s_by_point[k] for k in ids}
    savings = {k: 100.0 * (1.0 - r) for k, r in ratios.items()}
    return {"per_point_ratio": ratios, "per_point_saving_pct": savings,
            "ratio_min": min(ratios.values()), "ratio_max": max(ratios.values()),
            "saving_pct": 100.0 * (1.0 - cand / base), "speedup": base / cand,
            "spread_pct": max(savings.values()) - min(savings.values()),
            "n_points": len(ids), "dropped": sorted(map(str, dropped))}


# --------------------------------------------------------------------------------------------
# fidelity
# --------------------------------------------------------------------------------------------
def calibrate_tolerance(ref_wrfout, accepted_wrfout, *, fields=None, safety=3.0, time_index=-1):
    """tol_rel = max rel_rms over `fields` between a reference and a run the user already accepts
    (e.g. the same physics at another decomposition) x safety -- the gate's own `calibrate` rule."""
    res = gate.compare(ref_wrfout, accepted_wrfout, fields=list(fields or GATE_DEFAULT_FIELDS),
                       time_index=time_index)
    finite = [r["rel_rms"] for r in res if math.isfinite(r["rel_rms"])]
    if not finite or len(finite) != len(res):
        raise ValueError("calibrate_tolerance: non-finite or missing fields; refusing to calibrate")
    return max(finite) * safety


def _last_path(x):
    """The single wrfout path of `x` (the last/end frame of a list of window frames)."""
    return str(x if isinstance(x, (str, os.PathLike)) else list(x)[-1])


def _paths(x):
    return [x] if isinstance(x, (str, os.PathLike)) else list(x)


def _chaos_floor_verdict(baseline, cand, twin, family, out_prefix, top, thresholds,
                         judged_fields=None):
    own_tmp = out_prefix is None
    tmp = tempfile.TemporaryDirectory(prefix="wtune-fidelity-") if own_tmp else None
    try:
        prefix = os.path.join(tmp.name, "gate") if own_tmp else out_prefix
        payload = chaos_gate.run_gate(_paths(baseline), _paths(twin), _paths(cand), family,
                                      5, thresholds, prefix)
    finally:
        if tmp is not None:
            tmp.cleanup()
    label = payload["label"]
    per = payload.get("per_field_summary_at_last_time") or {}
    rows = {k: v for k, v in per.items() if v.get("status") == "ok"}
    snapshot, deciding = {}, payload.get("deciding")
    if judged_fields is not None and label != chaos_gate.LABEL_INVALID:
        # judge only the prognostic state fields; the others (OLR, TSK ...) are radiation
        # snapshots, reported separately and never judged
        keep = {k for k in judged_fields}
        snapshot = {k: {"floor_ratio": v.get("floor_ratio"), "nd": v.get("nd")}
                    for k, v in rows.items() if k not in keep}
        rows = {k: v for k, v in rows.items() if k in keep}
        if rows:
            label, deciding = chaos_gate.decide_label(rows, payload["thresholds"])
        else:
            label, deciding = chaos_gate.LABEL_INVALID, "no prognostic field comparable"
    order = sorted(rows.items(), key=lambda kv: chaos_gate._floor_sort_key(kv[1]), reverse=True)
    worst = [{"name": k, "floor_ratio": v.get("floor_ratio"), "nd": v.get("nd"),
              "rms_diff": v.get("rms_diff"), "floor": v.get("floor"),
              "rel_rms": v.get("nd")} for k, v in order[:top]]
    return {"passed": label in (chaos_gate.LABEL_WITHIN_FLOOR, chaos_gate.LABEL_SMALL_DRIFT),
            "verdict": label, "label": label, "gate": "chaos-floor",
            "tolerance_source": "chaos floor (twin = same physics, other decomposition)",
            "deciding": deciding, "last_time": payload.get("last_time"),
            "radiation_snapshot_not_judged": snapshot,
            "n_fields": len(rows), "failed_fields": [], "worst": worst,
            "per_field": payload.get("per_field_summary_at_last_time") or {},
            "thresholds": payload.get("thresholds"),
            "reason": "" if label != chaos_gate.LABEL_INVALID else (deciding or "")}


def fidelity_verdict(baseline_wrfout, cand_wrfout, *, twin=None, fields=None, tol_rel=None,
                     tol_abs=0.0, time_index=-1, top=3, family="all", out_prefix=None,
                     thresholds=None, judged_fields=None):
    """PRIMARY (twin given): decision 0051 chaos-floor gate (fidelity_gate.run_gate) -- the label
    WITHIN-FLOOR / SMALL-DRIFT / LARGE-DRIFT / INVALID against how far `twin` (same physics, other
    decomposition, same restart) drifts from the baseline. passed = WITHIN-FLOOR or SMALL-DRIFT.
    Each path argument may be one wrfout NetCDF or a list (matched by `Times`). Returns label,
    verdict(=label), deciding, per_field (last matched post-init time: floor_ratio, nd, bias...),
    worst (top fields by floor ratio; `rel_rms` there is nd = rms_diff/std(ref)). run_gate writes
    <out_prefix>.json/.csv (a temp dir, discarded, when out_prefix is None).

    FALLBACK (no twin): wtune.config_gate.compare + per-field rel_rms tolerance at the last record,
    labelled provisional: gate='config-gate', verdict 'PASS'|'FAIL', label None, tol_rel None ->
    DEFAULT_TOL_REL ('provisional-default', NOT calibrated); W excluded by default; zero comparable
    fields fails."""
    if twin is not None:
        return _chaos_floor_verdict(baseline_wrfout, cand_wrfout, twin, family, out_prefix, top,
                                    thresholds, judged_fields)
    source = "caller" if tol_rel is not None else "provisional-default"
    tol = DEFAULT_TOL_REL if tol_rel is None else float(tol_rel)
    # wtune.config_gate.compare takes ONE wrfout file (last record): a window gives a list of frames
    # (mid + end), so compare the END frames. (Passing the list through str() made the open fail:
    # "['/home/.../wrfout_d01_..._00:29:00.nc', ...]" -- live SC26-P1 run.)
    results = gate.compare(_last_path(baseline_wrfout), _last_path(cand_wrfout),
                           fields=list(fields) if fields is not None else list(GATE_DEFAULT_FIELDS),
                           time_index=time_index)
    failed = gate._check_results(results, tol, tol_abs)
    worst = sorted(results, key=lambda r: r["rel_rms"], reverse=True)[:top]
    reason = "" if results else "no comparable fields in the baseline wrfout"
    passed = bool(results) and not failed
    return {"passed": passed, "verdict": "PASS" if passed else "FAIL", "label": None,
            "gate": "config-gate", "tol_rel": tol,
            "tol_abs": tol_abs, "tolerance_source": source, "n_fields": len(results),
            "failed_fields": failed,
            "worst": [{k: r[k] for k in ("name", "rel_rms", "max_abs", "status", "reason")}
                      for r in worst],
            "reason": reason}


# --------------------------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------------------------
def advice_table(candidates, speed_results, fidelity_results):
    """One row per candidate (keyed by id into the two result dicts; either may lack an id):
    id, label, rationale, class, predicted_saving_pct, ratio_min, ratio_max, fidelity
    ('pass'|'fail'|'not run' from the provisional config-gate; the chaos-floor label WITHIN-FLOOR|
    SMALL-DRIFT|LARGE-DRIFT|INVALID when a twin was used), fidelity_state, fidelity_gate, worst_field, worst_rel_rms, recommendation, note.
    recommendation: 'consider' (saving >= 5% and fidelity pass); 'not worth it' (saving < 5%);
    'changes results beyond tolerance' (saving >= 5%, fidelity fail); 'not measured' (no speed
    result); 'fidelity not checked' (saving >= 5%, no fidelity result). Sorted by predicted saving
    descending (unmeasured last), stable. Every row: 'advice only -- not applied'."""
    rows = []
    for i, c in enumerate(candidates):
        sp = (speed_results or {}).get(c["id"])
        fd = (fidelity_results or {}).get(c["id"])
        saving = sp["saving_pct"] if sp else None
        estimate_only = c.get("restart_compatible") is False and sp is None
        fid_show = None
        if fd is None:
            fid, worst, worst_rms = "not run", None, None
        else:
            label = fd.get("label")
            if label == "INVALID":
                fid = "invalid"
            elif label == "LARGE-DRIFT":
                fid = "fail"
            else:
                fid = "pass" if fd["passed"] else "fail"
            if label:
                fid_show = label
            w = fd["worst"][0] if fd["worst"] else None
            worst, worst_rms = (w["name"], w["rel_rms"]) if w else (None, None)
        if estimate_only:
            rec = ESTIMATE_ONLY
        elif saving is None:
            rec = "not measured"
        elif saving < MIN_SAVING_PCT:
            rec = "not worth it"
        elif fid == "fail":
            rec = "changes results beyond tolerance"
        elif fid == "pass":
            rec = "consider"
        else:
            rec = "fidelity not checked"
        rows.append({"id": c["id"], "label": c["label"], "rationale": c["rationale"],
                     "class": CLASS, "predicted_saving_pct": saving,
                     "ratio_min": sp["ratio_min"] if sp else None,
                     "ratio_max": sp["ratio_max"] if sp else None,
                     "fidelity": fid_show or fid, "fidelity_state": fid,
                     "fidelity_gate": (fd or {}).get("gate"), "worst_field": worst, "worst_rel_rms": worst_rms,
                     "recommendation": rec, "note": ADVICE_ONLY,
                     "measured": sp is not None,
                     "prior_saving_pct": c.get("prior_gain_pct"),
                     "radiation_snapshot_not_judged": (fd or {}).get("radiation_snapshot_not_judged"),
                     "_i": i})
    rows.sort(key=lambda r: (r["predicted_saving_pct"] is None,
                             -(r["predicted_saving_pct"] or 0.0), r["_i"]))
    for r in rows:
        del r["_i"]
    return rows
