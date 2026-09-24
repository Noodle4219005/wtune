#!/usr/bin/env python3
"""predictor/physics_holdout_eval.py -- honest leave-one-case-out evaluation of the PHYSICS-GRAY
recommendation LOGIC (recommend.py --compliance PHYSICS-GRAY / physics.predict_physics_shortlist)
against REAL measured physics-campaign outcomes (decision 0051).

For every case_base present in the given --ledger file(s)' mode=physics rows, this holds that
case's own rows out of training entirely (the actual bug-1 fix: --holdout-case now really is
applied to physics rows -- see recommend.recommend_physics_gray), fits on the OTHER cases' rows,
and asks predict_physics_shortlist() for a shortlist restricted to the templates that are BOTH
applicable to the held-out case (physics.physics_candidates_for_case) AND were actually measured
for it in the ledger (so every comparison below has real ground truth, never a guess). For each
such template it compares:

  - predicted vs measured fidelity label
  - predicted vs measured time-to-solution percent change vs P00 (negative = faster)
  - predicted status (RECOMMENDABLE/UNCONFIRMED) vs "truly acceptable & feasible" (measured label
    in --allowed-labels AND not every replicate FAILED)

and flags a "dangerous recommendation": predicted RECOMMENDABLE but NOT truly acceptable &
feasible (measured LARGE-DRIFT, or failed) -- exactly the failure mode the real physics-campaign
smoke (job 1077893) exposed for lafe/P05 and P03's crashes.

Writes <out-dir>/holdout_eval_metrics.csv (one row per case x template) and
<out-dir>/holdout_eval_summary.md (per-case table + aggregate counts).

CLI:
    python3 physics_holdout_eval.py --ledger ledger_karina_....csv --ledger ledger_conus25_....csv \\
        ... --out-dir /home/u5727520/wrf-bench/wscore/physics/holdout_eval_<jobid>
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
for _p in (_HERE, _REPO_ROOT, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark                 # predictor/benchmark.py: load_ledger()/write_csv()
import physics as physicsmod     # predictor/physics.py: PHYSICS-GRAY models (decision 0051)
import configs as campaign_configs  # campaign/configs.py: PHYSICS_TEMPLATES_JSON_PATH

DEFAULT_ALLOWED_LABELS = physicsmod.DEFAULT_ALLOWED_LABELS

_NATIVE_FIELDS = ("mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
                   "sf_surface_physics", "ra_lw_physics", "ra_sw_physics", "radt", "cudt",
                   "time_step")
_GRID_FIELDS = ("e_we", "e_sn", "e_vert", "dx", "dy")

CSV_FIELDS = [
    "case", "template", "family", "measured_fidelity_label", "predicted_fidelity_label",
    "fidelity_label_match", "measured_time_to_solution_pct", "predicted_time_to_solution_pct",
    "time_to_solution_abs_error_pct", "measured_failed", "measured_truly_acceptable_feasible",
    "predicted_status", "dangerous_recommendation", "reasons",
]


# ---------------------------------------------------------------------------
# Reconstruct, from a case's own MEASURED ledger rows (never a namelist file -- this script
# evaluates already-collected campaign data, it does not need to re-derive anything from disk),
# exactly the inputs recommend.recommend_physics_gray would have built for a real run: the
# case's native (physics_suite-unresolved) physics dict, the resolved case_features row
# predict_physics_shortlist scores against, and the {knob: verdict} probe table.
# ---------------------------------------------------------------------------
def _native_physics_from_p00_row(p00_row):
    notes = physicsmod.parse_notes(p00_row.get("notes"))
    native = {f: p00_row.get(f) for f in _NATIVE_FIELDS}
    native["physics_suite"] = notes.get("physics_suite")
    return native


def _case_features_row(p00_row, native_resolved):
    row = {f: p00_row.get(f) for f in _GRID_FIELDS}
    row.update(
        hardware_id=p00_row.get("hardware_id"), case_base=None, time_step=p00_row.get("time_step"),
        mp_physics=native_resolved.get("mp_physics"), cu_physics=p00_row.get("cu_physics"),
        bl_pbl_physics=native_resolved.get("bl_pbl_physics"),
        sf_sfclay_physics=native_resolved.get("sf_sfclay_physics"),
        sf_surface_physics=native_resolved.get("sf_surface_physics"),
        ra_lw_physics=native_resolved.get("ra_lw_physics"),
        ra_sw_physics=native_resolved.get("ra_sw_physics"),
        radt=p00_row.get("radt"), cudt=p00_row.get("cudt"),
    )
    return row


def _case_probe_verdicts(case_rows):
    """Union of every candidate row's own recorded probe=<knob>:<verdict> notes for this case --
    a knob's probe verdict is a property of the case + knob value, not of which template happened
    to edit it, so this reconstructs the same {knob: verdict} table a real 'wtune.py screen' run
    for this case would have produced (no saved screen file needed -- the verdict was already
    applied and recorded per candidate at run time)."""
    verdicts = {}
    for r in case_rows:
        notes = physicsmod.parse_notes(r.get("notes"))
        for k, v in physicsmod.parse_probe(notes.get("probe")).items():
            verdicts[k] = v
    return verdicts


def _reference_group_for_case(all_rows, case):
    """(group_key, physics_groups()[key], performance_targets()[key]) for the one physics group of
    this case that actually has candidate (non-P00) rows attached -- i.e. the REFERENCE-decomp
    group, never the twin-decomp group (which is P00-only by construction, see physics.py's
    physics_groups docstring). (None, None, None) if this case has no candidates at all."""
    groups = physicsmod.physics_groups(all_rows)
    perf = physicsmod.performance_targets(all_rows)
    for key, g in groups.items():
        if key[0] == case and perf.get(key, {}).get("templates"):
            return key, g, perf[key]
    return None, None, None


def _measured_outcomes(perf_group):
    """{template: {"y": float|None, "n_ok": int, "n_failed": int, "label": str|None}} from one
    performance_targets() group entry -- "label" is the rep-1 row's fidelity_label (None if never
    labeled, e.g. every replicate FAILED, or the label is INVALID/absent)."""
    out = {}
    for tmpl, tdata in perf_group["templates"].items():
        rows_t = tdata["rows"]
        label_row = next((r for r in rows_t if r.get("rep_index") == 1), rows_t[0])
        label = label_row.get("fidelity_label") or None
        if label not in physicsmod.FIDELITY_ORDER:
            label = None
        out[tmpl] = dict(y=tdata["y"], n_ok=tdata["n_ok"], n_failed=tdata["n_failed"], label=label)
    return out


# ---------------------------------------------------------------------------
# One held-out case.
# ---------------------------------------------------------------------------
def evaluate_case(all_rows, case, allowed_labels=DEFAULT_ALLOWED_LABELS, max_infeasible_rate=0.0,
                   max_dangerous_miss=0.0,
                   max_template_large_rate=physicsmod.DEFAULT_MAX_TEMPLATE_LARGE_RATE,
                   min_speedup_pct=physicsmod.DEFAULT_MIN_SPEEDUP_PCT, templates_json_path=None):
    """Run the PHYSICS-GRAY recommendation logic for `case` held fully out of training, scored
    against `case`'s own real measured outcomes. Returns None when this case has no measured
    candidate templates at all (nothing to evaluate), else a dict with "case", "per_template"
    (list of CSV_FIELDS-shaped dicts), "top1" (the first per_template row with
    predicted_status=="RECOMMENDABLE", or None), and "n_dangerous"/"n_templates_evaluated"."""
    templates_json_path = templates_json_path or campaign_configs.PHYSICS_TEMPLATES_JSON_PATH

    key, group, perf_group = _reference_group_for_case(all_rows, case)
    if key is None:
        return None
    measured = _measured_outcomes(perf_group)

    p00_row = group["baseline"][0]
    native_physics = _native_physics_from_p00_row(p00_row)
    native_resolved = physicsmod.resolve_physics_suite(native_physics, native_physics.get("physics_suite"))
    case_features_row = _case_features_row(p00_row, native_resolved)

    case_rows = [r for r in physicsmod.physics_rows(all_rows) if r.get("case_base") == case]
    probe_verdicts = _case_probe_verdicts(case_rows)

    raw_candidates = physicsmod.physics_candidates_for_case(native_physics, case, 1, templates_json_path)
    candidates = [c for c in raw_candidates if c["template"] != "P00" and c.get("applicable", True)
                  and c["template"] in measured]
    if not candidates:
        return None

    # The actual defect-1 fix, exercised here directly: `case`'s own rows are dropped from
    # training BEFORE fitting anything, exactly as recommend.recommend_physics_gray now does for
    # --holdout-case.
    train_rows = [r for r in all_rows if r.get("case_base") != case]
    physicsmod.attach_change_context(train_rows)
    physicsmod.add_perf_targets(train_rows)
    physicsmod.add_fidelity_targets(train_rows)

    predicted = physicsmod.predict_physics_shortlist(
        train_rows, case_features_row, candidates, allowed_labels=allowed_labels,
        probe_verdicts=probe_verdicts, max_infeasible_rate=max_infeasible_rate,
        max_dangerous_miss=max_dangerous_miss, max_template_large_rate=max_template_large_rate,
        min_speedup_pct=min_speedup_pct)

    per_template = []
    for c in predicted:
        tmpl = c["template"]
        m = measured[tmpl]
        measured_tts_pct = ((math.exp(m["y"]) - 1) * 100.0) if m["y"] is not None else None
        predicted_tts_pct = c["predicted_time_to_solution_pct"]
        tts_abs_error_pct = (abs(predicted_tts_pct - measured_tts_pct)
                              if predicted_tts_pct is not None and measured_tts_pct is not None
                              else None)
        truly_acceptable_feasible = (m["label"] in allowed_labels) and (m["n_ok"] > 0)
        dangerous = (c["status"] == "RECOMMENDABLE") and not truly_acceptable_feasible
        per_template.append(dict(
            case=case, template=tmpl, family=c.get("family"),
            measured_fidelity_label=m["label"], predicted_fidelity_label=c["predicted_fidelity_label"],
            fidelity_label_match=(c["predicted_fidelity_label"] == m["label"]),
            measured_time_to_solution_pct=measured_tts_pct,
            predicted_time_to_solution_pct=predicted_tts_pct,
            time_to_solution_abs_error_pct=tts_abs_error_pct,
            measured_failed=(m["n_ok"] == 0),
            measured_truly_acceptable_feasible=truly_acceptable_feasible,
            predicted_status=c["status"], dangerous_recommendation=dangerous,
            reasons="; ".join(c.get("reasons") or []),
        ))

    # predict_physics_shortlist's own output is already sorted RECOMMENDABLE-first, then by
    # predicted_y ascending (faster first) -- the first RECOMMENDABLE row here IS the same top-1 a
    # real shortlist[:top_k] would show.
    top1 = next((r for r in per_template if r["predicted_status"] == "RECOMMENDABLE"), None)
    n_dangerous = sum(1 for r in per_template if r["dangerous_recommendation"])
    return dict(case=case, per_template=per_template, top1=top1, n_dangerous=n_dangerous,
                n_templates_evaluated=len(per_template))


# ---------------------------------------------------------------------------
# All cases + report.
# ---------------------------------------------------------------------------
def run_holdout_eval(ledger_paths, out_dir, allowed_labels=DEFAULT_ALLOWED_LABELS,
                      max_infeasible_rate=0.0, max_dangerous_miss=0.0,
                      max_template_large_rate=physicsmod.DEFAULT_MAX_TEMPLATE_LARGE_RATE,
                      min_speedup_pct=physicsmod.DEFAULT_MIN_SPEEDUP_PCT, templates_json_path=None):
    all_rows = []
    for p in ledger_paths:
        all_rows.extend(benchmark.load_ledger(p))
    cases = sorted({r.get("case_base") for r in physicsmod.physics_rows(all_rows)
                    if r.get("case_base") is not None})

    case_results = []
    for case in cases:
        res = evaluate_case(list(all_rows), case, allowed_labels=allowed_labels,
                             max_infeasible_rate=max_infeasible_rate,
                             max_dangerous_miss=max_dangerous_miss,
                             max_template_large_rate=max_template_large_rate,
                             min_speedup_pct=min_speedup_pct,
                             templates_json_path=templates_json_path)
        if res is not None:
            case_results.append(res)

    os.makedirs(out_dir, exist_ok=True)
    csv_rows = [row for res in case_results for row in res["per_template"]]
    benchmark.write_csv(os.path.join(out_dir, "holdout_eval_metrics.csv"), csv_rows, CSV_FIELDS)

    n_cases = len(case_results)
    n_with_top1 = sum(1 for r in case_results if r["top1"] is not None)
    n_top1_ok = sum(1 for r in case_results
                     if r["top1"] is not None and r["top1"]["measured_truly_acceptable_feasible"])
    total_dangerous = sum(r["n_dangerous"] for r in case_results)
    fidelity_pairs = [row for row in csv_rows if row["measured_fidelity_label"] is not None
                       and row["predicted_fidelity_label"] is not None]
    n_fidelity_match = sum(1 for row in fidelity_pairs if row["fidelity_label_match"])

    lines = ["# WTune PHYSICS-GRAY leave-one-case-out holdout evaluation\n"]
    lines.append("%d case(s) evaluated, %d template-candidate row(s) total (see "
                  "holdout_eval_metrics.csv for the full per-row detail).\n" % (n_cases, len(csv_rows)))
    lines.append("## Per case\n")
    lines.append("| case | top-1 RECOMMENDABLE | truly acceptable & feasible | dangerous "
                  "recommendations | templates evaluated |")
    lines.append("|---|---|---|---|---|")
    for r in case_results:
        top1_str = r["top1"]["template"] if r["top1"] else "(none)"
        ok_str = ("yes" if r["top1"] and r["top1"]["measured_truly_acceptable_feasible"] else
                  "no" if r["top1"] else "n/a")
        lines.append("| %s | %s | %s | %d | %d |" % (
            r["case"], top1_str, ok_str, r["n_dangerous"], r["n_templates_evaluated"]))

    lines.append("\n## Aggregate\n")
    lines.append("- cases with a top-1 RECOMMENDABLE template: %d/%d" % (n_with_top1, n_cases))
    lines.append("- of those, top-1 was truly acceptable & feasible: %d/%d"
                  % (n_top1_ok, n_with_top1 if n_with_top1 else 1))
    lines.append("- dangerous recommendations (RECOMMENDABLE but measured LARGE-DRIFT or failed), "
                  "across every template x case, not just top-1: %d" % total_dangerous)
    lines.append("- fidelity label match rate (both predicted and measured labels known): %d/%d"
                  % (n_fidelity_match, len(fidelity_pairs) if fidelity_pairs else 1))
    with open(os.path.join(out_dir, "holdout_eval_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("wrote holdout_eval_metrics.csv (%d rows), holdout_eval_summary.md to %s -- "
          "%d/%d cases had a top-1 RECOMMENDABLE (%d truly acceptable & feasible), "
          "%d dangerous recommendation(s) total"
          % (len(csv_rows), out_dir, n_with_top1, n_cases, n_top1_ok, total_dangerous))
    return dict(n_cases=n_cases, n_with_top1=n_with_top1, n_top1_ok=n_top1_ok,
                total_dangerous=total_dangerous, n_fidelity_match=n_fidelity_match,
                n_fidelity_pairs=len(fidelity_pairs), case_results=case_results)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Leave-one-case-out holdout evaluation of PHYSICS-GRAY recommendations "
                     "against real measured physics-campaign outcomes.")
    ap.add_argument("--ledger", action="append", required=True, dest="ledgers", metavar="PATH",
                     help="repeatable; a physics-campaign ledger CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allowed-labels", default="WITHIN-FLOOR,SMALL-DRIFT", dest="allowed_labels")
    ap.add_argument("--max-infeasible-rate", type=float, default=0.0, dest="max_infeasible_rate")
    ap.add_argument("--max-dangerous-miss", type=float, default=0.0, dest="max_dangerous_miss")
    ap.add_argument("--max-template-large-rate", type=float,
                     default=physicsmod.DEFAULT_MAX_TEMPLATE_LARGE_RATE,
                     dest="max_template_large_rate")
    ap.add_argument("--min-speedup-pct", type=float, default=physicsmod.DEFAULT_MIN_SPEEDUP_PCT,
                     dest="min_speedup_pct")
    ap.add_argument("--templates-json", default=None, dest="templates_json")
    args = ap.parse_args(argv)
    allowed = tuple(s.strip() for s in args.allowed_labels.split(",") if s.strip())
    run_holdout_eval(args.ledgers, args.out_dir, allowed_labels=allowed,
                      max_infeasible_rate=args.max_infeasible_rate,
                      max_dangerous_miss=args.max_dangerous_miss,
                      max_template_large_rate=args.max_template_large_rate,
                      min_speedup_pct=args.min_speedup_pct,
                      templates_json_path=args.templates_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
