"""The tune report: ranking table, the winner's parameter table under the STRICT guard (decision
0056), the text rendering of every section, and its emission to stdout + tune_report.json.

`report` stays a plain dict: it is the JSON boundary that run(), the tests and wtune_run read.
"""
from __future__ import annotations

import json
import math
import os

from ._deps import wc, wtune_extensions, wtune_predict
from .candidates import rank_predicted_entries
from .common import _fmt_hms
from .context import Ctx
from .estimates import format_level_estimates
from .jobs import MAXRSS_NOTE
from .prediction import predict_candidates
from .preflight import render_node_health_text
from . import prediction as _prediction

LAYOUT_ROWS_SHOWN = 40               # rows of node_selection.layouts printed (the JSON has all)


def local_param_table(config: dict, nodes: int) -> dict:
    """Parameter table of `config` at `nodes`: namelist lines, the mpirun line of campaign/sites/
    scc.sh site_launch (cyclic adds --rank-by node) and its env. wtune_predict.build_param_table
    is not used: it hardcodes dist=cyclic."""
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


def _assert_report_is_strict_only(ctx: Ctx, param_table: dict) -> None:
    """Decision 0056: every namelist key the recommended configuration changes must be STRICT
    (wtune_compliance) and no extension-owned group (&chem, &wrfvar*) may change at all. Skipped
    without wtune_predict or a readable namelist; raises wc.ComplianceError."""
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


def build_report(ctx: Ctx, nodes: int, nodes_note: str, all_results: dict,
                 confirm_results: dict | None, anchor_cid: str | None,
                 full_prediction: bool = True) -> dict:
    """The t0 report dict: candidates (confirm results when present) ranked by predicted
    whole-run time, the best one's parameter table (STRICT-guarded) and, unless
    `full_prediction` is False, its full-run prediction text."""
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
            # confirmed only with ALL confirm_reps measured (a cancelled confirm can leave fewer)
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
    param_table = local_param_table(best_config, nodes) if best_config is not None else None
    if param_table is not None:
        _assert_report_is_strict_only(ctx, param_table)

    return dict(
        case=ctx.case, nodes=nodes, nodes_note=nodes_note, cores_per_node=ctx.cores_per_node,
        budget_minutes=(ctx.budget_s / 60.0 if math.isfinite(ctx.budget_s) else None),
        elapsed_minutes=ctx.elapsed() / 60.0,
        job_ids=list(ctx.owned_jobs), cancelled_job_ids=list(ctx.cancelled_jobs),
        anchor_id=anchor_cid, ranking=table, best=best, param_table=param_table,
        full_run_prediction=(_prediction.build_full_run_prediction(ctx, best_config, nodes,
                                                                   all_results, confirm_results)
                             if full_prediction else None),
        ranking_start_note=ctx.ranking_start_note,
    )


def _ranking_lines(report: dict) -> list[str]:
    """The ranking table (predicted whole-run time first) and the failed predictions."""
    lines = ["%-10s %12s %6s %4s %5s %5s %5s %7s %10s %14s %7s %8s %4s %-11s %s"
             % ("id", "pred_total", "ranks", "omp", "nx", "ny", "tiles", "dist", "median_s",
                "range_s", "speedup", "pred_spd", "n", "pred_source", "confirmed")]
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
    return lines


def render_report_text(report: dict) -> str:
    """The printed t0 / node-search report, with the node selection, node health and pipeline
    sections when present."""
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
    lines += _ranking_lines(report)
    if report["best"] and report["param_table"]:
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


def _mb(v: float | None) -> str:
    return "%.0f" % v if v is not None else "-"


def _node_table_lines(ns: dict) -> list[str]:
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
                        _mb(r.get("maxrss_mb"))))
    for obj, c in (ns.get("choices") or {}).items():
        lines.append("choice %-5s%s N=%s  %s" % (obj, "*" if obj == ns.get("objective") else " ",
                                                 c["nodes"], c["note"]))
    lines.append("(* = objective used; finalists searched: %s; final N=%s cores/node=%s)"
                 % (ns.get("finalists"), ns.get("final", ns.get("chosen")),
                    ns.get("final_cores_per_node", "-")))
    return lines


def _core_table_lines(ns: dict) -> list[str]:
    lines = []
    for n, rows in sorted((ns.get("core_tables") or {}).items(), key=lambda kv: int(kv[0])):
        lines.append("")
        lines.append("--- cores per node at N=%s (ANCHOR; idle cores = hardware - c) ---" % n)
        lines.append("%7s %10s %4s %11s %6s %8s %9s"
                     % ("cores", "ranks/node", "omp", "median_s", "reps", "spread", "maxrss_MB"))
        for r in rows:
            lines.append("%7d %10d %4d %11.4f %6d %7.1f%% %9s %s"
                         % (r["cores"], r["ranks_per_node"], r["omp"], r["median"], r["n"],
                            r["spread"] * 100, _mb(r.get("maxrss_mb")),
                            "<- chosen" if r.get("chosen") else ""))
    return lines


def _layout_lines(ns: dict) -> list[str]:
    """The measured layouts (first LAYOUT_ROWS_SHOWN), the winner, the runner-up and whether
    they differ beyond noise."""
    rows = ns["layouts"]
    lines = ["", "--- measured layouts (%d; best first; the full table is in tune_report.json "
             "node_selection.layouts) ---" % len(rows),
             "%4s %5s %6s %4s %9s %5s %11s %7s %5s  %s"
             % ("N", "c/n", "ranks", "omp", "nx x ny", "tiles", "median_s", "spread",
                "reps", "source")]
    for r in rows[:LAYOUT_ROWS_SHOWN]:
        lines.append("%4d %5d %6s %4s %9s %5s %11.4f %6.1f%% %5d  %s"
                     % (r["nodes"], r["cores_per_node"], r["ranks"], r["omp"],
                        "%sx%s" % (r["nx"], r["ny"]), r["numtiles"], r["median"],
                        r["spread"] * 100, r["reps"], r["source"]))
    if len(rows) > LAYOUT_ROWS_SHOWN:
        lines.append("  ... %d more" % (len(rows) - LAYOUT_ROWS_SHOWN))
    w, ru, cmp_ = ns.get("winner"), ns.get("runner_up"), ns.get("winner_vs_runner_up")
    fmt = lambda r: ("N=%d c/n=%d ranks=%s omp=%s %sx%s tiles=%s %.4f s/step"   # noqa: E731
                     % (r["nodes"], r["cores_per_node"], r["ranks"], r["omp"], r["nx"], r["ny"],
                        r["numtiles"], r["median"]))
    if w:
        lines.append("WINNER:    " + fmt(w))
        if w.get("dirty"):
            lines.append("WARNING: the winner's number comes from a dirty node (%s); the "
                         "ranking may be wrong -- clean the node and tune again"
                         % "; ".join(w["dirty"]))
    n_dirty = sum(1 for r in rows if r.get("dirty"))
    if n_dirty:
        lines.append("note: %d of %d measured layouts were taken on a dirty node"
                     % (n_dirty, len(rows)))
    if ru:
        lines.append("RUNNER-UP: " + fmt(ru))
    if cmp_:
        lines.append("winner vs runner-up: %.1f%% apart -> %s (noise floor %.0f%%)"
                     % (cmp_["gap"] * 100,
                        "differ BEYOND noise" if cmp_["beyond_noise"]
                        else "NOT distinguishable from noise: either is fine",
                        cmp_["noise"] * 100))
    return lines


def _untried_lines(ns: dict) -> list[str]:
    """The enumeration counts, the untried layouts by reason, the MaxRSS caveat and the node
    counts / cores not run."""
    lines = []
    for k, v in sorted((ns.get("enumerated") or {}).items()):
        lines.append("enumerated %s: %d raw, %d unique (block, 1-D tiles), %d legal, %d excluded%s"
                     % (k, v["raw"], v["unique"], v["legal"], v["excluded"],
                        " (%s)" % "; ".join(v["excluded_reasons"][:2]) if v["excluded_reasons"]
                        else ""))
    untried = ns.get("untried") or []
    if untried:
        reasons: dict = {}
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
    return lines


def render_node_selection_text(ns: dict) -> str:
    """The node-selection section of the report (report["node_selection"])."""
    lines = _node_table_lines(ns) + _core_table_lines(ns)
    if ns.get("layouts"):
        lines += _layout_lines(ns)
    lines += _untried_lines(ns)
    return "\n".join(lines)


def _pipeline_phase_lines(pipe: dict) -> list[str]:
    lines = []
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
    return lines


def _pipeline_prediction_lines(pipe: dict) -> list[str]:
    """Band, --time suggestion, short-run note, phase-2 plan, physics gate and the prediction."""
    lines = []
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
        rng = pr.get("predicted_range_s")
        where = "at N=%d cores/node=%s %s" % (pr["nodes"], pr["cores_per_node"], pr["config"])
        if rng:
            lines.append("predicted whole run: %s (range %s - %s) %s; suggested --time=%s"
                         % (pr["predicted_total_hms"], _fmt_hms(rng[0]), _fmt_hms(rng[1]),
                            where, pr["time_suggestion"]))
        else:
            lines.append("predicted whole run: unavailable (run length unknown) %s" % where)
        if pr.get("breakdown"):
            lines.append("  " + pr["breakdown"] + ("  [init: %s]" % pr["init_source"]
                                                   if pr.get("init_source") else ""))
        if pr.get("segment_note"):
            lines.append("  NOTE: " + pr["segment_note"])
    return lines


def render_pipeline_text(pipe: dict) -> str:
    """The -O pipeline section (report["pipeline"]): level and estimates, phases, prediction,
    physics advice and the running / wall / queue / controller totals."""
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
    lines += _pipeline_phase_lines(pipe)
    lines += _pipeline_prediction_lines(pipe)
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


def _attach_advice(report: dict) -> str:
    """Decision 0061: the advice-only tail (report["advice"] + its text). Never fails the tune:
    an error goes to report["advice_error"]."""
    try:
        import wtune_advise          # looked up at call time (tests swap it in sys.modules)
        res = wtune_advise.advise_case_id(report["case"])
        report["advice"] = res["advice"]
        report["advice_unavailable"] = res["unavailable"]
        return "\n\n" + wtune_advise.render_text(res)
    except Exception as e:                                  # noqa: BLE001
        report["advice"] = []
        report["advice_error"] = "%s: %s" % (type(e).__name__, e)
        return ""


def final_winner(report: dict) -> dict | None:
    """report["winner"]: the final pick (nodes + best, after a phase-2 overturn too) as its
    node_selection.layouts row, else a row built from the best ranking row; None for an
    unmeasured pick (O0, aborted)."""
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


def _emit_report(report: dict, out_dir: str, node_health: dict | None = None) -> None:
    """Print the report (+ advice) and write <out_dir>/tune_report.json (sorted keys)."""
    if node_health and "node_health" not in report:
        report["node_health"] = node_health
    report["winner"] = final_winner(report)
    only = report.get("pipeline_aborted") or report.get("pipeline_only")
    advice_text = "" if only else _attach_advice(report)
    print("")
    print(render_pipeline_text(report["pipeline"]) if only
          else render_report_text(report) + advice_text)
    write_report_json(report, out_dir)


def write_report_json(report: dict, out_dir: str) -> None:
    """<out_dir>/tune_report.json: indented, keys sorted, non-JSON values as str()."""
    with open(os.path.join(out_dir, "tune_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

