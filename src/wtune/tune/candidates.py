"""Decomposition candidates: the shared legality / --config filter, diverse and full-space
enumeration, sweep anchors, and aggregation / ranking of measured rows."""
from __future__ import annotations

import contextlib
import dataclasses
import io
import re
import statistics

import wtune.config
import wtune.extensions
from ._deps import campaign_configs, decomp_space, sweep
from .common import _cluster_rank, _log_key, _notes_kv, parse_run_id
from . import shortlist
from .context import Ctx, cores_used

FULL_SPACE_CONFIGS_PER_JOB = 24      # configs in one full-space job (~1 h at 20 steps)


def _apply_search_filter(ctx: Ctx, configs: list[dict]) -> list[dict]:
    """The one candidate filter of every generation site: WRF-Chem j-patch guard, extension OMP
    cap, then wtune.config.candidate_allowed (--config search restrictions + exclude rules; a
    no-op without --config). The ANCHOR is exempt from the search restrictions only; dropped
    candidates are printed."""
    out = []
    chem_dropped = []
    omp_cap = getattr(ctx, "omp_cap", None)
    for c in configs:
        exempt = c.get("config_source") == "anchor" or c.get("id") == "ANCHOR"
        chem_e_sn = getattr(ctx, "chem_e_sn", None)
        if chem_e_sn and c.get("ny"):     # PROVISIONAL WRF-Chem narrow-j-patch hang guard
            why = wtune.extensions.chem_j_patch_reason(chem_e_sn, int(c["ny"]), c.get("nx"))
            if why:
                print("config: %s %s" % (c.get("id"), why))
                if not exempt:
                    chem_dropped.append(c)
                continue
        if omp_cap and int(c.get("omp", 1) or 1) > omp_cap:   # dmpar-only build
            if exempt:
                print("config: ANCHOR omp=%s dropped: above the thread cap of this WRF build"
                      % c.get("omp"))
            continue
        if wtune.config.candidate_allowed(c, ctx.search_cfg, ctx.exclude_rules, ctx.is_fire,
                                           exempt_search=exempt):
            out.append(c)
        elif exempt:
            print("config: ANCHOR excluded by an `exclude` rule")
    if chem_dropped and not out:
        # the guard is PROVISIONAL (one case): when it leaves nothing to measure, keep the layout
        # with the widest j-patch (smallest ny) instead of an empty search, and say so
        best = min(chem_dropped, key=lambda c: int(c["ny"]))
        print("config: every candidate is below the WRF-Chem j-patch guard; keeping %s (%sx%s, the "
              "widest j-patch) -- it may hang at the first history write" % (
                  best.get("id"), best.get("nx"), best.get("ny")))
        out.extend(_apply_search_filter(dataclasses.replace(ctx, chem_e_sn=None), [best]))
    return out


def sample_more_diverse_configs(ctx: Ctx, exclude_configs: list[dict], k: int,
                                seed: int) -> list[dict]:
    """Up to `k` not-yet-measured decomp_space candidates (filtered BEFORE sampling, identity =
    scc_sweep._identity): round0's top-up and the adaptive fallback once the proposer is
    exhausted. [] when the pool is exhausted."""
    total_cores = ctx.nodes * cores_used(ctx)
    candidates, _axis = decomp_space.enumerate_space(ctx.case_json, total_cores, ctx.nodes,
                                                       ctx.hardware_id)
    candidates = _apply_search_filter(ctx, candidates)
    seen = {sweep._identity(c) for c in exclude_configs if c}
    fresh = [c for c in candidates if sweep._identity(c) not in seen]
    if not fresh or k <= 0:
        return []
    k = min(k, len(fresh))
    model = shortlist.load(ctx.hardware_id)
    ranked = shortlist.rank(model, fresh, ctx.case_json.get("e_vert", 1))[:(k + 1) // 2] if model else []
    if ranked:       # the machine's ranker picks the first half, the design sample the rest
        taken = {id(c) for c in ranked}
        rest = [c for c in fresh if id(c) not in taken]
        more = decomp_space.sample_design(rest, k - len(ranked), seed, must_include=None)[0] \
            if k > len(ranked) and rest else []
        chosen = ranked + list(more)
        print("config: %d of %d new layouts are the %s ranker's first picks"
              % (len(ranked), len(chosen), ctx.hardware_id))
    else:
        chosen, _coverage = decomp_space.sample_design(fresh, k, seed, must_include=None)
    out = decomp_space.to_campaign_configs(chosen, prefix="D")
    # ids continue after the D ids already used in this tune: a restart at D0001 would overwrite
    # an earlier round's measurement of a different layout
    used = [int(m.group(1)) for c in exclude_configs if c
            for m in [re.match(r"D(\d+)$", str(c.get("id")))] if m]
    for i, c in enumerate(out, start=max(used, default=0) + 1):
        c["id"] = "D%04d" % i
    return out


def aggregate_by_cid(rows: list[dict], configs_by_cid: dict) -> dict:
    """{cid: {cid, config, s_values, median, lo, hi, n, run_ids[, dirty]}} over the rows with a
    real s_per_step_median (a failed run has none and is dropped). run_ids parallel s_values: each is
    its run's RUNDIR basename (see _rsl_paths_for_entry)."""
    by_cid: dict = {}
    dirty: dict = {}
    for r in rows:
        parsed = parse_run_id(r.get("run_id"))
        if not parsed:
            continue
        s = sweep._to_float(r.get("s_per_step_median"))
        if s is None or s <= 0:
            continue
        by_cid.setdefault(parsed["cid"], []).append((s, r.get("run_id")))
        kv = _notes_kv(r.get("notes"))
        if kv.get("node_dirty", "none") != "none":
            dirty.setdefault(parsed["cid"], set()).add("node_dirty=" + kv["node_dirty"])
        if kv.get("foreign_load") == "1":
            dirty.setdefault(parsed["cid"], set()).add("foreign_load=" + kv.get("foreign", "1"))
    out = {}
    for cid, pairs in by_cid.items():
        svals = [p[0] for p in pairs]
        out[cid] = dict(cid=cid, config=configs_by_cid.get(cid), s_values=svals,
                         median=statistics.median(svals), lo=min(svals), hi=max(svals),
                         n=len(svals), run_ids=[p[1] for p in pairs])
        if cid in dirty:          # measured while the node was not clean (hygiene census)
            out[cid]["dirty"] = sorted(dirty[cid])
    return out


def merge_aggs(a: dict, b: dict) -> dict:
    """`a` with the runs of `b` (the same layout measured again) added."""
    s = a["s_values"] + b["s_values"]
    out = dict(a, s_values=s, median=statistics.median(s), lo=min(s), hi=max(s), n=len(s),
               run_ids=(a.get("run_ids") or []) + (b.get("run_ids") or []))
    dirty = sorted(set(a.get("dirty") or []) | set(b.get("dirty") or []))
    if dirty:
        out["dirty"] = dirty
    return out


def rank_predicted_entries(predicted: dict, anchor_cid: str | None) -> list[dict]:
    """predict_candidates() entries by predicted whole-run time (2026-09-24 decision), NOISE-
    clustered (_cluster_rank); entries whose prediction failed follow, ranked by median s/step.
    The tie-break order is the sorted cid set, independent of dict construction order."""
    order_index = {cid: i for i, cid in enumerate(sorted(predicted))}
    ok = [e for e in predicted.values() if e.get("pred_total_s") is not None]
    failed = [e for e in predicted.values() if e.get("pred_total_s") is None]
    ranked_ok = _cluster_rank(ok, lambda e: _log_key(e["pred_total_s"]), anchor_cid, order_index)
    ranked_failed = _cluster_rank(failed, lambda e: _log_key(e["median"]), anchor_cid, order_index)
    return ranked_ok + ranked_failed


def enumerate_full_space(ctx: Ctx) -> tuple[list[dict], dict]:
    """Every legal layout at ctx.nodes x cores_used(ctx) (ranks x omp, nx x ny, numtiles), with
    dist=block and 1-D tiles only (blocktime is a no-op on scc; cyclic / 2-D tiles are left to
    the proposer). Returns (configs with the ANCHOR override first, info: counts + why
    candidates were dropped)."""
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


def _sweep_omp_options(ctx: Ctx, nodes: int, cores: int | None = None) -> list[int]:
    """decomp_space's OpenMP options at `nodes` x `cores`, minus those above the extension cap."""
    opts = decomp_space.default_omp_options(nodes * (cores or ctx.cores_per_node), nodes,
                                            ctx.hardware_id)
    cap = getattr(ctx, "omp_cap", None)
    return [o for o in opts if not cap or o <= cap]


def _sweep_anchor_fitting(ctx: Ctx, nodes: int) -> tuple[dict | None, int | None, str | None]:
    """(ANCHOR, cores per node, None): on all the cores of a node, or, when nothing is legal
    there, on the most cores per node that has a legal ANCHOR -- a small grid on big nodes
    with a low thread cap (91x100 cuts into 90 ranks at most; a 144-core node at omp 1 wants
    144). (None, None, reason) when no core count has one."""
    anchor, why = _sweep_anchor(ctx, nodes)
    if anchor is not None:
        return anchor, ctx.cores_per_node, None
    for c in range(ctx.cores_per_node - 1, 0, -1):
        anchor, _why = _sweep_anchor(ctx, nodes, cores=c)
        if anchor is not None:
            return anchor, c, None
    return None, None, why


def _hybrid_anchor(ctx: Ctx, nodes: int, omp: int, cores: int | None = None) -> dict | None:
    """ANCHOR at `nodes` x `cores` (default all) cores/node: ranks = nodes*cores/omp, numtiles =
    omp, the least-aspect legal (nx, ny), dist=block (scc_sweep.build_anchor conventions); None
    when no shape is legal."""
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


def _sweep_anchor(ctx: Ctx, nodes: int, cores: int | None = None) -> tuple[dict | None, str | None]:
    """(ANCHOR, None) at the smallest allowed omp whose hybrid anchor is legal and passes the
    filter, else (None, reason). Undersubscribed (c < C) runs are packed onto cores [0, c) by
    OMPI's ppr binding; whether spreading them is faster is unmeasured."""
    cn = dataclasses.replace(ctx, nodes=nodes)
    omps = _sweep_omp_options(ctx, nodes, cores)
    msgs: list[str] = []
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
