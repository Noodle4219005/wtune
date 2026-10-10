#!/usr/bin/env python3
"""predictor/experiment.py -- the generated-space baseline rule (GBASE anchor) and an HH:MM:00
formatter, used by src/wtune/tools/scc_sweep.py and the tuner.

The experiment engine that used to live here (plan/record/screen/reconcile/summarize) is in
archive/unused-2026-10/src/predictor/experiment_legacy.py: it printed sbatch commands
(MODE=timing,CONFIGS=...) that campaign/strict_campaign.sbatch no longer accepts.
"""
from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
_REPO_ROOT = os.path.dirname(_SRC)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
if not __package__:                    # run by path: this directory is not an import root
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
for _p in (_SRC, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as campaign_configs     # campaign/configs.py: CASES, resolve_case, COST_GUESS,
                                        # load_generated_configs -- stdlib only


GENERATED_BASELINE_RULE = (
    "generated-space baseline (C01 analogue, _generated_baseline()): omp=1, ranks = the MAXIMUM "
    "legal rank count <= total_cores (scanned descending via predictor/decomp_space.py's "
    "legal_shapes_exact, decomp_space.py:143), numtiles=1, dist=cyclic, blocktime='0'; among "
    "legal (nx,ny) shapes at that rank count, pick the least aspect of d01's RSL min-patch "
    "(max(patch_x,patch_y)/min(patch_x,patch_y) via decomp_space.rsl_patch_extents, "
    "decomp_space.py:107 -- the same metric decomp_space.seed_design()'s own aspect() helper "
    "uses, decomp_space.py:636-638, for the identical 'cold-start, no C01' scenario), tie-broken "
    "by legal_shapes_exact's own ascending-nx factor_pairs order (decomp_space.py:102,150)."
)


def _format_hms(total_minutes):
    total_minutes = int(math.ceil(total_minutes))
    hh, mm = divmod(total_minutes, 60)
    return "%02d:%02d:00" % (hh, mm)


# ---------------------------------------------------------------------------
# generated-space candidate assembly (predictor/decomp_space.py -- a case that resolves no C01
# STRICT template, see the module docstring's "Generated space" paragraph)
# ---------------------------------------------------------------------------
def _import_decomp_space():
    try:
        from wtune.predictor import decomp_space
    except ImportError as e:
        raise SystemExit(
            "predictor/decomp_space.py could not be imported (%s: %s) -- it should need only "
            "the stdlib plus campaign/configs.py and predictor/features.py; check those modules "
            "if this is unexpected" % (type(e).__name__, e))
    return decomp_space


def _max_legal_ranks(case_json, max_ranks):
    """Largest ranks in 1..max_ranks with >=1 legal WRF shape for `case_json`'s active domains
    (predictor/decomp_space.legal_shapes_exact -- the exact floor-based share/module_check_a_
    mundo.F gate), scanning downward from max_ranks: (ranks, shapes), or (None, []) if no rank
    count in that range has any legal shape at all. This is GENERATED_BASELINE_RULE's own
    downward scan (_generated_baseline below), factored out so PART D's
    _call_decomp_search_propose fix (a case whose max_omp caps every candidate to a SINGLE omp
    value whose exact-fit ranks=total_cores//omp has no legal shape -- see that function's
    docstring) can reuse the identical rule instead of a second implementation."""
    decomp_space = _import_decomp_space()
    we_list, sn_list = campaign_configs._active_domain_extents(case_json)
    for ranks in range(max_ranks, 0, -1):
        shapes = decomp_space.legal_shapes_exact(we_list, sn_list, ranks)
        if shapes:
            return ranks, shapes
    return None, []


def _generated_baseline(case_json, total_cores, nodes, hardware_id):
    """The generated-space C01 analogue -- see GENERATED_BASELINE_RULE. Always resolvable from
    predictor/decomp_space.enumerate_space's exact legal-shape gate: scoped to
    total_cores=<the chosen max-legal rank count>, not the caller's total_cores, because the
    caller's own total_cores can itself be illegal at omp=1 -- enumerate_space(total_cores=
    nodes*112, omp_options=[1]) would enumerate ZERO omp=1 candidates whenever ranks=total_cores
    itself has no legal (nx,ny) split (e.g. matthew 91x100 at total_cores=112: legal_shapes_exact
    is empty there; the true max legal rank count is 90, shape 9x10)."""
    decomp_space = _import_decomp_space()
    we_list, sn_list = campaign_configs._active_domain_extents(case_json)
    ranks, shapes = _max_legal_ranks(case_json, total_cores)
    if ranks is None:
        raise SystemExit(
            "no legal MPI decomposition exists at omp=1 for any rank count 1..%d -- cannot "
            "build a generated-space baseline" % total_cores)

    we0, sn0 = we_list[0], sn_list[0]

    def _aspect(shape):
        nx, ny = shape
        px, _ = decomp_space.rsl_patch_extents(we0, nx)
        py, _ = decomp_space.rsl_patch_extents(sn0, ny)
        return max(px, py) / float(min(px, py))

    shape = min(shapes, key=lambda s: (_aspect(s), s))

    base_candidates, _axis = decomp_space.enumerate_space(
        case_json, total_cores=ranks, nodes=nodes, hardware_id=hardware_id, omp_options=[1],
        tile_options=dict(numtiles_multiples=(), max_numtiles=1, include_2d=False))
    base = next(c for c in base_candidates
                if (c["nx"], c["ny"]) == shape and c["dist"] == "cyclic" and c["blocktime"] == "0")
    return dict(
        ranks=base["ranks"], omp=1, nx=base["nx"], ny=base["ny"], numtiles=1,
        dist="cyclic", blocktime="0",
        numtiles_x=base.get("numtiles_x", 0), numtiles_y=base.get("numtiles_y", 0),
        tile_sz_x=base.get("tile_sz_x", 0), tile_sz_y=base.get("tile_sz_y", 0),
        tile_strategy=base.get("tile_strategy", 0),
        patch_x=base.get("patch_x"), patch_y=base.get("patch_y"),
    )
