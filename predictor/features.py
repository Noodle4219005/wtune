#!/usr/bin/env python3
"""predictor/features.py -- derived features for the WTune configuration predictor.

Splits features into three groups, on purpose:
  case_features    grid, physics, cadence, hardware -- properties of the *weather case*,
                    known (or knowable) before any config sweep on it.
  config_features  parallel geometry and runtime knobs -- properties of one *run* of that case.
  profile_features passed through when present (Level-1 only).

This split matters for kNN case retrieval (which must compare cases on case_features only) and
for the Level-0 (case+config) vs Level-1 (+profile) ablation in validate.py.

All functions take one ledger row (a dict of strings/None, as read back by csv.DictReader, or
already-parsed values) and return a dict of floats/bools/None. Missing inputs propagate to None
rather than being silently zero-filled -- callers decide how to impute.

Stdlib only.
"""
from __future__ import annotations

import math

HALO_WIDTH = 3  # WRF halo width used throughout this corpus (dm parallel, standard 5th-order adv)

# f1 Sapphire Rapids icpnq nodes: 112 cores/node, 2 sockets, SNC 8 NUMA domains x 14 cores,
# SMT off (recorded HPC hardware facts; see CLAUDE.md / knowledge/hpc notes). Cache/bandwidth
# fields are left null -- not measured, not to be invented.
HARDWARE_TABLE = {
    "f1-spr-icpnq": dict(
        cores_per_node=112, cores_per_numa=14, numa_per_node=8, smt=False,
        cache_per_core_mb=None, mem_bw_per_core=None,
    ),
    # f1 cf448 icpnp nodes: same 8480+ silicon, 112 cores, but SNC off -> 2 NUMA domains x 56
    # cores (knowledge/hpc/2026-09-03-f1-compute-nodes-have-2-numa-not-8-...); 2026-09-14 campaign.
    "f1-spr-icpnp": dict(
        cores_per_node=112, cores_per_numa=56, numa_per_node=2, smt=False,
        cache_per_core_mb=None, mem_bw_per_core=None,
    ),
}

# Physics scheme vocabularies actually observed in this corpus (config-cost separability note:
# mp in {6,8,10}, bl in {1,2,5}; ra_lw in {4,5} seen in the 12h-segmented summary). An unseen
# value at inference time falls into "_other" rather than being silently dropped.
MP_VOCAB = (6, 8, 10)
BL_VOCAB = (1, 2, 5)
RA_LW_VOCAB = (4, 5)


def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _onehot(value, vocab, prefix):
    v = _num(value)
    out = {"%s_%s" % (prefix, k): 0.0 for k in vocab}
    out["%s_other" % prefix] = 0.0
    if v is None:
        return out
    iv = int(v)
    key = "%s_%d" % (prefix, iv)
    if key in out:
        out[key] = 1.0
    else:
        out["%s_other" % prefix] = 1.0
    return out


def physics_flags(row):
    """One-hot flags for the schemes known to vary in this corpus. See MP_VOCAB etc. above."""
    out = {}
    out.update(_onehot(row.get("mp_physics"), MP_VOCAB, "mp"))
    out.update(_onehot(row.get("bl_pbl_physics"), BL_VOCAB, "bl"))
    out.update(_onehot(row.get("ra_lw_physics"), RA_LW_VOCAB, "ralw"))
    return out


def case_features(row):
    """Grid, physics, cadence, and hardware features -- a property of the weather case, not of
    any one run's parallel config."""
    e_we, e_sn, e_vert = _num(row.get("e_we")), _num(row.get("e_sn")), _num(row.get("e_vert"))
    hw = HARDWARE_TABLE.get(row.get("hardware_id"), {})
    feat = {
        "e_we": e_we, "e_sn": e_sn, "e_vert": e_vert,
        "dx": _num(row.get("dx")), "dy": _num(row.get("dy")),
        "total_cells": (e_we * e_sn) if (e_we is not None and e_sn is not None) else None,
        "aspect": (e_we / e_sn) if (e_we is not None and e_sn not in (None, 0)) else None,
        "time_step": _num(row.get("time_step")), "radt": _num(row.get("radt")),
        "cudt": _num(row.get("cudt")), "history_interval": _num(row.get("history_interval")),
        "mp_physics": _num(row.get("mp_physics")), "cu_physics": _num(row.get("cu_physics")),
        "bl_pbl_physics": _num(row.get("bl_pbl_physics")),
        "sf_sfclay_physics": _num(row.get("sf_sfclay_physics")),
        "sf_surface_physics": _num(row.get("sf_surface_physics")),
        "ra_lw_physics": _num(row.get("ra_lw_physics")), "ra_sw_physics": _num(row.get("ra_sw_physics")),
        "cores_per_node": hw.get("cores_per_node"), "cores_per_numa": hw.get("cores_per_numa"),
        "numa_per_node": hw.get("numa_per_node"),
        "smt": (1.0 if hw.get("smt") else 0.0) if "smt" in hw else None,
        "cache_per_core_mb": hw.get("cache_per_core_mb"), "mem_bw_per_core": hw.get("mem_bw_per_core"),
    }
    feat.update(physics_flags(row))
    return feat


def _tile_dims(extent, nparts):
    """floor/ceil split of `extent` into `nparts`, i.e. (min tile size, max tile size). This is
    the same ceil/floor APPROXIMATION `config_features` already uses for per-rank patch sizes
    (patch_x/patch_y below), applied per tile instead -- not WRF's exact recursive tile-splitting
    algorithm (frame/module_machine.F's rlocproc, replicated exactly in
    predictor/decomp_space.py's `tile_extents`), kept out of this stdlib-only module on purpose
    to avoid a features.py -> decomp_space.py import (decomp_space.py itself imports this module,
    so the reverse import would cycle). Good enough for a coarse feature; use
    decomp_space.tile_geometry_for_patch directly when exact per-tile extents matter."""
    if not extent or not nparts:
        return None, None
    q, rem = divmod(int(extent), int(nparts))
    return q, (q + 1 if rem else q)


def tile_geometry_features(row):
    """Additive tile-geometry features (WTune D1 decomposition-space work): tiles_x, tiles_y,
    tile_sz_x, tile_sz_y, tile_strategy, cells_per_tile_min/max, tile_aspect, numa_aligned.
    Computed only when the corresponding raw keys (numtiles_x/numtiles_y at minimum) are present
    on `row` -- a ledger row from before this work simply lacks them, so every field here is None
    for it, and `models.py`'s `{k: v for k, v in d.items() if v is not None}` filter drops them,
    exactly like PROFILE_FIELDS already does for a profile-less row (see `has_profile_data`)."""
    numtiles_x, numtiles_y = _num(row.get("numtiles_x")), _num(row.get("numtiles_y"))
    tile_sz_x, tile_sz_y = _num(row.get("tile_sz_x")), _num(row.get("tile_sz_y"))
    tile_strategy = _num(row.get("tile_strategy"))
    e_we, e_sn, e_vert = _num(row.get("e_we")), _num(row.get("e_sn")), _num(row.get("e_vert"))
    nproc_x, nproc_y = _num(row.get("nproc_x")), _num(row.get("nproc_y"))

    patch_x = math.ceil(e_we / nproc_x) if (e_we is not None and nproc_x) else None
    patch_y = math.ceil(e_sn / nproc_y) if (e_sn is not None and nproc_y) else None

    tile_w_min = tile_w_max = tile_h_min = tile_h_max = None
    if patch_x is not None and numtiles_x:
        tile_w_min, tile_w_max = _tile_dims(patch_x, numtiles_x)
    if patch_y is not None and numtiles_y:
        tile_h_min, tile_h_max = _tile_dims(patch_y, numtiles_y)

    cells_per_tile_min = cells_per_tile_max = None
    if None not in (tile_w_min, tile_h_min, tile_w_max, tile_h_max, e_vert):
        cells_per_tile_min = tile_w_min * tile_h_min * e_vert
        cells_per_tile_max = tile_w_max * tile_h_max * e_vert

    tile_aspect = None
    if tile_w_max not in (None, 0) and tile_h_max not in (None, 0):
        tile_aspect = tile_w_max / float(tile_h_max)

    hw = HARDWARE_TABLE.get(row.get("hardware_id"), {})
    cores_per_numa = hw.get("cores_per_numa")
    omp_threads = _num(row.get("omp_threads"))
    numa_aligned = None
    if cores_per_numa and omp_threads:
        numa_aligned = 1.0 if cores_per_numa % omp_threads == 0 else 0.0

    return {
        "tiles_x": numtiles_x, "tiles_y": numtiles_y,
        "tile_sz_x": tile_sz_x, "tile_sz_y": tile_sz_y, "tile_strategy": tile_strategy,
        "cells_per_tile_min": cells_per_tile_min, "cells_per_tile_max": cells_per_tile_max,
        "tile_aspect": tile_aspect, "numa_aligned": numa_aligned,
    }


def config_features(row):
    """Parallel geometry and runtime-knob features -- a property of one run."""
    e_we, e_sn, e_vert = _num(row.get("e_we")), _num(row.get("e_sn")), _num(row.get("e_vert"))
    nproc_x, nproc_y = _num(row.get("nproc_x")), _num(row.get("nproc_y"))
    numtiles = _num(row.get("numtiles"))
    omp_threads = _num(row.get("omp_threads"))
    n_ranks, n_nodes = _num(row.get("n_ranks")), _num(row.get("n_nodes"))
    hw = HARDWARE_TABLE.get(row.get("hardware_id"), {})
    numa_per_node = hw.get("numa_per_node")

    patch_x = math.ceil(e_we / nproc_x) if (e_we is not None and nproc_x) else None
    patch_y = math.ceil(e_sn / nproc_y) if (e_sn is not None and nproc_y) else None
    patch_cells = (patch_x * patch_y) if (patch_x is not None and patch_y is not None) else None

    halo_interior_ratio = None
    if patch_x is not None and patch_y is not None and patch_x > 0 and patch_y > 0:
        h = HALO_WIDTH
        interior = patch_x * patch_y
        with_halo = (patch_x + 2 * h) * (patch_y + 2 * h)
        halo_interior_ratio = (with_halo - interior) / interior

    cells_per_tile = None
    if patch_cells is not None and e_vert is not None and numtiles:
        cells_per_tile = patch_cells * e_vert / numtiles

    tiles_per_thread = (numtiles / omp_threads) if (numtiles is not None and omp_threads) else None
    numtiles_mod_omp = (numtiles % omp_threads) if (numtiles is not None and omp_threads) else None
    tiles_divisible = (numtiles_mod_omp == 0) if numtiles_mod_omp is not None else None

    ranks_per_node = (n_ranks / n_nodes) if (n_ranks is not None and n_nodes) else None
    ranks_per_numa = (ranks_per_node / numa_per_node) if (ranks_per_node is not None and numa_per_node) else None

    grid_divisible_x = (e_we % nproc_x == 0) if (e_we is not None and nproc_x) else None
    grid_divisible_y = (e_sn % nproc_y == 0) if (e_sn is not None and nproc_y) else None
    decomp_aspect = (patch_x / patch_y) if (patch_x is not None and patch_y not in (None, 0)) else None

    out = {
        "nproc_x": nproc_x, "nproc_y": nproc_y, "numtiles": numtiles, "omp_threads": omp_threads,
        "n_ranks": n_ranks, "n_nodes": n_nodes, "kmp_blocktime": _num(row.get("kmp_blocktime")),
        "distribution_cyclic": 1.0 if row.get("distribution") == "cyclic" else (
            0.0 if row.get("distribution") else None),
        "patch_x": patch_x, "patch_y": patch_y, "patch_cells": patch_cells,
        "halo_interior_ratio": halo_interior_ratio, "cells_per_tile": cells_per_tile,
        "tiles_per_thread": tiles_per_thread, "numtiles_mod_omp": numtiles_mod_omp,
        "tiles_divisible": (1.0 if tiles_divisible else (0.0 if tiles_divisible is not None else None)),
        "ranks_per_node": ranks_per_node, "ranks_per_numa": ranks_per_numa,
        "grid_divisible_x": (1.0 if grid_divisible_x else (0.0 if grid_divisible_x is not None else None)),
        "grid_divisible_y": (1.0 if grid_divisible_y else (0.0 if grid_divisible_y is not None else None)),
        "decomp_aspect": decomp_aspect,
    }
    out.update(tile_geometry_features(row))
    return out


PROFILE_FIELDS = (
    "profile_micro_share", "profile_radiation_share", "profile_dynamics_share",
    "profile_mpi_share", "profile_io_share", "profile_rank_imbalance", "profile_thread_imbalance",
)


def profile_features(row):
    """Pass through profile fields when present; all-None when the source has no profile data
    (used by validate.py's Level-0/Level-1 ablation to detect the "no profile data" case)."""
    return {f: _num(row.get(f)) for f in PROFILE_FIELDS}


def has_profile_data(rows):
    return any(_num(r.get(f)) is not None for r in rows for f in PROFILE_FIELDS)


def config_key(row):
    """Identity of "the same configuration" for target/pairwise/kNN logic: the tuple of
    config_features that is not itself derived-redundant (drop the geometry derived from
    nproc_x/nproc_y/numtiles/omp_threads, which is fully determined by the raw knobs).
    Tile geometry (numtiles_x/y, tile_sz_x/y, tile_strategy) is appended only when non-default, so
    STRICT-campaign keys are unchanged but distinct D2 tile shapes are no longer merged (2026-09-15)."""
    key = (
        row.get("nproc_x"), row.get("nproc_y"), row.get("numtiles"), row.get("omp_threads"),
        row.get("distribution"), row.get("kmp_blocktime"), row.get("n_nodes"), row.get("n_ranks"),
    )
    tiles = tuple(_num(row.get(k)) or 0 for k in
                  ("numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy"))
    return key + (tiles,) if any(tiles) else key


# ---------------------------------------------------------------------------
# Decision groups: the resource allocation + comparison scope within which configs are directly
# comparable for baseline/ranking/pairwise/regret purposes. A 4-node and an 8-node run of the same
# case_base are NOT alternative configurations of one decision -- the 8-node run is faster because
# it has twice the hardware, not because its knobs are better. See
# knowledge/... "conus regret 0.78-0.81 under leave-one-run-out is meaningless" (2026-09-14
# predictor evaluation-validity fix).
# ---------------------------------------------------------------------------
def total_cores(row):
    """n_ranks * omp_threads, or None if either is unknown (never guessed)."""
    n_ranks, omp_threads = _num(row.get("n_ranks")), _num(row.get("omp_threads"))
    if n_ranks is None or omp_threads is None:
        return None
    return n_ranks * omp_threads


def resolved_n_nodes(row):
    """n_nodes as recorded, or inferred as total_cores / hardware cores-per-node when n_nodes is
    missing but total_cores and the hardware's cores-per-node are both known and the division is
    exact. Returns None rather than a fractional/guessed node count otherwise."""
    n_nodes = _num(row.get("n_nodes"))
    if n_nodes is not None:
        return n_nodes
    cores = total_cores(row)
    cores_per_node = HARDWARE_TABLE.get(row.get("hardware_id"), {}).get("cores_per_node")
    if cores is not None and cores_per_node and cores % cores_per_node == 0:
        return cores / cores_per_node
    return None


def decision_group_key(row):
    """Identity of the fixed resource allocation + comparison scope: (case_base, hardware_id,
    compliance_class, n_nodes, total_cores, binary_kind, io_form_history). Configs are only
    comparable (baseline/top-k/regret/pairwise) within one decision group.

    Returns None -- never a partial or guessed key -- when case_base/hardware_id, or both
    n_nodes and total_cores, cannot be determined. Callers must exclude such rows from ranking
    and count the exclusion (see decision_group_known / targets.count_unknown_decision_group)."""
    if row.get("case_base") is None or row.get("hardware_id") is None:
        return None
    n_nodes = resolved_n_nodes(row)
    cores = total_cores(row)
    if n_nodes is None or cores is None:
        return None
    io_form = row.get("io_form_history")
    if _num(io_form) is not None:  # ledger CSV gives "14", namelist/case rows give 14 (2026-09-15)
        io_form = str(int(_num(io_form)))
    return (row.get("case_base"), row.get("hardware_id"), row.get("compliance_class"),
            n_nodes, cores, row.get("binary_kind"), io_form)


def decision_group_known(row):
    return decision_group_key(row) is not None
