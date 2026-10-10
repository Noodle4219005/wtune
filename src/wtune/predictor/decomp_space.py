#!/usr/bin/env python3
"""predictor/decomp_space.py -- the full legal stock-WRF 4.8 decomposition/tiling space (step D1).

Enumerates every legal (ranks, omp) x (nproc_x, nproc_y) x tile geometry x (dist, blocktime)
candidate that campaign/configs.py's 16 fixed templates only sample, with exact WRF semantics
for two DIFFERENT remainder rules: MPI patches split the remainder at both ends
(RSL_LITE/task_for_point.c; legality gate floor(extent/nproc) >= 10, module_check_a_mundo.F
3014-3029 -- not configs.py's weaker ceil check), tiles use module_machine.F's recursive
region_bounds (10 into 3 -> 4,3,3; 11 into 3 -> 4,3,4). _rlocproc/_region_bounds are memoized
to keep the several-thousand-candidate space tractable. Python >= 3.7 (predictor pixi env)."""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
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

from wtune.predictor import features as feat                # predictor/features.py: HARDWARE_TABLE, HALO_WIDTH
import configs as campaign_configs     # campaign/configs.py: CASES, TEMPLATES, resolve_case

HALO_WIDTH = feat.HALO_WIDTH

# frame/module_machine.F:11
TILE_NONE, TILE_X, TILE_Y, TILE_XY = 0, 1, 2, 3
# frame/module_tiles.F:17 (#define MIN_TILE_SIZE 1)
MIN_TILE_SIZE = 1
# share/module_check_a_mundo.F:3017-3018 (Fortran integer division, i.e. floor for positive ints)
MIN_PATCH = 10   # mass points per patch, each direction

DEFAULT_HARDWARE = "f1-spr-icpnp"

# numtiles of the 16 STRICT templates, unioned into the default candidates so sample_design's
# must_include can match them (8, 12, 13, 14, 16 are not small multiples of every omp).
_TEMPLATE_NUMTILES = sorted(set(t["numtiles"] for t in campaign_configs.TEMPLATES))

DEFAULT_TILE_OPTIONS = dict(
    numtiles_multiples=(1, 2, 3, 4),   # numtiles candidates include {omp*k : k in this list}
    max_numtiles=64,                   # hard cap on any generated numtiles candidate
    include_2d=True,                   # also emit an explicit 2D (least_aspect) variant per numtiles>1
    tile_sz_variants=(),               # extra explicit (tile_sz_x, tile_sz_y) pairs to also try
)

DIST_OPTIONS = ("block", "cyclic")
BLOCKTIME_OPTIONS = ("0", "default")


# ---------------------------------------------------------------------------
# WRF semantics, replicated exactly
# ---------------------------------------------------------------------------

def factor_pairs(n: int) -> list[tuple[int, int]]:
    """Every (a, b) with a*b == n, a ascending."""
    return [(i, n // i) for i in range(1, n + 1) if n % i == 0]


def rsl_patch_extents(extent: int, nproc: int) -> tuple[int, int]:
    """(min_patch, max_patch) for one dimension of one domain split `nproc` ways, per RSL-lite's
    remainder-at-both-ends rule (external/RSL_LITE/task_for_point.c:6-15,87-100): `rem =
    extent % nproc` ranks get `extent//nproc + 1` points (split as `rem//2` at the front, the rest
    at the back -- task_for_point.c:88-90's a/b split), the other `nproc-rem` ranks get exactly
    `extent//nproc`. min_patch == extent//nproc is therefore EXACTLY the quantity WRF's own
    legality check divides (share/module_check_a_mundo.F:3017-3018, Fortran integer division)."""
    q, rem = divmod(extent, nproc)
    return q, (q + 1 if rem else q)


def rsl_patch_width_for_rank(extent: int, nproc: int, rank_index: int) -> int:
    """Width of 0-indexed rank `rank_index`'s own patch under RSL-lite's remainder-at-both-ends
    rule: rem = extent % nproc ranks get one extra point, rem//2 at the front, the rest at the back.
    Rank 0 can be in either group per dimension (karina 8x56: largest y-patch; 4x56: smallest
    x-patch), so neither envelope bound stands in for it (smoke job 1078036)."""
    q, rem = divmod(extent, nproc)
    front = rem // 2
    back = rem - front
    if rank_index < front or rank_index >= nproc - back:
        return q + 1
    return q


def legal_shapes_exact(we_list: list[int], sn_list: list[int], ranks: int,
                       min_patch: int = MIN_PATCH) -> list[tuple[int, int]]:
    """All (nx, ny) with nx*ny == ranks that pass WRF's REAL startup gate
    (share/module_check_a_mundo.F:3014-3029): floor(e_we(i)/nproc_x) >= 10 and
    floor(e_sn(i)/nproc_y) >= 10 for EVERY active domain i (paired by index). This is stricter
    than campaign/configs.py's own `legal_shapes` (ceil-based -- see that module's docstring) and
    is the one and only ground truth for "will WRF actually accept this decomposition"."""
    out = []
    for nx, ny in factor_pairs(ranks):
        ok = True
        for we, sn in zip(we_list, sn_list):
            if we // nx < min_patch or sn // ny < min_patch:
                ok = False
                break
        if ok:
            out.append((nx, ny))
    return out


@functools.lru_cache(maxsize=None)
def _rlocproc(p, maxi, nproc, ml=0, mr=0):
    """Transliteration of frame/module_machine.F:15-46 `rlocproc` (RECURSIVE). Returns the
    processor index (0..nproc-1) owning 0-indexed point `p` of `maxi` points. ml/mr (minimum
    left/right block widths) are always called with 0 by every caller in this codebase, exactly
    as in the Fortran source (module_machine.F:42's recursive call always passes zero,zero) --
    kept as parameters only for source fidelity."""
    rem = maxi % nproc
    width = maxi // nproc
    mid = maxi // 2
    adjust = 0
    if rem > 0 and (rem % 2 == 0 or rem > 2 or p <= mid):
        width += 1
    if p <= mid and rem % 2 != 0:
        adjust = 1
    bl = max(width, ml)
    br = max(width, mr)
    if p < bl:
        return 0
    if p > maxi - br - 1:
        return nproc - 1
    p_r = p - bl
    maxi_r = maxi - bl - br + adjust
    nproc_r = max(nproc - 2, 1)
    return _rlocproc(p_r, maxi_r, nproc_r, 0, 0) + 1


def _locproc(i, m, numpart):
    """frame/module_machine.F:48-59 `locproc`."""
    return _rlocproc(i, m, numpart, 0, 0)


@functools.lru_cache(maxsize=None)
def _region_bounds(region_start, region_end, num_p, p):
    """frame/module_machine.F:76-106 `region_bounds`: 1-indexed inclusive [region_start,
    region_end] split into `num_p` groups; returns group `p`'s (start, end), or (None, None) when
    `p` gets no point at all (module_tiles.F's callers then skip creating that tile entirely --
    "converse happens if number of tiles > number of points in dim", module_tiles.F:418,437)."""
    npoints = region_end - region_start + 1
    ts = te = None
    for i in range(npoints):
        if _locproc(i, npoints, num_p) == p:
            if ts is None or i < ts:
                ts = i
            if te is None or i > te:
                te = i
    if ts is None:
        return None, None
    return ts + region_start, te + region_start


def least_aspect(nparts: int, minparts_y: int = 1, minparts_x: int = 1) -> tuple[int, int]:
    """frame/module_machine.F:108-132 `least_aspect`. Returns (nparts_y, nparts_x) -- the same
    output order as the Fortran subroutine's argument list."""
    mini = 2 * nparts
    nparts_y, nparts_x = 1, nparts
    for y in range(1, nparts + 1):
        if nparts % y == 0:
            x = nparts // y
            if abs(y - x) < mini and y >= minparts_y and x >= minparts_x:
                mini = abs(y - x)
                nparts_y, nparts_x = y, x
    return nparts_y, nparts_x


def resolve_tile_geometry(patch_x: int, patch_y: int, numtiles: int = 1, numtiles_x: int = 0,
                          numtiles_y: int = 0, tile_sz_x: int = 0, tile_sz_y: int = 0,
                          tile_strategy: int = TILE_NONE) -> dict:
    """module_tiles.F:269-391's numtiles/numtiles_x/numtiles_y resolution, namelist branch only
    (this campaign always writes numtiles; WRF_NUM_TILES_INC assumed 1) -> {num_tiles, numtiles_x,
    numtiles_y, tile_strategy_used, went_2d (the 1D-Y -> 2D auto-switch, module_tiles.F:333-348)}."""
    num_tiles = numtiles
    ntx, nty = numtiles_x, numtiles_y
    went_2d = False
    if tile_strategy == TILE_NONE:
        strategy_used = TILE_Y
        if num_tiles > patch_y // MIN_TILE_SIZE and ntx == 0 and nty == 0:
            ntx = 1
            nty = patch_y // MIN_TILE_SIZE
            while ntx * nty < num_tiles:
                ntx += 1
            went_2d = True
            strategy_used = TILE_XY
    else:
        strategy_used = tile_strategy

    if tile_sz_x >= 1 and tile_sz_y >= 1:
        ntx = patch_x // tile_sz_x + (1 if patch_x % tile_sz_x else 0)
        nty = patch_y // tile_sz_y + (1 if patch_y % tile_sz_y else 0)
        num_tiles = ntx * nty
    elif ntx >= 1 or nty >= 1:
        if ntx >= 1 and nty >= 1:
            num_tiles = ntx * nty
        elif ntx >= 1:
            nty = num_tiles // ntx
        else:
            ntx = num_tiles // nty
    else:
        if strategy_used == TILE_X:
            ntx, nty = num_tiles, 1
        elif strategy_used == TILE_Y:
            ntx, nty = 1, num_tiles
        else:
            nty, ntx = least_aspect(num_tiles, 1, 1)

    num_tiles = max(num_tiles, 1)
    ntx = max(ntx, 1)
    nty = max(nty, 1)
    return dict(num_tiles=num_tiles, numtiles_x=ntx, numtiles_y=nty,
                tile_strategy_used=strategy_used, went_2d=went_2d)


def tile_extents(patch_x: int, patch_y: int, numtiles_x: int, numtiles_y: int) -> list[tuple[int, int, int, int]]:
    """[(i_start, i_end, j_start, j_end), ...] 1-indexed within the patch (1..patch_x, 1..patch_y)
    for each REALIZED tile, in module_tiles.F set_tiles2's own t=0..numtiles_x*numtiles_y-1 order
    (module_tiles.F:409-451: y is the outer/major loop, x inner). A (row, col) whose group is
    empty realizes NO tile (module_tiles.F:418,437's `IF (ts.LE.te)` guard) -- skipped here too,
    so the result can be shorter than numtiles_x*numtiles_y."""
    tiles = []
    for t in range(numtiles_x * numtiles_y):
        row = t // numtiles_x
        col = t % numtiles_x
        j_start, j_end = _region_bounds(1, patch_y, numtiles_y, row)
        if j_start is None:
            continue
        i_start, i_end = _region_bounds(1, patch_x, numtiles_x, col)
        if i_start is None:
            continue
        tiles.append((i_start, i_end, j_start, j_end))
    return tiles


def tile_geometry_for_patch(patch_x: int, patch_y: int, e_vert: int, numtiles: int = 1,
                            numtiles_x: int = 0, numtiles_y: int = 0, tile_sz_x: int = 0,
                            tile_sz_y: int = 0, tile_strategy: int = TILE_NONE) -> dict:
    """resolve_tile_geometry + tile_extents + summary stats (cells/aspect min/max across the
    realized tiles) for one patch. `e_vert` turns 2D tile area into a 3D cell count, matching
    predictor/features.py's `cells_per_tile` convention (patch_cells * e_vert / numtiles)."""
    resolved = resolve_tile_geometry(patch_x, patch_y, numtiles, numtiles_x, numtiles_y,
                                      tile_sz_x, tile_sz_y, tile_strategy)
    extents = tile_extents(patch_x, patch_y, resolved["numtiles_x"], resolved["numtiles_y"])
    out = dict(resolved, realized_num_tiles=len(extents),
               cells_per_tile_min=None, cells_per_tile_max=None,
               tile_aspect_min=None, tile_aspect_max=None,
               tile_w_min=None, tile_w_max=None, tile_h_min=None, tile_h_max=None)
    if not extents:
        return out
    widths = [ie - is_ + 1 for is_, ie, _js, _je in extents]
    heights = [je - js + 1 for _is, _ie, js, je in extents]
    areas = [w * h for w, h in zip(widths, heights)]
    aspects = [w / float(h) for w, h in zip(widths, heights) if h]
    out.update(
        cells_per_tile_min=min(areas) * e_vert, cells_per_tile_max=max(areas) * e_vert,
        tile_aspect_min=min(aspects) if aspects else None,
        tile_aspect_max=max(aspects) if aspects else None,
        tile_w_min=min(widths), tile_w_max=max(widths),
        tile_h_min=min(heights), tile_h_max=max(heights),
    )
    return out


def domain_patch_geometry(we_list: list[int], sn_list: list[int], nx: int, ny: int) -> dict:
    """[{...}] one dict per active domain: RSL-lite min/max patch extents (`rsl_patch_extents`)
    and the halo/interior ratio (HALO_WIDTH=3, predictor/features.py's own convention) at each
    extreme."""
    out = []
    for we, sn in zip(we_list, sn_list):
        px_min, px_max = rsl_patch_extents(we, nx)
        py_min, py_max = rsl_patch_extents(sn, ny)
        # rank 0's own patch (see rsl_patch_width_for_rank's docstring: NOT reliably equal to
        # either the min or max envelope bound above, independently per dimension).
        px_rank0 = rsl_patch_width_for_rank(we, nx, 0)
        py_rank0 = rsl_patch_width_for_rank(sn, ny, 0)
        out.append(dict(
            e_we=we, e_sn=sn,
            patch_x_min=px_min, patch_x_max=px_max,
            patch_y_min=py_min, patch_y_max=py_max,
            patch_x_rank0=px_rank0, patch_y_rank0=py_rank0,
            halo_interior_ratio_min=_halo_ratio(px_min, py_min),
            halo_interior_ratio_max=_halo_ratio(px_max, py_max),
        ))
    return out


def _halo_ratio(px, py, h=HALO_WIDTH):
    if not px or not py:
        return None
    interior = px * py
    with_halo = (px + 2 * h) * (py + 2 * h)
    return (with_halo - interior) / interior


# ---------------------------------------------------------------------------
# Space enumeration
# ---------------------------------------------------------------------------

def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def default_omp_options(total_cores: int, nodes: int, hardware_id: str) -> list[int]:
    """Every divisor of the hardware's cores-per-NUMA that gives an integer rank count divisible
    by `nodes` -- i.e. every omp value whose ranks pack a whole number of NUMA domains per node
    with no thread crossing a NUMA boundary (ranks_per_numa = cores_per_numa // omp is then
    always exact, see `enumerate_space`)."""
    hw = feat.HARDWARE_TABLE[hardware_id]
    cores_per_numa = hw["cores_per_numa"]
    return [d for d in _divisors(cores_per_numa)
            if total_cores % d == 0 and (total_cores // d) % nodes == 0]


def numtiles_candidates(omp: int, tile_options: dict | None) -> list[int]:
    """{1} u {omp*k : k in tile_options['numtiles_multiples']} u the 16 STRICT templates' own
    numtiles values (see _TEMPLATE_NUMTILES), capped at tile_options['max_numtiles'] -- except
    numtiles == omp (k=1), always kept: at omp=72 the cap left only numtiles <= 16 < omp."""
    multiples = tile_options.get("numtiles_multiples", DEFAULT_TILE_OPTIONS["numtiles_multiples"])
    max_nt = tile_options.get("max_numtiles", DEFAULT_TILE_OPTIONS["max_numtiles"])
    cands = set([1])
    for k in multiples:
        nt = omp * k
        if nt <= max_nt or k == 1:
            cands.add(nt)
    for nt in _TEMPLATE_NUMTILES:
        if nt <= max_nt:
            cands.add(nt)
    return sorted(cands)


def enumerate_space(case: dict, total_cores: int = 448, nodes: int = 4,
                    hardware_id: str = DEFAULT_HARDWARE, omp_options: list[int] | None = None,
                    tile_options: dict | None = None) -> tuple[list[dict], dict]:
    """Every legal stock-WRF candidate for a cases.json-shaped `case` (e_we/e_sn/e_vert[, max_dom,
    e_we_list, e_sn_list]) -> (candidates, axis_summary with per-axis counts). Axes: omp, ranks =
    total_cores // omp, (nx, ny) legal on every active domain, numtiles, tile_shape ('1d_y' WRF
    default, '2d' least-aspect TILE_XY, optional 'tile_sz' variants), dist, blocktime. Tile geometry
    is reported for d01's smallest patch (1D->2D risk) and largest patch (cells_per_tile bound)."""
    hw = feat.HARDWARE_TABLE[hardware_id]
    if omp_options is None:
        omp_options = default_omp_options(total_cores, nodes, hardware_id)
    topts = dict(DEFAULT_TILE_OPTIONS)
    if tile_options:
        topts.update(tile_options)

    we_list, sn_list = campaign_configs._active_domain_extents(case)
    e_vert = case["e_vert"]

    candidates = []
    shapes_count = {}
    numtiles_options = {}
    tile_shape_options = ["1d_y"] + (["2d"] if topts["include_2d"] else [])
    tile_shape_options += ["tile_sz"] * len(topts["tile_sz_variants"])

    for omp in omp_options:
        ranks = total_cores // omp
        ranks_per_node = ranks // nodes
        numa_per_node = hw.get("numa_per_node")
        cores_per_numa = hw.get("cores_per_numa")
        ranks_per_numa = (ranks_per_node / float(numa_per_node)) if numa_per_node else None
        numa_aligned = bool(cores_per_numa and cores_per_numa % omp == 0)
        threads_per_numa = (ranks_per_numa * omp) if ranks_per_numa is not None else None

        shapes = legal_shapes_exact(we_list, sn_list, ranks)
        shapes_count[omp] = len(shapes)
        nt_cands = numtiles_candidates(omp, topts)
        numtiles_options[omp] = nt_cands

        for nx, ny in shapes:
            domains = domain_patch_geometry(we_list, sn_list, nx, ny)
            d01 = domains[0]
            rep_px, rep_py = d01["patch_x_min"], d01["patch_y_min"]
            max_px, max_py = d01["patch_x_max"], d01["patch_y_max"]
            rank0_px, rank0_py = d01["patch_x_rank0"], d01["patch_y_rank0"]

            for numtiles in nt_cands:
                tile_variants = [("1d_y", 0, 0, 0, 0, TILE_NONE)]
                if topts["include_2d"] and numtiles > 1:
                    ly, lx = least_aspect(numtiles, 1, 1)
                    tile_variants.append(("2d", lx, ly, 0, 0, TILE_XY))
                for tsz_x, tsz_y in topts["tile_sz_variants"]:
                    tile_variants.append(("tile_sz", 0, 0, tsz_x, tsz_y, TILE_NONE))

                for shape_name, ntx, nty, tsx, tsy, tstrat in tile_variants:
                    geo_min = tile_geometry_for_patch(rep_px, rep_py, e_vert, numtiles,
                                                       ntx, nty, tsx, tsy, tstrat)
                    geo_max = tile_geometry_for_patch(max_px, max_py, e_vert, numtiles,
                                                       ntx, nty, tsx, tsy, tstrat)
                    # rank 0's own tile geometry, what WRF's rank-0 tile printout shows (the
                    # min/max envelopes need not match it; smoke job 1078036)
                    geo_rank0 = tile_geometry_for_patch(rank0_px, rank0_py, e_vert, numtiles,
                                                         ntx, nty, tsx, tsy, tstrat)
                    for dist in DIST_OPTIONS:
                        for blocktime in BLOCKTIME_OPTIONS:
                            candidates.append(dict(
                                ranks=ranks, omp=omp, nx=nx, ny=ny,
                                numtiles=numtiles, tile_shape=shape_name,
                                numtiles_x=ntx, numtiles_y=nty,
                                tile_sz_x=tsx, tile_sz_y=tsy, tile_strategy=tstrat,
                                dist=dist, blocktime=blocktime,
                                patch_x=rep_px, patch_y=rep_py,
                                patch_x_max=max_px, patch_y_max=max_py,
                                patch_x_rank0=rank0_px, patch_y_rank0=rank0_py,
                                halo_interior_ratio=d01["halo_interior_ratio_min"],
                                halo_interior_ratio_max=d01["halo_interior_ratio_max"],
                                numtiles_x_used=geo_min["numtiles_x"],
                                numtiles_y_used=geo_min["numtiles_y"],
                                went_2d=geo_min["went_2d"],
                                realized_num_tiles_min=geo_min["realized_num_tiles"],
                                realized_num_tiles_max=geo_max["realized_num_tiles"],
                                cells_per_tile_min=geo_min["cells_per_tile_min"],
                                cells_per_tile_max=geo_max["cells_per_tile_max"],
                                tile_aspect_min=geo_min["tile_aspect_min"],
                                tile_aspect_max=geo_min["tile_aspect_max"],
                                numtiles_x_rank0=geo_rank0["numtiles_x"],
                                numtiles_y_rank0=geo_rank0["numtiles_y"],
                                went_2d_rank0=geo_rank0["went_2d"],
                                realized_num_tiles_rank0=geo_rank0["realized_num_tiles"],
                                cells_per_tile_rank0_min=geo_rank0["cells_per_tile_min"],
                                cells_per_tile_rank0_max=geo_rank0["cells_per_tile_max"],
                                tile_aspect_rank0_min=geo_rank0["tile_aspect_min"],
                                tile_aspect_rank0_max=geo_rank0["tile_aspect_max"],
                                ranks_per_node=ranks_per_node, ranks_per_numa=ranks_per_numa,
                                threads_per_numa=threads_per_numa, numa_aligned=numa_aligned,
                                domains=domains,
                            ))

    axis_summary = dict(
        omp_options=list(omp_options),
        ranks_by_omp={omp: total_cores // omp for omp in omp_options},
        shapes_count=shapes_count,
        numtiles_options=numtiles_options,
        tile_shape_options=tile_shape_options,
        dist_options=list(DIST_OPTIONS),
        blocktime_options=list(BLOCKTIME_OPTIONS),
        total=len(candidates),
    )
    return candidates, axis_summary


def campaign_template_predicates(case: dict) -> list[dict]:
    """[{'ranks','omp','nx','ny','numtiles','dist','blocktime'}, ...] for every STRICT template
    (C01-C16, deduped) that resolves for `case`, via campaign/configs.py's OWN resolve_case (the
    same scorer/shape resolution the real campaign used) -- for use as `sample_design`'s
    `must_include`, so a ground-truth sample always covers the already-measured STRICT points."""
    kept, _skipped, _aliases = campaign_configs.resolve_case(case)
    return [dict(ranks=c["ranks"], omp=c["omp"], nx=c["nx"], ny=c["ny"],
                 numtiles=c["numtiles"], dist=c["dist"], blocktime=c["blocktime"])
            for c in kept]


def _matches_predicate(candidate, predicate):
    return all(candidate.get(k) == v for k, v in predicate.items())


def _candidate_key(c):
    return (c["omp"], c["nx"], c["ny"], c["numtiles"], c["tile_shape"], c["dist"], c["blocktime"])


# ---------------------------------------------------------------------------
# Ground-truth sample design
# ---------------------------------------------------------------------------

_STRATIFY_AXES = ("omp", "nx", "ny", "numtiles", "tile_shape", "dist", "blocktime")


def sample_design(space: list[dict], n: int, seed: int, must_include: list[dict] | None = None
                  ) -> tuple[list[dict], dict]:
    """Deterministic stratified subset of `space` of size n -> (chosen, coverage). Phase 1: every
    must_include match (a floor, may exceed n); phase 2: one representative per missing value of
    each _STRATIFY_AXES axis; phase 3: random fill. `space` is sorted by _candidate_key first, so
    the same (space, n, seed, must_include) always gives the same list."""
    rng = random.Random(seed)
    space_sorted = sorted(space, key=_candidate_key)
    chosen = []
    chosen_keys = set()

    n_must_matched = 0
    must_unmatched = 0
    if must_include:
        for predicate in must_include:
            matches = [c for c in space_sorted if _matches_predicate(c, predicate)]
            if not matches:
                must_unmatched += 1
                continue
            n_must_matched += 1
            pick = matches[0]
            k = _candidate_key(pick)
            if k not in chosen_keys:
                chosen.append(pick)
                chosen_keys.add(k)

    by_axis_value = {}
    for axis in _STRATIFY_AXES:
        values = {}
        for c in space_sorted:
            values.setdefault(c[axis], []).append(c)
        by_axis_value[axis] = values

    for axis in _STRATIFY_AXES:
        for value in sorted(by_axis_value[axis], key=repr):
            if len(chosen) >= n:
                break
            pool = [c for c in by_axis_value[axis][value] if _candidate_key(c) not in chosen_keys]
            if not pool:
                continue
            pick = rng.choice(pool)
            chosen.append(pick)
            chosen_keys.add(_candidate_key(pick))

    remaining = [c for c in space_sorted if _candidate_key(c) not in chosen_keys]
    rng.shuffle(remaining)
    for c in remaining:
        if len(chosen) >= n:
            break
        chosen.append(c)
        chosen_keys.add(_candidate_key(c))

    coverage = {
        axis: dict(levels_in_space=len(by_axis_value[axis]),
                   levels_in_sample=len(set(c[axis] for c in chosen)))
        for axis in _STRATIFY_AXES
    }
    coverage["n_requested"] = n
    coverage["n_chosen"] = len(chosen)
    coverage["must_include_total"] = len(must_include) if must_include else 0
    coverage["must_include_matched"] = n_must_matched
    coverage["must_include_unmatched"] = must_unmatched
    return chosen, coverage


def seed_design(space: list[dict], numtiles_targets: tuple = (None, 12), max_omp: int = 14
                ) -> list[dict]:
    """Cold-start screen for a case/allocation with no measurements (e.g. 1-node cases where the
    448-core templates do not exist): for every ranks x omp group, the most balanced legal patch
    shape (min patch aspect), default 1-D tiles, dist=block, KMP_BLOCKTIME=0, at numtiles nearest
    each target (None = one tile per thread), for BOTH the most balanced patch and the widest-in-x
    patch (see the comment in the body: balanced alone lost by 25% on zhengzhou). Thread counts above
    `max_omp` are skipped (lafe 8x56 segfaulted, job 1078094). Deterministic; ties by _candidate_key."""
    def aspect(c):
        px, py = c.get("patch_x") or 0, c.get("patch_y") or 0
        return max(px, py) / float(min(px, py)) if px and py else float("inf")

    base = [c for c in space if c["omp"] <= max_omp and c["tile_shape"] == "1d_y" and c["dist"] == "block"
            and str(c["blocktime"]) == "0"]
    chosen, seen = [], set()
    for omp in sorted({c["omp"] for c in base}):
        group = [c for c in base if c["omp"] == omp]
        if not group:
            continue
        # Two shapes per thread count: an x-wide patch vectorises better than the most balanced
        # one (zhengzhou 4x28 2.780 vs 14x8 3.477 s/step, jobs 1080108/1078389; lafe, karina too).
        shapes = []
        for key in (lambda c: (aspect(c), _candidate_key(c)),          # most balanced patch
                    lambda c: (c["nx"], -c["ny"], _candidate_key(c))):  # widest patch in x
            pick = min(group, key=key)
            if (pick["nx"], pick["ny"]) not in shapes:
                shapes.append((pick["nx"], pick["ny"]))
        for shape_xy in shapes:
            shape = [c for c in group if (c["nx"], c["ny"]) == shape_xy]
            for target in numtiles_targets:
                t = max(1, omp) if target is None else target
                c = min(shape, key=lambda c: (abs(c["numtiles"] - t), c["numtiles"]))
                if _candidate_key(c) not in seen:
                    seen.add(_candidate_key(c))
                    chosen.append(c)
    return chosen


def to_campaign_configs(candidates: list[dict], prefix: str = "G") -> list[dict]:
    """Config dicts campaign/configs.py's generated-config runner (CONFIG_FILE=... env,
    campaign/strict_campaign.sbatch) can execute directly: zero-padded ids (G0001, G0002, ...,
    width sized to len(candidates), minimum 4 digits) in the given order, and exactly the fields
    `edit_namelist`/the `edit-namelist` CLI and the TSV the sbatch runner reads expect."""
    width = max(4, len(str(len(candidates))))
    out = []
    for i, c in enumerate(candidates, start=1):
        cid = "%s%0*d" % (prefix, width, i)
        out.append(dict(
            id=cid, ranks=c["ranks"], omp=c["omp"], nx=c["nx"], ny=c["ny"],
            numtiles=c["numtiles"], dist=c["dist"], blocktime=c["blocktime"],
            patch_x=c.get("patch_x"), patch_y=c.get("patch_y"),
            numtiles_x=c.get("numtiles_x", 0), numtiles_y=c.get("numtiles_y", 0),
            tile_sz_x=c.get("tile_sz_x", 0), tile_sz_y=c.get("tile_sz_y", 0),
            tile_strategy=c.get("tile_strategy", 0),
            tile_shape=c.get("tile_shape"),
            config_source="generated",
        ))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_summary(argv):
    ap = argparse.ArgumentParser(prog="decomp_space.py summary")
    ap.add_argument("case_id")
    ap.add_argument("--total-cores", type=int, default=448)
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--hardware-id", default=DEFAULT_HARDWARE)
    a = ap.parse_args(argv)
    case = campaign_configs.CASES.get(a.case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    candidates, axis = enumerate_space(case, a.total_cores, a.nodes, a.hardware_id)
    print("case=%s total_candidates=%d" % (a.case_id, axis["total"]))
    print("omp_options=%s" % (axis["omp_options"],))
    for omp in axis["omp_options"]:
        print("  omp=%-3d ranks=%-4d legal_shapes=%-3d numtiles_candidates=%s"
              % (omp, axis["ranks_by_omp"][omp], axis["shapes_count"][omp],
                 axis["numtiles_options"][omp]))
    print("tile_shape_options=%s" % (axis["tile_shape_options"],))
    print("dist_options=%s  blocktime_options=%s" % (axis["dist_options"], axis["blocktime_options"]))
    preds = campaign_template_predicates(case)
    matched = sum(1 for p in preds if any(_matches_predicate(c, p) for c in candidates))
    print("campaign-template predicates: %d total, %d matched by >=1 candidate, %d unmatched"
          % (len(preds), matched, len(preds) - matched))
    return 0


def _cmd_sample(argv):
    ap = argparse.ArgumentParser(prog="decomp_space.py sample")
    ap.add_argument("case_id")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--total-cores", type=int, default=448)
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--hardware-id", default=DEFAULT_HARDWARE)
    ap.add_argument("--no-must-include", action="store_true",
                     help="skip campaign_template_predicates coverage (pure stratified sample)")
    ap.add_argument("--prefix", default="G")
    a = ap.parse_args(argv)
    case = campaign_configs.CASES.get(a.case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    candidates, axis = enumerate_space(case, a.total_cores, a.nodes, a.hardware_id)
    must_include = None if a.no_must_include else campaign_template_predicates(case)
    chosen, coverage = sample_design(candidates, a.n, a.seed, must_include=must_include)
    configs_out = to_campaign_configs(chosen, prefix=a.prefix)
    result = dict(case_id=a.case_id, n=a.n, seed=a.seed, total_space=axis["total"],
                  coverage=coverage, configs=configs_out)
    text = json.dumps(result, indent=2, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print("wrote %d configs (of %d requested, space=%d) to %s"
              % (len(configs_out), a.n, axis["total"], a.out))
    else:
        print(text)
    return 0


def _cmd_seed(argv):
    ap = argparse.ArgumentParser(prog="decomp_space.py seed")
    ap.add_argument("case_id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-cores", type=int, default=448)
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--hardware-id", default=DEFAULT_HARDWARE)
    ap.add_argument("--prefix", default="S")
    a = ap.parse_args(argv)
    case = campaign_configs.CASES.get(a.case_id)
    if case is None:
        sys.stderr.write("unknown case %r\n" % (a.case_id,))
        return 2
    candidates, axis = enumerate_space(case, a.total_cores, a.nodes, a.hardware_id)
    configs_out = to_campaign_configs(seed_design(candidates), prefix=a.prefix)
    with open(a.out, "w") as f:
        json.dump(dict(case_id=a.case_id, total_space=axis["total"], configs=configs_out), f,
                  indent=2, sort_keys=True)
    print("wrote %d seed configs (space=%d) to %s" % (len(configs_out), axis["total"], a.out))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.stderr.write(
            "usage: decomp_space.py summary CASE_ID [--total-cores N] [--nodes N] "
            "[--hardware-id ID] | "
            "decomp_space.py sample CASE_ID --n N --seed S [--out FILE.json] "
            "[--no-must-include] [--prefix P] | "
            "decomp_space.py seed CASE_ID --out FILE.json [--total-cores N] [--nodes N] "
            "[--hardware-id ID] [--prefix P]\n"
        )
        return 2
    if argv[0] == "summary":
        return _cmd_summary(argv[1:])
    if argv[0] == "sample":
        return _cmd_sample(argv[1:])
    if argv[0] == "seed":
        return _cmd_seed(argv[1:])
    sys.stderr.write("unknown subcommand %r\n" % (argv[0],))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
