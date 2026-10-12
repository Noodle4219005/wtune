#!/usr/bin/env python3
"""wtune decomp -- every decomposition axis WRF actually exposes, in one place.

`wtune/shape.py` ranks nproc_x x nproc_y and picks a numtiles. That is four of the eleven axes
the source actually has. This enumerates all of them, applies the constraint each one really
carries, and says WHY the rejected ones are rejected -- with a source line or a measurement,
never an assertion.

Everything here is integer arithmetic plus frozen coefficients: no run required.

AXES THAT EXIST AND ARE USABLE
    nproc_x, nproc_y        Cartesian MPI decomposition. TASK_FOR_POINT computes ownership by
                            integer division per axis; there is no table and no file hook, so
                            an external partitioner cannot supply the grid split.
    grid size               choosing e_we/e_sn divisible by nproc removes the remainder, which
                            is otherwise ceil(E/nx)*ceil(F/ny) and provably irreducible.
    numtiles                cache blocking. The invariant is per COLUMN-cell (cells x levels),
                            measured at e_vert=50; writing it per cell assumes 50 levels.
    numtiles_x/y            tile SHAPE. Registry.EM_COMMON; bypasses the automatic 1D-Y -> 2D
                            switch at module_tiles.F:337.
    tile_sz_x/y             tile size; overrides numtiles entirely (module_tiles.F:353).
    tile_strategy           TILE_NONE=0 TILE_X=1 TILE_Y=2 TILE_XY=3 (module_machine.F:11).
    WRF_NUM_TILES_X/_Y      environment, read per rank (module_tiles.F:298,310) -- so tile
                            shape can differ BETWEEN ranks without touching the namelist.
    ranks x threads         needs a dm+sm build. Threads of one rank must stay inside one NUMA
                            domain: rank counts that do not divide the domain count cost 1.9x.
    rank -> node            not a WRF knob at all; applied with srun --distribution=arbitrary.
                            Intel MPI's -machinefile does NOT honour a per-rank host order.

AXES THAT LOOK USABLE AND ARE NOT
    lats_to_mic             REAL and live: it makes two y-bands of different heights
                            (task_for_point.c, the minx==-99 branch, which is in the live #else).
                            But module_dm.F:171-173 forces ntasks_x=P/2, ntasks_y=2, and it is
                            mutually exclusive with nproc_x/nproc_y. On 901^2 with 448 ranks
                            that is a 4-cell-wide patch: below WRF's own 10-cell floor.
    nproc_z                 does not exist. Zero hits in Registry.EM_COMMON and module_dm.F.
    reorder_mesh            only in module_io_quilt_old.F, the old I/O quilting path. Not in
                            the decomposition.
    swap_x/y, cycle_x/y     the ARW point-redistribution call is inside #if 0 (solve_em.F).
    WSM5 CHUNK              no CHUNK in module_mp_wsm5.F in this build, and these cases run
                            mp_physics=8 (Thompson) anyway.
"""
from __future__ import annotations

import argparse

MIN_PATCH = 10                  # measured: 4x112 -> patch 225x8 -> FATAL on both sweep cases
FIT_W_MIN, FIT_W_MAX = 28, 128  # width model's fitted range; 225 was predicted best, came third
TILE_COLUMN_CELLS = 251 * 50    # cells x levels per tile; both anchors ran at e_vert=50
ALPHA, BETA, GAMMA = 0.820982, 13.728888, 1.023000    # R^2 0.9967, max resid 1.16 pp
NUMA_PER_NODE, CORES_PER_NUMA = 8, 14                 # Xeon 8480+, SMT off

# Measured wall time per step for whole (ranks, threads, numtiles) configurations, karina 901^2
# on 4 nodes. The shape model below was fitted at a FIXED rank count, so it describes how shape
# changes efficiency at constant work per rank -- it does NOT know that halving the rank count
# doubles each rank's work while halving the number of halo exchanges. Measured: 448->224 ranks
# at one thread costs only +18% (0.34065 -> 0.40205), not +100%. Using the shape model across
# rank counts put the shipping 8x56x1 at +34.2% when it is really +6.0%, so cross-group ranking
# uses these anchors only, and groups without one are reported as incomparable.
ANCHOR = {                       # (nranks, threads, numtiles) -> med s/step
    (448, 1, 6):  0.34065,       # jobs 1068311-12, dmsm
    (224, 1, 13): 0.40205,       # jobs 1068311-12, half the cores idle
    (224, 2, 13): 0.32135,       # jobs 1068311-12
    (64,  7, 46): 1.07090,       # jobs 1067877-78
}

# Measured thread scaling. Patch-dependent, and only two patch sizes have been measured, so
# anything else is reported as unmeasured rather than interpolated.
#   patch 12769 (64 ranks,  numtiles=46): jobs 1067877-78, 2 reps agreeing to 1%
#   patch  3584 (224 ranks, numtiles=13): jobs 1068311-12, speedup(2)=0.40205/0.32135
THREAD_SPEEDUP = {
    12769: {1: 1.000, 2: 0.670, 3: 0.831, 4: 0.935, 5: 0.986, 6: 1.051, 7: 0.998},
    3584:  {1: 1.000, 2: 1.251},
}


def patch_dims(extent: int, nproc: int) -> tuple[int, int]:
    q, rem = divmod(extent, nproc)
    return (q + 1 if rem else q), q


def factor_pairs(n: int):
    return [(i, n // i) for i in range(1, n + 1) if n % i == 0]


def tiles_for(cells: int, e_vert: int) -> int:
    return max(1, round(cells * e_vert / TILE_COLUMN_CELLS))


def speedup(patch_cells: int, threads: int):
    """Measured speedup or None. Nearest measured patch within 30%, else unmeasured."""
    if threads == 1:
        return 1.0, "exact"
    best = None
    for p, tab in THREAD_SPEEDUP.items():
        if threads in tab and abs(p - patch_cells) / p < 0.30:
            if best is None or abs(p - patch_cells) < abs(best[0] - patch_cells):
                best = (p, tab[threads])
    if best:
        return best[1], "measured at patch %d" % best[0]
    return None, "UNMEASURED"


def enumerate_configs(e_we, e_sn, e_vert, cores, nodes):
    doms = nodes * NUMA_PER_NODE
    out = []
    for threads in (1, 2, 3, 4, 6, 7, 14):
        if cores % threads:
            continue
        nranks = cores // threads
        if nranks % doms:                       # ranks must divide the NUMA domain count
            continue
        per_dom = nranks // doms
        if per_dom * threads != CORES_PER_NUMA:  # a domain's threads must fill it exactly
            continue
        for nx, ny in factor_pairs(nranks):
            wmax, wmin = patch_dims(e_we, nx)
            hmax, hmin = patch_dims(e_sn, ny)
            if wmin < MIN_PATCH or hmin < MIN_PATCH:
                continue
            cells_min, cells_max = wmin * hmin, wmax * hmax
            mean = e_we * e_sn / nranks
            shape = ALPHA + BETA / wmin + GAMMA / hmin
            sp, prov = speedup(cells_min, threads)
            out.append(dict(
                nranks=nranks, threads=threads, nx=nx, ny=ny, per_dom=per_dom,
                wmin=wmin, hmin=hmin, cells_max=cells_max, cells_min=cells_min,
                geo=100 * (cells_max / mean - 1),
                numtiles=tiles_for(cells_min, e_vert),
                even=(e_we % nx == 0 and e_sn % ny == 0),
                shape=shape, sp=sp, prov=prov,
                extrapolated=not (FIT_W_MIN <= wmin <= FIT_W_MAX),
                # within-group score only: shape efficiency x the fattest patch's share
                score=shape * (cells_max / mean),
            ))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="every decomposition axis WRF exposes")
    p.add_argument("--e-we", type=int, default=901)
    p.add_argument("--e-sn", type=int, default=901)
    p.add_argument("--e-vert", type=int, default=45)
    p.add_argument("--cores", type=int, default=448)
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args()

    cfg = enumerate_configs(a.e_we, a.e_sn, a.e_vert, a.cores, a.nodes)
    for c in cfg:
        key = (c["nranks"], c["threads"], c["numtiles"])
        c["anchor"] = ANCHOR.get(key)
    groups = {}
    for c in cfg:
        if c["extrapolated"]:
            continue
        groups.setdefault((c["nranks"], c["threads"]), []).append(c)
    for g in groups.values():
        g.sort(key=lambda c: c["score"])
        best = g[0]["score"]
        for c in g:
            c["rel_in_group"] = 100 * (c["score"] / best - 1)
    # order the GROUPS by their measured anchor; groups without one go last, marked
    anchored = sorted([k for k in groups if any(c["anchor"] for c in groups[k])],
                      key=lambda k: min(c["anchor"] for c in groups[k] if c["anchor"]))
    unanchored = [k for k in groups if k not in anchored]
    ranked = [c for k in anchored for c in groups[k][:3]]
    unk = [c for c in cfg if c["extrapolated"]] + [c for k in unanchored for c in groups[k][:1]]

    print("grid %dx%dx%d | %d cores on %d nodes | %d NUMA domains of %d cores\n"
          % (a.e_we, a.e_sn, a.e_vert, a.cores, a.nodes, a.nodes * NUMA_PER_NODE, CORES_PER_NUMA))
    print("%-9s %-4s %-11s %-9s %-8s %-8s %-11s %s"
          % ("nproc", "thr", "patch", "cells max", "geo imb", "numtiles", "measured", "within group"))
    if not ranked:
        print("  (no group has a measured anchor)")
    base = None
    for c in ranked[:a.top]:
        if c["anchor"] and base is None: base = c["anchor"]
        meas = ("%.5f (%+.1f%%)" % (c["anchor"], 100 * (c["anchor"] / base - 1))) if c["anchor"] else "-"
        tie = "  <- tied" if 0 < c["rel_in_group"] <= 1.2 else ""
        print("%-9s %-4d %-11s %-9d %7.2f%% %-8d %-11s %+.1f%%%s"
              % ("%dx%d" % (c["nx"], c["ny"]), c["threads"], "%dx%d" % (c["wmin"], c["hmin"]),
                 c["cells_max"], c["geo"], c["numtiles"], meas, c["rel_in_group"], tie))
    if unk:
        print("\nnot ranked (%d):" % len(unk))
        for c in unk[:6]:
            why = ("width %d outside the fitted range %d-%d" % (c["wmin"], FIT_W_MIN, FIT_W_MAX)
                   if c["extrapolated"] else
                   "no measured wall time for %d ranks x %d threads -- shape model cannot compare "
                   "across rank counts" % (c["nranks"], c["threads"]))
            print("  %-9s thr=%-3d patch %-10s %s"
                  % ("%dx%d" % (c["nx"], c["ny"]), c["threads"], "%dx%d" % (c["wmin"], c["hmin"]), why))

    if ranked:
        b = ranked[0]
        print("\n--- recommended ---")
        print("&domains")
        print("  nproc_x = %d, nproc_y = %d," % (b["nx"], b["ny"]))
        print("  numtiles = %d," % b["numtiles"])
        print("/")
        print("threads per rank : %d   (%d ranks, %d per NUMA domain, %d cores each)"
              % (b["threads"], b["nranks"], b["per_dom"], b["threads"]))
        if b["threads"] > 1:
            print("build            : a dm+sm tree -- tile knobs are inert without -qopenmp")
            print("launcher         : srun --mpi=pmi2 -n %d --ntasks-per-node=%d -c %d"
                  % (b["nranks"], b["nranks"] // a.nodes, b["threads"]))
            print("                   (Intel MPI -machinefile ignores per-rank host order)")
        print("tile shape       : leave numtiles_x/y unset -> TILE_Y keeps the full patch width")
        print("                   (width dominates cost: 64x28 and 28x64 have identical halo")
        print("                    and differ by 23 pp), and check rsl.error for")
        print("                   'Going 2D' -- module_tiles.F:337 switches silently when")
        print("                   numtiles > patch height %d" % b["hmin"])
        print("cost-model size  : %d  (max patch, not e_we*e_sn/ranks: the fattest rank sets"
              % b["cells_max"])
        print("                   makespan and carries %+.2f%% more cells)" % b["geo"])
        if not b["even"]:
            print("grid             : %dx%d leaves a remainder. ceil(E/nx)*ceil(F/ny) is the"
                  % (a.e_we, a.e_sn))
            print("                   minimum achievable maximum -- 200k random splits never beat")
            print("                   it -- so only a divisible grid removes it. That is a case")
            print("                   design change (fewer cells), not a tuner knob.")
    print("\n--- axes deliberately not offered ---")
    for name, why in (
        ("lats_to_mic",
         "live, and the only namelist path to unequal patches, but module_dm.F:171-173 forces "
         "ntasks_x=P/2 ntasks_y=2 -> patch width %d on this grid, below the %d floor"
         % (max(1, a.e_we // max(1, a.cores // 2)), MIN_PATCH)),
        ("nproc_z", "does not exist: zero hits in Registry.EM_COMMON and module_dm.F"),
        ("reorder_mesh", "only in module_io_quilt_old.F, not the decomposition path"),
        ("swap_x/y, cycle_x/y", "the ARW point-redistribution call is inside #if 0"),
        ("rank->node placement", "not a decomposition knob; separate tool, measured -0.99% at "
                                 "448x1 and null under hybrid"),
    ):
        print("  %-22s %s" % (name, why))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
