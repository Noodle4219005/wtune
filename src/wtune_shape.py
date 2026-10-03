#!/usr/bin/env python3
"""wtune shape -- decomposition and grid-shape advisor. Pure integer arithmetic, no model fit
at runtime, no measurement, no training data.

WHY THIS EXISTS AND WHY IT IS NOT A LEARNED MODEL
-------------------------------------------------
Seven WRF cases spanning hurricane, stable boundary layer, deep convection, atmospheric river,
Alaskan winter, Amazonian convection and Pacific synoptic were profiled per phase. The per-rank
imbalance splits cleanly in two, and the split is visible in one column -- the standard
deviation ACROSS cases:

    rk_tend_tim          imbalance 20.6% +- 0.8      <- seven different weathers
    small_step_prep_tim  imbalance 16.9% +- 1.2
    rk_scalar_tend_tim   imbalance 14.3% +- 2.0
    micro_driver_tim     imbalance 89.1% +- 33.8     <- the only weather-driven one

An imbalance that is constant to +-0.8 across seven unrelated weather situations is not
weather. It is geometry: the integer-division remainder of e_we/nproc_x and e_sn/nproc_y, plus
the boundary relaxation zone. And geometry does not need a predictor -- it needs division.

6 of those 7 cases put their globally slowest rank on a maximum-cell patch, and two unrelated
cases (arctic_sbl, amazon_conv) put it on the SAME rank #425. That is what this advisor
computes, in closed form, before any run exists.

THE COST MODEL, AND THE ONE THING IT IS ALLOWED TO CLAIM
--------------------------------------------------------
Coefficients come from a controlled sweep (jobs 1062661/1062662: one execution directory, one
real.exe, io_form=14, no history writes, only nproc_x x nproc_y changed) and are FROZEN here.
The model is a function of patch width because that is what the sweep says it is:

    64x28 and 28x64 have identical halo surface and differ by 23 percentage points.

So halo volume is not the driver; the length of the contiguous i-dimension inner loop is.
`--validate` replays the sweep and prints the residuals rather than asserting the fit is good.
"""
from __future__ import annotations

import argparse

MIN_PATCH = 10          # WRF refuses a patch thinner than this in either dimension;
                        # measured: 4x112 -> patch 225x8 -> FATAL on both cases in the sweep.

# Controlled sweep, jobs 1062661 / 1062662, 901^2 / 448 ranks.
# (nproc_x, nproc_y, patch_w, patch_h, karina med s/step, arctic med s/step); None = FATAL.
SWEEP = [
    (4, 112, None, None),
    (7, 64, 0.3703, 0.6079),
    (8, 56, 0.3666, 0.6069),
    (14, 32, 0.3977, 0.6492),
    (16, 28, 0.4018, 0.6691),
    (28, 16, 0.4548, 0.7724),
    (32, 14, 0.4824, 0.8224),
]
SWEEP_GRID = 901
# Model residual on its own training sweep. Two decompositions whose predicted costs sit inside
# this band are NOT distinguishable by the model and are reported as tied rather than ranked --
# the measured 8x56 vs 7x56 gap is 0.6%, below both this band and the 0.68% run-to-run floor.
TIE_BAND_PP = 1.2

# The width model was fitted on patch widths 28..128 and it EXTRAPOLATES BADLY above that.
# At 224 ranks the advisor ranked 4x56 (patch width 225) first at -2.0%; measured it came third
# at +4.5%, behind 8x28 (width 113). 1/W keeps promising a benefit that stops arriving once the
# inner loop is long enough. Candidates outside the fitted range are reported, not ranked.
FIT_W_MIN, FIT_W_MAX = 28, 128

# Dimensionless: rel_cost = ALPHA + BETA/patch_width + GAMMA/patch_height, least squares on the
# mean of the two cases' relative curves. Frozen; refit only from a new controlled sweep.
ALPHA, BETA, GAMMA = 0.820982, 13.728888, 1.023000   # R^2 = 0.9967, max resid 1.16 pp


# The tiling invariant is a CACHE-BLOCKING one, so its unit is column-cells (horizontal cells x
# vertical levels), not horizontal cells. Both measurements that established it ran at e_vert=50
# (1501x1201; namelist read back from wtune-numtiles-a-8x56-nt{8,16}-20260902), which is why it
# was first written down as "251 cells".
#
#   1 node  / 112 ranks, patch 16071 cells, numtiles=64 -> 251 cells/tile, -9.1% (3 reps, eps=0.5%)
#   4 nodes / 448 ranks, patch  3948 cells, numtiles=16 -> 247 cells/tile, -1.72%
#
# Writing it per-cell silently assumes every case has 50 levels. This corpus does not:
# e_vert is 45 for karina/alaska/amazon/synoptic, 60 for atmos_river, and 80 for arctic_sbl and
# deep_conv_ot -- a 78% swing in working set that a per-cell rule cannot see.
TILE_COLUMN_CELLS = 251 * 50    # = 12550 cells x levels per tile
REF_E_VERT = 50


def tiling(patch_w: int, patch_h: int, e_vert: int = REF_E_VERT) -> dict:
    """Tile count from the column-cell invariant, plus what WRF will actually do with it.

    Setting `numtiles` alone is a REQUEST. frame/module_tiles.F:337 silently converts 1D-Y
    tiling to 2D when `num_tiles > patch_height` (MIN_TILE_SIZE is 1), and any build without
    -qopenmp ignores the request entirely and runs one tile per patch.
    """
    n = max(1, round(patch_w * patch_h * e_vert / TILE_COLUMN_CELLS))
    return dict(numtiles=n, tile_cells=patch_w * patch_h / n,
                tile_col_cells=patch_w * patch_h * e_vert / n,
                # default strategy is TILE_Y: full patch width, split in y only
                tile_w=patch_w, tile_h=max(1, patch_h // n),
                auto_2d=n > patch_h)


def patch_dims(extent: int, nproc: int) -> tuple[int, int]:
    """WRF hands the integer-division remainder out as one extra cell to `rem` patches."""
    q, rem = divmod(extent, nproc)
    return (q + 1 if rem else q), q


def factor_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, n // i) for i in range(1, n + 1) if n % i == 0]


def evaluate(e_we: int, e_sn: int, nx: int, ny: int) -> dict:
    wmax, wmin = patch_dims(e_we, nx)
    hmax, hmin = patch_dims(e_sn, ny)
    cells_max, cells_mean = wmax * hmax, (e_we * e_sn) / (nx * ny)
    legal = wmin >= MIN_PATCH and hmin >= MIN_PATCH
    # cost is driven by the typical patch; imbalance by the fattest one
    step = ALPHA + BETA / wmin + GAMMA / hmin if legal else float("inf")
    return dict(
        nx=nx, ny=ny, wmin=wmin, wmax=wmax, hmin=hmin, hmax=hmax,
        cells_max=cells_max, cells_mean=cells_mean,
        geo_imb=100 * (cells_max / cells_mean - 1),
        even=(e_we % nx == 0 and e_sn % ny == 0),
        step=step, legal=legal,
        score=step * (cells_max / cells_mean),
        extrapolated=not (FIT_W_MIN <= wmin <= FIT_W_MAX),
    )


def rank(e_we: int, e_sn: int, ranks: int) -> list[dict]:
    rows = [evaluate(e_we, e_sn, nx, ny) for nx, ny in factor_pairs(ranks)]
    # extrapolated candidates sort last: the model has no evidence about them
    rows.sort(key=lambda r: (not r["legal"], r["extrapolated"], r["score"]))
    return rows


def rank_nested(doms: list[tuple[int, int]], ratios: list[int], ranks: int) -> list[dict]:
    """Nested runs share one nproc_x x nproc_y across every domain.

    Two things change versus a single grid, and both bite:
      * LEGALITY is set by the innermost domain, not the parent. On nest_move (901^2 parent,
        601^2 nest) 7x64 is legal on the parent (128x14) and ILLEGAL on the nest (85x9).
      * COST is a weighted sum: a nest at parent_grid_ratio r runs r steps per parent step, so
        it contributes cells x r, and for 901^2/601^2/r=3 the nest is 57% of the work despite
        being the smaller grid.
    """
    out = []
    for nx, ny in factor_pairs(ranks):
        per = [evaluate(we, sn, nx, ny) for we, sn in doms]
        w = [we * sn * r for (we, sn), r in zip(doms, ratios)]
        tot = sum(w)
        out.append(dict(
            nx=nx, ny=ny, per=per,
            legal=all(p["legal"] for p in per),
            binding=min(range(len(per)), key=lambda i: min(per[i]["wmin"], per[i]["hmin"])),
            score=sum(p["score"] * wi for p, wi in zip(per, w)) / tot
                  if all(p["legal"] for p in per) else float("inf"),
            weights=[100 * wi / tot for wi in w],
        ))
    out.sort(key=lambda r: (not r["legal"], r["score"]))
    return out


def even_grids(nx: int, ny: int, e_we: int, e_sn: int, span: int = 60) -> list[dict]:
    """Grid sizes near the requested one that divide exactly by this decomposition.

    A smaller grid is FEWER CELLS, not a faster code path. The cell delta is reported so the
    two can never be confused; this is a case-design suggestion, not a tuner knob.
    """
    out = []
    for we in range(e_we - span, e_we + span + 1):
        if we <= 0 or we % nx:
            continue
        for sn in range(e_sn - span, e_sn + span + 1):
            if sn <= 0 or sn % ny:
                continue
            if we // nx < MIN_PATCH or sn // ny < MIN_PATCH:
                continue
            out.append(dict(e_we=we, e_sn=sn,
                            dcells=100 * (we * sn) / (e_we * e_sn) - 100,
                            dist=abs(we - e_we) + abs(sn - e_sn)))
    out.sort(key=lambda r: (r["dist"], abs(r["dcells"])))
    return out


def validate() -> int:
    print("replaying the controlled sweep (jobs 1062661 / 1062662, 901^2 / 448 ranks)\n")
    print("%-8s %-10s %8s %9s %9s %8s" % ("shape", "patch", "measured", "predicted", "residual", "legal"))
    ref = next(i for i, r in enumerate(SWEEP) if (r[0], r[1]) == (8, 56))
    for nx, ny, kar, arc in SWEEP:
        r = evaluate(SWEEP_GRID, SWEEP_GRID, nx, ny)
        if kar is None:
            ok = "FATAL (caught)" if not r["legal"] else "FATAL (MISSED)"
            print("%-8s %-10s %8s %9s %9s %8s"
                  % (f"{nx}x{ny}", f"{r['wmin']}x{r['hmin']}", "FATAL", "--", "--", ok))
            continue
        mrel = 100 * ((kar / SWEEP[ref][2] + arc / SWEEP[ref][3]) / 2 - 1)
        prel = 100 * (r["step"] / evaluate(SWEEP_GRID, SWEEP_GRID, 8, 56)["step"] - 1)
        print("%-8s %-10s %+7.1f%% %+8.1f%% %+8.1f pp %8s"
              % (f"{nx}x{ny}", f"{r['wmin']}x{r['hmin']}", mrel, prel, prel - mrel,
                 "yes" if r["legal"] else "no"))
    rows = [r for r in rank(901, 901, 448) if r["legal"]]
    top = rows[0]
    print("\nadvisor's pick for 901^2 / 448 ranks: %dx%d" % (top["nx"], top["ny"]))
    if (top["nx"], top["ny"]) == (8, 56):
        print("MATCHES the measured optimum")
        return 0
    print("DOES NOT match the measured optimum 8x56 -- the model is wrong, do not ship")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description="WRF decomposition and grid-shape advisor")
    p.add_argument("--e-we", type=int, default=901)
    p.add_argument("--e-sn", type=int, default=901)
    p.add_argument("--ranks", type=int, default=448)
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--suggest-grid", action="store_true",
                   help="also list nearby grid sizes that divide exactly")
    p.add_argument("--domains", type=str, default=None,
                   help="nested run: comma list of WExSN per domain, e.g. 901x901,601x601")
    p.add_argument("--grid-ratio", type=str, default=None,
                   help="parent_grid_ratio per domain, e.g. 1,3")
    p.add_argument("--e-vert", type=int, default=REF_E_VERT,
                   help="vertical levels; the tiling invariant is per column-cell, not per cell")
    p.add_argument("--hybrid", action="store_true",
                   help="running the dmsm (MPI+OpenMP) tree, so the tile knobs actually apply")
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    if a.validate:
        return validate()

    if a.domains:
        doms = [tuple(int(v) for v in d.split("x")) for d in a.domains.split(",")]
        ratios = [int(v) for v in a.grid_ratio.split(",")] if a.grid_ratio else [1] * len(doms)
        cum = []
        acc = 1
        for r in ratios:
            acc *= r
            cum.append(acc)
        rows = rank_nested(doms, cum, a.ranks)
        legal = [r for r in rows if r["legal"]]
        print("%d domains %s, step multipliers %s, %d ranks -- %d/%d legal\n"
              % (len(doms), a.domains, cum, a.ranks, len(legal), len(rows)))
        print("work share: %s" % ", ".join("d%02d %.0f%%" % (i + 1, w)
                                           for i, w in enumerate(rows[0]["weights"])))
        print("\n%-9s %-26s %10s %9s" % ("nproc_x_y", "patch min per domain", "rel cost", "binding"))
        best = legal[0]["score"] if legal else 1.0
        for r in legal[:a.top]:
            print("%-9s %-26s %+9.1f%% %9s"
                  % (f"{r['nx']}x{r['ny']}",
                     " | ".join(f"{p['wmin']}x{p['hmin']}" for p in r["per"]),
                     100 * (r["score"] / best - 1), "d%02d" % (r["binding"] + 1)))
        for r in rows:
            if not r["legal"]:
                bad = [i + 1 for i, p in enumerate(r["per"]) if not p["legal"]]
                print("  rejected %-7s %-26s illegal on d%s"
                      % (f"{r['nx']}x{r['ny']}",
                         " | ".join(f"{p['wmin']}x{p['hmin']}" for p in r["per"]),
                         ",".join("%02d" % b for b in bad)))
        return 0 if legal else 2

    rows = rank(a.e_we, a.e_sn, a.ranks)
    legal = [r for r in rows if r["legal"]]
    print("grid %dx%d, %d ranks -- %d factorisations, %d legal (min patch >= %d cells)\n"
          % (a.e_we, a.e_sn, a.ranks, len(rows), len(legal), MIN_PATCH))
    if not legal:
        # This is the cont_diurnal failure mode: a 202^2 domain has no legal 448-rank split,
        # and Slurm only tells you after real.exe has already been launched.
        cap = (a.e_we // MIN_PATCH) * (a.e_sn // MIN_PATCH)
        best_r = max((r for r in range(1, a.ranks + 1)
                      if any(evaluate(a.e_we, a.e_sn, nx, ny)["legal"]
                             for nx, ny in factor_pairs(r))), default=0)
        print("NO legal decomposition of this grid across %d ranks." % a.ranks)
        print("  min patch is %d cells per side, so %dx%d admits at most %d ranks"
              % (MIN_PATCH, a.e_we, a.e_sn, cap))
        print("  largest rank count <= %d that works: %d" % (a.ranks, best_r))
        if best_r:
            b2 = [r for r in rank(a.e_we, a.e_sn, best_r) if r["legal"]][0]
            print("  at %d ranks the pick is nproc_x=%d, nproc_y=%d (patch %dx%d)"
                  % (best_r, b2["nx"], b2["ny"], b2["wmin"], b2["hmin"]))
        return 2

    print("%-9s %-12s %10s %9s %9s %7s" %
          ("nproc_x_y", "patch min", "max cells", "geo imb", "rel cost", "even"))
    best = legal[0]["step"]
    for r in legal[:a.top]:
        d = 100 * (r["step"] / best - 1)
        tie = " <- tied with the pick (model resid %.1f pp)" % TIE_BAND_PP \
            if 0 < d <= TIE_BAND_PP else ""
        if r["extrapolated"]:
            tie = " <- width %d outside the fitted range %d-%d; NOT RANKED" % (
                r["wmin"], FIT_W_MIN, FIT_W_MAX)
        print("%-9s %-12s %10d %8.2f%% %+8.1f%% %7s%s"
              % (f"{r['nx']}x{r['ny']}", f"{r['wmin']}x{r['hmin']}", r["cells_max"],
                 r["geo_imb"], d, "yes" if r["even"] else "-", tie))

    bad = [r for r in rows if not r["legal"]]
    if bad:
        print("\n%d factorisation(s) rejected: patch thinner than %d cells, e.g. %s"
              % (len(bad), MIN_PATCH,
                 ", ".join(f"{r['nx']}x{r['ny']} ({r['wmin']}x{r['hmin']})" for r in bad[:4])))
    b = legal[0]
    print("\nrecommended: nproc_x=%d, nproc_y=%d" % (b["nx"], b["ny"]))
    print("  patch %dx%d .. %dx%d ; fattest %d cells vs mean %.0f = %+.2f%% before any weather"
          % (b["wmin"], b["hmin"], b["wmax"], b["hmax"], b["cells_max"], b["cells_mean"], b["geo_imb"]))
    print("  cost-model size feature (use this, not e_we*e_sn/ranks): %d" % b["cells_max"])
    t = tiling(b["wmin"], b["hmin"], a.e_vert)
    print("\n  numtiles = %d   (patch %d cells x %d levels / %d-column-cell invariant"
          % (t["numtiles"], b["wmin"] * b["hmin"], a.e_vert, TILE_COLUMN_CELLS))
    print("                  -> %.0f cells x levels per tile)" % t["tile_col_cells"])
    if a.e_vert != REF_E_VERT:
        print("    (a per-cell rule would have said %d here -- e_vert %d vs the %d the invariant"
              % (max(1, round(b["wmin"] * b["hmin"] / 251)), a.e_vert, REF_E_VERT))
        print("     was measured at; the working set differs by %+.0f%%)"
              % (100 * (a.e_vert / REF_E_VERT - 1)))
    print("    default TILE_Y keeps the full patch width: tiles are %dx%d"
          % (t["tile_w"], t["tile_h"]))
    if t["auto_2d"]:
        print("    WARNING: %d > patch height %d, so module_tiles.F:337 will silently switch to"
              % (t["numtiles"], b["hmin"]))
        print("    2D tiling. Set numtiles_x/numtiles_y explicitly to keep control, and read back")
        print("    'Going 2D. New tiling is NxN' from rsl.error.0000.")
    if not a.hybrid:
        print("    INERT in this build: the shipping tree is dmpar (-qopenmp commented out in")
        print("    configure.wrf), so WRF runs one tile per patch whatever numtiles says. Pass")
        print("    --hybrid only when running the dmsm tree.")
    ties = [r for r in legal[1:] if 100 * (r["step"] / b["step"] - 1) <= TIE_BAND_PP]
    if ties:
        print("  NOT separable from: %s -- the model's own residual is %.1f pp and the measured"
              % (", ".join(f"{r['nx']}x{r['ny']}" for r in ties), TIE_BAND_PP))
        print("  run-to-run floor is 0.68%; the pick among these rests on geometric imbalance only")
    if b["even"]:
        print("  grid divides exactly -- no integer-division imbalance")
    elif a.suggest_grid:
        print("\n  grid does NOT divide exactly. Nearby sizes that do:")
        print("  %-14s %10s %12s" % ("e_we x e_sn", "cell delta", "geo imb"))
        for g in even_grids(b["nx"], b["ny"], a.e_we, a.e_sn)[:6]:
            print("  %-14s %+9.2f%% %11.2f%%" % (f"{g['e_we']}x{g['e_sn']}", g["dcells"], 0.0))
        print("\n  Removing the remainder deletes %.2f%% of geometric imbalance, but the measured"
              % b["geo_imb"])
        print("  wall gain on karina was only 1.06%: the fattest rank also carries weather, so")
        print("  flattening cells moves the makespan less than the imbalance figure suggests.")
        print("  A smaller grid is FEWER CELLS, not a faster code path -- never compare its")
        print("  wall clock against the original. This is a case-design suggestion.")
    else:
        print("  grid does not divide exactly; rerun with --suggest-grid for alternatives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
