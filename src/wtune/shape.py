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

Only `patch_dims` is still called (wtune_case.py). The ranking functions and the command line
are in archive/unused-2026-10/src/wtune_shape_legacy.py; the measured constants stay here.
"""
from __future__ import annotations

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


def patch_dims(extent: int, nproc: int) -> tuple[int, int]:
    """WRF hands the integer-division remainder out as one extra cell to `rem` patches."""
    q, rem = divmod(extent, nproc)
    return (q + 1 if rem else q), q
