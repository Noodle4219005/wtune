#!/usr/bin/env python3
"""VENDORED: sea-level pressure from a wrfout frame (NCL `wrf_slp` / RIP4 `DCOMPUTESEAPRS`).

# VENDORED for this project -- do NOT import the upstream copy at runtime.
#
#   upstream repo   git@github.com:Noodle4219005/my-sc26-wrf.git  ("SC26-WRF")
#   upstream path   docs/testcase-atlas/tools/wrf_slp.py
#   upstream commit 66b04c0fae511c49a8076971646ac8c09f07ddb3  (2026-09-07; last commit
#                   touching that file. Repo HEAD when vendored: 1f72e60ce07541f1d4c6c9cd
#                   8cd06a5fd1ecbc24, working tree clean for this path)
#   sha256          2aae722ce3baa25faf82c9069c82aa384d5e75151cf3a7bcd21ad88bc6820243
#   vendored on     2026-09-17
#   local checkout  /home/u5727520/SC26-WRF/docs/testcase-atlas/tools/wrf_slp.py
#
# Why vendored rather than imported: WTune must be self-contained. The same lesson as
# tools/wrf_adios2_to_netcdf4/convert.py -- an external copy that moved/changed silently
# broke a campaign. ~/SC26-WRF is a different repository on a different branch cadence and
# is not guaranteed to exist on a machine where WTune runs.
#
# Changes from upstream (kept deliberately minimal so the diff stays auditable):
#   1. docstring translated/extended with this provenance block;
#   2. `slp_from_frame` gained no new behaviour -- the numerics are byte-for-byte the
#      upstream expressions (RD, G, USSALR, TC, PCONST, the level search, the RIP
#      t_sea_level correction branch). Do not "improve" them: matching NCL exactly is the
#      whole point, otherwise a difference comes from the algorithm and not from the model.
#
# Upstream's own rationale, preserved: the reference solutions we compare against were
# produced with NCL, so the PCONST=100 hPa reference level, the 0.0065 K/m extrapolation
# and the RIP t_sea_level branch must all be reproduced rather than approximated.
"""
import numpy as np

RD = 287.04
G = 9.81
USSALR = 0.0065          # standard-atmosphere lapse rate, K/m
TC = 273.16 + 17.5
PCONST = 10000.0         # Pa


def slp_from_frame(P, PB, T, QVAPOR, PH, PHB):
    """One time level. All inputs (nz, ny, nx); PH/PHB are (nz+1, ny, nx). Returns hPa."""
    p = P + PB                                   # Pa
    tk = (T + 300.0) * (p / 100000.0) ** (2.0 / 7.0)
    q = QVAPOR
    gh = (PH + PHB) / G                          # full (staggered) geopotential height, m
    z = 0.5 * (gh[:-1] + gh[1:])                 # half (mass) levels

    nz, ny, nx = p.shape
    p0 = p[0]                                    # lowest model level pressure
    target = p0 - PCONST

    # level = first level with p < p0 - PCONST (NCL is 1-based, this is 0-based)
    below = p < target[None, :, :]
    level = np.where(below.any(axis=0), below.argmax(axis=0), -1)
    if (level < 0).any():
        raise ValueError('some column has no level above p0-100hPa -- too few vertical levels')

    klo = np.maximum(level - 1, 0)
    khi = np.minimum(klo + 1, nz - 1)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    plo, phi = p[klo, jj, ii], p[khi, jj, ii]
    tlo = tk[klo, jj, ii] * (1.0 + 0.608 * q[klo, jj, ii])
    thi = tk[khi, jj, ii] * (1.0 + 0.608 * q[khi, jj, ii])
    zlo, zhi = z[klo, jj, ii], z[khi, jj, ii]

    lr = np.log(target / phi) / np.log(plo / phi)
    t_at = thi - (thi - tlo) * lr
    z_at = zhi - (zhi - zlo) * lr

    t_surf = t_at * (p0 / target) ** (USSALR * RD / G)
    t_sea = t_at + USSALR * z_at

    # RIP correction branch: t_sea < TC and t_surf <= TC  =>  t_sea = TC - 0.005*(t_surf-TC)^2
    m = (t_sea < TC) & (t_surf <= TC)
    t_sea = np.where(m, TC - 0.005 * (t_surf - TC) ** 2, t_sea)

    return 0.01 * p0 * np.exp((2.0 * G * z[0]) / (RD * (t_sea + t_surf)))
