"""post/condensate.py -- hydrometeor totals shared by every cloud/TC structure figure.

"Total condensate" is the sum of EVERY hydrometeor mixing ratio the run wrote, not QCLOUD+QICE:
in SC26-P2 (Thompson, mp=8) QCLOUD+QICE was 10.1% of the column condensate mass at F048 (QSNOW
74.4%, QRAIN 14.9%) and in SC26-P1 (WSM5) the 41.3% held as QSNOW is the lake-effect snow
itself, so a cloud figure built from QCLOUD+QICE draws a small, misleading part of the cloud.

Water paths integrate over the DRY-air layer mass (WRF mixing ratios are per kg of dry air):
dm = -(C1H * (MU + MUB) + C2H) * DNW / g per layer (WRF v4 hybrid coordinate; C1H=1, C2H=0
reduces it to the terrain-following form when the file predates the hybrid coordinate).
"""
from __future__ import annotations

import numpy as np

G = 9.81
SIGMA_SB = 5.670374419e-8            # Stefan-Boltzmann, W m-2 K-4
SPECIES = ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QHAIL")
LIQUID = ("QCLOUD",)
RAIN = ("QRAIN",)
CLOUD_ICE = ("QICE",)
PRECIP_ICE = ("QSNOW", "QGRAUP", "QHAIL")
# cloud boundary on total condensate: 0.01 g/kg shows cloud edges and anvils; 0.1 g/kg (the old
# isosurface) is the dense core only
CLOUD_EDGE_KGKG = 1.0e-5


def species_present(ds):
    """The SPECIES this wrfout actually holds, in SPECIES order."""
    return [s for s in SPECIES if s in ds.variables]


def _field(ds, name, timeidx):
    return np.asarray(ds.variables[name][timeidx], dtype=float)


def total_condensate(ds, timeidx=0, species=None):
    """Sum of the hydrometeor mixing ratios (kg/kg, mass grid, (nz, ny, nx)). species=None uses
    every SPECIES present. Raises KeyError when none is present."""
    names = species if species is not None else species_present(ds)
    if not names:
        raise KeyError("no hydrometeor mixing ratio (%s) in this wrfout" % ", ".join(SPECIES))
    out = None
    for s in names:
        q = np.clip(_field(ds, s, timeidx), 0.0, None)
        out = q if out is None else out + q
    return out


def layer_dry_mass(ds, timeidx=0):
    """Dry-air mass per unit area of each model layer, kg m-2, (nz, ny, nx)."""
    mut = _field(ds, "MU", timeidx) + _field(ds, "MUB", timeidx)
    dnw = np.asarray(ds.variables["DNW"][timeidx], dtype=float)
    if "C1H" in ds.variables and "C2H" in ds.variables:
        c1 = np.asarray(ds.variables["C1H"][timeidx], dtype=float)
        c2 = np.asarray(ds.variables["C2H"][timeidx], dtype=float)
    else:
        c1, c2 = np.ones_like(dnw), np.zeros_like(dnw)
    return -(c1[:, None, None] * mut[None] + c2[:, None, None]) * dnw[:, None, None] / G


def water_paths(ds, timeidx=0):
    """Column paths in g m-2 for the groups present: LWP (QCLOUD), RWP (QRAIN), IWP (QICE),
    SWP (QSNOW+QGRAUP+QHAIL), TWP (all). Groups whose species are absent are omitted."""
    dm = layer_dry_mass(ds, timeidx)
    present = set(species_present(ds))
    out = {}
    for key, group in (("LWP", LIQUID), ("RWP", RAIN), ("IWP", CLOUD_ICE), ("SWP", PRECIP_ICE)):
        names = [s for s in group if s in present]
        if names:
            q = sum(np.clip(_field(ds, s, timeidx), 0.0, None) for s in names)
            out[key] = (q * dm).sum(axis=0) * 1000.0
    if out:
        out["TWP"] = sum(out.values())
    return out


def height_msl(ds, timeidx=0):
    """Mass-level geometric height above MSL (m), (nz, ny, nx), from PH+PHB destaggered."""
    z = (_field(ds, "PH", timeidx) + _field(ds, "PHB", timeidx)) / G
    return 0.5 * (z[1:] + z[:-1])


def cloud_top_height(ds, timeidx=0, threshold=CLOUD_EDGE_KGKG):
    """Height (m MSL) of the highest mass level with total condensate >= threshold; NaN where
    the column has none."""
    q = total_condensate(ds, timeidx)
    z = height_msl(ds, timeidx)
    cloudy = q >= threshold
    nz = q.shape[0]
    # index of the highest cloudy level: last True along axis 0
    top = nz - 1 - np.argmax(cloudy[::-1], axis=0)
    has = cloudy.any(axis=0)
    out = np.take_along_axis(z, top[None], axis=0)[0]
    return np.where(has, out, np.nan)


def olr_brightness_temperature(ds, timeidx=0):
    """Equivalent blackbody temperature from WRF's OLR, (OLR / sigma) ** 0.25 in K -- a broadband
    proxy for an IR satellite image, NOT a radiative-transfer simulation of a satellite channel
    (it is warmer than a window-channel Tb over high cloud). Raises KeyError without OLR."""
    olr = np.clip(_field(ds, "OLR", timeidx), 1e-6, None)
    return (olr / SIGMA_SB) ** 0.25
