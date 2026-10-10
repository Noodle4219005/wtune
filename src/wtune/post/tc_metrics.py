"""post/tc_metrics.py -- pure-numpy TC quadrant wind-radii computation, the one piece of
post/tc.py's original wind.py port that survives decision 0056's 2026-09-29 update (wrap
established tools, delete the self-written substitute). Everything else that wind.py/
fine.py used to compute -- Vmax/RMW, azimuthal-mean profile, core/ring means, finest-
domain track-error/intensity selection -- is now either read straight from
TempestExtremes' own track (analysis/tc_track.py's chosen track already carries
pmin_hpa/vmax_ms per point) or produced by MET tc_pairs/tc_stat (track/intensity/wind-
radii error) and tc_rmw (radius-height structure); see post/tc.py and post/met_tc.py.

Why quadrant_radii() stays self-written: no established tool computes a model's OWN
34/50/64 kt quadrant wind radii from a WRF wind field -- MET tc_pairs/tc_stat compare
radii two decks already agree on reporting, they don't derive them from raw model output.
This is the input side of that comparison: it produces the RAD1-4 columns
post/atcf.py's a-deck writer needs, ported FAITHFULLY (same formulas, radii, thresholds)
from the pre-registered reference script (~/wtune-runs/p2-nest/wind.py; pre-registered in
knowledge/experiments/2026-09-28-sc26p2-katrina-4km-nest-refinement-and-wind-structure.md).

Metric definitions (verbatim from the reference script's own math):

  distance     haversine great-circle distance, R = EARTH_RADIUS_KM = 6371.0 km (NOT
               predictor/verify_obs.py's 6371.0088 -- the reference script used 6371.0,
               and this port keeps its number bit-for-bit, not the more precise IUGG mean
               radius used elsewhere in this repo).

  quadrant     bearing from the tracked centre, clockwise from true north:
               ang = degrees(atan2((lon-lon0)*cos(lat0), lat-lat0)) % 360
               quadrant = floor(ang / 90): 0=NE, 1=SE, 2=SW, 3=NW.

  quadrant     R34/R50/R64: per compass quadrant (NE/SE/SW/NW), the maximum distance (km)
  radii        within 600 km of the centre at which wind speed is >= the threshold; 0.0
               (not None) when no point in that quadrant meets the threshold. Thresholds
               are the National Hurricane Center 34/50/64 kt radii converted to m/s:
               THRESHOLDS_MS = {"r34": 17.49, "r50": 25.72, "r64": 32.92}. Computed on d01
               (post/tc.py always has it; the reference script's own "d01-based" metrics).

Units: distances in km, wind speed in m/s, angles in degrees. Plain numpy arrays/floats
in, no netCDF4/wrf-python/xarray dependency.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0
QUADRANT_RADIUS_MAX_KM = 600.0
QUADRANT_NAMES = ("ne", "se", "sw", "nw")       # index 0..3, matches quadrant_index()
THRESHOLDS_MS = {"r34": 17.49, "r50": 25.72, "r64": 32.92}


def dist_km(lat0, lon0, lat, lon):
    """Haversine great-circle distance (km), R=EARTH_RADIUS_KM. Scalars or arrays."""
    lat0 = np.asarray(lat0, dtype=float)
    lon0 = np.asarray(lon0, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    r = np.pi / 180.0
    a = (np.sin((lat - lat0) * r / 2) ** 2
         + np.cos(lat0 * r) * np.cos(lat * r) * np.sin((lon - lon0) * r / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def quadrant_index(lat0, lon0, lat, lon):
    """0=NE, 1=SE, 2=SW, 3=NW, bearing clockwise from true north at the centre."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ang = np.degrees(np.arctan2((lon - lon0) * np.cos(np.radians(lat0)), lat - lat0)) % 360
    return (ang // 90).astype(int)


def quadrant_radii(lat0, lon0, lat, lon, rr, speed, thresholds=THRESHOLDS_MS,
                    max_radius_km=QUADRANT_RADIUS_MAX_KM):
    """{'r34_ne_km': km, 'r34_se_km': km, ...} -- max radius per quadrant meeting each
    threshold, within max_radius_km of the centre; 0.0 (never None) when unmet."""
    q = quadrant_index(lat0, lon0, lat, lon)
    out = {}
    for name, th in thresholds.items():
        for qi, qn in enumerate(QUADRANT_NAMES):
            mm = (q == qi) & (rr <= max_radius_km) & (speed >= th)
            out["%s_%s_km" % (name, qn)] = round(float(rr[mm].max()), 1) if mm.any() else 0.0
    return out
