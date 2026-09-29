"""post/recipes/tc_extra.py -- the one tc figure decision 0057 section 2 asks for beyond what
`wtune post tc` (post/tc.py + post/met_tc.py, owned by another delegate session as of
2026-09-29 -- called here read-only, never edited) already produces: a 10 m tangential-wind
radius-time plot. Reads the track straight from post/tc.py's own ATCF a-deck output (no second
TempestExtremes run), and derives storm-relative tangential wind from U10/V10 with the same
planar bearing/haversine-radius approximation post/tc_metrics.py's quadrant_radii() already
uses for the model's own quadrant wind radii -- azimuthally averaged in radius bins here,
instead of maxed per quadrant.

No established tool in this repo's toolbox produces a model's own 10 m storm-relative
azimuthal-mean wind as a function of BOTH radius and lead time. MET tc_rmw is the closest
established tool, but only for a single snapshot's 3-D radius-height structure -- post/tc.py
already wraps it for exactly that (_rmw_figure); this is the 2-D, time-resolved companion, and
stays self-written for the same reason tc_metrics.py's quadrant_radii() does (see its own
docstring): no tool derives a model's OWN structure from raw U10/V10.
"""
from __future__ import annotations

import os
from datetime import timedelta

from post import atcf
from post import tc_metrics as tm

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
RADIUS_BINS_KM = list(range(0, 401, 25))


def tangential_wind(lat0, lon0, lat, lon, u, v):
    """Storm-relative tangential (cyclonic-positive in the NH) wind component, from the same
    planar bearing approximation tc_metrics.quadrant_index() uses (a local, not a rigorous
    great-circle, bearing -- consistent with the rest of this repo's TC wind-structure math)."""
    import numpy as np
    dx = (np.asarray(lon) - lon0) * np.cos(np.radians(lat0))
    dy = np.asarray(lat) - lat0
    phi = np.arctan2(dy, dx)
    return -np.asarray(u) * np.sin(phi) + np.asarray(v) * np.cos(phi)


def radius_time_plot(run_dir, adeck_path, out_path, domain=1, radius_bins_km=RADIUS_BINS_KM):
    """Azimuthally-averaged 10 m tangential wind (m/s) per radius bin per lead hour, as a
    pcolormesh. Returns out_path, or None if no a-deck point has a matching wrfout frame.

    An ATCF a-deck's own "time" field (post/atcf.py's `time`) is the constant model-init/
    synoptic time, NOT the point's valid time -- the lead is the separate `tau_h` (TAU) field
    (post/atcf.py's write_lines(); see the NHC ATCF Technical Documentation). Deduping/sorting
    by `time` instead of `tau_h` collapsed every point in the track into one (all sharing the
    same init time), which is why an earlier version of this plot showed a single, near-zero-
    width lead axis -- fixed here by keying everything on tau_h and reconstructing each
    point's own valid time as time + tau_h for the wrfout lookup."""
    import netCDF4
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(atcf.parse_file(adeck_path), key=lambda r: r["tau_h"])
    seen, points = set(), []
    for r in rows:
        if r["tau_h"] in seen:
            continue
        seen.add(r["tau_h"])
        points.append(r)
    if not points:
        return None

    n_bins = len(radius_bins_km) - 1
    grid = np.full((n_bins, len(points)), np.nan)
    leads = []
    any_frame = False
    for k, r in enumerate(points):
        valid_time = r["time"] + timedelta(hours=r["tau_h"])
        time_str = valid_time.strftime(WRF_TIME_FMT)
        leads.append(float(r["tau_h"]))
        path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
        if not os.path.exists(path):
            continue
        any_frame = True
        with netCDF4.Dataset(path) as ds:
            lat = np.asarray(ds.variables["XLAT"][0])
            lon = np.asarray(ds.variables["XLONG"][0])
            u10 = np.asarray(ds.variables["U10"][0])
            v10 = np.asarray(ds.variables["V10"][0])
        rr = tm.dist_km(r["lat"], r["lon"], lat, lon)
        vt = tangential_wind(r["lat"], r["lon"], lat, lon, u10, v10)
        for b in range(n_bins):
            mask = (rr >= radius_bins_km[b]) & (rr < radius_bins_km[b + 1])
            if mask.any():
                grid[b, k] = float(np.nanmean(vt[mask]))

    if not any_frame or np.all(np.isnan(grid)):
        return None
    radius_centers = [(radius_bins_km[b] + radius_bins_km[b + 1]) / 2.0 for b in range(n_bins)]
    fig, ax = plt.subplots(figsize=(7, 4))
    pc = ax.pcolormesh(leads, radius_centers, grid, cmap="RdBu_r", shading="nearest")
    fig.colorbar(pc, ax=ax, label="tangential wind (m/s, cyclonic +)")
    ax.set_xlabel("lead (h)")
    ax.set_ylabel("radius (km)")
    ax.set_title("10 m tangential wind, radius-time")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def tracking_notes_for(run_label, out_dir):
    """The tracking-grid hints post/tc.py's run_tc_report() wrote for this run (tracking_notes.json
    in out_dir): coarser-domain fallback, grid-limited track, storm left the nest. [] if none."""
    import json
    path = os.path.join(out_dir, "tracking_notes.json")
    try:
        with open(path) as fh:
            return list(json.load(fh).get(run_label, []))
    except (OSError, ValueError):
        return []


def figures(run_dir, run_label, adeck_path, domain, out_dir):
    from post.recipe import fig, missing
    # The hints themselves reach the report as recipe notes (post/recipe.py note(), emitted by
    # _tc_figures from run_tc_report's summary["notes"]); here they only annotate this figure.
    notes = tracking_notes_for(run_label, out_dir)
    path = os.path.join(out_dir, "tc_radius_time_wind_%s.png" % run_label)
    written = radius_time_plot(run_dir, adeck_path, path, domain=domain)
    out = []
    if written:
        out.append(fig("tc_radius_time_wind", written, run_label, domain="d%02d" % domain,
                       extra={"note": " ".join(notes)} if notes else None))
    else:
        out.append(missing("tc_radius_time_wind", "wrfout U10/V10 at the a-deck's own times",
                           "no wrfout frame matched any a-deck time for %s" % run_label))
    return out
