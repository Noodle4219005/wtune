"""post/recipes/common.py -- the "common" figure set (decision 0057 section 2), always
produced: a domain/nest configuration map with terrain (wrf-python `terrain` + cartopy, one
call per report even with several run dirs), and a reproducibility-noise envelope (diffwrf,
`wtune post compare`'s own engine) when two or more run dirs that differ only in decomposition
are given. Domain boundaries are drawn from each nest's own XLAT/XLONG grid edge (the actual
curved boundary on the parent's map projection, not a straight lon/lat rectangle) -- no
established tool draws a WRF nest-configuration figure, so this one is self-written glue over
wrf-python's terrain diagnostic and plain numpy edge extraction.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frames

# what one figure of a set may raise on a reduced history stream / an out-of-range point
# (cloud.py's _ERRORS = tool/import errors + these)
FIELD_ERRORS = (KeyError, ValueError, IndexError)


def error_reason(e):
    """missing() reason for one failed figure (same rule as cloud.py's _reason)."""
    from post.extract import is_missing_field_error, missing_reason
    return missing_reason(e) if is_missing_field_error(e) else "%s: %s" % (type(e).__name__, e)


def _bucket_mm(ds):
    """BUCKET_MM when the frame carries the I_RAINC/I_RAINNC bucket counters (namelist
    bucket_mm > 0), else 0.0 -- for the CDO route; netCDF4 readers use
    post.extract.accumulated_rain."""
    bucket = float(getattr(ds, "BUCKET_MM", -1.0))
    has = "I_RAINC" in ds.variables and "I_RAINNC" in ds.variables
    return bucket if bucket > 0 and has else 0.0


def _domain_edge_latlon(ds):
    import numpy as np
    lat = np.asarray(ds.variables["XLAT"][0])
    lon = np.asarray(ds.variables["XLONG"][0])
    top = list(zip(lon[0, :], lat[0, :]))
    right = list(zip(lon[:, -1], lat[:, -1]))
    bottom = list(zip(lon[-1, ::-1], lat[-1, ::-1]))
    left = list(zip(lon[::-1, 0], lat[::-1, 0]))
    ring = top + right + bottom + left
    return [p[0] for p in ring], [p[1] for p in ring]


def plot_domain_map(run_dir, max_dom, out_dir, run_label):
    """d01 terrain (wrf.getvar 'terrain') filled with cartopy coastlines, every nest's own
    grid-edge boundary overlaid. Returns the PNG path. Built directly (not through
    post/recipes/mapfig.py's one-shot save_map()) because the nest-boundary overlay needs to
    draw on the SAME still-open axes save_map would otherwise have already closed."""
    import netCDF4
    import numpy as np
    import cartopy.crs as ccrs
    wrf_mod = __import__("wrf")
    from post.plotting import _coastline, _matplotlib, _natural_earth_ready, _set_cartopy_data_dir

    d1_frames = list_frames(run_dir, domain=1)
    if not d1_frames:
        return None
    time0, path0 = d1_frames[0]
    plt = _matplotlib()
    out_path = os.path.join(out_dir, "common_domain_map.png")

    with netCDF4.Dataset(path0) as ds:
        terrain = wrf_mod.getvar(ds, "terrain", timeidx=0)
        lats, lons = wrf_mod.latlon_coords(terrain, as_np=True)
        crs = wrf_mod.get_cartopy(var=terrain)
        terrain_np = np.asarray(terrain)

        _set_cartopy_data_dir()
        natural_earth_ok = _natural_earth_ready()
        fig, ax = plt.subplots(figsize=(8, 6.4), subplot_kw={"projection": crs},
                               layout="constrained")
        cf = ax.contourf(lons, lats, terrain_np, levels=20, cmap="terrain",
                         transform=ccrs.PlateCarree())
        fig.colorbar(cf, ax=ax, label="terrain height (m)")
        _coastline(ax, ds, ccrs, natural_earth_ok, terrain=False, nest_rings=[])  # fill = terrain; rings drawn below

        for dom in range(1, max_dom + 1):
            frames = list_frames(run_dir, domain=dom)
            if not frames:
                continue
            with netCDF4.Dataset(frames[0][1]) as dds:
                edge_lon, edge_lat = _domain_edge_latlon(dds)
            ax.plot(edge_lon, edge_lat, "-", linewidth=1.5, color="red" if dom > 1 else "black",
                    transform=ccrs.PlateCarree(), label="d%02d" % dom)
        # legend below the axes, outside the data area (it used to sit on the terrain corner)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=max_dom,
                   fontsize=8, frameon=False)
        ax.set_title("%s domain/nest configuration (%d domain(s))  valid %s"
                     % (run_label, max_dom, time0), fontsize=10)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_repro_envelope(run_dirs, labels, out_dir):
    """Max per-frame diffwrf RMSE across every field, for each (run_dirs[0], run_dirs[i])
    pair, plotted vs. time -- the "reproducibility-noise envelope" (decision 0057 section 2),
    drawn straight from post/compare.py's own diffwrf-backed comparator, no re-derivation.
    Returns the PNG path, or None if fewer than 2 run dirs were given."""
    if len(run_dirs) < 2:
        return None
    from post.compare import compare_runs
    plt = _plt()

    fig, ax = plt.subplots(figsize=(7, 4))
    any_data = False
    for i in range(1, len(run_dirs)):
        result = compare_runs(run_dirs[0], run_dirs[i])
        by_time = {}
        for r in result["results"]:
            by_time.setdefault(r["time"], []).append(r["rmse"])
        if not by_time:
            continue
        times = sorted(by_time)
        envelope = [max(by_time[t]) for t in times]
        ax.plot(range(len(times)), envelope, "-o", markersize=3,
               label="%s vs %s" % (labels[0], labels[i]))
        any_data = True
    if not any_data:
        plt.close(fig)
        return None
    ax.set_xlabel("frame index")
    ax.set_ylabel("max field RMSE (diffwrf)")
    ax.set_yscale("log")
    ax.set_title("reproducibility-noise envelope")
    ax.legend(fontsize=7)
    path = os.path.join(out_dir, "common_repro_envelope.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def figures(run_dirs, labels, out_dir, case_info, decomposition_only=False):
    """Returns the list of common-type figure records (post.recipe's fig()/missing() shape).
    decomposition_only: True when the caller asserts the extra run dirs differ only in
    decomposition (e.g. --repro-pairs) -- the envelope is only meaningful under that
    condition, so it is otherwise skipped rather than drawn on runs that also differ
    physically."""
    from post.recipe import fig, missing

    out = []
    run_label = labels[0]
    try:
        _, domains = _domain_physics(run_dirs[0])
        max_dom = len(domains)
        path = plot_domain_map(run_dirs[0], max_dom, out_dir, run_label)
        if path:
            out.append(fig("common_domain_map", path, run_label, domain="all",
                          valid_time=list_frames(run_dirs[0], domain=1)[0][0]))
        else:
            out.append(missing("common_domain_map", "d01 wrfout", "no wrfout_d01_* frames found"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("common_domain_map", "wrf-python/cartopy", str(e)))

    if decomposition_only and len(run_dirs) >= 2:
        from post.compare import CompareError       # diffwrf failed: compare_runs fails closed
        try:
            path = plot_repro_envelope(run_dirs, labels, out_dir)
            if path:
                out.append(fig("common_repro_envelope", path, ",".join(labels)))
            else:
                out.append(missing("common_repro_envelope", "diffwrf",
                                   "no common (domain,time) frames differed -- runs identical "
                                   "or no overlapping frames"))
        except (tools.ToolNotFoundError, ImportError, CompareError) as e:
            out.append(missing("common_repro_envelope", "diffwrf", str(e)))
    return out


def _domain_physics(run_dir):
    from post import caseinfo
    return caseinfo.domain_physics(run_dir)
