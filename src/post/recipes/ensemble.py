"""post/recipes/ensemble.py -- the "ensemble" figure set (decision 0057 section 2): a spaghetti
plot (one contour per member, wrf-python getvar + matplotlib), a probability-above-threshold
map (plain numpy fraction-of-members-exceeding, over the same wrf-python field each member
already computed), and MET Ensemble-Stat's rank histogram + spread-skill (skipped with a stated
reason when `ensemble_stat` is not installed -- same tools.find_postproc_tool contract as
post/met_tc.py's tc_pairs/tc_stat/tc_rmw).

Field/level/threshold default to SLP <= 1000 hPa contour (spaghetti) and precipitation
>= 25 mm (probability) -- reasonable, commonly-shown defaults for a tracked-system or
heavy-rain ensemble; a caller with a different variable of interest should draw it with
`wtune post plot`/`extract` per member instead, this figure set is the generic default view.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frames


def _member_field(run_dir, domain, time_str, field="slp"):
    """(values, lats, lons, crs, ds_landmask) of one member; the lateral boundary zone
    (spec_bdy_width cells, where the solution is the driving analysis rather than the member's
    own physics) is set to NaN so no boundary-band artefact enters the ensemble statistics."""
    import netCDF4
    import numpy as np
    from post import caseinfo
    wrf_mod = __import__("wrf")
    path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
    with netCDF4.Dataset(path) as ds:
        if field == "slp":
            var = wrf_mod.getvar(ds, "slp", timeidx=0, units="hPa")
        elif field == "rain_acc":
            var = (wrf_mod.getvar(ds, "RAINC", timeidx=0) + wrf_mod.getvar(ds, "RAINNC", timeidx=0))
        else:
            raise ValueError(field)
        lats, lons = wrf_mod.latlon_coords(var, as_np=True)
        crs = wrf_mod.get_cartopy(var=var)
        arr = np.array(var, dtype=float)
        w = caseinfo.spec_bdy_width(run_dir)
        arr[:w, :] = np.nan
        arr[-w:, :] = np.nan
        arr[:, :w] = np.nan
        arr[:, -w:] = np.nan
        landmask = np.asarray(ds.variables["LANDMASK"][0]) if "LANDMASK" in ds.variables else None
        return arr, lats, lons, crs, landmask


def _draw_coast(ax, ccrs, lats, lons, landmask):
    """Shared base map (post/basemap.py), LANDMASK contour when Natural Earth is not cached."""
    from post.basemap import add_base_map
    from post.plotting import _set_cartopy_data_dir
    _set_cartopy_data_dir()
    # labels=False: the figures draw their own labelled gridlines (no doubled tick labels)
    drawn = add_base_map(ax, ccrs, lats=lats, lons=lons, terrain=False, nest_rings=[],
                         caption=False, labels=False)
    if not drawn["layers"] and landmask is not None:
        ax.contour(lons, lats, landmask, levels=[0.5], colors="k", linewidths=0.7,
                   transform=ccrs.PlateCarree())


def plot_spaghetti(run_dirs, labels, domain, time_str, out_path, contour_level=None):
    """One SLP contour per member plus a marker at each member's SLP minimum. contour_level
    None = 10 hPa above the HIGHEST member minimum (every member then has a closed contour; a level tied to the deepest low drops the weaker members): a fixed 1000 hPa contour is the whole outer
    circulation of a hurricane (5-degree-wide rings that hide the member differences), while a
    level tied to the deepest low outlines the core the members actually disagree on."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    members = [_member_field(rd, domain, time_str, field="slp") for rd in run_dirs]
    mins = [np.nanmin(m[0]) for m in members]
    level = contour_level if contour_level is not None else float(np.floor(max(mins) + 10.0))
    crs, lats0, lons0, lm0 = members[0][3], members[0][1], members[0][2], members[0][4]
    fig, ax = plt.subplots(figsize=(8, 6.5), subplot_kw={"projection": crs})
    for idx, ((arr, lats, lons, _, _), label) in enumerate(zip(members, labels)):
        color = plt.get_cmap("tab10")(idx % 10)
        ax.contour(lons, lats, arr, levels=[level], colors=[color], linewidths=1.4,
                   transform=ccrs.PlateCarree())
        j, i = np.unravel_index(np.nanargmin(arr), arr.shape)
        ax.plot(lons[j, i], lats[j, i], "*", color=color, ms=13, mec="k",
                transform=ccrs.PlateCarree(), label="%s (min %.0f hPa)" % (label, arr[j, i]))
    _draw_coast(ax, ccrs, lats0, lons0, lm0)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    ax.set_title("ensemble spaghetti: SLP=%g hPa contour, stars = member SLP minimum\n%s (%d members)"
                 % (level, time_str, len(run_dirs)), fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_probability(run_dirs, domain, time_str, out_path, threshold_mm=25.0):
    """Fraction of members with accumulated rain >= threshold. With N members only the values
    k/N exist, so the fill uses N+1 discrete bins centred on k/N (colour bar ticks at those
    values) instead of a continuous 0-100 scale that would suggest precision that is not there."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs

    stack, lats, lons, crs, lm = [], None, None, None, None
    for run_dir in run_dirs:
        arr, lats, lons, crs, lm = _member_field(run_dir, domain, time_str, field="rain_acc")
        stack.append(arr)
    stack = np.array(stack)
    n = stack.shape[0]
    valid = np.isfinite(stack).all(axis=0)
    prob = np.where(valid, (stack >= threshold_mm).mean(axis=0) * 100.0, np.nan)
    edges = (np.arange(n + 2) - 0.5) / n * 100.0            # bin k covers k/N +- half a step
    cmap = mcolors.ListedColormap(plt.get_cmap("Blues")(np.linspace(0.05, 0.95, n + 1)))
    norm = mcolors.BoundaryNorm(edges, n + 1)
    fig, ax = plt.subplots(figsize=(8.5, 6.5), subplot_kw={"projection": crs})
    pm = ax.pcolormesh(lons, lats, np.ma.masked_invalid(prob), cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading="auto")
    cb = fig.colorbar(pm, ax=ax, ticks=np.arange(n + 1) / n * 100.0, shrink=0.85,
                      label="members exceeding (%%), N = %d" % n)
    cb.ax.set_yticklabels(["%g" % (k / n * 100.0) for k in range(n + 1)])
    _draw_coast(ax, ccrs, lats, lons, lm)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, x_inline=False, y_inline=False)
    gl.top_labels = gl.right_labels = False
    ax.set_title("fraction of %d members with rain accumulated since t0 >= %g mm, valid %s\n"
                 "(lateral boundary zone blanked)" % (n, threshold_mm, time_str), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def figures(run_dirs, labels, out_dir, case_info):
    from post.recipe import fig, missing
    if len(run_dirs) < 2:
        return [missing("ens_spaghetti", "ensemble members", "fewer than 2 run dirs given")]

    domain = case_info["domains"][-1]["domain"]
    common_times = None
    for run_dir in run_dirs:
        times = set(t for t, _ in list_frames(run_dir, domain=domain))
        common_times = times if common_times is None else (common_times & times)
    if not common_times:
        return [missing("ens_spaghetti", "wrfout_d%02d_*" % domain,
                        "no common valid time across all %d members" % len(run_dirs))]
    time_str = sorted(common_times)[-1]

    out = []
    try:
        path = plot_spaghetti(run_dirs, labels, domain, time_str,
                              os.path.join(out_dir, "ens_spaghetti.png"))
        out.append(fig("ens_spaghetti", path, ",".join(labels), domain="d%02d" % domain,
                       valid_time=time_str))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("ens_spaghetti", "wrf-python", str(e)))

    try:
        path = plot_probability(run_dirs, domain, time_str, os.path.join(out_dir, "ens_probability.png"))
        out.append(fig("ens_probability", path, ",".join(labels), domain="d%02d" % domain,
                       valid_time=time_str))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("ens_probability", "wrf-python", str(e)))

    try:
        tools.find_postproc_tool("ensemble_stat")
        out.append(missing("ens_rank_spread_skill", "MET ensemble_stat",
                           "ensemble_stat found but the obs-side wiring (rank histogram needs "
                           "an observation, spread-skill needs a verifying analysis) is not "
                           "implemented this round"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("ens_rank_spread_skill", "MET ensemble_stat", str(e)))
    return out
