"""post/obs_figs.py -- matplotlib/cartopy drawing shared by the model-vs-observation figures of
post/recipes/{precip,convection}.py: same-scale side-by-side maps, a difference map, score-vs-
threshold panels, FSS-vs-scale, station bias map and bias/RMSE time series. Drawing only: every
number plotted comes from MET output (post/met_gridstat.py, post/met_pointstat.py).
"""
from __future__ import annotations

import os

from post.plotting import _matplotlib, _natural_earth_ready, _set_cartopy_data_dir

PRECIP_LEVELS = [1, 2.5, 5, 10, 15, 25, 35, 50, 75, 100, 150, 200, 300]
DBZ_LEVELS = list(range(5, 75, 5))


def _cartopy_axes(fig, spec, extent):
    import cartopy.crs as ccrs
    _set_cartopy_data_dir()
    ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if _natural_earth_ready():
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    return ax, ccrs


def _extent(lat, lon, pad=0.3):
    import numpy as np
    ok = np.isfinite(lat) & np.isfinite(lon)
    return [float(lon[ok].min()) - pad, float(lon[ok].max()) + pad,
            float(lat[ok].min()) - pad, float(lat[ok].max()) + pad]


def side_by_side(lat, lon, left, right, titles, suptitle, out_path, levels, cmap="turbo",
                 cbar_label="", extend="max", under_color="white"):
    """Two maps of two fields on the same grid with ONE fixed discrete colour scale (`levels`) and
    one shared colour bar. Cells where either field is NaN are blanked in both, so the panels show
    exactly the matched points the scores were computed on."""
    import numpy as np
    from matplotlib.colors import BoundaryNorm
    plt = _matplotlib()
    both = np.isfinite(left) & np.isfinite(right)
    a, b = np.where(both, left, np.nan), np.where(both, right, np.nan)
    cm = plt.get_cmap(cmap).copy()
    cm.set_under(under_color)
    cm.set_over(cm(1.0))
    norm = BoundaryNorm(levels, cm.N, extend=extend)
    fig = plt.figure(figsize=(12.5, 5.6), layout="constrained")
    gs = fig.add_gridspec(1, 2)
    extent = _extent(np.where(both, lat, np.nan), np.where(both, lon, np.nan))
    mesh = None
    for i, (arr, title) in enumerate(((a, titles[0]), (b, titles[1]))):
        ax, ccrs = _cartopy_axes(fig, gs[0, i], extent)
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(arr), cmap=cm, norm=norm,
                             transform=ccrs.PlateCarree(), shading="auto")
        ax.set_title(title, fontsize=10)
    fig.colorbar(mesh, ax=fig.axes, shrink=0.85, label=cbar_label, extend=extend)
    fig.suptitle(suptitle, fontsize=10)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def diff_map(lat, lon, fcst, obs, title, out_path, cbar_label, vmax):
    """model - observation on matched points, symmetric colour scale +-vmax."""
    import numpy as np
    plt = _matplotlib()
    both = np.isfinite(fcst) & np.isfinite(obs)
    d = np.where(both, fcst - obs, np.nan)
    fig = plt.figure(figsize=(7.5, 5.8), layout="constrained")
    ax, ccrs = _cartopy_axes(fig, fig.add_gridspec(1, 1)[0, 0],
                             _extent(np.where(both, lat, np.nan), np.where(both, lon, np.nan)))
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(d), cmap="RdBu", vmin=-vmax, vmax=vmax,
                         transform=ccrs.PlateCarree(), shading="auto")
    fig.colorbar(mesh, ax=ax, shrink=0.85, label=cbar_label, extend="both")
    ax.set_title(title, fontsize=9)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def scores_vs_threshold(rows, title, out_path, thresh_label="threshold"):
    """rows: [{thresh, pod, far, csi, fbias, ets?}] -> 4 small panels (POD, FAR, CSI, frequency
    bias) against the event threshold; the FBIAS panel has the perfect-score line at 1."""
    plt = _matplotlib()
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), layout="constrained")
    x = [r["thresh"] for r in rows]
    for ax, key, name in zip(axs, ("pod", "far", "csi", "fbias"),
                             ("POD (PODY)", "FAR", "CSI", "frequency bias (FBIAS)")):
        ax.plot(x, [r[key] for r in rows], "-o")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(thresh_label)
        if min(x) > 0 and max(x) / min(x) > 20:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
        if key == "fbias":
            ax.axhline(1.0, color="k", lw=0.7, ls="--")
        else:
            ax.set_ylim(0, 1)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def fss_vs_scale(nbr_rows, cell_km, title, out_path, thresh_unit=""):
    """nbr_rows: parse_nbrcnt_table() output. One line per event threshold: FSS against
    neighbourhood width in km (width_points x cell_km). Dashed = 'useful skill' FSS = 0.5 + f0/2
    with f0 the observed event frequency (Roberts & Lean 2008), same colour as its line."""
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    threshes = sorted({r["thresh"] for r in nbr_rows})
    for i, th in enumerate(threshes):
        rs = [r for r in nbr_rows if r["thresh"] == th]
        c = "C%d" % (i % 10)
        ax.plot([r["width"] * cell_km for r in rs], [r["fss"] for r in rs], "-o", color=c,
                label="%g%s" % (th, thresh_unit))
        f0 = rs[0]["o_rate"]
        if f0 == f0:
            ax.axhline(0.5 + f0 / 2.0, color=c, ls=":", lw=0.9)
    ax.set_xlabel("neighbourhood width (km)")
    ax.set_ylabel("FSS")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(title="event threshold", fontsize=8, ncol=2)
    ax.set_title(title, fontsize=9)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def station_bias_maps(panels, suptitle, out_path):
    """One row of station maps. panels: [{lat, lon, bias, title, vmax}]; stations coloured by the
    mean model-minus-observation bias (diverging scale +-vmax, one per panel)."""
    import numpy as np
    plt = _matplotlib()
    n = len(panels)
    fig = plt.figure(figsize=(4.8 * n, 6.4), layout="constrained")
    gs = fig.add_gridspec(1, n)
    allx = np.concatenate([np.asarray(p["lon"]) for p in panels])
    ally = np.concatenate([np.asarray(p["lat"]) for p in panels])
    ext = _extent(ally, allx, pad=2.0)
    for i, p in enumerate(panels):
        ax, ccrs = _cartopy_axes(fig, gs[0, i], ext)
        sc = ax.scatter(p["lon"], p["lat"], c=p["bias"], s=42, cmap="RdBu_r", vmin=-p["vmax"],
                        vmax=p["vmax"], edgecolor="k", linewidth=0.4,
                        transform=ccrs.PlateCarree(), zorder=5)
        fig.colorbar(sc, ax=ax, shrink=0.8, extend="both")
        ax.set_title(p["title"], fontsize=9)
    fig.suptitle(suptitle, fontsize=10)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def bias_rmse_series(times_by_label, series, title, out_path):
    """series: {label: (unit, bias_list, rmse_list, n_list)}, times_by_label: {label: [datetime]}
    -> one row per variable: bias (model - obs, with zero line) and RMSE against valid time."""
    plt = _matplotlib()
    n = len(series)
    fig, axs = plt.subplots(n, 1, figsize=(9, 2.6 * n), sharex=True, layout="constrained",
                            squeeze=False)
    for ax, (label, (unit, bias, rmse, cnt)) in zip(axs[:, 0], series.items()):
        t = times_by_label[label]
        ax.plot(t, bias, "-o", ms=3, label="bias (model - obs)")
        ax.plot(t, rmse, "-s", ms=3, label="RMSE")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("%s (%s)" % (label, unit), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axs[-1, 0].set_xlabel("valid time (UTC)")
    for lab in axs[-1, 0].get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
