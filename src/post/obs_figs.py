"""post/obs_figs.py -- matplotlib/cartopy drawing shared by the model-vs-observation figures of
post/recipes/{precip,convection}.py: same-scale side-by-side maps, a difference map, score-vs-
threshold panels, FSS-vs-scale, station bias map and bias/RMSE time series. Drawing only: every
number plotted comes from MET output (post/met_gridstat.py, post/met_pointstat.py).
"""
from __future__ import annotations

import os

from post.plotting import _matplotlib, _set_cartopy_data_dir

# WPC QPF isohyet thresholds (0.01, 0.10, 0.25, 0.50, 1, 1.5, 2, 3, 4, 6, 8 in) in mm
# (https://www.wpc.ncep.noaa.gov/html/fam2.shtml; decision 0064)
PRECIP_LEVELS_IN = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
PRECIP_LEVELS = [round(x * 25.4, 2) for x in PRECIP_LEVELS_IN]
PRECIP_CMAP = "YlGnBu"
DBZ_LEVELS = list(range(5, 75, 5))


def _cartopy_axes(fig, spec, extent):
    import cartopy.crs as ccrs
    _set_cartopy_data_dir()
    ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # shared base map (coastline, lakes, borders, states, labelled gridlines); these maps carry
    # no model grid, so no terrain contours
    from post.basemap import add_base_map
    add_base_map(ax, ccrs, terrain=False, nest_rings=[], caption=False)
    return ax, ccrs


def _extent(lat, lon, pad=0.3):
    import numpy as np
    ok = np.isfinite(lat) & np.isfinite(lon)
    return [float(lon[ok].min()) - pad, float(lon[ok].max()) + pad,
            float(lat[ok].min()) - pad, float(lat[ok].max()) + pad]


def side_by_side(lat, lon, left, right, titles, suptitle, out_path, levels, cmap=PRECIP_CMAP,
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
    cb = fig.colorbar(mesh, ax=fig.axes, shrink=0.85, label=cbar_label, extend=extend)
    if list(levels) == PRECIP_LEVELS:
        # WPC thresholds: label the bar in both mm and inches
        cb.set_ticks(PRECIP_LEVELS)
        cb.set_ticklabels(["%g mm (%g in)" % (mm, inch)
                           for mm, inch in zip(PRECIP_LEVELS, PRECIP_LEVELS_IN)], fontsize=7)
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


def performance_diagram_points(rows):
    """[(thresh, success_ratio, pod, csi, fbias)] for the rows with finite POD and FAR;
    success ratio SR = 1 - FAR (Roebber 2009)."""
    out = []
    for r in rows:
        pod, far = r.get("pod"), r.get("far")
        if pod is None or far is None or pod != pod or far != far:
            continue
        out.append((r["thresh"], 1.0 - far, pod, r.get("csi"), r.get("fbias")))
    return out


def scores_vs_threshold(rows, title, out_path, thresh_label="threshold"):
    """rows: [{thresh, pod, far, csi, fbias, total?}] -> one Roebber (2009, WAF 24:601)
    performance diagram: x = success ratio (1-FAR), y = POD, curved CSI isolines (grey), straight
    frequency-bias rays (dashed, labelled); one marker per event threshold (decision 0064). The
    perfect forecast is the top-right corner."""
    import numpy as np
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 6.0), layout="constrained")
    sr = np.linspace(0.001, 1.0, 400)
    pod = np.linspace(0.001, 1.0, 400)
    SR, POD = np.meshgrid(sr, pod)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    cs = ax.contour(SR, POD, CSI, levels=np.arange(0.1, 1.0, 0.1), colors="0.6", linewidths=0.7)
    ax.clabel(cs, fmt="%.1f", fontsize=7)
    for b in (0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0):
        x = np.linspace(0.0, 1.0, 50)
        ax.plot(x, b * x, ls="--", color="0.45" if b != 1.0 else "k", lw=0.7 if b != 1.0 else 1.0)
        tx, ty = (1.0, b) if b <= 1.0 else (1.0 / b, 1.0)
        ax.text(min(tx, 0.99), min(ty, 0.99), "%g" % b, fontsize=7, color="0.3",
                ha="left" if b <= 1.0 else "center", va="center" if b <= 1.0 else "bottom")
    pts = performance_diagram_points(rows)
    cmap = plt.get_cmap("viridis")
    for k, (th, s, p, c, fb) in enumerate(pts):
        n = next((r.get("total") for r in rows if r["thresh"] == th), None)
        ax.plot(s, p, "o", ms=9, color=cmap(k / max(1, len(pts) - 1)), mec="k",
                label="%g%s%s" % (th, " " + thresh_label if thresh_label else "",
                                  ("  (n=%d)" % n) if n else ""))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("success ratio (1 - FAR)")
    ax.set_ylabel("probability of detection (POD)")
    ax.set_aspect("equal")
    ax.legend(title="event threshold", fontsize=7, loc="lower right")
    ax.set_title(title + "\ngrey curves: CSI; dashed rays: frequency bias", fontsize=9)
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
    for ax, (label, spec) in zip(axs[:, 0], series.items()):
        unit, bias, rmse, cnt = spec[:4]
        ci = spec[4] if len(spec) > 4 else None
        t = times_by_label[label]
        ax.plot(t, bias, "-o", ms=3, color="C0", label="bias (model - obs)")
        ax.plot(t, rmse, "-s", ms=3, color="C1", label="RMSE")
        if ci:
            # 95 % bootstrap interval, stations resampled at each valid time (decision 0064)
            for key, color in (("bias", "C0"), ("rmse", "C1")):
                lo, hi = ci.get(key + "_lo"), ci.get(key + "_hi")
                if lo and hi and len(lo) == len(t):
                    ax.fill_between(t, lo, hi, color=color, alpha=0.18, lw=0,
                                    label="95% bootstrap CI" if key == "bias" else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("%s (%s)" % (label, unit), fontsize=9)
        ax.grid(alpha=0.3)
        if cnt and len(cnt) == len(t):
            ax2 = ax.twinx()
            ax2.bar(t, cnt, width=0.03, color="0.75", alpha=0.6, zorder=0)
            ax2.set_ylabel("n stations", fontsize=7, color="0.45")
            ax2.tick_params(axis="y", labelsize=6, colors="0.45")
            ax2.set_ylim(0, max(cnt) * 4)          # keep the count bars in the bottom quarter
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
        ax.legend(fontsize=7, loc="upper right")
    axs[-1, 0].set_xlabel("valid time (UTC)")
    for lab in axs[-1, 0].get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
