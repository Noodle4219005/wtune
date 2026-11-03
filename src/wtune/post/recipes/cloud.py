"""post/recipes/cloud.py -- the "cloud" figure set: 2-D views of cloud amount and cloud vertical
structure built from TOTAL condensate (post/condensate.py: every hydrometeor species the run
wrote, not QCLOUD+QICE -- that pair was 10% of the condensate mass in SC26-P2/Thompson and P1/WSM5
held 41% of it as snow). Produced for every run whose wrfout carries any hydrometeor (caseinfo
type "cloud"); P1 (cold-air outbreak, lake-effect snow) is the motivating case.

Domains: d01 plus the finest nest (post/recipes/cf3d.cf_domains, same rule as the CF data
product). Times: the last frame; the water-path figure (cheap: no wrf-python) also at the
--every/--times frames.

  cloud_water_path       B1  LWP/RWP/IWP/SWP/TWP panels (g m-2, log 1-2000) + 10 m barbs (kt)
  cloud_top_height       B2  highest level with total condensate >= 0.01 g/kg (km MSL) +
                             composite reflectivity 15/25 dBZ (taken from the lake-effect
                             study of Gaudet et al. 2021, OWLeS; not cited in the title, which
                             every case type gets)
  cloud_xsec             B3  two sections through the max-TWP column: along the mean boundary-
                             layer wind (the fetch) and perpendicular to it
  cloud_ir_proxy         B4  OLR-derived equivalent blackbody temperature, enhanced-IR palette
  cloud_condensate_3d    B5  ParaView volume (post.recipes.paraview3d.render_condensate_volume,
                             another module; a missing entry when it is unavailable)

Section geometry is done in the NATIVE grid (index space x dx): the line direction is the
grid-relative mean BL wind (wrf-python ua/va are grid-relative, i.e. aligned with the grid
axes), so "along the wind" is exact on the projected plane. Heights are interpolated with
wrf.interplevel / wrf.vertcross. The shared helpers here (plan-view axes, IR palette, height
interpolation, section overlays) are also used by post/recipes/tc_structure.py.
"""
from __future__ import annotations

import math
import os

from wtune.post import tools
from wtune.post.extract import list_frame_refs
from wtune.post.focus import frame_at
from wtune.post import figdpi

CONDENSATE_LEVELS_GKG = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]   # = 1e-5 .. 1e-3 kg/kg
CONDENSATE_CMAP = "bone_r"
WP_LEVELS_GM2 = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
WP_CMAP = "Blues"
WP_PANELS = ("LWP", "RWP", "IWP", "SWP", "TWP")
WP_NAMES = {"LWP": "LWP (QCLOUD)", "RWP": "RWP (QRAIN)", "IWP": "IWP (QICE)",
            "SWP": "SWP (QSNOW+QGRAUP+QHAIL)", "TWP": "TWP (all species)"}
W_UP_LEVELS = [0.25, 0.5, 1.0]          # m/s, solid red
W_DOWN_LEVELS = [-1.0, -0.5, -0.25]     # m/s, dashed blue
ECHO_DBZ = (15.0, 25.0)                 # lake-effect echo thresholds (Gaudet et al. 2021, OWLeS)
XSEC_HALF_KM = 150.0
XSEC_MIN_TOP_KM = 4.0
XSEC_MAX_TOP_KM = 18.0
CALM_MS = 0.5
IR_TMIN_K, IR_TMAX_K, IR_ENHANCE_K = 180.0, 310.0, 240.0
IR_TITLE = ("OLR-derived equivalent blackbody temperature "
            "(proxy, not a satellite-channel simulation)")


# ---------------------------------------------------------------- shared helpers

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _wrf():
    try:
        return __import__("wrf")
    except ImportError as e:
        raise tools.ToolNotFoundError("wrf-python not importable: %s" % e)


def wrap_title(text, width=95):
    import textwrap
    return "\n".join(w for line in str(text).split("\n") for w in (textwrap.wrap(line, width) or [""]))


def dx_km_of(ds):
    return float(ds.getncattr("DX")) / 1000.0


def bdy_width_for(run_dir):
    from wtune.post import caseinfo
    try:
        return caseinfo.spec_bdy_width(run_dir)
    except Exception:                       # noqa: BLE001 -- no namelist: WRF's default
        return caseinfo.DEFAULT_SPEC_BDY_WIDTH


def grid_limit_note(dx_km):
    """Caption fragment for the TC structure figures on a coarse grid (None at dx <= 4 km)."""
    if dx_km > 4.0:
        return "dx = %g km: eye/eyewall are under-resolved" % dx_km
    return None


def has_hydrometeors(path):
    """(bool, [species]) -- whether this wrfout frame holds any condensate species."""
    import netCDF4
    from wtune.post import condensate
    with netCDF4.Dataset(path) as ds:
        sp = condensate.species_present(ds)
    return bool(sp), sp


def plan_figure(crs, nrows=1, ncols=1, figsize=(8, 6.4)):
    """(fig, [axes]) on the WRF-native cartopy projection, constrained layout."""
    plt = _plt()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": crs},
                             layout="constrained", squeeze=False)
    return fig, list(axes.ravel())


def finish_plan(ax, ds, lats, lons, crs, caption=True, extent=None, outer_labels=None):
    """Shared base map (post/basemap.add_base_map) and an explicit extent (default: the lat/lon
    points given) -- wrf.get_cartopy() on a nest can otherwise auto-scale to the globe (see
    terrain.py). outer_labels=(left, bottom) for a panel of a multi-panel figure: gridline
    labels only outside the map (latitude on the left column, longitude on the bottom row,
    never inline, never top/right); None keeps the base map's own single-map labels."""
    import cartopy.crs as ccrs
    from wtune.post.basemap import add_base_map
    from wtune.post.plotting import _set_cartopy_data_dir
    _set_cartopy_data_dir()
    ax.set_extent(extent or projected_extent(crs, lats, lons), crs=crs)
    info = add_base_map(ax, ccrs, ds=ds, lats=lats, lons=lons, caption=caption,
                        labels=outer_labels is None)
    if outer_labels is not None:
        info["gridliner"] = outer_gridlines(ax, *outer_labels)
    return info


def outer_gridlines(ax, left, bottom):
    """Outer-only lat/lon gridline labels; see post/basemap.outer_gridlines."""
    from wtune.post.basemap import outer_gridlines as _outer
    return _outer(ax, left=left, bottom=bottom)


def projected_extent(crs, lats, lons):
    """[x0, x1, y0, y1] of lat/lon points in the native projection (for set_extent)."""
    import numpy as np
    import cartopy.crs as ccrs
    pts = crs.transform_points(ccrs.PlateCarree(), np.asarray(lons, "f8").ravel(),
                               np.asarray(lats, "f8").ravel())
    return [float(np.nanmin(pts[:, 0])), float(np.nanmax(pts[:, 0])),
            float(np.nanmin(pts[:, 1])), float(np.nanmax(pts[:, 1]))]


def native_xy(crs, lats, lons):
    """(X, Y) of the lat/lon grid in the native projection's own coordinates. Fills are drawn
    with transform=crs on these: a contourf polygon that spans the whole domain with holes,
    reprojected from PlateCarree, was drawn inside-out (holes filled, body empty) on a Lambert
    grid; in native coordinates there is nothing to reproject."""
    import numpy as np
    import cartopy.crs as ccrs
    lats = np.asarray(lats, "f8")
    pts = crs.transform_points(ccrs.PlateCarree(), np.asarray(lons, "f8"), lats)
    return pts[..., 0], pts[..., 1]


def ir_colormap():
    """(cmap, norm, levels): reverse greyscale (warm = dark, cold = white) above IR_ENHANCE_K and
    a colour enhancement for colder tops below it (NOAA-style "enhanced IR": cyan just below
    240 K through yellow to dark red at the coldest tops), 1 K bins over 180-310 K."""
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.colors import BoundaryNorm, ListedColormap
    levels = np.arange(IR_TMIN_K, IR_TMAX_K + 1.0, 1.0)
    mids = 0.5 * (levels[1:] + levels[:-1])
    turbo = colormaps["turbo"]
    colors = []
    for t in mids:
        if t >= IR_ENHANCE_K:
            g = 0.97 - 0.87 * (t - IR_ENHANCE_K) / (IR_TMAX_K - IR_ENHANCE_K)
            colors.append((g, g, g, 1.0))
        else:
            frac = (IR_ENHANCE_K - t) / (IR_ENHANCE_K - IR_TMIN_K)
            colors.append(turbo(0.25 + 0.75 * frac))
    cmap = ListedColormap(colors, name="enhanced_ir").with_extremes(under=colors[0],
                                                                    over=colors[-1])
    return cmap, BoundaryNorm(levels, cmap.N), levels


def plot_ir_proxy(ds, time_str, out_path, run_label, domain, timeidx=0):
    """B4 / A2: whole-domain OLR-derived equivalent blackbody temperature (condensate.
    olr_brightness_temperature), enhanced-IR palette, colour bar in K with a degC axis."""
    import numpy as np
    from wtune.post import condensate
    from wtune.post.conventions import ds_time_label
    wrf_mod = _wrf()
    tb = condensate.olr_brightness_temperature(ds, timeidx)
    lats = np.asarray(ds.variables["XLAT"][timeidx], "f8")
    lons = np.asarray(ds.variables["XLONG"][timeidx], "f8")
    crs = wrf_mod.get_cartopy(wrfin=ds)
    cmap, norm, levels = ir_colormap()
    fig, (ax,) = plan_figure(crs)
    X, Y = native_xy(crs, lats, lons)
    pm = ax.pcolormesh(X, Y, tb, cmap=cmap, norm=norm, shading="nearest", transform=crs)
    cbar = fig.colorbar(pm, ax=ax, shrink=0.9, extend="both",
                        ticks=np.arange(IR_TMIN_K, IR_TMAX_K + 1, 20))
    cbar.set_label("equivalent blackbody temperature (K)")
    sec = cbar.ax.secondary_yaxis("left", functions=(lambda k: k - 273.15, lambda c: c + 273.15))
    sec.set_ylabel("(°C)")
    finish_plan(ax, ds, lats, lons, crs=crs)
    ax.set_title(wrap_title("%s  d%02d  %s; colour below %g K (%g °C)\n%s"
                            % (run_label, domain, IR_TITLE, IR_ENHANCE_K,
                               IR_ENHANCE_K - 273.15, ds_time_label(ds, time_str))), fontsize=9)
    fig.savefig(out_path, dpi=figdpi.dpi())
    _plt().close(fig)
    return out_path


def interp_heights(field, z, levels_m, hgt=None):
    """wrf.interplevel of a (nz, ny, nx) mass-level field onto heights `levels_m` (same
    reference as z: MSL for 'z', AGL for 'height_agl'), NaN outside the model column. When
    hgt (same reference) is given, the gap between the ground and the lowest model level is
    filled with the lowest-level value (constant extrapolation down to the ground, never below
    it), so a 0 km level over the sea is not blank. Returns (nlev, ny, nx) float ndarray."""
    import numpy as np
    wrf_mod = _wrf()
    f = np.asarray(field, "f8")
    zz = np.asarray(z, "f8")
    lev = np.asarray(levels_m, "f8")
    out = np.asarray(wrf_mod.interplevel(f, zz, lev, missing=np.nan, meta=False), "f8")
    if out.ndim == 2:
        out = out[None]
    out = np.where(np.abs(out) > 1e20, np.nan, out)
    if hgt is not None:
        h = np.asarray(hgt, "f8")
        gap = (np.isnan(out) & (lev[:, None, None] >= h[None] - 1e-6)
               & (lev[:, None, None] < zz[0][None]))
        out = np.where(gap, np.broadcast_to(f[0][None], out.shape), out)
    return out


def line_endpoints(ic, jc, ux, uy, half_cells, nx, ny):
    """Grid-space line through (ic, jc) along the unit vector (ux, uy), half_cells long on each
    side, each side shortened independently so its endpoint stays inside [0, nx-1] x [0, ny-1].
    Returns ((x0, y0), (x1, y1), (back_cells, fwd_cells)); x1 is the downwind/forward end."""
    def reach(sign):
        t = float(half_cells)
        for comp, c0, n in ((ux, ic, nx), (uy, jc, ny)):
            d = sign * comp
            if d > 1e-12:
                t = min(t, (n - 1 - c0) / d)
            elif d < -1e-12:
                t = min(t, (0 - c0) / d)
        return max(t, 0.0)
    fwd, back = reach(+1.0), reach(-1.0)
    return ((ic - back * ux, jc - back * uy), (ic + fwd * ux, jc + fwd * uy), (back, fwd))


def section(field, z, start, end, levels_m):
    """wrf.vertcross of a (nz, ny, nx) field between grid points start/end ((x, y) in grid
    index units) on heights levels_m; returns ((nlev, npts) array, (npts, 2) grid x/y)."""
    import numpy as np
    wrf_mod = _wrf()
    sp = wrf_mod.CoordPair(x=float(start[0]), y=float(start[1]))
    ep = wrf_mod.CoordPair(x=float(end[0]), y=float(end[1]))
    out = np.asarray(wrf_mod.vertcross(np.asarray(field, "f8"), np.asarray(z, "f8"),
                                       start_point=sp, end_point=ep,
                                       levels=np.asarray(levels_m, "f8"), missing=np.nan,
                                       meta=False), "f8")
    xy = np.asarray(wrf_mod.xy(np.asarray(field)[0], start_point=(float(start[0]), float(start[1])),
                               end_point=(float(end[0]), float(end[1])), meta=False), "f8")
    out = np.where(np.abs(out) > 1e20, np.nan, out)
    return out, xy


def line_values(field2d, xy):
    """wrf.interpline-equivalent sampling of a 2-D field at the section's own xy points
    (bilinear in grid index space)."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    a = np.asarray(field2d, "f8")
    return map_coordinates(a, [xy[:, 1], xy[:, 0]], order=1, mode="nearest")


def signed_distance_km(xy, ic, jc, ux, uy, dx_km):
    """Distance (km) of each grid-space point from (ic, jc), projected on (ux, uy)."""
    import numpy as np
    return ((np.asarray(xy)[:, 0] - ic) * ux + (np.asarray(xy)[:, 1] - jc) * uy) * dx_km


def _levels_label(levels):
    return "/".join("%g" % v for v in levels)


def w_contour_handles(w):
    """Legend handles for the w contours draw_w_contours draws on `w` (any shape; pass all
    panels of a multi-panel figure together): only the levels the field actually reaches."""
    import numpy as np
    from matplotlib.lines import Line2D
    w = np.asarray(w, "f8")
    if not np.isfinite(w).any():
        return []
    up = [v for v in W_UP_LEVELS if np.nanmax(w) >= v]
    down = [v for v in W_DOWN_LEVELS if np.nanmin(w) <= v]
    handles = []
    if up:
        handles.append(Line2D([], [], color="red", lw=0.8,
                              label="w up %s m s$^{-1}$" % _levels_label(up)))
    if down:
        handles.append(Line2D([], [], color="blue", lw=0.8, ls="--",
                              label="w down %s m s$^{-1}$" % _levels_label(sorted(down)[::-1])))
    return handles


def draw_w_contours(ax, x, y, w):
    """w contours: solid red up / dashed blue down at +-0.25, 0.5, 1 m/s. Returns legend
    handles for the levels reached only (w_contour_handles)."""
    import numpy as np
    w = np.asarray(w, "f8")
    if np.isfinite(w).any():
        if np.nanmax(w) >= W_UP_LEVELS[0]:
            ax.contour(x, y, w, levels=W_UP_LEVELS, colors="red", linewidths=0.8,
                       linestyles="solid")
        if np.nanmin(w) <= W_DOWN_LEVELS[-1]:
            ax.contour(x, y, w, levels=W_DOWN_LEVELS, colors="blue", linewidths=0.8,
                       linestyles="dashed")
    return w_contour_handles(w)


def isotherm_handles(temp_c, specs):
    """Legend handles for the isotherms of `specs` that `temp_c` (any shape) actually spans."""
    import numpy as np
    from matplotlib.lines import Line2D
    t = np.asarray(temp_c, "f8")
    if not np.isfinite(t).any():
        return []
    return [Line2D([], [], color="k", lw=lw, ls=ls, label=label)
            for val, ls, lw, label in specs if np.nanmin(t) <= val <= np.nanmax(t)]


def draw_isotherms(ax, x, y, temp_c, specs):
    """specs: [(degC, linestyle, linewidth, label)]. Returns legend handles of the isotherms
    actually drawn (isotherm_handles)."""
    import numpy as np
    t = np.asarray(temp_c, "f8")
    for val, ls, lw, label in specs:
        if np.isfinite(t).any() and np.nanmin(t) <= val <= np.nanmax(t):
            ax.contour(x, y, t, levels=[val], colors="k", linewidths=lw, linestyles=ls)
    return isotherm_handles(t, specs)


def flat_all(arrays):
    """All panels' section arrays as one flat array (legend handles for a multi-panel figure)."""
    import numpy as np
    return np.concatenate([np.asarray(a, "f8").ravel() for a in arrays]) if arrays else \
        np.array([np.nan])


def echo_thresholds_reached(mdbz):
    """ECHO_DBZ thresholds the composite reflectivity actually reaches (contours drawn)."""
    import numpy as np
    m = np.asarray(mdbz, "f8")
    if not np.isfinite(m).any():
        return []
    return [v for v in ECHO_DBZ if np.nanmax(m) >= v]


def figure_legend(fig, handles, **kw):
    """fig.legend only when there is something to list (no empty legend box)."""
    if handles:
        return fig.legend(handles=handles, **kw)
    return None


def draw_terrain(ax, x, ter_km):
    """Grey terrain silhouette (as post/recipes/moisture.py)."""
    import numpy as np
    ter = np.asarray(ter_km, "f8")
    if len(ter) == len(x) and np.nanmax(ter) > 0.0:
        ax.fill_between(x, 0, ter, color="0.45", zorder=5)


def condensate_fill(ax, x, y, q_gkg):
    """Log fill of total condensate, 0.01-1 g/kg (1e-5..1e-3 kg/kg)."""
    import numpy as np
    from matplotlib.colors import BoundaryNorm
    from matplotlib import colormaps
    q = np.asarray(q_gkg, "f8")
    cm = colormaps[CONDENSATE_CMAP].resampled(len(CONDENSATE_LEVELS_GKG))
    q = np.where(q < CONDENSATE_LEVELS_GKG[0], np.nan, q)
    return ax.contourf(x, y, q, levels=CONDENSATE_LEVELS_GKG, cmap=cm,
                       norm=BoundaryNorm(CONDENSATE_LEVELS_GKG, cm.N, extend="max"),
                       extend="max")


def condensate_colorbar(fig, cf, ax):
    cb = fig.colorbar(cf, ax=ax, ticks=CONDENSATE_LEVELS_GKG)
    cb.set_label("total condensate (g kg$^{-1}$, log scale)")
    return cb


# ---------------------------------------------------------------- B1 water paths

def water_path_panels(wp):
    """(panels, omitted): the WP_PANELS groups to draw, dropping a species group whose max over
    the plotted domain is below WP_LEVELS_GM2[0] (1 g m-2: an all-blank panel). TWP is always
    drawn and stays the sum of every group, omitted ones included."""
    import numpy as np
    panels, omitted = [], []
    for k in WP_PANELS:
        if k not in wp:
            continue
        a = np.asarray(wp[k], "f8")
        mx = float(np.nanmax(a)) if np.isfinite(a).any() else 0.0
        if k != "TWP" and mx < WP_LEVELS_GM2[0]:
            omitted.append(k)
        else:
            panels.append(k)
    return panels, omitted


def water_path_figure(path, time_str, out_path, run_label, domain, timeidx=0):
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm
    from matplotlib import colormaps
    from wtune.post import condensate
    from wtune.post.basemap import bdy_width_of, interior_mask, masked_barbs
    from wtune.post.conventions import KT_PER_MS, ds_time_label
    from wtune.post.plotting import WIND_SUBSAMPLE
    import netCDF4
    wrf_mod = _wrf()
    with netCDF4.Dataset(path) as ds:
        wp = condensate.water_paths(ds, timeidx)
        if not wp:
            raise KeyError("no hydrometeor mixing ratio in %s" % os.path.basename(path))
        panels, omitted = water_path_panels(wp)
        lats = np.asarray(ds.variables["XLAT"][timeidx], "f8")
        lons = np.asarray(ds.variables["XLONG"][timeidx], "f8")
        crs = wrf_mod.get_cartopy(wrfin=ds)
        u10, v10 = (np.asarray(a, "f8") for a in
                    wrf_mod.getvar(ds, "uvmet10", timeidx=timeidx, units="m s-1", meta=False))
        ncols = min(3, len(panels))
        nrows = int(math.ceil(len(panels) / float(ncols)))
        fig, axes = plan_figure(crs, nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows + 0.8))
        cm = colormaps[WP_CMAP].resampled(len(WP_LEVELS_GM2))
        norm = BoundaryNorm(WP_LEVELS_GM2, cm.N, extend="max")
        stride = max(WIND_SUBSAMPLE, min(lats.shape) // 12)
        mask = interior_mask(lats.shape, bdy_width_of(ds))
        cf = None
        X, Y = native_xy(crs, lats, lons)
        for k, (ax, key) in enumerate(zip(axes, panels)):
            val = np.where(wp[key] < WP_LEVELS_GM2[0], np.nan, wp[key])
            if np.isfinite(val).any():
                cf = ax.contourf(X, Y, val, levels=WP_LEVELS_GM2, cmap=cm, norm=norm,
                                 extend="max", transform=crs)
            masked_barbs(ax, lons, lats, u10 * KT_PER_MS, v10 * KT_PER_MS, stride, mask=mask,
                         length=4.5, linewidth=0.5, transform=ccrs.PlateCarree())
            info = finish_plan(ax, ds, lats, lons, crs=crs, caption=False,
                               outer_labels=(k % ncols == 0, k + ncols >= len(panels)))
            ax.set_title("%s  max %.0f g m$^{-2}$" % (WP_NAMES[key], float(np.nanmax(wp[key]))),
                         fontsize=8)
        for ax in axes[len(panels):]:
            ax.set_visible(False)
        # base-map caption as a figure x-label: constrained layout keeps it clear of the
        # bottom row's longitude labels (a free fig.text overlapped them)
        fig.supxlabel(info["caption"], fontsize=5.5, color="0.3", x=0.005, ha="left")
        if cf is None:
            from matplotlib.cm import ScalarMappable
            cf = ScalarMappable(norm=norm, cmap=cm)
        cb = fig.colorbar(cf, ax=axes[:len(panels)], shrink=0.85, ticks=WP_LEVELS_GM2)
        cb.set_label("water path (g m$^{-2}$, log scale 1-2000; < 1 unshaded)")
        omit_txt = ("\n%s < 1 g m$^{-2}$ everywhere (omitted; still in TWP)"
                    % "/".join(omitted)) if omitted else ""
        fig.suptitle(wrap_title("%s  d%02d  column water paths (dry-air-mass integral of each "
                                "species group) + 10 m wind barbs (kt)%s\n%s"
                                % (run_label, domain, omit_txt, ds_time_label(ds, time_str)),
                                120), fontsize=9)
        fig.savefig(out_path, dpi=figdpi.dpi())
        _plt().close(fig)
    return out_path


# ---------------------------------------------------------------- B2 cloud-top height

def cloud_top_figure(path, time_str, out_path, run_label, domain, timeidx=0):
    import numpy as np
    import netCDF4
    from matplotlib.lines import Line2D
    from wtune.post import condensate
    from wtune.post.conventions import ds_time_label
    from wtune.post.recipes.mapfig import nice_levels
    wrf_mod = _wrf()
    with netCDF4.Dataset(path) as ds:
        cth = condensate.cloud_top_height(ds, timeidx) / 1000.0
        mdbz = np.asarray(wrf_mod.getvar(ds, "mdbz", timeidx=timeidx, meta=False), "f8")
        lats = np.asarray(ds.variables["XLAT"][timeidx], "f8")
        lons = np.asarray(ds.variables["XLONG"][timeidx], "f8")
        crs = wrf_mod.get_cartopy(wrfin=ds)
        fig, (ax,) = plan_figure(crs)
        top = float(np.nanmax(cth)) if np.isfinite(cth).any() else 1.0
        levels = nice_levels(0.0, max(top, 1.0), nbins=12)
        X, Y = native_xy(crs, lats, lons)
        if np.isfinite(cth).any():
            cf = ax.contourf(X, Y, cth, levels=levels, cmap="viridis", extend="max",
                             transform=crs)
            fig.colorbar(cf, ax=ax, shrink=0.9, label="cloud-top height (km MSL)")
        handles = []
        reached = echo_thresholds_reached(mdbz)
        for val, color, lw in ((ECHO_DBZ[0], "magenta", 0.8), (ECHO_DBZ[1], "red", 1.3)):
            if val in reached:
                ax.contour(X, Y, mdbz, levels=[val], colors=color, linewidths=lw,
                           transform=crs)
                handles.append(Line2D([], [], color=color, lw=lw,
                                      label="composite reflectivity %g dBZ" % val))
        finish_plan(ax, ds, lats, lons, crs=crs)
        figure_legend(fig, handles, loc="outside lower center", ncol=2, fontsize=8,
                      frameon=False)
        absent = "" if reached else "; no composite reflectivity >= %g dBZ" % ECHO_DBZ[0]
        ax.set_title(wrap_title(
            "%s  d%02d  cloud-top height (highest level with total condensate >= %g g/kg; "
            "white = no cloud) + composite reflectivity %g/%g dBZ%s\n%s"
            % (run_label, domain, condensate.CLOUD_EDGE_KGKG * 1000.0, ECHO_DBZ[0], ECHO_DBZ[1],
               absent, ds_time_label(ds, time_str))), fontsize=9)
        fig.savefig(out_path, dpi=figdpi.dpi())
        _plt().close(fig)
    return out_path


# ---------------------------------------------------------------- B3 cross sections

def max_column(field2d, bdy_width, inside=None):
    """(j, i) of the field maximum over the interior (bdy_width edge cells excluded); within
    `inside` (bool array, post.focus) when the interior has a cell there."""
    import numpy as np
    a = np.array(field2d, "f8")
    ny, nx = a.shape
    bw = int(bdy_width) if 2 * int(bdy_width) < min(ny, nx) else 0
    if inside is not None and np.asarray(inside)[bw:ny - bw, bw:nx - bw].any():
        a = np.where(inside, a, -np.inf)
    sub = a[bw:ny - bw, bw:nx - bw]
    sub = np.where(np.isfinite(sub), sub, -np.inf)
    j, i = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return int(j + bw), int(i + bw)


def mean_bl_wind(u_col, v_col, zagl_col, pblh):
    """Mean grid-relative wind of the levels whose height AGL is <= PBLH (at least the lowest
    level). Returns (u, v)."""
    import numpy as np
    z = np.asarray(zagl_col, "f8")
    keep = z <= max(float(pblh), float(z[0]))
    return float(np.mean(np.asarray(u_col, "f8")[keep])), float(np.mean(np.asarray(v_col, "f8")[keep]))


def section_directions(u, v):
    """((ux, uy) along the wind, (px, py) perpendicular = wind rotated +90 deg, calm flag).
    A calm wind (< CALM_MS) falls back to the grid x axis."""
    sp = math.hypot(u, v)
    if sp < CALM_MS:
        return (1.0, 0.0), (0.0, 1.0), True
    ux, uy = u / sp, v / sp
    return (ux, uy), (-uy, ux), False


def cloud_xsec_figure(path, time_str, out_path, run_label, domain, bdy_width,
                      timeidx=0, focus_box=None):
    import numpy as np
    import netCDF4
    from matplotlib.lines import Line2D
    from wtune.post import condensate
    from wtune.post.conventions import ds_time_label
    from wtune.post.focus import box_mask
    wrf_mod = _wrf()
    with netCDF4.Dataset(path) as ds:
        dx = dx_km_of(ds)
        twp = condensate.water_paths(ds, timeidx)["TWP"]
        jc, ic = max_column(twp, bdy_width,
                            inside=box_mask(ds.variables["XLAT"][timeidx],
                                            ds.variables["XLONG"][timeidx], focus_box))
        q = condensate.total_condensate(ds, timeidx) * 1000.0
        z = np.asarray(wrf_mod.getvar(ds, "z", timeidx=timeidx, units="m", meta=False), "f8")
        zagl = np.asarray(wrf_mod.getvar(ds, "height_agl", timeidx=timeidx, units="m", meta=False), "f8")
        ua = np.asarray(wrf_mod.getvar(ds, "ua", timeidx=timeidx, meta=False), "f8")
        va = np.asarray(wrf_mod.getvar(ds, "va", timeidx=timeidx, meta=False), "f8")
        wa = np.asarray(wrf_mod.getvar(ds, "wa", timeidx=timeidx, meta=False), "f8")
        tc = np.asarray(wrf_mod.getvar(ds, "tc", timeidx=timeidx, meta=False), "f8")
        hgt = np.asarray(ds.variables["HGT"][timeidx], "f8")
        pblh = np.asarray(ds.variables["PBLH"][timeidx], "f8") if "PBLH" in ds.variables else None
        cth = condensate.cloud_top_height(ds, timeidx)
        lat = np.asarray(ds.variables["XLAT"][timeidx], "f8")
        lon = np.asarray(ds.variables["XLONG"][timeidx], "f8")
        tlabel = ds_time_label(ds, time_str)
    ny, nx = hgt.shape
    pb = float(pblh[jc, ic]) if pblh is not None else 1000.0
    u, v = mean_bl_wind(ua[:, jc, ic], va[:, jc, ic], zagl[:, jc, ic], pb)
    along, perp, calm = section_directions(u, v)
    half = XSEC_HALF_KM / dx
    lines = []
    for name, (ux, uy) in (("along mean BL wind (fetch)", along), ("perpendicular", perp)):
        start, end, _ = line_endpoints(ic, jc, ux, uy, half, nx, ny)
        lines.append((name, ux, uy, start, end))
    # height window: the cloud in these sections plus 1 km, 4..18 km
    tops = []
    for _, ux, uy, start, end in lines:
        xy = np.asarray(wrf_mod.xy(hgt, start_point=tuple(start), end_point=tuple(end),
                                   meta=False), "f8")
        tops.append(np.nanmax(line_values(np.nan_to_num(cth, nan=0.0), xy)))
    top_km = float(np.clip(math.ceil(max(tops) / 1000.0 + 1.0), XSEC_MIN_TOP_KM, XSEC_MAX_TOP_KM))
    levels = np.linspace(0.0, top_km * 1000.0, 61)

    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.6), layout="constrained")
    cf = None
    w_all, t_all = [], []
    iso = [(0.0, "solid", 1.8, "0 °C"), (-15.0, "dashed", 1.0, "-15 °C (dendritic growth)")]
    for ax, (name, ux, uy, start, end) in zip(axes, lines):
        qs, xy = section(q, z, start, end, levels)
        ws, _ = section(wa, z, start, end, levels)
        ts, _ = section(tc, z, start, end, levels)
        x = signed_distance_km(xy, ic, jc, ux, uy, dx)
        y = levels / 1000.0
        cf = condensate_fill(ax, x, y, qs)
        draw_w_contours(ax, x, y, ws)
        draw_isotherms(ax, x, y, ts, iso)
        w_all.append(ws)
        t_all.append(ts)
        ter = line_values(hgt, xy) / 1000.0
        if pblh is not None:
            ax.plot(x, ter + line_values(pblh, xy) / 1000.0, color="darkorange", lw=1.4)
        draw_terrain(ax, x, ter)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, top_km)
        ax.set_ylabel("height MSL (km)")
        ax.set_xlabel("distance from max-TWP column (km, + = %s)"
                      % ("downwind" if name.startswith("along") else "left of the wind"))
        ends = np.array([start, end], "f8")
        la, lo = line_values(lat, ends), line_values(lon, ends)
        a0, a1 = (la[0], lo[0]), (la[1], lo[1])
        ax.set_title("%s  (%.2f, %.2f) -> (%.2f, %.2f)" % (name, a0[0], a0[1], a1[0], a1[1]),
                     fontsize=8)
    condensate_colorbar(fig, cf, list(axes))
    # legend from BOTH panels, listing only the contour levels actually drawn
    handles = w_contour_handles(flat_all(w_all)) + isotherm_handles(flat_all(t_all), iso)
    if pblh is not None:
        handles.append(Line2D([], [], color="darkorange", lw=1.4, label="PBLH"))
    figure_legend(fig, handles, loc="outside lower center", ncol=max(len(handles), 1),
                  fontsize=7, frameon=False)
    wind_txt = ("calm BL wind: grid W-E / S-N" if calm else
                "mean BL wind (below PBLH %.0f m) %.1f m s$^{-1}$ grid-relative" % (pb, math.hypot(u, v)))
    fig.suptitle(wrap_title(
        "%s  d%02d  sections through the max-TWP column (%.2f, %.2f): total condensate "
        "(all species, log 0.01-1 g/kg) + w + 0/-15 °C + PBLH; %s; +-%g km (clipped to "
        "domain)\n%s" % (run_label, domain, lat[jc, ic], lon[jc, ic], wind_txt, XSEC_HALF_KM,
                         tlabel), 120), fontsize=9)
    fig.savefig(out_path, dpi=figdpi.dpi())
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- B5 3-D (other module)

def condensate_volume(path, out_png, work_dir, title, center_latlon=None, box_km=None,
                      cutaway=False, timeidx=0):
    """Calls post.recipes.paraview3d.render_condensate_volume (owned by another module).
    Raises ImportError/NotImplementedError/tools.ToolNotFoundError when unavailable (and that
    module's own ParaViewError/ValueError); callers turn any exception into a missing entry."""
    from wtune.post.recipes.paraview3d import render_condensate_volume
    return render_condensate_volume(path, out_png, work_dir, center_latlon=center_latlon,
                                    box_km=box_km, cutaway=cutaway, title=title, timeidx=timeidx)


# ---------------------------------------------------------------- figure set

def _cloud_domains(case_info):
    from wtune.post.recipes.cf3d import cf_domains
    return cf_domains(case_info)


def _select_times(all_times, every, times):
    if not (every or times):
        return all_times[-1:]
    from wtune.post.timeseries import select_times
    sel = select_times(all_times, every=every, times=times)
    return sel or all_times[-1:]


_ERRORS = (tools.ToolNotFoundError, ImportError, KeyError, ValueError, IndexError)


def _reason(e):
    from wtune.post.extract import is_missing_field_error, missing_reason
    return missing_reason(e) if is_missing_field_error(e) else "%s: %s" % (type(e).__name__, e)


IR_THRESHOLDS_K = (240, 220, 210)
IR_IDS = ("cloud_ir_compare", "cloud_ir_scores", "cloud_ir_fss")
MAX_IR_TIMES = 8
IR_TOL_MIN = 30


def _agent_ir_figures(run_dir, run_label, domain, case_info, out_dir, spinup_hours=6,
                      map_time=None, tag=""):
    """Model OLR-derived Tb vs infrared brightness temperature an agent fetched
    (post/agent_obs.py, satellite_ir), MET Grid-Stat on the model grid pooled over up to
    MAX_IR_TIMES times; events are cold tops (Tb <= threshold). The map shows the verified
    time nearest map_time (the rain peak), else the one with the most cold cloud. tag: file
    name suffix (one set per domain)."""
    import csv
    from datetime import datetime
    import netCDF4
    import numpy as np
    from wtune.post import agent_obs, condensate, met_gridstat, obs_figs, snapshot_verify
    from wtune.post.caseinfo import WRF_TIME_FMT, spec_bdy_width
    from wtune.post.recipe import fig, missing, note
    from wtune.post.recipes.convection import (_pool_counts, _pool_fss, _post_spinup_frames,
                                               _project_of)
    project_dir = _project_of(out_dir)
    if not project_dir:
        return []
    ok, bad = agent_obs.usable(project_dir, "satellite_ir")
    out = [note("cloud", "agent dataset %s is not used: %s (wtune event obs-check)" % (n, e))
           for n, e in bad]
    if not ok:
        return out
    times = [(datetime.strptime(t, WRF_TIME_FMT), (p, ti))
             for t, p, ti in _post_spinup_frames(run_dir, domain, spinup_hours)]
    dx = [d.get("dx_km") for d in case_info["domains"] if d["domain"] == domain]
    cell_km = dx[0] if dx and dx[0] else 10.0

    def model_field(path, ti):
        with netCDF4.Dataset(path) as ds:
            tb = np.asarray(condensate.olr_brightness_temperature(ds, ti), "f8")
            return (np.asarray(ds.variables["XLAT"][ti], "f8"),
                    np.asarray(ds.variables["XLONG"][ti], "f8"), tb, "K")

    for name, d, prov in ok:
        label = name.upper()        # the dataset string is a sentence; it ran off the panel
        idx = agent_obs.snapshot_index(d, "satellite_ir")
        picks = [(t, ref, f) for t, ref in times for f in [agent_obs.nearest(idx, t, IR_TOL_MIN)] if f]
        if not picks:
            out += [missing(i, label, "no irtb file within %d min of a model output time after "
                            "spin-up" % IR_TOL_MIN) for i in IR_IDS]
            continue
        step = max(1, -(-len(picks) // MAX_IR_TIMES))
        keep = set(range(0, len(picks), step))
        if map_time:                    # thinning dropped the rain peak's 00Z (haitang2017)
            peak = datetime.strptime(map_time, WRF_TIME_FMT)
            keep.add(min(range(len(picks)), key=lambda k: abs(picks[k][0] - peak)))
        picks = [picks[k] for k in sorted(keep)]
        work = os.path.join(out_dir, "gridstat_ir_%s%s" % (name, tag))
        os.makedirs(work, exist_ok=True)
        try:
            lat0, lon0, _tb, _u = model_field(*picks[0][1])
            bbox = (lat0.min(), lat0.max(), lon0.min(), lon0.max())
            poly = met_gridstat.write_interior_polygon(picks[0][1][0], spec_bdy_width(run_dir),
                                                       os.path.join(work, "interior.poly"))
            res = snapshot_verify.verify(
                picks, model_field,
                lambda p, t, griddes, onc: agent_obs.regrid_snapshot(
                    p, "tb", t, griddes, onc, "TB_OBS", "K", bbox, "remapcon"),
                ("TB_MODEL", "TB_OBS"), ["<=%g" % x for x in IR_THRESHOLDS_K], work, poly,
                obtype=name.upper())
        except (tools.ToolNotFoundError, tools.ToolError, met_gridstat.METError, ImportError,
                OSError, KeyError, ValueError) as e:
            out += [missing(i, label, "verification failed: %s" % e) for i in IR_IDS]
            continue
        if not res["fields"]:
            out += [missing(i, label, "no matched cells inside the model interior") for i in IR_IDS]
            continue
        pooled, pooled_fss = _pool_counts(res["per_ctc"]), _pool_fss(res["per_nbr"])
        csv_ts = os.path.join(out_dir, "cloud_ir_scores_by_time_%s%s.csv" % (name, tag))
        with open(csv_ts, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["valid_time_utc", "threshold_K_or_colder", "POD", "FAR", "CSI",
                        "frequency_bias", "n_points"])
            for t, r in res["cts_rows"]:
                w.writerow([t.strftime("%Y-%m-%d %H:%M"), r["thresh"], r["pod"], r["far"],
                            r["csi"], r["fbias"], r["total"]])
        csv_pool = os.path.join(out_dir, "cloud_ir_scores_pooled_%s%s.csv" % (name, tag))
        with open(csv_pool, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["threshold_K_or_colder", "POD", "FAR", "CSI", "frequency_bias",
                        "n_points_all_times"])
            for r in pooled:
                w.writerow([r["thresh"], r["pod"], r["far"], r["csi"], r["fbias"], r["total"]])
            w.writerow([])
            w.writerow(["threshold_K_or_colder", "width_points", "width_km", "FSS_pooled"])
            for r in pooled_fss:
                w.writerow([r["thresh"], r["width"], round(r["width"] * cell_km, 2), r["fss"]])
        nt = len(res["fields"])
        span = "%s..%s" % (res["fields"][0][0].strftime(WRF_TIME_FMT),
                           res["fields"][-1][0].strftime(WRF_TIME_FMT))
        common = dict(obs_sources=[agent_obs.source_tag(name, prov)], csv=[csv_ts, csv_pool],
                      n_times=nt, tool="MET grid_stat on the model grid (obs cdo remapcon), "
                                       "pooled over %d time(s)" % nt)
        if map_time:
            peak = datetime.strptime(map_time, WRF_TIME_FMT)
            tb_, glat, glon, gf, go = min(res["fields"], key=lambda f: abs(f[0] - peak))
        else:
            tb_, glat, glon, gf, go = max(res["fields"], key=lambda f: int(np.sum(f[4] <= 220)))
        cmap, _norm, _lev = ir_colormap()
        p = obs_figs.side_by_side(
            glat, glon, gf, go, ("%s d%02d model, OLR-derived Tb (broadband)" % (run_label, domain),
                                 "%s (obs)" % label),
            "IR brightness temperature %s UTC, matched points" % tb_.strftime("%Y-%m-%d %H:%M"),
            os.path.join(out_dir, "cloud_ir_compare_%s%s.png" % (name, tag)),
            np.arange(IR_TMIN_K, IR_TMAX_K + 1, 5.0), cmap=cmap, cbar_label="K", extend="both")
        out.append(fig(IR_IDS[0], p, run_label, domain="d%02d" % domain,
                       valid_time=tb_.strftime(WRF_TIME_FMT), extra=common))
        p = obs_figs.scores_vs_threshold(
            pooled, "Grid-Stat cold-cloud scores vs %s, %d time(s) pooled" % (label, nt),
            os.path.join(out_dir, "cloud_ir_scores_%s%s.png" % (name, tag)),
            "brightness temperature (K)", op="≤")
        out.append(fig(IR_IDS[1], p, run_label, domain="d%02d" % domain, valid_time=span,
                       extra=common))
        p = obs_figs.fss_vs_scale(pooled_fss, cell_km, "Grid-Stat FSS vs %s, %d time(s) pooled"
                                  % (label, nt), os.path.join(out_dir, "cloud_ir_fss_%s%s.png" % (name, tag)),
                                  " K or colder")
        out.append(fig(IR_IDS[2], p, run_label, domain="d%02d" % domain, valid_time=span,
                       extra=common))
    return out


def figures(run_dirs, labels, out_dir, case_info, every=None, times=None, paraview=True,
            map_time=None, focus_box=None):
    """The cloud figure set for run_dirs[0] (records per post.recipe fig()/missing()); the
    one-time figures at the frame nearest map_time, else the last frame."""
    import netCDF4
    from wtune.post.recipe import fig, missing
    run_dir, run_label = run_dirs[0], labels[0]
    os.makedirs(out_dir, exist_ok=True)
    bdy_width = bdy_width_for(run_dir)
    out = []
    for domain in _cloud_domains(case_info):
        dom = "d%02d" % domain
        frames = list_frame_refs(run_dir, domain=domain)      # [(time, path, timeidx)]
        if not frames:
            out.append(missing("cloud_water_path", "wrfout_%s_*" % dom, "no frames found",
                               run=run_label))
            continue
        last_time, last_path, last_ti = frame_at(frames, map_time)
        try:
            ok, _sp = has_hydrometeors(last_path)
        except (OSError, ImportError) as e:
            out.append(missing("cloud_water_path", "wrfout_%s" % dom, _reason(e), run=run_label))
            continue
        if not ok:
            out.append(missing("cloud_water_path", "hydrometeor mixing ratios in wrfout_%s" % dom,
                               "no QCLOUD/QRAIN/QICE/QSNOW/QGRAUP/QHAIL in this domain's "
                               "history stream", run=run_label))
            continue
        by_time = {t: (path, ti) for t, path, ti in frames}

        for t in _select_times([f[0] for f in frames], every, times):
            p = os.path.join(out_dir, "cloud_water_path_%s_%s.png" % (dom, t.replace(":", "")))
            try:
                water_path_figure(by_time[t][0], t, p, run_label, domain, timeidx=by_time[t][1])
                out.append(fig("cloud_water_path", p, run_label, domain=dom, valid_time=t))
            except _ERRORS as e:
                out.append(missing("cloud_water_path", "wrfout_%s %s" % (dom, t), _reason(e),
                                   run=run_label))

        p = os.path.join(out_dir, "cloud_top_height_%s.png" % dom)
        try:
            cloud_top_figure(last_path, last_time, p, run_label, domain, timeidx=last_ti)
            out.append(fig("cloud_top_height", p, run_label, domain=dom, valid_time=last_time))
        except _ERRORS as e:
            out.append(missing("cloud_top_height", "wrfout_%s (wrf-python mdbz)" % dom,
                               _reason(e), run=run_label))

        p = os.path.join(out_dir, "cloud_xsec_%s.png" % dom)
        try:
            cloud_xsec_figure(last_path, last_time, p, run_label, domain, bdy_width,
                              timeidx=last_ti, focus_box=focus_box)
            out.append(fig("cloud_xsec", p, run_label, domain=dom, valid_time=last_time))
        except _ERRORS as e:
            out.append(missing("cloud_xsec", "wrfout_%s (wrf-python z/ua/va/wa/tc, PBLH)" % dom,
                               _reason(e), run=run_label))

        p = os.path.join(out_dir, "cloud_ir_proxy_%s.png" % dom)
        try:
            with netCDF4.Dataset(last_path) as ds:
                plot_ir_proxy(ds, last_time, p, run_label, domain, timeidx=last_ti)
            out.append(fig("cloud_ir_proxy", p, run_label, domain=dom, valid_time=last_time,
                           extra={"note": IR_TITLE}))
        except _ERRORS as e:
            out.append(missing("cloud_ir_proxy", "wrfout_%s OLR" % dom, _reason(e),
                               run=run_label))

        if paraview:
            p = os.path.join(out_dir, "cloud_condensate_3d_%s.png" % dom)
            try:
                written = condensate_volume(
                    last_path, p, os.path.join(out_dir, "_paraview_%s" % dom),
                    "%s  %s  total condensate volume  %s" % (run_label, dom, last_time),
                    timeidx=last_ti)
                out.append(fig("cloud_condensate_3d", written or p, run_label, domain=dom,
                               valid_time=last_time))
            except Exception as e:    # noqa: BLE001 -- pvbatch absent/failed, not implemented: a gap, never a failed report
                out.append(missing("cloud_condensate_3d", "pvbatch / paraview3d."
                                   "render_condensate_volume", _reason(e), run=run_label))
    for k, dom_n in enumerate(_cloud_domains(case_info)):   # d01 and the finest nest
        out += _agent_ir_figures(run_dir, run_label, dom_n, case_info, out_dir,
                                 map_time=map_time, tag="" if k == 0 else "_d%02d" % dom_n)
    return out
