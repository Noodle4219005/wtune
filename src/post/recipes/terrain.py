"""post/recipes/terrain.py -- the "terrain" figure set (decision 0057 update, 2026-09-29):
domain/nest studies over complex terrain (any domain's HGT relief >= 1000 m or peak >=
1500 m -- post/caseinfo.py's detect_terrain()). Every figure is built from wrf-python
diagnostics (`ter`, `uvmet10`, `uvmet`, `theta`, `wa`, `vertcross`, `z`) plus cartopy, per
decision 0057's own tool list for this type; no new numerics beyond plain indexing/argmin are
written here.

Cross-section line default: through the domain's highest terrain point, along the domain-mean
700 hPa wind direction, extended +/-150 km (project.yaml's xsec_halfwidth_km) -- built
directly in lat/lon via cartopy.geodesic (post/recipes/xsection.py), not grid-index
extension, so there is no index-space-vs-geographic-direction approximation left in the
line's own placement (2026-09-29 lead review); project.yaml's cross_sections.terrain still
exists as a full override for a line the mean-700-hPa-wind default doesn't fit. The figure
itself follows the standard mountain-wave cross-section convention (w shading, theta
isentrope lines, a muted along-line-wind contour, a filled terrain silhouette, a locator
inset -- 2026-09-29 second lead review), height window project.yaml's xsec_top_km (default
10 km). Station-observation figures (lee-side T2/RH/wind, MET Point-Stat) are reported as
explicit missing entries: no station list is fetched or configured in this round, matching
the house convention in post/recipes/precip.py.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frames


DEFAULT_XSEC_TOP_KM = 10.0


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fill_below_valid(arr):
    """Per column: fill every NaN level BELOW the column's own lowest FINITE sample with that
    lowest finite value -- constant extrapolation downward all the way to the bottom of the
    array (index 0), including levels that are physically below ground. Never fills ABOVE
    the lowest valid sample.

    Sixth lead review: the previous version stopped filling at the local terrain height and
    left the true below-ground cells NaN/masked, which seemed right but wasn't -- matplotlib's
    contourf drops ANY grid quad that has even one masked corner, so a masked cell in a
    TALLER neighbouring column wedged out perfectly valid, above-ground cells in a shorter
    column right next to it (the "white wedge" persisted after the fifth review's ground-up
    fill because that fix only addressed the gap between terrain and the lowest valid
    *sample*, not the gap between terrain and the array's own bottom, z=0). Filling the whole
    column removes every masked corner from the plotted arrays entirely; the terrain
    silhouette polygon (drawn at a higher zorder) is what visually hides the underground
    portion now, not a data mask. `arr`: (n_levels, n_cols). Returns a new array; `arr` itself
    is not modified."""
    import numpy as np
    arr = arr.copy()
    n_cols = arr.shape[1]
    for col in range(n_cols):
        col_vals = arr[:, col]
        finite_idx = np.flatnonzero(np.isfinite(col_vals))
        if finite_idx.size == 0 or finite_idx[0] == 0:
            continue
        first_valid = finite_idx[0]
        col_vals[:first_valid] = col_vals[first_valid]
    return arr


def _terrain_wind_map(run_dir, run_label, domain, out_dir):
    """Terrain fill + 10 m wind (black barbs) + 700 hPa wind (red barbs), one figure
    (decision 0057 update item 1)."""
    import netCDF4
    import numpy as np
    import cartopy.crs as ccrs
    wrf_mod = __import__("wrf")
    from post.plotting import WIND_SUBSAMPLE, _coastline, _natural_earth_ready, _set_cartopy_data_dir

    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return None
    time0, path0 = frames[-1]
    plt = _plt()
    with netCDF4.Dataset(path0) as ds:
        ter = wrf_mod.getvar(ds, "ter", timeidx=0)
        lats, lons = wrf_mod.latlon_coords(ter, as_np=True)
        crs = wrf_mod.get_cartopy(var=ter)
        u10, v10 = wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")
        pressure = wrf_mod.getvar(ds, "pressure", timeidx=0)
        u_full, v_full = wrf_mod.getvar(ds, "uvmet", timeidx=0, units="m s-1")
        u700 = wrf_mod.interplevel(u_full, pressure, 700.0)
        v700 = wrf_mod.interplevel(v_full, pressure, 700.0)

        _set_cartopy_data_dir()
        natural_earth_ok = _natural_earth_ready()
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": crs})
        # wrf.get_cartopy() on a nested (non-d01) domain can return a Mercator CRS whose
        # cartopy-computed default extent is the whole globe rather than this domain's own
        # small footprint (observed on maria1km d02, 2026-09-29 scc T1 validation run --
        # d01's own CRS does not show this, which is why post/recipes/common.py's domain map
        # is unaffected: it always uses d01). Pin the extent to the actual lon/lat data range
        # explicitly, rather than relying on cartopy's autoscale.
        margin = 0.3
        ax.set_extent([float(lons.min()) - margin, float(lons.max()) + margin,
                      float(lats.min()) - margin, float(lats.max()) + margin],
                     crs=ccrs.PlateCarree())
        cf = ax.contourf(lons, lats, np.asarray(ter), levels=20, cmap="terrain",
                         transform=ccrs.PlateCarree())
        fig.colorbar(cf, ax=ax, label="terrain height (m)")
        _coastline(ax, ds, ccrs, natural_earth_ok)

        # WIND_SUBSAMPLE (post/plotting.py) is tuned for the coarser regional domains that
        # module usually plots; a 1 km nest has far more grid points across the same
        # displayed extent, so the fixed stride left barbs dense enough to look like a solid
        # colour fill (maria1km d02, 2026-09-29 scc T1 run). Scale the stride to the domain's
        # own grid size instead, aiming for roughly 25 barbs across the shorter axis.
        stride = max(WIND_SUBSAMPLE, min(lons.shape) // 25)
        s = slice(None, None, stride)
        ax.barbs(lons[s, s], lats[s, s], np.asarray(u10)[s, s], np.asarray(v10)[s, s],
                 length=5, color="black", transform=ccrs.PlateCarree(), label="10 m wind")
        ax.barbs(lons[s, s], lats[s, s], np.asarray(u700)[s, s], np.asarray(v700)[s, s],
                 length=5, color="red", transform=ccrs.PlateCarree(), label="700 hPa wind")
        ax.set_title("%s  d%02d  terrain + 10 m (black) / 700 hPa (red) wind  %s"
                     % (run_label, domain, time0), fontsize=9)
        out_path = os.path.join(out_dir, "terrain_wind_d%02d.png" % domain)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


class CrossSectionError(RuntimeError):
    pass


def default_cross_section_line(run_dir, domain, halfwidth_km=None, bdy_width=None):
    """A geodesic line through the domain INTERIOR's highest HGT point, extended
    halfwidth_km (default post/recipes/xsection.DEFAULT_HALFWIDTH_KM, 150 km) in each
    direction along the domain-mean 700 hPa wind bearing -- placed directly in lat/lon via
    post/recipes/xsection.line_from_point_bearing (cartopy.geodesic, WGS84).

    "Interior" excludes bdy_width cells (default post/caseinfo.spec_bdy_width(), WRF's own
    &bdy_control spec_bdy_width, normally 5) from every domain edge -- WRF's lateral boundary
    zone, where the solution is specified/relaxed toward the driving data, not produced by
    the model's own dynamics (2026-09-29 fourth lead review: katrina m1's d01 "peak" (3411 m)
    turned out to sit exactly in that zone; the whole default line -- crest, wind-direction
    average, AND the clip below -- must stay inside the interior, not just inside the domain).
    The crest search, the 700 hPa wind averaging, and clip_halfwidth_to_domain's containment
    test are therefore all restricted to (or evaluated against) the SAME interior, using a
    grid-index predicate (wrf.ll_to_xy against the interior's own index range) rather than a
    lat/lon bounding box: a Lambert-projected grid's true footprint is not a lat/lon
    rectangle, so a bbox-based clip could still accept a point the real interior excludes.

    Raises CrossSectionError if the domain is too small for bdy_width to leave any interior
    at all. Returns a dict: start_ll, end_ll, peak_ll, bearing_deg, halfwidth_km, bdy_width,
    time0, path0, hgt, lats2d, lons2d, interior_bounds (j0, j1, i0, i1)."""
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    from post.caseinfo import _interior_bounds, spec_bdy_width
    from post.recipes.xsection import (DEFAULT_HALFWIDTH_KM, clip_halfwidth_to_domain,
                                       line_from_point_bearing)

    halfwidth_km = halfwidth_km if halfwidth_km is not None else DEFAULT_HALFWIDTH_KM
    frames = list_frames(run_dir, domain=domain)
    time0, path0 = frames[-1]
    bdy_width = bdy_width if bdy_width is not None else spec_bdy_width(run_dir)

    with netCDF4.Dataset(path0) as ds:
        ter = wrf_mod.getvar(ds, "ter", timeidx=0)
        hgt = np.asarray(ter)
        lats2d, lons2d = wrf_mod.latlon_coords(ter, as_np=True)
        pressure = wrf_mod.getvar(ds, "pressure", timeidx=0)
        u_full, v_full = wrf_mod.getvar(ds, "uvmet", timeidx=0, units="m s-1")
        u700 = np.asarray(wrf_mod.interplevel(u_full, pressure, 700.0, missing=np.nan))
        v700 = np.asarray(wrf_mod.interplevel(v_full, pressure, 700.0, missing=np.nan))

        ny, nx = hgt.shape
        bounds = _interior_bounds(ny, nx, bdy_width)
        if bounds is None:
            raise CrossSectionError(
                "d%02d (%dx%d) is too small for spec_bdy_width=%d to leave any interior"
                % (domain, ny, nx, bdy_width))
        j0, j1, i0, i1 = bounds

        interior_hgt = hgt[j0:j1, i0:i1]
        j_local, i_local = np.unravel_index(np.nanargmax(interior_hgt), interior_hgt.shape)
        j_peak, i_peak = j_local + j0, i_local + i0
        peak_ll = (float(lats2d[j_peak, i_peak]), float(lons2d[j_peak, i_peak]))

        mean_u = float(np.nanmean(u700[j0:j1, i0:i1]))
        mean_v = float(np.nanmean(v700[j0:j1, i0:i1]))
        norm = (mean_u ** 2 + mean_v ** 2) ** 0.5
        # compass bearing (0=N, 90=E, ...) of the mean flow -- the direction the wind blows
        # TOWARD; downstream/positive-x per xsection.line_from_point_bearing's own convention.
        bearing_deg = (np.degrees(np.arctan2(mean_u, mean_v)) % 360.0) if norm > 1e-6 else 90.0

        def _inside_interior(lat, lon):
            xy = wrf_mod.ll_to_xy(ds, lat, lon)
            i, j = int(xy[0]), int(xy[1])
            return i0 <= i < i1 and j0 <= j < j1

        clipped_halfwidth_km = clip_halfwidth_to_domain(peak_ll[0], peak_ll[1], bearing_deg,
                                                        halfwidth_km, _inside_interior)
        start_ll, end_ll = line_from_point_bearing(peak_ll[0], peak_ll[1], bearing_deg,
                                                   clipped_halfwidth_km)

    return dict(start_ll=start_ll, end_ll=end_ll, peak_ll=peak_ll, bearing_deg=bearing_deg,
               halfwidth_km=clipped_halfwidth_km, bdy_width=bdy_width, time0=time0,
               path0=path0, hgt=hgt, lats2d=lats2d, lons2d=lons2d,
               interior_bounds=(j0, j1, i0, i1))


def _cross_section_figure(run_dir, run_label, domain, out_dir, line=None, top_km=None,
                          halfwidth_km=None):
    """Standard mountain-wave-style cross-section (2026-09-29 second lead review): w shading
    (RdBu_r, symmetric about 0), theta isentropes every 2 K as thin black lines, a muted
    along-line-wind contour, a filled terrain silhouette, a real-km distance axis centred on
    the ridge crest (negative = upstream of the mean 700 hPa wind), and a small locator
    inset. `line`: optional (lat0,lon0,lat1,lon1) override from project.yaml's
    cross_sections.terrain (the ridge-crest centring/inset marker are skipped for an explicit
    override, since it is not guaranteed to pass through the domain's own HGT peak).
    `top_km`/`halfwidth_km`: project.yaml's xsec_top_km/xsec_halfwidth_km, else this module's
    own defaults. missing=np.nan on every vertcross call: wrf.vertcross's own default
    "missing" value below ground is a huge finite sentinel (~9.97e36), not NaN -- the same
    landmine as post/ivt.py's fix for moisture_ivt."""
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    from post.recipes.xsection import (centred_distance_km, draw_locator_inset,
                                       line_unit_vector, smooth_horizontal)

    top_km = top_km if top_km is not None else DEFAULT_XSEC_TOP_KM
    line_info = default_cross_section_line(run_dir, domain, halfwidth_km=halfwidth_km)
    time0, path0 = line_info["time0"], line_info["path0"]
    hgt, lats2d, lons2d = line_info["hgt"], line_info["lats2d"], line_info["lons2d"]

    if line:
        start_ll = (line["lat0"], line["lon0"])
        end_ll = (line["lat1"], line["lon1"])
        peak_marker = None
    else:
        start_ll, end_ll = line_info["start_ll"], line_info["end_ll"]
        peak_marker = line_info["peak_ll"]

    start = wrf_mod.CoordPair(lat=start_ll[0], lon=start_ll[1])
    end = wrf_mod.CoordPair(lat=end_ll[0], lon=end_ll[1])
    di, dj = line_unit_vector(start_ll, end_ll)

    # explicit height levels, starting at 0 m with a fine (100 m) spacing, instead of
    # wrf.vertcross's own default autolevels=100 (spread evenly across the WHOLE line's
    # z-range, top to bottom) -- autolevels can leave a column's first few sample heights
    # below that column's own local terrain even though they are above the line's overall
    # minimum terrain, which is what produced the white gaps between the silhouette and the
    # shading reported in the fourth lead review. A dense, ground-up grid plus the explicit
    # below-ground mask right after removes both the cause and any residual effect.
    step_m = 100.0
    level_m = np.arange(0.0, top_km * 1000.0 + step_m, step_m)

    plt = _plt()
    with netCDF4.Dataset(path0) as ds:
        z = wrf_mod.getvar(ds, "z", timeidx=0, units="m")
        theta = wrf_mod.getvar(ds, "theta", timeidx=0, units="K")
        wa = wrf_mod.getvar(ds, "wa", timeidx=0, units="m s-1")
        ter = wrf_mod.getvar(ds, "ter", timeidx=0)
        u_full, v_full = wrf_mod.getvar(ds, "uvmet", timeidx=0, units="m s-1")

        theta_cross = wrf_mod.vertcross(theta, z, levels=level_m, wrfin=ds, start_point=start,
                                        end_point=end, latlon=True, missing=np.nan)
        w_cross = wrf_mod.vertcross(wa, z, levels=level_m, wrfin=ds, start_point=start,
                                    end_point=end, missing=np.nan)
        u_cross = wrf_mod.vertcross(u_full, z, levels=level_m, wrfin=ds, start_point=start,
                                    end_point=end, missing=np.nan)
        v_cross = wrf_mod.vertcross(v_full, z, levels=level_m, wrfin=ds, start_point=start,
                                    end_point=end, missing=np.nan)
        ter_line = wrf_mod.interpline(ter, wrfin=ds, start_point=start, end_point=end)

    theta_np = np.asarray(theta_cross)
    w_np_raw = np.asarray(w_cross)
    along_wind_np_raw = np.asarray(u_cross) * di + np.asarray(v_cross) * dj
    ter_np_km = np.asarray(ter_line) / 1000.0
    xy_loc = theta_cross.coords["xy_loc"].values
    x_km = centred_distance_km(xy_loc)
    y_km = np.asarray(theta_cross.coords["vertical"].values) / 1000.0

    # fill every column all the way down to z=0, no below-ground mask at all (sixth lead
    # review): matplotlib's contourf drops any grid quad with even one masked corner, so a
    # below-ground NaN cell in a TALLER neighbouring column was wedging out perfectly valid,
    # above-ground cells in a shorter column right next to it -- masking "correctly" was
    # itself the bug. The terrain silhouette polygon below is drawn at a higher zorder than
    # everything else and is what actually hides the underground portion now.
    theta_np = _fill_below_valid(theta_np)
    w_np_raw = _fill_below_valid(w_np_raw)
    along_wind_np_raw = _fill_below_valid(along_wind_np_raw)

    # trim to the requested height window BEFORE any level/percentile computation (item 1:
    # "symmetric limits ... over the plotted window").
    keep = y_km <= top_km * 1.02
    if keep.sum() < 2:
        keep = np.ones_like(y_km, dtype=bool)
    y_km, theta_np = y_km[keep], theta_np[keep, :]
    # w is the shading now (item 1), not a contour -- shade the raw wrf.vertcross values
    # unsmoothed ("keep all numbers from vertcross/interpline ... presentation only", second
    # lead review). Only the along-line-wind CONTOUR (item 3) still gets display smoothing,
    # the same reason as before: discrete contour LINES tangle on 1 km-resolution noise in a
    # way a continuous colour fill does not.
    w_np = w_np_raw[keep, :]
    along_wind_np = smooth_horizontal(along_wind_np_raw[keep, :])

    # main axes | colour bar | locator inset, side by side so the inset never covers the bar
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_axes([0.07, 0.15, 0.60, 0.72])
    cax = fig.add_axes([0.685, 0.15, 0.015, 0.72])

    # 1) w shading: diverging, centred on 0, symmetric at the 99th-percentile |w| over the
    # plotted window (item 1).
    finite_w = w_np[np.isfinite(w_np)]
    vmax = float(np.nanpercentile(np.abs(finite_w), 99)) if finite_w.size else 1.0
    vmax = max(vmax, 1e-3)
    from matplotlib.ticker import MaxNLocator
    w_cmap = plt.get_cmap("RdBu_r")
    # the axes facecolour (shown through any residual NaN/masked cell) must render IDENTICAL
    # to the zero-straddling bin's own colour, so a viewer can never mistake "no data" for
    # "data near zero" (fifth lead review, item 2) -- cmap(0.5) is that bin's colour exactly,
    # since the levels below are built symmetric about 0.
    ax.set_facecolor(w_cmap(0.5))
    w_levels = MaxNLocator(nbins=12, symmetric=True).tick_values(-vmax, vmax)
    # no boundary at 0: the bin straddling 0 is one white band instead of pale blue|pink
    # stripes from every |w| < first level
    w_levels = w_levels[np.abs(w_levels) > 1e-12]
    cf = ax.contourf(x_km, y_km, w_np, levels=w_levels, cmap=w_cmap, extend="both", zorder=1)
    cbar = fig.colorbar(cf, cax=cax, ticks=w_levels)
    cbar.set_label("w (m/s)")

    # 2) theta isentropes every 2 K, thin black LINES (not a fill) -- this is what shows
    # mountain waves (lifted/sinking isentropes); only every 5th line is labelled.
    finite_theta = theta_np[np.isfinite(theta_np)]
    if finite_theta.size:
        t0 = np.floor(np.nanmin(finite_theta) / 2.0) * 2.0
        t1 = np.ceil(np.nanmax(finite_theta) / 2.0) * 2.0 + 2.0
        theta_levels = np.arange(t0, t1, 2.0)
        cs_theta = ax.contour(x_km, y_km, theta_np, levels=theta_levels, colors="black",
                              linewidths=0.4, zorder=2)
        ax.clabel(cs_theta, levels=list(theta_levels[::5]), inline=True, fontsize=5, fmt="%g")

    # 3) along-line wind: 2-3 labelled contours in one muted colour (item 3 -- dropped the
    # earlier blue-dashed tangle; w is now the shading, not a second contour set).
    finite_wind = along_wind_np[np.isfinite(along_wind_np)]
    if finite_wind.size:
        wind_mag = float(np.nanpercentile(np.abs(finite_wind), 90))
        if wind_mag > 1e-3:
            cs_wind = ax.contour(x_km, y_km, along_wind_np, levels=sorted({-wind_mag, wind_mag}),
                                 colors="dimgray", linewidths=0.9, zorder=2)
            ax.clabel(cs_wind, inline=True, fontsize=6, fmt="%.0f m/s")

    # terrain silhouette: zorder above every data layer (1-2) so it hides the fake,
    # constant-filled-downward values below the true ground -- the shading/contour lines are
    # never masked any more (sixth lead review), so this polygon is now the ONLY thing
    # hiding the underground portion, not a data mask.
    ax.fill_between(x_km, 0.0, ter_np_km, color="saddlebrown", zorder=5)

    ax.set_xlim(x_km.min(), x_km.max())
    ax.set_ylim(0, top_km)
    ax.set_xlabel("distance from ridge crest (km, negative = upstream of 700 hPa wind)")
    ax.set_ylabel("height (km)")
    ax.set_title("%s  d%02d  cross-section, %s" % (run_label, domain, time0), fontsize=10)
    fig.text(0.5, 0.02, "(%.2f, %.2f) -> (%.2f, %.2f)"
             % (start_ll[0], start_ll[1], end_ll[0], end_ll[1]), ha="center", fontsize=7)

    draw_locator_inset(fig, [0.78, 0.50, 0.20, 0.37], lons2d, lats2d, hgt, start_ll, end_ll,
                       point_ll=peak_marker)

    out_path = os.path.join(out_dir, "terrain_crosssection_d%02d.png" % domain)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _rain_over_terrain(run_dir, run_label, domain, out_dir):
    """Accumulated rain filled contour + HGT contour lines overlaid (orographic precipitation,
    decision 0057 update item 3)."""
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    from post.recipes.mapfig import save_map

    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return None
    time0, path0 = frames[-1]
    with netCDF4.Dataset(path0) as ds:
        rainc = wrf_mod.getvar(ds, "RAINC", timeidx=0)
        rainnc = wrf_mod.getvar(ds, "RAINNC", timeidx=0)
        rain = np.asarray(rainc + rainnc)
        ter = np.asarray(wrf_mod.getvar(ds, "ter", timeidx=0))
        lats, lons = wrf_mod.latlon_coords(wrf_mod.getvar(ds, "ter", timeidx=0), as_np=True)
        crs = wrf_mod.get_cartopy(var=wrf_mod.getvar(ds, "ter", timeidx=0))
        out_path = os.path.join(out_dir, "terrain_rain_d%02d.png" % domain)
        save_map(out_path, lats, lons,
                "%s  d%02d  accumulated rain (fill) + terrain (contour)  %s"
                % (run_label, domain, time0), fill_field=rain, fill_label="rain (mm)",
                fill_cmap="Blues", contour_field=ter,
                contour_levels=[500, 1000, 1500, 2000, 2500, 3000], contour_color="brown",
                ds=ds, crs=crs, basemap="cartopy")
    return out_path


VALLEY_WINDOW = 5           # grid cells, odd -- the neighbourhood a "local minimum" is judged in
VALLEY_RIDGE_DELTA_M = 300.0  # required relief between the point and its own window's max


def _find_valley_point(hgt, landmask, bounds, window=VALLEY_WINDOW,
                       ridge_delta_m=VALLEY_RIDGE_DELTA_M):
    """(j, i) of a genuine valley: LANDMASK==1, inside the domain interior (`bounds` =
    (j0, j1, i0, i1), excluding the lateral boundary zone), a local HGT minimum in a
    `window`x`window` neighbourhood, with that neighbourhood's own max at least
    ridge_delta_m higher -- real surrounding relief, not just a locally flat low patch
    (2026-09-29 fourth lead review: the previous global-HGT-minimum pick landed on a sea
    point, HGT=-1 m, sitting in the lateral boundary row, with a flat theta profile). None if
    no interior land point qualifies -- the caller reports that as missing(), not a forced
    pick."""
    import numpy as np
    from scipy.ndimage import maximum_filter, minimum_filter

    j0, j1, i0, i1 = bounds
    local_min = minimum_filter(hgt, size=window, mode="nearest")
    local_max = maximum_filter(hgt, size=window, mode="nearest")
    candidate = (hgt <= local_min + 1e-6) & ((local_max - hgt) >= ridge_delta_m)
    if landmask is not None:
        candidate &= (landmask == 1)
    mask = np.zeros_like(candidate)
    mask[j0:j1, i0:i1] = True
    candidate &= mask
    if not candidate.any():
        return None
    hgt_masked = np.where(candidate, hgt, np.inf)
    j, i = np.unravel_index(np.argmin(hgt_masked), hgt_masked.shape)
    return int(j), int(i)


def _short_time_label(wrf_time_str):
    """'2005-08-27_18:00:00' -> '27/18Z' (fourth lead review item 2: thin the tick labels,
    format like 27/00Z)."""
    from datetime import datetime
    dt = datetime.strptime(wrf_time_str, "%Y-%m-%d_%H:%M:%S")
    return dt.strftime("%d/%HZ")


VALLEY_TOP_AGL_KM = 3.0            # hard-coded per fifth lead review: "do not add a new flag"
VALLEY_THETA_CONTOUR_STEP_K = 1.0
VALLEY_BARB_HEIGHTS_KM = (0.25, 1.0, 1.75, 2.5)   # a few representative heights inside 0-3 km


def _valley_time_height(run_dir, run_label, domain, out_dir, bdy_width=None):
    """Time-height diagram (theta) at a genuine valley point (_find_valley_point) -- a
    boundary-layer diagnostic (cold pool, nocturnal inversion, daytime mixing; decision 0057
    update item 5). Plots 0..VALLEY_TOP_AGL_KM (fixed 3 km AGL -- the boundary layer, not the
    whole troposphere: the fourth lead review's 0-20 km/290-500 K version squeezed the only
    thing this figure exists to show into a few bottom pixels), with the colour range taken
    from that column's own values inside that window, 1 K black isentrope lines, optional
    3-hourly wind barbs (skipped if uvmet is not available), the valley's lat/lon + HGT in
    the title, and ticks pinned to whole 6 h marks. Returns (path, None) on success, (None,
    reason) when no qualifying point exists or fewer than 2 frames are available -- the
    caller turns `reason` into an explicit missing() entry rather than silently skipping or
    drawing a meaningless figure at a sea/boundary point."""
    import netCDF4
    import numpy as np
    from datetime import datetime
    wrf_mod = __import__("wrf")
    from post.caseinfo import _interior_bounds, spec_bdy_width

    frames = list_frames(run_dir, domain=domain)
    if len(frames) < 2:
        # a time-height diagram needs >=2 times; matplotlib's contourf refuses a
        # (n_levels, 1) array outright (a real crash caught on the scc T1 validation run,
        # 2026-09-29, against a single-frame maria1km wrfout).
        return None, "fewer than 2 frames available"

    bdy_width = bdy_width if bdy_width is not None else spec_bdy_width(run_dir)
    with netCDF4.Dataset(frames[0][1]) as ds:
        ter_var = wrf_mod.getvar(ds, "ter", timeidx=0)
        hgt = np.asarray(ter_var)
        lats2d, lons2d = wrf_mod.latlon_coords(ter_var, as_np=True)
        landmask = np.asarray(ds.variables["LANDMASK"][0]) if "LANDMASK" in ds.variables else None
    bounds = _interior_bounds(hgt.shape[0], hgt.shape[1], bdy_width)
    if bounds is None:
        return None, ("d%02d is too small for spec_bdy_width=%d to leave any interior"
                      % (domain, bdy_width))

    point = _find_valley_point(hgt, landmask, bounds)
    if point is None:
        return None, ("no interior land point is a genuine valley (LANDMASK==1, a local HGT "
                      "minimum in a %dx%d window with >=%g m surrounding relief, outside the "
                      "%d-cell lateral boundary zone)"
                      % (VALLEY_WINDOW, VALLEY_WINDOW, VALLEY_RIDGE_DELTA_M, bdy_width))
    j, i = point
    hgt_here = float(hgt[j, i])
    lat_here, lon_here = float(lats2d[j, i]), float(lons2d[j, i])

    times, theta_profiles, z_agl_profiles, u_profiles, v_profiles = [], [], [], [], []
    have_wind = True
    for t, path in frames:
        with netCDF4.Dataset(path) as ds:
            theta = np.asarray(wrf_mod.getvar(ds, "theta", timeidx=0, units="K"))[:, j, i]
            z = np.asarray(wrf_mod.getvar(ds, "z", timeidx=0, units="m"))[:, j, i]
            if have_wind:
                try:
                    u_full, v_full = wrf_mod.getvar(ds, "uvmet", timeidx=0, units="m s-1")
                    u_profiles.append(np.asarray(u_full)[:, j, i])
                    v_profiles.append(np.asarray(v_full)[:, j, i])
                except Exception:
                    have_wind = False   # "skip it if not [cheap/available]" -- fifth lead review
        times.append(t)
        theta_profiles.append(theta)
        z_agl_profiles.append(z - hgt_here)

    n_levels = min(len(p) for p in theta_profiles)
    theta_arr = np.array([p[:n_levels] for p in theta_profiles]).T          # (levels, time)
    z_agl_arr = np.mean([p[:n_levels] for p in z_agl_profiles], axis=0)     # (levels,)

    top_km = VALLEY_TOP_AGL_KM
    keep = (z_agl_arr / 1000.0) <= top_km * 1.05
    if keep.sum() < 2:
        return None, ("fewer than 2 model levels fall within 0-%g km AGL at this point"
                      % top_km)
    z_km = z_agl_arr[keep] / 1000.0
    theta_plot = theta_arr[keep, :]

    if have_wind and len(u_profiles) == len(times):
        u_arr = np.array([p[:n_levels] for p in u_profiles]).T[keep, :]
        v_arr = np.array([p[:n_levels] for p in v_profiles]).T[keep, :]
    else:
        have_wind = False

    finite_theta = theta_plot[np.isfinite(theta_plot)]
    t_min, t_max = (float(np.nanmin(finite_theta)), float(np.nanmax(finite_theta))) \
        if finite_theta.size else (290.0, 291.0)
    if t_max - t_min < 1.0:
        t_max = t_min + 1.0

    # integer-K levels (floor(min)..ceil(max) at 1 K steps) instead of 21 evenly-spaced float
    # levels across the exact [t_min, t_max] range -- the latter produced ugly non-integer
    # colour-bar ticks (297.99, 300.57, ...); this also makes the fill bins and the 1 K
    # contour lines below share the exact same level set.
    theta_contour_levels = np.arange(np.floor(t_min), np.ceil(t_max) + VALLEY_THETA_CONTOUR_STEP_K,
                                     VALLEY_THETA_CONTOUR_STEP_K)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cf = ax.contourf(range(len(times)), z_km, theta_plot, levels=theta_contour_levels,
                     cmap="viridis", extend="both")
    cbar = fig.colorbar(cf, ax=ax, label="theta (K)")
    cbar.set_ticks(theta_contour_levels[::2])   # every 2 K

    if len(theta_contour_levels) >= 2:
        cs = ax.contour(range(len(times)), z_km, theta_plot, levels=theta_contour_levels,
                        colors="black", linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=5, fmt="%g")

    if have_wind:
        barb_time_idx = [k for k, t in enumerate(times)
                         if datetime.strptime(t, "%Y-%m-%d_%H:%M:%S").hour % 3 == 0]
        for h_km in VALLEY_BARB_HEIGHTS_KM:
            if h_km > top_km or not barb_time_idx:
                continue
            level_idx = int(np.argmin(np.abs(z_km - h_km)))
            us = [u_arr[level_idx, k] for k in barb_time_idx]
            vs = [v_arr[level_idx, k] for k in barb_time_idx]
            ax.barbs(barb_time_idx, [h_km] * len(barb_time_idx), us, vs, length=5,
                     color="dimgray")

    # ticks at whole 6-hour marks (00/06/12/18Z), not an arbitrary evenly-spaced count.
    tick_idx = [k for k, t in enumerate(times)
               if datetime.strptime(t, "%Y-%m-%d_%H:%M:%S").hour % 6 == 0]
    if not tick_idx:
        tick_idx = list(range(0, len(times), max(1, len(times) // 8)))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([_short_time_label(times[k]) for k in tick_idx], rotation=45, fontsize=7)
    ax.set_ylim(0, top_km)
    ax.set_ylabel("height AGL (km)")
    ax.set_title("%s  d%02d  valley time-height (theta) at (%.2f, %.2f)  HGT=%.0f m"
                 % (run_label, domain, lat_here, lon_here, hgt_here), fontsize=9)
    out_path = os.path.join(out_dir, "terrain_valley_timeheight_d%02d.png" % domain)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path, None


def _finest_domain(domains):
    return domains[-1]["domain"]


def _terrain_domain(case_info):
    """The finest domain that ITSELF satisfies the terrain detection rule
    (case_info["terrain_domains"], from post/caseinfo.detect_terrain()) -- e.g. katrina m1:
    d01 has the relief, d02 does not, so every terrain figure must target d01, not the run's
    overall finest domain (2026-09-29 third lead review). Falls back to the overall finest
    domain only when terrain_domains is empty, which happens only when "terrain" was forced
    via --types on a case where no domain actually meets the relief/peak rule -- there is no
    "correct" domain in that situation, so the pre-existing default is kept rather than
    failing outright."""
    terrain_domains = case_info.get("terrain_domains") or []
    if terrain_domains:
        return max(terrain_domains)
    return _finest_domain(case_info["domains"])


def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6, cross_section=None,
           xsec_top_km=None, xsec_halfwidth_km=None):
    from post.recipe import fig, missing

    run_dir, run_label = run_dirs[0], labels[0]
    domain = _terrain_domain(case_info)
    out = []

    try:
        path = _terrain_wind_map(run_dir, run_label, domain, out_dir)
        if path:
            out.append(fig("terrain_wind_map", path, run_label, domain="d%02d" % domain))
        else:
            out.append(missing("terrain_wind_map", "wrfout_d%02d_*" % domain, "no frames found"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("terrain_wind_map", "wrf-python/cartopy", str(e)))

    try:
        path = _cross_section_figure(run_dir, run_label, domain, out_dir, line=cross_section,
                                     top_km=xsec_top_km, halfwidth_km=xsec_halfwidth_km)
        if path:
            out.append(fig("terrain_crosssection", path, run_label, domain="d%02d" % domain))
        else:
            out.append(missing("terrain_crosssection", "wrfout_d%02d_*" % domain, "no frames found"))
    except (tools.ToolNotFoundError, ImportError, ValueError, CrossSectionError) as e:
        out.append(missing("terrain_crosssection", "wrf-python", str(e)))

    try:
        path = _rain_over_terrain(run_dir, run_label, domain, out_dir)
        if path:
            out.append(fig("terrain_rain", path, run_label, domain="d%02d" % domain))
        else:
            out.append(missing("terrain_rain", "wrfout_d%02d_*" % domain, "no frames found"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("terrain_rain", "wrf-python/cartopy", str(e)))

    try:
        path, reason = _valley_time_height(run_dir, run_label, domain, out_dir)
        if path:
            out.append(fig("terrain_valley_timeheight", path, run_label, domain="d%02d" % domain))
        else:
            out.append(missing("terrain_valley_timeheight", "wrfout_d%02d_*" % domain, reason))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("terrain_valley_timeheight", "wrf-python", str(e)))

    out.append(missing("terrain_leeward_station", "station T2/RH/wind observations",
                       "no station list fetched or configured this round -- foehn diagnostic "
                       "needs a named lee-side point (not implemented, matching the "
                       "precip/synoptic obs-gap convention)"))
    out.append(missing("terrain_point_stat", "MET Point-Stat vs. station obs",
                       "no station observations fetched this round"))
    return out
