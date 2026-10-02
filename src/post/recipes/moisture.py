"""post/recipes/moisture.py -- the "moisture" figure set (decision 0057 update, 2026-09-29):
detected whenever precip, convection, or tc is detected (precip is itself always produced, so
this set is effectively always on alongside it -- matching decision 0057's own wording
literally: "偵測到 precip、convection 或 tc 時就做"). PW comes from wrf-python's own `pw`
diagnostic (post/plotting.py's plot_field, same engine `wtune post plot` uses); its 6 h change
reuses post/changemap.py (cdo sub). IVT is the one self-written integral in this package --
see post/ivt.py's docstring for why. The vertical cross-section and single-point time-height
figures use wrf.vertcross/getvar the same way post/recipes/terrain.py does.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frame_refs, list_frames


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _pw_figure(run_dir, run_label, domain, out_dir):
    from post.plotting import plot_field
    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return None
    last_time = frames[-1][0]
    written = plot_field(run_dir, "pw", out_dir, domain=domain, times=[last_time],
                         basemap="cartopy")
    return written[0] if written else None


def _pw_change_figure(run_dir, run_label, domain, out_dir, hours=6.0):
    """The 6 h PW change (decision 0057 update item 1) -- t_prev is the available frame
    nearest `hours` before the last frame, NOT simply the second-to-last frame: on hourly
    wrfout output (as opposed to 6-hourly), "the previous frame" is only 1 h earlier, which
    silently produced a wrongly-labelled 'moisture_pw_chg6h' entry actually showing a 1 h
    change (caught on the scc P7 validation run, 2026-09-29)."""
    from datetime import datetime, timedelta
    from post.changemap import change_map
    from post.timeseries import WRF_TIME_FMT

    frames = list_frames(run_dir, domain=domain)
    if len(frames) < 2:
        return None
    times = [t for t, _ in frames]
    t_curr = times[-1]
    target = datetime.strptime(t_curr, WRF_TIME_FMT) - timedelta(hours=hours)
    t_prev = min(times, key=lambda t: abs((datetime.strptime(t, WRF_TIME_FMT) - target).total_seconds()))
    if t_prev == t_curr:
        return None
    record = change_map(run_dir, run_label, "pw", domain, out_dir, t_prev, t_curr, times[0],
                        fig_id="moisture_pw_chg6h")
    return record


def _default_moisture_point(run_dir, domain, tc_centre=None):
    """(lat, lon) for the cross-section/time-height figures: the TC centre when given
    (decision 0057 update: "剖面線預設過颶風中心或最大雨量區"), else the grid cell with the
    largest accumulated rain at the last frame. Rule: INTERIOR cells only -- the
    spec_bdy_width cells at every domain edge are boundary-relaxed, not model physics
    (post/caseinfo.py::_interior_bounds, same lesson as the terrain figures). The storm moves,
    so a fixed point is only a sampling location, not a track: the title says which rule
    chose it (_point_reason)."""
    import netCDF4
    import numpy as np
    from post import caseinfo
    wrf_mod = __import__("wrf")
    if tc_centre:
        return tc_centre
    frames = list_frame_refs(run_dir, domain=domain)
    time0, path0, ti = frames[-1]
    with netCDF4.Dataset(path0) as ds:
        rainc = wrf_mod.getvar(ds, "RAINC", timeidx=ti)
        rainnc = wrf_mod.getvar(ds, "RAINNC", timeidx=ti)
        rain = np.asarray(rainc + rainnc)
        lats, lons = wrf_mod.latlon_coords(rainc, as_np=True)
    bounds = caseinfo._interior_bounds(rain.shape[0], rain.shape[1], caseinfo.spec_bdy_width(run_dir))
    if bounds is not None:
        j0, j1, i0, i1 = bounds
        sub = rain[j0:j1, i0:i1]
        j, i = np.unravel_index(np.nanargmax(sub), sub.shape)
        j, i = j + j0, i + i0
    else:
        j, i = np.unravel_index(np.nanargmax(rain), rain.shape)
    return float(lats[j, i]), float(lons[j, i])


def _point_reason(tc_centre):
    return ("TC centre (best track, last frame)" if tc_centre
            else "max accumulated rain at last frame, interior cells only")


def _ivt_figure(run_dir, run_label, domain, out_dir):
    import netCDF4
    import numpy as np
    import cartopy.crs as ccrs
    wrf_mod = __import__("wrf")
    from post.ivt import IVT_LEVELS_HPA, ivt_field
    from post.plotting import WIND_SUBSAMPLE, _coastline, _natural_earth_ready, _set_cartopy_data_dir

    frames = list_frame_refs(run_dir, domain=domain)
    if not frames:
        return None
    time0, path0, ti = frames[-1]
    with netCDF4.Dataset(path0) as ds:
        pressure = wrf_mod.getvar(ds, "pressure", timeidx=ti)
        qvapor = wrf_mod.getvar(ds, "QVAPOR", timeidx=ti)
        u_full, v_full = wrf_mod.getvar(ds, "uvmet", timeidx=ti, units="m s-1")

        # missing=np.nan: wrf.interplevel()'s own default ("missing", a huge finite sentinel
        # ~9.97e36, not NaN) below the local surface pressure is what made this figure blank
        # right over the storm on the P7 katrina run (2026-09-29 lead review) -- a low's own
        # centre has psfc < 1000 hPa, so the 1000 hPa level is locally underground, and that
        # sentinel (not NaN) getting summed into ivt_field's trapezoid integral produced huge
        # bogus values that dominated the colour scale. Requesting real NaN here, paired with
        # ivt_field's own NaN-aware integration (post/ivt.py), integrates only the above-
        # ground part of each column instead.
        q_levels, u_levels, v_levels = {}, {}, {}
        for lev in IVT_LEVELS_HPA:
            q_levels[lev] = np.asarray(wrf_mod.interplevel(qvapor, pressure, lev, missing=np.nan))
            u_levels[lev] = np.asarray(wrf_mod.interplevel(u_full, pressure, lev, missing=np.nan))
            v_levels[lev] = np.asarray(wrf_mod.interplevel(v_full, pressure, lev, missing=np.nan))
        ivt_u, ivt_v, ivt_mag = ivt_field(q_levels, u_levels, v_levels)

        u850 = np.asarray(wrf_mod.interplevel(u_full, pressure, 850.0, missing=np.nan))
        v850 = np.asarray(wrf_mod.interplevel(v_full, pressure, 850.0, missing=np.nan))
        ter = wrf_mod.getvar(ds, "ter", timeidx=ti)
        lats, lons = wrf_mod.latlon_coords(ter, as_np=True)
        crs = wrf_mod.get_cartopy(var=ter)

        _set_cartopy_data_dir()
        natural_earth_ok = _natural_earth_ready()
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": crs})
        # see post/recipes/terrain.py's _terrain_wind_map for why this is needed: wrf.
        # get_cartopy() on a nested domain's own CRS can auto-scale to the whole globe
        # instead of this domain's footprint (caught on maria1km's Mercator d02, 2026-09-29).
        margin = 0.3
        ax.set_extent([float(lons.min()) - margin, float(lons.max()) + margin,
                      float(lats.min()) - margin, float(lats.max()) + margin],
                     crs=ccrs.PlateCarree())
        # CW3E AR-scale levels (decision 0064); weaker IVT unshaded. A cold/dry case never
        # reaches 100 kg m-1 s-1 -- then fall back to round levels and say so in the title.
        from post.conventions import IVT_LEVELS
        from post.recipes.mapfig import nice_levels
        ivt_max = float(np.nanmax(ivt_mag)) if np.isfinite(ivt_mag).any() else 0.0
        ar_scale = ivt_max >= IVT_LEVELS[0]
        ivt_levels = IVT_LEVELS if ar_scale else nice_levels(0.0, max(ivt_max, 1.0), nbins=10)
        cf = ax.contourf(lons, lats, ivt_mag, levels=ivt_levels, cmap="YlGnBu", extend="max",
                         transform=ccrs.PlateCarree())
        fig.colorbar(cf, ax=ax, label="IVT (kg m$^{-1}$ s$^{-1}$)")
        _coastline(ax, ds, ccrs, natural_earth_ok)
        # see post/recipes/terrain.py's _terrain_wind_map: a fine nest has far more grid
        # points than WIND_SUBSAMPLE was tuned for, so scale the stride to this domain.
        stride = max(WIND_SUBSAMPLE, min(lons.shape) // 25)
        # barbs only inside the plotted interior: finite IVT fill, spec_bdy_width edge cells out
        from post.basemap import bdy_width_of, interior_mask, masked_barbs
        from post.conventions import KT_PER_MS, ds_time_label
        masked_barbs(ax, lons, lats, np.asarray(u850) * KT_PER_MS, np.asarray(v850) * KT_PER_MS,
                     stride, mask=interior_mask(lons.shape, bdy_width_of(ds), fill=ivt_mag),
                     length=5, transform=ccrs.PlateCarree())
        scale_note = ("AR scale, 250 = AR threshold" if ar_scale else
                      "max %.0f, below the AR scale (100+)" % ivt_max)
        ax.set_title("%s  d%02d  IVT (shaded; %s) + 850 hPa wind (kt)\n%s"
                     % (run_label, domain, scale_note, ds_time_label(ds, time0)), fontsize=9)
        out_path = os.path.join(out_dir, "moisture_ivt_d%02d.png" % domain)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


DEFAULT_XSEC_TOP_KM = 16.0   # a TC's eyewall updraft is deep -- decision per 2026-09-29 review


def grid_inside_predicate(xy_fn, ny, nx):
    """inside_fn(lat, lon) for clip_halfwidth_to_domain: True iff the nearest mass-grid index
    xy_fn(lat, lon) -> (i, j) lies in [0, nx-1] x [0, ny-1]. wrf.vertcross rejects an
    endpoint whose index is >= the grid shape, and a lat/lon bbox of a Lambert grid admits
    such points (d02 99x99: end_point (99, 70) was 'outside of domain')."""
    def inside(lat, lon):
        i, j = (int(v) for v in xy_fn(lat, lon))
        return 0 <= i <= nx - 1 and 0 <= j <= ny - 1
    return inside


def _cross_section_figure(run_dir, run_label, domain, out_dir, point, line=None, top_km=None,
                          halfwidth_km=None):
    """RH shading + theta_e isentrope lines, same presentation convention as
    post/recipes/terrain.py's cross-section (2026-09-29 second lead review, "apply 4-7
    consistently to moisture_crosssection too"): real-km distance axis centred on `point`
    (default: due east/west, +/-halfwidth_km), a 0..top_km height window (default 16 km, a
    TC's eyewall updraft is deep), a one-line title, and a locator inset. `line`: optional
    (lat0,lon0,lat1,lon1) override from project.yaml's cross_sections.moisture."""
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    from post.recipes.xsection import (DEFAULT_HALFWIDTH_KM,
                                       centred_distance_km, clip_halfwidth_to_domain,
                                       draw_locator_inset, line_from_point_bearing)

    top_km = top_km if top_km is not None else DEFAULT_XSEC_TOP_KM
    halfwidth_km = halfwidth_km if halfwidth_km is not None else DEFAULT_HALFWIDTH_KM

    frames = list_frame_refs(run_dir, domain=domain)
    if not frames:
        return None
    time0, path0, ti = frames[-1]
    with netCDF4.Dataset(path0) as ds:
        ter = wrf_mod.getvar(ds, "ter", timeidx=ti)
        hgt = np.asarray(ter)
        lats2d, lons2d = wrf_mod.latlon_coords(ter, as_np=True)

    point_marker = None
    if line:
        start_ll = (line["lat0"], line["lon0"])
        end_ll = (line["lat1"], line["lon1"])
    else:
        lat, lon = point
        # due east/west through `point` -- decision 0057 update gives no direction for the
        # moisture cross-section (unlike terrain's mean-wind line), so this keeps the
        # previous version's west-east convention, now bounded to +/-halfwidth_km instead of
        # the full domain width. Clipped to this domain's own bbox (2026-09-29 third lead
        # review, same fix as terrain.py's default line) so a point near the edge never
        # produces a line that runs outside the domain it is drawn on.
        # Containment is tested in GRID-INDEX space (grid_inside_predicate), not against the
        # lat/lon bbox: a Lambert grid's bbox admits endpoints whose index is == nx/ny, which
        # wrf.vertcross rejects ("end_point (99, 70) is outside of domain with shape (99, 99)").
        with netCDF4.Dataset(path0) as ds_g:
            inside = grid_inside_predicate(lambda la, lo: wrf_mod.ll_to_xy(ds_g, la, lo),
                                           hgt.shape[0], hgt.shape[1])
            clipped_halfwidth_km = clip_halfwidth_to_domain(lat, lon, 90.0, halfwidth_km, inside)
        start_ll, end_ll = line_from_point_bearing(lat, lon, 90.0, clipped_halfwidth_km)
        point_marker = (lat, lon)

    start = wrf_mod.CoordPair(lat=start_ll[0], lon=start_ll[1])
    end = wrf_mod.CoordPair(lat=end_ll[0], lon=end_ll[1])

    with netCDF4.Dataset(path0) as ds:
        z = wrf_mod.getvar(ds, "z", timeidx=ti, units="m")
        rh = wrf_mod.getvar(ds, "rh", timeidx=ti)
        theta_e = wrf_mod.getvar(ds, "theta_e", timeidx=ti, units="K")
        rh_cross = wrf_mod.vertcross(rh, z, wrfin=ds, start_point=start, end_point=end,
                                     latlon=True, missing=np.nan)
        thetae_cross = wrf_mod.vertcross(theta_e, z, wrfin=ds, start_point=start, end_point=end,
                                         missing=np.nan)
        ter_line = np.asarray(wrf_mod.interpline(wrf_mod.getvar(ds, "ter", timeidx=ti), wrfin=ds,
                                                 start_point=start, end_point=end)) / 1000.0
        from post.conventions import ds_time_label
        tlabel = ds_time_label(ds, time0)

    rh_np = np.asarray(rh_cross)
    thetae_np = np.asarray(thetae_cross)
    xy_loc = rh_cross.coords["xy_loc"].values
    x_km = centred_distance_km(xy_loc)
    y_km = np.asarray(rh_cross.coords["vertical"].values) / 1000.0

    keep = y_km <= top_km * 1.02
    if keep.sum() < 2:
        keep = np.ones_like(y_km, dtype=bool)
    y_km, rh_np, thetae_np = y_km[keep], rh_np[keep, :], thetae_np[keep, :]

    plt = _plt()
    # main axes | colour bar | locator inset, side by side so the inset never covers the bar
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_axes([0.07, 0.15, 0.60, 0.72])
    cax = fig.add_axes([0.685, 0.15, 0.015, 0.72])
    # RH every 10 % and theta-e every 5 K (stated, decision 0064); terrain as a grey silhouette
    cf = ax.contourf(x_km, y_km, rh_np, levels=np.arange(0, 110, 10), cmap="BuGn")
    cbar = fig.colorbar(cf, cax=cax)
    cbar.set_label("RH (%)")
    if np.isfinite(thetae_np).any():
        te_lv = np.arange(5.0 * np.floor(np.nanmin(thetae_np) / 5.0), np.nanmax(thetae_np) + 5.0, 5.0)
        cs = ax.contour(x_km, y_km, thetae_np, levels=te_lv, colors="k", linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%d")
    if len(ter_line) == len(x_km):
        ax.fill_between(x_km, 0, ter_line, color="0.45", zorder=5)

    ax.set_xlim(x_km.min(), x_km.max())
    ax.set_ylim(0, top_km)
    ax.set_xlabel("distance from centre (km, negative = west)")
    ax.set_ylabel("height (km)")
    ax.set_title("%s  d%02d  cross-section: RH (shaded, %%) + theta-e (K, every 5 K), terrain grey\n%s"
                 % (run_label, domain, tlabel), fontsize=10)
    fig.text(0.5, 0.02, "(%.2f, %.2f) -> (%.2f, %.2f)"
             % (start_ll[0], start_ll[1], end_ll[0], end_ll[1]), ha="center", fontsize=7)

    draw_locator_inset(fig, [0.78, 0.50, 0.20, 0.37], lons2d, lats2d, hgt, start_ll, end_ll,
                       point_ll=point_marker, cmap="terrain")

    out_path = os.path.join(out_dir, "moisture_crosssection_d%02d.png" % domain)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _point_time_height(run_dir, run_label, domain, out_dir, point, reason=None, top_m=12000.0):
    import netCDF4
    import numpy as np
    from datetime import datetime
    from post.timeseries import WRF_TIME_FMT
    wrf_mod = __import__("wrf")

    frames = list_frame_refs(run_dir, domain=domain)
    if len(frames) < 2:
        # a time-height diagram needs >=2 times; matplotlib's contourf refuses a
        # (n_levels, 1) array outright -- same fix as terrain.py's _valley_time_height
        # (caught on the scc T1 validation run, 2026-09-29).
        return None
    lat, lon = point
    with netCDF4.Dataset(frames[0][1]) as ds:
        xy = wrf_mod.ll_to_xy(ds, lat, lon)
        i, j = int(xy[0]), int(xy[1])

    times, q_profiles, z_profiles = [], [], []
    hgt = 0.0
    for t, path, ti in frames:
        with netCDF4.Dataset(path) as ds:
            q = np.asarray(ds.variables["QVAPOR"][ti, :, j, i]) * 1000.0  # kg/kg -> g/kg
            z = np.asarray(wrf_mod.getvar(ds, "z", timeidx=ti, units="m"))[:, j, i]
            hgt = float(ds.variables["HGT"][ti, j, i]) if "HGT" in ds.variables else 0.0
        times.append(datetime.strptime(t, WRF_TIME_FMT))
        q_profiles.append(q)
        z_profiles.append(z)

    n_levels = min(len(p) for p in q_profiles)
    q_arr = np.array([p[:n_levels] for p in q_profiles]).T
    # height above ground level (z is MSL), mean over time -- levels move by a few metres only
    z_agl = (np.mean([p[:n_levels] for p in z_profiles], axis=0) - hgt) / 1000.0

    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.08)   # room for the two-line title
    qmax = float(np.nanmax(q_arr))
    cf = ax.contourf(times, z_agl, q_arr, levels=MaxNLocator(nbins=16).tick_values(0.0, qmax),
                     cmap="BuPu")
    fig.colorbar(cf, ax=ax, label="specific humidity (g/kg)")
    cs = ax.contour(times, z_agl, q_arr, levels=MaxNLocator(nbins=8).tick_values(0.0, qmax),
                    colors="k", linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
    ax.set_ylim(0.0, top_m / 1000.0)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("valid time (day/hour UTC)")
    ax.set_ylabel("height AGL (km)")
    ax.set_title("%s  d%02d  specific humidity time-height at %.2f%s, %.2f%s\n%s"
                 % (run_label, domain, abs(lat), "N" if lat >= 0 else "S",
                    abs(lon), "E" if lon >= 0 else "W", reason or "user-given point"), fontsize=9, pad=8)
    out_path = os.path.join(out_dir, "moisture_timeheight_d%02d.png" % domain)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _finest_domain(domains):
    return domains[-1]["domain"]


def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6, cross_section=None,
           tc_centre=None, xsec_top_km=None, xsec_halfwidth_km=None):
    from post.recipe import fig, missing

    run_dir, run_label = run_dirs[0], labels[0]
    domain = _finest_domain(case_info["domains"])
    out = []

    try:
        path = _pw_figure(run_dir, run_label, domain, out_dir)
        if path:
            out.append(fig("moisture_pw", path, run_label, domain="d%02d" % domain))
        else:
            out.append(missing("moisture_pw", "wrfout_d%02d_*" % domain, "no frames found"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("moisture_pw", "wrf-python/cartopy", str(e)))

    try:
        record = _pw_change_figure(run_dir, run_label, domain, out_dir)
        out.append(record if record else missing("moisture_pw_chg6h", "wrfout_d%02d_*" % domain,
                                                 "fewer than 2 frames available (or no "
                                                 "earlier frame nearer 6 h than the last one)"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("moisture_pw_chg6h", "cdo/wrf-python", str(e)))

    try:
        path = _ivt_figure(run_dir, run_label, domain, out_dir)
        if path:
            out.append(fig("moisture_ivt", path, run_label, domain="d%02d" % domain))
        else:
            out.append(missing("moisture_ivt", "wrfout_d%02d_*" % domain, "no frames found"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("moisture_ivt", "wrf-python/cartopy", str(e)))

    try:
        point = _default_moisture_point(run_dir, domain, tc_centre=tc_centre)
        path = _cross_section_figure(run_dir, run_label, domain, out_dir, point,
                                     line=cross_section, top_km=xsec_top_km,
                                     halfwidth_km=xsec_halfwidth_km)
        if path:
            out.append(fig("moisture_crosssection", path, run_label, domain="d%02d" % domain))
        else:
            out.append(missing("moisture_crosssection", "wrfout_d%02d_*" % domain,
                               "no frames found"))
    except (tools.ToolNotFoundError, ImportError, ValueError) as e:
        out.append(missing("moisture_crosssection", "wrf-python", str(e)))
        point = None

    try:
        if point is None:
            point = _default_moisture_point(run_dir, domain, tc_centre=tc_centre)
        path = _point_time_height(run_dir, run_label, domain, out_dir, point,
                                  reason=_point_reason(tc_centre))
        if path:
            out.append(fig("moisture_timeheight", path, run_label, domain="d%02d" % domain))
        else:
            out.append(missing("moisture_timeheight", "wrfout_d%02d_*" % domain, "no frames found"))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("moisture_timeheight", "wrf-python", str(e)))

    return out
