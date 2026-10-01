"""post/recipes/synoptic.py -- the "synoptic" figure set (decision 0057 section 2): 500 hPa
geopotential height + absolute vorticity, SLP + 1000-500 hPa thickness, 850 hPa equivalent
potential temperature + wind, 250 hPa jet (wind speed), T2 + 10 m wind -- all wrf-python
getvar/interplevel (z, avo, theta_e, wspd_wdir, slp, T2, uvmet10) through
post/recipes/mapfig.py. The ERA5 500 hPa difference map and its MET Grid-Stat RMSE/bias are
produced only when post/recipes/era5.py can open the ARCO store (gcsfs/zarr); otherwise both are
reported as one missing-observation entry with that exact reason, per decision 0057 section 3.

1000-500 hPa thickness has no wrf-python getvar name of its own -- it is z(500hPa) - z(1000hPa),
two ordinary wrf.interplevel calls differenced in numpy; that arithmetic is the only
self-written piece in the model-only maps.

The ERA5 diff map's RMSE/bias come from MET Grid-Stat (post/met_gridstat.py: CNT line, ERA5 as
OBS on its 0.25 deg grid, bilinear regrid done by MET, masked to the model domain's interior).
ERA5 is read from <project>/input/era5/ when present (compute nodes have no internet; fetch it
on a login node with `python -m post.recipes.era5`), else from ARCO via gcsfs.
"""
from __future__ import annotations

import os

from post import met_gridstat, tools
from post.extract import list_frames


def _post_spinup_time(run_dir, domain, spinup_hours):
    from datetime import datetime
    from post.caseinfo import WRF_TIME_FMT
    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return None, []
    t0 = datetime.strptime(frames[0][0], WRF_TIME_FMT)
    post = [(t, p) for t, p in frames
           if (datetime.strptime(t, WRF_TIME_FMT) - t0).total_seconds() / 3600.0 >= spinup_hours]
    chosen = post[-1] if post else frames[-1]
    return chosen, frames


def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6):
    from post.recipe import fig, missing
    from post.recipes.mapfig import save_map

    run_dir, run_label = run_dirs[0], labels[0]
    domain = 1   # synoptic diagnostics are read on d01, the domain the extent test itself used
    chosen, frames = _post_spinup_time(run_dir, domain, spinup_hours)
    out = []
    if chosen is None:
        return [missing("syn_maps", "wrfout_d01_*", "no frames found")]
    time_str, path = chosen

    try:
        import netCDF4
        import numpy as np
        wrf_mod = __import__("wrf")
        with netCDF4.Dataset(path) as ds:
            from post.conventions import ds_time_label
            tlabel = ds_time_label(ds, time_str)     # init / valid / lead (decision 0064)
            pressure = wrf_mod.getvar(ds, "pressure", timeidx=0)
            z = wrf_mod.getvar(ds, "z", timeidx=0, units="m")
            z500 = wrf_mod.interplevel(z, pressure, 500)
            # 1000 hPa is below ground wherever SLP < 1000 hPa (the whole hurricane core):
            # plain interplevel returns missing there, so the thickness contours vanished.
            # wrf.vinterp(extrapolate=True) is wrf-python's standard below-ground
            # extrapolation (standard-atmosphere lapse rate) for a height field.
            z_pl = wrf_mod.vinterp(ds, z, "pressure", [1000.0, 500.0], extrapolate=True,
                                   field_type="z", timeidx=0, log_p=True)
            z1000_ext = np.asarray(z_pl[0])
            z500_ext = np.asarray(z_pl[1])
            # avo needs MAPFAC_U/V/M and F; long climate runs often drop them via iofields
            try:
                avo500 = wrf_mod.interplevel(wrf_mod.getvar(ds, "avo", timeidx=0), pressure, 500)
            except KeyError as e:
                avo500 = None
                avo_missing = str(e)
            lats, lons = wrf_mod.latlon_coords(z500, as_np=True)
            crs = wrf_mod.get_cartopy(var=z500)
            xlim = wrf_mod.cartopy_xlim(var=z500)
            ylim = wrf_mod.cartopy_ylim(var=z500)

            # classic 500 hPa chart (decision 0064): heights every 60 m (6 dam), absolute
            # vorticity shaded from 16e-5 s-1 every 4e-5 s-1 (weaker values unshaded)
            from matplotlib import colormaps
            avo_np = np.asarray(avo500, dtype="f8")
            avo_levels = np.arange(16.0, 52.0, 4.0)
            cm = colormaps["YlOrRd"]
            avo_colors = [cm(x) for x in np.linspace(0.12, 0.95, len(avo_levels))]
            z500_np = np.asarray(z500, dtype="f8")
            z_levels = np.arange(60.0 * np.floor(np.nanmin(z500_np) / 60.0),
                                 np.nanmax(z500_np) + 60.0, 60.0)

            if avo500 is None:
                out.append(missing("syn_z500_vort", "wrfout MAPFAC_U/V/M, F",
                                   "absolute vorticity needs map-factor fields not in this wrfout "
                                   "(%s) -- re-run with them in the history stream" % avo_missing))
            else:
                p1 = os.path.join(out_dir, "syn_z500_vort.png")
                save_map(p1, lats, lons,
                        "%s d01 500 hPa height (every 60 m) + absolute vorticity (shaded)  %s"
                        % (run_label, tlabel),
                        fill_field=avo_np, fill_label="absolute vorticity (10$^{-5}$ s$^{-1}$)",
                        fill_levels=avo_levels, fill_colors=avo_colors, fill_extend="max",
                        contour_field=z500_np, contour_levels=z_levels, contour_label_fmt="%d",
                        ds=ds, crs=crs, gridline_labels=True, terrain=False)
                out.append(fig("syn_z500_vort", p1, run_label, domain="d01", valid_time=time_str))

            slp = np.asarray(wrf_mod.getvar(ds, "slp", timeidx=0, units="hPa"), dtype="f8")
            thickness_dam = (z500_ext - z1000_ext) / 10.0
            # classic surface chart (decision 0064): isobars every 4 hPa with H/L, 1000-500 hPa
            # thickness dashed every 6 dam, blue <= 540 dam, red above, 540 bold
            slp_levels = np.arange(4.0 * np.floor(np.nanmin(slp) / 4.0), np.nanmax(slp) + 4.0, 4.0)
            th_lo = 6.0 * np.floor(np.nanmin(thickness_dam) / 6.0)
            th_levels = np.arange(th_lo, np.nanmax(thickness_dam) + 6.0, 6.0)
            cold = th_levels[(th_levels < 540.0)]
            warm = th_levels[(th_levels > 540.0)]
            th_sets = []
            if len(cold):
                th_sets.append(dict(field=thickness_dam, levels=cold, colors="tab:blue",
                                    linestyles="dashed", linewidths=0.8, fmt="%d"))
            if len(warm):
                th_sets.append(dict(field=thickness_dam, levels=warm, colors="tab:red",
                                    linestyles="dashed", linewidths=0.8, fmt="%d"))
            if th_levels.min() <= 540.0 <= th_levels.max():
                th_sets.append(dict(field=thickness_dam, levels=[540.0], colors="tab:blue",
                                    linestyles="dashed", linewidths=2.0, fmt="%d"))
            # one legend entry describing what is actually drawn
            th_sets[-1]["label"] = ("1000-500 hPa thickness every 6 dam (dashed; blue <= 540, "
                                    "red > 540%s)" % (", 540 bold" if th_levels.min() <= 540.0
                                                      <= th_levels.max() else ""))
            p2 = os.path.join(out_dir, "syn_slp_thickness.png")
            save_map(p2, lats, lons,
                    "%s d01 SLP (isobars every 4 hPa) + 1000-500 hPa thickness (dashed, dam)  %s\n"
                    "1000 hPa height extrapolated below ground (wrf.vinterp extrapolate=True)"
                    % (run_label, tlabel),
                    contour_field=slp, contour_levels=slp_levels, contour_label_fmt="%d",
                    contours=th_sets, hl_field=slp,
                    # SLP over high terrain is a reduction to sea level, not an analysed centre
                    hl_exclude=np.asarray(ds.variables["HGT"][0]) > 500.0,
                    ds=ds, crs=crs, gridline_labels=True,
                    terrain=False)
            out.append(fig("syn_slp_thickness", p2, run_label, domain="d01", valid_time=time_str,
                           extra={"note": "1000 hPa height extrapolated below ground "
                                          "(wrf.vinterp extrapolate=True)"}))

            theta_e = wrf_mod.getvar(ds, "theta_e", timeidx=0)
            theta_e850 = wrf_mod.interplevel(theta_e, pressure, 850)
            u850 = np.asarray(wrf_mod.interplevel(
                wrf_mod.getvar(ds, "ua", timeidx=0, units="m s-1"), pressure, 850))
            v850 = np.asarray(wrf_mod.interplevel(
                wrf_mod.getvar(ds, "va", timeidx=0, units="m s-1"), pressure, 850))
            # 850 hPa is below ground where the surface pressure is < 850 hPa
            below850 = np.asarray(ds.variables["PSFC"][0]) / 100.0 < 850.0
            p3 = os.path.join(out_dir, "syn_850_thetae_wind.png")
            te = np.asarray(theta_e850, dtype="f8")
            te_levels = np.arange(4.0 * np.floor(np.nanmin(te) / 4.0), np.nanmax(te) + 4.0, 4.0)
            save_map(p3, lats, lons,
                    "%s d01 850 hPa theta-e (K, every 4 K) + wind (kt)  %s" % (run_label, tlabel),
                    fill_field=te, fill_label="theta-e (K)", fill_cmap="magma",
                    fill_levels=te_levels, fill_extend="both",
                    u=np.ma.masked_where(below850, u850), v=np.ma.masked_where(below850, v850),
                    barb_factor=KT_PER_MS, barb_label="wind barbs in kt",
                    ds=ds, crs=crs, gridline_labels=True, hatch_mask=below850,
                    hatch_label="850 hPa below ground")
            out.append(fig("syn_850_thetae_wind", p3, run_label, domain="d01", valid_time=time_str))

            wspd250 = np.asarray(wrf_mod.interplevel(
                wrf_mod.getvar(ds, "wspd_wdir", timeidx=0, units="kt")[0], pressure, 250),
                dtype="f8")
            z250 = np.asarray(wrf_mod.interplevel(z, pressure, 250), dtype="f8")
            # jet chart (decision 0064): isotachs shaded in kt from 70 kt (40 kt when the jet
            # is weaker) every 10 kt, 250 hPa heights every 120 m
            jet_lo = 70.0 if np.nanmax(wspd250) >= 80.0 else 40.0
            jet_levels = np.arange(jet_lo, max(jet_lo + 20.0, np.nanmax(wspd250)) + 10.0, 10.0)
            z250_levels = np.arange(120.0 * np.floor(np.nanmin(z250) / 120.0),
                                    np.nanmax(z250) + 120.0, 120.0)
            p4 = os.path.join(out_dir, "syn_250_jet.png")
            save_map(p4, lats, lons,
                    "%s d01 250 hPa isotachs (kt, from %d kt) + height (every 120 m)  %s"
                    % (run_label, jet_lo, tlabel),
                    fill_field=wspd250, fill_label="wind speed (kt)", fill_cmap="PuBu",
                    fill_levels=jet_levels, fill_extend="max",
                    contour_field=z250, contour_levels=z250_levels, contour_label_fmt="%d",
                    ds=ds, crs=crs, gridline_labels=True, terrain=False)
            out.append(fig("syn_250_jet", p4, run_label, domain="d01", valid_time=time_str))

            t2c = np.asarray(wrf_mod.getvar(ds, "T2", timeidx=0), dtype="f8") - 273.15
            u10 = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")[0])
            v10 = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")[1])
            # surface chart (decision 0064): T2 in degC on a diverging scale centred on 0 degC
            # with the 0 degC isotherm bold, barbs in kt
            t2_levels = celsius_levels(t2c)
            p5 = os.path.join(out_dir, "syn_t2_wind10.png")
            save_map(p5, lats, lons,
                    "%s d01 T2 (degC, 0 degC bold) + 10 m wind (kt)  %s" % (run_label, tlabel),
                    fill_field=t2c, fill_label="T2 (degC)", fill_cmap="RdBu_r",
                    fill_levels=t2_levels, fill_extend="both",
                    contours=[dict(field=t2c, levels=[0.0], colors="k", linewidths=1.6,
                                   label="0 degC isotherm")],
                    u=u10, v=v10, barb_factor=KT_PER_MS, barb_label="wind barbs in kt",
                    ds=ds, crs=crs, gridline_labels=True)
            out.append(fig("syn_t2_wind10", p5, run_label, domain="d01", valid_time=time_str))

            model_bbox = (float(np.asarray(ds.variables["XLAT"][0]).min()),
                         float(np.asarray(ds.variables["XLAT"][0]).max()),
                         float(np.asarray(ds.variables["XLONG"][0]).min()),
                         float(np.asarray(ds.variables["XLONG"][0]).max()))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("syn_maps", "wrf-python/cartopy", str(e)))
        return out

    try:
        from post.recipes.era5 import Era5UnavailableError, z500_for
        from post.caseinfo import WRF_TIME_FMT, spec_bdy_width
        from datetime import datetime
        try:
            from wtune_project import find_project
            project_dir = find_project(out_dir)
        except Exception:
            project_dir = None
        e_lat, e_lon, e_z, e_valid, e_src = z500_for(
            datetime.strptime(time_str, WRF_TIME_FMT), model_bbox, project_dir)
        gs = met_gridstat.grid_stat_z500(
            path, e_lat, e_lon, e_z, e_valid, os.path.join(out_dir, "gridstat"),
            spec_bdy_width(run_dir))
        p6 = os.path.join(out_dir, "syn_era5_diff.png")
        from post.conventions import path_time_label
        _era5_diff_map(p6, run_label, path_time_label(path, time_str), lats, lons, np.asarray(z500), e_lat, e_lon, e_z,
                       gs, crs, xlim, ylim)
        out.append(fig("syn_era5_diff", p6, run_label, domain="d01", valid_time=time_str,
                       extra={"rmse_m": gs["rmse_m"], "bias_m": gs["bias_m"], "n": gs["n"],
                              "stat_file": gs["stat_file"], "era5_source": e_src}))
    except Era5UnavailableError as e:
        out.append(missing("syn_era5_diff", "ERA5 (ARCO)", str(e)))
    except (tools.ToolNotFoundError, tools.ToolError, met_gridstat.METError) as e:
        out.append(missing("syn_era5_diff", "MET grid_stat", str(e)))
    return out


KT_PER_MS = 1.943844


def celsius_levels(t_c, step=None):
    """Temperature levels in degC symmetric about 0 degC (diverging map centred on freezing),
    covering the field's 1st-99th percentile; step 2/3/5 degC by range (decision 0064)."""
    import numpy as np
    finite = np.asarray(t_c, dtype="f8")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.arange(-20.0, 22.0, 2.0)
    lo, hi = np.percentile(finite, [1, 99])
    vmax = max(abs(lo), abs(hi), 2.0)
    step = step or (2.0 if vmax <= 20 else 3.0 if vmax <= 36 else 5.0)
    vmax = step * np.ceil(vmax / step)
    return np.arange(-vmax, vmax + step, step)


def symmetric_levels(vmax, nbins=12):
    """Contour levels symmetric about 0 with NO boundary at 0, so the bin straddling 0 is one
    band (white in RdBu_r; same construction as terrain.py's w shading)."""
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    levels = MaxNLocator(nbins=nbins, symmetric=True).tick_values(-vmax, vmax)
    return levels[np.abs(levels) > 1e-12]


def _era5_diff_map(out_path, run_label, time_str, lats, lons, z500, e_lat, e_lon, e_z, gs, crs,
                   xlim, ylim):
    """Grid-Stat DIFF (model - ERA5, matched-pair NetCDF) filled on the model domain's own
    projection, extent = the model domain; model z500 solid black / ERA5 z500 dashed every
    60 m so a displaced trough shows as a dipole in the fill AND as offset contour lines."""
    import netCDF4
    import numpy as np
    import cartopy.crs as ccrs
    from post.plotting import (_coastline, _matplotlib, _natural_earth_ready,
                               _set_cartopy_data_dir)
    plt = _matplotlib()
    with netCDF4.Dataset(gs["pairs_nc"]) as ds:
        name = next(v for v in ds.variables if v.startswith("DIFF_"))
        d_lat = np.asarray(ds.variables["lat"][:])
        d_lon = np.asarray(ds.variables["lon"][:])
        diff = np.ma.filled(ds.variables[name][:], np.nan).astype("f8")
    _set_cartopy_data_dir()
    pc = ccrs.PlateCarree()
    fig_, ax = plt.subplots(figsize=(8.5, 6.5), subplot_kw={"projection": crs})
    ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=crs)
    finite = diff[np.isfinite(diff)]
    vmax = max(float(np.percentile(np.abs(finite), 99)), 1e-3) if finite.size else 1.0
    levels = symmetric_levels(vmax)
    cmap = plt.get_cmap("RdBu_r")
    ax.set_facecolor(cmap(0.5))
    cf = ax.contourf(d_lon, d_lat, np.ma.masked_invalid(diff), levels=levels, cmap=cmap,
                     extend="both", transform=pc)
    fig_.colorbar(cf, ax=ax, ticks=levels, label="model - ERA5 z500 (m)", shrink=0.85)
    lo = 60.0 * np.floor(min(np.nanmin(z500), np.nanmin(e_z)) / 60.0)
    hi = 60.0 * np.ceil(max(np.nanmax(z500), np.nanmax(e_z)) / 60.0)
    clev = np.arange(lo, hi + 60.0, 60.0)
    cm = ax.contour(lons, lats, z500, levels=clev, colors="k", linewidths=0.9,
                    linestyles="solid", transform=pc)
    ce = ax.contour(e_lon, e_lat, e_z, levels=clev, colors="k", linewidths=0.9,
                    linestyles="dashed", transform=pc)
    ax.clabel(cm, inline=True, fontsize=7, fmt="%d")
    ax.clabel(ce, inline=True, fontsize=7, fmt="%d")
    if _natural_earth_ready():
        # shared base map; no terrain contours on top of two sets of z500 contours
        _coastline(ax, None, ccrs, True, lats=lats, lons=lons, terrain=False, nest_rings=[])
    else:
        print("syn_era5_diff: Natural Earth data not found (WTUNE_CARTOPY_DATA=%s): no coastline"
              % (os.environ.get("WTUNE_CARTOPY_DATA") or "<unset>"))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="k", lw=0.9, label="model z500 (60 m)"),
                       Line2D([], [], color="k", lw=0.9, ls="--", label="ERA5 z500 (60 m)")],
              loc="lower left", fontsize=7, framealpha=0.85)
    ax.set_title("%s d01 500 hPa height, model - ERA5 (m)  %s\nRMSE=%.1f m  bias=%.1f m  "
                 "(MET Grid-Stat, ERA5 0.25\u00b0, interior)"
                 % (run_label, time_str, gs["rmse_m"], gs["bias_m"]), fontsize=9)
    fig_.tight_layout()
    fig_.savefig(out_path, dpi=130)
    plt.close(fig_)
    return out_path
