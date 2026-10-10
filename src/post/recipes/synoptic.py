"""post/recipes/synoptic.py -- the "synoptic" figure set (decision 0057 section 2): 500 hPa height +
absolute vorticity, SLP + 1000-500 hPa thickness, 850 hPa theta-e + wind, 250 hPa jet, T2 + 10 m
wind, all wrf-python getvar/interplevel through post/recipes/mapfig.py. The ERA5 500 hPa
difference map (MET Grid-Stat RMSE/bias) is reported as one missing-observation entry when ERA5
is unavailable (decision 0057 section 3); ERA5 comes from <project>/input/era5/ when present
(compute nodes have no internet: `python -m post.recipes.era5` on a login node), else ARCO.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable

from post import met_gridstat, tools
from post.extract import list_frame_refs
from post.recipes.common import FIELD_ERRORS, error_reason

KT_PER_MS = 1.943844          # kt per m s-1
_SYNOPTIC_DOMAIN = 1          # read on d01, the domain the extent test used
_THICKNESS_DIVIDE = 540.0     # dam; rain/snow line of the classic surface chart
_HGT_HIGH_TERRAIN_M = 500.0   # no H/L centres where SLP is a reduction over terrain above this


def _post_spinup_time(run_dir: str, domain: int, spinup_hours: float) -> tuple:
    """(last frame at least spinup_hours after the first, else the last frame; all frames)."""
    from datetime import datetime
    from post.caseinfo import WRF_TIME_FMT
    frames = list_frame_refs(run_dir, domain=domain)      # [(time, path, timeidx)]
    if not frames:
        return None, []
    t0 = datetime.strptime(frames[0][0], WRF_TIME_FMT)
    post = [f for f in frames
           if (datetime.strptime(f[0], WRF_TIME_FMT) - t0).total_seconds() / 3600.0 >= spinup_hours]
    chosen = post[-1] if post else frames[-1]
    return chosen, frames


@dataclasses.dataclass
class _Chart:
    """What every synoptic chart needs: the open frame, its shared fields and map geometry."""
    wrf: Any
    ds: Any
    ti: int
    np: Any
    save_map: Callable
    out_dir: str
    run_label: str
    tlabel: str
    time_str: str
    pressure: Any = None
    z: Any = None
    z500: Any = None
    z1000_ext: Any = None
    z500_ext: Any = None
    avo500: Any = None
    avo_missing: str | None = None
    lats: Any = None
    lons: Any = None
    crs: Any = None
    xlim: Any = None
    ylim: Any = None


def _read_shared_fields(c: _Chart) -> None:
    """Pressure, height (500 hPa, plus 1000/500 hPa extrapolated below ground), 500 hPa vorticity
    and the map geometry, in wrf-python's call order."""
    wrf_mod, ds, ti, np = c.wrf, c.ds, c.ti, c.np
    c.pressure = wrf_mod.getvar(ds, "pressure", timeidx=ti)
    c.z = wrf_mod.getvar(ds, "z", timeidx=ti, units="m")
    c.z500 = wrf_mod.interplevel(c.z, c.pressure, 500)
    # 1000 hPa is below ground wherever SLP < 1000 hPa (a hurricane core): plain interplevel
    # leaves it missing, so extrapolate with wrf.vinterp (standard-atmosphere lapse rate).
    z_pl = wrf_mod.vinterp(ds, c.z, "pressure", [1000.0, 500.0], extrapolate=True,
                           field_type="z", timeidx=ti, log_p=True)
    c.z1000_ext = np.asarray(z_pl[0])
    c.z500_ext = np.asarray(z_pl[1])
    try:   # avo needs MAPFAC_U/V/M and F, which long climate runs often drop via iofields
        c.avo500 = wrf_mod.interplevel(wrf_mod.getvar(ds, "avo", timeidx=ti), c.pressure, 500)
    except KeyError as e:
        c.avo500 = None
        c.avo_missing = str(e)
    c.lats, c.lons = wrf_mod.latlon_coords(c.z500, as_np=True)
    c.crs = wrf_mod.get_cartopy(var=c.z500)
    c.xlim = wrf_mod.cartopy_xlim(var=c.z500)
    c.ylim = wrf_mod.cartopy_ylim(var=c.z500)


def cyclonic_avo(np: Any, avo: Any, lats: Any) -> tuple[Any, str]:
    """(absolute vorticity with the sign flipped south of the equator, label): cyclonic is
    positive in both hemispheres, so the fixed positive levels also shade a SH trough. A grid
    wholly north of the equator is returned unchanged with the plain label."""
    label = "absolute vorticity (10$^{-5}$ s$^{-1}$)"
    south = np.asarray(lats) < 0.0
    if not south.any():
        return avo, label
    return (np.where(south, -avo, avo),
            label[:-1] + "; sign flipped south of the equator: cyclonic +)")


def _z500_vort_chart(c: _Chart, fig: Callable, missing: Callable) -> dict:
    """Classic 500 hPa chart (decision 0064): heights every 60 m, absolute vorticity shaded from
    16e-5 s-1 every 4e-5 s-1."""
    from matplotlib import colormaps
    np = c.np
    avo_np = np.asarray(c.avo500, dtype="f8")
    avo_levels = np.arange(16.0, 52.0, 4.0)        # 1e-5 s-1
    cm = colormaps["YlOrRd"]
    avo_colors = [cm(x) for x in np.linspace(0.12, 0.95, len(avo_levels))]
    z500_np = np.asarray(c.z500, dtype="f8")
    z_levels = np.arange(60.0 * np.floor(np.nanmin(z500_np) / 60.0),
                         np.nanmax(z500_np) + 60.0, 60.0)
    if c.avo500 is None:
        return missing("syn_z500_vort", "wrfout MAPFAC_U/V/M, F",
                       "absolute vorticity needs map-factor fields not in this wrfout "
                       "(%s) -- re-run with them in the history stream" % c.avo_missing)
    avo_np, avo_label = cyclonic_avo(np, avo_np, c.lats)
    p1 = os.path.join(c.out_dir, "syn_z500_vort.png")
    c.save_map(p1, c.lats, c.lons,
               "%s d01 500 hPa height (every 60 m) + absolute vorticity (shaded)  %s"
               % (c.run_label, c.tlabel),
               fill_field=avo_np, fill_label=avo_label,
               fill_levels=avo_levels, fill_colors=avo_colors, fill_extend="max",
               contour_field=z500_np, contour_levels=z_levels, contour_label_fmt="%d",
               ds=c.ds, crs=c.crs, gridline_labels=True, terrain=False)
    return fig("syn_z500_vort", p1, c.run_label, domain="d01", valid_time=c.time_str)


def _thickness_contours(np: Any, thickness_dam: Any) -> list[dict]:
    """Thickness contour sets every 6 dam: blue <= 540, red above, 540 bold; the last set carries
    the one legend label, naming only the lines the data range draws."""
    th_lo = 6.0 * np.floor(np.nanmin(thickness_dam) / 6.0)
    th_levels = np.arange(th_lo, np.nanmax(thickness_dam) + 6.0, 6.0)
    cold = th_levels[(th_levels < _THICKNESS_DIVIDE)]
    warm = th_levels[(th_levels > _THICKNESS_DIVIDE)]
    th_sets = []
    if len(cold):
        th_sets.append(dict(field=thickness_dam, levels=cold, colors="tab:blue",
                            linestyles="dashed", linewidths=0.8, fmt="%d"))
    if len(warm):
        th_sets.append(dict(field=thickness_dam, levels=warm, colors="tab:red",
                            linestyles="dashed", linewidths=0.8, fmt="%d"))
    th_min, th_max = np.nanmin(thickness_dam), np.nanmax(thickness_dam)
    has_540 = th_min <= _THICKNESS_DIVIDE <= th_max
    if has_540:
        th_sets.append(dict(field=thickness_dam, levels=[_THICKNESS_DIVIDE], colors="tab:blue",
                            linestyles="dashed", linewidths=2.0, fmt="%d"))
    parts = []
    if th_min < _THICKNESS_DIVIDE:
        parts.append("blue <= 540")
    if th_max > _THICKNESS_DIVIDE:
        parts.append("red > 540")
    if has_540:
        parts.append("540 bold")
    th_sets[-1]["label"] = ("1000-500 hPa thickness every 6 dam (dashed; %s)" % ", ".join(parts))
    return th_sets


def _slp_thickness_chart(c: _Chart, fig: Callable) -> dict:
    """Classic surface chart (decision 0064): isobars every 4 hPa with H/L + thickness
    (z500 - z1000 differenced here; wrf-python has no thickness getvar)."""
    np = c.np
    slp = np.asarray(c.wrf.getvar(c.ds, "slp", timeidx=c.ti, units="hPa"), dtype="f8")
    thickness_dam = (c.z500_ext - c.z1000_ext) / 10.0
    slp_levels = np.arange(4.0 * np.floor(np.nanmin(slp) / 4.0), np.nanmax(slp) + 4.0, 4.0)
    th_sets = _thickness_contours(np, thickness_dam)
    p2 = os.path.join(c.out_dir, "syn_slp_thickness.png")
    c.save_map(p2, c.lats, c.lons,
               "%s d01 SLP (isobars every 4 hPa) + 1000-500 hPa thickness (dashed, dam)  %s\n"
               "1000 hPa height extrapolated below ground (wrf.vinterp extrapolate=True)"
               % (c.run_label, c.tlabel),
               contour_field=slp, contour_levels=slp_levels, contour_label_fmt="%d",
               contours=th_sets, hl_field=slp,
               hl_exclude=np.asarray(c.ds.variables["HGT"][0]) > _HGT_HIGH_TERRAIN_M,
               ds=c.ds, crs=c.crs, gridline_labels=True,
               terrain=False)
    return fig("syn_slp_thickness", p2, c.run_label, domain="d01", valid_time=c.time_str,
               extra={"note": "1000 hPa height extrapolated below ground "
                              "(wrf.vinterp extrapolate=True)"})


def _thetae850_chart(c: _Chart, fig: Callable) -> dict:
    """850 hPa theta-e every 4 K + wind barbs (kt), hatched where 850 hPa is below ground."""
    np, wrf_mod, ds, ti = c.np, c.wrf, c.ds, c.ti
    theta_e850 = wrf_mod.interplevel(wrf_mod.getvar(ds, "theta_e", timeidx=ti), c.pressure, 850)
    # earth-relative wind: save_map draws the barbs on lat/lon (ua/va are along the grid axes)
    uvmet = wrf_mod.getvar(ds, "uvmet", timeidx=ti, units="m s-1")
    u850 = np.asarray(wrf_mod.interplevel(uvmet[0], c.pressure, 850))
    v850 = np.asarray(wrf_mod.interplevel(uvmet[1], c.pressure, 850))
    below850 = np.asarray(ds.variables["PSFC"][ti]) / 100.0 < 850.0   # surface pressure, hPa
    p3 = os.path.join(c.out_dir, "syn_850_thetae_wind.png")
    te = np.asarray(theta_e850, dtype="f8")
    te_levels = np.arange(4.0 * np.floor(np.nanmin(te) / 4.0), np.nanmax(te) + 4.0, 4.0)
    c.save_map(p3, c.lats, c.lons,
               "%s d01 850 hPa theta-e (K, every 4 K) + wind (kt)  %s" % (c.run_label, c.tlabel),
               fill_field=te, fill_label="theta-e (K)", fill_cmap="magma",
               fill_levels=te_levels, fill_extend="both",
               u=np.ma.masked_where(below850, u850), v=np.ma.masked_where(below850, v850),
               barb_factor=KT_PER_MS, barb_label="wind barbs in kt",
               ds=ds, crs=c.crs, gridline_labels=True, hatch_mask=below850,
               hatch_label="850 hPa below ground")
    return fig("syn_850_thetae_wind", p3, c.run_label, domain="d01", valid_time=c.time_str)


def _jet250_chart(c: _Chart, fig: Callable) -> dict:
    """Jet chart (decision 0064): isotachs in kt from 70 kt (40 kt for a jet under 80 kt) every
    10 kt, 250 hPa heights every 120 m."""
    np, wrf_mod = c.np, c.wrf
    wspd250 = np.asarray(wrf_mod.interplevel(
        wrf_mod.getvar(c.ds, "wspd_wdir", timeidx=c.ti, units="kt")[0], c.pressure, 250),
        dtype="f8")
    z250 = np.asarray(wrf_mod.interplevel(c.z, c.pressure, 250), dtype="f8")
    jet_lo = 70.0 if np.nanmax(wspd250) >= 80.0 else 40.0
    jet_levels = np.arange(jet_lo, max(jet_lo + 20.0, np.nanmax(wspd250)) + 10.0, 10.0)
    z250_levels = np.arange(120.0 * np.floor(np.nanmin(z250) / 120.0),
                            np.nanmax(z250) + 120.0, 120.0)
    p4 = os.path.join(c.out_dir, "syn_250_jet.png")
    c.save_map(p4, c.lats, c.lons,
               "%s d01 250 hPa isotachs (kt, from %d kt) + height (every 120 m)  %s"
               % (c.run_label, jet_lo, c.tlabel),
               fill_field=wspd250, fill_label="wind speed (kt)", fill_cmap="PuBu",
               fill_levels=jet_levels, fill_extend="max",
               contour_field=z250, contour_levels=z250_levels, contour_label_fmt="%d",
               ds=c.ds, crs=c.crs, gridline_labels=True, terrain=False)
    return fig("syn_250_jet", p4, c.run_label, domain="d01", valid_time=c.time_str)


def _t2_wind10_chart(c: _Chart, fig: Callable) -> dict:
    """Surface chart (decision 0064): T2 (degC) diverging about 0 degC, 0 degC bold, barbs in kt."""
    np, wrf_mod, ds, ti = c.np, c.wrf, c.ds, c.ti
    t2c = np.asarray(wrf_mod.getvar(ds, "T2", timeidx=ti), dtype="f8") - 273.15
    u10 = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=ti, units="m s-1")[0])
    v10 = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=ti, units="m s-1")[1])
    t2_levels = celsius_levels(t2c)
    p5 = os.path.join(c.out_dir, "syn_t2_wind10.png")
    c.save_map(p5, c.lats, c.lons,
               "%s d01 T2 (degC, 0 degC bold) + 10 m wind (kt)  %s" % (c.run_label, c.tlabel),
               fill_field=t2c, fill_label="T2 (degC)", fill_cmap="RdBu_r",
               fill_levels=t2_levels, fill_extend="both",
               contours=[dict(field=t2c, levels=[0.0], colors="k", linewidths=1.6,
                              label="0 degC isotherm")],
               u=u10, v=v10, barb_factor=KT_PER_MS, barb_label="wind barbs in kt",
               ds=ds, crs=c.crs, gridline_labels=True)
    return fig("syn_t2_wind10", p5, c.run_label, domain="d01", valid_time=c.time_str)


def _model_bbox(np: Any, ds: Any) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) of the frame's XLAT/XLONG."""
    return (float(np.asarray(ds.variables["XLAT"][0]).min()),
            float(np.asarray(ds.variables["XLAT"][0]).max()),
            float(np.asarray(ds.variables["XLONG"][0]).min()),
            float(np.asarray(ds.variables["XLONG"][0]).max()))


def _era5_diff_entry(c: _Chart, run_dir: str, path: str, model_bbox: tuple, fig: Callable,
                     missing: Callable) -> dict:
    """ERA5 z500 difference map with MET Grid-Stat RMSE/bias (interior of the model domain), or
    the missing-observation entry naming why ERA5 or MET was unavailable."""
    np = c.np
    try:
        from post.recipes.era5 import Era5UnavailableError, z500_for
    except ImportError as e:
        return missing("syn_era5_diff", "ERA5 (ARCO)", str(e))
    try:
        from post.caseinfo import WRF_TIME_FMT, spec_bdy_width
        from datetime import datetime
        try:
            from wtune.project import find_project
            project_dir = find_project(c.out_dir)
        except Exception:
            project_dir = None
        e_lat, e_lon, e_z, e_valid, e_src = z500_for(
            datetime.strptime(c.time_str, WRF_TIME_FMT), model_bbox, project_dir)
        gs = met_gridstat.grid_stat_z500(
            path, e_lat, e_lon, e_z, e_valid, os.path.join(c.out_dir, "gridstat"),
            spec_bdy_width(run_dir), timeidx=c.ti)
        p6 = os.path.join(c.out_dir, "syn_era5_diff.png")
        from post.conventions import path_time_label
        _era5_diff_map(p6, c.run_label, path_time_label(path, c.time_str), c.lats, c.lons,
                       np.asarray(c.z500), e_lat, e_lon, e_z, gs, c.crs, c.xlim, c.ylim)
        return fig("syn_era5_diff", p6, c.run_label, domain="d01", valid_time=c.time_str,
                   extra={"rmse_m": gs["rmse_m"], "bias_m": gs["bias_m"], "n": gs["n"],
                          "stat_file": gs["stat_file"], "era5_source": e_src})
    except Era5UnavailableError as e:
        return missing("syn_era5_diff", "ERA5 (ARCO)", str(e))
    except (tools.ToolNotFoundError, tools.ToolError, met_gridstat.METError) as e:
        return missing("syn_era5_diff", "MET grid_stat", str(e))


def figures(run_dirs: list[str], labels: list[str], out_dir: str, case_info: dict,
            spinup_hours: float = 6) -> list[dict]:
    """The synoptic charts of the first run's last post-spin-up d01 frame -> figure/missing
    entries; wrf-python/cartopy missing -> one 'syn_maps' missing entry."""
    from post.recipe import fig, missing
    from post.recipes.mapfig import save_map

    run_dir, run_label = run_dirs[0], labels[0]
    chosen, frames = _post_spinup_time(run_dir, _SYNOPTIC_DOMAIN, spinup_hours)
    out = []
    if chosen is None:
        return [missing("syn_maps", "wrfout_d01_*", "no frames found")]
    time_str, path, ti = chosen

    try:
        import netCDF4
        import numpy as np
        wrf_mod = __import__("wrf")
        with netCDF4.Dataset(path) as ds:
            from post.conventions import ds_time_label
            c = _Chart(wrf=wrf_mod, ds=ds, ti=ti, np=np, save_map=save_map, out_dir=out_dir,
                       run_label=run_label, time_str=time_str,
                       tlabel=ds_time_label(ds, time_str))   # init / valid / lead (decision 0064)
            _read_shared_fields(c)
            for fid, chart, args in (("syn_z500_vort", _z500_vort_chart, (c, fig, missing)),
                                     ("syn_slp_thickness", _slp_thickness_chart, (c, fig)),
                                     ("syn_850_thetae_wind", _thetae850_chart, (c, fig)),
                                     ("syn_250_jet", _jet250_chart, (c, fig)),
                                     ("syn_t2_wind10", _t2_wind10_chart, (c, fig))):
                try:       # one chart's missing field is one missing entry, the others survive
                    out.append(chart(*args))
                except FIELD_ERRORS as e:
                    out.append(missing(fid, "wrfout_d01", error_reason(e)))
            model_bbox = _model_bbox(np, ds)
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("syn_maps", "wrf-python/cartopy", str(e)))
        return out

    out.append(_era5_diff_entry(c, run_dir, path, model_bbox, fig, missing))
    return out


def celsius_levels(t_c: Any, step: float | None = None) -> Any:
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


def symmetric_levels(vmax: float, nbins: int = 12) -> Any:
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
    from post.basemap import caption_rect
    fig_.tight_layout(rect=caption_rect(fig_))
    fig_.savefig(out_path, dpi=130)
    plt.close(fig_)
    return out_path
