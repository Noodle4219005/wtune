"""post/plotting.py -- `wtune post plot RUN_DIR --field NAME --out DIR`: filled-contour
maps, drawn the way the wrf-python plotting docs do it: wrf.get_cartopy(var) for the WRF
map projection, wrf.latlon_coords(var) for the coordinates, cartopy Natural Earth features
for coastlines/borders (decision 0056, 2026-09-29 update: wrap established tools).
matplotlib/cartopy are imported lazily, inside functions only. Reuses post.extract's
wrf-python field lookups so plot and extract compute the same field the same way.

Coastline: `--basemap cartopy` (default) draws Natural Earth coastlines/borders when the
shapefiles are available locally (WTUNE_CARTOPY_DATA, bin/fetch-natural-earth.sh) -- if
they are not (no cache and no internet), it falls back to a LANDMASK=0.5 contour on the
same cartopy-projected axes and prints that it did; this is the only fallback that happens
silently-by-design, since it is explicitly required, never a cartopy import failure.
`--basemap landmask` skips cartopy/wrf.get_cartopy entirely and plots on plain lon/lat axes.
"""
from __future__ import annotations

import os

from post.extract import _level_base_field, _surface_field, list_frames, UNITS

WIND_SUBSAMPLE = 8      # every Nth grid point in each direction for the wind overlay


class PlotError(RuntimeError):
    pass


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _field_2d(ds, wrf_mod, field, level_hpa, timeidx=0):
    """Returns the wrf.getvar()-style xarray result (NOT ndarray) -- wrf.get_cartopy/
    latlon_coords need its projection/coordinate metadata."""
    if level_hpa is not None:
        base = _level_base_field(ds, wrf_mod, field, timeidx)
        pressure = wrf_mod.getvar(ds, "pressure", timeidx=timeidx)
        return wrf_mod.interplevel(base, pressure, level_hpa)
    return _surface_field(ds, wrf_mod, field, timeidx)


def _natural_earth_ready(resolution="110m"):
    """True if Natural Earth's coastline shapefile is already cached locally -- checking
    this way (shapereader.natural_earth) never triggers a network call when it IS cached,
    and only raises (caught here) when it is not and cartopy tries to download it."""
    import cartopy.io.shapereader as shpreader
    try:
        path = shpreader.natural_earth(resolution=resolution, category="physical",
                                       name="coastline")
        return os.path.exists(path)
    except Exception:
        return False


def _set_cartopy_data_dir():
    data_dir = os.environ.get("WTUNE_CARTOPY_DATA")
    if data_dir:
        import cartopy
        cartopy.config["data_dir"] = data_dir
    return data_dir


def _gridlines(ax, draw_labels):
    """Lat/lon gridlines; with draw_labels the degree labels go on the left/bottom edges only
    (cartopy's own gridliner labels, works on the WRF Lambert/Mercator projections)."""
    gl = ax.gridlines(draw_labels=draw_labels, linewidth=0.3, alpha=0.5)
    if draw_labels:
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}
    return gl


def _coastline(ax, ds, ccrs_mod, natural_earth_ok, draw_labels=False):
    """ccrs_mod=None means plain lon/lat axes (--basemap landmask): no transform kwarg.
    draw_labels: also label the lat/lon gridlines (cartopy axes only)."""
    import numpy as np
    if natural_earth_ok:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        _gridlines(ax, draw_labels)
        return
    if ccrs_mod is not None and draw_labels:
        _gridlines(ax, True)
    if ds is None or "LANDMASK" not in ds.variables:
        return
    lat2d = np.asarray(ds.variables["XLAT"][0])
    lon2d = np.asarray(ds.variables["XLONG"][0])
    landmask = np.asarray(ds.variables["LANDMASK"][0])
    kwargs = {"transform": ccrs_mod.PlateCarree()} if ccrs_mod is not None else {}
    ax.contour(lon2d, lat2d, landmask, levels=[0.5], colors="k", linewidths=0.6, **kwargs)


def plot_field(run_dir, field, out_dir, domain=1, times=None, level_hpa=None,
               overlay_wind=False, basemap="cartopy", fmt="png", levels=None, cmap=None,
               extend="neither", gridline_labels=False):
    """Writes one figure per requested time to out_dir. Returns the list of written paths.
    levels/cmap/extend: a FIXED colour scale (explicit contour levels) so frames of one time
    series share one colour bar; default is per-frame auto levels, viridis. gridline_labels:
    label the lat/lon gridlines (cartopy basemap only)."""
    import numpy as np
    wrf_mod = __import__("wrf")
    netcdf4 = __import__("netCDF4")
    plt = _matplotlib()

    if basemap not in ("cartopy", "landmask"):
        raise PlotError("--basemap must be 'cartopy' or 'landmask', got %r" % basemap)

    ccrs = None
    natural_earth_ok = False
    if basemap == "cartopy":
        try:
            import cartopy.crs as ccrs
        except ImportError as e:
            raise PlotError("--basemap cartopy requested but cartopy is not importable "
                            "(pixi env may not have it installed): %s" % e)
        data_dir = _set_cartopy_data_dir()
        natural_earth_ok = _natural_earth_ready()
        if not natural_earth_ok:
            print("post plot: Natural Earth coastline data not found (WTUNE_CARTOPY_DATA=%s) "
                 "-- falling back to LANDMASK contour. Run bin/fetch-natural-earth.sh and "
                 "export WTUNE_CARTOPY_DATA to fix this." % (data_dir or "<unset>"))

    os.makedirs(out_dir, exist_ok=True)
    frames = list_frames(run_dir, domain=domain)
    if times:
        wanted = set(times)
        frames = [(t, p) for t, p in frames if t in wanted]
    if not frames:
        raise PlotError("no wrfout_d%02d_* frames found in %s" % (domain, run_dir))

    written = []
    run_name = os.path.basename(os.path.normpath(run_dir))
    for time_str, path in frames:
        with netcdf4.Dataset(path) as ds:
            var = _field_2d(ds, wrf_mod, field, level_hpa)
            data = np.asarray(var)

            if basemap == "cartopy":
                crs = wrf_mod.get_cartopy(var=var)
                lats, lons = wrf_mod.latlon_coords(var, as_np=True)
                fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": crs})
                cf = ax.contourf(lons, lats, data, levels=20 if levels is None else levels,
                                 cmap=cmap or "viridis", extend=extend,
                                 transform=ccrs.PlateCarree())
            else:
                lat2d = np.asarray(ds.variables["XLAT"][0])
                lon2d = np.asarray(ds.variables["XLONG"][0])
                fig, ax = plt.subplots(figsize=(8, 6))
                cf = ax.contourf(lon2d, lat2d, data, levels=20 if levels is None else levels,
                                 cmap=cmap or "viridis", extend=extend)

            cbar = fig.colorbar(cf, ax=ax)
            cbar.set_label(UNITS.get(field, ""))
            _coastline(ax, ds, ccrs, natural_earth_ok, draw_labels=gridline_labels)

            if overlay_wind:
                lat2d = np.asarray(ds.variables["XLAT"][0])
                lon2d = np.asarray(ds.variables["XLONG"][0])
                u = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")[0])
                v = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=0, units="m s-1")[1])
                s = slice(None, None, WIND_SUBSAMPLE)
                barb_kwargs = {"transform": ccrs.PlateCarree()} if basemap == "cartopy" else {}
                ax.barbs(lon2d[s, s], lat2d[s, s], u[s, s], v[s, s], length=5, **barb_kwargs)

            level_note = (" @ %g hPa" % level_hpa) if level_hpa is not None else ""
            ax.set_title("%s  d%02d  %s  %s%s" % (run_name, domain, time_str, field, level_note))
            if basemap != "cartopy":
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")

            lev_tag = ("_%dhPa" % int(level_hpa)) if level_hpa is not None else ""
            out_path = os.path.join(out_dir, "%s_d%02d_%s%s.%s"
                                    % (field, domain, time_str.replace(":", ""), lev_tag, fmt))
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            written.append(out_path)
    return written
