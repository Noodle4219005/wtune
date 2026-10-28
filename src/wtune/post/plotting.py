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

from wtune.post.extract import _level_base_field, _surface_field, list_frame_refs, UNITS
from wtune.post import figdpi
from wtune.post.conventions import continuous_lons

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
    out = _surface_field(ds, wrf_mod, field, timeidx)
    if field == "t2":
        # maps show T2 in degC (decision 0064); `wtune post extract` keeps K via _surface_field
        attrs = dict(out.attrs)
        out = out - 273.15
        out.attrs = attrs
        out.attrs["units"] = "degC"
    return out


# map colour-bar units (extract.UNITS stays the data-file contract)
MAP_UNITS = dict(UNITS, t2="degC")
FIELD_TITLES = dict(slp="sea-level pressure (isobars every 4 hPa)", t2="2 m temperature", td2="2 m dew point",
                    rh2="2 m relative humidity", wspd10="10 m wind speed",
                    rain_acc="accumulated precipitation since init", pw="precipitable water",
                    mcape="MCAPE")


def _natural_earth_ready(resolution="110m"):
    """True if Natural Earth's coastline shapefile is cached locally. Only looks on disk:
    shapereader.natural_earth downloads a missing file, and a compute node without internet
    hung in it (f1-arm job 14882, 12 min); bin/fetch-natural-earth.sh fetches on a login node."""
    from wtune.post import basemap
    return basemap.shapefile_path("physical", "coastline", resolution) is not None


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


def _coastline(ax, ds, ccrs_mod, natural_earth_ok=None, draw_labels=False, **kw):
    """The shared base map (post/basemap.py: Natural Earth coastline/lakes/borders/states,
    terrain contours, labelled lat/lon gridlines, nest outline, caption), kept under its old
    name because every map builder calls it. ccrs_mod=None means plain lon/lat axes
    (--basemap landmask). natural_earth_ok/draw_labels are ignored: layers are detected from
    the local cache and cartopy axes always get gridline labels."""
    from wtune.post.basemap import add_base_map
    return add_base_map(ax, ccrs_mod, ds=ds, labels=True, **kw)


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
    frames = list_frame_refs(run_dir, domain=domain)
    if times:
        wanted = set(times)
        frames = [f for f in frames if f[0] in wanted]
    if not frames:
        raise PlotError("no wrfout_d%02d_* frames found in %s" % (domain, run_dir))

    written = []
    run_name = os.path.basename(os.path.normpath(run_dir))
    for time_str, path, ti in frames:
        with netcdf4.Dataset(path) as ds:
            var = _field_2d(ds, wrf_mod, field, level_hpa, timeidx=ti)
            data = np.asarray(var)

            tkw = {}
            if basemap == "cartopy":
                crs = wrf_mod.get_cartopy(var=var)
                lats, lons = wrf_mod.latlon_coords(var, as_np=True)
                lons = continuous_lons(lons)
                fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": crs})
                tkw = {"transform": ccrs.PlateCarree()}
            else:
                lats = np.asarray(ds.variables["XLAT"][0])
                lons = continuous_lons(ds.variables["XLONG"][0])
                fig, ax = plt.subplots(figsize=(8, 6))
            if field == "slp" and level_hpa is None:
                # surface chart convention (decision 0064): black isobars every 4 hPa with H/L
                # (none over terrain > 500 m, where SLP is a reduction), not a colour fill
                from wtune.post.recipes.mapfig import highs_lows
                from wtune.post.recipes.mapfig import slp_isobar_field
                iso = np.arange(4.0 * np.floor(np.nanmin(data) / 4.0), np.nanmax(data) + 4.0, 4.0)
                hgt = np.asarray(ds.variables["HGT"][0]) if "HGT" in ds.variables else None
                cs = ax.contour(lons, lats, slp_isobar_field(data, hgt), levels=iso, colors="k",
                                linewidths=0.8, **tkw)
                ax.clabel(cs, inline=True, fontsize=7, fmt="%d")
                for j, i, kind in highs_lows(data, exclude=None if hgt is None else hgt > 500.0):
                    ax.text(lons[j, i], lats[j, i], kind, color="b" if kind == "H" else "r",
                            fontsize=13, fontweight="bold", ha="center", va="center", **tkw)
            else:
                norm = None
                if levels is not None and cmap is not None and not isinstance(cmap, str):
                    from matplotlib.colors import BoundaryNorm     # one colour per bin
                    norm = BoundaryNorm(levels, cmap.N, extend=extend
                                        if cmap.N > len(levels) - 1 else "neither")
                cf = ax.contourf(lons, lats, data, levels=20 if levels is None else levels,
                                 cmap=cmap or "viridis", norm=norm, extend=extend, **tkw)
                cbar = fig.colorbar(cf, ax=ax)
                cbar.set_label(MAP_UNITS.get(field, ""))
            _coastline(ax, ds, ccrs, muted=field == "slp" and level_hpa is None)

            if overlay_wind:
                from wtune.post.basemap import bdy_width_of, interior_mask, masked_barbs
                lat2d = np.asarray(ds.variables["XLAT"][0])
                lon2d = continuous_lons(ds.variables["XLONG"][0])
                u = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=ti, units="m s-1")[0])
                v = np.asarray(wrf_mod.getvar(ds, "uvmet10", timeidx=ti, units="m s-1")[1])
                barb_kwargs = {"transform": ccrs.PlateCarree()} if basemap == "cartopy" else {}
                from wtune.post.conventions import KT_PER_MS
                masked_barbs(ax, lon2d, lat2d, u * KT_PER_MS, v * KT_PER_MS, WIND_SUBSAMPLE,
                             mask=interior_mask(data.shape, bdy_width_of(ds), fill=data),
                             length=5, **barb_kwargs)

            from wtune.post.conventions import ds_time_label
            level_note = (" @ %g hPa" % level_hpa) if level_hpa is not None else ""
            wind_note = " + 10 m wind (kt)" if overlay_wind else ""
            ax.set_title("%s  d%02d  %s%s%s\n%s" % (run_name, domain, FIELD_TITLES.get(field, field),
                                                    level_note, wind_note,
                                                    ds_time_label(ds, time_str)), fontsize=10)
            if basemap != "cartopy":
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")

            lev_tag = ("_%dhPa" % int(level_hpa)) if level_hpa is not None else ""
            out_path = os.path.join(out_dir, "%s_d%02d_%s%s.%s"
                                    % (field, domain, time_str.replace(":", ""), lev_tag, fmt))
            from wtune.post.basemap import caption_rect
            fig.tight_layout(rect=caption_rect(fig))
            fig.savefig(out_path, dpi=figdpi.dpi(), bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            written.append(out_path)
    return written
