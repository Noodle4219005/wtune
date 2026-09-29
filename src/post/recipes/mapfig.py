"""post/recipes/mapfig.py -- one cartopy map-figure writer shared by every recipe figure that
draws a lat/lon field (decision 0057 section 2's synoptic/convection/precip/chem/pbl maps).
Reuses post/plotting.py's own coastline/Natural-Earth-fallback logic (_coastline,
_natural_earth_ready, _set_cartopy_data_dir, WIND_SUBSAMPLE) so every map in this package falls
back the same way `wtune post plot` does (LANDMASK contour when the Natural Earth cache is
missing, never a silent cartopy import failure) instead of each recipe module re-deriving it.
The field values themselves always come from wrf-python getvar/interplevel -- this module only
draws what it is given.
"""
from __future__ import annotations

from post.plotting import (WIND_SUBSAMPLE, _coastline, _matplotlib, _natural_earth_ready,
                           _set_cartopy_data_dir)


def save_map(out_path, lats, lons, title, fill_field=None, fill_label="", fill_cmap="viridis",
            contour_field=None, contour_levels=None, contour_color="k", contour_label_fmt="%g",
            u=None, v=None, ds=None, crs=None, basemap="cartopy"):
    """Writes one PNG. lats/lons: plain 2-D ndarrays. fill_field: optional filled contourf.
    contour_field/contour_levels: optional overlaid contour lines (e.g. geopotential height on
    a vorticity fill). u/v: optional wind barbs, subsampled by WIND_SUBSAMPLE. ds: netCDF4
    Dataset, only used for the LANDMASK coastline fallback. crs: a cartopy CRS (e.g. from
    wrf.get_cartopy(var=...)) for the WRF-native map projection; default PlateCarree when
    basemap=='cartopy' and none is given. basemap: 'cartopy' (default) or 'landmask', matching
    `wtune post plot`'s own --basemap contract.
    """
    plt = _matplotlib()
    ccrs = None
    natural_earth_ok = False
    subplot_kw = {}
    transform_kw = {}
    if basemap == "cartopy":
        import cartopy.crs as ccrs
        _set_cartopy_data_dir()
        natural_earth_ok = _natural_earth_ready()
        proj = crs if crs is not None else ccrs.PlateCarree()
        subplot_kw = {"projection": proj}
        transform_kw = {"transform": ccrs.PlateCarree()}
    elif basemap != "landmask":
        raise ValueError("basemap must be 'cartopy' or 'landmask', got %r" % basemap)

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=subplot_kw)

    if fill_field is not None:
        cf = ax.contourf(lons, lats, fill_field, levels=20, cmap=fill_cmap, **transform_kw)
        fig.colorbar(cf, ax=ax, label=fill_label)
    if contour_field is not None:
        cs = ax.contour(lons, lats, contour_field, levels=contour_levels, colors=contour_color,
                        linewidths=0.8, **transform_kw)
        ax.clabel(cs, inline=True, fontsize=7, fmt=contour_label_fmt)
    if u is not None and v is not None:
        s = slice(None, None, WIND_SUBSAMPLE)
        ax.barbs(lons[s, s], lats[s, s], u[s, s], v[s, s], length=5, **transform_kw)
    if ds is not None:
        _coastline(ax, ds, ccrs, natural_earth_ok)

    ax.set_title(title, fontsize=10)
    if basemap != "cartopy":
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
