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


def nice_levels(lo, hi, nbins=14):
    """Round contour levels spanning [lo, hi] (MaxNLocator: steps 1/2/2.5/5/10 x 10^n)."""
    from matplotlib.ticker import MaxNLocator
    return MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)


def pooled_levels(arrays, lo_pct=2.0, hi_pct=98.0, nbins=14, floor=None):
    """ONE fixed scale for a whole time series: nice levels spanning the lo_pct..hi_pct
    percentiles of every frame pooled together (so frames are comparable and an animation
    does not flicker). floor: clamp the lower bound (e.g. 0 for non-negative fields)."""
    import numpy as np
    vals = np.concatenate([np.asarray(a, dtype="f8").ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if floor is not None:
        lo = max(lo, floor)
    if hi <= lo:
        hi = lo + 1.0
    return nice_levels(lo, hi, nbins)


def save_map(out_path, lats, lons, title, fill_field=None, fill_label="", fill_cmap="viridis",
            contour_field=None, contour_levels=None, contour_color="k", contour_label_fmt="%g",
            u=None, v=None, ds=None, crs=None, basemap="cartopy", fill_levels=None,
            fill_extend="neither", fill_colors=None, gridline_labels=False, hatch_mask=None,
            hatch_label="below ground", legend_handles=None, hgt=None, bdy_width=None):
    """Writes one PNG. lats/lons: plain 2-D ndarrays. fill_field: optional filled contourf.
    contour_field/contour_levels: optional overlaid contour lines (e.g. geopotential height on
    a vorticity fill). u/v: optional wind barbs, subsampled by WIND_SUBSAMPLE. ds: netCDF4
    Dataset, only used for the LANDMASK coastline fallback. crs: a cartopy CRS (e.g. from
    wrf.get_cartopy(var=...)) for the WRF-native map projection; default PlateCarree when
    basemap=='cartopy' and none is given. basemap: 'cartopy' (default) or 'landmask', matching
    `wtune post plot`'s own --basemap contract.
    fill_levels/fill_extend: explicit filled-contour levels (default: 20 auto levels) -- pass
    one fixed scale for every frame of a time series. fill_colors: one colour per level bin
    (overrides fill_cmap; needed for unevenly spaced levels). gridline_labels: label lat/lon
    gridlines. hatch_mask: bool array, True cells are drawn light-grey hatched (e.g. pressure
    level below ground) and a `hatch_label` legend entry is added; legend_handles: extra
    legend entries. Legends go below the axes, outside the data area.
    Every map gets the shared base map (post/basemap.add_base_map: coastline, lakes, borders,
    states, terrain contours from `hgt` or ds HGT, labelled gridlines, nest outline). Barbs are
    drawn only inside the plotted interior (bdy_width edge cells excluded, default from ds).
    """
    import numpy as np
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

    # constrained layout: title/colour bar/gridline labels/legend all get room (tight_layout
    # clipped long titles)
    fig, ax = plt.subplots(figsize=(8, 6.4), subplot_kw=subplot_kw, layout="constrained")

    if fill_field is not None:
        kw = dict(transform_kw)
        if fill_colors is not None:
            # one colour per bin (+ optionally one more for the extend="max" overflow arrow);
            # BoundaryNorm keeps colours per bin even when the levels are unevenly spaced
            from matplotlib.colors import BoundaryNorm, ListedColormap
            nb = len(fill_levels) - 1
            cm_ = ListedColormap(list(fill_colors[:nb]))
            if fill_extend in ("max", "both"):
                cm_.set_over(fill_colors[nb] if len(fill_colors) > nb else fill_colors[nb - 1])
            kw["cmap"] = cm_
            kw["norm"] = BoundaryNorm(fill_levels, cm_.N)
        else:
            kw["cmap"] = fill_cmap
        cf = ax.contourf(lons, lats, fill_field, levels=20 if fill_levels is None else fill_levels,
                         extend=fill_extend, **kw)
        fig.colorbar(cf, ax=ax, label=fill_label, shrink=0.9)
    handles = list(legend_handles or [])
    if hatch_mask is not None and np.any(hatch_mask):
        from matplotlib.patches import Patch
        ax.contourf(lons, lats, np.asarray(hatch_mask, dtype="f8"), levels=[0.5, 1.5],
                    colors=["0.85"], hatches=["////"], **transform_kw)
        handles.append(Patch(facecolor="0.85", edgecolor="0.4", hatch="////", label=hatch_label))
    if contour_field is not None:
        cs = ax.contour(lons, lats, contour_field, levels=contour_levels, colors=contour_color,
                        linewidths=0.8, **transform_kw)
        ax.clabel(cs, inline=True, fontsize=7, fmt=contour_label_fmt)
    if u is not None and v is not None:
        from post.basemap import bdy_width_of, interior_mask, masked_barbs
        bw = bdy_width if bdy_width is not None else (bdy_width_of(ds) if ds is not None else 0)
        masked_barbs(ax, lons, lats, u, v, WIND_SUBSAMPLE,
                     mask=interior_mask(np.shape(lats), bw, fill=fill_field), length=5,
                     **transform_kw)
    _coastline(ax, ds, ccrs, natural_earth_ok, lats=lats, lons=lons, hgt=hgt)
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8,
                   frameon=False)

    ax.set_title(title, fontsize=9)
    if basemap != "cartopy":
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
