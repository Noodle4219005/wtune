"""The domain/nest configuration map -- the Fig. 1 of almost every WRF paper (decision 0057;
WRF figure conventions in the wiki note how-wrf-papers-choose-their-figures-per-research-domain).

Drawn in the model's own projection with every domain as a box in projected coordinates, so a
d01 that crosses the date line (haitang2017: 221 x 126 at 45 km) stays a rectangle instead of a
ring around the globe, the extent is d01 itself, every domain has its own colour and a label
"dNN  dx km  nx x ny", and terrain is shaded when a grid file is at hand. Two sources:

- files (wrfout / wrfinput / geo_em): the grid each file holds (domains_from_files);
- namelist.wps alone (domains_from_namelist): before WPS has run, e.g. `wtune post preview` on
  a fresh case -- the same projection maths as wtune.geog's namelist bounds.
"""
from __future__ import annotations

import os
import re

from wtune.post import figdpi

EARTH_R = 6370000.0                 # WRF's sphere
DOMAIN_COLOURS = ("black", "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b")


def _nl_values(text, key):
    m = re.search(r"^\s*%s\s*=\s*([^\n/&]+)" % re.escape(key), text, re.M | re.I)
    if not m:
        return []
    return [v.strip().strip("'\"") for v in m.group(1).split(",") if v.strip()]


def namelist_crs(text):
    """Cartopy CRS of a namelist.wps &geogrid (lambert, mercator, polar), WRF's sphere."""
    import cartopy.crs as ccrs
    g = lambda k, cast=str: [cast(v) for v in _nl_values(text, k)]  # noqa: E731
    proj = (g("map_proj") or ["lambert"])[0].lower()
    ref_lat, ref_lon = g("ref_lat", float)[0], g("ref_lon", float)[0]
    t1 = (g("truelat1", float) or [ref_lat])[0]
    t2 = (g("truelat2", float) or [t1])[0]
    sl = (g("stand_lon", float) or [ref_lon])[0]
    globe = ccrs.Globe(ellipse=None, semimajor_axis=EARTH_R, semiminor_axis=EARTH_R)
    if proj == "lambert":
        return ccrs.LambertConformal(central_longitude=sl, central_latitude=ref_lat,
                                     standard_parallels=(t1, t2), globe=globe)
    if proj == "mercator":
        return ccrs.Mercator(central_longitude=sl, latitude_true_scale=t1, globe=globe)
    if proj == "polar":
        return ccrs.Stereographic(central_latitude=90 if ref_lat >= 0 else -90,
                                  central_longitude=sl, true_scale_latitude=t1, globe=globe)
    raise ValueError("map_proj %r is not drawn from namelist.wps (lambert, mercator, polar)" % proj)


def domains_from_namelist(nlw_path):
    """(crs, [dict(dom, x0, y0, x1, y1, dx_km, nx, ny)]) from namelist.wps, boxes in the
    projection's metres. d01 is centred on ref_lat/ref_lon (WPS's default ref_x/ref_y)."""
    import cartopy.crs as ccrs
    text = open(nlw_path).read()
    crs = namelist_crs(text)
    g = lambda k, cast=str: [cast(v) for v in _nl_values(text, k)]  # noqa: E731
    dx = g("dx", float)[0]
    dy = (g("dy", float) or [dx])[0]
    we, sn = g("e_we", int), g("e_sn", int)
    n = int((g("max_dom", int) or [len(we)])[0])
    ratio = g("parent_grid_ratio", int) or [1] * n
    ips, jps = g("i_parent_start", int) or [1] * n, g("j_parent_start", int) or [1] * n
    parent = g("parent_id", int) or [1] * n
    cx, cy = crs.transform_point(g("ref_lon", float)[0], g("ref_lat", float)[0], ccrs.PlateCarree())
    org = {1: (cx - (we[0] - 1) * dx / 2.0, cy - (sn[0] - 1) * dy / 2.0, dx, dy)}
    for d in range(2, n + 1):
        px, py, pdx, pdy = org[parent[d - 1]]
        org[d] = (px + (ips[d - 1] - 1) * pdx, py + (jps[d - 1] - 1) * pdy,
                  pdx / ratio[d - 1], pdy / ratio[d - 1])
    doms = []
    for d in range(1, n + 1):
        x0, y0, ddx, ddy = org[d]
        doms.append(dict(dom=d, x0=x0, y0=y0, x1=x0 + (we[d - 1] - 1) * ddx,
                         y1=y0 + (sn[d - 1] - 1) * ddy, dx_km=ddx / 1000.0,
                         nx=we[d - 1], ny=sn[d - 1]))
    return crs, doms


def _grid(ds):
    """(lat, lon, terrain or None) 2-D arrays of a wrfout / wrfinput / geo_em dataset."""
    import numpy as np
    for la, lo, h in (("XLAT", "XLONG", "HGT"), ("XLAT_M", "XLONG_M", "HGT_M")):
        if la in ds.variables:
            lat, lon = np.asarray(ds.variables[la][0]), np.asarray(ds.variables[lo][0])
            hgt = np.asarray(ds.variables[h][0]) if h in ds.variables else None
            return lat, lon, hgt
    raise ValueError("no XLAT/XLONG or XLAT_M/XLONG_M")


def domains_from_files(paths):
    """(crs, doms, terrain) from {dom: wrfout|wrfinput|geo_em path}: the CRS of d01's file
    (wrf-python's get_cartopy), each domain's box as the min/max of its grid in projected
    metres, terrain = (lat, lon, hgt) of d01."""
    import netCDF4
    import numpy as np
    import cartopy.crs as ccrs
    wrf_mod = __import__("wrf")
    doms, terrain, crs = [], None, None
    for d in sorted(paths):
        with netCDF4.Dataset(paths[d]) as ds:
            if crs is None:
                crs = (wrf_mod.get_cartopy(wrfin=ds) if "XLAT" in ds.variables
                       else namelist_crs(_geo_em_namelist_text(ds)))
            lat, lon, hgt = _grid(ds)
            pts = crs.transform_points(ccrs.PlateCarree(), lon.ravel(), lat.ravel())
            dx_km = float(getattr(ds, "DX", 0.0)) / 1000.0
            doms.append(dict(dom=d, x0=float(np.nanmin(pts[:, 0])), x1=float(np.nanmax(pts[:, 0])),
                             y0=float(np.nanmin(pts[:, 1])), y1=float(np.nanmax(pts[:, 1])),
                             dx_km=dx_km, nx=lat.shape[1] + 1, ny=lat.shape[0] + 1))
            if d == min(paths) and hgt is not None:
                terrain = (lat, lon, hgt)
    return crs, doms, terrain


def _geo_em_namelist_text(ds):
    """A minimal &geogrid text from a geo_em's global attributes (MAP_PROJ, CEN_LAT, ...)."""
    proj = {1: "lambert", 2: "polar", 3: "mercator"}.get(int(getattr(ds, "MAP_PROJ", 1)), "lambert")
    return ("map_proj = '%s'\nref_lat = %f\nref_lon = %f\ntruelat1 = %f\ntruelat2 = %f\n"
            "stand_lon = %f\n" % (proj, ds.MOAD_CEN_LAT, ds.CEN_LON, ds.TRUELAT1, ds.TRUELAT2,
                                  ds.STAND_LON))


def draw(crs, doms, out_path, title, terrain=None, caption=None):
    """The map: d01's box (+3 %) as the extent, terrain shaded (if given), coastlines, each
    domain's box in its own colour with a 'dNN  dx km  nx x ny' label. Returns out_path."""
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle
    from wtune.post.plotting import _matplotlib, _set_cartopy_data_dir
    plt = _matplotlib()
    _set_cartopy_data_dir()
    d1 = doms[0]
    w, h = d1["x1"] - d1["x0"], d1["y1"] - d1["y0"]
    aspect = h / w if w else 1.0
    fig = plt.figure(figsize=(8.0, max(4.0, min(9.0, 8.0 * aspect + 1.2))))
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_extent([d1["x0"] - 0.03 * w, d1["x1"] + 0.03 * w,
                   d1["y0"] - 0.03 * h, d1["y1"] + 0.03 * h], crs=crs)
    if terrain is not None:
        lat, lon, hgt = terrain
        top = max(500.0, float(np.nanmax(hgt)))
        levels = [0, 1, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
        levels = [lv for lv in levels if lv < top] + [top]
        cf = ax.contourf(lon, lat, hgt, levels=levels, cmap="terrain", extend="max",
                         transform=ccrs.PlateCarree())
        fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02, label="terrain height (m)")
    from wtune.post.basemap import add_base_map
    lat_span = (float(np.nanmax(terrain[0])) - float(np.nanmin(terrain[0]))
                if terrain is not None else h / 111e3)
    base = add_base_map(ax, ccrs, lats=None if terrain is None else terrain[0],
                        lons=None if terrain is None else terrain[1], labels=True,
                        nest_rings=[], terrain=False, caption=False)
    from wtune.post.basemap import gridline_step
    gridline_step(base.get("gridliner"), lat_span)
    for i, d in enumerate(doms):
        colour = DOMAIN_COLOURS[i % len(DOMAIN_COLOURS)]
        ax.add_patch(Rectangle((d["x0"], d["y0"]), d["x1"] - d["x0"], d["y1"] - d["y0"],
                               fill=False, edgecolor=colour, linewidth=2.0 if i else 2.4,
                               transform=crs, zorder=5))
        label = "d%02d  %g km  %d×%d" % (d["dom"], round(d["dx_km"], 3), d["nx"], d["ny"])
        ax.text(d["x0"] + 0.01 * w, d["y1"] - 0.01 * h, label, transform=crs, ha="left",
                va="top", fontsize=9, color=colour, fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))
    ax.set_title(title, fontsize=11)
    if caption:
        fig.text(0.01, 0.005, caption, fontsize=7, ha="left", va="bottom", wrap=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=figdpi.dpi(), bbox_inches="tight")
    plt.close(fig)
    return out_path
