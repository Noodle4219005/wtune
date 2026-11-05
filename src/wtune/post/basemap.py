"""post/basemap.py -- the ONE base-map layer drawn on every plan-view map (user request
2026-10-01: "overlay the original terrain or the region boundaries under the figures, otherwise I
can't tell which areas the colour blocks apply to").

add_base_map(ax, ...) draws, above the filled field:
  * Natural Earth coastline, lake shorelines, country borders, state/province borders (50 m, or
    10 m when it is cached and the domain is small) as thin dark lines -- read from the local
    cartopy data dir (WTUNE_CARTOPY_DATA, bin/fetch-natural-earth.sh; the same cache
    post/xwrf_convert._write_coast uses) and drawn as plain lon/lat LineCollections, so it works
    on any projection and never triggers a download;
  * terrain contours (HGT of the same domain), light grey, interval adapted to the relief, a few
    labelled;
  * lat/lon gridlines with tick labels outside the frame only (cartopy axes; outer_gridlines);
  * the outline of every nested child domain as a dashed box (derived from the open wrfout);
  * a one-line caption (figure bottom) naming the layers, source and licence.
Without any Natural Earth coastline file it falls back to the LANDMASK=0.5 contour and says so in
the caption. masked_barbs() draws wind barbs only inside a validity mask (plotted interior).
"""
from __future__ import annotations

import os

# (category, NE name, label, edge colour, linewidth)
LAYERS = (
    ("physical", "coastline", "coastline", "0.05", 0.7),
    ("physical", "lakes", "lake shorelines", "0.15", 0.5),
    ("cultural", "admin_0_boundary_lines_land", "country borders", "0.2", 0.6),
    ("cultural", "admin_1_states_provinces_lakes", "state/province borders", "0.35", 0.4),
)
MUTED = "0.6"                # every layer's colour with muted=True
FINE_SPAN_DEG = 8.0          # use 10 m below this domain span when cached
TERRAIN_STEPS_M = (50, 100, 200, 250, 500, 1000, 2000)
TERRAIN_MAX_LINES = 12
LICENCE = "Natural Earth, public domain"

_cache = {}


def _data_dirs():
    dirs = []
    try:
        import cartopy
        for key in ("data_dir", "pre_existing_data_dir"):
            d = cartopy.config.get(key)
            if d:
                dirs.append(d)
    except ImportError:
        pass
    env = os.environ.get("WTUNE_CARTOPY_DATA")
    if env:
        dirs.insert(0, env)
    return dirs


def shapefile_path(category, name, resolution):
    """Path of the locally cached Natural Earth shapefile, or None (never downloads)."""
    for d in _data_dirs():
        p = os.path.join(d, "shapefiles", "natural_earth", category,
                         "ne_%s_%s.shp" % (resolution, name))
        if os.path.isfile(p):
            return p
    return None


def _all_segments(path):
    """[(lon0, lat0, lon1, lat1, ndarray Nx2)] of every line/ring in the shapefile, cached."""
    if path not in _cache:
        import numpy as np
        from cartopy.io import shapereader
        out = []
        for g in shapereader.Reader(path).geometries():
            for part in getattr(g, "geoms", [g]):
                rings = [part.exterior, *part.interiors] if hasattr(part, "exterior") else [part]
                for r in rings:
                    xy = np.asarray(r.coords)[:, :2]
                    xy = xy[np.abs(xy[:, 1]) <= 85.0]       # Mercator-safe
                    if len(xy) >= 2:
                        out.append((xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max(), xy))
        _cache[path] = out
    return _cache[path]


def layer_segments(category, name, resolution, bbox):
    """Line segments (list of Nx2 lon/lat arrays) of one Natural Earth layer that touch bbox
    (lon0, lon1, lat0, lat1); None when the file is not cached locally."""
    path = shapefile_path(category, name, resolution)
    if path is None:
        return None
    import numpy as np
    lon0, lon1, lat0, lat1 = bbox
    out = []
    # a box past 180E (continuous_lons of a domain across it: 165..190) also takes the
    # segments 360 deg away, shifted onto it
    for shift in [0.0] + [360.0] * (lon1 > 180.0) + [-360.0] * (lon0 < -180.0):
        a0, a1 = lon0 - shift, lon1 - shift
        for s in _all_segments(path):
            if s[2] >= a0 and s[0] <= a1 and s[3] >= lat0 and s[1] <= lat1:
                out.append(s[4] + np.array([shift, 0.0]) if shift else s[4])
    return out


def centred_plate_carree(ccrs, lon0, lon1):
    """(PlateCarree centred on lon0..lon1, its central longitude). With central_longitude 0 a
    box across 180E (170..190) is two boxes at the map's ends."""
    c = ((lon0 + lon1) / 2.0 + 180.0) % 360.0 - 180.0
    return ccrs.PlateCarree(central_longitude=c), c


def set_lonlat_extent(ax, extent, c):
    """set_extent of (lon0, lon1, lat0, lat1), lon0..lon1 continuous (may pass 180), on an axes
    from centred_plate_carree(..., c): in the axes' own x (lon - c), so nothing is cut at 180."""
    lon0, lon1, lat0, lat1 = extent
    x0 = (lon0 - c + 180.0) % 360.0 - 180.0
    ax.set_extent([x0, x0 + (lon1 - lon0), lat0, lat1], crs=ax.projection)


def terrain_interval(hgt_max, hgt_min=0.0):
    """Contour interval (m) from the relief: the smallest step in TERRAIN_STEPS_M that gives at
    most TERRAIN_MAX_LINES lines; None when the domain is flat (top < 100 m, or relief < 300 m:
    contours of a few tens of metres over a plain are clutter, not orientation)."""
    rng = float(hgt_max) - max(float(hgt_min), 0.0)
    if hgt_max < 100.0 or rng < 300.0:
        return None
    for step in TERRAIN_STEPS_M:
        if rng / step <= TERRAIN_MAX_LINES:
            return step
    return TERRAIN_STEPS_M[-1]


def domain_edge_ring(ds):
    """(lons, lats) of the grid-edge ring of an open wrfout Dataset."""
    import numpy as np
    lat = np.asarray(ds.variables["XLAT"][0])
    lon = np.asarray(ds.variables["XLONG"][0])
    top = list(zip(lon[0, :], lat[0, :]))
    right = list(zip(lon[:, -1], lat[:, -1]))
    bottom = list(zip(lon[-1, ::-1], lat[-1, ::-1]))
    left = list(zip(lon[::-1, 0], lat[::-1, 0]))
    ring = top + right + bottom + left + top[:1]
    return [p[0] for p in ring], [p[1] for p in ring]


def child_rings(ds):
    """[(domain number, lons, lats)] of every domain nested inside the open wrfout `ds`
    (found as wrfout_dNN_* next to it, NN > this GRID_ID)."""
    try:
        import netCDF4
        from wtune.post.extract import list_frames
        gid = int(ds.getncattr("GRID_ID"))
        run_dir = os.path.dirname(os.path.abspath(ds.filepath()))
        out = []
        for dom in range(gid + 1, gid + 6):
            frames = list_frames(run_dir, domain=dom)
            if not frames:
                break
            with netCDF4.Dataset(frames[0][1]) as dds:
                lo, la = domain_edge_ring(dds)
            out.append((dom, lo, la))
        return out
    except Exception:
        return []


def _bbox(lats, lons, pad=1.0):
    import numpy as np
    ok = np.isfinite(lats) & np.isfinite(lons)
    if not ok.any():
        return None
    return (float(lons[ok].min()) - pad, float(lons[ok].max()) + pad,
            float(lats[ok].min()) - pad, float(lats[ok].max()) + pad)


def outer_gridlines(ax, left=True, bottom=True, **line_kw):
    """Lat/lon gridlines whose labels sit OUTSIDE the map frame only: longitude along the bottom
    edge, latitude along the left edge, never inline (x/y_inline=False: on a Lambert-conformal
    WRF map cartopy otherwise writes a row of longitude labels across the interior), never
    top/right, never rotated. left/bottom=False drop that edge (inner panels of a multi-panel
    figure)."""
    sides = {}
    if bottom:
        sides["bottom"] = "x"
    if left:
        sides["left"] = "y"
    kw = dict(linewidth=0.3, alpha=0.5, color="0.4")
    kw.update(line_kw)
    gl = ax.gridlines(draw_labels=sides or False, x_inline=False, y_inline=False,
                      rotate_labels=False, **kw)
    gl.top_labels = gl.right_labels = False
    gl.left_labels = "y" if left else False
    gl.bottom_labels = "x" if bottom else False
    gl.xlabel_style = {"size": 7, "rotation": 0}
    gl.ylabel_style = {"size": 7, "rotation": 0}
    return gl


def _gridlines(ax, ccrs_mod):
    return outer_gridlines(ax)


CAPTION_FONTSIZE = 5.5


def caption_rect(fig):
    """Layout rect (left, bottom, right, top) keeping a strip at the figure bottom free for the
    base-map caption, so it never overprints the bottom-edge longitude labels. Pass it to
    fig.tight_layout(rect=...); constrained-layout figures get it from add_base_map itself."""
    return (0.0, 2.2 * CAPTION_FONTSIZE / 72.0 / fig.get_figheight(), 1.0, 1.0)


def add_base_map(ax, ccrs_mod=None, ds=None, lats=None, lons=None, hgt=None, labels=True,
                 nest_rings=None, terrain=True, caption=True, muted=False):
    """Draw the base layers on `ax`. ccrs_mod: cartopy.crs module (None = plain lon/lat axes).
    ds: open wrfout Dataset of THIS domain (gives XLAT/XLONG/HGT/LANDMASK and the nest rings);
    alternatively pass lats/lons (+ hgt) arrays. nest_rings: explicit [(dom, lons, lats)] (None =
    derive from ds; [] = none). terrain=False skips the HGT contours (the field already IS
    terrain). muted: grey coastlines and borders, for a map whose own black lines (isobars)
    must not be mistaken for them. Returns dict(layers=[labels drawn], resolution,
    terrain_interval_m, gridliner, caption)."""
    import numpy as np
    from matplotlib.collections import LineCollection
    if ds is not None:
        if lats is None and "XLAT" in ds.variables:
            lats = np.asarray(ds.variables["XLAT"][0])
            lons = np.asarray(ds.variables["XLONG"][0])
        if hgt is None and "HGT" in ds.variables:
            ds_hgt = np.asarray(ds.variables["HGT"][0])
            # a field regridded off the model grid (e.g. ERA5 differences) has other lat/lon
            # shapes: model terrain cannot be contoured on them
            if ds_hgt.shape == np.shape(lats):
                hgt = ds_hgt
    if hgt is not None and lats is not None and np.shape(hgt) != np.shape(lats):
        hgt = None
    lats = None if lats is None else np.asarray(lats, dtype="f8")
    lons = None if lons is None else np.asarray(lons, dtype="f8")
    transform_kw = {"transform": ccrs_mod.PlateCarree()} if ccrs_mod is not None else {}
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    box = _bbox(lats, lons) if lats is not None else None
    if box is None:
        try:
            ext = ax.get_extent(crs=ccrs_mod.PlateCarree())
            box = (ext[0] - 1, ext[1] + 1, ext[2] - 1, ext[3] + 1)
        except Exception:
            box = (-180.0, 180.0, -85.0, 85.0)
    span = max(box[1] - box[0], box[3] - box[2])
    resolutions = (("10m", "50m", "110m") if span < FINE_SPAN_DEG else ("50m", "110m"))

    drawn = []
    used_res = None
    for category, name, label, color, lw in LAYERS:
        for res in resolutions:
            segs = layer_segments(category, name, res, box)
            if segs is not None:
                break
        if segs is None:
            continue
        if segs:
            ax.add_collection(LineCollection(segs, colors=MUTED if muted else color,
                                             linewidths=lw, zorder=3, **transform_kw),
                              autolim=False)
        drawn.append(label)
        used_res = used_res or res
    landmask_fallback = False
    if "coastline" not in drawn and ds is not None and "LANDMASK" in ds.variables \
            and lats is not None:
        landmask = np.asarray(ds.variables["LANDMASK"][0])
        # lats/lons may be a cropped window of the file's grid; the mask is the whole grid
        if landmask.shape == np.shape(lats) == np.shape(lons):
            ax.contour(lons, lats, landmask, levels=[0.5],
                       colors=MUTED if muted else "k", linewidths=0.6, zorder=3, **transform_kw)
            landmask_fallback = True

    interval = None
    if terrain and hgt is not None and lats is not None:
        hgt = np.asarray(hgt, dtype="f8")
        if np.isfinite(hgt).any():
            interval = terrain_interval(np.nanmax(hgt), np.nanmin(hgt))
        if interval:
            levels = np.arange(interval, np.nanmax(hgt) + interval, interval)
            if len(levels):
                cs = ax.contour(lons, lats, hgt, levels=levels, colors="0.55", linewidths=0.35,
                                zorder=2.5, **transform_kw)
                try:
                    ax.clabel(cs, levels=levels[::3], fontsize=5, fmt="%d", inline=True)
                except Exception:
                    pass

    rings = nest_rings if nest_rings is not None else (child_rings(ds) if ds is not None else [])
    for dom, ring_lon, ring_lat in rings:
        ax.plot(ring_lon, ring_lat, linestyle="--", color="k", linewidth=1.1, zorder=4,
                **transform_kw)
    gl = _gridlines(ax, ccrs_mod) if (ccrs_mod is not None and labels) else None
    if ccrs_mod is None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    parts = []
    if drawn:
        parts.append("%s %s: %s (%s)" % ("Natural Earth", used_res, ", ".join(drawn), LICENCE))
    elif landmask_fallback:
        parts.append("coastline = WRF LANDMASK 0.5 contour (Natural Earth data not cached)")
    else:
        parts.append("no coastline data available (Natural Earth not cached)")
    if interval:
        parts.append("grey lines: model terrain HGT every %d m" % interval)
    if rings:
        parts.append("dashed: %s outline" % "/".join("d%02d" % d for d, _, _ in rings))
    text = "Base map -- " + "; ".join(parts)
    if caption:
        fig = ax.figure
        fig.text(0.005, 0.003, text, fontsize=CAPTION_FONTSIZE, color="0.3", ha="left",
                 va="bottom")
        from matplotlib.layout_engine import ConstrainedLayoutEngine
        if isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine):
            # constrained-layout rect is (left, bottom, WIDTH, HEIGHT), unlike tight_layout's
            bottom = caption_rect(fig)[1]
            fig.get_layout_engine().set(rect=(0.0, bottom, 1.0, 1.0 - bottom))
    return dict(layers=drawn, resolution=used_res, terrain_interval_m=interval, gridliner=gl,
                caption=text, landmask_fallback=landmask_fallback, nest=[d for d, _, _ in rings])


def interior_mask(shape, bdy_width=0, fill=None):
    """Bool mask of the plotted interior: True away from the bdy_width edge cells and (when
    `fill` is given) where the filled field is finite."""
    import numpy as np
    m = np.ones(shape, dtype=bool)
    if bdy_width:
        m[:bdy_width, :] = False
        m[-bdy_width:, :] = False
        m[:, :bdy_width] = False
        m[:, -bdy_width:] = False
    if fill is not None:
        m &= np.isfinite(np.asarray(fill, dtype="f8"))
    return m


def gridline_step(gl, lat_span):
    """Lat/lon gridlines every 2, 5 or 10 degrees by the map's latitude span: cartopy's own
    choice on a Lambert map labelled one meridian (120E) on haitang2017's d01."""
    import matplotlib.ticker as mticker
    if gl is None:
        return
    step = 2 if lat_span < 12 else 5 if lat_span < 30 else 10
    gl.xlocator = mticker.FixedLocator(range(-180, 181, step))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, step))


BARBS_ACROSS = 15           # barbs along a map's shorter side


def barb_stride(shape, across=BARBS_ACROSS, minimum=4) -> int:
    """Grid stride giving about `across` barbs along the shorter side (haitang2017 d03's 25,
    with a second set on the same points, read as a solid fill)."""
    return max(minimum, min(shape) // across)


def above_ground(ds, level_hpa, timeidx=0):
    """True where the surface pressure is above level_hpa: wind interpolated to that level is
    not underground (850/700 hPa barbs drawn over the Central Mountain Range)."""
    import numpy as np
    return np.asarray(ds.variables["PSFC"][timeidx], dtype="f8") / 100.0 > level_hpa


def masked_barbs(ax, lons, lats, u, v, stride, mask=None, offset=0, **kw):
    """ax.barbs on every `stride`-th point (from `offset`, to interleave a second set), but
    ONLY where `mask` (plotted interior), u and v are valid -- no barb is ever drawn outside
    the filled interior. Returns the Barbs or None when nothing is left."""
    import numpy as np
    s = slice(offset, None, stride)
    lo, la = np.asarray(lons)[s, s], np.asarray(lats)[s, s]
    uu, vv = np.asarray(u, dtype="f8")[s, s], np.asarray(v, dtype="f8")[s, s]
    # wrf.interplevel marks below-ground points with a ~9.97e36 fill value, not NaN
    keep = np.isfinite(uu) & np.isfinite(vv) & (np.abs(uu) < 1e20) & (np.abs(vv) < 1e20)
    if mask is not None:
        keep &= np.asarray(mask)[s, s]
    if not keep.any():
        return None
    return ax.barbs(lo[keep], la[keep], uu[keep], vv[keep], **kw)


def bdy_width_of(ds, default=5):
    """spec_bdy_width of the run an open wrfout belongs to (namelist), else `default`."""
    try:
        from wtune.post.caseinfo import spec_bdy_width
        return spec_bdy_width(os.path.dirname(os.path.abspath(ds.filepath())))
    except Exception:
        return default
