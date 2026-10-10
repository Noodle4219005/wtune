"""post/recipes/xsection.py -- shared vertical-cross-section helpers for terrain.py and
moisture.py: geodesic line placement/distance (2026-09-29 lead review), and the standard
mountain-wave-style presentation both figures share (2026-09-29 second lead review, "redraw
in the standard mountain-wave cross-section style"). Uses cartopy.geodesic (WGS84 ellipsoid,
already a dependency of the cartopy-based map figures elsewhere in this package) -- an
established tool, not a hand-rolled haversine -- for every distance/bearing computation.
"""
from __future__ import annotations

DEFAULT_HALFWIDTH_KM = 150.0


def along_line_km(xy_loc):
    """xy_loc: wrf.vertcross()/interpline()'s own `xy_loc` coordinate values (pass
    latlon=True) -- an array of CoordPair(x, y, lat, lon) objects, one per along-line sample
    point. Returns a 1-D ndarray of cumulative geodesic distance in km, same length as
    xy_loc, starting at 0.0."""
    import numpy as np
    import cartopy.geodesic as cgeo

    lons = np.array([p.lon for p in xy_loc], dtype=float)
    lats = np.array([p.lat for p in xy_loc], dtype=float)
    if len(lons) < 2:
        return np.zeros(len(lons))
    geod = cgeo.Geodesic()
    pts = np.column_stack([lons[:-1], lats[:-1]])
    ends = np.column_stack([lons[1:], lats[1:]])
    seg_km = geod.inverse(pts, ends)[:, 0] / 1000.0
    return np.concatenate([[0.0], np.cumsum(seg_km)])


def centred_distance_km(xy_loc):
    """along_line_km(), re-centred so the line's own midpoint (the ridge crest / TC-centre
    point the line was built around, by construction the exact geometric midpoint of a
    symmetric halfwidth_km line) sits at 0 -- negative before it, positive after (2026-09-29
    lead review item 5: "x axis distance from ridge crest, negative = upstream")."""
    raw = along_line_km(xy_loc)
    return raw - raw[-1] / 2.0


def line_from_point_bearing(lat, lon, bearing_deg, halfwidth_km=DEFAULT_HALFWIDTH_KM):
    """(start_ll, end_ll), each an (lat, lon) tuple: a geodesic line centred on (lat, lon),
    extended halfwidth_km in each direction along bearing_deg (compass bearing, 0=N, 90=E)
    and its reciprocal -- shared by terrain.py's ridge-crest line (bearing = mean 700 hPa
    wind direction) and moisture.py's TC-centre/max-rain-point line (bearing = due east,
    matching its previous west-east convention). `end` is the downstream/forward side
    (positive distance), `start` is the upstream/reciprocal side (negative distance)."""
    import cartopy.geodesic as cgeo
    geod = cgeo.Geodesic()
    dist_m = halfwidth_km * 1000.0
    fwd = geod.direct(points=[[lon, lat]], azimuths=[bearing_deg], distances=[dist_m])[0]
    back = geod.direct(points=[[lon, lat]], azimuths=[(bearing_deg + 180.0) % 360.0],
                       distances=[dist_m])[0]
    end_ll = (float(fwd[1]), float(fwd[0]))
    start_ll = (float(back[1]), float(back[0]))
    return start_ll, end_ll


def bbox_predicate(bbox):
    """(lat, lon) -> bool closure over bbox=(lat_min, lat_max, lon_min, lon_max) -- the
    simple containment test for clip_halfwidth_to_domain() when no better one is available
    (e.g. moisture.py's default line, which was never reported to have the Lambert-grid
    mismatch problem index_predicate below exists for). A plain lat/lon rectangle is only an
    approximation of a curvilinear WRF domain's true footprint (2026-09-29 fourth lead
    review: "the lat/lon bbox of a Lambert grid is not the grid") -- adequate when the caller
    has no cheap way to ask the grid itself, but post/recipes/terrain.py uses the grid-index
    predicate below instead, precisely because it does have one."""
    lat_min, lat_max, lon_min, lon_max = bbox

    def _inside(lat, lon):
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

    return _inside


def clip_halfwidth_to_domain(lat, lon, bearing_deg, halfwidth_km, inside_fn):
    """Largest half-width <= halfwidth_km such that BOTH endpoints of
    line_from_point_bearing(lat, lon, bearing_deg, that half-width) satisfy
    inside_fn(lat, lon) -- so the cross-section line never draws a point outside the region
    inside_fn defines, and (by construction of line_from_point_bearing's symmetric
    +/-half-width) the centre point stays the exact midpoint of whatever line results
    (2026-09-29 third/fourth lead review: the default line was running past the edge of the
    domain -- and, once fixed with a lat/lon bbox, still past the true grid interior, since a
    Lambert-projected grid's bbox is not the same shape as the grid itself). inside_fn: a
    callable(lat, lon) -> bool; see bbox_predicate() for a simple rectangle, or build one from
    the grid's own nearest-index lookup (wrf.ll_to_xy) for the precise version. Returns 0.0 if
    (lat, lon) itself already fails inside_fn (should not normally happen: the centre point is
    always found from that domain's own HGT/rain field, already restricted to the same
    region)."""
    if not inside_fn(lat, lon):
        return 0.0
    start, end = line_from_point_bearing(lat, lon, bearing_deg, halfwidth_km)
    if inside_fn(*start) and inside_fn(*end):
        return halfwidth_km

    lo, hi = 0.0, halfwidth_km
    for _ in range(30):
        mid = (lo + hi) / 2.0
        s, e = line_from_point_bearing(lat, lon, bearing_deg, mid)
        if inside_fn(*s) and inside_fn(*e):
            lo = mid
        else:
            hi = mid
    return lo


def line_unit_vector(start_ll, end_ll):
    """(di, dj): east/north unit-vector components of the direction from start_ll to end_ll,
    via a flat-earth approximation at the line's mean latitude -- adequate for projecting
    wind onto a few-hundred-km line (the line's own placement uses the proper geodesic above;
    this is only used for the along-line wind overlay)."""
    import numpy as np
    lat0, lon0 = start_ll
    lat1, lon1 = end_ll
    mean_lat_rad = np.radians((lat0 + lat1) / 2.0)
    east = (lon1 - lon0) * np.cos(mean_lat_rad)
    north = lat1 - lat0
    norm = (east ** 2 + north ** 2) ** 0.5 or 1.0
    return east / norm, north / norm


def smooth_horizontal(field, window=15):
    """A light along-line moving average (scipy.ndimage.uniform_filter1d, already a wtune
    dependency), NaN-aware, applied only to a CONTOUR overlay -- never to the base shading.
    At 1 km resolution, real small-scale turbulent/gravity-wave structure oscillates sign
    every few grid columns; contouring it raw draws a dense, unreadable tangle (caught in
    lead review, 2026-09-29). This is display-only smoothing, the same convention a synoptic
    chart uses before contouring a noisy field -- it never touches the shading's own values."""
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    if field.shape[-1] < window:
        return field
    valid = np.isfinite(field)
    filled = np.where(valid, field, 0.0)
    smoothed_sum = uniform_filter1d(filled, size=window, axis=-1, mode="nearest")
    smoothed_count = uniform_filter1d(valid.astype(float), size=window, axis=-1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed_sum / smoothed_count
    return np.where(smoothed_count > 0, smoothed, np.nan)


def draw_locator_inset(fig, rect, lons2d, lats2d, backdrop, start_ll, end_ll, point_ll=None,
                       cmap="terrain"):
    """A small inset axes (plain lon/lat axes, not a real map projection -- it only needs to
    show roughly where the section is, not be geographically precise) filling `backdrop`
    (e.g. HGT) with the cross-section line and its centre point marked in red (2026-09-29
    lead review item 6: "a small inset map ... showing HGT and the section line so the reader
    knows where it is"). Returns the inset Axes."""
    ax_inset = fig.add_axes(rect)
    ax_inset.contourf(lons2d, lats2d, backdrop, levels=12, cmap=cmap)
    ax_inset.plot([start_ll[1], end_ll[1]], [start_ll[0], end_ll[0]], "r-", linewidth=1.2)
    if point_ll:
        ax_inset.plot([point_ll[1]], [point_ll[0]], "r*", markersize=6)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.8)
    return ax_inset
