"""post/xwrf_convert.py -- one wrfout frame -> CF-ish NetCDF for ParaView 3-D rendering
(decision 0057 item 3: tc/convection cloud isosurface). xWRF (`ds.xwrf.postprocess()`, the
established WRF->xarray tool this repo already depends on for exactly this conversion --
"build on established tools") supplies the projected x/y coordinates (metres) and destaggered
mass-grid fields; this module only sums QCLOUD+QICE into one QCLOUD_QICE field (no tool
provides that specific sum) and adds wrf-python's own 'z' diagnostic (geopotential height, m)
as a physical vertical coordinate field, since the NetCDF's own 'z' dimension is WRF's eta
level index, not a height -- post/paraview/render_cloud.py needs the real height to warp the
grid into.

Writes two separate NetCDF files -- cloud fields on the full 3-D grid, terrain on its own 2-D
grid -- rather than one mixed-rank file, so render_cloud.py can open each with a plain,
reader-configuration-free `OpenDataFile()` call.

XLAT/XLONG are deliberately NOT attached as coordinates here (fixed 2026-09-29, after a real
ParaView run on scc): when they are present, xarray's CF writer tags CLOUD_TOTAL/HGT with a
`coordinates = "XLAT XLONG"` attribute, and ParaView's vtkNetCDFReader then builds point
geometry from those 2-D degree-valued arrays -- worse, it first ran its own spherical-coordinate
heuristic on them (`SphericalCoordinates=1`), collapsing X/Y into a near-zero, unit-sphere-like
range (~-0.15..0.18) and producing an empty contour downstream ("Could not determine array
range" in pvbatch's own log). Leaving x(x)/y(y) as the only spatial coordinates makes the reader
use them directly: real, WRF-projection metres, on the same physical scale as Z_HEIGHT (also
metres) -- the correct geometry for a 3-D render, and the reason render_cloud.py no longer needs
degree-to-metre conversion math of its own. xWRF's own CRS DataArray (`ds['wrf_projection']`, a
non-serializable pyproj object) is still not copied into the output.
"""
from __future__ import annotations

import os


def coast_polylines(lines, transform, bbox):
    """Natural Earth coastline parts -> projected polylines clipped to the grid.

    lines: iterable of (N, 2) lon/lat arrays; transform(lon, lat) -> (x, y) in the grid's
    projected metres; bbox: (xmin, xmax, ymin, ymax) in those metres. Points outside bbox are
    dropped and split the line there, so nothing is drawn past the domain edge. Returns a list
    of (M, 2) arrays with M >= 2."""
    import numpy as np
    xmin, xmax, ymin, ymax = bbox
    out = []
    for ll in lines:
        ll = np.asarray(ll, dtype=float)
        if ll.ndim != 2 or len(ll) < 2:
            continue
        x, y = transform(ll[:, 0], ll[:, 1])
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        inside = (np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
                  & (y >= ymin) & (y <= ymax))
        start = None
        for i in range(len(x) + 1):
            ok = i < len(x) and inside[i]
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                if i - start >= 2:
                    out.append(np.column_stack([x[start:i], y[start:i]]))
                start = None
    return out


def write_vtk_polylines(path, polylines, z=0.0):
    """Legacy ASCII VTK POLYDATA of the polylines: (N, 2) arrays at constant height z, or
    (N, 3) arrays carrying their own per-vertex z (written with 3 decimals)."""
    npts = sum(len(p) for p in polylines)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ncoastline\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS %d float\n" % npts)
        for pl in polylines:
            for pt in pl:
                if len(pt) == 3:
                    f.write("%.3f %.3f %.3f\n" % tuple(pt))
                else:
                    f.write("%.1f %.1f %.1f\n" % (pt[0], pt[1], z))
        f.write("LINES %d %d\n" % (len(polylines), len(polylines) + npts))
        i = 0
        for pl in polylines:
            f.write("%d %s\n" % (len(pl), " ".join(str(i + k) for k in range(len(pl)))))
            i += len(pl)


def _write_coast(ds, out_path, resolution="50m"):
    """Natural Earth coastline (cached under WTUNE_CARTOPY_DATA) in the grid's projected
    coordinates -> out_path. Returns out_path, or None when the data or the projection is
    unavailable (the render then simply has no coastline)."""
    import numpy as np
    try:
        import cartopy.io.shapereader as shpreader
        from post.plotting import _set_cartopy_data_dir
        from pyproj import Transformer
        _set_cartopy_data_dir()
        shp = shpreader.natural_earth(resolution=resolution, category="physical",
                                      name="coastline")
        if not os.path.exists(shp):
            return None
        crs = ds["wrf_projection"].item()
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        lines = []
        for g in shpreader.Reader(shp).geometries():
            for part in getattr(g, "geoms", [g]):
                lines.append(np.asarray(part.coords)[:, :2])
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        pad = 0.0
        polys = coast_polylines(lines, tr.transform,
                                (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad))
    except Exception:
        return None
    if not polys:
        return None
    write_vtk_polylines(out_path, polys, z=30.0)
    return out_path


def convert_frame(wrfout_path, out_prefix):
    """Writes out_prefix + '_cloud.nc' (QCLOUD_QICE, Z_HEIGHT, Z_KM on Time,z,y,x),
    out_prefix + '_terrain.nc' (HGT, LANDMASK, TERRAIN_H on Time,y,x: TERRAIN_H is HGT over land
    and -500 over sea, for one land/sea colour map) and, when Natural Earth 50m coastline is
    cached, out_prefix + '_coast.vtk' (polylines in the grid's projected metres).
    Returns (cloud_nc_path, terrain_nc_path, coast_vtk_path_or_None)."""
    import xarray as xr
    import xwrf  # noqa: F401 -- registers the .xwrf accessor
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")

    raw = xr.open_dataset(wrfout_path)
    ds = raw.xwrf.postprocess()

    # drop_vars(..., errors="ignore"): QCLOUD/QICE/QSNOW/HGT inherit xwrf's full non-dimension
    # coordinate set (XLAT/XLONG/CLAT/XTIME) from ds itself -- passing coords= to xr.Dataset()
    # below is NOT enough to keep them out, since xr.Dataset() re-unions each DataArray's own
    # attached coords regardless. See the module docstring for why they must not survive here.
    _extra_coords = ["XLAT", "XLONG", "CLAT", "XTIME"]

    cloud_total = (ds["QCLOUD"] + ds["QICE"]).astype("float32")
    cloud_total = cloud_total.drop_vars(_extra_coords, errors="ignore")
    cloud_total.attrs = dict(units="kg kg-1", description="QCLOUD+QICE")

    with netCDF4.Dataset(wrfout_path) as nc_ds:
        z_height = np.asarray(wrf_mod.getvar(nc_ds, "z", timeidx=0, units="m"), dtype="float32")
    z_da = xr.DataArray(z_height[None, ...], dims=("Time", "z", "y", "x"),
                        coords={"Time": ds["Time"], "z": ds["z"], "y": ds["y"], "x": ds["x"]},
                        attrs=dict(units="m", description="geopotential height (wrf-python 'z')"))

    z_km = (z_da / 1000.0).astype("float32")
    z_km.attrs = dict(units="km", description="geopotential height (km)")
    cloud_ds = xr.Dataset({"QCLOUD_QICE": cloud_total, "Z_HEIGHT": z_da, "Z_KM": z_km})
    cloud_nc = out_prefix + "_cloud.nc"
    cloud_ds.to_netcdf(cloud_nc)

    terrain = ds["HGT"].astype("float32").drop_vars(_extra_coords, errors="ignore")
    landmask = ds["LANDMASK"].astype("float32").drop_vars(_extra_coords, errors="ignore")
    terrain_h = xr.where(landmask >= 0.5, terrain.clip(min=0.0), -500.0).astype("float32")
    terrain_h.attrs = dict(units="m", description="HGT over land, -500 over sea")
    terrain_ds = xr.Dataset({"HGT": terrain, "LANDMASK": landmask, "TERRAIN_H": terrain_h})
    terrain_nc = out_prefix + "_terrain.nc"
    terrain_ds.to_netcdf(terrain_nc)
    coast = _write_coast(ds, out_prefix + "_coast.vtk")
    return cloud_nc, terrain_nc, coast


# --------------------------------------------------------------------------------------
# Total-condensate volume (2026-10-02): one wrfout frame -> a regular 3-D grid for ParaView
# volume rendering (post/paraview/render_volume.py). The old figure contoured QCLOUD+QICE,
# 10% of the condensate mass in SC26-P2 (post/condensate.py); this grids the sum of every
# hydrometeor instead. Everything below except build_volume() is pure numpy so it is unit-
# testable without ParaView, wrf-python or a real wrfout.
# --------------------------------------------------------------------------------------

VOLUME_TOP_KM = 18.0
VOLUME_DZ_KM = 0.25
LOGQ_MIN = -2.0          # log10(g/kg): 0.01 g/kg = condensate.CLOUD_EDGE_KGKG, opacity starts
LOGQ_MAX = 1.0           # 10 g/kg
MAX_HORIZONTAL_POINTS = 400   # per axis; larger grids are subsampled (stride) for the volume


STORM_BOX_KM = 600.0


VOLUME_MIN_TOP_KM = 4.0


def default_vexag(width_km, ztop_km=VOLUME_TOP_KM, box_km=None):
    """Vertical exaggeration so the rendered column is ~1/5 of the horizontal width:
    ztop_km * vexag ~= width_km / 5, clipped to [1, 30]. A storm box (box_km given) keeps the
    larger of that and box_km / 75 clipped to [2, 30] (600 km -> 8). Rounded to an integer."""
    v = float(min(30, max(1, round(width_km / (5.0 * ztop_km)))))
    if box_km:
        v = max(v, float(min(30, max(2, round(box_km / 75.0)))))
    return v


def volume_top_km(cloud_top_m, top_km=VOLUME_TOP_KM, floor_km=VOLUME_MIN_TOP_KM):
    """Adaptive top of the gridded volume: the 99.9th percentile of the cloud-top heights
    (m MSL, NaN = clear column) + 1 km, rounded up to a whole km, within [floor_km, top_km].
    No cloud at all -> top_km. A shallow (lake-effect, 2-3 km) cloud deck is otherwise a thin
    film at the bottom of an 18 km box."""
    import math
    import numpy as np
    ct = np.asarray(cloud_top_m, dtype=float)
    ct = ct[np.isfinite(ct)]
    if ct.size == 0:
        return float(top_km)
    z = math.ceil(np.percentile(ct, 99.9) / 1000.0 + 1.0 - 1e-9)
    return float(min(top_km, max(floor_km, z)))


def volume_levels_km(top_km=VOLUME_TOP_KM, dz_km=VOLUME_DZ_KM):
    """0, dz, ..., top (km MSL)."""
    import numpy as np
    return np.round(np.arange(0.0, top_km + dz_km / 2.0, dz_km), 6)


def nearest_index(xlat, xlong, lat, lon):
    """(j, i, distance_km) of the mass-grid point nearest to (lat, lon)."""
    import numpy as np
    xlat = np.asarray(xlat, dtype=float)
    dlon = (np.asarray(xlong, dtype=float) - lon + 180.0) % 360.0 - 180.0
    dx = np.radians(dlon) * np.cos(np.radians(0.5 * (xlat + lat)))
    dy = np.radians(xlat - lat)
    d2 = dx * dx + dy * dy
    j, i = np.unravel_index(int(np.argmin(d2)), d2.shape)
    return int(j), int(i), float(np.sqrt(d2[j, i]) * 6371.0)


def crop_bounds(ny, nx, dx_km, dy_km, bdy_width, center_ji=None, box_km=None):
    """(j0, j1, i0, i1) python-slice bounds. Without center_ji: the domain interior (bdy_width
    cells dropped on every edge, WRF's lateral boundary zone). With center_ji: a box_km x box_km
    square centred there, clipped to that same interior. Raises ValueError when nothing is
    left."""
    j0, j1, i0, i1 = bdy_width, ny - bdy_width, bdy_width, nx - bdy_width
    if center_ji is not None and box_km:
        jc, ic = center_ji
        hj = int(round(0.5 * box_km / dy_km))
        hi = int(round(0.5 * box_km / dx_km))
        j0, j1 = max(j0, jc - hj), min(j1, jc + hj + 1)
        i0, i1 = max(i0, ic - hi), min(i1, ic + hi + 1)
    if j1 - j0 < 2 or i1 - i0 < 2:
        raise ValueError("crop leaves no grid (ny=%d nx=%d bdy=%d centre=%s box=%s km)"
                         % (ny, nx, bdy_width, center_ji, box_km))
    return j0, j1, i0, i1


def horizontal_stride(ny, nx, max_points=MAX_HORIZONTAL_POINTS):
    return max(1, -(-max(ny, nx) // max_points))


def regrid_to_heights(field, z_m, hgt_m, levels_km):
    """Linear-in-height interpolation of field (nk, ny, nx) on model mass levels z_m (m MSL,
    increasing with k) to levels_km (km MSL). Levels below the terrain (hgt_m) or above the
    model top are 0; between the terrain and the lowest mass level the lowest-level value is
    used. Returns float32 (nl, ny, nx)."""
    import numpy as np
    f = np.asarray(field, dtype=np.float32)
    z = np.asarray(z_m, dtype=np.float32)
    hgt = np.asarray(hgt_m, dtype=np.float32)
    nk = z.shape[0]
    out = np.zeros((len(levels_km),) + z.shape[1:], dtype=np.float32)
    for n, zk in enumerate(np.asarray(levels_km, dtype=float)):
        zt = np.float32(zk * 1000.0)
        above = z >= zt
        has = above.any(axis=0)                       # False: above the model top
        k = np.argmax(above, axis=0)                  # first level at or above zt
        k1 = np.clip(k, 1, nk - 1)[None]
        k0 = k1 - 1
        z0 = np.take_along_axis(z, k0, 0)[0]
        z1 = np.take_along_axis(z, k1, 0)[0]
        f0 = np.take_along_axis(f, k0, 0)[0]
        f1 = np.take_along_axis(f, k1, 0)[0]
        w = np.clip((zt - z0) / np.where(z1 > z0, z1 - z0, 1.0), 0.0, 1.0)
        val = f0 + w * (f1 - f0)
        lowest = has & (k == 0)                       # zt at or below the lowest mass level
        val = np.where(lowest, np.where(zt >= hgt, f[0], 0.0), val)
        out[n] = np.where(has, val, 0.0)
    return out


def log_condensate(q_gkg, lo=LOGQ_MIN, hi=LOGQ_MAX):
    """log10(q in g/kg) clipped to [lo, hi]; q <= 10**lo (including 0) maps to lo."""
    import numpy as np
    q = np.asarray(q_gkg, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.log10(np.maximum(q, np.float32(10.0 ** lo)))
    return np.clip(lq, lo, hi).astype(np.float32)


def tangential_wind(u, v, x_km, y_km, xc_km, yc_km, cyclonic_sign=1.0):
    """Tangential wind about (xc, yc) for grid-relative u, v (..., ny, nx) and the grid's own
    projected 1-D axes: (-dy*u + dx*v)/r, counter-clockwise positive, times cyclonic_sign
    (pass -1 in the southern hemisphere so cyclonic is positive). 0 at the centre point.

    Grid-relative wind with grid-relative offsets is the consistent pair on WRF's conformal
    projections (the same number as earth-relative wind with earth-relative offsets)."""
    import numpy as np
    dx = np.asarray(x_km, dtype=float)[None, :] - xc_km
    dy = np.asarray(y_km, dtype=float)[:, None] - yc_km
    r = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        vt = np.where(r > 0, (-dy * u + dx * v) / np.where(r > 0, r, 1.0), 0.0)
    return (cyclonic_sign * vt).astype(np.float32)


def camera_side_mask(x_km, y_km, xc_km, yc_km, bearing_deg, beyond_km=0.0):
    """(ny, nx) True on the camera's side of the vertical plane through (xc, yc) that faces a
    camera standing at compass bearing bearing_deg from the centre -- the half a cutaway
    removes -- plus a slab beyond_km thick behind the plane (signed distance > -beyond_km)."""
    import math
    import numpy as np
    b = math.radians(bearing_deg)
    nx_, ny_ = round(math.sin(b), 12), round(math.cos(b), 12)   # exact 0 at 90/180 deg
    dx = np.asarray(x_km, dtype=float)[None, :] - xc_km
    dy = np.asarray(y_km, dtype=float)[:, None] - yc_km
    return dx * nx_ + dy * ny_ > -beyond_km


def write_vti(path, origin, spacing, arrays):
    """VTK XML ImageData, point data, raw binary appended (UInt64 block headers, little
    endian). arrays: {name: (nz, ny, nx) array} -- C order, so x varies fastest as VTK
    requires. All arrays are written as Float32."""
    import numpy as np
    shapes = {np.asarray(a).shape for a in arrays.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 3:
        raise ValueError("arrays must share one (nz, ny, nx) shape: %s" % shapes)
    nz, ny, nx = next(iter(shapes))
    blobs, offset, decl = [], 0, []
    for name, a in arrays.items():
        b = np.ascontiguousarray(a, dtype="<f4").tobytes()
        decl.append('        <DataArray type="Float32" Name="%s" NumberOfComponents="1" '
                    'format="appended" offset="%d"/>' % (name, offset))
        blobs.append(np.uint64(len(b)).astype("<u8").tobytes() + b)
        offset += 8 + len(b)
    ext = "0 %d 0 %d 0 %d" % (nx - 1, ny - 1, nz - 1)
    head = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">\n'
        '  <ImageData WholeExtent="%s" Origin="%r %r %r" Spacing="%r %r %r">\n'
        '    <Piece Extent="%s">\n'
        '      <PointData Scalars="%s">\n%s\n      </PointData>\n'
        '      <CellData/>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '  <AppendedData encoding="raw">\n   _' % (
            ext, float(origin[0]), float(origin[1]), float(origin[2]),
            float(spacing[0]), float(spacing[1]), float(spacing[2]), ext,
            next(iter(arrays)), "\n".join(decl)))
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        for b in blobs:
            f.write(b)
        f.write(b"\n  </AppendedData>\n</VTKFile>\n")
    return path


def write_terrain_vtk(path, x_km, y_km, hgt_m, landmask, vexag):
    """Legacy ASCII VTK STRUCTURED_GRID of the terrain surface at z = HGT(km) * vexag, with
    point data TERRAIN_H (HGT over land, -500 m over sea, for one land/sea colour map) and
    HGT (m)."""
    import numpy as np
    hgt = np.asarray(hgt_m, dtype=float)
    land = np.asarray(landmask, dtype=float) >= 0.5
    terrain_h = np.where(land, np.clip(hgt, 0.0, None), -500.0)
    ny, nx = hgt.shape
    X, Y = np.meshgrid(np.asarray(x_km, dtype=float), np.asarray(y_km, dtype=float))
    Z = np.clip(hgt, 0.0, None) / 1000.0 * vexag
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nterrain\nASCII\nDATASET STRUCTURED_GRID\n")
        f.write("DIMENSIONS %d %d 1\nPOINTS %d float\n" % (nx, ny, nx * ny))
        np.savetxt(f, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), fmt="%.3f")
        f.write("POINT_DATA %d\n" % (nx * ny))
        for name, a in (("TERRAIN_H", terrain_h), ("HGT", hgt)):
            f.write("SCALARS %s float 1\nLOOKUP_TABLE default\n" % name)
            np.savetxt(f, a.ravel(), fmt="%.1f")
    return path


SHORELINE_LIFT_M = 50.0


def sample_bilinear(x_km, y_km, field, px, py):
    """Bilinear sample of field (ny, nx) on the uniform axes x_km (nx,), y_km (ny,) at points
    (px, py); points outside the grid take the nearest edge value."""
    import numpy as np
    f = np.asarray(field, dtype=float)
    x = np.asarray(x_km, dtype=float)
    y = np.asarray(y_km, dtype=float)
    fi = np.clip((np.asarray(px, dtype=float) - x[0]) / (x[1] - x[0]), 0, len(x) - 1)
    fj = np.clip((np.asarray(py, dtype=float) - y[0]) / (y[1] - y[0]), 0, len(y) - 1)
    i0 = np.minimum(np.floor(fi).astype(int), len(x) - 2)
    j0 = np.minimum(np.floor(fj).astype(int), len(y) - 2)
    wi, wj = fi - i0, fj - j0
    return ((1 - wj) * ((1 - wi) * f[j0, i0] + wi * f[j0, i0 + 1])
            + wj * ((1 - wi) * f[j0 + 1, i0] + wi * f[j0 + 1, i0 + 1]))


def drape_polylines(polylines, x_km, y_km, hgt_m, vexag, lift_m=SHORELINE_LIFT_M):
    """(N, 2) km polylines -> (N, 3) with z = (HGT + lift_m) * vexag / 1000 (rendered km),
    HGT sampled bilinearly and clipped at 0 like the terrain surface write_terrain_vtk draws.
    A lake shoreline (Lake Michigan, HGT ~176 m) otherwise sits under the exaggerated
    terrain/lake surface and is invisible (scc, SC26-P1 d02, x13)."""
    import numpy as np
    hgt = np.clip(np.asarray(hgt_m, dtype=float), 0.0, None)
    out = []
    for pl in polylines:
        pl = np.asarray(pl, dtype=float)
        h = sample_bilinear(x_km, y_km, hgt, pl[:, 0], pl[:, 1])
        out.append(np.column_stack([pl[:, 0], pl[:, 1], (h + lift_m) * vexag / 1000.0]))
    return out


def _projected_axes(wrfout_path, ny, nx, dx_km, dy_km):
    """(x_km (nx,), y_km (ny,), lonlat->km transform or None). xWRF's projected x/y (the grid's
    own projection, metres) when xwrf/pyproj work on this file; otherwise index * DX with no
    transform (the render then has no coastline)."""
    import numpy as np
    try:
        import xarray as xr
        import xwrf  # noqa: F401
        from pyproj import Transformer
        with xr.open_dataset(wrfout_path) as raw:
            ds = raw.xwrf.postprocess()
            x = np.asarray(ds["x"].values, dtype=float) / 1000.0
            y = np.asarray(ds["y"].values, dtype=float) / 1000.0
            crs = ds["wrf_projection"].item()
        if x.shape == (nx,) and y.shape == (ny,):
            tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

            def to_km(lon, lat):
                px, py = tr.transform(lon, lat)
                return np.asarray(px) / 1000.0, np.asarray(py) / 1000.0
            return x, y, to_km
    except Exception:
        pass
    return np.arange(nx) * dx_km, np.arange(ny) * dy_km, None


def _shoreline_lines(lonlat_bbox, resolution="50m", layers=("coastline", "lakes")):
    """Natural Earth physical coastline and lake shorelines (post/basemap.py's local cache
    lookup; never downloads) as lon/lat arrays whose bounds touch lonlat_bbox
    (lon0, lon1, lat0, lat1); [] when nothing is cached."""
    from post import basemap
    lon0, lon1, lat0, lat1 = lonlat_bbox
    out = []
    for name in layers:
        try:
            path = basemap.shapefile_path("physical", name, resolution)
            segs = basemap._all_segments(path) if path else []
        except Exception:   # noqa: BLE001 -- a broken cache only loses the shoreline
            segs = []
        for x0, y0, x1, y1, xy in segs:
            if x1 >= lon0 and x0 <= lon1 and y1 >= lat0 and y0 <= lat1:
                out.append(xy)
    return out


def build_volume(wrfout_path, out_prefix, *, center_latlon=None, box_km=None, cutaway=False,
                 bearing_deg=202.5, vexag=None, bdy_width=5, timeidx=0,
                 max_points=MAX_HORIZONTAL_POINTS):
    """One wrfout frame -> out_prefix + '_condensate.vti' (point data log10_q = log10(q g/kg)
    clipped to [-2, 1], q_gkg, and vt (tangential wind, m s-1, cyclonic positive) when
    cutaway; spacing in km, z spacing times vexag; 0 .. ztop_km, volume_top_km() of the box's
    cloud tops), out_prefix + '_terrain.vtk' and, when Natural Earth is cached,
    out_prefix + '_coast.vtk' (coastline AND lake shorelines). q is condensate.
    total_condensate (every hydrometeor). With cutaway, log10_q is set to -2 (transparent) on
    the camera's side of the vertical plane through the centre and in a 1.5 dx slab behind it;
    q_gkg and vt are not masked.

    Returns a dict of the files and the geometry the render script needs (horizontal km;
    vexag is the exaggeration already applied to every z in the files). vexag=None:
    default_vexag() of the cropped width and ztop (plus the storm-box rule with a centre); with
    center_latlon and no box_km the box is STORM_BOX_KM."""
    import netCDF4
    import numpy as np
    from post import condensate

    with netCDF4.Dataset(wrfout_path) as nc:
        q = condensate.total_condensate(nc, timeidx) * 1000.0          # g/kg
        z = condensate.height_msl(nc, timeidx)
        hgt = np.asarray(nc.variables["HGT"][timeidx], dtype=float)
        land = (np.asarray(nc.variables["LANDMASK"][timeidx], dtype=float)
                if "LANDMASK" in nc.variables else (hgt > 0).astype(float))
        xlat = np.asarray(nc.variables["XLAT"][timeidx], dtype=float)
        xlong = np.asarray(nc.variables["XLONG"][timeidx], dtype=float)
        cloud_top = condensate.cloud_top_height(nc, timeidx)              # m MSL, NaN clear
        dx_km = float(nc.getncattr("DX")) / 1000.0
        dy_km = float(nc.getncattr("DY")) / 1000.0
        u = v = None
        if cutaway:
            uu = np.asarray(nc.variables["U"][timeidx], dtype=float)
            vv = np.asarray(nc.variables["V"][timeidx], dtype=float)
            u = 0.5 * (uu[..., 1:] + uu[..., :-1])
            v = 0.5 * (vv[..., 1:, :] + vv[..., :-1, :])
    ny, nx = hgt.shape
    x_all, y_all, to_km = _projected_axes(wrfout_path, ny, nx, dx_km, dy_km)

    center_ji = None
    if center_latlon is not None:
        j, i, dist = nearest_index(xlat, xlong, center_latlon[0], center_latlon[1])
        if dist > 2.0 * max(dx_km, dy_km):
            raise ValueError("centre %s is outside the domain (nearest grid point %.0f km away)"
                             % (center_latlon, dist))
        center_ji = (j, i)
    if center_ji is not None and not box_km:
        box_km = STORM_BOX_KM
    j0, j1, i0, i1 = crop_bounds(ny, nx, dx_km, dy_km, bdy_width, center_ji, box_km)
    stride = horizontal_stride(j1 - j0, i1 - i0, max_points)
    sj, si = slice(j0, j1, stride), slice(i0, i1, stride)
    x_km, y_km = x_all[si], y_all[sj]
    if center_ji is not None:
        xc, yc = float(x_all[center_ji[1]]), float(y_all[center_ji[0]])
    else:
        xc, yc = float(0.5 * (x_km[0] + x_km[-1])), float(0.5 * (y_km[0] + y_km[-1]))

    width_km = float(max(x_km[-1] - x_km[0], y_km[-1] - y_km[0]))
    ztop = volume_top_km(cloud_top[sj, si])
    if vexag is None:
        vexag = default_vexag(width_km, ztop, box_km if center_ji is not None else None)
    levels = volume_levels_km(ztop)
    zc, hc = z[:, sj, si], hgt[sj, si]
    q3 = regrid_to_heights(q[:, sj, si], zc, hc, levels)
    logq = log_condensate(q3)
    arrays = {"log10_q": logq, "q_gkg": q3}
    vt_max = None
    if cutaway:
        sign = -1.0 if center_latlon is not None and center_latlon[0] < 0 else 1.0
        vt_model = tangential_wind(u[:, sj, si], v[:, sj, si], x_km, y_km, xc, yc, sign)
        vt = regrid_to_heights(vt_model, zc, hc, levels)
        # transparent: the camera-side half AND a 1.5 dx slab behind the cut plane. With the
        # transparency edge exactly on the plane, trilinear sampling of the volume leaves a
        # thin semi-opaque sheet at the plane that shows through the slice as vertical white
        # streaks (scc, SC26-P2 F048); the slab moves that sheet behind the opaque cut face,
        # which itself stays exactly through the centre.
        slab = 1.5 * max(dx_km, dy_km) * stride
        logq[:, camera_side_mask(x_km, y_km, xc, yc, bearing_deg, slab)] = LOGQ_MIN
        arrays["vt"] = vt
        cloudy = q3 >= 10.0 ** LOGQ_MIN
        ref = np.abs(vt[cloudy]) if cloudy.any() else np.abs(vt).ravel()
        vt_max = float(max(10.0, 10.0 * np.ceil(np.percentile(ref, 99.5) / 10.0)))

    vti = write_vti(out_prefix + "_condensate.vti", (x_km[0], y_km[0], 0.0),
                    (dx_km * stride, dy_km * stride, VOLUME_DZ_KM * vexag), arrays)
    terrain = write_terrain_vtk(out_prefix + "_terrain.vtk", x_km, y_km, hc, land[sj, si],
                                vexag)
    coast = None
    if to_km is not None:
        lat_c, lon_c = xlat[sj, si], xlong[sj, si]
        bbox_ll = (float(lon_c.min()) - 1, float(lon_c.max()) + 1,
                   float(lat_c.min()) - 1, float(lat_c.max()) + 1)
        polys = coast_polylines(_shoreline_lines(bbox_ll), to_km,
                                (x_km[0], x_km[-1], y_km[0], y_km[-1]))
        if polys:
            coast = out_prefix + "_coast.vtk"
            write_vtk_polylines(coast, drape_polylines(polys, x_km, y_km, hc, vexag))
    return dict(vti=vti, terrain=terrain, coast=coast, center_km=(xc, yc),
                extent_km=(float(x_km[0]), float(x_km[-1]), float(y_km[0]), float(y_km[-1])),
                width_km=width_km,
                dx_km=dx_km, stride=stride, vexag=vexag, vt_max=vt_max, ztop_km=ztop,
                shape=tuple(int(s) for s in q3.shape))
