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
    """Legacy ASCII VTK POLYDATA of the polylines at constant height z (metres)."""
    npts = sum(len(p) for p in polylines)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ncoastline\nASCII\nDATASET POLYDATA\n")
        f.write("POINTS %d float\n" % npts)
        for pl in polylines:
            for px, py in pl:
                f.write("%.1f %.1f %.1f\n" % (px, py, z))
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
