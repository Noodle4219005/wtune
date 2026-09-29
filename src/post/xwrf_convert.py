"""post/xwrf_convert.py -- one wrfout frame -> CF-ish NetCDF for ParaView 3-D rendering
(decision 0057 item 3: tc/convection cloud isosurface). xWRF (`ds.xwrf.postprocess()`, the
established WRF->xarray tool this repo already depends on for exactly this conversion --
"build on established tools") supplies the projected x/y coordinates (metres) and destaggered
mass-grid fields; this module only sums QCLOUD+QICE+QSNOW into one CLOUD_TOTAL field (no tool
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


def convert_frame(wrfout_path, out_prefix):
    """Writes out_prefix + '_cloud.nc' (CLOUD_TOTAL, Z_HEIGHT on Time,z,y,x) and
    out_prefix + '_terrain.nc' (HGT on Time,y,x). Returns (cloud_nc_path, terrain_nc_path)."""
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

    cloud_total = (ds["QCLOUD"] + ds["QICE"] + ds["QSNOW"]).astype("float32")
    cloud_total = cloud_total.drop_vars(_extra_coords, errors="ignore")
    cloud_total.attrs = dict(units="kg kg-1", description="QCLOUD+QICE+QSNOW")

    with netCDF4.Dataset(wrfout_path) as nc_ds:
        z_height = np.asarray(wrf_mod.getvar(nc_ds, "z", timeidx=0, units="m"), dtype="float32")
    z_da = xr.DataArray(z_height[None, ...], dims=("Time", "z", "y", "x"),
                        coords={"Time": ds["Time"], "z": ds["z"], "y": ds["y"], "x": ds["x"]},
                        attrs=dict(units="m", description="geopotential height (wrf-python 'z')"))

    cloud_ds = xr.Dataset({"CLOUD_TOTAL": cloud_total, "Z_HEIGHT": z_da})
    cloud_nc = out_prefix + "_cloud.nc"
    cloud_ds.to_netcdf(cloud_nc)

    terrain = ds["HGT"].astype("float32").drop_vars(_extra_coords, errors="ignore")
    terrain_ds = xr.Dataset({"HGT": terrain})
    terrain_nc = out_prefix + "_terrain.nc"
    terrain_ds.to_netcdf(terrain_nc)
    return cloud_nc, terrain_nc
