"""A d01 across 180E (haitang2017 reaches 180.01E): maps without wedges, a fetch box that is not
the whole globe."""
import os

import netCDF4
import numpy as np

from wtune.post.conventions import continuous_lons

NY, NX = 30, 40
T0 = "2017-07-31_00:00:00"


def _grid():
    lat = (20.0 + 0.5 * np.arange(NY))[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * (165.0 + 0.6 * np.arange(NX))[None, :]      # 165 .. 188.4 E
    return lat, (lon + 180.0) % 360.0 - 180.0                            # as WRF writes XLONG


def test_continuous_lons_across_180():
    _lat, lon = _grid()
    c = continuous_lons(lon)
    assert (np.diff(c, axis=1) > 0).all()
    assert c.min() == 165.0 and abs(c.max() - 188.4) < 1e-9
    assert np.array_equal(continuous_lons([[-80.0, -70.0]]), [[-80.0, -70.0]])


def test_run_window_box_is_not_the_globe(tmp_path, capsys):
    from wtune.post.sources import run_window
    lat, lon = _grid()
    with netCDF4.Dataset(os.path.join(tmp_path, "wrfout_d01_" + T0), "w") as d:
        for name, n in (("Time", 1), ("DateStrLen", 19), ("south_north", NY), ("west_east", NX)):
            d.createDimension(name, n)
        d.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(T0)], "S1")
        dims = ("Time", "south_north", "west_east")
        d.createVariable("XLAT", "f4", dims)[:] = lat[None]
        d.createVariable("XLONG", "f4", dims)[:] = lon[None]
    lat0, lat1, lon0, lon1 = run_window(str(tmp_path))["bbox"]
    assert (lon0, lon1) == (165.0, 180.0)
    assert "across 180E" in capsys.readouterr().out


def test_filled_map_has_no_wedge_at_180(tmp_path):
    import cartopy.crs as ccrs
    import matplotlib.image as mpimg
    from wtune.post.recipes.mapfig import save_map
    lat, lon = _grid()
    out = str(tmp_path / "m.png")
    save_map(out, lat, lon, "t", fill_field=np.ones((NY, NX)), fill_levels=[0, 2],
             fill_colors=["#ff0000"], terrain=False,
             crs=ccrs.LambertConformal(central_longitude=177.0, standard_parallels=(30, 60)))
    img = mpimg.imread(out)[..., :3]
    h, w = img.shape[:2]
    core = img[int(0.35 * h):int(0.6 * h), int(0.3 * w):int(0.7 * w)]
    white = (core > 0.97).all(axis=-1).mean()
    assert white < 0.01, white
