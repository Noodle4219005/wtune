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


def test_tc_detection_box_crosses_180(tmp_path, monkeypatch):
    # raw min/max of XLONG made d01 -180..180 wide: any fix at that latitude was "inside"
    from datetime import datetime
    from wtune.post import caseinfo
    from wtune.post.conventions import lon_in_range
    lat, lon = _grid()
    with netCDF4.Dataset(os.path.join(tmp_path, "wrfout_d01_" + T0), "w") as d:
        for name, n in (("Time", 1), ("DateStrLen", 19), ("south_north", NY), ("west_east", NX)):
            d.createDimension(name, n)
        d.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(T0)], "S1")
        dims = ("Time", "south_north", "west_east")
        d.createVariable("XLAT", "f4", dims)[:] = lat[None]
        d.createVariable("XLONG", "f4", dims)[:] = lon[None]
    box = caseinfo._d01_bbox_and_window(str(tmp_path))
    assert abs(box[2] - 165.0) < 1e-4 and abs(box[3] - 188.4) < 1e-4
    t = datetime(2017, 7, 31)
    for fix, inside in ((-175.0, True), (185.0, True), (170.0, True), (0.0, False),
                        (-100.0, False), (150.0, False)):
        monkeypatch.setattr(caseinfo, "_best_track_points", lambda *a, f=fix: [(t, 25.0, f)])
        assert caseinfo.detect_tc(str(tmp_path), bdeck_path="b")[0] is inside, fix
    assert list(lon_in_range([-175.0, 0.0], 165.0, 188.4)) == [True, False]


def test_base_map_takes_coastline_past_180_and_axes_hold_one_box(monkeypatch):
    import cartopy.crs as ccrs
    from wtune.post import basemap
    seg = np.array([[-176.0, 52.0], [-174.0, 52.5]])          # an Aleutian island east of 180
    monkeypatch.setattr(basemap, "shapefile_path", lambda *a: "x")
    monkeypatch.setattr(basemap, "_all_segments", lambda p: [(-176.0, 52.0, -174.0, 52.5, seg)])
    got = basemap.layer_segments("physical", "coastline", "50m", (165.0, 190.0, 45.0, 60.0))
    assert len(got) == 1 and np.allclose(got[0][:, 0], [184.0, 186.0])
    assert basemap.layer_segments("physical", "coastline", "50m", (-180.0, -170.0, 45, 60))[0] is seg
    proj, c = basemap.centred_plate_carree(ccrs, 165.0, 190.0)
    assert c == 177.5
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection=proj)
    basemap.set_lonlat_extent(ax, (165.0, 190.0, 45.0, 60.0), c)
    x0, x1, _y0, _y1 = ax.get_extent(crs=proj)
    assert abs(x0 + 12.5) < 1e-6 and abs(x1 - 12.5) < 1e-6      # 25 deg wide, not the globe
    plt.close("all")


def test_track_map_across_180_is_one_line(tmp_path, monkeypatch):
    from datetime import datetime, timedelta
    import matplotlib.axes
    from wtune.post import tc
    t0 = datetime(2017, 7, 31)
    pts = [(t0 + timedelta(hours=6 * k), (20.0 + k, 176.0 + 2 * k - 360 * (k > 2)), 1)
           for k in range(5)]                                  # 176, 178, 180, -178, -176
    xs = []
    real = matplotlib.axes.Axes.plot
    monkeypatch.setattr(matplotlib.axes.Axes, "plot",
                        lambda self, x, *a, **k: xs.extend(np.ravel(x)) or real(self, x, *a, **k))
    tc._track_map_figure(str(tmp_path), {"run0": pts}, None, "T01", dom_dx={1: 15.0})
    assert xs and max(xs) - min(xs) == 8.0                     # 176..184, no 356 deg jump
