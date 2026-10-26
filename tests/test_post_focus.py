"""post/focus.py: the point and time the recipes' point figures and one-time maps use."""
import os

import netCDF4
import numpy as np
import pytest

from wtune.post import focus

NY = NX = 60
TIMES = ["2017-07-31_00:00:00", "2017-07-31_06:00:00", "2017-07-31_07:00:00",
         "2017-07-31_08:00:00", "2017-07-31_09:00:00"]


def _core(j, i, amp):
    y, x = np.mgrid[0:NY, 0:NX]
    return amp * np.exp(-((x - i) ** 2 + (y - j) ** 2) / 20.0)


def _write(run, rain_by_time):
    lat = (20.0 + 0.03 * np.arange(NY))[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * (118.0 + 0.03 * np.arange(NX))[None, :]
    for t, rain in zip(TIMES, rain_by_time):
        with netCDF4.Dataset(os.path.join(run, "wrfout_d03_%s" % t), "w") as d:
            d.DX = 3000.0
            for name, n in (("Time", 1), ("DateStrLen", 19), ("south_north", NY),
                            ("west_east", NX)):
                d.createDimension(name, n)
            d.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(t)], "S1")
            dims = ("Time", "south_north", "west_east")
            d.createVariable("XLAT", "f4", dims)[:] = lat[None]
            d.createVariable("XLONG", "f4", dims)[:] = lon[None]
            d.createVariable("RAINC", "f4", dims)[:] = np.zeros((1, NY, NX))
            d.createVariable("RAINNC", "f4", dims)[:] = rain[None]


def test_the_wet_core_away_from_the_edge_and_its_peak_hour(tmp_path):
    # haitang2017 d03 (2026-10-26): the wettest single cell sat at the west edge and every point
    # figure was drawn at F072, after the rain had ended
    edge = _core(30, 2, 900.0)                     # boundary-zone rain, the old argmax
    early = _core(10, 10, 300.0)                   # spin-up rain, gone from the event total
    real = _core(30, 40, 1.0)
    acc = [early, early + 10 * real, early + 30 * real, early + 130 * real, early + 140 * real]
    _write(str(tmp_path), [a + edge * k for k, a in enumerate(acc)])
    f = focus.rain_focus(str(tmp_path), 3, spinup_hours=6)
    assert (abs(f["j"] - 30), abs(f["i"] - 40)) <= (1, 1)
    assert f["peak_time"] == "2017-07-31_08:00:00" and f["start"] == TIMES[0]
    assert "in post.focus" not in f["rule"]


def test_post_focus_box_restricts_the_search(tmp_path):
    acc = [np.zeros((NY, NX))] + [k * (_core(30, 40, 50.0) + _core(20, 20, 10.0))
                                  for k in range(1, 5)]
    _write(str(tmp_path), acc)
    lat = 20.0 + 0.03 * 20
    lon = 118.0 + 0.03 * 20
    f = focus.rain_focus(str(tmp_path), 3, spinup_hours=0,
                         box=[lat - 0.2, lat + 0.2, lon - 0.2, lon + 0.2])
    assert (abs(f["j"] - 20), abs(f["i"] - 20)) <= (1, 1) and "in post.focus" in f["rule"]


def test_post_focus_must_be_a_box():
    from wtune import project
    assert project.project_post_defaults({"post": {"focus": [21.8, 25.4, 119.9, 122.1]}})[
        "focus"] == [21.8, 25.4, 119.9, 122.1]
    with pytest.raises(project.ProjectError, match="post.focus"):
        project.project_post_defaults({"post": {"focus": [25.4, 21.8, 119.9, 122.1]}})
