"""Pure-logic tests for the 3-D cloud figure: title text, camera direction, coastline clipping
and VTK polyline output. pvbatch itself is exercised on scc."""
import importlib.util
import os

import numpy as np
import pytest

from post.recipes import paraview3d
from post import xwrf_convert


def _render_cloud_module():
    path = os.path.join(os.path.dirname(paraview3d.__file__), "..", "paraview",
                        "render_cloud.py")
    spec = importlib.util.spec_from_file_location("render_cloud", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_title_has_run_domain_time_quantity_threshold_and_exaggeration():
    t = paraview3d.build_title("katrina-base", 2, "2005-08-28_12:00:00")
    assert "katrina-base" in t and "d02" in t and "2005-08-28_12:00:00" in t
    assert "QCLOUD+QICE ≥ 0.1 g/kg" in t
    assert "×30" in t


def test_camera_is_south_south_east_and_above_the_horizon():
    rc = _render_cloud_module()
    x, y, z = rc.camera_offset()
    assert x > 0 and y < 0           # east and south of the target
    assert abs(x / y) == pytest.approx(np.tan(np.radians(22.5)), rel=1e-6)   # SSE bearing
    assert np.degrees(np.arcsin(z)) == pytest.approx(35.0)
    assert x * x + y * y + z * z == pytest.approx(1.0)


def test_coast_polylines_split_at_bbox_and_drop_short_pieces():
    line = np.array([[0, 0], [1, 0], [2, 0], [9, 0], [3, 0], [4, 0], [5, 5]], dtype=float)
    polys = xwrf_convert.coast_polylines([line], lambda lon, lat: (lon, lat), (0, 6, -1, 1))
    assert [len(p) for p in polys] == [3, 2]     # the lone (5,5) and the (9,0) gap are dropped


def test_write_vtk_polylines_counts(tmp_path):
    p = tmp_path / "c.vtk"
    xwrf_convert.write_vtk_polylines(str(p), [np.array([[0, 0], [1, 1]]),
                                              np.array([[2, 2], [3, 3], [4, 4]])], z=30.0)
    txt = p.read_text().splitlines()
    assert "POINTS 5 float" in txt
    assert "LINES 2 7" in txt
    assert txt[-1] == "3 2 3 4"
