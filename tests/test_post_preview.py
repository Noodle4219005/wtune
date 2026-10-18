"""wtune post preview: the figure list, the Stage IV tiling of the run window, and the case
facts read from wrfinput before any wrfout exists."""
from datetime import datetime, timedelta

import netCDF4
import numpy as np

from wtune.post import caseinfo, preview


def _case(tmp_path, hours=48, species=("QCLOUD",)):
    (tmp_path / "namelist.input").write_text(
        "&time_control\n run_days = 0, run_hours = %d,\n/\n&domains\n max_dom = 1,\n/\n" % hours)
    with netCDF4.Dataset(tmp_path / "wrfinput_d01", "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 4)
        ds.createDimension("bottom_top", 2)
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = np.array(list("2016-10-06_00:00:00"), dtype="S1")
        for name, val in (("XLAT", np.linspace(20, 30, 12)), ("XLONG", np.linspace(-80, -70, 12))):
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = \
                val.reshape(3, 4)
        for s in species:
            ds.createVariable(s, "f4", ("Time", "bottom_top", "south_north", "west_east"))[:] = 0
    return str(tmp_path)


def test_before_the_run_the_case_facts_come_from_wrfinput(tmp_path):
    case = _case(tmp_path)
    lat0, lat1, lon0, lon1, start, end = caseinfo._d01_bbox_and_window(case)
    assert (lat0, lat1, lon0, lon1) == (20, 30, -80, -70)
    assert (start, end) == (datetime(2016, 10, 6), datetime(2016, 10, 8))
    assert caseinfo.detect_cloud(case)[0]                   # hydrometeor variables in wrfinput


def test_expected_figures_list_the_detected_types_and_the_time_series():
    catalog = {"tc_track_map": {"zh": {"question": "q1"}}, "syn_slp": {"zh": {"question": "q2"}},
               "conv_x": {}, "timeseries_slp": {}, "chg_slp": {}, "_meta": {}}
    assert preview.expected_figures(["tc", "synoptic"], catalog) == [
        ("tc", "tc_track_map", "q1"), ("synoptic", "syn_slp", "q2")]
    assert [f for _t, f, _q in preview.expected_figures(["tc"], catalog, every="6h")] == [
        "tc_track_map", "timeseries_slp", "chg_slp"]


def test_stage4_takes_the_shortest_accumulation_that_tiles_the_window():
    t0 = datetime(2016, 10, 6)
    window = (t0, t0 + timedelta(hours=12))
    hourly = {(1, t0 + timedelta(hours=h)): "h%d" % h for h in range(1, 13)}
    six = {(6, t0 + timedelta(hours=6)): "a", (6, t0 + timedelta(hours=12)): "b"}
    assert preview.stage4_tiles({**hourly, **six}, window) == ["h%d" % h for h in range(1, 13)]
    del hourly[(1, t0 + timedelta(hours=5))]                  # a gap: the 6 h files still tile it
    assert preview.stage4_tiles({**hourly, **six}, window) == ["a", "b"]
    assert preview.stage4_tiles(hourly, window) == []
