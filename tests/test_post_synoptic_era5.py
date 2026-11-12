"""Tests for the synoptic ERA5 diff pieces: symmetric levels with a white zero bin, the Grid-Stat
CNT parser (fixtures captured from a real MET 12.1.1 run on scc, Katrina m1 d01 z500,
2005-08-29 00Z), and the local ERA5 subset taking precedence over gcsfs."""
import os
from datetime import datetime

import numpy as np
import pytest

from wtune.post import met_gridstat
from wtune.post.recipes import era5

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def test_parse_cnt_real_output():
    with open(os.path.join(FIX, "grid_stat_z500_cnt.txt")) as fh:
        res = met_gridstat.parse_cnt(fh.read())
    assert res == {"rmse_m": pytest.approx(13.25065), "bias_m": pytest.approx(9.69062),
                   "n": 3825}


def test_real_stat_line_carries_the_same_cnt_values():
    # the .stat file's CNT row: values after LINE_TYPE are TOTAL, FBAR, ... (unnamed columns)
    with open(os.path.join(FIX, "grid_stat_z500.stat")) as fh:
        lines = fh.read().splitlines()
    head = lines[0].split()
    vals = lines[1].split()[len(head):]
    assert head[-1] == "LINE_TYPE" and lines[1].split()[len(head) - 1] == "CNT"
    assert vals[0] == "3825" and "13.25065" in vals and "9.69062" in vals


def test_parse_cnt_without_cnt_row_raises():
    with pytest.raises(met_gridstat.METError):
        met_gridstat.parse_cnt("VERSION LINE_TYPE TOTAL\nV12.1.1 SL1L2 5\n")


def test_local_subset_takes_precedence_over_gcsfs(tmp_path, monkeypatch):
    t = datetime(2005, 8, 29, 0)
    lat, lon = np.linspace(20, 22, 5), np.linspace(-95, -93, 7)
    z = np.full((5, 7), 5800.0)
    path = era5.subset_path(str(tmp_path), t)
    era5.write_subset(lat, lon, z, t, path)

    def boom():
        raise AssertionError("gcsfs/ARCO must not be opened when the local subset exists")
    monkeypatch.setattr(era5, "open_era5", boom)
    lat2, lon2, z2, valid, src = era5.z500_for(t, (20, 22, -95, -93), str(tmp_path))
    assert src == path and valid == t
    assert np.allclose(z2, 5800.0) and z2.shape == (5, 7)
    assert np.allclose(lat2, lat) and np.allclose(lon2, lon)


def test_era5_diff_entry_missing_when_era5_import_fails(monkeypatch):
    import sys
    import types
    from wtune.post.recipes import synoptic
    monkeypatch.setitem(sys.modules, "wtune.post.recipes.era5", None)
    calls = []
    out = synoptic._era5_diff_entry(types.SimpleNamespace(np=None), "", "", None, None,
                                    lambda *a: calls.append(a) or "missing")
    assert out == "missing" and calls[0][:2] == ("syn_era5_diff", "ERA5 (ARCO)")
