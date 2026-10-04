"""Tests for post/compare.py: diffwrf table parsing (against real captured output) and the
compare_runs orchestration with diffwrf mocked via post.tools.run.
"""
import os

import pytest

import post.compare as compare
from post import tools

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "post")
DIFFWRF_IDENTICAL = open(os.path.join(FIXTURES, "diffwrf_identical.txt")).read()
DIFFWRF_DIFFERENT = open(os.path.join(FIXTURES, "diffwrf_different.txt")).read()
NCDUMP_H_TEXT = open(os.path.join(FIXTURES, "ncdump_h_sample.txt")).read()


def test_parse_diffwrf_identical_capture_has_no_rows():
    assert compare.parse_diffwrf(DIFFWRF_IDENTICAL) == []


def test_parse_diffwrf_different_capture():
    rows = compare.parse_diffwrf(DIFFWRF_DIFFERENT)
    assert len(rows) == 78
    u = next(r for r in rows if r["field"] == "U")
    assert u["ndifs"] == 832103
    assert u["dims"] == 3
    assert u["rms1"] == pytest.approx(8.162436562)
    assert u["digits"] == 3
    assert u["rmse"] == pytest.approx(0.5795)
    assert u["pntwise_max"] == pytest.approx(0.4923)


def _touch(path):
    open(path, "w").close()


def _mock_diffwrf(monkeypatch, table_text):
    monkeypatch.setattr(tools, "find_diffwrf", lambda: "/fake/diffwrf")
    monkeypatch.setattr(tools, "run", lambda cmd, cwd=None, check=True: (0, table_text, ""))


def test_compare_identical_runs(tmp_path, monkeypatch):
    _mock_diffwrf(monkeypatch, DIFFWRF_IDENTICAL)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _touch(a / "wrfout_d01_2005-08-27_00:00:00")
    _touch(b / "wrfout_d01_2005-08-27_00:00:00")

    result = compare.compare_runs(str(a), str(b))
    assert result["identical"] is True
    assert result["n_different"] == 0
    assert "IDENTICAL" in compare.render_summary(result)


def test_compare_different_runs(tmp_path, monkeypatch):
    _mock_diffwrf(monkeypatch, DIFFWRF_DIFFERENT)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _touch(a / "wrfout_d01_2005-08-27_00:00:00")
    _touch(b / "wrfout_d01_2005-08-27_00:00:00")

    result = compare.compare_runs(str(a), str(b))
    assert result["identical"] is False
    assert result["n_different"] == 78
    summary = compare.render_summary(result)
    assert "DIFFERENT (78 vars)" in summary


def test_compare_vars_filter(tmp_path, monkeypatch):
    _mock_diffwrf(monkeypatch, DIFFWRF_DIFFERENT)
    monkeypatch.setattr(tools, "ncdump_header", lambda path: NCDUMP_H_TEXT)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _touch(a / "wrfout_d01_2005-08-27_00:00:00")
    _touch(b / "wrfout_d01_2005-08-27_00:00:00")

    result = compare.compare_runs(str(a), str(b), variables=["T2"])
    assert result["n_different"] == 1
    assert result["results"][0]["field"] == "T2"


def test_compare_unknown_var_raises(tmp_path, monkeypatch):
    _mock_diffwrf(monkeypatch, DIFFWRF_DIFFERENT)
    monkeypatch.setattr(tools, "ncdump_header", lambda path: NCDUMP_H_TEXT)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _touch(a / "wrfout_d01_2005-08-27_00:00:00")
    _touch(b / "wrfout_d01_2005-08-27_00:00:00")

    with pytest.raises(compare.CompareError):
        compare.compare_runs(str(a), str(b), variables=["NOT_A_REAL_VAR"])


def test_compare_no_common_frames(tmp_path, monkeypatch):
    _mock_diffwrf(monkeypatch, DIFFWRF_IDENTICAL)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _touch(a / "wrfout_d01_2005-08-27_00:00:00")
    _touch(b / "wrfout_d01_2005-08-27_06:00:00")

    result = compare.compare_runs(str(a), str(b))
    assert result["results"] == []
    assert "no common" in compare.render_summary(result)


def test_compare_adios2_bp_input_raises_clear_error(tmp_path, monkeypatch):
    from test_post_inventory import _no_bp_converter
    _no_bp_converter(monkeypatch)
    monkeypatch.setattr(tools, "find_diffwrf", lambda: "/fake/diffwrf")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "wrfout_d01_2005-08-27_00:00:00.bp").mkdir()
    (a / "wrfout_d01_2005-08-27_00:00:00.bp").joinpath("md.idx").touch()        # an empty dir is not a frame
    _touch(b / "wrfout_d01_2005-08-27_00:00:00")

    with pytest.raises(compare.CompareError, match="wrf_adios2_to_netcdf4"):
        compare.compare_runs(str(a), str(b))
