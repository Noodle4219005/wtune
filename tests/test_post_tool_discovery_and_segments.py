"""post/tools.py tool discovery outside bin/wtune (WTUNE_POSTPROC_VIEW precedence, then the
repo's own postproc spack view and pixi env bin) and segmented `wtune run --out` dirs
(OUT/seg_<k>/wrfout_*) seen as one run by post/extract.py, post/caseinfo.py and post/compare.py.
All synthetic: empty files stand in for binaries and wrfout frames (listing never opens them)."""
import os

import pytest

from wtune.post import compare, extract, tools
from wtune.post.custom_api import Context
from wtune.post.recipes import paraview3d


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w"):
        pass
    return str(path)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A synthetic <repo> with nothing installed; PATH resolves nothing; env var unset."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(tools, "REPO_ROOT", str(root))
    monkeypatch.delenv("WTUNE_POSTPROC_VIEW", raising=False)
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    return root


def _repo_view_bin(root):
    return os.path.join(str(root), "envs", "postproc", ".spack-env", "view", "bin")


def _repo_pixi_bin(root):
    return os.path.join(str(root), "envs", "wtune", ".pixi", "envs", "default", "bin")


# ---------------------------------------------------------------------------------------------
# tool discovery
# ---------------------------------------------------------------------------------------------


def test_postproc_tool_then_path_then_pixi_bin(repo, monkeypatch):
    pixi = _touch(os.path.join(_repo_pixi_bin(repo), "nccmp"))
    assert tools.find_postproc_tool("nccmp") == pixi
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/" + name)
    assert tools.find_postproc_tool("nccmp") == "/usr/bin/nccmp"


def test_find_diffwrf_falls_back_to_repo_bins(repo, monkeypatch):
    monkeypatch.delenv("WRF_TREE", raising=False)
    want = _touch(os.path.join(_repo_pixi_bin(repo), "diffwrf"))
    assert tools.find_diffwrf() == want


def test_find_pvbatch_falls_back_to_postproc_view(repo, monkeypatch):
    monkeypatch.delenv("WTUNE_PARAVIEW_BIN", raising=False)
    want = _touch(os.path.join(_repo_view_bin(repo), "pvbatch"))
    assert paraview3d.find_pvbatch() == want


# ---------------------------------------------------------------------------------------------
# segmented run dirs
# ---------------------------------------------------------------------------------------------

NAMELIST = """&time_control
 run_hours = 2,
/
&domains
 max_dom = 2,
/
&bdy_control
 spec_bdy_width = 7,
/
"""


def _frame(d, dom, hh):
    return _touch(os.path.join(str(d), "wrfout_d%02d_2005-08-28_%02d:00:00" % (dom, hh)))


def _segmented(tmp_path):
    """o2/seg_0: 00..02 (d01, d02), o2/seg_1: 02..04 (the 02 boundary frame in both)."""
    out = tmp_path / "o2"
    seg0, seg1 = out / "seg_0", out / "seg_1"
    for hh in (0, 1, 2):
        _frame(seg0, 1, hh)
        _frame(seg0, 2, hh)
    for hh in (2, 3, 4):
        _frame(seg1, 1, hh)
        _frame(seg1, 2, hh)
    (seg0 / "namelist.input").write_text(NAMELIST)
    (seg1 / "namelist.input").write_text(NAMELIST.replace("run_hours = 2", "run_hours = 9"))
    (out / "run_report.json").write_text("{}")
    return out


def test_frame_path_and_compare_frames_resolve_into_segments(tmp_path):
    out = _segmented(tmp_path)
    assert extract.frame_path(str(out), 2, "2005-08-28_03:00:00") == str(
        out / "seg_1" / "wrfout_d02_2005-08-28_03:00:00")
    assert extract.frame_path(str(out), 1, "2005-08-28_07:00:00") == str(
        out / "wrfout_d01_2005-08-28_07:00:00")     # absent: the flat path, for the error
    frames = compare._frames(str(out))
    assert len(frames) == 10
    assert frames[(1, "2005-08-28_02:00:00")] == str(out / "seg_1" / "wrfout_d01_2005-08-28_02:00:00")
    ctx = Context(str(tmp_path), "s", str(tmp_path / "s_out"), [("o2", str(out))])
    assert ctx.finest_domain(str(out)) == 2


def test_find_run_track_csv_finds_the_file_post_tc_writes(tmp_path):
    _touch(tmp_path / "post" / "20261028100000-recipe" / "tc" / "track_domains_run72.csv")
    newer = _touch(tmp_path / "post" / "20261028174912-recipe" / "tc" / "track_domains_run72.csv")
    _touch(tmp_path / "post" / "20261028174912-recipe" / "tc" / "quadrant_run72.csv")
    ctx = Context(str(tmp_path), "s", str(tmp_path / "s_out"), [])
    assert ctx.find_run_track_csv("run72") == newer
    assert ctx.find_run_track_csv("run7") is None
