"""post/tools.py tool discovery outside bin/wtune (WTUNE_POSTPROC_VIEW precedence, then the
repo's own postproc spack view and pixi env bin) and segmented `wtune run --out` dirs
(OUT/seg_<k>/wrfout_*) seen as one run by post/extract.py, post/caseinfo.py and post/compare.py.
All synthetic: empty files stand in for binaries and wrfout frames (listing never opens them)."""
import os

import pytest

from post import caseinfo, compare, extract, tools
from post.custom_api import Context
from post.recipes import paraview3d


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

def test_postproc_tool_env_var_wins_over_repo_view_and_path(repo, tmp_path, monkeypatch):
    env_view = tmp_path / "envview"
    want = _touch(env_view / "bin" / "tc_pairs")
    _touch(os.path.join(_repo_view_bin(repo), "tc_pairs"))
    monkeypatch.setenv("WTUNE_POSTPROC_VIEW", str(env_view))
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/" + name)
    assert tools.find_postproc_tool("tc_pairs") == want


def test_postproc_tool_env_var_set_never_falls_back_to_repo_view(repo, tmp_path, monkeypatch):
    _touch(os.path.join(_repo_view_bin(repo), "tc_pairs"))
    monkeypatch.setenv("WTUNE_POSTPROC_VIEW", str(tmp_path / "envview"))
    with pytest.raises(tools.ToolNotFoundError) as e:
        tools.find_postproc_tool("tc_pairs")
    msg = str(e.value)
    assert os.path.join(str(tmp_path / "envview"), "bin", "tc_pairs") in msg
    assert _repo_view_bin(repo) not in msg


def test_postproc_tool_falls_back_to_repo_view_when_env_var_unset(repo):
    want = _touch(os.path.join(_repo_view_bin(repo), "grid_stat"))
    assert tools.find_postproc_tool("grid_stat") == want


def test_postproc_tool_then_path_then_pixi_bin(repo, monkeypatch):
    pixi = _touch(os.path.join(_repo_pixi_bin(repo), "nccmp"))
    assert tools.find_postproc_tool("nccmp") == pixi
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/" + name)
    assert tools.find_postproc_tool("nccmp") == "/usr/bin/nccmp"


def test_postproc_tool_error_lists_every_location(repo):
    with pytest.raises(tools.ToolNotFoundError) as e:
        tools.find_postproc_tool("point_stat")
    msg = str(e.value)
    for loc in (os.path.join(_repo_view_bin(repo), "point_stat"), "PATH",
                os.path.join(_repo_pixi_bin(repo), "point_stat")):
        assert loc in msg


def test_find_on_path_prefers_path_then_pixi_then_view(repo, monkeypatch):
    view_cdo = _touch(os.path.join(_repo_view_bin(repo), "cdo"))
    assert tools.find_on_path("cdo") == view_cdo
    pixi_cdo = _touch(os.path.join(_repo_pixi_bin(repo), "cdo"))
    assert tools.find_on_path("cdo") == pixi_cdo
    monkeypatch.setattr(tools.shutil, "which", lambda name: "/usr/bin/" + name)
    assert tools.find_on_path("cdo") == "/usr/bin/cdo"


def test_find_on_path_uses_env_view_bin(repo, tmp_path, monkeypatch):
    want = _touch(tmp_path / "envview" / "bin" / "ffmpeg")
    monkeypatch.setenv("WTUNE_POSTPROC_VIEW", str(tmp_path / "envview"))
    assert tools.find_on_path("ffmpeg") == want


def test_find_on_path_error_lists_every_location_and_hint(repo):
    with pytest.raises(tools.ToolNotFoundError) as e:
        tools.find_on_path("cdo", install_hint="envs/wtune/pixi.toml provides it.")
    msg = str(e.value)
    for loc in ("PATH", os.path.join(_repo_pixi_bin(repo), "cdo"),
                os.path.join(_repo_view_bin(repo), "cdo"), "envs/wtune/pixi.toml provides it."):
        assert loc in msg


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


def test_list_frames_merges_segments_later_segment_wins(tmp_path):
    out = _segmented(tmp_path)
    frames = extract.list_frames(str(out), domain=1)
    assert [t for t, _ in frames] == ["2005-08-28_%02d:00:00" % h for h in range(5)]
    paths = dict(frames)
    assert os.path.dirname(paths["2005-08-28_00:00:00"]) == str(out / "seg_0")
    assert os.path.dirname(paths["2005-08-28_02:00:00"]) == str(out / "seg_1")
    assert os.path.dirname(paths["2005-08-28_04:00:00"]) == str(out / "seg_1")
    assert all(os.path.isfile(p) for p in paths.values())
    assert len(extract.list_frames(str(out), domain=2)) == 5
    assert extract.list_frames(str(out), domain=3) == []


def test_segments_sort_numerically_and_rescue_wins_over_its_segment(tmp_path):
    out = tmp_path / "o2"
    for k in range(11):
        _frame(out / ("seg_%d" % k), 1, k)
    _frame(out / "seg_10", 1, 9)               # boundary frame: seg_10 must beat seg_9
    _frame(out / "seg_3", 1, 4)
    _frame(out / "seg_3_rescue", 1, 3)         # CFL rescue of seg_3 re-ran the same window
    _frame(out / "seg_3_rescue", 1, 4)         # seg_3 also wrote the 04 boundary...
    _frame(out / "seg_4", 1, 4)                # ...but seg_4 is still the later segment
    (out / "seg_x").mkdir()
    _touch(str(out / "seg_7.log"))
    assert [os.path.basename(p) for p in extract.segment_dirs(str(out))] == (
        ["seg_%d" % k for k in range(3)] + ["seg_3", "seg_3_rescue"]
        + ["seg_%d" % k for k in range(4, 11)])
    paths = dict(extract.list_frames(str(out), domain=1))
    assert os.path.basename(os.path.dirname(paths["2005-08-28_09:00:00"])) == "seg_10"
    assert os.path.basename(os.path.dirname(paths["2005-08-28_03:00:00"])) == "seg_3_rescue"
    assert os.path.basename(os.path.dirname(paths["2005-08-28_04:00:00"])) == "seg_4"


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


def test_caseinfo_reads_seg0_namelist_and_uses_frame_span(tmp_path):
    out = _segmented(tmp_path)
    assert caseinfo.find_namelist(str(out)) == str(out / "seg_0" / "namelist.input")
    assert caseinfo.spec_bdy_width(str(out)) == 7
    assert caseinfo.run_length_hours(str(out)) == pytest.approx(4.0)   # not seg_0's run_hours=2


def test_flat_run_dir_unchanged_and_ignores_seg_subdirs(tmp_path):
    flat = tmp_path / "flat"
    for hh in (0, 6):
        _frame(flat, 1, hh)
    _frame(flat / "seg_0", 1, 3)               # stray subdir: a flat dir never looks inside
    (flat / "namelist.input").write_text(NAMELIST)
    (flat / "seg_0" / "namelist.output").write_text(NAMELIST)
    assert extract.segment_dirs(str(flat)) == []
    assert extract.list_frames(str(flat), domain=1) == [
        ("2005-08-28_00:00:00", str(flat / "wrfout_d01_2005-08-28_00:00:00")),
        ("2005-08-28_06:00:00", str(flat / "wrfout_d01_2005-08-28_06:00:00"))]
    assert extract.list_frames(str(flat), domain=2) == []
    assert caseinfo.find_namelist(str(flat)) == str(flat / "namelist.input")
    assert caseinfo.run_length_hours(str(flat)) == pytest.approx(2.0)   # namelist run_hours


def test_empty_dir_has_no_frames_and_no_namelist(tmp_path):
    assert extract.list_frames(str(tmp_path), domain=1) == []
    assert extract.list_frames(str(tmp_path / "missing"), domain=1) == []
    with pytest.raises(caseinfo.CaseInfoError):
        caseinfo.find_namelist(str(tmp_path))
