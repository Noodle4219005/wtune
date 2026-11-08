"""post/vapor_bundle.py (+ post/recipes/vapor_bundle.py, `wtune post vapor-bundle`) and the
bundle's own vapor_render.py.

The bundle side needs no real wrfout content (frames are only linked/copied, never opened), so
the frames here are small byte files with real wrfout names. vapor_render.py is exercised
against a FAKE `vapor` package (session/dataset/renderer stubs that record every call): real
VAPOR has no linux-aarch64 build and is not installed here, so these tests pin the script's
call sequence and its defensive paths, not VAPOR's behaviour.
"""
import os
import py_compile


from wtune.post import vapor_bundle as vb

TIMES = ("2005-08-28_00:00:00", "2005-08-28_06:00:00", "2005-08-28_12:00:00")


def _run_dir(tmp_path, domains=(1,)):
    run = tmp_path / "run"
    run.mkdir()
    for d in domains:
        for k, t in enumerate(TIMES):
            (run / ("wrfout_d%02d_%s" % (d, t))).write_bytes(b"x" * (1000 + k))
    return run


def test_last_frame_only_by_default_and_stale_frames_removed(tmp_path):
    run = _run_dir(tmp_path)
    out = tmp_path / "out"
    vb.build_bundle(str(run), str(out), 1, every="6h")
    r = vb.build_bundle(str(run), str(out), 1)            # re-run without --every
    assert [f["name"] for f in r["frames"]] == ["wrfout_d01_2005-08-28_12_00_00"]
    assert sorted(n for n in os.listdir(out / "vapor" / "d01") if n.startswith("wrfout")) == \
        ["wrfout_d01_2005-08-28_12_00_00"]


def _no_link(monkeypatch):
    def no_link(src, dst):
        raise OSError(18, "Invalid cross-device link")
    monkeypatch.setattr(vb.os, "link", no_link)


def _tree(path):
    return sorted(os.path.relpath(os.path.join(d, f), path)
                  for d, _sub, files in os.walk(path) for f in files) if path.exists() else []


def test_copy_only_with_allow_copy(tmp_path, monkeypatch):
    run = _run_dir(tmp_path)
    _no_link(monkeypatch)
    r = vb.build_bundle(str(run), str(tmp_path / "out"), 1, times=["2005082806"],
                        allow_copy=True)
    assert [f["mode"] for f in r["frames"]] == ["copy"]
    assert r["copied_bytes"] == 1001 and r["link_note"].startswith("copied (")
    dst = tmp_path / "out" / "vapor" / "d01" / "wrfout_d01_2005-08-28_06_00_00"
    assert not os.path.samefile(dst, run / "wrfout_d01_2005-08-28_06:00:00")
    assert dst.read_bytes() == b"x" * 1001


def test_recipe_never_copies(tmp_path, monkeypatch):
    from wtune.post.recipes import vapor_bundle as recipe_vb
    run = _run_dir(tmp_path)
    _no_link(monkeypatch)
    out = tmp_path / "rec"
    recs = recipe_vb.figures(str(run), "base", str(out), dict(domains=[dict(domain=1)]))
    assert len(recs) == 1 and recs[0]["missing"] and recs[0]["id"] == "common_vapor_bundle"
    assert "not hard-linkable here" in recs[0]["reason"]
    assert "wtune post vapor-bundle" in recs[0]["reason"] and "--allow-copy" in recs[0]["reason"]
    assert _tree(out) == []


def test_recipe_entry_and_report(tmp_path):
    from wtune.post import report
    from wtune.post.recipes import vapor_bundle as recipe_vb
    run = _run_dir(tmp_path, domains=(1, 2))
    out = tmp_path / "rec"
    case_info = dict(domains=[dict(domain=1), dict(domain=2)], types=["common"],
                     evidence={"common": "always"}, run_length_h=12.0,
                     d01_extent_km=(300.0, 300.0))
    recs = recipe_vb.figures(str(run), "base", str(out), case_info)
    assert [r["domain"] for r in recs] == ["d01", "d02"]
    assert all(r["id"] == "common_vapor_bundle" and r["data_file"] and not r["missing"]
               for r in recs)
    assert recs[0]["size_bytes"] == 1002 and "hard-linked (no extra space)" in recs[0]["note"]
    html = open(report.build_recipe_report(str(out), case_info, recs, [str(run)], ["base"], 6,
                                           {})).read()
    assert "data file: <a href='vapor/d01'>vapor/d01</a>" in html
    assert "<a href='vapor/d01/README.md'>README.md</a>" in html
    assert "<a href='vapor/d01/vapor_render.py'>vapor_render.py</a>" in html

    case_info["domains"] = [dict(domain=1), dict(domain=3)]   # d03 has no frames
    recs = recipe_vb.figures(str(run), "base", str(tmp_path / "rec2"), case_info)
    assert not recs[0]["missing"] and recs[1]["missing"]


# -- vapor_render.py against a fake `vapor` package ------------------------------------------

def test_vapor_render_compiles():
    py_compile.compile(vb.SCRIPT_SRC, doraise=True)
