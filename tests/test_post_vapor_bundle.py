"""post/vapor_bundle.py (+ post/recipes/vapor_bundle.py, `wtune post vapor-bundle`) and the
bundle's own vapor_render.py.

The bundle side needs no real wrfout content (frames are only linked/copied, never opened), so
the frames here are small byte files with real wrfout names. vapor_render.py is exercised
against a FAKE `vapor` package (session/dataset/renderer stubs that record every call): real
VAPOR has no linux-aarch64 build and is not installed here, so these tests pin the script's
call sequence and its defensive paths, not VAPOR's behaviour.
"""
import importlib.util
import json
import os
import py_compile
import sys
import types

import pytest

from wtune.post import vapor_bundle as vb

TIMES = ("2005-08-28_00:00:00", "2005-08-28_06:00:00", "2005-08-28_12:00:00")


def _run_dir(tmp_path, domains=(1,)):
    run = tmp_path / "run"
    run.mkdir()
    for d in domains:
        for k, t in enumerate(TIMES):
            (run / ("wrfout_d%02d_%s" % (d, t))).write_bytes(b"x" * (1000 + k))
    return run


def test_bundle_layout_names_and_manifest(tmp_path):
    run = _run_dir(tmp_path)
    r = vb.build_bundle(str(run), str(tmp_path / "out"), 1, every="6h", run_label="base")
    bdir = tmp_path / "out" / "vapor" / "d01"
    assert r["dir"] == str(bdir)
    names = sorted(os.listdir(bdir))
    assert names == ["README.md", "bundle.json", "vapor_render.py",
                     "wrfout_d01_2005-08-28_00_00_00", "wrfout_d01_2005-08-28_06_00_00",
                     "wrfout_d01_2005-08-28_12_00_00"]
    assert not any(":" in n for n in names)
    assert r["link_note"] == "hard-linked (no extra space)" and r["copied_bytes"] == 0
    assert os.path.samefile(bdir / "wrfout_d01_2005-08-28_06_00_00",
                            run / "wrfout_d01_2005-08-28_06:00:00")

    m = json.loads((bdir / "bundle.json").read_text())
    assert m["run_dir"] == str(run) and m["domain"] == 1 and m["selection"] == "--every 6h"
    assert [f["valid_time"] for f in m["frames"]] == list(TIMES)
    f0 = m["frames"][0]
    assert f0["original"] == str(run / "wrfout_d01_2005-08-28_00:00:00")
    assert f0["name"] == "wrfout_d01_2005-08-28_00_00_00" and f0["size_bytes"] == 1000
    assert f0["mode"] == "hardlink" and "sha256" not in f0
    assert m["total_bytes"] == 3003 and m["wtune_version"] and m["created_utc"]

    readme = (bdir / "README.md").read_text(encoding="utf-8")
    assert "rsync -av scc:%s/ ./" % bdir in readme
    assert "conda create -n vapor -c conda-forge -c ncar-vapor vapor" in readme
    assert "linux-aarch64" in readme and "SetTimestep" in readme and "xvfb-run -a" in readme
    assert "Times" in readme                       # the time-from-Times-not-name assumption


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


def test_no_copy_without_allow_copy_and_nothing_left(tmp_path, monkeypatch):
    run = _run_dir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    _no_link(monkeypatch)
    with pytest.raises(vb.NotHardLinkable) as ei:
        vb.build_bundle(str(run), str(out), 1, every="6h")
    msg = str(ei.value)
    assert "not hard-linkable here (Invalid cross-device link)" in msg
    assert "--allow-copy` to copy 3 files (~0.0 MB)" in msg and "--every 6h" in msg
    assert os.listdir(out) == []                   # vapor/ and vapor/d01/ removed again


def test_refused_relink_keeps_an_earlier_bundle(tmp_path, monkeypatch):
    run = _run_dir(tmp_path)
    out = tmp_path / "out"
    vb.build_bundle(str(run), str(out), 1)
    before = _tree(out)
    _no_link(monkeypatch)
    with pytest.raises(vb.NotHardLinkable):          # different selection, first frame refused
        vb.build_bundle(str(run), str(out), 1, times=["2005082800"])
    assert _tree(out) == before


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


def test_cli_vapor_bundle(tmp_path, capsys):
    from wtune.post import cli
    run = _run_dir(tmp_path)
    out = tmp_path / "cli"
    assert cli.main(["vapor-bundle", str(run), "--every", "12h", "--out", str(out)]) == 0
    assert sorted(n for n in os.listdir(out / "vapor" / "d01") if n.startswith("wrfout")) == \
        ["wrfout_d01_2005-08-28_00_00_00", "wrfout_d01_2005-08-28_12_00_00"]
    assert "rsync -av" in capsys.readouterr().out
    assert cli.main(["vapor-bundle", str(run), "--domain", "4", "--out", str(out)]) == 1


def test_cli_vapor_bundle_copies_only_with_allow_copy(tmp_path, monkeypatch, capsys):
    from wtune.post import cli
    run = _run_dir(tmp_path)
    _no_link(monkeypatch)
    out = tmp_path / "cli"
    assert cli.main(["vapor-bundle", str(run), "--out", str(out)]) == 1
    assert "--allow-copy` to copy 1 files" in capsys.readouterr().out
    assert not out.exists()                        # even the --out dir it created is gone
    assert cli.main(["vapor-bundle", str(run), "--out", str(out), "--allow-copy"]) == 0
    assert "copied (" in capsys.readouterr().out
    assert (out / "vapor" / "d01" / "wrfout_d01_2005-08-28_12_00_00").exists()


# -- vapor_render.py against a fake `vapor` package ------------------------------------------

def test_vapor_render_compiles():
    py_compile.compile(vb.SCRIPT_SRC, doraise=True)


class _Rec:
    def __init__(self, calls, label):
        self.calls, self.label = calls, label

    def __getattr__(self, name):
        def call(*a):
            self.calls.append((self.label, name) + a)
        return call


def _fake_vapor(monkeypatch, names, n_steps, timestep_raises=False):
    calls = []

    class Data:
        def GetDataVarNames(self):
            return list(names)

        def GetNumTimeSteps(self, var):
            calls.append(("data", "GetNumTimeSteps", var))
            return n_steps

        def NewRenderer(self, kind):
            calls.append(("data", "NewRenderer", kind))
            r = _Rec(calls, kind)
            r.GetPrimaryTransferFunction = lambda: _Rec(calls, kind + ".tf")
            return r

    class Camera(_Rec):
        def GetPosition(self):
            return [0.0, 0.0, 10.0]

        def GetTarget(self):
            return [0.0, 0.0, 0.0]

    class Session:
        def OpenDataset(self, kind, files):
            calls.append(("ses", "OpenDataset", kind, list(files)))
            return Data()

        def GetCamera(self):
            return Camera(calls, "cam")

        def SetResolution(self, w, h):
            calls.append(("ses", "SetResolution", w, h))

        def SetTimestep(self, i):
            calls.append(("ses", "SetTimestep", i))
            if timestep_raises:
                raise AttributeError("no SetTimestep")

        def Render(self, path):
            calls.append(("ses", "Render", path))
            open(path, "w").write("png")

        def Save(self, path):
            calls.append(("ses", "Save", path))
            open(path, "w").write("vs3")

    pkg = types.ModuleType("vapor")
    pkg.session = types.SimpleNamespace(Session=Session)
    pkg.dataset = types.SimpleNamespace(WRF="WRF")
    pkg.renderer = types.SimpleNamespace(ImageRenderer="ImageRenderer",
                                         VolumeIsoRenderer="VolumeIsoRenderer")
    monkeypatch.setitem(sys.modules, "vapor", pkg)
    return calls


def _bundle_script(tmp_path):
    run = _run_dir(tmp_path)
    r = vb.build_bundle(str(run), str(tmp_path / "out"), 1, every="6h")
    spec = importlib.util.spec_from_file_location("vapor_render_under_test", r["script"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, r


NAMES = ("HGT", "QCLOUD", "QICE", "T")


def test_vapor_render_call_sequence(tmp_path, monkeypatch, capsys):
    calls = _fake_vapor(monkeypatch, NAMES, 3)
    mod, r = _bundle_script(tmp_path)
    png_dir = tmp_path / "png"
    assert mod.main(["--out", str(png_dir), "--width", "800", "--height", "600"]) == 0
    open_call = [c for c in calls if c[1] == "OpenDataset"][0]
    assert open_call[2] == "WRF" and open_call[3] == sorted(
        os.path.join(r["dir"], f["name"]) for f in r["frames"])
    assert ("ses", "SetResolution", 800, 600) in calls
    assert ("ImageRenderer", "SetHeightVariableName", "HGT") in calls
    assert ("VolumeIsoRenderer", "SetVariableName", "QCLOUD") in calls
    assert ("VolumeIsoRenderer", "SetIsoValues", [1e-4]) in calls
    assert ("VolumeIsoRenderer", "SetVariableName", "QICE") in calls
    assert ("cam", "ViewAll") in calls
    assert [c[2] for c in calls if c[1] == "SetTimestep"] == [0, 1, 2]
    renders = [os.path.basename(c[2]) for c in calls if c[1] == "Render"]
    assert renders == ["wrfout_d01_2005-08-28_%s_00_00.png" % h for h in ("00", "06", "12")]
    assert calls[-1] == ("ses", "Save", str(png_dir / "wtune_session.vs3"))
    out = capsys.readouterr().out
    assert "rendered: 3 PNG(s)" in out and "every unverified call succeeded" in out


def test_vapor_render_list_and_session_only(tmp_path, monkeypatch, capsys):
    calls = _fake_vapor(monkeypatch, NAMES, 3)
    mod, _r = _bundle_script(tmp_path)
    assert mod.main(["--list"]) == 0
    assert not [c for c in calls if c[1] in ("NewRenderer", "Render", "Save")]
    assert "variables: HGT, QCLOUD, QICE, T" in capsys.readouterr().out

    del calls[:]
    assert mod.main(["--session-only", "--out", str(tmp_path / "s")]) == 0
    assert not [c for c in calls if c[1] == "Render"]
    assert ("ses", "Save", str(tmp_path / "s" / "wtune_session.vs3")) in calls


def test_vapor_render_qice_absent_and_settimestep_failure_only_warn(tmp_path, monkeypatch,
                                                                   capsys):
    calls = _fake_vapor(monkeypatch, ("HGT", "QCLOUD"), 3, timestep_raises=True)
    mod, _r = _bundle_script(tmp_path)
    assert mod.main(["--out", str(tmp_path / "png")]) == 0
    assert ("VolumeIsoRenderer", "SetVariableName", "QICE") not in calls
    assert len([c for c in calls if c[1] == "Render"]) == 3          # rendering went on
    out = capsys.readouterr().out
    assert "skipped: QICE isosurface" in out
    assert out.count("unverified call failed (continued without it): SetTimestep") == 3


def test_vapor_render_without_vapor_installed(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "vapor", None)
    mod, _r = _bundle_script(tmp_path)
    assert mod.main([]) == 3
    assert "conda activate vapor" in capsys.readouterr().out
