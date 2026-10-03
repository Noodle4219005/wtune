"""wtune_bp behaviour that needs no adios2 module: BP detection, manifest caching (converter
monkeypatched), adios2.xml, CLI wiring, the missing-module error, and the netCDF comparator."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import wtune_bp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = b"CDF\x01" + b"\0" * 60
HDF = b"\x89HDF\r\n\x1a\n" + b"\0" * 56


def _mkbp(path, engine="BP5"):
    os.makedirs(path)
    files = ["md.0", "md.idx", "data.0"] + (["mmd.0"] if engine == "BP5" else [])
    for f in files:
        open(os.path.join(path, f), "w").close()
    return str(path)


# ---------------------------------------------------------------- detection
def test_is_bp_engines_and_negatives(tmp_path):
    assert wtune_bp.bp_engine(_mkbp(tmp_path / "a", "BP5")) == "BP5"
    assert wtune_bp.bp_engine(_mkbp(tmp_path / "b", "BP4")) == "BP4"
    assert wtune_bp.is_bp(str(tmp_path / "a")) and wtune_bp.is_bp(str(tmp_path / "b"))
    os.symlink(tmp_path / "a", tmp_path / "link")
    assert wtune_bp.is_bp(str(tmp_path / "link"))
    nodata = tmp_path / "nodata"
    nodata.mkdir()
    (nodata / "md.idx").touch()
    (nodata / "md.0").touch()
    assert not wtune_bp.is_bp(str(nodata))
    (tmp_path / "f.nc").write_bytes(NC)
    assert not wtune_bp.is_bp(str(tmp_path / "f.nc"))
    assert not wtune_bp.is_bp(str(tmp_path / "missing"))


def test_is_netcdf_and_find_input_names(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    for n in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "wrflowinp_d01"):
        (case / n).write_bytes(NC)
    (case / "wrffdda_d01").write_bytes(HDF)
    (case / "wrfout_d01_2019-11-26_12:00:00").write_bytes(NC)
    (case / "namelist.input").write_text("&time_control\n/\n")
    (case / "wrfinput_d03").write_text("not netcdf")
    _mkbp(case / "wrfinput_d01.bp")
    _mkbp(case / "wrfbdy_d02")  # a BP dir with an input name is not a netCDF source
    assert wtune_bp.find_input_names(str(case)) == [
        "wrfbdy_d01", "wrffdda_d01", "wrfinput_d01", "wrfinput_d02", "wrflowinp_d01"]


# ---------------------------------------------------------------- cached conversion
FAKE_DATES = ["2026-01-01_00:00:00", "2026-01-01_06:00:00"]


@pytest.fixture
def fake_conv(monkeypatch):
    calls = []

    def fake(src, dst, engine="BP5", split_prefix=None):
        calls.append((src, os.path.basename(dst), engine))
        if split_prefix is not None:          # per-record: a plain dir of <prefix>_<date>.bp
            os.makedirs(dst)
            for d in FAKE_DATES:
                rec = os.path.join(dst, "%s_%s.bp" % (split_prefix, d))
                _mkbp(rec, engine)
                with open(os.path.join(rec, "data.0"), "wb") as f:
                    f.write(open(src, "rb").read())
            return dst
        _mkbp(dst, engine)
        with open(os.path.join(dst, "data.0"), "wb") as f:
            f.write(open(src, "rb").read())
        return dst
    monkeypatch.setattr(wtune_bp, "nc2bp", fake)
    return calls


def test_convert_inputs_caches_and_reconverts_only_on_change(tmp_path, fake_conv, monkeypatch):
    case, cache = tmp_path / "case", tmp_path / "cache"
    case.mkdir()
    (case / "wrfinput_d01").write_bytes(NC + b"A")
    (case / "wrfbdy_d01").write_bytes(NC + b"B")

    m = wtune_bp.convert_inputs(str(case), str(cache))
    assert m == {"wrfbdy_d01": str(cache / "wrfbdy_d01"), "wrfinput_d01": str(cache / "wrfinput_d01")}
    assert len(fake_conv) == 2 and wtune_bp.is_bp(m["wrfinput_d01"])
    assert wtune_bp.is_per_record_dir(m["wrfbdy_d01"])
    man = json.loads((cache / "bp_manifest.json").read_text())["entries"]
    e = man["wrfinput_d01"]
    assert e["source"] == os.path.realpath(case / "wrfinput_d01")
    assert e["size"] == len(NC) + 1 and e["engine"] == "BP4"   # DEFAULT_ENGINE: WRF cannot read BP5 inputs (scc 2026-10-03)
    assert e["converter_version"] == wtune_bp.CONVERTER_VERSION
    assert e["sha256"] == wtune_bp.sha256_file(str(case / "wrfinput_d01"))
    assert e["bp_path"] == str(cache / "wrfinput_d01")
    assert not [f for f in os.listdir(cache) if ".tmp-" in f]

    # unchanged -> nothing reconverted
    wtune_bp.convert_inputs(str(case), str(cache))
    assert len(fake_conv) == 2

    # touched but identical bytes -> no reconversion, manifest mtime refreshed
    st = os.stat(case / "wrfinput_d01")
    os.utime(case / "wrfinput_d01", (st.st_atime, st.st_mtime + 100))
    wtune_bp.convert_inputs(str(case), str(cache))
    assert len(fake_conv) == 2
    man = json.loads((cache / "bp_manifest.json").read_text())["entries"]
    assert man["wrfinput_d01"]["mtime"] == st.st_mtime + 100

    # same size, different content -> only that file reconverted, BP replaced
    (case / "wrfinput_d01").write_bytes(NC + b"Z")
    wtune_bp.convert_inputs(str(case), str(cache))
    assert [c[1] for c in fake_conv[2:]] != [] and len(fake_conv) == 3
    assert open(cache / "wrfinput_d01" / "data.0", "rb").read().endswith(b"Z")

    # engine change, converter version change, missing BP -> reconvert
    wtune_bp.convert_inputs(str(case), str(cache), names=["wrfbdy_d01"], engine="BP5")
    assert fake_conv[-1][2] == "BP5" and wtune_bp.bp_engine(
        str(cache / "wrfbdy_d01" / ("wrfbdy_d01_%s.bp" % FAKE_DATES[0]))) == "BP5"
    monkeypatch.setattr(wtune_bp, "CONVERTER_VERSION", "test-bump")
    wtune_bp.convert_inputs(str(case), str(cache), names=["wrfinput_d01"])
    assert len(fake_conv) == 5
    import shutil
    shutil.rmtree(cache / "wrfinput_d01")
    wtune_bp.convert_inputs(str(case), str(cache), names=["wrfinput_d01"])
    assert len(fake_conv) == 6
    assert not [f for f in os.listdir(cache) if ".tmp-" in f or f.endswith(".old")]


def test_convert_inputs_failure_leaves_no_partial_output(tmp_path, monkeypatch):
    case, cache = tmp_path / "case", tmp_path / "cache"
    case.mkdir()
    (case / "wrfinput_d01").write_bytes(NC)

    def boom(src, dst, engine="BP5"):
        _mkbp(dst)
        raise RuntimeError("converter died")
    monkeypatch.setattr(wtune_bp, "nc2bp", boom)
    with pytest.raises(RuntimeError):
        wtune_bp.convert_inputs(str(case), str(cache))
    assert [f for f in os.listdir(cache) if not f.startswith(".bp_manifest.lock")] == []


def test_convert_inputs_explicit_names_validated(tmp_path, fake_conv):
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text("x")
    with pytest.raises(FileNotFoundError):
        wtune_bp.convert_inputs(str(case), str(tmp_path / "c"), names=["wrfinput_d01"])
    with pytest.raises(ValueError):
        wtune_bp.convert_inputs(str(case), str(tmp_path / "c"), names=["namelist.input"])
    with pytest.raises(ValueError):
        wtune_bp.convert_inputs(str(case), str(tmp_path / "c"), engine="BP3")


# ---------------------------------------------------------------- adios2 interpreter
@pytest.fixture
def no_adios2(monkeypatch, tmp_path):
    """No envs/adios2 python, nothing on PATH, no WTUNE_ADIOS2_PYTHON."""
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.delenv("WTUNE_ADIOS2_PYTHON", raising=False)
    monkeypatch.setattr(wtune_bp, "_ADIOS2_ENV_PY", str(tmp_path / "no-such-python"))
    monkeypatch.setattr(wtune_bp, "_IMPORT_OK", {})
    monkeypatch.setenv("PATH", str(empty))


FAKE_TOOL = r"""
import os, sys
tool, args = sys.argv[1], sys.argv[2:]
if os.environ.get("PYTHONPATH") or os.environ.get("LD_LIBRARY_PATH"):
    sys.exit("caller env leaked into the adios2 child")
if tool.endswith("wrf_nc2bp.py"):
    dst = args[1]
    os.makedirs(dst)
    if os.environ.get("FAKE_FAIL"):
        sys.exit("boom: partial output left behind")
    eng = args[args.index("--engine") + 1]
    for f in ["md.0", "md.idx", "data.0"] + (["mmd.0"] if eng == "BP5" else []):
        open(os.path.join(dst, f), "w").close()
else:
    assert tool.endswith(os.path.join("wrf_adios2_to_netcdf4", "convert.py")), tool
    out = args[args.index("--output") + 1]
    open(out, "wb").write(b"CDF\x01" + open(os.path.join(args[args.index("--input") + 1], "data.0"), "rb").read())
    if os.environ.get("FAKE_FAIL"):
        sys.exit("boom")
"""


def _fake_python(tmp_path, import_ok=True):
    tool = tmp_path / "fake_tool.py"
    tool.write_text(FAKE_TOOL)
    py = tmp_path / ("fakepy" if import_ok else "badpy")
    py.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit %d; fi\nexec "%s" "%s" "$@"\n'
                  % (0 if import_ok else 1, sys.executable, tool))
    py.chmod(0o755)
    return str(py)


def test_require_adios2_python_message(no_adios2):
    assert wtune_bp.adios2_python() is None
    with pytest.raises(wtune_bp.Adios2Unavailable) as ei:
        wtune_bp.require_adios2_python()
    msg = str(ei.value)
    assert "pixi install" in msg and os.path.join("envs", "adios2") in msg


def test_adios2_python_resolution_order(no_adios2, monkeypatch, tmp_path):
    good, bad = _fake_python(tmp_path), _fake_python(tmp_path, import_ok=False)
    monkeypatch.setattr(wtune_bp, "_ADIOS2_ENV_PY", good)
    assert wtune_bp.require_adios2_python() == good
    # an explicit WTUNE_ADIOS2_PYTHON wins and never falls back
    monkeypatch.setenv("WTUNE_ADIOS2_PYTHON", bad)
    assert wtune_bp.adios2_python() is None
    with pytest.raises(wtune_bp.Adios2Unavailable) as ei:
        wtune_bp.require_adios2_python()
    assert "WTUNE_ADIOS2_PYTHON" in str(ei.value)
    # PATH fallback
    monkeypatch.delenv("WTUNE_ADIOS2_PYTHON")
    monkeypatch.setattr(wtune_bp, "_ADIOS2_ENV_PY", bad)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    os.symlink(good, bindir / "python")
    monkeypatch.setenv("PATH", str(bindir))
    assert wtune_bp.adios2_python() == str(bindir / "python")


def test_converters_run_in_adios2_subprocess(no_adios2, monkeypatch, tmp_path):
    monkeypatch.setenv("WTUNE_ADIOS2_PYTHON", _fake_python(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/should/not/reach/child")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/spack/adios2/lib")
    src = tmp_path / "wrfinput_d01"
    src.write_bytes(NC)
    for eng in ("BP4", "BP5"):
        bp = str(tmp_path / ("w.%s" % eng))
        assert wtune_bp.nc2bp(str(src), bp, engine=eng) == bp
        assert wtune_bp.bp_engine(bp) == eng
    nc = str(tmp_path / "w.nc")
    assert wtune_bp.bp2nc(str(tmp_path / "w.BP5"), nc) == nc and wtune_bp.is_netcdf(nc)
    monkeypatch.setenv("FAKE_FAIL", "1")
    with pytest.raises(wtune_bp.Adios2ConversionError) as ei:
        wtune_bp.nc2bp(str(src), str(tmp_path / "f.bp"))
    assert "boom" in str(ei.value)
    with pytest.raises(wtune_bp.Adios2ConversionError):
        wtune_bp.bp2nc(str(tmp_path / "w.BP5"), str(tmp_path / "f.nc"))
    assert not (tmp_path / "f.bp").exists() and not (tmp_path / "f.nc").exists()
    assert not [f for f in os.listdir(tmp_path) if ".tmp-" in f]
    assert "adios2" not in sys.modules


def test_converters_need_adios2_but_check_args_first(tmp_path, no_adios2):
    (tmp_path / "x.nc").write_bytes(NC)
    _mkbp(tmp_path / "x.bp")
    with pytest.raises(FileExistsError):
        wtune_bp.nc2bp(str(tmp_path / "x.nc"), str(tmp_path / "x.bp"))
    with pytest.raises(ValueError):
        wtune_bp.nc2bp(str(tmp_path / "x.nc"), str(tmp_path / "y.bp"), engine="HDF5")
    with pytest.raises(wtune_bp.Adios2Unavailable):
        wtune_bp.nc2bp(str(tmp_path / "x.nc"), str(tmp_path / "y.bp"))
    with pytest.raises(wtune_bp.Adios2Unavailable):
        wtune_bp.bp2nc(str(tmp_path / "x.bp"), str(tmp_path / "y.nc"))
    assert not (tmp_path / "y.bp").exists() and not (tmp_path / "y.nc").exists()


# ---------------------------------------------------------------- adios2.xml
def test_adios2_xml():
    root = ET.fromstring(wtune_bp.adios2_xml())
    assert root.tag == "adios-config" and root.findall("io") == []

    names = wtune_bp.restart_io_names(["2019-11-26_19:00:00"], 2)
    assert names == ["wrfrst_d01_2019-11-26_19:00:00", "wrfrst_d02_2019-11-26_19:00:00"]
    text = wtune_bp.adios2_xml(names + ["wrfout_d01_2019-11-26_19:00:00"])
    ios = ET.fromstring(text).findall("io")
    assert [(i.get("name"), i.find("engine").get("type")) for i in ios] == [(n, "BP4") for n in names]
    # same entry text as wtune_run.render_adios2_restart_xml / strict_campaign.sbatch
    assert '  <io name="wrfrst_d01_2019-11-26_19:00:00">\n    <engine type="BP4"/>\n  </io>\n' in text

    text = wtune_bp.adios2_xml({"wrfout_d01_x": "BP5", "wrfrst_d01_x": None, "wrfinput_d01": None})
    got = {i.get("name"): i.find("engine").get("type") for i in ET.fromstring(text).findall("io")}
    assert got == {"wrfout_d01_x": "BP5", "wrfrst_d01_x": "BP4"}
    with pytest.raises(ValueError):
        wtune_bp.adios2_xml({"wrfrst_d01_x": "SST"})
    assert wtune_bp.ADIOS2_IO_FORM == 14


# ---------------------------------------------------------------- comparator (netCDF4 only)
def _nc(path, u, times=("2026-01-01_00:00:00", "2026-01-01_01:00:00"), attr="m s-1"):
    from netCDF4 import Dataset
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("x", 3)
        ds.setncattr("DX", np.float32(3000.0))
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[:, :] = np.asarray([list(s) for s in times], dtype="S1")
        v = ds.createVariable("U", "f4", ("Time", "x"))
        v[:] = u
        v.setncattr("units", attr)
        i = ds.createVariable("I", "i4", ("Time", "x"))
        i[:] = np.arange(6, dtype=np.int32).reshape(2, 3)


def test_compare_netcdf(tmp_path):
    pytest.importorskip("netCDF4")
    base = np.array([[1, np.nan, 3], [4, 5, 6]], np.float32)
    a, b = str(tmp_path / "a.nc"), str(tmp_path / "b.nc")
    _nc(a, base)
    _nc(b, base.copy())
    r = wtune_bp.compare_netcdf(a, b)
    assert r["ok"] and r["vars_compared"] == 3 and r["max_abs_diff"] == 0.0

    c = str(tmp_path / "c.nc")
    d2 = base.copy()
    d2[1, 2] += 0.5
    _nc(c, d2)
    r = wtune_bp.compare_netcdf(a, c)
    assert not r["ok"] and r["max_abs_diff"] == 0.5 and r["mismatches"][0].startswith("U:")

    d = str(tmp_path / "d.nc")
    _nc(d, base, times=("2026-01-01_00:00:00", "2026-01-01_02:00:00"), attr="m/s")
    r = wtune_bp.compare_netcdf(a, d)
    assert any(m.startswith("Times:") for m in r["mismatches"])
    assert any(m.startswith("U:units") for m in r["mismatches"])


# ---------------------------------------------------------------- CLI
def test_cli_parsing_and_targets():
    p = wtune_bp.build_parser()
    a = p.parse_args(["nc2bp", "a", "b", "--out", "o", "--engine", "BP4", "--check"])
    assert (a.action, a.src, a.out, a.engine, a.check) == ("nc2bp", ["a", "b"], "o", "BP4", True)
    a = p.parse_args(["bp2nc", "x.bp"])
    assert (a.action, a.src, a.out) == ("bp2nc", ["x.bp"], None)
    a = p.parse_args(["inputs", "case", "--cache", "c", "--check"])
    assert (a.case_dir, a.cache, a.check, a.engine, a.names) == ("case", "c", True, "BP4", None)
    with pytest.raises(SystemExit):
        p.parse_args(["inputs", "case"])          # --cache required
    with pytest.raises(SystemExit):
        p.parse_args(["nc2bp", "a", "--engine", "BP3"])
    assert wtune_bp.nc2bp_target("/d/wrfinput_d01") == "/d/wrfinput_d01.bp"
    assert wtune_bp.nc2bp_target("/d/wrfinput_d01", "/o") == "/o/wrfinput_d01"
    assert wtune_bp.bp2nc_target("/d/wrfinput_d01.bp") == "/d/wrfinput_d01.nc"
    assert wtune_bp.bp2nc_target("/d/wrfout_d01_2019-11-26_12:00:00/", "/o") == \
        "/o/wrfout_d01_2019-11-26_12:00:00.nc"


def test_cli_main_without_adios2_exits_2(no_adios2, capsys):
    assert wtune_bp.main(["bp2nc", "x"]) == 2
    assert "pixi install" in capsys.readouterr().err


def test_wtune_cli_registers_convert():
    r = subprocess.run([sys.executable, os.path.join(REPO, "wtune.py"), "convert", "inputs", "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--cache" in r.stdout and "wtune convert" in r.stdout


# ---------------------------------------------------------------- per-record layout
def test_per_record_layout_manifest_links_and_legacy_reconvert(tmp_path, fake_conv):
    """wrfbdy/wrflowinp/wrffdda convert per record (WRF's reader returns BP step 0 for every
    field read, so one record per file); a manifest entry from before the per-record layout
    (single BP, no "layout") is reconverted; wrfinput stays one BP."""
    case, cache = tmp_path / "case", tmp_path / "cache"
    case.mkdir()
    for n in ("wrfinput_d01", "wrfbdy_d01", "wrflowinp_d01", "wrffdda_d01"):
        (case / n).write_bytes(NC + n.encode())
    # legacy cache: single-file BP for wrfbdy, entry without "layout"
    _mkbp(cache / "wrfbdy_d01", "BP4")
    src = os.path.realpath(case / "wrfbdy_d01")
    st = os.stat(src)
    (cache / "bp_manifest.json").write_text(json.dumps({"version": 1, "entries": {"wrfbdy_d01": {
        "source": src, "size": st.st_size, "mtime": st.st_mtime,
        "sha256": wtune_bp.sha256_file(src), "engine": "BP4",
        "converter_version": wtune_bp.CONVERTER_VERSION,
        "bp_path": str(cache / "wrfbdy_d01")}}}))
    m = wtune_bp.convert_inputs(str(case), str(cache))
    assert sorted(os.path.basename(c[0]) for c in fake_conv) == ["wrfbdy_d01", "wrffdda_d01", "wrfinput_d01",
                                               "wrflowinp_d01"]
    man = json.loads((cache / "bp_manifest.json").read_text())["entries"]
    assert man["wrfinput_d01"]["layout"] == "single"
    links = {}
    for name in ("wrfbdy_d01", "wrflowinp_d01", "wrffdda_d01"):
        assert man[name]["layout"] == "per-record"
        assert man[name]["records"] == ["%s_%s.bp" % (name, d) for d in FAKE_DATES]
        assert wtune_bp.is_per_record_dir(m[name]) and not wtune_bp.is_bp(m[name])
        links.update(wtune_bp.bp_input_links(name, m[name]))
    links.update(wtune_bp.bp_input_links("wrfinput_d01", m["wrfinput_d01"]))
    assert links["wrfinput_d01.bp"] == m["wrfinput_d01"]
    assert links["wrfbdy_d01_2026-01-01_06:00:00.bp"] == os.path.join(
        m["wrfbdy_d01"], "wrfbdy_d01_2026-01-01_06:00:00.bp")
    assert len(links) == 7 and all(wtune_bp.is_bp(t) for t in links.values())
    # cached now; a lost record file forces that name's reconversion only
    wtune_bp.convert_inputs(str(case), str(cache))
    assert len(fake_conv) == 4
    import shutil
    shutil.rmtree(os.path.join(m["wrflowinp_d01"], "wrflowinp_d01_%s.bp" % FAKE_DATES[1]))
    wtune_bp.convert_inputs(str(case), str(cache))
    assert [os.path.basename(c[0]) for c in fake_conv[4:]] == ["wrflowinp_d01"]
    assert not [f for f in os.listdir(cache) if ".tmp-" in f or f.endswith(".old")]


def test_per_record_namelist_edits():
    e = wtune_bp.per_record_namelist_edits(["wrfinput_d01", "wrfbdy_d01", "wrflowinp_d01",
                                            "wrflowinp_d02"], max_dom=2)
    assert e == {
        "bdy_control": {"multi_bdy_files": ".true."},
        "time_control": {"bdy_inname": '"wrfbdy_d<domain>_<date>.bp"',
                         "auxinput4_inname": '"wrflowinp_d<domain>_<date>.bp"',
                         "frames_per_auxinput4": "1, 1"},
    }
    # WRF's check_a_mundo demands the literal "_<date>" in bdy_inname with multi_bdy_files
    assert "_<date>" in e["time_control"]["bdy_inname"]
    f = wtune_bp.per_record_namelist_edits(["wrfffdda_d01", "wrffdda_d01"])
    assert f == {"fdda": {"gfdda_inname": '"wrffdda_d<domain>_<date>.bp"'},
                 "time_control": {"frames_per_auxinput10": "1"}}
    assert wtune_bp.per_record_namelist_edits(["wrfinput_d01"]) == {}


# ---------------------------------------------------------------- fresh copy (NFS open stall)
class _FakeCp:
    """Records cp commands; reflink fails when `reflink` is False; copies with shutil."""

    def __init__(self, reflink=True):
        self.reflink, self.calls = reflink, []

    def __call__(self, cmd, **kw):
        import shutil
        self.calls.append(list(cmd))
        if "--reflink=always" in cmd and not self.reflink:
            os.makedirs(cmd[-1])                 # a partial target must be cleaned up
            return subprocess.CompletedProcess(cmd, 1, b"", b"Operation not supported")
        shutil.copytree(cmd[-2], cmd[-1], symlinks=True)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")


def _container(path, name):
    for d in FAKE_DATES:
        rec = os.path.join(path, "%s_%s.bp" % (name, d))
        _mkbp(rec, "BP4")
        with open(os.path.join(rec, "data.0"), "wb") as f:
            f.write(b"x" * 1000)
    return str(path)


def test_fresh_input_copy_reflink(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    src = _container(tmp_path / "cache" / "wrfbdy_d01", "wrfbdy_d01")
    cp, stats = _FakeCp(), {}
    out = wtune_bp.fresh_input_copy({"wrfbdy_d01": src}, str(tmp_path / "fresh"), runner=cp,
                                    stats=stats)
    assert out == {"wrfbdy_d01": str(tmp_path / "fresh" / "wrfbdy_d01")}
    assert cp.calls == [["cp", "-a", "--reflink=always", src, out["wrfbdy_d01"]]]
    assert stats == {"reflink": True, "bytes": 0}
    assert wtune_bp.record_files(out["wrfbdy_d01"]) == wtune_bp.record_files(src)
    # a list of dirs is keyed by basename
    out2 = wtune_bp.fresh_input_copy([src], str(tmp_path / "fresh2"), runner=cp)
    assert list(out2) == ["wrfbdy_d01"]


def test_fresh_input_copy_falls_back_to_cp_a_with_size_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    src = _container(tmp_path / "wrflowinp_d01", "wrflowinp_d01")
    cp, stats = _FakeCp(reflink=False), {}
    out = wtune_bp.fresh_input_copy([src], str(tmp_path / "fresh"), runner=cp, stats=stats)
    assert [c[:3] for c in cp.calls] == [["cp", "-a", "--reflink=always"], ["cp", "-a", src]]
    assert stats["reflink"] is False and stats["bytes"] == 2000
    err = capsys.readouterr().err
    assert err.count("WARNING") == 1 and "full cp -a of 0.0 MB" in err
    assert wtune_bp.is_per_record_dir(out["wrflowinp_d01"])


def test_fresh_input_copy_warns_inside_a_slurm_job(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    src = _container(tmp_path / "wrfbdy_d01", "wrfbdy_d01")
    wtune_bp.fresh_input_copy([src], str(tmp_path / "fresh"), runner=_FakeCp())
    assert "inside Slurm job 1234 (a compute node)" in capsys.readouterr().err


def test_cached_inputs_only_when_current(tmp_path, fake_conv):
    case, cache = tmp_path / "case", tmp_path / "cache"
    case.mkdir()
    (case / "wrfbdy_d01").write_bytes(NC + b"B")
    assert wtune_bp.cached_inputs(str(case), str(cache), ["wrfbdy_d01"]) is None
    wtune_bp.convert_inputs(str(case), str(cache))
    assert wtune_bp.cached_inputs(str(case), str(cache), ["wrfbdy_d01"]) == {
        "wrfbdy_d01": str(cache / "wrfbdy_d01")}
    (case / "wrfbdy_d01").write_bytes(NC + b"changed")
    assert wtune_bp.cached_inputs(str(case), str(cache), ["wrfbdy_d01"]) is None
