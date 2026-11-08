"""ADIOS2 BP history (io_form_history=14) read through post.extract's netCDF cache.

The BP directories here are fakes (data.0 holds the Times as JSON, md.0/md.idx/mmd.0 stand in
for the metadata) and wtune.bp is a fake module whose bp2nc writes the small multi-frame
wrfout of tests/test_post_multiframe.py -- so the index, cache, invalidation, worker pool and
error paths are tested without adios2. Every conversion appends "<pid> <bp>" to $FAKE_BP_LOG,
which also counts conversions done inside pool workers.
"""
from conftest import swap_wtune_module
import json
import os
import types
from concurrent.futures import ProcessPoolExecutor

import pytest

netCDF4 = pytest.importorskip("netCDF4")

from wtune.post import extract  # noqa: E402
from test_post_multiframe import T, _times_file  # noqa: E402


def _bp(path, times):
    """A fake BP directory holding `times`."""
    path = str(path)
    os.makedirs(path)
    with open(os.path.join(path, "data.0"), "w") as fh:
        json.dump(times, fh)
    for name in ("md.0", "md.idx", "mmd.0"):
        with open(os.path.join(path, name), "w") as fh:
            fh.write("m")
    return path


def _rewrite(bp, times, metadata=True):
    """WRF rewrote bp: new data (and, by default, new metadata), mtimes moved forward."""
    with open(os.path.join(bp, "data.0"), "w") as fh:
        json.dump(times, fh)
    names = ["data.0"]
    if metadata:
        with open(os.path.join(bp, "md.0"), "a") as fh:
            fh.write("more")
        names.append("md.0")
    for name in names:
        p = os.path.join(bp, name)
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2 * 10 ** 9))


def _fake_bp2nc(src, dst):
    if os.path.lexists(dst):                 # the real bp2nc's contract
        raise FileExistsError(dst)
    with open(os.path.join(src, "data.0")) as fh:
        times = json.load(fh)
    _times_file(dst, times)
    with open(os.environ["FAKE_BP_LOG"], "a") as fh:
        fh.write("%d %s\n" % (os.getpid(), src))
    return dst


class _Adios2Unavailable(RuntimeError):
    pass


def _no_adios2():
    raise _Adios2Unavailable("no Python that can `import adios2` was found. The BP converters run "
                             "under the separate envs/adios2 environment: run `pixi install` in "
                             "envs/adios2")


@pytest.fixture
def fake_bp(tmp_path, monkeypatch):
    log = tmp_path / "bp2nc.log"
    log.write_text("")
    monkeypatch.setenv("FAKE_BP_LOG", str(log))
    monkeypatch.delenv(extract.NC_CACHE_ENV, raising=False)
    mod = types.ModuleType("wtune_bp")
    mod.bp2nc = _fake_bp2nc
    mod.require_adios2_python = lambda: "/fake/adios2/python"
    mod.is_bp = os.path.isdir
    swap_wtune_module(monkeypatch, "bp", mod)
    monkeypatch.setattr(extract, "_BP_MEMO", {})
    monkeypatch.setattr(extract, "_BP_ORIGIN", {})
    monkeypatch.setattr(extract, "_cpu_budget", lambda: 1)
    mod.conversions = lambda: [ln.split(" ", 1)[1] for ln in log.read_text().splitlines()]
    mod.pids = lambda: [int(ln.split(" ", 1)[0]) for ln in log.read_text().splitlines()]
    return mod


def _fresh_process(monkeypatch):
    """What a new `wtune post` process sees: no in-process memo, only the cache on disk."""
    monkeypatch.setattr(extract, "_BP_MEMO", {})
    monkeypatch.setattr(extract, "_BP_ORIGIN", {})


def _cache(run, root=None):
    root = root or os.path.join(str(run), "_nc_cache")
    return os.path.join(str(root), extract._run_id(str(run)))


def test_a_rewritten_bp_is_reconverted_alone(tmp_path, fake_bp, monkeypatch):
    run = tmp_path / "run"
    a = _bp(run / ("wrfout_d01_%s" % T(0)), [T(0), T(6)])
    b = _bp(run / ("wrfout_d01_%s" % T(12)), [T(12), T(18)])
    extract.list_frame_refs(str(run))
    _rewrite(a, [T(0), T(6), T(9)])                                  # the run was re-done
    assert [t for t, _p, _i in extract.list_frame_refs(str(run))] == \
        [T(0), T(6), T(9), T(12), T(18)]                             # same process
    assert sorted(fake_bp.conversions()) == sorted([a, b, a])
    _rewrite(a, [T(0)], metadata=False)                              # only a data file changed
    _fresh_process(monkeypatch)
    assert [t for t, _p, _i in extract.list_frame_refs(str(run))] == [T(0), T(12), T(18)]
    assert fake_bp.conversions().count(a) == 3 and fake_bp.conversions().count(b) == 1


def test_conversion_runs_in_a_bounded_process_pool(tmp_path, fake_bp, monkeypatch):
    pools = []

    class RecordingPool(ProcessPoolExecutor):
        def __init__(self, max_workers=None, **kw):
            pools.append(max_workers)
            super().__init__(max_workers=max_workers, **kw)

    monkeypatch.setattr(extract, "ProcessPoolExecutor", RecordingPool)
    monkeypatch.setattr(extract, "_cpu_budget", lambda: 3)
    run = tmp_path / "run"
    bps = [_bp(run / ("wrfout_d01_%s" % T(h)), [T(h)]) for h in (0, 6, 12, 18)]
    refs = extract.list_frame_refs(str(run))
    assert [t for t, _p, _i in refs] == [T(0), T(6), T(12), T(18)]
    assert pools == [3]                                  # 4 files, bounded by the CPU budget
    assert sorted(fake_bp.conversions()) == sorted(bps)
    assert os.getpid() not in fake_bp.pids()             # converted in the workers
    for _t, p, _i in refs:
        assert extract.file_times(p)                     # complete, readable netCDF


def test_mixed_dir_prefers_the_netcdf_file(tmp_path, fake_bp):
    run = tmp_path / "run"
    nc = _times_file(run / ("wrfout_d01_%s" % T(0)), [T(0)])
    _bp(run / ("wrfout_d01_%s.bp" % T(0)), [T(0)])                   # same time, BP too
    b6 = _bp(run / ("wrfout_d01_%s.bp" % T(6)), [T(6)])
    refs = extract.list_frame_refs(str(run))
    assert refs == [(T(0), nc, 0),
                    (T(6), os.path.join(_cache(run), "wrfout_d01_%s.nc" % T(6)), 0)]
    assert fake_bp.conversions() == [b6]                 # the shadowed BP is never converted


def test_cli_nc_cache_flag_and_default_base(tmp_path, fake_bp, monkeypatch, capsys):
    from wtune.post import cli
    from wtune.post import inventory
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.setattr(inventory, "_domain_grid", lambda path: (
        dict(dx=12000.0, dy=12000.0, e_we=5, e_sn=4, e_vert=None, history_interval_attr=None,
             lon_range=None, lat_range=None), ["Times", "XLAT", "XLONG"]))
    run = tmp_path / "run"
    _bp(run / ("wrfout_d01_%s" % T(0)), [T(0), T(6)])
    assert cli.main(["inventory", str(run), "--json", "--nc-cache", str(tmp_path / "c")]) == 0
    inv = json.loads(capsys.readouterr().out)
    assert inv["format"] == "adios2_bp" and inv["nc_cache"] == [_cache(run, tmp_path / "c")]
    assert inv["domains"]["d01"]["times"] == [T(0)]          # per FILE (file name time)
    args = cli.build_parser().parse_args(["recipe", str(run), "--out", str(tmp_path / "o")])
    assert cli._nc_cache_base(args) == str(tmp_path / "o")
    args = cli.build_parser().parse_args(["extract", str(run), "--vars", "t2", "--out",
                                          str(tmp_path / "x" / "p.csv"), "--points", "1,2"])
    assert cli._nc_cache_base(args) == str(tmp_path / "x")


def test_missing_adios2_is_one_clear_error(tmp_path, fake_bp, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)                          # no project.yaml discovery
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    fake_bp.require_adios2_python = _no_adios2
    run = tmp_path / "run"
    _bp(run / ("wrfout_d01_%s" % T(0)), [T(0)])
    _bp(run / ("wrfout_d01_%s" % T(6)), [T(6)])
    with pytest.raises(extract.BPConversionError) as e:
        extract.list_frames(str(run))
    msg = str(e.value)
    assert str(run) in msg and "2 file(s)" in msg and "pixi install" in msg
    assert "wrf_adios2_to_netcdf4" in msg and fake_bp.conversions() == []

    # the recipe: ONE missing entry naming the fix, a report, no traceback
    from wtune.post import recipe
    res = recipe.run_recipe([str(run)], str(tmp_path / "out"), labels=["r"])
    assert len(res["figures"]) == 1
    gap = res["figures"][0]
    assert gap["missing"] and gap["run"] == "r" and "pixi install" in gap["reason"]
    assert "pixi install" in open(res["report_path"]).read()

    # whole-run CLI commands: an error line and exit 2; inventory/compare report it
    from wtune.post import cli, compare, inventory
    assert cli.main(["cf3d", str(run), "--out", str(tmp_path / "cf")]) == 2
    assert "pixi install" in capsys.readouterr().err
    inv = inventory.inventory(str(run))
    assert inv["format"] == "adios2_bp" and "pixi install" in inv["note"]
    with pytest.raises(compare.CompareError, match="pixi install"):
        compare._frames(str(run))

    # no wtune.bp at all
    swap_wtune_module(monkeypatch, "bp", None)
    with pytest.raises(extract.BPConversionError, match="cannot import wtune.bp"):
        extract.list_frames(str(run))


def test_a_failed_conversion_leaves_that_file_out_and_is_not_retried(tmp_path, fake_bp,
                                                                     monkeypatch, capsys):
    run = tmp_path / "run"
    _bp(run / ("wrfout_d01_%s" % T(0)), [T(0)])
    bad = _bp(run / ("wrfout_d01_%s" % T(6)), [T(6)])
    with open(os.path.join(bad, "data.0"), "w") as fh:
        fh.write("truncated")                                       # unreadable BP
    assert [t for t, _p in extract.list_frames(str(run))] == [T(0)]
    assert "conversion failed" in capsys.readouterr().err
    calls = []
    monkeypatch.setattr(extract, "_convert_all", lambda *a: calls.append(a) or {})
    assert [t for t, _p in extract.list_frames(str(run))] == [T(0)]
    assert calls == []                                              # unchanged: not retried
    failed = extract.prepare_history(str(run))["failed"]
    assert [bp for bp, _err in failed] == [bad]


def test_vapor_bundle_links_the_cached_netcdf_and_says_so(tmp_path, fake_bp):
    from wtune.post import vapor_bundle as vb
    run = tmp_path / "run"
    b0 = _bp(run / ("wrfout_d01_%s" % T(0)), [T(0)])
    b6 = _bp(run / ("wrfout_d01_%s" % T(6)), [T(6)])
    r = vb.build_bundle(str(run), str(tmp_path / "out"), 1, every="6h")
    assert [f["native"] for f in r["frames"]] == [b0, b6]
    assert {f["mode"] for f in r["frames"]} == {"hardlink"}
    for f in r["frames"]:
        assert f["original"].startswith(_cache(run))
        assert os.stat(os.path.join(r["dir"], f["name"])).st_ino == os.stat(f["original"]).st_ino
    assert "netCDF cache" in r["link_note"] and "ADIOS2 BP" in r["link_note"]
    readme = open(r["readme"], encoding="utf-8").read()
    assert "ADIOS2 BP" in readme and "複本" in readme and _cache(run) in readme
    assert "就地改寫原檔" not in readme             # not the same data as the run's history
    manifest = json.load(open(r["manifest"]))
    assert [f["native"] for f in manifest["frames"]] == [b0, b6]
