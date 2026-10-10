"""Golden snapshots of wtune_bp.main (CLI stdout/stderr/rc, fake converters), the convert_inputs
manifest and the full compare_netcdf result dicts, pinned before the round-3 tidy-up split
convert_inputs/compare_netcdf. Tmp paths and timings are normalised.

Regenerate (only for an intended behaviour change):
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_bp.py
"""
import json
import os
import re

import numpy as np
import pytest

from wtune import bp as wtune_bp

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden", "bp",
                      "bp_snapshots.json")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
NC = b"CDF\x01" + b"\0" * 60
DATES = ["2026-01-01_00:00:00", "2026-01-01_06:00:00"]


def _mkbp(path, engine="BP4"):
    os.makedirs(path)
    for f in ["md.0", "md.idx", "data.0"] + (["mmd.0"] if engine == "BP5" else []):
        open(os.path.join(path, f), "w").close()


def _norm(text, tmp):
    text = text.replace(str(tmp), "<TMP>")
    return re.sub(r"\d+\.\d s", "<T> s", text)


@pytest.fixture
def fakes(monkeypatch):
    def fake_nc2bp(src, dst, engine="BP4", split_prefix=None):
        if split_prefix is not None:
            os.makedirs(dst)
            for d in DATES:
                _mkbp(os.path.join(dst, "%s_%s.bp" % (split_prefix, d)), engine)
            return dst
        _mkbp(dst, engine)
        return dst

    def fake_bp2nc(src, dst):
        open(dst, "wb").write(NC)
        return dst

    monkeypatch.setattr(wtune_bp, "require_adios2_python", lambda: "/fake/python")
    monkeypatch.setattr(wtune_bp, "nc2bp", fake_nc2bp)
    monkeypatch.setattr(wtune_bp, "bp2nc", fake_bp2nc)
    monkeypatch.setattr(wtune_bp, "roundtrip_check", lambda s, b, t: {
        "ok": not s.endswith("wrfbdy_d01"), "vars_compared": 3, "max_abs_diff": 0.25,
        "mismatches": ["U: data differs", "V: shape"] if s.endswith("wrfbdy_d01") else []})


def _case(tmp):
    case = tmp / "case"
    case.mkdir()
    (case / "wrfinput_d01").write_bytes(NC + b"A")
    (case / "wrfbdy_d01").write_bytes(NC + b"B")
    (case / "wrfinput_d02").write_bytes(b"not netcdf")
    return str(case)


def _run(argv, tmp, capsys):
    rc = wtune_bp.main(argv)
    cap = capsys.readouterr()
    return {"rc": rc, "stdout": _norm(cap.out, tmp), "stderr": _norm(cap.err, tmp)}


def _cli_cases(tmp):
    case = _case(tmp)
    src = str(tmp / "wrfinput_d01")
    open(src, "wb").write(NC)
    bp = str(tmp / "x.bp")
    _mkbp(bp)
    os.makedirs(str(tmp / "empty"), exist_ok=True)
    return {
        "nc2bp-default": ["nc2bp", src],
        "nc2bp-out-engine-check": ["nc2bp", src, "--out", str(tmp / "o"), "--engine", "BP5",
                                   "--check"],
        "bp2nc-ok": ["bp2nc", bp, "--out", str(tmp / "o2")],
        "bp2nc-not-bp": ["bp2nc", str(tmp)],
        "inputs-all": ["inputs", case, "--cache", str(tmp / "cache")],
        "inputs-check": ["inputs", case, "--cache", str(tmp / "cache2"), "--check"],
        "inputs-names": ["inputs", case, "--cache", str(tmp / "cache3"), "--names",
                         "wrfinput_d01"],
        "inputs-none": ["inputs", str(tmp / "empty"), "--cache", str(tmp / "cache4")],
    }


def _manifest(tmp):
    case = _case(tmp)
    cache = tmp / "cache"
    wtune_bp.convert_inputs(case, str(cache))
    man = json.load(open(cache / "bp_manifest.json"))
    for e in man["entries"].values():
        e["mtime"] = "<mtime>"
        e["converted_at"] = "<ts>"
    return json.loads(_norm(json.dumps(man, indent=1, sort_keys=True), tmp))


def _nc(path, u, times=DATES[:1] * 2, attr="m s-1", extra=False, dtype="f4", glob=3000.0,
        xlen=3, nrec=2):
    from netCDF4 import Dataset
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("x", xlen)
        ds.setncattr("DX", np.float32(glob))
        t = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[:nrec, :] = np.asarray([list(s) for s in times[:nrec]], dtype="S1")
        v = ds.createVariable("U", dtype, ("Time", "x"))
        v[:nrec, :] = u[:nrec, :xlen]
        v.setncattr("units", attr)
        if extra:
            ds.createVariable("EXTRA", "f4", ("x",))
        else:
            i = ds.createVariable("I", "i4", ("Time", "x"))
            i[:nrec, :] = np.arange(6, dtype=np.int32).reshape(2, 3)[:nrec, :xlen]


def _compare(tmp):
    pytest.importorskip("netCDF4")
    base = np.array([[1, np.nan, 3], [4, 5, 6]], np.float32)
    mod = base.copy()
    mod[1, 2] += 0.5
    paths = {}

    def mk(name, *a, **k):
        paths[name] = str(tmp / (name + ".nc"))
        _nc(paths[name], *a, **k)

    mk("base", base)
    mk("same", base.copy())
    mk("data", mod)
    mk("attrs", base, times=[DATES[0], DATES[1]], attr="m/s", glob=1.0)
    mk("extra", base, extra=True)
    mk("dtype", base, dtype="f8")
    mk("narrow", base, xlen=2)
    mk("one", base, nrec=1)
    pairs = {"same": ("base", "same", None), "data": ("base", "data", None),
             "attrs": ("base", "attrs", None), "extra": ("base", "extra", None),
             "dtype": ("base", "dtype", None), "narrow": ("base", "narrow", None),
             "record0": ("base", "one", 0), "record1-bad": ("base", "one", 1)}
    return {k: wtune_bp.compare_netcdf(paths[a], paths[b], left_record=r)
            for k, (a, b, r) in pairs.items()}


def _collect(tmp_factory, capsys):
    out = {"cli": {}}
    for cid in _cli_cases(tmp_factory("probe")):
        t = tmp_factory(cid)
        out["cli"][cid] = _run(_cli_cases(t)[cid], t, capsys)
    out["manifest"] = _manifest(tmp_factory("manifest"))
    out["compare"] = _compare(tmp_factory("compare"))
    return json.loads(json.dumps(out))


def test_bp_matches_golden(tmp_path, fakes, capsys):
    def tmp_factory(name):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        return d
    # _collect re-creates each cli case's tree in its own dir (cases write into it)
    got = _collect(tmp_factory, capsys)
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(got, fh, indent=1, sort_keys=True)
            fh.write("\n")
    with open(GOLDEN) as fh:
        want = json.load(fh)
    assert got == want
