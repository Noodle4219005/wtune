"""Golden snapshots of the full outputs of post/cf_pressure (convert_frame/export NetCDF
structure + data checksums) and post/vapor_bundle (build_bundle result, bundle.json, README.md,
vapor_render.py checksum, tree), pinned before the round-3 tidy-up split write_cf/read_frame/
build_bundle. Synthetic inputs come from test_post_cf_pressure.py / test_post_vapor_bundle.py.

Regenerate (only for an intended behaviour change):
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_post_outputs.py
"""
import hashlib
import json
import os
import re

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("wrf")
pytest.importorskip("pyproj")
pytest.importorskip("scipy")

import test_post_cf_pressure as tcp  # noqa: E402
import test_post_vapor_bundle as tvb  # noqa: E402
from post import cf_pressure as cp  # noqa: E402
from post import vapor_bundle as vb  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden",
                      "post_outputs", "post_outputs.json")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
_TS = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|\+00:00)?")


def _norm(text, tmp):
    text = _TS.sub("<TS>", str(text).replace(str(tmp), "<TMP>"))
    text = re.sub(r"[0-9a-f]{40}(-dirty)?", "<SHA>", text)                # wtune_commit: git HEAD
    text = re.sub(r'("wtune_commit": )null', r'\1"<SHA>"', text)        # exported tree without .git
    text = re.sub(r'(\\"wtune_commit\\": )null', r'\1\\"<SHA>\\"', text)   # same, inside a JSON string
    return re.sub(r"(<TS>: )(?!wtune)\S.*", r"\1<argv>", text)  # history: sys.argv of the runner


def _nc_dump(path, tmp):
    out = {"dims": {}, "vars": {}, "attrs": {}}
    with netCDF4.Dataset(path) as d:
        for n, dim in d.dimensions.items():
            out["dims"][n] = [len(dim), dim.isunlimited()]
        for k in d.ncattrs():
            out["attrs"][k] = _norm(d.getncattr(k), tmp)
        for n, v in d.variables.items():
            arr = v[:]
            filled = np.ma.filled(arr, 1e30) if np.ma.isMaskedArray(arr) else np.asarray(arr)
            out["vars"][n] = {
                "dtype": str(v.dtype), "dims": list(v.dimensions), "shape": list(v.shape),
                "attrs": {k: _norm(v.getncattr(k), tmp) for k in v.ncattrs()},
                "filters": {k: str(x) for k, x in sorted(v.filters().items())},
                "masked": int(np.ma.count_masked(arr)) if np.ma.isMaskedArray(arr) else 0,
                "sha256": hashlib.sha256(np.ascontiguousarray(filled).tobytes()).hexdigest(),
            }
    return out


def _result(r, tmp, mask_size=False):
    text = _norm(json.dumps(r, sort_keys=True, default=str), tmp)
    if mask_size:   # a written NetCDF's size depends on the tmp path length stored in its attributes
        text = re.sub(r'"size_bytes": \d+', '"size_bytes": "<n>"', text)
    return json.loads(text)


def _cf(tmp):
    out = {}
    src = tmp / "wrfout_d01_2005-08-28_06:00:00"
    tcp._write_wrfout(str(src))
    r = cp.convert_frame(str(src), str(tmp / "a.nc"), domain=1, bdy_width=5, command="wtune post cf3d x")
    out["convert"] = {"result": _result(r, tmp, True), "nc": _nc_dump(str(tmp / "a.nc"), tmp)}
    src2 = tmp / "noqice" / "wrfout_d01_2005-08-28_06:00:00"
    src2.parent.mkdir()
    tcp._write_wrfout(str(src2), qice=False)
    r = cp.convert_frame(str(src2), str(tmp / "b.nc"), domain=1, bdy_width=5)
    out["convert_noqice"] = {"result": _result(r, tmp, True), "nc": _nc_dump(str(tmp / "b.nc"), tmp)}
    run = tcp._run_dir(tmp)
    res = cp.export(str(run), str(tmp / "exp"), [1], 5, every="6h", command="c")
    out["export"] = {"results": _result(res, tmp, True),
                     "nc": {os.path.relpath(x["path"], str(tmp)): _nc_dump(x["path"], tmp)
                            for x in res if not x.get("error")}}
    return out


def _tree(root, tmp):
    out = {}
    for d, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, str(tmp))
            if f in ("README.md", "bundle.json"):       # embed the tmp path; pinned in full above
                out[rel] = {"size": None, "sha256": None}
                continue
            out[rel] = {"size": os.path.getsize(p),
                        "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    return out


def _vb(tmp, monkeypatch):
    out = {}
    run = tvb._run_dir(tmp)
    r = vb.build_bundle(str(run), str(tmp / "o1"), 1, every="6h", run_label="base")
    bdir = tmp / "o1" / "vapor" / "d01"
    man = json.loads((bdir / "bundle.json").read_text())
    out["every"] = {"result": _result(r, tmp), "manifest": _result(man, tmp),
                    "readme": _norm((bdir / "README.md").read_text(encoding="utf-8"), tmp),
                    "render_sha": hashlib.sha256((bdir / "vapor_render.py").read_bytes()).hexdigest(),
                    "bundle_json_text": _norm((bdir / "bundle.json").read_text(), tmp),
                    "names": sorted(os.listdir(bdir))}
    r = vb.build_bundle(str(run), str(tmp / "o2"), 1)
    out["last"] = {"result": _result(r, tmp), "names": sorted(os.listdir(tmp / "o2" / "vapor" / "d01"))}
    r = vb.build_bundle(str(run), str(tmp / "o3"), 1, times=["2005082806"])
    out["times"] = {"result": _result(r, tmp), "tree": _tree(tmp / "o3", tmp)}
    tvb._no_link(monkeypatch)
    try:
        vb.build_bundle(str(run), str(tmp / "o4"), 1, every="6h")
    except vb.NotHardLinkable as e:
        out["nolink_error"] = {"type": type(e).__name__, "msg": _norm(e, tmp),
                               "left": sorted(os.listdir(tmp / "o4")) if (tmp / "o4").exists() else None}
    r = vb.build_bundle(str(run), str(tmp / "o5"), 1, times=["2005082806"], allow_copy=True)
    out["copy"] = {"result": _result(r, tmp), "tree": _tree(tmp / "o5", tmp)}
    res = vb.export(str(run), str(tmp / "o6"), [1, 2], every="6h")
    out["export"] = _result(res, tmp)
    return out


def test_post_outputs_match_golden(tmp_path, monkeypatch):
    (tmp_path / "cf").mkdir()
    (tmp_path / "vb").mkdir()
    got = {"cf": _cf(tmp_path / "cf"), "vapor": _vb(tmp_path / "vb", monkeypatch)}
    got = json.loads(json.dumps(got, sort_keys=True))
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(got, fh, indent=1, sort_keys=True)
            fh.write("\n")
    with open(GOLDEN) as fh:
        want = json.load(fh)
    assert got == want
