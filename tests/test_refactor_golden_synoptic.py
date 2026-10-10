"""Golden snapshot of post/recipes/synoptic.figures, pinned before the round-2 refactor split it
into per-chart helpers. wrf-python, the netCDF4 Dataset, save_map, ERA5 and MET Grid-Stat are
replaced by recording fakes, so the snapshot pins exactly what figures() computes and hands to
them (levels, titles, contour sets, figure/missing entries, call order), not the plotting.

Regenerate (only when a behaviour change is intended) with:
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_synoptic.py
"""
import json
import os
import sys

import netCDF4
import numpy as np
import pytest

from wtune.post import caseinfo, conventions, met_gridstat, tools
from wtune.post.recipes import era5, mapfig, synoptic

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fixtures", "golden", "synoptic")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
NY, NX, NZ = 4, 5, 3


def _grid(lo, hi, shape=(NY, NX)):
    return np.linspace(lo, hi, int(np.prod(shape))).reshape(shape)


def _summ(v):
    """JSON-safe, rounded summary of whatever figures() hands to a fake."""
    if isinstance(v, np.ma.MaskedArray):
        return {"masked_shape": list(v.shape), "n_masked": int(np.ma.count_masked(v)),
                "sum": round(float(np.nansum(np.ma.filled(v.astype("f8"), np.nan))), 6)}
    if isinstance(v, np.ndarray):
        if v.dtype == bool:
            return {"bool_shape": list(v.shape), "n_true": int(v.sum())}
        a = v.astype("f8")
        if a.size <= 12:
            return [round(float(x), 6) for x in a.ravel()]
        return {"shape": list(v.shape), "sum": round(float(np.nansum(a)), 6),
                "min": round(float(np.nanmin(a)), 6), "max": round(float(np.nanmax(a)), 6)}
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, dict):
        return {str(k): _summ(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_summ(x) for x in v]
    if isinstance(v, FakeDS):
        return "<DS>"
    return v


class FakeDS:
    def __init__(self, path):
        self.path = path
        self.variables = {
            "HGT": _grid(0.0, 900.0, (1, NY, NX)),
            "PSFC": np.concatenate([_grid(80000.0, 101000.0, (1, NY, NX))] * 2),
            "XLAT": _grid(20.0, 30.0, (1, NY, NX)),
            "XLONG": _grid(-100.0, -90.0, (1, NY, NX)),
        }

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeWrf:
    def __init__(self, calls, z500=5600.0, z1000=150.0, avo_missing=False, jet_max=95.0,
                 fail=None):
        self.calls, self.z500, self.z1000 = calls, z500, z1000
        self.avo_missing, self.jet_max, self.fail = avo_missing, jet_max, fail

    def getvar(self, ds, name, timeidx=None, units=None):
        self.calls.append(["getvar", name, timeidx, units])
        if self.fail is not None and name == "pressure":
            raise self.fail
        if name == "avo" and self.avo_missing:
            raise KeyError("MAPFAC_U")
        three = np.stack([_grid(i, i + 1.0) for i in range(NZ)])
        return {
            "pressure": three * 100.0 + 200.0,
            "z": three * 1000.0 + self.z500,
            "avo": three * 10.0 + 12.0,
            "slp": _grid(987.3, 1021.9),
            "theta_e": three + 330.0,
            "ua": three * 3.0 - 2.0,
            "va": three * 2.0 + 1.0,
            "uvmet": np.stack([three * 3.0 - 2.0, three * 2.0 + 1.0]),
            "wspd_wdir": np.stack([three * 10.0 + self.jet_max - 20.0, three]),
            "T2": _grid(260.0, 305.0),
            "uvmet10": np.stack([_grid(-5.0, 5.0), _grid(0.0, 8.0)]),
        }[name]

    def interplevel(self, field, pressure, level):
        self.calls.append(["interplevel", level])
        return np.asarray(field)[0] + level / 1000.0

    def vinterp(self, ds, z, vert, levels, extrapolate=False, field_type=None, timeidx=None,
                log_p=False):
        self.calls.append(["vinterp", vert, list(levels), extrapolate, field_type, log_p])
        return np.stack([_grid(self.z1000, self.z1000 + 40.0), _grid(self.z500, self.z500 + 120)])

    def latlon_coords(self, var, as_np=False):
        return _grid(20.0, 30.0), _grid(-100.0, -90.0)

    def get_cartopy(self, var=None):
        return "<CRS>"

    def cartopy_xlim(self, var=None):
        return (-1.0, 1.0)

    def cartopy_ylim(self, var=None):
        return (-2.0, 2.0)


def _run(monkeypatch, tmp_path, frames=None, era5_exc=None, gs_exc=None, project=True, **wrf_kw):
    calls = []
    frames = [("2005-08-28_00:00:00", "/run/wrfout_a", 0), ("2005-08-28_06:00:00",
              "/run/wrfout_b", 0), ("2005-08-28_12:00:00", "/run/wrfout_b", 1)] \
        if frames is None else frames
    monkeypatch.setattr(synoptic, "list_frame_refs",
                        lambda run_dir, domain=1: (calls.append(["frames", run_dir, domain])
                                                   or frames))
    monkeypatch.setitem(sys.modules, "wrf", FakeWrf(calls, **wrf_kw))
    monkeypatch.setattr(netCDF4, "Dataset", lambda p: (calls.append(["open", p]) or FakeDS(p)))
    monkeypatch.setattr(conventions, "ds_time_label", lambda ds, t: "LABEL(%s)" % t)
    monkeypatch.setattr(conventions, "path_time_label", lambda p, t: "PLABEL(%s,%s)" % (p, t))

    def save_map(path, lats, lons, title, **kw):
        calls.append(["save_map", os.path.basename(path), title, _summ(lats), _summ(lons),
                      {k: _summ(v) for k, v in kw.items()}])
    monkeypatch.setattr(mapfig, "save_map", save_map)

    from wtune import project as wtune_project

    def find_project(out_dir):
        if not project:
            raise RuntimeError("no project")
        return "/proj"
    monkeypatch.setattr(wtune_project, "find_project", find_project)

    def z500_for(t, bbox, project_dir):
        calls.append(["z500_for", t.isoformat(), _summ(list(bbox)), project_dir])
        if era5_exc is not None:
            raise era5_exc
        return _grid(19.0, 31.0), _grid(-101.0, -89.0), _grid(5500.0, 5900.0), t, "local.nc"
    monkeypatch.setattr(era5, "z500_for", z500_for)
    monkeypatch.setattr(caseinfo, "spec_bdy_width", lambda run_dir: 5)

    def grid_stat(path, e_lat, e_lon, e_z, valid, out_dir, bdy, timeidx=None):
        calls.append(["grid_stat", path, os.path.basename(out_dir), bdy, timeidx])
        if gs_exc is not None:
            raise gs_exc
        return {"rmse_m": 13.25, "bias_m": 9.69, "n": 3825, "stat_file": "gs.stat",
                "pairs_nc": "pairs.nc"}
    monkeypatch.setattr(met_gridstat, "grid_stat_z500", grid_stat)
    monkeypatch.setattr(synoptic, "_era5_diff_map",
                        lambda *a: calls.append(["era5_diff_map", _summ(list(a[1:]))]))
    out = synoptic.figures(["/run"], ["m1"], str(tmp_path), {"case": "x"}, spinup_hours=6)
    return {"out": out, "calls": calls}


CASES = {
    "full_straddle_540": dict(z500=5500.0),
    "cold_only_no_project": dict(z500=5200.0, project=False),
    "warm_only_weak_jet": dict(z500=5900.0, jet_max=60.0),
    "avo_missing_era5_unavailable": dict(avo_missing=True,
                                         era5_exc=era5.Era5UnavailableError("no network")),
    "met_error": dict(gs_exc=met_gridstat.METError("grid_stat failed")),
    "tool_error": dict(gs_exc=tools.ToolError("grid_stat rc=1")),
    "wrf_import_error": dict(fail=ImportError("No module named wrf")),
    "tool_not_found": dict(fail=tools.ToolNotFoundError("no wrf-python")),
    "spinup_not_reached": dict(frames=[("2005-08-28_00:00:00", "/run/wrfout_a", 0),
                                       ("2005-08-28_03:00:00", "/run/wrfout_a", 1)]),
    "no_frames": dict(frames=[]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_synoptic_figures_golden(name, monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, **CASES[name])
    text = json.dumps(_summ(result), indent=1, default=repr).replace(str(tmp_path), "<OUT>") + "\n"
    path = os.path.join(GOLDEN_DIR, "figures_%s.json" % name)
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    with open(path) as fh:
        expected = fh.read()
    assert text == expected, "figures() differs from golden %s" % path
