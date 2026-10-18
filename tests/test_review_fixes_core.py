"""Regression tests for the 8bdcd3c source-review findings in wtune.predict, wtune.preflight,
wtune.config, tools/metdata, tools/grib_state and tools/wrf_adios2_to_netcdf4/convert."""

import argparse
import datetime
import importlib
import json
import math
import sys
import types

import pytest

from wtune import config as wcfg
from wtune import predict as wp
from wtune import preflight as pf


def _nl(tmp_path, text):
    path = tmp_path / "namelist.input"
    path.write_text(text)
    return path


# ---- 1. parse_namelist ----------------------------------------------------------------

def test_parse_namelist_assignments_on_the_group_header_line(tmp_path):
    nl = wp.parse_namelist(_nl(tmp_path, "&domains time_step = 18, max_dom = 2, /\n"
                                         "&physics\n radt = 30,\n/\n"))
    assert nl["domains"] == {"time_step": [18], "max_dom": [2]}
    assert nl["physics"] == {"radt": [30]}


def test_parse_namelist_trailing_slash_ends_the_group(tmp_path):
    nl = wp.parse_namelist(_nl(tmp_path, "&physics\n radt = 30, /\n stray = 1\n"
                                         "&time_control\n io_form_history = 2 /\n"
                                         "&dynamics\n p = 'a/b/'\n q = 1\n/\n"))
    assert nl["physics"] == {"radt": [30]}          # '/' not a value; group closed
    assert nl["time_control"] == {"io_form_history": [2]}
    assert nl["dynamics"] == {"p": ["a/b/"], "q": [1]}   # a quoted '/' closes nothing


def test_parse_namelist_repeat_count_and_short_logicals(tmp_path):
    nl = wp.parse_namelist(_nl(tmp_path, "&domains\n e_we = 2*100, 151\n max_dom = 3\n"
                                         " flag = T, F, 2*.true.\n name = 2*'x'\n/\n"))
    dom = nl["domains"]
    assert dom["e_we"] == [100, 100, 151]
    assert dom["flag"] == [True, False, True, True]
    assert dom["name"] == ["x", "x"]
    assert wp.g_dom(nl, "domains", "e_we", 1) == 100


def test_parse_namelist_indexed_assignment_gets_its_own_key(tmp_path):
    nl = wp.parse_namelist(_nl(tmp_path, "&domains\n e_we = 100, 100\n e_we(2) = 151\n/\n"))
    assert nl["domains"]["e_we"] == [100, 100]       # previous key not corrupted
    assert nl["domains"]["e_we(2)"] == [151]


def test_out_of_block_scan_follows_the_same_group_boundaries(tmp_path):
    path = _nl(tmp_path, "&domains max_dom = 1,\n e_we = 100 /\n lost = 1\n&physics\n/\n")
    assert [(k, ln) for ln, k, _raw in pf.find_out_of_block_assignments(path)] == [("lost", 3)]


# ---- 2. C1: short per-domain start_*/end_* lists --------------------------------------

C1_NAMELIST = """&time_control
 run_hours = 6,
 start_year = 2024, 2024, start_month = 09, 09, start_day = 05, 05,
 start_hour = 12, 12,
 end_year = 2024, 2024, end_month = 09, 09, end_day = 05, 05,
 end_hour = %s
 end_minute = 00,
/
&domains
 max_dom = 2,
/
"""


def test_c1_warns_when_end_hour_has_fewer_values_than_max_dom(tmp_path):
    findings = []
    pf.check_c1_time_windows(wp.parse_namelist(_nl(tmp_path, C1_NAMELIST % "18,")), findings)
    hits = [f for f in findings if f.severity == pf.WARN and "end_hour" in f.message]
    assert len(hits) == 1
    assert "domain 2" in hits[0].message and "Registry default 12" in hits[0].message
    # end_minute = 00 is short too, but 0 IS the Registry default: nothing to warn about
    assert not [f for f in findings if "end_minute" in f.message]


def test_c1_silent_when_every_domain_has_its_own_value(tmp_path):
    findings = []
    pf.check_c1_time_windows(wp.parse_namelist(_nl(tmp_path, C1_NAMELIST % "18, 18,")),
                             findings)
    assert not [f for f in findings if f.severity in (pf.WARN, pf.FATAL)]


# ---- 3. C7: nproc_x/nproc_y = -1 means not set ----------------------------------------

def test_c7_treats_nonpositive_nproc_as_not_set(tmp_path):
    nl = wp.parse_namelist(_nl(tmp_path, "&domains\n e_we = 100,\n e_sn = 100,\n e_vert = 40,\n"
                                         " nproc_x = -1,\n nproc_y = -1,\n/\n"))
    findings = []
    pf.check_c7_decomposition(nl, findings, ranks=72)
    assert not [f for f in findings if f.severity == pf.FATAL]
    findings = []
    pf.check_c7_decomposition(nl, findings)
    assert [f.severity for f in findings] == [pf.INFO]


# ---- 4. TOTAL range with an I/O bracket includes the s/step source noise --------------

PRED_NAMELIST = """&time_control
 run_hours = 1,
 start_year = 2024, start_month = 09, start_day = 05, start_hour = 12,
 history_interval = 60,
/
&domains
 time_step = 60, max_dom = 1, e_we = 100, e_sn = 100, e_vert = 40, dx = 12000, dy = 12000,
/
&physics
 radt = 30,
/
"""
LEDGER_HEAD = "case_base,n_ranks,omp_threads,nproc_x,nproc_y,numtiles,s_per_step_median\n"


def _predict(tmp_path, ledger_rows, config):
    case_dir = tmp_path / "revcase"
    case_dir.mkdir(exist_ok=True)
    (case_dir / "namelist.input").write_text(PRED_NAMELIST)
    ledgers = None
    if ledger_rows:
        ledger = tmp_path / "ledger.csv"
        ledger.write_text(LEDGER_HEAD + "".join(ledger_rows))
        ledgers = [str(ledger)]
    return wp.build_prediction(
        namelist_path=str(case_dir / "namelist.input"), case_dir=str(case_dir), nodes=1,
        config=config, ledger_paths=ledgers, calibration_path=str(wp.DEFAULT_CALIBRATION),
        io_form_override=None, io_target="nfs")


@pytest.mark.parametrize("rows, source, noise_pct", [
    ([], "no_probe_fallback", wp.PROBE_NOISE_FALLBACK_PCT),
    (["revcase,36,1,6,6,1,0.5\n"], "extrapolated", wp.PROBE_NOISE_EXTRAPOLATED_PCT),
    (["revcase,72,1,8,9,1,0.5\n"], "measured", wp.PROBE_NOISE_MEASURED_PCT),
])
def test_io_bracket_range_includes_compute_noise(tmp_path, rows, source, noise_pct):
    pred = _predict(tmp_path, rows, {"ranks": 72, "omp": 1, "nx": 8, "ny": 9, "numtiles": 1})
    assert pred["io_range"] is not None and pred["compute"]["source"] == source
    compute = pred["compute"]["compute_total_s"]
    band = compute * math.hypot(pred["drift_pct"], noise_pct) / 100.0
    assert pred["range_high_s"] == pytest.approx(compute + band + pred["init"]["total_s"])
    assert pred["range_low_s"] == pytest.approx(
        max(0.0, compute - band) + pred["init"]["total_s"])
    # never narrower than the drift-only band it replaces
    assert pred["range_high_s"] - pred["total_s"] >= compute * pred["drift_pct"] / 100.0


# ---- 5. resolve_probe extrapolates by cores, not by rank count ------------------------

def _row(ranks, omp, s):
    return {"case_base": "c", "n_ranks": str(ranks), "omp_threads": str(omp), "nproc_x": "1",
            "nproc_y": str(ranks), "numtiles": "1", "s_per_step_median": str(s)}


def test_resolve_probe_same_cores_different_omp_is_not_rescaled():
    config = {"ranks": 72, "omp": 1, "nx": 8, "ny": 9, "numtiles": 1}
    probe = wp.resolve_probe([_row(36, 2, 1.0)], "c", config)
    assert probe["source"] == "extrapolated"
    assert probe["s_normal"] == pytest.approx(1.0)   # 72 cores -> 72 cores


def test_resolve_probe_picks_the_nearest_row_by_cores():
    config = {"ranks": 72, "omp": 1, "nx": 8, "ny": 9, "numtiles": 1}
    probe = wp.resolve_probe([_row(60, 4, 9.0), _row(36, 2, 1.0)], "c", config)
    assert probe["from_ranks"] == 36                  # 72 cores, not the 240-core row
    # omp 1 everywhere: unchanged rank scaling
    probe = wp.resolve_probe([_row(36, 1, 1.0)], "c", config)
    assert probe["s_normal"] == pytest.approx(0.5)


# ---- 6. resolve_config: --config with nproc_x/nproc_y and no ranks --------------------

def test_resolve_config_derives_ranks_from_nproc_keys():
    args = argparse.Namespace(config=json.dumps({"nproc_x": 8, "nproc_y": 9}), dist=None)
    assert wp.resolve_config(args)["ranks"] == 72
    args = argparse.Namespace(config=json.dumps({"nx": 8, "ny": 9, "ranks": 36}), dist=None)
    assert wp.resolve_config(args)["ranks"] == 36


# ---- 7. budget.nodes: search in the `config show` preview ----------------------------

def test_resolve_node_counts_accepts_search():
    assert wcfg.resolve_node_counts(dict(nodes="search", max_nodes=4)) == \
        wcfg.resolve_node_counts(dict(nodes="auto", max_nodes=4))


# ---- 8/9. metdata check/fetch with no source; grib_state passes grib_dir --------------

def _no_source_plan(md):
    when = datetime.datetime(2010, 6, 1, 0)
    source, reason = md.select_source(when)
    assert source is None
    return [{"valid_time": when, "cycle": when, "fhour": 0, "source": source,
             "reason": reason, "messages": []}]


def test_metdata_check_fails_loudly_without_a_source(capsys):
    md = importlib.import_module("wtune.tools.metdata")
    assert md.do_check(_no_source_plan(md)) == 1
    out = capsys.readouterr().out
    assert "2010-06-01_00" in out and "--grib-dir" in out


def test_metdata_fetch_exits_nonzero_without_a_source(tmp_path, monkeypatch, capsys):
    md = importlib.import_module("wtune.tools.metdata")
    monkeypatch.setattr(md, "build_plan", lambda *a, **k: (_no_source_plan(md), []))
    monkeypatch.setattr(md, "load_case", lambda *a, **k: ({}, None, None))
    rc = md.main(["fetch", "--case-dir", str(tmp_path), "--cache-dir", str(tmp_path / "c")])
    assert rc == 1
    out = capsys.readouterr().out
    assert "2010-06-01_00" in out and "--grib-dir" in out


# ---- 10. ADIOS2 -> netCDF converter: double variables, --step bounds ------------------

@pytest.fixture
def convert_mod(monkeypatch):
    pytest.importorskip("netCDF4")
    if "adios2" not in sys.modules:
        try:
            importlib.import_module("adios2")
        except ImportError:   # only the module object is needed, never a real reader
            monkeypatch.setitem(sys.modules, "adios2", types.ModuleType("adios2"))
    name = "wtune.tools.wrf_adios2_to_netcdf4.convert"
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


class _FakeReader:
    def __init__(self, num_steps):
        self._n = num_steps

    def available_variables(self):
        return {"XLAT_D": {"Type": "double"}, "T": {"Type": "float"}}

    def num_steps(self):
        return self._n

    def read_attribute_string(self, _name, _var):
        return ["west_east", "south_north"]


def test_convert_maps_double_variables(convert_mod):
    out = convert_mod.r_metadata(_FakeReader(3), {"_DIM_west_east": {"Value": "4"}})
    assert out[4] == {"XLAT_D": "d", "T": "f"}


def test_convert_puts_one_frame_in_each_chunk(convert_mod, tmp_path):
    from netCDF4 import Dataset
    dims = {"T": ("Time", "bottom_top", "south_north", "west_east"), "Times": ("Time", "DateStrLen")}
    with Dataset(str(tmp_path / "out.nc"), "w") as d:
        convert_mod.create_nc_dims(d, 121, {"bottom_top": 4, "south_north": 5,
                                            "west_east": 6, "DateStrLen": 19})
        convert_mod.create_nc_vars(d, ["T", "Times"], {"T": "f", "Times": "c"}, dims)
        assert d.variables["T"].chunking() == [1, 4, 5, 6]
        assert d.variables["T"].filters()["zlib"]
        assert not d.variables["Times"].filters()["zlib"]


@pytest.mark.parametrize("step", [3, 7, -4])
def test_convert_rejects_out_of_range_step(convert_mod, monkeypatch, step):
    monkeypatch.setattr(convert_mod, "open_files", lambda *a, **k: (_FakeReader(3), None))
    monkeypatch.setattr(convert_mod, "r_attrs", lambda _f: ({}, {}, {}))

    def must_not_run(*_a, **_k):
        raise AssertionError("conversion started for an out-of-range --step")

    monkeypatch.setattr(convert_mod, "create_nc_dims", must_not_run)
    with pytest.raises(ValueError, match="out of range"):
        convert_mod.convert("in.bp", "out.nc", step=step)
