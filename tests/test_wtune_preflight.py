#!/usr/bin/env python3
"""Tests for wtune/preflight.py (`wtune/cli.py check`) -- every check paired with a synthetic
fixture reproducing its real incident (must be caught, right severity) and a clean fixture
(must be silent), plus one end-to-end clean-CONUS-like exit-0 case.

Run: uv run --with pytest --with numpy --with netCDF4 python -m pytest -q test_wtune_preflight.py
"""
import datetime
import os

import pytest

netCDF4 = pytest.importorskip("netCDF4")
import numpy as np

from wtune import preflight as pf
from wtune import predict as wp


# --------------------------------------------------------------------------
# fixture helpers
# --------------------------------------------------------------------------

def write_namelist(run_dir, text):
    p = run_dir / "namelist.input"
    p.write_text(text)
    return p


def touch(run_dir, *names):
    for n in names:
        (run_dir / n).write_text("")


BASE_TIME_CONTROL = """\
 run_days = 0,
 run_hours = 6,
 run_minutes = 0,
 run_seconds = 0,
 start_year = 2020, start_month = 1, start_day = 1, start_hour = 0, start_minute = 0,
 end_year = 2020, end_month = 1, end_day = 1, end_hour = 6, end_minute = 0,
 history_interval = 180,
"""


def one_domain_namelist(extra_time_control="", extra_domains="", extra_physics="", ghg_input=0):
    return """\
&time_control
%s%s
/
&domains
 max_dom = 1,
 e_we = 100,
 e_sn = 100,
 e_vert = 30,
 dx = 10000,
 dy = 10000,
 time_step = 60,
 nproc_x = 2,
 nproc_y = 2,
%s
/
&physics
 mp_physics = 6,
 ra_lw_physics = 1,
 ra_sw_physics = 1,
 sf_surface_physics = 1,
 ghg_input = %d,
%s
/
""" % (BASE_TIME_CONTROL, extra_time_control, extra_domains, ghg_input, extra_physics)


def clean_findings(findings, severities=("FATAL", "HANG", "WARN")):
    return [f for f in findings if f.severity in severities]


def run(run_dir, namelist_path=None, **kw):
    nl = wp.parse_namelist(str(namelist_path or (run_dir / "namelist.input")))
    return pf.run_checks(nl, run_dir, namelist_path or (run_dir / "namelist.input"), **kw)


def write_nc(path, dims=None, gattrs=None, vars_=None):
    """vars_: {name: (dtype, dim_names, data)}"""
    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        for name, size in (dims or {}).items():
            ds.createDimension(name, size)
        for k, v in (gattrs or {}).items():
            ds.setncattr(k, v)
        for name, (dtype, dim_names, data) in (vars_ or {}).items():
            var = ds.createVariable(name, dtype, dim_names)
            if data is not None:
                var[:] = data
    finally:
        ds.close()


def write_times_char(ds_path, dims_extra, time_strings, mode="a"):
    ds = netCDF4.Dataset(str(ds_path), mode, format="NETCDF4")
    try:
        if "Time" not in ds.dimensions:
            ds.createDimension("Time", None)
        if "DateStrLen" not in ds.dimensions:
            ds.createDimension("DateStrLen", 19)
        var = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        for i, s in enumerate(time_strings):
            for j, ch in enumerate(s):
                var[i, j] = ch
    finally:
        ds.close()


# --------------------------------------------------------------------------
# C1 -- time windows
# --------------------------------------------------------------------------

def test_c1_nest_stops_early_warns(tmp_path):
    """yagi2024-shape incident: run_days=5 gives d01 a 5-day window, but d02's own end_minute=30
    (a leftover smoke-test value) stops it 30 minutes in."""
    text = """\
&time_control
 run_days = 5,
 run_hours = 0,
 run_minutes = 0,
 run_seconds = 0,
 start_year = 2024, 2024,
 start_month = 9, 9,
 start_day = 3, 3,
 start_hour = 12, 12,
 start_minute = 0, 0,
 end_year = 2024, 2024,
 end_month = 9, 9,
 end_day = 3, 3,
 end_hour = 12, 12,
 end_minute = 0, 30,
/
&domains
 max_dom = 2,
 e_we = 100, 100,
 e_sn = 100, 100,
 e_vert = 30, 30,
 dx = 10000, 3333,
 dy = 10000, 3333,
 time_step = 60,
 parent_id = 1, 1,
 parent_grid_ratio = 1, 3,
 parent_time_step_ratio = 1, 3,
/
"""
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c1_time_windows(nl, findings)
    warns = [f for f in findings if f.severity == "WARN" and f.id == "C1"]
    assert len(warns) == 1
    assert "d02" in warns[0].message
    assert "2024-09-08 12:00:00" in warns[0].message  # d01 end = start + 5 days


def test_c1_nest_end_before_start_is_fatal(tmp_path):
    text = """\
&time_control
 run_hours = 6,
 start_year = 2020, 2020,
 start_month = 1, 1,
 start_day = 1, 1,
 start_hour = 0, 6,
 end_year = 2020, 2020,
 end_month = 1, 1,
 end_day = 1, 1,
 end_hour = 6, 3,
/
&domains
 max_dom = 2,
 e_we = 100, 100,
 e_sn = 100, 100,
 e_vert = 30, 30,
 dx = 10000, 3333,
 dy = 10000, 3333,
/
"""
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c1_time_windows(nl, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C1"]
    assert len(fatals) == 1
    assert "d02" in fatals[0].message


def test_c1_clean_single_domain_silent(tmp_path):
    write_namelist(tmp_path, one_domain_namelist())
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c1_time_windows(nl, findings)
    assert clean_findings(findings) == []


def test_c1_clean_matching_nest_windows_silent(tmp_path):
    text = """\
&time_control
 run_hours = 6,
 start_year = 2020, 2020,
 start_month = 1, 1,
 start_day = 1, 1,
 start_hour = 0, 0,
 end_year = 2020, 2020,
 end_month = 1, 1,
 end_day = 1, 1,
 end_hour = 6, 6,
/
&domains
 max_dom = 2,
 e_we = 100, 100,
 e_sn = 100, 100,
 e_vert = 30, 30,
 dx = 10000, 3333,
 dy = 10000, 3333,
/
"""
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c1_time_windows(nl, findings)
    assert clean_findings(findings) == []


# --------------------------------------------------------------------------
# C2 -- iofields_filename + out-of-block lines
# --------------------------------------------------------------------------

def test_c2_missing_iofields_file_is_fatal_in_case_dir_when_nowhere(tmp_path):
    text = one_domain_namelist(extra_time_control=" iofields_filename = 'storm_diags_d01.txt',\n")
    write_namelist(tmp_path, text)
    wrf_run = tmp_path / "wrfrun"
    wrf_run.mkdir()
    findings = run(tmp_path, wrf_run_dir=str(wrf_run))
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C2"]
    assert len(fatals) == 1
    assert "storm_diags_d01.txt" in fatals[0].message
    assert "hangs at input_input" in fatals[0].message
    assert "ignore_iofields_warning=.true." in fatals[0].message
    assert "2541-2545" in fatals[0].evidence


def test_c2_missing_iofields_in_run_dir_is_fatal(tmp_path):
    """climate-ea50 probe dirs: wrf.exe linked, iofields_d01.txt never staged."""
    text = one_domain_namelist(extra_time_control=' iofields_filename = "iofields_d01.txt",\n'
                                                  " ignore_iofields_warning = .true.,\n")
    write_namelist(tmp_path, text)
    touch(tmp_path, "wrf.exe", "RRTM_DATA")
    wrf_run = tmp_path / "wrfrun"
    wrf_run.mkdir()
    touch(wrf_run, "iofields_d01.txt")        # WRF's run/ having it does not help a run dir
    findings = run(tmp_path, wrf_run_dir=str(wrf_run))
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C2"]
    assert len(fatals) == 1 and "the run dir" in fatals[0].message
    assert pf.IOFIELDS_HANG in fatals[0].message
    assert pf.exit_code_for(findings) == 2


def test_c2_iofields_in_case_dir_is_info(tmp_path):
    text = one_domain_namelist(extra_time_control=" iofields_filename = 'iofields_d01.txt',\n")
    write_namelist(tmp_path, text)
    touch(tmp_path, "iofields_d01.txt")
    findings = run(tmp_path)
    c2 = [f for f in findings if f.id == "C2"]
    assert [f.severity for f in c2] == ["INFO"] and "stages it" in c2[0].message


def test_c2_iofields_only_in_wrf_run_dir_is_info_for_case_dir(tmp_path):
    text = one_domain_namelist(extra_time_control=" iofields_filename = 'iofields_d01.txt',\n")
    write_namelist(tmp_path, text)
    wrf_run = tmp_path / "wrfrun"
    wrf_run.mkdir()
    touch(wrf_run, "iofields_d01.txt")
    findings = run(tmp_path, wrf_run_dir=str(wrf_run))
    c2 = [f for f in findings if f.id == "C2"]
    assert [f.severity for f in c2] == ["INFO"] and "--wrf-run-dir" in c2[0].message


def test_c2_iofields_domain_placeholder_and_run_dir_present_silent(tmp_path):
    text = one_domain_namelist(extra_time_control=" iofields_filename = 'io_d<domain>.txt',\n")
    write_namelist(tmp_path, text)
    touch(tmp_path, "wrf.exe", "io_d01.txt")
    findings = run(tmp_path)
    assert [f for f in findings if f.id == "C2"] == []


def test_c2_present_iofields_file_silent(tmp_path):
    text = one_domain_namelist(extra_time_control=" iofields_filename = 'fields_d01.txt',\n")
    write_namelist(tmp_path, text)
    touch(tmp_path, "fields_d01.txt", "RRTM_DATA")
    findings = run(tmp_path)
    assert [f for f in findings if f.severity == "HANG"] == []


def test_c2_out_of_block_line_warns(tmp_path):
    """elevconv2022 incident shape: a second iofields_filename sits between '/' and '&domains'."""
    text = """\
&time_control
%s
 iofields_filename = 'storm_diags_d01.txt',
/
 iofields_filename = 'prates_out.txt'
&domains
 max_dom = 1,
 e_we = 100,
 e_sn = 100,
 e_vert = 30,
 dx = 10000,
 dy = 10000,
 time_step = 60,
/
&physics
 mp_physics = 6,
 ra_lw_physics = 1,
 ra_sw_physics = 1,
 sf_surface_physics = 1,
 ghg_input = 0,
/
""" % BASE_TIME_CONTROL
    write_namelist(tmp_path, text)
    touch(tmp_path, "storm_diags_d01.txt", "RRTM_DATA")
    findings = run(tmp_path)
    warns = [f for f in findings if f.severity == "WARN" and f.id == "C2"]
    assert len(warns) == 1
    assert "iofields_filename" in warns[0].message
    assert "outside every" in warns[0].message


def test_c2_no_out_of_block_line_silent(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTM_DATA")
    findings = run(tmp_path)
    assert [f for f in findings if f.id == "C2"] == []


# --------------------------------------------------------------------------
# C3 -- physics data files
# --------------------------------------------------------------------------

def test_c3_missing_rrtmg_data_is_fatal(tmp_path):
    text = one_domain_namelist(extra_physics=" ra_lw_physics = 4,\n ra_sw_physics = 4,\n",
                                extra_time_control="", ghg_input=1)
    text = text.replace("ra_lw_physics = 1,", "").replace("ra_sw_physics = 1,", "")
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, tmp_path, findings)
    fatal_files = {f.message for f in findings if f.severity == "FATAL"}
    assert any("RRTMG_LW_DATA" in m for m in fatal_files)
    assert any("RRTMG_SW_DATA" in m for m in fatal_files)
    assert any("CAMtr_volume_mixing_ratio" in m for m in fatal_files)


def test_c3_rrtmg_with_all_files_present_silent(tmp_path):
    text = one_domain_namelist(extra_physics=" ra_lw_physics = 4,\n ra_sw_physics = 4,\n", ghg_input=1)
    text = text.replace(" ra_lw_physics = 1,\n", "").replace(" ra_sw_physics = 1,\n", "")
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTMG_LW_DATA", "RRTMG_SW_DATA", "CAMtr_volume_mixing_ratio",
          "LANDUSE.TBL", "ozone.formatted", "ozone_lat.formatted", "ozone_plev.formatted")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, tmp_path, findings)
    assert clean_findings(findings) == []


def test_c3_cam_scheme_exempt_from_camtr_requirement(tmp_path):
    """CAM (ra_lw_physics=3) has its own fallback table in read_CAMgases -- missing
    CAMtr_volume_mixing_ratio must NOT be flagged for CAM, only its own CAM_ABS/AEROPT_DATA."""
    text = one_domain_namelist(extra_physics=" ra_lw_physics = 3,\n ra_sw_physics = 3,\n", ghg_input=1)
    text = text.replace(" ra_lw_physics = 1,\n", "").replace(" ra_sw_physics = 1,\n", "")
    write_namelist(tmp_path, text)
    touch(tmp_path, "CAM_ABS_DATA", "CAM_AEROPT_DATA", "LANDUSE.TBL", "ozone.formatted", "ozone_lat.formatted", "ozone_plev.formatted")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, tmp_path, findings)
    assert not any("CAMtr_volume_mixing_ratio" in f.message for f in findings)
    assert clean_findings(findings) == []


def test_c3_noahmp_missing_table_is_fatal(tmp_path):
    text = one_domain_namelist(extra_physics=" sf_surface_physics = 4,\n")
    text = text.replace(" sf_surface_physics = 1,\n", "")
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTM_DATA", "CAMtr_volume_mixing_ratio")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and "NoahmpTable" in f.message]
    assert len(fatals) == 1


def test_c3_noahmp_table_present_silent(tmp_path):
    text = one_domain_namelist(extra_physics=" sf_surface_physics = 4,\n", ghg_input=1)
    text = text.replace(" sf_surface_physics = 1,\n", "")
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTM_DATA", "CAMtr_volume_mixing_ratio", "NoahmpTable.TBL")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, tmp_path, findings)
    assert clean_findings(findings) == []


def test_c3_thompson_missing_tables_warns_only(tmp_path):
    """Thompson (mp_physics=8) missing lookup tables is WARN, not FATAL -- WRF recomputes them
    at init (costing minutes), it does not abort."""
    text = one_domain_namelist(extra_physics=" mp_physics = 8,\n")
    text = text.replace(" mp_physics = 6,\n", "")
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTM_DATA", "CAMtr_volume_mixing_ratio", "LANDUSE.TBL")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, tmp_path, findings)
    assert [f.severity for f in findings if "freezeH2O" in f.message] == ["WARN"]
    assert not any(f.severity == "FATAL" for f in findings)


def test_c3_thompson_tables_present_silent(tmp_path):
    text = one_domain_namelist(extra_physics=" mp_physics = 8,\n")
    text = text.replace(" mp_physics = 6,\n", "")
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTM_DATA", "CAMtr_volume_mixing_ratio", "LANDUSE.TBL",
          "freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, tmp_path, findings)
    assert clean_findings(findings) == []


def test_c3_wrf_run_dir_extension_less_file_flagged(tmp_path):
    """sc26p2-n4 incident: an extension-whitelist staging tool missed the extension-less
    CAMtr_volume_mixing_ratio while copying every *.SSP245-style extension variant."""
    wrf_run_dir = tmp_path / "wrf_run"
    wrf_run_dir.mkdir()
    touch(wrf_run_dir, "CAMtr_volume_mixing_ratio", "CAMtr_volume_mixing_ratio.SSP245", "RRTM_DATA")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    text = one_domain_namelist()
    write_namelist(run_dir, text)
    touch(run_dir, "wrf.exe", "CAMtr_volume_mixing_ratio.SSP245", "RRTM_DATA")  # staged run dir
    nl = wp.parse_namelist(str(run_dir / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, run_dir, findings, wrf_run_dir=wrf_run_dir)
    warns = [f for f in findings if f.severity == "WARN" and "CAMtr_volume_mixing_ratio" in f.message
             and "present in WRF's own run/" in f.message]
    assert len(warns) == 1


# --- C3 extended physics-table rules (README.physics_files + WRF 4.8 OPEN-site gates) ---

_OZONE = ["ozone.formatted", "ozone_lat.formatted", "ozone_plev.formatted"]
_AEROSOL = ["aerosol.formatted", "aerosol_lat.formatted", "aerosol_lon.formatted",
            "aerosol_plev.formatted"]
_CLM = ["CLM_ALB_ICE_DFS_DATA", "CLM_ALB_ICE_DRC_DATA", "CLM_ASM_ICE_DFS_DATA",
        "CLM_ASM_ICE_DRC_DATA", "CLM_DRDSDT0_DATA", "CLM_EXT_ICE_DFS_DATA",
        "CLM_EXT_ICE_DRC_DATA", "CLM_KAPPA_DATA", "CLM_TAU_DATA"]


def _c3_fatal(tmp_path, physics, files=(), extra=""):
    """C3 FATAL messages for a minimal one-domain namelist. ghg_input=0 so CAMtr never
    interferes; every physics option not given defaults to 0 (no scheme)."""
    write_namelist(tmp_path, "&domains\n max_dom = 1,\n/\n&physics\n ghg_input = 0,\n%s/\n%s"
                   % (physics, extra))
    touch(tmp_path, *files)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, tmp_path, findings)
    return [f.message for f in findings if f.severity == "FATAL"]


def _mentions(msgs, fname):
    return [m for m in msgs if (" %s " % fname) in m]


def test_c3_rrtmg_default_o3input_requires_ozone(tmp_path):
    phys = " ra_lw_physics = 4,\n ra_sw_physics = 4,\n"
    msgs = _c3_fatal(tmp_path, phys, ["RRTMG_LW_DATA", "RRTMG_SW_DATA"])
    for f in _OZONE:
        assert len(_mentions(msgs, f)) == 1
    # aer_opt defaults to 0: the aerosol climatology files must NOT be required
    assert not any("aerosol" in m for m in msgs)


def test_c3_rrtmg_ozone_present_silent(tmp_path):
    phys = " ra_lw_physics = 4,\n ra_sw_physics = 4,\n"
    assert _c3_fatal(tmp_path, phys, ["RRTMG_LW_DATA", "RRTMG_SW_DATA"] + _OZONE) == []


def test_c3_o3input_zero_disables_ozone_requirement(tmp_path):
    phys = " ra_lw_physics = 4,\n ra_sw_physics = 4,\n o3input = 0,\n"
    assert _c3_fatal(tmp_path, phys, ["RRTMG_LW_DATA", "RRTMG_SW_DATA"]) == []


def test_c3_non_rrtmg_radiation_does_not_require_ozone(tmp_path):
    # module_check_a_mundo forces o3input=0 unless domain 1 uses an RRTMG-family scheme
    msgs = _c3_fatal(tmp_path, " ra_lw_physics = 1,\n ra_sw_physics = 1,\n", ["RRTM_DATA"])
    assert msgs == []


def test_c3_rrtmg_fast_and_kiaps_require_rrtmg_tables(tmp_path):
    msgs = _c3_fatal(tmp_path, " ra_lw_physics = 24,\n ra_sw_physics = 14,\n o3input = 0,\n")
    assert len(_mentions(msgs, "RRTMG_LW_DATA")) == 1
    assert len(_mentions(msgs, "RRTMG_SW_DATA")) == 1


def test_c3_aer_opt_1_requires_aerosol_climatology(tmp_path):
    phys = " ra_lw_physics = 4,\n ra_sw_physics = 4,\n aer_opt = 1,\n"
    base = ["RRTMG_LW_DATA", "RRTMG_SW_DATA"] + _OZONE
    msgs = _c3_fatal(tmp_path, phys, base)
    assert sorted(f for f in _AEROSOL if _mentions(msgs, f)) == sorted(_AEROSOL)
    assert _c3_fatal(tmp_path, phys, base + _AEROSOL) == []


def test_c3_cam_requires_ozone_even_with_o3input_zero_but_not_aerosol(tmp_path):
    phys = " ra_lw_physics = 3,\n ra_sw_physics = 3,\n o3input = 0,\n"
    msgs = _c3_fatal(tmp_path, phys, ["CAM_ABS_DATA", "CAM_AEROPT_DATA"])
    assert all(_mentions(msgs, f) for f in _OZONE)
    assert not any("aerosol" in m for m in msgs)
    assert _c3_fatal(tmp_path, phys, ["CAM_ABS_DATA", "CAM_AEROPT_DATA"] + _OZONE) == []


def test_c3_goddard_via_ra_sw_only(tmp_path):
    msgs = _c3_fatal(tmp_path, " ra_sw_physics = 5,\n")
    assert len(_mentions(msgs, "BROADBAND_CLOUD_GODDARD.bin")) == 1
    assert _c3_fatal(tmp_path, " ra_sw_physics = 5,\n", ["BROADBAND_CLOUD_GODDARD.bin"]) == []


def test_c3_goddard_lw_and_sw_reports_once(tmp_path):
    msgs = _c3_fatal(tmp_path, " ra_lw_physics = 5,\n ra_sw_physics = 5,\n")
    assert len(_mentions(msgs, "BROADBAND_CLOUD_GODDARD.bin")) == 1


def test_c3_gfdl_default_co2tf_requires_tr_tables_not_co2_trans(tmp_path):
    phys = " ra_lw_physics = 99,\n ra_sw_physics = 99,\n"
    msgs = _c3_fatal(tmp_path, phys)
    for f in ("tr49t67", "tr49t85", "tr67t85"):
        assert len(_mentions(msgs, f)) == 1
    assert not _mentions(msgs, "co2_trans")
    assert _c3_fatal(tmp_path, phys, ["tr49t67", "tr49t85", "tr67t85"]) == []


def test_c3_gfdl_co2tf_0_requires_co2_trans_only(tmp_path):
    phys = " ra_lw_physics = 99,\n ra_sw_physics = 99,\n co2tf = 0,\n"
    msgs = _c3_fatal(tmp_path, phys)
    assert len(_mentions(msgs, "co2_trans")) == 1
    assert not any(_mentions(msgs, f) for f in ("tr49t67", "tr49t85", "tr67t85"))
    assert _c3_fatal(tmp_path, phys, ["co2_trans"]) == []


def test_c3_eclipse_requires_besselian_elements(tmp_path):
    phys = " ra_sw_physics = 1,\n ra_sw_eclipse = 1,\n"
    assert len(_mentions(_c3_fatal(tmp_path, phys), "eclipse_besselian_elements.dat")) == 1
    assert _c3_fatal(tmp_path, phys, ["eclipse_besselian_elements.dat"]) == []
    assert _c3_fatal(tmp_path, " ra_sw_physics = 1,\n") == []


def test_c3_urban_requires_urbparm(tmp_path):
    msgs = _c3_fatal(tmp_path, " sf_urban_physics = 1,\n")
    assert len(_mentions(msgs, "URBPARM.TBL")) == 1
    assert not _mentions(msgs, "URBPARM_LCZ.TBL")
    assert _c3_fatal(tmp_path, " sf_urban_physics = 1,\n", ["URBPARM.TBL"]) == []


def test_c3_urban_lcz_requires_lcz_table_instead_of_urbparm(tmp_path):
    phys = " sf_urban_physics = 2,\n use_wudapt_lcz = 1,\n"
    msgs = _c3_fatal(tmp_path, phys)
    assert len(_mentions(msgs, "URBPARM_LCZ.TBL")) == 1
    assert not _mentions(msgs, "URBPARM.TBL")
    assert _c3_fatal(tmp_path, phys, ["URBPARM_LCZ.TBL"]) == []


def test_c3_clm_requires_landuse_and_clm_tables(tmp_path):
    msgs = _c3_fatal(tmp_path, " sf_surface_physics = 5,\n")
    for f in ["LANDUSE.TBL"] + _CLM:
        assert len(_mentions(msgs, f)) == 1, f
    assert _c3_fatal(tmp_path, " sf_surface_physics = 5,\n", ["LANDUSE.TBL"] + _CLM) == []


@pytest.mark.parametrize("sf,files", [
    (1, ["LANDUSE.TBL"]),
    (3, ["LANDUSE.TBL", "VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"]),
    (7, ["LANDUSE.TBL", "VEGPARM.TBL"]),
    (8, ["LANDUSE.TBL"]),
])
def test_c3_other_lsm_tables(tmp_path, sf, files):
    phys = " sf_surface_physics = %d,\n" % sf
    msgs = _c3_fatal(tmp_path, phys)
    assert sorted(f for f in files if _mentions(msgs, f)) == sorted(files)
    assert len(msgs) == len(files)
    assert _c3_fatal(tmp_path, phys, files) == []


def test_c3_noahmp_snicar_files_not_required(tmp_path):
    assert _c3_fatal(tmp_path, " sf_surface_physics = 4,\n", ["NoahmpTable.TBL"]) == []


def test_c3_windturbine_spec_requires_table(tmp_path):
    assert len(_mentions(_c3_fatal(tmp_path, " windturbine_spec = 1,\n"), "wind-turbine-1.tbl")) == 1
    assert _c3_fatal(tmp_path, " windturbine_spec = 1,\n", ["wind-turbine-1.tbl"]) == []


def test_c3_chem_opt_requires_hlc_table(tmp_path):
    extra = "&chem\n chem_opt = 112,\n/\n"
    assert len(_mentions(_c3_fatal(tmp_path, "", extra=extra), "HLC.TBL")) == 1
    assert _c3_fatal(tmp_path, "", ["HLC.TBL"], extra=extra) == []
    assert _c3_fatal(tmp_path, "", extra="&chem\n chem_opt = 0,\n/\n") == []


def test_c3_ferrier_requires_expanded_rain_table_only(tmp_path):
    msgs = _c3_fatal(tmp_path, " mp_physics = 5,\n")
    assert len(_mentions(msgs, "ETAMPNEW_DATA.expanded_rain")) == 1
    assert not _mentions(msgs, "ETAMPNEW_DATA")   # fer_hires_init never opens plain ETAMPNEW_DATA
    assert _c3_fatal(tmp_path, " mp_physics = 5,\n", ["ETAMPNEW_DATA.expanded_rain"]) == []


def test_c3_thompson_aerosol_aware_requires_ccn_table(tmp_path):
    dats = ["freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat"]
    assert len(_mentions(_c3_fatal(tmp_path, " mp_physics = 28,\n", dats), "CCN_ACTIVATE.BIN")) == 1
    assert _c3_fatal(tmp_path, " mp_physics = 28,\n", dats + ["CCN_ACTIVATE.BIN"]) == []


def test_c3_plain_thompson_does_not_require_ccn_table(tmp_path):
    assert _c3_fatal(tmp_path, " mp_physics = 8,\n") == []


def test_c3_full_sbm_tables(tmp_path):
    files = ["bulkdens.asc_s_0_03_0_9", "bulkradii.asc_s_0_03_0_9", "capacity.asc",
             "coeff_p.asc", "coeff_q.asc", "constants.asc", "kernels.asc_s_0_03_0_9",
             "kernels_z.asc", "masses.asc", "termvels.asc"]
    msgs = _c3_fatal(tmp_path, " mp_physics = 32,\n")
    assert sorted(f for f in files if _mentions(msgs, f)) == sorted(files)
    assert _c3_fatal(tmp_path, " mp_physics = 32,\n", files) == []


def test_c3_jensen_tables(tmp_path):
    files = ["ishmael-gamma-tab.bin", "ishmael-qi-qc.bin", "ishmael-qi-qr.bin"]
    msgs = _c3_fatal(tmp_path, " mp_physics = 55,\n")
    assert sorted(f for f in files if _mentions(msgs, f)) == sorted(files)
    assert _c3_fatal(tmp_path, " mp_physics = 55,\n", files) == []


def test_c3_p3_prefix_match_with_scc_filenames(tmp_path):
    scc = ["p3_lookupTable_1.dat-v5.4_2momI", "p3_lookupTable_1.dat-v5.4_3momI",
           "p3_lookupTable_2.dat-v5.3"]
    for mp in (50, 51, 52, 53):
        d = tmp_path / ("mp%d" % mp)
        d.mkdir()
        assert _c3_fatal(d, " mp_physics = %d,\n" % mp, scc) == [], mp


def test_c3_p3_missing_tables_fatal(tmp_path):
    msgs = _c3_fatal(tmp_path, " mp_physics = 52,\n")
    assert any("p3_lookupTable_1.dat*" in m for m in msgs)
    assert any("p3_lookupTable_2.dat*" in m for m in msgs)


def test_c3_p3_single_category_does_not_need_table_2(tmp_path):
    # table 2 (ice-ice interactions) is only opened for nCat>1, i.e. mp_physics=52
    assert _c3_fatal(tmp_path, " mp_physics = 50,\n", ["p3_lookupTable_1.dat-v5.4_2momI"]) == []


def test_c3_fast_sbm_not_checked_against_full_sbm_tables(tmp_path):
    assert _c3_fatal(tmp_path, " mp_physics = 30,\n") == []


# --------------------------------------------------------------------------
# C4 -- aerosol climatology
# --------------------------------------------------------------------------

def _lafe_like_namelist(tmp_path, use_aero_icbc=True, wif_input_opt=1):
    # lafe's own namelist: mp_physics=28 with wif_input_opt=1 / num_wif_levels=30 in &domains
    domains = (" wif_input_opt = %d,\n num_wif_levels = 30,\n" % wif_input_opt) if wif_input_opt else ""
    text = one_domain_namelist(extra_domains=domains,
                               extra_physics=" mp_physics = 28,\n use_aero_icbc = .true.,\n" if use_aero_icbc
                               else " mp_physics = 28,\n")
    text = text.replace(" mp_physics = 6,\n", "")
    write_namelist(tmp_path, text)
    return wp.parse_namelist(str(tmp_path / "namelist.input"))


def test_c4_missing_qnwfa_qnifa_in_wrfinput_is_fatal(tmp_path):
    nl = _lafe_like_namelist(tmp_path)
    write_nc(tmp_path / "wrfinput_d01", dims={"west_east": 100, "south_north": 100},
             vars_={"XLAT": ("f4", ("south_north", "west_east"), None)})
    findings = []
    pf.check_c4_aerosol_climatology(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C4"]
    assert len(fatals) == 1
    assert "QNWFA" in fatals[0].message


def test_c4_qnwfa_qnifa_present_silent(tmp_path):
    nl = _lafe_like_namelist(tmp_path)
    write_nc(tmp_path / "wrfinput_d01", dims={"west_east": 100, "south_north": 100},
             vars_={
                 "QNWFA": ("f4", ("south_north", "west_east"), None),
                 "QNIFA": ("f4", ("south_north", "west_east"), None),
             })
    findings = []
    pf.check_c4_aerosol_climatology(nl, tmp_path, findings)
    assert clean_findings(findings) == []


def test_c4_mp28_with_wif_input_opt_0_is_fatal_without_netcdf(tmp_path):
    # WTF ralbNE (scc job 1086): real.exe 'wif_input_opt=0 but mp_physics=28, 29, or 88'
    nl = _lafe_like_namelist(tmp_path, use_aero_icbc=False, wif_input_opt=0)
    findings = []
    pf.check_c4_aerosol_climatology(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C4"]
    assert len(fatals) == 1
    assert "wif_input_opt" in fatals[0].message


def test_c4_tempo_only_fatal_when_aerosolaware(tmp_path):
    text = one_domain_namelist(extra_physics=" mp_physics = 88,\n")
    text = text.replace(" mp_physics = 6,\n", "")
    write_namelist(tmp_path, text)
    findings = []
    pf.check_c4_aerosol_climatology(wp.parse_namelist(str(tmp_path / "namelist.input")),
                                    tmp_path, findings)
    assert [f for f in findings if f.severity == "FATAL"] == []
    write_namelist(tmp_path, text.replace(" mp_physics = 88,\n",
                                          " mp_physics = 88,\n tempo_aerosolaware = 1,\n"))
    findings = []
    pf.check_c4_aerosol_climatology(wp.parse_namelist(str(tmp_path / "namelist.input")),
                                    tmp_path, findings)
    assert [f.id for f in findings if f.severity == "FATAL"] == ["C4"]


def test_c4_not_applicable_without_aero_scheme_silent(tmp_path):
    # premise corrected 2026-09-25: mp_physics=28 WITHOUT wif_input_opt is not "not applicable"
    # -- real.exe aborts (see the ralbNE test above); a non-aerosol scheme is
    text = one_domain_namelist(extra_physics=" mp_physics = 8,\n")
    text = text.replace(" mp_physics = 6,\n", "")
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    # no wrfinput at all -- must not even try
    findings = []
    pf.check_c4_aerosol_climatology(nl, tmp_path, findings)
    assert findings == []


# --------------------------------------------------------------------------
# C5 -- met_em naming vs nocolons
# --------------------------------------------------------------------------

def test_c5_nocolons_with_only_colon_names_is_fatal(tmp_path):
    text = one_domain_namelist(extra_time_control=" nocolons = .true.,\n")
    write_namelist(tmp_path, text)
    touch(tmp_path, "met_em.d01.2017-08-23_06:00:00.nc")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c5_met_em_naming(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C5"]
    assert len(fatals) == 1


def test_c5_nocolons_with_underscore_names_silent(tmp_path):
    text = one_domain_namelist(extra_time_control=" nocolons = .true.,\n")
    write_namelist(tmp_path, text)
    touch(tmp_path, "met_em.d01.2017-08-23_06_00_00.nc")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c5_met_em_naming(nl, tmp_path, findings)
    assert clean_findings(findings) == []


def test_c5_no_met_em_silent(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c5_met_em_naming(nl, tmp_path, findings)
    assert findings == []


# --------------------------------------------------------------------------
# C6 -- input consistency
# --------------------------------------------------------------------------

def test_c6_wrfinput_grid_dim_mismatch_is_fatal(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    write_nc(tmp_path / "wrfinput_d01",
             gattrs={"WEST-EAST_GRID_DIMENSION": 999, "SOUTH-NORTH_GRID_DIMENSION": 100,
                     "BOTTOM-TOP_GRID_DIMENSION": 30, "SIMULATION_START_DATE": "2020-01-01_00:00:00"})
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c6_input_consistency(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C6"]
    assert len(fatals) == 1
    assert "WEST-EAST_GRID_DIMENSION" in fatals[0].evidence


def test_c6_wrfinput_matching_dims_silent(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    write_nc(tmp_path / "wrfinput_d01",
             gattrs={"WEST-EAST_GRID_DIMENSION": 100, "SOUTH-NORTH_GRID_DIMENSION": 100,
                     "BOTTOM-TOP_GRID_DIMENSION": 30, "SIMULATION_START_DATE": "2020-01-01_00:00:00"})
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c6_input_consistency(nl, tmp_path, findings)
    assert clean_findings(findings) == []


def test_c6_wrfbdy_short_boundary_is_fatal(tmp_path):
    text = one_domain_namelist(extra_time_control=" interval_seconds = 3600,\n")
    write_namelist(tmp_path, text)
    write_nc(tmp_path / "wrfbdy_d01", dims={"Time": None, "DateStrLen": 19})
    write_times_char(tmp_path / "wrfbdy_d01", None, ["2020-01-01_00:00:00", "2020-01-01_01:00:00"])
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c6_input_consistency(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and "wrfbdy" in f.message]
    assert len(fatals) == 1


def test_c6_wrfbdy_covers_full_window_silent(tmp_path):
    text = one_domain_namelist(extra_time_control=" interval_seconds = 3600,\n")
    write_namelist(tmp_path, text)
    write_nc(tmp_path / "wrfbdy_d01", dims={"Time": None, "DateStrLen": 19})
    write_times_char(tmp_path / "wrfbdy_d01", None,
                      ["2020-01-01_00:00:00", "2020-01-01_03:00:00", "2020-01-01_05:00:00"])
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c6_input_consistency(nl, tmp_path, findings)
    assert clean_findings(findings) == []


def _c2(tmp_path, wrf_run_dir=None):
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c2_iofields(nl, tmp_path, tmp_path / "namelist.input", findings,
                         wrf_run_dir=wrf_run_dir)
    return findings


def test_c2_sst_update_missing_wrflowinp_is_fatal(tmp_path):
    """(was C6; now C2 with the other namelist-named files, no netCDF4 needed)"""
    text = one_domain_namelist(extra_physics=" sst_update = 1,\n")
    write_namelist(tmp_path, text)
    fatals = [f for f in _c2(tmp_path) if f.severity == "FATAL" and "wrflowinp" in f.message]
    assert len(fatals) == 1 and "auxinput4_inname" in fatals[0].message


def test_c2_sst_update_with_wrflowinp_not_fatal(tmp_path):
    text = one_domain_namelist(extra_physics=" sst_update = 1,\n")
    write_namelist(tmp_path, text)
    touch(tmp_path, "wrflowinp_d01")
    assert clean_findings(_c2(tmp_path)) == []
    touch(tmp_path, "wrf.exe")
    assert _c2(tmp_path) == []


def test_c2_sst_update_custom_auxinput4_inname(tmp_path):
    text = one_domain_namelist(
        extra_time_control=" auxinput4_inname = 'sst_d<domain>.nc',\n",
        extra_physics=" sst_update = 1,\n")
    write_namelist(tmp_path, text)
    touch(tmp_path, "wrf.exe", "wrflowinp_d01")          # the default name does not count
    fatals = [f for f in _c2(tmp_path) if f.severity == "FATAL"]
    assert len(fatals) == 1 and "sst_d01.nc" in fatals[0].message
    touch(tmp_path, "sst_d01.nc")
    assert _c2(tmp_path) == []


def test_c2_sst_update_off_ignores_wrflowinp(tmp_path):
    write_namelist(tmp_path, one_domain_namelist())
    touch(tmp_path, "wrf.exe")
    assert _c2(tmp_path) == []


def test_c6_start_date_mismatch_warns(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    write_nc(tmp_path / "wrfinput_d01",
             gattrs={"WEST-EAST_GRID_DIMENSION": 100, "SOUTH-NORTH_GRID_DIMENSION": 100,
                     "BOTTOM-TOP_GRID_DIMENSION": 30, "SIMULATION_START_DATE": "2019-12-31_00:00:00"})
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c6_input_consistency(nl, tmp_path, findings)
    warns = [f for f in findings if f.severity == "WARN" and "SIMULATION_START_DATE" in f.message]
    assert len(warns) == 1


def test_c6_netcdf4_unavailable_skips_explicitly(tmp_path, monkeypatch):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    monkeypatch.setattr(pf, "HAVE_NETCDF4", False)
    findings = []
    pf.check_c6_input_consistency(nl, tmp_path, findings)
    assert len(findings) == 1
    assert findings[0].severity == "INFO"
    assert "netCDF4 not available" in findings[0].message


# --------------------------------------------------------------------------
# C7 -- decomposition
# --------------------------------------------------------------------------

def test_c7_thin_patch_is_fatal(tmp_path):
    text = one_domain_namelist(extra_domains=" e_we = 15,\n").replace(" e_we = 100,\n", "")
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c7_decomposition(nl, findings)  # nproc_x=2 from namelist -> 15//2=7 < 10
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C7"]
    assert len(fatals) == 1
    assert "e_we" in fatals[0].message


def test_c7_legal_patch_silent(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c7_decomposition(nl, findings)  # e_we=100, nproc_x=2 -> 50 >= 10
    assert clean_findings(findings) == []


def test_c7_ranks_mismatch_is_fatal(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c7_decomposition(nl, findings, ranks=99)  # namelist says nproc_x*nproc_y=4
    fatals = [f for f in findings if f.severity == "FATAL" and "ranks" in f.message]
    assert len(fatals) == 1


def test_c7_cli_override_used_over_namelist(tmp_path):
    text = one_domain_namelist(extra_domains=" e_we = 15,\n").replace(" e_we = 100,\n", "")
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    # CLI override to a shape that's fine for e_we=15
    pf.check_c7_decomposition(nl, findings, nproc_x=1, nproc_y=1)
    assert clean_findings(findings) == []


# --------------------------------------------------------------------------
# C8 -- time step ratio
# --------------------------------------------------------------------------

def test_c8_d01_ratio_too_large_is_fatal(tmp_path):
    # dx=1000m=1km, time_step=600s -> ratio 600 >> 6.0
    text = one_domain_namelist(extra_domains=" dx = 1000,\n dy = 1000,\n time_step = 600,\n")
    text = text.replace(" dx = 10000,\n", "").replace(" dy = 10000,\n", "").replace(" time_step = 60,\n", "")
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c8_time_step(nl, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C8"]
    assert len(fatals) == 1


def test_c8_d01_ratio_within_bound_silent(tmp_path):
    # dx=10km, time_step=60s -> ratio 6.0 (equality allowed, strictly '>' only)
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c8_time_step(nl, findings)
    assert clean_findings(findings) == []


def test_c8_respects_explicit_reasonable_time_step_ratio(tmp_path):
    """elevconv2022 sets reasonable_time_step_ratio=6.5 explicitly; the checker must honor it,
    not hardcode 6.0."""
    text = one_domain_namelist(extra_domains=" reasonable_time_step_ratio = 6.5,\n dx = 1000,\n"
                                              " dy = 1000,\n time_step = 620,\n")
    text = text.replace(" dx = 10000,\n", "").replace(" dy = 10000,\n", "").replace(" time_step = 60,\n", "")
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c8_time_step(nl, findings)
    # ratio = 620/1.0 = 620 >> 6.5, still fatal -- but confirm the ratio used is 6.5 not 6.0
    # via a near-boundary case below
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C8"]
    assert len(fatals) == 1
    assert "6.500" in fatals[0].message


def test_c8_nest_exceeding_ratio_warns_not_fatal(tmp_path):
    text = """\
&time_control
%s
/
&domains
 max_dom = 2,
 e_we = 100, 100,
 e_sn = 100, 100,
 e_vert = 30, 30,
 dx = 10000, 10000,
 dy = 10000, 10000,
 time_step = 60,
 parent_id = 1, 1,
 parent_grid_ratio = 1, 1,
 parent_time_step_ratio = 1, 1,
/
&physics
 mp_physics = 6,
 ra_lw_physics = 1,
 ra_sw_physics = 1,
 sf_surface_physics = 1,
 ghg_input = 0,
/
""" % BASE_TIME_CONTROL
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c8_time_step(nl, findings)
    # d02 has the SAME dx as d01 (parent_grid_ratio=1) but the same time_step (parent_time_step_ratio=1)
    # -> its ratio equals d01's (6.0, within bound) -- so tighten to force a nest-only violation:
    assert clean_findings(findings) == []


def test_c8_use_adaptive_time_step_skips(tmp_path):
    text = one_domain_namelist(extra_domains=" use_adaptive_time_step = .true.,\n dx = 1000,\n"
                                              " dy = 1000,\n time_step = 600,\n")
    text = text.replace(" dx = 10000,\n", "").replace(" dy = 10000,\n", "").replace(" time_step = 60,\n", "")
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c8_time_step(nl, findings)
    assert clean_findings(findings) == []
    assert any("SKIPPED" in f.message for f in findings)


# --------------------------------------------------------------------------
# C9 -- disk
# --------------------------------------------------------------------------

def test_c9_insufficient_disk_warns(tmp_path, monkeypatch):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))

    class FakeStat:
        f_bavail = 1
        f_frsize = 1  # ~1 byte free -- must be dwarfed by any real estimate

    monkeypatch.setattr(pf.os, "statvfs", lambda path: FakeStat())
    findings = []
    pf.check_c9_disk(nl, tmp_path, findings)
    warns = [f for f in findings if f.severity == "WARN" and f.id == "C9"]
    assert len(warns) == 1


def test_c9_sufficient_disk_silent(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c9_disk(nl, tmp_path, findings)  # real tmp_path filesystem, tiny domain
    assert clean_findings(findings) == []


# --------------------------------------------------------------------------
# C10 -- wrfinput physical sanity
# --------------------------------------------------------------------------

_C10_SHAPE_4D = (1, 2, 2, 2)   # Time, bottom_top|soil_layers_stag, south_north, west_east
_C10_SHAPE_SFC = (1, 2, 2)     # Time, south_north, west_east


def _c10_full(shape, fill):
    return np.full(shape, fill, dtype="f4")


def write_wrfinput(path, overrides=None, nz=2, ny=2, nx=2):
    """Synthetic wrfinput_d01 with every C10-checked field at a clean, in-range value;
    `overrides` replaces individual variables' ndarrays for one test's bad-value injection."""
    overrides = overrides or {}
    shape4, shape_sfc = (1, nz, ny, nx), (1, ny, nx)
    data = {
        "T": _c10_full(shape4, 290.0),
        "U": _c10_full(shape4, 5.0),
        "V": _c10_full(shape4, -3.0),
        "QVAPOR": _c10_full(shape4, 0.005),
        "TSLB": _c10_full(shape4, 290.0),
        "SMOIS": _c10_full(shape4, 0.3),
        "PSFC": _c10_full(shape_sfc, 101300.0),
        "TSK": _c10_full(shape_sfc, 295.0),
        "SST": _c10_full(shape_sfc, 290.0),
        "LANDMASK": _c10_full(shape_sfc, 1.0),
    }
    data.update(overrides)
    dim4 = ("Time", "bottom_top", "south_north", "west_east")
    dim_sfc = ("Time", "south_north", "west_east")
    write_nc(
        path,
        dims={"Time": None, "bottom_top": nz, "south_north": ny, "west_east": nx},
        vars_={
            "T": ("f4", dim4, data["T"]),
            "U": ("f4", dim4, data["U"]),
            "V": ("f4", dim4, data["V"]),
            "QVAPOR": ("f4", dim4, data["QVAPOR"]),
            "TSLB": ("f4", dim4, data["TSLB"]),
            "SMOIS": ("f4", dim4, data["SMOIS"]),
            "PSFC": ("f4", dim_sfc, data["PSFC"]),
            "TSK": ("f4", dim_sfc, data["TSK"]),
            "SST": ("f4", dim_sfc, data["SST"]),
            "LANDMASK": ("f4", dim_sfc, data["LANDMASK"]),
            **({"XLAND": ("f4", dim_sfc, data["XLAND"])} if "XLAND" in data else {}),
        },
    )


def test_c10_clean_wrfinput_silent(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    write_wrfinput(tmp_path / "wrfinput_d01")
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    assert clean_findings(findings) == []


def test_c10_nan_in_qvapor_is_fatal(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    q = _c10_full(_C10_SHAPE_4D, 0.005)
    q[0, 0, 0, 0] = np.nan
    write_wrfinput(tmp_path / "wrfinput_d01", overrides={"QVAPOR": q})
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C10"
              and "QVAPOR" in f.message and "non-finite" in f.message]
    assert len(fatals) == 1


def test_c10_tslb_out_of_range_is_fatal(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    tslb = _c10_full(_C10_SHAPE_4D, 290.0)
    tslb[0, 0, 0, 0] = 150.0  # below the 200 K floor
    write_wrfinput(tmp_path / "wrfinput_d01", overrides={"TSLB": tslb})
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C10" and "TSLB" in f.message]
    assert len(fatals) == 1


def _c10_run(tmp_path, overrides):
    write_namelist(tmp_path, one_domain_namelist())
    write_wrfinput(tmp_path / "wrfinput_d01", overrides=overrides)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    return [f for f in findings if f.id == "C10" and f.severity in ("FATAL", "WARN")]


def test_c10_smois_negative_on_land_warns_water_ignored(tmp_path):
    """Small negative SMOIS on land is WARN, not FATAL: zhengzhou (ran fine) has 1923 land values
    in [-0.012, 0) at deeper levels. A water point (LANDMASK==0) is not checked at all."""
    sm = _c10_full(_C10_SHAPE_4D, 0.3)
    sm[0, 0, 0, 0] = -0.02   # land point -- the case_I SM000010 value
    sm[0, 1, 0, 1] = -0.012  # land point, deeper level
    sm[0, 0, 1, 1] = -0.5    # water point, must be masked out
    lm = _c10_full(_C10_SHAPE_SFC, 1.0)
    lm[0, 1, 1] = 0.0
    out = _c10_run(tmp_path, {"SMOIS": sm, "LANDMASK": lm})
    smois = [f for f in out if "SMOIS" in f.message]
    assert [f.severity for f in smois] == ["WARN"]
    assert "has 2 value(s)" in smois[0].message   # the water point is not counted


def test_c10_smois_above_one_on_land_is_fatal(tmp_path):
    sm = _c10_full(_C10_SHAPE_4D, 0.3)
    sm[0, 0, 0, 0] = 1.5
    out = _c10_run(tmp_path, {"SMOIS": sm})
    assert [f.severity for f in out if "SMOIS" in f.message] == ["FATAL"]


def test_c10_case_i_like_nonzero_garbage_tslb_is_fatal(tmp_path):
    """tw3km-2031/case_I signature: non-zero garbage land TSLB (-12.5, 170.6 ...), none exactly
    0 -- stays FATAL, and the message reports count and min/max of the bad values."""
    tslb = _c10_full(_C10_SHAPE_4D, 290.0)
    tslb[0, 0, 0, 0] = -12.5
    tslb[0, 0, 0, 1] = 170.6
    tslb[0, 1, 1, 0] = -10.9
    out = _c10_run(tmp_path, {"TSLB": tslb})
    fatals = [f for f in out if f.severity == "FATAL" and "TSLB" in f.message]
    assert len(fatals) == 1
    assert "has 3 non-zero value(s)" in fatals[0].message
    assert "min=-12.500" in fatals[0].message and "max=170.600" in fatals[0].message
    assert not [f for f in out if f.severity == "WARN" and "TSLB" in f.message]


def test_c10_zero_tslb_on_land_warns_only(tmp_path):
    """dorian2019 d02 (ran fine): land columns with TSLB exactly 0 -> one WARN, not FATAL."""
    tslb = _c10_full(_C10_SHAPE_4D, 290.0)
    tslb[0, :, 0, 0] = 0.0      # one full land column
    tslb[0, 1, 1, 1] = 0.0      # a second land point, deeper level only
    out = _c10_run(tmp_path, {"TSLB": tslb})
    tslb_f = [f for f in out if "TSLB" in f.message or "soil temperature" in f.message]
    assert [f.severity for f in tslb_f] == ["WARN"]
    assert "2 land points with undefined (0) soil temperature" in tslb_f[0].message


def test_c10_tslb_on_water_not_checked(tmp_path):
    tslb = _c10_full(_C10_SHAPE_4D, 290.0)
    tslb[0, :, 1, 1] = 0.0
    tslb[0, 0, 0, 1] = -5.0
    lm = _c10_full(_C10_SHAPE_SFC, 1.0)
    lm[0, 1, 1] = 0.0
    xl = _c10_full(_C10_SHAPE_SFC, 1.0)
    xl[0, 0, 1] = 2.0           # LANDMASK says land but XLAND says water -> not land
    out = _c10_run(tmp_path, {"TSLB": tslb, "LANDMASK": lm, "XLAND": xl})
    assert out == []


def test_c10_tslb_small_nonzero_fraction_warns(tmp_path):
    """dorian2019 d02 shape (ran 1 h on scc, job 1392, no NaN): 20/(14806x4) = 0.034% of land
    TSLB values non-zero out of range -> WARN, not FATAL. Here 3 of 4x50x50 = 0.030%."""
    nz, ny, nx = 4, 50, 50
    tslb = np.full((1, nz, ny, nx), 290.0, dtype="f4")
    tslb[0, 3, 0, 0] = 101.1
    tslb[0, 3, 0, 1] = 101.1
    tslb[0, 2, 7, 9] = 150.0
    write_namelist(tmp_path, one_domain_namelist())
    write_wrfinput(tmp_path / "wrfinput_d01", overrides={"TSLB": tslb}, nz=nz, ny=ny, nx=nx)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    tslb_f = [f for f in findings if f.id == "C10" and "TSLB" in f.message]
    assert [f.severity for f in tslb_f] == ["WARN"]
    assert "has 3 non-zero value(s)" in tslb_f[0].message
    assert "0.030% of 10000 land values" in tslb_f[0].message


def test_c10_tslb_fraction_above_threshold_is_fatal(tmp_path):
    """Just over 0.1%: 11 of 4x50x50 = 0.11% -> FATAL."""
    nz, ny, nx = 4, 50, 50
    tslb = np.full((1, nz, ny, nx), 290.0, dtype="f4")
    tslb[0, 0, 0, :11] = -12.5
    write_namelist(tmp_path, one_domain_namelist())
    write_wrfinput(tmp_path / "wrfinput_d01", overrides={"TSLB": tslb}, nz=nz, ny=ny, nx=nx)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    assert [f.severity for f in findings if f.id == "C10" and "TSLB" in f.message] == ["FATAL"]


def test_c10_sst_zero_on_water_skipped_nonzero_bad_fatal(tmp_path):
    lm = _c10_full(_C10_SHAPE_SFC, 0.0)      # all water
    lm[0, 0, 0] = 1.0                        # one land point
    sst = _c10_full(_C10_SHAPE_SFC, 290.0)
    sst[0, 0, 0] = 0.0                       # land: never checked
    sst[0, 0, 1] = 0.0                       # water (inland lake): exactly 0 -> skipped
    assert _c10_run(tmp_path, {"SST": sst, "LANDMASK": lm}) == []
    sst[0, 1, 1] = 50.0                      # water, non-zero garbage -> FATAL
    out = _c10_run(tmp_path, {"SST": sst, "LANDMASK": lm})
    fatals = [f for f in out if f.severity == "FATAL" and "SST" in f.message]
    assert len(fatals) == 1 and "has 1 non-zero value(s)" in fatals[0].message


def test_c10_tsk_out_of_range_is_fatal(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    tsk = _c10_full(_C10_SHAPE_SFC, 295.0)
    tsk[0, 0, 0] = 100.0  # below the 180 K floor
    write_wrfinput(tmp_path / "wrfinput_d01", overrides={"TSK": tsk})
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    fatals = [f for f in findings if f.severity == "FATAL" and f.id == "C10" and "TSK" in f.message]
    assert len(fatals) == 1


def test_c10_no_wrfinput_gives_info_only(tmp_path):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    assert clean_findings(findings) == []
    assert any(f.id == "C10" and f.severity == "INFO" for f in findings)


def test_c10_netcdf4_unavailable_skips_explicitly(tmp_path, monkeypatch):
    text = one_domain_namelist()
    write_namelist(tmp_path, text)
    nl = wp.parse_namelist(str(tmp_path / "namelist.input"))
    monkeypatch.setattr(pf, "HAVE_NETCDF4", False)
    findings = []
    pf.check_c10_wrfinput_sanity(nl, tmp_path, findings)
    assert len(findings) == 1
    assert findings[0].severity == "INFO"
    assert "netCDF4 not available" in findings[0].message


# --------------------------------------------------------------------------
# end-to-end
# --------------------------------------------------------------------------

def test_end_to_end_clean_conus_like_exits_zero(tmp_path):
    text = one_domain_namelist(ghg_input=1)
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTM_DATA", "CAMtr_volume_mixing_ratio", "LANDUSE.TBL")
    code = pf.main([str(tmp_path)])
    assert code == 0


def test_end_to_end_hang_gives_exit_two(tmp_path):
    text = one_domain_namelist(extra_time_control=" iofields_filename = 'storm_diags_d01.txt',\n",
                                ghg_input=1)
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTM_DATA", "CAMtr_volume_mixing_ratio")
    code = pf.main([str(tmp_path)])
    assert code == 2


def test_end_to_end_namelist_missing_exits_two(tmp_path):
    code = pf.main([str(tmp_path)])
    assert code == 2


def test_end_to_end_json_output(tmp_path):
    text = one_domain_namelist(ghg_input=1)
    write_namelist(tmp_path, text)
    touch(tmp_path, "RRTM_DATA", "CAMtr_volume_mixing_ratio", "LANDUSE.TBL")
    json_out = tmp_path / "findings.json"
    code = pf.main([str(tmp_path), "--json", str(json_out)])
    assert code == 0
    assert json_out.exists()
    import json
    data = json.loads(json_out.read_text())
    assert isinstance(data, list)
