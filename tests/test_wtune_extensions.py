"""WRF extensions (WRF-Chem, WRF-Hydro, WRFDA, moving nest): detection, refusal, compliance,
STRICT-only tuning, cost factors. No job is submitted; every directory is a tmp dir.

Fixtures are transcribed in the shape of the documented examples:
  * WRF-Chem: WRF-Chem User's Guide, namelist.input &chem section (chem_opt=301 GOCART simple,
    chemdt/photdt in minutes, io_style_emissions/emiss_inpt_opt, kemit).
  * moving nest: WRF User's Guide (Chapter 5, "Vortex-following/moving nests"): &domains
    vortex_interval, max_vortex_speed, corral_dist, track_level, time_to_move, num_moves.
  * WRF-Hydro: WRF-Hydro technical description / hydro.namelist (&HYDRO_nlist: AGGFACTRT,
    DTRT_CH, DTRT_TER, SUBRTSWCRT, ...).
  * WRFDA: WRFDA tutorial namelist.input (&wrfvar1..&wrfvar7: var4d, ntmax, cv_options, use_*obs).
"""
import os
import sys
import types

import pytest

import wtune.cli
from wtune import compliance as wc
from wtune import extensions as we
from wtune import predict as wp
from wtune import run as wr
import wtune.tune

BASE = """ &time_control
 run_days                            = 0,
 run_hours                           = 12,
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = 2020, 2020,
 start_month                         = 06,   06,
 start_day                           = 01,   01,
 start_hour                          = 00,   00,
 end_year                            = 2020, 2020,
 end_month                           = 06,   06,
 end_day                             = 01,   01,
 end_hour                            = 12,   12,
 interval_seconds                    = 21600
 history_interval                    = 60,   60,
 io_form_history                     = 2
 io_form_restart                     = 2
 /

 &domains
 time_step                           = 30,
 max_dom                             = 2,
 e_we                                = 200,  301,
 e_sn                                = 200,  301,
 e_vert                              = 45,   45,
 dx                                  = 3000, 1000,
 dy                                  = 3000, 1000,
 nproc_x                             = -1,
 nproc_y                             = -1,
 numtiles                            = 1,
%(domains_extra)s /

 &physics
 mp_physics                          = 8,    8,
 ra_lw_physics                       = 4,    4,
 ra_sw_physics                       = 4,    4,
 radt                                = 10,   10,
 bl_pbl_physics                      = 1,    1,
 cu_physics                          = 0,    0,
 /

%(groups)s
 &bdy_control
 spec_bdy_width                      = 5,
 /
"""

CHEM_GROUP = """ &chem
 kemit                               = 19,
 chem_opt                            = 301,
 chemdt                              = 3,
 photdt                              = 30,
 io_style_emissions                  = 1,
 emiss_inpt_opt                      = 1,
 chem_in_opt                         = 0,
 gas_drydep_opt                      = 0,
 aer_drydep_opt                      = 0,
 bio_emiss_opt                       = 0,
 dust_opt                            = 1,
 seas_opt                            = 1,
 /
"""
VORTEX_DOMAINS = """ vortex_interval                     = 15,   15,
 max_vortex_speed                    = 40,   40,
 corral_dist                         = 8,    8,
 track_level                         = 50000,
 time_to_move                        = 0,    0,
"""
HYDRO_NAMELIST = """&HYDRO_nlist
sys_cpl = 2
GEO_STATIC_FLNM = "./DOMAIN/geo_em.d02.nc"
DXRT = 250.0
AGGFACTRT = 4
DTRT_CH = 10
DTRT_TER = 10
SUBRTSWCRT = 1
OVRTSWCRT = 1
CHANRTSWCRT = 1
/
"""
WRFVAR = """ &wrfvar1
 write_increments                    = false,
 var4d                               = false,
 /
 &wrfvar3
 ob_format                           = 2,
 /
 &wrfvar5
 check_max_iv                        = true,
 /
 &wrfvar6
 max_ext_its                         = 1,
 ntmax                               = 100,
 eps                                 = 0.01,
 /
 &wrfvar7
 cv_options                          = 5,
 as1                                 = 0.25, 1.0, 1.5,
 /
 &wrfvar4
 use_synopobs                        = true,
 use_soundobs                        = true,
 /
"""


def make(tmp_path, groups="", domains_extra="", name="case"):
    d = tmp_path / name
    d.mkdir()
    (d / "namelist.input").write_text(BASE % dict(groups=groups, domains_extra=domains_extra))
    return d


def parsed(d):
    return wp.parse_namelist(str(d / "namelist.input"))


def fake_exe(path, symbols=(), control=True):
    data = b"\x7fELF" + b"\0" * 64
    if control:
        data += b"solve_em_\0start_domain_\0"
    for s in symbols:
        data += s + b"_\0"
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------- detection
def test_plain_arw_detects_nothing(tmp_path):
    d = make(tmp_path)
    assert we.detect(parsed(d), str(d)) == []
    assert we.check_report(str(d / "namelist.input"), str(d)) == ([], 0)


def test_chem_detected_and_active(tmp_path):
    d = make(tmp_path, CHEM_GROUP)
    (det,) = we.detect(parsed(d), str(d))
    assert det["id"] == "chem" and det["active"] and det["detail"]["chem_opt"] == 301
    assert "&chem group present" in det["evidence"]


def test_chem_group_with_chem_opt_zero_is_inactive(tmp_path):
    d = make(tmp_path, " &chem\n chem_opt = 0,\n /\n")
    (det,) = we.detect(parsed(d), str(d))
    assert det["id"] == "chem" and not det["active"]


def test_moving_nest_detected_by_vortex_keys_and_by_num_moves(tmp_path):
    d = make(tmp_path, domains_extra=VORTEX_DOMAINS)
    (det,) = we.detect(parsed(d), str(d))
    assert det["id"] == "movenest" and det["active"]
    assert "vortex_interval" in det["detail"]["vortex_keys"]
    d2 = make(tmp_path, domains_extra=" num_moves = 3,\n move_id = 1, 1, 1,\n", name="c2")
    (det2,) = we.detect(parsed(d2), str(d2))
    assert det2["id"] == "movenest" and det2["detail"]["num_moves"] == 3
    d3 = make(tmp_path, domains_extra=" num_moves = -1,\n", name="c3")   # default sentinel: no moving nest
    assert we.detect(parsed(d3), str(d3)) == []


def test_hydro_detected_by_hydro_namelist_file(tmp_path):
    d = make(tmp_path)
    (d / "hydro.namelist").write_text(HYDRO_NAMELIST)
    (det,) = we.detect(parsed(d), str(d))
    assert det["id"] == "hydro" and det["active"]


def test_wrfda_detected_by_groups_and_by_exe(tmp_path):
    d = make(tmp_path, WRFVAR)
    (det,) = we.detect(parsed(d), str(d))
    assert det["id"] == "wrfda" and "wrfvar1" in det["detail"]["wrfvar_groups"]
    d2 = make(tmp_path, name="c2")
    (d2 / "da_wrfvar.exe").write_text("x")
    (det2,) = we.detect(parsed(d2), str(d2))
    assert det2["id"] == "wrfda" and det2["detail"]["da_wrfvar_exe"]


@pytest.mark.parametrize("fname,uid", [("wrfndi_d02", "ndown"), ("real_nmm.exe", "nmm"),
                                         ("atmosphere_model", "mpas"),
                                         ("wrf_hydro_NoahMP.exe", "hydro-standalone")])
def test_unsupported_families(tmp_path, fname, uid):
    d = make(tmp_path)
    (d / fname).write_text("x")
    assert [u[0] for u in we.detect_unsupported(str(d))] == [uid]


# ---------------------------------------------------------------- check output / build mismatch
def test_check_prints_exact_support_line_for_chem(tmp_path):
    d = make(tmp_path, CHEM_GROUP)
    lines, code = we.check_report(str(d / "namelist.input"), str(d))
    assert code == 0
    assert lines[0] == (
        "INFO EXT-CHEM WRF-Chem detected (&chem group present; chem_opt=301): OpenMP not supported "
        "by WRF-Chem; tuning ranks/decomposition/tiles/I-O only; chem_opt/chemdt/photdt/emissions are left "
        "exactly as given; time prediction for this extension is UNCALIBRATED until calibration "
        "runs exist. Decomposition constraint (PROVISIONAL, one case): splits with a j-patch < 21 "
        "rows are never proposed (WRF-Chem hung at narrow j-patches).")


def test_check_wrfda_marks_advisory_only(tmp_path):
    d = make(tmp_path, WRFVAR)
    lines, _ = we.check_report(str(d / "namelist.input"), str(d))
    text = "\n".join(lines)
    assert "does not run or orchestrate DA cycles" in text and "ADVISORY ONLY" in text


def test_build_mismatch_is_fatal_and_present_build_is_clean(tmp_path):
    d = make(tmp_path, CHEM_GROUP)
    fake_exe(d / "wrf.exe")
    lines, code = we.check_report(str(d / "namelist.input"), str(d))
    assert code == 2 and any(l.startswith("FATAL EXT-BUILD") and "WRF-Chem" in l for l in lines)
    fake_exe(d / "wrf.exe", symbols=[b"chem_driver"])
    lines, code = we.check_report(str(d / "namelist.input"), str(d))
    assert code == 0 and not any("FATAL" in l for l in lines)


def test_stripped_binary_is_unknown_not_a_mismatch(tmp_path):
    d = make(tmp_path, CHEM_GROUP)
    fake_exe(d / "wrf.exe", control=False)
    lines, code = we.check_report(str(d / "namelist.input"), str(d))
    assert code == 0 and any("could not determine" in l for l in lines)


def test_inactive_chem_group_never_a_build_error(tmp_path):
    d = make(tmp_path, " &chem\n chem_opt = 0,\n /\n")
    fake_exe(d / "wrf.exe")
    assert we.check_report(str(d / "namelist.input"), str(d))[1] == 0


def test_unsupported_family_flagged_in_check(tmp_path):
    d = make(tmp_path)
    (d / "wrfndi_d02").write_text("x")
    lines, code = we.check_report(str(d / "namelist.input"), str(d))
    assert code == 1 and "UNSUPPORTED ndown" in lines[0]


# ---------------------------------------------------------------- refusals
def _cli(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["wtune"] + list(argv))
    with pytest.raises(SystemExit) as e:
        wtune.cli.main()
    return e.value.code, capsys.readouterr()


@pytest.mark.parametrize("groups,dom", [(CHEM_GROUP, ""), ("", VORTEX_DOMAINS), (WRFVAR, "")])
def test_cli_predict_rebuild_refused(tmp_path, monkeypatch, capsys, groups, dom):
    d = make(tmp_path, groups, dom)
    code, out = _cli(monkeypatch, capsys, "predict-rebuild", str(d), "--nodes", "1",
                     "--nx", "4", "--ny", "4")
    assert code == 2 and "predict-rebuild refused" in out.err


def test_cli_predict_rebuild_refused_for_hydro(tmp_path, monkeypatch, capsys):
    d = make(tmp_path)
    (d / "hydro.namelist").write_text(HYDRO_NAMELIST)
    code, out = _cli(monkeypatch, capsys, "predict-rebuild", str(d), "--nodes", "1",
                     "--nx", "4", "--ny", "4")
    assert code == 2 and "WRF-Hydro" in out.err


def run_args(d, *extra):
    return [str(d), "--nodes", "1", "--nx", "4", "--ny", "4", "--out", str(d) + "_out", *extra]


@pytest.mark.parametrize("groups,dom,label", [(CHEM_GROUP, "", "WRF-Chem"),
                                              ("", VORTEX_DOMAINS, "Moving nest")])
def test_run_cfl_rescue_refused_on_dt_coupled_extensions(tmp_path, capsys, groups, dom, label):
    d = make(tmp_path, groups, dom)
    assert wr.main(run_args(d, "--cfl-rescue", "--dry-run")) == 2
    err = capsys.readouterr().err
    assert "--cfl-rescue would change time_step" in err and label in err


def test_run_refuses_wrfda_cycling_unless_forecast_only_acknowledged(tmp_path, capsys):
    d = make(tmp_path, WRFVAR)
    assert wr.main(run_args(d, "--dry-run")) == 2
    err = capsys.readouterr().err
    assert "does not run or orchestrate DA cycles" in err and "--wrfda-forecast-only" in err
    assert we.refusal_run(str(d), wrfda_forecast_only=True) is None


def test_run_refuses_ndown(tmp_path, capsys):
    d = make(tmp_path)
    (d / "wrfndi_d02").write_text("x")
    assert wr.main(run_args(d, "--dry-run")) == 2
    assert "UNSUPPORTED ndown" in capsys.readouterr().err


def test_run_cfl_rescue_still_allowed_for_plain_case(tmp_path):
    d = make(tmp_path)
    assert we.refusal_run(str(d), cfl_rescue=True) is None


# ---------------------------------------------------------------- compliance
EXT_NAMES = ["chem_opt", "chemdt", "photdt", "emiss_inpt_opt", "io_style_emissions", "kemit",
             "aggfactrt", "dtrt_ch", "dtrt_ter", "var4d", "ntmax", "cv_options", "use_synopobs",
             "vortex_interval", "max_vortex_speed", "corral_dist", "track_level", "time_to_move",
             "num_moves"]


@pytest.mark.parametrize("name", EXT_NAMES)
def test_extension_parameters_are_physics_gray(name):
    assert wc.classify(name) == wc.PHYSICS_GRAY


def test_catalog_still_validates_and_strict_knobs_stay_strict():
    assert wc.validate() == []
    for k in ("nproc_x", "nproc_y", "numtiles"):
        assert wc.classify(k) == wc.STRICT


@pytest.mark.parametrize("name", ["chem_opt", "chemdt", "aggfactrt", "vortex_interval", "ntmax"])
def test_assert_strict_only_rejects_extension_change(name):
    with pytest.raises(wc.ComplianceError):
        wc.assert_strict_only({name: [1]}, {name: [2]})


# ---------------------------------------------------------------- STRICT-only tuning output
STRICT_CHANGES = {"nproc_x", "nproc_y", "numtiles"}


def flat(nl):
    return {k: v for g in nl.values() for k, v in g.items()}


@pytest.mark.parametrize("groups,dom", [(CHEM_GROUP, ""), ("", VORTEX_DOMAINS), (WRFVAR, "")])
def test_param_table_only_touches_strict_keys(tmp_path, groups, dom):
    d = make(tmp_path, groups, dom)
    cfg = dict(ranks=16, omp=1, nx=4, ny=4, numtiles=1, dist="block", id="c")
    table = wp.build_param_table(cfg, 1, 2, "nfs")
    assert {ln.split("=")[0].strip() for ln in table["namelist_lines"]} <= STRICT_CHANGES
    ctx = types.SimpleNamespace(namelist_path=str(d / "namelist.input"))
    wtune.tune._assert_report_is_strict_only(ctx, table)      # no raise


def test_tune_guard_rejects_changed_extension_key(tmp_path):
    d = make(tmp_path, CHEM_GROUP)
    ctx = types.SimpleNamespace(namelist_path=str(d / "namelist.input"))
    with pytest.raises(wc.ComplianceError):
        wtune.tune._assert_report_is_strict_only(
            ctx, {"namelist_lines": ["nproc_x=4,", "chem_opt=1,"]})


def test_extension_group_guard_flags_any_change_inside_owned_group():
    a = {"chem": {"some_new_chem_key": [1]}, "domains": {"nproc_x": [4]}}
    b = {"chem": {"some_new_chem_key": [2]}, "domains": {"nproc_x": [8]}}
    assert we.extension_keys_changed(a, b) == [("chem", "some_new_chem_key")]
    assert we.extension_keys_changed({"domains": {"nproc_x": [4]}}, {"domains": {"nproc_x": [8]}}) == []


@pytest.mark.parametrize("groups,dom", [(CHEM_GROUP, ""), ("", VORTEX_DOMAINS), (WRFVAR, "")])
def test_segment_namelist_preserves_extension_parameters(tmp_path, groups, dom):
    import datetime as dt
    d = make(tmp_path, groups, dom)
    text = (d / "namelist.input").read_text()
    out = wr.render_segment_namelist(text, dt.datetime(2020, 6, 1, 0), dt.datetime(2020, 6, 1, 6), 0,
                                     wr.make_config(4, 4, 1, 1, 1), 360)
    o = tmp_path / "seg.input"
    o.write_text(out)
    changed = we.extension_keys_changed(wp.parse_namelist(str(d / "namelist.input")),
                                        wp.parse_namelist(str(o)))
    assert changed == []
    # byte-for-byte: every extension line survives verbatim
    for block in (groups, dom):
        for line in block.splitlines():
            if "=" in line:
                assert line in out


def test_recommend_carries_extension_note(tmp_path):
    d = make(tmp_path, CHEM_GROUP)
    notes, refusal = we.tune_gate(str(d / "namelist.input"))
    assert refusal is None and notes and notes[0].startswith("WRF-Chem detected")


# ---------------------------------------------------------------- cost factors
def base_compute(source):
    return {"source": source, "s_normal": 1.0, "rad_extra_s": 10.0, "compute_total_s": 110.0,
            "noise_pct": 50.0, "noise_abs_s": 55.0}


def test_moving_nest_factor_is_measured_6_percent_on_fallback(tmp_path):
    d = make(tmp_path, domains_extra=VORTEX_DOMAINS)
    c = we.apply_cost_factors(base_compute("no_probe_fallback"), parsed(d), 30, str(d))
    (f,) = c["extension_factors"]
    assert f["status"] == "measured" and f["applied"]
    assert f["multiplier"] == pytest.approx(1.06)
    assert c["compute_total_s"] == pytest.approx(110.0 * 1.06)
    assert c["noise_pct"] > 50.0


def test_factor_not_applied_when_probe_measured_on_same_case(tmp_path):
    d = make(tmp_path, domains_extra=VORTEX_DOMAINS)
    c = we.apply_cost_factors(base_compute("measured"), parsed(d), 30, str(d))
    assert c["compute_total_s"] == 110.0 and not c["extension_factors"][0]["applied"]
    assert "already included" in c["extension_factors"][0]["note"]


def test_chem_factor_is_placeholder_with_wide_uncertainty(tmp_path):
    d = make(tmp_path, CHEM_GROUP)
    c = we.apply_cost_factors(base_compute("no_probe_fallback"), parsed(d), 30, str(d))
    (f,) = c["extension_factors"]
    assert f["status"] == "placeholder" and f["uncertainty_pct"] >= 50
    assert f["multiplier"] > 1.0 and c["extension_uncertainty_pct"] == f["uncertainty_pct"]
    text = "\n".join(we.render_factor_lines(c))
    assert "PLACEHOLDER" in text and "UNCALIBRATED" in text


def test_hydro_factor_reads_aggfactrt_from_hydro_namelist(tmp_path):
    d = make(tmp_path)
    (d / "hydro.namelist").write_text(HYDRO_NAMELIST)
    (f,) = we.cost_factors(parsed(d), 30, str(d))
    assert f["id"] == "hydro" and f["status"] == "placeholder" and f["aggfactrt"] == 4


def test_wrfda_has_no_cost_model(tmp_path):
    d = make(tmp_path, WRFVAR)
    c = we.apply_cost_factors(base_compute("no_probe_fallback"), parsed(d), 30, str(d))
    (f,) = c["extension_factors"]
    assert f["status"] == "none" and c["compute_total_s"] == 110.0


def test_coefficients_overridable_without_code_change(tmp_path, monkeypatch):
    d = make(tmp_path, CHEM_GROUP)
    p = tmp_path / "coef.json"
    p.write_text('{"chem": {"status": "measured", "frac_per_tracer": 0.0, "frac_per_chem_step": 0.0,'
                 ' "uncertainty_pct": 7.0}}')
    monkeypatch.setenv("WTUNE_EXTENSION_COEFFS", str(p))
    (f,) = we.cost_factors(parsed(d), 30, str(d))
    assert f["status"] == "measured" and f["multiplier"] == 1.0 and f["uncertainty_pct"] == 7.0
    monkeypatch.delenv("WTUNE_EXTENSION_COEFFS")
    (g,) = we.cost_factors(parsed(d), 30, str(d), calibration={"extension_coefficients":
                                                                 {"chem": {"default_n_tracers": 0,
                                                                           "frac_per_chem_step": 0.0}}})
    assert g["multiplier"] == 1.0


def test_plain_arw_compute_untouched(tmp_path):
    d = make(tmp_path)
    c = base_compute("no_probe_fallback")
    assert we.apply_cost_factors(dict(c), parsed(d), 30, str(d)) == c


# ---------------------------------------------------------------- coldstart blocker
def test_coldstart_chem_blocks_only_on_missing_input_files(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
    import test_coldstart_sample as t
    from wtune.tools import coldstart_sample as cs
    case = tmp_path / "cs"
    case.mkdir()
    t.write_namelist(str(case / "namelist.input"), extra=CHEM_GROUP)
    t.write_namelist_wps(str(case / "namelist.wps"))
    geog = t._make_geog_dir(str(case), names=t.SCC_GEOG_LISTING)
    _rows, blockers, _nl = cs.inspect_case(str(case), geog_path=geog)
    assert any("chem_opt=301" in b and "wrfchemi_d*" in b and "crash" in b for b in blockers)
    (case / "wrfchemi_d01_2020-06-01_00:00:00").write_text("x")
    rows, blockers, _nl = cs.inspect_case(str(case), geog_path=geog)
    assert not any("chem_opt" in b for b in blockers)
    assert any(r[0] == "WRF-Chem" and "grid/rank" in r[3] for r in rows)


# ---------------------------------------------------------------- direct invocation (no wtune.cli)
def test_preflight_main_direct_prints_extension_lines_and_fails_on_build_mismatch(tmp_path, capsys):
    import wtune.preflight
    d = make(tmp_path, CHEM_GROUP)
    fake_exe(d / "wrf.exe")
    code = wtune.preflight.main([str(d)])
    out = capsys.readouterr().out
    assert code == 2 and "INFO EXT-CHEM WRF-Chem detected" in out and "FATAL EXT-BUILD" in out


def test_preflight_main_direct_plain_case_has_no_extension_output(tmp_path, capsys):
    import wtune.preflight
    d = make(tmp_path)
    wtune.preflight.main([str(d)])
    assert "EXT-" not in capsys.readouterr().out


def test_rebuild_predict_main_direct_refuses(tmp_path, capsys):
    from wtune import rebuild_predict as rb
    d = make(tmp_path, CHEM_GROUP)
    assert rb.main([str(d), "--nodes", "1", "--nx", "4", "--ny", "4", "--dry-run"]) == 2
    assert "predict-rebuild refused" in capsys.readouterr().err


# ---------------------------------------------------------------- advice-only CFL coupling warning
def test_cfl_coupling_warning_chem(tmp_path):
    d = make(tmp_path, CHEM_GROUP)                 # chemdt=3 min=180 s, photdt=30 min=1800 s
    (w,) = we.cfl_coupling_warnings(str(d), 30, 22)
    assert "chemdt=3 min (NOT an integer multiple" in w and "photdt=30 min (NOT" in w
    assert "WTune will NOT change these" in w and "you must check" in w
    (w2,) = we.cfl_coupling_warnings(str(d), 30, 20)
    assert "chemdt=3 min (divisible)" in w2


def test_cfl_coupling_warning_hydro_and_movenest(tmp_path):
    d = make(tmp_path, domains_extra=VORTEX_DOMAINS)
    (d / "hydro.namelist").write_text(HYDRO_NAMELIST)
    ws = we.cfl_coupling_warnings(str(d), 30, 22)
    text = "\n".join(ws)
    assert "DTRT_TER=10 s (NOT an integer multiple" in text
    assert "vortex_interval=15 min (NOT" in text and "move schedule" in text


def test_cfl_coupling_warning_absent_for_plain_arw_and_no_advice(tmp_path):
    d = make(tmp_path)
    assert we.cfl_coupling_warnings(str(d), 30, 22) == []
    d2 = make(tmp_path, CHEM_GROUP, name="c2")
    assert we.cfl_coupling_warnings(str(d2), 30, None) == []


# ---------------------------------------------------------------- no OpenMP on dmpar-only extensions
@pytest.mark.parametrize("groups,hydro,expect", [(CHEM_GROUP, False, 1), (WRFVAR, False, 1),
                                                 ("", True, 1), ("", False, None)])
def test_omp_cap(tmp_path, groups, hydro, expect):
    d = make(tmp_path, groups)
    if hydro:
        (d / "hydro.namelist").write_text(HYDRO_NAMELIST)
    assert we.omp_cap_for_namelist(str(d / "namelist.input")) == expect


def test_moving_nest_keeps_openmp_and_support_line_says_so(tmp_path):
    d = make(tmp_path, domains_extra=VORTEX_DOMAINS)
    assert we.omp_cap_for_namelist(str(d / "namelist.input")) is None
    (det,) = we.detect(parsed(d), str(d))
    assert "ranks/decomposition/tiles/OpenMP/I-O only" in we.support_line(det)


def test_support_line_states_no_openmp_for_hydro_and_wrfda(tmp_path):
    d = make(tmp_path, WRFVAR)
    (d / "hydro.namelist").write_text(HYDRO_NAMELIST)
    for det in we.detect(parsed(d), str(d)):
        assert ("OpenMP not supported by %s; tuning ranks/decomposition/tiles/I-O only" % det["label"]
                in we.support_line(det))


def test_tune_search_filter_drops_omp_gt_1_on_chem(tmp_path):
    d = make(tmp_path, CHEM_GROUP)
    cap = we.omp_cap_for_namelist(str(d / "namelist.input"))
    import wtune.config
    ctx = types.SimpleNamespace(omp_cap=cap, search_cfg={}, exclude_rules=[], is_fire=False)
    cfgs = [{"id": "a", "omp": 1}, {"id": "b", "omp": 2}, {"id": "ANCHOR", "omp": 4}]
    assert [c["id"] for c in wtune.tune._apply_search_filter(ctx, cfgs)] == ["a"]
    ctx.omp_cap = None
    assert len(wtune.tune._apply_search_filter(ctx, cfgs)) == 3


# ---------------------------------------------------------------- recorded real-binary evidence (agent E)
def test_movenest_signature_is_med_nest_move_only():
    assert we.BUILD_SIGNATURES["movenest"] == (b"med_nest_move",)


def test_vortex_center_alarm_string_is_not_a_move_nests_build(tmp_path):
    # production wrf.exe carries "...(COMPUTE_VORTEX_CENTER_ALARM) FAILED" but no med_nest_move
    exe = tmp_path / "wrf.exe"
    exe.write_bytes(b"\x7fELF" + b"\0" * 8 + b"solve_em_\0start_domain_\0"
                    b"WRFU_AlarmCreate(COMPUTE_VORTEX_CENTER_ALARM) FAILED\0vortex_center\0")
    assert we.scan_binary(str(exe))["movenest"] is False
    exe.write_bytes(exe.read_bytes() + b"med_nest_move_\0")
    assert we.scan_binary(str(exe))["movenest"] is True


# ---------------------------------------------------------------- WRF-Chem narrow j-patch guard (PROVISIONAL)
KATRINA_E_SN = [129]          # 169 x 129 d01 (failed-idea note 2026-09-30)


def test_chem_min_j_patch_constant_and_note():
    assert we.CHEM_MIN_J_PATCH == 21
    assert "wrf-chem-gocart-hangs-at-small-j-patch" in we.CHEM_J_PATCH_NOTE


@pytest.mark.parametrize("nx,ny", [(8, 9), (6, 12)])
def test_chem_hanging_splits_rejected_with_reason(nx, ny):
    why = we.chem_j_patch_reason(KATRINA_E_SN, ny, nx)
    assert why and "PROVISIONAL" in why and "wrf-chem-gocart-hangs" in why and "< 21" in why


@pytest.mark.parametrize("nx,ny", [(12, 6), (6, 6), (4, 4), (2, 4)])
def test_chem_passing_splits_accepted(nx, ny):
    assert we.chem_j_patch_reason(KATRINA_E_SN, ny, nx) is None


def test_wrf_auto_mesh_matches_mpaspect():
    assert we.wrf_auto_mesh(72) == (8, 9)      # the automatic split that hung
    assert we.wrf_auto_mesh(36) == (6, 6)
    assert we.wrf_auto_mesh(7) == (1, 7)


def chem_katrina(tmp_path, chem_opt=300):
    d = tmp_path / "kat"
    d.mkdir()
    txt = (BASE % dict(groups=" &chem\n chem_opt = %d,\n /\n" % chem_opt, domains_extra="")
           ).replace("e_we                                = 200,  301,", "e_we = 169,").replace(
               "e_sn                                = 200,  301,", "e_sn = 129,").replace(
               "max_dom                             = 2,", "max_dom = 1,")
    (d / "namelist.input").write_text(txt)
    return d


def test_check_flags_automatic_split_that_hangs_and_passes_wide_j(tmp_path):
    d = chem_katrina(tmp_path)
    lines, code = we.check_report(str(d / "namelist.input"), str(d), ranks=72)
    assert code == 1 and any("EXT-CHEM-JPATCH" in l and "rejected 8x9" in l and "MPASPECT" in l for l in lines)
    lines, code = we.check_report(str(d / "namelist.input"), str(d), ranks=36)
    assert code == 0 and not any("JPATCH" in l for l in lines)
    lines, code = we.check_report(str(d / "namelist.input"), str(d), nproc_x=6, nproc_y=12)
    assert any("rejected 6x12" in l for l in lines)


def test_chem_off_never_constrained(tmp_path):
    d = chem_katrina(tmp_path, chem_opt=0)
    assert we.chem_e_sn_for_namelist(str(d / "namelist.input")) is None
    assert not any("JPATCH" in l for l in we.check_report(str(d / "namelist.input"), str(d), ranks=72)[0])


def test_tune_filter_rejects_hanging_chem_splits(tmp_path, capsys):
    d = chem_katrina(tmp_path)
    ctx = types.SimpleNamespace(omp_cap=1, chem_e_sn=we.chem_e_sn_for_namelist(str(d / "namelist.input")),
                                search_cfg={}, exclude_rules=[], is_fire=False)
    cfgs = [{"id": "a", "nx": 8, "ny": 9, "omp": 1}, {"id": "b", "nx": 6, "ny": 12, "omp": 1},
            {"id": "c", "nx": 12, "ny": 6, "omp": 1}, {"id": "d", "nx": 6, "ny": 6, "omp": 1}]
    assert [c["id"] for c in wtune.tune._apply_search_filter(ctx, cfgs)] == ["c", "d"]
    assert "CHEM_MIN_J_PATCH" in capsys.readouterr().out


def test_post_help_lists_every_post_subcommand():
    # it listed 9 of them while post/cli.py had 16 (fetch-era5 is an alias of `fetch era5`)
    import argparse
    from wtune.post import cli as post_cli
    sub = [a for a in post_cli.build_parser()._actions
           if isinstance(a, argparse._SubParsersAction)][0]
    text = [h for name, _d, h, _s, _m in wtune.cli._FORWARDED_COMMANDS if name == "post"][0]
    listed = [w.strip() for w in text.split(":")[-1].split("|")]
    assert sorted(listed) == sorted(set(sub.choices) - {"fetch-era5"})
    tune = [h for name, _d, h, _s, _m in wtune.cli._FORWARDED_COMMANDS if name == "tune"][0]
    assert "(scc)" not in tune


# ---------------------------------------------------------------- no OpenMP in a dmpar wrf.exe
def _wrf_tree(tmp_path, body):
    (tmp_path / "WRF" / "main").mkdir(parents=True)
    (tmp_path / "WRF" / "main" / "wrf.exe").write_bytes(b"\x7fELF" + body)
    return str(tmp_path / "WRF")


@pytest.mark.parametrize("body,expect", [
    (b"solve_em libnvomp.so start_domain", False),       # NVHPC links libnvomp.so into dmpar too
    (b"solve_em __kmpc_fork_call", True), (b"start_domain GOMP_parallel", True),
    (b"__kmpc_fork_call", None)])                          # no WRF symbol: stripped, cannot tell
def test_build_has_openmp_reads_the_parallel_region_entry(tmp_path, body, expect):
    tree = _wrf_tree(tmp_path, body)
    assert we.build_has_openmp(os.path.join(tree, "main", "wrf.exe")) is expect
    assert we.omp_cap_for_build(tree) == (1 if expect is False else None)


def test_omp_cap_for_build_is_none_without_a_wrf_exe(tmp_path):
    assert we.omp_cap_for_build(str(tmp_path / "none")) is None
    assert we.omp_cap_for_build("") is None and we.omp_cap_for_build(None) is None


@pytest.mark.parametrize("body,cap", [(b"solve_em", 1), (b"solve_em __kmpc_fork_call", None)])
def test_tune_caps_omp_at_1_on_a_dmpar_build(tmp_path, monkeypatch, body, cap):
    from wtune.tune import cli
    monkeypatch.setenv("WTUNE_MAX_OMP", "0")
    tree = _wrf_tree(tmp_path, body)
    monkeypatch.setattr(cli, "_probe_tree", lambda args: tree)
    captured = {}
    monkeypatch.setattr(cli, "Ctx", lambda **kw: captured.update(kw) or types.SimpleNamespace(**kw))
    args = cli.build_arg_parser().parse_args(["c1"])
    o = types.SimpleNamespace(case="c1", minutes=10.0, confirm_reps=1, restart_io_form=14,
                              cfg={"search": {}, "exclude": [], "prediction": {}, "probe": {}})
    ns = types.SimpleNamespace(nodes=1, node_seeds=None, hotspot_fill=None)
    (tmp_path / "namelist.input").write_text("&domains\n/\n")
    cli._build_ctx(args, o, ns, {}, str(tmp_path / "namelist.input"), False, 10, None, 4,
                   str(tmp_path / "tout"), None, None, None, None, None, None)
    assert captured["omp_cap"] == cap and captured["extension_omp_cap"] is None
