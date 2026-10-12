"""Tests for wtune.calc (decision 0062): known-value conversions, dt/radt/nest/cfl arithmetic,
physics-scale formulas, rules-file integrity and `check` behaviour.

Run: envs/wtune/.pixi/envs/default/bin/python -m pytest -q tests/test_wtune_calc.py
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from wtune import calc as wc

REPO = Path(__file__).resolve().parent.parent
LEVELS = {"wrf-doc", "wtune-measured", "rule-of-thumb"}


def val(res, label_part):
    for it in res["items"]:
        if label_part in it["label"]:
            return it["value"]
    raise KeyError(label_part)


def write_nl(tmp_path, domains="", physics="", dynamics="", extra="", time_control="", name="namelist.input"):
    text = """&time_control
 run_hours = 6, interval_seconds = 21600,
 start_year = 2022, 2022, start_month = 7, 7, start_day = 20, 20, start_hour = 0, 0,
 end_year = 2022, 2022, end_month = 7, 7, end_day = 20, 20, end_hour = 6, 6,
%s/
&domains
%s/
&physics
%s/
&dynamics
%s/
%s""" % (time_control, domains, physics, dynamics, extra)
    p = tmp_path / name
    p.write_text(text)
    return str(p)


NEST_DOMAINS = """ time_step = 54, max_dom = 2,
 e_we = 200, 301, e_sn = 200, 301, e_vert = 45, 45,
 dx = 9000, 3000, dy = 9000, 3000,
 parent_id = 0, 1, parent_grid_ratio = 1, 3, parent_time_step_ratio = 1, 3,
 i_parent_start = 1, 50, j_parent_start = 1, 50,
"""


def statuses(rep, rid):
    return [(f["domain"], f["status"]) for f in rep["findings"] if f["rule_id"] == rid]


# ------------------------------------------------------------------------------ rules integrity

def test_every_rule_has_level_and_source():
    rs = wc.rules()
    assert len(rs) >= 40
    for rid, r in rs.items():
        assert r["source_level"] in LEVELS, rid
        assert r["source_ref"].strip(), rid
        assert r["formula"].strip() and r["function"] and r["topic"] in wc.TOPICS, rid


def test_wrf_doc_rules_cite_a_file_and_line():
    for rid, r in wc.rules().items():
        if r["source_level"] == "wrf-doc":
            assert re.search(r"(README\.namelist|Registry|module_check_a_mundo\.F|start_em\.F|solve_em\.F|"
                             r"mediation_nest_move\.F|module_model_constants\.F):\d+", r["source_ref"]), rid


def test_only_wrf_doc_rules_can_be_hard():
    for rid, r in wc.rules().items():
        if r.get("hard"):
            assert r["source_level"] == "wrf-doc", rid


def test_wtune_measured_rules_cite_a_wiki_note_or_calibration():
    for rid, r in wc.rules().items():
        if r["source_level"] == "wtune-measured":
            assert re.search(r"\.md|calibration|wtune.predict", r["source_ref"]), rid


def test_rule_of_thumb_not_disguised_as_official():
    for rid in ("dt.steep_terrain", "cfl.limit", "nudge.moisture", "nudge.wavenumber", "scales.effective_resolution",
                "spinup.mesoscale", "longrun.threshold", "geog.res_vs_dx", "physics.cumulus"):
        assert wc.rule(rid)["source_level"] == "rule-of-thumb"


# ------------------------------------------------------------------------------------- units (C4)

def test_deg_km_and_arcsec_known_values():
    assert wc.deg_to_km(1.0) == pytest.approx(111.177, abs=0.01)            # pi/180*6370
    assert wc.deg_to_km(1.0, lat=60.0, axis="lon") == pytest.approx(55.589, abs=0.01)
    assert wc.km_to_deg(111.177) == pytest.approx(1.0, abs=1e-4)
    assert wc.arcsec_to_m(30.0) == pytest.approx(926.5, abs=0.5)            # 30 arcsec ~ 0.926 km
    assert wc.m_to_arcsec(wc.arcsec_to_m(30.0)) == pytest.approx(30.0)


def test_icao_known_values_and_roundtrip():
    assert wc.icao_height_m(1013.25) == pytest.approx(0.0, abs=1e-6)
    assert wc.icao_height_m(500.0) == pytest.approx(5574.4, abs=1.0)
    assert wc.icao_height_m(226.32) == pytest.approx(11000.0, abs=1.0)
    assert wc.icao_height_m(100.0) == pytest.approx(16181.0, abs=5.0)
    for p in (850.0, 500.0, 300.0, 100.0, 50.0):
        assert wc.icao_pressure_hpa(wc.icao_height_m(p)) == pytest.approx(p, rel=1e-3)


def test_knots_and_time_units():
    assert wc.kt_to_ms(100.0) == pytest.approx(51.4444, abs=1e-3)
    assert wc.ms_to_kt(wc.kt_to_ms(37.0)) == pytest.approx(37.0)
    assert wc.convert_time(1.0, "d", "h") == 24.0
    assert wc.convert_time(90.0, "min", "h") == 1.5


def test_units_function_shape_and_unknown_kind():
    r = wc.units("hpa-m", 500.0)
    assert r["topic"] == "units" and r["items"][0]["unit"] == "m"
    assert r["items"][0]["source_level"] == "rule-of-thumb"
    with pytest.raises(ValueError):
        wc.units("furlong-m", 1.0)


# ------------------------------------------------------------------------------ dt / radt / cfl

@pytest.mark.parametrize("dx,dt_s", [(3, 18), (9, 54), (27, 162)])
def test_dt_6dx_and_radt(dx, dt_s):
    r = wc.dt(dx)
    assert val(r, "recommended") == pytest.approx(dt_s)
    assert val(r, "steep") == [pytest.approx(4.0 * dx), pytest.approx(5.0 * dx)]
    assert val(r, "abort limit") == pytest.approx(dt_s)
    assert val(wc.radt(dx), "radt recommended") == pytest.approx(dx)


def test_dt_map_factor_and_nest_dt_and_adaptive_defaults():
    r = wc.dt(9, max_map_factor=1.5, parent_grid_ratio=3)
    assert val(r, "recommended") == pytest.approx(36.0)                      # 6 * 9 / 1.5
    assert val(r, "nest time_step") == pytest.approx(12.0)
    assert val(wc.dt(9), "adaptive") == [36.0, 72.0, 27.0]


def test_cfl_arithmetic():
    r = wc.cfl(u=30.0, dx_m=3000.0, dt_s=18.0, w=5.0, dz_m=100.0)
    assert val(r, "horizontal Courant") == pytest.approx(0.18)
    assert val(r, "vertical Courant") == pytest.approx(0.9)
    assert val(r, "max dt") == pytest.approx(0.84 * 3000.0 / 30.0)
    hi = wc.cfl(u=100.0, dx_m=3000.0, dt_s=60.0, w=10.0, dz_m=100.0)
    assert val(hi, "horizontal Courant") == pytest.approx(2.0)
    assert "NO" in [i for i in hi["items"] if "Courant" in i["label"]][0]["note"]
    assert "w_damping" in [i for i in hi["items"] if i["label"] == "vertical Courant number"][0]["note"]


# ---------------------------------------------------------------------------------------- nest

def test_nest_dimension_validity():
    assert wc.nest_dim_ok(91, 3) and wc.nest_dim_ok(301, 3) and wc.nest_dim_ok(101, 5) and wc.nest_dim_ok(100, 3)
    assert not wc.nest_dim_ok(92, 3) and not wc.nest_dim_ok(101, 3) and not wc.nest_dim_ok(100, 5)
    assert wc.nearest_nest_dim(92, 3) == 91 and wc.nearest_nest_dim(101, 3) == 100
    assert wc.nest_dim_ok(wc.nearest_nest_dim(98, 5), 5)


def test_nest_dims_from_parent_cells_and_margin():
    r = wc.nest(3, parent_cells_x=100, parent_cells_y=60, parent_dx_km=9.0)
    assert val(r, "e_we / e_sn") == [301, 181]
    assert val(r, "nest dx") == pytest.approx(3.0)
    assert val(r, "parent_time_step_ratio") == 3
    assert wc.nest_footprint(301, 181, 3, 50, 40) == (50, 149, 40, 99)
    assert wc.nest_margin(301, 181, 3, 50, 40, 200, 200) == min(49, 199 - 149, 39, 199 - 99)
    assert wc.nest_margin(301, 181, 3, 150, 40, 200, 200) < 0               # sticks out of the parent
    bad = wc.nest(3, e_we=92, e_sn=101)
    assert val(bad, "legal") is False
    assert "nearest legal: 91 x 100" in [i for i in bad["items"] if "legal" in i["label"]][0]["note"]
    even = wc.nest(2)
    assert "NO" in even["items"][0]["note"]


def test_bdy_and_domain_helpers():
    b = wc.bdy(5)
    assert val(b, "spec_bdy_width") == [5, 1, 4] and val(b, "distance") == 10
    d = wc.domain(25.0, -80.0, 3000.0, 3000.0, 9.0)
    assert val(d, "e_we, e_sn") == [334, 334]                                # 3000/9 + 1
    assert val(d, "map_proj") == "lambert"
    t1, t2 = val(d, "truelat")
    assert t1 < 25.0 < t2
    assert val(wc.domain(5.0, 100.0, 1000.0, 1000.0, 10.0), "map_proj") == "mercator"
    assert val(wc.domain(75.0, 10.0, 1000.0, 1000.0, 10.0), "map_proj") == "polar"


# ----------------------------------------------------------------------------- vert/physics/geog

def test_vert_model_top_and_eta():
    r = wc.vert(45, 5000.0, eta=[1.0, 0.0])
    assert val(r, "model top") == pytest.approx(wc.icao_height_m(50.0), abs=1.0)
    assert val(r, "damp_opt") == [3, 0.2, 5000.0]
    eta = val(r, "eta")
    assert eta[0][2] == pytest.approx(0.0, abs=1e-6) and eta[1][1] == pytest.approx(50.0)


def test_physics_geog():
    assert wc.dx_regime(3.0)[1] == [0] and wc.dx_regime(7.0)[1] == [3, 11] and wc.dx_regime(12.0)[1] == [1, 6]
    g = wc.geog(0.6, 30.0)
    assert val(g, "geog resolution") == pytest.approx(926.5, abs=0.5)
    assert "coarser" in [i for i in g["items"] if i["label"].startswith("geog resolution")][0]["note"]
    assert val(wc.geog(9.0, 30.0), "coarsest standard") == "2m"


# ---------------------------------------------------------------------------------- decomp/io/run

def test_decomp_reuses_wtune_case_legality():
    r = wc.decomp([42, 52], [42, 31])
    assert val(r, "max MPI ranks") == 12                                     # Front Range: 4x3 (measured)
    r2 = wc.decomp([91], [100], ranks=8, cores_per_node=112)
    legal = val(r2, "legal nproc_x")
    assert (4, 2) in legal and (2, 4) in legal
    assert val(wc.decomp([91], [100], ranks=91), "legal nproc_x") == []       # 91 ranks has no legal shape
    assert [8, 14] in val(r2, "ranks_per_node") or [14, 8] in val(r2, "ranks_per_node")


def test_io_frame_size_estimate_and_scaling():
    fb = wc.io_frame_bytes(91, 100, 45)
    assert fb == pytest.approx(38.0e6, rel=0.01)                              # vs 40.3e6 uncompressed matthew wrfinput
    r = wc.io(91, 100, 45, frames=24)
    assert val(r, "total for 24 frame") == pytest.approx(24 * fb / 1e9)
    assert wc.io_frame_bytes(91, 100, 45, n3d=0, n2d=1) == pytest.approx(4 * 91 * 100)
    assert "ESTIMATE" in r["items"][0]["label"]


def test_run_steps_delegates_to_predict(tmp_path):
    nl = write_nl(tmp_path, domains=NEST_DOMAINS)
    r = wc.run(nl)
    assert val(r, "d01 steps") == 6 * 3600 // 54
    assert val(r, "d02 steps") == 6 * 3600 // 54 * 3


# ------------------------------------------------------------------------------ physics scales

def test_nudge_wavenumbers_coefficients():
    r = wc.nudge(334, 334, 9000.0)                                           # L = 2997 km, lambda 1000 km
    assert val(r, "xwavenum, ywavenum") == [3, 3]
    assert val(r, "range") == [[2, 3], [2, 3]]
    assert wc.nudge(334, 334, 9000.0, wavelength_km=1500.0)["items"][0]["value"] == [2, 2]
    assert val(r, "guv = gt = gph") == 3e-4
    assert "55.6 min" in [i for i in r["items"] if i["label"] == "guv = gt = gph"][0]["note"]
    assert val(r, "gq") == 0.0
    assert wc.rule("nudge.coefficients")["source_level"] == "wrf-doc"
    assert wc.rule("nudge.moisture")["source_level"] == "rule-of-thumb"
    assert "never searches" in val(r, "policy")


def test_scales_known_values():
    assert wc.coriolis(25.0) == pytest.approx(6.1637e-5, rel=1e-3)
    assert wc.coriolis(0.0) == 0.0
    assert wc.coriolis(-30.0) < 0
    assert wc.brunt_vaisala(288.0, 4.0) == pytest.approx(0.014056, rel=1e-3)
    assert math.isnan(wc.brunt_vaisala(288.0, 12.0))                          # unstable
    assert wc.froude(10.0, 0.01, 1000.0) == pytest.approx(1.0)
    r = wc.scales(lat=45.0, N=0.01, H=10000.0, dx_km=3.0, U=5.0, h=1000.0, feature_km=50.0)
    assert val(r, "L_R") == pytest.approx(0.01 * 10000.0 / wc.coriolis(45.0) / 1000.0, rel=1e-6)
    assert val(r, "effective resolution") == pytest.approx(21.0)
    assert "blocked" in [i for i in r["items"] if "Fr" in i["label"]][0]["note"]
    assert val(r, "minimum domain") == pytest.approx(max(500.0, val(r, "L_R")))


def test_les_regimes_and_khdif():
    assert wc.les_regime(100.0)[0] == "les" and wc.les_regime(500.0)[0] == "gray" and wc.les_regime(3000.0)[0] == "pbl"
    assert val(wc.les(500.0), "khdif") == pytest.approx(50.0)


def test_acoustic_sound_steps_match_wrf_formula():
    assert wc.sound_steps_auto(18.0, 3000.0) == 4                             # 6*dx -> 4 (solve_em.F comment)
    assert wc.sound_steps_auto(100.0, 10000.0) == 6                           # 10*dx at 10 km -> 6
    assert wc.sound_steps_auto(5.0, 3000.0) == 4                              # floor of 4
    r = wc.acoustic(18.0, 3000.0, c_s=340.0)
    assert val(r, "dts") == pytest.approx(4.5)
    assert val(r, "Courant") == pytest.approx(340.0 * 4.5 / 3000.0)


def test_movenest_search_radius():
    r = wc.movenest(3000.0)
    assert val(r, "search radius") == pytest.approx(36.0)                     # 60*15 min*40 m/s
    assert "12.0 nest cells" in r["items"][0]["note"]
    r2 = wc.movenest(3000.0, storm_speed=10.0, parent_dx_km=27.0)
    assert val(r2, "suggestion") == [15.0, 20.0]
    assert val(r2, "vortex_interval so") == pytest.approx(45.0)


def test_longrun_and_spinup():
    assert val(wc.longrun(10.0), "this run") == 10.0
    assert val(wc.longrun(2.0, 10800), "sst_update") == [1, 180.0]
    s = wc.spinup(36.0)
    assert val(s, "discard") == [6, 12] and val(s, "usable") == 24.0
    assert wc.rule("spinup.cost_measured")["source_level"] == "wtune-measured"


# ---------------------------------------------------------------------------------- check engine

def test_check_clean_namelist_has_no_error_and_no_false_alarm(tmp_path):
    nl = write_nl(tmp_path, domains=NEST_DOMAINS, physics=" radt = 9, 9, cu_physics = 1, 0,\n",
                  dynamics=" damp_opt = 3, zdamp = 5000.,\n")
    rep = wc.check(nl, geog_res_arcsec=30.0)
    assert rep["counts"]["ERROR"] == 0 and rep["counts"]["DEVIATES"] == 0, wc.render_check(rep)
    assert ("d02", "OK") in statuses(rep, "nest.inside_parent")


def test_check_hard_wrf_doc_constraints_are_errors(tmp_path):
    nl = write_nl(tmp_path, domains=NEST_DOMAINS.replace("301, e_sn = 200, 301", "300, e_sn = 200, 301"))
    rep = wc.check(nl)
    assert ("d02", "ERROR") in statuses(rep, "nest.dim_legal")               # 300 != n*3+1
    nl2 = write_nl(tmp_path, domains=NEST_DOMAINS.replace("time_step = 54", "time_step = 72"), name="n2.input")
    rep2 = wc.check(nl2)
    assert ("d01", "ERROR") in statuses(rep2, "dt.reasonable_ratio")         # 8 s/km > 6
    assert ("d01", "DEVIATES") in statuses(rep2, "dt.six_dx")
    nl3 = write_nl(tmp_path, domains=NEST_DOMAINS + " nproc_x = 30, nproc_y = 2,\n", name="n3.input")
    rep3 = wc.check(nl3)
    assert ("all", "ERROR") in statuses(rep3, "decomp.min_patch")            # 200//30 = 6 < 10
    assert all(f["source_level"] == "wrf-doc" for f in rep3["findings"] if f["status"] == "ERROR")


def test_check_rule_of_thumb_deviation_is_never_error(tmp_path):
    nl = write_nl(tmp_path, domains=NEST_DOMAINS, physics=" radt = 30, 30, cu_physics = 1, 1,\n")
    rep = wc.check(nl)
    assert ("d01", "DEVIATES") in statuses(rep, "radt.one_min_per_km")        # 30 min at 9 km
    assert ("d02", "DEVIATES") in statuses(rep, "physics.cumulus")            # cu on at 3 km
    assert rep["counts"]["ERROR"] == 0
    for f in rep["findings"]:
        if f["source_level"] == "rule-of-thumb":
            assert f["status"] != "ERROR"


def test_check_c3_600m_domain_with_30_arcsec_geog_is_flagged(tmp_path):
    dom = (" time_step = 18, max_dom = 3,\n e_we = 200, 301, 301, e_sn = 200, 301, 301, e_vert = 45, 45, 45,\n"
           " dx = 5400, 1800, 600, dy = 5400, 1800, 600,\n parent_id = 0, 1, 2, parent_grid_ratio = 1, 3, 3,\n"
           " parent_time_step_ratio = 1, 3, 3, i_parent_start = 1, 50, 100, j_parent_start = 1, 50, 100,\n")
    rep = wc.check(write_nl(tmp_path, domains=dom), geog_res_arcsec=30.0)
    st = dict(statuses(rep, "geog.res_vs_dx"))
    assert st["d03"] == "DEVIATES" and st["d01"] == "OK" and st["d02"] == "OK"   # 926 m > 600 m; d02: 926 <= 1800 but not 2x finer -> OK
    d03 = [f for f in rep["findings"] if f["rule_id"] == "geog.res_vs_dx" and f["domain"] == "d03"][0]
    assert d03["source_level"] == "rule-of-thumb" and "粗" in d03["reason"]
    assert wc.check(write_nl(tmp_path, domains=dom, name="b.input"), geog_res_arcsec=3.0)["counts"]["DEVIATES"] < rep["counts"]["DEVIATES"]


def test_check_nudging_rules(tmp_path):
    dom = " time_step = 54, max_dom = 1, e_we = 334, e_sn = 334, e_vert = 45, dx = 9000, dy = 9000,\n"
    fdda = ("&fdda\n grid_fdda = 2, xwavenum = 3, ywavenum = 3, guv = 0.0003, gt = 0.0003, gph = 0.0003, gq = 0.00001,\n"
            " gfdda_interval_m = 360, gfdda_end_h = 6,\n/\n")
    rep = wc.check(write_nl(tmp_path, domains=dom, extra=fdda))
    assert ("d01", "OK") in statuses(rep, "nudge.wavenumber")
    assert ("d01", "OK") in statuses(rep, "nudge.coefficients")
    assert ("d01", "DEVIATES") in statuses(rep, "nudge.moisture")             # gq > 0
    assert ("d01", "DEVIATES") in statuses(rep, "nudge.pbl")                  # PBL nudging left on
    assert ("d01", "OK") in statuses(rep, "nudge.timing")
    assert rep["counts"]["ERROR"] == 0
    wrong = fdda.replace("xwavenum = 3", "xwavenum = 9")
    assert ("d01", "DEVIATES") in statuses(wc.check(write_nl(tmp_path, domains=dom, extra=wrong, name="w.input")), "nudge.wavenumber")
    unset = "&fdda\n grid_fdda = 2, xwavenum = 3, ywavenum = 3,\n/\n"
    rep3 = wc.check(write_nl(tmp_path, domains=dom, extra=unset, name="u.input"))
    assert ("d01", "DEVIATES") in statuses(rep3, "nudge.coefficients")        # Registry default 0
    short = fdda.replace("gfdda_end_h = 6", "gfdda_end_h = 3")
    assert ("d01", "DEVIATES") in statuses(wc.check(write_nl(tmp_path, domains=dom, extra=short, name="s.input")), "nudge.timing")


def test_check_les_acoustic_longrun_movenest(tmp_path):
    dom = " time_step = 1, max_dom = 1, e_we = 301, e_sn = 301, e_vert = 80, dx = 100, dy = 100,\n"
    rep = wc.check(write_nl(tmp_path, domains=dom, physics=" bl_pbl_physics = 1,\n", dynamics=" diff_opt = 1, km_opt = 4,\n"))
    assert ("d01", "DEVIATES") in statuses(rep, "les.regime")
    ok = wc.check(write_nl(tmp_path, domains=dom, physics=" bl_pbl_physics = 0,\n", dynamics=" diff_opt = 2, km_opt = 2,\n", name="l.input"))
    assert ("d01", "OK") in statuses(ok, "les.regime")
    big = " time_step = 18, max_dom = 1, e_we = 100, e_sn = 100, e_vert = 45, dx = 3000, dy = 3000,\n"
    r2 = wc.check(write_nl(tmp_path, domains=big, dynamics=" km_opt = 2, diff_opt = 2,\n time_step_sound = 2,\n", name="a.input"))
    assert ("d01", "DEVIATES") in statuses(r2, "les.km_opt_dx") and ("d01", "DEVIATES") in statuses(r2, "acoustic.sound_steps")
    lr = write_nl(tmp_path, domains=big, time_control="", name="lr.input").replace("run_hours = 6", "run_hours = 6")
    text = Path(lr).read_text().replace("run_hours = 6", "run_days = 10, run_hours = 0")
    Path(lr).write_text(text)
    assert ("all", "DEVIATES") in statuses(wc.check(lr), "longrun.threshold")
    mv = write_nl(tmp_path, domains=NEST_DOMAINS + " vortex_interval = 15, 15, max_vortex_speed = 40, 40,\n", name="m.input")
    assert ("d02", "OK") in statuses(wc.check(mv), "movenest.vortex")


def test_check_does_not_flag_physics_suite_defaults(tmp_path):
    rep = wc.check(write_nl(tmp_path, domains=NEST_DOMAINS, physics=" physics_suite = 'conus', cu_physics = -1, radt = 9,\n"))
    assert rep["counts"]["ERROR"] == 0
    assert all(f["status"] != "DEVIATES" for f in rep["findings"] if f["rule_id"] == "physics.cumulus")


def test_check_real_local_namelists_have_no_error():
    """C2 subset: namelists that exist on this machine (the scc copies must be run by the lead)."""
    cands = [REPO / "tests/fixtures/post/m1-namelist.input", REPO / ".route1-ref/namelist.input.real",
             Path("/home/nk/wrf-local/WRF/run/namelist.input")]
    found = [c for c in cands if c.exists()]
    if not found:
        pytest.skip("no local namelists")
    for c in found:
        rep = wc.check(str(c))
        assert rep["counts"]["ERROR"] == 0, wc.render_check(rep)
        assert rep["counts"]["OK"] > 5


# ------------------------------------------------------------------------------------------ CLI

def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "wtune.calc", *args], cwd=str(REPO / "src"), capture_output=True, text=True)


def test_cli_help_lists_every_topic():
    p = run_cli("--help")
    assert p.returncode == 0
    for t in wc.TOPICS + ("check",):
        assert t in p.stdout


def test_cli_text_output_has_value_formula_level_ref(capsys):
    assert wc.main(["dt", "--dx-km", "9"]) == 0
    out = capsys.readouterr().out
    assert "54" in out and "公式" in out and "來源等級: wrf-doc" in out and "README.namelist:155" in out


def test_cli_json_shape(capsys):
    assert wc.main(["radt", "--dx-km", "3", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"topic", "inputs", "items"} and data["topic"] == "radt"
    for it in data["items"]:
        assert set(it) == {"rule_id", "label", "value", "unit", "note", "formula", "source_level", "source_ref"}
        assert it["source_level"] in LEVELS
    assert wc.main(["--json", "units", "kt-ms", "100"]) == 0
    assert json.loads(capsys.readouterr().out)["items"][0]["value"] == pytest.approx(51.4444, abs=1e-3)


def test_cli_check_exit_codes_and_json(tmp_path, capsys):
    good = write_nl(tmp_path, domains=NEST_DOMAINS, physics=" radt = 9, 9,\n")
    assert wc.main(["check", good]) == 0
    assert "合計" in capsys.readouterr().out
    bad = write_nl(tmp_path, domains=NEST_DOMAINS.replace("time_step = 54", "time_step = 90"), name="bad.input")
    assert wc.main(["check", bad, "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert set(rep) == {"namelist", "geog_res_arcsec", "max_dom", "findings", "counts"}
    assert rep["counts"]["ERROR"] >= 1
    assert set(rep["findings"][0]) == {"rule_id", "domain", "status", "reason", "source_level", "source_ref"}
    assert {f["status"] for f in rep["findings"]} <= {"OK", "DEVIATES", "ERROR"}


def test_cli_geog_res_arcsec_option_and_bad_args(tmp_path, capsys):
    dom = " time_step = 3, max_dom = 1, e_we = 200, e_sn = 200, e_vert = 45, dx = 600, dy = 600,\n"
    nl = write_nl(tmp_path, domains=dom)
    assert wc.main(["check", nl, "--geog-res-arcsec", "30"]) == 0             # deviation is not an error
    out = capsys.readouterr().out
    assert "偏離" in out and "geog.res_vs_dx" in out
    assert wc.main(["units", "hpa-m", "500"]) == 0
    with pytest.raises(SystemExit):
        wc.main(["dt"])                                                      # --dx-km required


def test_wtune_cli_dispatches_calc():
    p = subprocess.run([sys.executable, str(REPO / "src/wtune/cli.py"), "calc", "units", "kt-ms", "10"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "5.14444" in p.stdout
