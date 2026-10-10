"""Tests for wtune/advise.py (decision 0061): detectors with fixtures/fakes, each rule firing and
not firing, A4 (an ordinary non-TC case fires no TC rule), the JSON shape, the CLI.

Real inputs: tests/fixtures/post/m1-namelist.input (Katrina m1: 12->4 km, d02 cu_physics=0,
start 2005-08-27_00, run_hours=48) and the AL122005 HURDAT2 excerpt (959 hPa at 2005-08-27 00Z).
Grid/SLP readers need netCDF4/wrf-python and a real wrfinput, so they are monkeypatched with
numpy fakes; everything else (namelist parsing, best-track readers, TC-Pairs parsing, cost model)
is the real code.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from wtune import advise as adv

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "post")
NAMELIST = os.path.join(FIXTURES, "m1-namelist.input")
HURDAT2 = os.path.join(FIXTURES, "al122005-hurdat2-excerpt.txt")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# d01 of m1 (real XLAT/XLONG extent) as a coarse fake grid; d02 chosen per test
D01_BOX = (17.9, 31.7, -99.0, -79.0)
BIG_NEST = (10.0, 40.0, -110.0, -70.0)       # every Katrina fix of the run is inside
FAR_NEST = (30.0, 32.0, -130.0, -128.0)      # none is inside


def _grid(box, ny=None, nx=None):
    ny = ny or int((box[1] - box[0]) * 111 / 4.0) + 1        # ~4 km spacing, like the real d02
    nx = nx or int((box[3] - box[2]) * 100 / 4.0) + 1
    lat = np.linspace(box[0], box[1], ny)[:, None] * np.ones((1, nx))
    lon = np.ones((ny, 1)) * np.linspace(box[2], box[3], nx)[None, :]
    return lat, lon


def make_case(tmp_path, namelist_edit=None, d01_box=D01_BOX, d02_box=BIG_NEST, slp=(996.4, (24.0, -88.0)),
              monkeypatch=None, **kw):
    d = tmp_path / "case"
    d.mkdir()
    text = open(NAMELIST).read()
    if namelist_edit:
        text = namelist_edit(text)
    (d / "namelist.input").write_text(text)
    (d / "wrfinput_d01").write_text("")
    (d / "wrfinput_d02").write_text("")

    def fake_latlon(path):
        box = d01_box if path.endswith("d01") else d02_box
        return "t", *_grid(box)

    def fake_slp(path, lat, lon, radius):
        if slp is None:
            raise AssertionError("SLP reader must not be called")
        return slp

    if monkeypatch is not None:
        monkeypatch.setattr(adv, "_read_latlon", fake_latlon)
        monkeypatch.setattr(adv, "_read_min_slp", fake_slp)
    return adv.build_case(str(d), **kw)


def set_nl(text, key, value):
    return re.sub(r"(?m)(\b%s\s*=\s*)[^,\n]+" % key, r"\g<1>%s" % value, text, count=1)


def preset(case, **dets):
    """Pre-populate detector outputs so a rule is tested without any I/O."""
    case.cache["det"] = {k: dict(v, available=True) if v is not None else
                         {"available": False, "reason": "preset"} for k, v in dets.items()}
    return case


def fired(result):
    return {m["id"] for m in result["advice"]}


# ---------------------------------------------------------------------------- rules file
def test_rules_file_has_the_eight_rules_with_all_fields():
    doc = adv.load_rules()
    assert len(doc["rules"]) == 8
    assert "只提示，未套用" in doc["footer"]
    for r in doc["rules"]:
        for key in ("id", "trigger", "title", "recommendation", "how", "cost", "evidence",
                    "side_effects", "trigger_text"):
            assert r[key], (r["id"], key)
        assert r["cost"]["method"] in ("cost_model", "measured", "uncalibrated")
        assert r["evidence"]["slug"] and r["evidence"]["text"]
        assert isinstance(r["how"], list) and r["how"]
        for c in r["trigger"]["all"] + r["trigger"]["any"]:
            assert c["detector"] in adv.DETECTORS and c["op"] in adv._OPS


def test_uncalibrated_costs_say_so_and_quote_the_measured_numbers():
    rules = {r["id"]: r for r in adv.load_rules()["rules"]}
    for rid in ("need_chem_wrf_chem", "need_hydro_wrf_hydro", "convection_radar_wrfda",
                "long_run_spectral_nudging"):
        assert rules[rid]["cost"]["method"] == "uncalibrated" and "未校正" in rules[rid]["cost"]["text"]
    mv = rules["tc_leaves_inner_nest_movenest"]["cost"]["text_template"]
    assert "+3.1%" in mv and "+4.6%" in mv
    ev = rules["tc_weak_vortex_early_start"]["evidence"]["text"]
    assert "944–949" in ev
    assert "389" in rules["tc_bogus_vortex_tc_exe"]["side_effects"]


# ---------------------------------------------------------------------------- detectors
def test_case_types_namelist_convection_and_tc_from_best_track(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005")
    ct = adv.run_detector(case, "case_types")
    assert ct["available"] and ct["tc"] and ct["convection"] and not ct["climate"]
    assert "best-track fix" in ct["tc_evidence"] and "d02 dx=4 km" in ct["convection_evidence"]


def test_case_types_storm_autofound_from_hurdat2_alone(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, hurdat2=HURDAT2)
    assert adv.run_detector(case, "case_types")["tc"] is True
    assert case.cache["storm_found"] == "AL122005"


def test_case_types_no_track_is_not_tc_but_registry_family_is(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch)
    assert adv.run_detector(case, "case_types")["tc"] is False
    (tmp_path / "b").mkdir()
    case = make_case(tmp_path / "b", monkeypatch=monkeypatch)
    case.registry = {"case_family": "tropical_cyclone"}
    ct = adv.run_detector(case, "case_types")
    assert ct["tc"] is True and "registry" in ct["tc_evidence"]


def test_case_types_climate_from_run_length(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch,
                     namelist_edit=lambda t: set_nl(t, "run_hours", 24 * 40))
    assert adv.run_detector(case, "case_types")["climate"] is True


def test_tc_vortex_deficit_delta_against_best_track(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005",
                     slp=(996.4, (24.5, -87.5)))
    d = adv.run_detector(case, "tc_vortex_deficit")
    assert d["available"]
    assert d["best_pmin_hpa"] == 959.0 and d["best_fix_time"] == "2005-08-27_00:00:00"
    assert d["delta_hpa"] == pytest.approx(37.4)
    assert d["model_slp_hpa"] == 996.4


def test_tc_vortex_deficit_unavailable_without_track_or_state(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch)
    assert "最佳路徑" in adv.run_detector(case, "tc_vortex_deficit")["reason"]
    (tmp_path / "x").mkdir()
    case2 = make_case(tmp_path / "x", monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005")
    case2.first_state = None
    assert "wrfinput" in adv.run_detector(case2, "tc_vortex_deficit")["reason"]


def test_tc_nest_exit_inside_and_outside(tmp_path, monkeypatch):
    inside = make_case(tmp_path, monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005",
                       d02_box=BIG_NEST)
    r = adv.run_detector(inside, "tc_nest_exit")
    assert r["available"] and r["n_total"] > 0 and r["n_outside"] == 0 and r["nest_domain"] == 2
    (tmp_path / "o").mkdir()
    outside = make_case(tmp_path / "o", monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005",
                        d02_box=FAR_NEST)
    r = adv.run_detector(outside, "tc_nest_exit")
    assert r["n_outside"] == r["n_total"] > 0 and r["first_exit"] == "2005-08-27_00:00:00"


def test_tc_nest_exit_single_domain_has_no_nest(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005",
                     namelist_edit=lambda t: set_nl(t, "max_dom", 1))
    r = adv.run_detector(case, "tc_nest_exit")
    assert r["available"] and r["n_outside"] == 0 and r["nest_domain"] is None


def test_radar_coverage_us_in_window_and_not(tmp_path, monkeypatch):
    def edit(t):
        for k, v in (("start_year", "2022, 2022"), ("end_year", "2022, 2022"),
                     ("start_month", "05, 05"), ("end_month", "05, 05")):
            t = set_nl(t, k, v)
        return t
    us = make_case(tmp_path, monkeypatch=monkeypatch, namelist_edit=edit, d01_box=(34.0, 40.0, -102.0, -94.0))
    r = adv.run_detector(us, "radar_coverage")
    assert r["us_domain"] and r["in_mrms"] and r["in_nexrad"] and r["covered"]
    (tmp_path / "a").mkdir()
    asia = make_case(tmp_path / "a", monkeypatch=monkeypatch, namelist_edit=edit,
                     d01_box=(20.0, 30.0, 110.0, 125.0))
    assert adv.run_detector(asia, "radar_coverage")["covered"] is False
    (tmp_path / "o").mkdir()
    old = make_case(tmp_path / "o", monkeypatch=monkeypatch, d01_box=(34.0, 40.0, -102.0, -94.0))
    r = adv.run_detector(old, "radar_coverage")      # 2005-08-27: before MRMS and before the verified NEXRAD date
    assert r["us_domain"] and not r["in_mrms"] and not r["in_nexrad"] and not r["covered"]


def test_radar_coverage_namelist_fallback_needs_ref_latlon(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch)
    case.first_state = None
    assert "ref_lat" in adv.run_detector(case, "radar_coverage")["reason"]


def test_run_length_days(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch)
    assert adv.run_detector(case, "run_length")["run_days"] == 2.0


def _write_tcst(path, leads, tk_err_nmi, amslp, bmslp):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["VERSION  LEAD     TK_ERR  AMSLP  BMSLP"]
    for ld, e, a, b in zip(leads, tk_err_nmi, amslp, bmslp):
        lines.append("V11.0.0 %s %s %s %s" % (ld, e, a, b))
    path.write_text("\n".join(lines) + "\n")


def test_post_detectors_read_tc_pairs(tmp_path, monkeypatch):
    post_tc = tmp_path / "tc"
    # LEAD as HHMMSS strings (met_tc/tc.py lead_h): 000000, 240000, 480000, 720000, 960000
    _write_tcst(post_tc / "met_m1" / "tc_pairs.tcst",
                ["000000", "240000", "480000", "720000", "960000"],
                [10, 30, 60, 100, 150], [980, 975, 970, 972, 968], [950, 945, 940, 935, 930])
    case = make_case(tmp_path, monkeypatch=monkeypatch, post_tc_dir=str(post_tc))
    g = adv.run_detector(case, "post_track_growth")
    assert g["available"] and g["slope_km_per_day"] > 30 and g["max_track_err_km"] == pytest.approx(150 * 1.852)
    b = adv.run_detector(case, "post_intensity_bias")
    assert b["median_weak_hpa"] == 30.0 and b["max_weak_hpa"] == 38.0


def test_post_detectors_unavailable_without_output(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, post_tc_dir=str(tmp_path / "nope"))
    assert not adv.run_detector(case, "post_track_growth")["available"]
    assert not adv.run_detector(case, "post_intensity_bias")["available"]


# ---------------------------------------------------------------------------- rules fire / not
TC_ON = {"case_types": {"tc": True, "convection": False, "tc_evidence": "fix", "convection_evidence": "",
                        "types": ["common", "tc"]}}


def _run(tmp_path, monkeypatch, **dets):
    case = make_case(tmp_path, monkeypatch=monkeypatch)
    base = {"case_types": {"tc": False, "convection": False, "tc_evidence": "", "convection_evidence": "",
                           "types": ["common"]},
            "run_length": {"run_days": 2.0, "run_hours": 48.0},
            "user_need": {n: False for n in adv.NEEDS}}
    base.update(dets)
    preset(case, **base)
    return adv.advise(case)


def test_rule_early_start_and_tc_exe_fire_on_init_deficit_and_not_when_small(tmp_path, monkeypatch):
    on = _run(tmp_path, monkeypatch, **TC_ON, tc_vortex_deficit={
        "delta_hpa": 37.4, "model_slp_hpa": 996.4, "best_pmin_hpa": 959.0})
    assert {"tc_weak_vortex_early_start", "tc_bogus_vortex_tc_exe"} <= fired(on)
    m = {x["id"]: x for x in on["advice"]}["tc_weak_vortex_early_start"]
    assert "37.4" in m["trigger"] and "996.4" in m["trigger"] and "959" in m["trigger"]
    assert "+50%" in m["cost"]["text"] and "+75%" in m["cost"]["text"]       # 24/48, 36/48
    (tmp_path / "n").mkdir()
    off = _run(tmp_path / "n", monkeypatch, **TC_ON, tc_vortex_deficit={
        "delta_hpa": 10.0, "model_slp_hpa": 969.0, "best_pmin_hpa": 959.0})
    assert not ({"tc_weak_vortex_early_start", "tc_bogus_vortex_tc_exe"} & fired(off))


def test_rule_early_start_fires_on_post_intensity_only(tmp_path, monkeypatch):
    on = _run(tmp_path, monkeypatch, **TC_ON, tc_vortex_deficit=None,
              post_intensity_bias={"median_weak_hpa": 25.0})
    assert "tc_weak_vortex_early_start" in fired(on)
    assert "tc_vortex_deficit" not in on["unavailable"]            # rule fired: no noise
    (tmp_path / "n").mkdir()
    off = _run(tmp_path / "n", monkeypatch, **TC_ON, tc_vortex_deficit=None,
               post_intensity_bias={"median_weak_hpa": 20.0})          # threshold is strict >
    assert "tc_weak_vortex_early_start" not in fired(off)
    assert "tc_vortex_deficit" in off["unavailable"]               # absence is reported, not "no"


def test_rule_movenest_fires_only_when_track_leaves_nest(tmp_path, monkeypatch):
    on = _run(tmp_path, monkeypatch, **TC_ON, tc_nest_exit={
        "n_total": 9, "n_outside": 4, "nest_domain": 2, "first_exit": "2005-08-28_00:00:00"})
    m = {x["id"]: x for x in on["advice"]}["tc_leaves_inner_nest_movenest"]
    assert "4 個" in m["trigger"] and "d02" in m["trigger"]
    assert "+3.1%" in m["cost"]["text"] and "6.0%" in m["cost"]["text"]
    (tmp_path / "n").mkdir()
    off = _run(tmp_path / "n", monkeypatch, **TC_ON, tc_nest_exit={
        "n_total": 9, "n_outside": 0, "nest_domain": 2, "first_exit": None})
    assert "tc_leaves_inner_nest_movenest" not in fired(off)


CONV = {"case_types": {"tc": False, "convection": True, "convection_evidence": "d02 dx=3 km, cu_physics=0",
                       "tc_evidence": "", "types": ["common", "convection"]}}


def test_rule_radar_wrfda_needs_convection_coverage_and_not_tc(tmp_path, monkeypatch):
    cov = {"covered": True, "text": "MRMS 可用"}
    assert "convection_radar_wrfda" in fired(_run(tmp_path, monkeypatch, **CONV, radar_coverage=cov))
    (tmp_path / "a").mkdir()
    assert "convection_radar_wrfda" not in fired(_run(tmp_path / "a", monkeypatch, **CONV,
                                                      radar_coverage={"covered": False, "text": "x"}))
    (tmp_path / "b").mkdir()
    noconv = {"case_types": {**CONV["case_types"], "convection": False}}
    assert "convection_radar_wrfda" not in fired(_run(tmp_path / "b", monkeypatch, **noconv, radar_coverage=cov))
    (tmp_path / "c").mkdir()
    tc = {"case_types": {**CONV["case_types"], "tc": True}}
    assert "convection_radar_wrfda" not in fired(_run(tmp_path / "c", monkeypatch, **tc, radar_coverage=cov))
    (tmp_path / "d").mkdir()
    r = _run(tmp_path / "d", monkeypatch, **CONV, radar_coverage=cov)
    m = [x for x in r["advice"] if x["id"] == "convection_radar_wrfda"][0]
    assert m["cost"]["method"] == "uncalibrated" and "觀測數" in m["cost"]["text"]


def test_rule_nudging_on_run_length_or_post_growth(tmp_path, monkeypatch):
    long_ = _run(tmp_path, monkeypatch, run_length={"run_days": 396.0, "run_hours": 9504.0})
    assert "long_run_spectral_nudging" in fired(long_)
    assert "396.0 天" in [x for x in long_["advice"] if x["id"] == "long_run_spectral_nudging"][0]["trigger"]
    (tmp_path / "n").mkdir()
    short = _run(tmp_path / "n", monkeypatch, run_length={"run_days": 4.9, "run_hours": 117.6})
    assert "long_run_spectral_nudging" not in fired(short)
    (tmp_path / "g").mkdir()
    grow = _run(tmp_path / "g", monkeypatch, run_length={"run_days": 2.0, "run_hours": 48.0},
                post_track_growth={"slope_km_per_day": 45.0})
    assert "long_run_spectral_nudging" in fired(grow)


@pytest.mark.parametrize("need,rid", [("chem", "need_chem_wrf_chem"), ("hydro", "need_hydro_wrf_hydro"),
                                      ("surge", "need_surge_holland_ocean_model")])
def test_need_rules_fire_only_when_declared(tmp_path, monkeypatch, need, rid):
    off = _run(tmp_path, monkeypatch)
    assert rid not in fired(off)
    (tmp_path / "x").mkdir()
    case = make_case(tmp_path / "x", monkeypatch=monkeypatch, needs=[need])
    on = adv.advise(case)
    assert fired(on) >= {rid} and sum(1 for i in fired(on) if i.startswith("need_")) == 1
    m = on["advice"][0]
    assert m["how"] and m["cost"]["text"] and m["cost"]["method"] == "uncalibrated"


def test_chem_cost_is_labelled_uncalibrated_and_hydro_too(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, needs=["chem", "hydro"])
    res = {m["id"]: m for m in adv.advise(case)["advice"]}
    assert "未校正" in res["need_chem_wrf_chem"]["cost"]["text"]
    assert "未校正" in res["need_hydro_wrf_hydro"]["cost"]["text"]


# ---------------------------------------------------------------------------- A4
def test_ordinary_non_tc_case_fires_no_tc_rule_and_never_reads_slp(tmp_path, monkeypatch):
    # Pacific box, Katrina best track given: no fix inside d01 -> not a TC case -> TC rules silent
    case = make_case(tmp_path, monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005",
                     d01_box=(25.0, 35.0, 125.0, 140.0), slp=None)
    res = adv.advise(case)
    assert not [i for i in fired(res) if i.startswith("tc_")]
    assert res["case_types"] and "tc" not in res["case_types"]


def test_ordinary_case_without_any_track_fires_no_tc_rule(tmp_path, monkeypatch):
    res = adv.advise(make_case(tmp_path, monkeypatch=monkeypatch, slp=None))
    assert not [i for i in fired(res) if i.startswith("tc_")]


# ---------------------------------------------------------------------------- end to end (Katrina-like)
def test_katrina_like_case_end_to_end(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005",
                     d02_box=FAR_NEST)
    res = adv.advise(case)
    ids = fired(res)
    assert {"tc_weak_vortex_early_start", "tc_bogus_vortex_tc_exe", "tc_leaves_inner_nest_movenest"} <= ids
    text = adv.render_text(res)
    assert "37.4" in text and "389" in text and "未套用" in text and "怎麼做" in text
    early = {m["id"]: m for m in res["advice"]}["tc_weak_vortex_early_start"]
    assert "內層 nest" in early["cost"]["text"]
    assert re.search(r"先驗成本 \+\d+%", early["cost"]["text"])      # nested refinement from the cost model


def test_nest_refinement_cost_none_when_already_fine(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch,
                     namelist_edit=lambda t: set_nl(t, "dx", "12000, 3000"))
    assert adv.nest_refinement_extra_pct(case) is None
    (tmp_path / "c").mkdir()
    case2 = make_case(tmp_path / "c", monkeypatch=monkeypatch)
    pct, ratio, dx_km = adv.nest_refinement_extra_pct(case2)      # 4 km finest -> ratio 2 -> 2 km
    assert pct > 0 and ratio == 2 and dx_km == 2.0


# ---------------------------------------------------------------------------- JSON shape / surfaces
def test_json_shape(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, hurdat2=HURDAT2, storm="AL122005", needs=["surge"])
    res = json.loads(json.dumps(adv.advise(case), default=str))
    assert set(res) == {"case", "surface", "advice", "not_fired", "unavailable", "case_types", "footer"}
    assert res["advice"]
    for m in res["advice"]:
        assert set(m) == {"id", "title", "trigger", "recommendation", "how", "cost", "evidence",
                          "side_effects", "footer", "measured"}
        assert set(m["cost"]) == {"method", "text"} and set(m["evidence"]) == {"text", "slug"}
        assert m["footer"] == res["footer"] and "只提示，未套用" in m["footer"]


def test_post_surface_keeps_only_verification_rules(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, needs=["chem", "hydro", "surge"])
    res = adv.advise(case, surface="post")
    assert not [i for i in fired(res) if i.startswith("need_")]
    allowed = {"tc_weak_vortex_early_start", "tc_bogus_vortex_tc_exe", "long_run_spectral_nudging"}
    assert set(res["not_fired"]) <= allowed


def test_render_html_and_report_hook(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch=monkeypatch, needs=["chem"])
    html = adv.render_html(adv.advise(case))
    assert "Advice" in html and "WRF-Chem" in html and "只提示，未套用" in html
    assert adv.render_html({"advice": []}) == ""
    from wtune.post import report
    out = tmp_path / "rep"
    out.mkdir()
    info = {"types": ["common"], "evidence": {"common": "always"}, "run_length_h": 48.0,
            "d01_extent_km": (1000.0, 1000.0)}
    path = report.build_recipe_report(str(out), info, [], ["/r"], ["run0"], 6, {}, advice_html=html)
    body = open(path).read()
    assert "Advice" in body and "WRF-Chem" in body
    path2 = report.build_recipe_report(str(out), info, [], ["/r"], ["run0"], 6, {})
    assert "Advice" not in open(path2).read()


def test_tune_hook_appends_text_and_json(monkeypatch):
    from wtune import tune as wtune_tune
    fake = {"case": "k", "surface": "advise", "case_types": ["common"], "unavailable": {"x": "r"},
            "footer": adv.load_rules()["footer"], "not_fired": [],
            "advice": [{"id": "i", "title": "T", "trigger": "t", "recommendation": "r", "how": ["h"],
                        "cost": {"method": "uncalibrated", "text": "未校正"},
                        "evidence": {"text": "e", "slug": "s"}, "side_effects": "se",
                        "footer": adv.load_rules()["footer"], "measured": {}}]}
    monkeypatch.setattr(adv, "advise_case_id", lambda cid, **kw: fake)
    report = {"case": "k"}
    text = wtune_tune._attach_advice(report)
    assert "只提示，未套用" in text and report["advice"] == fake["advice"]
    monkeypatch.setattr(adv, "advise_case_id", lambda cid, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    report = {"case": "k"}
    assert wtune_tune._attach_advice(report) == "" and report["advice"] == [] \
        and "boom" in report["advice_error"]


# ---------------------------------------------------------------------------- CLI
def test_cli_help_lists_need_and_json(capsys):
    with pytest.raises(SystemExit) as e:
        adv.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--need" in out and "--json" in out and "advice" in out.lower()


def test_cli_rejects_unknown_need_and_unknown_case(capsys):
    assert adv.main(["x", "--need", "bogus"]) == 2
    assert adv.main(["no-such-case-xyz"]) == 1
    assert "unknown case" in capsys.readouterr().err


def test_cli_json_and_need_comma_list(tmp_path, monkeypatch, capsys):
    d = tmp_path / "c"
    d.mkdir()
    shutil.copy(NAMELIST, d / "namelist.input")
    assert adv.main([str(d), "--need", "chem,surge", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert {"need_chem_wrf_chem", "need_surge_holland_ocean_model"} <= {m["id"] for m in res["advice"]}


def test_wtune_cli_dispatches_advise(tmp_path):
    out = subprocess.run([sys.executable, os.path.join(REPO, "src", "wtune"), "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0 and "advise" in out.stdout
    d = tmp_path / "c"
    d.mkdir()
    shutil.copy(NAMELIST, d / "namelist.input")
    r = subprocess.run([sys.executable, os.path.join(REPO, "src", "wtune"), "advise", str(d), "--need", "hydro"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and "WRF-Hydro" in r.stdout


def test_dir_with_only_namelist_wps_gives_valid_json_and_exit_0(tmp_path, capsys):
    d = tmp_path / "era5"
    d.mkdir()
    (d / "namelist.wps").write_text("&share\n/\n")
    assert adv.main([str(d), "--json", "--hurdat2", HURDAT2, "--storm", "AL122005"]) == 0
    res = json.loads(capsys.readouterr().out)          # whole stdout is one JSON document
    assert res["advice"] == [] or isinstance(res["advice"], list)
    assert "case_types" in res and "unavailable" in res
    d2 = adv.run_detector(adv.build_case(str(d), hurdat2=HURDAT2, storm="AL122005"), "tc_vortex_deficit")
    assert d2["available"] is False and d2["reason"]


def test_namelist_less_dir_tc_vortex_reported_unavailable_when_tc_forced(tmp_path):
    d = tmp_path / "e"
    d.mkdir()
    case = adv.build_case(str(d), hurdat2=HURDAT2, storm="AL122005")
    preset(case, case_types={"tc": True, "convection": False, "tc_evidence": "", "convection_evidence": "",
                             "types": ["common", "tc"]}, tc_vortex_deficit=None)
    res = adv.advise(case)
    assert "tc_vortex_deficit" in res["unavailable"] and not res["advice"]


def test_json_error_stays_valid_on_unknown_case(capsys):
    assert adv.main(["no-such-case-xyz", "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_how_lines_flag_unverified_values_and_cite_sources():
    rules = {r["id"]: r for r in adv.load_rules()["rules"]}
    tcexe = " ".join(rules["tc_bogus_vortex_tc_exe"]["how"])
    assert "real.exe 之前" in tcexe and "insert_bogus_storm" in tcexe and "README.namelist" in tcexe
    assert "VORTEX_CENTER" in " ".join(rules["tc_leaves_inner_nest_movenest"]["how"])
    assert "經驗值，未查證" in " ".join(rules["long_run_spectral_nudging"]["how"])
