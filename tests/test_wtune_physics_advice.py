"""Tests for src/wtune_physics_advice.py (decision 0059 phase 3). Synthetic namelists and numbers
only; the field-by-field gate is monkeypatched at wtune_config_gate.compare (its tolerance check,
_check_results, stays real)."""
import datetime
import math

import pytest

import wtune_physics_advice as adv
import wtune_probe_plan as pp


def _nl(tmp_path, *, dx, dt, mp=8, cu=1, bl=1, sfclay=1, lsm=2, lw=4, sw=4, radt=None,
        max_dom=1, extra="", chem=""):
    """Minimal single-/two-domain namelist; per-domain lists for max_dom=2 (dx given as a list)."""
    dxs = dx if isinstance(dx, list) else [dx]
    n = len(dxs)
    rep = lambda v: ", ".join(str(v) for _ in range(n))
    radt = radt if radt is not None else max(1, round(dxs[-1] / 1000.0))
    text = f"""&time_control
 run_hours = 6,
/
&domains
 time_step = {dt},
 max_dom = {n},
 e_we = {", ".join("100" for _ in dxs)},
 e_sn = {", ".join("100" for _ in dxs)},
 e_vert = {rep(40)},
 dx = {", ".join(str(v) for v in dxs)},
 dy = {", ".join(str(v) for v in dxs)},
 parent_grid_ratio = {", ".join(["1"] + ["3"] * (n - 1))},
 parent_time_step_ratio = {", ".join(["1"] * n)},
/
&physics
 mp_physics = {rep(mp)},
 cu_physics = {rep(cu)},
 bl_pbl_physics = {rep(bl)},
 sf_sfclay_physics = {rep(sfclay)},
 sf_surface_physics = {rep(lsm)},
 ra_lw_physics = {rep(lw)},
 ra_sw_physics = {rep(sw)},
 radt = {rep(radt)},
{extra}/
{chem}"""
    p = tmp_path / "namelist.input"
    p.write_text(text)
    return str(p)


def _ids(cands):
    return [c["id"] for c in cands]


# ---------------------------------------------------------------------------- plan_candidates
def test_dx3km_offers_cu_off_radt_and_cheaper_mp(tmp_path):
    c = adv.plan_candidates(_nl(tmp_path, dx=3000, dt=18, mp=8, cu=1, radt=3))
    ids = _ids(c)
    assert "CU-OFF" in ids and "P02" in ids and "P04" in ids
    cu = next(x for x in c if x["id"] == "CU-OFF")
    assert cu["edits"] == {"cu_physics": "0"} and cu["class"] == "PHYSICS-GRAY"
    p02 = next(x for x in c if x["id"] == "P02")
    assert p02["edits"]["radt"] == "6" and p02["edits"]["swint_opt"] == "1"
    p04 = next(x for x in c if x["id"] == "P04")
    assert "double->single" in p04["rationale"]          # Thompson -> WSM6 changes moment class
    assert all(x["rationale"] and "\n" not in x["rationale"] for x in c)
    # order: restart-compatible (measurable) first, then estimate-only; each block by prior gain
    assert all(g > 0 for g in (x["prior_gain_pct"] for x in c))
    for flag in (True, False):
        gains = [x["prior_gain_pct"] for x in c if x["restart_compatible"] is flag]
        assert gains == sorted(gains, reverse=True)
    flags = [x["restart_compatible"] for x in c]
    assert flags == sorted(flags, reverse=True)
    assert cu["restart_compatible"] is False and p02["restart_compatible"] is True
    assert _ids(adv.plan_candidates(_nl(tmp_path, dx=3000, dt=18, mp=8, cu=1, radt=3))) == ids


def test_dx12km_never_offers_cu_off(tmp_path):
    c = adv.plan_candidates(_nl(tmp_path, dx=12000, dt=72, mp=8, cu=1, radt=12))
    assert "CU-OFF" not in _ids(c)
    assert "P02" in _ids(c)


def test_cu_already_off_not_offered(tmp_path):
    c = adv.plan_candidates(_nl(tmp_path, dx=3000, dt=18, cu=0, radt=3))
    assert "CU-OFF" not in _ids(c)


def test_same_class_cheaper_mp_single_moment(tmp_path):
    c = adv.plan_candidates(_nl(tmp_path, dx=3000, dt=18, mp=4, cu=0, radt=3))   # WSM5 -> WSM3
    mp = next(x for x in c if x["id"] == "MP-3")
    assert mp["edits"] == {"mp_physics": "3"} and "double->single" not in mp["rationale"]
    assert "MP-1" not in _ids(c)             # Kessler (warm-rain only) is not a substitute


def test_two_domain_cu_off_only_where_dx_le_4km(tmp_path):
    p = _nl(tmp_path, dx=[9000, 3000], dt=10, mp=4, cu=5, bl=2, sfclay=2, lw=1, sw=2, radt=9)
    c = adv.plan_candidates(p, frozen_keys=["time_step"])
    cu = next(x for x in c if x["id"] == "CU-OFF")
    assert cu["edits"] == {"cu_physics": "5, 0"}
    assert "P03" not in _ids(c)               # time_step is frozen -> excluded, not ranked low
    assert any(s[0] == "P03" and "frozen" in s[1] for s in c.skipped)
    text = open(p).read()
    out = adv.render_candidate_namelist(text, cu)
    assert "cu_physics = 5, 0" in out.replace("cu_physics=", "cu_physics = ").replace("  ", " ")


def test_time_step_candidate_present_when_not_frozen(tmp_path):
    p = _nl(tmp_path, dx=3000, dt=18, radt=3)
    assert "P03" in _ids(adv.plan_candidates(p))
    assert "P03" not in _ids(adv.plan_candidates(p, frozen_keys=["TIME_STEP"]))


def test_max_candidates_truncates_and_reports(tmp_path):
    p = _nl(tmp_path, dx=3000, dt=18, radt=3)
    full = adv.plan_candidates(p, max_candidates=50, max_estimate_only=50)
    c2 = adv.plan_candidates(p, max_candidates=2, max_estimate_only=0)
    assert _ids(c2) == [i for i in _ids(full) if i in _ids(c2)]       # same relative order
    assert len(c2) == 2 and all(c["restart_compatible"] for c in c2)
    assert any("max_candidates" in r for _, r in c2.skipped)


def test_chem_case_returns_empty_with_reason(tmp_path):
    p = _nl(tmp_path, dx=3000, dt=18, chem="&chem\n chem_opt = 2,\n/\n")
    c = adv.plan_candidates(p)
    assert c == [] and "WRF-Chem" in c.note


def test_inactive_chem_group_is_not_an_extension_case(tmp_path):
    p = _nl(tmp_path, dx=3000, dt=18, radt=3, chem="&chem\n chem_opt = 0,\n/\n")
    assert adv.plan_candidates(p) != []


@pytest.mark.parametrize("chem,why", [
    ("&hydro_nlist\n sys_cpl = 1,\n/\n", "WRF-Hydro"),
    ("&wrfvar1\n var4d = .false.,\n/\n", "WRFDA"),
])
def test_other_extensions_return_empty(tmp_path, chem, why):
    c = adv.plan_candidates(_nl(tmp_path, dx=3000, dt=18, chem=chem))
    assert c == [] and why in c.note


def test_moving_nest_returns_empty(tmp_path):
    c = adv.plan_candidates(_nl(tmp_path, dx=3000, dt=18, extra=" vortex_interval = 15,\n"))
    assert c == [] and "Moving nest" in c.note


# ---------------------------------------------------------------------------- predicted_speedup
def test_predicted_speedup_weighted_saving_and_spread():
    base = {"a": 1.0, "b": 2.0, "c": 4.0}
    cand = {"a": 0.9, "b": 1.6, "c": 3.8}
    w = {"a": 100.0, "b": 100.0, "c": 200.0}
    r = adv.predicted_speedup(base, cand, w)
    assert r["per_point_ratio"] == pytest.approx({"a": 0.9, "b": 0.8, "c": 0.95})
    tb, tc = 100 * 1 + 100 * 2 + 200 * 4, 100 * 0.9 + 100 * 1.6 + 200 * 3.8
    assert r["saving_pct"] == pytest.approx(100 * (1 - tc / tb))
    assert r["speedup"] == pytest.approx(tb / tc)
    assert (r["ratio_min"], r["ratio_max"]) == pytest.approx((0.8, 0.95))
    assert r["spread_pct"] == pytest.approx(15.0)
    assert r["n_points"] == 3 and r["dropped"] == []


def test_predicted_speedup_drops_unmatched_points_and_rejects_empty():
    r = adv.predicted_speedup({"a": 1.0, "b": 1.0}, {"a": 0.5, "z": 1.0}, {"a": 1.0, "b": 1.0})
    assert r["n_points"] == 1 and r["dropped"] == ["b", "z"]
    assert r["saving_pct"] == pytest.approx(50.0)
    with pytest.raises(ValueError):
        adv.predicted_speedup({"a": 1.0}, {"b": 1.0}, {"a": 1.0})
    with pytest.raises(ValueError):
        adv.predicted_speedup({"a": 0.0}, {"a": 1.0}, {"a": 1.0})


def test_weights_from_combine_reproduce_combine_total():
    t0 = datetime.datetime(2026, 1, 1)
    run_end = t0 + datetime.timedelta(hours=2)
    pms = [pp.PointMeasurement(at=t0 + datetime.timedelta(minutes=m), reason="t", source="x",
                               s_normal=s, n_rad=1, r=3.0)
           for m, s in ((0, 0.10), (60, 0.20), (120, 0.15))]
    cal = {"radiation": {"r_rad_default": 3.0}}
    w = adv.weights_from_combine(pms, t0, run_end, 30, 10, calibration=cal)
    assert len(w) == 3 and all(v > 0 for v in w.values())
    # candidate = every point 20% cheaper -> whole run exactly 20% cheaper
    base_tot = pp.combine(pms, t0, run_end, 30, 10, calibration=cal)["compute_total_s"]
    assert sum(w[p.at.isoformat()] * p.s_normal for p in pms) == pytest.approx(base_tot, rel=1e-3)
    b = {p.at.isoformat(): p.s_normal for p in pms}
    r = adv.predicted_speedup(b, {k: 0.8 * v for k, v in b.items()}, w)
    assert r["saving_pct"] == pytest.approx(20.0) and r["spread_pct"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------- fidelity
def _res(name, rel, status="pass", reason=""):
    return {"name": name, "n": 10, "n_diff": 1, "max_abs": rel, "rms_diff": rel, "rms_ref": 1.0,
            "rel_rms": rel, "status": status, "reason": reason}


def test_fidelity_verdict_pass_and_fail(monkeypatch):
    seen = {}

    def fake(ref, cand, fields=None, skip=None, time_index=-1):
        seen["args"] = (ref, cand, fields, time_index)
        return [_res("T2", 0.001), _res("U10", 0.2), _res("QVAPOR", 0.01)]
    monkeypatch.setattr(adv.gate, "compare", fake)
    ok = adv.fidelity_verdict("ref.nc", "c.nc", tol_rel=0.5)
    assert ok["passed"] and ok["verdict"] == "PASS" and ok["tolerance_source"] == "caller"
    assert [w["name"] for w in ok["worst"]] == ["U10", "QVAPOR", "T2"]
    assert seen["args"][:2] == ("ref.nc", "c.nc") and "W" not in seen["args"][2]
    bad = adv.fidelity_verdict("ref.nc", "c.nc", tol_rel=0.05)
    assert not bad["passed"] and bad["failed_fields"] == ["U10"]
    assert bad["worst"][0]["name"] == "U10" and bad["worst"][0]["status"] == "fail"
    dflt = adv.fidelity_verdict("ref.nc", "c.nc")
    assert dflt["tol_rel"] == adv.DEFAULT_TOL_REL and dflt["tolerance_source"] == "provisional-default"


def test_fidelity_verdict_structural_and_empty(monkeypatch):
    monkeypatch.setattr(adv.gate, "compare", lambda *a, **k: [
        {**_res("T2", math.inf, "fail", "missing"), "max_abs": math.inf}])
    r = adv.fidelity_verdict("r", "c", tol_rel=1e9)
    assert not r["passed"] and r["failed_fields"] == ["T2"]
    monkeypatch.setattr(adv.gate, "compare", lambda *a, **k: [])
    r = adv.fidelity_verdict("r", "c")
    assert not r["passed"] and "no comparable" in r["reason"]


def test_calibrate_tolerance(monkeypatch):
    monkeypatch.setattr(adv.gate, "compare", lambda *a, **k: [_res("T2", 0.01), _res("U10", 0.04)])
    assert adv.calibrate_tolerance("r", "a") == pytest.approx(0.12)
    monkeypatch.setattr(adv.gate, "compare", lambda *a, **k: [_res("T2", math.inf, "fail", "missing")])
    with pytest.raises(ValueError):
        adv.calibrate_tolerance("r", "a")


# ---------------------------------------------------------------------------- advice_table
def _speed(saving, lo=0.9, hi=1.0):
    return {"saving_pct": saving, "ratio_min": lo, "ratio_max": hi}


def _fid(passed, worst="U10", rms=0.02):
    return {"passed": passed, "worst": [{"name": worst, "rel_rms": rms}]}


def test_advice_table_recommendations_and_columns():
    cands = [{"id": i, "label": "L" + i, "rationale": "R" + i} for i in "ABCDEF"]
    speed = {"A": _speed(8.0, 0.88, 0.97), "B": _speed(2.0), "C": _speed(12.0), "D": _speed(6.0),
             "E": _speed(-1.0)}
    fid = {"A": _fid(True), "B": _fid(True), "C": _fid(False, "W", 0.9), "E": _fid(False)}
    rows = {r["id"]: r for r in adv.advice_table(cands, speed, fid)}
    assert rows["A"]["recommendation"] == "consider"
    assert rows["B"]["recommendation"] == "not worth it"
    assert rows["C"]["recommendation"] == "changes results beyond tolerance"
    assert rows["D"]["recommendation"] == "fidelity not checked"
    assert rows["E"]["recommendation"] == "not worth it"            # slower: saving < 5% wins
    assert rows["F"]["recommendation"] == "not measured"
    a = rows["A"]
    assert (a["predicted_saving_pct"], a["ratio_min"], a["ratio_max"]) == (8.0, 0.88, 0.97)
    assert (a["fidelity"], a["worst_field"], a["worst_rel_rms"]) == ("pass", "U10", 0.02)
    assert rows["C"]["fidelity"] == "fail" and rows["C"]["worst_field"] == "W"
    assert rows["D"]["fidelity"] == "not run"
    assert all(r["note"] == "advice only — not applied" and r["class"] == "PHYSICS-GRAY"
               for r in rows.values())


def test_advice_table_order_saving_desc_unmeasured_last_stable():
    cands = [{"id": i, "label": i, "rationale": i} for i in "XYZW"]
    out = adv.advice_table(cands, {"Y": _speed(9.0), "Z": _speed(9.0), "X": _speed(1.0)}, {})
    assert [r["id"] for r in out] == ["Y", "Z", "X", "W"]


def test_threshold_boundary_is_inclusive_at_5():
    c = [{"id": "A", "label": "a", "rationale": "r"}]
    assert adv.advice_table(c, {"A": _speed(5.0)}, {"A": _fid(True)})[0]["recommendation"] == "consider"
    assert adv.advice_table(c, {"A": _speed(4.99)}, {"A": _fid(True)})[0]["recommendation"] == "not worth it"


# ---------------------------------------------------------------------------- chaos-floor gate
def _payload(label, rows=None, deciding="d"):
    return {"label": label, "deciding": deciding, "last_time": "2026-01-01_01:00:00",
            "thresholds": {"floor_ratio_max": 2.0},
            "per_field_summary_at_last_time": rows or {}}


def _row(fr, nd, status="ok"):
    return {"status": status, "floor_ratio": fr, "nd": nd, "rms_diff": 1.0, "floor": 0.5}


@pytest.mark.parametrize("label,passed", [("WITHIN-FLOOR", True), ("SMALL-DRIFT", True),
                                          ("LARGE-DRIFT", False), ("INVALID", False)])
def test_fidelity_verdict_twin_uses_chaos_floor_label(monkeypatch, label, passed):
    seen = {}

    def fake(ref, twin, cand, family, bdy, thr, prefix, time_tol_s=None):
        seen.update(ref=ref, twin=twin, cand=cand, prefix=prefix)
        return _payload(label, {"T2": _row(1.0, 0.01), "QSNOW": _row(3.5, 0.2),
                                "U10": _row(1.5, 0.05), "bad": _row(99, 9, "missing")})
    monkeypatch.setattr(adv.chaos_gate, "run_gate", fake)
    monkeypatch.setattr(adv.gate, "compare", lambda *a, **k: pytest.fail("config gate used"))
    r = adv.fidelity_verdict("ref.nc", "cand.nc", twin="twin.nc")
    assert r["label"] == r["verdict"] == label and r["passed"] is passed
    assert r["gate"] == "chaos-floor" and "chaos floor" in r["tolerance_source"]
    assert seen["ref"] == ["ref.nc"] and seen["twin"] == ["twin.nc"] and seen["cand"] == ["cand.nc"]
    assert [w["name"] for w in r["worst"]] == ["QSNOW", "U10", "T2"]     # by floor_ratio, ok rows only
    assert r["worst"][0]["floor_ratio"] == 3.5 and set(r["per_field"]) >= {"T2", "QSNOW"}


def test_fidelity_verdict_twin_persists_outputs_when_prefix_given(monkeypatch):
    got = {}
    monkeypatch.setattr(adv.chaos_gate, "run_gate",
                        lambda *a, **k: got.setdefault("prefix", a[6]) and _payload("WITHIN-FLOOR"))
    adv.fidelity_verdict("r", "c", twin="t", out_prefix="/x/out")
    assert got["prefix"] == "/x/out"


def test_fallback_without_twin_is_labelled_provisional(monkeypatch):
    monkeypatch.setattr(adv.gate, "compare", lambda *a, **k: [_res("T2", 0.001)])
    monkeypatch.setattr(adv.chaos_gate, "run_gate", lambda *a, **k: pytest.fail("chaos gate used"))
    r = adv.fidelity_verdict("r", "c")
    assert r["gate"] == "config-gate" and r["label"] is None
    assert r["tolerance_source"] == "provisional-default"


def _chaos(label, fr=1.0):
    return {"passed": label in ("WITHIN-FLOOR", "SMALL-DRIFT"), "label": label, "gate": "chaos-floor",
            "worst": [{"name": "QSNOW", "rel_rms": 0.04, "floor_ratio": fr}]}


def test_advice_table_maps_chaos_floor_labels():
    cands = [{"id": i, "label": i, "rationale": i} for i in "ABCDE"]
    speed = {"A": _speed(8.0), "B": _speed(8.0), "C": _speed(8.0), "D": _speed(2.0), "E": _speed(8.0)}
    fid = {"A": _chaos("WITHIN-FLOOR"), "B": _chaos("SMALL-DRIFT"), "C": _chaos("LARGE-DRIFT", 9),
           "D": _chaos("WITHIN-FLOOR"), "E": _chaos("INVALID")}
    rows = {r["id"]: r for r in adv.advice_table(cands, speed, fid)}
    assert (rows["A"]["recommendation"], rows["A"]["fidelity"]) == ("consider", "WITHIN-FLOOR")
    assert (rows["B"]["recommendation"], rows["B"]["fidelity"]) == ("consider", "SMALL-DRIFT")
    assert rows["C"]["recommendation"] == "changes results beyond tolerance"
    assert rows["C"]["fidelity"] == "LARGE-DRIFT" and rows["C"]["fidelity_state"] == "fail"
    assert rows["D"]["recommendation"] == "not worth it"
    assert rows["E"]["recommendation"] == "fidelity not checked" and rows["E"]["fidelity_state"] == "invalid"
    assert all(r["note"] == adv.ADVICE_ONLY and r["fidelity_gate"] == "chaos-floor" for r in rows.values())


# ---------------------------------------------------------------------------- 0059 runner wiring
def test_restart_compatible_flag_and_slot_filling(tmp_path):
    assert adv.restart_compatible({"radt": "6, 6", "cudt": "0"}) is True
    assert adv.restart_compatible({"RA_LW_PHYSICS": "4"}) is False
    assert adv.restart_compatible({"mp_physics": "8, 8"}) is False
    p = _nl(tmp_path, dx=3000, dt=18, radt=3)
    got = adv.plan_candidates(p, max_candidates=1, max_estimate_only=2)
    flags = [c["restart_compatible"] for c in got]
    assert flags[0] is True                                   # measured slot -> compatible first
    assert flags.count(True) <= 1 and flags.count(False) <= 2
    assert all(c["restart_compatible"] is adv.restart_compatible(c["edits"]) for c in got)


def test_advice_table_marks_scheme_swaps_estimate_only():
    cands = [{"id": "R", "label": "radt", "rationale": "r", "restart_compatible": True,
              "prior_gain_pct": 6.0},
             {"id": "CU", "label": "cu off", "rationale": "r", "restart_compatible": False,
              "prior_gain_pct": 12.0}]
    rows = {r["id"]: r for r in adv.advice_table(cands, {"R": _speed(8.0)}, {})}
    assert rows["CU"]["recommendation"] == adv.ESTIMATE_ONLY
    assert rows["CU"]["measured"] is False and rows["CU"]["fidelity"] == "not run"
    assert rows["CU"]["prior_saving_pct"] == 12.0 and rows["CU"]["predicted_saving_pct"] is None
    assert rows["R"]["measured"] is True and rows["R"]["recommendation"] == "fidelity not checked"


def test_gate_judges_prognostic_fields_and_reports_radiation_snapshots(monkeypatch):
    # matthew radt-x2: OLR (floor_ratio 29) and TSK (54) are radiation snapshots, not a result change
    rows = {"T2": _row(1.0, 0.01), "U10": _row(0.8, 0.02), "OLR": _row(29.0, 0.4),
            "TSK": _row(54.0, 0.5)}
    monkeypatch.setattr(adv.chaos_gate, "run_gate", lambda *a, **k: _payload("LARGE-DRIFT", rows))
    monkeypatch.setattr(adv.chaos_gate, "decide_label",
                        lambda r, thr: ("WITHIN-FLOOR", "judged only %s" % sorted(r)))
    r = adv.fidelity_verdict("r", "c", twin="t", judged_fields=adv.PROGNOSTIC_FIELDS)
    assert r["label"] == "WITHIN-FLOOR" and r["passed"] is True
    assert set(r["radiation_snapshot_not_judged"]) == {"OLR", "TSK"}
    assert [w["name"] for w in r["worst"]] == ["T2", "U10"]
    assert r["deciding"] == "judged only ['T2', 'U10']"
    # without judged_fields nothing is filtered (the earlier behaviour)
    r2 = adv.fidelity_verdict("r", "c", twin="t")
    assert r2["label"] == "LARGE-DRIFT" and r2["radiation_snapshot_not_judged"] == {}
    # no prognostic field comparable -> INVALID, not a silent pass
    monkeypatch.setattr(adv.chaos_gate, "run_gate",
                        lambda *a, **k: _payload("WITHIN-FLOOR", {"OLR": _row(1.0, 0.1)}))
    assert adv.fidelity_verdict("r", "c", twin="t",
                                judged_fields=adv.PROGNOSTIC_FIELDS)["label"] == "INVALID"


def test_fallback_gate_gets_a_single_path_not_a_stringified_list(monkeypatch):
    # live SC26-P1: phase 3 had no twin, so the config-gate fallback received the LISTS of window
    # frames and opened "['/home/.../wrfout_d01_...nc', ...]" -> FileNotFoundError
    seen = {}

    def fake_compare(ref, cand, **kw):
        seen.update(ref=ref, cand=cand)
        return [_res("T2", 0.001)]

    monkeypatch.setattr(adv.gate, "compare", fake_compare)
    run = "/home/nkli/wtune-runs/campaign/sc26p1p0_%s_r1_2327/wrfout_d01_2019-01-28_00:%d:00.nc"
    base = [run % ("ANCHOR", 29), run % ("ANCHOR", 39)]
    cand = [run % ("PH1", 29), run % ("PH1", 39)]
    r = adv.fidelity_verdict(base, cand)
    assert seen == {"ref": base[-1], "cand": cand[-1]}           # strings: the END frames
    assert all(isinstance(v, str) and not v.startswith("[") for v in seen.values())
    assert r["gate"] == "config-gate" and r["passed"] is True
