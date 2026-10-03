"""Golden snapshots of tools/case_rebuild.complete_namelist, namelist_fields_from_wrfinput_facts
and render_rebuild_sbatch, pinned before the round-2 readability refactor split them. Fixtures
under tests/fixtures/golden/case_rebuild/ were written by the UNCHANGED functions.

Regenerate (only when a behaviour change is intended) with:
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_case_rebuild.py
"""
import copy
import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import case_rebuild as cr  # noqa: E402
import coldstart_sample as cs  # noqa: E402
import test_case_rebuild as tcr  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fixtures", "golden", "case_rebuild")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
_TOOL_PATH = os.path.abspath(cr.__file__)


def _check_golden(name, text):
    path = os.path.join(GOLDEN_DIR, name)
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    with open(path) as fh:
        expected = fh.read()
    assert text == expected, "result differs from golden %s" % path


def _json_golden(name, obj):
    _check_golden(name + ".json", json.dumps(obj, indent=1, default=repr) + "\n")


# ---------------------------------------------------------------------------
# complete_namelist
# ---------------------------------------------------------------------------
_NESTED_NL = """\
&time_control
 run_hours = 6,
 history_interval_m = 30, 30,
 start_year = 2019, 2019,
/
&domains
 max_dom = 2,
 e_we = 100, 61,
 e_sn = 100, 61,
 e_vert = 40, 40,
 dx = 9000, 3000,
 parent_grid_ratio = 1, 3,
/
&physics
 physics_suite = 'CONUS',
 mp_physics = -1, -1,
 ra_lw_physics = -1, -1,
 cu_physics = 6, 0,
/
&bdy_control
 spec_bdy_width = 5,
/
"""

_NESTED_FACTS = dict(
    max_dom=2, num_metgrid_levels="34", num_metgrid_soil_levels=None, num_land_cat=None,
    interval_seconds=10800,
    start_dt=datetime.datetime(2019, 11, 26, 12, 0, 0),
    end_dt=datetime.datetime(2019, 11, 26, 19, 30, 15),
    dx_km=9.0, parent_grid_ratio=[1, 3],
)

COMPLETE_CASES = {
    "base": (tcr.BASE_NL_TEXT, tcr.BASE_FACTS, None),
    "base_vertical_grid": (tcr.BASE_NL_TEXT, tcr.BASE_FACTS,
                           {"eta_levels": [1.0, 0.6, 0.3, 0.0], "p_top_pa": 4000.0}),
    "dzstretch_vertical_grid": (
        tcr.TestCompleteNamelistVerticalGridOverride._nl_with_dzstretch(None),
        tcr.BASE_FACTS, {"eta_levels": [1.0, 0.5, 0.0], "p_top_pa": 5000}),
    "nested_suite": (_NESTED_NL, _NESTED_FACTS, None),
    "nested_suite_vertical_grid": (_NESTED_NL, _NESTED_FACTS,
                                   {"eta_levels": [1.0, 0.7, 0.2, 0.0], "p_top_pa": 5000.0}),
    "empty_text": ("", dict(tcr.BASE_FACTS, dx_km=0.1), None),
}


@pytest.mark.parametrize("name", sorted(COMPLETE_CASES))
def test_complete_namelist_golden(name):
    text, facts, vertical_grid = COMPLETE_CASES[name]
    txt, changes = cr.complete_namelist(text, facts, vertical_grid=vertical_grid)
    _check_golden("complete_%s.txt" % name, txt)
    _json_golden("complete_%s_changes" % name, changes)


# ---------------------------------------------------------------------------
# namelist_fields_from_wrfinput_facts
# ---------------------------------------------------------------------------
def _facts(n_dom=1, dt=72.0, times_offset_h=0, drop=(), extra_attrs=None):
    d01 = copy.deepcopy(tcr.TestNamelistFieldsFromWrfinputFactsPure._facts(None)[0])
    d01["attrs"]["DT"] = dt
    for a in drop:
        d01["attrs"].pop(a, None)
    d01["attrs"].update(extra_attrs or {})
    if times_offset_h:
        d01["times"] = [d01["times"][0] + datetime.timedelta(hours=times_offset_h)]
    out = [d01]
    for d in range(2, n_dom + 1):
        nest = {"attrs": dict(d01["attrs"], GRID_ID=d, PARENT_ID=d - 1, I_PARENT_START=30,
                              J_PARENT_START=40, PARENT_GRID_RATIO=3, DX=d01["attrs"]["DX"] / 3,
                              DY=d01["attrs"]["DY"] / 3, MP_PHYSICS=6)}
        for a in drop:
            nest["attrs"].pop(a, None)
        out.append(nest)
    return out


_BDY = [datetime.datetime(2019, 11, 26, 12) + datetime.timedelta(hours=3 * i) for i in range(3)]

FIELDS_CASES = {
    "wrfbdy_times": (dict(), dict(wrfbdy_times=_BDY)),
    "hours_override": (dict(), dict(hours=6.0)),
    "no_wrfbdy_no_hours": (dict(), dict()),
    "fract_dt_times_mismatch": (dict(dt=45.5, times_offset_h=1), dict(hours=12.5)),
    "nested_bdy_width": (dict(n_dom=3), dict(wrfbdy_times=_BDY, wrfbdy_bdy_width=10)),
    "missing_physics_attr": (dict(n_dom=2, drop=("GWD_OPT", "SF_URBAN_PHYSICS")),
                             dict(wrfbdy_times=_BDY[:1])),
}


@pytest.mark.parametrize("name", sorted(FIELDS_CASES))
def test_namelist_fields_from_wrfinput_facts_golden(name):
    fkw, kw = FIELDS_CASES[name]
    groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(_facts(**fkw), **kw)
    _json_golden("fields_%s" % name, {"groups": groups, "provenance": provenance,
                                       "notes": notes})
    _check_golden("fields_%s_namelist.txt" % name, cr.render_namelist_input(groups))


def test_namelist_fields_with_unknown_key_registry_golden():
    registry = cr.load_registry()
    registry = {"groups": {g: [e for e in entries if e.get("name", "").lower() != "radt"]
                           for g, entries in registry["groups"].items()}}
    registry["groups"]["extra_group"] = [{"name": "nio_groups", "per_domain": False}]
    groups, provenance, notes = cr.namelist_fields_from_wrfinput_facts(
        _facts(n_dom=2), hours=3.0, registry=registry)
    _json_golden("fields_custom_registry", {"groups": groups, "provenance": provenance,
                                            "notes": notes})


def test_namelist_fields_errors():
    with pytest.raises(cr.RebuildError) as exc:
        cr.namelist_fields_from_wrfinput_facts([])
    assert str(exc.value) == "namelist-from-wrfinput: no domain facts given"
    with pytest.raises(cr.RebuildError) as exc:
        cr.namelist_fields_from_wrfinput_facts(_facts(drop=("DX",)), hours=1.0)
    assert str(exc.value) == ("namelist-from-wrfinput: a domain's wrfinput is missing global "
                              "attr DX")
    with pytest.raises(cr.RebuildError) as exc:
        cr.namelist_fields_from_wrfinput_facts(_facts(drop=("SIMULATION_START_DATE",)), hours=1.0)
    assert str(exc.value) == "namelist-from-wrfinput: wrfinput_d01 has no SIMULATION_START_DATE"


# ---------------------------------------------------------------------------
# render_rebuild_sbatch
# ---------------------------------------------------------------------------
def _plan(**kw):
    plan = {"nodes": 2, "config": {"ranks": 100, "omp": 2, "nx": 10, "ny": 10, "numtiles": 1},
            "max_dom": 1, "verify_steps": 0, "span_start": "2020-01-01_00:00:00",
            "span_end": "2020-01-01_06:00:00", "interval_seconds": 10800}
    plan.update(kw)
    return plan


SBATCH_CASES = {
    "minimal": (_plan(), {"prefix": "GFS", "vtable": "Vtable.GFS", "forcing_interval_h": 3}, {}),
    "out_dir_verify_nested": (
        _plan(max_dom=3, verify_steps=20, _out_dir="/scratch/rebuild out/case"),
        {"prefix": "FNL", "vtable": "Vtable.GFS_PRMSL", "forcing_interval_h": 6},
        {"estimate_min": 45}),
    "ml_model": (_plan(_out_dir="/scratch/rb", ml_model="/plan/report.json",
                       config={"ranks": 72, "omp": 1, "nx": 8, "ny": 9, "numtiles": 1}, nodes=1),
                 {}, {"site": "scc"}),
    "ml_verify_short": (_plan(ml_model="model.json", verify_steps=5, nodes=3,
                              config={"ranks": 200, "omp": 1, "nx": 10, "ny": 20, "numtiles": 1}),
                        {"prefix": "ERA5", "vtable": "none", "forcing_interval_h": 1},
                        {"estimate_min": 10}),
}


@pytest.fixture
def _pinned_scm(monkeypatch):
    monkeypatch.setattr(cs, "_SCM_PATH", "/opt/wtune/src/tools/state_correction_model.py")


@pytest.mark.parametrize("name", sorted(SBATCH_CASES))
def test_render_rebuild_sbatch_golden(name, _pinned_scm):
    plan, chosen, kw = SBATCH_CASES[name]
    text = cr.render_rebuild_sbatch(copy.deepcopy(plan), chosen, **kw)
    text = text.replace(_TOOL_PATH, "<TOOL>").replace(os.getcwd(), "<CWD>")
    _check_golden("rebuild_sbatch_%s.sh" % name, text)


def test_render_rebuild_sbatch_overpacked_assertion():
    with pytest.raises(AssertionError) as exc:
        cr.render_rebuild_sbatch(_plan(nodes=1), {})
    assert str(exc.value) == "ppn=100 exceeds CORES_PER_NODE=72 (ranks=100 nodes=1)"
