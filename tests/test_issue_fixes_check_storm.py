"""Regression tests for four `wtune check` / storm-lookup fixes (2026-10-02):

1. a WRF run/ link farm (ndown.exe, tc.exe, ... symlinked by `wtune run`) is not an ndown run;
   ndown is reported only on wrfndi_d0* data.
2. C3 on a case dir (no wrf.exe): tables missing there but present in --wrf-run-dir are INFO
   ("linked at run time"), not FATAL; the per-file WARN flood becomes one INFO summary.
3. --storm by NAME / NAME+YEAR, filtered by the case window.
4. `wtune project set` and advise taking storm/hurdat2 from project.yaml.
"""
import os
from datetime import datetime

import pytest
import yaml

from wtune import advise as adv
from wtune import predict as wp
from wtune import preflight as pf
from wtune import project as proj
from wtune.post import tc

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "post")
M1_NAMELIST = os.path.join(FIXTURES, "m1-namelist.input")   # Katrina m1: 2005-08-27_00, 48 h

_PAD = ", -999" * 13
HURDAT2_TEXT = (
    "AL211981,            KATRINA,      3,\n"
    "19811102, 1200,  , TD, 17.0N,  80.0W,  25, 1006" + _PAD + "\n"
    "19811103, 0000,  , TS, 18.0N,  80.5W,  45,  999" + _PAD + "\n"
    "19811104, 0000,  , HU, 19.5N,  79.0W,  75,  980" + _PAD + "\n"
    "AL151999,            KATRINA,      3,\n"
    "19991028, 1800,  , TD, 12.5N,  81.0W,  25, 1005" + _PAD + "\n"
    "19991029, 1200,  , TS, 13.5N,  83.0W,  35, 1000" + _PAD + "\n"
    "19991030, 1200,  , TD, 15.0N,  86.0W,  25, 1004" + _PAD + "\n"
    "AL122005,            KATRINA,      4,\n"
    "20050823, 1800,  , TD, 23.1N,  75.1W,  30, 1008" + _PAD + "\n"
    "20050827, 0000,  , HU, 24.4N,  84.4W,  95,  959" + _PAD + "\n"
    "20050828, 1800,  , HU, 26.3N,  88.6W, 150,  902" + _PAD + "\n"
    "20050829, 1110, L, HU, 29.3N,  89.6W, 110,  920" + _PAD + "\n"
    "AL132005,               RITA,      2,\n"
    "20050918, 0000,  , TD, 22.2N,  70.0W,  30, 1006" + _PAD + "\n"
    "20050919, 0000,  , TS, 23.0N,  74.0W,  40,  999" + _PAD + "\n"
)


@pytest.fixture
def hurdat2(tmp_path):
    p = tmp_path / "hurdat2.txt"
    p.write_text(HURDAT2_TEXT)
    return str(p)


@pytest.fixture(autouse=True)
def _no_ambient_project(monkeypatch, tmp_path):
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.delenv("WTUNE_HURDAT2", raising=False)
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ ISSUE 1: ndown false positive
def _wrf_tree(tmp_path, names):
    run = tmp_path / "WRF" / "run"
    run.mkdir(parents=True)
    for n in names:
        (run / n).write_text("x")
    return run


_RRTMG_NL = ("&domains\n max_dom = 1,\n/\n&physics\n ra_lw_physics = 4,\n ra_sw_physics = 4,\n"
             " ghg_input = 1,\n/\n")


def _c3(run_dir, wrf_run_dir=None):
    (run_dir / "namelist.input").write_text(_RRTMG_NL)
    nl = wp.parse_namelist(str(run_dir / "namelist.input"))
    findings = []
    pf.check_c3_physics_files(nl, run_dir, findings, wrf_run_dir=wrf_run_dir)
    return findings


def test_c3_case_dir_missing_everywhere_stays_fatal(tmp_path):
    wrf_run = _wrf_tree(tmp_path, ["RRTMG_LW_DATA"])
    case = tmp_path / "case"
    case.mkdir()
    findings = _c3(case, wrf_run)
    fatal = [f.message for f in findings if f.severity == "FATAL"]
    assert any("RRTMG_SW_DATA does not exist" in m for m in fatal)
    assert not any("RRTMG_LW_DATA" in m for m in fatal)
    assert len(fatal) == 5


def test_bare_name_with_several_storms_lists_candidates(hurdat2):
    with pytest.raises(tc.TCError) as e:
        tc.resolve_storm(hurdat2, "KATRINA")
    msg = str(e.value)
    for sid, year in (("AL211981", 1981), ("AL151999", 1999), ("AL122005", 2005)):
        assert "%s (%d)" % (sid, year) in msg
    assert "ATCF id" in msg and "NAME+YEAR" in msg


def test_window_disambiguates_bare_name(hurdat2, capsys):
    win = (datetime(2005, 8, 27), datetime(2005, 8, 29))
    assert tc.resolve_storm(hurdat2, "katrina", window=win) == "AL122005"
    assert "storm katrina -> AL122005" in capsys.readouterr().err
    with pytest.raises(tc.TCError, match="overlaps the case window"):
        tc.resolve_storm(hurdat2, "KATRINA", window=(datetime(2010, 1, 1), datetime(2010, 1, 2)))


def test_unknown_name_and_unique_name(hurdat2):
    with pytest.raises(tc.TCError, match="no storm named 'WILMA'"):
        tc.resolve_storm(hurdat2, "WILMA")
    assert tc.resolve_storm(hurdat2, "rita") == "AL132005"


def test_parse_hurdat2_full_accepts_name_year_and_id(hurdat2):
    by_name = tc.parse_hurdat2_full(hurdat2, "KATRINA2005")
    by_id = tc.parse_hurdat2_full(hurdat2, "AL122005")
    assert by_name == by_id and len(by_id) == 4 and by_id[1]["pmin_hpa"] == 959
    with pytest.raises(tc.TCError, match="storm id 'AL992005' not found"):
        tc.parse_hurdat2_full(hurdat2, "AL992005")


def _case_dir(tmp_path):
    d = tmp_path / "case"
    d.mkdir()
    (d / "namelist.input").write_text(open(M1_NAMELIST).read())
    return d


# ------------------------------------------------------------------ ISSUE 4: project storm settings
def _project(tmp_path):
    p = proj.init_project(str(tmp_path / "proj"))
    with open(os.path.join(p, proj.PROJECT_YAML)) as fh:
        doc = yaml.safe_load(fh)
    doc["extra_user_key"] = {"keep": 1}
    with open(os.path.join(p, proj.PROJECT_YAML), "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    return p


def test_project_set_errors(tmp_path, capsys):
    p = _project(tmp_path)
    assert proj.main(["set", "--project", p]) == 1
    assert proj.main(["set", "--project", p, "--hurdat2", str(tmp_path / "missing.txt")]) == 1
    assert proj.main(["set", "--storm", "AL122005"]) == 1          # cwd has no project
    assert "no project found" in capsys.readouterr().err


def test_advise_cli_project_flag(tmp_path, hurdat2, monkeypatch):
    p = _project(tmp_path)
    proj.set_project(p, storm="AL122005", hurdat2=hurdat2)
    seen = {}

    def fake_advise(case, *a, **k):
        seen["storm"], seen["hurdat2"] = case.storm, case.hurdat2
        return {"case": case.name, "advice": [], "unavailable": {}}

    monkeypatch.setattr(adv, "advise", fake_advise)
    assert adv.main([str(_case_dir(tmp_path)), "--json", "--project", p]) == 0
    assert seen == {"storm": "AL122005", "hurdat2": hurdat2}
