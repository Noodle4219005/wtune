"""Tests for post/tc.py: HURDAT2/EBTRK observation parsers (pure text), storm-id parsing,
b-deck resolution (NHC download / HURDAT2 fallback, both mocked), and a-deck construction
from a synthetic TempestExtremes track + quadrant-radii rows. model_quadrant_radii() and
run_tc_report()'s full orchestration need real multi-domain wrfout files, TempestExtremes
and (for the MET-dependent figures) MET, and are exercised manually (decision 0056's
report) -- the quadrant math itself is unit-tested against an analytic vortex in
test_post_tc_metrics.py, and the ATCF format in test_post_atcf.py.
"""
import textwrap
from datetime import datetime

import pytest

import wtune.post.tc as tc
from wtune.post import atcf

HURDAT2_TEXT = textwrap.dedent("""\
    AL122005,            KATRINA,     2,
    20050827, 0000,  , TS, 25.4, -74.6,  50, 995,  60,  40,  30,  50,   0,   0,   0,   0,   0,   0,   0,   0,   35
    20050827, 0600,  , HU, 25.7, -75.6,  70, 985,  90,  70,  50,  80,  40,  30,  20,  35,   0,   0,   0,   0,   25
    """)

EBTRK_TEXT = textwrap.dedent("""\
    AL122005 KATRINA        082700 2005  25.4  -74.6   50   995   20    10   1005   300
    AL122005 KATRINA        082706 2005  25.7  -75.6   70   985   15     8   1000   280
    """)


def test_parse_ebtrk_rmw_ignores_other_storms(tmp_path):
    f = tmp_path / "ebtrk.txt"
    f.write_text(EBTRK_TEXT + "AL999999 OTHER          082712 2005  30.0  -80.0   40   1000   10    5   1010   200\n")
    rows = tc.parse_ebtrk_rmw(str(f), "AL122005")
    assert len(rows) == 2


def test_build_obs_without_ebtrk(tmp_path):
    f = tmp_path / "hurdat2.txt"
    f.write_text(HURDAT2_TEXT)
    obs = tc.build_obs(str(f), "AL122005")
    assert len(obs["hurdat2"]) == 2
    assert obs["ebtrk_rmw"] is None


def test_storm_id_parts_rejects_bad_format():
    with pytest.raises(tc.TCError):
        tc.storm_id_parts("KATRINA")


def test_resolve_bdeck_prefers_local_file(tmp_path):
    local = tmp_path / "mybdeck.dat"
    local.write_text("AL, 12, 2005082700,   , BEST,   0, 246N,  833W,  90, 959, HU,   0,\n")
    resolved = tc.resolve_bdeck("AL122005", str(tmp_path), bdeck_path=str(local))
    assert resolved == str(local)


def test_resolve_bdeck_missing_local_file_raises(tmp_path):
    with pytest.raises(tc.TCError):
        tc.resolve_bdeck("AL122005", str(tmp_path), bdeck_path=str(tmp_path / "nope.dat"))


def test_resolve_bdeck_falls_back_to_hurdat2(tmp_path, monkeypatch):
    def _boom(*a, **kw):
        raise atcf.ATCFError("offline")
    monkeypatch.setattr(atcf, "fetch_bdeck", _boom)
    f = tmp_path / "hurdat2.txt"
    f.write_text(HURDAT2_TEXT)
    hurdat2_points = tc.parse_hurdat2_full(str(f), "AL122005")

    resolved = tc.resolve_bdeck("AL122005", str(tmp_path), hurdat2_points=hurdat2_points)
    rows = atcf.parse_file(resolved)
    assert rows[0]["tech"] == "BEST"
    assert rows[0]["basin"] == "AL" and rows[0]["cy"] == "12"


def test_build_adeck_writes_atcf_from_chosen_track(tmp_path):
    chosen = [
        dict(time=datetime(2005, 8, 27, 0), lat=24.6, lon=-83.3, pmin_hpa=959, vmax_ms=90 * tc.KT_TO_MS),
        dict(time=datetime(2005, 8, 27, 6), lat=24.4, lon=-84.0, pmin_hpa=950, vmax_ms=95 * tc.KT_TO_MS),
    ]
    quadrant_rows = [
        dict(time="2005-08-27_00:00:00", lead_h=0.0,
             r34_ne_km=90 * tc.NMI_TO_KM, r34_se_km=75 * tc.NMI_TO_KM,
             r34_sw_km=75 * tc.NMI_TO_KM, r34_nw_km=75 * tc.NMI_TO_KM,
             r50_ne_km=0.0, r50_se_km=0.0, r50_sw_km=0.0, r50_nw_km=0.0,
             r64_ne_km=0.0, r64_se_km=0.0, r64_sw_km=0.0, r64_nw_km=0.0),
    ]
    out_path = str(tmp_path / "adeck.dat")
    path = tc.build_adeck(chosen, quadrant_rows, out_path, "AL", "12", tech="ctrl",
                          storm_name="katrina")
    rows = atcf.parse_file(path)
    assert all(r["tech"] == "CTRL" for r in rows)
    # one forecast: every line carries the initialization time, lead grows with valid time
    assert {r["time"] for r in rows} == {datetime(2005, 8, 27, 0)}
    assert sorted({r["tau_h"] for r in rows}) == [0, 6]
    assert rows[0]["vmax_kt"] == 90
    assert rows[0]["rad_kt"] == 34
    assert rows[0]["radii"] == (90, 75, 75, 75)
    # second point has no quadrant row but vmax=95kt reaches all three thresholds -> one
    # line per threshold, each with an all-zero radii quadruplet (no data to fill it)
    second = [r for r in rows if r["tau_h"] == 6]
    assert len(second) == 3
    assert sorted(r["rad_kt"] for r in second) == [34, 50, 64]
    assert all(r["vmax_kt"] == 95 and r["radii"] == (0, 0, 0, 0) for r in second)


def test_a_bdeck_alone_is_enough_for_the_tc_report(tmp_path, monkeypatch):
    # Haitang WP122017 (2026-10-26): JMA best track as an ATCF b-deck, no HURDAT2 record exists
    # outside the Atlantic/E. Pacific; tc still demanded --hurdat2
    import argparse
    from datetime import datetime
    from wtune.post import atcf, cli, tc
    bdeck = tmp_path / "bwp122017.dat"
    rows = [dict(time=datetime(2017, 7, 31, h), tau_h=0, lat=24.0 + h / 10, lon=119.5,
                 vmax_kt=45, mslp_hpa=985) for h in (0, 6, 6, 12)]
    with open(bdeck, "w") as fh:
        for r in rows:
            fh.write(atcf.format_line("WP", "12", r["time"], "BEST", 0, r["lat"], r["lon"],
                                      r["vmax_kt"], r["mslp_hpa"]) + "\n")
    pts = tc.bdeck_points(str(bdeck))
    assert [p["time"].hour for p in pts] == [0, 6, 12]
    assert pts[0]["pmin_hpa"] == 985 and abs(pts[0]["vmax_ms"] - 45 * tc.KT_TO_MS) < 1e-9
    assert pts[0]["rmw_km"] is None
    seen = {}
    monkeypatch.setattr(tc, "run_tc_report", lambda *a, **k: seen.update(a=a, k=k) or {})
    monkeypatch.setattr(cli, "_print_tc_summary", lambda *a: None)
    monkeypatch.setattr(cli, "_record_manifest", lambda *a, **k: None)
    run = tmp_path / "run"
    run.mkdir()
    (run / "wrfout_d01_2017-07-31_00:00:00").write_bytes(b"x")
    base = dict(project=None, run_dirs=[str(run)], labels=None, out=str(tmp_path / "out"),
                ebtrk=None, track_smooth_km=None, nc_cache=None)
    assert cli.cmd_tc(argparse.Namespace(hurdat2=None, storm="WP122017", bdeck=str(bdeck), **base)) == 0
    assert seen["k"]["bdeck_path"] == str(bdeck) and seen["a"][2] is None
    with pytest.raises(SystemExit, match="--hurdat2 or --bdeck"):
        cli.cmd_tc(argparse.Namespace(hurdat2=None, storm="WP122017", bdeck=None, **base))
