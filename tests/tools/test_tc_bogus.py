"""tc_bogus: HURDAT2 parsing, interpolation, &tc rendering, tc.exe namelist -- offline."""
import datetime as dt
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import tc_bogus as tb  # noqa: E402

# Katrina 2005 (pre-2021: RMW column -999) + a synthetic 2022 storm with RMW + an extratropical
# stage, and one storm near the dateline.
HURDAT = """\
AL122005,            KATRINA,      4,
20050826, 1800,  , HU, 24.9N,  82.6W,  85,  968,   75,   75,   55,   35,   60,   60,   35,   20,   20,   20,   15,   15, -999
20050827, 0000,  , HU, 24.6N,  83.3W,  90,  959,   90,   75,   75,   75,   60,   60,   40,   30,   25,   25,   20,   20, -999
20050827, 0600,  , HU, 24.4N,  84.0W,  95,  950,  130,   90,   90,  130,   60,   60,   45,   60,   35,   30,   30,   35, -999
20050827, 0830, L, HU, 24.3N,  84.3W,  95, -999,  130,   90,   90,  130,   60,   60,   45,   60,   35,   30,   30,   35, -999
AL992022,            TESTSTORM,      2,
20220901, 0000,  , HU, 30.0N,  60.0W, 100,  950,  100,  100,   80,   80,   50,   50,   40,   40,   30,   30,   20,   20,   20
20220901, 0600,  , EX, 31.0N,  59.0W,  60,  980,  100,  100,   80,   80,   50,   50,   40,   40,    0,    0,    0,    0, -999
CP991999,            DATELINE,      2,
19990801, 0000,  , TS, 20.0N, 179.0W,  40, 1000,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -999
19990801, 0600,  , TS, 20.0N, 179.0E,  40, 1000,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -999
"""


@pytest.fixture
def storms():
    return tb.parse_hurdat2(HURDAT.splitlines())


def test_parse(storms):
    assert [s["name"] for s in storms] == ["KATRINA", "TESTSTORM", "DATELINE"]
    k = storms[0]["fixes"][1]
    assert k["time"] == dt.datetime(2005, 8, 27, 0)
    assert (k["lat"], k["lon"], k["vmax_kt"], k["pmin_mb"]) == (24.6, -83.3, 90, 959)
    assert k["r34_ne"] == 90 and k["r64_nw"] == 20 and k["rmw_nmi"] is None
    assert storms[0]["fixes"][3]["record"] == "L" and storms[0]["fixes"][3]["pmin_mb"] is None
    assert storms[1]["fixes"][0]["rmw_nmi"] == 20


def test_exact_and_interpolated(storms):
    f = tb.fix_at(storms[0], dt.datetime(2005, 8, 27, 0))
    assert not f["interpolated"] and f["pmin_mb"] == 959
    f = tb.fix_at(storms[0], dt.datetime(2005, 8, 27, 3))
    assert f["interpolated"]
    assert f["lat"] == pytest.approx(24.5) and f["lon"] == pytest.approx(-83.65)
    assert f["vmax_kt"] == pytest.approx(92.5) and f["pmin_mb"] == pytest.approx(954.5)
    # off-synoptic landfall record brackets 07Z; its missing pressure stays missing
    f = tb.fix_at(storms[0], dt.datetime(2005, 8, 27, 7))
    assert f["pmin_mb"] is None and f["vmax_kt"] == pytest.approx(95)
    assert tb.fix_at(storms[0], dt.datetime(2005, 8, 26, 12)) is None
    assert tb.fix_at(storms[0], dt.datetime(2005, 8, 28, 0)) is None


def test_dateline_interpolation(storms):
    f = tb.fix_at(storms[2], dt.datetime(1999, 8, 1, 3))
    assert abs(abs(f["lon"]) - 180.0) < 1e-9


def test_storms_in_bbox(storms):
    t = dt.datetime(2005, 8, 27, 0)
    assert [f["name"] for f in tb.storms_at(storms, t, (15, 35, -100, -70))] == ["KATRINA"]
    assert tb.storms_at(storms, t, (30, 45, -100, -70)) == []
    assert len(tb.storms_at(storms, t)) == 1


def test_defaults():
    assert tb.vmax_ratio_for_dx(45000) == pytest.approx(0.75)
    assert tb.vmax_ratio_for_dx(15000) == pytest.approx(0.9)
    assert tb.vmax_ratio_for_dx(100000) == pytest.approx(0.75)
    assert tb.vmax_ratio_for_dx(12000) == pytest.approx(0.915)
    assert tb.vmax_ratio_for_dx(1000) <= 1.0
    # Willoughby et al. (2006): 46.4 km at vmax=0, lat=0
    assert tb.default_rmax_m(0.0, 0.0) == pytest.approx(46400.0)
    assert tb.default_rmax_m(46.3, 24.6) == pytest.approx(34300, rel=0.01)


def test_render_block_katrina(storms):
    fixes = tb.storms_at(storms, dt.datetime(2005, 8, 27, 0), (15, 35, -100, -70))
    blk = tb.render_tc_block(fixes, dx_m=12000)
    assert " insert_bogus_storm = .true." in blk and " num_storm = 1," in blk
    assert " latc_loc = 24.60," in blk and " lonc_loc = -83.30," in blk
    assert " vmax_meters_per_second = 46.30," in blk          # 90 kt
    assert " vmax_ratio = 0.915," in blk
    assert "Willoughby" in blk
    rmax = float(blk.split(" rmax = ")[1].split(",")[0])
    assert rmax == pytest.approx(tb.default_rmax_m(90 * tb.KT_TO_MS, 24.6), abs=1)


def test_render_block_rmw_and_filters(storms):
    f = tb.fix_at(storms[1], dt.datetime(2022, 9, 1, 0))
    blk = tb.render_tc_block([f], vmax_ratio=1.0)
    assert " rmax = 37040," in blk and "best-track RMW" in blk      # 20 nmi
    ex = tb.fix_at(storms[1], dt.datetime(2022, 9, 1, 6))
    with pytest.raises(tb.TCBogusError):
        tb.render_tc_block([ex], vmax_ratio=1.0)                    # extratropical only
    with pytest.raises(tb.TCBogusError):
        tb.render_tc_block([f])                                     # no dx, no ratio
    blk = tb.render_tc_block([f], vmax_ratio=1.0, rmax_m=50000)
    assert " rmax = 50000," in blk and "override" in blk


CASE_NL = """&time_control
 run_days=0, run_hours=48, run_minutes=0, run_seconds=0,
 start_year=2019, 2019, start_month=08, 08, start_day=30, 30,
 start_hour=12, 12, start_minute=00, 00, start_second=00, 00,
 end_year=2019, 2019, end_month=09, 09, end_day=01, 01,
 end_hour=12, 12, end_minute=00, 00, end_second=00, 00,
 interval_seconds=10800,
/
&domains
 max_dom = 2,
 e_we = 100, 61,
 p_top_requested = 5000,
/
&tc
 insert_bogus_storm = .false.
/
&physics
 mp_physics = 8,
/
"""


def test_build_tc_namelist():
    blk = "&tc\n insert_bogus_storm = .true.\n num_storm = 1,\n/\n"
    out = tb.build_tc_namelist(CASE_NL, blk, dt.datetime(2019, 9, 1, 0))
    assert out.count("&tc") == 1 and ".false." not in out.split("&tc")[1]
    assert "&physics" in out and "mp_physics = 8" in out
    import coldstart_sample as cs
    assert cs.get_first(out, "max_dom") == "1"
    for k, v in (("start_year", "2019"), ("start_month", "09"), ("start_day", "01"),
                 ("start_hour", "00"), ("end_day", "01"), ("end_hour", "00"), ("run_hours", "0")):
        assert cs.get_first(out, k) == v, k
        assert cs.count_key(out, k) == 1, k


def test_met_em_date():
    assert tb.met_em_date("/x/met_em.d01.2005-08-27_00:00:00.nc") == (1, dt.datetime(2005, 8, 27))
    with pytest.raises(tb.TCBogusError):
        tb.met_em_date("/x/wrfinput_d01")


def _write_met(path, val, extra=True, title="OUTPUT FROM METGRID V4.6.0"):
    import numpy as np
    nc4 = pytest.importorskip("netCDF4")
    with nc4.Dataset(path, "w") as nc:
        nc.createDimension("Time", None)
        nc.createDimension("z", 3)
        nc.createDimension("y", 4)
        nc.createDimension("x", 5)
        nc.TITLE = title
        if extra:
            nc.FLAG_METGRID = 1
            nc.createVariable("LANDSEA", "f4", ("Time", "y", "x"))[:] = np.ones((1, 4, 5))
        for k in tb.TC_FIELDS:
            dims = ("Time", "y", "x") if k in ("PMSL", "PSFC") else ("Time", "z", "y", "x")
            v = nc.createVariable(k, "f4", dims)
            v[:] = np.full(v.shape if v.shape[0] else (1,) + v.shape[1:], val, np.float32)
        nc.createVariable("HGT_M", "f4", ("Time", "y", "x"))[:] = np.full((1, 4, 5), 7.0)


def test_apply_tc_with_fake_exe(tmp_path):
    """apply_tc's plumbing against a stand-in tc.exe: output keeps the metgrid file's attributes
    and extra variables, and takes only TC_FIELDS from tc.exe's output."""
    nc4 = pytest.importorskip("netCDF4")
    met = tmp_path / "met_em.d01.2005-08-27_00:00:00.nc"
    _write_met(str(met), 1.0)
    tcout = tmp_path / "tcout.nc"
    _write_met(str(tcout), 2.0, extra=False, title="OUTPUT FROM TC BOGUS")
    fake = tmp_path / "tc.exe"
    fake.write_text("#!/bin/sh\ncp %s auxinput1_d01_2005-08-27_00:00:00.nc\n"
                    "echo '         Bogus storm number   1 completed.' > rsl.out.0000\n" % tcout)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    blk = "! note\n&tc\n insert_bogus_storm = .true.\n num_storm = 1,\n/\n"
    out = tb.apply_tc(str(met), CASE_NL, blk, str(tmp_path / "out"), str(fake))
    assert os.path.basename(out) == met.name
    with nc4.Dataset(out) as nc:
        assert nc.TITLE == "OUTPUT FROM METGRID V4.6.0" and nc.FLAG_METGRID == 1
        assert "num_storm = 1" in nc.TC_BOGUS and "note" not in nc.TC_BOGUS
        assert float(nc["LANDSEA"][:].min()) == 1.0 and float(nc["HGT_M"][:].min()) == 7.0
        for k in tb.TC_FIELDS:
            assert float(nc[k][:].min()) == 2.0, k
    with nc4.Dataset(str(met)) as nc:                      # input untouched
        assert float(nc["UU"][:].max()) == 1.0
    nl = (tmp_path / "out" / "tc_work" / "namelist.input").read_text()
    assert "num_storm = 1" in nl
    bad = tmp_path / "bad.exe"
    bad.write_text("#!/bin/sh\nexit 0\n")
    bad.chmod(bad.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(tb.TCBogusError):
        tb.apply_tc(str(met), CASE_NL, blk, str(tmp_path / "out2"), str(bad))


def test_cli_plan(tmp_path, capsys):
    p = tmp_path / "h.txt"
    p.write_text(HURDAT)
    rc = tb.main(["plan", "--hurdat2", str(p), "--time", "2005-08-27_03", "--bbox",
                  "15,35,-100,-70", "--dx-m", "12000", "--out", str(tmp_path / "tc.nml")])
    out = capsys.readouterr().out
    assert rc == 0 and "KATRINA" in out and "(interpolated)" in out and "&tc" in out
    assert (tmp_path / "tc.nml").read_text().count("&tc") == 1
    assert tb.main(["plan", "--hurdat2", str(p), "--time", "2005-08-27_03", "--bbox",
                    "40,50,-100,-70", "--dx-m", "12000"]) == 1
