"""Tests for the observation fetch modules, provenance helpers, MET table parsers and the
precip/convection comparison helpers (no network, no MET, no Py-ART)."""
import json
import os
import sys
from argparse import Namespace
from datetime import datetime
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wtune.post import met_gridstat, met_pointstat, report, sources  # noqa: E402
from wtune.post.fetch import mrms, nexrad, stage4, stations  # noqa: E402
from wtune.post.recipes import convection, precip  # noqa: E402


def _args(**kw):
    d = dict(project=None, run_dir=None, start=None, end=None, bbox=None)
    d.update(kw)
    return Namespace(**d)


def test_sources_registry_complete():
    for name, rec in sources.REGISTRY.items():
        assert set(rec) == set(sources.PROVENANCE_KEYS), name
    assert "fetch-era5" in sources.REGISTRY["era5"]["access"]


def test_download_idempotent_and_provenance_merge(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"hello")
    dest = str(tmp_path / "out" / "a.bin")
    sha, dl = sources.download(src.as_uri(), dest)
    assert dl and sha == sources.sha256_file(dest)
    sha2, dl2 = sources.download("file:///nonexistent/never", dest, known_sha=sha)
    assert not dl2 and sha2 == sha                      # existing file with matching sha: no request
    proj = str(tmp_path / "proj")
    prov = dict(sources.REGISTRY["stage4"])
    sources.write_provenance(proj, "stage4", prov, ["u1"], {"a": "1"}, window={"s": 1}, bbox=[0, 1, 2, 3])
    sources.write_provenance(proj, "stage4", prov, ["u2"], {"b": "2"})
    rec = sources.read_provenance(proj, "stage4")
    assert rec["urls"] == ["u1", "u2"] and rec["files"] == {"a": "1", "b": "2"}
    assert rec["window"] == {"s": 1} and rec["fetched_utc"].endswith("Z")
    recs = sources.dataset_records(proj, ["stage4", "era5"])
    assert recs[0]["fetched_utc"] == rec["fetched_utc"]
    assert recs[1]["fetched_utc"].startswith("n/a")


def test_used_sources_from_figures():
    figs = [dict(id="precip_fss", missing=False, obs_sources=[{"name": "stage4"}]),
            dict(id="precip_diff", missing=True, obs_sources=[{"name": "mrms"}]),
            dict(id="syn_z500_diff", missing=False), dict(id="tc_track_map", missing=False)]
    assert sources.used_sources(figs) == ["stage4", "era5", "hurdat2"]


def test_stage4_windows_and_coverage(capsys, tmp_path):
    t0, t1 = datetime(2005, 8, 27, 0), datetime(2005, 8, 29, 0)
    assert len(stage4.valid_times(t0, t1, 1)) == 48
    assert stage4.valid_times(t0, t1, 24) == [datetime(2005, 8, 28, 12)]
    assert stage4.url_for(datetime(2005, 8, 28, 12), 24).endswith(
        "/2005/08/28/stage4/ST4.2005082812.24h.grib")
    out = stage4.run(_args(project=str(tmp_path), start="2017-09-20_00", end="2017-09-21_00",
                           bbox="17,19,-68,-64", accum="1,6,24", jobs=1))
    assert out == [] and "outside the Stage IV CONUS" in capsys.readouterr().out


def test_mrms_coverage_boundary(capsys, tmp_path):
    assert mrms.pick_region((17, 19, -68, -64)) == "CARIB"
    assert mrms.pick_region((25, 31, -95, -85)) == "CONUS"
    assert mrms.pick_region((0, 3, 0, 3)) is None
    out = mrms.run(_args(project=str(tmp_path), start="2005-08-27_00", end="2005-08-29_00",
                         bbox="24,32,-92,-84", products="refl,qpe", step_min=60, region=None))
    assert out == [] and "starts on 2020-10-14" in capsys.readouterr().out
    keys = ["CONUS/X/20240905/MRMS_X_20240905-000040.grib2.gz", "CONUS/X/20240905/MRMS_X_20240905-000240.grib2.gz",
            "CONUS/X/20240905/MRMS_X_20240905-010039.grib2.gz"]
    from datetime import timedelta
    got = mrms.choose_keys(keys, [datetime(2024, 9, 5, 0), datetime(2024, 9, 5, 1), datetime(2024, 9, 5, 2)],
                           timedelta(minutes=5))
    assert got == [keys[0], keys[2]]


def test_nexrad_helpers():
    text = ("NCDCID   ICAO WBAN  NAME  COUNTRY ST COUNTY LAT LON ELEV UTC STNTYPE\n"
            "30001794 KLIX 12930 SLIDELL             UNITED STATES        LA ST TAMMANY     30.336667  -89.825    54    -6    NEXRAD\n"
            "30001795 KMOB 13894 MOBILE              UNITED STATES        AL MOBILE         30.679444  -88.240    208   -6    NEXRAD\n"
            "30001796 KABR 14929 ABERDEEN            UNITED STATES        SD BROWN          45.455833  -98.413    1383  -6    NEXRAD\n")
    sites = nexrad.parse_homr(text)
    assert [s[0] for s in sites] == ["KLIX", "KMOB", "KABR"]
    assert nexrad.pick_radars(sites, (28, 31, -91, -87), 2) == ["KLIX", "KMOB"]
    keys = ["2005/08/28/KLIX/KLIX20050828_120514.gz", "2005/08/28/KLIX/KLIX20050828_115000.gz",
            "2024/01/01/KLIX/KLIX20240101_000000_V06_MDM"]
    from datetime import timedelta
    assert nexrad.choose(keys, [datetime(2005, 8, 28, 12)], timedelta(minutes=8)) == [keys[0]]
    assert nexrad.choose(keys, [datetime(2005, 8, 28, 11)], timedelta(minutes=8)) == []


def test_stations_select_and_isd_lite():
    hist = ('"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
            '"722350","03940","JACKSON INTL","US","MS","KJAN","+32.320","-090.078","+0101.2","19730101","20261231"\n'
            '"999999","00001","BUOY","US","","","+28.000","-088.000","+0000.0","19900101","20261231"\n'
            '"111111","22222","OLD","US","","","+30.000","-089.000","+0001.0","19500101","19600101"\n')
    rows = stations.parse_history(hist)
    got = stations.select(rows, (28, 33, -92, -85), datetime(2005, 8, 27), datetime(2005, 8, 29), 10)
    assert [stations.station_id(r) for r in got] == ["722350-03940"]
    text = "2005 08 28 12   250   240 10132   100    50     0    -1 -9999\n2005 08 28 13 -9999 -9999 -9999 -9999 -9999 0 -9999 -9999\n"
    recs = met_pointstat.parse_isd_lite(text)
    lines = met_pointstat.ascii2nc_lines(dict(id="722350-03940", lat="32.32", lon="-90.078", elev_m="101.2"),
                                         recs, datetime(2005, 8, 28, 0), datetime(2005, 8, 29, 0))
    assert len(lines) == 4                               # T, Td, MSLP, WIND; the all-missing hour is dropped
    assert all(len(l.split()) == 11 for l in lines)
    tmp = [l for l in lines if " TMP " in l][0]
    assert abs(float(tmp.split()[-1]) - 298.15) < 1e-6   # 25.0 degC in K
    wind = [l for l in lines if " WIND " in l][0]
    assert wind.split()[-1] == "5.000"


_CTS = ("VERSION MODEL DESC FCST_LEAD FCST_VALID_BEG FCST_VALID_END OBS_LEAD OBS_VALID_BEG OBS_VALID_END "
        "FCST_VAR FCST_UNITS FCST_LEV OBS_VAR OBS_UNITS OBS_LEV OBTYPE VX_MASK INTERP_MTHD INTERP_PNTS "
        "FCST_THRESH OBS_THRESH COV_THRESH ALPHA LINE_TYPE TOTAL BASER BASER_NCL BASER_NCU BASER_BCL BASER_BCU "
        "FMEAN FMEAN_NCL FMEAN_NCU FMEAN_BCL FMEAN_BCU ACC ACC_NCL ACC_NCU ACC_BCL ACC_BCU FBIAS FBIAS_BCL FBIAS_BCU "
        "PODY PODY_NCL PODY_NCU PODY_BCL PODY_BCU PODN PODN_NCL PODN_NCU PODN_BCL PODN_BCU POFD POFD_NCL POFD_NCU POFD_BCL POFD_BCU "
        "FAR FAR_NCL FAR_NCU FAR_BCL FAR_BCU CSI CSI_NCL CSI_NCU CSI_BCL CSI_BCU\n")


def _cts_row(th, total, fbias, pody, far, csi):
    vals = dict(FCST_THRESH=">=" + th, OBS_THRESH=">=" + th, LINE_TYPE="CTS", TOTAL=str(total),
                FBIAS=fbias, PODY=pody, FAR=far, CSI=csi)
    return " ".join(vals.get(h, "NA") for h in _CTS.split()) + "\n"


def test_parse_cts_and_nbrcnt(tmp_path):
    p = tmp_path / "cts.txt"
    p.write_text(_CTS + _cts_row("25.0", 1000, "1.5", "0.4", "0.6", "0.25"))
    rows = met_gridstat.parse_cts_table(str(p))
    assert rows[0]["thresh"] == 25.0 and rows[0]["pod"] == 0.4 and rows[0]["csi"] == 0.25
    assert rows[0]["fbias"] == 1.5 and rows[0]["total"] == 1000
    nb = tmp_path / "nbr.txt"
    nb.write_text("VERSION FCST_THRESH INTERP_PNTS LINE_TYPE FBS FSS F_RATE O_RATE\n"
                  "V12 >=10.0 9 NBRCNT 0.1 0.6 0.3 0.2\nV12 >=10.0 1 NBRCNT 0.2 0.4 0.3 0.2\n")
    r = met_gridstat.parse_nbrcnt_table(str(nb))
    assert [(x["thresh"], x["width"], x["fss"]) for x in r] == [(10.0, 1, 0.4), (10.0, 3, 0.6)]


def test_pooled_scores():
    ctc = [[dict(thresh=20.0, fy_oy=10, fy_on=10, fn_oy=10, fn_on=70)],
           [dict(thresh=20.0, fy_oy=30, fy_on=0, fn_oy=10, fn_on=60)]]
    p = convection._pool_counts(ctc)[0]
    assert p["pod"] == 40 / 60 and p["far"] == 10 / 50 and abs(p["csi"] - 40 / 70) < 1e-12
    assert p["fbias"] == 50 / 60 and p["total"] == 200
    nb = [[dict(thresh=20.0, width=1, fss=0.5, fbs=0.1, o_rate=0.1, f_rate=0.1)],
          [dict(thresh=20.0, width=1, fss=0.5, fbs=0.3, o_rate=0.3, f_rate=0.3)]]
    f = convection._pool_fss(nb)[0]
    assert abs(f["fss"] - 0.5) < 1e-12 and abs(f["fbs"] - 0.4) < 1e-12


def test_precip_pick_window_and_reasons():
    from datetime import timedelta
    base = datetime(2005, 8, 27)
    frames = [base + timedelta(hours=h) for h in range(0, 49)]
    idx = {(24, datetime(2005, 8, 28, 12)): "a", (6, datetime(2005, 8, 28, 12)): "b", (1, datetime(2005, 8, 29, 0)): "c"}
    assert precip.pick_window(frames, 6, idx) == (24, datetime(2005, 8, 28, 12), "a")
    assert precip.pick_window(frames, 18, idx)[0] == 6          # 24 h window would start before spin-up ends
    assert precip.pick_window(frames[:5], 6, idx) is None
    # around the rain peak: the window centred nearest it among those holding it, not the
    # latest (Stage IV used to ignore the peak; CMORPH already did this)
    two = {(24, datetime(2005, 8, 28, 12)): "a", (24, datetime(2005, 8, 29, 0)): "z",
           (24, datetime(2005, 8, 28, 6)): "e"}
    assert precip.pick_window(frames, 6, two)[1:] == (datetime(2005, 8, 29, 0), "z")
    # 27 14Z is in 27 06Z-28 06Z (centre 4 h away) and 27 12Z-28 12Z (10 h away)
    assert precip.pick_window(frames, 6, two, around=datetime(2005, 8, 27, 14))[2] == "e"
    assert precip.pick_window(frames, 6, two, around=datetime(2005, 8, 28, 4))[2] == "a"
    late = datetime(2005, 8, 29, 1)                 # in no window: the latest, as before
    assert precip.pick_window(frames, 6, two, around=late)[2] == "z"
    r = precip.stage4_missing_reason("/p", "/r", frames, (17, 19, -68, -64), False)
    assert "does not exist" in r and "coverage" in r
    r = precip.stage4_missing_reason("/p", "/r", frames, (25, 30, -90, -85), False)
    assert "wtune post fetch stage4 --project /p --run-dir /r" in r


def test_radar_missing_reason_pre_and_post_mrms():
    old = convection.radar_missing_reason("/p", "/r", [datetime(2005, 8, 27)], None)
    assert "fetch nexrad" in old and "MRMS does not exist before 2020-10-14" in old
    new = convection.radar_missing_reason("/p", "/r", [datetime(2024, 9, 5)], None)
    assert "fetch mrms" in new


def test_mcape_undefined_mask():
    import numpy as np
    a = np.array([0.0, 100.0, np.nan, 9.969e36])
    assert list(convection._undefined(a)) == [False, False, True, True]


def test_report_notes_sources_and_figure_note(tmp_path):
    figs = [dict(id="precip_fss", missing=False, path=str(tmp_path / "f.png"), run="m1", domain="d02",
                 valid_time="t", note="figure-level caveat",
                 obs_sources=sources.source_tags(None, ["stage4"]), tool="MET grid_stat",
                 window="w"),
            dict(id="precip_diff", missing=True, dataset="Stage IV", reason="not fetched", run=None)]
    (tmp_path / "f.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    case_info = dict(types=["precip"], evidence={"precip": "e"}, run_length_h=48.0, d01_extent_km=(1.0, 1.0))
    cat = json.load(open(os.path.join(os.path.dirname(report.__file__), "recipes", "catalog.json")))
    path = report.build_recipe_report(str(tmp_path), case_info, figs, ["/r"], ["m1"], 6, cat,
                                      notes=[dict(source="tc", level="warn", text="track note text")])
    html_text = open(path).read()
    assert "track note text" in html_text and "warnbox" in html_text
    assert "figure-level caveat" in html_text
    assert "資料來源與授權" in html_text and "doi:10.7289" not in html_text and "Lin, Y." in html_text
    assert "missing: precip_diff" in html_text
    assert html_text.count("missing:") == 1               # notes are not counted as missing observations


def test_station_map_has_a_panel_when_only_dew_point_paired(tmp_path, monkeypatch):
    """bb1f14f review: the map drew T2 / WSPD10 / MSLP only; with dew point the only paired
    variable, np.concatenate([]) raised and the whole station verification failed."""
    from datetime import datetime
    from wtune.post import met_pointstat, obs_figs
    t = datetime(2019, 2, 13, 21)
    d = tmp_path / ("ps_%s" % t.strftime(met_pointstat._TAG_FMT))
    d.mkdir()
    (d / "x_cnt.txt").write_text("FCST_VAR LINE_TYPE ME RMSE TOTAL\nTD2 CNT 5.8 5.8 1\n")
    (d / "x_mpr.txt").write_text("FCST_VAR LINE_TYPE OBS_SID OBS_LAT OBS_LON FCST OBS\n"
                                 "TD2 MPR 724646-03028 37.283 -102.614 262.0 256.2\n")
    seen = {}
    monkeypatch.setattr(obs_figs, "bias_rmse_series", lambda *a, **k: None)
    monkeypatch.setattr(obs_figs, "station_bias_maps",
                        lambda panels, title, out: seen.setdefault("panels", panels))
    met_pointstat.summarize(str(tmp_path), [(t, None)], str(tmp_path), str(tmp_path))
    assert [p["title"].split()[0] for p in seen["panels"]] == ["Td2"]


def test_station_bias_maps_follow_the_stations_box(tmp_path):
    # a fixed 4.8n x 6.4 drew short wide panels with ~5 pt text in blank margins (haitang2017)
    pytest.importorskip("cartopy")
    from PIL import Image
    from wtune.post import obs_figs
    assert obs_figs.station_panel_height_in((114.0, 124.0, 20.0, 27.0)) == pytest.approx(3.8 * 0.7)
    assert obs_figs.station_panel_height_in((0.0, 1.0, 0.0, 10.0)) == 6.0
    panel = dict(lat=[22.7, 25.1], lon=[116.7, 121.8], bias=[1.0, -1.0], title="T2", vmax=3)
    out = str(tmp_path / "m.png")
    obs_figs.station_bias_maps([panel] * 3, "s", out)
    w, h = Image.open(out).size
    assert 2.5 < w / h < 4.0          # one row of three maps, no tall blank band
