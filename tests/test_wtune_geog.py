"""wtune.geog: synthetic GeoTIFF import, tile decode, TBL copy, list, fetch (fake downloader), G3."""
import io
import json
import os
import re
import sys
import tarfile

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from wtune import geog as g  # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

TBL = """# test table
===============================
name = HGT_M
        priority = 1
        dest_type = continuous
        fill_missing=0.
        interp_option = gmted2010_30s:average_gcell(4.0)+four_pt+average_4pt
        interp_option =       default:average_gcell(4.0)+four_pt+average_4pt
        rel_path = gmted2010_30s:topo_gmted2010_30s/
        rel_path =       default:topo_gmted2010_30s/
===============================
name=LANDUSEF
        priority=1
        rel_path =      modis_15s:modis_landuse_20class_15s/
        rel_path =       default:modis_landuse_20class_30s_with_lakes/
        rel_path =    usgs_30s:landuse_30s/
===============================
"""


def make_tiff(path, arr, lon_ul, lat_ul, sx, sy, nodata=None, point=False):
    im = Image.fromarray(arr.astype(np.float32))
    info = {33550: (sx, sy, 0.0), 33922: (0.0, 0.0, 0.0, lon_ul, lat_ul, 0.0),
            34735: (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 2 if point else 1, 2048, 0, 1, 4326)}
    if nodata is not None:
        info[42113] = str(nodata)
    im.save(path, tiffinfo=info)


@pytest.fixture
def wps(tmp_path):
    d = tmp_path / "WPS" / "geogrid"
    d.mkdir(parents=True)
    (d / "GEOGRID.TBL.ARW").write_text(TBL)
    os.symlink("GEOGRID.TBL.ARW", d / "GEOGRID.TBL")
    return tmp_path / "WPS"


def hill(ny=30, nx=40):
    y, x = np.mgrid[0:ny, 0:nx]
    return (500 + 400 * np.exp(-(((x - 20) / 8.0) ** 2 + ((y - 12) / 6.0) ** 2))).astype(np.float32)


def test_import_roundtrip_and_index(tmp_path, wps):
    a = hill()
    a[0:3, 0:4] = -9999                                   # nodata block, top-left (north-west)
    tif = tmp_path / "dem.tif"
    make_tiff(tif, a, lon_ul=-100.0, lat_ul=40.0, sx=0.01, sy=0.01, nodata=-9999)
    r = g.import_dem([str(tif)], "srtm_test", dest=str(tmp_path / "out"), tile=16, wps_dir=str(wps),
                     geog_path=str(tmp_path / "geog"))
    ds = r["dataset_dir"]
    idx = g.parse_index(os.path.join(ds, "index"))
    for k, v in dict(type="continuous", signed="yes", projection="regular_ll", wordsize="2", tile_x="16",
                     tile_y="16", tile_z="1", known_x="1", known_y="1", units="meters MSL",
                     missing_value="-32768", dx="0.01", dy="0.01").items():
        assert idx[k] == v, k
    # SW pixel centre of an area-registered raster whose NW corner is (-100, 40)
    assert float(idx["known_lat"]) == pytest.approx(40.0 - 29.5 * 0.01)
    assert float(idx["known_lon"]) == pytest.approx(-100.0 + 0.5 * 0.01)
    assert "00001-00016.00001-00016" in os.listdir(ds)
    # decode: row 0 of the first tile is the southernmost row
    wt = g.WpsTiles(ds)
    raw, i0, j0 = wt.assemble()
    assert (i0, j0) == (1, 1)
    first = wt.tile(1, 1)
    assert first[0, 0] == int(round(a[29, 0]))            # bottom-left source pixel
    # nodata -> missing_value, valid pixels within 0.5 m
    dec = raw[:30, :40].astype(float)
    flipped = a[::-1]
    assert (dec[27:30, 0:4] == -32768).all()
    ok = flipped != -9999
    assert np.abs(dec[ok] - flipped[ok]).max() <= 0.5
    v = g.verify_dataset(ds)
    assert v["ok"], v
    assert v["max_abs_diff"] <= 0.5 and v["valid_decoded_as_missing"] == 0
    # coverage bounds are pixel edges
    b = r["bounds"]
    assert (b["west"], b["north"]) == (pytest.approx(-100.0), pytest.approx(40.0))
    assert b["south"] == pytest.approx(39.7) and b["east"] == pytest.approx(-99.6)


def test_verify_detects_corruption(tmp_path, wps):
    tif = tmp_path / "dem.tif"
    make_tiff(tif, hill(), -100.0, 40.0, 0.01, 0.01)
    r = g.import_dem([str(tif)], "dem1", dest=str(tmp_path / "o"), tile=64, wps_dir=str(wps))
    t = os.path.join(r["dataset_dir"], "00001-00064.00001-00064")
    raw = np.fromfile(t, dtype=">i2")
    raw[100] += 7
    raw.astype(">i2").tofile(t)
    assert not g.verify_dataset(r["dataset_dir"])["ok"]


def test_name_collision_and_duplicate_key(tmp_path, wps):
    tif = tmp_path / "d.tif"; make_tiff(tif, hill(), 0.0, 10.0, 0.01, 0.01)
    with pytest.raises(SystemExit):
        g.import_dem([str(tif)], "modis_15s", dest=str(tmp_path / "o"), wps_dir=str(wps))
    with pytest.raises(ValueError):
        g.tbl_copy_text(TBL, "HGT_M", "default", "x/", False)


def test_require_bounds_and_fill_from(tmp_path, wps):
    tif = tmp_path / "d.tif"
    a = hill(); a[5:8, 5:8] = -9999
    make_tiff(tif, a, -100.0, 40.0, 0.01, 0.01, nodata=-9999)
    with pytest.raises(SystemExit):                      # domain wider than the DEM
        g.import_dem([str(tif)], "x1", dest=str(tmp_path / "o"), wps_dir=str(wps),
                     require_bounds=(39.6, 40.0, -100.1, -99.7))
    # base dataset: constant 123 m on a coarser 0.05 deg lattice around the same area
    base = tmp_path / "base"
    make_tiff(tmp_path / "b.tif", np.full((20, 20), 123.0), -100.5, 40.5, 0.05, 0.05)
    rb = g.import_dem([str(tmp_path / "b.tif")], "basedem", dest=str(base), wps_dir=str(wps), tile=8)
    r = g.import_dem([str(tif)], "x2", dest=str(tmp_path / "o2"), wps_dir=str(wps), tile=64,
                     fill_from=rb["dataset_dir"])
    assert r["voids"] == 9 and r["filled"] == 9
    raw, _, _ = g.WpsTiles(r["dataset_dir"]).assemble()
    assert (raw[:30, :40] != -32768).all()
    assert g.verify_dataset(r["dataset_dir"])["ok"]


def test_hgt_mosaic_and_nodata(tmp_path):
    def hgt(path, lat, lon, base):
        n = 5
        a = (base + np.arange(n * n).reshape(n, n)).astype(">i2")
        a[0, 0] = -32768
        a.tofile(path)
    p1, p2 = tmp_path / "N10E020.hgt", tmp_path / "N10E021.hgt"
    hgt(p1, 10, 20, 0); hgt(p2, 10, 21, 0)
    r = g.mosaic([g.read_raster(str(p1)), g.read_raster(str(p2))])
    assert r.shape == (5, 9) and r.dx == pytest.approx(0.25)
    assert r.lat0 == 10 and r.lon0 == 20
    assert np.isnan(r.arr[4, 0]) and not np.isnan(r.arr[4, 4])     # tile 1 void kept; shared column filled from tile 1
    assert r.arr[0, 0] == 20.0                                                      # SW point of tile 1 = last row first col


def test_list_fake_geog(tmp_path, wps):
    geog = tmp_path / "geog"
    for d, dx in (("topo_gmted2010_30s", "0.00833333"), ("modis_landuse_20class_30s_with_lakes", "0.00833333"),
                  ("extra_thing", "0.1")):
        (geog / d).mkdir(parents=True)
        (geog / d / "index").write_text("type=continuous\nprojection=regular_ll\ndx=%s\ndy=%s\n" % (dx, dx))
    r = g.build_list(str(geog), str(wps))
    have = {d["dir"]: d for d in r["present"]}
    assert set(have) == {"topo_gmted2010_30s", "modis_landuse_20class_30s_with_lakes", "extra_thing"}
    assert "HGT_M:gmted2010_30s" in have["topo_gmted2010_30s"]["tbl_uses"]
    assert have["extra_thing"]["tbl_uses"] == []
    miss = {m["dir"]: m for m in r["missing"]}
    assert miss["modis_landuse_20class_15s"]["fetch_keys"] == ["modis_15s"]
    assert miss["modis_landuse_20class_15s"]["fetch_bytes"] == g.CATALOG["modis_15s"]["bytes"]
    assert "landuse_30s" in miss and miss["landuse_30s"]["fetch_keys"] == [] and "no stand-alone" in miss["landuse_30s"]["note"]
    assert "topo_gmted2010_30s" not in miss
    rc = g.main(["geog", "list", "--geog-path", str(geog), "--wps-dir", str(wps)])
    assert rc == 0


def fake_tar(dirname, with_wps_prefix=False):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("index", b"type=categorical\nprojection=regular_ll\ndx=0.00416667\ndy=0.00416667\n"),
                           ("00001-00002.00001-00002", b"\0\0")):
            ti = tarfile.TarInfo(("WPS_GEOG/" if with_wps_prefix else "") + dirname + "/" + name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        ti = tarfile.TarInfo("../evil"); ti.size = 1
        tf.addfile(ti, io.BytesIO(b"x"))
    return buf.getvalue()


def test_fetch_dry_run_confirm_and_extract(tmp_path, wps, monkeypatch, capsys):
    geog = tmp_path / "geog"; geog.mkdir()
    calls = []
    monkeypatch.setattr(g, "_head_size", lambda url, timeout=30: 12345678)

    def fake_dl(url, dest):
        calls.append(url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(fake_tar("modis_landuse_20class_15s", True))
        return "ab" * 32, True
    monkeypatch.setattr(g, "_download", fake_dl)
    base = ["geog", "fetch", "modis_15s", "--geog-path", str(geog), "--wps-dir", str(wps)]
    # no --yes and stdin not a tty -> size shown, nothing downloaded
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert g.main(base) == 2
    out = capsys.readouterr().out
    assert "11.8 MB" in out and "modis_landuse_20class_15s.tar.bz2" in out and not calls
    assert g.main(base + ["--yes"]) == 0
    d = geog / "modis_landuse_20class_15s"
    assert (d / "index").exists() and not (tmp_path / "evil").exists() and not (geog.parent / "evil").exists()
    prov = json.loads((d / "provenance.json").read_text())
    for k in ("dataset", "provider", "url", "license", "citation", "coverage_time", "coverage_space",
              "resolution", "access", "urls", "files", "fetched_utc", "window", "bbox"):
        assert k in prov, k
    assert prov["files"]["modis_landuse_20class_15s.tar.bz2"] == "ab" * 32
    assert "modis_15s+default" in capsys.readouterr().out
    assert len(calls) == 1
    assert g.main(base + ["--yes"]) == 0 and len(calls) == 1            # already present


def test_fetch_unwritable_falls_back_and_writes_abs_tbl(tmp_path, wps, monkeypatch, capsys):
    geog = tmp_path / "ro"; geog.mkdir(); os.chmod(geog, 0o555)
    monkeypatch.setattr(g, "FALLBACK_DEST", str(tmp_path / "extra"))
    monkeypatch.setattr(g, "_head_size", lambda url, timeout=30: 1000)
    monkeypatch.setattr(g, "_download", lambda url, dest: (os.makedirs(os.path.dirname(dest), exist_ok=True),
                                                           open(dest, "wb").write(fake_tar("modis_landuse_20class_15s")),
                                                           ("cd" * 32, True))[2])
    try:
        if os.access(geog, os.W_OK):
            pytest.skip("running as a user that can write read-only dirs")
        rc = g.main(["geog", "fetch", "modis_15s", "--geog-path", str(geog), "--wps-dir", str(wps), "--yes"])
    finally:
        os.chmod(geog, 0o755)
    assert rc == 0
    out = capsys.readouterr().out
    assert "not writable" in out and "abs_path" in out and "symlink" in out
    tbl = (tmp_path / "extra" / "GEOGRID.TBL.d" / "GEOGRID.TBL").read_text()
    assert "abs_path = modis_15s:%s/" % (tmp_path / "extra" / "modis_landuse_20class_15s") in tbl
    assert "rel_path =      modis_15s" not in tbl


def test_fetch_unknown_and_undistributed():
    with pytest.raises(SystemExit):
        g.fetch_plan("nope", "/x")
    with pytest.raises(SystemExit) as e:
        g.fetch_plan("usgs_30s", "/x")
    assert "no stand-alone tarball" in str(e.value)


def test_domain_bounds_namelist_wps(tmp_path):
    (tmp_path / "namelist.wps").write_text(
        "&geogrid\n parent_id = 1, 1,\n parent_grid_ratio = 1, 3,\n i_parent_start = 1, 11,\n j_parent_start = 1, 11,\n"
        " e_we = 41, 31,\n e_sn = 41, 31,\n dx = 3000,\n dy = 3000,\n map_proj = 'mercator',\n"
        " ref_lat = 0.0,\n ref_lon = 100.0,\n truelat1 = 0.0,\n stand_lon = 100.0,\n/\n")
    s, n, w, e = g.domain_bounds(str(tmp_path / "namelist.wps"), domain=1)
    assert (n - s) == pytest.approx(40 * 3000 / 6370000 * 180 / np.pi, rel=1e-3) and (s + n) / 2 == pytest.approx(0, abs=1e-6)
    s2, n2, w2, e2 = g.domain_bounds(str(tmp_path), domain=2)       # a dir with namelist.wps
    assert (e2 - w2) == pytest.approx(30 * 1000 / 6370000 * 180 / np.pi, rel=1e-3)
    assert w < w2 < e2 < e
    assert [t[0] for t in g.srtm_tiles((9.5, 10.5, 99.5, 100.5))] == ["N09E099", "N09E100", "N10E099", "N10E100"]
    assert g.srtm_tiles((-0.5, -0.1, -0.5, -0.1))[0][0] == "S01W001"


def test_g3_no_tune_or_advise_path_reaches_geog():
    """G3 (static): no module outside the explicit CLI dispatch imports wtune.geog, and wtune.geog
    imports nothing from the tune/advise code."""
    importers = []
    for root, _, files in os.walk(SRC):
        for f in files:
            if not f.endswith(".py") or (f == "geog.py" and root.endswith("wtune")):
                continue
            txt = open(os.path.join(root, f), errors="replace").read()
            if re.search(r"^\s*from\s+wtune\s+import\s+geog\b|^\s*(import|from)\s+wtune\.geog\b", txt, re.M):
                importers.append(f)
    assert importers == ["cli.py"], importers
    cli = open(os.path.join(SRC, "wtune", "cli.py")).read()
    m = re.search(r"if args.command == 'prep':\n\s+import wtune\.geog\n", cli)
    assert m, "wtune.geog must be imported only inside the 'prep' dispatch"
    assert cli.count("import wtune.geog") == 1
    mine = open(os.path.join(SRC, "wtune", "geog.py")).read()
    for banned in ("tune", "advise", "calc", "probe_plan", "cost_drivers"):
        assert not re.search(r"^\s*from\s+wtune(\.%s\b|\s+import\s+%s\b)" % (banned, banned), mine, re.M)
