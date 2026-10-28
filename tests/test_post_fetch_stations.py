"""wtune post fetch stations keeps only real ISD-Lite gzip files (f1-arm c34 2026-10-26: NCEI's
isd-lite path answered 301 to the AWS bucket's index page, and that page was saved as both
stations' .gz; terrain_point_stat then failed 'Not a gzipped file')."""
import argparse
import gzip
import json
import os

from wtune.post import sources
from wtune.post.fetch import stations

HISTORY = ("USAF,WBAN,STATION NAME,CTRY,STATE,ICAO,LAT,LON,ELEV(M),BEGIN,END\n"
           "724604,93076,GOOD,US,CO,KAAA,37.0,-102.5,1200,20000101,20251231\n"
           "724646,03028,MOVED,US,CO,KBBB,37.1,-102.4,1300,20000101,20251231\n")
PAGE = b"<!DOCTYPE html><html>index</html>"


def test_html_in_place_of_isd_lite_is_not_kept(tmp_path, monkeypatch, capsys):
    urls = []

    def download(url, dest, known_sha=None, **k):
        urls.append(url)
        if "724604" in url:
            with gzip.open(dest, "wb") as fh:
                fh.write(b"2019 02 13 20    10   -20 10200   270    30     0     0     0\n")
        else:
            with open(dest, "wb") as fh:
                fh.write(PAGE)
        return sources.sha256_file(dest), True
    monkeypatch.setattr(sources, "http_get", lambda url, **k: HISTORY.encode())
    monkeypatch.setattr(sources, "download", download)
    base = tmp_path / "input" / "obs" / "stations"
    base.mkdir(parents=True)
    (base / "724604-93076-2019.gz").write_bytes(PAGE)        # kept by an earlier fetch
    args = argparse.Namespace(project=str(tmp_path), run_dir=None, start="2019-02-13_20",
                              end="2019-02-13_22", bbox="36.5,37.6,-103.2,-101.9", max_stations=5)
    stations.run(args)
    assert all(u.startswith("https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/2019/") for u in urls)
    assert sorted(os.listdir(base)) == ["724604-93076-2019.gz", "provenance.json", "stations.csv"]
    assert (base / "stations.csv").read_text().count("\n") == 2     # header + the good station
    assert "is not a gzip file" in capsys.readouterr().out
    assert set(json.loads((base / "provenance.json").read_text())["files"]) == {
        "724604-93076-2019.gz", "stations.csv"}


def test_when_no_station_is_kept_the_listing_drops_the_removed_ones(tmp_path, monkeypatch, capsys):
    """bb1f14f reproduction: with every station skipped, run() returned before rewriting
    stations.csv / provenance.json, which still listed the HTML file it had just deleted."""
    def download(url, dest, known_sha=None, **k):
        with open(dest, "wb") as fh:
            fh.write(PAGE)
        return sources.sha256_file(dest), True
    monkeypatch.setattr(sources, "http_get", lambda url, **k: HISTORY.encode())
    monkeypatch.setattr(sources, "download", download)
    base = tmp_path / "input" / "obs" / "stations"
    base.mkdir(parents=True)
    (base / "724604-93076-2019.gz").write_bytes(PAGE)
    (base / "stations.csv").write_text("id,name,icao,lat,lon,elev_m\n724604-93076,GOOD,KAAA,37,-102.5,1200\n")
    sources.write_provenance(str(tmp_path), "stations", stations.PROVENANCE, [],
                             {"724604-93076-2019.gz": "x", "stations.csv": "y"})
    args = argparse.Namespace(project=str(tmp_path), run_dir=None, start="2019-02-13_20",
                              end="2019-02-13_22", bbox="36.5,37.6,-103.2,-101.9", max_stations=5)
    assert stations.run(args) == []
    assert (base / "stations.csv").read_text() == "id,name,icao,lat,lon,elev_m\n"
    assert set(json.loads((base / "provenance.json").read_text())["files"]) == {"stations.csv"}
    assert "724646-03028 has no ISD-Lite file" in capsys.readouterr().out


def test_refetch_removes_files_of_stations_no_longer_selected(tmp_path, monkeypatch, capsys):
    """haitang2017-r2 2026-10-28: a fetch with a wrong bbox kept 110 North-African stations;
    the refetch rewrote stations.csv but left their .gz files beside the new ones."""
    def download(url, dest, known_sha=None, **k):
        with gzip.open(dest, "wb") as fh:
            fh.write(b"2019 02 13 20    10   -20 10200   270    30     0     0     0\n")
        return sources.sha256_file(dest), True
    monkeypatch.setattr(sources, "http_get", lambda url, **k: HISTORY.encode())
    monkeypatch.setattr(sources, "download", download)
    base = tmp_path / "input" / "obs" / "stations"
    base.mkdir(parents=True)
    with gzip.open(base / "605710-99999-2019.gz", "wb") as fh:
        fh.write(b"x")
    args = argparse.Namespace(project=str(tmp_path), run_dir=None, start="2019-02-13_20",
                              end="2019-02-13_22", bbox="36.5,37.6,-103.2,-101.9", max_stations=5)
    stations.run(args)
    assert sorted(os.listdir(base)) == ["724604-93076-2019.gz", "724646-03028-2019.gz",
                                        "provenance.json", "stations.csv"]
    assert "removed 1 file(s)" in capsys.readouterr().out
    assert "605710-99999-2019.gz" not in json.loads((base / "provenance.json").read_text())["files"]
