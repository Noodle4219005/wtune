"""Independent reproduction: `wtune post fetch stations` saved NCEI's 301-to-HTML page as each
station's ISD-Lite .gz. The HTTP layer is faked below urllib.request.urlopen (what
wtune.post.sources.http_get/download call), so urllib's own redirect handling follows the 301."""
from __future__ import annotations

import argparse
import email.message
import gzip
import hashlib
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import urllib.response

import pytest

from wtune.post import sources
from wtune.post.fetch import stations
from wtune.post.met_pointstat import parse_isd_lite

BUCKET_LITE = "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/%d/%s-%d.gz"
BUCKET_INDEX = "https://noaa-isd-pds.s3.amazonaws.com/index.html#isd-lite/"
HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
HTML = (b"<!DOCTYPE html><html><head><title>noaa-isd-pds</title></head><body>"
        b"<div id='listing'>isd-lite/</div></body></html>\n")
YEAR = 2024
A, B, C = "722020-12839", "722050-12815", "722106-12897"   # C is not in the bucket


def isd_lite_bytes(seed):
    lines = []
    for d in (1, 2, 3):
        for h in range(0, 24, 3):
            lines.append("%4d %02d %02d %02d %5d %5d %5d %5d %5d %5d %5d %5d" % (
                YEAR, 6, d, h, 250 + seed + h, 180 + seed, 10132, 270, 46, 0, 0, -9999))
    return gzip.compress(("\n".join(lines) + "\n").encode())


GZ = {A: isd_lite_bytes(1), B: isd_lite_bytes(2), C: isd_lite_bytes(3)}

HISTORY_CSV = (
    '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
    '"722020","12839","MIAMI INTL AP","US","FL","KMIA","+25.788","-080.317","+0008.8","19730101","20261020"\n'
    '"722050","12815","ORLANDO INTL AP","US","FL","KMCO","+28.434","-081.325","+0027.4","19730101","20261020"\n'
    '"722106","12897","FORT MYERS PAGE FLD","US","FL","KFMY","+26.586","-081.861","+0004.6","19730101","20261020"\n'
    '"999999","99999","BOGUS","US","FL","","+27.000","-081.000","","19730101","20261020"\n'
)


class FakeNet(urllib.request.HTTPSHandler):
    """Serves the URLs below; everything else is 404. NCEI isd-lite answers 301 to the bucket's
    HTML index (as on 2026-10-26); the bucket serves real gzip for stations in `in_bucket`."""

    def __init__(self, in_bucket, bucket_body=None):
        super().__init__()
        self.in_bucket = set(in_bucket)
        self.bucket_body = bucket_body or {}
        self.log = []

    def _resp(self, req, code, msg, body, extra=None):
        hdrs = email.message.Message()
        hdrs["Content-Length"] = str(len(body))
        for k, v in (extra or {}).items():
            hdrs[k] = v
        r = urllib.response.addinfourl(io.BytesIO(body), hdrs, req.full_url, code)
        r.msg = msg
        return r

    def https_open(self, req):
        url = urllib.parse.urldefrag(req.full_url)[0]
        self.log.append(url)
        if url == HISTORY:
            return self._resp(req, 200, "OK", HISTORY_CSV.encode())
        if url == BUCKET_INDEX.split("#")[0]:
            return self._resp(req, 200, "OK", HTML, {"Content-Type": "text/html"})
        if url.startswith("https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/"):
            return self._resp(req, 301, "Moved Permanently", b"<html>301</html>",
                              {"Location": BUCKET_INDEX})
        for sid in GZ:
            if url == BUCKET_LITE % (YEAR, sid, YEAR) and sid in self.in_bucket:
                body = self.bucket_body.get(sid, GZ[sid])
                return self._resp(req, 200, "OK", body)
        return self._resp(req, 404, "Not Found", b"<Error><Code>NoSuchKey</Code></Error>")

    def http_open(self, req):
        raise AssertionError("plain http request: %s" % req.full_url)


@pytest.fixture
def net(monkeypatch):
    holder = {}

    def install(in_bucket, bucket_body=None):
        fake = FakeNet(in_bucket, bucket_body)
        opener = urllib.request.build_opener(fake)

        def urlopen(req, timeout=None, **kw):
            return opener.open(req, timeout=timeout)

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(sources.time, "sleep", lambda s: None)
        holder["fake"] = fake
        return fake

    return install


def args_for(project):
    return argparse.Namespace(project=str(project), run_dir=None, start="2024-06-01_00",
                              end="2024-06-03_00", bbox="24.5,29.5,-83,-79", max_stations=10)


def base(project):
    return os.path.join(str(project), "input", "obs", "stations")


def prov(project):
    with open(os.path.join(base(project), "provenance.json")) as fh:
        return json.load(fh)


def name(sid):
    return "%s-%d.gz" % (sid, YEAR)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def assert_good_file(project, sid):
    path = os.path.join(base(project), name(sid))
    assert os.path.isfile(path), "%s missing" % path
    raw = open(path, "rb").read()
    assert raw[:2] == b"\x1f\x8b", "%s is not gzip; starts %r" % (name(sid), raw[:40])
    assert raw == GZ[sid]
    with gzip.open(path, "rt") as fh:
        recs = parse_isd_lite(fh.read())
    assert len(recs) == 24 and recs[0][1][:4] == [YEAR, 6, 1, 0]
    assert prov(project)["files"][name(sid)] == sha(raw)


def listed_files_exist_and_are_gzip(project):
    for fn in prov(project)["files"]:
        path = os.path.join(base(project), fn)
        assert os.path.isfile(path), "provenance.json lists %s, which is not on disk" % fn
        if fn.endswith(".gz"):
            assert open(path, "rb").read(2) == b"\x1f\x8b", "provenance lists non-gzip %s" % fn


def catalogue_ids(project):
    with open(os.path.join(base(project), "stations.csv")) as fh:
        return [line.split(",")[0] for line in fh.read().splitlines()[1:]]


def test_bucket_answers_non_gzip_station_skipped_with_message(tmp_path, net, capsys):
    net([A, B], bucket_body={B: HTML})                # 200 but an HTML body
    stations.run(args_for(tmp_path))
    assert_good_file(tmp_path, A)
    assert not os.path.exists(os.path.join(base(tmp_path), name(B)))
    assert name(B) not in prov(tmp_path)["files"]
    assert B not in catalogue_ids(tmp_path)
    assert "not a gzip" in capsys.readouterr().out
    listed_files_exist_and_are_gzip(tmp_path)
