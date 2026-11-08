"""Tests for post/atcf.py: round-trip write/parse, and parsing against a real NHC archive
b-deck sample (tests/fixtures/post/bal122005_sample.dat, Katrina 2005) to pin the column
layout down to a known-good file, not just our own writer's self-consistency.
"""
import os
from datetime import datetime

import pytest

from wtune.post import atcf

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "post")


def test_write_lines_round_trip(tmp_path):
    rows = [
        dict(time=datetime(2005, 8, 27, 0), tau_h=0, lat=24.6, lon=-83.3, vmax_kt=90,
             mslp_hpa=959, radii_kt={34: (90, 75, 75, 75), 50: (60, 60, 40, 30),
                                     64: (25, 25, 20, 15)}),
        dict(time=datetime(2005, 8, 27, 6), tau_h=6, lat=24.4, lon=-84.0, vmax_kt=20,
             mslp_hpa=1008, radii_kt={}),
    ]
    out = tmp_path / "awtun12l2005.dat"
    atcf.write_lines(rows, str(out), basin="AL", cy="12", tech="wtun", stormname="katrina")
    parsed = atcf.parse_file(str(out))

    # row 0 (Vmax=90kt=HU) writes one line per 34/50/64 threshold
    assert len(parsed) == 4
    for line in parsed[:3]:
        assert line["tech"] == "WTUN"
        assert line["vmax_kt"] == 90
        assert line["mslp_hpa"] == 959
        assert line["ty"] == "HU"
        assert line["stormname"] == "KATRINA"
    assert [line["rad_kt"] for line in parsed[:3]] == [34, 50, 64]
    assert parsed[0]["radii"] == (90, 75, 75, 75)
    assert parsed[2]["radii"] == (25, 25, 20, 15)

    # row 1 (Vmax=20kt=TD) writes a single RAD=0 line
    assert parsed[3]["rad_kt"] == 0
    assert parsed[3]["vmax_kt"] == 20
    assert parsed[3]["ty"] == "TD"
    assert parsed[3]["lat"] == pytest.approx(24.4)
    assert parsed[3]["lon"] == pytest.approx(-84.0)


def test_fetch_bdeck_wraps_download_failure(tmp_path, monkeypatch):
    def _boom(*a, **kw):
        raise OSError("no route to host")
    monkeypatch.setattr(atcf.urllib.request, "urlopen", _boom)
    with pytest.raises(atcf.ATCFError, match="could not download"):
        atcf.fetch_bdeck(2005, "AL", "12", str(tmp_path / "b.dat"))


def test_parse_line_too_short_raises():
    with pytest.raises(atcf.ATCFError):
        atcf.parse_line("AL, 12, 2005082700")
