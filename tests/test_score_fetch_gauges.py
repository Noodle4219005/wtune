"""campaign/event_eval/score_fetch.py rain_gauges scoring (bb1f14f review: an empty reference
passed, one run gauge stood in for several, an id at the wrong place matched, longer rows the
figures use were not checked, and 95 % let a shifted dry gauge through)."""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("score_fetch", os.path.join(REPO, "campaign", "event_eval", "score_fetch.py"))
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)
TASK = {"window": ["2009-08-07 00:00", "2009-08-07 04:00"]}
HDR = "id,lat,lon,elev_m,start_utc,end_utc,precip_mm\n"


def _write(d, rows):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "gauges.csv"), "w") as fh:
        fh.write(HDR + "".join("%s,%s,%s,0,%s,%s,%s\n" % r for r in rows))
    return d


def _hours(gid, lat, lon, vals, shift=0):
    return [(gid, lat, lon, "2009-08-07 %02d:00" % (h + shift), "2009-08-07 %02d:00" % (h + 1 + shift), v)
            for h, v in enumerate(vals)]


REF = _hours("467530", 23.5, 120.8, [10, 20, 0, 5]) + _hours("C0M530", 23.49, 120.7, [0, 0, 0, 1])


def test_rain_gauges_scoring(tmp_path):
    ref = _write(str(tmp_path / "ref"), REF)
    score = lambda rows: sf._gauges(TASK, _write(str(tmp_path / "run"), rows), ref)[0]  # noqa: E731
    assert score(REF)
    assert score([("0467530",) + r[1:] if r[0] == "467530" else r for r in REF])      # leading zero
    assert not sf._gauges(TASK, _write(str(tmp_path / "run"), REF), _write(str(tmp_path / "e"), []))[0]
    two = _write(str(tmp_path / "ref2"), _hours("A1", 23.50, 120.70, [0, 0, 0, 0]) +
                 _hours("A2", 23.51, 120.70, [0, 0, 0, 0]))
    assert not sf._gauges(TASK, _write(str(tmp_path / "run"), _hours("B", 23.505, 120.70, [0, 0, 0, 0])),
                          two)[0]                                                # one gauge for two
    assert not score([(r[0], 30.0, 130.0) + r[3:] if r[0] == "C0M530" else r for r in REF])  # wrong place
    assert not score(REF + [("467530", 23.5, 120.8, "2009-08-07 00:00", "2009-08-07 04:00", 1.0)])
    assert score(REF + [("467530", 23.5, 120.8, "2009-08-07 00:00", "2009-08-07 04:00", 35.0)])
    assert not score(_hours("467530", 23.5, 120.8, [10, 20, 0, 5]) +
                     _hours("C0M530", 23.49, 120.7, [0, 0, 1, 0]))                    # dry gauge, shifted
