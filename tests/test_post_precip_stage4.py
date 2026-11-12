"""Regression: Stage IV index/window on the scc SC26-P1 layout (relative --project . semantics),
station Point-Stat helper reuse."""
import os
from datetime import datetime, timedelta

from wtune.post.recipes import precip


def _project(tmp_path):
    proj = tmp_path / "proj"
    s4 = proj / "input" / "obs" / "stage4"
    s4.mkdir(parents=True)
    (proj / "project.yaml").write_text("name: x\n")
    t = datetime(2019, 1, 28, 1)
    while t <= datetime(2019, 1, 31, 0):
        (s4 / ("ST4.%s.01h.grib" % t.strftime("%Y%m%d%H"))).write_text("")
        if t.hour % 6 == 0:
            (s4 / ("ST4.%s.06h.grib" % t.strftime("%Y%m%d%H"))).write_text("")
        t += timedelta(hours=1)
    for d in ("2019012912", "2019013012"):
        (s4 / ("ST4.%s.24h.grib" % d)).write_text("")
    (s4 / "provenance.json").write_text("{}")
    (s4 / "nexrad_sites.txt").write_text("")
    (proj / "post" / "o2" / "precip").mkdir(parents=True)
    return proj


def _frames():
    t, out = datetime(2019, 1, 28), []
    while t <= datetime(2019, 1, 31):
        out.append(t)
        t += timedelta(hours=1)
    return out


def test_index_found_with_relative_project(tmp_path, monkeypatch):
    proj = _project(tmp_path)
    monkeypatch.chdir(proj)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    pd = precip._project_of("post/o2/precip")          # relative out dir, cwd = project
    assert pd == str(proj)
    idx = precip.stage4_index(".")                      # relative --project . also works
    assert (24, datetime(2019, 1, 30, 12)) in idx and (1, datetime(2019, 1, 28, 1)) in idx
    assert len([k for k in idx if k[0] == 24]) == 2
    assert os.path.isfile(idx[(24, datetime(2019, 1, 30, 12))])


def test_station_verification_reports_no_project():
    res, dataset, reason = precip.station_verification("r", 2, [], "/tmp/o", 6, None)
    assert res is None and "no wtune project" in reason and "ISD-Lite" in dataset
