"""`wtune prep wps|real`: one job each over the case's window, outputs linked into case/."""
import json
import os

import pytest

import wtune.geog
from wtune import prep
from wtune.tools import case_rebuild as cr

PLAN = {"nodes": 1, "config": {"ranks": 4, "omp": 1, "nx": 2, "ny": 2, "numtiles": 1},
        "max_dom": 1, "verify_steps": 0, "span_start": "2020-01-01_00:00:00",
        "span_end": "2020-01-01_06:00:00", "interval_seconds": 10800}
CHOSEN = {"prefix": "GFS", "vtable": "Vtable.GFS", "forcing_interval_h": 3}


@pytest.fixture
def project(tmp_path, monkeypatch):
    p = tmp_path / "p"
    (p / "case").mkdir(parents=True)
    (p / "project.yaml").write_text("name: p\n")
    (p / "case" / "namelist.wps").write_text("&share\n/\n")
    monkeypatch.setenv("WTUNE_PROJECT", str(p))
    monkeypatch.setenv("WPS_DIR", "/wps")
    monkeypatch.setenv("WRF_TREE", "/wrf")
    return p


def _prep_dir(project):
    out = project / "input" / "prep"
    (out / "wps").mkdir(parents=True, exist_ok=True)
    (out / "real").mkdir(exist_ok=True)
    (out / "plan.json").write_text(json.dumps(PLAN))
    (out / "forcing.json").write_text(json.dumps({"chosen": CHOSEN}))
    return out


class Slurm:
    def __init__(self, make):
        self.make, self.scripts = make, []

    def sbatch(self, cmd):
        path = cmd.split()[-1]
        self.scripts.append(open(path).read())
        self.make()
        return "Submitted batch job 77"

    def sacct(self, cmd):
        return "77|COMPLETED|00:01:00\n"


def test_prep_wps_runs_wps_only_and_checks_met_em(project, monkeypatch):
    out = project / "input" / "prep"
    monkeypatch.setattr(cr, "cmd_rebuild_plan", lambda case, o, forcing=None: _prep_dir(project))
    monkeypatch.setattr(cr, "cmd_rebuild_fetch", lambda o: (0, {}))
    monkeypatch.setattr(cr, "cmd_rebuild_stage", lambda o: None)

    def met_em():
        for h in (0, 3, 6):
            (out / "wps" / ("met_em.d01.2020-01-01_%02d:00:00.nc" % h)).write_text("x")
    s = Slurm(met_em)
    got = prep.prep_wps(poll_s=0, sbatch_runner=s.sbatch, sacct_runner=s.sacct,
                        sleep=lambda _s: None)
    assert len(got) == 3
    text = s.scripts[0]
    assert "geogrid.exe" in text and "metgrid.exe" in text and "./real.exe" not in text
    assert "--job-name=wtune-prep-wps" in text


def test_prep_real_links_its_outputs_into_case(project):
    out = _prep_dir(project)
    (out / "wps" / "met_em.d01.2020-01-01_00:00:00.nc").write_text("x")
    (project / "case" / "wrfbdy_d01").symlink_to("/elsewhere/wrfbdy_d01")

    def real():
        for n in ("wrfinput_d01", "wrfbdy_d01"):
            (out / "real" / n).write_text("x")
    s = Slurm(real)
    with pytest.raises(prep.PrepError, match="--replace"):
        prep.prep_real(poll_s=0, sbatch_runner=s.sbatch, sacct_runner=s.sacct)
    assert not s.scripts                                   # nothing submitted
    made = prep.prep_real(replace=True, poll_s=0, sbatch_runner=s.sbatch, sacct_runner=s.sacct,
                          sleep=lambda _s: None)
    assert made == ["wrfbdy_d01", "wrfinput_d01"]
    assert "./real.exe" in s.scripts[0] and "geogrid.exe" not in s.scripts[0]
    for n in made:
        assert os.readlink(project / "case" / n) == str(out / "real" / n)


def test_prep_real_needs_met_em(project):
    _prep_dir(project)
    with pytest.raises(prep.PrepError, match="prep wps"):
        prep.prep_real(poll_s=0)


def test_wtune_prep_forwards_wps_and_real(monkeypatch):
    seen = []
    monkeypatch.setattr(prep, "main", lambda argv: seen.append(argv) or 0)
    assert wtune.geog.main(["real", "--replace"]) == 0 and seen == [["real", "--replace"]]
