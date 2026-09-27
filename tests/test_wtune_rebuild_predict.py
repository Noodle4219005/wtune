import datetime as dt
import types

import pytest

import wtune_rebuild_predict as wrp


def _fake_plan(case_dir, forcing, spinup_min, max_points, sources=("rebuild",)):
    t0 = dt.datetime(2021, 7, 19, 18)
    pts = [types.SimpleNamespace(at=t0 + dt.timedelta(hours=h, minutes=spinup_min), reason="auto",
                                 source="rebuild") for h in (0, 7)]
    return None, t0, t0 + dt.timedelta(hours=30), 36.0, 17, pts, []


def test_dry_run_prints_plan_and_defaults(monkeypatch, capsys, tmp_path):
    seen = {}

    def plan(case_dir, forcing, spinup_min, max_points, sources=("rebuild",)):
        seen.update(forcing=forcing, spinup=spinup_min, max_points=max_points)
        return _fake_plan(case_dir, forcing, spinup_min, max_points)

    monkeypatch.setattr(wrp, "plan", plan)
    assert wrp.main([str(tmp_path), "--nodes", "4", "--nx", "16", "--ny", "18", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert seen == {"forcing": "era5_arco", "spinup": 20, "max_points": None}
    assert "288 ranks, 16x18" in out and "2 parallel jobs x 4 nodes" in out
    wrp.main([str(tmp_path), "--nodes", "4", "--nx", "16", "--ny", "18", "--forcing", "auto",
              "--max-points", "3", "--dry-run"])
    assert seen == {"forcing": "auto", "spinup": 60, "max_points": 3}


def test_refuses_output_inside_case_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(wrp, "plan", _fake_plan)
    monkeypatch.setenv("WPS_DIR", "/x")
    monkeypatch.setenv("WRF_TREE", "/x")
    assert wrp.main([str(tmp_path), "--nodes", "4", "--nx", "16", "--ny", "18",
                     "--out", str(tmp_path / "inside")]) == 2
    assert not (tmp_path / "inside").exists()


def test_rebuild_points_fetch_into_each_point_plan_dir(monkeypatch, tmp_path):
    import datetime as _dt
    import wtune_tune
    seen = []
    monkeypatch.setenv("WPS_DIR", "/x")
    monkeypatch.setenv("WRF_TREE", "/x")
    monkeypatch.setattr(wtune_tune.wtune_probe_plan, "plan_rebuild_segments",
                        lambda case_dir, pts, cfg, pdir, **k: {"windows": [], "pdir": pdir})
    monkeypatch.setattr(wtune_tune.coldstart, "resolve_and_fetch_forcing",
                        lambda plan, pdir, cache, grib_dir=None: (seen.append((pdir, cache)) or (0, {})))
    monkeypatch.setattr(wtune_tune.coldstart, "stage_plan", lambda pdir, site="scc": pdir + "/run.sbatch")
    monkeypatch.setattr(wtune_tune.sweep, "submit_job", lambda cmd, sbatch_runner=None: "1")
    monkeypatch.setattr(wtune_tune, "wait_for_jobs", lambda ctx, ids, label: {j: ("COMPLETED", "") for j in ids})
    monkeypatch.setattr(wtune_tune.wtune_predict, "read_wrfinput_sample_points", lambda d: ([], None))
    monkeypatch.setattr(wtune_tune.wtune_predict, "load_calibration", lambda p: {})
    monkeypatch.setattr(wtune_tune.coldstart, "score_windows_from_rsl", lambda plan, sp, cal: ([], None, []))
    ctx = types.SimpleNamespace(sbatch_runner=None, owned_jobs=[])
    pts = [types.SimpleNamespace(at=_dt.datetime(2021, 7, 20, h)) for h in (1, 8)]
    wtune_tune.run_rebuild_points(ctx, pts, dict(ranks=288, omp=1, nx=16, ny=18), 4, str(tmp_path),
                                  str(tmp_path / "out"), _dt.datetime(2021, 7, 19, 18))
    assert sorted(seen) == [(str(tmp_path / "out" / ("pt_%d" % i)), str(tmp_path / "out" / ("pt_%d" % i) / "grib"))
                            for i in (0, 1)]
