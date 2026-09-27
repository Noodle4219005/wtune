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
