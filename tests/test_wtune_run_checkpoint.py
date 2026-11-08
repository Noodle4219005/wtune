#!/usr/bin/env python3
"""`wtune run --checkpoint-minutes`: the restarts --resume resumes from, when the case's own
restart_interval puts none inside a segment. Fakes only."""


import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
from wtune import run as wr
from test_wtune_run import (_geo, _make_case, _Spy, fake_bp)  # noqa: F401  (autouse fixture)

ALIGN = {"steps": 360, "seconds": 21600.0, "label": "interval_seconds 06:00:00"}   # dt 60 s
HOUR = 60                                                                          # d01 steps


def _main(tmp_path, extra=(), fakes=None, case_text=None):
    if not (tmp_path / "case").exists():
        _make_case(tmp_path, *([case_text] if case_text else []))
    f, lines, out = fakes or _Spy(), [], tmp_path / "out"
    rc = wr.main([str(tmp_path / "case"), "--nodes", "1", "--nx", "8", "--ny", "9", "--out",
                  str(out), "--poll-interval", "0"] + list(extra), sbatch_runner=f.sbatch,
                 sacct_runner=f.sacct, snapshot_fn=f.snapshot, sleep=lambda s: None,
                 log=lines.append)
    return rc, out, f, lines


def test_setting_auto_for_a_p1_like_run(tmp_path):
    g, n = _geo(tmp_path), 72 * HOUR
    rst = wr.restart_setting(g, None, n, 100000 * 60.0, "auto", ALIGN)   # case: 100000 min
    assert rst == {"set_min": 720, "cadence_s": 43200.0,
                   "label": "checkpoints every 720 min (auto, for --resume)"}
    assert wr.restart_write_steps(g, 0, n, rst["cadence_s"]) == [h * HOUR for h in
                                                                range(12, 73, 12)]
    assert wr.restart_setting(g, None, n, 100000 * 60.0)["set_min"] == wr.RESTART_OFF_MIN
