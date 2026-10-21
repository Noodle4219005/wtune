"""wtune.run.space: the output-volume check before the first job and the shrunk-history check
after each one (f1-arm marshall 2026-10-21: past the GPFS soft quota WRF+ADIOS2 printed SUCCESS
with 17 KB history frames and run_report said completed)."""
import glob
import json
import os
import types

from wtune import run as wr
from wtune.run import cli as run_cli
from wtune.run import space
from wtune.run.geometry import HistoryStream

import test_wtune_run as T
from test_wtune_run import fake_bp  # noqa: F401  (autouse: a fake src/wtune/bp.py)

LAYOUT = ["--nodes", "1", "--nx", "8", "--ny", "9"]


def test_estimate_counts_history_frames_and_restart_sets():
    geo = types.SimpleNamespace(hist=[HistoryStream(iv_s=3600.0, begin_s=0.0), None,
                                      HistoryStream(iv_s=60.0, begin_s=600.0)],
                                points=[100, 10, 1000])
    need, detail = space.estimate_output_bytes(geo, 1200.0, 1)
    fb = space.frame_bytes
    # d01: t=0 only; d02: no history, one restart; d03: 600..1200 s every 60 s = 11 frames
    assert need == fb(100) * 2 + fb(10) + fb(1000) * 12
    assert detail == "d01 1 x %s, d03 11 x %s" % (space._fmt_bytes(fb(100)),
                                                  space._fmt_bytes(fb(1000)))


def test_quota_headroom_reads_the_soft_limit(monkeypatch):
    monkeypatch.setattr(space, "_gpfs_device", lambda p: "home2")
    out = ("Filesystem type   KB  quota  limit in_doubt grace | files\n"
           "home2      USR   85543152  104857600  209715200  2815232  none |  788266  0  0\n")
    assert space.quota_headroom("/home/x", runner=lambda argv: out) == (104857600 - 85543152) * 1024.0
    no_soft = out.replace("104857600", "0")
    assert space.quota_headroom("/home/x", runner=lambda argv: no_soft) == (209715200 - 85543152) * 1024.0
    monkeypatch.setattr(space, "_gpfs_device", lambda p: None)
    assert space.quota_headroom("/tmp", runner=lambda argv: out) is None


def test_writable_bytes_takes_the_smaller(monkeypatch, tmp_path):
    monkeypatch.setattr(space, "quota_headroom", lambda p, r=None: 5e9)
    assert space.writable_bytes(str(tmp_path)) == (5e9, "GPFS quota headroom")
    monkeypatch.setattr(space, "quota_headroom", lambda p, r=None: None)
    assert space.writable_bytes(str(tmp_path))[1] == "filesystem free space"


def _write(path, n):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "data.0"), "wb") as fh:
        fh.write(b"x" * n)


def test_shrunk_outputs_per_stream_and_kind(tmp_path):
    for t, n in (("00", 1000), ("01", 990), ("02", 50)):
        _write(str(tmp_path / ("wrfout_d01_2020-06-01_%s:00:00" % t)), n)
    _write(str(tmp_path / "wrfout_d02_2020-06-01_00:00:00"), 20)          # its own stream
    (tmp_path / "wrfout_d01_2020-06-01_03:00:00").write_bytes(b"x" * 5)  # netCDF: other kind
    names = sorted(os.listdir(tmp_path)) + ["wrfout_d01_2020-06-01_04:00:00"]  # absent: ignored
    assert space.shrunk_outputs(str(tmp_path), names) == ["wrfout_d01_2020-06-01_02:00:00"]


class LosesWrites(T.Fakes):
    """A job that ends SUCCESS but whose last history frames lost their data."""

    def sbatch(self, cmd):
        out = super().sbatch(cmd)
        seg = os.path.dirname(cmd.split()[-1].strip("'"))
        frames = sorted(glob.glob(os.path.join(seg, "wrfout_d01_*")))
        for i, f in enumerate(frames):
            _write(f, 10 if i >= len(frames) - 2 else 10000)
        return out


def _run(tmp_path, args, fakes):
    case = T._make_case(tmp_path, T.NAMELIST_3DOM)
    out, lines = tmp_path / "out", []
    rc = wr.main([str(case), "--out", str(out), "--poll-interval", "0"] + LAYOUT + args,
                 sbatch_runner=fakes.sbatch, sacct_runner=fakes.sacct,
                 snapshot_fn=fakes.snapshot, sleep=lambda s: None, log=lines.append)
    rep = out / "run_report.json"
    return rc, lines, json.loads(rep.read_text()) if rep.exists() else None


def test_segment_with_shrunk_history_fails_without_resume(tmp_path, capsys):
    f = LosesWrites()
    rc, _lines, rep = _run(tmp_path, [], f)
    assert rc == 1 and len(f.submitted) == 1
    assert "history outputs are under 10%" in rep["error"]
    att = rep["segments"][0]["attempts"][0]
    assert att["ok"] is False and len(att["shrunk"]) == 2


def test_run_refuses_when_the_output_does_not_fit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_cli, "writable_bytes", lambda p, r=None: (1e3, "GPFS quota headroom"))
    f = T.Fakes()
    rc, lines, _rep = _run(tmp_path, [], f)
    assert rc == 2 and f.submitted == []
    assert any(line.startswith("output estimate") for line in lines)
    assert "only 0.0 GB is writable (GPFS quota headroom)" in capsys.readouterr().err
    f = T.Fakes()
    (tmp_path / "again").mkdir()
    rc, _lines, _rep = _run(tmp_path / "again", ["--no-space-check"], f)
    assert rc == 0 and f.submitted


def test_plan_warns_about_debug_level(tmp_path):
    case = T._make_case(tmp_path, T.NAMELIST_3DOM)
    text = (case / "namelist.input").read_text()
    (case / "namelist.input").write_text(text.replace("&time_control", "&time_control\n debug_level = 50,", 1))
    lines = []
    rc = wr.main([str(case), "--out", str(tmp_path / "out"), "--dry-run"] + LAYOUT, log=lines.append)
    assert rc == 0 and any("debug_level=50" in ln for ln in lines)
