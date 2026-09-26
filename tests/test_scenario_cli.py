"""Tests for scenario.cli: `list` output and `generate --dry-run` (mocked analog selection, no
network/downloads/ungrib.exe)."""

import datetime

import pytest

import scenario.cli as scenario_cli
from scenario.base import list_generators


def test_cli_list_contains_every_registered_name(capsys):
    rc = scenario_cli.main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    for row in list_generators():
        assert row["name"] in out


def test_cli_generate_dry_run_analog(tmp_path, monkeypatch, capsys):
    fixed_date = datetime.datetime(2025, 8, 29, 12)
    monkeypatch.setattr(scenario_cli, "select_analog_date", lambda start, **kw: fixed_date)

    rc = scenario_cli.main([
        "generate", "--generator", "analog",
        "--start", "2031-07-15T12:00:00", "--end", "2031-07-15T12:00:00",
        "--bbox", "30,40,-110,-100",
        "--out", str(tmp_path / "out"),
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "analog date: %s" % fixed_date.isoformat() in out
    assert "generator: analog" in out
    assert "files expected: 1" in out
    assert "license:" in out
    # no files/network side effects on a dry run
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


def test_cli_generate_dry_run_reports_missing_fields(tmp_path, monkeypatch, capsys):
    fixed_date = datetime.datetime(2025, 8, 29, 12)
    monkeypatch.setattr(scenario_cli, "select_analog_date", lambda start, **kw: fixed_date)

    rc = scenario_cli.main([
        "generate", "--generator", "analog",
        "--start", "2031-07-15T00:00:00", "--end", "2031-07-15T12:00:00",
        "--bbox", "30,40,-110,-100",
        "--out", str(tmp_path / "out"),
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "files expected: 3" in out


def test_cli_generate_noncommercial_gated_without_flag(tmp_path, capsys):
    rc = scenario_cli.main([
        "generate", "--generator", "graphcast",
        "--start", "2031-07-15T00:00:00", "--end", "2031-07-15T00:00:00",
        "--bbox", "30,40,-110,-100",
        "--out", str(tmp_path / "out"),
        "--dry-run",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "allow_noncommercial" in err


def _case_dir_with_interval(tmp_path, interval_seconds):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "namelist.wps").write_text("&share\n interval_seconds = %d,\n/\n" % interval_seconds)
    return case_dir


def test_interval_h_derived_from_case_namelist(tmp_path, monkeypatch, capsys):
    # conus25-style case: 3h cadence (interval_seconds=10800) -> 0,3,6h = 3 files over a 6h span.
    case_dir = _case_dir_with_interval(tmp_path, 10800)
    monkeypatch.setattr(scenario_cli, "select_analog_date",
                         lambda start, **kw: datetime.datetime(2025, 8, 29, 12))

    rc = scenario_cli.main([
        "generate", "--generator", "analog",
        "--start", "2031-07-15T00:00:00", "--end", "2031-07-15T06:00:00",
        "--bbox", "30,40,-110,-100", "--out", str(tmp_path / "out"),
        "--case-dir", str(case_dir), "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "interval_h: 3 (interval_seconds=10800)" in out
    assert "files expected: 3" in out


def test_interval_h_falls_back_to_namelist_input_when_no_namelist_wps(tmp_path, monkeypatch, capsys):
    # most user cases ship only namelist.input, not a namelist.wps.
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "namelist.input").write_text("&time_control\n interval_seconds = 10800,\n/\n")
    monkeypatch.setattr(scenario_cli, "select_analog_date",
                         lambda start, **kw: datetime.datetime(2025, 8, 29, 12))

    rc = scenario_cli.main([
        "generate", "--generator", "analog",
        "--start", "2031-07-15T00:00:00", "--end", "2031-07-15T06:00:00",
        "--bbox", "30,40,-110,-100", "--out", str(tmp_path / "out"),
        "--case-dir", str(case_dir), "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "interval_h: 3 (interval_seconds=10800)" in out
    assert "namelist.input" in out
    assert "&time_control.interval_seconds" in out
    assert "files expected: 3" in out


def test_interval_h_explicit_flag_wins_over_case_dir(tmp_path, monkeypatch, capsys):
    case_dir = _case_dir_with_interval(tmp_path, 10800)
    monkeypatch.setattr(scenario_cli, "select_analog_date",
                         lambda start, **kw: datetime.datetime(2025, 8, 29, 12))

    rc = scenario_cli.main([
        "generate", "--generator", "analog",
        "--start", "2031-07-15T00:00:00", "--end", "2031-07-15T06:00:00",
        "--bbox", "30,40,-110,-100", "--out", str(tmp_path / "out"),
        "--case-dir", str(case_dir), "--interval-h", "6", "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "interval_h: 6 (interval_seconds=21600)" in out
    assert "files expected: 2" in out


def test_interval_h_coarsened_for_gpu_generator_prints_note(tmp_path, capsys):
    # aifs's Caps.min_interval_h=6h is coarser than the case's own 3h cadence -- must coarsen,
    # not interpolate, and say so.
    case_dir = _case_dir_with_interval(tmp_path, 10800)

    rc = scenario_cli.main([
        "generate", "--generator", "aifs",
        "--start", "2031-07-15T00:00:00", "--end", "2031-07-15T06:00:00",
        "--bbox", "30,40,-110,-100", "--out", str(tmp_path / "out"),
        "--case-dir", str(case_dir), "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "coarsening to 6h" in out
    assert "interval_h: 6 (interval_seconds=21600)" in out
    assert "files expected: 2" in out


def test_interval_h_coarsened_note_and_provenance_written(tmp_path, monkeypatch):
    """Non-dry-run: interval_seconds and interval_coarsened land in forcing.json's provenance,
    for a downstream namelist-rendering stage to read -- never touching the formal case dir."""
    from scenario.base import Caps, ForcingSet

    class FakeGpuGen:
        def capabilities(self):
            return Caps(fields=(), levels_pa=(), has_soil=False, has_sst=False,
                        has_hydrometeors=False, min_interval_h=6, max_lead_h=240,
                        needs_gpu=True, license="x", noncommercial=False)

        def generate(self, req):
            req.out_dir.mkdir(parents=True, exist_ok=True)
            return ForcingSet(files=[], prefix="FAKE", valid_times=[req.start])

    monkeypatch.setattr(scenario_cli, "get_generator",
                         lambda name, allow_noncommercial=False: FakeGpuGen())

    case_dir = _case_dir_with_interval(tmp_path, 10800)
    out_dir = tmp_path / "out"
    rc = scenario_cli.main([
        "generate", "--generator", "fake-gpu",
        "--start", "2031-07-15T00:00:00", "--end", "2031-07-15T00:00:00",
        "--bbox", "30,40,-110,-100", "--out", str(out_dir),
        "--case-dir", str(case_dir),
    ])
    assert rc == 0
    back = ForcingSet.from_json(out_dir / "forcing.json")
    assert back.provenance["interval_seconds"] == 21600
    assert back.provenance["interval_coarsened"] == {"requested_h": 3.0, "used_h": 6.0}
