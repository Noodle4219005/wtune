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
