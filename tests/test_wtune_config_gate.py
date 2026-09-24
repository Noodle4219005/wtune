import json
import math
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np

import wtune_config_gate


MODULE = Path(wtune_config_gate.__file__).resolve()


def _write_file(path, drop=()):
    drop = set(drop)
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("Time", 2)
        data.createDimension("south_north", 3)
        data.createDimension("west_east", 4)
        data.createDimension("DateStrLen", 19)

        if "Times" not in drop:
            times = data.createVariable("Times", "S1", ("Time", "DateStrLen"))
            times[:] = np.array(
                [list("2026-01-01_00:00:00"), list("2026-01-01_01:00:00")],
                dtype="S1",
            )
        if "T2" not in drop:
            t2 = data.createVariable(
                "T2", "f4", ("Time", "south_north", "west_east")
            )
            t2[:] = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1.0
        if "U10" not in drop:
            u10 = data.createVariable("U10", "f4", ("south_north", "west_east"))
            u10[:] = np.arange(12, dtype=np.float32).reshape(3, 4) + 0.5
        if "RAINC" not in drop:
            rainc = data.createVariable(
                "RAINC", "f4", ("Time", "south_north", "west_east")
            )
            rainc[:] = np.full((2, 3, 4), 2.0, dtype=np.float32)
        if "scalar" not in drop:
            scalar = data.createVariable("scalar", "i4", ())
            scalar.assignValue(7)


def _pair(tmp_path, drop=()):
    ref = tmp_path / "ref.nc"
    cand = tmp_path / "cand.nc"
    _write_file(ref)
    _write_file(cand, drop=drop)
    return ref, cand


def _run(*args):
    return subprocess.run(
        [sys.executable, str(MODULE), *map(str, args)],
        cwd=MODULE.parent,
        text=True,
        capture_output=True,
    )


def test_identical_files_compare_and_check(tmp_path):
    ref, cand = _pair(tmp_path)
    results = wtune_config_gate.compare(ref, cand)
    assert {item["name"] for item in results} == {"T2", "U10", "RAINC"}
    assert all(item["rel_rms"] == 0.0 for item in results)

    result = _run("check", "--ref", ref, "--cand", cand)
    assert result.returncode == 0
    assert "GATE PASS" in result.stdout


def test_relative_tolerance(tmp_path):
    ref, cand = _pair(tmp_path)
    with netCDF4.Dataset(cand, "a") as data:
        data.variables["T2"][:] *= np.float32(1.001)

    tight = _run("check", "--ref", ref, "--cand", cand, "--tol-rel", "1e-6")
    assert tight.returncode == 1
    assert "failed_fields=T2" in tight.stdout

    loose = _run("check", "--ref", ref, "--cand", cand, "--tol-rel", "1e-2")
    assert loose.returncode == 0
    assert "GATE PASS" in loose.stdout


def test_missing_field_is_reported(tmp_path):
    ref, cand = _pair(tmp_path, drop={"RAINC"})
    output = tmp_path / "result.json"
    result = _run(
        "check",
        "--ref",
        ref,
        "--cand",
        cand,
        "--json",
        output,
    )
    assert result.returncode == 1
    payload = json.loads(output.read_text())
    missing = next(item for item in payload["fields"] if item["name"] == "RAINC")
    assert missing["status"] == "fail"
    assert missing["reason"] == "missing"


def test_calibrate_suggests_scaled_maximum(tmp_path):
    ref, cand = _pair(tmp_path)
    with netCDF4.Dataset(cand, "a") as data:
        data.variables["T2"][:] *= np.float32(1.001)
    output = tmp_path / "calibration.json"

    result = _run(
        "calibrate",
        "--ref",
        ref,
        "--cand",
        cand,
        "--safety",
        "2.5",
        "--json",
        output,
    )
    assert result.returncode == 0
    payload = json.loads(output.read_text())
    maximum = max(item["rel_rms"] for item in payload["fields"])
    assert math.isclose(payload["tol_rel"], maximum * 2.5)
    assert "suggested --tol-rel" in result.stdout


def test_time_index_selects_different_records(tmp_path):
    ref, cand = _pair(tmp_path)
    with netCDF4.Dataset(cand, "a") as data:
        data.variables["T2"][0, :, :] *= np.float32(1.01)

    last = _run("check", "--ref", ref, "--cand", cand, "--tol-rel", "1e-6")
    first = _run(
        "check",
        "--ref",
        ref,
        "--cand",
        cand,
        "--time-index",
        "0",
        "--tol-rel",
        "1e-6",
    )
    assert last.returncode == 0
    assert first.returncode == 1
    assert "failed_fields=T2" in first.stdout
