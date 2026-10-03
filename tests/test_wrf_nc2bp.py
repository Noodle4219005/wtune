"""Synthetic WRF netCDF -> BP -> netCDF bit-exact regression (restored from
8ba1629:tools/wrf_nc2bp_roundtrip_test.py). Runs under the wtune env: the converters go through
wtune_bp, which runs them as a subprocess of the envs/adios2 interpreter (or
$WTUNE_ADIOS2_PYTHON); skipped when no adios2-capable interpreter is found."""

import os
import sys

import numpy as np
import pytest

import wtune_bp

netCDF4 = pytest.importorskip("netCDF4")
Dataset = netCDF4.Dataset

if wtune_bp.adios2_python() is None:
    pytest.skip("no adios2-capable interpreter (pixi install in envs/adios2, or set "
                "WTUNE_ADIOS2_PYTHON)", allow_module_level=True)


def _make_source(path):
    dates = ["2026-01-01_00:00:00", "2026-01-01_03:00:00"]
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 3)
        ds.createDimension("west_east_stag", 4)
        ds.createDimension("bottom_top", 2)
        ds.createDimension("bdy_width", 1)

        ds.setncattr("TITLE", "synthetic WRF input")
        ds.setncattr("MAP_PROJ", np.int32(3))
        ds.setncattr("DX", np.float32(3000.25))
        ds.setncattr("INTS", np.array([1, -2, 7], dtype=np.int32))
        ds.setncattr("REALS", np.array([1.5, -2.25], dtype=np.float32))

        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:, :] = np.asarray([list(x) for x in dates], dtype="S1")

        xlat = ds.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        xlat[:] = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10.0
        xlat.setncattr("FieldType", np.int32(104))
        xlat.setncattr("MemoryOrder", "XY ")
        xlat.setncattr("description", "latitude")
        xlat.setncattr("units", "degree_north")
        xlat.setncattr("stagger", "")
        xlat.setncattr("coordinates", "XLONG XLAT")

        u = ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))
        u[:] = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 4)
        u.setncattr("FieldType", np.int32(104))
        u.setncattr("MemoryOrder", "XYZ")
        u.setncattr("description", "x wind")
        u.setncattr("units", "m s-1")
        u.setncattr("stagger", "X")

        bdy = ds.createVariable("U_BXS", "f4", ("Time", "bdy_width", "bottom_top", "south_north"))
        bdy[:] = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2) - 1.0
        bdy.setncattr("FieldType", np.int32(104))
        bdy.setncattr("MemoryOrder", "XSZ")
        bdy.setncattr("description", "bdy x wind")
        bdy.setncattr("units", "m s-1")
        bdy.setncattr("stagger", "X")

        integer = ds.createVariable("I_FIELD", "i4", ("Time", "south_north", "west_east"))
        integer[:] = np.arange(12, dtype=np.int32).reshape(2, 2, 3) - 4
        integer.setncattr("FieldType", np.int32(106))
        integer.setncattr("MemoryOrder", "XY ")
        integer.setncattr("description", "integer field")

        scalar = ds.createVariable("SCALAR", "f4", ("Time",))
        scalar[:] = np.array([2.5, -7.0], dtype=np.float32)
        scalar.setncattr("FieldType", np.int32(104))
        scalar.setncattr("MemoryOrder", "0")
        scalar.setncattr("description", "scalar field")

        for name, values in {
            "md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_": dates,
            "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_": [
                "2026-01-01_03:00:00",
                "2026-01-01_06:00:00",
            ],
        }.items():
            var = ds.createVariable(name, "S1", ("Time", "DateStrLen"))
            var[:, :] = np.asarray([list(x) for x in values], dtype="S1")


@pytest.mark.parametrize("engine", ["BP5", "BP4"])
def test_roundtrip_bit_exact(tmp_path, engine):
    source = tmp_path / "wrfinput_d01"
    bp = tmp_path / "wrfinput_d01.bp"
    _make_source(str(source))
    assert wtune_bp.nc2bp(str(source), str(bp), engine=engine) == str(bp)
    assert wtune_bp.bp_engine(str(bp)) == engine
    back = tmp_path / "back.nc"
    wtune_bp.bp2nc(str(bp), str(back))
    r = wtune_bp.compare_netcdf(str(source), str(back))
    assert r["ok"], r["mismatches"]
    assert r["vars_compared"] == 8 and r["max_abs_diff"] == 0.0


def test_api_roundtrip_check_and_cached_inputs(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    _make_source(str(case / "wrfinput_d01"))
    _make_source(str(case / "wrfbdy_d01"))
    cache = tmp_path / "cache"
    m = wtune_bp.convert_inputs(str(case), str(cache))
    assert sorted(m) == ["wrfbdy_d01", "wrfinput_d01"]
    assert wtune_bp.is_bp(m["wrfinput_d01"])                 # single-record: one BP, unchanged
    assert wtune_bp.is_per_record_dir(m["wrfbdy_d01"])       # multi-record: one BP per record
    for name, bp in m.items():
        r = wtune_bp.roundtrip_check(str(case / name), bp, str(tmp_path / "rt"))
        assert r["ok"], r["mismatches"]
    assert not os.listdir(tmp_path / "rt")
    assert "adios2" not in sys.modules     # never imported into the wtune process


def test_cli_nc2bp_check(tmp_path, capsys):
    source = tmp_path / "wrfinput_d01"
    _make_source(str(source))
    out = tmp_path / "out"
    assert wtune_bp.main(["nc2bp", str(source), "--out", str(out), "--check"]) == 0
    assert wtune_bp.is_bp(str(out / "wrfinput_d01"))
    assert "CHECK PASS" in capsys.readouterr().out
    assert wtune_bp.main(["bp2nc", str(out / "wrfinput_d01"), "--out", str(out)]) == 0
    assert (out / "wrfinput_d01.nc").is_file()


_PROBE = r"""
import json, sys
import adios2
out = {}
with adios2.FileReader(sys.argv[1]) as fr:
    out["nsteps"] = fr.num_steps()
    av = fr.available_variables()
    out["steps"] = {k: int(v["AvailableStepsCount"]) for k, v in av.items()}
    out["Times"] = [fr.read("Times", step_selection=[s, 1]) for s in range(out["nsteps"])]
    md = "md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_"
    out["md"] = [fr.read(md, step_selection=[s, 1]) for s in range(out["nsteps"])]
    out["U_BXS"] = [fr.read("U_BXS", step_selection=[s, 1]).tolist() for s in range(out["nsteps"])]
json.dump(out, sys.stdout)
"""


def test_multi_record_boundary_layout_matches_wrf_writer(tmp_path, capsys):
    """Boundary files: one BP step per Time record, every time-dependent variable (fields,
    Times, md___ strings) present in every step with that record's value -- the layout
    WRF's own io_form=14 real.exe wrfbdy has (scc 2026-10-03). The multi-record WARNING
    (WRF's reader returns step 0 for every field read) reaches the caller's stderr."""
    import json
    import subprocess
    source = tmp_path / "wrfbdy_d01"
    bp = tmp_path / "wrfbdy_d01.bp"
    _make_source(str(source))
    wtune_bp.nc2bp(str(source), str(bp))
    assert "returns BP step 0" in capsys.readouterr().err
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE)
    r = subprocess.run([wtune_bp.require_adios2_python(), str(probe), str(bp)],
                       env=wtune_bp._child_env(), capture_output=True, text=True, check=True)
    got = json.loads(r.stdout)
    assert got["nsteps"] == 2
    assert set(got["steps"].values()) == {2}
    assert got["Times"] == ["2026-01-01_00:00:00", "2026-01-01_03:00:00"]
    assert got["md"] == ["2026-01-01_03:00:00", "2026-01-01_06:00:00"]
    with Dataset(str(source)) as ds:
        for k in range(2):
            assert np.array_equal(np.asarray(got["U_BXS"][k], dtype=np.float32),
                                  ds.variables["U_BXS"][k])


_RECORD_PROBE = r"""
import json, sys
import adios2
out = {}
with adios2.FileReader(sys.argv[1]) as fr:
    out["nsteps"] = fr.num_steps()
    out["Times"] = fr.read("Times")
    out["this"] = fr.read("md___thisbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
    out["next"] = fr.read("md___nextbdytimee_x_t_d_o_m_a_i_n_m_e_t_a_data_")
    out["U_BXS"] = fr.read("U_BXS").tolist()
json.dump(out, sys.stdout)
"""

_DATES = ["2026-01-01_00:00:00", "2026-01-01_03:00:00"]
_NEXT = ["2026-01-01_03:00:00", "2026-01-01_06:00:00"]


def _probe_record(tmp_path, bp):
    import json
    import subprocess
    probe = tmp_path / "record_probe.py"
    probe.write_text(_RECORD_PROBE)
    r = subprocess.run([wtune_bp.require_adios2_python(), str(probe), str(bp)],
                       env=wtune_bp._child_env(), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def test_split_one_bp_per_record_named_by_wrf_date(tmp_path, capsys):
    """split_prefix: one single-step BP per record, named <prefix>_<Times[k]>.bp (the name
    WRF's construct_filename2a builds from "wrfbdy_d<domain>_<date>.bp"), each holding
    exactly record k: fields, Times and both md___ boundary-time strings."""
    source = tmp_path / "wrfbdy_d01"
    out = tmp_path / "split"
    _make_source(str(source))
    wtune_bp.nc2bp(str(source), str(out), split_prefix="wrfbdy_d01")
    assert "returns BP step 0" not in capsys.readouterr().err    # no single-file warning
    assert wtune_bp.record_files(str(out)) == ["wrfbdy_d01_%s.bp" % d for d in _DATES]
    assert not wtune_bp.is_bp(str(out)) and wtune_bp.is_per_record_dir(str(out))
    with Dataset(str(source)) as ds:
        for k, rec in enumerate(wtune_bp.record_files(str(out))):
            assert wtune_bp.bp_engine(str(out / rec)) == "BP4"
            got = _probe_record(tmp_path, out / rec)
            assert got["nsteps"] == 1
            assert got["Times"] == _DATES[k] and got["this"] == _DATES[k]
            assert got["next"] == _NEXT[k]
            assert np.array_equal(np.asarray(got["U_BXS"], dtype=np.float32),
                                  ds.variables["U_BXS"][k])
    r = wtune_bp.roundtrip_check(str(source), str(out), str(tmp_path / "rt"))
    assert r["ok"], r["mismatches"]
    assert r["vars_compared"] == 2 * 8


def test_split_roundtrip_detects_wrong_record(tmp_path):
    """The per-record round trip pairs file i with source record i: swapping two records'
    contents must fail."""
    import shutil
    source = tmp_path / "wrfbdy_d01"
    out = tmp_path / "split"
    _make_source(str(source))
    wtune_bp.nc2bp(str(source), str(out), split_prefix="wrfbdy_d01")
    a, b = [out / r for r in wtune_bp.record_files(str(out))]
    tmp = out / "swap"
    shutil.move(str(a), str(tmp)); shutil.move(str(b), str(a)); shutil.move(str(tmp), str(b))
    r = wtune_bp.roundtrip_check(str(source), str(out), str(tmp_path / "rt"))
    assert not r["ok"]


def test_convert_inputs_per_record_manifest_links_and_cache(tmp_path):
    import json
    case = tmp_path / "case"
    case.mkdir()
    _make_source(str(case / "wrfinput_d01"))
    _make_source(str(case / "wrfbdy_d01"))
    _make_source(str(case / "wrflowinp_d01"))
    cache = tmp_path / "cache"
    m = wtune_bp.convert_inputs(str(case), str(cache))
    man = json.loads((cache / wtune_bp.MANIFEST_NAME).read_text())["entries"]
    assert man["wrfinput_d01"]["layout"] == "single" and "records" not in man["wrfinput_d01"]
    for name in ("wrfbdy_d01", "wrflowinp_d01"):
        assert man[name]["layout"] == "per-record"
        assert man[name]["records"] == ["%s_%s.bp" % (name, d) for d in _DATES]
    links = {}
    for name, target in m.items():
        links.update(wtune_bp.bp_input_links(name, target))
    assert sorted(links) == sorted(["wrfinput_d01.bp"]
                                   + ["wrfbdy_d01_%s.bp" % d for d in _DATES]
                                   + ["wrflowinp_d01_%s.bp" % d for d in _DATES])
    assert all(wtune_bp.is_bp(t) for t in links.values())
    # cached: a second call converts nothing
    stamp = {n: e["converted_at"] for n, e in man.items()}
    mt = os.path.getmtime(links["wrfbdy_d01_%s.bp" % _DATES[0]])
    wtune_bp.convert_inputs(str(case), str(cache))
    man2 = json.loads((cache / wtune_bp.MANIFEST_NAME).read_text())["entries"]
    assert {n: e["converted_at"] for n, e in man2.items()} == stamp
    assert os.path.getmtime(links["wrfbdy_d01_%s.bp" % _DATES[0]]) == mt
