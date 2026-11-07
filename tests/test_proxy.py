import json
import os

import pytest

from wtune import proxy


def test_proxy_shape_matches_the_validated_conus25_and_karina_boxes():
    # conus25 (1501x1201): the boxes used by the passed proxy runs (jobs 2024351/2024531)
    assert proxy.proxy_shape(1501, 1201, 18, 64, 8, 144) == dict(ranks=144, a=6, b=24, nx=504, ny=456)
    assert proxy.proxy_shape(1501, 1201, 9, 64, 4, 144) == dict(ranks=144, a=4, b=36, nx=668, ny=684)
    # karina (901^2) hybrid 224x2 on 4 x 112 cores (job 2024533)
    assert proxy.proxy_shape(901, 901, 8, 28, 4, 112, omp=2) == dict(ranks=56, a=4, b=14, nx=452, ny=462)


def test_proxy_shape_refuses_what_one_node_cannot_reproduce():
    assert proxy.proxy_shape(1501, 1201, 71, 16, 8, 144) is None      # 142 ranks: 2x71, box too wide
    assert proxy.proxy_shape(1501, 1201, 18, 63, 8, 144) is None      # 1134 ranks not divisible by 8
    assert proxy.proxy_shape(901, 901, 8, 28, 4, 112, omp=4) is None  # 56 ranks x 4 threads > 112 cores


def test_nl_set_handles_several_pairs_per_line_and_inserts_decomposition():
    text = (" &time_control\n run_days=0, run_hours=48, run_minutes=0, run_seconds=0,\n /\n"
            " &domains\n time_step=15, max_dom=1, e_we=901, e_sn=901, e_vert=45,\n /\n")
    out = proxy.nl_set(text, e_we=453, run_hours=0, nproc_x=4, nproc_y=28, numtiles=14)
    assert "e_we = 453, e_sn=901" in out and "run_hours = 0, run_minutes=0" in out
    assert " nproc_x = 4,\n nproc_y = 28,\n numtiles = 14,\n" in out
    assert proxy.nl_get(out, "e_vert") == "45"


def test_nl_set_replaces_every_domain_column_and_keeps_one_comma():
    out = proxy.nl_set(" &domains\n e_sn = 1201, 1201,\n e_we                = 1501,\n /\n", e_sn=457, e_we=505)
    assert " e_sn = 457,\n" in out and " e_we = 505,\n" in out


def test_nl_set_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        proxy.nl_set(" &domains\n e_we = 10,\n /\n", radt=5)


def test_time_step_includes_the_fraction():
    assert proxy.time_step_seconds(" time_step = 0,\n time_step_fract_num = 2,\n time_step_fract_den = 3,\n") \
        == pytest.approx(2 / 3)


def _tree(tmp_path, name, omp_line):
    t = tmp_path / name
    (t / "run").mkdir(parents=True)
    (t / "configure.wrf").write_text("FC = nvfortran\nOMP             =       %s\n" % omp_line)
    return str(t)


def test_build_openmp_detection_reads_configure_wrf(tmp_path):
    assert proxy.build_has_openmp(_tree(tmp_path, "dmsm", "-mp -Minfo=mp")) is True
    assert proxy.build_has_openmp(_tree(tmp_path, "dmpar", "# -mp -Minfo=mp -Mrecursive")) is False
    assert proxy.build_has_openmp(str(tmp_path / "missing")) is None


def _case(tmp_path, max_dom=1):
    c = tmp_path / "case"
    c.mkdir()
    (c / "namelist.input").write_text(
        " &domains\n time_step = 15,\n max_dom = %d,\n e_we = 1501,\n e_sn = 1201,\n /\n" % max_dom)
    return str(c)


def test_plan_refuses_omp_on_a_build_without_openmp(tmp_path):
    dmpar = _tree(tmp_path, "dmpar", "# -mp")
    plan = proxy.make_plan(_case(tmp_path), 4, ["9x64", "9x32:omp=2:nt=24", "71x8"], 144, wrf_tree=dmpar)
    assert [r["tag"] for r in plan["runs"]] == ["9x64_o1_nt1", "9x32_o2_nt24"]
    assert plan["skipped"] == ["71x8"]
    errs = proxy.check_builds(plan)
    assert len(errs) == 1 and "9x32_o2_nt24" in errs[0] and "without OpenMP" in errs[0]
    dmsm = _tree(tmp_path, "dmsm", "-mp")
    os.makedirs(tmp_path / "b")
    plan = proxy.make_plan(_case(tmp_path / "b"), 4, ["9x32:omp=2:nt=24"], 144, wrf_tree=dmpar,
                           wrf_tree_omp=dmsm)
    assert plan["runs"][0]["wrf_tree"] == dmsm and proxy.check_builds(plan) == []


def test_plan_refuses_nested_cases(tmp_path):
    with pytest.raises(ValueError, match="single-domain"):
        proxy.make_plan(_case(tmp_path, max_dom=2), 4, ["9x64"], 144)


def test_collect_ranks_by_median_and_drops_failed_runs(tmp_path):
    rows = [dict(tag="A", rep=1, rc=0, median=0.47), dict(tag="A", rep=2, rc=0, median=0.48),
            dict(tag="H", rep=1, rc=0, median=0.40), dict(tag="X", rep=1, rc=4, median=None)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = proxy.collect(str(tmp_path))
    assert [r["tag"] for r in out] == ["H", "A"] and out[1]["reps"] == [0.47, 0.48]


def test_render_sbatch_runs_the_python_worker_under_site_env(tmp_path, monkeypatch):
    import wtune.machine
    from wtune.tools import scc_sweep
    monkeypatch.setattr(wtune.machine, "partition", lambda: "arm144")
    monkeypatch.setattr(wtune.machine, "cores_per_node", lambda: 144)
    monkeypatch.setattr(wtune.machine, "site", lambda: "f1arm")
    monkeypatch.setattr(scc_sweep, "scheduler_args", lambda: "--account=acd110018")
    s = proxy.render_sbatch(str(tmp_path / "plan.json"), "/env/bin/python", "/repo/campaign")
    assert "#SBATCH -p arm144 -N 1 --exclusive --ntasks-per-node=144 --time=01:00:00 --account=acd110018" in s
    assert 'source "$CAMPAIGN/sites/f1arm.sh"\n' in s and "\nsite_env\n" in s
    assert "/env/bin/python -m wtune.proxy _worker %s" % (tmp_path / "plan.json") in s


def test_crop_couples_boundaries_with_dry_air_mass(tmp_path):
    nc = pytest.importorskip("netCDF4")
    np = pytest.importorskip("numpy")
    NX, NY, NZ, BW = 12, 10, 3, 2
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    src.mkdir()
    I = nc.Dataset(src / "wrfinput_d01", "w", format="NETCDF3_64BIT_OFFSET")
    for d, n in (("Time", None), ("DateStrLen", 19), ("west_east", NX), ("west_east_stag", NX + 1),
                 ("south_north", NY), ("south_north_stag", NY + 1), ("bottom_top", NZ)):
        I.createDimension(d, n)
    I.setncattr("WEST-EAST_GRID_DIMENSION", np.int32(NX + 1))
    I.setncattr("SOUTH-NORTH_GRID_DIMENSION", np.int32(NY + 1))
    t = I.createVariable("Times", "S1", ("Time", "DateStrLen"))
    t[0, :] = np.array(list("2019-11-26_18:00:00"), dtype="S1")
    shapes = {"U": ("bottom_top", "south_north", "west_east_stag"), "V": ("bottom_top", "south_north_stag", "west_east"),
              "T": ("bottom_top", "south_north", "west_east"), "MU": ("south_north", "west_east"),
              "MUB": ("south_north", "west_east"), "MAPFAC_UY": ("south_north", "west_east_stag"),
              "MAPFAC_VX": ("south_north_stag", "west_east"), "MAPFAC_MY": ("south_north", "west_east")}
    vals = {}
    for k, dims in shapes.items():
        v = I.createVariable(k, "f4", ("Time",) + dims)
        shape = [I.dimensions[d].size for d in dims]
        a = rng.uniform(0.9, 1.1, shape) if k.startswith("MAPFAC") else rng.normal(1e3 if k == "MUB" else 5, 1, shape)
        vals[k] = a
        v[0] = a
    I.close()
    B = nc.Dataset(src / "wrfbdy_d01", "w", format="NETCDF3_64BIT_OFFSET")
    for d, n in (("Time", None), ("DateStrLen", 19), ("west_east", NX), ("west_east_stag", NX + 1),
                 ("south_north", NY), ("south_north_stag", NY + 1), ("bottom_top", NZ), ("bdy_width", BW)):
        B.createDimension(d, n)
    B.setncattr("INTERVAL_SECONDS", np.int32(10800))
    B.createVariable("Times", "S1", ("Time", "DateStrLen"))
    for name, dims in (("U_BXS", ("bdy_width", "bottom_top", "south_north")), ("U_BTXS", ("bdy_width", "bottom_top", "south_north")),
                       ("T_BYE", ("bdy_width", "bottom_top", "west_east")), ("MU_BXE", ("bdy_width", "south_north")),
                       ("HT_SHAD_BXS", ("bdy_width", "south_north"))):
        B.createVariable(name, "f4", ("Time",) + dims)
    B.close()
    out = tmp_path / "out"
    out.mkdir()
    zero = proxy.crop(str(src / "wrfinput_d01"), str(src / "wrfbdy_d01"), str(out), 8, 6)
    assert zero == ["HT_SHAD"]
    C = nc.Dataset(out / "wrfinput_d01")
    assert C.dimensions["west_east"].size == 8 and C.getncattr("WEST-EAST_GRID_DIMENSION") == 9
    u, mu = C.variables["U"][0].astype("f8"), (C.variables["MU"][0] + C.variables["MUB"][0]).astype("f8")
    np.testing.assert_allclose(u, vals["U"][:, 2:8, 2:11], rtol=1e-6)     # centre crop: i0=2, j0=2
    D = nc.Dataset(out / "wrfbdy_d01")
    mu_u0 = mu[:, 0]                                                       # west edge u point: nearest mass column
    expect = u[:, :, 0] * mu_u0[None, :] / C.variables["MAPFAC_UY"][0][:, 0]
    np.testing.assert_allclose(D.variables["U_BXS"][0, 0], expect, rtol=1e-5)
    np.testing.assert_allclose(D.variables["MU_BXE"][0, 0], C.variables["MU"][0][:, -1], rtol=1e-6)
    assert not D.variables["U_BTXS"][:].any() and D.variables["T_BYE"][0, 0].mean() > 1e3   # coupled, not raw ~5
