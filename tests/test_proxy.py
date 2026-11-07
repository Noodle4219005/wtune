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


def test_band_shape_keeps_full_width_and_three_node_rows():
    # Big Chill d01 (561x450) at N=5, 24x30: 144 ranks per node = 6 whole rows of 24 -> 3 nodes x 6 rows
    assert proxy.band_shape(561, 450, 24, 30, 5, 144) == dict(
        ranks=432, a=24, b=18, nx=560, ny=270, nodes=3, ranks_per_node=144)
    # conus25 (1501x1201) at N=8, 18x64: 8 rows per node
    assert proxy.band_shape(1501, 1201, 18, 64, 8, 144) == dict(
        ranks=432, a=18, b=24, nx=1500, ny=456, nodes=3, ranks_per_node=144)


def test_band_shape_refuses_partial_rows_and_targets_no_larger_than_the_band():
    assert proxy.band_shape(561, 450, 16, 18, 4, 144, omp=2) is None   # 72 ranks: 4.5 rows of 16
    assert proxy.band_shape(561, 450, 18, 24, 3, 144) is None          # target has only 3 nodes
    assert proxy.band_shape(561, 450, 24, 30, 5, 144, omp=2) is None   # 144 ranks x 2 threads > 144 cores


def test_band_plan_runs_on_several_nodes(tmp_path, monkeypatch):
    import wtune.machine
    from wtune.tools import scc_sweep
    plan = proxy.make_plan(_case(tmp_path), 8, ["18x64", "16x27"], 144, proxy_nodes=3)
    assert [(r["tag"], r["nodes"], r["ranks_per_node"], r["ranks"]) for r in plan["runs"]] == [
        ("18x64_o1_nt1", 3, 144, 432)]
    assert plan["skipped"] == ["16x27"] and plan["proxy_nodes"] == 3   # 54 ranks per node: not whole rows of 16
    monkeypatch.setattr(wtune.machine, "partition", lambda: "arm576")
    monkeypatch.setattr(wtune.machine, "cores_per_node", lambda: 144)
    monkeypatch.setattr(wtune.machine, "site", lambda: "f1arm")
    monkeypatch.setattr(scc_sweep, "scheduler_args", lambda: "")
    s = proxy.render_sbatch(str(tmp_path / "plan.json"), "/env/bin/python", "/repo/campaign", nodes=3)
    assert "#SBATCH -p arm576 -N 3 --exclusive" in s and "export SLURM_JOB_NUM_NODES=3\n" in s


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


def _tree(tmp_path, name):
    t = tmp_path / name
    (t / "run").mkdir(parents=True)
    return str(t)


def _case(tmp_path, max_dom=1):
    c = tmp_path / "case"
    c.mkdir()
    (c / "namelist.input").write_text(
        " &domains\n time_step = 15,\n max_dom = %d,\n e_we = 1501,\n e_sn = 1201,\n /\n" % max_dom)
    return str(c)


def test_plan_refuses_omp_on_a_build_without_openmp(tmp_path, monkeypatch):
    import wtune.extensions
    monkeypatch.setattr(wtune.extensions, "omp_cap_for_build", lambda t: 1 if t.endswith("dmpar") else None)
    dmpar = _tree(tmp_path, "dmpar")
    plan = proxy.make_plan(_case(tmp_path), 4, ["9x64", "9x32:omp=2:nt=24", "71x8"], 144, wrf_tree=dmpar)
    assert [r["tag"] for r in plan["runs"]] == ["9x64_o1_nt1", "9x32_o2_nt24"]
    assert plan["skipped"] == ["71x8"]
    errs = proxy.check_builds(plan)
    assert len(errs) == 1 and "9x32_o2_nt24" in errs[0] and "no OpenMP" in errs[0]
    dmsm = _tree(tmp_path, "dmsm")
    os.makedirs(tmp_path / "b")
    plan = proxy.make_plan(_case(tmp_path / "b"), 4, ["9x32:omp=2:nt=24"], 144, wrf_tree=dmpar,
                           wrf_tree_omp=dmsm)
    assert plan["runs"][0]["wrf_tree"] == dmsm and proxy.check_builds(plan) == []


MARSHALL = (" &time_control\n run_hours = 5,\n history_interval = -1, 1,\n /\n &domains\n time_step = 6,\n"
            " time_step_fract_num = 0,\n time_step_fract_den = 1,\n max_dom = 2,\n e_we = 505, 505,\n"
            " e_sn = 505, 505,\n e_vert = 45, 45,\n dx = 1000, 111.1111,\n dy = 1000, 111.1111,\n"
            " grid_id = 1, 2,\n parent_id = 0, 1,\n i_parent_start = 1, 239,\n parent_grid_ratio = 1, 9,\n"
            " parent_time_step_ratio = 1, 9,\n /\n &physics\n mp_physics = 6, 8,\n bl_pbl_physics = 1, 0,\n /\n"
            " &bdy_control\n specified = .true., .false.,\n nested = .false., .true.,\n /\n")


def test_nested_case_is_split_into_weighted_single_domain_runs(tmp_path):
    c = tmp_path / "case"
    c.mkdir()
    (c / "namelist.input").write_text(MARSHALL)
    plan = proxy.make_plan(str(c), 8, ["32x32"], 144)
    assert [(r["tag"], r["domain"], r["weight"]) for r in plan["runs"]] == [("32x32_o1_nt1_d1", 1, 1), ("32x32_o1_nt1_d2", 2, 9)]
    assert plan["runs"][1]["dt"] == pytest.approx(6 / 9) and plan["max_dom"] == 2


def test_single_domain_namelist_takes_the_nest_column():
    layer = proxy.domain_layers(MARSHALL)[1]
    out = proxy.single_domain_namelist(MARSHALL, layer, 128, 256, 8, 16, 14, 27)
    for key, val in (("max_dom", "1"), ("e_we", "129"), ("e_sn", "257"), ("time_step", "0"),
                     ("time_step_fract_num", "2"), ("time_step_fract_den", "3"), ("mp_physics", "8"),
                     ("bl_pbl_physics", "0"), ("specified", ".true."), ("nested", ".false."),
                     ("parent_time_step_ratio", "1"), ("numtiles", "14"), ("run_seconds", "27")):
        assert proxy.nl_get(out, key) == val, key
    assert float(proxy.nl_get(out, "dx")) > 111.1111          # nudged past WRF's dt <= 6*dx[km] bound
    assert proxy.nl_values(out, "e_we") == ["129"]


def test_collect_adds_domains_by_weight_and_drops_incomplete_layouts(tmp_path):
    json.dump(dict(max_dom=2), open(tmp_path / "plan.json", "w"))
    rows = [dict(tag="A_d1", layout="A", domain=1, weight=1, rep=1, rc=0, median=0.06),
            dict(tag="A_d2", layout="A", domain=2, weight=9, rep=1, rc=0, median=0.05),
            dict(tag="B_d1", layout="B", domain=1, weight=1, rep=1, rc=0, median=0.05),
            dict(tag="B_d2", layout="B", domain=2, weight=9, rep=1, rc=1, median=None)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = proxy.collect(str(tmp_path))
    assert [r["tag"] for r in out] == ["A"] and out[0]["median"] == pytest.approx(0.06 + 9 * 0.05)


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


def test_nest_dx_only_moves_when_dt_sits_on_the_bound():
    assert proxy.nest_dx(dict(domain=2, dx=600.0, dy=600.0, dt=3.0)) == (600.0, 600.0)   # Big Chill d02
    dx, dy = proxy.nest_dx(dict(domain=2, dx=111.1111, dy=111.1111, dt=6 / 9))            # marshall d02
    assert 6 * dx / 1000 >= 6 / 9 and dx < 111.2
    assert proxy.nest_dx(dict(domain=1, dx=1000.0, dy=1000.0, dt=6.0)) == (1000.0, 1000.0)
