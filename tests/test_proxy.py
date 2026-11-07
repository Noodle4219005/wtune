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
    assert proxy.band_shape(561, 450, 18, 24, 2, 144, proxy_nodes=3) is None   # band bigger than the target
    # proxy_nodes == nodes: the whole domain on the target's own nodes, any rank layout
    assert proxy.band_shape(561, 450, 16, 18, 2, 144, proxy_nodes=2) == dict(
        ranks=288, a=16, b=18, nx=560, ny=449, nodes=2, ranks_per_node=144)
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
    plan = proxy.make_plan(str(c), 8, ["32x32"], 144, nested="split")
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
    D.close()
    wide = tmp_path / "wide"
    wide.mkdir()
    proxy.crop(str(src / "wrfinput_d01"), str(src / "wrfbdy_d01"), str(wide), 8, 6, bdy_width=3)
    W = nc.Dataset(wide / "wrfbdy_d01")
    assert W.dimensions["bdy_width"].size == 3                            # wider than the source's 2
    np.testing.assert_allclose(W.variables["U_BXS"][0, 2], u[:, :, 2] * (0.5 * (mu[:, 1] + mu[:, 2]))[None, :]
                               / C.variables["MAPFAC_UY"][0][:, 2], rtol=1e-5)


def test_flat_offset_moves_the_crop_edges_off_the_ridge():
    np = pytest.importorskip("numpy")
    hgt = np.zeros((40, 50))
    hgt[:, 20:26] = np.linspace(0, 3000, 6)[None, :]          # a ridge crossing the domain centre
    i0, j0 = proxy.flat_offset(hgt, 15, 30, 3)
    assert i0 + 15 <= 20 or i0 >= 26                           # the window avoids the ridge
    assert proxy.flat_offset(np.zeros((40, 50)), 15, 30, 3) == (17, 5)   # flat everywhere: the centre


def test_wide_boundary_keys_go_to_bdy_control():
    nl = proxy.nl_force(" &domains\n e_we = 10,\n /\n &bdy_control\n specified = .true.,\n /\n",
                        spec_bdy_width=10, relax_zone=9, spec_exp=0.33)
    bdy = nl.split("&bdy_control")[1]
    assert "spec_bdy_width = 10" in bdy and "relax_zone = 9" in bdy and "spec_exp = 0.33" in bdy


def test_nest_dx_only_moves_when_dt_sits_on_the_bound():
    assert proxy.nest_dx(dict(domain=2, dx=600.0, dy=600.0, dt=3.0)) == (600.0, 600.0)   # Big Chill d02
    dx, dy = proxy.nest_dx(dict(domain=2, dx=111.1111, dy=111.1111, dt=6 / 9))            # marshall d02
    assert 6 * dx / 1000 >= 6 / 9 and dx < 111.2
    assert proxy.nest_dx(dict(domain=1, dx=1000.0, dy=1000.0, dt=6.0)) == (1000.0, 1000.0)


def test_null_rsl_links_every_rank_but_zero_to_dev_null(tmp_path):
    proxy._null_rsl(str(tmp_path), 3)
    assert sorted(os.listdir(tmp_path)) == ["rsl.error.0001", "rsl.error.0002", "rsl.out.0001", "rsl.out.0002"]
    assert all(os.readlink(tmp_path / f) == os.devnull for f in os.listdir(tmp_path))


BIGCHILL = (" &time_control\n run_days = 0, run_hours = 4, run_minutes = 0, run_seconds = 0,\n history_interval = 60, 60,\n /\n"
            " &domains\n time_step = 15,\n"
            " max_dom = 2,\n e_we = 561, 671,\n e_sn = 450, 791,\n e_vert = 86, 86,\n dx = 3000, 600,\n"
            " dy = 3000, 600,\n grid_id = 1, 2,\n parent_id = 0, 1,\n i_parent_start = 1, 150,\n"
            " j_parent_start = 1, 80,\n parent_grid_ratio = 1, 5,\n parent_time_step_ratio = 1, 5,\n /\n"
            " &bdy_control\n spec_bdy_width = 5,\n specified = .true., .false.,\n nested = .false., .true.,\n /\n")


def _check_nest_crop(text, nb, px, py, margin):
    layers = proxy.domain_layers(text)
    ratio = [1] + [int(v) for v in proxy.nl_values(text, "parent_grid_ratio")[1:]]
    ips = [int(v) for v in proxy.nl_values(text, "i_parent_start")]
    jps = [int(v) for v in proxy.nl_values(text, "j_parent_start")]
    for ly, bx in zip(layers, nb["boxes"]):
        d, r = ly["domain"], ratio[ly["domain"] - 1]
        mx, my = ly["e_we"] - 1, ly["e_sn"] - 1
        # each rank owns the target's patch, up to the rounding to the parent ratio
        assert 0 <= bx["nx"] - nb["a"] * -(-mx // px) < r and 0 <= bx["ny"] - nb["b"] * -(-my // py) < r
        assert bx["nx"] % r == 0 and bx["ny"] % r == 0 and bx["i0"] % r == 0 and bx["j0"] % r == 0
        assert 0 <= bx["i0"] <= mx - bx["nx"] and 0 <= bx["j0"] <= my - bx["ny"]
        if d == 1:
            assert (bx["i_parent_start"], bx["j_parent_start"]) == (1, 1)
            continue
        p = nb["boxes"][d - 2]
        # the same parent cell before and after cropping
        assert p["i0"] + bx["i_parent_start"] == ips[d - 1] + bx["i0"] // r
        assert p["j0"] + bx["j_parent_start"] == jps[d - 1] + bx["j0"] // r
        assert bx["i_parent_start"] - 1 >= margin and p["nx"] - (bx["i_parent_start"] - 1 + bx["nx"] // r) >= margin
        assert bx["j_parent_start"] - 1 >= margin and p["ny"] - (bx["j_parent_start"] - 1 + bx["ny"] // r) >= margin


def test_nest_crop_boxes_give_every_rank_the_targets_patch_in_both_big_chill_domains():
    nb = proxy.nest_crop_boxes(BIGCHILL, 16, 18, 2, 144)
    assert (nb["ranks"], nb["a"], nb["b"]) == (144, 12, 12)
    assert nb["boxes"] == [dict(domain=1, nx=420, ny=300, i0=6, j0=8, i_parent_start=1, j_parent_start=1),
                           dict(domain=2, nx=505, ny=530, i0=80, j0=130, i_parent_start=160, j_parent_start=98)]
    _check_nest_crop(BIGCHILL, nb, 16, 18, 6)
    nb = proxy.nest_crop_boxes(BIGCHILL, 12, 12, 2, 144, omp=2)
    assert nb["ranks"] == 72 and nb["a"] * nb["b"] == 72
    _check_nest_crop(BIGCHILL, nb, 12, 12, 6)


def test_nest_crop_boxes_walk_up_a_three_domain_chain():
    text = BIGCHILL.replace("max_dom = 2", "max_dom = 3").replace("671,", "671, 601,").replace("791,", "791, 601,")
    for k, v in (("parent_id = 0, 1", "0, 1, 2"), ("i_parent_start = 1, 150", "1, 150, 30"),
                 ("j_parent_start = 1, 80", "1, 80, 90"), ("parent_grid_ratio = 1, 5", "1, 5, 3"),
                 ("parent_time_step_ratio = 1, 5", "1, 5, 3")):
        text = text.replace(k, k.split("=")[0] + "= " + v)
    nb = proxy.nest_crop_boxes(text, 16, 18, 2, 144)
    assert [bx["domain"] for bx in nb["boxes"]] == [1, 2, 3]
    _check_nest_crop(text, nb, 16, 18, 6)


def test_nest_crop_boxes_refuse_what_does_not_fit():
    # d02 over nearly all of d01: its crop's footprint leaves no 6-cell margin in d01's crop
    big = BIGCHILL.replace("671,", "2741,").replace("i_parent_start = 1, 150", "i_parent_start = 1, 7")
    assert proxy.nest_crop_boxes(big, 16, 18, 2, 144) is None
    assert proxy.nest_crop_boxes(BIGCHILL, 71, 16, 8, 144) is None                  # no one-node shape
    assert proxy.nest_crop_boxes(BIGCHILL.replace(" j_parent_start = 1, 80,\n", ""), 16, 18, 2, 144) is None
    sib = BIGCHILL.replace("max_dom = 2", "max_dom = 3").replace("parent_id = 0, 1", "parent_id = 0, 1, 1")
    assert proxy.nest_crop_boxes(sib, 16, 18, 2, 144) is None                        # siblings, not a chain


def test_nested_namelist_sets_every_domain_column():
    nb = proxy.nest_crop_boxes(BIGCHILL, 16, 18, 2, 144)
    out = proxy.nested_namelist(BIGCHILL, nb["boxes"], 12, 12, 2, 450)
    for key, vals in (("max_dom", ["2"]), ("e_we", ["421", "506"]), ("e_sn", ["301", "531"]),
                      ("i_parent_start", ["1", "160"]), ("j_parent_start", ["1", "98"]),
                      ("history_interval", ["100000", "100000"]), ("run_hours", ["0"]), ("run_seconds", ["450"]),
                      ("nproc_x", ["12"]), ("nproc_y", ["12"]), ("numtiles", ["2"]),
                      ("parent_grid_ratio", ["1", "5"]), ("dx", ["3000", "600"]), ("nested", [".false.", ".true."])):
        assert proxy.nl_values(out, key) == vals, key


def test_nested_plan_is_one_cropped_run_per_layout_and_split_still_works(tmp_path):
    c = tmp_path / "case"
    c.mkdir()
    (c / "namelist.input").write_text(BIGCHILL)
    plan = proxy.make_plan(str(c), 2, ["16x18", "12x12:omp=2"], 144, steps=70)
    assert [(r["tag"], r["layout"], r["domain"], r["weight"], len(r["boxes"])) for r in plan["runs"]] == [
        ("16x18_o1_nt1", "16x18_o1_nt1", 1, 1, 2), ("12x12_o2_nt1", "12x12_o2_nt1", 1, 1, 2)]
    assert (plan["runs"][0]["nx"], plan["runs"][0]["ny"], plan["runs"][0]["ranks"]) == (420, 300, 144)
    assert plan["steps"] == proxy.PROXY_MAX_STEPS == 10
    split = proxy.make_plan(str(c), 2, ["16x18"], 144, nested="split")
    assert [(r["tag"], r["weight"], "boxes" in r) for r in split["runs"]] == [
        ("16x18_o1_nt1_d1", 1, False), ("16x18_o1_nt1_d2", 5, False)]
    full = proxy.make_plan(str(c), 2, ["16x18"], 144, proxy_nodes=2)       # the whole nested case on 2 nodes
    assert [(r["tag"], r["nodes"], r["ranks"], len(r["boxes"])) for r in full["runs"]] == [("16x18_o1_nt1", 2, 288, 2)]
    split = proxy.make_plan(str(c), 2, ["16x18"], 144, proxy_nodes=2, nested="split")
    assert [r["tag"] for r in split["runs"]] == ["16x18_o1_nt1_d1", "16x18_o1_nt1_d2"]


def test_collect_takes_a_nested_crop_run_as_the_whole_layout(tmp_path):
    json.dump(dict(max_dom=2), open(tmp_path / "plan.json", "w"))
    rows = [dict(tag="A", layout="A", domain=1, weight=1, rep=1, rc=0, median=0.9, nested="crop"),
            dict(tag="B_d1", layout="B", domain=1, weight=1, rep=1, rc=0, median=0.3)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [(r["tag"], r["median"]) for r in proxy.collect(str(tmp_path))] == [("A", 0.9)]


def test_whole_domain_box_uses_the_cases_own_files(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(proxy, "crop", lambda *a, **k: calls.append((os.path.basename(a[0]), a[1], k)))
    plan = dict(case_dir="/case", e_we=561, e_sn=450, max_dom=2, place="centre", bdy_width=10)
    full = dict(domain=1, nx=560, ny=449)
    assert proxy.whole_domain(plan, full) and not proxy.whole_domain(plan, dict(full, domain=2))
    assert not proxy.whole_domain(plan, dict(full, ny=448))
    assert proxy._inputs(plan, full, str(tmp_path), {}) == {"wrfinput_d01": "/case/wrfinput_d01",
                                                         "wrfbdy_d01": "/case/wrfbdy_d01"}
    assert calls == []
    part = proxy._inputs(plan, dict(full, nx=420, ny=300), str(tmp_path), {})
    assert calls[0][0] == "wrfinput_d01" and part["wrfbdy_d01"].startswith(str(tmp_path))


def test_nested_crop_run_crops_every_domain_at_its_offsets(tmp_path, monkeypatch):
    case = tmp_path / "case"
    case.mkdir()
    (case / "wrfinput_d02").write_text("x")
    calls = []
    monkeypatch.setattr(proxy, "crop", lambda *a, **k: calls.append((os.path.basename(a[0]), a[1], a[3:5], k)))
    nb = proxy.nest_crop_boxes(BIGCHILL, 16, 18, 2, 144)
    plan = dict(case_dir=str(case), e_we=561, e_sn=450, max_dom=2)
    files = proxy._inputs(plan, dict(domain=1, nx=420, ny=300, boxes=nb["boxes"]), str(tmp_path / "s"), {})
    assert sorted(files) == ["wrfbdy_d01", "wrfinput_d01", "wrfinput_d02"]
    (n1, bdy1, box1, k1), (n2, bdy2, box2, k2) = calls
    assert (n1, box1, k1["i0"], k1["j0"], k1["parent_start"]) == ("wrfinput_d01", (420, 300), 6, 8, None)
    assert bdy1.endswith("wrfbdy_d01") and bdy2 is None
    assert (n2, box2, k2["i0"], k2["j0"], k2["parent_start"]) == ("wrfinput_d02", (505, 530), 80, 130, (160, 98))


def test_crop_at_offsets_keeps_a_nest_a_nest(tmp_path):
    nc = pytest.importorskip("netCDF4")
    np = pytest.importorskip("numpy")
    src = tmp_path / "wrfinput_d02"
    I = nc.Dataset(src, "w", format="NETCDF3_64BIT_OFFSET")
    for d, n in (("Time", None), ("west_east", 12), ("west_east_stag", 13), ("south_north", 10),
                 ("south_north_stag", 11)):
        I.createDimension(d, n)
    for k, v in (("WEST-EAST_GRID_DIMENSION", 13), ("SOUTH-NORTH_GRID_DIMENSION", 11), ("GRID_ID", 2),
                 ("PARENT_ID", 1), ("PARENT_GRID_RATIO", 3), ("I_PARENT_START", 40), ("J_PARENT_START", 50)):
        I.setncattr(k, np.int32(v))
    u = np.arange(10 * 13, dtype="f4").reshape(10, 13)
    I.createVariable("U", "f4", ("Time", "south_north", "west_east_stag"))[0] = u
    I.close()
    out = tmp_path / "out"
    out.mkdir()
    assert proxy.crop(str(src), None, str(out), 6, 3, i0=3, j0=6, parent_start=(7, 9)) == []
    assert sorted(os.listdir(out)) == ["wrfinput_d02"]
    C = nc.Dataset(out / "wrfinput_d02")
    np.testing.assert_array_equal(C.variables["U"][0], u[6:9, 3:10])
    assert (C.getncattr("I_PARENT_START"), C.getncattr("J_PARENT_START"), C.getncattr("GRID_ID")) == (7, 9, 2)
    assert C.getncattr("WEST-EAST_GRID_DIMENSION") == 7 and C.getncattr("PARENT_GRID_RATIO") == 3


def test_nested_namelist_keeps_a_history_interval_per_domain():
    text = (" &time_control\n run_days = 0,\n run_hours = 11,\n run_minutes = 0,\n run_seconds = 0,\n"
            " history_interval = 60, 10, 10,\n restart_interval = 720,\n /\n"
            " &domains\n max_dom = 2,\n e_we = 561, 671, 400,\n e_sn = 450, 791, 400,\n"
            " i_parent_start = 1, 240, 1,\n j_parent_start = 1, 115, 1,\n /\n")
    boxes = [dict(domain=1, nx=320, ny=406, i_parent_start=1, j_parent_start=1),
             dict(domain=2, nx=385, ny=700, i_parent_start=122, j_parent_start=124)]
    nl = proxy.nested_namelist(text, boxes, 8, 14, 1, 150)
    assert proxy.nl_values(nl, "history_interval") == ["100000", "100000", "10"]
    assert proxy.nl_values(nl, "restart_interval") == ["100000"]
    assert proxy.nl_values(nl, "e_we") == ["321", "386", "400"]


BIGCHILL_ISC = BIGCHILL.replace("i_parent_start = 1, 150", "i_parent_start = 1, 240").replace(
    "j_parent_start = 1, 80", "j_parent_start = 1, 115")


def test_nested_band_is_full_width_with_three_nodes_of_rows_in_every_domain():
    # Big Chill on x86 (112 cores) at N=4, 16x28: 7 rows of 16 per node -> 16 x 21 ranks on 3 nodes
    nb = proxy.nest_crop_boxes(BIGCHILL_ISC, 16, 28, 4, 112, proxy_nodes=3)
    assert (nb["ranks"], nb["a"], nb["b"], nb["nodes"], nb["ranks_per_node"]) == (336, 16, 21, 3, 112)
    assert nb["boxes"] == [dict(domain=1, nx=560, ny=357, i0=0, j0=15, i_parent_start=1, j_parent_start=1),
                           dict(domain=2, nx=670, ny=610, i0=0, j0=90, i_parent_start=240, j_parent_start=118)]
    d1, d2 = nb["boxes"]
    assert d2["ny"] == -(-21 * -(-790 // 28) // 5) * 5          # b x ceil(m/py), rounded up to the ratio
    # the same d01 cell under the nest's corner, and the margin to the crop's edge
    assert d1["j0"] + d2["j_parent_start"] == 115 + d2["j0"] // 5
    assert d2["j_parent_start"] - 1 >= 6 and d1["ny"] - (d2["j_parent_start"] - 1 + d2["ny"] // 5) >= 6
    assert proxy.nest_crop_boxes(BIGCHILL_ISC, 16, 18, 4, 144, omp=2, proxy_nodes=3) is None   # 4.5 rows per node


def test_nested_band_on_the_targets_own_nodes_is_the_whole_case(tmp_path, monkeypatch):
    nb = proxy.nest_crop_boxes(BIGCHILL_ISC, 16, 28, 4, 112, proxy_nodes=4)
    assert nb["boxes"] == [dict(domain=1, nx=560, ny=449, i0=0, j0=0, i_parent_start=1, j_parent_start=1),
                           dict(domain=2, nx=670, ny=790, i0=0, j0=0, i_parent_start=240, j_parent_start=115)]
    case = tmp_path / "case"
    case.mkdir()
    (case / "wrfinput_d02").write_text("x")
    monkeypatch.setattr(proxy, "crop", lambda *a, **k: pytest.fail("a whole case is not cropped"))
    plan = dict(case_dir=str(case), e_we=561, e_sn=450, max_dom=2)
    layers = {ly["domain"]: ly for ly in proxy.domain_layers(BIGCHILL_ISC)}
    r = dict(domain=1, nx=560, ny=449, boxes=nb["boxes"])
    assert proxy._inputs(plan, r, str(tmp_path / "s"), layers) == {
        f: str(case / f) for f in ("wrfinput_d01", "wrfbdy_d01", "wrfinput_d02")}
    nl = proxy.nested_namelist(BIGCHILL_ISC, nb["boxes"], 16, 28, 1, 150)
    assert proxy.nl_values(nl, "e_we") == ["561", "671"] and proxy.nl_values(nl, "j_parent_start") == ["1", "115"]


def test_nested_band_plan_runs_one_nested_run_on_the_band_nodes(tmp_path):
    c = tmp_path / "case"
    c.mkdir()
    (c / "namelist.input").write_text(BIGCHILL_ISC)
    plan = proxy.make_plan(str(c), 4, ["16x28", "16x18"], 112, proxy_nodes=3)
    assert [(r["tag"], r["nodes"], r["ranks_per_node"], r["ranks"], (r["nx"], r["ny"])) for r in plan["runs"]] == [
        ("16x28_o1_nt1", 3, 112, 336, (560, 357))]
    # 16x18: 72 ranks per node, not whole rows of 16 -- no band, nested or split
    assert plan["skipped"] == ["16x18"]


def test_a_failed_run_is_rerun_once_with_every_ranks_rsl_and_the_job_goes_on(tmp_path, monkeypatch):
    import subprocess
    import wtune.machine
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(" &time_control\n run_days = 0, run_hours = 1, run_minutes = 0, run_seconds = 0,\n /\n &domains\n time_step = 15,\n"
                                         " max_dom = 1,\n e_we = 101,\n e_sn = 81,\n /\n")
    tree = _tree(tmp_path, "dmpar")
    run = dict(domain=1, weight=1, dt=15.0, px=2, py=2, omp=1, numtiles=1, ranks=4, a=2, b=2, nx=100, ny=80,
               wrf_tree=tree)
    plan = dict(case_dir=str(case), e_we=101, e_sn=81, max_dom=1, dt=15.0, steps=10, reps=1,
                runs=[dict(run, tag="A", layout="A"), dict(run, tag="B", layout="B")])
    out = tmp_path / "job"
    out.mkdir()
    json.dump(plan, open(out / "plan.json", "w"))
    monkeypatch.setenv("WTUNE_PROXY_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setattr(wtune.machine, "load_user", lambda: {})
    monkeypatch.setattr(wtune.machine, "export", lambda m: None)
    monkeypatch.setattr(wtune.machine, "site_launch_line", lambda *a: "mpirun ./wrf.exe")
    seen = []

    def fake(cmd, cwd=None, **k):
        if cmd[0] == "rm":
            return subprocess.CompletedProcess(cmd, 0)
        seen.append((os.path.basename(cwd), os.path.islink(os.path.join(cwd, "rsl.error.0001"))))
        bad = os.path.basename(cwd) == "A-1"                   # A's first attempt dies at step 3
        open(os.path.join(cwd, "rsl.error.0000"), "w").write("".join(
            "Timing for main: time 2019 on domain   1:    %.5f elapsed seconds\n" % (0.5 + 0.01 * i)
            for i in range(3 if bad else 10)))
        if cwd.endswith("-retry"):                               # rank 1 writes its own rsl file
            open(os.path.join(cwd, "rsl.error.0001"), "w").write("x")
        return subprocess.CompletedProcess(cmd, 1 if bad else 0)
    monkeypatch.setattr(subprocess, "run", fake)
    assert proxy.worker(str(out / "plan.json")) == 0
    assert seen == [("A-1", True), ("A-1-retry", False), ("B-1", True)]   # B still runs after A failed
    recs = [json.loads(line) for line in open(out / "results.jsonl")]
    assert [(r["tag"], r["ok"], r.get("retry", False)) for r in recs] == [
        ("A", False, False), ("A", True, True), ("B", True, False)]
    assert recs[1]["run_dir"] == str(out / "runs" / "A-1-retry")
    assert os.path.exists(out / "runs" / "A-1-retry" / "rsl.error.0001")     # the retry keeps every rank's rsl
    assert [r["tag"] for r in proxy.collect(str(out))] == ["A", "B"]


def test_collect_drops_runs_marked_failed(tmp_path):
    rows = [dict(tag="A", rep=1, rc=0, median=0.40, ok=False), dict(tag="A", rep=1, rc=0, median=0.47, ok=True,
                                                                     retry=True), dict(tag="B", rep=1, rc=0, median=0.45)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [(r["tag"], r["median"]) for r in proxy.collect(str(tmp_path))] == [("B", 0.45), ("A", 0.47)]
