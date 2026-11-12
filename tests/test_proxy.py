import json
import re
import os

import pytest

from wtune import proxy


def test_proxy_shape_refuses_what_one_node_cannot_reproduce():
    assert proxy.proxy_shape(1501, 1201, 71, 16, 8, 144) is None      # 142 ranks: 2x71, box too wide
    assert proxy.proxy_shape(1501, 1201, 18, 63, 8, 144) is None      # 1134 ranks not divisible by 8
    assert proxy.proxy_shape(901, 901, 8, 28, 4, 112, omp=4) is None  # 56 ranks x 4 threads > 112 cores


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


def test_nl_set_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        proxy.nl_set(" &domains\n e_we = 10,\n /\n", radt=5)


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


def test_nest_dx_only_moves_when_dt_sits_on_the_bound():
    assert proxy.nest_dx(dict(domain=2, dx=600.0, dy=600.0, dt=3.0)) == (600.0, 600.0)   # Big Chill d02
    dx, dy = proxy.nest_dx(dict(domain=2, dx=111.1111, dy=111.1111, dt=6 / 9))            # marshall d02
    assert 6 * dx / 1000 >= 6 / 9 and dx < 111.2
    assert proxy.nest_dx(dict(domain=1, dx=1000.0, dy=1000.0, dt=6.0)) == (1000.0, 1000.0)


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
    assert plan["steps"] == proxy.PROXY_MAX_STEPS == 8
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


BIGCHILL_ISC = BIGCHILL.replace("i_parent_start = 1, 150", "i_parent_start = 1, 240").replace(
    "j_parent_start = 1, 80", "j_parent_start = 1, 115")


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
            open(os.path.join(cwd, "rsl.error.0001"), "w").write("forrtl: severe (154)")
        return subprocess.CompletedProcess(cmd, 1 if bad else 0)
    monkeypatch.setattr(subprocess, "run", fake)
    _route_launch(monkeypatch)
    assert proxy.worker(str(out / "plan.json")) == 0
    assert seen == [("A-1", True), ("A-1-retry", False), ("B-1", True)]   # B still runs after A failed
    recs = [json.loads(line) for line in open(out / "results.jsonl")]
    assert [(r["tag"], r["ok"], r.get("retry", False)) for r in recs] == [
        ("A", False, False), ("A", True, True), ("B", True, False)]
    assert recs[1]["run_dir"] == str(out / "runs" / "A-1-retry")
    assert os.path.exists(out / "runs" / "A-1-retry" / "rsl.error.0001")     # the retry keeps a rank's rsl naming an error
    assert [r["tag"] for r in proxy.collect(str(out))] == ["A", "B"]
    # a run whose setup fails (here its crop) is recorded and the job goes on, without a re-run
    seen.clear()
    plan["runs"] = [dict(run, tag="C", layout="C", nx=50), dict(run, tag="D", layout="D")]
    json.dump(plan, open(out / "plan.json", "w"))
    monkeypatch.setattr(proxy, "crop", lambda *a, **k: (_ for _ in ()).throw(OSError("no wrfinput_d01")))
    assert proxy.worker(str(out / "plan.json")) == 0
    assert seen == [("D-1", True)]
    last = [json.loads(line) for line in open(out / "results.jsonl")][-2]
    assert (last["tag"], last["ok"], last["error"]) == ("C", False, "OSError: no wrfinput_d01")


def test_drop_runs_keeps_the_records_and_the_failed_reruns(tmp_path):
    for d in ("runs/A-1", "runs/B-1", "runs/B-1-retry"):
        os.makedirs(tmp_path / d)
        (tmp_path / d / "rsl.error.0000").write_text("x")
    os.symlink(str(tmp_path / "plan.json"), str(tmp_path / "runs" / "A-1" / "wrfinput_d01"))
    for f in ("results.jsonl", "plan.json", "proxy-7.out"):
        (tmp_path / f).write_text("x")
    proxy.drop_runs(str(tmp_path))
    assert sorted(os.listdir(tmp_path / "runs")) == ["B-1-retry"]
    assert sorted(os.listdir(tmp_path)) == ["plan.json", "proxy-7.out", "results.jsonl", "runs"]
    proxy.drop_runs(str(tmp_path / "nothing-here"))


def test_shifted_nested_crop_moves_the_deepest_domain_and_replaces_its_parents():
    nb = proxy.nest_crop_boxes(BIGCHILL_ISC, 16, 28, 4, 112, proxy_nodes=3, pos=1)
    d1, d2 = nb["boxes"]
    assert (d2["i0"], d2["j0"]) == (0, 135) and (d2["nx"], d2["ny"]) == (670, 610)
    assert d1["j0"] + d2["j_parent_start"] == 115 + 135 // 5                    # the same d01 cell
    assert d2["j_parent_start"] - 1 >= 6 and d1["ny"] - (d2["j_parent_start"] - 1 + d2["ny"] // 5) >= 6
    one = proxy.nest_crop_boxes(BIGCHILL, 16, 18, 2, 144, pos=1)
    _check_nest_crop(BIGCHILL, one, 16, 18, 6)
    assert one["boxes"][1]["j0"] != 130 or one["boxes"][1]["i0"] != 80        # moved off the centre
    assert proxy.nest_crop_boxes(BIGCHILL_ISC, 16, 28, 4, 112, proxy_nodes=4, pos=1) is None   # the whole case


def test_collect_scores_a_layout_by_the_median_over_its_crop_positions(tmp_path):
    rows = [dict(tag="A", rep=1, rc=0, median=0.40), dict(tag="A", rep=2, rc=0, median=0.42),
            dict(tag="A_p1", layout="A", pos=1, rep=1, rc=0, median=0.50),
            dict(tag="B", rep=1, rc=0, median=0.45), dict(tag="B_p1", layout="B", pos=1, rep=1, rc=1, median=None)]
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = {r["tag"]: r for r in proxy.collect(str(tmp_path))}
    assert out["A"]["positions"] == {0: pytest.approx(0.41), 1: 0.50}
    assert out["A"]["median"] == pytest.approx((0.41 + 0.50) / 2) and out["B"]["positions"] == {0: 0.45}


def test_time_points_add_position_zero_runs_cropped_from_each_state(tmp_path, monkeypatch):
    plan = proxy.make_plan(_case(tmp_path), 8, ["18x64"], 144, positions=2, time_points=["/pts/a", "/pts/b"])
    assert [(r["tag"], r["pos"], r.get("time", 0), r.get("src")) for r in plan["runs"]] == [
        ("18x64_o1_nt1", 0, 0, None), ("18x64_o1_nt1_p1", 1, 0, None),
        ("18x64_o1_nt1_t1", 0, 1, "/pts/a"), ("18x64_o1_nt1_t2", 0, 2, "/pts/b")]
    calls = []
    monkeypatch.setattr(proxy, "crop", lambda *a, **k: calls.append((a[0], a[1], a[2])))
    files = proxy._inputs(plan, plan["runs"][2], str(tmp_path / "s"), {})
    assert calls[0][:2] == ("/pts/a/wrfinput_d01", "/pts/a/wrfbdy_d01") and calls[0][2].endswith("_t1")
    whole = dict(case_dir="/case", e_we=561, e_sn=450, max_dom=1)
    assert proxy._inputs(whole, dict(domain=1, nx=560, ny=449, time=1, src="/pts/a"), str(tmp_path), {}) == {
        "wrfinput_d01": "/pts/a/wrfinput_d01", "wrfbdy_d01": "/pts/a/wrfbdy_d01"}
    assert files["wrfinput_d01"].startswith(str(tmp_path / "s"))


def test_time_point_needs_both_files_on_the_cases_grid(tmp_path):
    case = _case(tmp_path)
    d = tmp_path / "pt"
    d.mkdir()
    (d / "wrfinput_d01").write_text("x")
    assert proxy.time_point_ok(case, str(d)) == "no wrfbdy_d01"
    (d / "wrfbdy_d01").write_text("x")
    assert proxy.time_point_ok(case, str(d)) is None                       # no namelist: taken as is
    (d / "namelist.input").write_text(" &domains\n e_we = 1501,\n e_sn = 1201,\n /\n")
    assert proxy.time_point_ok(case, str(d)) is None
    (d / "namelist.input").write_text(" &domains\n e_we = 1501,\n e_sn = 1001,\n /\n")
    assert "differ" in proxy.time_point_ok(case, str(d))


def test_a_time_point_run_starts_at_its_states_time(tmp_path):
    nc = pytest.importorskip("netCDF4")
    np = pytest.importorskip("numpy")
    I = nc.Dataset(tmp_path / "wrfinput_d01", "w", format="NETCDF3_64BIT_OFFSET")
    I.createDimension("Time", None)
    I.createDimension("DateStrLen", 19)
    I.createVariable("Times", "S1", ("Time", "DateStrLen"))[0, :] = np.array(list("2024-07-02_06:30:00"), dtype="S1")
    I.close()
    text = (" &time_control\n start_year = 2024, 2024,\n start_month = 07, 07,\n start_day = 01, 01,\n"
            " start_hour = 00, 00,\n start_minute = 00, 00,\n /\n")
    nl = proxy.nl_set(text, **proxy._start_kv(text, str(tmp_path / "wrfinput_d01")))
    for k, v in (("start_day", "2"), ("start_hour", "6"), ("start_minute", "30"), ("start_month", "7")):
        assert proxy.nl_values(nl, k) == [v], k
    assert "start_second" not in nl                                        # absent keys are not added


def test_tidy_keeps_only_the_rsl_files_that_name_an_error(tmp_path):
    for f, text in (("rsl.error.0000", "ok"), ("rsl.error.0001", "fine"), ("rsl.error.0049", "forrtl: severe (154)"),
                    ("rsl.out.0002", "x"), ("wrfout_d01_2022", "")):
        (tmp_path / f).write_text(text)
    proxy._tidy(str(tmp_path), keep_logs=True)
    assert sorted(os.listdir(tmp_path)) == ["rsl.error.0000", "rsl.error.0049"]


def test_a_run_failing_twice_with_tiles_runs_once_more_at_half_the_tiles(tmp_path, monkeypatch):
    import subprocess
    import wtune.machine
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(" &time_control\n run_days = 0, run_hours = 1, run_minutes = 0, run_seconds = 0,\n /\n &domains\n time_step = 15,\n"
                                         " max_dom = 1,\n e_we = 101,\n e_sn = 81,\n /\n")
    tree = _tree(tmp_path, "dmpar")
    run = dict(domain=1, weight=1, dt=15.0, px=2, py=2, omp=1, numtiles=16, ranks=4, a=2, b=2, nx=100, ny=80,
               wrf_tree=tree, tag="A", layout="A")
    plan = dict(case_dir=str(case), e_we=101, e_sn=81, max_dom=1, dt=15.0, steps=10, reps=1, runs=[run])
    out = tmp_path / "job"
    out.mkdir()
    json.dump(plan, open(out / "plan.json", "w"))
    monkeypatch.setenv("WTUNE_PROXY_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setattr(wtune.machine, "load_user", lambda: {})
    monkeypatch.setattr(wtune.machine, "export", lambda m: None)
    monkeypatch.setattr(wtune.machine, "site_launch_line", lambda *a: "mpirun ./wrf.exe")
    tiles = []

    def fake(cmd, cwd=None, **k):
        if cmd[0] == "rm":
            return subprocess.CompletedProcess(cmd, 0)
        nt = int(re.search(r"numtiles\s*=\s*(\d+)", open(os.path.join(cwd, "namelist.input")).read()).group(1))
        tiles.append(nt)
        open(os.path.join(cwd, "rsl.error.0000"), "w").write("".join(
            "Timing for main: time 2019 on domain   1:    0.50000 elapsed seconds\n" for _ in range(10 if nt < 16 else 0)))
        return subprocess.CompletedProcess(cmd, 0 if nt < 16 else 255)
    monkeypatch.setattr(subprocess, "run", fake)
    _route_launch(monkeypatch)
    assert proxy.worker(str(out / "plan.json")) == 0
    assert tiles == [16, 16, 8]
    recs = [json.loads(line) for line in open(out / "results.jsonl")]
    assert [(r["ok"], r.get("nt_fallback")) for r in recs] == [(False, None), (False, None), (True, 16)]
    assert [r["tag"] for r in proxy.collect(str(out))] == ["A"]


def _route_launch(monkeypatch):
    """Send the launcher through the patched subprocess.run, a timeout read as the cap (124)."""
    import subprocess

    def launch(argv, d, env, cap):
        try:
            return subprocess.run(argv, cwd=d, env=env, timeout=cap).returncode
        except subprocess.TimeoutExpired:
            return 124
    monkeypatch.setattr(proxy, "_launch", launch)


def test_launch_returns_the_exit_code_and_stops_a_run_at_the_cap(tmp_path):
    import time
    assert proxy._launch(["sh", "-c", "echo $SLURM_KILL_BAD_EXIT; exit 3"], str(tmp_path),
                         dict(os.environ, SLURM_KILL_BAD_EXIT="1"), 10) == 3
    assert open(tmp_path / "run.log").read().strip() == "1"
    t = time.time()
    assert proxy._launch(["sleep", "30"], str(tmp_path), dict(os.environ), 1) == 124
    assert time.time() - t < 10                                   # SIGTERM ends it, no 30 s wait
