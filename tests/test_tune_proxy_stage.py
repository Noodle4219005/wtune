import dataclasses
import os
import types

from wtune import proxy as P
from wtune.tune import proxy_stage as S


def _cfg(cid, nx, ny, omp=1, nt=1, **kw):
    return dict(id=cid, nx=nx, ny=ny, ranks=nx * ny, omp=omp, numtiles=nt, dist="block", **kw)


def _ctx(tmp_path, with_inputs=True):
    case = tmp_path / "case"
    case.mkdir()
    (case / "namelist.input").write_text(" &domains\n time_step = 15,\n max_dom = 1,\n e_we = 1501,\n e_sn = 1201,\n /\n")
    if with_inputs:
        (case / "wrfinput_d01").write_text("x")
        (case / "wrfbdy_d01").write_text("x")
    return types.SimpleNamespace(namelist_path=str(case / "namelist.input"), wrf_tree="/trees/dmsm", nodes=4,
                                 steps=40, max_nodes=8, out_dir=str(tmp_path / "out"), owned_jobs=[], seed=1)


def _patch(monkeypatch, extra, timings, band=None):
    submitted = []
    monkeypatch.setattr(S, "_apply_search_filter", lambda ctx, cs: cs)
    monkeypatch.setattr(S, "sample_more_diverse_configs", lambda ctx, ex, k, seed: extra[:k])
    monkeypatch.setattr(S, "cores_used", lambda ctx: 144)
    monkeypatch.setattr(P, "check_builds", lambda plan: [])

    def submit(plan, d, *a, **k):
        submitted.append(plan)
        return str(100 + len(submitted)), "Submitted batch job"
    monkeypatch.setattr(P, "submit_plan", submit)
    monkeypatch.setattr(S._jobs, "wait_for_jobs", lambda *a, **k: {})
    monkeypatch.setattr(P, "collect", lambda d: [dict(tag=t, median=m) for t, m in
                        (timings if d.endswith("job0") else (band or {}) if d.endswith("band0") else {}).items()])
    return submitted


def test_proxy_keeps_the_anchor_and_the_proxys_fastest(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    extra = [_cfg("D0001", 12, 24, omp=2, nt=24), _cfg("D0002", 9, 64, nt=13), _cfg("D0003", 18, 32)]
    timings = {"24x24_o1_nt1": 0.51, "32x18_o1_nt1": 0.56, "16x36_o1_nt1": 0.65,
               "12x24_o2_nt24": 0.40, "9x64_o1_nt13": 0.42, "18x32_o1_nt1": 0.50}
    submitted = _patch(monkeypatch, extra, timings)
    ctx = _ctx(tmp_path)
    out = S.select(ctx, configs)
    assert [c["id"] for c in out] == ["ANCHOR", "D0001", "D0002"]       # anchor + the proxy's 2 fastest
    one = [p for p in submitted if p.get("proxy_nodes", 1) == 1]
    # the one-node pool: 6 layouts at one crop position, one job each
    assert sum(len(p["runs"]) for p in one) == 6 and ctx.owned_jobs[:2] == ["101", "102"]
    assert all(r["pos"] == 0 for p in one for r in p["runs"]) and all(len(p["runs"]) == 1 for p in one)
    bands = [p for p in submitted if p.get("proxy_nodes") == 3]            # N=4 > 3: the 4 leaders as bands
    assert sorted(r["layout"] for p in bands for r in p["runs"] if r["pos"] == 0) == [
        "12x24_o2_nt24", "18x32_o1_nt1", "9x64_o1_nt13"]   # 32x18: 144 ranks per node, not whole rows
    assert sum(r["pos"] == 1 for p in bands for r in p["runs"]) == 3       # a band shifts along y


def test_band_round_reorders_the_one_node_leaders(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    extra = [_cfg("D0001", 12, 24, omp=2, nt=24), _cfg("D0002", 9, 64, nt=13), _cfg("D0003", 18, 32)]
    timings = {"24x24_o1_nt1": 0.51, "32x18_o1_nt1": 0.56, "16x36_o1_nt1": 0.65,
               "12x24_o2_nt24": 0.40, "9x64_o1_nt13": 0.42, "18x32_o1_nt1": 0.50}
    band = {"12x24_o2_nt24": 0.47, "9x64_o1_nt13": 0.43, "18x32_o1_nt1": 0.52}   # one node was 15 % low on D0001
    _patch(monkeypatch, extra, timings, band)
    out = S.select(_ctx(tmp_path), configs)
    assert [c["id"] for c in out] == ["ANCHOR", "D0002", "D0001"]


def test_no_band_round_at_or_below_three_nodes(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 12, 24), _cfg("G0001", 18, 16), _cfg("G0002", 8, 36)]
    submitted = _patch(monkeypatch, [], {"12x24_o1_nt1": 0.5, "18x16_o1_nt1": 0.6, "8x36_o1_nt1": 0.7})
    ctx = _ctx(tmp_path)
    ctx.nodes = 2
    S.select(ctx, configs)
    assert all(p.get("proxy_nodes", 1) == 1 for p in submitted)


def test_proxy_falls_back_without_netcdf_inputs(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18)]
    _patch(monkeypatch, [], {})
    assert S.select(_ctx(tmp_path, with_inputs=False), configs) is configs


def test_proxy_skips_2d_tiles(tmp_path, monkeypatch):
    assert S._spec(_cfg("X", 8, 8, tile_strategy=3, numtiles_x=4)) is None
    assert S._spec(_cfg("Y", 8, 8, omp=2, nt=14)) == "8x8:omp=2:nt=14"


def test_a_cyclic_twin_does_not_inherit_the_block_timing(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    cyc = dict(_cfg("C0001", 12, 24, omp=2, nt=24), dist="cyclic")
    extra = [cyc, _cfg("D0001", 12, 24, omp=2, nt=24), _cfg("D0002", 18, 32)]
    timings = {"24x24_o1_nt1": 0.51, "32x18_o1_nt1": 0.56, "16x36_o1_nt1": 0.65, "12x24_o2_nt24": 0.40,
               "18x32_o1_nt1": 0.50}
    _patch(monkeypatch, extra, timings)
    out = S.select(_ctx(tmp_path), configs)
    assert [c["id"] for c in out] == ["ANCHOR", "D0001", "D0002"]


def test_proxy_cuts_round0_to_anchor_plus_two(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24)] + [_cfg("G%04d" % i, 24, 24 * 0 + 24 - i, nt=i) for i in range(1, 6)]
    timings = {S._tag(c): 0.5 - 0.01 * i for i, c in enumerate(configs)}
    _patch(monkeypatch, [], timings)
    out = S.select(_ctx(tmp_path), configs)
    assert len(out) == 3 and out[0]["id"] == "ANCHOR" and [c["id"] for c in out[1:]] == ["G0005", "G0004"]


def test_short_budget_drops_the_anchor_then_keeps_the_proxys_best(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    timings = {"24x24_o1_nt1": 0.51, "32x18_o1_nt1": 0.45, "16x36_o1_nt1": 0.48}
    _patch(monkeypatch, [], timings)
    ctx = _ctx(tmp_path)
    ctx.budget_s, ctx.startup_s, ctx.max_nodes, ctx.nodes = 100.0, 30.0, 4, 4
    ctx.elapsed = lambda: 0.0
    ctx.estimate_s_per_step = lambda: (0.5, "x", 1)          # one real probe = 30 + 40*0.5 = 50 s
    out = S.select(ctx, configs)
    assert [c["id"] for c in out] == ["G0001", "G0002"]       # 2 probes fit: anchor dropped
    ctx.proxy_used = False
    ctx.budget_s = 60.0
    assert [c["id"] for c in S.select(ctx, configs)] == ["G0001"]   # 1 fits: the proxy's best only


@dataclasses.dataclass
class _JCtx:
    namelist_path: str
    wrf_tree: str
    out_dir: str
    nodes: int = 1
    steps: int = 40
    max_nodes: int = 8
    seed: int = 1
    owned_jobs: list = dataclasses.field(default_factory=list)
    anchor_override: dict | None = None


def _joint(tmp_path, monkeypatch, pools, timings, band):
    ns = _ctx(tmp_path)
    ctx = _JCtx(namelist_path=ns.namelist_path, wrf_tree=ns.wrf_tree, out_dir=ns.out_dir)
    submitted = _patch(monkeypatch, [], {}, None)
    monkeypatch.setattr(S, "_sweep_anchor", lambda c, n: (pools[n][0], None) if n in pools else (None, "x"))
    monkeypatch.setattr(S, "_seed_layouts", lambda c: [])
    monkeypatch.setattr(S, "sample_more_diverse_configs", lambda c, ex, k, seed: pools[c.nodes][1:k + 1])
    monkeypatch.setattr(P, "collect", lambda d: [dict(tag=t, median=m) for t, m in
                        (timings if d.endswith("jobs0") else band if "band" in os.path.basename(d) else {}).items()])
    return ctx, submitted


POOLS = {2: [_cfg("ANCHOR", 16, 18), _cfg("D0001", 12, 24), _cfg("D0002", 24, 12)],
         4: [_cfg("ANCHOR", 24, 24), _cfg("D0001", 12, 24, omp=2, nt=24), _cfg("D0002", 18, 32)],
         5: [_cfg("ANCHOR", 24, 30), _cfg("D0001", 20, 36), _cfg("D0002", 30, 24)]}


def test_rank_across_nodes_times_every_n_in_one_plan_and_bands_the_leaders(tmp_path, monkeypatch):
    timings = {"N2_16x18_o1_nt1": 0.80, "N2_12x24_o1_nt1": 0.78, "N2_24x12_o1_nt1": 0.85,
               "N4_24x24_o1_nt1": 0.45, "N4_12x24_o2_nt24": 0.40, "N4_18x32_o1_nt1": 0.47,
               "N5_24x30_o1_nt1": 0.42, "N5_20x36_o1_nt1": 0.38, "N5_30x24_o1_nt1": 0.44}
    band = {"N4_24x24_o1_nt1": 0.46, "N4_12x24_o2_nt24": 0.47, "N5_24x30_o1_nt1": 0.44,
            "N2_12x24_o1_nt1": 0.90, "N2_16x18_o1_nt1": 0.88}          # N=2's leaders on 2 whole nodes
    ctx, submitted = _joint(tmp_path, monkeypatch, POOLS, timings, band)
    out, same = S.rank_across_nodes(ctx, [1, 2, 4, 5])               # N=1: no ANCHOR, skipped
    one = [r for p in submitted if p["proxy_nodes"] == 1 for r in p["runs"]]
    tags = [r["tag"] for r in one if r["pos"] == 0]
    assert sorted(tags) == sorted(timings) and all(r["layout"] == r["tag"] for r in one if r["pos"] == 0)
    assert all(r["pos"] == 0 for r in one)                          # the pool: one crop position
    assert all(len({r["layout"] for r in p["runs"]}) == 1 for p in submitted)   # one layout per job
    bands = [r["layout"] for p in submitted if p["proxy_nodes"] == 3 for r in p["runs"] if r["pos"] == 0]
    # the 2 one-node leaders of each N > 3; N=5's 20x36 holds no whole rows per node
    assert sorted(bands) == ["N4_12x24_o2_nt24", "N4_24x24_o1_nt1", "N5_24x30_o1_nt1"]
    full = [(r["layout"], r["pos"]) for p in submitted if p["proxy_nodes"] == 2 for r in p["runs"]]
    # N=2: the whole domain on 2 nodes, one position only
    assert sorted(full) == [("N2_12x24_o1_nt1", 0), ("N2_16x18_o1_nt1", 0)]
    assert sorted(out) == [2, 4, 5] and same == {2: True, 4: True, 5: True}
    assert [(S._tag(c), t, f) for c, t, f in out[4]] == [
        ("24x24_o1_nt1", 0.46, "band"), ("12x24_o2_nt24", 0.47, "band"), ("18x32_o1_nt1", 0.47, "one-node")]
    assert [(c["id"], f) for c, _t, f in out[5]] == [("ANCHOR", "band"), ("D0001", "one-node"),
                                                     ("D0002", "one-node")]
    assert [(c["id"], f) for c, _t, f in out[2]] == [("ANCHOR", "full"), ("D0001", "full"), ("D0002", "one-node")]


def test_rank_across_nodes_keeps_the_domain_suffix_of_a_nested_case(tmp_path, monkeypatch):
    ctx, _ = _joint(tmp_path, monkeypatch, POOLS, {}, {})
    (tmp_path / "case" / "namelist.input").write_text(
        " &domains\n time_step = 15,\n max_dom = 2,\n e_we = 1501, 1501,\n e_sn = 1201, 1201,\n"
        " parent_id = 0, 1,\n parent_time_step_ratio = 1, 3,\n /\n")
    _plan, runs = S._joint_plan(ctx, str(tmp_path / "case"), 2, {"16x18_o1_nt1": POOLS[2][0]}, 1)
    assert [(r["tag"], r["layout"]) for r in runs] == [("N2_16x18_o1_nt1_d1", "N2_16x18_o1_nt1"),
                                                       ("N2_16x18_o1_nt1_d2", "N2_16x18_o1_nt1")]


def test_rank_across_nodes_is_empty_without_timings(tmp_path, monkeypatch):
    ctx, _ = _joint(tmp_path, monkeypatch, POOLS, {}, {})
    assert S.rank_across_nodes(ctx, [2, 4]) == ({}, {})


def test_select_with_a_joint_ranking_submits_nothing(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    submitted = _patch(monkeypatch, [], {})
    ctx = _ctx(tmp_path, with_inputs=False)
    ctx.proxy_ranked = [(_cfg("D0001", 12, 24, omp=2, nt=24), 0.46, "band"), (_cfg("ANCHOR", 24, 24), 0.47, "band"),
                        (_cfg("D0002", 18, 32), 0.47, "one-node"), (_cfg("D0003", 9, 64, nt=13), 0.49, "one-node")]
    out = S.select(ctx, configs)
    assert [c["id"] for c in out] == ["ANCHOR", "D0001", "D0002"] and ctx.proxy_used
    assert submitted == [] and ctx.owned_jobs == []


def test_run_jobs_drops_each_jobs_run_directories_after_collecting(tmp_path, monkeypatch):
    _patch(monkeypatch, [], {"A": 0.5})
    dropped = []
    monkeypatch.setattr(P, "drop_runs", lambda d: dropped.append(os.path.basename(d)))
    runs = [dict(tag=t) for t in "ABCDEF"]
    ctx = _ctx(tmp_path)
    assert S._run_jobs(ctx, dict(runs=runs), runs, str(tmp_path / "o"), "job", 1) == {"A": 0.5}
    assert dropped == ["job%d" % j for j in range(6)]               # one job per layout, max_nodes 8


def test_run_jobs_keeps_a_layouts_runs_in_one_job(tmp_path, monkeypatch):
    submitted = _patch(monkeypatch, [], {})
    runs = [dict(tag=t + s, layout=t) for t in "ABCDE" for s in ("", "_p1", "_t1")]
    S._run_jobs(_ctx(tmp_path), dict(runs=runs), runs, str(tmp_path / "o"), "job", 1)
    assert len(submitted) == 5                                     # one job per layout (5), max_nodes 8
    for p in submitted:
        lays = {r["layout"] for r in p["runs"]}
        assert all(sum(r["layout"] == x for r in p["runs"]) == 3 for x in lays)
    assert sorted(r["tag"] for p in submitted for r in p["runs"]) == sorted(r["tag"] for r in runs)


def test_rank_across_nodes_flags_an_n_whose_crop_positions_disagree(tmp_path, monkeypatch):
    timings = {"N4_24x24_o1_nt1": (0.45, {0: 0.45, 1: 0.45}), "N4_12x24_o2_nt24": (0.40, {0: 0.40, 1: 0.40}),
               "N4_18x32_o1_nt1": (0.47, {0: 0.47, 1: 0.47}),
               "N5_24x30_o1_nt1": (0.42, {0: 0.41, 1: 0.43}), "N5_20x36_o1_nt1": (0.38, {0: 0.38, 1: 0.39})}
    # N=4's band leaders: 24x24 fastest at the centre, 12x24 at the shifted crop
    band = {"N4_24x24_o1_nt1": (0.465, {0: 0.44, 1: 0.49}), "N4_12x24_o2_nt24": (0.47, {0: 0.46, 1: 0.48})}
    ctx, _ = _joint(tmp_path, monkeypatch, POOLS, {}, {})
    monkeypatch.setattr(P, "collect", lambda d: [dict(tag=t, median=m, positions=p) for t, (m, p) in (
        timings if d.endswith("jobs0") else band if "band" in os.path.basename(d) else {}).items()])
    out, same = S.rank_across_nodes(ctx, [4, 5])
    assert same == {4: False, 5: True}             # N=5: one band leader only, its one-node pool agrees
    assert [(S._tag(c), t) for c, t, _f in out[4][:2]] == [("24x24_o1_nt1", 0.465), ("12x24_o2_nt24", 0.47)]
    assert S._same_fastest({"a": {0: 1.0}, "b": {0: 2.0, 1: 1.0}}, {"a": {0: 1, 1: 2}, "b": {0: 2, 1: 1}}) is False


def test_time_points_are_the_completed_rebuilds_of_this_tune(tmp_path, capsys):
    ctx = _ctx(tmp_path)
    case = os.path.dirname(ctx.namelist_path)
    assert S._time_points(ctx, case) == [] and "no completed rebuilt state" in capsys.readouterr().out
    for i in (0, 1, 2, 10):
        real = os.path.join(ctx.out_dir, "rebuilt", "pt_%d" % i, "win_0", "real")
        os.makedirs(real)
        for f in ("wrfinput_d01", "wrfbdy_d01"):
            open(os.path.join(real, f), "w").write("x")
        if i != 1:                                       # pt_1's real_only job has not completed
            open(os.path.join(ctx.out_dir, "rebuilt", "pt_%d" % i, "namelist.template.input"), "w").write("x")
    got = S._time_points(ctx, case)
    assert [p.split(os.sep)[-3] for p in got] == ["pt_0", "pt_2"]


def test_band_leaders_are_also_timed_at_the_time_points(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    extra = [_cfg("D0001", 12, 24, omp=2, nt=24), _cfg("D0002", 9, 64, nt=13), _cfg("D0003", 18, 32)]
    timings = {"24x24_o1_nt1": 0.51, "32x18_o1_nt1": 0.56, "16x36_o1_nt1": 0.65,
               "12x24_o2_nt24": 0.40, "9x64_o1_nt13": 0.42, "18x32_o1_nt1": 0.50}
    submitted = _patch(monkeypatch, extra, timings)
    monkeypatch.setattr(S, "_time_points", lambda ctx, case: ["/pts/a", "/pts/b"])
    S.select(_ctx(tmp_path), configs)
    one = [r for p in submitted if p.get("proxy_nodes", 1) == 1 for r in p["runs"]]
    band = [r for p in submitted if p.get("proxy_nodes") == 3 for r in p["runs"]]
    assert not [r for r in one if r.get("time")]                                # one-node pool: own time only
    assert sorted(r["tag"] for r in band if r.get("time")) == sorted(
        "%s_t%d" % (t, k) for t in ("12x24_o2_nt24", "9x64_o1_nt13", "18x32_o1_nt1") for k in (1, 2))


def test_the_joint_pool_skips_layouts_the_proxy_cannot_run_and_fills_the_slots(tmp_path, monkeypatch):
    shapes = [(20, 36), (30, 24), (36, 20), (40, 18), (18, 40), (45, 16), (16, 45), (48, 15)]
    cyc = [dict(_cfg("C%04d" % i, *shapes[i]), dist="cyclic") for i in range(7)]
    good = [_cfg("D%04d" % i, *xy) for i, xy in enumerate(shapes)]
    pools = {5: [_cfg("ANCHOR", 24, 30)] + cyc + good}
    asked = []
    ctx, submitted = _joint(tmp_path, monkeypatch, pools, {}, {})

    def sample(c, ex, k, seed):
        asked.append(k)
        return pools[c.nodes][1:k + 1]
    monkeypatch.setattr(S, "sample_more_diverse_configs", sample)
    S.rank_across_nodes(ctx, [5])
    assert asked == [S.PROXY_POOL_OVERSAMPLE * (S.PROXY_JOINT_PER_N - 1)]
    ids = sorted({r["layout"] for p in submitted for r in p["runs"]})
    # the 7 cyclic twins skipped; the ANCHOR and the first 5 block layouts fill the 6 slots
    assert ids == sorted(["N5_24x30_o1_nt1"] + ["N5_%dx%d_o1_nt1" % xy for xy in shapes[:5]])


def test_run_jobs_caps_the_job_count_and_waits_out_a_refused_submit(tmp_path, monkeypatch):
    runs = [dict(tag="L%02d" % i, layout="L%02d" % i) for i in range(30)]
    calls, waits = [], []

    def submit(plan, d, *a, **k):
        calls.append(os.path.basename(d))
        if len(calls) == 3:                                      # the QOS refuses the 3rd once
            return None, "sbatch: error: QOSMaxSubmitJobPerUserLimit"
        return str(500 + len(calls)), "Submitted batch job"
    monkeypatch.setattr(P, "submit_plan", submit)
    monkeypatch.setattr(S._jobs, "wait_for_jobs", lambda ctx, ids, stage, **k: waits.append(list(ids)) or {})
    monkeypatch.setattr(P, "collect", lambda d: [])
    ctx = types.SimpleNamespace(max_nodes=8, owned_jobs=[], steps=8)
    monkeypatch.setattr(S, "_proxy_time_limit", lambda ctx: "0:10:00")
    S._run_jobs(ctx, dict(runs=[]), runs, str(tmp_path), "jobs", 1)
    assert calls[:4] == ["jobs0", "jobs1", "jobs2", "jobs2"]     # retried after a wait
    assert len(set(calls)) == S.PROXY_MAX_JOBS                  # 30 layouts on 24 jobs
    assert waits[0] == ["501"]                                   # the oldest job waited for first
    assert "501" not in waits[1] and len(waits[1]) == S.PROXY_MAX_JOBS - 1


def test_the_whole_domain_and_band_rounds_are_submitted_together(tmp_path, monkeypatch):
    timings = {"N2_16x18_o1_nt1": 0.80, "N2_12x24_o1_nt1": 0.78, "N4_24x24_o1_nt1": 0.45,
               "N4_12x24_o2_nt24": 0.40}
    ctx, submitted = _joint(tmp_path, monkeypatch, POOLS, timings, {})
    events = []
    real = P.submit_plan

    def submit(plan, d, *a, **k):
        events.append(("submit", plan.get("proxy_nodes", 1)))
        return real(plan, d, *a, **k)
    monkeypatch.setattr(P, "submit_plan", submit)
    monkeypatch.setattr(S._jobs, "wait_for_jobs", lambda c, ids, stage, **k: events.append(("wait", len(ids))) or {})
    S.rank_across_nodes(ctx, [2, 4])
    after_one = events[[i for i, e in enumerate(events) if e[0] == "wait"][0] + 1:]
    assert [e for e in after_one if e[0] == "submit"] == [("submit", 2), ("submit", 2), ("submit", 3), ("submit", 3)]
    assert after_one[-1] == ("wait", 4)                                # one wait for both sizes


def test_an_n_under_the_cut_is_timed_with_its_anchor_alone_and_dear_layouts_get_their_own_job(
        tmp_path, monkeypatch):
    ctx, submitted = _joint(tmp_path, monkeypatch, POOLS, {}, {})
    asked = []
    monkeypatch.setattr(S, "sample_more_diverse_configs",
                        lambda c, ex, k, seed: asked.append(c.nodes) or POOLS[c.nodes][1:k + 1])
    monkeypatch.setattr(S, "PROXY_MAX_JOBS", 4)
    S.rank_across_nodes(ctx, [2, 4, 5], anchor_only_below=4)
    assert asked == [4, 5]                                               # N=2: no sample drawn
    jobs = [sorted({r["layout"] for r in p["runs"]}) for p in submitted if p["proxy_nodes"] == 1]
    assert ["N2_16x18_o1_nt1"] in jobs                                    # the dearest runs alone
    assert sum(len(j) for j in jobs) == 7 and len(jobs) == 4


def test_the_seed_shapes_lead_the_joint_pool_ahead_of_the_sampled_layouts(tmp_path, monkeypatch):
    ctx, submitted = _joint(tmp_path, monkeypatch, POOLS, {}, {})
    seeds = [_cfg("S0001", 8, 45, omp=2, nt=12), _cfg("S0002", 4, 90, omp=2, nt=12)]
    monkeypatch.setattr(S, "_seed_layouts", lambda c: seeds if c.nodes == 5 else [])
    S.rank_across_nodes(ctx, [5])
    tags = [r["layout"] for p in submitted for r in p["runs"]]
    assert tags[:4] == ["N5_24x30_o1_nt1", "N5_8x45_o2_nt12", "N5_4x90_o2_nt12", "N5_20x36_o1_nt1"]
    assert len(set(tags)) == 5                         # 2 seeds on top of the anchor + 2 sampled


def test_seed_layouts_take_the_balanced_and_the_x_wide_patch_per_thread_count(monkeypatch):
    import types
    space = [dict(nx=nx, ny=ny, ranks=nx * ny, omp=omp, numtiles=nt, tile_shape="1d_y", dist="block",
                  blocktime="0", patch_x=800 // nx, patch_y=800 // ny)
             for omp, shapes in ((1, [(28, 32), (14, 64), (8, 112)]), (2, [(16, 28), (8, 56), (28, 16)]))
             for nx, ny in shapes for nt in (1, 12)]
    monkeypatch.setattr(S.decomp_space, "enumerate_space", lambda case, cores, n, hw: (space, None))
    monkeypatch.setattr(S, "_apply_search_filter", lambda c, cs: cs)
    monkeypatch.setattr(S, "cores_used", lambda c: 112)
    ctx = types.SimpleNamespace(case_json={}, nodes=8, hardware_id="x")
    got = {(c["nx"], c["ny"], c["omp"], c["numtiles"]) for c in S._seed_layouts(ctx)}
    # balanced + widest-in-x (seed_design) and the next x-wide shape per thread count
    assert got == {(28, 32, 1, 12), (8, 112, 1, 12), (14, 64, 1, 12), (16, 28, 2, 12), (8, 56, 2, 12)}


def test_band_leaders_are_distinct_patch_shapes():
    tags = ["7x64_o2_nt12", "7x64_o2_nt6", "16x28_o2_nt12", "14x64_o1_nt12"]
    assert S._distinct_shapes(tags, 2) == ["7x64_o2_nt12", "16x28_o2_nt12"]
    assert S._distinct_shapes(["7x64_o2_nt12", "7x64_o2_nt6"], 2) == ["7x64_o2_nt12", "7x64_o2_nt6"]
    assert S._distinct_shapes(["8x56_o2_nt14", "8x56_o1_nt14"], 2) == ["8x56_o2_nt14", "8x56_o1_nt14"]


def test_select_with_a_joint_ranking_probes_distinct_shapes(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    _patch(monkeypatch, [], {})
    ctx = _ctx(tmp_path, with_inputs=False)
    ctx.proxy_ranked = [(_cfg("D0001", 12, 24, omp=2, nt=12), 0.40, "band"),
                        (_cfg("D0002", 12, 24, omp=2, nt=6), 0.41, "band"),
                        (_cfg("D0003", 18, 32), 0.45, "one-node")]
    assert [c["id"] for c in S.select(ctx, configs)] == ["ANCHOR", "D0001", "D0003"]


def test_a_layout_timed_at_the_tiles_fallback_is_ranked_and_band_timed_at_that_numtiles(tmp_path, monkeypatch):
    pools = {5: [_cfg("ANCHOR", 24, 30), _cfg("D0001", 12, 60, omp=1, nt=12)]}
    ctx, submitted = _joint(tmp_path, monkeypatch, pools, {}, {})
    real = P.collect

    def collect(d):
        if d.endswith("jobs0") or d.endswith("jobs1"):
            return [dict(tag="N5_24x30_o1_nt1", median=0.50), dict(tag="N5_12x60_o1_nt12", median=0.40, numtiles=6)]
        if "band" in os.path.basename(d):
            return [dict(tag="N5_12x60_o1_nt6", median=0.42), dict(tag="N5_24x30_o1_nt1", median=0.52)]
        return real(d)
    monkeypatch.setattr(P, "collect", collect)
    out, _same = S.rank_across_nodes(ctx, [5])
    bands = sorted({r["layout"] for p in submitted if p["proxy_nodes"] == 3 for r in p["runs"]})
    assert "N5_12x60_o1_nt6" in bands and "N5_12x60_o1_nt12" not in bands
    assert [(S._tag(c), t, f) for c, t, f in out[5]][0] == ("12x60_o1_nt6", 0.42, "band")
