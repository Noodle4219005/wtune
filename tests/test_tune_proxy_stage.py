import dataclasses
import os
import types
import pytest

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


def test_no_band_round_at_or_below_three_nodes(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 12, 24), _cfg("G0001", 18, 16), _cfg("G0002", 8, 36)]
    submitted = _patch(monkeypatch, [], {"12x24_o1_nt1": 0.5, "18x16_o1_nt1": 0.6, "8x36_o1_nt1": 0.7})
    ctx = _ctx(tmp_path)
    ctx.nodes = 2
    S.select(ctx, configs)
    assert all(p.get("proxy_nodes", 1) == 1 for p in submitted)


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
    core_budget: int | None = None
    cores_per_node: int = 64


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


def test_rank_across_nodes_keeps_the_domain_suffix_of_a_nested_case(tmp_path, monkeypatch):
    ctx, _ = _joint(tmp_path, monkeypatch, POOLS, {}, {})
    (tmp_path / "case" / "namelist.input").write_text(
        " &domains\n time_step = 15,\n max_dom = 2,\n e_we = 1501, 1501,\n e_sn = 1201, 1201,\n"
        " parent_id = 0, 1,\n parent_time_step_ratio = 1, 3,\n /\n")
    _plan, runs = S._joint_plan(ctx, str(tmp_path / "case"), 2, {"16x18_o1_nt1": POOLS[2][0]}, 1)
    assert [(r["tag"], r["layout"]) for r in runs] == [("N2_16x18_o1_nt1_d1", "N2_16x18_o1_nt1"),
                                                       ("N2_16x18_o1_nt1_d2", "N2_16x18_o1_nt1")]


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


class _FakePool:
    """_Pool stand-in: each wait() runs the first queued batch (by `first`) to completion;
    after `stop_after` waits it reports the deadline instead."""

    def __init__(self, stop_after=99):
        self.nodes, self.added, self.ran, self.closed, self.left = 8, [], set(), False, stop_after
        self.ran_first = set()

    def add(self, batch_dir, plan, runs, nodes, first, rep0=1):
        self.added.append((os.path.basename(batch_dir), first, nodes, sorted({r["layout"] for r in runs})))

    def done(self, d):
        return [a for a in self.added if a[0] == os.path.basename(d)][-1][1] in self.ran_first

    def wait(self, pred):
        if self.left <= 0:
            return False
        self.left -= 1
        todo = sorted((first, b) for b, first, _n, _l in self.added if first not in self.ran_first)
        if todo:
            self.ran_first.add(todo[0][0])
            self.ran.add(todo[0][1])
        return pred()

    def close(self, stopped=False):
        self.closed, self.stopped = True, stopped


def _pool_joint(tmp_path, monkeypatch, timings, band, fake):
    ctx, _ = _joint(tmp_path, monkeypatch, POOLS, timings, band)
    ctx.budget_s = float("inf")
    ctx.elapsed = lambda: 0.0
    monkeypatch.setattr(S, "PROXY_JOINT_LEADERS", 2)
    monkeypatch.setattr(S, "_pool_nodes", lambda c, k: 8)
    monkeypatch.setattr(S._Pool, "start", classmethod(lambda cls, c, root, n: fake))

    def collect(d):
        b = os.path.basename(d)
        src = band if b.startswith("band-") else timings
        if b not in fake.ran:
            return []
        n = b.split("-N")[1]
        return [dict(tag=t, median=m) for t, m in src.items() if t.startswith("N%s_" % n)]
    monkeypatch.setattr(P, "collect", collect)
    return ctx


TIMINGS = {"N2_16x18_o1_nt1": 0.80, "N2_12x24_o1_nt1": 0.78, "N2_24x12_o1_nt1": 0.85,
           "N4_24x24_o1_nt1": 0.45, "N4_12x24_o2_nt24": 0.40, "N4_18x32_o1_nt1": 0.47,
           "N5_24x30_o1_nt1": 0.42, "N5_20x36_o1_nt1": 0.38, "N5_30x24_o1_nt1": 0.44}
BAND = {"N4_24x24_o1_nt1": 0.46, "N4_12x24_o2_nt24": 0.47, "N5_24x30_o1_nt1": 0.44,
        "N2_12x24_o1_nt1": 0.90, "N2_16x18_o1_nt1": 0.88}


def test_the_node_pool_bands_an_n_as_soon_as_its_one_node_runs_are_timed(tmp_path, monkeypatch):
    fake = _FakePool()
    ctx = _pool_joint(tmp_path, monkeypatch, TIMINGS, BAND, fake)
    out, same = S.rank_across_nodes(ctx, [1, 2, 4, 5])
    # every N's one-node runs queued at once, the largest N first; each N's leaders as soon as its
    # one-node runs are timed, ahead of the one-node runs still waiting. N=4's two leaders, 2 %
    # apart and within 5 % of the best band (N=5), are re-timed once ahead of the one-node runs;
    # N=2's are as close, but 100 % slower than N=5
    assert [(b, f, k) for b, f, k, _l in fake.added] == [
        ("one-N5", "1-994", 1), ("one-N4", "1-995", 1), ("one-N2", "1-997", 1),
        ("band-N5", "0-994", 3), ("band-N4", "0-995", 3), ("band-N4", "0-r995", 3),
        ("band-N2", "0-997", 2)]
    assert fake.added[5][3] == ["N4_12x24_o2_nt24", "N4_24x24_o1_nt1"]
    assert dict((b, ls) for b, _f, _k, ls in fake.added)["band-N4"] == ["N4_12x24_o2_nt24", "N4_24x24_o1_nt1"]
    assert fake.closed and not fake.stopped
    # the same ranking as one batch of jobs per round
    assert [(S._tag(c), t, f) for c, t, f in out[4]] == [
        ("24x24_o1_nt1", 0.46, "band"), ("12x24_o2_nt24", 0.47, "band"), ("18x32_o1_nt1", 0.47, "one-node")]
    assert [(c["id"], f) for c, _t, f in out[2]] == [("ANCHOR", "full"), ("D0001", "full"), ("D0002", "one-node")]


def test_the_node_pool_ranks_what_finished_at_the_deadline(tmp_path, monkeypatch, capsys):
    fake = _FakePool(stop_after=2)                      # N=5's one-node runs, then its band
    ctx = _pool_joint(tmp_path, monkeypatch, TIMINGS, BAND, fake)
    out, _same = S.rank_across_nodes(ctx, [2, 4, 5])
    assert sorted(out) == [5] and out[5][0][2] == "band" and fake.closed and fake.stopped
    assert "stopped with N=[2, 4] not fully timed" in capsys.readouterr().out


def test_pool_worker_runs_each_task_on_its_own_free_nodes_in_queue_order(tmp_path):
    import json as _json
    import threading as _th
    import time as _time
    q, batch = tmp_path / "pool", tmp_path / "b"
    (q / "tasks").mkdir(parents=True)
    batch.mkdir()
    (batch / "plan.json").write_text(_json.dumps(dict(runs=[])))
    tasks = [("0-001-000001", "BAND", 3)] + [("1-001-%06d" % i, "L%d" % i, 1) for i in range(2, 7)] + \
            [("1-002-000007", "HUGE", 9)]
    for name, tag, k in tasks:
        (q / "tasks" / (name + ".json")).write_text(_json.dumps(dict(batch=str(batch), run=dict(tag=tag), nodes=k)))
    (q / "CLOSE").write_text("")
    spans, guard = [], _th.Lock()

    def fake_tries(plan, r, rep, out, scratch, write, hosts):
        t0 = _time.monotonic()
        _time.sleep(0.05)
        with guard:
            spans.append((r["tag"], tuple(hosts), t0, _time.monotonic()))
        write(_json.dumps(dict(tag=r["tag"])) + "\n")
        return 1

    assert P.pool_worker(str(q), hosts=["a", "b", "c", "d"], run_tries=fake_tries, poll_s=0.01) == 0
    assert sorted(os.listdir(q / "done")) == sorted(n + ".json" for n, _t, _k in tasks)
    assert sorted(_json.loads(l)["tag"] for l in open(batch / "results.jsonl")) == sorted(
        t for _n, t, k in tasks if k <= 4)
    first = min(spans, key=lambda s: s[2])
    assert first[0] == "BAND" and len(set(first[1])) == 3
    for i, (t1, h1, a1, b1) in enumerate(spans):            # no node holds two runs at once
        for t2, h2, a2, b2 in spans[i + 1:]:
            assert not (set(h1) & set(h2) and a1 < b2 and a2 < b1), (t1, t2)


def test_on_hosts_restricts_the_launcher_to_the_pools_nodes():
    assert P.on_hosts(["srun", "--mpi=pmi2", "-n", "56", "./wrf.exe"], ["n1", "n2"], 112) == [
        "srun", "-N", "2", "-w", "n1,n2", "--exact", "--mpi=pmi2", "-n", "56", "./wrf.exe"]
    assert P.on_hosts(["mpirun", "-np", "8", "./wrf.exe"], ["n1"], 144) == [
        "mpirun", "--host", "n1:144", "-np", "8", "./wrf.exe"]
    assert P.on_hosts(["./wrf.exe"], ["n1"], 1) == ["./wrf.exe"]
    assert P.on_hosts(["srun", "./wrf.exe"], None, 1) == ["srun", "./wrf.exe"]


def test_pool_size_and_time_limit(tmp_path):
    import types as _types
    real = S._jobs.sweep.default_sbatch_runner
    ctx = _types.SimpleNamespace(sbatch_runner=real, max_nodes=8, proxy_max_nodes=0, budget_s=float("inf"),
                                 elapsed=lambda: 0.0)
    assert S._pool_nodes(ctx, 70) == 32 and S._pool_nodes(ctx, 2) == 3
    ctx.proxy_max_nodes = 8
    assert S._pool_nodes(ctx, 70) == 8
    assert S._pool_nodes(_types.SimpleNamespace(sbatch_runner=lambda c: None), 70) == 0   # fake runner
    assert S._pool_time_limit(ctx) == "03:00:00"
    ctx.budget_s, ctx.elapsed = 15 * 60.0, lambda: 120.0
    assert S._pool_time_limit(ctx) == "00:18:00"


def test_a_pool_run_crops_on_its_own_first_node_as_a_slurm_step(tmp_path, monkeypatch):
    calls = []

    class _R:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr(P.subprocess, "run", lambda argv, **k: calls.append(argv) or _R())
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.setattr(P, "crop", lambda *a, **k: pytest.fail("not in the worker's process"))
    P._crop_on(["n5", "n6"], "/c/wrfinput_d01", "/c/wrfbdy_d01", "/s/box", 10, 20, place="centre", parent_start=(3, 4))
    argv = calls[0]
    assert argv[:8] == ["srun", "-N", "1", "-n", "1", "-w", "n5", "--exact"] and argv[-3:-1] == ["wtune.proxy", "_crop"]
    got = []
    monkeypatch.setattr(P, "crop", lambda *a, **k: got.append((a, k)))
    assert P.main(["_crop", argv[-1]]) == 0
    assert got == [(("/c/wrfinput_d01", "/c/wrfbdy_d01", "/s/box", 10, 20), dict(place="centre", parent_start=(3, 4)))]
    monkeypatch.delenv("SLURM_JOB_ID")
    P._crop_on(["n5"], "a", None, "b", 1, 2)                     # no Slurm: in-process
    assert got[-1] == (("a", None, "b", 1, 2), {})


def test_pool_worker_backfills_one_node_runs_that_cannot_delay_a_waiting_band(tmp_path):
    import json as _json
    import threading as _th
    import time as _time
    q, batch = tmp_path / "pool", tmp_path / "b"
    (q / "tasks").mkdir(parents=True)
    batch.mkdir()
    (batch / "plan.json").write_text(_json.dumps(dict(runs=[])))
    tasks = [("0-001-000001", "B1", 3), ("0-001-000002", "B2", 3)] + \
            [("1-001-%06d" % i, "L%d" % i, 1) for i in range(3, 7)]
    for name, tag, k in tasks:
        (q / "tasks" / (name + ".json")).write_text(_json.dumps(dict(batch=str(batch), run=dict(tag=tag), nodes=k)))
    (q / "CLOSE").write_text("")
    starts, guard = {}, _th.Lock()

    def fake_tries(plan, r, rep, out, scratch, write, hosts):
        with guard:
            starts[r["tag"]] = _time.monotonic()
        _time.sleep(0.15)
        return 1

    P.pool_worker(str(q), hosts=list("abcde"), run_tries=fake_tries, poll_s=0.01)
    order = sorted(starts, key=starts.get)
    # B2 waits for 3 nodes; the 2 left beside B1 take L3 and L4 (B1 ending frees B2's 3 nodes alone).
    # Runs started in one poll start their threads in no fixed order: compare rounds, not ranks.
    assert set(order[:3]) == {"B1", "L3", "L4"} and starts["B2"] < starts["L5"] + 0.1
    assert not P._backfill_ok(3, 0, {})                                 # nothing running: no backfill
    assert not P._backfill_ok(3, 0, {"x": (None, ["a"], 0.0), "y": (None, list("bcd"), 1.0)})


def test_a_pool_stopped_at_the_deadline_leaves_no_crops_behind(tmp_path):
    pool = S._Pool(None, str(tmp_path), 8, "1")
    os.makedirs(tmp_path / "scratch" / "1-991-000001")
    pool.close()
    assert (tmp_path / "scratch").exists()
    pool.close(stopped=True)
    assert (tmp_path / "CLOSE").exists() and not (tmp_path / "scratch").exists()


def test_a_pool_re_time_queues_further_reps_of_the_same_runs(tmp_path):
    import json as _json
    os.makedirs(tmp_path / "pool" / "tasks")
    pool = S._Pool(None, str(tmp_path / "pool"), 8, "1")
    batch, runs = str(tmp_path / "band-N8"), [dict(tag="N8_a"), dict(tag="N8_b")]
    pool.add(batch, dict(reps=1), runs, 3, "0-991")
    pool.add(batch, dict(reps=1, changed=True), runs[:1], 3, "0-r991", rep0=2)
    tasks = sorted(os.listdir(tmp_path / "pool" / "tasks"))
    reps = [_json.load(open(tmp_path / "pool" / "tasks" / t))["rep"] for t in tasks]
    assert [t[:6] for t in tasks] == ["0-991-", "0-991-", "0-r991"] and reps == [1, 1, 2]
    assert len(_json.load(open(os.path.join(batch, "plan.json")))["runs"]) == 2
    assert pool.batches[batch] == [tasks[2]]


def test_a_pool_run_that_raises_frees_its_nodes_and_the_queue_goes_on(tmp_path, capsys):
    import json as _json
    root, batch = tmp_path / "pool", str(tmp_path / "one-N8")
    os.makedirs(root / "tasks")
    pool = S._Pool(None, str(root), 1, "1")
    pool.add(batch, dict(reps=1, max_dom=1), [dict(tag="N8_bad"), dict(tag="N8_ok")], 1, "1-991")

    def run_tries(plan, r, rep, out, scratch, write, hosts):
        if r["tag"] == "N8_bad":
            raise OSError("crop failed")
        write(_json.dumps(dict(tag=r["tag"], rep=rep, rc=0, median=1.0)) + "\n")
        return 1

    open(root / "CLOSE", "w").close()
    P.pool_worker(str(root), hosts=["h0"], run_tries=run_tries, poll_s=0.01, idle_exit_s=5)
    assert pool.done(batch) and [r["tag"] for r in P.collect(batch)] == ["N8_ok"]
    assert "failed: OSError: crop failed" in capsys.readouterr().out


def test_a_band_re_timed_in_the_pool_is_scored_over_both_runs(tmp_path):
    # the controller's queue, the worker and collect() together: a re-time queued as rep 2 of the
    # same runs runs on its own nodes and each crop position scores the median of its two runs
    import json as _json
    root, batch = tmp_path / "pool", str(tmp_path / "band-N8")
    os.makedirs(root / "tasks")
    pool = S._Pool(None, str(root), 6, "1")
    runs = [dict(tag="N8_a", layout="N8_a", pos=0), dict(tag="N8_a_p1", layout="N8_a", pos=1),
            dict(tag="N8_b", layout="N8_b", pos=0)]
    secs = {("N8_a", 1): 2.0, ("N8_a_p1", 1): 2.2, ("N8_a", 2): 2.4, ("N8_a_p1", 2): 2.2, ("N8_b", 1): 3.0}

    def run_tries(plan, r, rep, out, scratch, write, hosts):
        assert len(hosts) == 3
        write(_json.dumps(dict(tag=r["tag"], layout=r["layout"], pos=r["pos"], rep=rep, rc=0,
                               median=secs[(r["tag"], rep)])) + "\n")
        return 1

    def drain():
        open(root / "CLOSE", "w").close()
        P.pool_worker(str(root), hosts=["h%d" % i for i in range(6)], run_tries=run_tries,
                      poll_s=0.01, idle_exit_s=5)
        os.remove(root / "CLOSE")

    pool.add(batch, dict(reps=1, max_dom=1), runs, 3, "0-991")
    drain()
    assert pool.done(batch)
    assert {r["tag"]: r["median"] for r in P.collect(batch)} == {"N8_a": 2.1, "N8_b": 3.0}
    pool.add(batch, dict(reps=1, max_dom=1), runs[:2], 3, "0-r991", rep0=2)
    assert not pool.done(batch)
    drain()
    got = {r["tag"]: r for r in P.collect(batch)}
    assert pool.done(batch) and got["N8_a"]["positions"] == {0: 2.2, 1: 2.2}
    assert got["N8_a"]["median"] == pytest.approx(2.2) and len(got["N8_a"]["reps"]) == 4
    assert not os.path.exists(root / "scratch") and len(os.listdir(root / "done")) == 5


def test_a_stop_or_queue_cap_while_the_pool_waits_ranks_what_finished(tmp_path, monkeypatch, capsys):
    # wtune stop and --max-queue-hours raise QueueCapExceeded inside the wait; the real sweep keeps
    # its partial results and so must the pool, not lose every timing with the phase
    from wtune.tune.context import StopRequested
    calls = []

    def wait_for_jobs(ctx, jobs, stage, until=None):
        calls.append(jobs)
        raise StopRequested("stopped by wtune stop")
    monkeypatch.setattr(S._jobs, "wait_for_jobs", wait_for_jobs)
    pool = S._Pool(None, str(tmp_path), 4, "77")
    assert pool.wait(lambda: False) is False and calls == [["77"]]
    assert "wtune stop" in str(pool.halted)


def test_the_jobs_path_keeps_the_one_node_runs_that_finished_before_a_queue_cap(tmp_path, monkeypatch, capsys):
    from wtune.tune.context import QueueCapExceeded

    def run_jobs(*a, **k):
        e = QueueCapExceeded("queue-only wait 1:00:00 reached --max-queue-hours")
        e.partial = {"N4_24x24_o1_nt1": 0.45}
        raise e
    monkeypatch.setattr(S, "_run_jobs", run_jobs)
    monkeypatch.setattr(S, "_one_node_scores", lambda label, pool, timed, *a: dict(timed) if label == "N4" else {})
    plans = {4: ({"runs": []}, []), 5: ({"runs": []}, [])}
    one, band, _p, _b = S._time_in_jobs(None, str(tmp_path), {4: [], 5: []}, plans, str(tmp_path), 1, 1)
    assert one == {4: {"N4_24x24_o1_nt1": 0.45}} and band == {}
    assert "none re-timed" in capsys.readouterr().out


def test_rank_across_nodes_times_each_core_count_under_its_own_label_and_merges_by_n(tmp_path, monkeypatch):
    at48 = [_cfg("ANCHOR", 12, 16), _cfg("D0001", 16, 12)]                 # 4 x 48 = 192 ranks
    ctx, submitted = _joint(tmp_path, monkeypatch, POOLS, {}, {})
    monkeypatch.setattr(S, "_sweep_anchor", lambda c, n: ((at48 if c.core_budget == 48 else POOLS[n])[0], None))
    monkeypatch.setattr(S, "cores_used", lambda c: c.core_budget or 144)
    monkeypatch.setattr(S, "sample_more_diverse_configs",
                        lambda c, ex, k, seed: (at48 if c.core_budget == 48 else POOLS[c.nodes])[1:k + 1])
    timings = {"N4_24x24_o1_nt1": 0.45, "N4_12x24_o2_nt24": 0.40, "N4_18x32_o1_nt1": 0.47,
               "N4c48_12x16_o1_nt1": 0.43, "N4c48_16x12_o1_nt1": 0.41}
    band = {"N4_12x24_o2_nt24": 0.47, "N4c48_16x12_o1_nt1": 0.44}
    monkeypatch.setattr(P, "collect", lambda d: [dict(tag=t, median=m) for t, m in
                        (timings if d.endswith("jobs0") else band if "band" in os.path.basename(d) else {}).items()])
    monkeypatch.setattr(S, "PROXY_JOINT_LEADERS", 1)
    out, _same = S.rank_across_nodes(ctx, [4], cores=[144, 48])
    assert [(S._tag(c), S.cores_of(c, 4), t, f) for c, t, f in out[4][:2]] == [
        ("16x12_o1_nt1", 48, 0.44, "band"), ("12x24_o2_nt24", 144, 0.47, "band")]
    # each core count re-timed in a plan of its own: one plan's cores would launch the other's runs
    bands = [p for p in submitted if p.get("proxy_nodes") == 3]
    assert sorted(sorted({r["layout"].split("_")[0] for r in p["runs"]}) for p in bands) == [["N4"], ["N4c48"]]


def test_the_pool_job_holds_the_tunes_cores_per_node_not_the_machines(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(P, "_submit_script", lambda sb, nodes, sbatch: (seen.update(text=open(sb).read()) or ("9", "ok")))
    ctx = types.SimpleNamespace(cores_per_node=36, owned_jobs=[], budget_s=600.0, elapsed=lambda: 0.0)
    pool = S._Pool.start(ctx, str(tmp_path), 4)
    assert pool is not None and "--ntasks-per-node=36" in seen["text"]
