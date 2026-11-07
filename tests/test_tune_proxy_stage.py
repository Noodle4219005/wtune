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
    assert sum(len(p["runs"]) for p in one) == 6 and ctx.owned_jobs[:2] == ["101", "102"]
    bands = [p for p in submitted if p.get("proxy_nodes") == 3]            # N=4 > 3: the 4 leaders as bands
    assert [r["layout"] for p in bands for r in p["runs"]] == ["12x24_o2_nt24", "9x64_o1_nt13",
                                                               "18x32_o1_nt1"]   # 32x18: 144 ranks per node, not whole rows


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
    out = S.rank_across_nodes(ctx, [1, 2, 4, 5])                     # N=1: no ANCHOR, skipped
    one = [p for p in submitted if p["proxy_nodes"] == 1]
    tags = [r["tag"] for p in one for r in p["runs"]]
    assert sorted(tags) == sorted(timings) and all(r["layout"] == r["tag"] for p in one for r in p["runs"])
    assert any(len({r["tag"].split("_")[0] for r in p["runs"]}) > 1 for p in one)   # N mixed in a job
    bands = [r["layout"] for p in submitted if p["proxy_nodes"] == 3 for r in p["runs"]]
    # the 2 one-node leaders of each N > 3; N=5's 20x36 holds no whole rows per node
    assert sorted(bands) == ["N4_12x24_o2_nt24", "N4_24x24_o1_nt1", "N5_24x30_o1_nt1"]
    full = [r["layout"] for p in submitted if p["proxy_nodes"] == 2 for r in p["runs"]]
    assert sorted(full) == ["N2_12x24_o1_nt1", "N2_16x18_o1_nt1"]     # N=2: the whole domain on 2 nodes
    assert sorted(out) == [2, 4, 5]
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
    assert S.rank_across_nodes(ctx, [2, 4]) == {}


def test_select_with_a_joint_ranking_submits_nothing(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    submitted = _patch(monkeypatch, [], {})
    ctx = _ctx(tmp_path, with_inputs=False)
    ctx.proxy_ranked = [(_cfg("D0001", 12, 24, omp=2, nt=24), 0.46, "band"), (_cfg("ANCHOR", 24, 24), 0.47, "band"),
                        (_cfg("D0002", 18, 32), 0.47, "one-node"), (_cfg("D0003", 9, 64, nt=13), 0.49, "one-node")]
    out = S.select(ctx, configs)
    assert [c["id"] for c in out] == ["ANCHOR", "D0001", "D0002"] and ctx.proxy_used
    assert submitted == [] and ctx.owned_jobs == []
