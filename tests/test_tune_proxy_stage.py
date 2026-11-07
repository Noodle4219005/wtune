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
