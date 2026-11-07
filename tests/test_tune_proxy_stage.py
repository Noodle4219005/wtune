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


def _patch(monkeypatch, extra, timings):
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
    monkeypatch.setattr(P, "collect", lambda d: [dict(tag=t, median=m) for t, m in timings.items()]
                        if d.endswith("job0") else [])
    return submitted


def test_proxy_keeps_the_anchor_and_the_proxys_fastest(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18), _cfg("G0002", 16, 36)]
    extra = [_cfg("D0001", 12, 24, omp=2, nt=24), _cfg("D0002", 9, 64, nt=13), _cfg("D0003", 18, 32)]
    timings = {"24x24_o1_nt1": 0.51, "32x18_o1_nt1": 0.56, "16x36_o1_nt1": 0.65,
               "12x24_o2_nt24": 0.40, "9x64_o1_nt13": 0.42, "18x32_o1_nt1": 0.50}
    submitted = _patch(monkeypatch, extra, timings)
    ctx = _ctx(tmp_path)
    out = S.select(ctx, configs)
    assert [c["id"] for c in out] == ["ANCHOR", "D0001", "D0002"]       # same count; anchor stays first
    assert sum(len(p["runs"]) for p in submitted) == 6 and ctx.owned_jobs == ["101", "102"]


def test_proxy_falls_back_without_netcdf_inputs(tmp_path, monkeypatch):
    configs = [_cfg("ANCHOR", 24, 24), _cfg("G0001", 32, 18)]
    _patch(monkeypatch, [], {})
    assert S.select(_ctx(tmp_path, with_inputs=False), configs) is configs


def test_proxy_skips_2d_tiles(tmp_path, monkeypatch):
    assert S._spec(_cfg("X", 8, 8, tile_strategy=3, numtiles_x=4)) is None
    assert S._spec(_cfg("Y", 8, 8, omp=2, nt=14)) == "8x8:omp=2:nt=14"
