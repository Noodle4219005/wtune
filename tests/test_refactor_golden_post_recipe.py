"""Golden snapshot of post/recipe.py's orchestration (run_recipe/_run_recipe): which recipe
module is called for each detected type, in which order, with which arguments, how notes are split
from figures and what the report builder receives -- all figure backends stubbed. Pinned before the
round-3 tidy-up split _run_recipe.

Regenerate (only for an intended behaviour change):
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_post_recipe.py
"""
import json
import os


from post import caseinfo, recipe

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden",
                      "post_recipe", "recipe_calls.json")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"

ALL_TYPES = ["common", "tc", "convection", "cloud", "terrain", "moisture", "precip", "synoptic",
             "pbl", "chem", "climate", "ensemble", "unknown-type"]

CASES = {
    "common-only": dict(types=["common"]),
    "all-types-track": dict(types=ALL_TYPES, hurdat2_path="h.txt", storm_id="AL012020",
                            bdeck_path="b", ebtrk_path="e", paraview=False, spinup_hours=3,
                            cross_sections={"terrain": {"t": 1}, "moisture": {"m": 2}},
                            xsec_top_km=12.0, xsec_halfwidth_km=300.0, project_dir="P", run_dirs=["r1", "r2"],
                            spinup_days=7.0, track_smooth_km=5.0, every="6h", times=["t1"],
                            animate=True, labels=["a", "b"], title="T"),
    "tc-no-track": dict(types=["common", "tc"], every=None),
    "moisture-centre": dict(types=["common", "moisture"], hurdat2_path="h.txt", storm_id="S"),
    "moisture-centre-fails": dict(types=["common", "moisture"], bdeck_path="b", centre="raise"),
    "pre-notes-and-advice": dict(types=["common", "precip"], pre_notes=[
        {"kind": "note", "source": "history", "level": "warning", "text": "x"}], advice=True),
    "advice-skipped": dict(types=["common", "convection"], advice="none"),
    "label-mismatch": dict(types=["common"], labels=["only-one"], run_dirs=["r1", "r2"]),
}


def _recorder(log, name, ret):
    def f(*a, **k):
        log.append([name, repr(a), repr(sorted(k.items()))])
        return list(ret) if isinstance(ret, list) else ret
    return f


def _fig(i):
    return recipe.fig("f_" + i, "/p/" + i + ".png", "run")


def _run(tmp, monkeypatch, name, cfg):
    cfg = dict(cfg)
    types = cfg.pop("types")
    run_dirs = cfg.pop("run_dirs", ["r1"])
    centre = cfg.pop("centre", None)
    advice = cfg.pop("advice", None)
    log = []
    monkeypatch.setattr(caseinfo, "detect_types", _recorder(
        log, "detect_types", dict(types=types, domains=[{"domain": 1}, {"domain": 2}], evidence={})))
    import post.recipes.common as common_mod
    import post.recipes.vapor_bundle as vb_mod
    monkeypatch.setattr(common_mod, "figures", _recorder(log, "common", [_fig("common")]))
    monkeypatch.setattr(vb_mod, "figures", _recorder(log, "vapor", [_fig("vapor")]))
    monkeypatch.setattr(recipe, "_cf3d_data", _recorder(log, "cf3d", [_fig("cf3d")]))
    for t in list(recipe.TYPE_DISPATCH):
        monkeypatch.setitem(recipe.TYPE_DISPATCH, t, _recorder(
            log, "dispatch-" + t, [_fig(t), recipe.note("src-" + t, "n-" + t, level="info")]))
    monkeypatch.setattr(recipe, "_tc_structure_figures", _recorder(log, "tc_structure", [_fig("tcs")]))
    monkeypatch.setattr(recipe, "_time_series_figures", _recorder(log, "timeseries", [_fig("ts")]))
    if centre == "raise":
        def boom(*a, **k):
            raise RuntimeError("no track")
        monkeypatch.setattr(recipe, "_tc_centre_from_track", boom)
    else:
        monkeypatch.setattr(recipe, "_tc_centre_from_track", _recorder(log, "tc_centre", (1.5, 2.5)))
    adv = {"advice": [{"id": "x"}]} if advice is True else None
    if advice == "none":
        monkeypatch.setattr(recipe, "_verification_advice", lambda *a: log.append(["advice-none"]) or None)
    else:
        monkeypatch.setattr(recipe, "_verification_advice",
                            lambda *a, **k: log.append(["advice", repr(a[1:5])]) or adv)
    if advice is True:
        from wtune import advise as wtune_advise
        monkeypatch.setattr(wtune_advise, "render_html", lambda a: "<advice/>")
    monkeypatch.setattr(recipe, "load_catalog", lambda: {"cat": 1})
    import post.report as report

    def fake_report(out_dir, case_info, figures, run_dirs, labels, spinup_hours, catalog,
                    title=None, notes=None, advice_html=None):
        log.append(["report", repr((case_info, figures, run_dirs, labels, spinup_hours, catalog,
                                    title, notes, advice_html))])
        return "REPORT"
    monkeypatch.setattr(report, "build_recipe_report", fake_report)
    out = str(tmp / name)
    try:
        res = recipe._run_recipe(run_dirs, out, **cfg)
        result = {k: (v if k != "out_dir" else "<OUT>") for k, v in res.items()}
    except Exception as e:  # noqa: BLE001 -- pinned by type + message
        result = "%s: %s" % (type(e).__name__, e)
    text = json.dumps({"result": result, "log": log, "out_exists": os.path.isdir(out)},
                      sort_keys=True, default=str)
    return json.loads(text.replace(out, "<OUT>"))


def test_recipe_orchestration_matches_golden(tmp_path, monkeypatch):
    got = {n: _run(tmp_path, monkeypatch, n, c) for n, c in CASES.items()}
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(got, fh, indent=1, sort_keys=True)
            fh.write("\n")
    with open(GOLDEN) as fh:
        want = json.load(fh)
    assert got == want
