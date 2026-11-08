"""post/recipe.py: project_dir reaches a recipe module only when its figures() accepts it, and
notes are split from figures in run_recipe's result."""
import os
import sys
import types

from wtune.post import recipe


def _fake_module(name, with_project):
    mod = types.ModuleType("wtune.post.recipes." + name)
    seen = {}
    if with_project:
        def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6, project_dir=None):
            seen["project_dir"] = project_dir
            return [recipe.note("fake", "hello")]
    else:
        def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6):
            seen["called"] = True
            return []
    mod.figures = figures
    sys.modules["wtune.post.recipes." + name] = mod
    return seen


def test_spinup_days_passed_only_when_accepted_and_given(tmp_path):
    mod = types.ModuleType("wtune.post.recipes.fakeclim")
    seen = {}

    def figures(run_dirs, labels, out_dir, case_info, spinup_days=30):
        seen["d"] = spinup_days
        return []
    mod.figures = figures
    sys.modules["wtune.post.recipes.fakeclim"] = mod
    run = recipe._simple_dispatch("fakeclim", "sub")
    run(["r"], ["l"], str(tmp_path), {}, spinup_days=10.0)
    assert seen["d"] == 10.0
    run(["r"], ["l"], str(tmp_path), {})              # not given -> module default
    assert seen["d"] == 30
    _fake_module("fakenoproj2", False)
    recipe._simple_dispatch("fakenoproj2", "sub")(["r"], ["l"], str(tmp_path), {}, spinup_days=5.0)


def test_every_figure_is_also_in_one_flat_folder(tmp_path):
    out = tmp_path / "post"
    paths = ["tc/track_map.png", "synoptic/syn_slp.png", "timeseries/f048/slp_d01_f048.png",
             "common_domain_map.png", "terrain/a/x.png", "terrain/b/x.png", "tc/obs.csv"]
    for p in paths:
        (out / p).parent.mkdir(parents=True, exist_ok=True)
        (out / p).write_text(p)
    figs = [recipe.fig("f", str(out / p), "run0") for p in paths]
    figs.append(dict(id="m", missing=True, path=None))
    dest = recipe.figures_folder(str(out), figs)
    assert sorted(os.listdir(dest)) == ["common_domain_map.png", "synoptic-syn_slp.png",
                                        "tc-track_map.png", "terrain-x-2.png", "terrain-x.png",
                                        "timeseries-slp_d01_f048.png"]
    assert (out / "figures" / "tc-track_map.png").read_text() == "tc/track_map.png"
    recipe.figures_folder(str(out), figs[:1])                  # a re-run drops stale names
    assert os.listdir(dest) == ["tc-track_map.png"]
