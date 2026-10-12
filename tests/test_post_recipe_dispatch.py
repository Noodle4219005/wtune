"""post/recipe.py: project_dir reaches a recipe module only when its figures() accepts it, and
notes are split from figures in run_recipe's result."""
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


def test_simple_dispatch_passes_project_dir_only_when_accepted(tmp_path, monkeypatch):
    seen = _fake_module("fakeproj", True)
    out = recipe._simple_dispatch("fakeproj", "sub")(["r"], ["l"], str(tmp_path), {},
                                                    project_dir="/some/project")
    assert seen["project_dir"] == "/some/project"
    assert out == [recipe.note("fake", "hello")]
    seen2 = _fake_module("fakenoproj", False)
    recipe._simple_dispatch("fakenoproj", "sub")(["r"], ["l"], str(tmp_path), {},
                                                 project_dir="/some/project")
    assert seen2["called"]


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


def test_cli_exposes_climate_spinup_days():
    from wtune.post import cli
    a = cli.build_parser().parse_args(["recipe", "--climate-spinup-days", "12"])
    assert a.climate_spinup_days == 12.0
    assert cli.build_parser().parse_args(["recipe"]).climate_spinup_days is None
