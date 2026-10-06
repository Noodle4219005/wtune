"""Golden snapshots of src/post/cli.py, pinned before the round-3 tidy-up table-drove
build_parser and split cmd_recipe/cmd_tc: --help of every post subcommand (fixed COLUMNS),
parsed Namespace (defaults and a few explicit options) per subcommand, and stdout/rc/call kwargs
of cmd_recipe/cmd_tc/cmd_cf3d/cmd_vapor_bundle with the heavy backends stubbed.

Regenerate (only for an intended behaviour change):
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_post_cli.py
"""
import contextlib
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "fixtures", "golden", "post_cli", "post_cli.json")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"

_HELP_SNIPPET = (
    "import sys; sys.path[:0] = [%r, %r]; from post import cli; sys.exit(cli.main(sys.argv[1:]))"
    % (os.path.join(REPO, "src"), os.path.join(REPO, "src", "tools")))

SUBS = ["inventory", "extract", "plot", "sounding", "tc", "compare", "report", "recipe", "cf3d",
        "vapor-bundle", "custom", "fetch", "fetch-era5"]


def _help(argv):
    env = dict(os.environ, COLUMNS="80", LC_ALL="C")
    p = subprocess.run([sys.executable, "-c", _HELP_SNIPPET] + argv, env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def _fetch_names():
    from post import fetch
    return sorted(fetch.discover()[0])


# argv per subcommand (minimal valid, then with explicit options) -> parsed Namespace
PARSE = {
    "inventory": ["inventory", "r"],
    "inventory-json": ["inventory", "r", "--json", "--nc-cache", "c"],
    "extract": ["extract", "r", "--vars", "a,b", "--out", "o"],
    "extract-full": ["extract", "r", "--vars", "a", "--out", "o", "--domain", "2", "--times",
                     "t1,t2", "--levels", "500,850", "--points", "1,2", "--points", "3,4,N",
                     "--box", "1,2,3,4", "--workers", "4"],
    "plot": ["plot", "r", "--field", "f"],
    "plot-full": ["plot", "r", "--field", "f", "--domain", "2", "--times", "t", "--level", "500",
                  "--overlay", "wind", "--basemap", "landmask", "--format", "svg", "--out", "o",
                  "--project", "p"],
    "sounding": ["sounding", "r", "--point", "1,2", "--time", "t", "--out", "o"],
    "tc": ["tc"],
    "tc-full": ["tc", "a", "b", "--hurdat2", "h", "--storm", "S", "--bdeck", "bd", "--ebtrk", "e",
                "--track-smooth-km", "0", "--labels", "x,y", "--out", "o", "--project", "p"],
    "compare": ["compare", "a", "b"],
    "compare-full": ["compare", "a", "b", "--vars", "v", "--domain", "2", "--json"],
    "report": ["report", "o", "--title", "T"],
    "recipe": ["recipe"],
    "recipe-full": ["recipe", "a", "--out", "o", "--project", "p", "--types", "tc,x",
                    "--ensemble", "--spinup-hours", "3", "--hurdat2", "h", "--storm", "S",
                    "--bdeck", "b", "--ebtrk", "e", "--labels", "l", "--repro-pairs",
                    "--no-paraview", "--every", "6h", "--times", "t", "--animate",
                    "--track-smooth-km", "5", "--climate-spinup-days", "7", "--title", "T"],
    "cf3d": ["cf3d", "r"],
    "cf3d-full": ["cf3d", "r", "--domain", "2", "--every", "6h", "--times", "t", "--out", "o",
                  "--project", "p"],
    "vapor-bundle": ["vapor-bundle", "r"],
    "vapor-bundle-full": ["vapor-bundle", "r", "--domain", "2", "--every", "1d", "--times", "t",
                          "--allow-copy", "--out", "o", "--project", "p"],
    "custom": ["custom", "--project", "p", "--out", "o"],
    "fetch-list": ["fetch", "--list"],
    "fetch-era5": ["fetch-era5", "--project", "p", "--run-dir", "r"],
    "none": [],
}


def _parse_all():
    from post import cli
    out = {}
    for cid, argv in PARSE.items():
        ns = cli.build_parser().parse_args(argv)
        out[cid] = {k: repr(v) for k, v in sorted(vars(ns).items())}
    for name in _fetch_names():
        try:
            ns = cli.build_parser().parse_args(["fetch", name, "--project", "p"])
            out["fetch-" + name] = {k: repr(v) for k, v in sorted(vars(ns).items())}
        except SystemExit as e:
            out["fetch-" + name] = {"exit": repr(e.code)}
    return out


def _call(argv, tmp_path, monkeypatch):
    """main(argv) with the heavy backends stubbed; returns rc, stdout, stderr, backend calls."""
    from post import cf_pressure, fetch, recipe, tc, vapor_bundle
    import wtune_project
    from post import cli
    calls = []

    def rec(name, ret):
        def f(*a, **k):
            calls.append([name, repr(a), repr(sorted(k.items()))])
            return ret
        return f

    monkeypatch.setattr(recipe, "run_recipe", rec("run_recipe", {
        "figures": [{"missing": False}, {"missing": True}, {"missing": False}],
        "case_info": {"types": ["common", "tc"]}, "report_path": "REPORT",
        "notes": [{"level": "note", "source": "src", "text": "txt"}]}))
    monkeypatch.setattr(tc, "run_tc_report", rec("run_tc_report", {
        "adecks": ["a"], "figures": ["f", "g"], "warnings": ["w1"], "notes": ["n1"],
        "track_steps": {"lab": {"x": 1}}}))
    monkeypatch.setattr(tc, "format_track_steps", lambda label, st: "steps %s %r" % (label, st))
    monkeypatch.setattr(cf_pressure, "export", rec("cf_export", [
        {"domain": 1, "valid_time": "T", "error": "boom"},
        {"domain": 2, "valid_time": "T", "path": "P", "levels": [1, 2], "nlat": 3, "nlon": 4,
         "size_bytes": 3 * 2 ** 20}]))
    monkeypatch.setattr(vapor_bundle, "export", rec("vb_export", [
        {"domain": 1, "error": "bad"},
        {"domain": 2, "dir": "D", "n_files": 2, "total_bytes": 2 ** 21, "link_note": "linked",
         "readme": "/x/README.md"}]))
    monkeypatch.setattr(fetch, "offline_summary", lambda p, r: "offline-note")
    monkeypatch.setattr(wtune_project, "manifest_append",
                        rec("manifest_append", None))
    out, err = io.StringIO(), io.StringIO()
    rc = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(argv)
        except SystemExit as e:
            rc = "SystemExit(%r)" % (e.code,)
        except Exception as e:  # noqa: BLE001 -- pinned by type and message
            rc = "%s(%s)" % (type(e).__name__, e)
    norm = lambda s: s.replace(str(tmp_path), "<TMP>")  # noqa: E731
    return {"rc": rc, "stdout": norm(out.getvalue()), "stderr": norm(err.getvalue()),
            "calls": [[norm(c) for c in call] for call in calls]}


def _calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    proj = tmp_path / "proj"
    (proj / "runs" / "r1").mkdir(parents=True)
    (proj / "runs" / "r1" / "wrfout_d01_2020-01-01_00:00:00").write_text("")
    (proj / "project.yaml").write_text("name: x\nstorm:\n  id: AL012020\n  hurdat2: /h.txt\n")
    P = str(proj)
    cases = {
        "recipe-plain": ["recipe", "a", "b", "--out", "o"],
        "recipe-no-rundirs": ["recipe", "--out", "o"],
        "recipe-no-out": ["recipe", "a"],
        "recipe-label-mismatch": ["recipe", "a", "b", "--labels", "x", "--out", "o"],
        "recipe-full": ["recipe", "a", "--out", "o", "--types", "tc,x", "--ensemble",
                        "--spinup-hours", "3", "--hurdat2", "h", "--storm", "S", "--bdeck", "b",
                        "--ebtrk", "e", "--labels", "l", "--repro-pairs", "--no-paraview",
                        "--every", "6h", "--times", "t1,t2", "--animate", "--title", "T",
                        "--track-smooth-km", "5", "--climate-spinup-days", "7"],
        "recipe-types-auto": ["recipe", "a", "--out", "o", "--types", "auto"],
        "recipe-project": ["recipe", "--project", P],
        "recipe-project-explicit-runs": ["recipe", "x/y", "--project", P, "--out", "o"],
        "tc-plain": ["tc", "a", "--hurdat2", "h", "--storm", "S", "--out", "o"],
        "tc-no-rundirs": ["tc", "--out", "o"],
        "tc-no-storm": ["tc", "a", "--out", "o"],
        "tc-no-out": ["tc", "a", "--hurdat2", "h", "--storm", "S"],
        "tc-label-mismatch": ["tc", "a", "b", "--hurdat2", "h", "--storm", "S", "--labels", "x",
                              "--out", "o"],
        "tc-project": ["tc", "--project", P],
        "cf3d-plain": ["cf3d", "r", "--out", "o", "--every", "6h", "--times", "t1,t2"],
        "cf3d-no-out": ["cf3d", "r"],
        "cf3d-project": ["cf3d", "r", "--project", P, "--domain", "2"],
        "vapor-plain": ["vapor-bundle", "r", "--out", "o", "--allow-copy", "--times", "t"],
        "vapor-no-out": ["vapor-bundle", "r"],
        "vapor-project": ["vapor-bundle", "r", "--project", P],
        "no-command": [],
    }
    out = {}
    for cid, argv in cases.items():
        out[cid] = _call(argv, tmp_path, monkeypatch)
    text = re.sub(r"\d{12,14}-(vapor|cf3d|recipe|tc)", r"<ID>-\1", json.dumps(out).replace(P, "<PROJ>"))
    return json.loads(text)


def _collect(tmp_path, monkeypatch):
    helps = {"top": _help(["--help"])}
    for s in SUBS:
        helps[s] = _help([s, "--help"])
    for n in _fetch_names():
        helps["fetch-" + n] = _help(["fetch", n, "--help"])
    return {"help": helps, "parse": _parse_all(), "calls": _calls(tmp_path, monkeypatch)}


def test_post_cli_matches_golden(tmp_path, monkeypatch):
    got = json.loads(json.dumps(_collect(tmp_path, monkeypatch)))
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(got, fh, indent=1, sort_keys=True)
            fh.write("\n")
    with open(GOLDEN) as fh:
        want = json.load(fh)
    assert got == want
