#!/usr/bin/env python3
"""Characterization (golden) tests for campaign/configs.py, pinned before it was split.

Generated from the UNCHANGED code (integration-083 cf740cb). Three snapshots:
  cli.json      `python campaign/configs.py <subcommand>` as strict_campaign.sbatch / nc2bp.sbatch
                run it: rc, stdout, stderr and the files written, one subprocess per case
  api.json      one fresh interpreter: edit_namelist over fixtures/golden/configs/namelists x
                option sets, _apply_adios2 under each WTUNE_* environment, physics templates,
                BP input plans, namelist file staging, main() for every (case, template)
  surface.json  every name on the module: kind, parameters, constant values (names may be
                added by a refactor, never removed or changed)
Paths are normalised to <TMP> / <FIX> / <REPO>; dict key order is compared.

The namelists are the repo's own (tests/fixtures/post, tests/fixtures/golden/case_rebuild:
repo__*), the constants of campaign/test_configs.py (tc__*) and one packed two-domain
sst_update + fdda namelist (syn__*). Regenerate only for an intended behaviour change:
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_configs.py
"""
import collections
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CAMPAIGN = os.path.join(REPO, "campaign")
CONFIGS_PY = os.path.join(CAMPAIGN, "configs.py")
REAL_CASES = os.path.join(CAMPAIGN, "cases.json")
FIX = os.path.join(HERE, "fixtures", "golden", "configs")
NML = os.path.join(FIX, "namelists")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
OD = collections.OrderedDict

RECORD_TIMES = ("2020-01-01_00:00:00", "2020-01-01_06:00:00")
STREAM_NL = "tc__Adios2InputStaging__NL.nml"      # sst_update + grid_fdda, 2 domains
MMLSQUOTA = (
    "                         Block Limits                                    |     File Limits\n"
    "Filesystem type         KB      quota      limit   in_doubt    grace |    files   quota\n"
    "home1      USR   943718400 1048576000 1153433600          0     none |   812345       0\n"
    "work2      USR            none\n")
RSL = "".join("Timing for main: time 2020-01-01_00:%02d:00 on domain   1: %8.5f elapsed seconds\n"
              % (m, 1.5 + 0.25 * (m % 3) + (9.0 if m == 4 else 0.0)) for m in range(1, 9)) + \
      "Timing for main: time 2020-01-01_00:01:00 on domain   2:    0.50000 elapsed seconds\n"
GENERATED = [
    OD([("id", "G0001"), ("ranks", 96), ("omp", 3), ("nx", 8), ("ny", 12), ("numtiles", 4),
        ("dist", "cyclic"), ("blocktime", "0"), ("patch_x", 113), ("patch_y", 76),
        ("numtiles_x", 2), ("numtiles_y", 2), ("tile_strategy", 3),
        ("physics_namelists", {"karina": "/pts/karina/namelist.P02", "lafe": ""}),
        ("window_frame", "mid_end")]),
    OD([("id", "G0002"), ("ranks", 72), ("omp", 4), ("nx", 8), ("ny", 9), ("numtiles", 12),
        ("dist", "block"), ("blocktime", "default"), ("config_source", "seed"),
        ("window_frame", True)]),
    OD([("id", "G0001"), ("ranks", 48), ("omp", 6), ("nx", 6), ("ny", 8), ("numtiles", 1),
        ("dist", "cyclic"), ("blocktime", "0")]),
    OD([("ranks", 1), ("omp", 1)]),
]
FIDELITY = OD([("label", "SMALL DRIFT"), ("deciding", "T2 at 6 h"), ("per_field_summary_at_last_time", OD([
    ("T2", OD([("nd", 0.25), ("floor_ratio", 3.5), ("floor_zero", False)])),
    ("U10", OD([("nd", 1.125), ("floor_ratio", True)])), ("bad", "x")]))])
PHYSICS = OD([("mp_physics", 8), ("cu_physics", 1), ("bl_pbl_physics", 1), ("sf_sfclay_physics", 1),
              ("sf_surface_physics", 2), ("ra_lw_physics", 4), ("ra_sw_physics", 4), ("radt", 12)])


def fixture_cases(nml=NML):
    """A cases.json over the fixture namelists; real ids, since templates name cases."""
    def case(namelist, **kw):
        c = OD([("prep_dir", os.path.join(nml, "prep_unused")),
                ("namelist_template", os.path.join(nml, namelist)),
                ("e_we", 425), ("e_sn", 300), ("e_vert", 45), ("dt", 72), ("steps", 40),
                ("physics", PHYSICS)])
        c.update(kw)
        return c
    conus12 = case("unused")
    del conus12["namelist_template"]
    conus12["prep_dir"] = os.path.join(FIX, "prep_conus12")
    return OD([
        ("karina", case("repo__rebuild_wrfbdy_times.nml")),
        ("maria1km", case("repo__m1.nml", max_dom=2, e_we_list=[170, 244], e_sn_list=[130, 175])),
        ("isc21", case("repo__rebuild_nested_suite.nml", max_dom=3)),
        ("lafe", case("syn__sst_fdda_packed.nml", max_dom=3,
                      namelist_overrides=OD([("max_dom", 1)]))),
        ("sc26p1", case("repo__m1.nml", max_dom=2, frozen_keys=["time_step", "Debug_Level"])),
        ("bigchill", case("tc__DiagnosticsAndCadenceTemplates__NL.nml", max_dom=2)),
        ("conus12", conus12),
        ("gone", case("does_not_exist.nml")),
        ("off", case("repo__m1.nml", enabled=False, reason="fixture: switched off")),
        ("noreason", case("repo__m1.nml", enabled=False)),
    ])


# --------------------------------------------------------------------------- shared helpers
def _env(tmp, **extra):
    drop = ("SITE", "CASES_JSON", "CONFIG_FILE", "PYTHONPATH", "SLURM_JOB_ID")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("WTUNE_", "FAKE_BP_")) and k not in drop}
    env.update(COLUMNS="80", LC_ALL="C", PYTHONHASHSEED="0", NO_COLOR="1", HOME=tmp,
               PYTHONDONTWRITEBYTECODE="1")
    env.update(extra)
    return env


def _paths(text, tmp):
    for path, token in ((tmp, "<TMP>"), (FIX, "<FIX>"), (REPO, "<REPO>")):
        text = text.replace(path, token)
    return re.sub(r"(bp-cache/[\w.-]+/)[0-9a-f]{12}\b", r"\1<SHA12>", text)


def _norm(text, tmp):
    """_paths, and an uncaught exception reduced to its last line."""
    text = _paths(text, tmp)
    if "Traceback (most recent call last):" in text:     # frames move with the code: last line
        head = text.split("Traceback (most recent call last):")[0]
        last = [ln for ln in text.splitlines() if ln.strip()][-1]
        text = head + "Traceback (most recent call last):\n  ...\n" + last + "\n"
    return text


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


def _mk_bp(path):
    os.makedirs(path)
    for name in ("md.idx", "data.0"):
        open(os.path.join(path, name), "w").close()


def _mk_records(container, name, times=RECORD_TIMES):
    for t in times:
        _mk_bp(os.path.join(container, "%s_%s.bp" % (name, t)))


def _tree(root, tmp):
    """{relative path: 'dir' | 'link -> target' | file text} below root, sorted."""
    out = OD()
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(dirs + files):
            path = os.path.join(base, name)
            rel = re.sub(r"(^|/)[0-9a-f]{12}(?=/|$)", r"\1<SHA12>", os.path.relpath(path, root))
            if os.path.islink(path):
                out[rel] = "link -> " + _norm(os.readlink(path), tmp)
            elif os.path.isdir(path):
                out[rel] = "dir"
            else:
                with open(path, errors="replace") as f:
                    out[rel] = _norm(f.read(), tmp)
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(base, d))]
    return out


def _read(name):
    with open(os.path.join(NML, name)) as f:
        return f.read()


def _namelists():
    return OD((n, _read(n)) for n in sorted(os.listdir(NML)) if n.endswith(".nml"))


FAKE_BP = '''"""Stand-in for src/wtune_bp: the real per-record helpers, a fake conversion."""
import importlib.util, os
_spec = importlib.util.spec_from_file_location("wtune_bp_real", os.environ["FAKE_BP_REAL"])
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)
PER_RECORD_NAME_RE = _real.PER_RECORD_NAME_RE
is_per_record_dir, record_files = _real.is_per_record_dir, _real.record_files
bp_input_links = _real.bp_input_links
per_record_namelist_edits = _real.per_record_namelist_edits
ensure_dir, open_lock = _real.ensure_dir, _real.open_lock
def _mk_bp(p):
    os.makedirs(p)
    for f in ("md.idx", "data.0"):
        open(os.path.join(p, f), "w").close()
def require_adios2_python():
    if os.environ.get("FAKE_BP_FAIL"):
        raise RuntimeError("no Python that can import adios2")
    return "/fake/python"
def cached_inputs(case_dir, cache_dir, names, engine="BP4"):
    if all(os.path.isdir(os.path.join(cache_dir, n)) for n in names):
        return {n: os.path.join(cache_dir, n) for n in names}
    return None
def convert_inputs(case_dir, cache_dir, names=None, engine="BP4"):
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for n in names:
        dst = os.path.join(cache_dir, n)
        if not os.path.isdir(dst):
            with open(os.environ["FAKE_BP_LOG"], "a") as f:
                f.write("convert %s %s\\n" % (n, engine))
            if PER_RECORD_NAME_RE.match(n):
                for t in ("2020-01-01_00:00:00", "2020-01-01_06:00:00"):
                    _mk_bp(os.path.join(dst, "%s_%s.bp" % (n, t)))
            else:
                _mk_bp(dst)
        out[n] = dst
    return out
'''


def _mk_prep(prep, names=("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01", "wrflowinp_d01",
                          "wrflowinp_d02"), namelist=STREAM_NL):
    for n in names:
        _write(os.path.join(prep, n), "CDF")
    return _write(os.path.join(prep, "namelist.input"), _read(namelist))


def _mk_bp_layouts(root):
    """Input dirs covering every branch of resolve_bp_inputs -> {layout: (namelist text, dir)}."""
    text = _read(STREAM_NL)
    out = OD()
    d = os.path.join(root, "netcdf")
    _mk_prep(d)
    out["netcdf"] = (text, d)
    d = os.path.join(root, "bp")                  # BP only: single, suffix-less, per record
    for n in ("wrfinput_d01.bp", "wrfinput_d02.bp", "wrfbdy_d01.bp"):
        _mk_bp(os.path.join(d, n))
    _mk_records(os.path.join(d, "wrflowinp_d01"), "wrflowinp_d01")
    _mk_records(os.path.join(d, "wrflowinp_d02.bp"), "wrflowinp_d02")
    out["bp"] = (text, d)
    d = os.path.join(root, "both")                # netCDF next to a single multi-record BP
    _mk_prep(d, names=("wrfinput_d01", "wrfbdy_d01", "wrflowinp_d01"))
    for n in ("wrfinput_d01.bp", "wrfbdy_d01.bp", "wrflowinp_d02.bp"):
        _mk_bp(os.path.join(d, n))
    _mk_records(os.path.join(d, "wrffdda_d01"), "wrffdda_d01")
    out["both"] = (text, d)
    d = os.path.join(root, "missing")
    _mk_prep(d, names=("wrfinput_d01",))
    out["missing"] = (text, d)
    d = os.path.join(root, "dated")               # a '<date>' inname is globbed
    _mk_prep(d, names=("wrfinput_d01", "wrfbdy_d01", "wrfchemi_d01_2020-01-01_00:00:00"))
    _mk_bp(os.path.join(d, "wrfchemi_d01_2020-01-01_06:00:00.bp"))
    out["dated"] = (
        "&time_control\n auxinput5_inname = 'wrfchemi_d<domain>_<date>',\n"
        " io_form_auxinput5 = 2,\n/\n&domains\n max_dom = 1,\n/\n", d)
    return out


def _mk_stage_run(d):
    """A run dir + two search dirs for link-namelist-files -> (namelist, run dir, search dirs)."""
    src, other, run = (os.path.join(d, n) for n in ("src", "other", "run"))
    for n in ("io/fields_d02.txt", "wrflowinp_d01", "wrflowinp_d02", "OBS_DOMAIN201", "tslist",
              "wrfchemi_d01_2020-01-01_00:00:00"):
        _write(os.path.join(src, n), "x")
    _write(os.path.join(other, "iofields_d01.txt"),
           "+:h:0:T2,Q2\n+:h:0:U10\n# note\n-:h:0:W\n-:h:0:P,PB\n")
    _write(os.path.join(other, "wrflowinp_d01"), "second dir: not used")
    _mk_records(os.path.join(src, "wrffdda_d01"), "wrffdda_d01")      # a directory: not staged
    _write(os.path.join(run, "wrflowinp_d02"), "already staged")
    nl = _write(os.path.join(run, "namelist.input"), _read("tc__NamelistFileStaging__NML.nml"))
    return nl, run, [src, other]


# --------------------------------------------------------------------------- CLI snapshot
def _cli_cases(tmp):
    """[(id, argv, env, stdin, [paths to snapshot after the run])]; inputs are made here."""
    inp = os.path.join(tmp, "in")
    cases_json = _write(os.path.join(inp, "cases.json"),
                        json.dumps({"cases": fixture_cases()}, indent=1))
    gen_list = _write(os.path.join(inp, "gen_list.json"), json.dumps(GENERATED))
    gen_dict = _write(os.path.join(inp, "gen_dict.json"), json.dumps({"configs": GENERATED[:2]}))
    gen_empty = _write(os.path.join(inp, "gen_empty.json"), json.dumps({"configs": [{"x": 1}]}))
    steps = _write(os.path.join(inp, "steps.txt"),
                   "".join("%s\n" % v for v in (30.5, 1.5, 1.75, 2.0, 12.25, 1.5, 1.75, 2.0)) + "\n")
    one_step = _write(os.path.join(inp, "one_step.txt"), "30.5\n")
    rsl = _write(os.path.join(inp, "rsl.out.0000"), RSL)
    fake_src = os.path.join(inp, "fake_src")
    _write(os.path.join(fake_src, "wtune", "bp.py"), FAKE_BP)
    bp_rec = os.path.join(inp, "bp_rec")
    for n in ("wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02"):
        _mk_records(os.path.join(bp_rec, n), n)
    fx = dict(CASES_JSON=cases_json)
    nl = lambda name: os.path.join(NML, name)
    out = []

    def add(cid, argv, env=None, stdin=None, snap=()):
        d = os.path.join(tmp, "cli", cid)
        os.makedirs(d, exist_ok=True)
        out.append((cid, [a.replace("{D}", d) for a in argv],
                    {k: v.replace("{D}", d) for k, v in (env or {}).items()}, stdin,
                    [p.replace("{D}", d) for p in snap]))
        return d

    # --- the STRICT table on the real cases.json (no file is read) -------------------------
    add("usage", [])
    add("all", ["--all"])
    for case in ("karina", "conus12", "isc21", "sc26p1", "c34"):
        add("case-" + case, [case])
    add("case-unknown", ["nosuchcase"])
    add("case-scc-288", ["conus12"], dict(SITE="scc", WTUNE_TOTAL_CORES="288"))
    add("case-72-cores", ["karina"], dict(WTUNE_TOTAL_CORES="72"))
    add("case-disabled", ["off"], fx)
    add("case-disabled-noreason", ["noreason"], fx)
    add("order", ["order", "karina"])
    add("order-reps3", ["order", "karina", "--reps", "3"])
    add("order-reps0", ["order", "karina", "--reps", "0"])
    add("order-unknown", ["order", "nosuchcase"])
    add("order-disabled", ["order", "off"], fx)
    add("order-noargs", ["order"])
    add("order-help", ["order", "--help"])
    add("generated-case", ["karina"], dict(CONFIG_FILE=gen_list))
    add("generated-case-dict", ["karina"], dict(CONFIG_FILE=gen_dict))
    add("generated-case-unknown", ["nosuchcase"], dict(CONFIG_FILE=gen_list))
    add("generated-order", ["order", "karina", "--reps", "2"], dict(CONFIG_FILE=gen_dict))
    add("config-extras", ["config-extras", "karina"], dict(CONFIG_FILE=gen_list))
    add("config-extras-other-case", ["config-extras", "lafe"], dict(CONFIG_FILE=gen_dict))
    add("config-extras-no-env", ["config-extras", "karina"])
    add("config-extras-argc", ["config-extras"], dict(CONFIG_FILE=gen_list))
    # --- physics templates on the fixture cases ---------------------------------------------
    for case in ("karina", "maria1km", "isc21", "sc26p1", "gone"):
        add("physics-list-" + case, ["physics-list", case], fx)
    add("physics-list-json", ["physics-list", "karina", "--json"], fx)
    add("physics-list-unknown", ["physics-list", "nosuchcase"], fx)
    add("physics-list-help", ["physics-list", "--help"])
    add("restart-time", ["restart-time", "maria1km", "30"], fx)
    add("restart-time-fraction", ["restart-time", "lafe", "7.5"], fx)
    add("restart-time-prep-dir", ["restart-time", "conus12", "90"], fx)
    add("restart-time-unknown", ["restart-time", "nosuchcase", "30"], fx)
    add("restart-time-argc", ["restart-time", "maria1km"], fx)
    add("restart-time-no-namelist", ["restart-time", "gone", "30"], fx)
    # --- parsers the sbatch pipes into `read` -------------------------------------------------
    add("step-stats", ["step-stats", steps])
    add("step-stats-one", ["step-stats", one_step])
    add("step-stats-missing", ["step-stats", "{D}/nothing.txt"])
    add("step-stats-help", ["step-stats", "--help"])
    add("fused-steps", ["fused-steps", rsl, "--spin-seconds", "180", "--dt", "60"])
    add("fused-steps-all-spinup", ["fused-steps", rsl, "--spin-seconds", "3600", "--dt", "60"])
    add("fused-steps-missing", ["fused-steps", "{D}/nothing", "--spin-seconds", "1", "--dt", "1"])
    add("fused-steps-help", ["fused-steps", "--help"])
    add("point-plan", ["point-plan", "ANCHOR", "G1", "G2", "ANCHOREND"])
    add("point-plan-fused", ["point-plan", "--fused", "1", "ANCHOR", "G1", "ANCHOREND"])
    add("point-plan-fused-anchor-end", ["point-plan", "--fused", "1", "--anchor-end", "1",
                                        "ANCHOR", "G1"])
    add("point-plan-fused-many", ["point-plan", "--fused", "1", "ANCHOR", "G1", "G2"])
    add("point-plan-fused-frame", ["point-plan", "--fused", "1", "--window-frame", "1", "G'1"])
    add("point-plan-no-anchor-end", ["point-plan", "--anchor-end", "0", "ANCHOR", "G1",
                                     "ANCHOREND"])
    add("point-plan-empty", ["point-plan"])
    add("point-plan-help", ["point-plan", "--help"])
    add("quota-free-gib", ["quota-free-gib"], stdin=MMLSQUOTA)
    add("quota-free-gib-work2", ["quota-free-gib", "--filesystem", "work2"], stdin=MMLSQUOTA)
    add("quota-free-gib-garbage", ["quota-free-gib"], stdin="mmlsquota: command not found\n")
    add("quota-free-gib-help", ["quota-free-gib", "--help"])
    # --- edit-namelist: the command line strict_campaign.sbatch builds -----------------------
    base = ["--nproc-x", "7", "--nproc-y", "32", "--numtiles", "12", "--run-minutes", "8"]
    edit = lambda cid, name, args, env=None: add(
        "edit-" + cid, ["edit-namelist", nl(name), "{D}/namelist.input"] + base + args, env,
        snap=["{D}"])
    edit("timing", "repo__m1.nml", ["--ranks", "224", "--max-dom", "2"])
    edit("sbatch-full", "syn__sst_fdda_packed.nml", [
        "--ranks", "72", "--max-dom", "2", "--history-mode", "end_d01", "--set", "max_dom=2",
        "--set", "restart=.false.", "--delete-key", "opt_run", "--delete-key",
        "BL_MYNN_TKEBUDGET", "--numtiles-x", "2", "--numtiles-y", "6", "--tile-sz-x", "0",
        "--tile-sz-y", "0", "--tile-strategy", "3"])
    edit("interval-physics", "repo__m1.nml", [
        "--ranks", "224", "--max-dom", "2", "--history-mode", "interval", "--history-minutes",
        "30", "--physics-template", "P05", "--physics-case", "karina", "--nest-history-on"])
    edit("mid-end", "repo__rebuild_nested_suite.nml", ["--ranks", "112", "--max-dom", "2",
                                                       "--history-mode", "mid_end_d01"])
    edit("no-adios2", "tc__QuiltingOff__LAFE_LIKE.nml", ["--no-adios2", "--max-dom", "2"])
    edit("anchor-spinup", "repo__m1.nml", ["--ranks", "224", "--max-dom", "2",
                                           "--write-restart-interval", "30",
                                           "--restart-io-form", "2", "--run-seconds", "45"])
    edit("restart-from", "repo__m1.nml", ["--ranks", "224", "--max-dom", "2", "--restart-from",
                                          "2005-08-27_00:30:00", "--restart-io-form", "2"])
    edit("netcdf-inputs", STREAM_NL, ["--ranks", "72", "--max-dom", "2"],
         dict(WTUNE_INPUT_FORMAT="netcdf"))
    edit("aux-streams", STREAM_NL, ["--ranks", "72", "--max-dom", "2"],
         dict(WTUNE_AUX_BP_STREAMS="auxinput4,gfdda"))
    per_record = dict(WTUNE_AUX_BP_STREAMS="auxinput4", WTUNE_BP_DIR=bp_rec,
                      WTUNE_PER_RECORD_BP="wrfbdy_d01 wrflowinp_d01 wrflowinp_d02")
    edit("per-record", "tc__PerRecordInputs__NL.nml", ["--ranks", "72", "--max-dom", "2"],
         per_record)
    edit("per-record-off-record", "tc__PerRecordInputs__NL.nml", [
        "--ranks", "72", "--max-dom", "2", "--restart-from", "2020-01-01_03:00:00"], per_record)
    edit("unknown-template", "repo__m1.nml", ["--ranks", "224", "--physics-template", "P99"])
    edit("err-no-ranks", "repo__m1.nml", [])
    edit("err-set", "repo__m1.nml", ["--ranks", "224", "--set", "max_dom"])
    edit("err-restart-from", "repo__m1.nml", ["--ranks", "224", "--restart-from", "2005-08-27"])
    edit("err-run-seconds", "repo__m1.nml", ["--ranks", "224", "--history-mode", "end_d01",
                                             "--run-seconds", "30"])
    edit("err-interval", "repo__m1.nml", ["--ranks", "224", "--history-mode", "interval"])
    add("edit-err-no-infile", ["edit-namelist", "{D}/nothing", "{D}/out"] + base + ["--ranks", "4"])
    add("edit-help", ["edit-namelist", "--help"])
    # --- staging: files the namelist names, BP inputs ----------------------------------------
    d = add("link-namelist-files", [], snap=["{D}/run"])
    nml_path, run, search = _mk_stage_run(d)
    out[-1][1][:] = ["link-namelist-files", nml_path, run] + search + ["", "/nonexistent/dir"]
    add("link-namelist-files-argc", ["link-namelist-files", "{D}/namelist.input"])
    add("link-namelist-files-no-namelist", ["link-namelist-files", "{D}/nothing", "{D}"])
    bp_env = dict(WTUNE_SRC=fake_src, FAKE_BP_LOG="{D}/convert.log",
                  FAKE_BP_REAL=os.path.join(REPO, "src", "wtune", "bp.py"),
                  WTUNE_BP_CACHE="{D}/bp-cache")

    def stage(cid, layout, args, env=None, twice=False):
        d = add("stage-" + cid, [], dict(bp_env, **(env or {})),
                snap=["{D}/cache", "{D}/bp-cache", "{D}/convert.log"])
        text, src = _mk_bp_layouts(os.path.join(d, "layouts"))[layout]
        nml_path = _write(os.path.join(d, "namelist.input"), text)
        out[-1][1][:] = ["stage-bp-inputs", "sc26p1", nml_path, src] + [
            a.replace("{D}", d) for a in args]
        if twice:                                  # the pinned run finds the cache converted
            subprocess.run([sys.executable, CONFIGS_PY] + out[-1][1], env=_env(tmp, **out[-1][2]),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    stage("convert", "netcdf", ["--cache-dir", "{D}/cache"])
    stage("convert-cached", "netcdf", ["--cache-dir", "{D}/cache"], twice=True)
    stage("default-cache", "netcdf", [])
    stage("both", "both", ["--cache-dir", "{D}/cache"])
    stage("bp-only", "bp", ["--cache-dir", "{D}/cache"])
    stage("dated", "dated", ["--cache-dir", "{D}/cache"])
    stage("check-bp", "bp", ["--check"])
    stage("check-netcdf", "netcdf", ["--check", "--cache-dir", "{D}/cache"])
    stage("check-both", "both", ["--check", "--cache-dir", "{D}/cache"])
    stage("missing-bdy", "missing", ["--cache-dir", "{D}/cache"])
    stage("convert-fails", "netcdf", ["--cache-dir", "{D}/cache"], dict(FAKE_BP_FAIL="1"))
    add("stage-no-namelist", ["stage-bp-inputs", "sc26p1", "{D}/nothing", "{D}"], bp_env)
    add("stage-help", ["stage-bp-inputs", "--help"])
    return out


def cli_snapshot(tmp):
    out = OD()
    for cid, argv, env, stdin, snap in _cli_cases(tmp):
        p = subprocess.run([sys.executable, CONFIGS_PY] + argv, env=_env(tmp, **env),
                           cwd=os.path.join(tmp, "cli", cid), input=stdin or "",
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=120)
        rec = OD([("argv", [_norm(a, tmp) for a in argv]),
                  ("env", OD((k, _norm(v, tmp)) for k, v in sorted(env.items()))),
                  ("rc", p.returncode), ("stdout", _norm(p.stdout, tmp)),
                  ("stderr", _norm(p.stderr, tmp))])
        files = OD()
        for path in snap:
            if os.path.isdir(path):
                files[_norm(path, tmp)] = _tree(path, tmp)
            elif os.path.isfile(path):
                with open(path) as f:
                    files[_norm(path, tmp)] = _norm(f.read(), tmp)
        if files:
            rec["files"] = files
        out[cid] = rec
    return out


# --------------------------------------------------------------------------- API snapshot
def _enc(o):
    """JSON-able form that keeps what a caller can observe: tuple vs list, float vs int."""
    if isinstance(o, dict):
        return OD((k if isinstance(k, str) else repr(k), _enc(v)) for k, v in o.items())
    if isinstance(o, tuple):
        return {"(tuple)": [_enc(v) for v in o]}
    if isinstance(o, list):
        return [_enc(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return {"(set)": sorted(repr(v) for v in o)}
    if isinstance(o, datetime.datetime):
        return {"(datetime)": o.isoformat()}
    if isinstance(o, float):
        return {"(float)": repr(o)}
    if o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, re.Pattern):
        return {"(regex)": [o.pattern, o.flags]}
    return {"(repr)": repr(o)}


def _call(fn, *args, **kwargs):
    """fn's encoded result, or its exception; with what it printed when it printed anything."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            result = _enc(fn(*args, **kwargs))
        except SystemExit as e:
            result = {"(exit)": e.code}
        except Exception as e:                              # noqa: BLE001 -- the snapshot pins it
            result = {"(raises)": "%s: %s" % (type(e).__name__, e)}
    if out.getvalue() or err.getvalue():
        return OD([("result", result), ("stdout", out.getvalue()), ("stderr", err.getvalue())])
    return result


@contextlib.contextmanager
def _environ(**values):
    keys = ("WTUNE_INPUT_FORMAT", "WTUNE_AUX_BP_STREAMS", "WTUNE_PER_RECORD_BP", "WTUNE_BP_DIR",
            "WTUNE_BP_CACHE", "CONFIG_FILE")
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(values)
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


DT = datetime.datetime
EDIT_VARIANTS = OD([
    ("default", {}),
    ("max_dom2", dict(max_dom=2)),
    ("max_dom3_nest_on", dict(max_dom=3, nest_history_on=True)),
    ("end_d01", dict(history_mode="end_d01", max_dom=2)),
    ("end_d01_nest_on", dict(history_mode="end_d01", max_dom=2, nest_history_on=True)),
    ("end_d01_single", dict(history_mode="end_d01")),
    ("interval", dict(history_mode="interval", history_minutes=30, max_dom=2)),
    ("interval_single", dict(history_mode="interval", history_minutes=30)),
    ("interval_no_minutes", dict(history_mode="interval")),
    ("mid_end", dict(history_mode="mid_end_d01", run_minutes=9, max_dom=2)),
    ("mid_end_nest_on", dict(history_mode="mid_end_d01", run_minutes=9, max_dom=2,
                             nest_history_on=True)),
    ("mid_end_too_short", dict(history_mode="mid_end_d01", run_minutes=1)),
    ("bad_mode", dict(history_mode="hourly")),
    ("no_adios2", dict(adios2=False)),
    ("no_ranks", dict(ranks=None)),
    ("overrides_delete", dict(
        overrides=OD([("max_dom", 1), ("restart", ".false."), ("io_form_input", 2)]),
        delete_keys=["opt_run", "BL_MYNN_TKEBUDGET", "frames_per_outfile"])),
    ("tiles", dict(numtiles_x=2, numtiles_y=6, tile_sz_x=0, tile_sz_y=0, tile_strategy=3)),
    ("tiles_one", dict(numtiles_x=0)),
    ("restart_from", dict(restart_from=DT(2019, 11, 26, 13, 30, 5), restart_io_form=2,
                          max_dom=2)),
    ("write_restart", dict(write_restart_interval=30, restart_io_form=2)),
    ("run_seconds", dict(run_seconds=45)),
    ("physics_P02", dict(physics_template="P02", max_dom=2)),
    ("physics_P04_lafe", dict(physics_template="P04", physics_case="lafe")),
    ("physics_P05_karina", dict(physics_template="P05", physics_case="karina", max_dom=2)),
    ("physics_A01", dict(physics_template="A01", max_dom=2)),
    ("physics_D01", dict(physics_template="D01", max_dom=3)),
    ("physics_unknown", dict(physics_template="P99")),
])
ADIOS2_NAMELISTS = (
    "repo__m1.nml", "syn__sst_fdda_packed.nml", STREAM_NL, "tc__PerRecordInputs__NL.nml",
    "tc__NamelistFileStaging__NML.nml")
PHYSICS_NAMELISTS = (
    "repo__m1.nml", "repo__rebuild_base.nml", "repo__rebuild_nested_suite.nml",
    "repo__rebuild_wrfbdy_times.nml", "syn__sst_fdda_packed.nml",
    "tc__ApplyPhysicsEdit__THREE_DOMAIN.nml", "tc__ApplyPhysicsEdit__SUITE.nml",
    "tc__DiagnosticsAndCadenceTemplates__NL.nml")
PHYSICS_EDITS = OD([
    ("delete", dict(op="delete", key="radt")),
    ("delete_absent", dict(op="delete", key="nosuchkey")),
    ("set_scalar", dict(op="set", key="mp_physics", value=6)),
    ("set_insert", dict(op="set", key="swint_opt", value=1)),
    ("set_insert_dynamics", dict(op="set", key="rk_ord", value=2, section="&dynamics")),
    ("set_insert_per_domain", dict(op="set", key="time_step_sound", value=3,
                                   section="&dynamics", per_domain=True)),
    ("set_by_domain_list", dict(op="set", key="radt", values_by_domain=[10, 5])),
    ("set_by_domain_map", dict(op="set", key="radt", values_by_domain={"1": 10, 2: 5})),
    ("set_by_domain_short", dict(op="set", key="radt", values_by_domain=[10])),
    ("set_by_domain_bad_index", dict(op="set", key="radt", values_by_domain={"d1": 10})),
    ("set_by_domain_outside", dict(op="set", key="radt", values_by_domain={1: 10, 2: 5, 9: 1})),
    ("mul", dict(op="mul", key="radt", factor=0.5)),
    ("mul_round_clamped", dict(op="mul", key="time_step", factor=1.37, round="int", min=20,
                               max=70, section="&domains")),
    ("mul_suite", dict(op="mul", key="mp_physics", factor=2)),
    ("mul_absent", dict(op="mul", key="nosuchkey", factor=2)),
    ("bad_op", dict(op="swap", key="radt")),
])


def _edit(configs, text, kw):
    kw = dict(kw)
    return configs.edit_namelist(text, 7, 32, 12, kw.pop("run_minutes", 8),
                                 ranks=kw.pop("ranks", 224), **kw)


def _api_edit(configs, put, nml):
    for name, text in nml.items():
        for vname, kw in EDIT_VARIANTS.items():
            put("edit_namelist|%s|%s" % (name, vname), _call(_edit, configs, text, kw))


def _api_adios2(configs, put, nml, tmp):
    bp_rec = os.path.join(tmp, "api", "bp_rec")
    for n in ("wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02", "wrffdda_d01"):
        _mk_records(os.path.join(bp_rec, n), n)
    _mk_bp(os.path.join(bp_rec, "wrflowinp_d02_2020-01-01_12:00:00.bp"))      # a flat record
    names = "wrfbdy_d01 wrflowinp_d01 wrflowinp_d02"
    envs = OD([
        ("plain", {}),
        ("netcdf", dict(WTUNE_INPUT_FORMAT="netcdf")),
        ("aux4", dict(WTUNE_AUX_BP_STREAMS="auxinput4")),
        ("aux4_gfdda_aux5", dict(WTUNE_AUX_BP_STREAMS="auxinput4,gfdda,auxinput5,sgfdda")),
        ("per_record", dict(WTUNE_AUX_BP_STREAMS="auxinput4", WTUNE_PER_RECORD_BP=names,
                            WTUNE_BP_DIR=bp_rec)),
        ("per_record_bdy", dict(WTUNE_PER_RECORD_BP="wrfbdy_d01,junk_d01", WTUNE_BP_DIR=bp_rec)),
        ("per_record_fdda", dict(WTUNE_AUX_BP_STREAMS="auxinput4,gfdda", WTUNE_BP_DIR=bp_rec,
                                 WTUNE_PER_RECORD_BP=names + " wrffdda_d01")),
        ("per_record_no_dir", dict(WTUNE_AUX_BP_STREAMS="auxinput4", WTUNE_PER_RECORD_BP=names)),
    ])
    starts = OD([("namelist_start", None), ("on_record", DT(2020, 1, 1, 6)),
                 ("off_record", DT(2020, 1, 1, 3))])
    for name in ADIOS2_NAMELISTS:
        for ename, env in envs.items():
            for sname, start in starts.items():
                if start is not None and "WTUNE_PER_RECORD_BP" not in env:
                    continue                      # the start only matters for per-record inputs
                with _environ(**env):
                    first = _call(configs._apply_adios2, nml[name], 72, 2, start=start)
                    put("apply_adios2|%s|%s|%s" % (name, ename, sname), first)
                    text = first.get("result") if isinstance(first, dict) else first
                    if ename.startswith("per_record") and isinstance(text, str):
                        # the RANKING restart pass edits the first pass's output
                        put("apply_adios2|%s|%s|%s|then_off_record" % (name, ename, sname),
                            _call(configs._apply_adios2, text, 72, 2, start=starts["off_record"]))
    with _environ(WTUNE_AUX_BP_STREAMS="auxinput4", WTUNE_PER_RECORD_BP=names,
                  WTUNE_BP_DIR=bp_rec):
        for sname, start in starts.items():
            put("edit_namelist_per_record|%s" % sname, _call(
                configs.edit_namelist, nml["tc__PerRecordInputs__NL.nml"], 8, 9, 1, 60,
                max_dom=2, ranks=72, restart_from=start))
    for name, text in nml.items():
        streams = _call(configs.adios2_input_streams, text)
        put("adios2_input_streams|" + name, streams)
        if isinstance(streams, list):
            put("adios2_stream_edits|" + name,
                [_call(configs.adios2_stream_edits, st) for st in configs.adios2_input_streams(text)])
        put("namelist_file_refs|" + name, _call(configs.namelist_file_refs, text))
        put("case_start_datetime|" + name, _call(configs.case_start_datetime, text))
        for sname, start in starts.items():
            put("per_record_streams|%s|%s" % (name, sname), _call(
                configs.per_record_streams, names + " wrffdda_d01", start, text, bp_rec,
                aux_ok={"auxinput4"}))
        for key in ("time_step", "radt", "io_form_history", "history_interval", "max_dom",
                    "iofields_filename", "nosuchkey"):
            put("find_namelist_key|%s|%s" % (name, key), _call(configs._find_namelist_key, text, key))
            put("namelist_string_values|%s|%s" % (name, key),
                _call(configs._namelist_string_values, text, key))


def _sha_text(result):
    """(text, summary) -> [sha1 of the text, summary]: the edit texts are pinned elsewhere."""
    if isinstance(result, dict) and "(tuple)" in result and isinstance(result["(tuple)"][0], str):
        rest = result["(tuple)"][1:]
        return [hashlib.sha1(result["(tuple)"][0].encode()).hexdigest()] + rest
    return result


def _api_physics(configs, put, nml):
    for name in PHYSICS_NAMELISTS:
        for tid in configs.PHYSICS_TEMPLATE_ORDER:
            template = configs.PHYSICS_TEMPLATES[tid]
            case_ids = [None] + sorted(template.get("extra_edits") or {})[:2]
            for max_dom in (1, 2):
                for cid in case_ids:
                    put("apply_physics_template|%s|%s|max_dom%d|%s" % (name, tid, max_dom, cid),
                        _sha_text(_call(configs.apply_physics_template, nml[name], template,
                                        max_dom, case_id=cid)))
            put("apply_physics_template|%s|%s|frozen" % (name, tid), _sha_text(_call(
                configs.apply_physics_template, nml[name], template, 2, case_id="karina",
                frozen_keys=["Time_Step", " radt "])))
        for ename, edit in PHYSICS_EDITS.items():
            for max_dom in (None, 1, 2):
                put("apply_physics_edit|%s|%s|max_dom%s" % (name, ename, max_dom),
                    _sha_text(_call(configs.apply_physics_edit, nml[name], edit, max_dom)))
    cases = fixture_cases()
    for cid in list(cases) + ["nosuchcase"]:
        for tid in configs.PHYSICS_TEMPLATE_ORDER + ["P99"]:
            summary = _call(configs.resolve_physics_config, cid, tid)
            put("resolve_physics_config|%s|%s" % (cid, tid), summary)
            template = configs.PHYSICS_TEMPLATES.get(tid)
            if template is None or cid not in cases:
                continue
            put("physics_template_applicability|%s|%s" % (cid, tid), _call(
                configs.physics_template_applicability, cid, template,
                frozen_keys=configs.case_frozen_keys(cases[cid])))
            put("template_frozen_key_conflicts|%s|%s" % (cid, tid), _call(
                configs.template_frozen_key_conflicts, template, ["TIME_STEP", "radt"], cid))
            ok, _reason = configs.resolve_physics_config(cid, tid)
            if ok is not None:
                put("format_physics_edits|%s|%s" % (cid, tid), _call(configs.format_physics_edits, ok))
                put("physics_ledger_columns|%s|%s" % (cid, tid),
                    _call(configs.physics_ledger_columns, cases[cid], ok))
                put("probe_verdicts_for_case|%s|%s" % (cid, tid), _call(
                    configs.probe_verdicts_for_case, cid, [e["key"] for e in ok],
                    screen_dir=os.path.join(FIX, "screen")))
        if cid in cases:
            put("case_effective_max_dom|" + cid, _call(configs._case_effective_max_dom, cases[cid]))
            put("case_namelist_path|" + cid, _call(configs._case_namelist_path, cases[cid]))
            put("case_frozen_keys|" + cid, _call(configs.case_frozen_keys, cases[cid]))
    with open(os.path.join(FIX, "screen", "screen_karina.txt")) as f:
        put("parse_probe_verdicts", _call(configs.parse_probe_verdicts, f.read()))
    for tag in ("P00_ref", "P00_twin", "P02", "P00"):
        put("physics_tag_template|" + tag, _call(configs.physics_tag_template, tag))


def _api_main(configs, put, tmp):
    """main() in-process for every (case, template): the same code the CLI snapshot spawns."""
    cases = list(fixture_cases()) + ["nosuchcase"]
    screen = os.path.join(FIX, "screen")
    for cid in cases:
        for argv in (["physics-list", cid], ["physics-list", cid, "--json"],
                     ["restart-time", cid, "45"]):
            put("main|" + " ".join(argv), _call(configs.main, argv))
    gen = _write(os.path.join(tmp, "api", "gen.json"), json.dumps({"configs": GENERATED}))
    for env in ({}, {"CONFIG_FILE": gen}):
        with _environ(**env):
            for argv in (["karina"], ["off"], ["order", "maria1km", "--reps", "3"],
                         ["config-extras", "karina"], ["--all"], []):
                put("main|%s|%s" % ("CONFIG_FILE" if env else "strict", " ".join(argv)),
                    _call(configs.main, argv))


def _api_layout(configs, put):
    real = configs.load_cases(REAL_CASES, site="f1")
    put("load_cases|f1|ids", list(real))
    scc = configs.load_cases(REAL_CASES, site="scc")
    put("load_cases|scc|overlaid", _enc(OD(
        (cid, OD((k, v) for k, v in sorted(c.items()) if real.get(cid, {}).get(k) != v))
        for cid, c in scc.items() if real.get(cid) != c)))
    for cid, case in real.items():
        put("resolve_case|" + cid, _call(configs.resolve_case, case))
        put("active_domain_extents|" + cid, _call(configs._active_domain_extents, case))
        if cid in ("karina", "sc26p1", "isc21"):
            put("available_decomps|" + cid, _call(configs.available_decomps, cid, cases=real))
        put("resolve_decomp_pair|%s|C02|C06" % cid,
            _call(configs.resolve_decomp_pair, cid, "C02", "C06", cases=real))
    for grid in ((901, 901, 45), (425, 300, 50), (100, 100, 40), ([2000, 150], [2000, 900], 50)):
        for ranks, omp in ((448, 1), (224, 2), (112, 4), (96, 3), (7, 1)):
            if isinstance(grid[0], int):
                put("ranked_shapes|%s|%d|%d" % (grid, ranks, omp),
                    _call(configs.ranked_shapes, grid[0], grid[1], grid[2], ranks, omp))
            put("legal_shapes|%s|%d" % (grid, ranks),
                _call(configs.legal_shapes, grid[0], grid[1], ranks))
    template = dict(id="X01", ranks=448, omp=1, shape="alt2", numtiles=1, dist="cyclic",
                    blocktime="0")
    for name, case in (("d01", dict(e_we=901, e_sn=901, e_vert=45)),
                       ("max_omp", dict(e_we=901, e_sn=901, e_vert=45, max_omp=0)),
                       ("tiny", dict(e_we=30, e_sn=30, e_vert=45, max_dom=2))):
        put("resolve_config|" + name, _call(configs.resolve_config, case, template))
    put("resolve_config|cores", _call(configs.resolve_config, dict(e_we=901, e_sn=901, e_vert=45),
                                      dict(template, ranks=100)))
    ids = ["C01", "C02", "C05"]
    for reps in (0, 1, 2, 3, "4", -1):
        put("build_order|%r" % (reps,), _call(configs.build_order, ids, reps))
    cfg = lambda i, **kw: dict(dict(id=i, ranks=4, omp=1, nx=2, ny=2, numtiles=1, dist="cyclic",
                                    blocktime="0"), **kw)
    put("dedupe", _call(configs.dedupe, [cfg("a"), cfg("b"), cfg("c", numtiles=2), cfg("d")]))
    put("describe_decomp", _call(configs.describe_decomp, cfg("a")))
    put("decomp_identity", _call(configs._decomp_identity, cfg("a", tile_sz_x="", numtiles_x=2)))
    put("config_extras_rows", _call(configs.config_extras_rows, GENERATED, "karina"))
    for value in (None, False, True, 0, 1, 2, "0", "1", "2", "", "off", "end", "mid_end", "false",
                  "yes"):
        put("window_frame_level|%r" % (value,), _call(configs.window_frame_level, value))
    orders = (["ANCHOR", "G1", "ANCHOREND"], ["ANCHOR", "G1", "G2", "ANCHOREND"], ["G1"],
              ["ANCHOR"], [])
    for order in orders:
        for fused in (False, True):
            for anchor_end in (None, True, False):
                for frame in (False, True):
                    put("resolve_point_plan|%s|%s|%s|%s" % (",".join(order), fused, anchor_end, frame),
                        _call(configs.resolve_point_plan, order, fused, anchor_end, frame))


def _api_reports(configs, put):
    for name, values in (("empty", []), ("one", [30.0]), ("plain", [30.0, 1.5, 1.75, 2.0]),
                         ("radiation", [30.0, 1.5, 1.75, 2.0, 12.25, 1.5]), ("two", [9.0, 2.0])):
        put("step_stats|" + name, _call(configs.step_stats, values))
    for spin, dt in ((0, 60), (180, 60), (179.5, 60), (3600, 60), (60, "30")):
        put("post_spinup_steps|%s|%s" % (spin, dt), _call(configs.post_spinup_steps, RSL, spin, dt))
    put("post_spinup_steps|empty", _call(configs.post_spinup_steps, "", 0, 60))
    for fs in ("home1", "work2", "nosuchfs"):
        put("quota_free_gib|" + fs, _call(configs.quota_free_gib, MMLSQUOTA, filesystem=fs))
    put("quota_free_gib|zero_quota", _call(configs.quota_free_gib, "home1 USR 5 0 0\n"))
    for name, payload in (("full", FIDELITY), ("empty", {}), ("no_fields", {"label": "X Y"})):
        put("fidelity_notes_tokens|" + name, _call(configs.fidelity_notes_tokens, payload))
    for value in (None, "", 0, "x", 1.5):
        put("describe_field|%r" % (value,), _call(configs.describe_field, value))
    put("shq", _call(configs._shq, "it's a 'path'"))
    put("bp_cache_dir|root", _call(configs.bp_cache_dir, "karina", "/nonexistent/prep", "/c"))
    with _environ(WTUNE_BP_CACHE="/env/cache"):
        put("bp_cache_dir|env", _call(configs.bp_cache_dir, "karina", "/nonexistent/prep"))
    put("bp_cache_dir|home", _call(configs.bp_cache_dir, "karina", "/nonexistent/prep"))


def _fake_bp(real, log, current=True, reflink=True):
    def convert_inputs(case_dir, cache_dir, names=None, engine="BP4"):
        os.makedirs(cache_dir, exist_ok=True)
        out = {}
        for n in names:
            dst = os.path.join(cache_dir, n)
            if not os.path.isdir(dst):
                log.append("convert %s %s" % (n, engine))
                if real.PER_RECORD_NAME_RE.match(n):
                    _mk_records(dst, n)
                else:
                    _mk_bp(dst)
            out[n] = dst
        return out

    def fresh_input_copy(sources, dest, stats=None):
        log.append("fresh copy %s -> %s" % (sorted(sources.items()), dest))
        stats.update(reflink=reflink, bytes=0 if reflink else 10)
        return {n: os.path.join(dest, n) for n in sources}

    def cached_inputs(case_dir, cache_dir, names):
        log.append("cached_inputs %s %s" % (case_dir, names))
        return {n: os.path.join(cache_dir, n) for n in names} if current else None
    return types.SimpleNamespace(
        PER_RECORD_NAME_RE=real.PER_RECORD_NAME_RE, is_per_record_dir=real.is_per_record_dir,
        bp_input_links=real.bp_input_links, require_adios2_python=lambda: "/fake/python",
        convert_inputs=convert_inputs, cached_inputs=cached_inputs,
        fresh_input_copy=fresh_input_copy)


def _api_bp(configs, put, tmp):
    real = configs._import_wtune_bp()
    root = os.path.join(tmp, "api", "bp")
    for layout, (text, src) in _mk_bp_layouts(os.path.join(root, "layouts")).items():
        cache = os.path.join(root, "cache", layout)
        put("resolve_bp_inputs|%s|plan" % layout,
            _call(configs.resolve_bp_inputs, text, src, cache, convert=False))
        put("resolve_bp_inputs|%s|plan_fake_bp" % layout, _call(
            configs.resolve_bp_inputs, text, src, cache, convert=False, bp=_fake_bp(real, [])))
        log = []
        put("resolve_bp_inputs|%s|convert" % layout, _call(
            configs.resolve_bp_inputs, text, src, cache, bp=_fake_bp(real, log), log=log.append))
        put("resolve_bp_inputs|%s|convert|log" % layout, list(log))
        put("resolve_bp_inputs|%s|convert|cache" % layout,
            _tree(cache, tmp) if os.path.isdir(cache) else None)
        del log[:]
        put("resolve_bp_inputs|%s|convert_again" % layout, _call(
            configs.resolve_bp_inputs, text, src, cache, bp=_fake_bp(real, log), log=log.append))
        put("resolve_bp_inputs|%s|convert_again|log" % layout, list(log))
    # a relative-path layout: the names resolve against src_dir
    put("resolve_bp_inputs|absent_dir", _call(
        configs.resolve_bp_inputs, _read(STREAM_NL), os.path.join(root, "nothing"),
        os.path.join(root, "cache", "nothing"), convert=False))
    prep = os.path.join(root, "prep")
    _mk_prep(prep, names=("wrfinput_d01", "wrfbdy_d01", "wrflowinp_d01", "wrflowinp_d02"),
             namelist="tc__PerRecordInputs__NL.nml")
    bp_dir = os.path.join(root, "bp_dir")
    for n in ("wrfinput_d01.bp", "wrfinput_d02.bp"):
        _mk_bp(os.path.join(bp_dir, n))
    _mk_records(os.path.join(bp_dir, "wrfbdy_d01"), "wrfbdy_d01")
    cases = _write(os.path.join(root, "cases.json"), json.dumps({"cases": OD([
        ("pt", {"prep_dir": prep, "bp_dir": ""}),
        ("with_bp_dir", {"prep_dir": prep, "bp_dir": bp_dir}),
        ("no_namelist", {"prep_dir": os.path.join(root, "nothing")})])}))
    with _environ(WTUNE_BP_CACHE=os.path.join(root, "bp-cache")):
        for cid in ("pt", "with_bp_dir", "no_namelist", "nosuchcase"):
            for state, current in (("current", True), ("stale", False)):
                log = []
                bp = _fake_bp(real, log, current=current)
                tag = "%s|%s" % (cid, state)
                put("case_bp_plan|" + tag, _call(configs._case_bp_plan, cid, cases, bp=bp))
                put("bp_cache_status|" + tag, _call(configs.bp_cache_status, cid, cases, bp=bp))
                put("per_record_sources|" + tag, _call(configs.per_record_sources, cid, cases, bp=bp))
                del log[:]
                for copies, reflink in ((3, True), (3, False), (0, True)):
                    dest = os.path.join(root, "fresh", cid, state)
                    put("fresh_probe_inputs|%s|%d|%s" % (tag, copies, reflink), _call(
                        configs.fresh_probe_inputs, cid, dest, copies, cases_json=cases,
                        bp=_fake_bp(real, log, current=current, reflink=reflink)))
                put("fresh_probe_inputs|%s|log" % tag, list(log))
        with _environ(WTUNE_INPUT_FORMAT="netcdf"):
            put("bp_cache_status|netcdf", _call(configs.bp_cache_status, "pt", cases))
    put("FRESH_COPY_PREFIX", configs.FRESH_COPY_PREFIX)
    link = os.path.join(root, "links")
    os.makedirs(link)
    target = os.path.join(root, "prep")
    for name, dst, tgt in (("new", "a", target), ("same", "a", target), ("stale", "a", root),
                           ("self", "prep_self", os.path.join(link, "prep_self"))):
        put("link_idempotent|" + name, _call(configs._link_idempotent, os.path.join(link, dst), tgt))
    _write(os.path.join(link, "real_file"), "x")
    put("link_idempotent|real_file",
        _call(configs._link_idempotent, os.path.join(link, "real_file"), target))
    put("link_idempotent|tree", _tree(link, tmp))


def _api_stage(configs, put, nml, tmp):
    _nl, run, search = _mk_stage_run(os.path.join(tmp, "api", "stage"))
    text = nml["tc__NamelistFileStaging__NML.nml"]
    result = configs.stage_namelist_files(text, run, search + ["", "/nonexistent/dir"])
    put("stage_namelist_files", _enc(result))
    put("format_namelist_stage", _call(configs.format_namelist_stage, result, search + [""]))
    put("format_namelist_stage|no_dirs", _call(configs.format_namelist_stage, result, ["", None]))
    put("stage_namelist_files|again", _call(configs.stage_namelist_files, text, run, search))
    put("merge_iofields", _call(configs.merge_iofields, text, run))
    put("merge_iofields|again", _call(configs.merge_iofields, text, run))
    put("stage_namelist_files|tree", _tree(run, tmp))
    escape = ('&time_control\n iofields_filename = "../up.txt", "/nonexistent/abs.txt",\n/\n'
              "&domains\n max_dom = 2,\n/\n")
    put("stage_namelist_files|outside_run_dir", _call(configs.stage_namelist_files, escape, run, search))


def _surface(configs):
    """{name: what an importer sees}; taken before anything is called."""
    out = OD()
    for name in sorted(vars(configs)):
        value = getattr(configs, name)
        if name.startswith("__") or inspect.ismodule(value):
            continue                    # stdlib aliases are not an interface
        if inspect.isfunction(value):
            out[name] = ["function"] + [
                [p.name, p.kind.name, "<required>" if p.default is p.empty else _enc(p.default)]
                for p in inspect.signature(value).parameters.values()]
        elif name == "PHYSICS_TEMPLATES":
            blob = json.dumps(value, sort_keys=True).encode()
            out[name] = ["sha1", hashlib.sha1(blob).hexdigest(), sorted(value)]
        else:
            out[name] = [type(value).__name__, _enc(value)]
    return out


def api_snapshot(tmp):
    """Run inside a fresh interpreter (see __main__): CASES_JSON is the fixture case table."""
    sys.path.insert(0, CAMPAIGN)
    import configs
    surface = _surface(configs)
    api = OD()
    put = api.__setitem__
    nml = _namelists()
    _api_edit(configs, put, nml)
    _api_adios2(configs, put, nml, tmp)
    _api_physics(configs, put, nml)
    _api_main(configs, put, tmp)
    _api_layout(configs, put)
    _api_reports(configs, put)
    _api_bp(configs, put, tmp)
    _api_stage(configs, put, nml, tmp)
    return OD([("surface", surface), ("api", api)])


def _dump(obj, tmp):
    return _paths(json.dumps(obj, indent=1), tmp) + "\n"


if __name__ == "__main__":             # the API snapshot's interpreter: python <this file> TMP
    sys.stdout.write(_dump(api_snapshot(sys.argv[1]), sys.argv[1]))
    sys.exit(0)


# --------------------------------------------------------------------------- the tests
import pytest  # noqa: E402


def _pairs(text):
    """Parsed JSON with every object as ['(dict)', [key, value], ...]: key order is compared."""
    return json.loads(text, object_pairs_hook=lambda p: ["(dict)"] + [[k, v] for k, v in p])


def _items(pairs):
    return OD((k, v) for k, v in pairs[1:])


def _golden(name, text):
    path = os.path.join(FIX, name)
    if REGEN:
        with open(path, "w") as f:
            f.write(text)
    with open(path) as f:
        return f.read()


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    return os.path.realpath(str(tmp_path_factory.mktemp("configs_golden")))


@pytest.fixture(scope="module")
def api(sandbox):
    env = _env(sandbox, CASES_JSON=_write(os.path.join(sandbox, "api", "cases.json"),
                                          json.dumps({"cases": fixture_cases()}, indent=1)))
    p = subprocess.run([sys.executable, os.path.abspath(__file__), sandbox], env=env, cwd=sandbox,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr
    doc = json.loads(p.stdout, object_pairs_hook=OD)
    out = {}
    for part in ("surface", "api"):
        text = json.dumps(doc[part], indent=1) + "\n"
        out["got_" + part] = _items(_pairs(text))
        out["want_" + part] = _items(_pairs(_golden(part + ".json", text)))
    return out


@pytest.fixture(scope="module")
def cli(sandbox):
    text = _dump(cli_snapshot(sandbox), sandbox)
    return _items(_pairs(text)), _items(_pairs(_golden("cli.json", text)))


def _differences(got, want):
    lines = []
    for key in want:
        if key not in got:
            lines.append("missing: %s" % key)
        elif got[key] != want[key]:
            lines.append("differs: %s\n  want %r\n  got  %r" % (key, want[key], got[key]))
    return lines


def test_cli_matches_golden(cli):
    got, want = cli
    assert list(got) == list(want)
    diff = _differences(got, want)
    assert not diff, "%d CLI case(s) differ:\n%s" % (len(diff), "\n".join(diff[:5]))


def test_cli_covers_every_subcommand(cli):
    for sub in ("edit-namelist", "step-stats", "fused-steps", "point-plan", "quota-free-gib",
                "link-namelist-files", "stage-bp-inputs", "physics-list", "restart-time",
                "config-extras", "--all", "order"):
        assert any(sub == _items(rec)["argv"][:1][0] for rec in cli[1].values()
                   if _items(rec)["argv"]), sub


SECTIONS = ["edit_namelist", "apply_adios2", "edit_namelist_per_record", "adios2_input_streams",
            "adios2_stream_edits", "namelist_file_refs", "case_start_datetime",
            "per_record_streams", "find_namelist_key", "namelist_string_values",
            "apply_physics_template", "apply_physics_edit", "resolve_physics_config", "main",
            "resolve_case", "resolve_bp_inputs", "fresh_probe_inputs", "stage_namelist_files"]


@pytest.mark.parametrize("section", SECTIONS + ["(everything else)"])
def test_api_matches_golden(api, section):
    def pick(d):
        if section in SECTIONS:
            return OD((k, v) for k, v in d.items() if k.split("|")[0] == section)
        return OD((k, v) for k, v in d.items() if k.split("|")[0] not in SECTIONS)
    got, want = pick(api["got_api"]), pick(api["want_api"])
    assert want, "no golden entries for %s" % section
    assert list(got) == list(want)
    diff = _differences(got, want)
    assert not diff, "%d of %d entries differ:\n%s" % (len(diff), len(want), "\n".join(diff[:5]))


def test_every_name_is_still_exported_unchanged(api):
    diff = _differences(api["got_surface"], api["want_surface"])
    assert not diff, "\n".join(diff)
