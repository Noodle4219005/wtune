#!/usr/bin/env python3
"""wtune_calibrate.py -- produce a machine calibration file for wtune_predict.py by running
the user's own WRF case at several rank counts on a new cluster and fitting the calibration
constants wtune_predict.py's offline wall-time model consumes, instead of hand-editing JSON
the way machine/calibration/scc.json was built.

Read first: wtune_predict.py's module docstring ('Calibration' section) and its
load_calibration/predict_init/predict_compute/build_prediction for exactly how each key below
is consumed; machine/calibration/scc.json for the schema and its *_source/_note provenance
convention (every derived number cites the raw measured fact(s) it came from, or is labelled
UNCALIBRATED GUESS/ASSUMED/TRANSFERRED); campaign/sites/scc.sh for site_env/site_launch/
site_hygiene; predictor/decomp_space.py for legal_shapes_exact (WRF's own real, floor-based
startup legality gate); campaign/configs.py's edit-namelist CLI for the namelist edits this
tool reuses instead of re-implementing namelist surgery.

Two subcommands:

  wtune_calibrate.py plan --site SITE --case-dir CASE --out DIR
      [--ranks-list 1,36,72,144,288] [--cores-per-node 72] [--steps 20] [--min-patch 10]
      [--time-limit 00:15:00]

    Writes one sbatch script per requested rank count into DIR (plan_<ranks>.sbatch), plus
    MANIFEST.json describing what was planned/skipped, plus machine_profile.sbatch (reuses
    machine/run_machine_profile.sh). Each per-rank job runs the user's own case (namelist
    copied from --case-dir, wrfinput_d*/wrfbdy_d* symlinked in) for exactly --steps steps
    (restart off, history off) EXCEPT the single largest successfully-decomposed rank count,
    whose job also writes exactly one d01 history frame at the end (--history-mode end_d01) so
    a write rate can be measured. Decomposition is WRF's own legal, near-square (nx, ny) for
    that rank count (predictor/decomp_space.py legal_shapes_exact); a rank count with no legal
    decomposition for this case's domain geometry is SKIPPED with a reason recorded in
    MANIFEST.json and printed as a warning -- not silently dropped, not a crash (e.g. a small
    idealized/regional case's e_we/e_sn can cap legal ranks well below a template rank list;
    WRF's real gate is floor(e_we/nproc_x) >= min_patch and floor(e_sn/nproc_y) >= min_patch on
    EVERY active domain, share/module_check_a_mundo.F:3014-3029).

    plan never calls sbatch itself -- submission (respecting whatever concurrent-job limit the
    site imposes) is a separate, explicit step the caller controls.

  wtune_calibrate.py fit --runs DIR --out machine/calibration/<name>.json
      [--base machine/calibration/scc.json] [--cores-per-node 72] [--host HOST]

    Reads back the run directories plan's sbatch scripts create under DIR (run_<ranks>_<jobid>/,
    each with wall.txt + rsl.error.0000/rsl.out.0000, matched against MANIFEST.json when
    present), fits startup_s/startup_s_base/startup_s_per_rank (same min(cap, base +
    per_rank*ranks) model wtune_predict.py already implements), measures a history-write rate
    from the one written frame when present, and attempts to separate a read rate from the
    startup window (kept from --base, clearly marked TRANSFERRED, when it is not separable).
    Every emitted field carries provenance: freshly measured fields get a MEASURED ... source
    string naming host/date/runs; everything else is carried over from --base and listed in
    calibration_provenance.transferred_fields instead of being silently presented as measured.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import re
import socket
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # HERE is src/; machine/ and campaign/ stay at the true repo root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "predictor"))

import wtune_predict as wp          # noqa: E402  (path insert above)
import decomp_space as ds           # noqa: E402  (predictor/decomp_space.py)

DEFAULT_CALIBRATION = REPO_ROOT / "machine" / "calibration" / "scc.json"
DEFAULT_RANKS_LIST = "1,36,72,144,288"
DEFAULT_CORES_PER_NODE = 72
DEFAULT_STEPS = 20
DEFAULT_MIN_PATCH = 10
DEFAULT_TIME_LIMIT = "00:15:00"

# WRF rsl.error.0000/rsl.out.0000 "Timing for main: time <t> on domain <d>: <s> elapsed
# seconds" lines are what wp.parse_rsl_d01_step_seconds already parses (d01 only). The
# write/read regexes below are this module's own, for lines that line does not cover.
_RSL_WRITE_RE = re.compile(
    r"Timing for Writing\s+\S+\s+for domain\s+(\d+):\s+([\d.]+)\s+elapsed seconds")
# Candidate "read the input" timing lines WRF sometimes prints; none of these is guaranteed to
# exist in a given build/version, which is exactly why read-rate fitting degrades to
# "not separable" (see fit_read_rate) rather than assuming one of these patterns is present.
_RSL_READ_RE = re.compile(
    r"Timing for (?:input|processing lateral boundary)\S*\s+for domain\s+(\d+):\s+([\d.]+)\s+"
    r"elapsed seconds")


class CalibrateError(Exception):
    pass


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def pick_near_square(shapes):
    """Of the legal (nx, ny) shapes for one rank count, pick the one closest to square:
    minimum |nx-ny|, tie-broken by the smaller max(nx,ny) (fewer ranks along the more
    constrained dimension), then by nx ascending for full determinism."""
    return min(shapes, key=lambda xy: (abs(xy[0] - xy[1]), max(xy), xy[0]))


def build_decomp(nl, max_dom, ranks, min_patch):
    we_list, sn_list = [], []
    for d in range(max_dom):
        e_we, e_sn, _e_vert = wp.get_domain_geom(nl, d)
        we_list.append(int(e_we))
        sn_list.append(int(e_sn))
    shapes = ds.legal_shapes_exact(we_list, sn_list, ranks, min_patch=min_patch)
    if not shapes:
        return None
    return pick_near_square(shapes)


PLAN_SBATCH_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name=wtcal-{ranks}
#SBATCH --partition=compute
#SBATCH --exclude=head
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tpn_header}
#SBATCH --time={time_limit}
#SBATCH --output={out_dir}/wtcal_{ranks}_%j.out
#
# Generated by wtune_calibrate.py plan (site={site} case_dir={case_dir} steps={steps}
# run_minutes={run_minutes} nx={nx} ny={ny} history_mode={history_mode}).
# Runs the case for exactly {steps} d01 steps (restart off, {history_mode_note}), then writes
# wall.txt (WALL_S/RC/OK) and leaves rsl.error.0000/rsl.out.0000 in RUNDIR for
# `wtune_calibrate.py fit --runs {out_dir}` to parse. Does not submit itself -- run
# `sbatch {sbatch_path}` (respecting your site's concurrent-job limit) when ready.
set -uo pipefail

SITE={site}
CAMPAIGN={campaign_dir}
export SITE CAMPAIGN
source "$CAMPAIGN/sites/$SITE.sh"
site_env

RUNDIR="{out_dir}/run_{ranks}_${{SLURM_JOB_ID:-manual}}"
mkdir -p "$RUNDIR"
cd "$RUNDIR" || exit 1

# WRF run-time support files (LOOKUP tables etc.), same link pattern as
# campaign/strict_campaign.sbatch -- never overwrite an existing namelist/output file.
for f in "$W"/run/*; do
  b=$(basename "$f")
  case "$b" in
    namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;
    *) [ -e "$b" ] || ln -sf "$f" "$b" ;;
  esac
done

cp {case_dir}/namelist.input ./namelist.input.orig
for wf in {case_dir}/wrfinput_d* {case_dir}/wrfbdy_d*; do
  [ -e "$wf" ] && ln -sf "$wf" "$(basename "$wf")"
done

python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.orig ./namelist.input \\
  --nproc-x {nx} --nproc-y {ny} --numtiles 1 --run-minutes {run_minutes} --max-dom {max_dom} \\
  --history-mode {history_mode} --set restart=.false. --no-adios2
rm -f ./namelist.input.orig
ln -sf "$W/main/wrf.exe" ./wrf.exe

export OMP_NUM_THREADS=1 OMP_STACKSIZE=1G OMP_PLACES=cores OMP_PROC_BIND=close

# preflight reap: this allocation's own nodes only, before any launch (see
# campaign/strict_campaign.sbatch's identical preflight comment -- ProctrackType=pgid on scc
# leaves a TIMEOUT'd mpirun's ranks invisible to Slurm's own cleanup).
if command -v site_hygiene >/dev/null 2>&1; then
  site_hygiene reap >/dev/null 2>&1 || true
  ALLOC_NODES="${{SLURM_JOB_NUM_NODES:-{nodes}}}"
  [ -n "${{SITE_HYGIENE_CMD:-}}" ] && timeout 90 "${{SITE_HYGIENE_CMD[@]}}" >/dev/null 2>&1 || true
fi

T0=$(date +%s.%N)
site_launch {ranks} {tpn_launch} 1 block
timeout 840 "${{SITE_LAUNCH_CMD[@]}}" > wrf.out 2>&1 &
WRF_PID=$!
wait "$WRF_PID"
RC=$?
T1=$(date +%s.%N)

OK=0
if [ -f rsl.error.0000 ] && grep -qa "SUCCESS COMPLETE WRF" rsl.error.0000; then
  OK=1
fi

{{
  echo "WALL_S $(python3 -c "print('%.3f' % (${{T1}} - ${{T0}}))")"
  echo "RC $RC"
  echo "OK $OK"
  echo "RANKS {ranks}"
  echo "NODES {nodes}"
  echo "NX {nx}"
  echo "NY {ny}"
}} > wall.txt

# post-run reap: this allocation's own nodes only, same as strict_campaign.sbatch.
if [ -n "${{SITE_HYGIENE_CMD:-}}" ]; then
  timeout 60 "${{SITE_HYGIENE_CMD[@]}}" >/dev/null 2>&1 || true
fi

exit 0
"""

MACHINE_PROFILE_SBATCH_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --job-name=wtcal-mprof
#SBATCH --partition=compute
#SBATCH --exclude=head
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output={out_dir}/wtcal_mprof_%j.out
#
# Generated by wtune_calibrate.py plan. Reuses machine/run_machine_profile.sh as-is (that
# script is not owned by wtune_calibrate.py and is never edited by it) to produce a
# machine_profile_<host>_<date>.json topology/bandwidth snapshot alongside the rank-count
# runs. nvhpc-only per this cluster's build rule: CC/FC point at nvhpc's own compiler names,
# not gcc/gfortran (run_machine_profile.sh's own defaults). Runs entirely as the submitting
# user -- no root is required or assumed; if a site's toolchain ever did need root this job
# would fail loudly (compiler/module errors in the .out file), not silently no-op.
set -uo pipefail

SITE={site}
CAMPAIGN={campaign_dir}
export SITE CAMPAIGN
source "$CAMPAIGN/sites/$SITE.sh"
site_env

export MACHINE_DIR="{machine_dir}"
export CC=nvc FC=nvfortran MPICC=mpicc
export OPENMP_FLAG=-mp MODDIR_FLAG=-module WARN_FLAGS=
export MACHINE_PROFILE_OUT_DIR="{out_dir}"

bash "$MACHINE_DIR/run_machine_profile.sh"
"""


def _fmt_shape_warning(ranks, we_list, sn_list, min_patch):
    max_nx = min(w // min_patch for w in we_list)
    max_ny = min(s // min_patch for s in sn_list)
    return ("no legal (nx,ny) with nx*ny=%d passes WRF's floor(e/nproc)>=%d gate on every "
            "active domain (e_we=%s e_sn=%s); legal ranks for this case top out around %d "
            "(e.g. %dx%d)" % (ranks, min_patch, we_list, sn_list, max_nx * max_ny, max_nx, max_ny))


def plan_cmd(args):
    case_dir = Path(args.case_dir).resolve()
    namelist_path = case_dir / "namelist.input"
    if not namelist_path.exists():
        raise CalibrateError("no namelist.input in --case-dir %s" % case_dir)
    nl = wp.parse_namelist(str(namelist_path))
    max_dom = wp.get_max_dom(nl)
    dt = wp.get_timestep_seconds(nl)
    we_list, sn_list = [], []
    for d in range(max_dom):
        e_we, e_sn, _e_vert = wp.get_domain_geom(nl, d)
        we_list.append(int(e_we))
        sn_list.append(int(e_sn))

    steps = args.steps
    run_minutes = max(1, int(round(steps * dt / 60.0)))

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ranks_requested = sorted({int(r) for r in args.ranks_list.split(",") if r.strip()})
    planned = []
    skipped = []
    for ranks in ranks_requested:
        shape = build_decomp(nl, max_dom, ranks, args.min_patch)
        if shape is None:
            reason = _fmt_shape_warning(ranks, we_list, sn_list, args.min_patch)
            print("WARNING: skipping ranks=%d: %s" % (ranks, reason), file=sys.stderr)
            skipped.append({"ranks": ranks, "reason": reason})
            continue
        nx, ny = shape
        planned.append({"ranks": ranks, "nx": nx, "ny": ny})

    if not planned:
        raise CalibrateError(
            "no requested rank count has a legal decomposition for this case's domain "
            "geometry (e_we=%s e_sn=%s, --min-patch=%d) -- pass a smaller --ranks-list" %
            (we_list, sn_list, args.min_patch))

    largest_ranks = max(p["ranks"] for p in planned)

    campaign_dir = REPO_ROOT / "campaign"
    machine_dir = REPO_ROOT / "machine"
    runs = []
    for p in planned:
        ranks, nx, ny = p["ranks"], p["nx"], p["ny"]
        nodes = int(math.ceil(ranks / args.cores_per_node))
        tpn_launch = ranks // nodes  # site_launch itself rounds this up when it under-covers
        tpn_header = int(math.ceil(ranks / nodes))
        history_mode = "end_d01" if ranks == largest_ranks else "off"
        history_mode_note = ("writes ONE d01 history frame at the end" if history_mode == "end_d01"
                              else "history off")
        sbatch_path = out_dir / ("plan_%d.sbatch" % ranks)
        text = PLAN_SBATCH_TEMPLATE.format(
            ranks=ranks, nodes=nodes, tpn_header=tpn_header, tpn_launch=tpn_launch,
            time_limit=args.time_limit, out_dir=out_dir, site=args.site,
            case_dir=case_dir, steps=steps, run_minutes=run_minutes, nx=nx, ny=ny,
            max_dom=max_dom, history_mode=history_mode, history_mode_note=history_mode_note,
            campaign_dir=campaign_dir, sbatch_path=sbatch_path)
        sbatch_path.write_text(text)
        runs.append({"ranks": ranks, "nodes": nodes, "nx": nx, "ny": ny,
                     "tpn_launch": tpn_launch, "history_mode": history_mode,
                     "sbatch_path": str(sbatch_path)})
        print("wrote %s (ranks=%d nodes=%d decomp=%dx%d history_mode=%s)" %
              (sbatch_path, ranks, nodes, nx, ny, history_mode))

    mprof_path = out_dir / "machine_profile.sbatch"
    mprof_path.write_text(MACHINE_PROFILE_SBATCH_TEMPLATE.format(
        out_dir=out_dir, site=args.site, campaign_dir=campaign_dir, machine_dir=machine_dir))
    print("wrote %s (machine/run_machine_profile.sh topology/bandwidth snapshot)" % mprof_path)

    manifest = {
        "site": args.site, "case_dir": str(case_dir), "out_dir": str(out_dir),
        "steps": steps, "run_minutes": run_minutes, "dt_s": dt, "max_dom": max_dom,
        "e_we": we_list, "e_sn": sn_list, "cores_per_node": args.cores_per_node,
        "min_patch": args.min_patch, "largest_ranks": largest_ranks,
        "runs": runs, "skipped": skipped,
        "machine_profile_sbatch": str(mprof_path),
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("wrote %s" % manifest_path)
    if skipped:
        print("skipped %d/%d requested rank count(s) -- see MANIFEST.json 'skipped'" %
              (len(skipped), len(ranks_requested)), file=sys.stderr)
    print("\nNothing was submitted. Submit each plan_<ranks>.sbatch (and machine_profile.sbatch) "
          "yourself, respecting your site's concurrent-job limit, then run:\n"
          "  wtune_calibrate.py fit --runs %s --out <calibration.json>" % out_dir)
    return 0


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------

def _read_wall_txt(path):
    vals = {}
    for line in path.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            vals[parts[0]] = parts[1].strip()
    return vals


def _find_rsl(run_dir):
    for name in ("rsl.error.0000", "rsl.out.0000"):
        p = run_dir / name
        if p.exists():
            return p
    return None


def _dir_bytes(run_dir, patterns):
    total = 0
    found = []
    for pat in patterns:
        for p in sorted(run_dir.glob(pat)):
            try:
                total += p.stat().st_size
            except OSError:
                continue
            found.append(p.name)
    return total, found


def collect_runs(runs_dir, manifest):
    """Returns (used, excluded): used is [{ranks, startup_measured, wall_s, write_s,
    write_bytes, read_s, read_bytes, dir}], excluded is [{ranks, dir, reason}]."""
    run_dirs = sorted(runs_dir.glob("run_*_*"))
    if not run_dirs:
        raise CalibrateError("no run_<ranks>_<jobid> directories found under %s "
                              "(nothing to fit -- did the plan_*.sbatch jobs run yet?)" % runs_dir)

    manifest_by_ranks = {}
    if manifest:
        for r in manifest.get("runs", []):
            manifest_by_ranks[r["ranks"]] = r

    used, excluded = [], []
    for run_dir in run_dirs:
        m = re.match(r"run_(\d+)_", run_dir.name)
        if not m:
            continue
        ranks = int(m.group(1))
        wall_path = run_dir / "wall.txt"
        if not wall_path.exists():
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": "no wall.txt (job did not finish or was never run)"})
            continue
        wall = _read_wall_txt(wall_path)
        try:
            wall_s = float(wall["WALL_S"])
            rc = int(wall["RC"])
            ok = int(wall["OK"])
        except (KeyError, ValueError) as exc:
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": "wall.txt malformed (%s)" % exc})
            continue
        if rc != 0 or ok != 1:
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": "run did not complete (RC=%d OK=%d)" % (rc, ok)})
            continue
        rsl_path = _find_rsl(run_dir)
        if rsl_path is None:
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": "no rsl.error.0000/rsl.out.0000"})
            continue
        main_steps = wp.parse_rsl_d01_step_seconds(str(rsl_path))
        if not main_steps:
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": "rsl log has no d01 'Timing for main' lines"})
            continue
        sum_main = sum(s for _dt, s in main_steps)
        startup_measured = wall_s - sum_main
        if startup_measured < 0:
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": ("startup_measured < 0 (wall=%.3f sum_main=%.3f) -- "
                                         "clock/parsing inconsistency" % (wall_s, sum_main))})
            continue

        text = rsl_path.read_text()
        write_s, write_bytes = None, None
        writes = [float(mm.group(2)) for mm in _RSL_WRITE_RE.finditer(text)
                  if int(mm.group(1)) == 1]
        if writes:
            write_s = sum(writes)
            write_bytes, _names = _dir_bytes(run_dir, ["wrfout_d01_*"])
            if write_bytes == 0:
                write_bytes = None

        read_s, read_bytes = None, None
        reads = [float(mm.group(2)) for mm in _RSL_READ_RE.finditer(text)
                 if int(mm.group(1)) == 1]
        if reads:
            read_s = sum(reads)
            read_bytes, _names = _dir_bytes(run_dir, ["wrfinput_d*", "wrfbdy_d*"])
            if read_bytes == 0:
                read_bytes = None

        used.append({
            "ranks": ranks, "wall_s": wall_s, "sum_main_s": sum_main,
            "startup_measured": startup_measured, "write_s": write_s, "write_bytes": write_bytes,
            "read_s": read_s, "read_bytes": read_bytes, "dir": str(run_dir),
            "manifest": manifest_by_ranks.get(ranks),
        })
    if not used:
        raise CalibrateError("every run under %s was excluded -- see reasons above; nothing to fit" %
                              runs_dir)
    return used, excluded


def fit_startup(used, cores_per_node):
    below = sorted((u["ranks"], u["startup_measured"]) for u in used if u["ranks"] < cores_per_node)
    at_or_above = [u["startup_measured"] for u in used if u["ranks"] >= cores_per_node]

    cap = statistics.median(at_or_above) if at_or_above else None

    base = per_rank = None
    if len(below) >= 2:
        n = len(below)
        sum_r = sum(r for r, _s in below)
        sum_s = sum(s for _r, s in below)
        sum_rr = sum(r * r for r, _s in below)
        sum_rs = sum(r * s for r, s in below)
        denom = n * sum_rr - sum_r * sum_r
        if denom != 0:
            per_rank = (n * sum_rs - sum_r * sum_s) / denom
            base = (sum_s - per_rank * sum_r) / n
        else:
            per_rank = 0.0
            base = sum_s / n
    elif len(below) == 1:
        r0, s0 = below[0]
        per_rank = s0 / r0 if r0 else 0.0
        base = 0.0
    base_clipped = max(0.0, base) if base is not None else None

    return {"cap": cap, "cap_n": len(at_or_above), "base": base_clipped,
            "base_unclipped": base, "per_rank": per_rank,
            "below_points": below, "at_or_above_points": at_or_above}


def fit_history_write_rate(used, base_calibration, geom):
    for u in used:
        if u["write_s"]:
            bytes_ = u["write_bytes"]
            source = "measured wrfout_d01 file size"
            if bytes_ is None:
                bpcl = base_calibration.get("derived", {}).get("bytes_per_cell_level")
                if bpcl is not None and geom is not None:
                    e_we, e_sn, e_vert = geom
                    bytes_ = bpcl * e_we * e_sn * e_vert
                    source = ("ESTIMATED via base calibration's bytes_per_cell_level "
                               "(no wrfout_d01_* file found on disk)")
            if bytes_ is None:
                continue
            rate_bps = bytes_ / u["write_s"]
            return {"ranks": u["ranks"], "bytes": bytes_, "seconds": u["write_s"],
                    "rate_MBps": rate_bps / 1e6, "bytes_source": source}
    return None


def fit_read_rate(used):
    for u in used:
        if u["read_s"] and u["read_bytes"]:
            return {"ranks": u["ranks"], "bytes": u["read_bytes"], "seconds": u["read_s"],
                     "rate_bytes_per_s": u["read_bytes"] / u["read_s"]}
    return None


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def build_output_calibration(args, used, excluded, startup_fit, write_fit, read_fit,
                              base_calibration, base_path, host):
    out = copy.deepcopy(base_calibration)
    date = _now_iso()
    measured_fields = []
    transferred_fields = sorted(
        k for k in base_calibration.keys()
        if k not in ("site", "description")
    )

    out["site"] = args.site
    out["description"] = (
        "Auto-generated by wtune_calibrate.py fit on %s from runs under %s "
        "(base template: %s). Fields listed in calibration_provenance.measured_fields below "
        "were fitted from this machine's own runs; everything else is copied verbatim from the "
        "base file and is TRANSFERRED, not measured here -- see calibration_provenance." %
        (date, args.runs, base_path))

    if startup_fit["cap"] is not None:
        out["startup_s"] = startup_fit["cap"]
        out["startup_s_source"] = (
            "MEASURED on %s %s by wtune_calibrate.py fit: median wall - sum(d01 'Timing for "
            "main') over %d run(s) with ranks >= cores_per_node=%d (%s)" %
            (host, date, startup_fit["cap_n"], args.cores_per_node,
             startup_fit["at_or_above_points"]))
        measured_fields += ["startup_s", "startup_s_source"]
    else:
        out.setdefault("startup_s_note", "")
        out["startup_s_note"] = ("no run with ranks >= cores_per_node=%d completed -- startup_s "
                                  "TRANSFERRED from %s, not re-measured" %
                                  (args.cores_per_node, base_path))

    if startup_fit["base"] is not None and startup_fit["per_rank"] is not None:
        out["startup_s_base"] = startup_fit["base"]
        out["startup_s_per_rank"] = startup_fit["per_rank"]
        out["startup_s_scaling_source"] = (
            "MEASURED on %s %s by wtune_calibrate.py fit: least-squares wall - sum(d01 'Timing "
            "for main') over runs below one node (ranks < cores_per_node=%d): %s; model "
            "min(startup_s, startup_s_base + startup_s_per_rank*ranks)" %
            (host, date, args.cores_per_node, startup_fit["below_points"]))
        measured_fields += ["startup_s_base", "startup_s_per_rank", "startup_s_scaling_source"]
    else:
        out["startup_s_scaling_note"] = (
            "fewer than 2 runs below one node (cores_per_node=%d) completed -- "
            "startup_s_base/startup_s_per_rank TRANSFERRED from %s, not re-fitted" %
            (args.cores_per_node, base_path))

    if write_fit is not None:
        out["history_write_rate_MBps_measured"] = write_fit["rate_MBps"]
        out["history_write_rate_MBps_measured_note"] = (
            "MEASURED on %s %s by wtune_calibrate.py fit: one d01 history frame from the "
            "ranks=%d run, %.0f bytes (%s) / %.3fs 'Timing for Writing' = %.3f MB/s (decimal). "
            "Informational only -- wtune_predict.py does not add history I/O to the total by "
            "default (see --include-io); history_write_rate_MBps_low/_high below are UNCHANGED "
            "(still TRANSFERRED from %s), a single measurement is not enough to set a bracket." %
            (host, date, write_fit["ranks"], write_fit["bytes"], write_fit["bytes_source"],
             write_fit["seconds"], write_fit["rate_MBps"], base_path))
        measured_fields += ["history_write_rate_MBps_measured", "history_write_rate_MBps_measured_note"]
    else:
        out["history_write_rate_MBps_measured_note"] = (
            "no run wrote a history frame with a parseable 'Timing for Writing' line -- "
            "no measured write rate available; history_write_rate_MBps_low/_high TRANSFERRED "
            "from %s" % base_path)

    derived = out.setdefault("derived", {})
    if read_fit is not None:
        derived["read_rate_bytes_per_s"] = read_fit["rate_bytes_per_s"]
        derived["read_rate_formula"] = (
            "MEASURED on %s %s by wtune_calibrate.py fit: ranks=%d run, %.0f bytes of real "
            "on-disk wrfinput_d*/wrfbdy_d* / %.3fs separable read-timing line(s) = %.3f B/s" %
            (host, date, read_fit["ranks"], read_fit["bytes"], read_fit["seconds"],
             read_fit["rate_bytes_per_s"]))
        measured_fields += ["derived.read_rate_bytes_per_s", "derived.read_rate_formula"]
        if "derived" in transferred_fields:
            pass
    else:
        derived["read_rate_transferred_note"] = (
            "no run's rsl log had a separable read-only timing line distinct from the "
            "startup/init window -- derived.read_rate_bytes_per_s TRANSFERRED verbatim from "
            "%s, NOT measured on this machine" % base_path)

    out["calibration_provenance"] = {
        "generated_by": "wtune_calibrate.py fit", "generated_at": date, "host": host,
        "base_file": str(base_path), "runs_dir": str(args.runs),
        "measured_fields": measured_fields,
        "transferred_fields": transferred_fields,
    }
    out["validity"] = {
        "n_run_dirs_found": len(used) + len(excluded),
        "n_runs_used": len(used),
        "n_runs_excluded": len(excluded),
        "excluded": excluded,
        "used_ranks": sorted(u["ranks"] for u in used),
    }
    return out


def render_summary(used, excluded, startup_fit, write_fit, read_fit, out_path):
    lines = []
    lines.append("runs found: %d used, %d excluded" % (len(used), len(excluded)))
    lines.append("%-8s %10s %12s %12s" % ("ranks", "wall_s", "sum_main_s", "startup_s"))
    for u in sorted(used, key=lambda x: x["ranks"]):
        lines.append("%-8d %10.3f %12.3f %12.3f" %
                      (u["ranks"], u["wall_s"], u["sum_main_s"], u["startup_measured"]))
    for e in excluded:
        lines.append("EXCLUDED ranks=%s dir=%s reason=%s" % (e["ranks"], e["dir"], e["reason"]))
    lines.append("")
    lines.append("startup fit: cap=%s (n=%d, >=cores_per_node)  base=%s  per_rank=%s" %
                  (startup_fit["cap"], startup_fit["cap_n"], startup_fit["base"],
                   startup_fit["per_rank"]))
    if write_fit:
        lines.append("history write rate: %.3f MB/s (ranks=%d, %.0f bytes / %.3fs)" %
                      (write_fit["rate_MBps"], write_fit["ranks"], write_fit["bytes"],
                       write_fit["seconds"]))
    else:
        lines.append("history write rate: not measured (no separable 'Timing for Writing' line)")
    if read_fit:
        lines.append("read rate: %.3f B/s (ranks=%d, %.0f bytes / %.3fs)" %
                      (read_fit["rate_bytes_per_s"], read_fit["ranks"], read_fit["bytes"],
                       read_fit["seconds"]))
    else:
        lines.append("read rate: not separable -- kept from --base, TRANSFERRED not measured")
    lines.append("")
    lines.append("written: %s" % out_path)
    return "\n".join(lines)


def fit_cmd(args):
    runs_dir = Path(args.runs).resolve()
    manifest = None
    manifest_path = runs_dir / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    used, excluded = collect_runs(runs_dir, manifest)
    startup_fit = fit_startup(used, args.cores_per_node)

    base_calibration = wp.load_calibration(args.base)
    geom = None
    if manifest and manifest.get("e_we") and manifest.get("e_sn"):
        # d01 geometry only (write_fit's bytes_per_cell_level fallback is a d01-frame estimate,
        # same convention as wtune_predict.predict_init's own no-real-wrfinput fallback).
        geom = (manifest["e_we"][0], manifest["e_sn"][0], None)
        e_vert = base_calibration.get("derived", {}).get("bytes_per_cell_level")
        # e_vert is not in MANIFEST.json (plan doesn't record it); without it the bytes_per_cell_level
        # fallback can't be computed, so geom stays usable only when a real wrfout file is on disk.
        geom = None if e_vert is None else geom

    write_fit = fit_history_write_rate(used, base_calibration, geom)
    read_fit = fit_read_rate(used)

    host = args.host or socket.gethostname()
    out = build_output_calibration(args, used, excluded, startup_fit, write_fit, read_fit,
                                    base_calibration, args.base, host)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(render_summary(used, excluded, startup_fit, write_fit, read_fit, out_path))
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_plan_subparser(sub):
    p = sub.add_parser("plan", help="write sbatch scripts that measure a case at several rank counts")
    p.add_argument("--site", required=True, help="campaign/sites/<site>.sh to source")
    p.add_argument("--case-dir", required=True,
                    help="directory with the user's own namelist.input (+ wrfinput_d*/wrfbdy_d*)")
    p.add_argument("--out", required=True, help="directory to write plan_<ranks>.sbatch + MANIFEST.json into")
    p.add_argument("--ranks-list", default=DEFAULT_RANKS_LIST,
                    help="comma-separated rank counts (default %s)" % DEFAULT_RANKS_LIST)
    p.add_argument("--cores-per-node", type=int, default=DEFAULT_CORES_PER_NODE,
                    help="default %d" % DEFAULT_CORES_PER_NODE)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="default %d" % DEFAULT_STEPS)
    p.add_argument("--min-patch", type=int, default=DEFAULT_MIN_PATCH,
                    help="WRF's own floor(e/nproc)>=min_patch legality gate (default %d)" %
                         DEFAULT_MIN_PATCH)
    p.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT,
                    help="sbatch --time for each rank-count job (default %s)" % DEFAULT_TIME_LIMIT)
    return p


def add_fit_subparser(sub):
    p = sub.add_parser("fit", help="fit a calibration JSON from plan's completed runs")
    p.add_argument("--runs", required=True, help="the --out directory plan wrote into (now containing run_<ranks>_<jobid>/)")
    p.add_argument("--out", required=True, help="output calibration JSON path")
    p.add_argument("--base", default=str(DEFAULT_CALIBRATION),
                    help="template calibration JSON to copy untouched fields from "
                         "(default machine/calibration/scc.json)")
    p.add_argument("--cores-per-node", type=int, default=DEFAULT_CORES_PER_NODE,
                    help="default %d" % DEFAULT_CORES_PER_NODE)
    p.add_argument("--site", default=None, help="'site' field to stamp in the output (default: base file's site + '-auto')")
    p.add_argument("--host", default=None, help="override the host name recorded in provenance strings (default: this machine's hostname)")
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wtune_calibrate.py")
    sub = parser.add_subparsers(dest="command")
    add_plan_subparser(sub)
    add_fit_subparser(sub)
    args = parser.parse_args(argv)

    if args.command == "fit" and args.site is None:
        try:
            base_site = wp.load_calibration(args.base).get("site", "unknown")
        except wp.PredictError:
            base_site = "unknown"
        args.site = base_site + "-auto"

    try:
        if args.command == "plan":
            return plan_cmd(args)
        if args.command == "fit":
            return fit_cmd(args)
    except CalibrateError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
