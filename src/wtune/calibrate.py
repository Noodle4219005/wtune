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

    --runs is repeatable: pass it once per plan --out directory (e.g. once for a small case's
    1/36/72/100-rank sweep that anchors startup_s/_base/_per_rank, and once for a second case's
    multi-node sweep -- 1/2/3/4 nodes at that case's own legal rank counts -- that anchors
    startup_s_per_node). Every run across every --runs dir is pooled before fitting: which runs
    end up feeding which constant falls out naturally from each run's own (ranks, nodes), not
    from which --runs directory it came from.

    Reads back the run directories plan's sbatch scripts create under each DIR
    (run_<ranks>_<jobid>/, each with wall.txt + rsl.error.0000/rsl.out.0000, matched against
    that DIR's MANIFEST.json when present), fits:
      - startup_s (the 1-node cap): median of wall - sum(d01 'Timing for main') over every run
        with ranks>=cores_per_node AND nodes==1 (a fully-loaded single node; from wall.txt's own
        NODES field) -- a sub-node run feeds startup_s_base/_per_rank instead, a multi-node run
        feeds startup_s_per_node instead, neither should also pull the cap's own median.
      - startup_s_base/startup_s_per_rank: least-squares wall - sum(main) over runs with
        ranks < --cores-per-node (same min(cap, base + per_rank*ranks) sub-node model
        wtune_predict.py already implements).
      - startup_s_per_node: the cap's growth per additional node beyond one (predict_init:
        startup_term = startup_s + startup_s_per_node*(nodes-1), evidence: a 4-node/192-rank
        sc26p2 run's wall - sum(main) ~32s vs a 1-node run's ~19.4s). Fitted from every run with
        a known node count AND real on-disk wrfinput_d*/wrfbdy_d* bytes: each run's own predicted
        read_s (those bytes / the effective read rate) is subtracted from wall - sum(main) first
        -- reads happen once from shared storage and do not scale with node count, so leaving
        them in would contaminate the per-node slope with per-run I/O noise instead of MPI/launch
        overhead. The 1-node points set a read-subtracted "cap_pure" baseline; a least-squares
        slope of (read-subtracted startup - cap_pure) against (nodes-1), forced through the
        origin, gives startup_s_per_node. None of this touches startup_s itself, which stays the
        RAW (not read-subtracted) 1-node median -- predict_init still adds read_s separately on
        top, unchanged from the existing scc.json convention.
      - a history-write rate from the one written frame when present (informational only).
      - a read rate, only when a run's rsl log has a separable read-only timing line distinct
        from the startup window; otherwise kept from --base and clearly marked TRANSFERRED.
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
import re
import socket
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO_ROOT = HERE.parent  # HERE is src/; machine/ and campaign/ stay at the true repo root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "predictor"))

from wtune import predict as wp          # noqa: E402  (path insert above)
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

def pick_near_square(shapes, aspect=1.0):
    """Of the legal (nx, ny) shapes for one rank count, pick the one whose PATCH is closest to
    square: patch_x/patch_y = (e_we/nx)/(e_sn/ny) = aspect*(ny/nx), so patch_x==patch_y exactly
    when nx/ny == aspect (e_we/e_sn) -- minimising |log(nx/ny) - log(aspect)| picks that shape
    (log scale so nx/ny=2*aspect and nx/ny=aspect/2 are equally bad, not asymmetric). aspect
    defaults to 1.0 (square domain -> minimise |log(nx/ny)|, i.e. the old rank-count-only
    behaviour). Ties broken by the smaller max(nx,ny), then by nx ascending, for full
    determinism. On a non-square domain (e.g. e_we=170 e_sn=130, aspect~1.31) this is why 72
    ranks picks (9,8) over (8,9): patch (170/9, 130/8)=(18.9, 16.25) is more square than
    (170/8, 130/9)=(21.25, 14.4).

    Uses log(nx)-log(ny) rather than log(nx/ny): IEEE754 subtraction is exactly antisymmetric
    (fl(a-b) == -fl(b-a)), so at the default aspect=1.0 a genuine tie like (3,4) vs (4,3) stays
    an EXACT tie and falls through to the deterministic tie-break below; log(nx/ny) does not
    have that guarantee (4/3 is not exactly representable, so log(3/4) and log(4/3) can differ
    from each other's negation in the last bit, silently deciding a tie that should not be
    decided by floating-point noise)."""
    log_aspect = math.log(aspect)
    return min(shapes, key=lambda xy: (abs((math.log(xy[0]) - math.log(xy[1])) - log_aspect),
                                        max(xy), xy[0]))


def build_decomp(nl, max_dom, ranks, min_patch):
    we_list, sn_list = [], []
    for d in range(max_dom):
        e_we, e_sn, _e_vert = wp.get_domain_geom(nl, d)
        we_list.append(int(e_we))
        sn_list.append(int(e_sn))
    shapes = ds.legal_shapes_exact(we_list, sn_list, ranks, min_patch=min_patch)
    if not shapes:
        return None
    # d01's own aspect ratio drives the "square patch" target for every domain at once (the
    # decomposition is shared across all active domains -- see legal_shapes_exact above).
    aspect = we_list[0] / sn_list[0]
    return pick_near_square(shapes, aspect=aspect)


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

# WRF run-time support files (LOOKUP tables etc.): link EVERY file in $W/run -- no
# extension/name whitelist -- except namelist.input, which is skipped because $W/run/namelist.input
# is a REAL file in this build (confirmed 2026-09-25) and `configs.py edit-namelist` writes its
# OUT path with a plain open(..., "w"), which follows a symlink and would corrupt that shared
# source-tree file; every other name is linked unconditionally, matching
# campaign/strict_campaign.sbatch's own file-linking loop.
for f in "$W"/run/*; do
  b=$(basename "$f")
  [ "$b" = "namelist.input" ] && continue
  [ -e "$b" ] || ln -sf "$f" "$b"
done

# Thompson microphysics lookup tables (freezeH2O.dat/qr_acr_qg_V4.dat/qr_acr_qsV2.dat):
# module_mp_thompson.F computes and WRITES these into the run dir at init whenever a
# same-named file is not already there, and this build's $W/run ships none of them (confirmed
# 2026-09-25 by the generic loop above finding nothing to link) -- computing them fresh is not
# fast (qr_acr_qg alone measured 32.3s at 1 rank) and, because it happened silently inside
# every calibration run's own timed window, it inflated every measured startup by however long
# that run's table computation took, contaminating jobs 1090-1099's whole calibration fit (see
# campaign/strict_campaign.sbatch's THOMPSON_TABLE_NAMES/link_thompson_tables/OUT for the
# identical production pattern this mirrors -- $OUT is site.sh's own campaign run-output dir,
# already exported by site_env above; $OUT/thompson_tables is where prior campaign runs on
# THIS site already cached them). A brand-new site with nothing cached yet just falls through
# ([ -f ... ] guard) and WRF computes fresh, same as before this fix -- no worse, not harvested
# back into the cache by this script (only linked, never written).
for tbl in freezeH2O.dat qr_acr_qg_V4.dat qr_acr_qsV2.dat; do
  [ -e "$tbl" ] && continue
  [ -f "$OUT/thompson_tables/$tbl" ] && ln -sf "$OUT/thompson_tables/$tbl" "$tbl"
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
# leaves a TIMEOUT'd mpirun's ranks invisible to Slurm's own cleanup). ALLOC_NODES MUST be set
# before the first site_hygiene call: scc.sh's site_hygiene() body reads $ALLOC_NODES itself,
# and `set -u` turns a reference to it before this line into a silent whole-script abort (its
# error goes to hygiene.log below, not to this job's own .out, which is why it once looked like
# the job died with no output at all).
ALLOC_NODES="${{SLURM_JOB_NUM_NODES:-{nodes}}}"
if command -v site_hygiene >/dev/null 2>&1; then
  site_hygiene reap >> hygiene.log 2>&1 || true
  [ -n "${{SITE_HYGIENE_CMD:-}}" ] && ( timeout 90 "${{SITE_HYGIENE_CMD[@]}}" >> hygiene.log 2>&1 || true )
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
  timeout 60 "${{SITE_HYGIENE_CMD[@]}}" >> hygiene.log 2>&1 || true
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
    """Returns (used, excluded): used is [{ranks, nodes, startup_measured, wall_s, sum_main_s,
    write_s, write_bytes, read_s, read_bytes, real_input_bytes, manifest_geom, dir}], excluded
    is [{ranks, dir, reason}]. Does not raise when every run it finds is excluded -- fit_cmd
    pools runs across every --runs dir given and only raises once nothing usable survived any
    of them; it DOES raise when this one directory has no run_<ranks>_<jobid>/ at all (almost
    always a wrong --runs path, worth failing on immediately rather than silently contributing
    nothing)."""
    run_dirs = sorted(runs_dir.glob("run_*_*"))
    if not run_dirs:
        raise CalibrateError("no run_<ranks>_<jobid> directories found under %s "
                              "(nothing to fit -- did the plan_*.sbatch jobs run yet?)" % runs_dir)

    manifest_by_ranks = {}
    manifest_geom = None
    if manifest:
        for r in manifest.get("runs", []):
            manifest_by_ranks[r["ranks"]] = r
        if manifest.get("e_we") and manifest.get("e_sn"):
            manifest_geom = {"e_we": manifest["e_we"][0], "e_sn": manifest["e_sn"][0],
                              "max_dom": manifest.get("max_dom")}

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
        nodes = None
        if "NODES" in wall:
            try:
                nodes = int(wall["NODES"])
            except ValueError:
                nodes = None
        if rc != 0 or ok != 1:
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": "run did not complete (RC=%d OK=%d)" % (rc, ok)})
            continue
        rsl_path = _find_rsl(run_dir)
        if rsl_path is None:
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": "no rsl.error.0000/rsl.out.0000"})
            continue
        text = rsl_path.read_text()

        # Thompson-table contamination (scc jobs 1090-1099, 2026-09-25): before this run dir's
        # $OUT/thompson_tables link fix, a missing cached table made WRF compute it from
        # scratch at init (module_mp_thompson.F "ThompMP: computing ..."), silently adding
        # tens of seconds INSIDE the measured startup window every time it happened. A run
        # whose rsl still shows this (a stale run dir from before the fix, or a genuinely new
        # site with no cache yet) cannot be trusted for a startup/init fit and is excluded
        # outright, even though it completed successfully (RC=0/OK=1) -- "ran to completion" and
        # "measures what we think it measures" are different questions.
        contam_markers = [ln for ln in text.splitlines()
                          if "ThompMP: computing" in ln or "Timing for table computation" in ln]
        if contam_markers:
            excluded.append({"ranks": ranks, "dir": str(run_dir),
                              "reason": ("contaminated: Thompson lookup table(s) computed at "
                                         "init instead of linked from cache (%d marker line(s), "
                                         "e.g. %r) -- startup window is not comparable to a run "
                                         "with the table already linked" %
                                         (len(contam_markers), contam_markers[0]))})
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

        # Real on-disk wrfinput_d*/wrfbdy_d* bytes, independent of whether a separable read
        # TIMING line exists (that's read_bytes above, for fit_read_rate's own use) -- this one
        # feeds fit_startup_per_node's read_s subtraction, which needs the bytes even when no
        # such line is present (it predicts read_s from the RATE, not from a measured read_s).
        real_input_bytes, _names = _dir_bytes(run_dir, ["wrfinput_d*", "wrfbdy_d*"])
        real_input_bytes = real_input_bytes or None

        used.append({
            "ranks": ranks, "nodes": nodes, "wall_s": wall_s, "sum_main_s": sum_main,
            "startup_measured": startup_measured, "write_s": write_s, "write_bytes": write_bytes,
            "read_s": read_s, "read_bytes": read_bytes, "real_input_bytes": real_input_bytes,
            "manifest_geom": manifest_geom, "dir": str(run_dir),
            "manifest": manifest_by_ranks.get(ranks),
        })
    return used, excluded


def _is_cap_point(u, cores_per_node):
    """The cap is the FULLY-LOADED single-node plateau: ranks>=cores_per_node (excludes the
    sub-node ranks that startup_s_base/_per_rank models instead) AND nodes==1 (excludes the
    multi-node runs that startup_s_per_node models instead -- with that field now modelling
    growth beyond node 1, blending a 2+-node run's larger startup into this median would double
    count that growth once startup_s_per_node is added back on top in predict_init). wall.txt
    (written by plan's sbatch template) always carries NODES; nodes==1 is ASSUMED when it is
    missing (a run directory from before that field existed), matching this function's
    pre-existing ranks-only behaviour exactly."""
    if u["ranks"] < cores_per_node:
        return False
    return u.get("nodes") in (None, 1)


def fit_startup(used, cores_per_node):
    below = sorted((u["ranks"], u["startup_measured"]) for u in used if u["ranks"] < cores_per_node)
    one_node = [u["startup_measured"] for u in used if _is_cap_point(u, cores_per_node)]

    cap = statistics.median(one_node) if one_node else None

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

    # predict_init (wtune_predict.py) computes startup_term = min(startup_s + ...,
    # startup_s_base + startup_s_per_rank*ranks) -- that min() assumes per_rank >= 0 (a
    # sub-node line that rises toward the cap and gets clamped by it). A NEGATIVE per_rank
    # makes that second term fall WITHOUT BOUND as ranks grows, so min() picks it over the cap
    # at large rank counts and predict_init returns a deeply negative "total_s" -- confirmed on
    # scc 2026-09-25 (a contaminated severe2023 1-rank point gave per_rank=-1.93, and predicting
    # conus12 at 288 ranks with the resulting calibration returned total_s=-376.6s). A negative
    # slope is not a smaller sub-node effect, it is measurement noise (most likely NFS/fabric
    # contention skewing a single small-sample point) masquerading as a fit -- reject it rather
    # than let it silently corrupt every prediction at a rank count larger than the points it
    # was fitted from.
    rejected_reason = None
    if per_rank is not None and per_rank < 0:
        rejected_reason = (
            "least-squares fit gave a NEGATIVE startup_s_per_rank=%.6g from points %s -- "
            "physically implausible (startup cannot fall without bound as ranks grow) and "
            "would make predict_init's min(startup_s+..., startup_s_base+per_rank*ranks) return "
            "unbounded-negative totals at rank counts beyond these points; rejected, not written"
            % (per_rank, below))
        base_clipped = None
        per_rank = None

    return {"cap": cap, "cap_n": len(one_node), "base": base_clipped,
            "base_unclipped": base, "per_rank": per_rank,
            "below_points": below, "one_node_points": one_node,
            "rejected_reason": rejected_reason}


def fit_startup_per_node(used, effective_read_rate):
    """startup_s_per_node: seconds the 1-node cap grows per additional node (predict_init:
    startup_term = startup_s + startup_s_per_node*(nodes-1), evidence: sc26p2 4 nodes/192 ranks
    wall-sum(main) ~32s vs the 1-node ~19.4s cap). Isolates the growth from read-time noise: each
    run's OWN predicted read_s (its real on-disk wrfinput_d*/wrfbdy_d* bytes / effective_read_rate)
    is subtracted from wall - sum(main) first, since a read from shared storage happens once and
    does not scale with node count -- leaving it in would fold per-run I/O variance into what is
    supposed to be a pure launch/MPI-coordination slope. The 1-node points set a read-subtracted
    baseline (cap_pure); the multi-node points' distance from that baseline, divided by
    (nodes-1) via least squares forced through the origin (no intercept -- by construction a
    0-node run does not exist), is the slope. Returns None without enough of both kinds of point."""
    pts = []
    for u in used:
        if u.get("nodes") is None or not u.get("real_input_bytes") or not effective_read_rate:
            continue
        read_s_pred = u["real_input_bytes"] / effective_read_rate
        startup_pure = (u["wall_s"] - u["sum_main_s"]) - read_s_pred
        pts.append({"nodes": u["nodes"], "ranks": u["ranks"], "startup_pure": startup_pure,
                    "read_s_pred": read_s_pred, "dir": u["dir"]})

    one_node = [p["startup_pure"] for p in pts if p["nodes"] == 1]
    multi_node = [p for p in pts if p["nodes"] > 1]
    if not one_node or not multi_node:
        return None

    cap_pure = statistics.median(one_node)
    num = sum((p["nodes"] - 1) * (p["startup_pure"] - cap_pure) for p in multi_node)
    den = sum((p["nodes"] - 1) ** 2 for p in multi_node)
    if den == 0:
        return None
    slope = num / den

    result = {"per_node": slope, "cap_pure": cap_pure, "one_node_n": len(one_node),
              "multi_node_points": [(p["nodes"], p["ranks"], round(p["startup_pure"], 3))
                                     for p in multi_node],
              "effective_read_rate": effective_read_rate, "rejected_reason": None}

    # Symmetric guard to fit_startup's per_rank rejection above: predict_init documents this
    # field as "the cap grows with node count" (startup_term += per_node*(nodes-1)) -- a
    # NEGATIVE slope means the model would predict a SHRINKING cap at higher node counts,
    # contradicting the evidence the field was added for (sc26p2 4 nodes ~32s vs 1 node
    # ~19.4s) and, same as a negative per_rank, is far more likely to be a contaminated
    # 1-node cap_pure (one noisy single-node run pulling the median off) than a real effect.
    if slope < 0:
        result["per_node"] = None
        result["rejected_reason"] = (
            "least-squares fit gave a NEGATIVE startup_s_per_node=%.6g (cap_pure=%.3fs from "
            "%d nodes==1 point(s), multi-node points (nodes,ranks,startup_pure)=%s) -- "
            "physically implausible (the cap should grow, not shrink, with node count) and "
            "most likely means cap_pure itself is contaminated by a noisy 1-node run; "
            "rejected, not written" %
            (slope, cap_pure, len(one_node), result["multi_node_points"]))

    return result


def fit_history_write_rate(used, base_calibration):
    for u in used:
        if u["write_s"]:
            bytes_ = u["write_bytes"]
            source = "measured wrfout_d01 file size"
            if bytes_ is None:
                bpcl = base_calibration.get("derived", {}).get("bytes_per_cell_level")
                geom = u.get("manifest_geom")
                if bpcl is not None and geom and geom.get("e_we") and geom.get("e_sn"):
                    # e_vert is not in MANIFEST.json (plan doesn't record it); without a real
                    # e_vert this ESTIMATED path can't be computed either, so it only ever fires
                    # when a real wrfout file was NOT found on disk AND geometry is known but
                    # e_vert still is not -- i.e. effectively never with the current MANIFEST.json
                    # schema. Left in place (rather than removed) because MANIFEST.json gaining
                    # e_vert later should make this branch start working with no code change here.
                    e_vert = geom.get("e_vert")
                    if e_vert:
                        bytes_ = bpcl * geom["e_we"] * geom["e_sn"] * e_vert
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


def build_output_calibration(args, used, excluded, startup_fit, per_node_fit, write_fit,
                              read_fit, base_calibration, base_path, host,
                              read_rate_disabled_reason=None):
    out = copy.deepcopy(base_calibration)
    date = _now_iso()
    measured_fields = []
    transferred_fields = sorted(
        k for k in base_calibration.keys()
        if k not in ("site", "description")
    )

    runs_dirs = args.runs if isinstance(args.runs, list) else [args.runs]
    out["site"] = args.site
    out["description"] = (
        "Auto-generated by wtune_calibrate.py fit on %s from runs under %s "
        "(base template: %s). Fields listed in calibration_provenance.measured_fields below "
        "were fitted from this machine's own runs; everything else is copied verbatim from the "
        "base file and is TRANSFERRED, not measured here -- see calibration_provenance." %
        (date, runs_dirs, base_path))

    if startup_fit["cap"] is not None:
        out["startup_s"] = startup_fit["cap"]
        out["startup_s_source"] = (
            "MEASURED on %s %s by wtune_calibrate.py fit: median wall - sum(d01 'Timing for "
            "main') over %d run(s) with nodes==1 (%s)" %
            (host, date, startup_fit["cap_n"], startup_fit["one_node_points"]))
        measured_fields += ["startup_s", "startup_s_source"]
    else:
        out.setdefault("startup_s_note", "")
        out["startup_s_note"] = ("no completed run had nodes==1 -- startup_s TRANSFERRED from "
                                  "%s, not re-measured" % base_path)

    if startup_fit["base"] is not None and startup_fit["per_rank"] is not None:
        out["startup_s_base"] = startup_fit["base"]
        out["startup_s_per_rank"] = startup_fit["per_rank"]
        out["startup_s_scaling_source"] = (
            "MEASURED on %s %s by wtune_calibrate.py fit: least-squares wall - sum(d01 'Timing "
            "for main') over runs below one node (ranks < cores_per_node=%d): %s; model "
            "min(startup_s, startup_s_base + startup_s_per_rank*ranks)" %
            (host, date, args.cores_per_node, startup_fit["below_points"]))
        measured_fields += ["startup_s_base", "startup_s_per_rank", "startup_s_scaling_source"]
    elif startup_fit.get("rejected_reason"):
        out["startup_s_scaling_note"] = (
            "REJECTED, not fitted: %s -- startup_s_base/startup_s_per_rank TRANSFERRED from %s "
            "instead" % (startup_fit["rejected_reason"], base_path))
    else:
        out["startup_s_scaling_note"] = (
            "fewer than 2 runs below one node (cores_per_node=%d) completed -- "
            "startup_s_base/startup_s_per_rank TRANSFERRED from %s, not re-fitted" %
            (args.cores_per_node, base_path))

    if per_node_fit is not None and per_node_fit.get("rejected_reason"):
        out["startup_s_per_node_note"] = (
            "REJECTED, not fitted: %s -- startup_s_per_node left ABSENT (predict_init treats a "
            "missing key as 0, i.e. no cap growth with node count, same as before this field "
            "existed); this is not a TRANSFERRED value, --base has no such key to copy" %
            per_node_fit["rejected_reason"])
    elif per_node_fit is not None:
        out["startup_s_per_node"] = per_node_fit["per_node"]
        out["startup_s_per_node_source"] = (
            "MEASURED on %s %s by wtune_calibrate.py fit: read_s-subtracted (wall - sum(d01 "
            "'Timing for main') - real_input_bytes/effective_read_rate[=%.6g B/s]) startup at "
            "%d node(s)==1 point(s) (median=cap_pure=%.3fs) vs %d multi-node point(s) "
            "[(nodes,ranks,startup_pure), ...]=%s; least-squares slope of "
            "(startup_pure - cap_pure) against (nodes-1), forced through the origin. Consumed by "
            "predict_init as startup_term = startup_s + startup_s_per_node*(nodes-1), BEFORE the "
            "min(...) against startup_s_base + startup_s_per_rank*ranks -- startup_s itself stays "
            "the raw (not read-subtracted) 1-node median, unchanged." %
            (host, date, per_node_fit["effective_read_rate"], per_node_fit["one_node_n"],
             per_node_fit["cap_pure"], len(per_node_fit["multi_node_points"]),
             per_node_fit["multi_node_points"]))
        measured_fields += ["startup_s_per_node", "startup_s_per_node_source"]
    else:
        out["startup_s_per_node_note"] = (
            "not fitted: need at least one nodes==1 run and one nodes>1 run, both with real "
            "on-disk wrfinput_d*/wrfbdy_d* bytes -- startup_s_per_node left ABSENT (predict_init "
            "treats a missing key as 0, i.e. no cap growth with node count, same as before this "
            "field existed); this is not a TRANSFERRED value, --base has no such key to copy")

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
    if read_rate_disabled_reason:
        derived["read_rate_transferred_note"] = (
            "read-rate measurement disabled for this fit (--no-read-rate): %s -- "
            "derived.read_rate_bytes_per_s TRANSFERRED verbatim from %s, NOT measured on this "
            "machine even though a separable read-timing line may have been present" %
            (read_rate_disabled_reason, base_path))
    elif read_fit is not None:
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
        "base_file": str(base_path), "runs_dirs": [str(d) for d in runs_dirs],
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


def render_summary(used, excluded, startup_fit, per_node_fit, write_fit, read_fit, out_path):
    lines = []
    lines.append("runs found: %d used, %d excluded" % (len(used), len(excluded)))
    lines.append("%-8s %6s %10s %12s %12s" % ("ranks", "nodes", "wall_s", "sum_main_s", "startup_s"))
    for u in sorted(used, key=lambda x: (x["ranks"], x.get("nodes") or 0)):
        lines.append("%-8d %6s %10.3f %12.3f %12.3f" %
                      (u["ranks"], u.get("nodes"), u["wall_s"], u["sum_main_s"],
                       u["startup_measured"]))
    for e in excluded:
        lines.append("EXCLUDED ranks=%s dir=%s reason=%s" % (e["ranks"], e["dir"], e["reason"]))
    lines.append("")
    lines.append("startup fit: cap=%s (n=%d, nodes==1)  base=%s  per_rank=%s" %
                  (startup_fit["cap"], startup_fit["cap_n"], startup_fit["base"],
                   startup_fit["per_rank"]))
    if startup_fit.get("rejected_reason"):
        lines.append("startup_s_base/per_rank REJECTED: %s" % startup_fit["rejected_reason"])
    if per_node_fit is not None and per_node_fit.get("rejected_reason"):
        lines.append("startup_s_per_node REJECTED: %s" % per_node_fit["rejected_reason"])
    elif per_node_fit:
        lines.append("startup_s_per_node fit: per_node=%.4fs/node  cap_pure=%.3fs (n=%d nodes==1)  "
                      "multi-node points (nodes,ranks,startup_pure)=%s" %
                      (per_node_fit["per_node"], per_node_fit["cap_pure"],
                       per_node_fit["one_node_n"], per_node_fit["multi_node_points"]))
    else:
        lines.append("startup_s_per_node fit: not enough nodes==1/nodes>1 points -- left unset")
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
    runs_arg = args.runs if isinstance(args.runs, list) else [args.runs]

    used, excluded = [], []
    for runs_dir_str in runs_arg:
        runs_dir = Path(runs_dir_str).resolve()
        manifest = None
        manifest_path = runs_dir / "MANIFEST.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        u, e = collect_runs(runs_dir, manifest)
        used += u
        excluded += e
    if not used:
        raise CalibrateError(
            "every run across all --runs dirs (%s) was excluded -- see reasons above; nothing "
            "to fit" % runs_arg)

    startup_fit = fit_startup(used, args.cores_per_node)

    base_calibration = wp.load_calibration(args.base)
    write_fit = fit_history_write_rate(used, base_calibration)
    read_rate_disabled_reason = args.no_read_rate_reason if args.no_read_rate_reason else (
        "--no-read-rate was given" if args.no_read_rate else None)
    read_fit = None if read_rate_disabled_reason else fit_read_rate(used)
    effective_read_rate = (read_fit["rate_bytes_per_s"] if read_fit is not None
                            else base_calibration.get("derived", {}).get("read_rate_bytes_per_s"))
    per_node_fit = fit_startup_per_node(used, effective_read_rate)

    host = args.host or socket.gethostname()
    out = build_output_calibration(args, used, excluded, startup_fit, per_node_fit, write_fit,
                                    read_fit, base_calibration, args.base, host,
                                    read_rate_disabled_reason=read_rate_disabled_reason)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(render_summary(used, excluded, startup_fit, per_node_fit, write_fit, read_fit, out_path))
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
    p.add_argument("--runs", required=True, action="append",
                    help="a --out directory plan wrote into (now containing run_<ranks>_<jobid>/); "
                         "repeatable -- give one per plan run (e.g. once for a small case's "
                         "1/36/72/100-rank sweep, once for a second case's multi-node sweep)")
    p.add_argument("--out", required=True, help="output calibration JSON path")
    p.add_argument("--base", default=str(DEFAULT_CALIBRATION),
                    help="template calibration JSON to copy untouched fields from "
                         "(default machine/calibration/scc.json)")
    p.add_argument("--cores-per-node", type=int, default=DEFAULT_CORES_PER_NODE,
                    help="default %d" % DEFAULT_CORES_PER_NODE)
    p.add_argument("--site", default=None, help="'site' field to stamp in the output (default: base file's site + '-auto')")
    p.add_argument("--host", default=None, help="override the host name recorded in provenance strings (default: this machine's hostname)")
    p.add_argument("--no-read-rate", action="store_true",
                    help="never write a MEASURED derived.read_rate_bytes_per_s from this batch's "
                         "runs, even if a separable read-timing line exists -- always TRANSFER "
                         "it from --base instead (use when this batch's reads are known-suspect, "
                         "e.g. NFS client-cache warmed by back-to-back same-file runs, as scc's "
                         "2026-09-25 396 MB/s outlier was: 7x the base file's measured rate)")
    p.add_argument("--no-read-rate-reason", default=None,
                    help="custom explanation recorded in derived.read_rate_transferred_note; "
                         "implies --no-read-rate")
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
