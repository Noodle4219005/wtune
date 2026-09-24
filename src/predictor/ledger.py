#!/usr/bin/env python3
"""predictor/ledger.py -- canonical run ledger for the WTune configuration predictor (phase A).

Builds one row per *measured* WRF run from the processed CSVs under the wiki's
`data/` tree, in a single schema (LEDGER_FIELDS below). Every adapter documents its
unit conversions and cites the wiki note it borrowed a fixed-config field from.

Stdlib only. Read-only on the wiki data tree; writes only under this repo's
`predictor/` directory.

CLI:
    python3 ledger.py build --wiki-data /home/u5727520/my-llm-wiki/data --out predictor/ledger/runs.csv
    python3 ledger.py validate --ledger predictor/ledger/runs.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import wtune_compliance  # repo root, read-only import
except ImportError:
    wtune_compliance = None  # classify() falls back to UNCLASSIFIED-safe heuristics below

HARDWARE_ID = "f1-spr-icpnq"  # f1 Sapphire Rapids icpnq nodes: 112 cores/node, 2 sockets,
                               # SNC 8 NUMA domains x 14 cores, SMT off (see AGENTS/CLAUDE.md HPC notes)
CORES_PER_NODE = 112

LEDGER_FIELDS = [
    # identity
    "run_id", "source_dataset", "job_id", "rep_index", "case_raw", "case_base", "case_family",
    "hardware_id",
    # binary / classification
    "binary_kind", "compliance_class",
    # grid and physics
    "e_we", "e_sn", "e_vert", "dx", "dy", "time_step",
    "mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics",
    "ra_lw_physics", "ra_sw_physics", "radt", "cudt",
    "history_interval", "io_form_history",
    # parallel config
    "n_nodes", "n_ranks", "omp_threads", "nproc_x", "nproc_y", "numtiles", "distribution",
    "kmp_blocktime",
    # timing
    "s_per_step_median", "s_per_step_mean", "n_steps_measured",
    # correctness
    "diffwrf_differing_fields", "fidelity_label",
    # profile
    "profile_micro_share", "profile_radiation_share", "profile_dynamics_share",
    "profile_mpi_share", "profile_io_share", "profile_rank_imbalance", "profile_thread_imbalance",
    "notes",
]

# Fields not in LEDGER_FIELDS, carried per-row only to decide compliance_class (never written out).
_EXTRA_KNOB_KEY = "_extra_knobs"

# ledger field -> wtune_compliance.py catalog knob name, used to classify a config that differs
# from its case's baseline. Only fields that ever vary within one case_base are listed; grid
# geometry (e_we/e_sn/e_vert/dx) is a case property, not a config knob, so it is excluded here.
KNOB_CATALOG_NAME = {
    "nproc_x": "nproc_x", "nproc_y": "nproc_y", "numtiles": "numtiles",
    "omp_threads": "OMP_NUM_THREADS", "distribution": "srun_distribution",
    "kmp_blocktime": "KMP_BLOCKTIME",
    "mp_physics": "mp_physics", "cu_physics": "cu_physics", "bl_pbl_physics": "bl_pbl_physics",
    "sf_sfclay_physics": "sf_sfclay_physics", "sf_surface_physics": "sf_surface_physics",
    "ra_lw_physics": "ra_lw_physics", "ra_sw_physics": "ra_sw_physics",
    "radt": "radt", "cudt": "cudt", "time_step": "time_step",
    "io_form_history": "io_form_history",
    # history_interval has no dedicated catalog entry (it changes output cadence, not the
    # computed solution or the execution path) -> deliberately not classified as a config knob.
}

_RANK = {"STRICT": 0, "PHYSICS-GRAY": 1, "RESEARCH": 2}


def _f(x):
    """Parse a CSV cell to float, or None for blank/NA-like values."""
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in ("na", "nan", "none", "-"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _i(x):
    v = _f(x)
    return int(v) if v is not None else None


def new_row(**kw):
    row = {k: None for k in LEDGER_FIELDS}
    row[_EXTRA_KNOB_KEY] = {}
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# case_base / case_family derivation
#
# knowledge/2026-09-05-loco-over-directories-leaks-when-most-of-the-corpus-is-variants-of-the-same-case.md:
# strip scheme variants (__mpXblY) and decomposition variants (_NNxMM[hNN]); this task's brief
# additionally requires stripping probe_/decay_/floor_ prefixes and _48h/_cat4 suffixes. All of
# these are the *same weather*, just a different scheme/decomposition/probe-length/intensity tag.
# ---------------------------------------------------------------------------
_SCHEME_RE = re.compile(r"__mp\d+bl\d+$")
_DECOMP_RE = re.compile(r"_\d+x\d+(h\d+)?$")
_PREFIXES = ("probe_", "decay_", "floor_")
_SUFFIXES = ("_48h", "_cat4")


def case_base_of(name: str) -> str:
    b = name
    b = _SCHEME_RE.sub("", b)
    b = _DECOMP_RE.sub("", b)
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if b.startswith(p):
                b = b[len(p):]
                changed = True
        for s in _SUFFIXES:
            if b.endswith(s):
                b = b[: -len(s)]
                changed = True
    return b


# Coarse, documented-as-a-heuristic regime grouping for leave-one-family-out CV. This is a
# judgment call (the brief does not define "family"); it groups idealized WRF cases by the kind
# of weather system so that e.g. "leave out all tropical cyclones" is a meaningful fold. Case
# names not recognised fall back to their own case_base (singleton family) rather than a
# fabricated guess.
_FAMILY_MAP = {
    "karina": "cyclone", "edouard_landfall": "cyclone", "lowell_cat4": "cyclone",
    "marie_ts": "cyclone", "extratrop_cyc": "cyclone",
    "deep_conv_ot": "convective", "dry_conv": "convective", "squall_line3k": "convective",
    "tropical_mcs": "convective",
    "arctic_sbl": "boundary_layer", "marine_sc_inv": "boundary_layer", "marine_cao": "boundary_layer",
    "lake_effect": "boundary_layer", "foehn_dry": "boundary_layer", "steep_lapse": "boundary_layer",
    "polar_low": "boundary_layer", "mountain_wave": "boundary_layer",
    "atmos_river": "synoptic", "synoptic_pacific": "synoptic", "monsoon_indian": "synoptic",
    "nest1km": "synoptic",
    "base448": "calibration", "m448": "calibration",
}


def case_family_of(case_base: str) -> str:
    return _FAMILY_MAP.get(case_base, case_base)


# ---------------------------------------------------------------------------
# Adapter 1: data/wtune/2026-09-07-wtune-config-cost.csv (76 rows)
# knowledge/experiments/2026-09-07-config-cost-is-multiplicatively-separable-to-2-percent.md
#
# s_per_model_h is wall-seconds per simulated model-hour (namelist-derived model_h); time_step is
# already in *seconds* in this CSV (15/6, matching the corpus's dt). So:
#   s_per_step = s_per_model_h * time_step / 3600
# n_nodes is not a column; every n_ranks value here (448 or 896) is exactly filled MPI-only nodes
# at 112 cores/node (omp_threads=1, no OMP column in this CSV), so n_nodes = n_ranks / 112.
# ---------------------------------------------------------------------------
def adapt_config_cost(path):
    rows = []
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            case_raw = r["run"]
            case_base = case_base_of(r["case"])
            n_ranks = _i(r["n_ranks"])
            omp_threads = 1
            n_nodes = n_ranks / CORES_PER_NODE if n_ranks and n_ranks % CORES_PER_NODE == 0 else None
            time_step = _f(r["time_step"])
            s_per_model_h = _f(r["s_per_model_h"])
            s_per_step = s_per_model_h * time_step / 3600.0 if (s_per_model_h and time_step) else None
            row = new_row(
                run_id="cost:%s" % case_raw,
                source_dataset=os.path.basename(path),
                job_id=None, rep_index=0,
                case_raw=case_raw, case_base=case_base, case_family=case_family_of(case_base),
                hardware_id=HARDWARE_ID,
                binary_kind="pristine", compliance_class=None,
                e_we=_f(r["e_we"]), e_sn=_f(r["e_sn"]), e_vert=_f(r["e_vert"]),
                dx=_f(r["dx"]), dy=_f(r["dx"]),  # square cells in this corpus (dy column absent)
                time_step=time_step,
                mp_physics=_i(r["mp_physics"]), cu_physics=_i(r["cu_physics"]),
                bl_pbl_physics=_i(r["bl_pbl_physics"]), sf_sfclay_physics=_i(r["sf_sfclay_physics"]),
                sf_surface_physics=_i(r["sf_surface_physics"]), ra_lw_physics=_i(r["ra_lw_physics"]),
                ra_sw_physics=_i(r["ra_sw_physics"]), radt=_f(r["radt"]), cudt=None,
                history_interval=_f(r["history_interval"]), io_form_history=_i(r["io_form_history"]),
                n_nodes=n_nodes, n_ranks=n_ranks, omp_threads=omp_threads,
                nproc_x=_i(r["nproc_x"]), nproc_y=_i(r["nproc_y"]), numtiles=_i(r["numtiles"]),
                distribution=None, kmp_blocktime=None,
                s_per_step_median=s_per_step, s_per_step_mean=s_per_step, n_steps_measured=None,
                diffwrf_differing_fields=None, fidelity_label=None,
                profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
                profile_mpi_share=None, profile_io_share=None,
                profile_rank_imbalance=_f(r["imbalance_M_pct"]), profile_thread_imbalance=None,
                notes="variant=%s comp_frac_pct=%s; s_per_step derived from s_per_model_h*time_step/3600"
                      % (r["variant"] or "-", r["comp_frac_pct"]),
            )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Adapter 2: data/wtune/2026-09-14-numtiles-12-13-14-abab-pristine-karina-1074183.csv (9 rows)
# knowledge/experiments/2026-09-14-even-numtiles-12-or-14-is-1-4-percent-faster-than-13-on-224x2-...md
#
# Case is explicitly "karina" (title: "224x2 hybrid"). e_we/e_sn=901/901 and e_vert=45 are stated
# in the linked elastic-core note (same karina case, "karina 901x901x45"); the rest of the physics
# stack is not restated in either note, so it is filled from the karina baseline row in
# 2026-09-07-wtune-config-cost.csv (case=karina, mp=8/bl=1/cu=0/ra_lw=4/ra_sw=4/sfclay=1/surf=2),
# cited in `notes`. distribution=cyclic is stated in the note body ("224 rank x 2 OpenMP thread,
# 4 nodes, cyclic"). nproc_x/nproc_y for this 224-rank hybrid decomposition are NOT stated anywhere
# -> left null rather than guessed.
# ---------------------------------------------------------------------------
_KARINA_PHYSICS = dict(mp_physics=8, cu_physics=0, bl_pbl_physics=1, sf_sfclay_physics=1,
                        sf_surface_physics=2, ra_lw_physics=4, ra_sw_physics=4)


def adapt_numtiles_12_13_14(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            tag = r["tag"]  # e.g. V13_r1
            diff = r["diffwrf_vs_first_V13"]
            fidelity = "reference" if diff == "ref" else ("bit_identical" if diff == "0" else None)
            row = new_row(
                run_id="numtiles121314:%s" % tag,
                source_dataset=os.path.basename(path),
                job_id="1074183", rep_index=_i(r["order"]),
                case_raw=tag, case_base="karina", case_family=case_family_of("karina"),
                hardware_id=HARDWARE_ID,
                binary_kind="pristine", compliance_class=None,
                e_we=901.0, e_sn=901.0, e_vert=45.0, dx=3000.0, dy=3000.0, time_step=15.0,
                radt=15.0, cudt=None, history_interval=None, io_form_history=None,
                n_nodes=4, n_ranks=224, omp_threads=2, nproc_x=None, nproc_y=None,
                numtiles=_i(r["numtiles"]), distribution="cyclic", kmp_blocktime=None,
                s_per_step_median=_f(r["med"]), s_per_step_mean=_f(r["mean_skip1"]),
                n_steps_measured=_i(r["steps"]),
                diffwrf_differing_fields=(0.0 if diff == "0" else None), fidelity_label=fidelity,
                profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
                profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
                profile_thread_imbalance=None,
                notes="physics filled from data/wtune/2026-09-07-wtune-config-cost.csv karina row "
                      "(same case); grid/e_vert/distribution from 2026-09-13 elastic-core note and "
                      "2026-09-14 numtiles-12-13-14 note; med_skip1/mean_skip1 also available but "
                      "not carried (median/mean-of-full-run chosen for consistency with other "
                      "sources); srun_wall=%ss" % r["srun_wall"],
                **_KARINA_PHYSICS,
            )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Adapter 3: data/wtune/2026-09-13-elastic-core-wrf-score-arms.csv (25 data rows)
# knowledge/experiments/2026-09-13-no-wrap-elastic-core-loses-on-real-wrf-...md
#
# Same karina 224x2x4-node case as adapter 2 (title: "karina 901x901x45, 224 rank x 4 nodes,
# cyclic"). exe=wrf_wtimer.exe -> binary_kind=wtimer; plov==1 -> binary_kind=plov (both RESEARCH
# per wtune_compliance's PLOV/wtimer carve-out, decided in classify_rows()). `bind` (cpu-bind
# mask) and `ctl` (elastic-core repin mode: none/static/dynamic) are not ledger columns; they are
# recorded in `notes` and fed to compliance classification as extra knobs
# (cpu_bind -> STRICT, elastic_core_repin -> STRICT per the catalog, so they do not by themselves
# force a row out of STRICT -- only plov/wtimer do).
# Rows with ok=0 (failed run, sentinel med_s_step=-1.0) are skipped: no valid timing exists.
# ---------------------------------------------------------------------------
def adapt_elastic_core(path):
    rows = []
    rep = Counter()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["ok"] != "1":
                continue  # failed run (P13 s1_readiness, rc=137): no valid s_per_step
            arm = r["arm"]
            rep[arm] += 1
            binary_kind = "plov" if r["plov"] == "1" else ("wtimer" if "wtimer" in r["exe"] else "pristine")
            extra = {}
            if r["bind"] and r["bind"] != "default":
                extra["cpu_bind"] = r["bind"]
            if r["ctl"] and r["ctl"] != "none":
                extra["elastic_core_repin"] = r["ctl"]
            diff = _f(r["diffwrf_differing_fields"])
            fidelity = None
            if r["diffwrf_differing_fields"] == "0":
                fidelity = "bit_identical"
            elif diff is not None and diff > 0:
                fidelity = "differs"
            row = new_row(
                run_id="elasticcore:%s:%s:%d" % (r["job"], arm, rep[arm]),
                source_dataset=os.path.basename(path),
                job_id=r["job"], rep_index=rep[arm] - 1,
                case_raw="%s/%s" % (r["stage"], arm), case_base="karina",
                case_family=case_family_of("karina"), hardware_id=HARDWARE_ID,
                binary_kind=binary_kind, compliance_class=None,
                e_we=901.0, e_sn=901.0, e_vert=45.0, dx=3000.0, dy=3000.0, time_step=15.0,
                radt=15.0, cudt=None, history_interval=None, io_form_history=None,
                n_nodes=4, n_ranks=224, omp_threads=_i(r["omp_default"]), nproc_x=None, nproc_y=None,
                numtiles=_i(r["numtiles"]), distribution="cyclic", kmp_blocktime=None,
                s_per_step_median=_f(r["med_s_step"]), s_per_step_mean=_f(r["mean_skip1_s_step"]),
                n_steps_measured=_i(r["steps"]),
                diffwrf_differing_fields=diff, fidelity_label=fidelity,
                profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
                profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
                profile_thread_imbalance=None,
                notes="stage=%s plan=%s bind=%s ctl=%s rel_to_v=%s; physics filled from "
                      "2026-09-07-wtune-config-cost.csv karina row" %
                      (r["stage"], r["plan"] or "-", r["bind"], r["ctl"], r["rel_to_v"]),
                **_KARINA_PHYSICS,
            )
            row[_EXTRA_KNOB_KEY] = extra
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Adapter 4: data/wrf/2026-08-26-wrf-decomposition-sweep.csv (8 rows)
# knowledge/2026-08-26-wrf-8x56-decomposition-zero-code-win.md +
# knowledge/wrf/2026-09-09-numtiles-16-is-a-1501x1201-constant-...md
#
# This is the SC26-WRF "CONUS" 1501x1201 grid (this repo has no other name for it than the task
# brief's own "(CONUS)" label); e_vert=50 is the one directly stated fact ("wtune-numtiles-a-8x56-
# nt{8,16}-20260902 的 namelist 讀回: e_we=1501, e_vert=50"). numtiles and physics are not given in
# this CSV or note -> left null. median_s_240step is preferred over median_s_60step_mean when
# present (longer run, closer to a real coldstart); n_steps_measured is set accordingly and the
# unused column is kept in `notes`.
# ---------------------------------------------------------------------------
def adapt_decomposition_sweep(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            m240, m60 = _f(r["median_s_240step"]), _f(r["median_s_60step_mean"])
            s_per_step = m240 if m240 is not None else m60
            n_steps = 240 if m240 is not None else 60
            tag = "%sx%s" % (r["nproc_x"], r["nproc_y"])
            row = new_row(
                run_id="conusdecomp:%s" % tag,
                source_dataset=os.path.basename(path),
                job_id=None, rep_index=0,
                case_raw="conus_%s" % tag, case_base="conus", case_family=case_family_of("conus"),
                hardware_id=HARDWARE_ID,
                binary_kind="pristine", compliance_class=None,
                e_we=1501.0, e_sn=1201.0, e_vert=50.0, dx=None, dy=None, time_step=None,
                mp_physics=None, cu_physics=None, bl_pbl_physics=None, sf_sfclay_physics=None,
                sf_surface_physics=None, ra_lw_physics=None, ra_sw_physics=None, radt=None,
                cudt=None, history_interval=None, io_form_history=None,
                n_nodes=4, n_ranks=_i(r["ranks"]), omp_threads=1,
                nproc_x=_i(r["nproc_x"]), nproc_y=_i(r["nproc_y"]), numtiles=None,
                distribution=None, kmp_blocktime=None,
                s_per_step_median=s_per_step, s_per_step_mean=None, n_steps_measured=n_steps,
                diffwrf_differing_fields=None, fidelity_label=None,
                profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
                profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
                profile_thread_imbalance=None,
                notes="%s; vs_auto_60step_pct=%s; both step counts: 240step=%s 60step_mean=%s" %
                      (r["note"], r["vs_auto_60step_pct"], r["median_s_240step"],
                       r["median_s_60step_mean"]),
            )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Adapter 5: data/wrf/2026-09-02-wtune-numtiles-8x56-4n.csv (2 rows)
# knowledge/experiments/2026-09-02-wtune-numtiles-16-banks-1.7pct.md
#
# Same CONUS 1501x1201/e_vert=50 case/job family (job 1053092/1053093 are exactly the two arms
# this note describes). `grid` column "8x56" is parsed into nproc_x/nproc_y.
# ---------------------------------------------------------------------------
def adapt_numtiles_8x56_4n(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            nx, ny = r["grid"].split("x")
            row = new_row(
                run_id="numtiles8x56:%s" % r["arm"],
                source_dataset=os.path.basename(path),
                job_id=r["job_id"], rep_index=0,
                case_raw=r["arm"], case_base="conus", case_family=case_family_of("conus"),
                hardware_id=HARDWARE_ID,
                binary_kind="pristine", compliance_class=None,
                e_we=1501.0, e_sn=1201.0, e_vert=50.0, dx=None, dy=None, time_step=None,
                mp_physics=None, cu_physics=None, bl_pbl_physics=None, sf_sfclay_physics=None,
                sf_surface_physics=None, ra_lw_physics=None, ra_sw_physics=None, radt=None,
                cudt=None, history_interval=None, io_form_history=None,
                n_nodes=_i(r["nodes"]), n_ranks=_i(r["ranks"]), omp_threads=1,
                nproc_x=_i(nx), nproc_y=_i(ny), numtiles=_i(r["numtiles"]),
                distribution=None, kmp_blocktime=None,
                s_per_step_median=_f(r["solver_s_per_step_median"]), s_per_step_mean=None,
                n_steps_measured=_i(r["steps"]),
                diffwrf_differing_fields=None, fidelity_label=None,
                profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
                profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
                profile_thread_imbalance=None,
                notes="tile_cells=%s partition=%s wrf_status=%s rel_to_nt8=%s" %
                      (r["tile_cells"], r["partition"], r["wrf_status"], r["rel_to_nt8"]),
            )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Adapter 6: data/wrf/2026-09-02-wtune-8n-scaling-validation.csv (2 rows)
# knowledge/experiments/2026-09-02-wtune-8n-scaling-validation.md, which links (and is linked by)
# the numtiles-16 CONUS note -> same case_base="conus" family, e_vert=50.
# ---------------------------------------------------------------------------
def adapt_8n_scaling(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            nx, ny = r["grid"].split("x")
            row = new_row(
                run_id="8nscaling:%s" % r["arm"],
                source_dataset=os.path.basename(path),
                job_id=r["job_id"], rep_index=0,
                case_raw=r["arm"], case_base="conus", case_family=case_family_of("conus"),
                hardware_id=HARDWARE_ID,
                binary_kind="pristine", compliance_class=None,
                e_we=1501.0, e_sn=1201.0, e_vert=50.0, dx=None, dy=None, time_step=None,
                mp_physics=None, cu_physics=None, bl_pbl_physics=None, sf_sfclay_physics=None,
                sf_surface_physics=None, ra_lw_physics=None, ra_sw_physics=None, radt=None,
                cudt=None, history_interval=None, io_form_history=None,
                n_nodes=_i(r["nodes"]), n_ranks=_i(r["ranks"]), omp_threads=1,
                nproc_x=_i(nx), nproc_y=_i(ny), numtiles=_i(r["numtiles"]),
                distribution=None, kmp_blocktime=None,
                s_per_step_median=_f(r["solver_s_per_step_median"]), s_per_step_mean=None,
                n_steps_measured=_i(r["steps"]),
                diffwrf_differing_fields=None, fidelity_label=None,
                profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
                profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
                profile_thread_imbalance=None,
                notes="tasks_per_node=%s partition=%s run_mode=%s backend=%s wrf_status=%s" %
                      (r["tasks_per_node"], r["partition"], r["run_mode"], r["backend"],
                       r["wrf_status"]),
            )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Adapter 7: data/wrf/2026-09-03-wtune-12h-segmented-runtime-summary.csv (10 rows)
# knowledge/experiments/2026-09-03-wtune-12h-segmented-transfers-across-schemes-and-across-the-
# terminator.md ("3km/901^2/448-rank(8x56)/12h/dt15" is the shared fixed config across all 10
# cases in this file).
#
# ms_per_step is loop_wall_s*1000/n_steps -- verified by hand (arctic_sbl: 2685.5*1000/2880 =
# 932.47, matches the column). That is a run-wide MEAN, not a median; no median is available in
# this summary file, so both s_per_step_median and s_per_step_mean are set to it (flagged in
# `notes`). dt column overrides the nominal 15s time_step for marine_cao (dt=6, per the note:
# "marine_cao 需要 dt=6 ... 才穩定").
# ---------------------------------------------------------------------------
def adapt_12h_segmented(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            case_base = case_base_of(r["case"])
            s_mean = _f(r["ms_per_step"]) / 1000.0 if r["ms_per_step"] else None
            row = new_row(
                run_id="seg12h:%s" % r["case"],
                source_dataset=os.path.basename(path),
                job_id=None, rep_index=0,
                case_raw=r["case"], case_base=case_base, case_family=case_family_of(case_base),
                hardware_id=HARDWARE_ID,
                binary_kind="pristine", compliance_class=None,
                e_we=901.0, e_sn=901.0, e_vert=_f(r["e_vert"]), dx=3000.0, dy=3000.0,
                time_step=_f(r["dt"]),
                mp_physics=_i(r["mp_physics"]), cu_physics=None, bl_pbl_physics=_i(r["bl_pbl_physics"]),
                sf_sfclay_physics=None, sf_surface_physics=None, ra_lw_physics=_i(r["ra_lw_physics"]),
                ra_sw_physics=None, radt=None, cudt=None, history_interval=None, io_form_history=None,
                n_nodes=4, n_ranks=448, omp_threads=1, nproc_x=8, nproc_y=56, numtiles=None,
                distribution=None, kmp_blocktime=None,
                s_per_step_median=s_mean, s_per_step_mean=s_mean, n_steps_measured=_i(r["n_steps"]),
                diffwrf_differing_fields=None, fidelity_label=None,
                profile_micro_share=None, profile_radiation_share=None, profile_dynamics_share=None,
                profile_mpi_share=None, profile_io_share=None, profile_rank_imbalance=None,
                profile_thread_imbalance=None,
                notes="s_per_step is loop_wall_s/n_steps (run-wide MEAN, no median available in this "
                      "summary file); e_we/e_sn/nproc from 2026-09-03 12h-segmented note "
                      "('3km/901^2/448-rank(8x56)')",
            )
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Adapter 8: data/wtune/2026-09-14-predictive-plov-r0-r3-node-steps-1074192.csv -- SKIPPED.
# knowledge/experiments/2026-09-14-predictive-plov-uses-cold-slack-...md
# This file is per-(arm, step, node) PLOV internal telemetry (max_ms/spread_ms/owner_wait_ms/...);
# it has no run-level s_per_step field to aggregate into one ledger row without inventing an
# aggregation the source note does not itself report, so per the brief's instruction ("otherwise
# skip with a note") it is not adapted. RESEARCH-only anyway (PLOV).
# ---------------------------------------------------------------------------
def adapt_predictive_plov(path):
    with open(path, newline="") as fh:
        n = sum(1 for _ in fh) - 1
    print("SKIP %s: %d rows are per-(arm,step,node) PLOV telemetry with no run-level "
          "s_per_step field; not adapted (RESEARCH-only, see 2026-09-14 predictive-plov note)."
          % (os.path.basename(path), n))
    return []


# ---------------------------------------------------------------------------
# campaign/strict_campaign.sbatch ledgers (ledger_<case>_<job>.csv) are already written in
# LEDGER_FIELDS order, with hardware_id taken from the node pool (f1-spr-icpnp on cf448,
# f1-spr-icpnq on ct448). Only type conversion is needed. Failed / timed-out runs (empty
# s_per_step_median) are counted and skipped: they carry no timing to rank.
# ---------------------------------------------------------------------------
_CAMPAIGN_INT_FIELDS = (
    "rep_index", "e_we", "e_sn", "e_vert", "mp_physics", "cu_physics", "bl_pbl_physics",
    "sf_sfclay_physics", "sf_surface_physics", "ra_lw_physics", "ra_sw_physics", "radt", "cudt",
    "io_form_history", "n_nodes", "n_ranks", "omp_threads", "nproc_x", "nproc_y", "numtiles",
    "n_steps_measured", "diffwrf_differing_fields",
    # kmp_blocktime: reuses _i's existing "non-numeric string -> None" behavior so a CSV cell of
    # "default" or "" both normalize to None, matching recommend.py's in-memory candidate rows
    # (None means unset/"default", else an int) -- otherwise features.config_key tuples built from
    # a campaign ledger row ("0", a str) and an in-memory candidate row (0, an int) compare unequal
    # for what is otherwise the identical configuration.
    "kmp_blocktime",
)
_CAMPAIGN_FLOAT_FIELDS = (
    "dx", "dy", "time_step", "history_interval", "s_per_step_median", "s_per_step_mean",
    "profile_micro_share", "profile_radiation_share", "profile_dynamics_share",
    "profile_mpi_share", "profile_io_share", "profile_rank_imbalance", "profile_thread_imbalance",
)


# The MEASURED fidelity error rides in the `notes` column, not in a column of its own: the
# 48-field LEDGER_FIELDS header (= strict_campaign.sbatch's LEDGER_HEADER) is depended on by every
# campaign ledger CSV already written and by the predictor/campaign tests, so adding
# fidelity_nd_max/fidelity_floor_ratio_max as real columns would invalidate all of them. `notes`
# already carries space-separated key=value tokens (mode=, template=, edits=, decomp=, ...), and
# strict_campaign.sbatch MODE=physics writes the two fidelity numbers there via
# configs.fidelity_notes_tokens(). This turns them back into numeric row fields.
#
# Only `fidelity_label` (the ordinal WITHIN-FLOOR/SMALL-DRIFT/LARGE-DRIFT verdict) has a real
# column; the numbers behind it did not, which is why the fidelity models could never rank two
# candidates that shared a label. predictor/physics.py already read fidelity_nd_max straight out
# of the raw notes string; these fields give every consumer the parsed value without re-splitting.
FIDELITY_NOTE_FIELDS = ("fidelity_floor_ratio_max", "fidelity_nd_max")


def parse_fidelity_notes(notes):
    """{'fidelity_floor_ratio_max': float|None, 'fidelity_nd_max': float|None} from a ledger
    `notes` string. Both default to None when the token is absent (every pre-physics-campaign
    row, and every STRICT/RESEARCH row), when it is present but empty (`fidelity_nd_max=`, which
    is what a MODE=physics run with no fidelity JSON writes), and when it is the gate's own "NA"
    placeholder (no field carried a numeric value for it) -- _f() maps all three to None, so a
    consumer never has to distinguish "not measured" from "measured as nothing"."""
    out = {k: None for k in FIDELITY_NOTE_FIELDS}
    if not notes:
        return out
    for tok in str(notes).split():
        key, sep, value = tok.partition("=")
        if sep and key in out:
            out[key] = _f(value)
    return out


def adapt_campaign_ledger(path):
    rows, failed, correctness = [], 0, 0
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            notes = r.get("notes") or ""
            is_physics = "mode=physics" in notes
            # A physics-campaign row with no s_per_step_median is a FAILED run (status=FAILED in
            # notes), not a timing gap -- predictor/physics.py needs it to count INFEASIBLE
            # candidates, so it is kept (with s_per_step_median=None) rather than dropped here.
            # Every other row (STRICT/RESEARCH, or a physics row with real timing) keeps the
            # existing drop-if-no-timing behaviour unchanged.
            if _f(r.get("s_per_step_median")) is None and not is_physics:
                failed += 1
                continue
            # correctness-mode rows (history=end_d01, C02/C06/C13 only) are diffwrf evidence, not
            # timing replicates: counting them made C02/C06/C13 the modal config, i.e. the baseline.
            if "mode=correctness" in notes:
                correctness += 1
                continue
            row = new_row(**{k: (r.get(k) or None) for k in LEDGER_FIELDS})
            for k in _CAMPAIGN_INT_FIELDS:
                row[k] = _i(r.get(k))
            for k in _CAMPAIGN_FLOAT_FIELDS:
                row[k] = _f(r.get(k))
            row["case_base"] = row["case_base"] or case_base_of(row["case_raw"])
            # Extra keys beyond LEDGER_FIELDS are safe: build()'s DictWriter uses
            # extrasaction="ignore", so these never widen the written CSV.
            row.update(parse_fidelity_notes(notes))
            rows.append(row)
    if failed:
        print("NOTE %s: skipped %d failed/timed-out run(s) with no s_per_step_median"
              % (os.path.basename(path), failed))
    if correctness:
        print("NOTE %s: skipped %d correctness-mode row(s) (diffwrf evidence, not timing)"
              % (os.path.basename(path), correctness))
    return rows


ADAPTERS = [
    ("wtune/2026-09-07-wtune-config-cost.csv", adapt_config_cost),
    ("wtune/2026-09-14-numtiles-12-13-14-abab-pristine-karina-1074183.csv", adapt_numtiles_12_13_14),
    ("wtune/2026-09-13-elastic-core-wrf-score-arms.csv", adapt_elastic_core),
    ("wrf/2026-08-26-wrf-decomposition-sweep.csv", adapt_decomposition_sweep),
    ("wrf/2026-09-02-wtune-numtiles-8x56-4n.csv", adapt_numtiles_8x56_4n),
    ("wrf/2026-09-02-wtune-8n-scaling-validation.csv", adapt_8n_scaling),
    ("wrf/2026-09-03-wtune-12h-segmented-runtime-summary.csv", adapt_12h_segmented),
    ("wtune/2026-09-14-predictive-plov-r0-r3-node-steps-1074192.csv", adapt_predictive_plov),
]


# ---------------------------------------------------------------------------
# Compliance classification
#
# "A run row is STRICT if every knob that differs from its case baseline is STRICT, and
# PHYSICS-GRAY if any differing knob is physics." PLOV/wtimer/--wrap binaries are always RESEARCH
# (overrides everything else), per the brief and per wtune_compliance.py's own RESEARCH carve-out
# for plov_*/link_wrap/ld_preload_shim.
#
# "Case baseline" here is computed per (case_base, n_ranks, omp_threads) group -- i.e. among runs
# that are actually the same execution shape -- as the most frequent full config-knob tuple in
# that group, ties broken by smallest s_per_step_median (mirrors the targets.py case-baseline
# rule, scoped tighter so grid-incomparable groups, e.g. 448x1 vs 224x2 karina, never mix).
# ---------------------------------------------------------------------------
def _classify(name):
    if wtune_compliance is not None:
        return wtune_compliance.classify(name)
    return "STRICT"  # degrades safely only if the repo import ever fails; ledger stays usable


def _config_tuple(row):
    return tuple(row.get(f) for f in KNOB_CATALOG_NAME) + tuple(sorted(row[_EXTRA_KNOB_KEY].items()))


def classify_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        # hardware_id + source_dataset: a campaign's case physics must not be judged against the
        # modal physics of a different dataset/hardware for the same case_base (2026-09-14: 26
        # cf448 karina campaign rows were labelled PHYSICS-GRAY against the old icpnq karina data).
        groups[(row["case_base"], row["hardware_id"], row["source_dataset"],
                row["n_ranks"], row["omp_threads"])].append(row)

    for key, grp in groups.items():
        counts = Counter(_config_tuple(r) for r in grp)
        best_count = max(counts.values())
        candidates = [t for t, c in counts.items() if c == best_count]

        def _tiebreak(t):
            rows_t = [r for r in grp if _config_tuple(r) == t]
            steps = [r["s_per_step_median"] for r in rows_t if r["s_per_step_median"] is not None]
            return min(steps) if steps else float("inf")

        baseline = min(candidates, key=_tiebreak)
        knob_names = list(KNOB_CATALOG_NAME)

        for row in grp:
            if row["binary_kind"] in ("plov", "wtimer"):
                row["compliance_class"] = "RESEARCH"
                continue
            # physics-campaign rows (mode=physics in notes) are never re-derived here: their
            # namelist edits (swint_opt, PBL/sfclay pair swaps, aer_opt, ...) are not all
            # KNOB_CATALOG_NAME/LEDGER_FIELDS columns, so this knob-diff-from-baseline scan would
            # silently miss e.g. a swint_opt-only template and misclassify it STRICT. The campaign
            # itself already writes the correct compliance_class (STRICT for P00/twin,
            # PHYSICS-GRAY for Pxx candidates) -- keep it as-is.
            if "mode=physics" in (row.get("notes") or ""):
                continue
            tup = _config_tuple(row)
            worst = "STRICT"
            for name, val in zip(knob_names, tup[: len(knob_names)]):
                base_val = baseline[knob_names.index(name)]
                if val != base_val and val is not None:
                    cls = _classify(KNOB_CATALOG_NAME[name])
                    if _RANK.get(cls, 0) > _RANK[worst]:
                        worst = cls
            for extra_name in row[_EXTRA_KNOB_KEY]:
                cls = _classify(extra_name)
                if _RANK.get(cls, 0) > _RANK[worst]:
                    worst = cls
            row["compliance_class"] = worst
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# 2026-09-14 user decision: these four cases are removed from training (disk data deleted too);
# lowell_cat4 maps to case_base "lowell" via case_base_of's _cat4 suffix strip.
EXCLUDED_CASE_BASES = frozenset({"deep_conv_ot", "arctic_sbl", "edouard_landfall", "lowell"})


def build(wiki_data_dir, out_path, campaign_ledgers=()):
    all_rows = []
    per_source = []
    sources = [(rel, os.path.join(wiki_data_dir, rel), fn) for rel, fn in ADAPTERS]
    sources += [(os.path.basename(p), p, adapt_campaign_ledger) for p in campaign_ledgers]
    for rel, path, fn in sources:
        if not os.path.exists(path):
            print("MISSING %s -- skipped" % path)
            continue
        rows = fn(path)
        dropped = [r for r in rows if r["case_base"] in EXCLUDED_CASE_BASES]
        if dropped:
            print("EXCLUDE %s: dropped %d row(s) of removed training cases %s"
                  % (rel, len(dropped), sorted({r["case_base"] for r in dropped})))
            rows = [r for r in rows if r["case_base"] not in EXCLUDED_CASE_BASES]
        per_source.append((rel, len(rows)))
        all_rows.extend(rows)

    classify_rows(all_rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in LEDGER_FIELDS})

    print("\n== per-source row counts ==")
    for rel, n in per_source:
        print("  %-70s %d" % (rel, n))
    print("  %-70s %d" % ("TOTAL", len(all_rows)))

    print("\n== per case_base: distinct configs / replicates, by compliance class ==")
    by_case = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        by_case[row["case_base"]][row["compliance_class"]].append(row)
    for case in sorted(by_case):
        parts = []
        for cls in ("STRICT", "PHYSICS-GRAY", "RESEARCH"):
            rs = by_case[case].get(cls, [])
            if not rs:
                continue
            n_config = len({_config_tuple(r) for r in rs})
            parts.append("%s: %d rows / %d configs" % (cls, len(rs), n_config))
        print("  %-20s %s" % (case, "; ".join(parts)))

    return all_rows


def validate(ledger_path):
    errs = []
    with open(ledger_path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in LEDGER_FIELDS if f not in reader.fieldnames]
        if missing:
            errs.append("missing columns: %s" % missing)
        n = 0
        for i, r in enumerate(reader):
            n += 1
            sps = _f(r.get("s_per_step_median"))
            if sps is None or sps <= 0:
                errs.append("row %d (%s): s_per_step_median must be > 0, got %r"
                            % (i, r.get("run_id"), r.get("s_per_step_median")))
            if not (r.get("case_base") or "").strip():
                errs.append("row %d (%s): case_base is empty" % (i, r.get("run_id")))
    if errs:
        print("INVALID: %d error(s)" % len(errs))
        for e in errs[:50]:
            print("  " + e)
        return False
    print("VALID: %d rows" % n)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--wiki-data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--campaign-ledger", nargs="*", default=[],
                   help="campaign ledger_<case>_<job>.csv files (already in LEDGER_FIELDS schema)")

    v = sub.add_parser("validate")
    v.add_argument("--ledger", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(args.wiki_data, args.out, args.campaign_ledger)
        return 0
    if args.cmd == "validate":
        return 0 if validate(args.ledger) else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
