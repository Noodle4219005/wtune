#!/usr/bin/env python3
"""wtune_preflight.py -- pre-submission WRF run_dir sanity checker (`wtune.py check`).

A real WRF user runs this BEFORE `sbatch`. Every check here is motivated by a real incident
that silently wasted a job in this project (see each check's docstring for the specific job/
directory). The unifying failure mode this file exists to catch: WRF either aborts with a
message that does not point at the actual mistake, or -- worse -- hangs forever with NO
message at all, burning the full walltime allocation for nothing.

Severities (see SEVERITY_ORDER below for the exit-code mapping):
    FATAL  WRF will abort at/near startup with an error message.
    HANG   WRF will hang with NO error message -- the worst outcome: the job sits at 100%
           node occupancy producing nothing until the walltime kills it.
    WARN   WRF will run to completion, but the result or the wall time is not what the user
           thinks (wrong window, silently ignored setting, slow first step, tight disk).
    INFO   Informational / SKIPPED (a check could not run, e.g. netCDF4 unavailable).

Exit code: 0 if no FATAL/HANG/WARN finding, 1 if the worst finding is WARN, 2 if there is any
FATAL or HANG finding.

Dependency policy: the core of this module is stdlib-only (re, os, datetime, json, argparse,
pathlib) because scc's system `/usr/bin/python3` (3.14) has neither numpy nor netCDF4 --
confirmed 2026-09-24 via `ssh scc '/usr/bin/python3 -c "import numpy"'` (ModuleNotFoundError)
and the same for netCDF4. Every netCDF-backed check (C4 aerosol fields, C6 grid-dim/time
consistency) imports netCDF4 lazily and, when it is not importable, emits an explicit
"SKIPPED (netCDF4 not available)" finding for that check -- never silence.

Reuses, rather than re-implements, this repo's existing single-sourced parsers:
    wtune_predict.parse_namelist / g / g_dom / get_max_dom / get_domain_geom / get_run_seconds
        -- the list/per-domain-aware namelist reader and WRF's own broadcast-last-value rule.
    wtune_predict._split_assignments -- reused (not re-implemented) so "which lines does the
        real parser treat as assignments" and "which lines does C2 flag as out-of-block" are
        answered by the exact same regex, not two regexes that can silently drift apart.
    wtune_runlength.resolve_run_seconds -- WRF's run_days/hours/minutes/seconds-vs-start/end
        fallback rule (see that module's docstring for the lafe/isc21 evidence).
    wtune_physics.resolve_suite -- physics_suite sentinel resolution (-1 -> the suite's actual
        scheme numbers), needed because real templates like atlas-validate/conus25 set
        physics_suite='conus' with mp_physics=ra_lw_physics=...=-1 and the *_physics IDs C3
        needs to look up a scheme's required data files are only visible after this resolves.
    predictor.decomp_space.rsl_patch_extents -- WRF's own RSL-lite remainder-at-both-ends patch
        split (external/RSL_LITE/task_for_point.c), the same rule WRF's own startup legality
        gate (share/module_check_a_mundo.F:3014-3029, floor(extent/nproc) >= 10) divides.

Ground-truth verification performed 2026-09-24 against the real WRF 4.x tree on scc
(/nfs/home/nkli/wrf-stack/wrf/nvhpc-fast-adios2-dmsm/WRF): run/README.physics_files (per-scheme
data-file table), Registry/Registry.EM_COMMON (ghg_input default=1, reasonable_time_step_ratio,
use_adaptive_time_step, nocolons, sst_update, use_aero_icbc, all namelist,physics/domains/
time_control scalar-vs-per-domain-array shape), dyn_em/start_em.F:288-311 (the REAL time-step
FATAL gate: domain 1 ONLY, first trip, real-data cases only, non-adaptive-timestep only) and
phys/module_physics_init.F:1010-1021 + phys/module_ra_clWRF_support.F:100-204 (the REAL
CAMtr_volume_mixing_ratio gate: ghg_input==1 and NOT idealized, for EVERY non-CAM lw scheme --
CAM has its own built-in fallback table and is the one scheme exempted, model=="CAM" branch).
Two corrections to the originally assumed design came directly out of that verification; both
are called out at the top of the C8 and C3 docstrings below.

C3's per-scheme table (check_c3_physics_files docstring lists the rule families) follows
README.physics_files except where the WRF 4.8 source's own OPEN statements are narrower (checked
2026-09-27 against a local V4.8.0 tree): the gate for each such case (Ferrier, Thompson CCN,
P3 table 2, GFDL co2tf, ozone/aerosol, urban LCZ, fast SBM) is cited next to its table entry, so
a staged run_dir that WRF would accept is never reported FATAL.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import wtune_predict as wp  # parse_namelist / g / g_dom / get_max_dom / get_domain_geom / ...
import wtune_physics as wphys  # resolve_suite (physics_suite sentinel resolution)
import wtune_runlength as runlen  # resolve_run_seconds

try:
    from predictor.decomp_space import rsl_patch_extents, MIN_PATCH
except Exception:  # pragma: no cover -- exercised only if predictor/ is not shipped alongside
    MIN_PATCH = 10

    def rsl_patch_extents(extent, nproc):
        q, rem = divmod(extent, nproc)
        return q, (q + 1 if rem else q)

try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    netCDF4 = None
    HAVE_NETCDF4 = False


FATAL = "FATAL"
HANG = "HANG"
WARN = "WARN"
INFO = "INFO"

# exit-code precedence, worst first
_SEVERITY_RANK = {HANG: 3, FATAL: 3, WARN: 1, INFO: 0}


class Finding:
    __slots__ = ("severity", "id", "message", "evidence", "fix")

    def __init__(self, severity, id, message, evidence="", fix=""):
        self.severity = severity
        self.id = id
        self.message = message
        self.evidence = evidence
        self.fix = fix

    def format(self):
        line = "%s %s %s" % (self.severity, self.id, self.message)
        if self.evidence:
            line += "\n    evidence: %s" % self.evidence
        if self.fix:
            line += "\n    fix: %s" % self.fix
        return line

    def to_dict(self):
        return {
            "severity": self.severity,
            "id": self.id,
            "message": self.message,
            "evidence": self.evidence,
            "fix": self.fix,
        }


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def _domain_datetime(nl, prefix, dom_index):
    """Per-domain <prefix>_year/month/day/hour/minute/second -> datetime, or None when
    year/month/day are not resolvable for this domain (WRF's own broadcast-last-value rule via
    g_dom -- a namelist that gives fewer values than max_dom reuses the last one, same as WRF).
    minute/second default to 0 (many real templates, e.g. lafe's, omit start_second/end_second
    when it is 0). hour/minute/second may legitimately overflow their clock range (WRF accepts
    e.g. end_minute = 120 in idealized cases) -- normalised via timedelta, never via %."""
    y = wp.g_dom(nl, "time_control", prefix + "_year", dom_index, None)
    mo = wp.g_dom(nl, "time_control", prefix + "_month", dom_index, None)
    d = wp.g_dom(nl, "time_control", prefix + "_day", dom_index, None)
    if y is None or mo is None or d is None:
        return None
    h = wp.g_dom(nl, "time_control", prefix + "_hour", dom_index, 0) or 0
    mi = wp.g_dom(nl, "time_control", prefix + "_minute", dom_index, 0) or 0
    s = wp.g_dom(nl, "time_control", prefix + "_second", dom_index, 0) or 0
    try:
        return (datetime.datetime(int(y), int(mo), int(d))
                + datetime.timedelta(hours=int(h), minutes=int(mi), seconds=int(s)))
    except (TypeError, ValueError):
        return None


def find_out_of_block_assignments(namelist_path):
    """Every 'key = value' assignment that sits OUTSIDE every &group.../ block, i.e. the exact
    complement of what wtune_predict.parse_namelist actually captures (its own comment: "else:
    stray line with no assignment and no prior key -- ignore."). Uses the SAME group-tracking
    regexes as parse_namelist and the SAME wp._split_assignments() key-detection the real parser
    uses inside a block, so "assignment" means the same thing in both places.

    Real incident: /nfs/data/atlas-b/elevconv2022/namelist.input line 39,
    ' iofields_filename = \\'prates_out.txt\\'' sits between &time_control's closing '/' and the
    next '&domains' -- Fortran's namelist read silently ignores it; the FIRST iofields_filename
    (line 31, storm_diags_d01.txt) is the one WRF actually uses.

    Returns [(lineno, key_lower, raw_line_text), ...].
    """
    text = Path(namelist_path).read_text()
    text = re.sub(r"!.*$", "", text, flags=re.M)
    out = []
    group = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"&([A-Za-z_]\w*)", line)
        if m:
            group = m.group(1).lower()
            continue
        if line.startswith("/"):
            group = None
            continue
        if group is not None:
            continue
        for key, _raw_value in wp._split_assignments(line):
            out.append((lineno, key.lower(), line))
    return out


def _resolved_physics(nl, dom_index):
    """physics_suite-resolved *_physics scheme IDs for one domain (wtune_physics.resolve_suite
    fills -1 sentinels from the suite table -- needed because real templates such as
    atlas-validate/conus25 set physics_suite='conus' with every *_physics=-1)."""
    case = {
        "physics_suite": wp.g(nl, "physics", "physics_suite", None),
        "mp_physics": wp.g_dom(nl, "physics", "mp_physics", dom_index, 0),
        "ra_lw_physics": wp.g_dom(nl, "physics", "ra_lw_physics", dom_index, 0),
        "ra_sw_physics": wp.g_dom(nl, "physics", "ra_sw_physics", dom_index, 0),
        "bl_pbl_physics": wp.g_dom(nl, "physics", "bl_pbl_physics", dom_index, 0),
        "cu_physics": wp.g_dom(nl, "physics", "cu_physics", dom_index, 0),
        "sf_surface_physics": wp.g_dom(nl, "physics", "sf_surface_physics", dom_index, 0),
        "sf_sfclay_physics": wp.g_dom(nl, "physics", "sf_sfclay_physics", dom_index, 0),
    }
    resolved, _filled = wphys.resolve_suite(case)
    return resolved


def _resolve_path(run_dir, fname):
    p = Path(fname)
    return p if p.is_absolute() else Path(run_dir) / p


def _fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return "%.1f%s" % (n, unit)
        n /= 1024.0
    return "%.1fPiB" % n


# --------------------------------------------------------------------------
# C1 -- per-domain time window (HANG/WARN/FATAL)
# --------------------------------------------------------------------------

def check_c1_time_windows(nl, findings):
    """d01's own run window comes from run_days/hours/minutes/seconds when their sum is > 0,
    else from d01's own start_date -> end_date span (wtune_runlength.resolve_run_seconds, WRF's
    own documented fallback). Every NESTED domain (2..max_dom) uses its OWN end_date regardless
    of what governs d01 -- run_days/hours/minutes/seconds NEVER extends a nest.

    Real incidents:
      - atlas-b/yagi2024 job 1038 (rebuild/wrf/namelist.input): run_days=5 sets d01's window to
        2024-09-03T12:00 -> 2024-09-08T12:00, but d02's own end_* (end_minute=30, all other end_*
        unset/0) is 2024-09-03T12:30 -- a smoke-test value that survived into the "truth" run.
        d02 silently freezes 30 minutes in while d01 keeps integrating for 5 days.
      - atlas-validate/maria1km: d02 has no end_minute at all (defaults to 0), so it stops on
        the hour while d01's window (from run_*) extends further.
    """
    max_dom = wp.get_max_dom(nl)
    d01_start = _domain_datetime(nl, "start", 0)
    if d01_start is None:
        findings.append(Finding(
            INFO, "C1",
            "d01 start_year/month/day not resolvable; cannot evaluate per-domain time windows",
        ))
        return

    days = wp.g(nl, "time_control", "run_days", 0) or 0
    hours = wp.g(nl, "time_control", "run_hours", 0) or 0
    minutes = wp.g(nl, "time_control", "run_minutes", 0) or 0
    seconds = wp.g(nl, "time_control", "run_seconds", 0) or 0
    total = (days or 0) * 86400 + (hours or 0) * 3600 + (minutes or 0) * 60 + (seconds or 0)

    if total > 0:
        d01_end = d01_start + datetime.timedelta(seconds=total)
        d01_source = "run_days/run_hours/run_minutes/run_seconds"
    else:
        d01_end = _domain_datetime(nl, "end", 0)
        d01_source = "start_date -> end_date span (run_* all 0/absent)"
        if d01_end is None:
            findings.append(Finding(
                FATAL, "C1",
                "d01 has no run_days/run_hours/run_minutes/run_seconds AND no resolvable "
                "end_date -- WRF has no run length to use",
            ))
            return

    if d01_end <= d01_start:
        findings.append(Finding(
            FATAL, "C1",
            "d01 end (%s) <= d01 start (%s) via %s" % (d01_end, d01_start, d01_source),
            evidence="start=%s end=%s" % (d01_start, d01_end),
            fix="fix &time_control's run_days/hours/minutes/seconds or end_year/month/day/"
                "hour/minute/second for domain 1",
        ))
        return

    for d in range(1, max_dom):
        start_d = _domain_datetime(nl, "start", d)
        end_d = _domain_datetime(nl, "end", d)
        if start_d is None or end_d is None:
            findings.append(Finding(
                INFO, "C1",
                "domain %d start/end not resolvable; skipped" % (d + 1),
            ))
            continue
        if end_d <= start_d:
            findings.append(Finding(
                FATAL, "C1",
                "nest d%02d end (%s) <= its own start (%s)" % (d + 1, end_d, start_d),
                evidence="start_*[%d]=%s end_*[%d]=%s" % (d, start_d, d, end_d),
                fix="fix end_year/month/day/hour/minute/second for domain %d" % (d + 1),
            ))
        elif end_d < d01_end:
            findings.append(Finding(
                WARN, "C1",
                "nest d%02d stops at %s (d01 runs to %s via %s)"
                % (d + 1, end_d, d01_end, d01_source),
                evidence="d%02d end_*=%s; d01 end=%s" % (d + 1, end_d, d01_end),
                fix="set end_year/month/day/hour/minute/second for domain %d to match or "
                    "exceed d01's window, or accept that the nest freezes at %s"
                    % (d + 1, end_d),
            ))


# --------------------------------------------------------------------------
# C2 -- iofields_filename (HANG) + out-of-block assignments (WARN)
# --------------------------------------------------------------------------

def check_c2_iofields(nl, run_dir, namelist_path, findings):
    """Each domain's iofields_filename (when not 'NONE_SPECIFIED') must exist in run_dir, or
    WRF 4.8 hangs FOREVER at init with no further output -- NOT a clean abort -- even with
    ignore_iofields_warning=.true..

    Real incident: /nfs/data/atlas-b/elevconv2022, 72 AND 576 ranks: log shows only 'Problem
    opening storm_diags_d01.txt' (frame/module_domain.F) and then nothing ever again; the job
    sat at full node occupancy until walltime. An empty file at that path is a legal iofields
    file (zero extra fields) and avoids the hang entirely.

    Also flags any out-of-block 'key = value' assignment (find_out_of_block_assignments) --
    elevconv2022's OWN namelist has a second iofields_filename line sitting between
    &time_control's '/' and '&domains' that Fortran's namelist read silently ignores; the first
    (in-block) iofields_filename line is the one that actually takes effect.
    """
    max_dom = wp.get_max_dom(nl)
    for d in range(max_dom):
        fname = wp.g_dom(nl, "time_control", "iofields_filename", d, None)
        if not fname or not isinstance(fname, str):
            continue
        if fname.strip().upper() in ("NONE_SPECIFIED", ""):
            continue
        full = _resolve_path(run_dir, fname)
        if not full.exists():
            findings.append(Finding(
                HANG, "C2",
                "iofields_filename=%r for domain %d does not exist" % (fname, d + 1),
                evidence=str(full),
                fix="create an empty file at %s (WRF 4.8 hangs at init instead of erroring "
                    "when this is missing, even with ignore_iofields_warning=.true. -- "
                    "elevconv2022 incident) or set iofields_filename='NONE_SPECIFIED'" % full,
            ))

    for lineno, key, raw_line in find_out_of_block_assignments(namelist_path):
        findings.append(Finding(
            WARN, "C2",
            "%s:%d: '%s' sits outside every &group.../ block and is SILENTLY IGNORED by "
            "Fortran's namelist read" % (Path(namelist_path).name, lineno, key),
            evidence=raw_line,
            fix="move this assignment inside the correct &group block",
        ))


# --------------------------------------------------------------------------
# C3 -- data files required by the chosen physics (FATAL, Thompson WARN-only)
# --------------------------------------------------------------------------

# scheme id -> required run_dir files, verified 2026-09-24 against
# /nfs/home/nkli/wrf-stack/wrf/nvhpc-fast-adios2-dmsm/WRF/run/README.physics_files
_RA_LW_FILES = {
    1: ["RRTM_DATA"],                              # RRTM
    3: ["CAM_ABS_DATA", "CAM_AEROPT_DATA"],        # CAM
    4: ["RRTMG_LW_DATA"],                          # RRTMG
    14: ["RRTMG_LW_DATA"],                         # RRTMG-KIAPS (module_ra_rrtmg_lwk.F opens it)
    24: ["RRTMG_LW_DATA"],                         # RRTMG fast (module_ra_rrtmg_lwf.F opens it)
}
_RA_SW_FILES = {
    3: ["CAM_ABS_DATA", "CAM_AEROPT_DATA"],        # CAM
    4: ["RRTMG_SW_DATA"],                          # RRTMG
    14: ["RRTMG_SW_DATA"],                         # RRTMG-KIAPS
    24: ["RRTMG_SW_DATA"],                         # RRTMG fast
}
# Files needed when EITHER ra_lw_physics or ra_sw_physics selects the scheme -- checked once per
# domain (not once per lw/sw) so lw=sw=5 does not double-report. Goddard: init_goddardrad is
# called from both the lw and sw branches of ra_init (phys/module_physics_init.F).
_RA_EITHER_FILES = {
    5: ["BROADBAND_CLOUD_GODDARD.bin"],            # Goddard
}
# GFDL (99): gfdletainit (lw or sw) reads co2 transmission tables, but WHICH ones depends on
# co2tf (Registry default 1): module_ra_gfdleta.F:141-145 -- co2tf==1 -> CO2O3 opens
# tr49t85/tr49t67/tr67t85; otherwise CONRAD opens co2_trans. Requiring all four would
# false-FATAL a run_dir that only stages the set its co2tf actually reads.
_GFDL_SCHEME = 99
_GFDL_CO2O3_FILES = ["tr49t67", "tr49t85", "tr67t85"]
_GFDL_CONRAD_FILES = ["co2_trans"]
# ozone*.formatted: read by oznini on domain 1 only (module_physics_init.F:2330, EM_CORE) when
# o3input==2 (Registry default 2) -- and share/module_check_a_mundo.F forces o3input=0 unless
# domain-1 ra_lw or ra_sw is an RRTMG-family scheme (4, 14, 24). CAM's own camradinit
# (module_ra_cam.F:1019) calls oznini UNCONDITIONALLY, independent of o3input.
_RRTMG_FAMILY = {4, 14, 24}
_CAM_SCHEME = 3
_OZONE_FILES = ["ozone.formatted", "ozone_lat.formatted", "ozone_plev.formatted"]
# aerosol*.formatted: read ONLY by aerosol_in (module_physics_init.F:2340-2345, 5476-5507),
# gated on aer_opt==1 (Tegen climatology) and domain 1, for any radiation scheme. Not read by
# CAM (its aerosol_init opens no *.formatted file) and not read for aer_opt!=1.
_AEROSOL_FILES = ["aerosol.formatted", "aerosol_lat.formatted", "aerosol_lon.formatted",
                  "aerosol_plev.formatted"]
_ECLIPSE_FILE = "eclipse_besselian_elements.dat"   # module_ra_eclipse.F:373-377 wrf_error_fatal

_LSM_FILES = {
    1: ["LANDUSE.TBL"],                                                     # slab (5-layer)
    2: ["LANDUSE.TBL", "VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"],       # Noah
    3: ["LANDUSE.TBL", "VEGPARM.TBL", "SOILPARM.TBL", "GENPARM.TBL"],       # RUC
    5: ["LANDUSE.TBL", "CLM_ALB_ICE_DFS_DATA", "CLM_ALB_ICE_DRC_DATA",      # CLM
        "CLM_ASM_ICE_DFS_DATA", "CLM_ASM_ICE_DRC_DATA", "CLM_DRDSDT0_DATA",
        "CLM_EXT_ICE_DFS_DATA", "CLM_EXT_ICE_DRC_DATA", "CLM_KAPPA_DATA", "CLM_TAU_DATA"],
    7: ["LANDUSE.TBL", "VEGPARM.TBL"],                                      # Pleim-Xiu
    8: ["LANDUSE.TBL"],                                                     # SSiB
}
_LSM_NAMES = {1: "slab", 2: "Noah", 3: "RUC", 5: "CLM", 7: "Pleim-Xiu", 8: "SSiB"}
# Noah-MP snicar_*.nc are only read by the SNICAR snow-albedo option -- deliberately NOT required.

# mp_physics -> required run_dir files (exact names), verified against the WRF 4.8 source's
# own OPEN statements where README.physics_files is broader than what is actually opened:
#   5/95 Ferrier: fer_hires_init / etanew open ONLY ETAMPNEW_DATA.expanded_rain
#        (module_mp_fer_hires.F:2969, module_mp_etanew.F:2266); plain ETAMPNEW_DATA and the
#        *_DBL variants are not opened by a real*4 build.
#   28/38 Thompson aerosol-aware: table_ccnAct (module_mp_thompson.F:1013-1016, 5145-5159,
#        wrf_error_fatal on open failure) runs only when is_aerosol_aware, i.e. when
#        thompson_init receives NWFA/NIFA -- mp 28 and 38, NOT plain Thompson (8).
#   32 full SBM: module_mp_full_sbm.F:2078-2712. Fast SBM (30) is deliberately absent -- its
#        4.8 reader opens SBM_input_33/*33.asc (module_mp_fast_sbm.F:5401-5646), a different
#        file set that README.physics_files does not describe.
#   55 Jensen-ISHMAEL: module_mp_jensen_ishmael.F:119-161.
_ETAMPNEW_FILES = ["ETAMPNEW_DATA.expanded_rain"]
_FULL_SBM_FILES = ["bulkdens.asc_s_0_03_0_9", "bulkradii.asc_s_0_03_0_9", "capacity.asc",
                   "coeff_p.asc", "coeff_q.asc", "constants.asc", "kernels.asc_s_0_03_0_9",
                   "kernels_z.asc", "masses.asc", "termvels.asc"]
_MP_FILES = {
    5: _ETAMPNEW_FILES,
    95: _ETAMPNEW_FILES,
    28: ["CCN_ACTIVATE.BIN"],
    38: ["CCN_ACTIVATE.BIN"],
    32: _FULL_SBM_FILES,
    55: ["ishmael-gamma-tab.bin", "ishmael-qi-qc.bin", "ishmael-qi-qr.bin"],
}
# P3: table file names carry a version suffix that drifts between WRF releases (README names
# ...-3momI_v5.1.7 / -5.3-2momI, scc's tree ships p3_lookupTable_1.dat-v5.4_2momI/_3momI and
# p3_lookupTable_2.dat-v5.3), so match by PREFIX. Table 1 is read by every P3 option; table 2
# only when nCat>1 (module_mp_p3.F:474-478), i.e. P3_2CATEGORY (52) only.
_P3_TABLE1_MP = {50, 51, 52, 53}
_P3_TABLE2_MP = {52}
_P3_TABLE1_PREFIX = "p3_lookupTable_1.dat"
_P3_TABLE2_PREFIX = "p3_lookupTable_2.dat"

# sf_urban_physics 1/2/3: urban_param_init (module_sf_urban.F:2997-3022) opens URBPARM.TBL when
# use_wudapt_lcz==0 and URBPARM_LCZ.TBL INSTEAD (not in addition) when use_wudapt_lcz==1.
_URBAN_SCHEMES = {1, 2, 3}
# chem_opt values whose chemistry reads HLC.TBL (README.physics_files).
_HLC_CHEM_OPTS = {111, 112, 114, 201, 202}
# Noah-MP (4): LANDUSE.TBL/VEGPARM.TBL/SOILPARM.TBL/GENPARM.TBL are NOT used by Noah-MP itself,
# but its own table (MPTABLE.TBL in older WRF, renamed NoahmpTable.TBL from ~4.x on -- confirmed
# on scc's WRF tree, which ships NoahmpTable.TBL and no MPTABLE.TBL) is required; accept either
# name so this doesn't false-FATAL across WRF versions.
_NOAHMP_TABLE_NAMES = ("NoahmpTable.TBL", "MPTABLE.TBL")

# ghg_input==1 (the WRF default -- Registry.EM_COMMON: 'rconfig integer ghg_input
# namelist,physics 1 1 rh') makes phys/module_physics_init.F:1010-1021 call
# read_CAMgases(...,.true.,name,...) once per domain at init for EVERY non-idealized run,
# independent of which lw scheme is active -- 'name' is only set to "CAM" when ra_lw_physics==3
# (CAMLWSCHEME); read_CAMgases's own fallback (module_ra_clWRF_support.F:187-201) uses a
# built-in table when model=="CAM" but calls wrf_error_fatal("CLWRF: 'CAMtr_volume_mixing_ratio'
# does not exist") for every OTHER model name if the file is missing. So: required for every
# ra_lw_physics != 3 when ghg_input resolves to 1 -- not just the RRTMG/RRTM subset this check
# was originally scoped to; CAM is the one scheme exempted via its own fallback table.
_CAMTR_EXEMPT_RA_LW = {3}

_THOMPSON_MP = {8, 28, 38}
_THOMPSON_DAT_FILES = ["freezeH2O.dat", "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat"]

# extension whitelist used by an earlier (buggy) staging tool -- kept here only as the
# incident's own evidence trail, not used by this checker (see --wrf-run-dir below).
# real incident: sc26p2-n4 staging used an extension whitelist and missed the extension-less
# CAMtr_volume_mixing_ratio -> FATAL 'CLWRF: CAMtr_volume_mixing_ratio does not exist'.


def _require_files(run_dir, dom_index, files, findings, note=""):
    for fname in files:
        full = _resolve_path(run_dir, fname)
        if not full.exists():
            findings.append(Finding(
                FATAL, "C3",
                "domain %d: required physics data file %s does not exist%s"
                % (dom_index + 1, fname, (" (%s)" % note) if note else ""),
                evidence=str(full),
                fix="copy/link %s from WRF's own run/ directory into the run_dir" % fname,
            ))


def _require_prefix(run_dir, dom_index, prefix, findings, note=""):
    """FATAL unless at least one run_dir entry's name starts with `prefix` (for version-suffixed
    tables such as P3's p3_lookupTable_1.dat-v5.4_2momI)."""
    try:
        names = os.listdir(run_dir)
    except OSError:
        names = []
    if not any(n.startswith(prefix) for n in names):
        findings.append(Finding(
            FATAL, "C3",
            "domain %d: required physics data file %s* does not exist%s"
            % (dom_index + 1, prefix, (" (%s)" % note) if note else ""),
            evidence=str(_resolve_path(run_dir, prefix + "*")),
            fix="copy/link %s* (the version WRF's own run/ ships) from WRF's own run/ directory "
                "into the run_dir" % prefix,
        ))


def _int_opt(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def check_c3_physics_files(nl, run_dir, findings, wrf_run_dir=None):
    """Required run_dir data files for the chosen physics, per active domain. Rule families
    (FATAL unless noted; see the table comments above for the WRF source anchor of each gate):
      radiation  ra_lw/ra_sw tables (RRTM, CAM, RRTMG 4/14/24), Goddard (5, lw or sw),
                 GFDL (99, co2tf-selected set), CAMtr_volume_mixing_ratio (ghg_input=1, non-CAM),
                 ozone*.formatted (domain 1: o3input=2 with RRTMG-family; any CAM domain),
                 aerosol*.formatted (domain 1: aer_opt=1), eclipse table (ra_sw_eclipse=1)
      surface    LSM tables (slab, Noah, RUC, CLM, PX, SSiB), Noah-MP table, URBPARM[_LCZ].TBL
                 (sf_urban_physics 1-3), wind-turbine-1.tbl (windturbine_spec=1)
      mp         Ferrier, aerosol-aware Thompson CCN, full SBM, P3 (prefix match), Jensen;
                 plain-Thompson prestaged .dat tables are WARN only
      chem       HLC.TBL for chem_opt 111/112/114/201/202
    """
    max_dom = wp.get_max_dom(nl)
    ghg_input = wp.g(nl, "physics", "ghg_input", 1)
    if ghg_input is None:
        ghg_input = 1
    o3input = _int_opt(wp.g(nl, "physics", "o3input", 2), 2)
    aer_opt = _int_opt(wp.g(nl, "physics", "aer_opt", 0), 0)
    ra_sw_eclipse = _int_opt(wp.g(nl, "physics", "ra_sw_eclipse", 0), 0)
    use_wudapt_lcz = _int_opt(wp.g(nl, "physics", "use_wudapt_lcz", 0), 0)
    windturbine_spec = _int_opt(wp.g(nl, "physics", "windturbine_spec", 0), 0)
    co2tf = _int_opt(wp.g(nl, "physics", "co2tf", 1), 1)

    # domain-1-only reads (oznini / aerosol_in are gated on id==1; eclipse is a scalar option)
    if aer_opt == 1:
        _require_files(run_dir, 0, _AEROSOL_FILES, findings,
                       note="aer_opt=1 (Tegen aerosol climatology)")
    if ra_sw_eclipse == 1:
        _require_files(run_dir, 0, [_ECLIPSE_FILE], findings, note="ra_sw_eclipse=1")
    if windturbine_spec == 1:
        _require_files(run_dir, 0, ["wind-turbine-1.tbl"], findings, note="windturbine_spec=1")

    for d in range(max_dom):
        phys = _resolved_physics(nl, d)
        ra_lw = phys.get("ra_lw_physics")
        ra_sw = phys.get("ra_sw_physics")
        mp = phys.get("mp_physics")
        sf = phys.get("sf_surface_physics")

        if ra_lw in _RA_LW_FILES:
            _require_files(run_dir, d, _RA_LW_FILES[ra_lw], findings,
                            note="ra_lw_physics=%s" % ra_lw)
        if ra_sw in _RA_SW_FILES:
            _require_files(run_dir, d, _RA_SW_FILES[ra_sw], findings,
                            note="ra_sw_physics=%s" % ra_sw)
        for scheme, files in _RA_EITHER_FILES.items():
            if scheme in (ra_lw, ra_sw):
                _require_files(run_dir, d, files, findings,
                               note="ra_lw/ra_sw_physics=%s" % scheme)
        if _GFDL_SCHEME in (ra_lw, ra_sw):
            if co2tf == 1:
                _require_files(run_dir, d, _GFDL_CO2O3_FILES, findings,
                               note="GFDL radiation (99) with co2tf=1 (default)")
            else:
                _require_files(run_dir, d, _GFDL_CONRAD_FILES, findings,
                               note="GFDL radiation (99) with co2tf=%s" % co2tf)

        if _CAM_SCHEME in (ra_lw, ra_sw):
            _require_files(run_dir, d, _OZONE_FILES, findings,
                           note="CAM radiation reads CAM ozone regardless of o3input")
        elif d == 0 and o3input == 2 and (ra_lw in _RRTMG_FAMILY or ra_sw in _RRTMG_FAMILY):
            _require_files(run_dir, d, _OZONE_FILES, findings,
                           note="o3input=2 (default) with RRTMG radiation; set o3input=0 "
                                "to use the scheme's built-in ozone profile instead")

        if int(ghg_input) == 1 and ra_lw not in _CAMTR_EXEMPT_RA_LW:
            full = _resolve_path(run_dir, "CAMtr_volume_mixing_ratio")
            if not full.exists():
                findings.append(Finding(
                    FATAL, "C3",
                    "domain %d: ghg_input=1 (default) with ra_lw_physics=%s (not CAM) requires "
                    "CAMtr_volume_mixing_ratio, which does not exist" % (d + 1, ra_lw),
                    evidence=str(full),
                    fix="link CAMtr_volume_mixing_ratio (WRF's run/ ships *.SSP245 as the "
                        "default) into the run_dir, or set ghg_input=0",
                ))

        if sf in _LSM_FILES:
            _require_files(run_dir, d, _LSM_FILES[sf], findings,
                            note="sf_surface_physics=%s (%s)" % (sf, _LSM_NAMES[sf]))
        elif sf == 4:  # Noah-MP
            if not any(_resolve_path(run_dir, n).exists() for n in _NOAHMP_TABLE_NAMES):
                findings.append(Finding(
                    FATAL, "C3",
                    "domain %d: sf_surface_physics=4 (Noah-MP) requires NoahmpTable.TBL "
                    "(or MPTABLE.TBL on older WRF), neither exists" % (d + 1),
                    evidence=str(_resolve_path(run_dir, _NOAHMP_TABLE_NAMES[0])),
                    fix="copy NoahmpTable.TBL from WRF's own run/ directory into the run_dir",
                ))

        sf_urban = _int_opt(wp.g_dom(nl, "physics", "sf_urban_physics", d, 0), 0)
        if sf_urban in _URBAN_SCHEMES:
            if use_wudapt_lcz == 1:
                _require_files(run_dir, d, ["URBPARM_LCZ.TBL"], findings,
                               note="sf_urban_physics=%s with use_wudapt_lcz=1" % sf_urban)
            else:
                _require_files(run_dir, d, ["URBPARM.TBL"], findings,
                               note="sf_urban_physics=%s" % sf_urban)

        chem_opt = _int_opt(wp.g_dom(nl, "chem", "chem_opt", d, 0), 0)
        if chem_opt in _HLC_CHEM_OPTS:
            _require_files(run_dir, d, ["HLC.TBL"], findings, note="chem_opt=%s" % chem_opt)

        if mp in _MP_FILES:
            _require_files(run_dir, d, _MP_FILES[mp], findings, note="mp_physics=%s" % mp)
        if mp in _P3_TABLE1_MP:
            _require_prefix(run_dir, d, _P3_TABLE1_PREFIX, findings,
                            note="mp_physics=%s (P3)" % mp)
        if mp in _P3_TABLE2_MP:
            _require_prefix(run_dir, d, _P3_TABLE2_PREFIX, findings,
                            note="mp_physics=%s (P3, nCat>1)" % mp)

        if mp in _THOMPSON_MP:
            missing = [f for f in _THOMPSON_DAT_FILES
                       if not _resolve_path(run_dir, f).exists()]
            if missing:
                findings.append(Finding(
                    WARN, "C3",
                    "domain %d: mp_physics=%s (Thompson) is missing prestaged lookup table(s) "
                    "%s -- WRF recomputes them at init instead of failing, costing minutes"
                    % (d + 1, mp, ", ".join(missing)),
                    evidence="checked in %s" % run_dir,
                    fix="prestage %s in the run_dir to skip the init-time table build"
                        % ", ".join(missing),
                ))

    if wrf_run_dir:
        wrf_run_dir = Path(wrf_run_dir)
        if wrf_run_dir.is_dir():
            for entry in sorted(wrf_run_dir.iterdir()):
                if not entry.is_file():
                    continue
                if not (Path(run_dir) / entry.name).exists():
                    findings.append(Finding(
                        WARN, "C3",
                        "%s is present in WRF's own run/ (%s) but absent from run_dir"
                        % (entry.name, wrf_run_dir),
                        evidence=str(entry),
                        fix="stage %s into run_dir if the active physics options need it "
                            "(real incident: sc26p2-n4 staging used an extension whitelist and "
                            "missed the extension-less CAMtr_volume_mixing_ratio)" % entry.name,
                    ))
        else:
            findings.append(Finding(
                INFO, "C3", "--wrf-run-dir %s is not a directory; skipped" % wrf_run_dir,
            ))


# --------------------------------------------------------------------------
# C4 -- aerosol climatology fields (FATAL, netCDF4 optional)
# --------------------------------------------------------------------------

def check_c4_aerosol_climatology(nl, run_dir, findings):
    """mp_physics=28 (Thompson aerosol-aware) with use_aero_icbc=.true. (or wif_input_opt>=1)
    requires QNWFA/QNIFA fields in wrfinput_d01 (wrf.exe stage) or in the met_em files
    (real.exe stage, when met_em.* are present and wrfinput does not exist yet).

    Real incident: atlas-validate/cs-lafe cold start -- real.exe: 'aerosol climatology
    field(s) missing'.
    """
    max_dom = wp.get_max_dom(nl)
    # wif_input_opt is a &domains key (Registry: 'rconfig integer wif_input_opt namelist,domains',
    # default 0); tempo_aerosolaware is &physics (default 0)
    wif_input_opt = int(wp.g(nl, "domains", "wif_input_opt", 0) or 0)
    tempo_aero = int(wp.g(nl, "physics", "tempo_aerosolaware", 0) or 0)
    mps = {_resolved_physics(nl, d).get("mp_physics") for d in range(max_dom)}
    # WTF test ralbNE (scc job 1086): real.exe FATAL 'wif_input_opt=0 but mp_physics=28, 29, or 88'
    # -- dyn_em/module_initialize_real.F ~2741-2748 (THOMPSONAERO, RCON, or TEMPO with
    # tempo_aerosolaware=1, and wif_input_opt == 0)
    aero_mp = sorted(m for m in mps if m in (28, 29) or (m == 88 and tempo_aero == 1))
    if aero_mp and wif_input_opt == 0:
        findings.append(Finding(
            FATAL, "C4",
            "mp_physics=%s needs wif_input_opt >= 1 (&domains); real.exe aborts with "
            "'wif_input_opt=0 but mp_physics=28, 29, or 88'" % ",".join(map(str, aero_mp)),
            evidence="dyn_em/module_initialize_real.F if_thompsonaero_3d; WTF ralbNE",
            fix="set wif_input_opt = 1 (and num_wif_levels = 30) in &domains, with the "
                "QNWFA_QNIFA climatology in metgrid constants_name"))
        return
    if 28 not in mps:
        return
    if not HAVE_NETCDF4:
        findings.append(Finding(INFO, "C4", "SKIPPED (netCDF4 not available)"))
        return

    use_aero_icbc = wp.g(nl, "physics", "use_aero_icbc", False)
    if not (use_aero_icbc or int(wif_input_opt) >= 1):
        return

    run_dir = Path(run_dir)
    met_em = sorted(run_dir.glob("met_em.d01.*"))
    wrfinput = run_dir / "wrfinput_d01"

    def _has_qnwfa_qnifa(path):
        try:
            ds = netCDF4.Dataset(str(path))
        except Exception as exc:
            return None, "could not open %s: %s" % (path, exc)
        try:
            names = set(ds.variables.keys())
        finally:
            ds.close()
        missing = [v for v in ("QNWFA", "QNIFA") if v not in names]
        return (len(missing) == 0), missing

    if wrfinput.exists():
        ok, missing = _has_qnwfa_qnifa(wrfinput)
        if ok is None:
            findings.append(Finding(INFO, "C4", missing))
        elif not ok:
            findings.append(Finding(
                FATAL, "C4",
                "mp_physics=28 with use_aero_icbc/wif_input_opt requires QNWFA/QNIFA in "
                "wrfinput_d01; missing %s" % missing,
                evidence=str(wrfinput),
                fix="regenerate wrfinput_d01 via real.exe with an aerosol climatology input "
                    "(cs-lafe cold-start incident: 'aerosol climatology field(s) missing')",
            ))
    elif met_em:
        ok, missing = _has_qnwfa_qnifa(met_em[0])
        if ok is None:
            findings.append(Finding(INFO, "C4", missing))
        elif not ok:
            findings.append(Finding(
                FATAL, "C4",
                "mp_physics=28 with use_aero_icbc/wif_input_opt requires QNWFA/QNIFA in "
                "met_em files (real.exe stage); missing %s in %s"
                % (missing, met_em[0].name),
                evidence=str(met_em[0]),
                fix="rerun metgrid with an aerosol climatology field, or set "
                    "use_aero_icbc=.false. if aerosol ICBC was not intended",
            ))
    else:
        findings.append(Finding(
            INFO, "C4",
            "mp_physics=28 with use_aero_icbc/wif_input_opt but neither wrfinput_d01 nor "
            "met_em.d01.* found in run_dir; cannot check QNWFA/QNIFA",
        ))


# --------------------------------------------------------------------------
# C5 -- met_em naming vs nocolons (FATAL, real.exe stage only)
# --------------------------------------------------------------------------

_MET_EM_RE = re.compile(
    r"^(met_em\.d\d\d\.\d{4}-\d\d-\d\d)_(\d\d)([_:])(\d\d)\3(\d\d)(\.nc)?$")


def check_c5_met_em_naming(nl, run_dir, findings):
    """real.exe's own filename convention follows nocolons: nocolons=.true. -> it opens
    'met_em.dNN.YYYY-MM-DD_HH_MM_SS[.nc]' (underscores); nocolons=.false./absent (WRF default)
    -> it opens the colon form 'met_em.dNN.YYYY-MM-DD_HH:MM:SS[.nc]'. Only the OTHER form
    existing is a real.exe-stage FATAL (file not found).

    Real incident: atlas-validate/cs-lafe cold start, nocolons=.true. in the namelist but only
    colon-named met_em files existed at first.
    """
    run_dir = Path(run_dir)
    met_em = sorted(p for p in run_dir.glob("met_em.*") if p.is_file())
    if not met_em:
        return

    nocolons = bool(wp.g(nl, "time_control", "nocolons", False))
    wanted_sep = "_" if nocolons else ":"
    have_seps = set()
    parsed = []
    for p in met_em:
        m = _MET_EM_RE.match(p.name)
        if not m:
            continue
        prefix, hh, sep, mm, ss, ext = m.groups()
        have_seps.add(sep)
        parsed.append((prefix, hh, mm, ss, ext or ""))

    if wanted_sep in have_seps:
        return  # the form real.exe will actually look for exists

    if not parsed:
        return  # nothing matched the expected met_em pattern at all; not this check's job

    other_sep = ":" if wanted_sep == "_" else "_"
    example_prefix, hh, mm, ss, ext = parsed[0]
    wanted_name = "%s_%s%s%s%s%s" % (
        example_prefix, hh, wanted_sep, mm, wanted_sep, ss) + ext
    findings.append(Finding(
        FATAL, "C5",
        "nocolons=%s but only %s-separated met_em filenames exist; real.exe opens "
        "'YYYY-MM-DD_HH%sMM%sSS' names and will not find them"
        % (nocolons, "colon" if other_sep == ":" else "underscore",
           wanted_sep, wanted_sep),
        evidence="e.g. %s (want %s)" % (met_em[0].name, wanted_name),
        fix="create %s-separated met_em links/copies alongside the existing ones "
            "(cs-lafe cold-start incident)" % ("underscore" if wanted_sep == "_" else "colon"),
    ))


# --------------------------------------------------------------------------
# C6 -- input consistency (FATAL/WARN, netCDF4 optional)
# --------------------------------------------------------------------------

def _open_nc(path):
    try:
        return netCDF4.Dataset(str(path)), None
    except Exception as exc:
        return None, str(exc)


def check_c6_input_consistency(nl, run_dir, findings):
    if not HAVE_NETCDF4:
        findings.append(Finding(INFO, "C6", "SKIPPED (netCDF4 not available)"))
        return

    run_dir = Path(run_dir)
    max_dom = wp.get_max_dom(nl)

    for d in range(max_dom):
        wrfinput = run_dir / ("wrfinput_d%02d" % (d + 1))
        if not wrfinput.exists():
            continue
        ds, err = _open_nc(wrfinput)
        if ds is None:
            findings.append(Finding(INFO, "C6", "could not open %s: %s" % (wrfinput, err)))
            continue
        try:
            try:
                e_we, e_sn, e_vert = wp.get_domain_geom(nl, d)
            except wp.PredictError as exc:
                findings.append(Finding(INFO, "C6", str(exc)))
                continue
            dims = {}
            for attr in ("WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
                         "BOTTOM-TOP_GRID_DIMENSION"):
                try:
                    dims[attr] = int(ds.getncattr(attr))
                except AttributeError:
                    dims[attr] = None
            expected = {
                "WEST-EAST_GRID_DIMENSION": int(e_we),
                "SOUTH-NORTH_GRID_DIMENSION": int(e_sn),
                "BOTTOM-TOP_GRID_DIMENSION": int(e_vert),
            }
            mismatches = [
                "%s: namelist=%s wrfinput=%s" % (k, expected[k], dims[k])
                for k in expected if dims[k] is not None and dims[k] != expected[k]
            ]
            if mismatches:
                findings.append(Finding(
                    FATAL, "C6",
                    "domain %d: %s grid dims do not match namelist e_we/e_sn/e_vert"
                    % (d + 1, wrfinput.name),
                    evidence="; ".join(mismatches),
                    fix="regenerate wrfinput_d%02d with real.exe using the SAME e_we/e_sn/"
                        "e_vert as this namelist, or fix the namelist to match" % (d + 1),
                ))

            try:
                sim_start = ds.getncattr("SIMULATION_START_DATE")
            except AttributeError:
                sim_start = None
            nl_start = _domain_datetime(nl, "start", d)
            if sim_start and nl_start is not None:
                nl_start_str = nl_start.strftime("%Y-%m-%d_%H:%M:%S")
                if sim_start != nl_start_str:
                    findings.append(Finding(
                        WARN, "C6",
                        "domain %d: %s SIMULATION_START_DATE=%s does not match namelist "
                        "start_date %s" % (d + 1, wrfinput.name, sim_start, nl_start_str),
                        evidence="SIMULATION_START_DATE=%s namelist=%s"
                                 % (sim_start, nl_start_str),
                        fix="regenerate wrfinput_d%02d for this namelist's start date, or fix "
                            "start_year/month/day/hour/minute/second" % (d + 1),
                    ))
        finally:
            ds.close()

    wrfbdy = run_dir / "wrfbdy_d01"
    if wrfbdy.exists():
        ds, err = _open_nc(wrfbdy)
        if ds is None:
            findings.append(Finding(INFO, "C6", "could not open %s: %s" % (wrfbdy, err)))
        else:
            try:
                if "Times" in ds.variables:
                    times_raw = ds.variables["Times"][:]
                    time_strs = []
                    for row in times_raw:
                        try:
                            time_strs.append(
                                b"".join(bytes(c) if not isinstance(c, bytes) else c
                                         for c in row).decode("ascii"))
                        except Exception:
                            pass
                    parsed_times = []
                    for s in time_strs:
                        try:
                            parsed_times.append(datetime.datetime.strptime(
                                s.strip(), "%Y-%m-%d_%H:%M:%S"))
                        except ValueError:
                            continue
                    if parsed_times:
                        last_bdy = max(parsed_times)
                        interval_s = wp.g(nl, "time_control", "interval_seconds", None)
                        d01_start = _domain_datetime(nl, "start", 0)
                        days = wp.g(nl, "time_control", "run_days", 0) or 0
                        hours = wp.g(nl, "time_control", "run_hours", 0) or 0
                        minutes = wp.g(nl, "time_control", "run_minutes", 0) or 0
                        seconds = wp.g(nl, "time_control", "run_seconds", 0) or 0
                        total = days * 86400 + hours * 3600 + minutes * 60 + seconds
                        d01_end = None
                        if d01_start is not None and total > 0:
                            d01_end = d01_start + datetime.timedelta(seconds=total)
                        elif d01_start is not None:
                            d01_end = _domain_datetime(nl, "end", 0)
                        if d01_end is not None and interval_s:
                            bdy_reach = last_bdy + datetime.timedelta(seconds=float(interval_s))
                            if bdy_reach < d01_end:
                                findings.append(Finding(
                                    FATAL, "C6",
                                    "wrfbdy_d01's last boundary time (%s) + interval_seconds "
                                    "(%ss) = %s, before d01's end (%s)"
                                    % (last_bdy, interval_s, bdy_reach, d01_end),
                                    evidence="last Times=%s interval_seconds=%s d01 end=%s"
                                             % (last_bdy, interval_s, d01_end),
                                    fix="rerun real.exe with boundary data covering the full "
                                        "d01 window (wrf_get_next_time will fail at run time "
                                        "otherwise)",
                                ))
            finally:
                ds.close()

    sst_update = wp.g(nl, "physics", "sst_update", 0) or 0
    if int(sst_update) == 1:
        for d in range(max_dom):
            wrflowinp = run_dir / ("wrflowinp_d%02d" % (d + 1))
            if not wrflowinp.exists():
                findings.append(Finding(
                    FATAL, "C6",
                    "sst_update=1 requires wrflowinp_d%02d, which does not exist" % (d + 1),
                    evidence=str(wrflowinp),
                    fix="generate wrflowinp_d%02d (real.exe with sst_update=1 inputs), or set "
                        "sst_update=0" % (d + 1),
                ))


# --------------------------------------------------------------------------
# C7 -- decomposition (FATAL)
# --------------------------------------------------------------------------

def check_c7_decomposition(nl, findings, ranks=None, nproc_x=None, nproc_y=None):
    """nproc_x*nproc_y must equal the rank count, and every domain's patch (WRF's own RSL-lite
    remainder-at-both-ends split, predictor.decomp_space.rsl_patch_extents) must be >= 10 mass
    points wide in BOTH directions on EVERY active domain (share/module_check_a_mundo.F:
    3014-3029, floor(e_we(i)/nproc_x) >= 10 using Fortran integer division).

    nproc_x/nproc_y default to the namelist's OWN &domains nproc_x/nproc_y (real templates here
    -- conus25, c34, marshall, isc21, lafe -- all set these directly); --nproc-x/--nproc-y/
    --ranks override the namelist when given (the actual sbatch resource request may differ
    from a stale namelist value).
    """
    nl_nproc_x = wp.g(nl, "domains", "nproc_x", None)
    nl_nproc_y = wp.g(nl, "domains", "nproc_y", None)
    eff_x = nproc_x if nproc_x is not None else nl_nproc_x
    eff_y = nproc_y if nproc_y is not None else nl_nproc_y

    if eff_x is None and eff_y is None and ranks is None:
        findings.append(Finding(
            INFO, "C7",
            "no nproc_x/nproc_y in the namelist and no --ranks/--nproc-x/--nproc-y given; "
            "decomposition not checked",
        ))
        return

    if eff_x is not None and eff_y is not None:
        eff_x, eff_y = int(eff_x), int(eff_y)
        implied_ranks = eff_x * eff_y
        if ranks is not None and int(ranks) != implied_ranks:
            findings.append(Finding(
                FATAL, "C7",
                "nproc_x*nproc_y (%d*%d=%d) does not equal --ranks (%d)"
                % (eff_x, eff_y, implied_ranks, ranks),
                fix="make --ranks match nproc_x*nproc_y, or change nproc_x/nproc_y",
            ))
        eff_ranks = implied_ranks
    elif ranks is not None:
        eff_ranks = int(ranks)
    else:
        findings.append(Finding(
            INFO, "C7",
            "only one of nproc_x/nproc_y is set (x=%s y=%s); cannot check patch legality"
            % (eff_x, eff_y),
        ))
        return

    if eff_x is None or eff_y is None:
        return  # ranks given but no shape to check patch sizes against

    max_dom = wp.get_max_dom(nl)
    for d in range(max_dom):
        try:
            e_we, e_sn, e_vert = wp.get_domain_geom(nl, d)
        except wp.PredictError as exc:
            findings.append(Finding(INFO, "C7", str(exc)))
            continue
        min_we, _max_we = rsl_patch_extents(int(e_we), eff_x)
        min_sn, _max_sn = rsl_patch_extents(int(e_sn), eff_y)
        if min_we < MIN_PATCH:
            findings.append(Finding(
                FATAL, "C7",
                "domain %d: nproc_x=%d gives a %d-point-wide patch in e_we=%d (< %d minimum)"
                % (d + 1, eff_x, min_we, int(e_we), MIN_PATCH),
                evidence="rsl_patch_extents(%d, %d) -> min=%d" % (int(e_we), eff_x, min_we),
                fix="reduce nproc_x or use a domain-1 shape with enough e_we per rank",
            ))
        if min_sn < MIN_PATCH:
            findings.append(Finding(
                FATAL, "C7",
                "domain %d: nproc_y=%d gives a %d-point-wide patch in e_sn=%d (< %d minimum)"
                % (d + 1, eff_y, min_sn, int(e_sn), MIN_PATCH),
                evidence="rsl_patch_extents(%d, %d) -> min=%d" % (int(e_sn), eff_y, min_sn),
                fix="reduce nproc_y or use a domain-1 shape with enough e_sn per rank",
            ))


# --------------------------------------------------------------------------
# C8 -- time step vs grid spacing (FATAL for d01, WARN for nests)
# --------------------------------------------------------------------------

def check_c8_time_step(nl, findings):
    """CORRECTED from the originally assumed design after reading dyn_em/start_em.F:280-311:
    WRF's OWN fatal gate (`IF (dt_s/dx_km > config_flags%reasonable_time_step_ratio)
    wrf_error_fatal('--- ERROR: Time step too large')`) checks ONLY domain 1 -- gated on
    `grid%id==1`, `first_trip_for_this_domain`, `config_flags%map_proj /= 0` (real-data grids
    only, never idealized) and `.NOT. use_adaptive_time_step`. It is NOT re-checked per nest
    with parent_time_step_ratio the way the request assumed. This checker reports the REAL
    FATAL for d01, and separately reports nests exceeding the same ratio (via their own
    effective dt/dx, derived through the parent_id/parent_time_step_ratio/parent_grid_ratio
    chain) as WARN, since WRF will not catch those at startup -- they can go numerically
    unstable mid-run instead of failing cleanly at init.

    reasonable_time_step_ratio itself is a real namelist,domains knob (default 6.0, confirmed
    via Registry.EM_COMMON) -- read from the namelist rather than hardcoded, because real
    templates override it (/nfs/data/atlas-b/elevconv2022/namelist.input sets it to 6.5).
    Equality is allowed (the check is strictly '>').
    """
    max_dom = wp.get_max_dom(nl)
    use_adaptive = bool(wp.g(nl, "domains", "use_adaptive_time_step", False))
    if use_adaptive:
        findings.append(Finding(
            INFO, "C8", "SKIPPED: use_adaptive_time_step=.true. (WRF does not apply the "
                        "fixed-ratio check)",
        ))
        return

    rrs = wp.g(nl, "domains", "reasonable_time_step_ratio", 6.0)
    if rrs is None:
        rrs = 6.0
    rrs = float(rrs)

    def _dt_seconds(dom_index):
        ts = wp.g_dom(nl, "domains", "time_step", dom_index, None)
        if ts is None:
            return None
        num = wp.g_dom(nl, "domains", "time_step_fract_num", dom_index, 0) or 0
        den = wp.g_dom(nl, "domains", "time_step_fract_den", dom_index, 1) or 1
        return float(ts) + float(num) / float(den)

    dt0 = _dt_seconds(0)
    dx0 = wp.g_dom(nl, "domains", "dx", 0, None)
    dy0 = wp.g_dom(nl, "domains", "dy", 0, None)
    if dt0 is None or dx0 is None or dy0 is None:
        findings.append(Finding(
            INFO, "C8", "domain 1 time_step/dx/dy not resolvable; C8 skipped",
        ))
        return

    dt_by_domain = [None] * max_dom
    dx_by_domain = [None] * max_dom
    dt_by_domain[0] = dt0
    dx_by_domain[0] = min(float(dx0), float(dy0))

    dx0_km = dx_by_domain[0] / 1000.0
    ratio0 = dt_by_domain[0] / dx0_km
    if ratio0 > rrs:
        findings.append(Finding(
            FATAL, "C8",
            "d01: time_step/dx ratio %.3f exceeds reasonable_time_step_ratio %.3f"
            % (ratio0, rrs),
            evidence="time_step=%.3fs dx=min(%s,%s)m=%.1fkm ratio=%.3f"
                     % (dt0, dx0, dy0, dx0_km, ratio0),
            fix="reduce time_step (dyn_em/start_em.F:305 aborts with 'Time step too large' "
                "at this ratio) or raise reasonable_time_step_ratio if you have verified "
                "stability",
        ))

    for d in range(1, max_dom):
        pid_raw = wp.g_dom(nl, "domains", "parent_id", d, d)
        pid = int(pid_raw) - 1 if pid_raw else d - 1
        if pid < 0 or pid >= d or dt_by_domain[pid] is None:
            pid = d - 1
        ptsr = wp.g_dom(nl, "domains", "parent_time_step_ratio", d, 1) or 1
        pgr = wp.g_dom(nl, "domains", "parent_grid_ratio", d, 1) or 1
        dt_by_domain[d] = dt_by_domain[pid] / float(ptsr)
        dx_by_domain[d] = dx_by_domain[pid] / float(pgr)
        dx_km = dx_by_domain[d] / 1000.0
        if dx_km <= 0:
            continue
        ratio = dt_by_domain[d] / dx_km
        if ratio > rrs:
            findings.append(Finding(
                WARN, "C8",
                "d%02d: effective time_step/dx ratio %.3f (via parent_time_step_ratio/"
                "parent_grid_ratio) exceeds reasonable_time_step_ratio %.3f -- WRF's own "
                "startup gate only checks d01 (dyn_em/start_em.F:288, grid%%id==1), so this "
                "will NOT abort at init and may go numerically unstable mid-run instead"
                % (d + 1, ratio, rrs),
                evidence="dt=%.3fs dx=%.1fkm ratio=%.3f" % (dt_by_domain[d], dx_km, ratio),
                fix="raise parent_time_step_ratio for domain %d or reduce d01's time_step"
                    % (d + 1),
            ))


# --------------------------------------------------------------------------
# C9 -- disk (WARN)
# --------------------------------------------------------------------------

_ASSUMED_3D_FIELDS = 40           # conservative: typical wrfout has ~40 full 3-D fields
_BYTES_PER_VALUE = 4              # io_form 2 (NetCDF) default REAL is 4 bytes


def _history_interval_minutes(nl, dom_index):
    v = wp.g_dom(nl, "time_control", "history_interval", dom_index, None)
    if v is not None:
        return float(v)
    v = wp.g_dom(nl, "time_control", "history_interval_m", dom_index, None)
    if v is not None:
        return float(v)
    v = wp.g_dom(nl, "time_control", "history_interval_h", dom_index, None)
    if v is not None:
        return float(v) * 60.0
    v = wp.g_dom(nl, "time_control", "history_interval_s", dom_index, None)
    if v is not None:
        return float(v) / 60.0
    return None


def check_c9_disk(nl, run_dir, findings):
    """Conservative estimate: bytes_per_frame = e_we*e_sn*e_vert*4 bytes*~40 3-D fields, summed
    over every active domain's history frames (floor(run_minutes/history_interval)+1, the same
    frame-count rule wtune_predict.py uses), compared against os.statvfs(run_dir) free space."""
    try:
        run_s = wp.get_run_seconds(nl)
    except wp.PredictError as exc:
        findings.append(Finding(INFO, "C9", "run length not resolvable: %s" % exc))
        return
    run_min = run_s / 60.0

    max_dom = wp.get_max_dom(nl)
    total_bytes = 0
    detail = []
    for d in range(max_dom):
        try:
            e_we, e_sn, e_vert = wp.get_domain_geom(nl, d)
        except wp.PredictError:
            continue
        hist_min = _history_interval_minutes(nl, d)
        if not hist_min or hist_min <= 0:
            continue
        frames = int(run_min // hist_min) + 1
        bytes_per_frame = e_we * e_sn * e_vert * _BYTES_PER_VALUE * _ASSUMED_3D_FIELDS
        domain_bytes = bytes_per_frame * frames
        total_bytes += domain_bytes
        detail.append("d%02d: %d frames x %s = %s"
                       % (d + 1, frames, _fmt_bytes(bytes_per_frame), _fmt_bytes(domain_bytes)))

    if total_bytes <= 0:
        return

    try:
        st = os.statvfs(str(run_dir))
    except (OSError, AttributeError) as exc:
        findings.append(Finding(INFO, "C9", "could not statvfs run_dir: %s" % exc))
        return
    free_bytes = st.f_bavail * st.f_frsize

    if total_bytes > free_bytes:
        findings.append(Finding(
            WARN, "C9",
            "estimated history output ~%s exceeds free space ~%s on run_dir's filesystem"
            % (_fmt_bytes(total_bytes), _fmt_bytes(free_bytes)),
            evidence="; ".join(detail),
            fix="reduce history_interval/frames_per_outfile, use iofields_filename to shrink "
                "the field set, or point run_dir at a filesystem with more free space",
        ))


# --------------------------------------------------------------------------
# C10 -- wrfinput physical sanity (FATAL, netCDF4 optional)
# --------------------------------------------------------------------------

# Range bounds. Masking differs per field (see check_c10_wrfinput_sanity's docstring for the
# scc truth-dir evidence behind each rule):
#   TSLB  land points only; exactly-0 values are real.exe's "undefined" fill -> WARN, non-zero
#         values outside the bounds -> FATAL (the tw3km-2031/case_I garbage signature).
#   SST   water points only; exactly-0 values skipped (inland lakes carry SST=0 in healthy runs).
#   SMOIS land points only; negative -> WARN, > 1 -> FATAL.
#   TSK   every point, FATAL (unchanged).
_C10_TSLB_RANGE = (200.0, 350.0)
_C10_SST_RANGE = (180.0, 350.0)
_C10_SMOIS_RANGE = (0.0, 1.0)
_C10_TSK_RANGE = (180.0, 350.0)
# TSLB non-zero out-of-range values are FATAL when they EXCEED this FRACTION of the domain's land
# TSLB values (land columns x soil levels), WARN when 0 < fraction <= it. Pre-registered 0.1%:
# dorian2019 d02 20/(14806x4) = 0.034% ran 1 h on scc (job 1392) with no NaN -> WARN;
# tw3km-2031/case_I 13186/(37588x4) = 8.8% went NaN by hour 1 -> FATAL.
_C10_TSLB_NONZERO_FATAL_FRACTION = 0.001
_C10_NAN_VARS = ("T", "U", "V", "QVAPOR", "TSLB", "SMOIS", "PSFC")


def _c10_filled(var):
    """netCDF4 Variable -> float64 ndarray, with any masked entries (fill values) turned into
    NaN rather than left as a sentinel like -1e30 that would otherwise fail the range check for
    the wrong reason and be indistinguishable from a real physical excursion."""
    import numpy as np
    raw = var[:]
    if np.ma.isMaskedArray(raw):
        return raw.filled(np.nan).astype("float64")
    return np.asarray(raw, dtype="float64")


def _c10_bcast(mask, arr):
    """Broadcast a (Time, sn, we) point mask to `arr` (inserting the level axis for 4-D soil
    fields); None means 'no mask available' -> every point."""
    import numpy as np
    if mask is None:
        return np.ones(arr.shape, dtype=bool)
    if mask.ndim < arr.ndim:
        mask = mask[:, np.newaxis, ...]
    return np.broadcast_to(mask, arr.shape)


def _c10_land_water(ds):
    """(land, water) boolean point masks. land = LANDMASK==1 and, when present, XLAND==1;
    water = LANDMASK==0. Either is None when LANDMASK is absent (then every point is checked,
    which is conservative)."""
    if "LANDMASK" not in ds.variables:
        return None, None
    lm = _c10_filled(ds.variables["LANDMASK"])
    land = lm > 0.5
    if "XLAND" in ds.variables:
        xl = _c10_filled(ds.variables["XLAND"])
        if xl.shape == lm.shape:
            land = land & (abs(xl - 1.0) < 0.5)
    return land, lm < 0.5


def _c10_range_finding(findings, severity, d, varname, wname, arr, bad, lo, hi, kind, where):
    import numpy as np
    idx = tuple(int(i) for i in np.argwhere(bad)[0])
    vals = arr[bad]
    findings.append(Finding(
        severity, "C10",
        "domain %d: %s in %s has %d %svalue(s) outside [%.0f, %.0f]%s (min=%.3f max=%.3f)"
        % (d + 1, varname, wname, int(bad.sum()), kind, lo, hi, where,
           float(vals.min()), float(vals.max())),
        evidence="first bad index=%s value=%.3f" % (idx, float(arr[idx])),
        fix="regenerate wrfinput_d%02d -- a mismatched LANDSEA mask between a fill source and "
            "the primary forcing can blend water-point soil state into land points "
            "(tw3km-2031/case_I incident)" % (d + 1),
    ))


def check_c10_wrfinput_sanity(nl, run_dir, findings):
    """wrfinput_d0N physical-plausibility gate, run BEFORE real.exe/wrf.exe ever start.

    Real incident: scc tw3km-2031 case_I (AIFS generator, analog-soil fill path). AIFS's
    regridded LANDSEA is fractional (~33% of points land/sea-mixed after interpolation) while
    the analog GFS's soil fields are binary land(=1)/water(=0)-masked; metgrid's LANDSEA-masked
    horizontal interpolation of the filled soil field then blended water-point state (TSLB/SMOIS
    effectively undefined over open ocean) into land points, producing ST000010 (TSLB level 1)
    around -23 C and SM000010 (SMOIS level 1) around -0.02 in wrfinput_d01. Neither real.exe nor
    wrf.exe checks this: wrf.exe went entirely NaN by simulation hour 1 (wrfout_d01 valid at
    2031-08-20_01:00:00 all-NaN in U/V/T/PSFC) while its own log still printed 'SUCCESS COMPLETE
    WRF' at the end of the 12 h run. This check catches it at `wtune check` time, before any
    node-hour is spent.

    Rules ("land" = LANDMASK==1 and, when present, XLAND==1; "water" = LANDMASK==0):
      TSLB  land only. Exactly 0 -> one WARN per domain (undefined soil temperature); non-zero
            outside [200, 350] K -> FATAL when they exceed _C10_TSLB_NONZERO_FATAL_FRACTION
            (0.1%) of the domain's land TSLB values (land columns x soil levels), else WARN.
      SST   water only, exactly-0 values skipped; non-zero outside [180, 350] K -> FATAL.
      SMOIS land only. < 0 -> WARN; > 1 -> FATAL.
      TSK   every point outside [180, 350] K -> FATAL.
    Any non-finite (NaN/Inf) value anywhere in T, U, V, QVAPOR, TSLB, SMOIS, or PSFC is FATAL
    regardless of range, land or water.

    Calibration (2026-09-27, scc truth run_dirs that ran vs case_I): healthy wrfinputs carry
    TSLB exactly 0 on water and on some land columns (dorian2019 d02: 350 land columns), SST
    exactly 0 on land and on some water points (inland lakes: zhengzhou 4663, dorian 16498), and
    small negative deep-level SMOIS on land (zhengzhou: 1923 values in [-0.012, 0)). case_I, which
    went NaN, instead had 13186 NON-ZERO garbage land TSLB values (-12.5, -10.9, 170.6 ...), none
    exactly 0 -- 13186/(37588x4) = 8.8% of its land values -> FATAL. dorian2019 d02 also carries
    20 non-zero out-of-range land TSLB values (min 101.1 K), but 20/(14806x4) = 0.034%, and its
    1 h scc run (job 1392) had no NaN at 01:00 in d02 -> WARN.
    """
    if not HAVE_NETCDF4:
        findings.append(Finding(INFO, "C10", "SKIPPED (netCDF4 not available)"))
        return
    try:
        import numpy as np
    except ImportError:
        findings.append(Finding(INFO, "C10", "SKIPPED (numpy not available)"))
        return

    run_dir = Path(run_dir)
    max_dom = wp.get_max_dom(nl)
    checked_any = False
    for d in range(max_dom):
        wrfinput = run_dir / ("wrfinput_d%02d" % (d + 1))
        if not wrfinput.exists():
            continue
        checked_any = True
        ds, err = _open_nc(wrfinput)
        if ds is None:
            findings.append(Finding(INFO, "C10", "could not open %s: %s" % (wrfinput, err)))
            continue
        try:
            for varname in _C10_NAN_VARS:
                if varname not in ds.variables:
                    continue
                arr = _c10_filled(ds.variables[varname])
                bad = ~np.isfinite(arr)
                if bad.any():
                    idx = tuple(int(i) for i in np.argwhere(bad)[0])
                    findings.append(Finding(
                        FATAL, "C10",
                        "domain %d: %s in %s has %d non-finite (NaN/Inf) value(s)"
                        % (d + 1, varname, wrfinput.name, int(bad.sum())),
                        evidence="first bad index=%s" % (idx,),
                        fix="regenerate wrfinput_d%02d -- real.exe's interpolation already "
                            "broke; wrf.exe propagates this silently and may still print "
                            "SUCCESS COMPLETE" % (d + 1),
                    ))

            land, water = _c10_land_water(ds)
            wname = wrfinput.name

            if "TSLB" in ds.variables:
                arr = _c10_filled(ds.variables["TSLB"])
                on_land = np.isfinite(arr) & _c10_bcast(land, arr)
                zero = on_land & (arr == 0.0)
                if zero.any():
                    cols = int(np.any(zero, axis=1).sum()) if arr.ndim == 4 else int(zero.sum())
                    findings.append(Finding(
                        WARN, "C10",
                        "domain %d: %d land points with undefined (0) soil temperature "
                        "(TSLB in %s)" % (d + 1, cols, wname),
                        evidence="%d TSLB value(s) exactly 0 on land points (LANDMASK==1%s)"
                                 % (int(zero.sum()),
                                    " and XLAND==1" if "XLAND" in ds.variables else ""),
                        fix="usually benign (real.exe leaves TSLB=0 where the source soil field "
                            "is undefined, e.g. coast/lake edges); regenerate wrfinput_d%02d if "
                            "the count is large relative to the land area" % (d + 1),
                    ))
                lo, hi = _C10_TSLB_RANGE
                bad = on_land & (arr != 0.0) & ((arr < lo) | (arr > hi))
                n_bad = int(bad.sum())
                if n_bad:
                    n_land = int(_c10_bcast(land, arr).sum())
                    frac = n_bad / float(n_land) if n_land else 1.0
                    sev = FATAL if frac > _C10_TSLB_NONZERO_FATAL_FRACTION else WARN
                    _c10_range_finding(findings, sev, d, "TSLB", wname, arr, bad, lo, hi,
                                       "non-zero ",
                                       " (land points; %.3f%% of %d land values, FATAL above "
                                       "%.1f%%)" % (100.0 * frac, n_land,
                                                    100.0 * _C10_TSLB_NONZERO_FATAL_FRACTION))

            if "SST" in ds.variables:
                arr = _c10_filled(ds.variables["SST"])
                lo, hi = _C10_SST_RANGE
                bad = (np.isfinite(arr) & _c10_bcast(water, arr) & (arr != 0.0)
                       & ((arr < lo) | (arr > hi)))
                if bad.any():
                    _c10_range_finding(findings, FATAL, d, "SST", wname, arr, bad, lo, hi,
                                       "non-zero ", " (water points)")

            if "SMOIS" in ds.variables:
                arr = _c10_filled(ds.variables["SMOIS"])
                lo, hi = _C10_SMOIS_RANGE
                on_land = np.isfinite(arr) & _c10_bcast(land, arr)
                neg = on_land & (arr < lo)
                if neg.any():
                    _c10_range_finding(findings, WARN, d, "SMOIS", wname, arr, neg, lo, hi,
                                       "", " (land points, negative)")
                high = on_land & (arr > hi)
                if high.any():
                    _c10_range_finding(findings, FATAL, d, "SMOIS", wname, arr, high, lo, hi,
                                       "", " (land points)")

            if "TSK" in ds.variables:
                arr = _c10_filled(ds.variables["TSK"])
                lo, hi = _C10_TSK_RANGE
                bad = np.isfinite(arr) & ((arr < lo) | (arr > hi))
                if bad.any():
                    _c10_range_finding(findings, FATAL, d, "TSK", wname, arr, bad, lo, hi, "", "")
        finally:
            ds.close()
    if not checked_any:
        findings.append(Finding(INFO, "C10", "no wrfinput_d0N found in run_dir; skipped"))


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def run_checks(nl, run_dir, namelist_path, ranks=None, nproc_x=None, nproc_y=None,
                wrf_run_dir=None):
    findings = []
    checks = [
        ("C1", lambda: check_c1_time_windows(nl, findings)),
        ("C2", lambda: check_c2_iofields(nl, run_dir, namelist_path, findings)),
        ("C3", lambda: check_c3_physics_files(nl, run_dir, findings, wrf_run_dir=wrf_run_dir)),
        ("C4", lambda: check_c4_aerosol_climatology(nl, run_dir, findings)),
        ("C5", lambda: check_c5_met_em_naming(nl, run_dir, findings)),
        ("C6", lambda: check_c6_input_consistency(nl, run_dir, findings)),
        ("C7", lambda: check_c7_decomposition(nl, findings, ranks=ranks, nproc_x=nproc_x,
                                               nproc_y=nproc_y)),
        ("C8", lambda: check_c8_time_step(nl, findings)),
        ("C9", lambda: check_c9_disk(nl, run_dir, findings)),
        ("C10", lambda: check_c10_wrfinput_sanity(nl, run_dir, findings)),
    ]
    for cid, fn in checks:
        try:
            fn()
        except Exception as exc:  # a checker bug must never take down the other checkers
            findings.append(Finding(
                INFO, cid, "internal error in %s: %r" % (cid, exc),
            ))
    return findings


def exit_code_for(findings):
    worst = max((_SEVERITY_RANK.get(f.severity, 0) for f in findings), default=0)
    if worst >= 3:
        return 2
    if worst >= 1:
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="wtune.py check",
        description="pre-submission WRF run_dir sanity checker",
    )
    parser.add_argument("run_dir", help="directory WRF will be run from")
    parser.add_argument("--namelist", default=None,
                         help="path to namelist.input (default: <run_dir>/namelist.input)")
    parser.add_argument("--ranks", type=int, default=None, help="total MPI ranks")
    parser.add_argument("--nproc-x", type=int, default=None, dest="nproc_x",
                         help="override namelist's &domains nproc_x")
    parser.add_argument("--nproc-y", type=int, default=None, dest="nproc_y",
                         help="override namelist's &domains nproc_y")
    parser.add_argument("--wrf-run-dir", default=None, dest="wrf_run_dir",
                         help="WRF's own run/ directory, to list files present there but "
                              "missing from run_dir (C3)")
    parser.add_argument("--json", default=None, help="also write findings as JSON to this path")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    namelist_path = Path(args.namelist) if args.namelist else run_dir / "namelist.input"

    if not namelist_path.exists():
        print("FATAL C0 namelist not found: %s" % namelist_path)
        return 2

    try:
        nl = wp.parse_namelist(str(namelist_path))
    except Exception as exc:
        print("FATAL C0 failed to parse %s: %r" % (namelist_path, exc))
        return 2

    findings = run_checks(
        nl, run_dir, namelist_path,
        ranks=args.ranks, nproc_x=args.nproc_x, nproc_y=args.nproc_y,
        wrf_run_dir=args.wrf_run_dir,
    )

    for f in findings:
        print(f.format())
    if not findings:
        print("(no findings)")

    if args.json:
        Path(args.json).write_text(json.dumps([f.to_dict() for f in findings], indent=2))

    # WRF extensions (chem/hydro/WRFDA/moving nest): support lines, unsupported families, build
    # mismatch (wtune_extensions.py); no-op for plain ARW.
    import wtune_extensions
    ext_lines, ext_code = wtune_extensions.check_report(
        str(namelist_path), str(run_dir), ranks=args.ranks, nproc_x=args.nproc_x, nproc_y=args.nproc_y)
    for line in ext_lines:
        print(line)

    return max(exit_code_for(findings), ext_code)


if __name__ == "__main__":
    sys.exit(main())
