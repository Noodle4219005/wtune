#!/usr/bin/env python3
"""wtune/preflight.py -- pre-submission WRF run_dir sanity checker (`wtune/cli.py check`).

Each check C1-C11 stems from an incident where WRF aborted misleadingly or hung (cited in its
docstring). Severities: FATAL, HANG (silent), WARN, INFO; exit 0 / 1 (worst WARN) / 2 (FATAL or
HANG). Stdlib core (scc's /usr/bin/python3 lacks numpy/netCDF4): netCDF checks import lazily and
report SKIPPED. Gates verified against WRF 4.x (2026-09-24) and 4.8 sources (2026-09-27)."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from wtune import predict as wp  # parse_namelist / g / g_dom / get_max_dom / get_domain_geom / ...
from wtune import physics as wphys  # resolve_suite (physics_suite sentinel resolution)

try:
    from wtune.predictor.decomp_space import rsl_patch_extents, MIN_PATCH
except Exception:  # pragma: no cover -- exercised only if predictor/ is not shipped alongside
    MIN_PATCH = 10

    def rsl_patch_extents(extent: int, nproc: int) -> tuple[int, int]:
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

    def format(self) -> str:
        line = "%s %s %s" % (self.severity, self.id, self.message)
        if self.evidence:
            line += "\n    evidence: %s" % self.evidence
        if self.fix:
            line += "\n    fix: %s" % self.fix
        return line

    def to_dict(self) -> dict[str, str]:
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


def find_out_of_block_assignments(namelist_path: str | Path) -> list[tuple[int, str, str]]:
    """'key = value' assignments outside every &group / block -> [(lineno, key_lower, raw_line)],
    the complement of what wtune.predict.parse_namelist captures (same regexes and
    _split_assignments). Fortran's namelist read ignores them (elevconv2022 line 39: a second
    iofields_filename between &time_control and &domains)."""
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
            line = line[m.end():].strip()  # same-line assignments belong to this group
        if line.startswith("/"):
            group = None
            continue
        if group is not None:
            if wp._mask_quotes(line).endswith("/"):  # trailing '/' closes the group
                group = None
            continue
        for key, _raw_value in wp._split_assignments(line):
            out.append((lineno, key.lower(), line))
    return out


def _resolved_physics(nl, dom_index):
    """physics_suite-resolved *_physics scheme IDs for one domain (wtune.physics.resolve_suite
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

def _run_seconds(nl: dict) -> int:
    """run_days/hours/minutes/seconds of &time_control as seconds (0 when all are absent/0)."""
    days = wp.g(nl, "time_control", "run_days", 0) or 0
    hours = wp.g(nl, "time_control", "run_hours", 0) or 0
    minutes = wp.g(nl, "time_control", "run_minutes", 0) or 0
    seconds = wp.g(nl, "time_control", "run_seconds", 0) or 0
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _c1_d01_end(nl: dict, d01_start: datetime.datetime, findings: list[Finding]):
    """(d01 end, where it came from), or None after a FATAL when d01 has no run length."""
    total = _run_seconds(nl)
    if total > 0:
        return (d01_start + datetime.timedelta(seconds=total),
                "run_days/run_hours/run_minutes/run_seconds")
    d01_end = _domain_datetime(nl, "end", 0)
    if d01_end is None:
        findings.append(Finding(
            FATAL, "C1",
            "d01 has no run_days/run_hours/run_minutes/run_seconds AND no resolvable "
            "end_date -- WRF has no run length to use",
        ))
        return None
    return d01_end, "start_date -> end_date span (run_* all 0/absent)"


# Registry.EM_COMMON defaults of &time_control start_*/end_* (year..second)
_C1_REGISTRY_DEFAULTS = {"start": (1993, 3, 13, 12, 0, 0), "end": (1993, 3, 14, 12, 0, 0)}


def _c1_warn_short_lists(nl: dict, d: int, findings: list[Finding]) -> None:
    """WRF does NOT broadcast start_*/end_*: an element the namelist does not give keeps its
    Registry default, while g_dom (used below) reuses the last value. WARN where they differ."""
    parts = ("year", "month", "day", "hour", "minute", "second")
    for prefix, defaults in _C1_REGISTRY_DEFAULTS.items():
        for part, default in zip(parts, defaults):
            key = "%s_%s" % (prefix, part)
            vals = nl.get("time_control", {}).get(key)
            if vals and len(vals) <= d and vals[-1] != default:
                findings.append(Finding(
                    WARN, "C1",
                    "%s has %d value(s), none for domain %d: WRF will use the Registry default "
                    "%s there (not %s), so the window computed below may be wrong"
                    % (key, len(vals), d + 1, default, vals[-1]),
                    evidence="%s = %s" % (key, ", ".join(str(v) for v in vals)),
                    fix="give %s one value per domain" % key,
                ))


def _c1_check_nest(nl: dict, d: int, d01_end: datetime.datetime, d01_source: str,
                   findings: list[Finding]) -> None:
    _c1_warn_short_lists(nl, d, findings)
    start_d = _domain_datetime(nl, "start", d)
    end_d = _domain_datetime(nl, "end", d)
    if start_d is None or end_d is None:
        findings.append(Finding(
            INFO, "C1",
            "domain %d start/end not resolvable; skipped" % (d + 1),
        ))
    elif end_d <= start_d:
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


def check_c1_time_windows(nl: dict, findings: list[Finding]) -> None:
    """d01's window comes from run_* (sum > 0) else its start -> end span; every nest uses its
    OWN end_* -- run_* never extends a nest. A nest ending before d01 freezes silently (yagi2024 job
    1038: d02 end 12:30 under run_days=5; maria1km: d02 without end_minute)."""
    max_dom = wp.get_max_dom(nl)
    d01_start = _domain_datetime(nl, "start", 0)
    if d01_start is None:
        findings.append(Finding(
            INFO, "C1",
            "d01 start_year/month/day not resolvable; cannot evaluate per-domain time windows",
        ))
        return
    window = _c1_d01_end(nl, d01_start, findings)
    if window is None:
        return
    d01_end, d01_source = window
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
        _c1_check_nest(nl, d, d01_end, d01_source, findings)


# --------------------------------------------------------------------------
# C2 -- namelist-named files (iofields, sst_update wrflowinp: FATAL) + out-of-block lines (WARN)
# --------------------------------------------------------------------------

IOFIELDS_HANG = ("WRF 4.8 hangs at input_input when the iofields file is missing, even with "
                 "ignore_iofields_warning=.true.")
IOFIELDS_EVIDENCE = ("climate-ea50 probe jobs 2541-2545 (scc, 2026-10-02: no step after "
                     "'med_initialdata_input: calling input_input', 1260 s timeout); "
                     "elevconv2022 (72 and 576 ranks)")


def _namelist_file_names(nl, key, default, dom_index):
    """Domain dom_index's (0-based) file name for namelist key `key` with WRF's '<domain>'
    placeholder resolved; None for unset/'NONE_SPECIFIED'."""
    fname = wp.g_dom(nl, "time_control", key, dom_index, default)
    if not fname or not isinstance(fname, str):
        return None
    fname = fname.strip()
    if fname.upper() in ("NONE_SPECIFIED", ""):
        return None
    return fname.replace("<domain>", "%02d" % (dom_index + 1))


def _check_referenced_file(run_dir, wrf_run_dir, key, fname, domains, findings, fatal_msg,
                           fix, evidence=""):
    """A file the namelist names. In a RUN dir (one holding wrf.exe) it must exist there:
    FATAL otherwise. In a case dir (no wrf.exe; wtune stages namelist-named files into every run
    dir it builds -- campaign/configs.stage_namelist_files): INFO when the case dir or
    --wrf-run-dir has it, FATAL when neither does."""
    doms = ",".join("%d" % (d + 1) for d in domains)
    full = _resolve_path(run_dir, fname)
    is_run_dir = os.path.lexists(os.path.join(str(run_dir), "wrf.exe"))
    if full.exists():
        if not is_run_dir:
            findings.append(Finding(
                INFO, "C2", "%s=%r (domain %s) is in the case dir; wtune stages it into every "
                "run dir it builds" % (key, fname, doms), evidence=str(full)))
        return
    if not is_run_dir and wrf_run_dir and not Path(fname).is_absolute() \
            and (Path(wrf_run_dir) / fname).exists():
        findings.append(Finding(
            INFO, "C2", "%s=%r (domain %s) is not in the case dir but in --wrf-run-dir, which "
            "every run dir links" % (key, fname, doms), evidence=str(Path(wrf_run_dir) / fname)))
        return
    where = ("the run dir" if is_run_dir else
             "the case dir" + (" nor --wrf-run-dir" if wrf_run_dir else ""))
    findings.append(Finding(
        FATAL, "C2",
        "%s=%r (domain %s) does not exist in %s: %s" % (key, fname, doms, where, fatal_msg),
        evidence=str(full) + ("; " + evidence if evidence else ""),
        fix=fix % full,
    ))


def check_c2_iofields(nl: dict, run_dir: str | Path, namelist_path: str | Path,
                      findings: list[Finding], wrf_run_dir: str | Path | None = None) -> None:
    """Files wrf.exe opens at init (_check_referenced_file: FATAL in a run dir, INFO in a case dir):
    iofields_filename per domain -- missing, WRF 4.8 HANGS at init with no output (climate-ea50
    jobs 2541-2545, elevconv2022); an empty file is legal -- and auxinput4_inname
    (wrflowinp_d<domain>) when sst_update=1. Also reports out-of-block assignments."""
    max_dom = wp.get_max_dom(nl)
    refs = {}
    for d in range(max_dom):
        fname = _namelist_file_names(nl, "iofields_filename", None, d)
        if fname:
            refs.setdefault(("iofields_filename", fname), []).append(d)
    sst_update = wp.g(nl, "physics", "sst_update", 0) or 0
    if int(sst_update) == 1:
        for d in range(max_dom):
            fname = _namelist_file_names(nl, "auxinput4_inname", "wrflowinp_d<domain>", d)
            if fname:
                refs.setdefault(("auxinput4_inname", fname), []).append(d)
    for (key, fname), doms in refs.items():
        if key == "iofields_filename":
            _check_referenced_file(
                run_dir, wrf_run_dir, key, fname, doms, findings, IOFIELDS_HANG,
                "stage the case's iofields file at %s, create an empty file there (zero extra "
                "fields), or set iofields_filename='NONE_SPECIFIED'", IOFIELDS_EVIDENCE)
        else:
            _check_referenced_file(
                run_dir, wrf_run_dir, key, fname, doms, findings,
                "sst_update=1 requires this wrflowinp (lower-boundary SST) file",
                "stage the wrflowinp file at %s (real.exe with sst_update=1 writes it), or set "
                "sst_update=0")

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

from wtune.preflight_tables import (  # noqa: F401 -- C3 physics-file tables (data only)
    _RA_LW_FILES,
    _RA_SW_FILES,
    _RA_EITHER_FILES,
    _GFDL_SCHEME,
    _GFDL_CO2O3_FILES,
    _GFDL_CONRAD_FILES,
    _RRTMG_FAMILY,
    _CAM_SCHEME,
    _OZONE_FILES,
    _AEROSOL_FILES,
    _ECLIPSE_FILE,
    _LSM_FILES,
    _LSM_NAMES,
    _ETAMPNEW_FILES,
    _FULL_SBM_FILES,
    _MP_FILES,
    _P3_TABLE1_MP,
    _P3_TABLE2_MP,
    _P3_TABLE1_PREFIX,
    _P3_TABLE2_PREFIX,
    _URBAN_SCHEMES,
    _HLC_CHEM_OPTS,
    _NOAHMP_TABLE_NAMES,
    _CAMTR_EXEMPT_RA_LW,
    _THOMPSON_MP,
    _THOMPSON_DAT_FILES,
)


def _linked_at_run_time(alt_dir, dom_index, name, findings, note=""):
    """True (and an INFO recorded) when `name` is absent from a case dir but present in
    --wrf-run-dir, which `wtune run` links into the run dir. alt_dir is None for a run dir."""
    if alt_dir is None:
        return False
    if name.endswith("*"):
        prefix = name[:-1]
        try:
            hit = any(n.startswith(prefix) for n in os.listdir(alt_dir))
        except OSError:
            hit = False
    else:
        hit = (Path(alt_dir) / name).exists()
    if hit:
        findings.append(Finding(
            INFO, "C3",
            "domain %d: required physics data file %s not in case dir; present in %s, linked "
            "at run time%s" % (dom_index + 1, name, alt_dir, (" (%s)" % note) if note else ""),
        ))
    return hit


def _require_files(run_dir, dom_index, files, findings, note="", alt_dir=None):
    for fname in files:
        full = _resolve_path(run_dir, fname)
        if not full.exists():
            if _linked_at_run_time(alt_dir, dom_index, fname, findings, note):
                continue
            findings.append(Finding(
                FATAL, "C3",
                "domain %d: required physics data file %s does not exist%s"
                % (dom_index + 1, fname, (" (%s)" % note) if note else ""),
                evidence=str(full),
                fix="copy/link %s from WRF's own run/ directory into the run_dir" % fname,
            ))


def _require_prefix(run_dir, dom_index, prefix, findings, note="", alt_dir=None):
    """FATAL unless at least one run_dir entry's name starts with `prefix` (for version-suffixed
    tables such as P3's p3_lookupTable_1.dat-v5.4_2momI)."""
    try:
        names = os.listdir(run_dir)
    except OSError:
        names = []
    if not any(n.startswith(prefix) for n in names) and \
            not _linked_at_run_time(alt_dir, dom_index, prefix + "*", findings, note):
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


@dataclass(frozen=True)
class _C3Facts:
    """What C3 reads from the namelist and the directory layout, plus the findings sink."""
    nl: dict
    run_dir: str | Path
    alt: str | None          # --wrf-run-dir that `wtune run` links into a case dir, else None
    case_dir: bool           # no wrf.exe here: a case/source dir, not what WRF runs from
    findings: list
    ghg_input: object
    o3input: int
    aer_opt: int
    eclipse: int
    use_wudapt_lcz: int
    windturbine: int
    co2tf: int


def _c3_facts(nl: dict, run_dir: str | Path, findings: list[Finding],
              wrf_run_dir: str | Path | None) -> _C3Facts:
    ghg_input = wp.g(nl, "physics", "ghg_input", 1)
    if ghg_input is None:
        ghg_input = 1
    # a dir without wrf.exe is a case/source dir: `wtune run` links WRF's run/ files into the real
    # run dir, so a table missing here but present in --wrf-run-dir is INFO, not FATAL.
    case_dir = not os.path.lexists(os.path.join(str(run_dir), "wrf.exe"))
    alt = str(wrf_run_dir) if case_dir and wrf_run_dir and Path(wrf_run_dir).is_dir() else None
    return _C3Facts(
        nl=nl, run_dir=run_dir, alt=alt, case_dir=case_dir, findings=findings,
        ghg_input=ghg_input,
        o3input=_int_opt(wp.g(nl, "physics", "o3input", 2), 2),
        aer_opt=_int_opt(wp.g(nl, "physics", "aer_opt", 0), 0),
        eclipse=_int_opt(wp.g(nl, "physics", "ra_sw_eclipse", 0), 0),
        use_wudapt_lcz=_int_opt(wp.g(nl, "physics", "use_wudapt_lcz", 0), 0),
        windturbine=_int_opt(wp.g(nl, "physics", "windturbine_spec", 0), 0),
        co2tf=_int_opt(wp.g(nl, "physics", "co2tf", 1), 1))


def _c3_need(f: _C3Facts, d: int, files: list[str], note: str) -> None:
    _require_files(f.run_dir, d, files, f.findings, alt_dir=f.alt, note=note)


def _c3_domain1_only(f: _C3Facts) -> None:
    """oznini/aerosol_in are gated on id==1; eclipse and wind turbine are scalar options."""
    if f.aer_opt == 1:
        _c3_need(f, 0, _AEROSOL_FILES, "aer_opt=1 (Tegen aerosol climatology)")
    if f.eclipse == 1:
        _c3_need(f, 0, [_ECLIPSE_FILE], "ra_sw_eclipse=1")
    if f.windturbine == 1:
        _c3_need(f, 0, ["wind-turbine-1.tbl"], "windturbine_spec=1")


def _c3_radiation(f: _C3Facts, d: int, ra_lw, ra_sw) -> None:
    if ra_lw in _RA_LW_FILES:
        _c3_need(f, d, _RA_LW_FILES[ra_lw], "ra_lw_physics=%s" % ra_lw)
    if ra_sw in _RA_SW_FILES:
        _c3_need(f, d, _RA_SW_FILES[ra_sw], "ra_sw_physics=%s" % ra_sw)
    for scheme, files in _RA_EITHER_FILES.items():
        if scheme in (ra_lw, ra_sw):
            _c3_need(f, d, files, "ra_lw/ra_sw_physics=%s" % scheme)
    if _GFDL_SCHEME in (ra_lw, ra_sw):
        if f.co2tf == 1:
            _c3_need(f, d, _GFDL_CO2O3_FILES, "GFDL radiation (99) with co2tf=1 (default)")
        else:
            _c3_need(f, d, _GFDL_CONRAD_FILES, "GFDL radiation (99) with co2tf=%s" % f.co2tf)
    if _CAM_SCHEME in (ra_lw, ra_sw):
        _c3_need(f, d, _OZONE_FILES, "CAM radiation reads CAM ozone regardless of o3input")
    elif d == 0 and f.o3input == 2 and (ra_lw in _RRTMG_FAMILY or ra_sw in _RRTMG_FAMILY):
        _c3_need(f, d, _OZONE_FILES,
                 "o3input=2 (default) with RRTMG radiation; set o3input=0 "
                 "to use the scheme's built-in ozone profile instead")


def _c3_camtr(f: _C3Facts, d: int, ra_lw) -> None:
    if not (int(f.ghg_input) == 1 and ra_lw not in _CAMTR_EXEMPT_RA_LW):
        return
    full = _resolve_path(f.run_dir, "CAMtr_volume_mixing_ratio")
    if not full.exists() and not _linked_at_run_time(
            f.alt, d, "CAMtr_volume_mixing_ratio", f.findings,
            "ghg_input=%s with ra_lw_physics=%s" % (f.ghg_input, ra_lw)):
        f.findings.append(Finding(
            FATAL, "C3",
            "domain %d: ghg_input=1 (default) with ra_lw_physics=%s (not CAM) requires "
            "CAMtr_volume_mixing_ratio, which does not exist" % (d + 1, ra_lw),
            evidence=str(full),
            fix="link CAMtr_volume_mixing_ratio (WRF's run/ ships *.SSP245 as the "
                "default) into the run_dir, or set ghg_input=0",
        ))


def _c3_surface(f: _C3Facts, d: int, sf) -> None:
    if sf in _LSM_FILES:
        _c3_need(f, d, _LSM_FILES[sf], "sf_surface_physics=%s (%s)" % (sf, _LSM_NAMES[sf]))
    elif sf == 4:  # Noah-MP
        if not any(_resolve_path(f.run_dir, n).exists() for n in _NOAHMP_TABLE_NAMES) and \
                not any(_linked_at_run_time(f.alt, d, n, f.findings, "sf_surface_physics=4")
                        for n in _NOAHMP_TABLE_NAMES):
            f.findings.append(Finding(
                FATAL, "C3",
                "domain %d: sf_surface_physics=4 (Noah-MP) requires NoahmpTable.TBL "
                "(or MPTABLE.TBL on older WRF), neither exists" % (d + 1),
                evidence=str(_resolve_path(f.run_dir, _NOAHMP_TABLE_NAMES[0])),
                fix="copy NoahmpTable.TBL from WRF's own run/ directory into the run_dir",
            ))
    sf_urban = _int_opt(wp.g_dom(f.nl, "physics", "sf_urban_physics", d, 0), 0)
    if sf_urban in _URBAN_SCHEMES:
        if f.use_wudapt_lcz == 1:
            _c3_need(f, d, ["URBPARM_LCZ.TBL"],
                     "sf_urban_physics=%s with use_wudapt_lcz=1" % sf_urban)
        else:
            _c3_need(f, d, ["URBPARM.TBL"], "sf_urban_physics=%s" % sf_urban)


def _c3_chem_and_microphysics(f: _C3Facts, d: int, mp) -> None:
    chem_opt = _int_opt(wp.g_dom(f.nl, "chem", "chem_opt", d, 0), 0)
    if chem_opt in _HLC_CHEM_OPTS:
        _c3_need(f, d, ["HLC.TBL"], "chem_opt=%s" % chem_opt)
    if mp in _MP_FILES:
        _c3_need(f, d, _MP_FILES[mp], "mp_physics=%s" % mp)
    if mp in _P3_TABLE1_MP:
        _require_prefix(f.run_dir, d, _P3_TABLE1_PREFIX, f.findings, alt_dir=f.alt,
                        note="mp_physics=%s (P3)" % mp)
    if mp in _P3_TABLE2_MP:
        _require_prefix(f.run_dir, d, _P3_TABLE2_PREFIX, f.findings, alt_dir=f.alt,
                        note="mp_physics=%s (P3, nCat>1)" % mp)
    if mp in _THOMPSON_MP:
        missing = [n for n in _THOMPSON_DAT_FILES
                   if not _resolve_path(f.run_dir, n).exists()
                   and not (f.alt and (Path(f.alt) / n).exists())]
        if missing:
            f.findings.append(Finding(
                WARN, "C3",
                "domain %d: mp_physics=%s (Thompson) is missing prestaged lookup table(s) "
                "%s -- WRF recomputes them at init instead of failing, costing minutes"
                % (d + 1, mp, ", ".join(missing)),
                evidence="checked in %s" % f.run_dir,
                fix="prestage %s in the run_dir to skip the init-time table build"
                    % ", ".join(missing),
            ))


def _c3_wrf_run_listing(f: _C3Facts, wrf_run_dir: str | Path) -> None:
    """Files of WRF's own run/ that the case/run dir lacks (INFO in a case dir, WARN otherwise)."""
    wrf_run_dir = Path(wrf_run_dir)
    if not wrf_run_dir.is_dir():
        f.findings.append(Finding(
            INFO, "C3", "--wrf-run-dir %s is not a directory; skipped" % wrf_run_dir,
        ))
        return
    absent = [entry for entry in sorted(wrf_run_dir.iterdir())
              if entry.is_file() and not (Path(f.run_dir) / entry.name).exists()]
    if f.case_dir and absent:
        names = [e.name for e in absent]
        f.findings.append(Finding(
            INFO, "C3",
            "%d file(s) of WRF's own run/ (%s) are not in the case dir (%s%s); "
            "wtune run links them at run time"
            % (len(names), wrf_run_dir, ", ".join(names[:5]),
               ", ..." if len(names) > 5 else ""),
        ))
    for entry in ([] if f.case_dir else absent):
        f.findings.append(Finding(
            WARN, "C3",
            "%s is present in WRF's own run/ (%s) but absent from run_dir"
            % (entry.name, wrf_run_dir),
            evidence=str(entry),
            fix="stage %s into run_dir if the active physics options need it "
                "(real incident: sc26p2-n4 staging used an extension whitelist and "
                "missed the extension-less CAMtr_volume_mixing_ratio)" % entry.name,
        ))


def check_c3_physics_files(nl: dict, run_dir: str | Path, findings: list[Finding],
                           wrf_run_dir: str | Path | None = None) -> None:
    """Data files the chosen physics opens at init, per active domain (FATAL; plain-Thompson
    .dat tables WARN): radiation, LSM/urban/Noah-MP tables, microphysics, chem HLC.TBL, plus
    the domain-1-only aerosol/ozone/eclipse inputs. Tables and WRF anchors: wtune.preflight_tables."""
    f = _c3_facts(nl, run_dir, findings, wrf_run_dir)
    _c3_domain1_only(f)
    for d in range(wp.get_max_dom(nl)):
        phys = _resolved_physics(nl, d)
        ra_lw, ra_sw = phys.get("ra_lw_physics"), phys.get("ra_sw_physics")
        _c3_radiation(f, d, ra_lw, ra_sw)
        _c3_camtr(f, d, ra_lw)
        _c3_surface(f, d, phys.get("sf_surface_physics"))
        _c3_chem_and_microphysics(f, d, phys.get("mp_physics"))
    if wrf_run_dir:
        _c3_wrf_run_listing(f, wrf_run_dir)


# --------------------------------------------------------------------------
# C4 -- aerosol climatology fields (FATAL, netCDF4 optional)
# --------------------------------------------------------------------------

def _has_qnwfa_qnifa(path: Path) -> tuple[bool | None, object]:
    """(True/False, missing names) for the aerosol climatology fields; (None, message) on open failure."""
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


def _c4_wif_finding(aero_mp: list[int]) -> Finding:
    # WTF test ralbNE (scc job 1086): real.exe FATAL 'wif_input_opt=0 but mp_physics=28, 29, or 88'
    # -- dyn_em/module_initialize_real.F ~2741-2748 (THOMPSONAERO, RCON, or TEMPO with
    # tempo_aerosolaware=1, and wif_input_opt == 0)
    return Finding(
        FATAL, "C4",
        "mp_physics=%s needs wif_input_opt >= 1 (&domains); real.exe aborts with "
        "'wif_input_opt=0 but mp_physics=28, 29, or 88'" % ",".join(map(str, aero_mp)),
        evidence="dyn_em/module_initialize_real.F if_thompsonaero_3d; WTF ralbNE",
        fix="set wif_input_opt = 1 (and num_wif_levels = 30) in &domains, with the "
            "QNWFA_QNIFA climatology in metgrid constants_name")


def _c4_wrfinput_finding(wrfinput: Path, missing) -> Finding:
    return Finding(
        FATAL, "C4",
        "mp_physics=28 with use_aero_icbc/wif_input_opt requires QNWFA/QNIFA in "
        "wrfinput_d01; missing %s" % missing,
        evidence=str(wrfinput),
        fix="regenerate wrfinput_d01 via real.exe with an aerosol climatology input "
            "(cs-lafe cold-start incident: 'aerosol climatology field(s) missing')",
    )


def _c4_met_em_finding(met_em: Path, missing) -> Finding:
    return Finding(
        FATAL, "C4",
        "mp_physics=28 with use_aero_icbc/wif_input_opt requires QNWFA/QNIFA in "
        "met_em files (real.exe stage); missing %s in %s" % (missing, met_em.name),
        evidence=str(met_em),
        fix="rerun metgrid with an aerosol climatology field, or set "
            "use_aero_icbc=.false. if aerosol ICBC was not intended",
    )


def _c4_check_inputs(run_dir: Path, findings: list[Finding]) -> None:
    """QNWFA/QNIFA in wrfinput_d01 (wrf.exe stage) or, without it, in the first met_em.d01.*."""
    met_em = sorted(run_dir.glob("met_em.d01.*"))
    wrfinput = run_dir / "wrfinput_d01"
    if wrfinput.exists():
        path, build = wrfinput, _c4_wrfinput_finding
    elif met_em:
        path, build = met_em[0], _c4_met_em_finding
    else:
        findings.append(Finding(
            INFO, "C4",
            "mp_physics=28 with use_aero_icbc/wif_input_opt but neither wrfinput_d01 nor "
            "met_em.d01.* found in run_dir; cannot check QNWFA/QNIFA",
        ))
        return
    ok, missing = _has_qnwfa_qnifa(path)
    if ok is None:
        findings.append(Finding(INFO, "C4", missing))
    elif not ok:
        findings.append(build(path, missing))


def check_c4_aerosol_climatology(nl: dict, run_dir: str | Path, findings: list[Finding]) -> None:
    """mp_physics=28/29 (88 with tempo_aerosolaware=1) needs wif_input_opt >= 1 (FATAL); mp 28 with
    use_aero_icbc/wif_input_opt needs QNWFA/QNIFA in wrfinput_d01 or the met_em files (cs-lafe:
    real.exe 'aerosol climatology field(s) missing'). The file check needs netCDF4."""
    max_dom = wp.get_max_dom(nl)
    # wif_input_opt is a &domains key (Registry default 0); tempo_aerosolaware is &physics (default 0)
    wif_input_opt = int(wp.g(nl, "domains", "wif_input_opt", 0) or 0)
    tempo_aero = int(wp.g(nl, "physics", "tempo_aerosolaware", 0) or 0)
    mps = {_resolved_physics(nl, d).get("mp_physics") for d in range(max_dom)}
    aero_mp = sorted(m for m in mps if m in (28, 29) or (m == 88 and tempo_aero == 1))
    if aero_mp and wif_input_opt == 0:
        findings.append(_c4_wif_finding(aero_mp))
        return
    if 28 not in mps:
        return
    if not HAVE_NETCDF4:
        findings.append(Finding(INFO, "C4", "SKIPPED (netCDF4 not available)"))
        return
    use_aero_icbc = wp.g(nl, "physics", "use_aero_icbc", False)
    if not (use_aero_icbc or int(wif_input_opt) >= 1):
        return
    _c4_check_inputs(Path(run_dir), findings)


# --------------------------------------------------------------------------
# C5 -- met_em naming vs nocolons (FATAL, real.exe stage only)
# --------------------------------------------------------------------------

_MET_EM_RE = re.compile(
    r"^(met_em\.d\d\d\.\d{4}-\d\d-\d\d)_(\d\d)([_:])(\d\d)\3(\d\d)(\.nc)?$")


def check_c5_met_em_naming(nl: dict, run_dir: str | Path, findings: list[Finding]) -> None:
    """real.exe opens met_em names per nocolons: .true. -> 'YYYY-MM-DD_HH_MM_SS', else the colon
    form. Only the other form on disk is FATAL (cs-lafe: nocolons=.true. with colon files)."""
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


def _c6_dims_finding(ds, geom, d: int, wrfinput: Path) -> Finding | None:
    """FATAL when wrfinput's grid-dimension attributes differ from the namelist's e_we/e_sn/e_vert."""
    e_we, e_sn, e_vert = geom
    expected = {
        "WEST-EAST_GRID_DIMENSION": int(e_we),
        "SOUTH-NORTH_GRID_DIMENSION": int(e_sn),
        "BOTTOM-TOP_GRID_DIMENSION": int(e_vert),
    }
    dims = {}
    for attr in expected:
        try:
            dims[attr] = int(ds.getncattr(attr))
        except AttributeError:
            dims[attr] = None
    mismatches = [
        "%s: namelist=%s wrfinput=%s" % (k, expected[k], dims[k])
        for k in expected if dims[k] is not None and dims[k] != expected[k]
    ]
    if not mismatches:
        return None
    return Finding(
        FATAL, "C6",
        "domain %d: %s grid dims do not match namelist e_we/e_sn/e_vert" % (d + 1, wrfinput.name),
        evidence="; ".join(mismatches),
        fix="regenerate wrfinput_d%02d with real.exe using the SAME e_we/e_sn/"
            "e_vert as this namelist, or fix the namelist to match" % (d + 1),
    )


def _c6_start_finding(ds, nl: dict, d: int, wrfinput: Path) -> Finding | None:
    """WARN when wrfinput's SIMULATION_START_DATE differs from the namelist's start for domain d."""
    try:
        sim_start = ds.getncattr("SIMULATION_START_DATE")
    except AttributeError:
        sim_start = None
    nl_start = _domain_datetime(nl, "start", d)
    if not (sim_start and nl_start is not None):
        return None
    nl_start_str = nl_start.strftime("%Y-%m-%d_%H:%M:%S")
    if sim_start == nl_start_str:
        return None
    return Finding(
        WARN, "C6",
        "domain %d: %s SIMULATION_START_DATE=%s does not match namelist "
        "start_date %s" % (d + 1, wrfinput.name, sim_start, nl_start_str),
        evidence="SIMULATION_START_DATE=%s namelist=%s" % (sim_start, nl_start_str),
        fix="regenerate wrfinput_d%02d for this namelist's start date, or fix "
            "start_year/month/day/hour/minute/second" % (d + 1),
    )


def _c6_check_wrfinput(nl: dict, d: int, wrfinput: Path, findings: list[Finding]) -> None:
    ds, err = _open_nc(wrfinput)
    if ds is None:
        findings.append(Finding(INFO, "C6", "could not open %s: %s" % (wrfinput, err)))
        return
    try:
        try:
            geom = wp.get_domain_geom(nl, d)
        except wp.PredictError as exc:
            findings.append(Finding(INFO, "C6", str(exc)))
            return
        for found in (_c6_dims_finding(ds, geom, d, wrfinput),
                      _c6_start_finding(ds, nl, d, wrfinput)):
            if found is not None:
                findings.append(found)
    finally:
        ds.close()


def _c6_bdy_times(ds) -> list[datetime.datetime]:
    """Valid times of a wrfbdy file's Times variable (rows that do not parse are skipped)."""
    if "Times" not in ds.variables:
        return []
    time_strs = []
    for row in ds.variables["Times"][:]:
        try:
            time_strs.append(b"".join(bytes(c) if not isinstance(c, bytes) else c
                                      for c in row).decode("ascii"))
        except Exception:
            pass
    parsed = []
    for t in time_strs:
        try:
            parsed.append(datetime.datetime.strptime(t.strip(), "%Y-%m-%d_%H:%M:%S"))
        except ValueError:
            continue
    return parsed


def _c6_d01_end(nl: dict) -> datetime.datetime | None:
    d01_start = _domain_datetime(nl, "start", 0)
    if d01_start is None:
        return None
    total = _run_seconds(nl)
    if total > 0:
        return d01_start + datetime.timedelta(seconds=total)
    return _domain_datetime(nl, "end", 0)


def _c6_bdy_finding(nl: dict, last_bdy: datetime.datetime) -> Finding | None:
    """FATAL when the last boundary time + interval_seconds stops short of d01's end."""
    interval_s = wp.g(nl, "time_control", "interval_seconds", None)
    d01_end = _c6_d01_end(nl)
    if d01_end is None or not interval_s:
        return None
    bdy_reach = last_bdy + datetime.timedelta(seconds=float(interval_s))
    if bdy_reach >= d01_end:
        return None
    return Finding(
        FATAL, "C6",
        "wrfbdy_d01's last boundary time (%s) + interval_seconds "
        "(%ss) = %s, before d01's end (%s)" % (last_bdy, interval_s, bdy_reach, d01_end),
        evidence="last Times=%s interval_seconds=%s d01 end=%s" % (last_bdy, interval_s, d01_end),
        fix="rerun real.exe with boundary data covering the full "
            "d01 window (wrf_get_next_time will fail at run time otherwise)",
    )


def _c6_check_wrfbdy(nl: dict, wrfbdy: Path, findings: list[Finding]) -> None:
    ds, err = _open_nc(wrfbdy)
    if ds is None:
        findings.append(Finding(INFO, "C6", "could not open %s: %s" % (wrfbdy, err)))
        return
    try:
        times = _c6_bdy_times(ds)
    finally:
        ds.close()
    if times:
        found = _c6_bdy_finding(nl, max(times))
        if found is not None:
            findings.append(found)


def check_c6_input_consistency(nl: dict, run_dir: str | Path, findings: list[Finding]) -> None:
    """wrfinput_dNN grid dims / SIMULATION_START_DATE vs the namelist, and wrfbdy_d01's last
    boundary time + interval_seconds vs d01's end. Needs netCDF4 (SKIPPED otherwise)."""
    if not HAVE_NETCDF4:
        findings.append(Finding(INFO, "C6", "SKIPPED (netCDF4 not available)"))
        return
    run_dir = Path(run_dir)
    for d in range(wp.get_max_dom(nl)):
        wrfinput = run_dir / ("wrfinput_d%02d" % (d + 1))
        if wrfinput.exists():
            _c6_check_wrfinput(nl, d, wrfinput, findings)
    wrfbdy = run_dir / "wrfbdy_d01"
    if wrfbdy.exists():
        _c6_check_wrfbdy(nl, wrfbdy, findings)
    # sst_update=1's wrflowinp (auxinput4_inname) is checked in C2 with the other files the
    # namelist names -- without needing netCDF4.


# --------------------------------------------------------------------------
# C7 -- decomposition (FATAL)
# --------------------------------------------------------------------------

def _c7_shape(nl: dict, findings: list[Finding], ranks: int | None, nproc_x: int | None,
              nproc_y: int | None) -> tuple[int, int] | None:
    """Effective (nproc_x, nproc_y) after the ranks-vs-shape check, or None when there is no
    shape to check patch sizes against (INFO emitted where the user should know)."""
    eff_x = nproc_x if nproc_x is not None else wp.g(nl, "domains", "nproc_x", None)
    eff_y = nproc_y if nproc_y is not None else wp.g(nl, "domains", "nproc_y", None)
    # Registry default -1 means "not set" (WRF then picks the shape itself)
    eff_x = None if isinstance(eff_x, int) and eff_x <= 0 else eff_x
    eff_y = None if isinstance(eff_y, int) and eff_y <= 0 else eff_y
    if eff_x is None and eff_y is None and ranks is None:
        findings.append(Finding(
            INFO, "C7",
            "no nproc_x/nproc_y in the namelist and no --ranks/--nproc-x/--nproc-y given; "
            "decomposition not checked",
        ))
        return None
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
        return eff_x, eff_y
    if ranks is not None:
        int(ranks)  # value unused; kept so a non-integer ranks still raises here as before
        return None  # ranks given but no shape to check patch sizes against
    findings.append(Finding(
        INFO, "C7",
        "only one of nproc_x/nproc_y is set (x=%s y=%s); cannot check patch legality"
        % (eff_x, eff_y),
    ))
    return None


def _c7_patch_findings(nl: dict, eff_x: int, eff_y: int, findings: list[Finding]) -> None:
    for d in range(wp.get_max_dom(nl)):
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


def check_c7_decomposition(nl: dict, findings: list[Finding], ranks: int | None = None,
                           nproc_x: int | None = None, nproc_y: int | None = None) -> None:
    """nproc_x*nproc_y must equal the ranks, and every active domain's RSL-lite patch must be >= 10
    mass points both ways (module_check_a_mundo.F:3014-3029). nproc_x/nproc_y/ranks default to the
    namelist's &domains values; --nproc-x/--nproc-y/--ranks override them."""
    shape = _c7_shape(nl, findings, ranks, nproc_x, nproc_y)
    if shape is not None:
        _c7_patch_findings(nl, shape[0], shape[1], findings)


# --------------------------------------------------------------------------
# C8 -- time step vs grid spacing (FATAL for d01, WARN for nests)
# --------------------------------------------------------------------------

def _dt_seconds(nl: dict, dom_index: int) -> float | None:
    ts = wp.g_dom(nl, "domains", "time_step", dom_index, None)
    if ts is None:
        return None
    num = wp.g_dom(nl, "domains", "time_step_fract_num", dom_index, 0) or 0
    den = wp.g_dom(nl, "domains", "time_step_fract_den", dom_index, 1) or 1
    return float(ts) + float(num) / float(den)


def _c8_d01_finding(dt0: float, dx0, dy0, dx0_km: float, ratio0: float, rrs: float) -> Finding:
    return Finding(
        FATAL, "C8",
        "d01: time_step/dx ratio %.3f exceeds reasonable_time_step_ratio %.3f" % (ratio0, rrs),
        evidence="time_step=%.3fs dx=min(%s,%s)m=%.1fkm ratio=%.3f" % (dt0, dx0, dy0, dx0_km, ratio0),
        fix="reduce time_step (dyn_em/start_em.F:305 aborts with 'Time step too large' "
            "at this ratio) or raise reasonable_time_step_ratio if you have verified "
            "stability",
    )


def _c8_nest_findings(nl: dict, dt0: float, dx0_m: float, rrs: float,
                      findings: list[Finding]) -> None:
    """WARN per nest whose effective dt/dx (through the parent chain) exceeds the ratio."""
    max_dom = wp.get_max_dom(nl)
    dt_by_domain = [None] * max_dom
    dx_by_domain = [None] * max_dom
    dt_by_domain[0], dx_by_domain[0] = dt0, dx0_m
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


def check_c8_time_step(nl: dict, findings: list[Finding]) -> None:
    """dt_s/dx_km > reasonable_time_step_ratio (namelist, default 6.0; equality allowed) is
    FATAL for d01 only -- WRF's own gate (start_em.F:280-311) checks domain 1 once, real-data and
    non-adaptive runs only. Nests over the ratio (effective dt/dx through the parent chain) are
    WARN: WRF does not stop them, they can go unstable mid-run."""
    wp.get_max_dom(nl)
    if bool(wp.g(nl, "domains", "use_adaptive_time_step", False)):
        findings.append(Finding(
            INFO, "C8", "SKIPPED: use_adaptive_time_step=.true. (WRF does not apply the "
                        "fixed-ratio check)",
        ))
        return
    rrs = wp.g(nl, "domains", "reasonable_time_step_ratio", 6.0)
    rrs = float(6.0 if rrs is None else rrs)
    dt0 = _dt_seconds(nl, 0)
    dx0 = wp.g_dom(nl, "domains", "dx", 0, None)
    dy0 = wp.g_dom(nl, "domains", "dy", 0, None)
    if dt0 is None or dx0 is None or dy0 is None:
        findings.append(Finding(
            INFO, "C8", "domain 1 time_step/dx/dy not resolvable; C8 skipped",
        ))
        return
    dx0_m = min(float(dx0), float(dy0))
    dx0_km = dx0_m / 1000.0
    ratio0 = dt0 / dx0_km
    if ratio0 > rrs:
        findings.append(_c8_d01_finding(dt0, dx0, dy0, dx0_km, ratio0, rrs))
    _c8_nest_findings(nl, dt0, dx0_m, rrs, findings)


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


def check_c9_disk(nl: dict, run_dir: str | Path, findings: list[Finding]) -> None:
    """Conservative estimate: bytes_per_frame = e_we*e_sn*e_vert*4 bytes*~40 3-D fields, summed
    over every active domain's history frames (floor(run_minutes/history_interval)+1, the same
    frame-count rule wtune/predict.py uses), compared against os.statvfs(run_dir) free space."""
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

# Range bounds (K; SMOIS m3/m3) -- masking per field in check_c10_wrfinput_sanity. Healthy scc
# runs carry TSLB/SST exactly 0 (undefined fill, inland lakes) and small negative deep SMOIS.
_C10_TSLB_RANGE = (200.0, 350.0)    # K
_C10_SST_RANGE = (180.0, 350.0)     # K
_C10_SMOIS_RANGE = (0.0, 1.0)       # m3 m-3
_C10_TSK_RANGE = (180.0, 350.0)     # K
# Pre-registered: dorian2019 d02 0.034 % ran fine (job 1392), tw3km-2031 case_I 8.8 % went NaN.
_C10_TSLB_NONZERO_FATAL_FRACTION = 0.001   # 0.1 % of land TSLB values
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


def _c10_nonfinite(ds, d: int, wname: str, findings: list[Finding]) -> None:
    import numpy as np
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
                % (d + 1, varname, wname, int(bad.sum())),
                evidence="first bad index=%s" % (idx,),
                fix="regenerate wrfinput_d%02d -- real.exe's interpolation already "
                    "broke; wrf.exe propagates this silently and may still print "
                    "SUCCESS COMPLETE" % (d + 1),
            ))


def _c10_tslb(ds, land, d: int, wname: str, findings: list[Finding]) -> None:
    import numpy as np
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
                     % (int(zero.sum()), " and XLAND==1" if "XLAND" in ds.variables else ""),
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
        _c10_range_finding(findings, sev, d, "TSLB", wname, arr, bad, lo, hi, "non-zero ",
                           " (land points; %.3f%% of %d land values, FATAL above %.1f%%)"
                           % (100.0 * frac, n_land, 100.0 * _C10_TSLB_NONZERO_FATAL_FRACTION))


def _c10_sst(ds, water, d: int, wname: str, findings: list[Finding]) -> None:
    import numpy as np
    arr = _c10_filled(ds.variables["SST"])
    lo, hi = _C10_SST_RANGE
    bad = (np.isfinite(arr) & _c10_bcast(water, arr) & (arr != 0.0)
           & ((arr < lo) | (arr > hi)))
    if bad.any():
        _c10_range_finding(findings, FATAL, d, "SST", wname, arr, bad, lo, hi,
                           "non-zero ", " (water points)")


def _c10_smois(ds, land, d: int, wname: str, findings: list[Finding]) -> None:
    import numpy as np
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


def _c10_tsk(ds, d: int, wname: str, findings: list[Finding]) -> None:
    import numpy as np
    arr = _c10_filled(ds.variables["TSK"])
    lo, hi = _C10_TSK_RANGE
    bad = np.isfinite(arr) & ((arr < lo) | (arr > hi))
    if bad.any():
        _c10_range_finding(findings, FATAL, d, "TSK", wname, arr, bad, lo, hi, "", "")


def _c10_check_domain(ds, d: int, wname: str, findings: list[Finding]) -> None:
    _c10_nonfinite(ds, d, wname, findings)
    land, water = _c10_land_water(ds)
    if "TSLB" in ds.variables:
        _c10_tslb(ds, land, d, wname, findings)
    if "SST" in ds.variables:
        _c10_sst(ds, water, d, wname, findings)
    if "SMOIS" in ds.variables:
        _c10_smois(ds, land, d, wname, findings)
    if "TSK" in ds.variables:
        _c10_tsk(ds, d, wname, findings)


def check_c10_wrfinput_sanity(nl: dict, run_dir: str | Path, findings: list[Finding]) -> None:
    """wrfinput_d0N plausibility before real.exe/wrf.exe (tw3km-2031 case_I: AIFS soil fill gave
    TSLB ~-23 C, wrf.exe all-NaN by hour 1 yet 'SUCCESS COMPLETE'). Land = LANDMASK==1 (and XLAND
    ==1). Per-variable bounds and severities: the _C10_* constants and _c10_* helpers above."""
    if not HAVE_NETCDF4:
        findings.append(Finding(INFO, "C10", "SKIPPED (netCDF4 not available)"))
        return
    try:
        import numpy  # noqa: F401 -- availability probe; the _c10_* helpers import it
    except ImportError:
        findings.append(Finding(INFO, "C10", "SKIPPED (numpy not available)"))
        return
    run_dir = Path(run_dir)
    checked_any = False
    for d in range(wp.get_max_dom(nl)):
        wrfinput = run_dir / ("wrfinput_d%02d" % (d + 1))
        if not wrfinput.exists():
            continue
        checked_any = True
        ds, err = _open_nc(wrfinput)
        if ds is None:
            findings.append(Finding(INFO, "C10", "could not open %s: %s" % (wrfinput, err)))
            continue
        try:
            _c10_check_domain(ds, d, wrfinput.name, findings)
        finally:
            ds.close()
    if not checked_any:
        findings.append(Finding(INFO, "C10", "no wrfinput_d0N found in run_dir; skipped"))


# --------------------------------------------------------------------------
# C11 -- iofields files are re-parsed at every I/O call (WARN when that costs real time)
# --------------------------------------------------------------------------

IOFIELDS_WARN_S = 60.0              # WARN above max(this, the fraction below of compute time)
IOFIELDS_WARN_COMPUTE_FRACTION = 0.02
IOFIELDS_PARSE_EVIDENCE = ("climate-ea50 on scc, 12 lines / 167 fields: 0.21 s per parse, 38% "
                           "of a 13-month run (perf job 3134); one merged line bit-identical and "
                           "10% faster, 30 kept fields listed additively 30% faster (jobs "
                           "3142-3144)")


def _iofields_files(nl, run_dir):
    """['<name> <lines> line(s) / <fields> field(s)', ...] of the distinct iofields files."""
    import wtune.iofields
    out = []
    for d in range(wp.get_max_dom(nl)):
        fname = _namelist_file_names(nl, "iofields_filename", None, d)
        path = _resolve_path(run_dir, fname) if fname else None
        if path and path.is_file():
            item = "%s %d line(s) / %d field(s)" % ((fname,) + wtune.iofields.counts(
                path.read_text()))
            if item not in out:
                out.append(item)
    return out


def check_c11_iofields_parse(nl: dict, run_dir: str | Path, findings: list[Finding],
                             compute_s: float | None = None) -> None:
    """WRF re-parses the iofields file on rank 0 at every input/output call while the other
    ranks wait. WARN when that costs the run more than max(60 s, 2% of compute_s), with the
    seconds as written and after the line merge `wtune run` does (wtune.io_model's term)."""
    from wtune import io_model as iom
    raw = iom.case_iofields_calls(nl, str(run_dir), merge=False)
    if not any(raw):
        return
    try:
        call_s = iom.iofields_call_s(wp.load_calibration(wp.calibration_path()))
    except wp.PredictError:
        call_s = iom.iofields_call_s()
    parses = iom.run_length_io(nl)["iofields_parses"]
    raw_s = iom.iofields_parse_s(parses, raw, call_s)
    merged_s = iom.iofields_parse_s(parses, iom.case_iofields_calls(nl, str(run_dir)), call_s)
    if raw_s <= max(IOFIELDS_WARN_S, IOFIELDS_WARN_COMPUTE_FRACTION * (compute_s or 0.0)):
        return
    findings.append(Finding(
        WARN, "C11",
        "iofields parsing costs about %.0f s of this run as written, %.0f s after `wtune run` "
        "merges the lines (automatic): %s; WRF re-parses the file at every input/output call "
        "(%d parses, %d get_fieldstr calls each as written)"
        % (raw_s, merged_s, "; ".join(_iofields_files(nl, run_dir)), sum(parses), max(raw)),
        evidence=IOFIELDS_PARSE_EVIDENCE,
        fix="to cut it further, list only the fields you KEEP, additively, on an auxiliary "
            "history stream (`+:h:<n>:FIELD,FIELD,...` with auxhist<n>_outname/_interval and "
            "the default history stream switched off): (5 + kept) calls per parse instead of "
            "%d. That restructures the output streams, so it stays your decision" % max(raw)))


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def run_checks(nl: dict, run_dir: str | Path, namelist_path: str | Path, ranks: int | None = None,
               nproc_x: int | None = None, nproc_y: int | None = None,
               wrf_run_dir: str | Path | None = None) -> list[Finding]:
    findings = []
    checks = [
        ("C1", lambda: check_c1_time_windows(nl, findings)),
        ("C2", lambda: check_c2_iofields(nl, run_dir, namelist_path, findings,
                                          wrf_run_dir=wrf_run_dir)),
        ("C3", lambda: check_c3_physics_files(nl, run_dir, findings, wrf_run_dir=wrf_run_dir)),
        ("C4", lambda: check_c4_aerosol_climatology(nl, run_dir, findings)),
        ("C5", lambda: check_c5_met_em_naming(nl, run_dir, findings)),
        ("C6", lambda: check_c6_input_consistency(nl, run_dir, findings)),
        ("C7", lambda: check_c7_decomposition(nl, findings, ranks=ranks, nproc_x=nproc_x,
                                               nproc_y=nproc_y)),
        ("C8", lambda: check_c8_time_step(nl, findings)),
        ("C9", lambda: check_c9_disk(nl, run_dir, findings)),
        ("C10", lambda: check_c10_wrfinput_sanity(nl, run_dir, findings)),
        ("C11", lambda: check_c11_iofields_parse(nl, run_dir, findings)),
    ]
    for cid, fn in checks:
        try:
            fn()
        except Exception as exc:  # a checker bug must never take down the other checkers
            findings.append(Finding(
                INFO, cid, "internal error in %s: %r" % (cid, exc),
            ))
    return findings


def exit_code_for(findings: list[Finding]) -> int:
    worst = max((_SEVERITY_RANK.get(f.severity, 0) for f in findings), default=0)
    if worst >= 3:
        return 2
    if worst >= 1:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wtune check",
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
    # mismatch (wtune/extensions.py); no-op for plain ARW.
    import wtune.extensions
    ext_lines, ext_code = wtune.extensions.check_report(
        str(namelist_path), str(run_dir), ranks=args.ranks, nproc_x=args.nproc_x, nproc_y=args.nproc_y)
    for line in ext_lines:
        print(line)

    return max(exit_code_for(findings), ext_code)


if __name__ == "__main__":
    sys.exit(main())
