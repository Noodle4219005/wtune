"""scenario.generators.analog -- the `analog` scenario generator: real GFS (Vtable.GFS) forcing
fetched at a REAL analog date, then RELABELLED (its intermediate files' HDATE rewritten) to the
request's FICTIONAL date.

Date SELECTION (matches the existing rebuild-analog flow exactly -- this part is shared code,
not duplicated logic):

    case_rebuild.cmd_analog (case_rebuild.py:2377-2409) is the existing selector:
    candidate_dates() (case_rebuild.py:2215-2231) lists every REAL date within +-doy_window days
    of the stated start's day-of-year, at the same UTC hour, across `years`, and feeds the
    wrfinput-target scored path. With NO wrfinput target (case_rebuild.py:2392-2398) it picks the
    CLIMATOLOGICAL analog -- same DOY, same UTC hour, in the MOST RECENT year of `years` -- via
    case_rebuild.climatological_analog() (case_rebuild.py, added 2026-09-26; fixes a pre-existing
    bug where this was computed as max(candidate_dates(...)), which is the +-doy_window window's
    LAST day, not "same DOY"). select_analog_date() below calls that SAME function directly
    rather than re-deriving the rule -- tests/test_scenario_analog.py asserts bit-for-bit
    equality against case_rebuild.climatological_analog() for identical inputs. The with-target
    scored path (case_rebuild.py:2280-2354, ranks by pattern-correlation/NRMSE against a real
    wrfinput) needs a live GRIB fetch per candidate and is documented in case_rebuild.py itself
    as "best-effort ... NOT independently exercised by the offline test suite"
    (case_rebuild.py:2200-2205); this generator does not reproduce that scored path, for the
    same reason.

Date SEMANTICS after selection -- this generator DIFFERS from case_rebuild's own flow here, by
architect decision, and this is the one place the two disagree:

    case_rebuild's existing flow does NOT relabel HDATE anywhere (verified: no `relabel`/`HDATE`
    rewrite exists in case_rebuild.py). It runs WRF directly AT the real analog date and then
    corrects the mismatch between "real weather" and "the stated/fictional event" by adjusting
    radiation to the STATED date's sun, not by touching the forcing files' own valid-time labels.

    This generator instead fetches/ungribs real forcing at the analog date and then RELABELS
    every resulting intermediate file's HDATE to the request's fictional valid time
    (wps_intermediate.relabel_date) -- SLAB (data) bytes are byte-for-byte identical to what
    ungrib produced at the analog date; only the 24-byte HDATE header changes. Architect
    decision: this gives metgrid the correct sun (real.exe/metgrid derive solar angle and monthly
    climatology interpolation from HDATE) and the correct monthly-climatology interpolation
    directly, without a separate downstream radiation correction step. See
    tests/test_scenario_analog.py::test_relabelled_slab_bytes_match_unrelabelled_ungrib_output
    for the byte-identity check.

External commands (GRIB download, ungrib.exe) are injectable (`download=`, `run_ungrib=`
keyword arguments to generate()) so tests never touch the network or need a real WPS install.
The module-level defaults below are best-effort production stand-ins over this repo's existing
fetch chain (src/tools/metdata.py: GFS/AWS -> RDA -> NCEI -> FNL -> CFSR) -- same
disclosed-limitation pattern case_rebuild.py itself uses for its own network orchestration
(case_rebuild.py:2200-2205): real network/subprocess, not exercised by the offline test suite.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .. import wps_intermediate as wi
from ..base import Caps, ForcingSet, ScenarioError, ScenarioGenerator, register, request_to_dict

_GENERATORS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(os.path.dirname(_GENERATORS_DIR))
_TOOLS_DIR = os.path.join(_SRC_DIR, "tools")


def _case_rebuild():
    """Lazy import of tools/case_rebuild.py -- needs src/tools/ on sys.path, exactly like
    wtune_cli.py's own 'metdata'/'analyze' subcommand dispatch does (src/wtune_cli.py:1801-1804)."""
    if _TOOLS_DIR not in sys.path:
        sys.path.insert(0, _TOOLS_DIR)
    import case_rebuild as cr
    return cr


DEFAULT_YEARS = (2021, 2025)
DEFAULT_DOY_WINDOW = 45

# Vtable.GFS's own verified field set (src/tools/coldstart_sample.py:319-332; cross-checked
# against ungrib/Variable_Tables/Vtable.GFS in the checked-out WPS source): pressure-level TT/
# UU/VV/RH/HGT, surface/near-surface fields, and 4 soil layers. NO hydrometeors.
PRESSURE_FIELDS = ("TT", "UU", "VV", "RH", "HGT")
SURFACE_FIELDS = ("PSFC", "PMSL", "SKINTEMP", "LANDSEA", "SOILHGT", "SEAICE", "SNOW", "SNOWH")
SOIL_FIELDS = ("ST000010", "ST010040", "ST040100", "ST100200",
               "SM000010", "SM010040", "SM040100", "SM100200")
DEFAULT_FIELDS = PRESSURE_FIELDS + SURFACE_FIELDS + SOIL_FIELDS

# GFS 0.25 deg pressure levels (Pa), 1000..10 hPa -- the WRF/WPS-standard 21-level cut.
PRESSURE_LEVELS_PA = tuple(int(round(hpa * 100)) for hpa in (
    1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250,
    200, 150, 100,
))


def select_analog_date(start: datetime.datetime, years: Tuple[int, int] = DEFAULT_YEARS,
                        doy_window: int = DEFAULT_DOY_WINDOW, hour: str = "same"
                        ) -> datetime.datetime:
    """Calls case_rebuild.climatological_analog() directly -- the SAME function
    case_rebuild.cmd_analog's own no-wrfinput branch calls (case_rebuild.py:2392-2398) -- so the
    rule (same day-of-year, same UTC hour, most recent year of `years`) lives in exactly one
    place. `doy_window` is accepted only for call-site/CLI signature compatibility with the
    with-target scored path this generator does not implement; climatological_analog() itself
    does not use a window (it is a same-DOY match, not a +-window one)."""
    cr = _case_rebuild()
    return cr.climatological_analog(start, years, hour=hour)


def _request_valid_times(start, end, interval_h):
    out = []
    t = start
    step = datetime.timedelta(hours=interval_h)
    while t <= end:
        out.append(t)
        t += step
    return out


def _default_download(analog_times, bbox, out_dir):
    """Best-effort production default: fetches real GFS analysis GRIB for each analog time via
    this repo's existing source chain (tools/metdata.py: GFS/AWS -> RDA -> NCEI -> FNL -> CFSR,
    metdata.select_source()/url_for()). Real network; NOT exercised by the offline test suite --
    pass an explicit `download=` callable (tests always do)."""
    if _TOOLS_DIR not in sys.path:
        sys.path.insert(0, _TOOLS_DIR)
    import metdata as md

    grib_dir = Path(out_dir) / "grib"
    grib_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in analog_times:
        source = md.select_source(t)
        url = md.url_for(source, t, 0)
        dest = grib_dir / ("%s.grib2" % t.strftime("%Y%m%d%H"))
        md.download_full(url, str(dest))
        paths.append(dest)
    return paths


def _default_run_ungrib(grib_paths, analog_times, bbox, out_dir, prefix):
    """Best-effort production default: would stage Vtable.GFS and run a real ungrib.exe over
    `grib_paths`. Needs a real WPS install and real GRIB input, so it is NOT reproduced here --
    pass an explicit `run_ungrib=` callable (tests always do); see coldstart_sample.py for this
    repo's own ungrib orchestration (rendered as an sbatch script, not run synchronously)."""
    raise ScenarioError(
        "analog generator's default run_ungrib needs a real WPS/ungrib.exe install; pass an "
        "explicit run_ungrib= callable, or use --dry-run on the CLI")


@register
class AnalogGenerator(ScenarioGenerator):
    name = "analog"
    license = "NOAA GFS: public domain"
    noncommercial = False
    needs_gpu = False
    implemented = True

    def capabilities(self) -> Caps:
        return Caps(fields=DEFAULT_FIELDS, levels_pa=PRESSURE_LEVELS_PA, has_soil=True,
                    has_sst=False, has_hydrometeors=False, min_interval_h=1, max_lead_h=384,
                    needs_gpu=False, license=self.license, noncommercial=False)

    def generate(self, req, *, download=None, run_ungrib=None,
                 years: Tuple[int, int] = DEFAULT_YEARS, doy_window: int = DEFAULT_DOY_WINDOW,
                 hour: str = "same") -> ForcingSet:
        """download(analog_times, bbox, out_dir) -> [grib_path, ...] (any order; run_ungrib gets
        the same analog_times list). run_ungrib(grib_paths, analog_times, bbox, out_dir, prefix)
        -> [intermediate_path, ...] IN THE SAME ORDER as `analog_times` -- one WPS intermediate
        file per analog time, still labelled with the analog (real) HDATE; generate() relabels
        each one to the corresponding fictional valid time itself."""
        analog_date = req.analog_date or select_analog_date(req.start, years=years,
                                                              doy_window=doy_window, hour=hour)
        valid_times = _request_valid_times(req.start, req.end, req.interval_h)
        analog_times = [analog_date + (t - req.start) for t in valid_times]

        download_fn = download or _default_download
        run_ungrib_fn = run_ungrib or _default_run_ungrib

        req.out_dir.mkdir(parents=True, exist_ok=True)
        grib_paths = download_fn(analog_times, req.bbox, req.out_dir)
        intermediate_paths = run_ungrib_fn(grib_paths, analog_times, req.bbox, req.out_dir,
                                            "FILE")
        if len(intermediate_paths) != len(valid_times):
            raise ScenarioError(
                "run_ungrib returned %d intermediate files for %d requested valid times"
                % (len(intermediate_paths), len(valid_times)))

        prefix = "FILE"
        out_files = []
        for src_path, fictional_time in zip(intermediate_paths, valid_times):
            dst_path = req.out_dir / ("%s:%s" % (prefix, fictional_time.strftime("%Y-%m-%d_%H")))
            wi.relabel_date(src_path, dst_path, fictional_time)
            out_files.append(dst_path)

        caps = self.capabilities()
        provenance = {
            "generator": self.name,
            "model": "GFS (Vtable.GFS)",
            "version": None,
            "weights_sha256": None,
            "license": self.license,
            "analog_date": analog_date,
            "created_utc": datetime.datetime.utcnow(),
            "request": request_to_dict(req),
        }
        return ForcingSet(files=out_files, prefix=prefix, valid_times=valid_times,
                           fields_present=list(caps.fields), fields_missing=[],
                           fill_rules={"hydrometeors": "spin-up"}, provenance=provenance)
