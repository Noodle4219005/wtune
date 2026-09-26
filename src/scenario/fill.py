"""scenario.fill -- given a generator's Caps, work out which surface/sub-surface fields a real
metgrid run needs that the generator does not itself produce, where each missing field would come
from, and how to merge an analog day's real forcing into a primary ForcingSet's files to cover
that gap.

Physical-quantity equivalence: different generators may name the SAME physical field differently
-- AIFS's netCDF-to-WPS contract (scenario/generators/aifs.py) writes specific humidity as
SPECHUMD and geopotential height as GHT; Vtable.GFS-based sources (scenario/generators/analog.py)
write RH (relative humidity) and HGT. ATMOSPHERIC_GROUPS below encodes those synonyms: a group
counts as present for a generator if Caps.fields contains ANY member, so compute_missing() never
reports "RH is missing" for a generator that provides the same physical quantity as SPECHUMD.

Architect rule (2026-09-26, aifs C3/C4b review -- REJECTED an earlier version of this module that
filled RH/HGT into an AIFS-primary merge): Fill must NEVER supply a 3-D atmospheric field
(temperature, wind, moisture, height), even under a different name, and even when the primary's
own field for that quantity uses a different name than the fill source's. Root cause of the
rejected defect: WPS's own metgrid renames HGT -> GHT on read (checked-out WPS-4.6.0 source,
metgrid/src/read_met_module.F:93), so naively copying analog's HGT records alongside AIFS's own
GHT records produced TWO competing GHT records per level in the merged intermediate file --
metgrid keeps whichever it reads last, silently splicing analog's real heights (and, via RH vs
SPECHUMD, analog's real moisture) into what was supposed to be an AIFS-only atmospheric state; the
"AIFS" WRF run was actually a silent AIFS/analog hybrid. A prior revision of this module tried to
paper over the resulting metgrid crash (mismatched num_metgrid_levels from AIFS's 13 fixed levels
vs analog's 21 GFS levels) by vertically interpolating the fill field onto the primary's own
levels -- rejected outright: interpolated real-analysis moisture/height is still not the primary
generator's own atmospheric state, regardless of which levels it sits on. If a generator's own
atmospheric group is entirely absent, that is a real capability gap this package must surface
loudly (ScenarioError naming the group and the generator), never paper over with fill.

Fill may ONLY supply the fields in FILLABLE_SURFACE_FIELDS: soil temperature/moisture layers,
SST, SEAICE, SNOW, SNOWH, LANDSEA, and SKINTEMP (only when the primary doesn't already have it).
PSFC/PMSL are atmospheric state, not ancillary surface data: real.exe's vertical interpolation
depends on them, so they get their own ATMOSPHERIC_GROUPS entry (present if EITHER exists --
real.exe can derive the other) and must never come from fill either, same as TT/UU/VV/moisture/
height. SOILHGT (static terrain-of-analysis) is separate again: it comes from geogrid's WPS_GEOG
static data, not from fill.merge_fill's analog-era5 records at all, since it is not carried as a
per-time ungrib record in the fictional generator's own output.

The required-field list below is this repo's own verified Vtable.GFS field set (see
src/tools/coldstart_sample.py:319-332's "Capability gaps" note, confirmed against the checked-out
WPS source's ungrib/Variable_Tables/Vtable.GFS): TT UU VV RH HGT on pressure levels (repeated at
surface XLVL for 2 m/10 m); PSFC PMSL SKINTEMP LANDSEA SOILHGT SEAICE SNOW SNOWH; 4 soil layers'
temperature (ST000010/ST010040/ST040100/ST100200) and moisture (SM000010/SM010040/SM040100/
SM100200). SST is listed separately (per this package's design spec) even though Vtable.GFS has
no distinct SST record of its own (SKINTEMP doubles for it over water in a GFS-only cold start) --
a generator that DOES carry SST (a reanalysis-backed one, say) reports it via Caps.fields and it
drops out of fields_missing normally.

Hydrometeors are NOT in the required list: no ungrib Vtable in this repo carries them (WRF
zero-fills them at init, i.e. spin-up) -- that fill rule is set directly by generators
(fill_rules["hydrometeors"] = "spin-up"), not computed here.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import replace as _replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .base import ScenarioError
from . import wps_intermediate as wi

# Equivalence groups: WPS field names that represent the SAME physical quantity under different
# generator-specific naming conventions, OR mutually-derivable atmospheric-state fields. A group
# is "present" for a generator if Caps.fields contains ANY member. Every member of every group is
# an atmospheric (or atmosphere-derived) field fill.merge_fill must NEVER supply -- see the module
# docstring's architect rule. PRESSURE_GROUP (PSFC/PMSL) is here because real.exe's vertical
# interpolation depends on surface/sea-level pressure, and real.exe can derive one from the other
# -- so, like RH vs SPECHUMD, having either is enough, but having NEITHER is a real gap, not
# something fill may cover with the analog's own (different) pressure state.
TEMPERATURE_GROUP = ("TT",)
UWIND_GROUP = ("UU",)
VWIND_GROUP = ("VV",)
MOISTURE_GROUP = ("RH", "SPECHUMD")
HEIGHT_GROUP = ("HGT", "GHT")
PRESSURE_GROUP = ("PSFC", "PMSL")
ATMOSPHERIC_GROUPS = (TEMPERATURE_GROUP, UWIND_GROUP, VWIND_GROUP, MOISTURE_GROUP, HEIGHT_GROUP,
                      PRESSURE_GROUP)
_NEVER_FILLABLE = frozenset(f for group in ATMOSPHERIC_GROUPS for f in group)

PRESSURE_LEVEL_FIELDS = ("TT", "UU", "VV", "RH", "HGT")  # display/back-compat; see ATMOSPHERIC_GROUPS
SOIL_TEMP_FIELDS = ("ST000010", "ST010040", "ST040100", "ST100200")
SOIL_MOIST_FIELDS = ("SM000010", "SM010040", "SM040100", "SM100200")
SOIL_FIELDS = SOIL_TEMP_FIELDS + SOIL_MOIST_FIELDS

# The fields fill.merge_fill is allowed to supply (architect rule, exhaustive): soil layers, SST,
# SEAICE, SNOW, SNOWH, LANDSEA, SKINTEMP (only when the primary doesn't already have it --
# "missing" already means "not in caps.fields"). PSFC/PMSL are NOT here -- see PRESSURE_GROUP.
FILLABLE_SURFACE_FIELDS = SOIL_FIELDS + ("SST", "SEAICE", "SNOW", "SNOWH", "LANDSEA", "SKINTEMP")

SURFACE_FIELDS = ("PSFC", "PMSL", "SKINTEMP", "LANDSEA", "SOILHGT", "SEAICE", "SNOW", "SNOWH",
                   "SST")

REQUIRED_FIELDS = PRESSURE_LEVEL_FIELDS + SURFACE_FIELDS + SOIL_FIELDS

_HDATE_FMT = "%Y-%m-%d_%H:%M:%S"


def fill_source_for(fieldname: str) -> str:
    """The category fill source for one missing field name: soil (ST*/SM*), SST, snow (SNOW/
    SNOWH), SEAICE and LANDSEA all come from the analog day's own real forcing ("analog-era5");
    SOILHGT (static terrain-of-analysis) comes from geogrid's WPS_GEOG static data instead, since
    it is not carried as a per-time ungrib record in the fictional generator's own output."""
    if fieldname == "SOILHGT":
        return "WPS_GEOG"
    return "analog-era5"


def compute_missing(caps, required: Sequence[str] = REQUIRED_FIELDS,
                     generator_name: str = "<generator>"
                     ) -> Tuple[List[str], Dict[str, str]]:
    """(fields_missing, fill_rules): fields_missing only ever contains FILLABLE_SURFACE_FIELDS
    members (plus SOILHGT, filled from WPS_GEOG, not analog-era5) -- never a 3-D atmospheric or
    atmosphere-derived field, per the architect rule in this module's docstring.

    Raises ScenarioError, naming `generator_name` and the missing physical quantity, if any
    ATMOSPHERIC_GROUPS member is entirely absent from `caps.fields`: that is a real generator
    capability gap, and fill may not paper over it with mismatched-source data."""
    present = set(caps.fields)

    for group in ATMOSPHERIC_GROUPS:
        if not any(member in present for member in group):
            raise ScenarioError(
                "scenario generator %r provides none of %s -- a required atmospheric (or "
                "atmosphere-derived) field; fill may not supply 3-D atmospheric data, so this is "
                "a real generator capability gap, not something to paper over with "
                "mismatched-source fill" % (generator_name, "/".join(group)))

    fillable_required = [f for f in required if f in FILLABLE_SURFACE_FIELDS or f == "SOILHGT"]
    missing = [f for f in fillable_required if f not in present]
    fill_rules = {f: fill_source_for(f) for f in missing}
    return missing, fill_rules


def _parse_hdate(hdate: str) -> datetime.datetime:
    return datetime.datetime.strptime(hdate[:19], _HDATE_FMT)


def merge_fill(primary_files: Iterable, fill_files: Iterable, missing_fields: Iterable[str],
                out_dir) -> List[Path]:
    """Copies `missing_fields`' records from `fill_files` (an analog day's WPS intermediate
    files, one per its own valid time) into copies of `primary_files`, choosing for each primary
    file the fill file with the NEAREST valid time (same hour-of-day preferred on ties/nearby
    matches), and relabelling the copied records' HDATE to the primary file's own HDATE.

    Refuses (ScenarioError) any `missing_fields` entry that names a 3-D atmospheric or
    atmosphere-derived field (any ATMOSPHERIC_GROUPS member) -- enforced here too, not just in
    compute_missing(), so this invariant holds regardless of what a caller passes in.

    Returns the list of new file paths written under `out_dir` (same basenames as
    `primary_files`). `primary_files` files that already carry a field in `missing_fields`
    still get that field's records duplicated from the fill source -- callers should only pass
    fields that are genuinely absent from the primary files (e.g. fill.compute_missing()'s
    output)."""
    primary_files = list(primary_files)
    fill_files = list(fill_files)
    missing_fields = list(missing_fields)

    disallowed = sorted(set(missing_fields) & _NEVER_FILLABLE)
    if disallowed:
        raise ScenarioError(
            "merge_fill: refusing to fill 3-D atmospheric / atmosphere-derived field(s) %s -- "
            "fill may only supply surface/sub-surface fields (%s)"
            % (", ".join(disallowed), ", ".join(FILLABLE_SURFACE_FIELDS)))

    if not fill_files:
        raise ValueError("merge_fill: no fill_files given")

    fill_records_by_file = {p: wi.read_fields(p) for p in fill_files}
    fill_times = {
        p: _parse_hdate(recs[0].hdate) for p, recs in fill_records_by_file.items() if recs
    }
    if not fill_times:
        raise ValueError("merge_fill: none of fill_files contain any records")

    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for primary_path in primary_files:
        primary_records = wi.read_fields(primary_path)
        if not primary_records:
            continue
        primary_hdate = primary_records[0].hdate
        primary_dt = _parse_hdate(primary_hdate)

        def sort_key(fp):
            dt = fill_times[fp]
            same_hour = 0 if dt.hour == primary_dt.hour else 1
            return (same_hour, abs((dt - primary_dt).total_seconds()))

        best_fill = min(fill_times, key=sort_key)
        fill_records = fill_records_by_file[best_fill]

        merged = list(primary_records)
        for fieldname in missing_fields:
            for rec in fill_records:
                if rec.field.strip() == fieldname:
                    merged.append(_replace(rec, hdate=primary_hdate))

        out_path = Path(out_dir) / Path(primary_path).name
        wi.write_fields(out_path, merged)
        out_paths.append(out_path)
    return out_paths
