"""scenario.fill -- given a generator's Caps, work out which fields a real metgrid run needs
that the generator does not itself produce, where each missing field would come from, and how to
merge an analog day's real forcing into a primary ForcingSet's files to cover the gap.

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

from . import wps_intermediate as wi

PRESSURE_LEVEL_FIELDS = ("TT", "UU", "VV", "RH", "HGT")
SOIL_TEMP_FIELDS = ("ST000010", "ST010040", "ST040100", "ST100200")
SOIL_MOIST_FIELDS = ("SM000010", "SM010040", "SM040100", "SM100200")
SOIL_FIELDS = SOIL_TEMP_FIELDS + SOIL_MOIST_FIELDS
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


def compute_missing(caps, required: Sequence[str] = REQUIRED_FIELDS
                     ) -> Tuple[List[str], Dict[str, str]]:
    """(fields_missing, fill_rules) for a generator with capabilities `caps` against the
    `required` field list (defaults to this module's REQUIRED_FIELDS). fields_missing preserves
    `required`'s order; fill_rules maps each missing field to fill_source_for(field)."""
    present = set(caps.fields)
    missing = [f for f in required if f not in present]
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

    Returns the list of new file paths written under `out_dir` (same basenames as
    `primary_files`). `primary_files` files that already carry a field in `missing_fields`
    still get that field's records duplicated from the fill source -- callers should only pass
    fields that are genuinely absent from the primary files (e.g. fill.compute_missing()'s
    output)."""
    primary_files = list(primary_files)
    fill_files = list(fill_files)
    missing_fields = list(missing_fields)
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
