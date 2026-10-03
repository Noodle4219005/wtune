"""scenario.generators.aifs -- ADAPTER only. AifsGenerator shells out to an external AIFS
inference runner (a separate script, `scenario.generators.aifs_infer`, owned by another agent and
NOT imported here) that writes one netCDF forecast file, then converts that netCDF into WPS
intermediate files with wps_intermediate.write_fields(). Missing soil/SST/snow/seaice fields are
then filled from the `analog` generator's real forcing for the same request (fill.merge_fill).

Fictional dates (2026-09-26 fix): AIFS is initialized from real ERA5, exactly like the analog
generator is initialized from real GFS -- so for a FICTIONAL req.start (any date with no real
forcing, e.g. a future date), generate() must run the AIFS model at the SAME climatological analog
date analog.select_analog_date() picks (real ERA5 exists there), not at req.start itself. Bug this
closes, found running a genuinely-future case (tw3km-2031, req.start=2031-08-20): the previous
version passed req.start straight through as the runner's init_time; ARCO-ERA5's zarr array is
pre-declared out to 2050 but any date that hasn't happened yet is unwritten placeholder, not
missing-key -- aifs_infer.py's ERA5 fetch silently returns an all-NaN slice instead of erroring,
so the "AIFS" forecast was entirely NaN. generate() now runs inference at the analog date and
relabels the resulting records' HDATE to the fictional valid time (wps_intermediate style, same
idea as analog.wi.relabel_date) -- SLAB bytes are the real AIFS forecast, only HDATE changes.

netCDF contract (what the runner's output file must look like):
    dims: (time, level, lat, lon)
    coords:
        time  -- CF datetime (any units num2date can parse)
        level -- pressure, hPa
        lat   -- ascending OR descending, either is accepted
        lon   -- 0..360 OR -180..180, either is accepted
    pressure-level variables (dims: time, level, lat, lon):
        t [K]        -> TT
        u [m s-1]     -> UU
        v [m s-1]     -> VV
        q [kg kg-1]   -> SPECHUMD
        z [m2 s-2]    -> GHT   (divided by 9.80665 to get geopotential HEIGHT in metres)
    single-level variables (dims: time, lat, lon), each written at a fixed WPS XLVL:
        2t  [K]       -> TT       @ SURFACE_PA (200100.0)
        10u [m s-1]   -> UU       @ SURFACE_PA
        10v [m s-1]   -> VV       @ SURFACE_PA
        msl [Pa]      -> PMSL     @ SEALEVEL_PA (201300.0)
        sp  [Pa]      -> PSFC     @ SURFACE_PA
        skt [K]       -> SKINTEMP @ SURFACE_PA
        lsm [0..1]    -> LANDSEA  @ SURFACE_PA
        z_sfc [m2 s-2]-> SOILHGT  @ SURFACE_PA (divided by 9.80665)
    Any variable may be ABSENT from the file; an absent variable's WPS field is reported in
    ForcingSet.fields_missing rather than raising.

Grid handling: cropped to req.bbox with a 5-degree margin on every side; output is always lat
ASCENDING with an SW-corner start point, regardless of the source file's own lat direction or
longitude convention (both -180..180 and 0..360 source longitudes are accepted and normalized to
-180..180 internally).
"""

from __future__ import annotations

import dataclasses
import datetime
import subprocess
from typing import Dict, Optional, Tuple

import numpy as np

from .. import fill as scenario_fill
from .. import wps_intermediate as wi
from ..base import Caps, ForcingSet, ScenarioError, ScenarioGenerator, register, request_to_dict

G = 9.80665
BBOX_MARGIN_DEG = 5.0

# var name in the netCDF -> (WPS field, units)
PRESSURE_VAR_MAP = {
    "t": ("TT", "K"),
    "u": ("UU", "m s-1"),
    "v": ("VV", "m s-1"),
    "q": ("SPECHUMD", "kg kg-1"),
    "z": ("GHT", "m"),
}

# var name in the netCDF -> (WPS field, units, XLVL)
SINGLE_VAR_MAP = {
    "2t": ("TT", "K", wi.SURFACE_PA),
    "10u": ("UU", "m s-1", wi.SURFACE_PA),
    "10v": ("VV", "m s-1", wi.SURFACE_PA),
    "msl": ("PMSL", "Pa", wi.SEALEVEL_PA),
    "sp": ("PSFC", "Pa", wi.SURFACE_PA),
    "skt": ("SKINTEMP", "K", wi.SURFACE_PA),
    "lsm": ("LANDSEA", "proprtn", wi.SURFACE_PA),
    "z_sfc": ("SOILHGT", "m", wi.SURFACE_PA),
}

ALL_FIELDS = tuple(dict.fromkeys(
    [v[0] for v in PRESSURE_VAR_MAP.values()] + [v[0] for v in SINGLE_VAR_MAP.values()]
))


def _normalize_lon180(lon):
    return ((np.asarray(lon, dtype=float) + 180.0) % 360.0) - 180.0


def _crop_1d(coord_sorted, lo, hi):
    mask = (coord_sorted >= lo) & (coord_sorted <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ScenarioError("bbox [%.3f, %.3f] does not intersect coordinate range [%.3f, %.3f]"
                             % (lo, hi, coord_sorted[0], coord_sorted[-1]))
    return int(idx[0]), int(idx[-1]) + 1


def _grid_crop_indices(lat, lon, bbox, margin_deg):
    """(lat_idx, lon_idx, lat_out, lon_out): fancy-index arrays into the ORIGINAL (lat, lon) axes
    that select an ascending-lat, ascending-normalized-lon, bbox+margin-cropped subgrid -- one
    gather that handles descending lat, either longitude convention, and the crop all at once."""
    lat = np.asarray(lat, dtype=float)
    lon_norm = _normalize_lon180(lon)
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon_norm)
    lat_sorted = lat[lat_order]
    lon_sorted = lon_norm[lon_order]

    lat_min, lat_max, lon_min, lon_max = bbox
    la0, la1 = _crop_1d(lat_sorted, lat_min - margin_deg, lat_max + margin_deg)
    lo0, lo1 = _crop_1d(lon_sorted, lon_min - margin_deg, lon_max + margin_deg)

    lat_idx = lat_order[la0:la1]
    lon_idx = lon_order[lo0:lo1]
    return lat_idx, lon_idx, lat_sorted[la0:la1], lon_sorted[lo0:lo1]


def _make_record(field, units, xlvl, valid_time, nx, ny, startlat, startlon, deltalat, deltalon,
                  slab):
    return wi.Record(hdate=valid_time.strftime("%Y-%m-%d_%H:%M:%S"), xfcst=0.0,
                      map_source="AIFS", field=field, units=units, desc=field, xlvl=float(xlvl),
                      nx=nx, ny=ny, iproj=0, startloc="SWCORNER", startlat=float(startlat),
                      startlon=float(startlon), deltalat=float(deltalat),
                      deltalon=float(deltalon), earth_radius=wi.EARTH_RADIUS_KM,
                      is_wind_earth_rel=True, slab=slab)


def netcdf_to_records(nc_path, bbox) -> Dict[int, Tuple[datetime.datetime, list, list]]:
    """Converts one AIFS-contract netCDF file into {time_index: (valid_time, [Record,...],
    [missing_field,...])}. Pure conversion -- no network, no subprocess; this is what
    tests/test_scenario_aifs.py exercises directly against a synthetic file."""
    import netCDF4

    ds = netCDF4.Dataset(nc_path)
    try:
        time_var = ds.variables["time"]
        raw_times = np.atleast_1d(netCDF4.num2date(
            time_var[:], time_var.units, getattr(time_var, "calendar", "standard")))
        times = [datetime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)
                 for t in raw_times]

        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
        level = (np.asarray(ds.variables["level"][:], dtype=float)
                 if "level" in ds.variables else np.array([]))

        lat_idx, lon_idx, lat_out, lon_out = _grid_crop_indices(lat, lon, bbox, BBOX_MARGIN_DEG)
        ny, nx = len(lat_idx), len(lon_idx)
        startlat, startlon = float(lat_out[0]), float(lon_out[0])
        deltalat = float(lat_out[1] - lat_out[0]) if ny > 1 else 1.0
        deltalon = float(lon_out[1] - lon_out[0]) if nx > 1 else 1.0

        out = {}
        for ti, t in enumerate(times):
            records = []
            missing = []

            for var_key, (field, units) in PRESSURE_VAR_MAP.items():
                if var_key not in ds.variables or level.size == 0:
                    missing.append(field)
                    continue
                data = np.asarray(ds.variables[var_key][ti], dtype=float)  # (level, lat, lon)
                for li, lvl_hpa in enumerate(level):
                    slab = data[li]
                    if var_key == "z":
                        slab = slab / G
                    slab_c = slab[np.ix_(lat_idx, lon_idx)].astype(np.float32)
                    records.append(_make_record(field, units, float(lvl_hpa) * 100.0, t, nx, ny,
                                                 startlat, startlon, deltalat, deltalon, slab_c))

            for var_key, (field, units, xlvl) in SINGLE_VAR_MAP.items():
                if var_key not in ds.variables:
                    missing.append(field)
                    continue
                slab = np.asarray(ds.variables[var_key][ti], dtype=float)
                if var_key == "z_sfc":
                    slab = slab / G
                slab_c = slab[np.ix_(lat_idx, lon_idx)].astype(np.float32)
                records.append(_make_record(field, units, xlvl, t, nx, ny, startlat, startlon,
                                             deltalat, deltalon, slab_c))

            out[ti] = (t, records, missing)
        return out
    finally:
        ds.close()


def _select_time_entry(per_time, target_time):
    if len(per_time) == 1:
        return next(iter(per_time.values()))
    return min(per_time.values(), key=lambda entry: abs((entry[0] - target_time).total_seconds()))


def _request_valid_times(start, end, interval_h):
    out = []
    t = start
    step = datetime.timedelta(hours=interval_h)
    while t <= end:
        out.append(t)
        t += step
    return out


def _default_runner(init_time, lead_h, out_nc):
    """Best-effort production default: shells out to the AIFS inference runner (owned by another
    agent; not imported here, only invoked as a subprocess). Real subprocess call; NOT exercised
    by the offline test suite -- pass an explicit `nc_paths=` or `runner=` argument in tests."""
    cmd = ["pixi", "run", "--manifest-path", "envs/ai/pixi.toml", "python", "-m",
           "scenario.generators.aifs_infer", "--init", init_time.strftime("%Y-%m-%dT%H"),
           "--lead-h", str(lead_h), "--out", str(out_nc)]
    subprocess.run(cmd, check=True)


@register
class AifsGenerator(ScenarioGenerator):
    name = "aifs"
    license = "CC BY 4.0 (weights); Apache-2.0 (code)"
    noncommercial = False
    needs_gpu = True
    implemented = True

    def capabilities(self) -> Caps:
        return Caps(fields=ALL_FIELDS, levels_pa=(), has_soil=False, has_sst=False,
                    has_hydrometeors=False, min_interval_h=6, max_lead_h=240, needs_gpu=True,
                    license=self.license, noncommercial=False)

    def generate(self, req, *, runner=None, nc_paths: Optional[Dict] = None,
                 analog_generator=None) -> ForcingSet:
        """nc_paths: optional {valid_time: nc_path} map that skips `runner` entirely (tests pass
        a synthetic file this way) -- when given, it is assumed to already be at the RIGHT
        (real) dates; no analog-date resolution or relabelling is applied on top of it.
        runner(init_time, lead_h, out_nc_path): default shells out to the AIFS inference runner
        (see _default_runner); called once per requested valid time, at the REAL analog date
        (see module docstring), when nc_paths is not given. analog_generator: an
        already-constructed `analog` ScenarioGenerator instance -- when given, its ForcingSet for
        the same bbox/window is used to fill soil/SST/snow/seaice/LANDSEA fields AIFS's own
        variable map never supplies (fill.merge_fill)."""
        from .analog import select_analog_date

        valid_times = _request_valid_times(req.start, req.end, req.interval_h)
        req.out_dir.mkdir(parents=True, exist_ok=True)

        if nc_paths is None:
            analog_date = req.analog_date or select_analog_date(req.start)
            real_times = [analog_date + (t - req.start) for t in valid_times]
            runner_fn = runner or _default_runner
            nc_paths = {}
            for t, real_t in zip(valid_times, real_times):
                lead_h = int((real_t - analog_date).total_seconds() // 3600)
                out_nc = req.out_dir / ("aifs_%s.nc" % real_t.strftime("%Y%m%d%H"))
                runner_fn(analog_date, lead_h, out_nc)
                nc_paths[t] = out_nc
        else:
            analog_date = None
            real_times = list(valid_times)

        prefix = "AIFS"
        out_files = []
        seen_missing = set()
        for t, real_t in zip(valid_times, real_times):
            per_time = netcdf_to_records(nc_paths[t], req.bbox)
            _, records, missing = _select_time_entry(per_time, real_t)
            if real_t != t:
                fictional_hdate = t.strftime("%Y-%m-%d_%H:%M:%S")
                records = [dataclasses.replace(r, hdate=fictional_hdate) for r in records]
            seen_missing.update(missing)
            dst_path = req.out_dir / ("%s:%s" % (prefix, t.strftime("%Y-%m-%d_%H")))
            wi.write_fields(dst_path, records)
            out_files.append(dst_path)

        fields_present = sorted(set(ALL_FIELDS) - seen_missing)
        cropped_caps = dataclasses.replace(self.capabilities(), fields=tuple(fields_present))
        analog_missing, fill_rules = scenario_fill.compute_missing(cropped_caps,
                                                                     generator_name=self.name)
        fields_missing = sorted(set(seen_missing) | set(analog_missing))

        if analog_generator is not None and analog_missing:
            analog_req = dataclasses.replace(req, out_dir=req.out_dir / "analog_fill",
                                              analog_date=analog_date)
            analog_set = analog_generator.generate(analog_req)
            merged_dir = req.out_dir / "merged"
            out_files = scenario_fill.merge_fill(out_files, analog_set.files, analog_missing,
                                                  merged_dir)

        provenance = {
            "generator": self.name,
            "model": "AIFS",
            "version": None,
            "weights_sha256": None,
            "license": self.license,
            "analog_date": analog_date,
            "created_utc": datetime.datetime.utcnow(),
            "request": request_to_dict(req),
        }
        return ForcingSet(files=out_files, prefix=prefix, valid_times=valid_times,
                           fields_present=fields_present, fields_missing=fields_missing,
                           fill_rules=fill_rules, provenance=provenance)
