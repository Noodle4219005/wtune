"""Tests for scenario.fill: missing-field computation against a generator's Caps, and
merge_fill's record copy + HDATE relabel + nearest-time matching."""

import datetime

import numpy as np
import pytest

import scenario.fill as fill
import scenario.wps_intermediate as wi
from scenario.base import Caps


def _caps(fields):
    return Caps(fields=tuple(fields), levels_pa=(), has_soil=False, has_sst=False,
                has_hydrometeors=False, min_interval_h=6, max_lead_h=240, needs_gpu=False,
                license="test", noncommercial=False)


def test_compute_missing_caps_without_soil():
    caps = _caps(fill.PRESSURE_LEVEL_FIELDS + ("PSFC", "PMSL", "SKINTEMP", "LANDSEA",
                                                 "SOILHGT", "SEAICE", "SNOW", "SNOWH", "SST"))
    missing, rules = fill.compute_missing(caps)
    assert set(missing) == set(fill.SOIL_FIELDS)
    for f in fill.SOIL_FIELDS:
        assert rules[f] == "analog-era5"


def test_compute_missing_static_terrain_rule():
    caps = _caps(fill.PRESSURE_LEVEL_FIELDS + fill.SOIL_FIELDS + ("PSFC", "PMSL", "SKINTEMP",
                 "LANDSEA", "SEAICE", "SNOW", "SNOWH", "SST"))
    missing, rules = fill.compute_missing(caps)
    assert missing == ["SOILHGT"]
    assert rules["SOILHGT"] == "WPS_GEOG"


def test_compute_missing_nothing_missing():
    caps = _caps(fill.REQUIRED_FIELDS)
    missing, rules = fill.compute_missing(caps)
    assert missing == []
    assert rules == {}


def _write_soil_record(path, hdate, field, value):
    slab = np.full((2, 2), value, dtype=np.float32)
    rec = wi.Record(hdate=hdate, xfcst=0.0, map_source="TEST", field=field, units="fraction",
                     desc=field, xlvl=wi.SURFACE_PA, nx=2, ny=2, iproj=0, startloc="SWCORNER",
                     startlat=0.0, startlon=0.0, deltalat=1.0, deltalon=1.0,
                     earth_radius=wi.EARTH_RADIUS_KM, is_wind_earth_rel=True, slab=slab)
    wi.write_fields(path, [rec])


def test_merge_fill_copies_soil_with_relabel_and_nearest_time(tmp_path):
    primary_dir = tmp_path / "primary"
    fill_dir = tmp_path / "fill"
    out_dir = tmp_path / "merged"
    primary_dir.mkdir()
    fill_dir.mkdir()

    primary_path = primary_dir / "FILE:2031-07-15_12"
    _write_soil_record(primary_path, "2031-07-15_12:00:00", "TT", 300.0)

    # Two candidate fill files: one 3h away (different hour-of-day), one 24h away but SAME
    # hour-of-day as the primary -- nearest-time-preferring-same-hour must pick the 24h one.
    near_diff_hour = fill_dir / "FILE:2025-08-29_09"
    _write_soil_record(near_diff_hour, "2025-08-29_09:00:00", "ST000010", 280.0)
    same_hour_far = fill_dir / "FILE:2025-08-28_12"
    _write_soil_record(same_hour_far, "2025-08-28_12:00:00", "ST000010", 281.5)

    out_paths = fill.merge_fill([primary_path], [near_diff_hour, same_hour_far],
                                 ["ST000010"], out_dir)
    assert len(out_paths) == 1
    merged = wi.read_fields(out_paths[0])
    fields = {r.field: r for r in merged}
    assert "TT" in fields and "ST000010" in fields
    assert fields["ST000010"].hdate == "2031-07-15_12:00:00"  # relabelled to primary's HDATE
    assert np.allclose(fields["ST000010"].slab, 281.5)  # picked the same-hour-of-day file


def _write_pressure_record(path, hdate, field, xlvl_pa, value, *, nx=2, ny=2):
    slab = np.full((ny, nx), value, dtype=np.float32)
    rec = wi.Record(hdate=hdate, xfcst=0.0, map_source="TEST", field=field, units="test",
                     desc=field, xlvl=float(xlvl_pa), nx=nx, ny=ny, iproj=0, startloc="SWCORNER",
                     startlat=0.0, startlon=0.0, deltalat=1.0, deltalon=1.0,
                     earth_radius=wi.EARTH_RADIUS_KM, is_wind_earth_rel=True, slab=slab)
    return rec


def _aifs_like_caps():
    # AIFS's own netCDF-to-WPS contract: moisture as SPECHUMD, height as GHT (see
    # scenario/generators/aifs.py PRESSURE_VAR_MAP) -- the OTHER member of MOISTURE_GROUP /
    # HEIGHT_GROUP from what Vtable.GFS-based sources use.
    return _caps(("TT", "UU", "VV", "SPECHUMD", "GHT", "PSFC", "PMSL", "SKINTEMP", "LANDSEA",
                  "SOILHGT"))


def test_compute_missing_recognizes_spechumd_ght_as_moisture_height_and_never_lists_them(tmp_path):
    """Architect rule (2026-09-26 aifs C3/C4b review): a primary using SPECHUMD/GHT must not have
    RH/HGT reported as missing (fillable) merely because it names the same physical quantity
    differently than Vtable.GFS-based sources do."""
    missing, rules = fill.compute_missing(_aifs_like_caps(), generator_name="aifs")
    assert "RH" not in missing and "HGT" not in missing
    assert set(missing) == set(fill.SOIL_FIELDS) | {"SST", "SEAICE", "SNOW", "SNOWH"}
    assert all(rules[f] == "analog-era5" for f in missing)


def test_compute_missing_raises_when_primary_lacks_an_entire_atmospheric_group(tmp_path):
    """No member of MOISTURE_GROUP (RH or SPECHUMD) at all -> a real capability gap, not
    something fill may paper over -- ScenarioError, not a silently-fillable field."""
    caps = _caps(("TT", "UU", "VV", "GHT", "PSFC", "PMSL", "SKINTEMP", "LANDSEA", "SOILHGT"))
    with pytest.raises(fill.ScenarioError, match="RH/SPECHUMD"):
        fill.compute_missing(caps, generator_name="some-generator")


def test_merge_fill_never_supplies_moisture_or_height_even_when_fill_source_has_them(tmp_path):
    """End-to-end: an AIFS-like primary (SPECHUMD/GHT) merged against a GFS-like fill source
    (which -- realistically -- also carries RH/HGT) must come out with NO RH/HGT records at all.
    Root cause this guards against: WPS's own metgrid renames HGT -> GHT on read
    (metgrid/src/read_met_module.F:93 in the checked-out WPS-4.6.0 source), so a merged file
    carrying both AIFS's GHT and analog's HGT would give metgrid two competing GHT records per
    level (last one read wins) -- silently splicing real analog heights/moisture into what must
    be an AIFS-only atmospheric state."""
    primary_dir = tmp_path / "primary"
    fill_dir = tmp_path / "fill"
    primary_dir.mkdir()
    fill_dir.mkdir()

    primary_path = primary_dir / "AIFS:2031-07-15_12"
    wi.write_fields(primary_path, [
        _write_pressure_record(primary_path, "2031-07-15_12:00:00", "TT", 85000, 290.0),
        _write_pressure_record(primary_path, "2031-07-15_12:00:00", "SPECHUMD", 85000, 0.01),
        _write_pressure_record(primary_path, "2031-07-15_12:00:00", "GHT", 85000, 1500.0),
    ])

    fill_path = fill_dir / "FILE:2031-07-15_12"
    soil_rec = wi.Record(hdate="2031-07-15_12:00:00", xfcst=0.0, map_source="TEST",
                          field="ST000010", units="fraction", desc="ST000010", xlvl=wi.SURFACE_PA,
                          nx=2, ny=2, iproj=0, startloc="SWCORNER", startlat=0.0, startlon=0.0,
                          deltalat=1.0, deltalon=1.0, earth_radius=wi.EARTH_RADIUS_KM,
                          is_wind_earth_rel=True, slab=np.full((2, 2), 280.0, dtype=np.float32))
    wi.write_fields(fill_path, [
        _write_pressure_record(fill_path, "2031-07-15_12:00:00", "RH", 85000, 55.0),
        _write_pressure_record(fill_path, "2031-07-15_12:00:00", "HGT", 85000, 1480.0),
        soil_rec,
    ])

    caps = _aifs_like_caps()
    missing, _ = fill.compute_missing(caps, generator_name="aifs")
    assert "RH" not in missing and "HGT" not in missing

    out_paths = fill.merge_fill([primary_path], [fill_path], missing, tmp_path / "merged")
    merged = wi.read_fields(out_paths[0])
    merged_fields = {r.field.strip() for r in merged}
    assert "RH" not in merged_fields
    assert "HGT" not in merged_fields
    assert "SPECHUMD" in merged_fields and "GHT" in merged_fields  # AIFS's own kept, untouched
    assert "ST000010" in merged_fields  # the actually-fillable soil field still came through

    # defense in depth: merge_fill itself refuses RH/HGT even if a caller passes them directly.
    with pytest.raises(fill.ScenarioError, match="RH"):
        fill.merge_fill([primary_path], [fill_path], ["RH"], tmp_path / "merged2")


def test_merge_fill_requires_fill_files(tmp_path):
    primary_path = tmp_path / "FILE:2031-07-15_12"
    _write_soil_record(primary_path, "2031-07-15_12:00:00", "TT", 300.0)
    with pytest.raises(ValueError):
        fill.merge_fill([primary_path], [], ["ST000010"], tmp_path / "merged")
