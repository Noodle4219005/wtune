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


def test_merge_fill_requires_fill_files(tmp_path):
    primary_path = tmp_path / "FILE:2031-07-15_12"
    _write_soil_record(primary_path, "2031-07-15_12:00:00", "TT", 300.0)
    with pytest.raises(ValueError):
        fill.merge_fill([primary_path], [], ["ST000010"], tmp_path / "merged")
