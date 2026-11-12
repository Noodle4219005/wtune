"""Tests for scenario.wps_intermediate -- pure-numpy WPS intermediate format v5 reader/writer.
Format verified against the real ungrib/src/output.F (see wps_intermediate.py's module
docstring for the exact source lines)."""


import numpy as np
import pytest

import wtune.scenario.wps_intermediate as wi


def _sample_record(nx=3, ny=2, slab=None):
    if slab is None:
        slab = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    return wi.Record(
        hdate="2031-07-15_12:00:00", xfcst=0.0, map_source="TEST", field="TT", units="K",
        desc="Temperature", xlvl=200100.0, nx=nx, ny=ny, iproj=0, startloc="SWCORNER",
        startlat=10.0, startlon=-100.0, deltalat=0.5, deltalon=0.5,
        earth_radius=wi.EARTH_RADIUS_KM, is_wind_earth_rel=True, slab=slab,
    )


def test_unsupported_iproj_rejected(tmp_path):
    rec = _sample_record()
    rec.iproj = 3  # lamcon, not implemented
    with pytest.raises(NotImplementedError):
        wi.write_fields(tmp_path / "FILE:x", [rec])


def test_field_string_too_long_rejected(tmp_path):
    rec = _sample_record()
    rec.field = "TOOLONGFIELDNAME"  # > 9 chars
    with pytest.raises(wi.WpsFormatError):
        wi.write_fields(tmp_path / "FILE:x", [rec])
