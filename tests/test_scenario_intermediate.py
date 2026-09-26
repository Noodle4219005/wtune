"""Tests for scenario.wps_intermediate -- pure-numpy WPS intermediate format v5 reader/writer.
Format verified against the real ungrib/src/output.F (see wps_intermediate.py's module
docstring for the exact source lines)."""

import datetime
import struct

import numpy as np
import pytest

import scenario.wps_intermediate as wi


def _sample_record(nx=3, ny=2, slab=None):
    if slab is None:
        slab = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    return wi.Record(
        hdate="2031-07-15_12:00:00", xfcst=0.0, map_source="TEST", field="TT", units="K",
        desc="Temperature", xlvl=200100.0, nx=nx, ny=ny, iproj=0, startloc="SWCORNER",
        startlat=10.0, startlon=-100.0, deltalat=0.5, deltalon=0.5,
        earth_radius=wi.EARTH_RADIUS_KM, is_wind_earth_rel=True, slab=slab,
    )


def test_roundtrip_bit_exact(tmp_path):
    rec = _sample_record()
    path = tmp_path / "FILE:2031-07-15_12"
    wi.write_fields(path, [rec])
    back = wi.read_fields(path)
    assert len(back) == 1
    got = back[0]
    assert got.hdate == rec.hdate
    assert got.field == rec.field
    assert got.units == rec.units
    assert got.desc == rec.desc
    assert got.map_source == rec.map_source
    assert got.startloc == rec.startloc
    assert got.xlvl == pytest.approx(rec.xlvl)
    assert got.startlat == pytest.approx(rec.startlat)
    assert got.startlon == pytest.approx(rec.startlon)
    assert got.deltalat == pytest.approx(rec.deltalat)
    assert got.deltalon == pytest.approx(rec.deltalon)
    assert got.earth_radius == pytest.approx(rec.earth_radius)
    assert got.iproj == rec.iproj
    assert got.nx == rec.nx
    assert got.ny == rec.ny
    assert got.is_wind_earth_rel == rec.is_wind_earth_rel
    assert np.array_equal(got.slab, rec.slab)


def test_raw_bytes_ifv_marker(tmp_path):
    rec = _sample_record()
    path = tmp_path / "FILE:2031-07-15_12"
    wi.write_fields(path, [rec])
    raw = path.read_bytes()
    # first 4 bytes: big-endian int32 record-length marker for the IFV record (payload is a
    # single int32 -> length 4); next 4 bytes: the IFV value itself, 5.
    assert raw[:4] == struct.pack(">i", 4)
    assert raw[4:8] == struct.pack(">i", 5)


def test_relabel_date_keeps_slab_bytes_identical(tmp_path):
    rec = _sample_record()
    src = tmp_path / "FILE:2031-07-15_12"
    dst = tmp_path / "FILE:2099-01-01_00"
    wi.write_fields(src, [rec])

    new_date = datetime.datetime(2099, 1, 1, 0)
    wi.relabel_date(src, dst, new_date)

    src_recs = wi.read_fields(src)
    dst_recs = wi.read_fields(dst)
    assert dst_recs[0].hdate == "2099-01-01_00:00:00"
    assert src_recs[0].hdate == "2031-07-15_12:00:00"
    assert np.array_equal(dst_recs[0].slab, src_recs[0].slab)

    # byte-for-byte: the SLAB record's raw payload bytes are identical between src and dst.
    def slab_payload(path):
        raw = path.read_bytes()
        # walk records: IFV(4+4+4), header(4+156+4), proj(4+28+4), wind(4+4+4), then SLAB.
        off = 4 + 4 + 4  # IFV record with markers
        off += 4 + 156 + 4  # header record with markers
        off += 4 + 28 + 4  # projection record with markers
        off += 4 + 4 + 4  # wind-flag record with markers
        (n,) = struct.unpack(">i", raw[off:off + 4])
        return raw[off + 4:off + 4 + n]

    assert slab_payload(src) == slab_payload(dst)


def test_multiple_records_and_levels(tmp_path):
    slab_a = np.zeros((2, 3), dtype=np.float32)
    slab_b = np.ones((2, 3), dtype=np.float32) * 42.0
    rec_a = _sample_record(slab=slab_a)
    rec_b = wi.Record(**{**rec_a.__dict__, "xlvl": 85000.0, "slab": slab_b})
    path = tmp_path / "FILE:2031-07-15_12"
    wi.write_fields(path, [rec_a, rec_b])
    back = wi.read_fields(path)
    assert len(back) == 2
    assert back[0].xlvl == pytest.approx(200100.0)
    assert back[1].xlvl == pytest.approx(85000.0)
    assert np.array_equal(back[1].slab, slab_b)


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
