"""scenario.wps_intermediate -- pure-numpy reader AND writer for the WPS intermediate file
format, version 5 ("WPS" format, as opposed to the older "SI"/"MM5" formats the same ungrib
`output` routine can also emit).

Format verified against the real WPS source tree checked out at /home/nk/devel/WPS (WRF-model/WPS
on GitHub), specifically the WP branch of `subroutine output` in ungrib/src/output.F:222-246
(the actual byte layout ungrib.exe writes) -- not from documentation, from the Fortran WRITE
statements themselves:

    write(iunit) 5                                                              -- rec 1 (IFV)
    write(iunit) hdate_output, xfcst, map%source, field, units, desc, level,
                 map%nx, map%ny, map%igrid                                      -- rec 2 (header)
    write(iunit) map%startloc, map%lat1, map%lon1, map%dy, map%dx, map%r_earth  -- rec 3 (IPROJ=0/4)
    write(iunit) map%grid_wind                                                  -- rec 4 (wind flag)
    write(iunit) scr2d                                                          -- rec 5 (SLAB)

Each Fortran unformatted sequential record is preceded and followed by a 4-byte record-length
marker (the record's payload length in bytes). This module treats every marker and every numeric
field as BIG-ENDIAN (the convention WPS/ungrib builds use -- e.g. gfortran's
`-fconvert=big-endian`/ifort's `-convert big_endian` -- specifically so intermediate files are
portable across machines of either endianness; it is a build flag, not something the unformatted
I/O format implies on its own).

Only IPROJ (map%igrid) == 0 (regular lat-lon) is implemented, per this package's design scope;
write_fields()/read_fields() raise NotImplementedError for any other projection code.

Level convention (WPS/ungrib's own, e.g. Vtable.GFS's surface-type entries): pressure levels are
Pa; the surface flag level is 200100.0 Pa; the sea-level flag (used by PMSL) is 201300.0 Pa.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

IFV = 5
IPROJ_LATLON = 0

SURFACE_PA = 200100.0
SEALEVEL_PA = 201300.0
EARTH_RADIUS_KM = 6367.470

_LEN_HDATE = 24
_LEN_MAP_SOURCE = 32
_LEN_FIELD = 9
_LEN_UNITS = 25
_LEN_DESC = 46
_LEN_STARTLOC = 8

_BE_I4 = ">i4"
_BE_F4 = ">f4"


class WpsFormatError(Exception):
    """Malformed WPS intermediate file (bad record-length marker, unsupported IPROJ, ...)."""


@dataclass
class Record:
    """One (field, level, valid time) 2-D slab plus its WPS-intermediate header. `slab` is a
    numpy array of shape (ny, nx) in ordinary (row=y, col=x) numpy/C order -- because a
    C-contiguous (ny, nx) array's byte order (x fastest) is IDENTICAL to Fortran's own (nx, ny)
    column-major layout (x fastest too), no transpose is needed on either read or write."""

    hdate: str            # "YYYY-MM-DD_HH:MM:SS" (unpadded logical value; padded to 24 on write)
    xfcst: float
    map_source: str
    field: str
    units: str
    desc: str
    xlvl: float
    nx: int
    ny: int
    iproj: int
    startloc: str
    startlat: float
    startlon: float
    deltalat: float
    deltalon: float
    earth_radius: float
    is_wind_earth_rel: bool
    slab: np.ndarray


def _pad(s: str, n: int, what: str) -> bytes:
    b = s.encode("ascii")
    if len(b) > n:
        raise WpsFormatError("%s %r is %d bytes, longer than the fixed field width %d"
                              % (what, s, len(b), n))
    return b.ljust(n, b" ")


def _write_record(f, payload: bytes) -> None:
    marker = struct.pack(">i", len(payload))
    f.write(marker)
    f.write(payload)
    f.write(marker)


def _read_record(f) -> bytes:
    head = f.read(4)
    if len(head) != 4:
        raise WpsFormatError("truncated file: expected a 4-byte record-length marker, got %d bytes"
                              % len(head))
    (n,) = struct.unpack(">i", head)
    payload = f.read(n)
    if len(payload) != n:
        raise WpsFormatError("truncated record: expected %d payload bytes, got %d" % (n, len(payload)))
    tail = f.read(4)
    if len(tail) != 4:
        raise WpsFormatError("truncated file: missing trailing record-length marker")
    (n2,) = struct.unpack(">i", tail)
    if n2 != n:
        raise WpsFormatError("mismatched record-length markers: leading %d, trailing %d" % (n, n2))
    return payload


def write_fields(path, records: List[Record]) -> None:
    """Writes `records` (one WPS intermediate file, one record per field/level) to `path`."""
    path = Path(path)
    with open(path, "wb") as f:
        for rec in records:
            if rec.iproj != IPROJ_LATLON:
                raise NotImplementedError(
                    "wps_intermediate only implements IPROJ=0 (lat-lon); got IPROJ=%d for field %r"
                    % (rec.iproj, rec.field))

            _write_record(f, struct.pack(">i", IFV))

            payload2 = b"".join((
                _pad(rec.hdate, _LEN_HDATE, "hdate"),
                struct.pack(">f", rec.xfcst),
                _pad(rec.map_source, _LEN_MAP_SOURCE, "map_source"),
                _pad(rec.field, _LEN_FIELD, "field"),
                _pad(rec.units, _LEN_UNITS, "units"),
                _pad(rec.desc, _LEN_DESC, "desc"),
                struct.pack(">f", rec.xlvl),
                struct.pack(">i", rec.nx),
                struct.pack(">i", rec.ny),
                struct.pack(">i", rec.iproj),
            ))
            _write_record(f, payload2)

            payload3 = b"".join((
                _pad(rec.startloc, _LEN_STARTLOC, "startloc"),
                struct.pack(">f", rec.startlat),
                struct.pack(">f", rec.startlon),
                struct.pack(">f", rec.deltalat),
                struct.pack(">f", rec.deltalon),
                struct.pack(">f", rec.earth_radius),
            ))
            _write_record(f, payload3)

            _write_record(f, struct.pack(">i", 1 if rec.is_wind_earth_rel else 0))

            slab = np.ascontiguousarray(rec.slab, dtype=_BE_F4)
            if slab.shape != (rec.ny, rec.nx):
                raise WpsFormatError("field %r: slab shape %s != (ny, nx) = (%d, %d)"
                                      % (rec.field, slab.shape, rec.ny, rec.nx))
            _write_record(f, slab.tobytes(order="C"))


def read_fields(path) -> List[Record]:
    """Reads every record from `path` (a single WPS intermediate PREFIX:HDATE file)."""
    records = []
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            if len(head) != 4:
                raise WpsFormatError("truncated file: partial record-length marker at EOF")
            (n,) = struct.unpack(">i", head)
            payload = f.read(n)
            tail = f.read(4)
            if len(tail) != 4 or struct.unpack(">i", tail)[0] != n:
                raise WpsFormatError("mismatched or missing record-length markers reading IFV record")
            (ifv,) = struct.unpack(">i", payload)
            if ifv != IFV:
                raise WpsFormatError("unsupported IFV=%d (only version 5 'WPS' format is implemented)"
                                      % ifv)

            payload2 = _read_record(f)
            off = 0

            def take(n_bytes):
                nonlocal off
                b = payload2[off:off + n_bytes]
                off += n_bytes
                return b

            hdate = take(_LEN_HDATE).decode("ascii").rstrip()
            (xfcst,) = struct.unpack(">f", take(4))
            map_source = take(_LEN_MAP_SOURCE).decode("ascii").rstrip()
            fieldname = take(_LEN_FIELD).decode("ascii").rstrip()
            units = take(_LEN_UNITS).decode("ascii").rstrip()
            desc = take(_LEN_DESC).decode("ascii").rstrip()
            (xlvl,) = struct.unpack(">f", take(4))
            (nx,) = struct.unpack(">i", take(4))
            (ny,) = struct.unpack(">i", take(4))
            (iproj,) = struct.unpack(">i", take(4))

            if iproj != IPROJ_LATLON:
                raise NotImplementedError(
                    "wps_intermediate only implements IPROJ=0 (lat-lon); got IPROJ=%d for field %r"
                    % (iproj, fieldname))

            payload3 = _read_record(f)
            off3 = 0

            def take3(n_bytes):
                nonlocal off3
                b = payload3[off3:off3 + n_bytes]
                off3 += n_bytes
                return b

            startloc = take3(_LEN_STARTLOC).decode("ascii").rstrip()
            (startlat,) = struct.unpack(">f", take3(4))
            (startlon,) = struct.unpack(">f", take3(4))
            (deltalat,) = struct.unpack(">f", take3(4))
            (deltalon,) = struct.unpack(">f", take3(4))
            (earth_radius,) = struct.unpack(">f", take3(4))

            payload4 = _read_record(f)
            (wind_flag,) = struct.unpack(">i", payload4)

            payload5 = _read_record(f)
            slab = np.frombuffer(payload5, dtype=_BE_F4).reshape(ny, nx).astype(np.float32)

            records.append(Record(
                hdate=hdate, xfcst=float(xfcst), map_source=map_source, field=fieldname,
                units=units, desc=desc, xlvl=float(xlvl), nx=int(nx), ny=int(ny),
                iproj=int(iproj), startloc=startloc, startlat=float(startlat),
                startlon=float(startlon), deltalat=float(deltalat), deltalon=float(deltalon),
                earth_radius=float(earth_radius), is_wind_earth_rel=bool(wind_flag), slab=slab,
            ))
    return records


def relabel_date(src, dst, new_date) -> None:
    """Rewrites every record's HDATE in `src` to `new_date` and writes the result to `dst`.
    Data bytes (SLAB) are byte-for-byte unchanged -- only the 24-byte HDATE header field
    differs between src and dst."""
    records = read_fields(src)
    new_hdate = new_date.strftime("%Y-%m-%d_%H:%M:%S")
    relabelled = [
        Record(**{**rec.__dict__, "hdate": new_hdate})
        for rec in records
    ]
    write_fields(dst, relabelled)
