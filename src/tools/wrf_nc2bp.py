#!/usr/bin/env python3
"""Convert WRF input netCDF files to BP files readable by WRF io_form=14.

This is deliberately a serial, one-time-slice-at-a-time converter.  It does
not use MPI and it never materialises the complete input file in memory.

Provenance: restored (layout logic unchanged) from ``git show
8ba1629:tools/wrf_nc2bp.py`` after the 2026-09-24 src/ layout refactor lost
it.  That version was validated on f1: conversion job 1071692 and regression
gate 1072441 (WRF with BP5 inputs and BP5 history, zero netCDF I/O inside
WRF, bit-identical to the netCDF path via diffwrf).  Library callers should
go through ``wtune_bp.nc2bp`` / ``wtune_bp.convert_inputs`` (atomic output,
manifest cache); the CLI front end is ``wtune convert nc2bp``.

Layout decisions are taken from the WRF ADIOS2 writer/reader in
``WRF/external/io_adios2`` (the line numbers below refer to the supplied
``intel-adios2-dmsm`` source tree):

* ``wrf_io.F90:1254-1272`` defines ``_DIM_DateStrLen``, a scalar string
  ``Times`` variable, and its ``Dims = [DateStrLen, Time]`` attribute.
* ``wrf_io.F90:2295-2523`` defines ordinary variables with a global shape,
  stores time as BP steps, and writes ``Dims`` in WRF's reversed netCDF
  dimension order followed by ``Time``.  The Python ADIOS2 binding exposes
  the resulting shape in netCDF order (as verified against a real WRF BP
  file), so the shape passed here is the netCDF spatial shape while the
  ``Dims`` attribute is explicitly reversed.
* ``wrf_io.F90:2530-2542`` writes ``FieldType`` and canonical
  ``MemoryOrder`` metadata.  The converter preserves the source attribute
  spelling/padding where present; WRF's reader obtains these values from the
  registry/call site, while the BP ``FieldType`` is required by
  ``wrf_io.F90:2733-2784``.
* ``wrf_io.F90:327-389`` and ``wrf_io.F90:1374-1411`` show that every input
  time is one BP step and that ``Times`` is written once per step.
* ``wrf_io.F90:949-1033`` opens the exact supplied path, counts BP steps,
  reads ``Times``, and ignores ordinary variables whose names start with
  ``md___``.  Such variables are nevertheless retained here because the
  boundary reader directly requests the ``md___thisbdytime...`` and
  ``md___nextbdytime...`` variables.
* ``ext_adios2_put_var_td.code:98-137`` and
  ``ext_adios2_put_var_ti.code:91-110`` establish ``_DIM_*``, ``Dims``, and
  per-variable attribute encodings.  Character metadata emitted by WRF is a
  one-element string attribute; numeric metadata remains typed numeric data.
* ``ext_adios2_put_dom_ti.code:82-87`` and ``wrf_io.F90:1714-1836`` establish
  that netCDF global attributes correspond to root BP attributes.  Logical
  values in the WRF build are represented as integer 0/1 values.

The source writer leaves the engine at the ADIOS2 default, which is BP5 in
the target ADIOS2 2.10.2 stack; only the optional burst-buffer path forces
BP4 (``wrf_io.F90:1306-1315``).  WRF's reader, however, opens inputs in
``adios2_mode_read`` and inquires ``Times`` before any BeginStep
(``wrf_io.F90:949-959``), which BP5 refuses (code 4); this CLI therefore
defaults to BP4 (scc, 2026-10-03).  Compression is intentionally not used:
the supplied Python wheel has no Blosc support, and uncompressed BP is fully
readable by WRF.

The WRF source does not interpret boundary suffixes in this directory.  The
converter consequently keeps ``_BXS/_BXE/_BYS/_BYE``, ``_BTXS...`` and all
``md___`` names byte-for-byte as netCDF variable names, applying the same
ordinary field/step contract.

Multi-record inputs (wrfbdy, wrflowinp, wrffdda): the layout written here is
structurally the one WRF itself writes (scc 2026-10-03: real.exe io_form 14
BP4 wrfbdy vs this converter on the same real.exe netCDF wrfbdy -- same
variables, types, shapes, step count, per-step Times and md___ strings,
attribute set).  WRF nevertheless cannot read such a file correctly:
``FieldIO`` (``wrf_io.F90:652-660``) calls ``adios2_set_step_selection`` only
when ``IO == 'write'``, and ``ext_adios2_read_field`` never sets it either, so
every field Get returns BP step 0 whatever record the DateStr names (only
``Times`` and the ``md___`` strings are step-selected, ``wrf_io.F90:978`` and
``ext_adios2_get_var_td.code:119``).  WRF fed with its own BP4 wrfbdy is
bit-identical to WRF fed with a netCDF wrfbdy whose records 2..N carry record
1's fields (13/13 hourly frames, scc job 2780) and departs from the netCDF run
one hour after the first boundary update.  No single-file layout can avoid
this; the converter warns on stderr for any multi-record single-file output.

The way around it without touching WRF is one record per file
(``--split-prefix``): with ``multi_bdy_files = .true.`` WRF builds the boundary
file name from ``bdy_inname`` with <date> = the current clock time, opens it,
reads it and closes it every boundary interval
(``share/mediation_integrate.F:1451-1516``); a dated ``auxinput4_inname``
plus ``frames_per_auxinput4 = 1`` does the same for sst_update (open_aux_u,
``share/mediation_integrate.F:1613-1682``, closes after
``frames_per_auxinput4`` frames, ``inc/med_auxinput_in_closes.inc``).  Every
read then hits the file's only step.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable

import adios2
import numpy as np
from netCDF4 import Dataset


DEFAULT_ENGINE = "BP4"
WRF_DATE_STR_LEN = 19
WRF_MAX_TIMES = 60000


def _is_char_dtype(dtype: np.dtype) -> bool:
    return np.dtype(dtype).kind in ("S", "U", "O")


def _attribute_value(value: Any) -> Any:
    """Make a netCDF attribute acceptable to the ADIOS2 Python binding."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    array = np.asarray(value)
    if array.ndim == 0:
        return array[()]
    return np.ascontiguousarray(array)


def _variable_attribute_value(name: str, value: Any) -> Any:
    """Match WRF's per-variable character-attribute encoding.

    ``put_var_ti_char`` calls the Fortran ADIOS2 binding with a count of one,
    which appears as a one-element string array to FileReader.  MemoryOrder
    is written separately by the field writer as a scalar character value.
    """
    value = _attribute_value(value)
    if isinstance(value, str) and name not in {"MemoryOrder", "Dims"}:
        return [value]
    return value


def _field_type_for_dtype(dtype: np.dtype) -> np.int32:
    """Infer WRF's field type only when a non-character input omits it."""
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float32"):
        return np.int32(104)  # WRF_REAL
    if dtype == np.dtype("float64"):
        return np.int32(105)  # WRF_DOUBLE
    if dtype == np.dtype("int32"):
        return np.int32(106)  # WRF_INTEGER
    raise ValueError(
        f"cannot infer a WRF FieldType for dtype {dtype}; add a FieldType "
        "attribute to the netCDF variable"
    )


def _decode_char(value: Any) -> str:
    """Decode one fixed-width netCDF character record without loading fields."""
    array = np.asarray(value)
    if array.dtype.kind == "S":
        raw = array.tobytes(order="C")
        return raw.decode("latin-1")
    if array.dtype.kind == "U":
        return "".join(str(x) for x in array.reshape(-1))
    if array.dtype.kind == "O":
        flat = array.reshape(-1)
        return "".join(
            x.decode("latin-1") if isinstance(x, bytes) else str(x) for x in flat
        )
    raise TypeError(f"unsupported character variable dtype {array.dtype}")


@dataclass
class VariableSpec:
    name: str
    dimensions: tuple[str, ...]
    spatial_dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    source_dtype: np.dtype
    is_string: bool
    has_time: bool
    source_variable: Any
    variable: Any = None

    @property
    def bp_shape(self) -> tuple[int, ...]:
        # ADIOS2's Python API exposes a Fortran-defined WRF variable in the
        # same order as the netCDF spatial slice (see the real BP probe in the
        # test/implementation notes).  A scalar field is represented by the
        # shape-one variable used by wrf_io.F90:2480-2489.
        return self.shape if self.shape else (1,)

    @property
    def dims_attribute(self) -> list[str]:
        return list(reversed(self.spatial_dimensions)) + ["Time"]


def _time_dimension(ds: Dataset) -> str | None:
    if "Time" in ds.dimensions:
        return "Time"
    return None


def _make_specs(ds: Dataset, time_name: str | None) -> list[VariableSpec]:
    specs: list[VariableSpec] = []
    for name, source in ds.variables.items():
        dimensions = tuple(source.dimensions)
        if time_name is not None and time_name in dimensions:
            if dimensions[0] != time_name:
                raise ValueError(
                    f"{name}: WRF input variables must have Time as the first "
                    f"dimension, got {dimensions}"
                )
            spatial_dimensions = dimensions[1:]
            has_time = True
        else:
            spatial_dimensions = dimensions
            has_time = False
        shape = tuple(len(ds.dimensions[d]) for d in spatial_dimensions)
        specs.append(
            VariableSpec(
                name=name,
                dimensions=dimensions,
                spatial_dimensions=spatial_dimensions,
                shape=shape,
                source_dtype=np.dtype(source.dtype),
                is_string=_is_char_dtype(source.dtype),
                has_time=has_time,
                source_variable=source,
            )
        )
    return specs


def _define_dimensions(io: Any, ds: Dataset) -> None:
    # Time is implicit in the BP step count.  WRF always defines DateStrLen
    # before any field; defining all physical netCDF dimensions is harmless,
    # preserves unused dimensions for the reverse converter, and gives the
    # same integer _DIM_* representation to every dimension used by a field.
    source_global_names = set(ds.ncattrs())
    for name, dimension in ds.dimensions.items():
        if name == "Time":
            continue
        length = len(dimension)
        internal_name = f"_DIM_{name}"
        if internal_name not in source_global_names:
            io.define_attribute(internal_name, np.int32(length))


def _define_variable(io: Any, ds: Dataset, spec: VariableSpec) -> None:
    if spec.name == "Times" or spec.is_string:
        spec.variable = io.define_variable(spec.name, "")
    else:
        # A zero-length probe supplies the exact dtype without allocating the
        # potentially multi-gigabyte field during metadata definition.
        probe = np.empty(0, dtype=spec.source_dtype)
        spec.variable = io.define_variable(
            spec.name,
            probe,
            shape=list(spec.bp_shape),
            start=[0] * len(spec.bp_shape),
            count=list(spec.bp_shape),
            is_constant_dims=True,
        )
    io.define_attribute("Dims", spec.dims_attribute, variable_name=spec.name)

    attrs = {attr: spec.source_variable.getncattr(attr) for attr in spec.source_variable.ncattrs()}
    # Dims is generated from the netCDF dimensions and must not be copied from
    # an unrelated user attribute.  It is not a netCDF WRF attribute anyway.
    attrs.pop("Dims", None)
    if not spec.is_string and "FieldType" not in attrs:
        attrs["FieldType"] = _field_type_for_dtype(spec.source_dtype)
    for attr_name, value in attrs.items():
        if attr_name == "_FillValue":
            # WRF's io_form=2 input fields are no-fill fields.  ADIOS2's
            # reader does not consume this netCDF implementation detail, and
            # the vendored reverse converter cannot set it after creation.
            continue
        io.define_attribute(
            attr_name,
            _variable_attribute_value(attr_name, value),
            variable_name=spec.name,
        )


def _source_string(spec: VariableSpec, step: int | None) -> str:
    source = spec.source_variable
    if step is None:
        value = source[...]
    else:
        value = source[step, ...]
    return _decode_char(value)


def _source_numeric(spec: VariableSpec, step: int | None) -> np.ndarray:
    source = spec.source_variable
    if step is None:
        value = source[...]
    else:
        value = source[step, ...]
    array = np.asarray(value, dtype=spec.source_dtype)
    if not spec.shape:
        return np.asarray(array, dtype=spec.source_dtype).reshape(1)
    return np.ascontiguousarray(array)


def _write_bp(
    ds: Dataset,
    specs: Iterable[VariableSpec],
    output: str,
    engine_name: str,
    records: Iterable[int],
) -> None:
    """Write one BP step per source Time record index in ``records``."""
    adios = adios2.Adios()
    io = adios.declare_io("wrf_nc2bp")
    io.set_engine(engine_name)
    _define_dimensions(io, ds)

    for spec in specs:
        _define_variable(io, ds, spec)

    for name in ds.ncattrs():
        io.define_attribute(name, _attribute_value(ds.getncattr(name)))

    engine = io.open(output, adios2.string_to_mode("w")[0])
    try:
        for record in records:
            engine.begin_step()
            for spec in specs:
                source_step = record if spec.has_time else None
                if spec.is_string:
                    value = _source_string(spec, source_step)
                    # ADIOS2's WRF string variables are scalar per BP step;
                    # DateStrLen remains metadata in _DIM_DateStrLen/Dims.
                    engine.put(spec.variable, value, adios2.bindings.Mode.Sync)
                else:
                    value = _source_numeric(spec, source_step)
                    engine.put(spec.variable, value, adios2.bindings.Mode.Sync)
            engine.end_step()
    finally:
        engine.close()


def record_file_name(prefix: str, date: str) -> str:
    """Per-record BP name: what WRF's construct_filename2a (share/module_io_domain.F:194-252)
    makes of ``<prefix-with-domain>_<date>.bp`` with <date> = the clock's current_timestr
    (19 characters, colons kept: maybe_remove_colons only acts with nocolons=.true.)."""
    if len(date) != WRF_DATE_STR_LEN:
        raise ValueError(f"bad WRF date string {date!r}")
    return f"{prefix}_{date}.bp"


def convert(
    input_path: str,
    output_path: str,
    engine_name: str = DEFAULT_ENGINE,
    split_prefix: str | None = None,
) -> list[str]:
    """Convert ``input_path``.  Without ``split_prefix``: one BP (one step per Time record)
    at ``output_path``.  With it: ``output_path`` becomes a directory holding one single-step
    BP per record, ``record_file_name(split_prefix, Times[k])``, each carrying record k's
    fields, Times and md___ strings unchanged (the per-record layout for WRF's
    multi_bdy_files / dated auxinput names, see the module docstring).  Returns the written
    BP paths."""
    if engine_name not in {"BP4", "BP5"}:
        raise ValueError(f"unsupported engine {engine_name!r}; use BP4 or BP5")
    if os.path.exists(output_path):
        raise FileExistsError(f"output already exists: {output_path}")

    with Dataset(input_path, "r") as ds:
        ds.set_auto_mask(False)
        ds.set_auto_scale(False)
        time_name = _time_dimension(ds)
        nsteps = len(ds.dimensions[time_name]) if time_name else 1
        if nsteps < 1:
            raise ValueError("input has no Time records")
        if nsteps > WRF_MAX_TIMES:
            raise ValueError(f"input has {nsteps} Time records; WRF limit is {WRF_MAX_TIMES}")
        if "Times" not in ds.variables:
            raise ValueError("WRF ADIOS2 input requires a Times variable")
        times = ds.variables["Times"]
        if times.ndim != 2 or times.dimensions[0] != "Time":
            raise ValueError(f"Times must have dimensions (Time, DateStrLen), got {times.dimensions}")
        if len(times.dimensions) != 2 or len(ds.dimensions[times.dimensions[1]]) != WRF_DATE_STR_LEN:
            raise ValueError("Times must use a 19-character DateStrLen dimension")
        if split_prefix is not None:
            specs = _make_specs(ds, time_name)
            dates = [_decode_char(times[k, :]) for k in range(nsteps)]
            names = [record_file_name(split_prefix, d) for d in dates]
            if len(set(names)) != len(names):
                raise ValueError(f"{input_path}: duplicate Times, cannot split per record")
            os.makedirs(output_path)
            written = []
            for k, name in enumerate(names):
                path = os.path.join(output_path, name)
                _write_bp(ds, specs, path, engine_name, [k])
                written.append(path)
            return written
        if nsteps > 1:
            print(
                f"WARNING: {input_path} has {nsteps} Time records; WRF's io_adios2 reader "
                "returns BP step 0 for every field read (wrf_io.F90:652-660), so WRF "
                "reading this BP file gets record 1's fields for every record; "
                "use --split-prefix (one BP per record)",
                file=sys.stderr,
            )
        specs = _make_specs(ds, time_name)
        _write_bp(ds, specs, output_path, engine_name, range(nsteps))
        return [output_path]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert WRF wrfinput/wrfbdy netCDF input to WRF-readable ADIOS2 BP"
    )
    parser.add_argument("input", help="input wrfinput_d0N / wrfbdy_d01 / wrflowinp_d0N / wrffdda_d0N netCDF file")
    parser.add_argument("output", help="output BP directory (for example wrfinput_d01.bp)")
    parser.add_argument("--engine", choices=("BP4", "BP5"), default=DEFAULT_ENGINE)
    parser.add_argument(
        "--split-prefix",
        metavar="NAME",
        help="write OUTPUT as a directory of one-record BP files NAME_<date>.bp "
        "(e.g. NAME=wrfbdy_d01) for multi_bdy_files / dated auxinput names",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.engine, args.split_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
