#!/usr/bin/env python3
"""Convert WRF input netCDF files to BP files readable by WRF io_form=14 (serial, one time slice
at a time, no MPI). Library callers use wtune.bp.nc2bp / convert_inputs; CLI: `wtune convert nc2bp`.

Layout follows WRF's own writer/reader (WRF/external/io_adios2/wrf_io.F90: Times 1254-1272, fields
2295-2523, FieldType/MemoryOrder 2530-2542, one BP step per time 327-389, reader 949-1033).
Restored from 8ba1629; validated f1 jobs 1071692/1072441 (wiki: adios2-bp4-restart-via-adios2-xml...).
Engine defaults to BP4: WRF's reader inquires Times before BeginStep, which BP5 refuses (scc 2026-10-03).
Multi-record files read back as step 0 for every field (wrf_io.F90:652-660; scc jobs 2778/2780), so
--split-prefix writes one BP per record (multi_bdy_files / dated auxinput names).
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
# Boundary suffixes (_BXS.., _BTXS.., md___*) are kept byte-for-byte: WRF does not interpret them
# here and the boundary reader requests md___thisbdytime/nextbdytime directly.


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
    """WRF's per-variable character attribute encoding: a one-element string array
    (put_var_ti_char, count 1); MemoryOrder and Dims stay scalars."""
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
    # Time is the BP step count; every other dimension becomes an integer _DIM_* attribute
    # (also the unused ones, which the reverse converter needs).
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
    """Per-record BP name <prefix>_<date>.bp as WRF's construct_filename2a builds it
    (share/module_io_domain.F:194-252; 19-character date, colons kept)."""
    if len(date) != WRF_DATE_STR_LEN:
        raise ValueError(f"bad WRF date string {date!r}")
    return f"{prefix}_{date}.bp"


def _check_times(ds: Dataset, time_name: str | None) -> tuple[int, Any]:
    """(number of Time records, the Times variable) of a WRF input; raises ValueError if unusable."""
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
    return nsteps, times


def _write_split(ds: Dataset, specs: list[VariableSpec], times: Any, nsteps: int,
                 input_path: str, output_path: str, engine_name: str, split_prefix: str) -> list[str]:
    """One single-step BP per Time record under the new directory output_path."""
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


def convert(
    input_path: str,
    output_path: str,
    engine_name: str = DEFAULT_ENGINE,
    split_prefix: str | None = None,
) -> list[str]:
    """Convert input_path to one BP (a step per Time record) at output_path, or, with
    split_prefix, to a directory of one-record BPs record_file_name(split_prefix, Times[k]).
    Returns the written BP paths."""
    if engine_name not in {"BP4", "BP5"}:
        raise ValueError(f"unsupported engine {engine_name!r}; use BP4 or BP5")
    if os.path.exists(output_path):
        raise FileExistsError(f"output already exists: {output_path}")

    with Dataset(input_path, "r") as ds:
        ds.set_auto_mask(False)
        ds.set_auto_scale(False)
        time_name = _time_dimension(ds)
        nsteps, times = _check_times(ds, time_name)
        if split_prefix is not None:
            specs = _make_specs(ds, time_name)
            return _write_split(ds, specs, times, nsteps, input_path, output_path,
                                engine_name, split_prefix)
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
