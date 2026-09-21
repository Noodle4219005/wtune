#!/usr/bin/env python3
"""Compare every WRF NetCDF variable and variable attribute bit for bit."""

import argparse
import ctypes
import hashlib
import itertools
import json
import os
import sys


NC_NOWRITE = 0
NC_GLOBAL = -1
NC_MAX_NAME = 256
NC_STRING = 12
MAX_SLAB_BYTES = 8 * 1024 * 1024
IGNORED_GLOBAL_ATTRIBUTES = {b"history", b"_NCProperties"}
TYPE_SIZES = {
    1: ("NC_BYTE", 1), 2: ("NC_CHAR", 1), 3: ("NC_SHORT", 2),
    4: ("NC_INT", 4), 5: ("NC_FLOAT", 4), 6: ("NC_DOUBLE", 8),
    7: ("NC_UBYTE", 1), 8: ("NC_USHORT", 2), 9: ("NC_UINT", 4),
    10: ("NC_INT64", 8), 11: ("NC_UINT64", 8),
}


class NcError(Exception):
    pass


class Different(Exception):
    def __init__(self, detail):
        super().__init__(detail.get("message", "files differ"))
        self.detail = detail


class NetCDF:
    def __init__(self, library):
        if not os.path.isabs(library):
            raise NcError("--library must be an absolute path")
        try:
            self.lib = ctypes.CDLL(library)
        except OSError as exc:
            raise NcError("cannot load libnetcdf: %s" % exc)
        self.path = os.path.realpath(library)
        c_int_p = ctypes.POINTER(ctypes.c_int)
        size_p = ctypes.POINTER(ctypes.c_size_t)
        self.lib.nc_strerror.argtypes = [ctypes.c_int]
        self.lib.nc_strerror.restype = ctypes.c_char_p
        self.lib.nc_inq_libvers.restype = ctypes.c_char_p
        self.lib.nc_open.argtypes = [ctypes.c_char_p, ctypes.c_int, c_int_p]
        self.lib.nc_close.argtypes = [ctypes.c_int]
        self.lib.nc_inq.argtypes = [ctypes.c_int, c_int_p, c_int_p, c_int_p, c_int_p]
        self.lib.nc_inq_grps.argtypes = [ctypes.c_int, c_int_p, c_int_p]
        self.lib.nc_inq_dim.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, size_p]
        self.lib.nc_inq_unlimdims.argtypes = [ctypes.c_int, c_int_p, c_int_p]
        self.lib.nc_inq_varname.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        self.lib.nc_inq_vartype.argtypes = [ctypes.c_int, ctypes.c_int, c_int_p]
        self.lib.nc_inq_varndims.argtypes = [ctypes.c_int, ctypes.c_int, c_int_p]
        self.lib.nc_inq_vardimid.argtypes = [ctypes.c_int, ctypes.c_int, c_int_p]
        self.lib.nc_inq_varnatts.argtypes = [ctypes.c_int, ctypes.c_int, c_int_p]
        self.lib.nc_inq_attname.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_char_p]
        self.lib.nc_inq_att.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                        c_int_p, size_p]
        self.lib.nc_get_att.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                        ctypes.c_void_p]
        self.lib.nc_get_var.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.nc_get_vara.argtypes = [ctypes.c_int, ctypes.c_int, size_p, size_p,
                                         ctypes.c_void_p]
        self.version = self.lib.nc_inq_libvers().decode("utf-8", "replace")

    def check(self, status, operation):
        if status:
            msg = self.lib.nc_strerror(status).decode("utf-8", "replace")
            raise NcError("%s: %s" % (operation, msg))

    def open(self, path):
        ncid = ctypes.c_int()
        self.check(self.lib.nc_open(os.fsencode(path), NC_NOWRITE, ctypes.byref(ncid)),
                   "open %s" % path)
        return ncid.value

    def close(self, ncid):
        self.check(self.lib.nc_close(ncid), "close NetCDF")

    def inq(self, ncid):
        values = [ctypes.c_int() for _ in range(4)]
        self.check(self.lib.nc_inq(ncid, *[ctypes.byref(x) for x in values]), "inq")
        return tuple(x.value for x in values)

    def child_group_count(self, ncid):
        count = ctypes.c_int()
        self.check(self.lib.nc_inq_grps(ncid, ctypes.byref(count), None),
                   "inq child groups")
        return count.value

    def dimensions(self, ncid, ndims):
        count = ctypes.c_int()
        self.check(self.lib.nc_inq_unlimdims(ncid, ctypes.byref(count), None),
                   "inq unlimited dimension count")
        ids = (ctypes.c_int * max(1, count.value))()
        if count.value:
            self.check(self.lib.nc_inq_unlimdims(ncid, ctypes.byref(count), ids),
                       "inq unlimited dimensions")
        unlimited = set(ids[:count.value])
        result = []
        for dimid in range(ndims):
            name = ctypes.create_string_buffer(NC_MAX_NAME + 1)
            length = ctypes.c_size_t()
            self.check(self.lib.nc_inq_dim(ncid, dimid, name, ctypes.byref(length)),
                       "inq dimension %d" % dimid)
            result.append((bytes(name.value), length.value, dimid in unlimited))
        return result

    def variable(self, ncid, varid):
        name = ctypes.create_string_buffer(NC_MAX_NAME + 1)
        xtype, ndims, natts = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self.check(self.lib.nc_inq_varname(ncid, varid, name), "inq variable name")
        self.check(self.lib.nc_inq_vartype(ncid, varid, ctypes.byref(xtype)),
                   "inq variable type")
        self.check(self.lib.nc_inq_varndims(ncid, varid, ctypes.byref(ndims)),
                   "inq variable dimensions")
        dimids = (ctypes.c_int * max(1, ndims.value))()
        if ndims.value:
            self.check(self.lib.nc_inq_vardimid(ncid, varid, dimids),
                       "inq variable dimension ids")
        self.check(self.lib.nc_inq_varnatts(ncid, varid, ctypes.byref(natts)),
                   "inq variable attribute count")
        return bytes(name.value), xtype.value, tuple(dimids[:ndims.value]), natts.value

    def attribute_names(self, ncid, varid, count):
        names = []
        for index in range(count):
            name = ctypes.create_string_buffer(NC_MAX_NAME + 1)
            self.check(self.lib.nc_inq_attname(ncid, varid, index, name),
                       "inq attribute name")
            names.append(bytes(name.value))
        return names

    def attribute(self, ncid, varid, name):
        xtype, length = ctypes.c_int(), ctypes.c_size_t()
        self.check(self.lib.nc_inq_att(ncid, varid, name, ctypes.byref(xtype),
                                      ctypes.byref(length)), "inq attribute")
        type_name, width = primitive_type(xtype.value)
        size = length.value * width
        buffer = (ctypes.c_ubyte * max(1, size))()
        self.check(self.lib.nc_get_att(ncid, varid, name, buffer), "read attribute")
        return xtype.value, type_name, length.value, bytes(buffer[:size])

    def data(self, ncid, varid, start, count, byte_count):
        buffer = (ctypes.c_ubyte * max(1, byte_count))()
        if not start:
            self.check(self.lib.nc_get_var(ncid, varid, buffer), "read scalar")
        else:
            starts = (ctypes.c_size_t * len(start))(*start)
            counts = (ctypes.c_size_t * len(count))(*count)
            self.check(self.lib.nc_get_vara(ncid, varid, starts, counts, buffer),
                       "read variable slab")
        return bytes(buffer[:byte_count])


def primitive_type(xtype):
    if xtype not in TYPE_SIZES:
        label = "NC_STRING" if xtype == NC_STRING else "user-defined type"
        raise NcError("unsupported %s id %d" % (label, xtype))
    return TYPE_SIZES[xtype]


def product(values):
    answer = 1
    for value in values:
        answer *= value
    return answer


def slabs(shape, width):
    if not shape:
        yield (), (), width
        return
    if any(length == 0 for length in shape):
        return
    max_elements = MAX_SLAB_BYTES // width
    chunk = list(shape)
    for index in range(len(chunk)):
        others = product(chunk[index + 1:]) * product(chunk[:index])
        if product(chunk) <= max_elements:
            break
        chunk[index] = max(1, min(chunk[index], max_elements // max(1, others)))
    ranges = [range(0, shape[i], chunk[i]) for i in range(len(shape))]
    for start in itertools.product(*ranges):
        count = tuple(min(chunk[i], shape[i] - start[i]) for i in range(len(shape)))
        yield start, count, product(count) * width


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def mismatch(category, message, **fields):
    detail = {"category": category, "message": message}
    detail.update(fields)
    raise Different(detail)


def compare_attributes(api, left, right, left_var, right_var, var_name, counts, stats,
                       ignored=None):
    left_names = api.attribute_names(left, left_var, counts[0])
    right_names = api.attribute_names(right, right_var, counts[1])
    ignored = ignored or set()
    left_names = [name for name in left_names if name not in ignored]
    right_names = [name for name in right_names if name not in ignored]
    if set(left_names) != set(right_names) or len(left_names) != len(right_names):
        mismatch("variable_attributes", "variable attribute inventory differs",
                 variable=var_name, left=sorted(decode(x) for x in left_names),
                 right=sorted(decode(x) for x in right_names))
    for name in left_names:
        a = api.attribute(left, left_var, name)
        b = api.attribute(right, right_var, name)
        if a[:3] != b[:3]:
            mismatch("variable_attribute", "variable attribute type or length differs",
                     variable=var_name, attribute=decode(name))
        if a[3] != b[3]:
            offset = first_byte(a[3], b[3])
            mismatch("variable_attribute", "variable attribute bytes differ",
                     variable=var_name, attribute=decode(name), byte_offset=offset)
        stats["compared_attribute_bytes"] += len(a[3])
        stats["compared_bytes"] += len(a[3])
        key = ("compared_global_attributes" if left_var == NC_GLOBAL else
               "compared_variable_attributes")
        stats[key] += 1


def first_byte(left, right):
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return min(len(left), len(right))


def decode(value):
    return value.decode("utf-8", "replace")


def compare_files(api, left_path, right_path, report):
    left = right = None
    try:
        left = api.open(left_path)
        right = api.open(right_path)
        if api.child_group_count(left) or api.child_group_count(right):
            raise NcError("child groups are unsupported; refusing to skip grouped state")
        li, ri = api.inq(left), api.inq(right)
        if li[1] == 0 or ri[1] == 0:
            raise NcError("refusing NetCDF file with no variables")
        ld, rd = api.dimensions(left, li[0]), api.dimensions(right, ri[0])
        if ld != rd:
            mismatch("dimensions", "dimension name, order, length, or unlimited status differs",
                     left=[(decode(x[0]), x[1], x[2]) for x in ld],
                     right=[(decode(x[0]), x[1], x[2]) for x in rd])
        left_global = api.attribute_names(left, NC_GLOBAL, li[2])
        right_global = api.attribute_names(right, NC_GLOBAL, ri[2])
        report["ignored_global_attributes"] = {
            "scope": "Only the named volatile provenance attributes are ignored.",
            "names": sorted(decode(x) for x in IGNORED_GLOBAL_ATTRIBUTES),
            "present_left": sorted(decode(x) for x in left_global
                                   if x in IGNORED_GLOBAL_ATTRIBUTES),
            "present_right": sorted(decode(x) for x in right_global
                                    if x in IGNORED_GLOBAL_ATTRIBUTES),
        }
        compare_attributes(api, left, right, NC_GLOBAL, NC_GLOBAL, "<global>",
                           (li[2], ri[2]), report["statistics"],
                           IGNORED_GLOBAL_ATTRIBUTES)
        if li[1] != ri[1]:
            mismatch("variable_inventory", "variable count differs", left=li[1], right=ri[1])
        shapes = [[x[1] for x in ld], [x[1] for x in rd]]
        stats = report["statistics"]
        for varid in range(li[1]):
            lv, rv = api.variable(left, varid), api.variable(right, varid)
            lname, rname = decode(lv[0]), decode(rv[0])
            if lv[:3] != rv[:3]:
                mismatch("variable_schema", "variable name, type, or dimension order differs",
                         variable_index=varid, left={"name": lname, "type": lv[1], "dims": lv[2]},
                         right={"name": rname, "type": rv[1], "dims": rv[2]})
            type_name, width = primitive_type(lv[1])
            compare_attributes(api, left, right, varid, varid, lname, (lv[3], rv[3]), stats)
            shape = tuple(shapes[0][dimid] for dimid in lv[2])
            if any(length == 0 for length in shape):
                raise NcError("refusing empty variable %s" % lname)
            for start, count, size in slabs(shape, width):
                a = api.data(left, varid, start, count, size)
                b = api.data(right, varid, start, count, size)
                if a != b:
                    mismatch("variable_data", "typed variable bytes differ", variable=lname,
                             netcdf_type=type_name, slab_start=start, slab_count=count,
                             slab_byte_offset=first_byte(a, b))
                stats["compared_data_bytes"] += size
                stats["compared_bytes"] += size
            stats["compared_variables"] += 1
    finally:
        errors = []
        for handle in (right, left):
            if handle is not None:
                try:
                    api.close(handle)
                except NcError as exc:
                    errors.append(str(exc))
        if errors and sys.exc_info()[0] is None:
            raise NcError("; ".join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="absolute libnetcdf shared-library path")
    parser.add_argument("--report", required=True, help="new JSON report path")
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args(argv)
    if os.path.exists(args.report):
        print("ERROR: report path already exists: %s" % args.report, file=sys.stderr)
        return 2
    report = {
        "left": {"path": os.path.abspath(args.left)},
        "right": {"path": os.path.abspath(args.right)},
        "library": {}, "result": "error", "first_difference": None,
        "comparison_scope": {
            "data": "libnetcdf primitive typed values compared as native-memory bytes",
            "container_bytes": "not compared; storage format, compression, and endianness may differ",
            "maximum_slab_bytes": MAX_SLAB_BYTES,
            "unsupported": ["child groups", "NC_STRING", "user-defined types", "VLEN"],
        },
        "statistics": {"compared_variables": 0, "compared_data_bytes": 0,
                       "compared_variable_attributes": 0,
                       "compared_global_attributes": 0,
                       "compared_attribute_bytes": 0, "compared_bytes": 0},
    }
    code = 2
    try:
        report["left"]["sha256"] = sha256(args.left)
        report["right"]["sha256"] = sha256(args.right)
        api = NetCDF(args.library)
        report["library"] = {"requested_path": args.library, "resolved_path": api.path,
                             "version": api.version}
        compare_files(api, args.left, args.right, report)
        report["result"] = "identical"
        code = 0
    except Different as exc:
        report["result"] = "different"
        report["first_difference"] = exc.detail
        code = 1
    except (NcError, OSError) as exc:
        report["error"] = str(exc)
        code = 2
    try:
        with open(args.report, "x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except OSError as exc:
        print("ERROR: cannot create report: %s" % exc, file=sys.stderr)
        return 2
    if code:
        message = report.get("error") or report["first_difference"]["message"]
        print(("DIFFERENT: " if code == 1 else "ERROR: ") + message, file=sys.stderr)
    else:
        print("IDENTICAL: %d variables, %d typed bytes" %
              (report["statistics"]["compared_variables"],
               report["statistics"]["compared_bytes"]))
    return code


if __name__ == "__main__":
    sys.exit(main())
