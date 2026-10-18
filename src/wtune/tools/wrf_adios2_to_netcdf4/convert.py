# PATCHED for this project (see UPSTREAM.md): upstream's `adios2.open(file,
# "rra")` random-access API no longer exists in the current `adios2` Python
# package (only `Stream`/`FileReader` remain) -- open_files()/r_metadata()/
# r_w_data_serial() below are rewritten against `adios2.FileReader`. Verified
# against real ADIOS2+Blosc2 wrfout output on this cluster (2026-08-07), see
# docs/io-adios2-zstd.md. The MPI-parallel path is NOT converted (upstream's
# own README already recommends the serial path; see open_files()) and will
# raise NotImplementedError if selected.
import sys
import argparse
from collections import defaultdict

import numpy as np
from netCDF4 import Dataset
import adios2

try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        parallel = True
    else:
        parallel = False

except ImportError:
    parallel = False


def progress(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    sys.stdout.write("\033[K")
    sys.stdout.write("[{0}] {1}% {2}\r".format(bar, percents, status))
    sys.stdout.flush()


def Locate(rank, nproc, datasize):
    extra = 0
    if rank == nproc - 1:
        extra = datasize % nproc
    num = datasize // nproc
    start = num * rank
    size = num + extra
    return start, size


def open_files(input_file, output_file, parallel=False, diskless=False):
    if parallel:
        # Not converted to the FileReader API -- untested, and upstream's own
        # README already recommends the serial path over parallel (no
        # measured performance benefit). Drop --parallel / mpirun to use it.
        raise NotImplementedError(
            "parallel conversion uses the removed adios2.open('rra', comm=...) "
            "API and was not ported; run this serially (no mpirun) instead"
        )
    adios2f = adios2.FileReader(input_file)
    netcdff = Dataset(
        output_file,
        "w",
        format="NETCDF4",
        parallel=parallel,
        diskless=diskless,
    )
    netcdff.set_fill_off()
    return (adios2f, netcdff)


def r_attrs(adios2f):
    attrs = adios2f.available_attributes()
    var_attrs = defaultdict(dict)  # init 2d dict
    global_attrs = {}
    for attr in attrs.keys():
        # "Dims" attribute not needed in NetCDF file
        if "/Dims" in attr:
            continue
        try:
            val = adios2f.read_attribute(attr)
        except ValueError:
            val = adios2f.read_attribute_string(attr)
        if "/" in attr:
            var, var_attrib = attr.split("/")
            var_attrs[var][var_attrib] = val
        else:
            if not attr.startswith("_DIM_"):
                global_attrs[attr] = val
    return (attrs, var_attrs, global_attrs)


def r_metadata(adios2f, attrs):
    vars = adios2f.available_variables()
    var_names = list(vars.keys())
    num_steps = adios2f.num_steps()  # was adios2f.steps() (old API: step count; new API: an iterator)
    dim_lens = {
        key[5:]: int(value["Value"])
        for key, value in attrs.items()
        if key.startswith("_DIM_")
    }

    var_dims = {}
    for var in vars:
        dims = adios2f.read_attribute_string("Dims", var)
        dims.reverse()
        var_dims[var] = dims
    typemap = {"float": "f", "double": "d", "int32_t": "i", "string": "c"}
    var_types = {}
    for var in vars:
        vtype = vars[var]["Type"]
        var_types[var] = typemap[vtype]
    return (var_names, num_steps, dim_lens, var_dims, var_types, vars)


def create_nc_dims(netcdff, num_steps, dim_lens):
    netcdff.createDimension("Time", size=num_steps)
    for dim in dim_lens:
        netcdff.createDimension(dim, size=dim_lens[dim])


def create_nc_vars(netcdff, var_names, var_types, var_dims):
    for var in var_names:
        # PATCHED (see UPSTREAM.md): compress like WRF's own io_form_history=2
        # writer does (zlib level 2 + shuffle -- read off a real wrfout of this
        # benchmark). Upstream creates variables uncompressed, which turns the
        # ~2.7 GB/frame WRF writes into ~8.5 GB/frame here: converting the
        # 7-frame CONUS 2.5km benchmark needed 59 GB and died partway through
        # on a 100 GB quota. Char/scalar variables keep the default (netCDF4
        # rejects the filters on scalars, and WRF leaves Times uncompressed).
        compress = var_types[var] != "c" and len(var_dims[var]) > 0
        kw = {"zlib": True, "complevel": 2, "shuffle": True} if compress else {}
        # PATCHED (see UPSTREAM.md): one frame per chunk. Frames are written one
        # at a time; netCDF-C's default chunks span many frames, so each write
        # recompressed them all (121 frames: 3 MB/s instead of 87 MB/s).
        if compress and var_dims[var][0] == "Time":
            kw["chunksizes"] = [1] + [len(netcdff.dimensions[d]) for d in var_dims[var][1:]]
        netcdff.createVariable(var, var_types[var], var_dims[var], **kw)


def decomp(var, vars):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    dshape = vars[var]["Shape"].split(",")
    dshape = list(map(int, dshape))
    max_ind = np.argmax(np.array(dshape))

    # do not decompose small arrays
    if dshape[max_ind] < 50:
        start_arr = np.zeros_like(dshape)
        count_arr = np.array(dshape)
    else:
        start, count = Locate(rank, size, dshape[max_ind])
        start_arr = np.zeros_like(dshape)
        start_arr[max_ind] = start
        count_arr = np.array(dshape)
        count_arr[max_ind] = count
    return start_arr, count_arr


def r_w_data_serial(adios2f, netcdff, var_names, steps, vars):
    # `adios2f.read(var)` (old "rra" API) implicitly tracked a per-object
    # current-step cursor advanced by successive calls. FileReader has no such
    # cursor -- each read is explicit and random-access via inquire_variable()
    # + step_selection=[step, 1]. There is also no read_string(): string
    # variables (e.g. Times) come back through the same .read() call, as a
    # plain Python str (verified against real data), decoded below just in
    # case a given engine/version returns bytes instead.
    # step_selection=[step, 1] already returns an array shaped exactly like
    # ADIOS2's own per-variable Shape (verified: no extra leading step axis
    # to drop), so -- same as the pre-patch code -- a single-element read
    # (true scalars, and 1-D "Shape": "1" variables like BATHYMETRY_FLAG
    # alike) is what decides the scalar-vs-array netCDF assignment, not the
    # Shape string itself.
    for i, var in enumerate(var_names):
        progress(i, len(var_names), status=var)
        variable = adios2f.inquire_variable(var)
        # PATCHED (see UPSTREAM.md): `steps` is the list of input steps to
        # convert, and its position in that list is the output Time index --
        # so --step writes a 1-frame file instead of skipping to index N of a
        # num_steps-long Time dimension. Without --step this is
        # range(num_steps) and out_step == step, i.e. upstream's behaviour.
        for out_step, step in enumerate(steps):
            raw = adios2f.read(variable, step_selection=[step, 1])
            if vars[var]["Type"] == "string":
                value = np.asarray(raw).reshape(-1)[0]
                if isinstance(value, bytes):
                    value = value.decode()
                netcdff.variables[var][out_step, :] = str(value)
            else:
                arr = np.asarray(raw)
                if arr.size == 1:
                    netcdff.variables[var][out_step] = arr.reshape(-1)[0]
                else:
                    netcdff.variables[var][out_step, :] = arr


def r_w_data_parallel(adios2f, netcdff, var_names, num_steps, vars):
    for i, var in enumerate(var_names):
        progress(i, len(var_names), status=var)
        if vars[var]["Type"] != "string":
            start_arr, count_arr = decomp(var, vars)
        for step in range(num_steps):
            if vars[var]["Type"] == "string":
                data = adios2f.read_string(var)
                netcdff.variables[var][step, :] = data
            else:
                data = adios2f.read(var, start=start_arr, count=count_arr)
                if data.ndim == 1 and len(data) == 1:
                    netcdff.variables[var][step] = data
                else:
                    netcdff.variables[var].set_collective(True)
                    if len(start_arr) == 3:
                        netcdff.variables[var][
                            step,
                            start_arr[0] : start_arr[0] + count_arr[0],
                            start_arr[1] : start_arr[1] + count_arr[1],
                            start_arr[2] : start_arr[2] + count_arr[2],
                        ] = data
                    elif len(start_arr) == 2:
                        netcdff.variables[var][
                            step,
                            start_arr[0] : start_arr[0] + count_arr[0],
                            start_arr[1] : start_arr[1] + count_arr[1],
                        ] = data
                    elif len(start_arr) == 1:
                        netcdff.variables[var][
                            step, start_arr[0] : start_arr[0] + count_arr[0]
                        ] = data


def w_global_attrs(netcdff, global_attrs):
    netcdff.setncatts(global_attrs)


def w_var_attrs(netcdff, var_attrs):
    for var in var_attrs.keys():
        for attr in var_attrs[var].keys():
            netcdff.variables[var].setncattr(attr, var_attrs[var][attr])


def close_files(adios2f, netcdff, diskless=False):
    if diskless == False:
        netcdff.close()
    adios2f.close()


def convert(input_file, output_file, parallel=False, diskless=False, step=None):
    adios2f, netcdff = open_files(
        input_file, output_file, parallel=parallel, diskless=diskless
    )
    attrs, var_attrs, global_attrs = r_attrs(adios2f)
    var_names, num_steps, dim_lens, var_dims, var_types, vars = r_metadata(
        adios2f, attrs
    )
    # PATCHED (see UPSTREAM.md): --step converts one Time frame instead of all
    # of them. A wrfout written with frames_per_outfile > 1 holds every history
    # time in one BP directory, but tools/verify_wrf.py compares against a
    # single-frame official reference, so it needs one frame -- and converting
    # only that frame costs one seventh of the disk and wall time.
    if step is not None and not -num_steps <= step < num_steps:
        raise ValueError(
            "--step %d out of range: %s has %d frame(s)" % (step, input_file, num_steps)
        )
    steps = range(num_steps) if step is None else [step % num_steps]
    create_nc_dims(netcdff, len(steps), dim_lens)
    create_nc_vars(netcdff, var_names, var_types, var_dims)
    if parallel:
        r_w_data_parallel(adios2f, netcdff, var_names, num_steps, vars)
    else:
        r_w_data_serial(adios2f, netcdff, var_names, steps, vars)
    w_global_attrs(netcdff, global_attrs)
    w_var_attrs(netcdff, var_attrs)
    close_files(adios2f, netcdff, diskless)

    if diskless == True:
        return netcdff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input: ADIOS2 file")
    parser.add_argument("--output", help="output: NetCDF file")
    parser.add_argument(
        "--step", type=int, default=None,
        help="convert only this Time frame (0-based, negative counts from the "
             "end) into a 1-frame NetCDF file; default converts every frame",
    )
    args = parser.parse_args()
    convert(args.input, args.output, parallel=parallel, step=args.step)
