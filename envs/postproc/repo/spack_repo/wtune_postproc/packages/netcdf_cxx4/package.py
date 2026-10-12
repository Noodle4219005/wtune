# Built-in netcdf-cxx4 with one fix: its `libs` searches only the Autotools-era name
# libnetcdf_c++4, but 4.3.1 is built with CMake, which installs libnetcdf-cxx4.so. MET's
# recipe then fails with NoLibrariesError before configuring (scc job 1582), and MET's own
# Makefiles link -lnetcdf_c++4 (scc job 1583), so also install the old name as a symlink.
from spack_repo.builtin.packages.netcdf_cxx4.package import NetcdfCxx4 as BuiltinNetcdfCxx4

import os

from spack.package import *


class NetcdfCxx4(BuiltinNetcdfCxx4):
    @property
    def libs(self):
        shared = "+shared" in self.spec
        libs = find_libraries(["libnetcdf-cxx4", "libnetcdf_c++4"], root=self.prefix,
                              shared=shared, recursive=True)
        if libs:
            return libs
        raise NoLibrariesError("Unable to locate netcdf-cxx4 libraries in %s" % self.prefix)

    @run_after("install")
    def old_library_name(self):
        lib = self.prefix.lib
        for f in os.listdir(lib):
            if f.startswith("libnetcdf-cxx4."):
                old = os.path.join(lib, f.replace("libnetcdf-cxx4", "libnetcdf_c++4", 1))
                if not os.path.lexists(old):
                    os.symlink(f, old)
