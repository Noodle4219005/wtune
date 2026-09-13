import os
import pathlib
import shlex
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class MPIChannelCTest(unittest.TestCase):
    def test_local_two_rank_channel_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "wt_mpi_channel_test"
            compile_command = shlex.split(os.environ.get("MPICC", "mpicc")) + [
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-pthread",
                "-I",
                str(ROOT / "plov"),
                str(ROOT / "plov" / "wt_mpi_channel.c"),
                str(ROOT / "plov" / "tests" / "test_wt_mpi_channel.c"),
                "-o",
                str(executable),
            ]
            subprocess.run(compile_command, check=True, cwd=ROOT, timeout=60)

            mpi_environment = os.environ.copy()
            for name in ("HYDRA_HOST_FILE", "HYDRA_HOSTFILE", "HYDRA_LAUNCHER"):
                mpi_environment.pop(name, None)
            run_command = shlex.split(os.environ.get("MPIEXEC", "mpiexec")) + [
                "-launcher",
                "fork",
                "-n",
                "2",
                str(executable),
            ]
            subprocess.run(
                run_command,
                check=True,
                cwd=ROOT,
                env=mpi_environment,
                timeout=15,
            )


if __name__ == "__main__":
    unittest.main()
