import os
import pathlib
import shlex
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class MPIRemoteCTest(unittest.TestCase):
    def test_local_two_rank_async_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "wt_mpi_remote_test"
            sources = [
                "wt_wire.c",
                "wt_policy.c",
                "wt_ledger.c",
                "wt_ledger_validate.c",
                "wt_mpi_channel.c",
                "wt_mpi_remote_common.c",
                "wt_mpi_remote_owner.c",
                "wt_mpi_remote_worker.c",
                "tests/test_wt_mpi_remote.c",
            ]
            subprocess.run(
                shlex.split(os.environ.get("MPICC", "mpicc"))
                + [
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    "-I",
                    str(ROOT / "plov"),
                ]
                + [str(ROOT / "plov" / source) for source in sources]
                + ["-lm", "-o", str(executable)],
                check=True,
                cwd=ROOT,
                timeout=60,
            )
            environment = os.environ.copy()
            for name in ("HYDRA_HOST_FILE", "HYDRA_HOSTFILE", "HYDRA_LAUNCHER"):
                environment.pop(name, None)
            subprocess.run(
                shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
                + ["-launcher", "fork", "-n", "2", str(executable)],
                check=True,
                cwd=ROOT,
                env=environment,
                timeout=30,
            )


if __name__ == "__main__":
    unittest.main()
