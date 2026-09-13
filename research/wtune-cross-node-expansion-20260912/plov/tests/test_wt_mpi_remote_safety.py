import os
import pathlib
import shlex
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MPIRemoteSafetyCTest(unittest.TestCase):
    def test_deadline_and_reentry_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "wt_mpi_remote_safety_test"
            sources = [
                "wt_wire.c", "wt_policy.c", "wt_ledger.c",
                "wt_ledger_validate.c", "wt_mpi_channel.c",
                "wt_mpi_remote_common.c", "wt_mpi_remote_owner.c",
                "wt_mpi_remote_worker.c", "tests/test_wt_mpi_remote_safety.c",
            ]
            subprocess.run(
                ["mpicc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-pedantic", "-I", str(ROOT)]
                + [str(ROOT / source) for source in sources]
                + ["-lm", "-o", str(executable)],
                check=True,
                timeout=60,
            )
            environment = os.environ.copy()
            environment.pop("HYDRA_HOST_FILE", None)
            environment.pop("HYDRA_HOSTFILE", None)
            subprocess.run(
                shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
                + ["-launcher", "fork", "-n", "2", str(executable)],
                check=True,
                env=environment,
                timeout=30,
            )


if __name__ == "__main__":
    unittest.main()
