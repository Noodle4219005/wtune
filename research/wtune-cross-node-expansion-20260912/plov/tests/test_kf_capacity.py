import json, os, shlex, subprocess, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class KFCapacityTest(unittest.TestCase):
    def test_schema_type_oracle(self):
        abi = json.loads((ROOT / "plov/kf_cps_schema.json").read_text())["abi"]
        self.assertIn("nca", abi["real_2d_required_inout"])
        self.assertIn("shall", abi["real_2d_optional_inout"])
        self.assertEqual(abi["logical_2d_required_inout"], ["cu_act_flag"])

    def test_c_fixture(self):
        cc = shlex.split(os.environ.get("CC", "cc"))
        with tempfile.TemporaryDirectory(prefix="wt-kf-capacity-") as td:
            exe = Path(td) / "capacity"
            command = cc + ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "plov"),
                str(ROOT / "plov/tests/test_kf_capacity.c"), str(ROOT / "plov/wt_kf_capacity.c"),
                str(ROOT / "plov/wt_live_spans.c"), "-o", str(exe)]
            subprocess.run(command, check=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=60)

if __name__ == "__main__": unittest.main()
