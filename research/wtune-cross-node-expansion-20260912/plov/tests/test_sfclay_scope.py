import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF/dyn_em/module_first_rk_step_part1.f90")
SHA = "4668213aaec35d72f7bdbd2473d810b4538370f8aadf5979ef940f315be9c43b"
GENERATOR = ROOT / "plov/gen_sfclay_scope_bridge.py"
SF_SOURCE = SOURCE.parents[1] / "phys/module_sf_sfclay.F"

class SfclayScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not GENERATOR.exists():
            raise AssertionError("Missing first-RK audited scope bridge generator")
        spec = importlib.util.spec_from_file_location("scopegen", GENERATOR)
        cls.gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gen)

    def test_complete_signature_and_declarations(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SHA)
        profile = self.gen.audit_text(SOURCE.read_text())
        self.assertEqual(len(profile), 84)
        self.assertEqual([x["name"] for x in profile[:3]], ["grid", "config_flags", "moist"])
        self.assertEqual([x["name"] for x in profile[-3:]], ["aerocu", "restart_flag", "feedback_is_ready"])
        self.assertEqual(profile[0]["type"], "type(domain)")
        self.assertEqual(profile[1]["type"], "type(grid_config_rec_type)")
        self.assertEqual(sum(x["optional"] for x in profile), 4)
        self.assertEqual(self.gen.generate(SOURCE), (ROOT / "plov/wt_sfclay_scope_bridge.c").read_text())

    def test_profile_digest_is_literal_pin_not_recomputed_on_import(self):
        """A future profile edit must fail closed until its audit pin is reviewed."""
        source = GENERATOR.read_text()
        self.assertIn(
            'PROFILE_SHA256 = "545d198388690ca64d46e4aa8a0fabf565d4e176bf02dd93f72c2ab9fb0de7cf"',
            source)
        with tempfile.TemporaryDirectory(prefix="wt-scope-profile-pin-") as tmp:
            altered = Path(tmp) / "altered_generator.py"
            altered.write_text(source.replace(
                'scalar scalar_tend fdda3d fdda2d aerod ru_tendf',
                'scalar scalar_tend fdda3d fdda2d aerod altered_formal ru_tendf', 1))
            spec = importlib.util.spec_from_file_location("altered_scopegen", altered)
            module = importlib.util.module_from_spec(spec)
            with self.assertRaisesRegex(RuntimeError, "declaration profile SHA-256"):
                spec.loader.exec_module(module)

    def test_reject_truncation_names_and_unsafe_declarations(self):
        text = SOURCE.read_text()
        changes = [
            text[:text.index(", restart_flag")],
            text.replace(", moist , moist_tend", ", moist_tend , moist", 1),
            text.replace("TYPE ( domain )", "CLASS ( domain )", 1),
            text.replace("TYPE ( grid_config_rec_type )", "TYPE ( unknown_type )", 1),
            text.replace(":: moist\n", ":: renamed_moist\n", 1),
            text.replace(":: moist\n", ":: moist, moist\n", 1),
            text.replace("LOGICAL ,INTENT(IN)", "CHARACTER ,INTENT(IN)", 1),
            text.replace("LOGICAL ,INTENT(IN)", "LOGICAL, VALUE,INTENT(IN)", 1),
            text.replace("LOGICAL ,INTENT(IN)", "LOGICAL, POINTER,INTENT(IN)", 1),
            text.replace("LOGICAL ,INTENT(IN)", "LOGICAL, ALLOCATABLE,INTENT(IN)", 1),
            text.replace("DIMENSION(ims:ime,kms:kme,jms:jme,num_moist)", "DIMENSION(:,:,:,:)", 1),
            text.replace("DIMENSION(ims:ime,kms:kme,jms:jme,num_moist)", "DIMENSION(ims:ime,kms:kme,jms:jme,*)", 1),
        ]
        for changed in changes:
            with self.subTest(change=changes.index(changed)):
                self.assertNotEqual(changed, text)
                with self.assertRaises(ValueError):
                    self.gen.audit_text(changed)
        with tempfile.TemporaryDirectory(prefix="wt-scope-refuse-") as tmp:
            changed = Path(tmp) / "changed.f90"
            changed.write_text(text + "! hash change\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.gen.generate(changed)
        with self.assertRaises(ValueError):
            self.gen.generate(SOURCE, "bad;symbol")

    def test_invalid_cli_inputs_preserve_existing_output(self):
        with tempfile.TemporaryDirectory(prefix="wt-scope-cli-") as tmp:
            output = Path(tmp) / "scope.c"
            original = "/* retain this audited output */\n"
            output.write_text(original)
            invalid_inputs = [
                ["--source", str(Path(tmp) / "missing.f90")],
                ["--source", str(SOURCE), "--symbol", "bad;symbol"],
                ["--source", str(SOURCE), "--symbol", "1bad"],
            ]
            for arguments in invalid_inputs:
                with self.subTest(arguments=arguments):
                    result = subprocess.run([sys.executable, str(GENERATOR), *arguments,
                                             "--output", str(output)], text=True, capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output.read_text(), original)

    def test_linked_runtime_scope_forwarding(self):
        cc = shlex.split(os.environ["CC"])
        def run(cmd):
            r = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 0, " ".join(map(str,cmd)) + "\n" + r.stdout + r.stderr)
            return r.stdout
        with tempfile.TemporaryDirectory(prefix="wt-sf-scope-") as tmp:
            tmp = Path(tmp)
            rk = tmp / "rk.c"
            rk.write_text(self.gen.generate(SOURCE, "scope_fixture_rk"))
            sf = tmp / "sf.c"
            run([sys.executable, "plov/gen_sfclay_bridge.py", "--source", str(SF_SOURCE),
                 "--schema", "plov/sfclay_schema.json", "--symbol", "scope_fixture_sf", "--output", str(sf)])
            formals = ", ".join("void *a%d" % i for i in range(84))
            args = ", ".join("a%d" % i for i in range(84))
            callee = tmp / "callee.c"
            callee.write_text("extern void fixture_rk_native(void **);\nextern void fixture_sf_native(void **);\n" +
                "void scope_fixture_rk(" + formals + ") { void *a[]={" + args + "}; fixture_rk_native(a); }\n" +
                "void scope_fixture_sf(" + formals + ") { void *a[]={" + args + "}; fixture_sf_native(a); }\n" +
                "void module_sf_sfclay_mp_sfclayinit_(int *a) { (void)a; }\n")
            flags = ["-std=c11", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-pthread", "-Iplov",
                     "-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
            default = tmp / "default.o"
            run(cc + flags + ["-c", "plov/wt_sfclay_scope_bridge.c", "-o", str(default)])
            symbols = run(["nm", "-g", str(default)])
            self.assertIn(" T __wrap_module_first_rk_step_part1_mp_first_rk_step_part1_", symbols)
            self.assertIn(" U __real_module_first_rk_step_part1_mp_first_rk_step_part1_", symbols)
            self.assertNotIn("wt_sfclay_scope_test_call", symbols)
            exe = tmp / "scope"
            sources = [str(rk), str(sf), str(callee), "plov/tests/sfclay_scope_fixture.c",
                "plov/wt_sfclay_scope.c", "plov/wt_sfclay_dispatch.c", "plov/wt_sfclay_live.c",
                "plov/wt_sfclay_init_lease.c", "plov/wt_sfclay_capacity.c", "plov/wt_sfclay.c", "plov/wt_live_spans.c"]
            run(cc + flags + ["-DWT_SFCLAY_SCOPE_TEST_HELPERS", "-DWT_SFCLAY_BRIDGE_TEST_HELPERS"] + sources +
                ["-Wl,--wrap=scope_fixture_rk", "-Wl,--wrap=scope_fixture_sf",
                 "-Wl,--wrap=module_sf_sfclay_mp_sfclayinit_", "-o", str(exe)])
            for case in ("forward", "optional-null", "direct", "nested-hook", "unmatched-hook", "nested-rk"):
                with self.subTest(case=case):
                    r = subprocess.run([str(exe), case], text=True, capture_output=True, timeout=5)
                    self.assertEqual(r.returncode, -signal.SIGABRT if case in ("nested-hook", "unmatched-hook", "nested-rk") else 0, r.stdout+r.stderr)
                    self.assertNotIn("Sanitizer", r.stderr)
                    self.assertNotIn("CHECK", r.stderr)

if __name__ == "__main__":
    unittest.main()
