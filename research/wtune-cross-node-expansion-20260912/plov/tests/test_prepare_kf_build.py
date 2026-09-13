import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/20260913-node-local-physics/build-reviewed"
SPEC = importlib.util.spec_from_file_location("prepare_kf_build", ROOT / "plov/prepare_kf_build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class PrepareKfBuildTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="wt-kf-plan-", dir=ARTIFACTS)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.wt = self.base / "arbitrary wtune"
        self.wrf = self.base / "frozen wrf"
        self.out = self.base / "new output"
        self.files = [
            "wtimer.c", "wtimer.S", "baseline.c", "wt_cumulus.c", "wt_cumulus.h",
            "wt_live_spans.c", "wt_live_spans.h", "wt_grid_spans.f90",
            "wt_grid_capture.f90", "wt_kf_native.c", "wt_kf_native.h",
            "wt_kf_live.c", "wt_kf_live.h", "wt_kf_capacity.c", "wt_kf_capacity.h",
            "wt_kf_dispatch.c", "wt_kf_dispatch.h", "wt_kf_bootstrap.c",
            "wt_kf_bootstrap.h", "wt_kf_cps_schema.h", "gen_kf_bridge.py",
            "gen_ref_bridge.py", "gen_phase_schema.py", "prepare_native_link.py"]
        for name in self.files:
            self.write(self.wt / "plov" / name, "fixture source\n")
        self.write(self.wt / "plov/kf_cps_schema.json", "{}")
        self.write(self.wrf / "phys/module_cu_kfeta.f90", "fixture KF source\n")
        self.write(self.wrf / "frame/module_domain_type.mod", "fixture module\n")
        self.write(self.wrf / "compile.log", "historical fixture\n")
        self.cc = self.base / "mpi compiler"
        self.fc = self.base / "fortran compiler"
        # A script that must never be executed by a recipe preparer.
        for path in (self.cc, self.fc):
            self.write(path, "#!/bin/sh\nexit 99\n")
            path.chmod(0o700)
        self.config = {
            "baseline_units": [
                {"source": "plov/wtimer.c", "language": "c", "output_name": "timer-c.o", "flags": []},
                {"source": "plov/wtimer.S", "language": "asm", "output_name": "timer-asm.o", "flags": []},
                {"source": "plov/baseline.c", "language": "c", "output_name": "baseline.o", "flags": []}],
            "wrap_symbols": ["module_cumulus_driver_mp_cumulus_driver_", "baseline_symbol_"],
            "c_flags": ["-O2", "-std=c11", "-fopenmp"],
            "fortran_flags": ["-O2", "-fopenmp"],
            "fortran_module_flag": "-J"}
        self.cfg = self.base / "compiler.json"

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def prepare(self):
        self.cfg.write_text(json.dumps(self.config))
        return build.prepare(wtune_root=self.wt, frozen_root=self.wrf,
                             cc=self.cc, fc=self.fc, compiler_config=self.cfg,
                             output_dir=self.out)

    def snapshot(self):
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for root in (self.wt, self.wrf) for path in root.rglob("*")
                if path.is_file()}

    def test_recipe_macro_scope_external_outputs_and_preservation(self):
        before = self.snapshot()
        plan = self.prepare()
        self.assertEqual(before, self.snapshot())
        self.assertEqual([p.name for p in self.out.iterdir()], ["build-manifest.json"])
        self.assertFalse(plan["attestation"]["source_abi_verified"])
        self.assertFalse(plan["attestation"]["caller_chain_verified"])
        self.assertFalse(plan["attestation"]["baseline_closure_verified"])
        units = [step for step in plan["steps"] if step["kind"] == "compile"]
        self.assertEqual(len(units), 13)
        guards, renames = [], []
        for unit in units:
            self.assertEqual(unit["cwd"], str(self.out))
            self.assertTrue(Path(unit["output"]).is_relative_to(self.out))
            argv = unit["argv"]
            self.assertEqual(argv[-2:], ["-o", unit["output"]])
            if "-DWT_CUMULUS_LIVE_SPANS=1" in argv:
                guards.append(Path(unit["source"]).name)
            if "-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch" in argv:
                renames.append(Path(unit["source"]).name)
            if unit["language"] == "fortran":
                self.assertIn(str(self.out / "modules"), argv)
                self.assertIn(str(self.wrf / "frame"), argv)
                self.assertIn("-J", argv)
        self.assertCountEqual(guards, ["wtimer.c", "wtimer.S"])
        self.assertEqual(renames, ["wt_kf_bridge.c"])
        boot = next(u for u in units if Path(u["source"]).name == "wt_kf_bootstrap.c")
        self.assertIn("-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=0", boot["argv"])
        self.assertIn("-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=0", boot["argv"])
        bridge = next(u for u in units if Path(u["source"]).name == "wt_kf_bridge.c")
        self.assertIn("generate-kf-bridge", bridge["depends_on"])
        link = plan["steps"][-1]
        self.assertEqual(link["kind"], "prepare_link")
        self.assertIn(str(self.wt / "plov/prepare_native_link.py"), link["argv"])
        self.assertEqual(len(link["depends_on"]), len(units))
        self.assertIn("module_cu_kfeta_mp_kf_eta_cps_", plan["wrap_symbols"])
        expected = dict(plan)
        digest = expected.pop("plan_sha256")
        self.assertEqual(digest, hashlib.sha256(json.dumps(expected, sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest())
        self.assertEqual(plan, json.loads((self.out / "build-manifest.json").read_text()))

    def test_unattested_node_local_profile_closes_units_flags_and_init_wrap(self):
        node_c = ("wt_kf_batch.c", "wt_kf_batch_codec.c", "wt_node_local.c",
                  "wt_kf_node.c", "wt_kf_node_live.c", "wt_kf_node_shm.c",
                  "wt_kf_init_bridge.c")
        for name in node_c:
            self.write(self.wt / "plov" / name, '#include "wt_kf_node.h"\n')
        self.write(self.wt / "plov/wt_kf_node.h", "fixture node header\n")
        self.config["node_local_profile"] = {"name": "unattested-v1"}
        plan = self.prepare()
        profile = plan["node_local_profile"]
        self.assertEqual(profile["name"], "unattested-v1")
        self.assertTrue(profile["enabled"])
        self.assertFalse(any(profile["attestation"].values()))
        self.assertFalse(profile["proof_activation_implemented"])
        units = [step for step in plan["steps"] if step["kind"] == "compile"]
        names = {Path(step["source"]).name: step for step in units}
        self.assertEqual(len(units), 20)
        for name in node_c + ("wt_kf_live.c", "wt_kf_native.c", "wt_kf_bootstrap.c", "wt_kf_bridge.c"):
            self.assertIn("-DWT_KF_NODE_LOCAL=1", names[name]["argv"])
        self.assertEqual([Path(step["source"]).name for step in units
                          if "-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch" in step["argv"]],
                         ["wt_kf_bridge.c"])
        self.assertEqual(plan["wrap_symbols"][-2:],
                         ["module_cu_kfeta_mp_kf_eta_cps_", "module_cu_kfeta_mp_kf_eta_init_"])

    def test_node_local_profile_rejects_proof_booleans_and_unknown_forms(self):
        for profile in ({}, {"name": "attested-v1"}, {"name": "unattested-v1", "proof": {}},
                        {"name": "unattested-v1", "source_abi_verified": True}, "unattested-v1"):
            with self.subTest(profile=profile):
                self.config["node_local_profile"] = profile
                with self.assertRaises(build.PrepareError):
                    self.prepare()
                self.assertFalse(self.out.exists())

    def test_module_directory_profile_defaults_and_preserves_order(self):
        provider = self.wrf / "external/provider modules"
        self.write(provider / "ESMF_Mod.MOD", "provider module\n")
        self.write(provider / "child.SMOD", "provider submodule\n")
        self.config["fortran_module_dirs"] = ["frame", "external/provider modules"]
        plan = self.prepare()
        self.assertEqual(plan["version"], 2)
        self.assertEqual(plan["compiler_config"]["fortran_module_dirs"],
                         ["frame", "external/provider modules"])
        records = plan["fortran_module_inputs"]
        self.assertEqual([record["relative_path"] for record in records],
                         ["frame", "external/provider modules"])
        self.assertEqual([member["name"] for member in records[1]["members"]],
                         ["ESMF_Mod.MOD", "child.SMOD"])
        hashes = {entry["path"] for entry in plan["input_hashes"]}
        self.assertTrue(all(member["path"] in hashes for record in records
                            for member in record["members"]))
        for step in plan["steps"]:
            if step.get("language") == "fortran":
                self.assertIn(["-I", str(self.wrf / "frame"), "-I", str(provider)],
                              [step["argv"][index:index + 4]
                               for index in range(len(step["argv"]) - 3)])

    def test_module_directory_profile_default_frame_is_snapshotted(self):
        plan = self.prepare()
        self.assertEqual(plan["compiler_config"]["fortran_module_dirs"], ["frame"])
        record = plan["fortran_module_inputs"][0]
        self.assertEqual(record["relative_path"], "frame")
        self.assertEqual(record["members"][0]["name"], "module_domain_type.mod")

    def test_module_directory_profile_rejects_invalid_or_empty_inputs(self):
        self.write(self.wrf / "external/good/provider.mod", "provider\n")
        bad_values = ([], ["external/good"], ["frame", "frame"],
                      ["frame", "/absolute"], ["frame", "external/../good"],
                      ["frame", "./external"], ["frame", "external//good"],
                      ["frame", ""], ["frame", 1], ["frame", "missing"])
        for value in bad_values:
            with self.subTest(value=value):
                self.config["fortran_module_dirs"] = value
                with self.assertRaises(build.PrepareError):
                    self.prepare()
                self.assertFalse(self.out.exists())
        self.config["fortran_module_dirs"] = ["frame", "external/good"]
        (self.wrf / "external/good/provider.mod").unlink()
        with self.assertRaises(build.PrepareError):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_module_directory_profile_rejects_symlinks_and_casefold_collisions(self):
        self.write(self.wrf / "external/one/Foo.MOD", "one\n")
        self.write(self.wrf / "external/two/foo.mod", "two\n")
        self.config["fortran_module_dirs"] = ["frame", "external/one", "external/two"]
        with self.assertRaises(build.PrepareError):
            self.prepare()
        self.assertFalse(self.out.exists())
        self.config["fortran_module_dirs"] = ["frame", "external/one"]
        alias = self.wrf / "external/alias"
        alias.symlink_to(self.wrf / "external/one", target_is_directory=True)
        self.config["fortran_module_dirs"] = ["frame", "external/alias"]
        with self.assertRaises(build.PrepareError):
            self.prepare()
        alias.unlink()
        (self.wrf / "external/one/Foo.MOD").unlink()
        module = self.wrf / "external/one/real.mod"
        self.write(module, "real\n")
        (self.wrf / "external/one/link.smod").symlink_to(module)
        with self.assertRaises(build.PrepareError):
            self.prepare()

    def test_module_directory_profile_enforces_directory_and_member_bounds(self):
        directories = ["frame"]
        for index in range(16):
            relative = f"providers/p{index}"
            self.write(self.wrf / relative / "provider.mod", "provider\n")
            directories.append(relative)
        self.config["fortran_module_dirs"] = directories
        with self.assertRaises(build.PrepareError):
            self.prepare()
        self.assertFalse(self.out.exists())
        self.config["fortran_module_dirs"] = ["frame", "providers/many"]
        for index in range(129):
            self.write(self.wrf / "providers/many" / f"m{index}.mod", "provider\n")
        with self.assertRaises(build.PrepareError):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_plov_module_shadow_is_rejected_before_output(self):
        self.write(self.wt / "plov/Shadow.MOD", "shadow\n")
        with self.assertRaises(build.PrepareError):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_existing_output_refused_without_changes(self):
        self.prepare()
        original = (self.out / "build-manifest.json").read_bytes()
        with self.assertRaises(build.PrepareError):
            self.prepare()
        self.assertEqual(original, (self.out / "build-manifest.json").read_bytes())

    def test_extra_libraries_forwarded_in_order_without_changing_baseline(self):
        original = json.loads(json.dumps(self.config["baseline_units"]))
        self.config["extra_libraries"] = ["rt", "m"]
        plan = self.prepare()
        self.assertEqual(plan["compiler_config"]["baseline_units"], original)
        self.assertEqual(plan["extra_libraries"], ["rt", "m"])
        self.assertEqual(plan["steps"][-1]["argv"][-4:],
                         ["--extra-library", "rt", "--extra-library", "m"])

    def test_extra_library_identifiers_are_bounded_and_fail_closed(self):
        for libraries in ("rt", ["-lrt"], ["../rt"], ["rt;touch"], [""],
                          [1], ["rt", "rt"], ["x" * 129],
                          ["lib" + str(i) for i in range(129)]):
            with self.subTest(libraries=libraries):
                self.config["extra_libraries"] = libraries
                with self.assertRaises(build.PrepareError):
                    self.prepare()
                self.assertFalse(self.out.exists())

    def test_cli_only_writes_recipe_and_accepts_intel_module_syntax(self):
        self.config["fortran_module_flag"] = "-module"
        self.config["fortran_flags"] = ["-real-size", "32", "-i4", "-qopenmp"]
        self.cfg.write_text(json.dumps(self.config))
        command = [sys.executable, str(ROOT / "plov/prepare_kf_build.py")]
        for key, value in (("wtune-root", self.wt), ("frozen-root", self.wrf),
                           ("cc", self.cc), ("fc", self.fc),
                           ("compiler-config", self.cfg), ("output-dir", self.out)):
            command += ["--" + key, str(value)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["manifest"],
                         str(self.out / "build-manifest.json"))
        plan = json.loads((self.out / "build-manifest.json").read_text())
        for step in plan["steps"]:
            if step.get("language") == "fortran":
                self.assertIn("-module", step["argv"])
        self.assertEqual([p.name for p in self.out.iterdir()], ["build-manifest.json"])

    def test_baseline_local_headers_are_hashed_and_missing_headers_refused(self):
        self.write(self.wt / "plov/baseline.c", '#include "baseline.h"\n')
        header = self.wt / "plov/baseline.h"
        self.write(header, '#include "inner.h"\n')
        self.write(self.wt / "plov/inner.h", "fixture header\n")
        plan = self.prepare()
        hashes = {entry["path"]: entry["sha256"] for entry in plan["input_hashes"]}
        self.assertEqual(hashes[str(header)], hashlib.sha256(header.read_bytes()).hexdigest())
        self.assertIn(str(self.wt / "plov/inner.h"), hashes)
        self.out = self.base / "second output"
        (self.wt / "plov/inner.h").unlink()
        with self.assertRaises(build.PrepareError):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_missing_input_creates_no_output(self):
        (self.wt / "plov/wt_kf_native.h").unlink()
        with self.assertRaises(build.PrepareError):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_reject_frozen_and_symlink_output(self):
        for target in (self.wrf / "new", self.wt / "new"):
            self.out = target
            with self.assertRaises(build.PrepareError):
                self.prepare()
        alias = self.base / "alias"
        alias.symlink_to(self.wrf, target_is_directory=True)
        self.out = alias / "new"
        with self.assertRaises(build.PrepareError):
            self.prepare()

    def test_source_escape_and_symlink_refused(self):
        outside = self.base / "outside.c"
        self.write(outside, "outside\n")
        for source in (str(outside), "../outside.c"):
            self.config["baseline_units"][2]["source"] = source
            with self.assertRaises(build.PrepareError):
                self.prepare()
        alias = self.wt / "plov/escape.c"
        alias.symlink_to(outside)
        self.config["baseline_units"][2]["source"] = "plov/escape.c"
        with self.assertRaises(build.PrepareError):
            self.prepare()

    def test_unsafe_flags_and_attestation_overrides_refused(self):
        for flag in ("-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=1", "-UFOO", "-Wp,-DX=1",
                     "@args", "-o/tmp/output", "-J/tmp/mod", "-module",
                     "-save-temps", "-MF/tmp/deps", "-include", "-Xclang", "-Wl,-o,x"):
            with self.subTest(flag=flag):
                self.config["c_flags"] = [flag]
                with self.assertRaises(build.PrepareError):
                    self.prepare()
                self.assertFalse(self.out.exists())

    def test_duplicates_reserved_kf_and_missing_timer_refused(self):
        original = json.loads(json.dumps(self.config))
        for replacement in (
            {"output_name": "timer-c.o"}, {"source": "plov/wt_kf_live.c"},
            {"output_name": "../escape.o"}, {"language": "shell"}):
            self.config = json.loads(json.dumps(original))
            self.config["baseline_units"][2].update(replacement)
            with self.assertRaises(build.PrepareError):
                self.prepare()
        self.config = original
        self.config["baseline_units"].pop(0)
        with self.assertRaises(build.PrepareError):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
