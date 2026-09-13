import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/20260913-node-local-physics/build-reviewed"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "plov" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("prepare_kf_build")
executor = load("execute_build_recipe")


class ExecuteBuildRecipeTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="wt-recipe-exec-", dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.wt, self.wrf, self.out = (self.base / name for name in
                                      ("wtune source", "frozen source", "recipe output"))
        self.plov = self.wt / "plov"
        self.plov.mkdir(parents=True)
        names = (*builder.KF_C, *builder.KF_F, *builder.HEADERS,
                 *builder.GENERATORS, "kf_cps_schema.json", "prepare_native_link.py",
                 "wtimer.c", "wtimer.S")
        for name in names:
            (self.plov / name).write_text("fixture input\n")
        (self.wrf / "frame").mkdir(parents=True)
        (self.wrf / "phys").mkdir()
        (self.wrf / "main").mkdir()
        for path in ("compile.log", "frame/module_domain_type.mod", "phys/module_cu_kfeta.f90"):
            (self.wrf / path).write_text("frozen fixture\n")
        mock = '''import json, pathlib, sys, time
args = sys.argv[1:]
assert sys.dont_write_bytecode, "source bytecode writes must be disabled"
with open("observed.jsonl", "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if "--manifest" in args:
    pathlib.Path(args[args.index("--manifest")+1]).write_text('{"argv":["DO_NOT_EXECUTE_LINK"]}')
else:
    flag = "-o" if "-o" in args else "--output"
    pathlib.Path(args[args.index(flag)+1]).write_text("mock output")
'''
        self.compiler = self.base / "mock compiler with spaces"
        self.compiler.write_text("#!" + sys.executable + "\n" + mock)
        self.compiler.chmod(0o700)
        for name in ("gen_kf_bridge.py", "prepare_native_link.py"):
            (self.plov / name).write_text(mock)
        self.config = self.base / "config.json"
        self.config.write_text(json.dumps({
            "baseline_units": [
                {"source": "plov/wtimer.c", "language": "c", "output_name": "timer.o", "flags": []},
                {"source": "plov/wtimer.S", "language": "asm", "output_name": "timer_asm.o", "flags": []}],
            "wrap_symbols": ["timer_"], "c_flags": ["-O2"], "fortran_flags": [],
            "fortran_module_flag": "-J", "extra_libraries": ["rt"]}))
        self.plan = builder.prepare(wtune_root=self.wt, frozen_root=self.wrf,
                                    cc=self.compiler, fc=self.compiler,
                                    compiler_config=self.config, output_dir=self.out)
        self.manifest = self.out / "build-manifest.json"
        self.digest = self.sha(self.manifest)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_recipe(self, execute=False, timeout=60):
        return executor.run_recipe(self.manifest, self.digest, execute=execute,
                                   timeout_seconds=timeout)

    def enable_node_profile(self):
        for name in builder.KF_NODE_C:
            (self.plov / name).write_text("fixture node input\n")
        self.config.write_text(json.dumps({
            "baseline_units": [
                {"source": "plov/wtimer.c", "language": "c", "output_name": "timer.o", "flags": []},
                {"source": "plov/wtimer.S", "language": "asm", "output_name": "timer_asm.o", "flags": []}],
            "wrap_symbols": ["timer_"], "c_flags": ["-O2"], "fortran_flags": [],
            "fortran_module_flag": "-J", "extra_libraries": ["rt"],
            "node_local_profile": {"name": "unattested-v1"}}))
        self.out = self.base / "node recipe output"
        self.plan = builder.prepare(wtune_root=self.wt, frozen_root=self.wrf,
                                    cc=self.compiler, fc=self.compiler,
                                    compiler_config=self.config, output_dir=self.out)
        self.manifest, self.digest = self.out / "build-manifest.json", self.sha(self.out / "build-manifest.json")

    def rewrite(self):
        canonical = dict(self.plan)
        canonical.pop("plan_sha256")
        self.plan["plan_sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        self.manifest.write_text(json.dumps(self.plan))
        self.digest = self.sha(self.manifest)

    def test_dry_validation_creates_no_artifacts(self):
        result = self.run_recipe()
        self.assertEqual(result["status"], "validated")
        self.assertEqual([path.name for path in self.out.iterdir()], ["build-manifest.json"])

    def test_cli_default_is_validation_only(self):
        result = subprocess.run([sys.executable, str(ROOT / "plov/execute_build_recipe.py"),
                                 "--manifest", str(self.manifest), "--expected-sha256", self.digest],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "validated")
        self.assertEqual([path.name for path in self.out.iterdir()], ["build-manifest.json"])

    def test_execute_exact_argv_spaces_and_never_link(self):
        before = {str(path): self.sha(path) for path in self.wrf.rglob("*") if path.is_file()}
        result = self.run_recipe(True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(before, {str(path): self.sha(path) for path in self.wrf.rglob("*") if path.is_file()})
        self.assertTrue((self.out / "execution.started").is_file())
        self.assertFalse((self.out / "wrf-wtune.exe").exists())
        self.assertTrue((self.out / "link-manifest.json").is_file())
        observed = [json.loads(line) for line in (self.out / "observed.jsonl").read_text().splitlines()]
        commands = [step for step in self.plan["steps"] if step["kind"] != "mkdir"]
        expected = [step["argv"][2:] if step["kind"] in ("generate", "prepare_link")
                    else step["argv"][1:] for step in commands]
        self.assertEqual(observed, expected)
        statuses = list((self.out / "execution-logs").glob("*.status.json"))
        self.assertEqual(len(statuses), len(commands))
        self.assertTrue(all(json.loads(path.read_text())["returncode"] == 0 for path in statuses))
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)

    def test_node_local_recipe_validates_and_fake_executes_full_closed_closure(self):
        self.enable_node_profile()
        self.assertEqual(executor.validate(self.manifest, self.digest), self.plan)
        result = self.run_recipe(True)
        self.assertEqual(result["status"], "complete")
        commands = [json.loads(line) for line in (self.out / "observed.jsonl").read_text().splitlines()]
        node_sources = {str(self.plov / name) for name in builder.KF_NODE_C}
        compiled = [command for command in commands if "-c" in command]
        self.assertTrue(node_sources <= {command[command.index("-c") + 1] for command in compiled})
        for command in compiled:
            if Path(command[command.index("-c") + 1]).suffix == ".c":
                self.assertIn("-DWT_KF_NODE_LOCAL=1", command)
        prepared_link = commands[-1]
        self.assertIn("module_cu_kfeta_mp_kf_eta_cps_", prepared_link)
        self.assertIn("module_cu_kfeta_mp_kf_eta_init_", prepared_link)

    def test_unattested_profile_cannot_request_link(self):
        self.enable_node_profile()
        with self.assertRaises(executor.RecipeError):
            executor.run_recipe(self.manifest, self.digest, execute=True, link=True)
        self.assertFalse((self.out / "execution.started").exists())

    def test_node_profile_manifest_drift_and_malformed_profile_are_rejected(self):
        self.enable_node_profile()
        self.plan["node_local_profile"]["attestation"]["init_verified"] = True
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe()
        self.plan["node_local_profile"] = {"enabled": True}
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe()

    def test_tampered_outer_and_canonical_hash_rejected(self):
        self.manifest.write_text(self.manifest.read_text() + " ")
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.digest = self.sha(self.manifest)
        self.plan["executed"] = True
        self.manifest.write_text(json.dumps(self.plan))
        self.digest = self.sha(self.manifest)
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.assertFalse((self.out / "execution.started").exists())

    def test_input_hash_change_rejected_before_marker(self):
        self.compiler.write_text("changed input")
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.assertFalse((self.out / "execution.started").exists())

    def test_added_module_rejected_before_execution(self):
        (self.wrf / "frame/Added.SMOD").write_text("added\n")
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.assertFalse((self.out / "execution.started").exists())

    def test_changed_module_rejected_before_execution(self):
        (self.wrf / "frame/module_domain_type.mod").write_text("changed\n")
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.assertFalse((self.out / "execution.started").exists())

    def test_deleted_module_rejected_before_execution(self):
        (self.wrf / "frame/module_domain_type.mod").unlink()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.assertFalse((self.out / "execution.started").exists())

    def test_module_drift_during_command_stops_at_boundary(self):
        module = self.wrf / "frame/module_domain_type.mod"
        self.replace_compiler(
            "import pathlib,sys\n"
            "args=sys.argv[1:]\n"
            "pathlib.Path(args[args.index('-o')+1]).write_text('mock object')\n"
            f"pathlib.Path({str(module)!r}).write_text('changed module')\n")
        result = self.run_recipe(True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "stale_input")
        self.assertEqual(result["step_id"], "compile-timer.o")
        self.assertFalse((self.out / "objects/timer_asm.o").exists())

    def test_module_addition_and_deletion_during_command_stop_at_boundary(self):
        module = self.wrf / "frame/module_domain_type.mod"
        added = self.wrf / "frame/added.SMOD"
        self.replace_compiler(
            "import pathlib,sys\n"
            "args=sys.argv[1:]\n"
            "pathlib.Path(args[args.index('-o')+1]).write_text('mock object')\n"
            f"pathlib.Path({str(module)!r}).unlink()\n"
            f"pathlib.Path({str(added)!r}).write_text('added module')\n")
        result = self.run_recipe(True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "stale_input")
        self.assertEqual(result["step_id"], "compile-timer.o")

    def test_plov_module_shadow_added_during_command_stops_at_boundary(self):
        shadow = self.plov / "Shadow.MOD"
        self.replace_compiler(
            "import pathlib,sys\n"
            "args=sys.argv[1:]\n"
            "pathlib.Path(args[args.index('-o')+1]).write_text('mock object')\n"
            f"pathlib.Path({str(shadow)!r}).write_text('shadow')\n")
        result = self.run_recipe(True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "stale_input")
        self.assertEqual(result["step_id"], "compile-timer.o")

    def test_module_profile_manifest_tampering_is_rejected(self):
        self.plan["compiler_config"]["fortran_module_dirs"] = ["frame", "phys"]
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe()

    def test_module_profile_ledger_and_argv_tampering_are_rejected(self):
        module = str(self.wrf / "frame/module_domain_type.mod")
        entry = next(entry for entry in self.plan["input_hashes"] if entry["path"] == module)
        entry["sha256"] = "0" * 64
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe()
        entry["sha256"] = self.sha(Path(module))
        for step in self.plan["steps"]:
            if step.get("language") == "fortran":
                position = step["argv"].index(str(self.wrf / "frame"))
                step["argv"][position] = str(self.wrf / "phys")
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe()
        self.plan["compiler_config"]["fortran_module_dirs"] = ["frame"]
        self.plan["fortran_module_inputs"][0]["members"][0]["name"] = "forged.mod"
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe()

    def test_symlink_output_and_existing_output_rejected(self):
        (self.out / "objects").symlink_to(self.wrf, target_is_directory=True)
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.assertFalse((self.out / "execution.started").exists())
        (self.out / "objects").unlink()
        (self.out / "objects").mkdir()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)

    def test_proof_enable_is_rejected_even_with_rehashed_manifest(self):
        self.plan["attestation"]["source_abi_verified"] = True
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.assertFalse((self.out / "execution.started").exists())

    def test_bad_dependencies_and_hidden_argv_output_rejected(self):
        self.plan["steps"][2]["depends_on"] = ["nonexistent"]
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.plan["steps"][2]["depends_on"] = ["prepare-directories"]
        self.plan["steps"][2]["argv"].insert(1, "-o" + str(self.wrf / "overwrite"))
        self.rewrite()
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)
        self.assertFalse((self.out / "execution.started").exists())

    def replace_compiler(self, body):
        self.compiler.write_text("#!" + sys.executable + "\n" + body)
        for entry in self.plan["input_hashes"]:
            if entry["path"] == str(self.compiler):
                entry["sha256"] = self.sha(self.compiler)
        self.rewrite()

    def test_failure_stops_and_cannot_replay(self):
        self.replace_compiler("import sys\nprint('native failure', file=sys.stderr)\nsys.exit(17)\n")
        result = self.run_recipe(True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 17)
        self.assertFalse((self.out / "link-manifest.json").exists())
        self.assertEqual(len(list((self.out / "execution-logs").glob("*.status.json"))), 2)
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)

    def test_compiler_mutating_pinned_source_stops_before_second_compile(self):
        source = str(self.plov / "wtimer.c")
        self.replace_compiler(
            "import pathlib,sys\n"
            "args=sys.argv[1:]\n"
            "pathlib.Path(args[args.index('-o')+1]).write_text('mock object')\n"
            f"pathlib.Path({source!r}).write_text('modified pinned fixture')\n")
        result = self.run_recipe(True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "stale_input")
        self.assertEqual(result["step_id"], "compile-timer.o")
        self.assertTrue((self.out / "objects/timer.o").is_file())
        self.assertFalse((self.out / "objects/timer_asm.o").exists())
        self.assertFalse((self.out / "link-manifest.json").exists())
        statuses = list((self.out / "execution-logs").glob("*.status.json"))
        self.assertEqual(len(statuses), 2)
        self.assertEqual(json.loads((self.out / "execution-logs/002.status.json").read_text())["reason"], "stale_input")
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)

    def test_timeout_stops_and_cannot_replay(self):
        self.replace_compiler("import time\ntime.sleep(10)\n")
        result = self.run_recipe(True, timeout=1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "timeout")
        self.assertFalse((self.out / "link-manifest.json").exists())
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(True)

    def test_timeout_must_be_bounded(self):
        for value in (0, -1, 601, True):
            with self.subTest(value=value), self.assertRaises(executor.RecipeError):
                self.run_recipe(True, timeout=value)


if __name__ == "__main__":
    unittest.main()
