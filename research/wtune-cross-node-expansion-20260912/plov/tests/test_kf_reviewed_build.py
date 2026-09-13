import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/20260913-node-local-physics/build-reviewed"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "plov" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("prepare_kf_build")
executor = load("execute_build_recipe")
reviewer = load("kf_review_record")


class ReviewedKfBuildTests(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="reviewed-kf-", dir=ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.wt, self.wrf = self.base / "wtune", self.base / "frozen"
        self.plov, self.out = self.wt / "plov", self.base / "reviewed output"
        self.plov.mkdir(parents=True)
        fixed = (*builder.KF_C, *builder.KF_F, *builder.KF_NODE_C, *builder.HEADERS,
                 *builder.GENERATORS, "kf_cps_schema.json", "wtimer.c", "wtimer.S")
        for name in fixed:
            self.write(self.plov / name, "fixture " + name + "\n")
        generator = ('import pathlib,sys\nargs=sys.argv[1:]\n'
                     'pathlib.Path(args[args.index("--output")+1]).write_text("bridge\\n")\n')
        self.write(self.plov / "gen_kf_bridge.py", generator)
        shutil.copyfile(ROOT / "plov/prepare_native_link.py", self.plov / "prepare_native_link.py")
        self.write(self.wrf / "phys/module_cu_kfeta.f90", "native KF\n")
        self.write(self.wrf / "frame/module_domain_type.mod", "module\n")
        self.write(self.wrf / "main/wrf.o", "baseline object\n")
        self.write(self.wrf / "main/libwrflib.a", "baseline archive\n")
        self.libdir = self.base / "libraries"
        self.library = self.libdir / "libreview.so"
        self.write(self.library, "reviewed library\n")
        link_line = " ".join(("mpiifx", "-o", "wrf.exe", str(self.wrf / "main/wrf.o"),
                              str(self.wrf / "main/libwrflib.a"), "-L" + str(self.libdir),
                              "-lreview"))
        self.write(self.wrf / "compile.log", link_line + "\n")
        compiler_body = '''import os,pathlib,sys
args=sys.argv[1:]
if "-c" in args:
    source=args[args.index("-c")+1]
    pathlib.Path(args[args.index("-o")+1]).write_text("object:"+pathlib.Path(source).name+"\\n")
    if os.environ.get("WT_FAKE_MUTATE_OBJECT") == "1" and pathlib.Path(source).name == "wtimer.S":
        pathlib.Path(args[args.index("-o")+1]).with_name("timer.o").write_text("mutated object\\n")
else:
    if os.environ.get("WT_FAKE_LINK_FAIL") == "1":
        sys.exit(17)
    pathlib.Path(args[args.index("-o")+1]).write_text("linked candidate\\n")
    map_arg=next(item for item in args if item.startswith("-Wl,-Map,"))
    pathlib.Path(map_arg.split(",",2)[2]).write_text("link map\\n")
'''
        self.compiler = self.base / "mpiifx"
        self.write(self.compiler, "#!" + sys.executable + "\n" + compiler_body)
        self.compiler.chmod(0o700)
        self.config_path = self.base / "compiler.json"
        self.record_path = self.base / "review.json"
        self.config = {
            "baseline_units": [
                {"source": "plov/wtimer.c", "language": "c", "output_name": "timer.o", "flags": []},
                {"source": "plov/wtimer.S", "language": "asm", "output_name": "timer-asm.o", "flags": []}],
            "wrap_symbols": ["baseline_symbol_"], "c_flags": ["-O2"],
            "fortran_flags": [], "fortran_module_flag": "-J", "extra_libraries": [],
            "node_local_profile": {"name": "unattested-v1"}}
        self._write_review_record()
        self.plan = builder.prepare(wtune_root=self.wt, frozen_root=self.wrf,
                                    cc=self.compiler, fc=self.compiler,
                                    compiler_config=self.config_path, output_dir=self.out)
        self.manifest = self.out / "build-manifest.json"
        self.digest = self.sha(self.manifest)

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def _write_review_record(self):
        self.config_path.write_text(json.dumps(self.config))
        probe_output = self.base / "unattested probe"
        probe = builder.prepare(wtune_root=self.wt, frozen_root=self.wrf,
                                cc=self.compiler, fc=self.compiler,
                                compiler_config=self.config_path, output_dir=probe_output)
        self.unattested_plan = probe
        modules = probe["fortran_module_inputs"]
        excluded = {str(self.compiler.resolve()), str(Path(sys.executable).resolve()),
                    str(self.config_path.resolve()), str(self.plov / "kf_cps_schema.json")}
        excluded.update(member["path"] for item in modules for member in item["members"])
        sources = [entry for entry in probe["input_hashes"] if entry["path"] not in excluded]
        evidence = {}
        claim_kinds = dict(reviewer.CLAIMS, concurrency=False)
        claim_kinds["concurrency"] = "concurrency-limit"
        for claim, kind in claim_kinds.items():
            path = self.base / "evidence" / (claim + ".md")
            self.write(path, "human reviewed evidence for " + claim + "\n")
            evidence[claim] = {"kind": kind, "path": str(path), "sha256": self.sha(path)}
        reviewed_config = json.loads(json.dumps(probe["compiler_config"]))
        reviewed_config["node_local_profile"] = {"name": "reviewed-g0-v1"}
        baseline = [{"path": str(path), "sha256": self.sha(path)} for path in
                    (self.wrf / "main/wrf.o", self.wrf / "main/libwrflib.a")]
        libraries = {
            "explicit_search_paths": [str(self.libdir)], "implicit_search_paths": [],
            "tokens": ["-lreview"],
            "resolutions": [{"token": "-lreview", "link_path": str(self.library),
                             "realpath": str(self.library), "sha256": self.sha(self.library),
                             "soname": "libreview.so", "shadows": []}]}
        lanes = {
            "sources": sources,
            "schema": {"path": str(self.plov / "kf_cps_schema.json"),
                       "sha256": self.sha(self.plov / "kf_cps_schema.json")},
            "compilers": [{"role": role, "path": str(path), "sha256": self.sha(path)}
                          for role, path in (("cc", self.compiler), ("fc", self.compiler),
                                             ("python", Path(sys.executable).resolve()))],
            "options": reviewed_config, "modules": modules,
            "baseline_objects": baseline, "libraries": libraries}
        fingerprints = {name: reviewer.canonical_hash(lanes[name])
                        for name in reviewer.FINGERPRINTS[:-1]}
        fingerprints["scientific_build_closure"] = reviewer.canonical_hash(fingerprints)
        claims = {claim: {"conclusion": "verified", "evidence": [evidence[claim]]}
                  for claim in reviewer.CLAIMS}
        claims["concurrency"] = {"conclusion": "not_verified",
                                  "reason": "Fortran I/O thread safety is not proven; serialize every CPS entry.",
                                  "evidence": [evidence["concurrency"]]}
        record = {
            "artifact_type": "wtune_kf_reviewed_g0_attestation", "version": 1,
            "profile": reviewer.PROFILE,
            "profile_id": reviewer.PROFILE + ":" + fingerprints["scientific_build_closure"],
            "human_review": {"reviewer": "fixture reviewer", "reviewed_utc": "2026-09-12T23:29:37+00:00",
                             "statement": reviewer.STATEMENT},
            "claims": claims, "fingerprints": fingerprints,
            "baseline_link_inputs": baseline, "library_contract": libraries,
            "link_output": {"candidate": str(self.out / "wrf-wtune.exe"),
                            "map": str(self.out / "wrf-wtune.map"),
                            "identity_record": str(self.out / "wrf-wtune.identity.json")}}
        self.record_path.write_text(json.dumps(record, indent=2))
        self.config = reviewed_config
        self.config["node_local_profile"]["review_record"] = str(self.record_path)
        self.config_path.write_text(json.dumps(self.config))

    def run_recipe(self, *, link=False):
        return executor.run_recipe(self.manifest, self.digest, execute=True, link=link)

    def test_reviewed_profile_macros_and_human_claims(self):
        self.assertEqual(self.plan["version"], 3)
        self.assertTrue(all(self.plan["attestation"].values()))
        profile = self.plan["node_local_profile"]
        self.assertTrue(profile["proof_activation_implemented"])
        self.assertFalse(profile["attestation"]["concurrency_verified"])
        c_steps = [step for step in self.plan["steps"]
                   if step.get("kind") == "compile" and step["language"] == "c"]
        macro = '-DWT_KF_BUILD_PROFILE_ID="' + profile["profile_id"] + '"'
        self.assertTrue(all(macro in step["argv"] for step in c_steps))
        self.assertTrue(all("-DWT_KF_BUILD_CONCURRENT_CPS_VERIFIED=0" in step["argv"]
                            for step in c_steps))
        renamed = [Path(step["source"]).name for step in c_steps
                   if "-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch" in step["argv"]]
        self.assertEqual(renamed, ["wt_kf_bridge.c"])
        init_enabled = [Path(step["source"]).name for step in c_steps
                        if "-DWT_KF_BUILD_INIT_TABLES_VERIFIED=1" in step["argv"]]
        self.assertEqual(init_enabled, ["wt_kf_init_bridge.c"])
        self.assertTrue(all("-DWT_KF_BUILD_INIT_TABLES_VERIFIED=1" not in step["argv"]
                            for step in self.unattested_plan["steps"]
                            if step.get("kind") == "compile"))

    def test_review_schema_rejects_boolean_and_local_counter_claims(self):
        lanes = {key: self.plan["scientific_build_closure"][key]
                 for key in ("sources", "schema", "compilers", "options", "modules")}
        original = json.loads(self.record_path.read_text())
        bad = json.loads(json.dumps(original))
        bad["human_review"]["automatic"] = True
        self.record_path.write_text(json.dumps(bad))
        with self.assertRaises(reviewer.ReviewError):
            reviewer.load(self.record_path, lanes=lanes, output_dir=self.out)
        bad = json.loads(json.dumps(original))
        bad["claims"]["node_init_static"]["evidence"][0]["kind"] = "local-counter"
        self.record_path.write_text(json.dumps(bad))
        with self.assertRaises(reviewer.ReviewError):
            reviewer.load(self.record_path, lanes=lanes, output_dir=self.out)

    def test_default_execute_prepares_link_only(self):
        result = self.run_recipe()
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["link_executed"])
        self.assertTrue((self.out / "link-manifest.json").is_file())
        self.assertFalse((self.out / "wrf-wtune.exe").exists())

    def test_end_to_end_fake_link_records_terminal_identity(self):
        result = self.run_recipe(link=True)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["link_executed"])
        for name in ("wrf-wtune.exe", "wrf-wtune.map", "wrf-wtune.identity.json"):
            self.assertTrue((self.out / name).is_file())
        identity = json.loads((self.out / "wrf-wtune.identity.json").read_text())
        self.assertIn("not native ABI", identity["attestation_note"])
        self.assertEqual(identity["candidate"]["sha256"], self.sha(self.out / "wrf-wtune.exe"))
        status = json.loads((self.out / "execution.status.json").read_text())
        self.assertEqual(status["native_identity_record"]["sha256"],
                         self.sha(self.out / "wrf-wtune.identity.json"))

    def test_review_source_library_module_and_flags_drift_refused_before_marker(self):
        targets = (self.record_path, self.plov / "wt_kf_live.c", self.library,
                   self.wrf / "frame/module_domain_type.mod", self.config_path)
        for target in targets:
            with self.subTest(target=target):
                original = target.read_bytes()
                target.write_bytes(original + b"drift")
                with self.assertRaises(executor.RecipeError):
                    self.run_recipe(link=True)
                self.assertFalse((self.out / "execution.started").exists())
                target.write_bytes(original)

    def test_existing_and_symlink_link_outputs_refused_before_marker(self):
        candidate = self.out / "wrf-wtune.exe"
        candidate.write_text("existing")
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(link=True)
        self.assertFalse((self.out / "execution.started").exists())
        candidate.unlink()
        candidate.symlink_to(self.library)
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(link=True)
        self.assertFalse((self.out / "execution.started").exists())

    def test_fake_link_failure_is_terminal_and_cannot_replay(self):
        with mock.patch.dict(os.environ, {"WT_FAKE_LINK_FAIL": "1"}):
            result = self.run_recipe(link=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["step_id"], "link-reviewed-candidate")
        self.assertEqual(result["returncode"], 17)
        self.assertFalse((self.out / "wrf-wtune.identity.json").exists())
        with self.assertRaises(executor.RecipeError):
            self.run_recipe(link=True)

    def test_changed_compiler_output_stops_at_next_boundary(self):
        with mock.patch.dict(os.environ, {"WT_FAKE_MUTATE_OBJECT": "1"}):
            result = self.run_recipe(link=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "stale_output")
        self.assertEqual(result["step_id"], "compile-timer-asm.o")
        self.assertFalse((self.out / "link-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
