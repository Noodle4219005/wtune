#!/usr/bin/env python3
"""Local fixtures for native-preproof ledger and loader-list parsing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("verify_native_preproof",
                                              HERE / "verify-native-preproof.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
SCRATCH = Path("/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/"
               "build-reviewed/native-preproof-verifier-test")


class NativePreproofVerifierTests(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(SCRATCH, ignore_errors=True)
        SCRATCH.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(SCRATCH, ignore_errors=True)

    def test_loader_output_controls_selection_and_conflicts_fail(self):
        first = SCRATCH / "first" / "libpick.so"
        selected = SCRATCH / "selected" / "libpick.so"
        first.parent.mkdir()
        selected.parent.mkdir()
        first.write_bytes(b"first")
        selected.write_bytes(b"selected")
        text = f"libpick.so => {selected} (0x1)\n"
        mapping = VERIFY.loader_mapping(text)
        self.assertEqual(mapping["libpick.so"], selected.resolve())
        self.assertNotEqual(mapping["libpick.so"], first.resolve())
        with self.assertRaisesRegex(ValueError, "ambiguous loader selection"):
            VERIFY.loader_mapping(text + f"libpick.so => {first} (0x2)\n")

    def test_object_must_match_executor_ledger(self):
        obj = SCRATCH / "unit.o"
        obj.write_bytes(b"object-v1")
        digest = hashlib.sha256(obj.read_bytes()).hexdigest()
        steps = [{"output": str(obj)}]
        status = {"generated_outputs": [{"path": str(obj), "sha256": digest,
                                          "size": obj.stat().st_size}]}
        VERIFY.validate_object_ledger(steps, status)
        obj.write_bytes(b"object-v2")
        with self.assertRaisesRegex(ValueError, "differs from executor ledger"):
            VERIFY.validate_object_ledger(steps, status)

    def test_changed_prospective_source_module_and_baseline_fail(self):
        source = SCRATCH / "source.c"
        baseline = SCRATCH / "baseline.o"
        modules = SCRATCH / "modules"
        module = modules / "physics.mod"
        modules.mkdir()
        source.write_bytes(b"source-v1")
        baseline.write_bytes(b"baseline-v1")
        module.write_bytes(b"module-v1")

        def entry(path, include_size=False):
            value = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if include_size:
                value["size"] = path.stat().st_size
            return value

        member = {"name": module.name, **entry(module, include_size=True)}
        membership = hashlib.sha256(json.dumps([member], sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
        prospective = {"lanes": {"sources": [entry(source)],
                                  "baseline_objects": [entry(baseline)],
                                  "modules": [{"path": str(modules), "members": [member],
                                               "membership_sha256": membership}]}}
        VERIFY.validate_prospective_inputs(prospective)
        for path in (source, module, baseline):
            original = path.read_bytes()
            path.write_bytes(original + b"-changed")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "changed"):
                VERIFY.validate_prospective_inputs(prospective)
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
