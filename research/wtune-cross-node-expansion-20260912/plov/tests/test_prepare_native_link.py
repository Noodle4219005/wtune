import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plov"))
import prepare_native_link as native


class PrepareNativeLinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="wt-native-link-")
        self.base = Path(self.tmp.name)
        self.frozen = self.base / "frozen WRF"
        self.main = self.frozen / "main"
        self.ext = self.frozen / "external lib"
        self.main.mkdir(parents=True)
        self.ext.mkdir()
        for path in (self.main / "wrf.o", self.main / "module top.o",
                     self.main / "libwrflib.a", self.ext / "libx.a"):
            path.write_bytes(path.name.encode())
        self.wrapper = self.base / "wrapper one.o"
        self.wrapper.write_bytes(b"wrapper")
        self.output = self.base / "out" / "wrf-wtune.exe"
        self.output.parent.mkdir()
        self.manifest = self.base / "out" / "command.json"
        self.log = self.base / "compile.log"
        self.command = (
            "time mpiifx -o wrf.exe -qopenmp wrf.o '../main/module top.o' "
            "libwrflib.a -L'../external lib' -lnetcdf "
            + repr(str(self.ext / "libx.a"))
        )
        self.log.write_text("noise\n" + self.command + "\nsuccess\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **changes):
        values = dict(compile_log=self.log, link_cwd=self.main,
                      output=self.output, manifest=self.manifest,
                      frozen_root=self.frozen, wrapper_objects=[self.wrapper],
                      wrap_symbols=["module_cu_kfeta_mp_kf_eta_cps_"])
        values.update(changes)
        return values

    def test_resolves_historical_inputs_hashes_and_preserves_order(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (self.main / "wrf.o", self.main / "module top.o",
                            self.main / "libwrflib.a", self.ext / "libx.a")}
        self.log.write_text("mpiifx -c -o module.o ignored.f90\nnoise\n" + self.command + "\n", encoding="utf-8")
        result = native.prepare(**self.args())
        self.assertEqual(result["source_line"], 3)
        self.assertEqual(result["argv"][0], "mpiifx")
        self.assertEqual(result["argv"][1:3], ["-o", str(self.output.resolve())])
        self.assertEqual(result["historical_cwd"], str(self.main.resolve()))
        self.assertEqual(result["link_cwd"], str(self.output.parent.resolve()))
        lib = str((self.main / "libwrflib.a").resolve())
        pos = result["argv"].index(lib)
        self.assertEqual(result["argv"][pos - 3:pos], ["-Wl,--start-group",
                         "-Wl,--wrap=module_cu_kfeta_mp_kf_eta_cps_", str(self.wrapper.resolve())])
        self.assertEqual(result["argv"][pos + 1], "-Wl,--end-group")
        self.assertIn(str((self.ext).resolve()), result["resolved_library_search_paths"])
        self.assertEqual(result["unresolved_implicit_dependencies"], ["-lnetcdf"])
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        self.assertFalse(self.output.exists())
        self.assertTrue(self.manifest.exists())
        self.assertEqual(json.loads(self.manifest.read_text()), result)

    def test_rejects_ambiguous_line_or_bad_output(self):
        self.log.write_text(self.command + "\n" + self.command + "\n", encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args())
        selected = native.prepare(**self.args(line=1, output=self.base / "selected.exe", manifest=self.base / "selected.json"))
        self.assertEqual(selected["source_line"], 1)
        self.log.write_text(self.command + " -o wrf.exe", encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args(manifest=self.base / "multi.json"))
        self.log.write_text(self.command.replace("-o wrf.exe", "-o other.exe"), encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args())

    def test_rejects_controls_missing_input_and_existing_manifest(self):
        self.log.write_text(self.command + " ; touch nope\n", encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args())
        self.log.write_text(self.command.replace("wrf.o", "wrf.o;touch"), encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args())
        self.log.write_text(self.command.replace(str(self.ext / "libx.a"), "missing.a"), encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args())
        self.log.write_text(self.command, encoding="utf-8")
        self.manifest.write_text("old", encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args())
        self.assertEqual(self.manifest.read_text(), "old")

    def test_rejects_symlink_into_frozen_and_invalid_wrapper_symbol(self):
        linked = self.base / "linked"
        linked.symlink_to(self.frozen, target_is_directory=True)
        with self.assertRaises(native.PrepareError): native.prepare(**self.args(output=linked / "new.exe"))
        with self.assertRaises(native.PrepareError): native.prepare(**self.args(wrap_symbols=["bad;symbol"]))

    def test_rejects_hidden_outputs_nonlink_driver_and_unsafe_wrapper(self):
        for addition in (" -oother", " --output evil", " -Wl,-o,evil", " -Wl,-o/evil", " -Wl,-Map,evil", " -Wl,--Map=evil", " -MF dep.d", " -save-temps", " @args.rsp", " -c", " -E"):
            self.log.write_text(self.command + addition, encoding="utf-8")
            with self.assertRaises(native.PrepareError, msg=addition): native.prepare(**self.args(manifest=self.base / ("m" + str(len(addition)) + ".json")))
        self.log.write_text(self.command.replace("mpiifx", "sh", 1), encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args(manifest=self.base / "shell.json"))
        bad = self.base / "not-object.a"; bad.write_bytes(b"x")
        self.log.write_text(self.command, encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args(wrapper_objects=[bad]))

    def test_rejects_same_output_and_manifest(self):
        same = self.base / "out" / "same"
        with self.assertRaises(native.PrepareError): native.prepare(**self.args(output=same, manifest=same))

    def test_resolves_slash_relative_allowlisted_driver(self):
        driver = self.main / "mpiifx"
        driver.write_bytes(b"not executed")
        self.log.write_text(self.command.replace("mpiifx", "./mpiifx", 1), encoding="utf-8")
        result = native.prepare(**self.args(output=self.base / "driver.exe", manifest=self.base / "driver.json"))
        self.assertEqual(result["driver"], str(driver.resolve()))

    def test_accepts_minimal_gnu_flags_and_resolves_object_symlink(self):
        target = self.main / "real-object.o"; target.write_bytes(b"real")
        alias = self.main / "alias.o"; alias.symlink_to(target.name)
        self.log.write_text(self.command.replace("time mpiifx", "gfortran -O2 -fopenmp").replace("wrf.o", "alias.o"), encoding="utf-8")
        result = native.prepare(**self.args(output=self.base / "gnu.exe", manifest=self.base / "gnu.json"))
        self.assertEqual(result["driver"], "gfortran")
        self.assertIn(str(target.resolve()), result["argv"])

    def test_rejects_source_operands_even_when_absolute(self):
        source = self.base / "source.f90"; source.write_text("end", encoding="utf-8")
        self.log.write_text(self.command + " '" + str(source) + "'", encoding="utf-8")
        with self.assertRaises(native.PrepareError): native.prepare(**self.args(manifest=self.base / "source.json"))

    def test_preserves_allowed_linker_options_and_separate_library_operand(self):
        self.log.write_text(self.command.replace("-lnetcdf", "-l netcdf") +
                            " -Wl,--as-needed,--wrap=extra_symbol", encoding="utf-8")
        result = native.prepare(**self.args())
        self.assertIn("-Wl,--as-needed,--wrap=extra_symbol", result["argv"])
        index = result["argv"].index("-l")
        self.assertEqual(result["argv"][index + 1], "netcdf")
        self.assertEqual(result["unresolved_implicit_dependencies"], ["-lnetcdf"])

    def test_response_file_cannot_hide_as_an_option_value(self):
        self.log.write_text(self.command + " -fp-model @flags.rsp", encoding="utf-8")
        with self.assertRaises(native.PrepareError):
            native.prepare(**self.args())

    def test_extra_libraries_are_recorded_and_validated(self):
        result = native.prepare(**self.args(extra_libraries=["rt", "m"]))
        self.assertEqual(result["argv"][-2:], ["-lrt", "-lm"])
        self.assertEqual(result["unresolved_implicit_dependencies"], ["-lnetcdf", "-lrt", "-lm"])
        for name in ("../evil", "-o", "rt,evil", "", "@file"):
            with self.assertRaises(native.PrepareError):
                native.prepare(**self.args(extra_libraries=[name], manifest=self.base / "badlib.json"))

    def test_cli_never_overwrites_existing_output(self):
        self.output.write_bytes(b"keep")
        command = [sys.executable, str(ROOT / "plov/prepare_native_link.py"),
                   "--compile-log", str(self.log), "--link-cwd", str(self.main),
                   "--frozen-root", str(self.frozen), "--output", str(self.output),
                   "--manifest", str(self.manifest), "--wrapper-object", str(self.wrapper),
                   "--wrap-symbol", "module_cu_kfeta_mp_kf_eta_cps_"]
        self.assertNotEqual(subprocess.run(command, timeout=20,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode, 0)
        self.assertEqual(self.output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
