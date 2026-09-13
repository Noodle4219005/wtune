#!/usr/bin/env python3
"""Validate a KF recipe and execute it once; linking needs reviewed-G0 plus --link.

The execution card must forbid concurrent writes to the manifest and pinned
inputs for the whole invocation. Hash checks detect drift at command boundaries;
they do not establish immutability between reads or replace that caller invariant.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import types


class RecipeError(ValueError):
    pass


REVIEWED_CLEARED_ENV = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
                        "INCLUDE", "LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX",
                        "IFORTCFG", "ICC_CFG", "ICPXCFG")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise RecipeError("recipe paths must be absolute strings")
    path = Path(value)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise RecipeError("symlink in recipe path: " + value)
    if path.resolve(strict=False) != path:
        raise RecipeError("recipe path must be canonical: " + value)
    return path


def _write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _environment(reviewed):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cleared = []
    if reviewed:
        for name in REVIEWED_CLEARED_ENV:
            if name in environment:
                cleared.append(name)
                del environment[name]
    overrides = {"PYTHONDONTWRITEBYTECODE": "1"}
    if reviewed:
        overrides["cleared"] = cleared
    return environment, overrides


def _builder(manifest):
    path = Path(__file__).with_name("prepare_kf_build.py")
    if _sha(path) != manifest.get("preparer_sha256"):
        raise RecipeError("recipe preparer version differs from the local validator")
    # Avoid importlib's __pycache__ write into the source tree during dry validation.
    module = types.ModuleType("_trusted_recipe_builder")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _check_pinned(manifest_path, expected_sha256, manifest):
    """Read-only boundary check, valid after the execution marker/output dirs exist."""
    paths = [(str(manifest_path), expected_sha256),
             (str(Path(__file__).with_name("prepare_kf_build.py")), manifest["preparer_sha256"])]
    if manifest.get("version") == 3:
        paths.append((str(Path(__file__).with_name("kf_review_record.py")),
                      manifest["review_helper_sha256"]))
    paths.extend((entry["path"], entry["sha256"]) for entry in manifest["input_hashes"])
    try:
        for value, digest in paths:
            path = _path(value)
            if not path.is_file() or _sha(path) != digest:
                raise RecipeError("pinned input changed: " + value)
    except OSError as error:
        raise RecipeError("pinned input unreadable: " + str(error)) from error
    builder = _builder(manifest)
    _check_module_inputs(manifest, builder)


def _check_module_inputs(manifest, builder):
    """Validate immutable configured module directories and the zero-shadow surface."""
    try:
        frozen = _path(manifest["frozen_root"])
        wtune = _path(manifest["wtune_root"])
        config = manifest["compiler_config"]
        if not isinstance(config, dict) or "fortran_module_dirs" not in config:
            raise RecipeError("missing canonical Fortran module directory configuration")
        directories = builder._module_directories(config["fortran_module_dirs"])
        actual = builder._module_snapshot(frozen, directories)
        if manifest.get("fortran_module_inputs") != actual:
            raise RecipeError("Fortran module membership or hash snapshot differs")
        expected_paths = {member["path"] for record in actual for member in record["members"]}
        module_paths = {entry["path"] for entry in manifest["input_hashes"]
                        if Path(entry["path"]).suffix.lower() in {".mod", ".smod"}}
        if module_paths != expected_paths:
            raise RecipeError("Fortran module members and input hash ledger differ")
        builder._assert_no_plov_modules(wtune / "plov")
    except (OSError, KeyError, TypeError, builder.PrepareError) as error:
        raise RecipeError("invalid Fortran module input profile: " + str(error)) from error


def validate(manifest_path, expected_sha256):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RecipeError("expected SHA256 must be 64 lowercase hexadecimal characters")
    path = _path(str(manifest_path))
    if _sha(path) != expected_sha256:
        raise RecipeError("manifest SHA256 mismatch")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise RecipeError("manifest must be an object")
    canonical = dict(manifest)
    digest = canonical.pop("plan_sha256", None)
    if hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != digest:
        raise RecipeError("canonical plan SHA256 mismatch")
    version = manifest.get("version")
    if manifest.get("artifact_type") != "wtune_kf_build_recipe" or \
            version not in {2, 3} or manifest.get("executed") is not False:
        raise RecipeError("unsupported or already executed recipe")
    reviewed = version == 3
    proof_keys = {"source_abi_verified", "caller_chain_verified", "baseline_closure_verified",
                  "compiler_module_abi_verified", "library_closure_verified"}
    proof = manifest.get("attestation")
    if not isinstance(proof, dict) or set(proof) != proof_keys or \
            any(value is not reviewed for value in proof.values()):
        raise RecipeError("recipe proof state does not match its profile version")
    node_enabled = manifest.get("compiler_config", {}).get("node_local_profile") is not None
    expected_node = {"enabled": node_enabled,
                     "name": ("reviewed-g0-v1" if reviewed else "unattested-v1")
                             if node_enabled else None,
                     "attestation": {"source_abi_verified": reviewed,
                                     "caller_chain_verified": reviewed,
                                     "init_verified": reviewed, "concurrency_verified": False},
                     "proof_activation_implemented": reviewed}
    actual_node = manifest.get("node_local_profile")
    if reviewed:
        if not isinstance(actual_node, dict):
            raise RecipeError("missing reviewed node-local profile")
        for key in ("profile_id", "review_record"):
            expected_node[key] = actual_node.get(key)
    if actual_node != expected_node or (reviewed and not node_enabled):
        raise RecipeError("invalid or proof-enabled node-local profile")
    wtune, frozen, output = (_path(manifest[name]) for name in ("wtune_root", "frozen_root", "output_dir"))
    if not all(p.is_dir() for p in (wtune, frozen, output)) or path.parent != output:
        raise RecipeError("source/output directory or manifest location mismatch")
    if wtune.is_relative_to(frozen) or frozen.is_relative_to(wtune) or \
            output.is_relative_to(wtune) or output.is_relative_to(frozen):
        raise RecipeError("source roots must be disjoint and output must be external")
    marker = output / "execution.started"
    if marker.exists() or marker.is_symlink():
        raise RecipeError("recipe execution was already started; replay is forbidden")
    entries = manifest.get("input_hashes")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 512:
        raise RecipeError("input hash list must contain 1..512 entries")
    inputs = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise RecipeError("invalid input hash entry")
        item = _path(entry["path"])
        if item in inputs or not item.is_file() or _sha(item) != entry["sha256"]:
            raise RecipeError("missing, duplicate or changed input: " + str(item))
        inputs.add(item)
    builder = _builder(manifest)
    config = manifest.get("compiler_config")
    matching = []
    for item in inputs:
        if item.suffix == ".json":
            try:
                if builder._configuration(item) == config:
                    matching.append(item)
            except (ValueError, OSError):
                pass
    if not matching:
        raise RecipeError("compiler config does not match a hashed input")
    _check_module_inputs(manifest, builder)

    def required(value, root=None):
        item = _path(str(value))
        if item not in inputs or (root is not None and not item.is_relative_to(root)):
            raise RecipeError("unrecorded or out-of-root input: " + str(item))
        return item

    def driver(value):
        item = required(value)
        if item.name in {"sh", "bash", "dash", "zsh", "ksh", "make", "gmake"} or not os.access(item, os.X_OK):
            raise RecipeError("non-executable or forbidden shell/make driver")
        return str(item)

    steps = manifest.get("steps")
    if not isinstance(steps, list) or not 4 <= len(steps) <= 128 or any(not isinstance(s, dict) for s in steps):
        raise RecipeError("invalid bounded step list")
    try:
        python = driver(steps[1]["argv"][0])
        cc = driver(next(s["argv"][0] for s in steps if s.get("kind") == "compile" and s.get("language") == "c"))
        fc = driver(next(s["argv"][0] for s in steps if s.get("kind") == "compile" and s.get("language") == "fortran"))
    except (KeyError, IndexError, StopIteration, TypeError) as error:
        raise RecipeError("missing compiler/generator step") from error
    plov = wtune / "plov"
    for name in (*builder.KF_C, *builder.KF_F, *(builder.KF_NODE_C if node_enabled else ()),
                 *builder.HEADERS, *builder.GENERATORS,
                 "kf_cps_schema.json", "prepare_native_link.py"):
        required(plov / name, wtune)
    native = required(frozen / "phys/module_cu_kfeta.f90", frozen)
    required(frozen / "frame/module_domain_type.mod", frozen)
    log = required(frozen / "compile.log", frozen)
    bridge = output / "generated/wt_kf_bridge.c"
    expected = [{"id": "prepare-directories", "kind": "mkdir", "depends_on": [],
                 "paths": [str(output / sub) for sub in ("objects", "modules", "generated")]},
                {"id": "generate-kf-bridge", "kind": "generate", "cwd": str(output),
                 "depends_on": ["prepare-directories"], "output": str(bridge),
                 "argv": [python, str(plov / "gen_kf_bridge.py"), "--source", str(native),
                          "--schema", str(plov / "kf_cps_schema.json"), "--output", str(bridge)]}]
    units, sources, names = [], set(), set()
    reserved = set(builder.KF_C + builder.KF_F + builder.KF_NODE_C + ("wt_kf_bridge.c",))
    reserved_names = {Path(name).stem + ".o" for name in reserved}
    for unit in config["baseline_units"]:
        if not isinstance(unit, dict) or set(unit) != {"source", "language", "output_name", "flags"}:
            raise RecipeError("invalid baseline unit schema")
        source = required(wtune / unit["source"], wtune)
        name, language = unit["output_name"], unit["language"]
        if language not in {"c", "asm", "fortran"} or not isinstance(name, str) or \
                not re.fullmatch(r"[A-Za-z0-9_-]+\.o", name) or source in sources or \
                source.name in reserved or name in names | reserved_names:
            raise RecipeError("invalid or duplicate baseline source/output")
        if source.suffix != {"c": ".c", "asm": ".S", "fortran": ".f90"}[language]:
            raise RecipeError("baseline language mismatch")
        sources.add(source)
        names.add(name)
        units.append((source, language, name, builder._flags(unit["flags"]), False))
    if not {plov / "wtimer.c", plov / "wtimer.S"} <= sources:
        raise RecipeError("missing timer baseline")
    for name in builder.KF_C + builder.KF_F + (builder.KF_NODE_C if node_enabled else ()):
        units.append((plov / name, "fortran" if name in builder.KF_F else "c", Path(name).stem + ".o", [], False))
    units.append((bridge, "c", "wt_kf_bridge.o", [], True))
    objects, compile_ids = [], []
    for source, language, name, flags, generated in units:
        target = output / "objects" / name
        key = {"c": "c_flags", "asm": "asm_flags", "fortran": "fortran_flags"}[language]
        argv = [fc if language == "fortran" else cc, *config[key], *flags, "-I", str(plov)]
        if source in (plov / "wtimer.c", plov / "wtimer.S"):
            argv += ["-DWT_CUMULUS_LIVE_SPANS=1"]
        if source.name == "wt_kf_bootstrap.c":
            proof_value = "1" if reviewed else "0"
            argv += ["-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=" + proof_value,
                     "-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=" + proof_value]
        if generated:
            argv += ["-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch"]
        if node_enabled and language == "c":
            argv += ["-DWT_KF_NODE_LOCAL=1"]
        if reviewed and language == "c":
            argv += ['-DWT_KF_BUILD_PROFILE_ID="' + actual_node["profile_id"] + '"',
                     "-DWT_KF_BUILD_CONCURRENT_CPS_VERIFIED=0"]
        if reviewed and source.name == "wt_kf_init_bridge.c":
            argv += ["-DWT_KF_BUILD_INIT_TABLES_VERIFIED=1"]
        if language == "fortran":
            for relative in config["fortran_module_dirs"]:
                argv += ["-I", str(frozen / relative)]
            argv += [config["fortran_module_flag"], str(output / "modules")]
        argv += ["-c", str(source), "-o", str(target)]
        compile_ids.append("compile-" + name)
        objects.append(str(target))
        expected.append({"id": compile_ids[-1], "kind": "compile", "source": str(source),
                         "language": language, "output": str(target), "cwd": str(output), "argv": argv,
                         "depends_on": ["generate-kf-bridge" if generated else "prepare-directories"]})
    symbols = config["wrap_symbols"] + [builder.SYMBOL] + (
        ["module_cu_kfeta_mp_kf_eta_init_"] if node_enabled else [])
    if manifest.get("wrap_symbols") != symbols or manifest.get("extra_libraries") != config["extra_libraries"]:
        raise RecipeError("wrapper/library metadata mismatch")
    argv = [python, str(plov / "prepare_native_link.py"), "--compile-log", str(log),
            "--link-cwd", str(frozen / "main"), "--frozen-root", str(frozen),
            "--output", str(output / "wrf-wtune.exe"), "--manifest", str(output / "link-manifest.json")]
    for obj in objects:
        argv += ["--wrapper-object", obj]
    for symbol in symbols:
        argv += ["--wrap-symbol", symbol]
    for library in config["extra_libraries"]:
        argv += ["--extra-library", library]
    expected.append({"id": "prepare-link", "kind": "prepare_link", "cwd": str(output),
                     "depends_on": compile_ids, "argv": argv})
    if steps != expected:
        raise RecipeError("step schema, order, dependencies or exact argv differ from version-2 recipe")
    destinations = [output / name for name in ("objects", "modules", "generated", "execution-logs",
                                               "execution.status.json", "link-manifest.json", "wrf-wtune.exe")]
    if reviewed:
        destinations += [output / "wrf-wtune.map", output / "wrf-wtune.identity.json"]
    for destination in destinations:
        _path(str(destination))
        if destination.exists():
            raise RecipeError("output already exists: " + str(destination))
    if reviewed:
        helper_path = Path(__file__).with_name("kf_review_record.py")
        if manifest.get("review_helper_sha256") != _sha(helper_path):
            raise RecipeError("review helper version differs")
        helper = builder._review_helper()
        try:
            helper.validate_manifest_closure(manifest, config, entries,
                                             [("cc", cc), ("fc", fc), ("python", python)], output)
        except (OSError, ValueError, helper.ReviewError) as error:
            raise RecipeError("review record validation failed: " + str(error)) from error
    return manifest


def _check_outputs(records):
    for record in records:
        path = _path(record["path"])
        if not path.is_file() or path.is_symlink() or _sha(path) != record["sha256"]:
            raise RecipeError("generated output changed: " + str(path))


def run_recipe(manifest_path, expected_sha256, *, execute=False, link=False, timeout_seconds=60):
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
        raise RecipeError("timeout_seconds must be an integer in 1..600")
    manifest = validate(manifest_path, expected_sha256)
    if link and (not execute or manifest["version"] != 3):
        raise RecipeError("--link requires execution of a reviewed-g0-v1 recipe")
    result = {"status": "validated", "manifest_sha256": expected_sha256,
              "plan_sha256": manifest["plan_sha256"], "attestation": manifest["attestation"],
              "link_executed": False, "timeout_seconds": timeout_seconds,
              "caller_invariant": "Concurrent writes to manifest and pinned inputs are forbidden."}
    if not execute:
        return result
    output = Path(manifest["output_dir"])
    try:
        _write(output / "execution.started", {**result, "status": "started",
                                              "started_utc": _now(), "pid": os.getpid()})
    except FileExistsError as error:
        raise RecipeError("execution marker already exists; replay forbidden") from error
    log_dir = output / "execution-logs"
    generated = []
    result["status"] = "failed"
    try:
        log_dir.mkdir()
        for index, step in enumerate(manifest["steps"]):
            if step["kind"] == "mkdir":
                for directory in step["paths"]:
                    _path(directory).mkdir()
                continue
            _path(step["cwd"])
            target = Path(step.get("output", output / "link-manifest.json"))
            _path(str(target))
            if target.exists():
                raise RecipeError("command output already exists: " + str(target))
            status = {"id": step["id"], "argv": step["argv"], "cwd": step["cwd"],
                      "started_utc": _now(), "returncode": None, "reason": "command_failed"}
            environment, status["environment_overrides"] = _environment(manifest["version"] == 3)
            start = time.monotonic()
            with (log_dir / f"{index:03d}.stdout").open("x") as stdout, \
                    (log_dir / f"{index:03d}.stderr").open("x") as stderr:
                try:
                    _check_pinned(manifest_path, expected_sha256, manifest)
                    _check_outputs(generated)
                    process = subprocess.Popen(step["argv"], cwd=step["cwd"], stdout=stdout,
                                               stderr=stderr, start_new_session=True, env=environment)
                    try:
                        status["returncode"] = process.wait(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        status["returncode"] = process.wait()
                        status["reason"] = "timeout"
                    except KeyboardInterrupt:
                        os.killpg(process.pid, signal.SIGKILL)
                        status["returncode"] = process.wait()
                        status["reason"] = "interrupted"
                except OSError as error:
                    status["reason"] = "launch_failed"
                    status["error"] = str(error)
                except RecipeError as error:
                    status["reason"] = "stale_input"
                    status["error"] = str(error)
                try:
                    _check_pinned(manifest_path, expected_sha256, manifest)
                except RecipeError as error:
                    status["reason"] = "stale_input"
                    status["error"] = str(error)
                try:
                    _check_outputs(generated)
                except RecipeError as error:
                    if status["reason"] != "stale_input":
                        status["reason"] = "stale_output"
                        status["error"] = str(error)
            status["elapsed_seconds"] = time.monotonic() - start
            status["finished_utc"] = _now()
            if status["returncode"] == 0 and status["reason"] == "command_failed":
                if target.is_file() and not target.is_symlink():
                    status["reason"] = "complete"
                    status["output_sha256"] = _sha(target)
                    generated.append({"path": str(target), "sha256": status["output_sha256"],
                                      "size": target.stat().st_size})
                else:
                    status["reason"] = "missing_output"
            _write(log_dir / f"{index:03d}.status.json", status)
            if status["reason"] != "complete":
                result.update(step_id=step["id"], returncode=status["returncode"], reason=status["reason"])
                break
        else:
            try:
                _check_pinned(manifest_path, expected_sha256, manifest)
                _check_outputs(generated)
                if link:
                    helper = _builder(manifest)._review_helper()
                    try:
                        argv, cwd, routes = helper.link_command(manifest, generated)
                    except helper.ReviewError as error:
                        raise RecipeError(str(error)) from error
                    for value in routes.values():
                        target = _path(value)
                        if target.exists() or target.is_symlink():
                            raise RecipeError("reviewed link output already exists: " + str(target))
                    index = len(manifest["steps"])
                    environment, overrides = _environment(True)
                    status = {"id": "link-reviewed-candidate", "argv": argv, "cwd": str(cwd),
                              "started_utc": _now(), "returncode": None, "reason": "command_failed",
                              "environment_overrides": overrides}
                    start = time.monotonic()
                    with (log_dir / f"{index:03d}.stdout").open("x") as stdout, \
                            (log_dir / f"{index:03d}.stderr").open("x") as stderr:
                        process = subprocess.Popen(argv, cwd=cwd, stdout=stdout, stderr=stderr,
                                                   start_new_session=True, env=environment)
                        try:
                            status["returncode"] = process.wait(timeout=timeout_seconds)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            status["returncode"] = process.wait()
                            status["reason"] = "timeout"
                        _check_pinned(manifest_path, expected_sha256, manifest)
                        _check_outputs(generated)
                    status["elapsed_seconds"] = time.monotonic() - start
                    status["finished_utc"] = _now()
                    candidate, link_map = (_path(routes[name]) for name in ("candidate", "map"))
                    if status["returncode"] == 0 and status["reason"] == "command_failed":
                        if all(path.is_file() and not path.is_symlink() for path in (candidate, link_map)):
                            status["reason"] = "complete"
                            for label, path in (("candidate", candidate), ("map", link_map)):
                                record = {"path": str(path), "sha256": _sha(path),
                                          "size": path.stat().st_size}
                                status[label] = record
                                generated.append(record)
                        else:
                            status["reason"] = "missing_output"
                    _write(log_dir / f"{index:03d}.status.json", status)
                    if status["reason"] != "complete":
                        result.update(step_id=status["id"], returncode=status["returncode"],
                                      reason=status["reason"])
                    else:
                        _check_pinned(manifest_path, expected_sha256, manifest)
                        _check_outputs(generated)
                        identity = helper.linked_identity(
                            manifest, expected_sha256, argv, routes, generated)
                        identity_path = _path(routes["identity_record"])
                        _write(identity_path, identity)
                        identity_record = {"path": str(identity_path), "sha256": _sha(identity_path),
                                           "size": identity_path.stat().st_size}
                        generated.append(identity_record)
                        _check_pinned(manifest_path, expected_sha256, manifest)
                        _check_outputs(generated)
                        result.update(status="complete", link_executed=True,
                                      linked_candidate=status["candidate"], link_map=status["map"],
                                      native_identity_record=identity_record)
                else:
                    result["status"] = "complete"
            except RecipeError as error:
                result.update(step_id="final-verification", reason="stale_input", error=str(error))
    except (OSError, RecipeError) as error:
        result.update(reason="execution_error", error=str(error))
    if manifest["version"] == 3:
        result["generated_outputs"] = generated
    result["finished_utc"] = _now()
    _write(output / "execution.status.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--link", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        result = run_recipe(args.manifest, args.expected_sha256, execute=args.execute, link=args.link,
                            timeout_seconds=args.timeout_seconds)
    except (RecipeError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
