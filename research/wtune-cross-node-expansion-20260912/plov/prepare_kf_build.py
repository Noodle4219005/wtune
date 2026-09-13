#!/usr/bin/env python3
"""Write a draft KF build recipe without running compilers, generators or links.

Compiler config requires baseline_units ({source, language, output_name, flags}),
wrap_symbols, c_flags, fortran_flags and fortran_module_flag (-J or -module).
Optional asm_flags and extra_libraries default to []. Extra libraries are bounded
linker identifiers, forwarded to the later link preparer. Flag fields are argv arrays, never shell
strings. Paths in baseline_units are relative to wtune_root. Baseline closure,
compiler/module ABI and library closure remain unverified, regardless of inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import sys
import types


class PrepareError(ValueError):
    pass


KF_C = ("wt_live_spans.c", "wt_cumulus.c", "wt_kf_capacity.c",
        "wt_kf_dispatch.c", "wt_kf_live.c", "wt_kf_native.c", "wt_kf_bootstrap.c")
KF_F = ("wt_grid_spans.f90", "wt_grid_capture.f90")
KF_NODE_C = ("wt_kf_batch.c", "wt_kf_batch_codec.c", "wt_node_local.c",
             "wt_kf_node.c", "wt_kf_node_live.c", "wt_kf_node_shm.c",
             "wt_kf_init_bridge.c")
HEADERS = ("wt_live_spans.h", "wt_cumulus.h", "wt_kf_capacity.h",
           "wt_kf_dispatch.h", "wt_kf_live.h", "wt_kf_native.h",
           "wt_kf_bootstrap.h", "wt_kf_cps_schema.h")
GENERATORS = ("gen_kf_bridge.py", "gen_ref_bridge.py", "gen_phase_schema.py")
SYMBOL = "module_cu_kfeta_mp_kf_eta_cps_"
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)
MAX_MODULE_DIRS = 16
MAX_MODULES_PER_DIR = 128


def _review_helper():
    path = Path(__file__).with_name("kf_review_record.py")
    module = types.ModuleType("_kf_review_record")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _file(path, root=None):
    try:
        path = Path(path).resolve(strict=True)
    except OSError as error:
        raise PrepareError(f"missing input: {path}") from error
    if not path.is_file() or (root is not None and not path.is_relative_to(root)):
        raise PrepareError(f"input is not a file inside its declared root: {path}")
    return path


def _module_name(name):
    return name.lower().endswith((".mod", ".smod"))


def _module_directories(value):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_MODULE_DIRS:
        raise PrepareError("fortran_module_dirs must contain 1..16 relative directories")
    result = []
    for raw in value:
        if not isinstance(raw, str) or not raw or raw.startswith("/") or "//" in raw:
            raise PrepareError("Fortran module directory must be a canonical relative path")
        path = PurePosixPath(raw)
        if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts) or \
                path.as_posix() != raw:
            raise PrepareError("Fortran module directory must be a canonical relative path")
        result.append(raw)
    if result[0] != "frame" or len(set(result)) != len(result):
        raise PrepareError("fortran_module_dirs must start with unique mandatory frame")
    return result


def _module_directory(frozen, relative):
    path = frozen
    for part in PurePosixPath(relative).parts:
        path /= part
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise PrepareError("symlink in configured Fortran module directory")
        except OSError as error:
            raise PrepareError("missing configured Fortran module directory: " + relative) from error
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PrepareError("missing configured Fortran module directory: " + relative) from error
    if not resolved.is_dir() or not resolved.is_relative_to(frozen):
        raise PrepareError("configured Fortran module directory escapes frozen root")
    return resolved


def _module_snapshot(frozen, directories):
    records, names = [], set()
    for relative in directories:
        directory = _module_directory(frozen, relative)
        members = []
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise PrepareError("unable to enumerate Fortran module directory") from error
        for item in entries:
            if not _module_name(item.name):
                continue
            try:
                mode = item.lstat().st_mode
            except OSError as error:
                raise PrepareError("unreadable Fortran module input: " + str(item)) from error
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise PrepareError("Fortran module input must be a regular non-symlink file")
            key = item.name.casefold()
            if key in names:
                raise PrepareError("case-insensitive Fortran module filename collision")
            names.add(key)
            members.append({"name": item.name, "path": str(item), "sha256": _hash(item),
                            "size": item.stat().st_size})
        if not members:
            raise PrepareError("configured Fortran module directory has no direct .mod/.smod files")
        if len(members) > MAX_MODULES_PER_DIR:
            raise PrepareError("configured Fortran module directory exceeds 128 members")
        digest = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        records.append({"relative_path": relative, "path": str(directory), "members": members,
                        "membership_sha256": digest})
    return records


def _assert_no_plov_modules(plov):
    try:
        if any(_module_name(path.name) for path in plov.iterdir()):
            raise PrepareError("plov must contain no direct .mod/.smod files")
    except OSError as error:
        raise PrepareError("unable to inspect plov module shadow surface") from error


def _flags(value):
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise PrepareError("flags must be an argv string array")
    simple = {"-g", "-Wall", "-Wextra", "-Werror", "-pedantic", "-fPIC",
              "-fopenmp", "-qopenmp", "-pthread", "-fpp", "-auto", "-i4",
              "-ip", "-ftz", "-fno-alias", "-FR", "-w", "-fno-inline",
              "-no-ip", "-fno-omit-frame-pointer", "-fcheck=all"}
    paired = {"-real-size": {"32"}, "-fp-model": {"precise", "strict"},
              "-align": {"all"}, "-convert": {"big_endian", "little_endian"}}
    index = 0
    while index < len(value):
        flag = value[index]
        if flag in paired:
            if index + 1 >= len(value) or value[index + 1] not in paired[flag]:
                raise PrepareError(f"unsupported value for {flag}")
            index += 2
            continue
        if flag not in simple and not re.fullmatch(
                r"-O[0-3sg]|-std=(?:c11|c17|gnu11|gnu17|f2008|f2018)|"
                r"-x[A-Z][A-Z0-9_]*|-m(?:arch|tune)=[A-Za-z0-9_.-]+|"
                r"-qopt-zmm-usage=(?:high|low)", flag):
            raise PrepareError(f"unsupported compile flag: {flag!r}")
        index += 1
    return list(value)


def _configuration(path):
    try:
        config = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise PrepareError("invalid compiler config JSON") from error
    required = {"baseline_units", "wrap_symbols", "c_flags", "fortran_flags",
                "fortran_module_flag"}
    if not isinstance(config, dict) or not required <= config.keys() or \
            config.keys() - required - {"asm_flags", "extra_libraries", "fortran_module_dirs", "node_local_profile"}:
        raise PrepareError("compiler config has missing or unsupported keys")
    for key in ("c_flags", "fortran_flags", "asm_flags"):
        config[key] = _flags(config.get(key, []))
    if config["fortran_module_flag"] not in ("-J", "-module"):
        raise PrepareError("Fortran module output flag must be -J or -module")
    config["fortran_module_dirs"] = _module_directories(config.get("fortran_module_dirs", ["frame"]))
    symbols = config["wrap_symbols"]
    if not isinstance(symbols, list) or len(symbols) > 128 or any(not isinstance(s, str) or not IDENT.fullmatch(s)
                                          for s in symbols) or len(set(symbols)) != len(symbols):
        raise PrepareError("wrap_symbols must contain unique linker identifiers")
    if SYMBOL in symbols:
        raise PrepareError("KF CPS wrapper is reserved for the added KF units")
    libraries = config.get("extra_libraries", [])
    if not isinstance(libraries, list) or len(libraries) > 128 or any(
            not isinstance(name, str) or len(name) > 128 or not IDENT.fullmatch(name)
            for name in libraries) or len(set(libraries)) != len(libraries):
        raise PrepareError("extra_libraries must contain unique identifiers (at most 128, each at most 128 characters)")
    config["extra_libraries"] = libraries
    profile = config.get("node_local_profile")
    if profile is not None:
        old = profile == {"name": "unattested-v1"}
        reviewed = isinstance(profile, dict) and set(profile) == {"name", "review_record"} and \
            profile.get("name") == "reviewed-g0-v1" and \
            isinstance(profile.get("review_record"), str) and \
            Path(profile["review_record"]).is_absolute()
        if not old and not reviewed:
            raise PrepareError("node_local_profile must be exact unattested-v1 or reviewed-g0-v1 form")
    if not isinstance(config["baseline_units"], list) or len(config["baseline_units"]) > 64:
        raise PrepareError("baseline_units must be a list of at most 64 units")
    return config


def prepare(*, wtune_root, frozen_root, cc, fc, compiler_config, output_dir, python=None):
    wtune, frozen = Path(wtune_root).resolve(strict=True), Path(frozen_root).resolve(strict=True)
    if not wtune.is_dir() or not frozen.is_dir() or wtune.is_relative_to(frozen) or frozen.is_relative_to(wtune):
        raise PrepareError("source roots must be separate directories")
    output = Path(output_dir).absolute()
    for part in (output, *output.parents):
        if part.is_symlink():
            raise PrepareError("output path must not contain symlinks")
    output = output.resolve(strict=False)
    if output.is_relative_to(frozen) or output.is_relative_to(wtune):
        raise PrepareError("output directory must be external to both source roots")
    if output.exists() or not output.parent.is_dir():
        raise PrepareError("output directory must be new with an existing parent")
    cc, fc = _file(cc), _file(fc)
    python = _file(python or sys.executable)
    if not all(os.access(path, os.X_OK) for path in (cc, fc, python)):
        raise PrepareError("compiler/Python executable must have execute permission")
    config_path = _file(compiler_config)
    config = _configuration(config_path)
    plov = (wtune / "plov").resolve(strict=True)
    _assert_no_plov_modules(plov)
    module_inputs = _module_snapshot(frozen, config["fortran_module_dirs"])
    module_paths = {Path(member["path"]) for record in module_inputs for member in record["members"]}
    inputs = {cc, fc, python, config_path}
    node_enabled = config.get("node_local_profile") is not None
    reviewed = node_enabled and config["node_local_profile"]["name"] == "reviewed-g0-v1"
    fixed = (*KF_C, *KF_F, *HEADERS, *GENERATORS, "kf_cps_schema.json", "prepare_native_link.py",
             *(KF_NODE_C if node_enabled else ()))
    inputs.update(_file(plov / name, wtune) for name in fixed)
    native_source = _file(frozen / "phys/module_cu_kfeta.f90", frozen)
    module = frozen / "frame/module_domain_type.mod"
    if module not in module_paths:
        raise PrepareError("mandatory frame/module_domain_type.mod is not a pinned module input")
    log = _file(frozen / "compile.log", frozen)
    inputs.update((native_source, log))
    inputs.update(Path(member["path"]) for record in module_inputs for member in record["members"])
    units, seen_sources, seen_outputs = [], set(), set()
    reserved = set(KF_C + KF_F + KF_NODE_C + ("wt_kf_bridge.c",))
    reserved_outputs = {Path(name).stem + ".o" for name in reserved}
    for unit in config["baseline_units"]:
        if not isinstance(unit, dict) or set(unit) != {"source", "language", "output_name", "flags"}:
            raise PrepareError("baseline unit requires source, language, output_name and flags")
        if not all(isinstance(unit[key], str) for key in ("source", "language", "output_name")):
            raise PrepareError("baseline unit paths/language must be strings")
        source = _file(wtune / unit["source"], wtune)
        name, language = unit["output_name"], unit["language"]
        if language not in {"c", "asm", "fortran"} or not re.fullmatch(r"[A-Za-z0-9_-]+\.o", name):
            raise PrepareError("invalid baseline language or object filename")
        suffix = {"c": ".c", "asm": ".S", "fortran": ".f90"}[language]
        if source.suffix != suffix:
            raise PrepareError("baseline source extension does not match language")
        if source.name in reserved or source in seen_sources or name in seen_outputs | reserved_outputs:
            raise PrepareError("duplicate/reserved source or object name")
        seen_sources.add(source)
        seen_outputs.add(name)
        units.append((source, language, name, _flags(unit["flags"]), False))
        inputs.add(source)
    if not {_file(plov / "wtimer.c", wtune), _file(plov / "wtimer.S", wtune)} <= seen_sources:
        raise PrepareError("baseline must contain wtimer.c and wtimer.S exactly once")
    for name in KF_C + KF_F + (KF_NODE_C if node_enabled else ()):
        units.append((plov / name, "fortran" if name in KF_F else "c",
                      Path(name).stem + ".o", [], False))
    bridge = output / "generated/wt_kf_bridge.c"
    units.append((bridge, "c", "wt_kf_bridge.o", [], True))
    # Literal local includes are part of the input snapshot, including baseline argmaps.
    pending = [source for source, language, *_ in units if language == "c" and source != bridge]
    visited = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        if len(visited) > 128:
            raise PrepareError("local include graph exceeds 128 files")
        for name in LOCAL_INCLUDE.findall(source.read_text()):
            candidate = source.parent / name
            if not candidate.exists():
                candidate = plov / name
            header = _file(candidate, wtune)
            inputs.add(header)
            pending.append(header)
    review_record = None
    review_lanes = None
    profile_id = None
    review_helper_sha256 = None
    if reviewed:
        helper = _review_helper()
        helper_path = _file(Path(__file__).with_name("kf_review_record.py"))
        review_helper_sha256 = _hash(helper_path)
        schema_path = plov / "kf_cps_schema.json"
        sources = sorted(inputs - {cc, fc, python, config_path, schema_path} - module_paths)
        options = dict(config)
        options["node_local_profile"] = {"name": "reviewed-g0-v1"}
        lanes = {
            "sources": [{"path": str(path), "sha256": _hash(path)} for path in sources],
            "schema": {"path": str(schema_path), "sha256": _hash(schema_path)},
            "compilers": [{"role": role, "path": str(path), "sha256": _hash(path)}
                          for role, path in (("cc", cc), ("fc", fc), ("python", python))],
            "options": options,
            "modules": module_inputs,
        }
        try:
            review_path = _file(config["node_local_profile"]["review_record"])
            review_record, profile_id, review_inputs, review_lanes = helper.load(
                review_path, lanes=lanes, output_dir=output)
        except (OSError, ValueError, helper.ReviewError) as error:
            raise PrepareError("invalid reviewed-G0 record: " + str(error)) from error
        inputs.update(review_inputs)
        inputs.add(helper_path)
    if len(inputs) > 512:
        raise PrepareError("module/input snapshot exceeds the executor limit of 512 inputs")
    steps = [{"id": "prepare-directories", "kind": "mkdir", "depends_on": [],
              "paths": [str(output / sub) for sub in ("objects", "modules", "generated")]},
             {"id": "generate-kf-bridge", "kind": "generate", "cwd": str(output),
              "depends_on": ["prepare-directories"], "output": str(bridge),
              "argv": [str(python), str(plov / "gen_kf_bridge.py"), "--source", str(native_source),
                       "--schema", str(plov / "kf_cps_schema.json"), "--output", str(bridge)]}]
    objects, compile_ids = [], []
    for source, language, name, flags, generated in units:
        target = output / "objects" / name
        compile_id = "compile-" + name
        objects.append(str(target))
        compile_ids.append(compile_id)
        flag_key = {"c": "c_flags", "asm": "asm_flags", "fortran": "fortran_flags"}[language]
        argv = [str(fc if language == "fortran" else cc), *config[flag_key], *flags,
                "-I", str(plov)]
        if source in (plov / "wtimer.c", plov / "wtimer.S"):
            argv.append("-DWT_CUMULUS_LIVE_SPANS=1")
        if source.name == "wt_kf_bootstrap.c":
            proof = "1" if reviewed else "0"
            argv += ["-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=" + proof,
                     "-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=" + proof]
        if generated:
            argv.append("-Dwt_kf_bridge_dispatch=wt_kf_live_dispatch")
        if node_enabled and language == "c":
            argv.append("-DWT_KF_NODE_LOCAL=1")
        if reviewed and language == "c":
            argv += ['-DWT_KF_BUILD_PROFILE_ID="' + profile_id + '"',
                     "-DWT_KF_BUILD_CONCURRENT_CPS_VERIFIED=0"]
        if reviewed and source.name == "wt_kf_init_bridge.c":
            argv.append("-DWT_KF_BUILD_INIT_TABLES_VERIFIED=1")
        if language == "fortran":
            for relative in config["fortran_module_dirs"]:
                argv += ["-I", str(frozen / relative)]
            argv += [config["fortran_module_flag"], str(output / "modules")]
        argv += ["-c", str(source), "-o", str(target)]
        steps.append({"id": compile_id, "kind": "compile", "source": str(source),
                      "language": language, "output": str(target), "cwd": str(output), "argv": argv,
                      "depends_on": ["generate-kf-bridge" if generated else "prepare-directories"]})
    symbols = config["wrap_symbols"] + [SYMBOL] + (
        ["module_cu_kfeta_mp_kf_eta_init_"] if node_enabled else [])
    link = [str(python), str(plov / "prepare_native_link.py"), "--compile-log", str(log),
            "--link-cwd", str(frozen / "main"), "--frozen-root", str(frozen),
            "--output", str(output / "wrf-wtune.exe"), "--manifest", str(output / "link-manifest.json")]
    for obj in objects:
        link += ["--wrapper-object", obj]
    for symbol in symbols:
        link += ["--wrap-symbol", symbol]
    for library in config["extra_libraries"]:
        link += ["--extra-library", library]
    steps.append({"id": "prepare-link", "kind": "prepare_link", "cwd": str(output),
                  "depends_on": compile_ids, "argv": link})
    node_attestation = {"source_abi_verified": reviewed, "caller_chain_verified": reviewed,
                        "init_verified": reviewed, "concurrency_verified": False}
    constraints = ["Draft only; no commands executed or native ABI attested.",
                   "Never write either source root or replace any frozen artifact.",
                   "Review baseline completeness and all input hashes before execution.",
                   "CC must provide MPI headers for MPI baseline units and match FC OpenMP runtime.",
                   "FC must match the frozen module producer and default kind/ABI options.",
                   "Generated bridge source-hash check is necessary but is not full attestation.",
                   "Prepare-link only writes a later manifest; it does not perform a link.",
                   "Configured frozen module directories are pinned at checked command boundaries only.",
                   "Generated output/modules and ambient compiler search inputs remain outside this immutable snapshot."]
    if reviewed:
        constraints = ["Claims come from an explicit human review; hashes only preserve its pinned inputs.",
                       "Never write either source root or replace any frozen artifact.",
                       "All native CPS entries remain serialized because concurrency is not verified.",
                       "Execution still stops after prepare-link unless the executor receives explicit --link.",
                       "A linked candidate, map and identity record are build provenance, not native ABI or WRF runtime proof.",
                       "Configured module directories and all reviewed inputs are checked at every command boundary."]
    plan = {"artifact_type": "wtune_kf_build_recipe", "version": 3 if reviewed else 2,
            "executed": False,
            "wtune_root": str(wtune), "frozen_root": str(frozen), "output_dir": str(output),
            "compiler_config": config, "wrap_symbols": symbols,
            "extra_libraries": config["extra_libraries"], "steps": steps,
            "fortran_module_inputs": module_inputs,
            "attestation": {"source_abi_verified": reviewed, "caller_chain_verified": reviewed,
                            "baseline_closure_verified": reviewed,
                            "compiler_module_abi_verified": reviewed,
                            "library_closure_verified": reviewed},
            "node_local_profile": {"enabled": node_enabled,
                                   "name": config["node_local_profile"]["name"] if node_enabled else None,
                                   "attestation": node_attestation,
                                   "proof_activation_implemented": reviewed},
            "constraints": constraints,
            "input_hashes": [{"path": str(path), "sha256": _hash(path)} for path in sorted(inputs)],
            "preparer_sha256": _hash(Path(__file__))}
    if reviewed:
        plan["node_local_profile"].update(
            profile_id=profile_id,
            review_record={"path": config["node_local_profile"]["review_record"],
                           "sha256": _hash(Path(config["node_local_profile"]["review_record"]))})
        plan["scientific_build_closure"] = review_lanes
        plan["review_helper_sha256"] = review_helper_sha256
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir()
    with (output / "build-manifest.json").open("x") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wtune-root", "frozen-root", "cc", "fc", "compiler-config", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--python")
    args = parser.parse_args(argv)
    try:
        result = prepare(**vars(args))
    except (PrepareError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"manifest": str(Path(result["output_dir"]) / "build-manifest.json"),
                      "plan_sha256": result["plan_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
