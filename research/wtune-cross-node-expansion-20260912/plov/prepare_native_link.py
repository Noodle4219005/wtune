#!/usr/bin/env python3
"""Produce a review manifest for a separate-output WRF link; never execute it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile


class PrepareError(ValueError):
    pass


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CONTROL = {";", "|", "||", "&", "&&", "(", ")", "<", ">", "<<", ">>"}
_FORTRAN_DRIVERS = {"ifx", "ifort", "mpiifx", "mpiifort", "gfortran", "mpifort",
                    "mpif90", "nvfortran", "flang", "flang-new", "ftn"}
_SINGLE_FLAGS = {"-qopenmp", "-fpp", "-auto", "-ip", "-w", "-ftz", "-fno-alias",
                 "-FR", "-fopenmp", "-fPIC", "-pthread", "-static", "-shared",
                 "-rdynamic", "-pie", "-no-pie"}
_VALUE_FLAGS = {"-align", "-fp-model", "-convert", "-real-size"}
_SAFE_WL = {"--start-group", "--end-group", "--as-needed", "--no-as-needed",
            "--whole-archive", "--no-whole-archive", "--gc-sections", "--no-undefined"}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved(path, cwd):
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return candidate.resolve(strict=False)


def _outside_frozen(path, frozen_root, label):
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(frozen_root)
    except ValueError:
        return resolved
    raise PrepareError(f"{label} resolves inside frozen root: {resolved}")


def _reject_shell_tokens(tokens):
    for token in tokens:
        if token.startswith("@"):
            raise PrepareError(f"response file rejected: {token!r}")
        if token in _CONTROL or any(char in token for char in ";|&$`><"):
            raise PrepareError(f"shell control or expansion token rejected: {token!r}")


def _parse_driver_line(raw):
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError as error:
        raise PrepareError(f"invalid shell quoting in link line: {error}") from error
    _reject_shell_tokens(tokens)
    if tokens[:1] == ["time"]:
        tokens = tokens[1:]
    if not tokens or tokens[0].startswith("-"):
        raise PrepareError("link line has no compiler driver")
    option_values = _reject_unsafe_options(tokens)
    outputs = [index for index, token in enumerate(tokens) if token == "-o"]
    if len(outputs) != 1 or outputs[0] + 1 >= len(tokens):
        raise PrepareError("link line must contain one complete '-o output'")
    if tokens[outputs[0] + 1] != "wrf.exe":
        raise PrepareError("historical link output must be exactly wrf.exe")
    return tokens, outputs[0], option_values


def _reject_unsafe_options(tokens):
    path_options = {"-I", "-isystem", "-include", "-J", "-module", "-B", "--sysroot"}
    values = set()
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if index in values:
            index += 1
            continue
        if token in {"-c", "-E", "-S", "-Xlinker"} or token.startswith("@"):
            raise PrepareError(f"non-link or unsupported option rejected: {token!r}")
        if token.startswith("-o") and token not in {"-o", "-openmp"}:
            raise PrepareError(f"hidden output option rejected: {token!r}")
        if token == "--output" or token.startswith("--output="):
            raise PrepareError(f"hidden output option rejected: {token!r}")
        if token == "-MF" or token.startswith("-MF") or token.startswith("-save-temps"):
            raise PrepareError(f"side-output option rejected: {token!r}")
        if token.startswith("-Wl,"):
            linker_args = token[4:].split(",")
            for argument in linker_args:
                if argument in _SAFE_WL:
                    continue
                if argument.startswith("--wrap=") and _IDENTIFIER.fullmatch(argument[7:]):
                    continue
                raise PrepareError(f"unsupported linker option rejected: {token!r}")
            index += 1
            continue
        if token in path_options or any(token.startswith(option) and token != option
                                        for option in path_options):
            raise PrepareError(f"unhandled path-bearing option rejected: {token!r}")
        if token in _VALUE_FLAGS:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                raise PrepareError(f"option requires a simple value: {token!r}")
            values.add(index + 1)
            index += 2
            continue
        if token == "-o":
            index += 2
            continue
        if token == "-L":
            if index + 1 >= len(tokens):
                raise PrepareError("-L without a path")
            index += 2
            continue
        if token == "-l":
            if index + 1 >= len(tokens) or not _IDENTIFIER.fullmatch(tokens[index + 1]):
                raise PrepareError("-l requires an identifier")
            values.add(index + 1)
            index += 2
            continue
        if token.startswith("-L") or (token.startswith("-l") and _IDENTIFIER.fullmatch(token[2:])):
            index += 1
            continue
        if token in _SINGLE_FLAGS or re.fullmatch(r"-O[0-3sg]", token) or \
           re.fullmatch(r"-x[A-Za-z0-9_]+", token) or \
           re.fullmatch(r"-qopt-zmm-usage=[A-Za-z0-9_-]+", token) or \
           re.fullmatch(r"-m(?:arch|tune)=[A-Za-z0-9_.-]+", token):
            index += 1
            continue
        if token.startswith("-"):
            raise PrepareError(f"unsupported compiler/link option rejected: {token!r}")
        index += 1
    return values


def _is_wrf_link_candidate(raw):
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError:
        return False
    if tokens[:1] == ["time"]:
        tokens = tokens[1:]
    return any(token == "-o" and index + 1 < len(tokens) and tokens[index + 1] == "wrf.exe"
               for index, token in enumerate(tokens))


def _select_line(log, explicit_line):
    lines = log.read_text(encoding="utf-8").splitlines()
    if explicit_line is not None:
        if explicit_line < 1 or explicit_line > len(lines):
            raise PrepareError("--line is outside compile log")
        raw = lines[explicit_line - 1]
        if not _is_wrf_link_candidate(raw):
            raise PrepareError("--line is not an exact '-o wrf.exe' link line")
        return explicit_line, raw
    candidates = [(number, raw) for number, raw in enumerate(lines, 1)
                  if _is_wrf_link_candidate(raw)]
    if len(candidates) != 1:
        raise PrepareError(f"expected one candidate link line, found {len(candidates)}; use --line")
    return candidates[0]


def _normalize_driver(tokens, cwd):
    driver = tokens[0]
    if "/" in driver:
        path = _resolved(driver, cwd)
        _validate_file(path, "compiler driver")
        driver_name = path.name
        tokens[0] = str(path)
    else:
        driver_name = driver
    if driver_name not in _FORTRAN_DRIVERS:
        raise PrepareError(f"unsupported compiler driver: {driver!r}")


def _direct_input(path):
    return path.endswith(".o") or path.endswith(".a")


def _validate_file(path, label):
    if not path.is_file():
        raise PrepareError(f"{label} is not a regular file: {path}")


def _atomic_create_json(path, data):
    if path.exists() or path.is_symlink():
        raise PrepareError(f"manifest already exists: {path}")
    if not path.parent.is_dir():
        raise PrepareError(f"manifest parent is absent: {path.parent}")
    payload = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".prepare-native-link-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise PrepareError(f"manifest already exists: {path}") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def prepare(*, compile_log, link_cwd, output, manifest, frozen_root,
            wrapper_objects, wrap_symbols, line=None, extra_libraries=None):
    extra_libraries = [] if extra_libraries is None else list(extra_libraries)
    if any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name)
           for name in extra_libraries):
        raise PrepareError("extra libraries must be linker identifiers")
    log = Path(compile_log).resolve(strict=True)
    cwd = Path(link_cwd).resolve(strict=True)
    if not cwd.is_dir():
        raise PrepareError(f"link cwd is not a directory: {cwd}")
    frozen = Path(frozen_root).resolve(strict=True)
    if not frozen.is_dir():
        raise PrepareError(f"frozen root is not a directory: {frozen}")
    output = _outside_frozen(Path(output), frozen, "output")
    manifest = _outside_frozen(Path(manifest), frozen, "manifest")
    if output == manifest:
        raise PrepareError("output and manifest must differ")
    if not output.parent.is_dir():
        raise PrepareError(f"output parent is absent: {output.parent}")
    if output.exists() or output.is_symlink():
        raise PrepareError(f"output already exists: {output}")
    if not wrapper_objects or not wrap_symbols:
        raise PrepareError("at least one wrapper object and wrap symbol are required")
    wrappers = []
    for item in wrapper_objects:
        path = Path(item).resolve(strict=False)
        _validate_file(path, "wrapper object")
        if path.suffix != ".o":
            raise PrepareError(f"wrapper object must end in .o: {path}")
        wrappers.append(str(path))
    for symbol in wrap_symbols:
        if not _IDENTIFIER.fullmatch(symbol):
            raise PrepareError(f"invalid wrap symbol: {symbol!r}")
    source_line, raw = _select_line(log, line)
    argv, output_index, option_values = _parse_driver_line(raw)
    _normalize_driver(argv, cwd)
    library_paths = []
    direct_paths = []
    resolved = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if index == 0:
            resolved.append(token)
            index += 1
            continue
        if index == output_index:
            resolved.extend(["-o", str(output)])
            index += 2
            continue
        if index in option_values:
            resolved.append(token)
            index += 1
            continue
        if token == "-L":
            if index + 1 >= len(argv):
                raise PrepareError("-L without a path")
            directory = _resolved(argv[index + 1], cwd)
            if not directory.is_dir():
                raise PrepareError(f"library path is not a directory: {directory}")
            resolved.extend(["-L", str(directory)])
            library_paths.append(str(directory))
            index += 2
            continue
        if token.startswith("-L") and token != "-L":
            directory = _resolved(token[2:], cwd)
            if not directory.is_dir():
                raise PrepareError(f"library path is not a directory: {directory}")
            resolved.append("-L" + str(directory))
            library_paths.append(str(directory))
            index += 1
            continue
        if _direct_input(token):
            path = _resolved(token, cwd)
            _validate_file(path, "direct link input")
            direct_paths.append(path)
            resolved.append(str(path))
        elif token.startswith("-"):
            resolved.append(token)
        else:
            raise PrepareError(f"unsupported source/path operand: {token!r}")
        index += 1
    libraries = ["-l" + resolved[index + 1] if token == "-l" else token
                 for index, token in enumerate(resolved) if token.startswith("-l")]
    libwrf = [index for index, token in enumerate(resolved)
              if Path(token).name == "libwrflib.a"]
    if len(libwrf) != 1:
        raise PrepareError("expected one direct libwrflib.a input")
    insertion = libwrf[0]
    group = ["-Wl,--start-group"]
    group.extend("-Wl,--wrap=" + symbol for symbol in wrap_symbols)
    group.extend(wrappers)
    group.extend([resolved[insertion], "-Wl,--end-group"])
    final_argv = resolved[:insertion] + group + resolved[insertion + 1:]
    additions = ["-l" + name for name in extra_libraries]
    final_argv.extend(additions)
    libraries.extend(additions)
    all_inputs = direct_paths + [Path(path) for path in wrappers]
    hashes = []
    seen = set()
    for path in all_inputs:
        path = path.resolve(strict=True)
        if path not in seen:
            hashes.append({"path": str(path), "sha256": _sha256(path)})
            seen.add(path)
    result = {
        "artifact_type": "wtune_native_link_command_manifest",
        "compile_log_sha256": _sha256(log),
        "source_line": source_line,
        "historical_cwd": str(cwd),
        "link_cwd": str(output.parent),
        "frozen_root": str(frozen),
        "driver": final_argv[0],
        "output": str(output),
        "argv": final_argv,
        "wrapper_objects": wrappers,
        "wrap_symbols": wrap_symbols,
        "direct_input_hashes": hashes,
        "resolved_library_search_paths": library_paths,
        "unresolved_implicit_dependencies": libraries,
        "attestation_note": "-l dependencies are unresolved; this manifest is not full attestation.",
    }
    _atomic_create_json(manifest, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-log", required=True)
    parser.add_argument("--link-cwd", required=True)
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--wrapper-object", action="append", default=[])
    parser.add_argument("--wrap-symbol", action="append", default=[])
    parser.add_argument("--extra-library", action="append", default=[])
    parser.add_argument("--line", type=int)
    args = parser.parse_args(argv)
    try:
        result = prepare(compile_log=args.compile_log, link_cwd=args.link_cwd,
            output=args.output, manifest=args.manifest, frozen_root=args.frozen_root,
            wrapper_objects=args.wrapper_object, wrap_symbols=args.wrap_symbol,
            line=args.line, extra_libraries=args.extra_library)
    except (PrepareError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
