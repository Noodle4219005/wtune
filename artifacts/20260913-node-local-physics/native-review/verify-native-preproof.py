#!/usr/bin/env python3
"""Verify the bounded KF compiler/link preproof without executing its candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(path):
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError("not a file: " + str(path))
    return {"path": str(path), "sha256": sha(path), "size": path.stat().st_size}


def validate_prospective_inputs(prospective):
    """Match all source, module, and baseline files pinned by the prospective record."""
    for lane in ("sources", "baseline_objects"):
        for expected in prospective["lanes"][lane]:
            path = Path(expected["path"])
            if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                    or path.resolve(strict=True) != path or sha(path) != expected["sha256"]):
                raise ValueError("prospective " + lane + " input changed: " + str(path))
    for group in prospective["lanes"]["modules"]:
        directory = Path(group["path"])
        if (not directory.is_absolute() or directory.is_symlink() or not directory.is_dir()
                or directory.resolve(strict=True) != directory):
            raise ValueError("prospective module directory changed: " + str(directory))
        members = []
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.name.lower().endswith((".mod", ".smod")):
                continue
            if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
                raise ValueError("prospective module input changed: " + str(path))
            members.append({"name": path.name, "path": str(path), "sha256": sha(path),
                            "size": path.stat().st_size})
        digest = hashlib.sha256(json.dumps(members, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
        if members != group["members"] or digest != group["membership_sha256"]:
            raise ValueError("prospective module membership changed: " + str(directory))


def dynamic(path):
    result = subprocess.run(["readelf", "-d", str(path)], text=True,
                            capture_output=True, check=True)
    needed = []
    rpath = []
    soname = None
    for line in result.stdout.splitlines():
        match = re.search(r"\(NEEDED\).*\[([^]]+)\]", line)
        if match:
            needed.append(match.group(1))
        match = re.search(r"\((?:RPATH|RUNPATH)\).*\[([^]]*)\]", line)
        if match:
            rpath.extend(part for part in match.group(1).split(":") if part)
        match = re.search(r"\(SONAME\).*\[([^]]+)\]", line)
        if match:
            soname = match.group(1)
    return needed, rpath, soname


def loader_mapping(text):
    """Parse PT_INTERP --list output and reject conflicting SONAME selections."""
    mapping = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(", line)
        if not match:
            continue
        name, raw = match.groups()
        selected = Path(raw).resolve(strict=True)
        previous = mapping.get(name)
        if previous is not None and previous != selected:
            raise ValueError("ambiguous loader selection for " + name)
        mapping[name] = selected
    return mapping


def loader_closure(candidate, mapping):
    """Recurse through DT_NEEDED using paths selected by the native loader."""
    queue = [candidate.resolve(strict=True)]
    seen = set()
    records = []
    unresolved = []
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        needed, rpaths, soname = dynamic(current)
        resolved = []
        for name in needed:
            if "/" in name:
                selected = Path(name)
                if not selected.is_absolute() or not selected.is_file():
                    unresolved.append({"parent": str(current), "needed": name})
                    continue
                selected = selected.resolve(strict=True)
            else:
                selected = mapping.get(name)
            if selected is None:
                unresolved.append({"parent": str(current), "needed": name})
                continue
            resolved.append({"needed": name, "selected": file_entry(selected)})
            if selected not in seen:
                queue.append(selected)
        records.append({**file_entry(current), "soname": soname, "needed": resolved,
                        "rpath_or_runpath": rpaths})
    if unresolved:
        raise ValueError("unresolved DT_NEEDED entries: " + json.dumps(unresolved, sort_keys=True))
    return records


def validate_object_ledger(compile_steps, status):
    ledger = {record["path"]: record for record in status.get("generated_outputs", [])}
    for step in compile_steps:
        path = Path(step["output"])
        record = ledger.get(str(path))
        if (record is None or not path.is_file() or path.is_symlink()
                or record.get("sha256") != sha(path) or record.get("size") != path.stat().st_size):
            raise ValueError("compiled object differs from executor ledger: " + str(path))


def trace_files(path):
    found = []
    for line in path.read_text(errors="replace").splitlines():
        for token in re.findall(r"/[^\s()]+", line):
            candidate = Path(token.rstrip(":,"))
            if candidate.exists() and candidate.is_file():
                resolved = candidate.resolve(strict=True)
                if resolved not in found:
                    found.append(resolved)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prospective", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--recipe")
    parser.add_argument("--raw")
    parser.add_argument("--output")
    args = parser.parse_args()
    prospective_path = Path(args.prospective).resolve(strict=True)
    prospective = json.loads(prospective_path.read_text())
    validate_prospective_inputs(prospective)
    if args.preflight:
        if any(value is not None for value in (args.recipe, args.raw, args.output)):
            parser.error("--preflight does not accept recipe, raw, or output")
        print(json.dumps({"status": "prospective_inputs_match",
                          "prospective_sha256": sha(prospective_path)}, sort_keys=True))
        return
    if any(value is None for value in (args.recipe, args.raw, args.output)):
        parser.error("--recipe, --raw, and --output are required without --preflight")
    recipe = Path(args.recipe).resolve(strict=True)
    raw = Path(args.raw).resolve(strict=True)
    output = Path(args.output)
    manifest_path = recipe / "build-manifest.json"
    link_manifest_path = recipe / "link-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    link_manifest = json.loads(link_manifest_path.read_text())
    status = json.loads((raw / "compile-status.json").read_text())
    if manifest.get("version") != 2 or manifest.get("compiler_config", {}).get(
            "node_local_profile") != {"name": "unattested-v1"}:
        raise ValueError("preproof manifest is not exact proof-disabled node-local v2")
    if status.get("status") != "complete" or status.get("link_executed") is not False:
        raise ValueError("compile/prepare terminal record differs")
    compile_steps = [step for step in manifest["steps"] if step.get("kind") == "compile"]
    if len(compile_steps) != 25 or any("WT_KF_BUILD_INIT_TABLES_VERIFIED=1" in arg
                                     for step in compile_steps for arg in step["argv"]):
        raise ValueError("preproof object set or proof-disabled flags differ")
    validate_object_ledger(compile_steps, status)
    argv = json.loads((raw / "link-argv.json").read_text())
    expected = list(link_manifest["argv"])
    expected[0] = "/pkg/compiler/intel/2024/mpi/2021.11/bin/mpiifx"
    expected[expected.index("-o") + 1] = str(recipe / "wrf-wtune-unattested.exe")
    expected[1:1] = ["-Wl,-t", "-Wl,-Map," + str(recipe / "wrf-wtune-unattested.map")]
    if argv != expected:
        raise ValueError("executed link argv differs from deterministic preproof argv")

    trace = trace_files(raw / "link-trace.stdout")
    trace_set = set(trace)
    direct = prospective["lanes"]["libraries"]["resolutions"]
    absent = sorted({item["realpath"] for item in direct
                     if Path(item["realpath"]).resolve(strict=True) not in trace_set})
    if absent:
        raise ValueError("ordered direct selections absent from linker trace: " + repr(absent))
    candidate = (recipe / "wrf-wtune-unattested.exe").resolve(strict=True)
    link_map = (recipe / "wrf-wtune-unattested.map").resolve(strict=True)
    loader_list_path = raw / "loader-list.txt"
    mapping = loader_mapping(loader_list_path.read_text(errors="replace"))
    closure = loader_closure(candidate, mapping)
    candidate_needed, candidate_rpath, _ = dynamic(candidate)
    baseline_needed = prospective["library_audit"]["baseline_needed"]
    missing_needed = sorted(set(baseline_needed) - set(candidate_needed))
    if missing_needed:
        raise ValueError("candidate lost baseline runtime dependencies: " + repr(missing_needed))
    loader_hash, loader_path = (raw / "loader-sha256.txt").read_text().strip().split(None, 1)
    loader_entry = file_entry(Path(loader_path))
    if loader_entry["sha256"] != loader_hash:
        raise ValueError("native loader changed after inspection")
    report = {
        "artifact_type": "wtune_kf_native_preproof_closure", "version": 1,
        "attestation_note": "Native facts for human review; not an ABI or library attestation.",
        "prospective_inputs": file_entry(prospective_path),
        "prospective_preflights": [file_entry(raw / name) for name in
                                   ("prospective-preflight-before.json",
                                    "prospective-preflight-after.json")],
        "build_manifest": file_entry(manifest_path), "link_manifest": file_entry(link_manifest_path),
        "compiler_hashes": (raw / "compiler-sha256.txt").read_text().splitlines(),
        "compiler_observations": {name: file_entry(raw / name) for name in
                                  ("icx-version.txt", "ifx-version.txt", "mpiicx-show.txt",
                                   "mpiifx-show.txt", "build-environment.txt")},
        "link_argv": argv, "link_trace_files": [file_entry(path) for path in trace],
        "candidate": file_entry(candidate), "map": file_entry(link_map),
        "native_loader": loader_entry,
        "native_loader_list": file_entry(loader_list_path),
        "loader_selected": {name: file_entry(path) for name, path in sorted(mapping.items())},
        "candidate_needed": candidate_needed, "candidate_runpath": candidate_rpath,
        "recursive_runtime_closure": closure,
        "wrf_executed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
