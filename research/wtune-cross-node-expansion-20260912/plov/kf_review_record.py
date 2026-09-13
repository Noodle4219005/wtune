#!/usr/bin/env python3
"""Validate the human-reviewed input closure for a KF reviewed-G0 recipe.

This module preserves a review decision; it does not infer scientific truth from
hashes.  The caller supplies the source/compiler/options/module lanes derived
from the recipe.  The record supplies reviewed baseline-link and resolved
library lanes plus concrete evidence files.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


class ReviewError(ValueError):
    pass


PROFILE = "reviewed-g0-v1"
STATEMENT = ("I reviewed the cited evidence for this exact pinned input closure; "
             "hashes preserve that review but do not independently prove the claims.")
CLAIMS = {
    "source_abi": "source-abi-audit",
    "caller_chain": "caller-chain-audit",
    "baseline_closure": "baseline-link-closure",
    "compiler_module_abi": "compiler-module-abi",
    "library_closure": "library-resolution",
    "node_init_static": "node-init-static-audit",
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
LIB = re.compile(r"-l([A-Za-z_][A-Za-z0-9_]*)\Z")
FINGERPRINTS = ("sources", "schema", "compilers", "options", "modules",
                "baseline_objects", "libraries", "scientific_build_closure")


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_booleans(value):
    if type(value) is bool:
        raise ReviewError("review records use conclusions, never proof booleans")
    if isinstance(value, dict):
        for item in value.values():
            _no_booleans(item)
    elif isinstance(value, list):
        for item in value:
            _no_booleans(item)


def _absolute(value, *, must_exist=True, directory=False, allow_leaf_symlink=False):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ReviewError("review paths must be absolute strings")
    path = Path(value)
    parents = path.parents if allow_leaf_symlink else (path, *path.parents)
    if any(part.is_symlink() for part in parents):
        raise ReviewError("unexpected symlink in review path: " + value)
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as error:
        raise ReviewError("missing review path: " + value) from error
    if not allow_leaf_symlink and resolved != path:
        raise ReviewError("review path must be canonical: " + value)
    if must_exist and (resolved.is_dir() != directory):
        raise ReviewError("review path has wrong file type: " + value)
    return path


def _file_entry(value, label="input"):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or \
            not isinstance(value.get("sha256"), str) or not HEX.fullmatch(value["sha256"]):
        raise ReviewError("invalid " + label + " file record")
    path = _absolute(value["path"])
    if file_hash(path) != value["sha256"]:
        raise ReviewError(label + " hash mismatch: " + str(path))
    return {"path": str(path), "sha256": value["sha256"]}, path


def _evidence(claim, value):
    expected_kind = CLAIMS.get(claim, "concurrency-limit")
    expected_keys = {"conclusion", "evidence"} if claim != "concurrency" else \
                    {"conclusion", "reason", "evidence"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ReviewError("invalid claim schema: " + claim)
    expected_conclusion = "verified" if claim != "concurrency" else "not_verified"
    if value["conclusion"] != expected_conclusion:
        raise ReviewError("invalid reviewed conclusion: " + claim)
    if claim == "concurrency" and (not isinstance(value["reason"], str) or
                                    not 8 <= len(value["reason"]) <= 1000):
        raise ReviewError("concurrency limitation requires a concrete reason")
    records = value["evidence"]
    if not isinstance(records, list) or not 1 <= len(records) <= 32:
        raise ReviewError("each reviewed claim needs 1..32 evidence files")
    paths = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"kind", "path", "sha256"} or \
                record.get("kind") != expected_kind:
            raise ReviewError("claim evidence kind/schema mismatch: " + claim)
        normalized, path = _file_entry({"path": record["path"], "sha256": record["sha256"]},
                                       claim + " evidence")
        paths.append(path)
        if normalized["path"] != record["path"]:
            raise ReviewError("non-canonical evidence path")
    if len(set(paths)) != len(paths):
        raise ReviewError("duplicate evidence path within claim: " + claim)
    return paths


def _library_candidates(name, directories):
    found = []
    for directory in directories:
        for filename in ("lib" + name + ".so", "lib" + name + ".a"):
            candidate = directory / filename
            if candidate.exists():
                if not candidate.is_file():
                    raise ReviewError("library candidate is not a file: " + str(candidate))
                found.append(candidate)
    return found


def _library_contract(value):
    keys = {"explicit_search_paths", "implicit_search_paths", "tokens", "resolutions"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ReviewError("invalid library contract schema")
    explicit = value["explicit_search_paths"]
    implicit = value["implicit_search_paths"]
    if not isinstance(explicit, list) or not isinstance(implicit, list) or \
            not explicit or len(explicit) + len(implicit) > 128:
        raise ReviewError("invalid bounded library search path lists")
    directories = [_absolute(item, directory=True) for item in explicit + implicit]
    tokens = value["tokens"]
    resolutions = value["resolutions"]
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= 256 or \
            any(not isinstance(token, str) or not LIB.fullmatch(token) for token in tokens) or \
            not isinstance(resolutions, list) or len(resolutions) != len(tokens):
        raise ReviewError("invalid unresolved library token ledger")
    pinned = []
    normalized = []
    for token, record in zip(tokens, resolutions):
        keys = {"token", "link_path", "realpath", "sha256", "soname", "shadows"}
        if not isinstance(record, dict) or set(record) != keys or record.get("token") != token or \
                not isinstance(record.get("soname"), str) or not record["soname"] or \
                not isinstance(record.get("shadows"), list):
            raise ReviewError("invalid ordered library resolution record")
        name = LIB.fullmatch(token).group(1)
        candidates = _library_candidates(name, directories)
        if not candidates:
            raise ReviewError("unresolved reviewed library: " + token)
        link_path = _absolute(record["link_path"], allow_leaf_symlink=True)
        if link_path != candidates[0]:
            raise ReviewError("reviewed library is not the ordered first candidate: " + token)
        realpath = link_path.resolve(strict=True)
        normalized_file, selected = _file_entry(
            {"path": record["realpath"], "sha256": record["sha256"]}, "library")
        if selected != realpath:
            raise ReviewError("reviewed library realpath differs from selected candidate: " + token)
        actual_shadows = []
        for candidate in candidates[1:]:
            resolved = candidate.resolve(strict=True)
            if resolved == realpath:
                continue
            entry = {"path": str(resolved), "sha256": file_hash(resolved)}
            if entry not in actual_shadows:
                actual_shadows.append(entry)
        if record["shadows"] != actual_shadows:
            raise ReviewError("library shadow ledger differs: " + token)
        pinned.extend([selected, *(Path(item["path"]) for item in actual_shadows)])
        normalized.append({**record, "realpath": normalized_file["path"]})
    return {**value, "resolutions": normalized}, pinned


def load(path, *, lanes, output_dir):
    """Return validated record, derived profile id, and all record-owned inputs."""
    record_path = _absolute(str(path))
    try:
        record = json.loads(record_path.read_text())
    except (OSError, UnicodeError, ValueError) as error:
        raise ReviewError("invalid reviewed record JSON") from error
    _no_booleans(record)
    keys = {"artifact_type", "version", "profile", "profile_id", "human_review", "claims",
            "fingerprints", "baseline_link_inputs", "library_contract", "link_output"}
    if not isinstance(record, dict) or set(record) != keys or \
            record.get("artifact_type") != "wtune_kf_reviewed_g0_attestation" or \
            record.get("version") != 1 or record.get("profile") != PROFILE:
        raise ReviewError("unsupported reviewed-G0 record")
    human = record["human_review"]
    if not isinstance(human, dict) or set(human) != {"reviewer", "reviewed_utc", "statement"} or \
            not isinstance(human.get("reviewer"), str) or not 1 <= len(human["reviewer"]) <= 256 or \
            human.get("statement") != STATEMENT:
        raise ReviewError("review record lacks the explicit human-review statement")
    try:
        reviewed = datetime.fromisoformat(human["reviewed_utc"])
    except (TypeError, ValueError) as error:
        raise ReviewError("reviewed_utc must be an ISO-8601 timestamp") from error
    if reviewed.tzinfo is None or reviewed.utcoffset() is None:
        raise ReviewError("reviewed_utc must include a timezone")
    claims = record["claims"]
    if not isinstance(claims, dict) or set(claims) != set(CLAIMS) | {"concurrency"}:
        raise ReviewError("review record must contain exactly six verified claims and concurrency")
    pinned = [record_path]
    for claim in (*CLAIMS, "concurrency"):
        pinned.extend(_evidence(claim, claims[claim]))
    baseline = record["baseline_link_inputs"]
    if not isinstance(baseline, list) or not 1 <= len(baseline) <= 128:
        raise ReviewError("baseline link input ledger must contain 1..128 files")
    normalized_baseline = []
    for item in baseline:
        normalized, input_path = _file_entry(item, "baseline link input")
        normalized_baseline.append(normalized)
        pinned.append(input_path)
    if len({item["path"] for item in normalized_baseline}) != len(normalized_baseline):
        raise ReviewError("duplicate baseline link input")
    libraries, library_paths = _library_contract(record["library_contract"])
    pinned.extend(library_paths)
    actual_lanes = dict(lanes)
    actual_lanes["baseline_objects"] = normalized_baseline
    actual_lanes["libraries"] = libraries
    actual = {name: canonical_hash(actual_lanes[name]) for name in FINGERPRINTS[:-1]}
    actual["scientific_build_closure"] = canonical_hash(actual)
    fingerprints = record["fingerprints"]
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(FINGERPRINTS) or \
            any(not isinstance(value, str) or not HEX.fullmatch(value)
                for value in fingerprints.values()) or fingerprints != actual:
        raise ReviewError("reviewed scientific/build closure fingerprint differs")
    profile_id = PROFILE + ":" + actual["scientific_build_closure"]
    if record["profile_id"] != profile_id:
        raise ReviewError("reviewed profile ID differs from canonical input closure")
    output = Path(output_dir)
    routes = record["link_output"]
    expected_routes = {"candidate": str(output / "wrf-wtune.exe"),
                       "map": str(output / "wrf-wtune.map"),
                       "identity_record": str(output / "wrf-wtune.identity.json")}
    if routes != expected_routes:
        raise ReviewError("reviewed link output route differs from the separate output directory")
    return record, profile_id, pinned, actual_lanes


def validate_manifest_closure(manifest, config, entries, compilers, output):
    """Independently join a version-3 manifest back to its review record."""
    closure = manifest.get("scientific_build_closure")
    if not isinstance(closure, dict):
        raise ReviewError("missing scientific build closure")
    expected_options = dict(config)
    expected_options["node_local_profile"] = {"name": PROFILE}
    if closure.get("options") != expected_options or \
            closure.get("modules") != manifest.get("fortran_module_inputs"):
        raise ReviewError("reviewed options/module closure differs")
    actual_compilers = closure.get("compilers")
    if not isinstance(actual_compilers, list) or \
            [(item.get("role"), item.get("path")) for item in actual_compilers
             if isinstance(item, dict)] != compilers:
        raise ReviewError("reviewed compiler closure differs")
    input_ledger = {(entry["path"], entry["sha256"]) for entry in entries}
    closure_files = list(closure.get("sources", []))
    closure_files.append(closure.get("schema"))
    closure_files.extend(actual_compilers)
    for record in closure.get("modules", []):
        closure_files.extend(record.get("members", []))
    closure_files.extend(closure.get("baseline_objects", []))
    for resolution in closure.get("libraries", {}).get("resolutions", []):
        closure_files.append({"path": resolution.get("realpath"),
                              "sha256": resolution.get("sha256")})
        closure_files.extend(resolution.get("shadows", []))
    if any(not isinstance(item, dict) or
           (item.get("path"), item.get("sha256")) not in input_ledger for item in closure_files):
        raise ReviewError("reviewed closure file is absent from the pinned input ledger")
    node = manifest.get("node_local_profile")
    review_ref = node.get("review_record") if isinstance(node, dict) else None
    if not isinstance(review_ref, dict) or set(review_ref) != {"path", "sha256"} or \
            (review_ref["path"], review_ref["sha256"]) not in input_ledger:
        raise ReviewError("invalid reviewed record reference")
    _, profile_id, _, actual = load(
        review_ref["path"], lanes={key: closure[key]
                                   for key in ("sources", "schema", "compilers", "options", "modules")},
        output_dir=output)
    if profile_id != node.get("profile_id") or actual != closure:
        raise ReviewError("reviewed profile/closure reconstruction differs")


def link_command(manifest, generated):
    """Validate the prepared manifest and return the one exact reviewed link."""
    output = Path(manifest["output_dir"])
    try:
        link = json.loads((output / "link-manifest.json").read_text())
        review = json.loads(Path(manifest["node_local_profile"]["review_record"]["path"]).read_text())
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ReviewError("unable to read reviewed link metadata") from error
    routes = review["link_output"]
    closure = manifest["scientific_build_closure"]
    libraries = closure["libraries"]
    objects = [item for item in generated if Path(item["path"]).parent == output / "objects"]
    direct = closure["baseline_objects"] + [
        {"path": item["path"], "sha256": item["sha256"]} for item in objects]
    wrappers = [step["output"] for step in manifest["steps"] if step.get("kind") == "compile"]
    if link.get("artifact_type") != "wtune_native_link_command_manifest" or \
            link.get("frozen_root") != manifest["frozen_root"] or \
            link.get("historical_cwd") != str(Path(manifest["frozen_root"]) / "main") or \
            link.get("link_cwd") != manifest["output_dir"] or link.get("output") != routes["candidate"] or \
            link.get("wrap_symbols") != manifest["wrap_symbols"] or \
            link.get("wrapper_objects") != wrappers or link.get("direct_input_hashes") != direct or \
            link.get("resolved_library_search_paths") != libraries["explicit_search_paths"] or \
            link.get("unresolved_implicit_dependencies") != libraries["tokens"]:
        raise ReviewError("generated link manifest differs from reviewed closure")
    argv = link.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
        raise ReviewError("invalid reviewed link argv")
    fc = next(item["path"] for item in closure["compilers"] if item["role"] == "fc")
    if Path(argv[0]).name != Path(fc).name or argv.count("-o") != 1 or \
            argv[argv.index("-o") + 1] != routes["candidate"]:
        raise ReviewError("link driver or separate output route differs")
    return [fc, "-Wl,-Map," + routes["map"], *argv[1:]], output, routes


def linked_identity(manifest, manifest_sha256, argv, routes, generated):
    """Create deterministic provenance for measured outputs; this is not an ABI claim."""
    output = Path(manifest["output_dir"])
    candidate = next(item for item in generated if item["path"] == routes["candidate"])
    link_map = next(item for item in generated if item["path"] == routes["map"])
    objects = [item for item in generated if Path(item["path"]).parent == output / "objects"]
    return {"artifact_type": "wtune_kf_linked_candidate_identity", "version": 1,
            "attestation_note": "Build provenance only; not native ABI or WRF runtime validation.",
            "profile_id": manifest["node_local_profile"]["profile_id"],
            "build_manifest_sha256": manifest_sha256, "plan_sha256": manifest["plan_sha256"],
            "link_manifest_sha256": file_hash(output / "link-manifest.json"), "link_argv": argv,
            "link_argv_sha256": canonical_hash(argv), "link_cwd": manifest["output_dir"],
            "candidate": candidate, "link_map": link_map, "objects": objects}
