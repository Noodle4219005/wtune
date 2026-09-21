# KF Fortran module-directory input contract review

Verdict: **CONDITIONAL ACCEPT** for a complete, fail-closed snapshot of the explicitly configured immutable module directories. Do not describe that alone as complete Fortran compiler input closure: the current argv also exposes `plov`, compiler-produced module output, the working directory, and ambient compiler search behavior.

Scope was limited to exact source review of:

- `plov/prepare_kf_build.py` (262 lines)
- `plov/execute_build_recipe.py` (340 lines)
- Prior knowledge in `knowledge/wrf/2026-09-12-wtune-kf-native-input-inventory.md`

No implementation, native-tree search, compiler invocation, cluster query, build, or submission was performed.

## Why directory-wide binary snapshotting is the correct input model

Do not make source-derived `USE` metadata the admission boundary. The failed r2 evidence already showed that an apparently direct missing module can itself import more modules, and preprocessing, conditional compilation, intrinsic/compiler modules, and binary naming rules can make a source graph incomplete. Source metadata may explain failures, but it is not sufficient to prove compiler binary inputs.

For each explicitly configured immutable `-I` directory, snapshotting every direct regular `.mod` and `.smod` entry is conservative and auditable. Direct-only enumeration is correct because `-I <dir>` does not recursively search subdirectories. `.smod` is required: Fortran submodule interface files are compiler-consumable binary inputs even if the current two WTune sources do not deliberately use submodules. Match both suffixes case-insensitively so `.MOD`/`.SMOD` cannot evade the snapshot.

This proves membership and bytes of the nominated directories at checked command boundaries. It does not prove which files were read, semantic `USE` closure, module/compiler ABI compatibility, or native correctness.

## Required contract invariants

### 1. Ordered configuration must preserve `frame`

`fortran_module_dirs` should be an optional compiler-config array of lexical paths relative to `frozen_root`. Canonical configuration must always contain a nonempty ordered list whose first entry is exactly `frame`:

- omitted key canonicalizes to `["frame"]`;
- an explicit list must begin with `frame`, then name zero or more additional directories in intended `-I` precedence;
- reject an empty list, duplicate directories, absolute paths, empty path components, `.`, `..`, and non-string values.

This should be additive rather than replaceable. Current preparation explicitly pins `frozen/frame/module_domain_type.mod` at `prepare_kf_build.py:137-141`, validation requires it at `execute_build_recipe.py:179-181`, and both builders put `-I frozen/frame` into every Fortran argv (`prepare_kf_build.py:203-214`, `execute_build_recipe.py:210-218`). Allowing a user-supplied list to omit or displace `frame` weakens that established input and can select an unintended provider.

The manifest should store the canonical relative list in `compiler_config`; reconstructed Fortran argv must emit one `-I <frozen_root/relative-dir>` pair per entry in exactly that order. Keep the order visible and exact in the step equality check at `execute_build_recipe.py:237-238`.

### 2. Reject symlinks and traversal lexically and after resolution

For every configured relative path:

1. reject absolute/anchored paths and any lexical `.` or `..` component before normalization;
2. walk each existing component with `lstat` and reject a symlink at the directory or any parent below `frozen_root`;
3. resolve strictly, require a directory, and require the result to remain below `frozen_root`;
4. enumerate direct entries without following symlinks; every selected module entry must be a regular non-symlink file;
5. reject directories with zero direct `.mod`/`.smod` files.

The existing preparer `_file` at `prepare_kf_build.py:42-50` resolves first and therefore can accept an in-root symlink. It is not sufficient for this stronger contract. The executor `_path` at `execute_build_recipe.py:31-41` already rejects symlink components for manifest paths; the new preparer helper must provide equivalent behavior before the manifest is created.

Hard links do not need a special rejection if the path is inside the frozen root, bytes are hashed, and the existing no-concurrent-write invariant holds. Record paths and content, not inode identity.

### 3. Snapshot exact membership as well as content

Each configured directory needs a manifest record containing:

- canonical absolute directory path and canonical relative config value;
- ordered direct member list;
- for each member: original basename, canonical absolute path, SHA-256, and preferably size;
- a deterministic membership digest over the ordered member records.

Also include every member path in `input_hashes`, so the existing per-file mechanism remains the single content-pin ledger. Do not record only the currently imported subset.

The preparer must enforce explicit bounds before creating the output directory: maximum directory count, per-directory member count, total member count, and compatibility with the executor's total input limit. Current `validate` accepts at most 512 `input_hashes` entries at `execute_build_recipe.py:111-125`; silently truncating a larger module set is forbidden. Either retain that bound and fail preparation clearly or introduce a reviewed larger bound in a new manifest version.

The executor must independently reconstruct configured directories from `frozen_root` plus canonical relative values, validate the membership-record schema and uniqueness, enumerate actual direct module names, compare the exact ordered name set, and then compare every file hash. This catches addition, deletion, rename, suffix-case change, and content drift. Current `_check_pinned` at `execute_build_recipe.py:61-73` checks only enumerated files, so it detects deletion/content changes but cannot detect an added module; membership comparison must be added there.

Run the full membership-plus-hash check during initial `validate`, before and after every non-mkdir command, and in final verification, preserving the current boundary pattern at `execute_build_recipe.py:291`, `:311`, and `:326`. The existing caller invariant against concurrent writes remains necessary because boundary hashing cannot prove immutability during compiler execution. The preparer should state the same no-concurrent-mutation requirement while it enumerates and hashes.

### 4. Collision policy must use the complete filename, casefolded

Across all explicitly configured immutable module directories, reject any duplicate key `member.name.casefold()`, including duplicates within one directory and byte-identical copies in different directories. Do not resolve these through `-I` precedence: Fortran module names are case-insensitive and compiler lookup details are not an ABI proof. Rejecting identical copies also prevents provenance from changing silently if one later diverges.

The collision key should include the extension. `foo.mod` and `FOO.mod` collide; `foo.smod` and `FOO.SMOD` collide; `foo.mod` and `foo.smod` are distinct artifacts and should not collide merely because their stems match. Preserve the original filename in the manifest. No source-derived module name is needed for this check.

### 5. Distinguish configured-directory closure from the actual compiler search surface

Every current compile argv begins with `-I plov` (`prepare_kf_build.py:203-205`, `execute_build_recipe.py:210-212`). A stray `plov/*.mod` or `plov/*.smod` can therefore shadow or supplement the frozen module inputs while remaining absent from the proposed configured-directory snapshot. At minimum, preparation and every executor boundary must require that `plov` contains zero direct `.mod`/`.smod` files. This zero-membership guard must detect later additions, not merely check once.

`output/modules` is intentionally compiler-writable through `fortran_module_flag` (`prepare_kf_build.py:213-214`; executor reconstruction at `execute_build_recipe.py:217-218`). Modules generated by an earlier Fortran step may become inputs to a later step. They cannot be treated as immutable pinned inputs known at recipe creation. If the claim is expanded to complete module input closure, the executor must record output-module membership/hashes after each successful producer step, verify that exact generated state before the next step, reject unexpected mutation/removal, and reject casefold collisions with immutable module inputs. Otherwise state plainly that generated module state is outside the immutable-directory closure.

The compiler may also search its working directory, configured environment variables, or toolchain defaults. Existing recipe constraints already say ambient include paths and toolchain dependencies remain unaudited (`prepare_kf_build.py:247-251`). Preserve that limitation. A directory-snapshot change alone supports this exact claim:

> All direct `.mod`/`.smod` members of the explicitly configured frozen module directories were pinned and unchanged at every checked command boundary.

It does not support “all Fortran compiler inputs were closed.” Textual Fortran `INCLUDE` files in those `-I` directories are also outside a module-binary-only contract.

### 6. Manifest validation must not trust duplicated declarations

The executor already pins the preparer implementation and executes it as a validation helper at `execute_build_recipe.py:51-59`. Put canonical parsing/enumeration rules in that preparer module and call them from validation, while still independently checking manifest structure and bounds before use.

Require exact agreement among:

- canonical `compiler_config.fortran_module_dirs`;
- directory membership records;
- module entries in `input_hashes`;
- reconstructed ordered `-I` argv for every Fortran compile step.

The canonical plan hash protects bytes, but validation must reject internally inconsistent signed data rather than trusting one manifest section. Since the manifest schema and exact step construction change, version 2 is clearer than silently changing version-1 semantics. The existing `preparer_sha256` already makes old recipes fail closed, but a version bump makes the new contract explicit to independent readers and prevents downgrade ambiguity.

Keep `compiler_module_abi_verified` false. Directory hashes are provenance, not ABI certification.

## Counterexamples the implementation tests must falsify

1. Omitted field produces canonical `["frame"]`; explicit `[]`, a list not beginning with `frame`, duplicate dirs, absolute paths, `a/../frame`, `./frame`, symlinked dir/component, and an empty/no-module directory all fail before output creation.
2. `.MOD` and `.SMOD` members are included. A symlinked module file is rejected even when its target remains under `frozen_root`.
3. Adding an otherwise unused `.mod`, deleting one, renaming only its case, adding `.smod`, or changing one byte after preparation fails initial validation. The same mutations injected immediately before or during a mocked step yield `stale_input` at the existing pre/post boundary and no later step launches.
4. `Foo.mod` versus `foo.MOD` in one or two configured dirs is rejected; identical hashes do not exempt it. `foo.mod` plus `foo.smod` is accepted as two distinct full filenames.
5. Directory order is preserved exactly in every reconstructed Fortran argv; swapped, omitted, duplicated, or injected `-I` pairs make `steps != expected` fail.
6. A new `plov/Shadow.mod` after preparation fails the zero-module guard before a compiler launches; the same injection during a mocked compiler step makes the post-step check fail.
7. Manifest tampering that edits only config dirs, only membership records, only `input_hashes`, or only step argv is rejected even after recomputing the outer plan digest in a test fixture.
8. Exceeding any directory/member/total-input bound fails preparation without a partial manifest or silently omitted module.
9. If generated output modules are brought into scope, mutation/removal between producer and consumer and a casefold collision with a frozen module both fail before the consumer launches.

With these invariants, adding `external/esmf_time_f90` after mandatory `frame` is a safe provenance expansion for a new recipe. It does not authorize reuse of the failed r2 recipe, establish the complete semantic import graph, prove Intel module ABI compatibility, or authorize another build.

