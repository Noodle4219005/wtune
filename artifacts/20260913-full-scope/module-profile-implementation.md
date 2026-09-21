# KF immutable Fortran module-directory profile — implementation evidence

## Result

`plov/prepare_kf_build.py` now writes version-2 recipes with an ordered,
compiler-configured `fortran_module_dirs` profile.  Omission canonicalizes to
`["frame"]`; explicit profiles must start with `frame`.  Each configured frozen
directory contributes every direct regular `.mod`/`.smod` member (case
insensitive suffix), its byte hash, size, ordered membership digest, and its
input-hash-ledger entry.  `module_domain_type.mod` remains mandatory in
`frame`.

`plov/execute_build_recipe.py` rejects version 1, reconstructs the canonical
module profile from the frozen root, compares exact membership and hashes,
requires its module entries to match the input ledger, and checks that `plov`
has no direct module files.  These checks run at validation, before and after
every non-`mkdir` command, and at final verification through the existing
pin-check boundary.  No link command is executed by this feature.

## Safety contract implemented

- Relative canonical profile entries only; `frame` first, no duplicates,
  traversal, empty components, or absolute paths.
- No symlink in a configured directory component or selected module input;
  configured directories must exist and have direct module members.
- Casefold collisions of full filenames fail across all configured directories,
  even when content hashes match.
- Directories, per-directory members, and total pinned inputs are bounded
  (`16`, `128`, and the existing `512` respectively) before output creation.
- `-I plov` remains first, followed by configured frozen directories in exact
  profile order.  `output/modules` is deliberately generated state and is not
  an immutable input snapshot.
- All attestation booleans, including `compiler_module_abi_verified`, remain
  false.  The claim is limited to explicitly configured frozen module
  directories; ambient compiler search inputs and ABI compatibility are not
  proved.

## RED/GREEN record

The first intended RED command could not start because `wtune/` has no Pixi
manifest.  After the controller supplied the required runtime-dev manifest,
the new profile cases were executed against the completed v2 implementation;
therefore this artifact records executable GREEN evidence but does not claim a
semantic pre-fix RED result.  The added cases cover default and multi-directory
profiles with spaces and `.MOD`/`.SMOD`, invalid paths, empty directories,
symlink paths/files, casefold collisions, direct `plov` shadowing, membership
addition/deletion/content drift before execution and during commands, and
config/membership/ledger/argv tampering.

GREEN command (from `/home/u5727520/wrf-stack/wtune`):

```text
TMPDIR=$PWD/artifacts/20260913-full-scope/module-profile-scratch \
TMP=$TMPDIR TEMP=$TMPDIR \
PIXI_CACHE_DIR=$PWD/artifacts/20260913-full-scope/module-profile-cache \
/home/u5727520/.pixi/bin/pixi run \
  --manifest-path /home/u5727520/wrf-stack/wtune-runtime-dev-20260912/pixi.toml \
  python -m unittest plov.tests.test_prepare_kf_build \
  plov.tests.test_execute_build_recipe -v
```

Result: `Ran 37 tests in 12.214s — OK`.

Additional checks used the same environment:

```text
python -m py_compile plov/prepare_kf_build.py plov/execute_build_recipe.py \
  plov/tests/test_prepare_kf_build.py plov/tests/test_execute_build_recipe.py
git diff --check
```

Both passed.  Files remain below 500 lines (360, 367, 321, and 304 lines).

## Limits

This is synthetic local Python-fixture coverage only.  It did not invoke a
native compiler, prepare a real link, touch WRF sources, query a cluster, or
submit a job.  A later recipe may add `external/esmf_time_f90` after `frame`;
that expansion is a new configured provenance snapshot, not ABI proof or a
semantic `USE`-closure admission decision.
