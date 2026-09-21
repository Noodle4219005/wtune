# KF version-2 Fortran module profile focused review

## Verdict: ACCEPT

I found no consequential counterexample in the reviewed version-2 module-profile delta. The implementation preserves the previously accepted boundary: it proves an exact snapshot only for explicitly configured frozen module directories at command boundaries. It does not claim semantic `USE` closure, compiler/module ABI compatibility, or immutability of the working directory, generated `output/modules`, or other ambient compiler search inputs.

Reviewed read-only:

- `plov/prepare_kf_build.py`
- `plov/execute_build_recipe.py`
- `plov/tests/test_prepare_kf_build.py`
- `plov/tests/test_execute_build_recipe.py`

No source was modified, and this review did not run compilers, builds, cluster operations, or tests.

## Consequential invariants checked

1. **Canonical, bounded directory profile.** `prepare_kf_build.py:62-76` accepts 1..16 canonical relative directories, requires `frame` first, and rejects absolute paths, empty elements, `.`/`..`, repeated separators, and duplicate entries. `prepare_kf_build.py:79-94` rejects a symlink in every configured path component, requires an existing directory, and verifies containment in the frozen root. Tests cover invalid forms, missing/empty directories, directory overflow, directory symlinks, and member symlinks at `tests/test_prepare_kf_build.py:143-197`.

2. **Exact direct-member snapshot.** `prepare_kf_build.py:97-128` enumerates direct `.mod` and `.smod` regular non-symlink files, requires 1..128 members per directory, records filename/path/SHA-256/size, records a membership digest, and rejects a case-folded full-filename collision across all configured directories. `prepare_kf_build.py:215-229` folds every recorded member into the common input set, and `prepare_kf_build.py:267-268` caps that set at 512. The profile and ordered `-I` arguments are exercised at `tests/test_prepare_kf_build.py:113-197`, including uppercase `.MOD`, `.SMOD`, cross-directory collision, and both bounds.

3. **Version-2 manifest and tamper consistency.** `prepare_kf_build.py:319-342` emits version 2, the canonical compiler configuration, module profile, input hashes, false proof flags, the preparer hash, and a canonical plan hash before creating the new output directory. `execute_build_recipe.py:56-64` refuses a different local preparer before using its validation logic. `execute_build_recipe.py:105-260` verifies the outer digest, canonical plan digest, exact version/proof state, all input hashes, the hashed compiler configuration, and reconstructs the complete step list and ordered Fortran search arguments. Rehashed changes to the directory list, profile member, member ledger, argv, and proof flags are refused by `tests/test_execute_build_recipe.py:195-220` and `:232-237`.

4. **Membership additions, deletions, substitutions, and case collisions fail closed.** `execute_build_recipe.py:83-102` re-runs the canonical directory parser and snapshotter, requires exact profile equality, and requires exact set equality between module members and `.mod`/`.smod` entries in the general input ledger. Thus a member addition or deletion changes the live record even if all original hashes still match; a byte change also fails the ordinary input hash check. Pre-execution add/change/delete falsifiers are present at `tests/test_execute_build_recipe.py:138-154`. The same shared snapshotter re-applies path, symlink, bound, type, and case-collision checks in the executor.

5. **Pre/post command checking and injection containment.** `execute_build_recipe.py:67-102` checks the manifest, preparer, every ordinary input, exact live module membership, and the direct `plov` zero-module surface. `execute_build_recipe.py:263-347` invokes that check immediately before and after every external command and once at final verification. Tests make a command change a module, replace membership by delete-plus-add, and add a `plov` shadow; execution becomes terminal `stale_input` before the next compile (`tests/test_execute_build_recipe.py:156-193`). This is boundary checking, as declared; an add-and-remove wholly within one command interval remains outside the stated guarantee and is also excluded by the returned caller invariant forbidding concurrent writes.

6. **Search-scope claim is accurate.** Preparation puts `plov` first and the configured frozen directories after it in explicit order, then directs generated compiler modules to `output/modules` (`prepare_kf_build.py:280-299`). Direct module files in `plov` are forbidden during preparation and at each execution boundary (`prepare_kf_build.py:131-136`; `execute_build_recipe.py:83-102`). The manifest explicitly says configured frozen directories are pinned only at checked boundaries and says generated `output/modules` and ambient compiler search inputs remain outside the immutable snapshot (`prepare_kf_build.py:325-334`). Because each compile runs with `cwd=output`, the working directory is one of those ambient, excluded inputs. The artifact therefore makes no unsupported whole-search-path closure claim.

7. **False proofs, one-shot execution, and output preservation.** `execute_build_recipe.py:105-122` accepts only the exact five-key, all-false attestation and only version 2, so an older failed recipe is not accepted as a new version-2 recipe. Validation refuses an existing execution marker and every planned destination before any execution write (`execute_build_recipe.py:123-136`, `:252-258`). Execution creates the marker exclusively before commands and never removes it; failure and timeout are terminal and cannot replay (`execute_build_recipe.py:263-347`; `tests/test_execute_build_recipe.py:258-266`, `:288-295`). Dry validation leaves only the manifest, successful execution refuses replay, and target files are required absent before launch (`tests/test_execute_build_recipe.py:88-118`). Preparation likewise performs validation and hashing before creating a new output directory, and preserves an already existing manifest (`tests/test_prepare_kf_build.py:205-210`, `:265-269`).

## Evidence limitation

The reported 37-test green run was supplied by the implementation lane and was not repeated under this read-only review. The implementation lane also disclosed that its attempted historical-prefix RED did not produce a usable test result because the first environment command failed. That missing historical run does not change this source-level verdict, but it should not be cited as RED evidence.
