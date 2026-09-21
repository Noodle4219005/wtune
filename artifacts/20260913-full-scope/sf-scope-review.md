# SFCLAY first-RK scope bridge review — 2026-09-13

## Scope and conclusion

Reviewed only the six assigned PLOV scope-bridge files.  The generated
first-RK wrapper is a compiler-managed 84-reference `--wrap` trampoline: it
uses `a0`/`a1` as the grid/config scope token inputs, calls `__real_*` once,
forwards `a0` through `a83` in order (including `feedback_is_ready`), and
always pairs `wt_sfclay_first_rk_end(token)` after that call.  The paired
surface hook retains its thread-local live token and leaves direct surface
calls to the existing disabled/fallback live-runtime path.  No frozen WRF
source, production enablement, or native Intel build/link was changed.

The first-RK audit accepts only the complete ordered signature and the audited
declaration profile.  It pins source SHA-256
`4668213aaec35d72f7bdbd2473d810b4538370f8aadf5979ef940f315be9c43b` and now
pins declaration-profile SHA-256
`545d198388690ca64d46e4aa8a0fabf565d4e176bf02dd93f72c2ab9fb0de7cf` as a
literal.  A changed generator profile fails at import rather than silently
emitting a new self-derived digest.

## Review changes

- `plov/gen_sfclay_scope_bridge.py`: replaced self-derived profile hash with
  the audited literal and a fail-closed import-time consistency check.
- `plov/tests/test_sfclay_scope.py`: added a profile-pin regression and CLI
  refusal cases that retain an existing output for missing input and two
  invalid C symbols.

## Evidence

Command, run from `/home/u5727520/wrf-stack/wtune-runtime-dev-20260912` with
`TMPDIR`, `PIXI_HOME`, and `PIXI_CACHE_DIR` under this artifact directory:

```text
PYTHONPATH=/home/u5727520/wrf-stack/wtune \
CC='/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/.pixi/envs/default/bin/x86_64-conda-linux-gnu-gcc -std=c11' \
pixi run python -m unittest plov.tests.test_sfclay_scope -v
```

The compiler identifies as `conda-forge gcc 16.2.0-4`.  Result: 5 tests passed
in 2.337 s.  The linked GCC 16 C11 fixture compiles
with `-Wall -Wextra -Werror -fsanitize=address,undefined` and exercises normal
forwarding, optional NULL references, direct fallback, nested/unmatched
surface hooks, and nested first-RK abort cases.  It also checks the default
object exposes the production `__wrap_*`/`__real_*` symbols without test helper
exports.

Before the pin change, the focused regression failed because the generator
contained `PROFILE_SHA256 = profile_digest(expected_profile())`; after the
literal pin and guard it passed.

## Prior-art and limits

`git log --all -- ...<six assigned files>` returned no committed history for
these new files.  At review start, `artifacts/20260913-full-scope/` did not
exist; the only nearby WTune artifact metadata was the prior
`20260912-kf-native-attestation` inventory.  This test proves the GNU fixture
bridge and parser rejection behavior; it does not certify the frozen Intel WRF
object ABI or activate `production.env`.
