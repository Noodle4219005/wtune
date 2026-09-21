# Changelog

Versions are declared in `wtune_version.py`; `test_wtune_version.py` fails if the top section
here does not match it. Development before 0.1.0 is in the git history, not in releases.

## 0.1.0 — 2026-09-20

First tagged version. WTune narrows a WRF configuration search and hands back a ranked
shortlist with evidence; it never selects or applies the final configuration.

### What is measured and settled

- **Decomposition** — 5–21.5% against an untuned default (all-MPI, one tile) across 13 cases, but
  only 0–4.1% against a good fixed template. The largest single knob is the **tile count**, not the
  rank count. The best decomposition does not move with simulated length (4 cases, 5–48 h), which
  closes phase-adaptive switching.
- **Only `nproc_x` changes WRF's arithmetic.** `nproc_y`, `numtiles`, thread count, rank count and
  rank placement are all bit-neutral, because the default tiling splits in J alone. This is why a
  fidelity twin that keeps `nproc_x` has a chaos floor of exactly zero.
- **Physics** — 0 of 6 templates recommendable. `radt`×2 gives no speed-up even on the case where
  radiation is 47% of the profile, which refutes the "low radiation share" explanation.
- **`&dynamics`** — one free lunch: turning off output diagnostics (`nwp_diagnostics`,
  `hailcast_opt`, `do_radar_ref`) is −2.7 to −3.5% with bit-identical prognostics.
  `time_step_sound` 4→2 is −5.6% and is the stability model's first validated forward prediction.
  `rk_ord`=2 is −10.8% but LARGE-DRIFT by 24 h. RK3/RK2 alternation across restart boundaries is
  closed: its drift is 99.9% of pure RK2's for half the speed-up.
- **I/O** is a guarantee, not a tuning axis: ADIOS2 with zstd, written explicitly into every
  namelist, with a checker that asserts it actually applied.

### Added

- `campaign/` — the STRICT and PHYSICS-GRAY measurement campaigns, 19 templates in three families
  (P physics, D dynamics-cost, A accuracy mirrors), the decomposition-search campaign, the
  per-window ranker and the decomposition summariser.
- `predictor/` — ledger, grouped-CV benchmark, `recommend.py` with `--objective perf|accuracy`,
  the decomposition space (`enumerate`/`sample`/`seed`) and the D3 active search,
  `stability.py` (acoustic-CFL margin), `verify_obs.py` (station T2 vs observations, storm track
  vs HURDAT2 — the only path that scores against something outside WRF).
- `fidelity_gate.py` — chaos-floor scoring against a reference and a twin.
- `experiments/` — the 9 harnesses that produced the conclusions above, vendored with sha256
  provenance and a test forbidding any runtime reference to another checkout.
- `tools/bp_compression_check.py` — asserts ADIOS2 compression applied, from the logical/stored
  byte ratio, because this ADIOS2 build carries no codec marker in BP metadata at all.
- `results/derived-data/` — the canonical processed CSV/JSON behind every number reported here.

### Fixed

- `campaign/configs.py` now writes `adios2_compression_enable` and `adios2_blosc_compressor`
  explicitly instead of inheriting the Registry default `lz4`. zstd is 8.0% smaller and faster to
  write.
- Four bugs that silently corrupted measurements rather than failing: `debug_level=50` made
  `rsl.out.0000` binary so `grep` harvested zero timings while recording `rc=0 ok=1` (all four
  reads now use `grep -a`); hailcast aborts when a nest's `history_interval` is 0, which collided
  with the nest-history-off convention; `fidelity_gate.py --family` rejected `diagnostics` and
  `nesting` only *after* WRF had run; and ADIOS2 compression reaches only the first write stream
  of a run.
- `tools/run_tests.sbatch` ran ~10% of the suite: it collected 74 tests where the repo has 762,
  because `test_fidelity_gate.py` matched no glob — the core fidelity logic's own test file had
  never run under the "full suite" — and `campaign/`, `predictor/`, `tools/`, `experiments/` and
  `deprecated/` had no stage at all. Now 744 passed / 18 skipped, job 1088417.
- `.gitignore`'s blanket `*.csv` excluded `results/derived-data/`, so the processed data behind
  the reported numbers was one `git clean` from gone.

### Known limitations

- Only the first ADIOS2 write stream of a run is compressed. The fix is two lines in
  `external/io_adios2/wrf_io.F90`, which is a WRF source change and is **not** applied here.
- `profile_*_share` is never written anywhere, so the cost-bucket demotion filter in
  `recommend.py` is a documented no-op.
- `campaign/cases.json` holds absolute paths; the tree is not portable to another machine as-is.
- Cross-hardware prediction does not generalise, and conus12 (the smallest grid) is a known weak
  spot.
- acc-first has not been validated against conus12's official reference solution, and the SC26-P1
  station observations are not in the tree, so `verify_obs.py station` cannot be exercised
  end-to-end.
