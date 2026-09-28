# Changelog

Versions are declared in `wtune_version.py`; `test_wtune_version.py` fails if the top section
here does not match it. Development before 0.1.0 is in the git history, not in releases.

## 0.2.0 — 2026-09-27

Whole-run prediction from rebuilt segments, a run monitor, segmented execution with restart
hand-offs, and an opt-in tuner mode that ranks by predicted whole-run time. Details and all
evidence: README section 6. Everything is on scc; errors are against a truth run of the same
case and configuration. The 85 commits between 0.1.0 and 2026-09-26 (`wtune check`,
`wtune calibrate`, multi-point probe planning, scenario generators, restart-based ranking start)
are in the git history, not listed here.

### Added

- **`wtune predict-rebuild`** (decision 0054) — every sampled point rebuilt from ERA5 with cloud
  water/ice (20-min spin-up; `--forcing auto` = GFS chain, 60 min), measured on the target
  decomposition as parallel jobs, combined into compute+init. conus25 +0.98%, maria1km −0.04%,
  zhengzhou −5.78%, lafe (24 h) +0.16%; ≈ ±2% after same-day drift correction (scc varies ±2–4%
  between days). lafe driver 32:41. `--measure-steps short` = `max(30, stepra+10)` is opt-in
  (out-of-sample −1.11% dorian2019, +0.38% lafe vs the default window).
- **Measured I/O probe and a recommended `--time`** — one history frame per domain (+ a restart)
  timed with the case's own output settings: −0.5% / −0.5% / +3.1% of the truth's Σ writes
  (dorian2019 / maria1km / zhengzhou). `--time` = range_high + 2 × I/O + 10% margin clamped to
  2–5 min: 1.33× / 1.25× / 1.09× of the real wall.
- **`wtune watch`** — read-only monitor of `rsl.*.0000`: progress, remaining compute and writes
  separately, `--prediction` scaling, HANG / HANG_INIT / NEST_END_EARLY / NEST_STOPPED / FAILED /
  CFL, replay mode. 25% replay: 8 of 9 truths within 10% rate-only, 5 of 5 with prediction.
- **`wtune run`** — segmented execution on one restart quantum L, ADIOS2 BP4 restarts, per-segment
  config from a predicted-mode tune report, opt-in CFL rescue (PHYSICS-GRAY), `--dry-run`.
  dorian2019 30 + 30 min chain vs one-shot: 202 + 202 variables bit-identical (jobs 1405+1410 vs
  1392).
- **`wtune tune --rank-by predicted`** (decision 0055, opt-in; default `t0` unchanged) — one shared
  spin-up per point, ANCHOR/ANCHOREND in every job, top 3 measured at every point and ranked by
  predicted whole-run compute+init. Running-time budget (`--max-wall-minutes` cap, default 3×),
  stage-2 reserve, stage-1 shrink-to-fit, feasibility refusal before any submission,
  `--max-points` default 3.
- Ranking restarts via ADIOS2 BP4 (`adios2.xml`, no code change): write 1.05 s vs 323.5 s netCDF,
  read 2.3–2.5 s vs 120 s, bit-identical 60 min later (jobs 1277/1288/1289); readable across
  288/144/72 ranks (G0, jobs 1371/1372: ratios within 0.38%).
- `WTUNE_SBATCH_EXCLUDE` adds nodes to `--exclude=head` in every generated sbatch;
  `node_hygiene.sh` reports `FOREIGN_LOAD` (non-own process, lifetime %CPU > 20, ≥ 30 s). scc
  work5 carried a root VS Code server outside Slurm that slowed a 576-rank `wrf.exe` from 1.25 to
  6.3 s/step.
- `wtune check`: C3 physics-table rules from WRF 4.8's own `OPEN` statements (README.physics_files
  over-lists); C10 wrfinput sanity with land/water masks and a 0.1% fraction rule for non-zero
  out-of-range land TSLB (dorian2019 d02 0.034% → WARN, ran with no NaN; tw3km case_I 8.8% →
  FATAL). 11 healthy truth dirs: zero FATAL.
- `tools/era5_intermediate.py` (ARCO-ERA5 → WPS intermediate, `--rain-snow` from the 137-level
  store), `tools/rescore_rebuild.py`, `tools/tc_bogus.py` (opt-in: failed its phase-5 criteria),
  scenario generator default `aifs` (7.46% vs analog 7.53%, within noise).

### Fixed

- **netCDF thread race.** Rebuild points are prepared in a thread pool; concurrent netCDF4 opens
  of the same wrfinput corrupted netCDF-C's ID table (`NetCDF: Not a valid ID`; one driver died
  without a traceback) and, in `vertical_grid_from_wrfinput`, the failed read was swallowed as a
  warning: **4 of 50 scc plans silently ran WRF's default eta levels** instead of the case's.
  Both reads now share one module lock. Of cited results only dorian2019's short-window pair was
  affected (one clean pair left).
- `run_seconds` is zeroed next to `run_minutes`: WRF adds all four `run_*` keys, and rebuild
  namelists carry a non-zero `run_seconds`, which would silently lengthen a ranking run.
- Ranking candidates starting from a restart set `override_restart_timers = .true.`: WRF otherwise
  kept the spin-up's 20-min restart interval and wrote a restart inside every measured window
  (netCDF 299 s on a radiation step). `wtune run` segments deliberately do **not** set it —
  with it, d01's 60-min history alarm re-anchored at a 00:30 boundary and the 01:00 frame was
  never written.
- `DEFAULT_SBATCH_SCRIPT` / the campaign dir follow the source tree: scc `~/wtune` is a stale
  checkout (G0's first jobs 1369/1370 ran it and were cancelled).
- Rebuild prep is `real_only` on one node: real.exe output does not depend on the candidates'
  decomposition, and 4-node prep jobs ran one at a time on a busy cluster.
- Rebuild windows ran WRF's default decomposition when the case namelist had no
  `nproc_x`/`nproc_y`/`numtiles` (conus25: +21% s/step, which alone pushed a prediction to +5.9%);
  radiation steps in restart windows are located by global step; each point fetches its GRIB into
  its own plan dir (every GFS-chain point had dangling links); each segment's watchdog is sized
  from the case's cost (a fixed 15 min killed all maria1km segments); the prediction's range text
  no longer claims an I/O bracket when I/O is excluded.
- Fictional-date scenarios: AIFS runs at the analog date (ARCO-ERA5 returns all-NaN for future
  dates) and LANDSEA follows the soil source (fractional LANDSEA put −23 K soil on land; WRF went
  NaN while printing SUCCESS).

### Known limitations and PENDING

- `tune --rank-by predicted` stays **opt-in** (default `t0`). End-to-end on conus25: G1 and G2
  passed (winner's whole-run prediction +0.75% vs a same-day full run); G3 failed because the
  predicted-mode search pool was narrow on a busy cluster (the old tuner's G0014 is 10.6% faster),
  and within one pool the whole-run ranking chose the same config as the t0 ranking. Recommended:
  `tune` (t0) then `predict-rebuild` on the winner. `--minutes` in stage1=t0 is the search budget
  only; stage 2 is announced and run separately (`--stage2-minutes` caps it).
- `wtune run` R2 (segment wall inside its predicted range) passed on the rerun with an io-block
  prediction: +2.0% / −5.3% per segment. The first attempt overestimated 2–3× on constant I/O
  fallbacks, now scaled from measured anchors.
- elevconv2022 out-of-sample: `--time` 1.25× the wall and I/O probe −3.8% passed; `watch
  --prediction` failed (−13.6% at 25%) because rebuild underestimates the convective peak (~15% at
  07 h) — a known limit for strong-convection cases. Short window vs old window −1.14%.
- `wtune run --tune-report` switching and `--cfl-rescue` are now validated on hardware (dorian2019);
  decompositions are not bitwise on this build, and a mid-run switch adds no error beyond that.
- predict-rebuild windows: `nwp_diagnostics`-safe history off, out-of-group namelist keys commented,
  iofields files linked into every window, WRF stopped 30 s after FATAL or 120 s after an init
  'Problem opening' hang. Running metgrid on the whole allocation was 4.7× slower and was reverted.
- `watch` rate-only misses a rising per-step cost (elevconv −15.6% at 25%); it reads rank 0's rsl
  only.
- A "small patch hangs at init" preflight rule was not added: refuted by 102 successful runs with
  ≤ 10-cell patches; only sc26p2 17×13 hangs.
- For a run under ~1 h, `predict-rebuild` costs a large share of just running it (conus25: 51%).

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
