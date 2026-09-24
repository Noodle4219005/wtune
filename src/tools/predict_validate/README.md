# predict_validate — pre-registered validation harness for `wtune predict`

Used 2026-09-23 on scc. Workflow:

1. `stage.py` / `stage_heldout.py` — stage full-length ground-truth run dirs from a clean campaign probe run dir;
   namelist = case template with only the decomposition, io_form 2 and restart_interval=run/4 changed.
2. Run `wtune.py predict` on the staged namelist and write the numbers to a CSV **before** any truth run.
3. `truth.sbatch` (1 node) / `truth_n.sbatch` (N nodes) — the full run; wall = mpirun launch → exit.
4. `window_stage.py` + `windows.sbatch` — K=4 sampling windows (t0 cold start + restarts at 25/50/75%),
   all in one job, pinned (`-w`) to the truth run's node so K=1 vs K=4 differ only in sample count.
5. `score.py` — whole-run wall error and normal-step compute error for K1 / K4.

Results and the pre-registration record: wiki
`knowledge/experiments/2026-09-23-wtune-predict-validation-in-sample-and-held-out.md`.
Paths inside the scripts assume scc's `~/wtune-runs/predict-validate/`.
