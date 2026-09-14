# WTune configuration predictor -- benchmark summary

Every metric below is computed PER DECISION GROUP (case_base, hardware_id, compliance_class, n_nodes, total_cores, binary_kind, io_form_history) -- configs are only ranked/regretted/paired against alternatives that hold the resource allocation fixed. See predictor/features.py `decision_group_key`.


## Scope: STRICT

- decision-group data notes: 0 row(s) excluded from ranking (unknown n_nodes/total_cores); io_form_history splits 6 group(s) that would otherwise merge different I/O backends.
- **best model (leave-one-case-out median regret): `catboost_point`**
- n_eval_cases (decision groups, >=2 configs) = 9
- top-3 hit rate = 0.375 (eligible groups with >3 configs: n=8)
- median regret = 0.010660483243594587, worst regret = 0.7954303640424778
- naive baseline (always the group's default config) median regret = 0.128816392274834; regret improvement over naive = 0.1181559090312394
- empirical conformal coverage: 95% nominal -> 0.996 (n=265), 90% nominal -> 0.996 (n=265)
- top features (permutation importance, |MAE delta|): nproc_x=0.0119, numtiles=0.0118, nproc_y=0.0086, e_sn=0.0065, omp_threads=0.0064
- profile (Level-1) ablation: level0 median_regret=0.010660483243594587 -> level1 median_regret=0.01480071675632555

## Scope: PHYSICS-GRAY

- decision-group data notes: 0 row(s) excluded from ranking (unknown n_nodes/total_cores); io_form_history splits 4 group(s) that would otherwise merge different I/O backends.
- **no model selected (underpowered)**: every model's leave-one-case-out result had <5 eligible decision groups -- see metrics.csv `n_eval_cases`/`status` columns.

- UNDERPOWERED (model, scheme) pairs: catboost_pairwise/leave-one-case-out, catboost_pairwise/leave-one-family-out, catboost_pairwise/leave-one-run-out, catboost_point/leave-one-case-out, catboost_point/leave-one-family-out, catboost_point/leave-one-run-out, elasticnet/leave-one-case-out, elasticnet/leave-one-family-out, elasticnet/leave-one-run-out, heuristic/leave-one-case-out, heuristic/leave-one-family-out, heuristic/leave-one-run-out, knn/leave-one-case-out, knn/leave-one-family-out, knn/leave-one-run-out, ridge/leave-one-case-out, ridge/leave-one-family-out, ridge/leave-one-run-out

## Exact n per metric

See metrics.csv `n_eval_cases`/`n_eval_groups_top3` columns per (scope, model, scheme) row; per_case.csv has one row per held-out DECISION GROUP actually evaluated (n_nodes/total_cores columns distinguish groups sharing a case_base).

