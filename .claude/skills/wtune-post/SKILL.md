---
name: wtune-post
description: Review and fix WTune's figures before and after a WRF run -- the `wtune post preview` figures and their figure plan, then every `wtune post recipe` figure, the case's own observation-vs-model figures as analysis/*.py scripts, and a layered, cited assessment that `wtune post check` accepts. Use when the user asks for a preview, post-processing, figures, verification or an assessment of a WRF run in a WTune project.
---

# wtune-post

WTune draws the standard figures and computes the numbers. You judge the figures, add the
figures this case needs, and write the assessment. WTune checks your files with a program.
You never change WRF output or the namelist. The figure checklist and the fix actions are in
`AGENT.md` at the repository root; read it first.

## Before the run: the preview

1. `wtune post preview`, then `wtune post brief post/preview`; read `post/preview/brief.json`.
2. Review every preview figure against the checklist in AGENT.md, and check the domains (the
   target area well inside the innermost domain, the storm inside d01), the initial state and
   the observation coverage.
3. Write `post/preview/review.json` with `figures` (verdicts) and `plan` (each figure the case
   needs after the run: `figure`, `why`, `how`, `action`).
4. Run every `project_set` action now (`wtune project set --post-focus ...`).
5. `wtune post check post/preview` until it exits 0.

## After the run

1. `wtune post recipe` (writes `post/<id>/` with `figures.json` and `report.html`).
   Over Taiwan set `post.focus: [lat_min, lat_max, lon_min, lon_max]` in project.yaml
   first, so the point figures sample the study area instead of the model's wettest corner.
2. `wtune post brief post/<id>` and read `post/<id>/brief.json` once. It has the case types,
   the runs, which observations passed their checks, every drawn figure with the question it
   answers, the missing ones with the reason, and key numbers (rain core and peak hour,
   lowest sea-level pressure). Quote those numbers; do not read values off a colour bar.
3. Review every drawn figure. Open the PNG; write `post/<id>/review.json`:

   ```json
   {"schema": "wtune-post-review/1",
    "figures": [{"path": "precip/rain_acc_d03_....png", "verdict": "fix",
                 "issues": ["cross-section on the domain edge, not over the rain"],
                 "action": {"kind": "project_set",
                            "args": ["--post-focus", "21.8,25.4,119.9,122.1"]}}]}
   ```

   A `fix` needs an `action`: `project_set` (a drawing flag of `wtune project set`; then rerun
   the recipe and review again), `analysis` (the analysis/*.py that redraws it) or
   `wtune_defect` (WTune draws it wrong and no option fixes it; tell the user).

   Look for: unreadable text or overlapping labels; a colour scale that saturates or is in
   the wrong units; an extent cut at the domain edge; a valid time after the event (F072
   when the rain ended at F048); a cross-section, sounding or box placed away from the
   phenomenon; barbs too dense or under the ground; letters (H/L) over mountains. `fix` and
   `drop` need issues.
4. Write the case's own figures as `<project>/analysis/*.py`, each with `run(ctx)`, and run
   `wtune post custom`. Use `ctx.kit` instead of rewriting:
   - `ctx.kit.obs_model_panels(obs, model, levels, title)`: observation and model side by
     side on one scale (CWA's over Taiwan, WPC's elsewhere).
   - `ctx.kit.box_mean`, `ctx.kit.match_points` (gauges to nearest cell),
     `ctx.kit.scores_by_threshold(obs, fcst, [50, 75, 90, 95], normalized=True)` next to
     the fixed-mm thresholds, `ctx.kit.hovmoller`, `ctx.kit.track_error`.
   - `ctx.observations("rain_gauges")` and `ctx.obs_module.read_gauges(...)` for checked
     data; `ctx.rain_focus(run_dir, domain)` for the core and peak hour.
   - Every number you will cite goes into a table with `ctx.save_table(...)`.

   Which figures (pick what the case needs, not all):

   | case | figures |
   |---|---|
   | heavy rain, Meiyu, SW flow | obs/model accumulated rain side by side; hourly rain box mean over the target area, obs and model; ETS and bias vs threshold, fixed and percentile; 850 hPa wind and moisture flux at the peak; Hovmoller of rain along the front-normal or the coast |
   | typhoon | track vs best track with error per 6 h; min SLP and max wind vs best track; rain obs/model; wind field at landfall |
   | convection | radar vs model reflectivity at the peak hour; timing of first echo; CAPE before onset |

5. Write `post/<id>/assessment.md` with these sections, in this order:
   setup (domains, physics, spin-up: is the configuration reasonable for the question) →
   large scale (does the model have the synoptic pattern; ERA5 difference) →
   target variables (where, when, how much, against observations) →
   scores (the table, fixed and normalized thresholds) →
   mechanism (why the model got it right or wrong, from model fields) →
   conclusion and limits (what the run can and cannot be used for).
   Rules: every paragraph links at least one figure or CSV as `[text](relative/path)`; a
   decimal number in a paragraph that links a CSV must be in that CSV at the precision you
   write; say "not verified" when no observation covers a claim.
6. `wtune post check post/<id>`. Fix every `error:` and run it again. Stop after 3 tries and
   report the remaining errors.
7. Reply to the user in their language: the 3 main findings, the figures to look at first,
   and any WTune figure defect you found.
