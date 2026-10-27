# AGENT.md -- what an agent does in WTune

WTune measures, runs WRF, computes the numbers, draws the standard figures and checks files
with programs. You do what needs judgement: read the user's description, find data WTune cannot
fetch, decide which figures the case needs and how they must look, review every figure WTune
draws (the preview before the run too), fix what is wrong, and write the assessment.

Never edit WRF source, the physics of the namelist, or WRF output. Never write a number you did
not get from a WTune command or a CSV a script wrote. Every stage ends with a WTune check; a
stage is done only when its check exits 0.

## Stages

| stage | WTune command | your job | your output | WTune check |
|---|---|---|---|---|
| 1 event | `wtune event init event.txt` | turn the user's words into event.yaml (skill `wtune-event`, part 1) | `event.yaml` | `wtune event check event.yaml` |
| 2 data | `wtune event plan event.yaml` | fetch every `agent_fetch` row: rain, radar, satellite, gusts, non-NHC best tracks (skill `wtune-event`, part 2) | `<project>/input/obs/<name>/` in `wtune-obs/1`; a best track as an ATCF b-deck | `wtune event obs-check` |
| 3 preview | `wtune post preview`, `wtune post brief post/preview` | review the preview figures; write the figure plan for after the run; apply the drawing options | `post/preview/review.json` with `plan` | `wtune post check post/preview` |
| 4 post | `wtune post recipe`, `wtune post brief post/<id>` | review every figure; fix; draw the case's own figures; write the assessment (skill `wtune-post`) | `review.json`, `analysis/*.py`, `assessment.md` | `wtune post check post/<id>` |
| 5 iterate | rerun the command of the stage you changed | redraw, review again | the same files | the same check |

Stop after 3 failed checks of one stage and report the remaining errors to the user.

## Stage 3: the preview (before WRF runs)

The preview shows what is known before the run: the domains and terrain, the initial state, the
best track and the observations of the window. It is the cheapest place to catch a figure that
will be wrong after a 30-minute run.

Check, and write each problem into `review.json`:

1. **Domains.** The target area of the case (event.yaml `region`, the user's text, the paper
   the case reproduces) lies inside the innermost domain with at least ~10 cells to spare on
   every side; a storm's track stays inside d01 (and inside the nest that must resolve it).
2. **Initial state.** The storm, front or low the case is about is in the t=0 fields, near the
   best-track position at the start time.
3. **Observations.** Every dataset of stage 2 covers the window and the area; the gauges are
   where the rain is supposed to be.
4. **Expected figures** (`brief.json` `figures.after_the_run`). Each answers a question of the
   case; a type that does not belong (`--post-types`) or a missing comparison goes in the plan.

Then write the **plan**: the figures the case needs after the run, each with what to draw and
how, and the action that makes it:

```json
{"schema": "wtune-post-review/1",
 "figures": [{"path": "figures/common_domain_map.png", "verdict": "ok", "issues": []}],
 "plan": [
   {"figure": "point figures over southern Taiwan",
    "why": "the heavy rain of the case is over SW Taiwan, not the model's wettest corner",
    "how": "cross-section, time-height and skew-T at the rain core inside the box",
    "action": {"kind": "project_set", "args": ["--post-focus", "21.8,25.4,119.9,122.1"]}},
   {"figure": "12 h rain, CWA gauges vs model",
    "why": "paper Fig. 4: where and how much it rained in each 12 h",
    "how": "4 panels 07-31 00Z..08-02 00Z, gauges as dots and model shading on CWA scale",
    "action": {"kind": "analysis", "script": "analysis/rain12h_taiwan.py"}}]}
```

Run every `project_set` action now (`wtune project set ...`); write the `analysis` scripts now
or after the run. `wtune post check post/preview` checks the review, the plan, and that each
`project_set` uses a valid drawing flag.

## What a correct figure looks like (the review checklist)

Open every PNG. A figure is `ok` only if all of these hold:

- **Readable.** Text at least ~7 pt at the saved size, nothing overlaps (labels, legend, colour
  bar, caption), no large blank margin. The title says the variable, units, domain and valid
  time (init + lead) in at most two lines.
- **Colour scale.** Amounts on a sequential scale, differences on a diverging scale centred on
  0. Rain over Taiwan on CWA's 16 levels, elsewhere WPC's. The top bin is not where most of the
  signal sits (a saturated scale hides the maximum). Observation and model panels share one
  scale. Units on the colour bar.
- **Extent.** The domain or the study area, nothing important cut at the edge, no map running
  off to the poles.
- **Time.** The time of the phenomenon (rain peak, landfall, strongest wind), not the last
  frame by default. An accumulation says its window. Spin-up hours are excluded or marked.
- **Where it samples.** A cross-section crosses the feature (normal to the front, the coast or
  the ridge; through the rain core or the storm centre), a sounding is where it says it is
  (inflow or rain core), a box is the one of the case description.
- **Vectors.** About 15 barbs along the short side or fewer, two sets never on the same points,
  none where the level is under the ground.
- **Pressure.** Isobars distinguishable from coastlines; no H/L along the boundary or over
  terrain above ~500 m, where sea-level pressure is a reduction.
- **Observation comparison.** Same grid or matched points, the source and time window in the
  caption, and the comparison the case asks for (amount, timing, position).

## How to fix a figure

Every `fix` in `review.json` carries one `action`, in this order of preference:

| action | when | example |
|---|---|---|
| `project_set` | a WTune option draws it right: study area, cross-section line, figure types, time step of the series | `{"kind": "project_set", "args": ["--cross-section", "moisture=22.0,119.5,23.5,121.5"]}` |
| `analysis` | the case needs a figure WTune does not draw, or draws differently from the paper | `{"kind": "analysis", "script": "analysis/rain_box_timeseries.py"}` |
| `wtune_defect` | WTune draws it wrong and no option fixes it (a colour scale, a crash, a layout bug) | `{"kind": "wtune_defect", "text": "rain_acc colour bar tops at 203.2 mm over Taiwan"}` |

Flags `project_set` may use: `--post-focus`, `--cross-section`, `--post-types`, `--post-every`.
After a `project_set` fix, rerun the stage's command and review the new figure. Report every
`wtune_defect` to the user; do not work around it silently.

## Writing an analysis script

`<project>/analysis/<name>.py` defines `run(ctx)`; `wtune post custom` runs all of them. Use
`ctx.kit` (post/analysis_kit.py) instead of rewriting: `obs_model_panels` (side by side, one
scale), `box_mean`, `match_points`, `scores_by_threshold` (fixed and `normalized=True`
percentile thresholds), `hovmoller`, `track_error`; `ctx.observations(kind)` for checked data;
`ctx.rain_focus(run_dir, domain)` for the rain core and peak hour. Save every number you will
cite with `ctx.save_table(...)`. Hold your figures to the checklist above.

## Figure plans by case type

| case | figures |
|---|---|
| heavy rain, Meiyu, SW flow | obs/model accumulated rain side by side (12 h and event total); hourly rain box mean over the target area, obs and model; ETS and bias vs threshold, fixed and percentile; low-level wind and moisture flux at the peak; Hovmöller of rain along the front-normal or the coast |
| typhoon | track vs best track with the 6-hourly error; minimum SLP and maximum wind vs best track; rain obs/model; wind and rain at landfall; radius-height structure |
| convection | radar vs model reflectivity at the peak hour; time of first echo; CAPE/CIN before onset; updraft helicity swaths |
| terrain flow | wind across the ridge (cross-section normal to it); rain vs terrain height; valley wind time-height |

## The assessment (stage 4)

`assessment.md`, in this order: setup, large scale, target variables (where, when, how much,
against observations), scores, mechanism, conclusion and limits. Every paragraph links at least
one figure or CSV as `[text](relative/path)`; a decimal next to a CSV link must be in that CSV
at the precision written; say "not verified" when no observation covers a claim. Details:
`.claude/skills/wtune-post/SKILL.md`.

## Connecting an agent or harness

WTune has no SDK and no server. The interface is the shell, files and exit codes, so any agent
that can do three things can drive it:

1. run shell commands on the cluster's login node (internet there for fetching data; WTune
   sends the heavy work to Slurm itself);
2. read and write files in the project;
3. look at PNG images (for the reviews). Without image input, send each PNG with the checklist
   above to a vision model and write its answer into `review.json`.

Setup, once per project:

| harness | how it gets these instructions |
|---|---|
| any harness that reads `AGENTS.md` (Codex, opencode, ...) | start it in the project directory; `wtune project init` writes `<project>/AGENTS.md` pointing here (`wtune project set --agents-md` for an older project) |
| Claude Code | the same `AGENTS.md`, plus the skills: copy or link `<wtune>/.claude/skills/wtune-*` into `<project>/.claude/skills/` |
| your own loop or a plain chat model | put this file and the stage's `SKILL.md` in the system prompt |

The loop for every stage:

1. Run the stage's WTune command.
2. Read the JSON WTune wrote (`brief.json`, `figures.json`, `event.yaml` template).
3. Write your file (`event.yaml`, `review.json`, `analysis/*.py`, `assessment.md`).
4. Run the check. Exit 0: the stage is done. Otherwise read the `error:` lines, fix, check again
   (at most 3 times).

Long commands (`wtune prep wps`, `wtune tune`, `wtune run`, `wtune post ... --srun`) take
minutes to hours. Start them in the background with their output in a log file (`nohup wtune
run ... > run.log 2>&1 &`), and poll instead of waiting in one call: `wtune watch <run dir>
--once --json` for a run, the log for the others. Stop one with `wtune stop` (it cancels its
own Slurm jobs), never with `kill` or `scancel` on jobs you did not start.

Files with a schema:

| file | schema | written by |
|---|---|---|
| `event.yaml` | `wtune-event/1` | agent, checked by `wtune event check` |
| `input/obs/<name>/provenance.json` | `wtune-obs/1` | agent, checked by `wtune event obs-check` |
| `post/<id>/figures.json` | `wtune-post-figures/1` (`kind` recipe or preview) | WTune |
| `post/<id>/brief.json` | `wtune-post-brief/1` | WTune |
| `post/<id>/review.json` | `wtune-post-review/1` | agent, checked by `wtune post check` |
| `post/<id>/assessment.md` | six sections, cited | agent, checked by `wtune post check` |

## Files

- `.claude/skills/wtune-event/SKILL.md` -- stages 1-2.
- `.claude/skills/wtune-post/SKILL.md` -- stages 3-4, step by step.
- `docs/tutorial/post.md` -- the commands.
