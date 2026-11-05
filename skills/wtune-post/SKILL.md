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

   Drawing flags a `project_set` action may use (replace the whole value; `none` clears):
   `--post-focus`, `--cross-section`, `--post-types`, `--post-every`, plus
   - `--post-times 2019-01-30T09,2019-01-30T12:30` (UTC): draw the time series at exactly these
     times, e.g. the hour of the strongest downslope wind, instead of every N hours. The FIRST
     entry also sets the time of the one-time maps (terrain, convection, cloud, synoptic,
     moisture PW/IVT, soundings), snapped to the nearest wrfout frame of the finest domain;
     without `post.times` those maps use the rain peak, and when the heaviest post-spin-up
     interval averages less than 0.1 mm/h (focus.MIN_PEAK_RATE_MM_H) they use the last frame and
     the report carries a note saying so. Accepted spellings: `2022-04-05T06`,
     `2022-04-05T06:00`, `2022-04-05_06:00:00` (WRF), `2022-04-05 06:00`; stored as
     `YYYY-MM-DDTHH[:MM]`. Precedence: `wtune post recipe --times` > `post.times` > `--every`. A
     time outside the wrfout range is an error that lists the available range.
   - `--post-map-domains d01,d03` (or `all`): shade terrain for those nests in the domain map
     and draw one zoomed terrain map per nest (`common_domain_map_zoom`,
     `common_domain_map_d03.png`). Default is d01 only. Sea (LANDMASK = 0, else HGT <= 0.5 m)
     is light blue, not the white of the 0 m end of the terrain scale.
   - `--post-overlay wind10`: 10 m wind barbs on those zoomed maps (at the first `post.times`
     entry, else the rain-peak time) and on the time-series `t2` and `wspd10` maps. Other
     figures are not affected.
   - `--post-figures-off ID,ID`: drop figures by catalog id from `figures.json`, the report and
     the folder. `--post-figures-on` only validates ids (nothing is off by default yet).
   - `--post-stations ID,ID --post-station-vars t2,wspd10,wdir10,gust
     --post-station-method nearest|bilinear --post-calm-kt 3`: with `post.stations` set,
     `wtune post recipe` also runs the station comparison (first `input/obs/` dataset of kind
     `stations` that passes obs-check; without one, a note) and registers the per-station
     figures; `wtune post stations --project P` reads the same keys for flags it is not given.
   A `project set` call with any bad flag changes nothing and the error names every bad flag
   with its value and an accepted example.

   Example: the downslope wind peaks at 09 UTC, d03 shows no terrain, `timeseries_mcape` is noise.
   ```bash
   wtune project set --post-times 2019-01-30T09 --post-map-domains d01,d03 \
       --post-overlay wind10 --post-figures-off timeseries_mcape
   wtune post recipe          # redraws into a new post/<id>/
   wtune post brief post/<id> # then review the new figures and write review.json
   wtune post check post/<id> # project_set args are parsed with the same parser
   ```
   A malformed value or an unknown figure id is rejected at `project set` and by `post check`
   with the accepted form. These flags only choose times, nests and figure switches; a new
   kind of figure still needs `analysis/*.py`.

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

## Analyse all of the data, not only what the task names

A task statement (a SPEC, an exam question) lists the least that is wanted. The analysis covers
everything the run and the observations contain:

- every station in the observation folder, every variable it has, every domain that contains the
  station (`wtune post stations` defaults to all of them; naming a subset makes the tool print
  "NOT analysed (partial analysis)");
- every output time of the run, not only the hours the task mentions; look for the strongest
  event in the whole series and plot that instant (`--post-times`);
- every domain, including the ones the task does not score, when they change the story (nest
  feedback, terrain resolution, where a front enters);
- what could not be compared (a variable with no model counterpart, a station outside every
  domain, observations missing for a stretch) is written down with the reason.

- gust is a reports-only sample: ASOS fills the gust column only when a gust occurred (`M` means
  no gust, not a missing value), so a gust bias/RMSE (e.g. SBA n=4, bias -8.03) describes
  gusty hours, not all hours. `station_stats.csv` marks those rows in its `note` column
  ("obs only when gusts occur; model gust source = WSPD10MAX|estimated"), the gust panel title
  says "(gust reports only)", and the compare notes carry n. Quote n and the caveat; do not
  call it gust skill.

`wtune post check` warns when `station_stats.csv` has a station the assessment never names. A
section that quotes only the requested stations or hours is a partial analysis; say what was left
out and why, or analyse it.

## Reviewing many files with a fast model

A recipe draws 200-300 figures. Screening them one by one is bulk work, so give it to the fast
model of the same harness (Claude Code: a Sonnet or Haiku subagent; Codex: `gpt-6-luna` at low
effort), in batches of about 20 figures, with the checklist below and the figure's caption from
`figures.json`:

1. Is the map extent, colour scale and units right for the variable named in the caption?
2. Is anything cut off, overlapped (labels over a domain box), empty, or all one colour?
3. Is the field physically plausible at this time (spin-up frame, wind direction against the
   terrain, rain where the terrain forces it)?
4. Does it answer the question in its caption, and does the number in its title match
   `brief.json`?

Each batch returns one row per figure: `path`, `verdict` (ok | fix | drop), `issues`. The lead
agent then opens every figure marked fix or drop, a sample of about 10 % of those marked ok, and
the figures that carry the conclusions; the lead writes `review.json` and the assessment itself.
The fast model screens; it never signs off a figure the lead did not look at when the
conclusion rests on it. `wtune post check` stays the programmatic gate.
