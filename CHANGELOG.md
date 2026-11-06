# Changelog

Versions are declared in `wtune_version.py`; `test_wtune_version.py` fails if the top section
here does not match it. Development before 0.1.0 is in the git history, not in releases.

## Unreleased

Reconciliation of the old open-defect list (2026-11-06).

- `wtune predict` no longer stops on a dangling `wrfbdy_d01*.bp` symlink in the case folder, and sizes an ADIOS2 `.bp` directory by its contents.
- `timeseries_pw` has a caption entry (a test now requires one for every time-series field).
- `wtune doctor` warns when `WTUNE_CARTOPY_DATA` is a folder without a coastline shapefile.
- Node preflight asks an idle node a second time before it counts a probe timeout against it.
- When the WRF-Chem j-patch guard rejects every candidate, the layout with the widest j-patch is kept (with a warning) instead of an empty search.

Fixes from validating SCC26 Sundowner Run 1 (three domains, Santa Barbara) with `wtune post`.

- `post.times` now also picks the one-time maps (terrain, convection, cloud, synoptic, moisture, soundings): the first entry, snapped to the nearest frame. Without it, a rain peak under 0.1 mm/h counts as no rain and those maps use the last frame, with a note in the report. The convection maps previously looked for their own rain peak and ignored the recipe's map time.
- `--post-times` accepts WRF and space spellings (`2022-04-05_06:00:00`, `2022-04-05 06:00`, `2022-04-05T06`). `project set` still rejects a bad flag atomically; the error now lists every bad flag with its value and example formats, and drawing flags are validated before `--register-case` writes anything.
- `--post-overlay wind10` also draws 10 m barbs on the time-series `t2` and `wspd10` maps.
- `wtune post recipe --out <relative path>`: `figures.json` paths are relative to the post directory (they were relative to the cwd, so `post check` could not find the figures).
- Station comparison: gust statistics carry a sampling-bias note (ASOS reports gusts only when they occur): `note` column in `station_stats.csv`, a warning in the notes with n, "(gust reports only)" in the panel title, and the catalog caption.
- Figures: T2 maps use a sequential scale unless the data crosses 0 degC (was a -24..+24 diverging scale for 0-25 degC data); sea is light blue on the domain maps; small domains get at least 3 labelled lat/lon ticks (°N/°W). The zoomed domain map drew the d01 extent (a duplicated block in `domain_map.draw` redrew the figure and dropped `zoom` and the nest terrain); fixed.
- `post.stations` makes `wtune post recipe` run the station comparison (first usable `stations` dataset under `input/obs/`; a note when there is none). `wtune post stations --project P` reads `post.stations/station_vars/station_method/calm_kt` for flags not given.

- `wtune post stations`: per-station model vs. observation series, bias/RMSE (circular wind direction, calm winds excluded) from a new `stations` obs kind; `wtune event obs-convert asos` turns an IEM ASOS CSV into one.

Fixes from the 2026-10-26 mock competition on f1-arm (haitang2017, MeiYu-2020), and the agent
half of `wtune post` (decision 0078).

**Agent-set drawing choices.** `wtune project set --post-times/--post-map-domains/--post-figures-off/--post-figures-on/--post-overlay` (and the station keys `--post-stations/--post-station-vars/--post-station-method/--post-calm-kt`, stored only) pick the times, the nests with terrain and zoomed maps, and figure switches for `wtune post recipe`; `post check` accepts them in a `project_set` action. Domain labels now sit outside their boxes.

**`wtune post` with an agent.** `wtune post brief POST_DIR` writes `brief.json` (case, runs,
the observations that pass obs-check, drawn figures with the question each answers, missing
ones with the reason, the rain core and peak hour, the lowest sea-level pressure). The agent
(skill `wtune-post`) writes `review.json`, its own `analysis/*.py` figures, and
`assessment.md`; `wtune post check POST_DIR` fails on a missing section, a paragraph that
cites no existing figure or CSV, or a decimal beside a CSV link that the CSV does not hold.
The preview goes through the same loop: `post/preview/figures.json`, `wtune post brief` and
`wtune post check post/preview`, whose review also carries the figure plan for after the run.
Every `fix` names how: `project_set` (new `wtune project set --post-focus` and
`--cross-section`), an `analysis/*.py` script, or a `wtune_defect` report. `AGENT.md` is the
agent's guide to all of it, including how to connect any agent or harness; projects get an
`AGENTS.md` pointing to it (`wtune project set --agents-md` for older ones).
Skill `wtune-operate` covers the rest for an agent: installing, the project, inputs (`prep`,
`metdata`, `scenario`, `convert`), checks and advice (`check`, `predict`, `advise`, `calc`,
`physics`, `compliance`), `tune`/`run`/`submit`/`watch`/`stop`, and the single `post` tools;
`AGENT.md` and the project `AGENTS.md` point to it.
The skills moved from `.claude/skills/` to `skills/` (`.claude/skills/` keeps links for
Claude Code), and `AGENT.md` says how any harness, a plain chat model or a person loads them.
`wtune doctor` checks every path a job uses (site file, partition, the env file and whether its
`WRF_TREE` matches machine.yaml's `wrf_tree`, the libraries `wrf.exe` cannot load after sourcing
it, WPS, geog, the site's adios2 python, Natural Earth), each with its fix; `--json` prints the
checks for an agent and `--find` searches the disk for WRF trees, WPS, geog data and the missing
libraries, so an agent can set the paths up itself (skill `wtune-operate`, "Paths"). `f1.sh`
now takes `WTUNE_PIXI_ENV` like the other sites.
Scripts get `ctx.kit`: CWA rain scale, contingency scores with fixed or percentile thresholds,
box means, nearest points, Hovmöller, track error, observation/model panels. `wtune post
recipe` now also writes `figures.json`.

**Standard figures.**
- The domain map is drawn in the model projection, one colour and a `dx`/size label per
  domain, cropped to d01; a d01 across the date line no longer draws a ring round the globe.
- Rain over Taiwan uses CWA's 16-level scale (the WPC one stops at 203.2 mm).
- SLP charts: grey coastlines under the black isobars; no H/L within half a window of the
  edge or 2 cells of terrain above 500 m.
- Cross-section, time-height, skew-T and max-reflectivity map: at the heaviest post-spin-up
  rain (30 km mean, off the boundary zone) and its peak hour, not the last frame's wettest
  cell at F072. `post.focus: [lat_min, lat_max, lon_min, lon_max]` in project.yaml limits the
  search to the study area.
- Wind barbs: ~15 along the short side, the 700 hPa set between the 10 m one, none where the
  level is underground.
- Synoptic charts, PW, IVT, MCAPE/MCIN, the terrain wind map and cross-section, and the cloud
  maps: at the same rain peak (finest domain), not at the last frame.
- A d01 across 180E: filled maps no longer leave white wedges, and `post fetch --run-dir`
  takes the larger side of the date line instead of lon -180..180 (stations were picked in
  North Africa for haitang2017).
- TC structure figures: at the last frame the track covers. Past its end they had centred on
  the deepest low anywhere in the domain (haitang2017 F072: a low over Guangxi).
- The valley time-height point and the cloud section's max-TWP column are taken inside
  `post.focus` when one qualifies there (haitang2017 picked both in Fujian).
- Gridded-rain comparisons with an agent dataset (CMORPH): the window is centred on the rain
  peak instead of the run's last hours, Taiwan gets CWA's scale, and the figures name the
  dataset briefly (its full title overlapped the panel titles). The 6-h rain change maps use
  CWA's scale over Taiwan too.
- `post fetch igra` takes the run's finest domain by default (the one the pbl figures use),
  not d01; `post fetch stations` removes the files of stations a refetch no longer selects.
- `ctx.find_run_track_csv` finds `tc/track_domains_<run_id>.csv`, the file post/tc.py writes.

**TC tracking on a Lambert grid.** TempestExtremes' `GenerateConnectivityFile --in_data`
writes every point with no neighbours, so DetectNodes took every cell as a local minimum (a
d03 frame took 186 s) and the 2-degree maximum 10 m wind was the centre cell's own (haitang2017
vmax 1-4 m/s; 21.8 m/s within 2 degrees). WTune now writes the connectivity of the WRF grid's
edge neighbours itself.

**Probe crashes on f1-arm.** 30 of 40 haitang2017 tune probes died on SIGSEGV about 5 s in, so
the node sweep had no measurement and tune fell back to its uncalibrated proxy. The cause is
WRF's Goddard 4-ice microphysics (`mp_physics = 7`): it writes `refl_10cm` over the whole tile,
but that array is allocated only when `do_radar_ref = 1`, so with the default 0 it overruns the
heap. WTune now sets `do_radar_ref = 1` whenever `mp_physics` has a 7 (tune probes, `wtune run`,
`case_rebuild`); history gains `REFL_10CM` and `REFD_MAX`, the other fields are bit-identical.
The crash was first blamed on Open MPI's shared-memory BTL, so the f1-arm site still runs with
`btl=self,tcp` (~3 % more s/step on one node) until btl/sm is measured again; a probe that dies
on signal 11 is run once more; `WTUNE_KEEP_ALL_RSL=1` keeps every rank's rsl files.

**Tune measures one start point, then predicts with rebuilt points.** By default `wtune tune` no
longer rebuilds prediction points before the layout search: phase 1 compares layouts on the t0
window alone, the winner's time decides how many points the precision needs, and phase 2
rebuilds only those points at the winner, `wtune predict-rebuild`'s way (the case's own forcing,
60 min spin-up, 40 steps), in waves within `max_nodes`. `--physics-advice` keeps the earlier
flow, whose shared spun-up restarts the physics candidates need.

**Whole-run prediction for short runs.** `wtune tune --precision` defaults to +-10 % of the run
(it was a fixed +-15 min, which on a 30 min run accepted any estimate: haitang2017 got no
prediction point and ran 38 % over). The tc bands follow that run's own step times (t0 only
30 %, base set 15 %), each rebuilt point measures 40 steps, and tune now sees a project's
`storm.bdeck`, so a JMA-tracked typhoon is planned as tc. `wtune post fetch` finds the project it
runs in.

**Observations behind every claim (haitang2017 review).** The comparisons an assessment needs
are WTune commands, not one-off scripts:
- `wtune post fetch era5` fetches z500 and 850 hPa wind and specific humidity at the recipe's
  rain peak and its last post-spin-up frame (`--times` adds more); it fetched one z500 time,
  and after the one-time maps moved to the rain peak `syn_era5_diff` found no subset. New
  `syn_era5_850_diff`: 850 hPa wind speed and humidity, model - ERA5, MET Grid-Stat RMSE and
  bias (the low-level jet and its moisture).
- `wtune post fetch gridsat`: NOAA GridSat-B1 IR brightness temperature (0.07 deg, 3-hourly,
  70S-70N) written as an agent `satellite_ir` dataset, so `cloud_ir_compare/scores/fss` work
  for any region without a conversion step.
- `common_sounding_obs`: WRF against IGRA radiosondes for every case type with IGRA data (the
  pbl set, LES-type runs only, was the only one), each station on the finest domain holding
  it, the soundings nearest the rain peak. `fetch igra` takes the stations nearest
  `post.focus` when the project has one.
- `precip_hovmoller`: rain against distance and time across the band (default line through the
  rain core across the 0-1.5 km wind; `--cross-section hovmoller=...`), each agent gridded
  rain dataset beside the model over the same windows, plus the model's hourly rain.
- The missing-radar message names NEXRAD/MRMS as US-only and points a domain elsewhere to an
  agent `radar` dataset instead of `fetch nexrad`.

**Figure fixes from the haitang2017 review.** tc_structure uses the strongest tracked frame past
spin-up, not the last one; tc_radius_time_wind has a column only where a frame is (nearest
within 10 min); the cloud_top_height title no longer cites lake-effect thresholds; maps
(`save_map`, `plot_field`) follow the domain's aspect instead of a fixed 8x6.4; precip_stn_map
panels follow the stations' box; the Stage IV window is centred on the rain peak; gridded rain
scores over Taiwan use CWA's warning levels (1/10/40/80/200/350 mm). Across 180E: the d01 box,
TC detection, the preview Stage IV mask, coastlines past 180, PlateCarree axes and the cdo
regrid target. `doctor --find` lists only env files that set WRF_TREE; `post` help lists all
its subcommands.

**Other fixes.**
- Maps keep to the grid's own box in the map projection (a Lambert d01 reaching 7 S drew to the
  pole), have 2/5/10-degree gridlines, and draw no isobars over ground above 1500 m.
- `wtune project set --bdeck FILE`: a storm outside HURDAT2 (e.g. a JMA track) is a project
  setting the preview, recipe, tc and brief use; the preview offers Stage IV only for a grid
  with cells over the contiguous US.
- A truncated, unreadable or dangling-link history frame is left out and listed in the report
  instead of stopping the whole recipe (a frame written past a full quota gave 0 figures).
- `wtune post preview` works before `prep real`: domains from `geo_em` or `namelist.wps`.
- `wtune post tc` and the recipe's tc set accept a b-deck without `--hurdat2`.
- `wtune post fetch` without `--run-dir` uses the project's newest run.
- `prep real` fits `plan.json`'s ranks to the machine's cores; the default layout is
  near-square.
- `wtune run` pre-creates only the ADIOS2 output directories WRF will write (an off-step
  history time, e.g. d01 dt 270 s with hourly output, left empty on-hour directories behind),
  and links an `--out` outside the project into `runs/` so `wtune stop` and `wtune post` find
  it.

## 0.9.3 — 2026-10-26

**`wtune event`: describe the event in your own words.** Put an `event.txt` in the case or
project folder ("the 2021 Zhengzhou rain; check hourly and 24 h rain and the surface wind; my
CMA gauges are in ~/obs/cma_rain.csv"). An agent following the new `wtune-event` skill turns it
into `event.yaml`, quoting the sentence behind every value. `wtune event check` rejects a quote
that is not in `event.txt`, a type or variable outside `wtune event vocab`, a storm id the text
does not state, a stale `source_sha256`, a tropical cyclone with no storm and no question, and
an `idealized` case that is also compared with observations. `wtune event plan` then says, for
each comparison, whether a fetch module gets the data (`module`), the agent has to find it
(`agent_fetch`: radar and rain outside the US, gusts, satellite IR, non-NHC best tracks), it is
the user's file (`local`), WTune has no such comparison (`unsupported`), or it is a model-only
figure. 22 event types; the eval set (17 cases plus 8 held-out) and the scripts that score a
model on it are in `tests/fixtures/events*` and `campaign/event_eval/`.

**`wtune post` reads observations an agent fetched.** The agent converts what it finds to
`wtune-obs/1` in `<project>/input/obs/<name>/` (a `provenance.json` with the source, licence and
sha256 of every file); `wtune event obs-check` checks names, units, ranges, axes, UTC times and
checksums, and warns on magnitudes that usually mean a unit, scale or coverage mistake (no rain
anywhere, no radar cell outside coverage, gusts under 2 m/s). The recipes use only folders
without errors:

| kind | file | compared in |
|---|---|---|
| `gridded_precip` | `precip_YYYYMMDDHH_NNh.nc` | precip: Grid-Stat on the dataset's grid, files summed to the longest window (`precip_agent_*`) |
| `rain_gauges` | `gauges.csv` | precip: Point-Stat per period (`precip_gauge_*`) |
| `radar` | `refl_YYYYMMDDHHMM.nc` | convection: the radar Grid-Stat when there is no MRMS / NEXRAD |
| `satellite_ir` | `irtb_YYYYMMDDHHMM.nc` | cloud: model OLR-derived Tb vs IR, cold tops <= 240/220/210 K (`cloud_ir_*`) |
| `stations_extra` | `stations_extra.csv` | precip: gust and visibility Point-Stat (`stn_gust_*`, `stn_vis_*`) |

WRF writes no gust or visibility: the model gust is `WSPD10MAX` when the run has it, else NCEP
UPP's CALGUST; visibility is Stoelinga-Warner (1999) at the lowest level, model and obs capped
at 16.1 km. Rain-gauge point obs are named `APCP_<hours>`: MET 12.1 Point-Stat rejects a point
obs named `APCP` by name and never matches an `A6` level against ascii2nc's level in seconds.
Checked on f1-arm with Matthew: CMORPH, ISD 6 h totals, the IEM NEXRAD N0Q mosaic, GridSat-B1
and NCEI Global Hourly gusts and visibility.

**`post preview` writes the values it draws.** Each preview figure gets a netCDF file of the
plotted fields in `post/preview/data/`, named like the figure in `figures/` (SLP and 10 m wind,
terrain per domain, Stage IV total, ERA5 z500); the best track stays `tc/best_track.dat`.

**`prep wps` checks the static data first.** It looks for the WPS static data (`machine.geog`,
or `WPS_GEOG`) before downloading the forcing, and a download that still misses files after its
retries is an error message, not a traceback. Run on f1-arm: WPS and real.exe for a 12 h window
(matthew, `geog_data_res='lowres'`).

**Serial WPS under `mpirun`.** geogrid and metgrid were always started with `mpirun -np <ppn>`;
a serial WPS build (f1-arm's) then ran ppn independent copies that all wrote the same file, and
the job failed. They now run under `mpirun` only when the executable calls `MPI_Init`. Linking
`libmpi` is no evidence: a serial build gets it through a parallel netCDF/HDF5.

**MET on RHEL.** `bin/install-postproc.sh` failed after building netcdf-cxx4 where CMake installs
to `lib64` (f1-arm): the recipe's post-install step looked only in `lib`. MET and ParaView are now
installed on f1-arm; the guide has the steps.

**`wtune run` checks disk space.** Before the first job it estimates the history and restart
output (frames x a conservative 40 3-D fields per frame) and refuses to submit when that does not
fit the space writable at `--out`; on GPFS this is the user's quota headroom, not `df`
(`--no-space-check` overrides); the estimate includes the BP conversion of inputs not yet in
the cache (uncompressed netCDF size: elevconv's 2.0 GB became 7.9 GB) and one per-segment input
copy. After each job it compares every history frame with the largest
of its stream: past the f1-arm soft quota WRF+ADIOS2 printed `SUCCESS COMPLETE WRF` while d02
frames shrank from 1.8 GB to 17 KB, and the run was reported completed. Such a segment now fails.
The plan also warns when the case sets `debug_level` above 0: sc26p1 (`debug_level=50`) wrote
~31 MB of `rsl.out` per rank per 6 model hours, which the estimate does not count.

**A failed segment names the WRF FATAL.** The error now quotes the first `FATAL CALLED` message of
any rank; c34 ended at 02:44 when WRF-Fire stopped the model ("Fire has reached domain boundary",
rank 39 only) and the report said only "warnings none".

**`wtune run` segments get the probes' environment.** The segment job sourced the site file but
never called its `site_env`, which on f1-arm pins Open MPI's TCP to the IPoIB network. Multi-node
segments therefore talked over the 1 Gb/s management network: c34 on 5 nodes ran 1.52 s per d01
step against the 0.246 s tune measured with the same layout, hit its time limit and was resumed.

**`--compress-history` with more than one segment.** After every segment had passed, deleting
the run's own restarts removed the link a later segment started from and then tried to size it:
the run ended `failed` with `FileNotFoundError` (f1-arm severe2023, 2 segments).

**`wtune run` after `wtune tune -O0`.** `-O0` writes its unmeasured layout as `best`; `run`
looked only for a measured winner and asked for `--nodes/--nx/--ny`. It now takes the `-O0`
layout.

**One station verification per domain.** When the terrain domain is also the finest domain,
`terrain_point_stat` ran the same MET Point-Stat as `precip_stn_series` and drew the same figure
twice. It is now left out with an info note pointing to the precip figures. The
`precip_ets_fbias` caption named thresholds 1/10/25/50/100 mm; it now names the ones used,
0.254/2.54/6.35/12.7/25.4/50 mm.

**Tune and run share one BP input conversion.** In a project, `wtune run` now uses the tune's
cache `<project>/input/bp-cache/<case>/<key>/` (or `$WTUNE_BP_CACHE`, as the tune does) instead
of its own `<project>/input/bp/`; marshall held the same 3 GB conversion twice. A project whose
`input/bp/` already has a conversion keeps using it until the tune's cache has one; the next
tune converts once into the cache, and `input/bp/` can then be deleted once no run of the
project is in progress.

**`wtune tune` checks disk space.** Before any job, the node pre-flight included, it estimates
the BP conversion of the case's inputs not yet in the cache and stops when it does not fit the
space writable there (GPFS quota headroom; `--no-space-check` overrides). When a job's per-record
input copy (reflinks, or one full copy) would not fit, the job reads the shared cache instead,
with a warning.

**The confirm round includes the fastest measured layout** that has a configuration. It took the top 3 by predicted
run time and the anchor; the prediction does not see `dist`, so on f1-arm conus12 a cyclic
layout measured at 0.132 s/step was never confirmed against block layouts at 0.187.

**Node pre-flight on shared clusters.** A probe the scheduler refuses or does not start within
30 s (`srun --immediate=30`: a reservation, a busy node, a per-user submit limit) no longer
excludes the node as unhealthy; it is kept and reported as not probed. A probe that times out
has its job cancelled by name and user (`wtune-preflight-<pid>-<node>`): a killed `srun` left a PENDING
job behind on scc. A refusal that names an invalid account, partition or QOS is reported as a configuration error, not as a busy node.

**`wtune run --thompson-cache`.** Every segment linked no `qr_acr_q*.dat` / `freezeH2O.dat`, so
Thompson recomputed them at each start (~70 s on scc). With `--thompson-cache` (also on
`wtune submit`) segments link them from the per-build cache and add the ones a completed segment
computed. It is off by default: on f1-arm the tables take under a second (matthew: 20 s for the
segment that computed them, 21 s for the one that read them). Without it `wtune run` deletes the
tables WRF computed into a segment once its job has ended, however it ended; they are 357 MiB and
were kept in every segment directory.

**One Thompson table cache per build, keyed by the `wrf.exe` contents.** The tune's probes kept
their tables in `$OUT/thompson_tables`, shared by every build, and the coldstart probes and
`wtune run` keyed theirs by the sha1 of the `wrf.exe` path, so a rebuild in place with other flags
kept the old build's tables. All of them, and `wtune calibrate`, now use
`~/wtune-runs/thompson_tables/<sha1 of wrf.exe>/` (`$WTUNE_TABLE_CACHE` moves it). The coldstart
probes' copy of the link step also linked a `freezeH2O.dat` the cache did not have, which makes
WRF fail writing the table (the same defect as `wtune run`'s below). Every job now caches a
table through a temporary name in the cache, so a concurrent job never links a partial file; the
tune's probes cache only from a run that got past WRF's init (one that died writing a table
cached it before), and `wtune calibrate` also caches what a completed run computed. After the
upgrade every build starts with an empty cache: run a tune before `wtune calibrate` on a new
build, or the calibration runs submitted together all compute the tables and are excluded from
the fit. The old `$OUT/thompson_tables` and the path-keyed coldstart directories are no longer
read and can be deleted.

**Probe ledgers carry the build.** Each ledger row's notes get `build=<sha1 of wrf.exe>`. The
s/step estimate a tune sizes its first measurements with (round 0, the node-search finalists,
the `-O0` cold-start prediction) used every row of the case on the machine, so a vanilla and a
tuned-flag build were averaged together; it now uses the rows of the same build, then rows
written before this change, never another build's, and so does the `-O3` cross-case prior. The
tune takes the build from the tree its jobs will run (`env.sh`, then the site file's `$W`) and
prints it; with no `wrf.exe` there it says that every build's rows are used.

**A project's probes stay in the project.** Inside a project the probes write their ledgers and
WRF run directories to `<project>/probes/` (`--ledger-dir` still wins; outside a project,
`$WTUNE_CAMPAIGN_OUT`, else `~/wtune-runs/campaign`). The tune exports `WTUNE_CAMPAIGN_OUT` and
the site files use it for `$OUT`, so the probes write where the tune reads: f1's site file wrote
to `~/wrf-bench/wscore/campaign` while the tune read `~/wtune-runs/campaign`. Ledgers of earlier
tunes in `~/wtune-runs/campaign` are still read for the estimate and the `-O3` prior.

**`wtune predict` uses the machine's cores per node.** The calibration file's `cores_per_node`
came first, so on f1-arm (144 cores, no calibration of its own) scc's 72 made predict warn that
90 ranks per node exceed 72 cores. The machine setting now wins when it is set.

**ISD-Lite moved to AWS.** NCEI's `pub/data/noaa/isd-lite/` now answers 301 with the
`noaa-isd-pds` bucket's index page, and `wtune post fetch stations` saved that page as every
station's `.gz` (f1-arm c34: `terrain_point_stat` and `terrain_leeward_station` failed with "Not a
gzipped file"). It now fetches from `noaa-isd-pds.s3.amazonaws.com/isd-lite/data/`, skips a
download that is not gzip, and replaces such a file left by an earlier fetch.

**`rsl` files are compressed.** After a segment succeeds, the `rsl.out` and `rsl.error` files of
every rank but 0 are gzipped (`gzip -1`, about 17x on sc26p1's `debug_level=50` logs, which wrote
17.6 GB in 22 minutes). A failed segment keeps them plain for the diagnosis. When the quota is
full the gzip stops with a warning and the run goes on.

**`wtune submit` passes the run's options.** `--resume`, `--checkpoint-minutes` and
`--stall-minutes` go to `wtune run`; `--no-space-check` goes to both tune and run.

**Launch lines in reports.** The tune report and `wtune predict` printed scc's
`mpirun ... ppr:N:node` on every machine; they now print the command the site file's
`site_launch` builds (f1-arm: `ppr:45:numa`).

**Terrain captions.** `terrain_point_stat` and `terrain_leeward_station` said no station
observations were fetched; they now describe the ISD-Lite / MET Point-Stat figures and the lee
station rule (30-100 km downstream of the crest, within 25 km of the section line).

**`-O0` time breakdown.** The whole run from `wtune predict` carries predict's own init time, but
the breakdown subtracted the 120 s startup constant from it, so a short case showed a negative
compute (sc26p2: `compute -20s`). Compute is now predict's compute, and init the rest.

**A node count that measured nothing is reported.** A probe that timed out or failed (isc21 on
scc: N=1, rc=124) was only "submitted" in the log, and the table's speedup and efficiency then
started from N=2 without saying so. The sweep now prints the failed N with its job and state,
and the log and report say which N speedup and efficiency are relative to.

**Older checkpoints are deleted without `--compress-history` too.** `--resume` (on by default)
writes a restart every 1/8 of the run; only `--compress-history` deleted the older ones, so a
one-segment run kept about 8 sets. After an attempt that finished, the restart sets WTune asked
for (its checkpoints, or `--checkpoint-minutes`) are cut to the newest valid one, which the next
segment starts from; the case's own restart interval is not touched. A failed attempt keeps all
of them until its resume finished, then only the set the resume started from: a resume cannot use
the newest set when the job died before stepping past it, or when it is the segment end, and
takes the one before. For the same reason `--compress-history` now keeps
the two newest sets while a job runs, not one.

**`use_adaptive_time_step`.** The step count was run length / `time_step`, though WRF picks d01's
dt each step. `wtune predict` now gives the bounds from `min_time_step` and `max_time_step` (as
WRF's `start_em.F`: -1 is NINT(3 or 8 x min(dx, dy) in km) s, `*_den` divides), widens the range
to them and says so; `wtune tune` prints the same warning. They are approximate: WRF shortens dt
below the minimum to land on output times.

**`wtune run` converts its inputs on a compute node.** The netCDF -> BP conversion ran in the
`wtune run` process, on the login node (tens of GB for dorian). When the BP cache is not current,
`wtune run` now submits one conversion job (`<out>/nc2bp/`), waits for it, and stops with nothing
submitted if it fails; a current cache submits nothing; `wtune stop`, Ctrl-C and an sacct that
never answers cancel it. Run
inside a Slurm job (`SLURM_JOB_ID` set, which an `salloc` shell also sets), it converts in place
as before. The login node still reads the inputs to check the cache, and hashes one whose
modification time changed.

**Network only on login nodes.** Two post steps reached the internet from a compute-node job:
the tc report downloaded the NHC b-deck (30 s timeout, then HURDAT2), and the map check asked
cartopy for the Natural Earth coastline, which downloads a missing file (f1-arm job 14882 hung
12 min). The map check now only looks on disk (`bin/fetch-natural-earth.sh` fetches);
`wtune post fetch bdeck` puts the storm's b-deck in `<project>/input/obs/bdeck/`, which tc and
the recipe read, and inside a Slurm job tc no longer downloads. `wtune submit` fetches it when
`post.fetch` lists `bdeck`; a storm of the current year is fetched again each time (NHC still
revises it; a failed download keeps the copy it has), an older one only when its file changed. The test suite, which runs on compute nodes, now fails a test that
connects to a non-loopback address or looks up a host name, instead of letting it wait for a
timeout.

**`wtune calibrate` links the files the namelist names.** The calibration jobs linked `$W/run`,
the Thompson tables and the inputs, but not `iofields_filename`, `wrflowinp`, `tslist` and the
like; WRF 4.8 hangs at `input_input` without the iofields file. They now run the probes'
`configs.py link-namelist-files`.

**Startup and init per machine.** `machine.measured.startup_s` and `init_s` replace scc's 120 s.
On f1-arm 45 segments took 2-15 s from job start to WRF's first step (median 5.4 s), so every
prediction there was about 100 s high; the f1-arm guide now sets both to 15 s.

**Ledger estimates scale from the nearest node count only.** The s/step estimate scaled every
ledger row linearly to the node count asked for; WRF does not scale linearly, and matthew at
N=1 was estimated at 0.061 s/step against 0.016 measured. Only the rows at the nearest measured
node count are used now.

**Docs.** `-O3` is described as what it does: it measures the whole space, with no model rounds.
The workflow, tune and post flowcharts have colour legends.

## 0.9.2 — 2026-10-19

**Stopping a tune keeps what it measured.** Ctrl-C, or `wtune stop` for a tune running in the
background, now stops `wtune tune` within one poll: it submits nothing more, cancels only its
own jobs, and writes `tune_report.json` from the best of what was measured -- the layouts
already measured at a finalist included -- so `wtune run` can use it right away. Before, Ctrl-C
killed the controller, left its jobs running in Slurm and wrote no report, and `wtune run` then
used the previous tune's report. A second Ctrl-C quits at once; `wtune stop` then cancels the
jobs that tune left (it keeps its pid and job ids in `tune/<id>/tune.state.json` while it runs).
Under `wtune submit`, Ctrl-C no longer kills the running step: the step stops itself, then
submit stops (`wtune submit --skip-tune` continues with the report). The same applies to a
`--max-queue-hours` stop: the jobs of the interrupted node-sweep wave that already finished now
count. `wtune run` skips a tune report with no result (a tune stopped before anything was
measured) instead of failing on it while an earlier report exists.

**`env_source` with `$HOME`.** `machine.env_source: $HOME/...` reached the jobs unexpanded, so a
job's site file could not source it (f1-arm, `f1arm.sh: line 23: $HOME/wtune-f1arm/env.sh: No such
file or directory`); jobs worked only because they inherit the submitting shell. It is now expanded
like `wrf_tree`.

**`wtune run` stops the same way.** Ctrl-C or `wtune stop` cancels the running segment at the
next poll, submits no further segment and marks `run_report.json` `stopped` (exit 130). Before,
Ctrl-C left the segment job running in Slurm and the report said `failed`. `wtune stop` finds
runs by `runs/<id>/run.state.json`. The prediction-rebuild jobs of `wtune tune` are now recorded
as they are submitted, so a stop cancels them too.

**`wtune prep wps` and `wtune prep real`.** WPS (one job: geogrid, ungrib, metgrid) and then
real.exe (one job) for the project's case over its whole namelist window, reusing the
`case_rebuild` steps; the forcing is downloaded on the login node. `prep real` links the new
`wrfinput`/`wrfbdy` into `case/` (`--replace` when it already has them). On f1-arm only the forcing
download has run so far (it stalled on two files, see the project page); the WPS and real.exe
jobs of `prep` have not yet run on a cluster.

**`wtune run --restart-io-form 2` kept no restart.** With ADIOS2 history, each `wrfrst` name was
pre-created as a directory; WRF's netCDF writer failed on it without an error and the empty
directory was removed afterwards. Only io_form 14 outputs are pre-created now.

**Restart-only cases.** A case with `wrfrst_d<NN>_<start>` and no `wrfinput_d01` now works
end to end. `wtune project init` keeps those restart files, sets `restart = .true.` in the
project's namelist copy and records the restart time and format in `cases.json`. Every tune
probe starts from them (`--restart-from`), and `wtune run` starts segment 0 from them with
`io_form_restart` matching their format. ADIOS2 input staging no longer requires wrfinput when
the namelist restarts. `wtune post preview` and the observation window read a netCDF wrfrst
when there is no wrfinput. `--compress-history` now removes restart *links* instead of failing
on a linked BP directory.

## 0.9.1 — 2026-10-19

**`wtune post preview`: what post will draw, before the run.** It lists the figures of the
detected case types with the question each answers (`expected_figures.csv`) and draws what is
known without the model: terrain, the t=0 SLP and 10 m wind from wrfinput, the best track over
the simulated window, Stage IV rain over that window and the fetched ERA5 z500, into
`post/preview/figures/` with a `preview.html`. Case-type detection reads wrfinput when there
are no wrfout frames yet (matthew on f1-arm: the same seven types as the post of its run, under
a minute on the login node, no job).

**Every figure in one folder.** `wtune post recipe` also hard-links each figure of the report into
`post/<id>/figures/` as `<type>-<file>`; the per-type folders and the report are unchanged.

**The tune's BP inputs stay in the project.** A tune run inside a project converts its inputs
into `<project>/input/bp-cache` instead of `~/wtune-runs/bp-cache`; `WTUNE_BP_CACHE` still wins.

**Docs.** A new first tutorial page on how the machine, project and command layers fit and what
tune, run and post each do; the project folder in full, with what lives outside it and what to
keep or delete; how to stop a tune and cancel its jobs.

**`--minutes` fills the cap instead of fixing the plan up front.** `-O2` keeps adding
round0 top-up waves, each sized from the measured startup and s/step, until the cap (less one
race job) or the candidates run out; a case with no prediction point gives the prediction
share to the layout search; and `--minutes` no longer turns on the short-run step-down, which
is `--stop-if-short` only (conus25 on f1-arm stopped at 16.7 of 45 minutes, +5.1% slower than
uncapped).

**Time left under `--minutes` goes to the node counts never tried.** One ANCHOR goes to every
legal N the sweep never sent (after an early stop, outside the refinement bracket). When the
sweep's own waves say that takes at most 15% of the time left, it runs right after the sweep and
the finalists are chosen from every N; otherwise it waits until the finalists' layouts run out,
and an N more than 3% faster than the best becomes a finalist tuned with what is left. Without
`--minutes` nothing changes (yagi on f1-arm used 11 of 45 minutes and never measured N=5, 9.5%
faster than its pick; c34 had 123 layouts to measure at N=1 and reached N=5 at minute 28).

**`--max-queue-hours` keeps the node counts already measured.** A queue cap reached during the
node sweep, the core sweep or a finalist used to abort the whole layout phase with no N; now
nothing more is submitted and the fastest N measured so far is reported with its ANCHOR. Only
a cap reached before any N is measured still aborts (on f1-arm, c34 and yagi had N=1, 2, 4
measured when their second wave's expected start passed the 10 h cap).

**Machine settings.** Cores per node (72), node cap (13), partition (`compute`) and login
nodes (`head`) are no longer fixed in the code: they are the built-in values of
`wtune.machine`, replaced by a project's `project.yaml` `machine:` block
(`wtune project set --cores-per-node/--max-nodes/--partition/--login-nodes/--account`) or by
`WTUNE_CORES_PER_NODE`, `WTUNE_MAX_NODES`, `WTUNE_PARTITION`, `WTUNE_LOGIN_NODES`,
`WTUNE_ACCOUNT`. `account` is new: sites that bill by project refuse a job without one.
Without these settings every sbatch is unchanged; the post-processing `srun` and the node
pre-flight's `sinfo`/`srun` now name the partition. `wtune tune --max-nodes` was silently
capped at 13; the cap is now the machine's.

**Calibration and hardware of another machine.** `machine.calibration` (`WTUNE_CALIBRATION`)
names the calibration file tune, run, check and calc read; it was always
`machine/calibration/scc.json`, so a file fitted with `wtune calibrate fit` was never used.
`machine.hardware_id` and `machine.numa_per_node` describe a node type the built-in hardware
table does not list, which used to end in a KeyError. Calling a machine `scc-gh200` while
giving it another node size is refused. A calibration without `cores_per_node` now falls back
to the machine setting, not to 112. The node search seeds are the powers of two below the node
cap and the cap (unchanged at 13).

**Site, paths and a per-user file.** `machine.site`, `machine.env_source`, `machine.wrf_tree`
and `machine.geog` replace the fixed `SITE=scc`, `~/wtune-scc/env.sh`, the scc WRF tree and
`/nfs/data/wrf/WPS_GEOG/`; `wtune tune` accepts any site that has a `campaign/sites/<site>.sh`.
Generated rebuild and cold-start jobs default `CAMPAIGN` and `GRIB_PYTHON` to the checkout
they were written from, not to `$HOME/wtune`. Machine settings can also live in
`~/.config/wtune/machine.yaml` (`wtune project set --user`), shared by every project on the
machine; a project's block overrides it.

**Measured numbers and the longest job.** The numbers the estimates fall back on (repeatability
floor, startup and init time, I/O anchors, radiation ratios, prediction bands: 31 of them) were
literals in a dozen modules. They are `wtune.machine.MEASURED`; a machine replaces the ones it
knows with `machine.measured` (`wtune project set --measured NAME=NUMBER`). `machine.max_hours`
is the longest job the partition accepts: `wtune run` segments to fit and probe and rebuild
jobs are capped at it; unset, nothing changes. `wtune calibrate plan --time-limit` now also
moves the launch timeout inside the job, which stayed at 14 minutes.

**f1 leftovers.** `campaign/strict_campaign.sbatch`, the probe job scc uses too, no longer
carries f1's account, partition, node shape and a personal output path in its header (the
sbatch command line supplies them) and finds `CAMPAIGN` under `$SLURM_SUBMIT_DIR` instead of
a fixed home directory; its thread setup follows the site file's `OMP_RUNTIME`, not the name
`scc`. `campaign/sites/f1.sh` uses `$HOME`. The predictor tools default to the machine's node
size, partition and hardware id instead of 448 cores on `cf448`.
`campaign/decomp_search_campaign.sbatch`, which nothing referenced, moved to
`archive/unused-2026-10/campaign/`.

**f1-arm.** `campaign/sites/f1arm.sh` (144-core Grace nodes, two NUMA domains). `wtune submit`
tunes with the project's own `wtune.yaml` when there is one.

**Tests on a cluster.** With Slurm on `PATH`, about 110 tests ran the real node preflight:
`sinfo`, then one `srun` per idle node (1430 on scc in one run), and the nodes it then excluded
leaked into four later tests. The tests now see no Slurm unless they pass their own runner.

**Found on f1-arm.** Job ids there started over, so `sacct -j <id>` answered with another user's
old job of the same number until slurmdbd had recorded the new one: a job just submitted read
as FAILED. State, timing and MaxRSS queries now look at the last 30 days only. The node
pre-flight no longer excludes a `mixed` node whose probe timed out: on a busy cluster that
shares nodes it excluded every node that was not fully taken. New machine setting `max_omp`
(`--max-omp`, `WTUNE_MAX_OMP`): 1 for a WRF built without OpenMP. `search.omp` in a config does
not bind the node sweep's ANCHOR; this cap does. New machine setting `node_preflight`
(`--node-preflight off`, `WTUNE_NODE_PREFLIGHT=0`): f1-arm lets a user have 2 to 12 jobs
submitted, so of the 40 probes all but two were refused and 38 nodes were excluded.

**`wtune run` starts WRF with the site file's launcher.** A segment wrote its own
`mpirun --map-by ppr:<ranks per node>:node` line while the probes of `wtune tune` went through
`site_launch`: on f1-arm (two NUMA domains per node) 90 ranks ran as 72 + 18 after being tuned as
45 + 45, and a site that launches with `srun` could not run at all. scc's commands are unchanged.

**A case that was a run directory.** `project init` no longer links its executables, `wrf.out`
or tslist time series, and a segment never takes an executable from the case, nor a link into
some WRF tree for a file `WRF_TREE/run` has: linked first, the old `wrf.exe` shadowed the
machine's. A table the case ships as its own file is still the case's.

**The winner's rank order.** The winner row of a tune report had no `dist`, so `wtune run`
always started as `block` and a layout measured as `cyclic` would have run with another rank
placement; the report also warned "dist not provided" for every sweep ANCHOR. The row keeps
it and the run follows it. Report text no longer names scc.

**A grid too small for a full node.** The node sweep measured one ANCHOR per node count on all
the cores of a node and skipped a count with no legal decomposition there: `matthew` (91x100,
90 ranks at most) on a 144-core node with one thread per rank left the whole tune with a single
layout. Such a count now gets its ANCHOR on the most cores per node that have a legal one, and
the cores/node sweep compares the usual fractions of that count.

**WRF-Fire without a config.** The rule that keeps a fire case at one thread was written only
into the `wtune.yaml` template; `wtune tune` and `wtune submit` without `--config` had no rule
and swept the node counts of `c34` at omp 2, 4 and 8 (one probe crashed, one hung for an hour).
It is the built-in `exclude` now; `exclude: []` in a config lifts it.

**Probes write no aux output.** A probe's history policy set only `history_interval`, so a
case's own `auxhistN` streams still wrote: every `isc21` probe launch wrote all six frames of
its `auxhist24` validation output on three domains, about 160 s of a 390 s launch at 8 nodes.
Every stream with a nonzero interval now begins after the run ends; the measured s/step does
not change (the median already left the write steps out).

**Aux output in ADIOS2.** `wtune run` switched history, restart, input, boundary and the aux
input streams to ADIOS2 but left `io_form_auxhistN` as the case had it: `isc21` wrote its
`auxhist24` validation output in serial netCDF, 162.5 s of a 911 s run at about 40 MB/s, while
ADIOS2 history on the same machine writes a frame in 0.03-0.07 s. Every aux output the case
writes is now ADIOS2 too (`--io netcdf` keeps the case's formats); `wtune convert bp2nc`
turns a frame back into netCDF.

**`wtune calibrate` in ADIOS2.** The plan's jobs still ran `--no-adios2`: netCDF inputs and a
netCDF history frame, so the measured write and read rates were netCDF's while every run is
ADIOS2. Each job now stages the case's inputs as BP the way the probe jobs do (one conversion
per case, shared cache), copies the per-record boundary files into its run directory, and
writes ADIOS2. `fit` counts the bytes of a BP directory, not the directory entry.

**Two numbers left in the code.** The spin-up time limit of a ranking point padded its estimate
with a literal x1.5 and 120 s; it now uses the machine's `measured` `radiation_factor` and
`init_s`. The probe job's disk guard asked `mmlsquota` about `home1`, the old f1 filesystem: on
f1-arm that is `home2` and scc has no GPFS quota, so every probe printed a warning and guarded
nothing. A site file now names its `QUOTA_FS` (f1 `home1`, f1-arm `home2`, scc none: no guard).

**Post-processing in parallel.** `wtune post recipe` drew every figure set one after another
on one core: dorian's post job sat 56 minutes in a single `DetectNodes` on a 72-core node with
one figure written. The figure sets (common, the CF/VAPOR data products, each case type, the
time series) now run in separate processes, one per CPU of the job, and their entries are
merged in the old order; `WTUNE_POST_WORKERS` sets the count (1 = the old serial path).
`DetectNodes` runs on time slices of the fields file at once and the node files are joined in
time order, which is the file a single run writes (DetectNodes treats each time on its own).
On f1-arm matthew's post went from 4:57 to 2:38 with the same 149 figures.

**Configuration search: idle nodes and an early stop.** Each adaptive round was one job while
the other node slots sat idle, and the rounds only stopped at 75% of the budget, after six
rounds or when the proposer ran dry. A round's batch (unchanged in size) is now dealt over the
free slots like round 0, so it ends sooner; and the rounds stop once the leader is ahead of the
runner-up by more than `noise_frac` (3%) and a round found nothing faster, as the node search
already did for larger node counts.

**WRF's own ADIOS2 compression off.** Probes wrote `adios2_compression_enable=.true.` and
`wtune run` kept the case's value (the Registry default is `.true.`). With compression on, stock
WRF 4.8 writes the 256-character compressor name into a 32-character buffer
(`external/io_adios2/wrf_io.F90`; the SC26-WRF patch `e6472a9` widens it), so a vanilla WRF must
not take that path. Every probe and segment namelist now says `.false.`; `--io netcdf` still
keeps the case's own `&namelist_adios2`.

**A stalled segment is cancelled.** On a full quota WRF+ADIOS2 stops writing and waits
instead of failing: on f1-arm job 15170 idled 75 minutes to its time limit (40 SU) while
`wtune run` kept polling. While a segment job runs, `wtune run` now watches `rsl.error.0000`;
when it has not changed for `--stall-minutes` (default 20, 0 disables) the job is cancelled,
the attempt is recorded as `stalled` and the run stops without a resume.

**History compressed outside WRF.** `wtune run --compress-history` is `--sidecar` where each
converted frame (zlib netCDF) is first compared with its BP variable by variable
(`tools/bp_nc_same.py`) and then takes the BP directory's place, so `wtune post` reads it
directly and keeps no second copy. matthew's 17 frames: 915.7 MB as BP, 237.4 MB as zlib
netCDF. On a full node the conversion shares the last core at the lowest priority. After a
run, of the restart sets WTune added only the newest validated one per directory is kept (a
resume starts from it); after a run that succeeded they are all deleted.

## 0.9.0 — 2026-10-12

Main code made small enough to review; behaviour of the tune -> run -> post path is unchanged
unless listed here.

**One package.** Everything lives under `src/wtune/` (`wtune.cli`, `wtune.tune`, `wtune.run`,
`wtune.post`, ...); the `wtune_*` top-level modules and import aliases are gone. Code no command
reaches moved to `archive/unused-2026-10/`; nine early commands and the predicted-ranking mode
of `wtune tune` are retired. `docs/REVIEW_MAP.md` lists what is left and where to start.

**New commands.** `wtune submit` runs tune -> run -> post for a project in one command;
`wtune doctor` checks that the machine and project are ready. Commands read their defaults
from `project.yaml`. `wtune submit` tunes at `-O2`.

**Run.** Auto-resume from the last valid restart, checkpoints and an optional sidecar;
`--out` that already holds a run is refused before planning; ADIOS2 output directories are
created up front; job-state, restart and layout-override fixes.

**Tune.** Every measurement is kept and a `sacct` failure no longer loses a round; a dirty
winner is re-measured; the first node sweep fits a small `--minutes`; BP inputs are converted
at the first real probe submission.

**Post and checks.** `post compare` fails closed, figures report their own errors, winds are
earth-relative, a partial Stage IV download is reported. `check`/`predict` validate namelist
syntax, ranges and `nproc -1`; iofields lines are merged when staged.

**Docs.** MkDocs tutorial site under `docs/` (`pixi run --manifest-path envs/docs/pixi.toml
serve`), with a public build config.

## 0.8.2 — 2026-10-03

Found on a long real case (climate-ea50: 13-month East-Asia 50 km run, 14 restart segments).

**Post-processing of multi-frame wrfout.** The frame index is built from each file's `Times`
(`frames_per_outfile > 1`; a restarted segment's first file starts one history interval after
the restart): every recipe reads the right time index, climate monthly means use every time
once, and the run length / last time are right (was 395.25 d and 11 months). A history stream
reduced by an iofields file gives "history stream lacks ..." missing entries, not exceptions.

**Prediction.** The whole-run prediction adds history writes, boundary updates, `sst_update`
reads and restart writes, and uses round0's measured startup as init (climate-ea50 46:14 ->
75:45 against 78:09 single-segment equivalent; SC26-P2 2:36 -> 2:12 against 2:21).

**Staging and checks.** Run and probe directories get the files the namelist names (iofields,
wrflowinp, fdda, obs nudging, tslist): WRF 4.8 hangs at input without the iofields file even with
`ignore_iofields_warning`, and `wtune check` makes that FATAL. `wtune project init` no longer
links a former run's outputs. `wtune run` keeps the case's `restart_interval` for a
one-segment run.

## 0.8.1 — 2026-10-02

**Figures.** Gridline labels sit outside every map (cartopy 0.26 drew longitude labels inline
across Lambert maps), and the base-map caption no longer overlaps them. Cloud and TC-structure
legends list only the contour levels that are drawn; the cloud-top map says when no echo
reaches 15 dBZ; two-panel section legends cover both panels.

**Fixes.** The cold-start rebuild passes `--bbox=` as one token: a bbox south of the equator
read as an option and failed the tune's phase-0 rebuild. `climate-ea50` (13-month East-Asia
50 km run) is registered in the scc case overlay.

## 0.8.0 — 2026-10-02

**One-pass tuning levels.** `wtune tune -O0..-O3` (decision 0059): plan, layout search,
whole-run prediction and opt-in physics advice in one call. -O3 (default) searches every legal
(N, cores per node, ranks x threads, nx x ny, numtiles) with no time cap; queue and controller
time are reported, not charged. The node sweep uses hybrid MPI+OpenMP layouts at any node count
(coarse to fine), measures cores per node, and checks node health before each phase. Prediction
points are chosen by case type and weather-cost indicators (decision 0060). `tune_report.json`
carries a top-level `winner` that follows a phase-2 overturn.

**Advice and calculators.** `wtune advise` prints WRF-mode advice (WRFDA, nudging, moving nest,
early start, tc.exe vortex insertion) with measured evidence and side effects, never applied
(decision 0061); it reads the project's storm and accepts `--storm NAME` or `NAME+YEAR`
(e.g. KATRINA -> AL122005, narrowed by the case window). `wtune calc` holds rules of thumb and
conversions incl. nudging wavenumbers (decision 0062). `wtune prep geog` optionally imports
higher-resolution DEMs (decision 0063). `wtune project set --storm --hurdat2`.

**Run and watch.** `wtune run --tune-report` takes the layout from the report's winner (flags
override) and uses the winner's measured s/step when the report has no ranking points.
`wtune watch` follows `OUT/seg_<k>` of a run root (`seg k/K`, whole-run percentage) and uses
the newest wrfout frame as a progress floor when the rsl lags on NFS (`RSL_LAGGING`).

**Figures (decisions 0064, 0065).** Every plan view gets a shared base map (coast, lakes,
borders, terrain contours, nest box). Units, contour intervals and colour tables follow WRF/NWP
verification practice (WPC precipitation thresholds, Roebber performance diagram, FSS useful-skill
line, SLP every 4 hPa with H/L, kt barbs, NWS reflectivity, CW3E IVT, init/valid/Fhhh titles).
New tropical-cyclone structure set: 2 km reflectivity with 34/50/64 kt and RMW, azimuthal-mean
radius-height tangential/radial wind, warm-core anomaly and total condensate, centre-passing
sections, warm-core Hovmöller. New cloud set for every case with hydrometeors: liquid/ice/snow
water paths, cloud-top height, along/across-wind condensate sections, OLR-derived brightness
temperature. "Total condensate" now sums every hydrometeor (QCLOUD+QICE alone was 10% of the
condensate mass in SC26-P2). The 3-D figure is a ParaView volume rendering of total condensate
(storm-centred cutaway coloured by tangential wind; height ruler; adaptive vertical extent).

**3-D data products.** `wtune post cf3d`: CF-1.8 NetCDF on 21 isobaric levels and a regular
lon-lat grid for Met.3D and ParaView. `wtune post vapor-bundle`: hard-linked wrfout frames with
`vapor_render.py` and a README for official VAPOR on a workstation (no linux-aarch64 build
exists); the recipe never copies, the CLI copies only with `--allow-copy`. Both are recipe data
products.

**Fixes.** TC tracking scales the StitchNodes range to the output interval (a fixed 8 deg let a
1007 hPa low 653 km away take Katrina's t0 fix). Post tools (cdo, MET, TempestExtremes) are found
without `bin/wtune` or `WTUNE_POSTPROC_VIEW`; post and fetch read segmented runs as one run.
`wtune check` no longer reports ndown for a linked `ndown.exe`, and a case directory's missing
WRF tables that `--wrf-run-dir` has are INFO, not FATAL. Tune tests use a per-session temp dir
(a fixed /tmp path owned by another user failed them on a shared node).

## 0.7.0 — 2026-10-01

**Measured node-count search.** `wtune tune --nodes search` is the new default: the ANCHOR
configuration is measured at every legal node count (one job per N, REPS=2, waves within
`--max-nodes`, extension guards applied, early stop once more nodes stop helping, sweep capped
at `--sweep-budget-frac` of `--minutes`, default 0.4), and the node count is chosen by
`--node-objective time|cost|knee` (`--deadline-minutes`, `--min-efficiency`,
`--node-finalists`). The full search then runs at the finalist node counts. `tune_report.json`
gets `node_selection` with all three objectives. `--nodes N` and `--nodes auto` are unchanged;
`budget.nodes: search` and the new keys are accepted in YAML. **Not yet verified on the
cluster** (pre-registered checks: matthew picks N <= 4 and stops early; conus25 picks N >= 8).

**WRF extensions.** `bin/build-wrf-extensions.sh` gains `COMPILER=gnu` for WRFDA (nvfortran 26.5
rejects the GNU `isnan` in the GOES-ABI reader); WRFDA con200 3DVAR converges with the gcc build.
WRF-Chem gets a provisional decomposition guard (`CHEM_MIN_J_PATCH=21`) after GOCART hung at
narrow j-patches; wider splits still hang intermittently (about 2 in 11 runs). Extension cost
calibration (72 jobs on scc) left all coefficients as placeholders under its pre-registered
rules; measured ratios are in `results/derived-data/2026-10-01-extension-calibration-results.csv`.

**Docs.** README gains a hands-on tutorial in Traditional Chinese (`## 教學：從零開始用 WTune`).

## 0.6.0 — 2026-09-30

**Observation comparison and data fetch.** `wtune post fetch NAME --project P [--run-dir R]`,
`fetch --list` and `fetch-era5` download into `<project>/input/obs/<name>/` with a
`provenance.json` each: `stage4` (IEM, 2001+, CONUS), `nexrad` (level-2; 2005 volumes take the
radar position from NCEI HOMR), `mrms` (from 2020-10-14), `stations` (ISD-Lite), `igra`,
`era5`, `era5_monthly`, `gpcp_monthly`. `post recipe` scores precip against Stage IV and
convection against a NEXRAD (Py-ART, `radar_composite.py`) or MRMS composite with MET Grid-Stat
(CTC/CTS, FSS), stations with `ascii2nc` + Point-Stat, all inside the model interior (precip also
within 230 km of a NEXRAD site). The report gains a source/license section and a notes box that
collects every warning; offline runs print the exact fetch command for what is missing.
`arm_pyart<2` (1.19.5), `xradar>=0.9` and `dask` join `envs/wtune/pixi.toml` (xarray stays at
2026.7); `WTUNE_PYART_PATH` stays as a fallback.

**Recipes.** `pbl`: WRF against IGRA2 radiosondes (Skew-T overlay, bulk-Richardson PBL height,
T/Td/wind RMSE). `ensemble`: spaghetti at the highest member minimum + 10 hPa, k/N probability
bins, boundary zone blanked. `climate`: annual cycle for four boxes and seasonal bias against
ERA5 monthly and GPCP v2.3, `--climate-spinup-days`. `post sounding` and the pbl overlay draw the
surface-based parcel profile with CAPE/CIN shaded (`post/sounding.py add_parcel`).

**TC tracking.** Per-time finest-nest selection with a recorded fallback
(`track_domains_<label>.csv`), position from CDO-smoothed SLP (`--track-smooth-km`, default 30,
0 = raw), MSLP/Vmax from raw fields, `warning: tracking grid: ...` notes and a track-step
quality line.

**Figures.** Fixed per-field scales across time-series frames, symmetric change maps with a white
zero bin on the domain projection, moisture time-height 0-12 km AGL, 500 hPa vorticity on a
sequential scale, below-ground 850 hPa hatched, 1000 hPa height extrapolated for thickness.

**WRF extensions.** `wtune check` reports support for WRF-Chem, WRF-Hydro, WRFDA and moving nest:
grid/rank-only tuning, extension parameters untouched, OpenMP capped at 1 for Chem/Hydro/WRFDA,
moving-nest cost factor +6.0 % measured, Chem/Hydro factors placeholders
(`src/wtune_extension_coefficients.json`, `$WTUNE_EXTENSION_COEFFS`). Refused: `predict-rebuild`
on any of them, WRFDA cycling in `wtune run` (`--wrfda-forecast-only`), `--cfl-rescue` on
Chem/Hydro/moving nest. `bin/build-wrf-extensions.sh` builds them on scc.

## 0.5.0 — 2026-09-29

`wtune project init|info PATH` (decision 0058): a user-named project folder that collects one
WRF case's `case/` (namelists copied, everything else -- wrfinput/wrfbdy/met_em/... --
symlinked; the source case dir is never opened for writing), `input/`, `tune/`, `predict/`,
`runs/<id>/`, `post/<id>/`, `analysis/` and a `project.yaml` (case/site/storm/obs/time +
optional `cross_sections.terrain`/`cross_sections.moisture` line overrides) under one directory,
with a `manifest.jsonl` line (time, `wtune` version, argv, key input paths + sha256 for files
<50 MB, output dir, exit code) appended by every project-aware command
(`src/wtune_project.py`). `wtune post recipe/tc/plot/custom` resolve their project via
`--project DIR` / `$WTUNE_PROJECT` / walking up from cwd for `project.yaml`, default their
`--out` to `post/<id>/`, and auto-discover RUN_DIRs from `runs/*/` (else `case/`) and
storm/HURDAT2 from `project.yaml` when not given on the command line; `tune`/`run`/
`predict-rebuild` are unchanged (decision 0058 explicitly scopes them to output-directory
defaults only, not attempted this round beyond that). Outside a project, every subcommand's
pre-existing behaviour is unchanged (decision 0058 P6).

`wtune post recipe --every 6h|--times T1,T2,...` (decision 0058 P3): a full per-valid-time
figure set (SLP/T2/10 m wind/rain, +MCAPE when convection was detected) written to
`post/<id>/timeseries/fNNN_<YYYYmmddHH>Z/`, named `<field>_d<domain>_fNNN_<YYYYmmddHH>Z.png`
(forecast-hour tag + valid time, no colon -- ParaView's reader fails on one, confirmed on scc
2026-09-29) — `post/timeseries.py`. Consecutive and vs-t0 **change maps** for SLP/T2/PW/MCAPE
plus a 6 h rain accumulation, every difference computed with `cdo sub` on NetCDF extracted via
the existing `wtune post extract` engine, never re-derived in Python — `post/changemap.py`,
named `<field>-chg<N>h_d<domain>_fNNN_<validZ>.png`. `--animate` builds one ffmpeg MP4 per
time-series field (`post/anim.py`, `ffmpeg` added to `envs/wtune/pixi.toml`), referenced from
the report by relative path (not base64-embedded). The report's new "Time series" section
groups these by forecast tag with prev/next navigation links.

`wtune post custom` (decision 0058 section 2): runs every `analysis/*.py` script in a project,
each handed a `post/custom_api.Context` (run list, `open_wrfout`, `list_frames`,
`find_run_track_csv`, `save_figure`/`save_table`), folding whatever it registers into one HTML
report; a script that raises gets one error entry instead of aborting the batch.
`wtune project init` writes an example script (`analysis/example_tc_track.py`): storm-following
central pressure & max wind, from an existing track csv if `wtune post tc`/`recipe` has already
run, else a per-frame SLP-minimum fallback via wrf-python.

Two new case types (decision 0057 update, 2026-09-29): **terrain** (any domain's HGT relief
>=1000 m or peak >=1500 m — terrain+10 m/700 hPa wind map, a theta/w cross-section through the
peak along the mean 700 hPa wind, rain-over-terrain-contours, and a time-height diagram at the
domain's lowest point; `post/recipes/terrain.py`) and **moisture** (triggers whenever
precip/convection/tc is detected, i.e. effectively always alongside precip — PW + its 6 h
`cdo sub` change, IVT + 850 hPa wind, an RH/theta_e cross-section through the TC centre or
max-rain point, and a specific-humidity time-height diagram; `post/recipes/moisture.py`).
IVT (`post/ivt.py`) is this package's one self-written integral (neither wrf-python nor MetPy
exposes a moisture-flux integral) — `IVT = (1/g) integral q*V dp` via `numpy.trapezoid`, exact
for a piecewise-linear profile and therefore exact for a constant synthetic profile
(`tests/test_post_ivt.py`, decision 0057 update T3). Station-observation-dependent terrain/
moisture figures (lee-side foehn timeseries, MET Point-Stat) are reported as explicit missing
entries, matching the house convention already used for precip's Stage IV/IMERG gap.

`gcsfs`/`zarr` added to `envs/wtune/pixi.toml` alongside `ffmpeg` (previously only in
`envs/ai/pixi.toml`) so `post/recipes/era5.py`'s anonymous ARCO-ERA5 access works from this
env too — confirmed opening the public `gs://gcp-public-data-arco-era5` zarr store and cropping
a real z500 field, both locally and on scc's login node, with no credentials.

## 0.4.0 — 2026-09-29

`wtune post recipe RUN_DIR [RUN_DIR ...] --out DIR` (decision 0057): case-type-driven standard
figure sets, replacing "pick a figure by hand" with "detect what kind of case this is and produce
its whole standard set, all matching types when they overlap". `post/caseinfo.py` reads
namelist.input/namelist.output + a HURDAT2/b-deck best-track check to detect, multi-label, any of
common (always) / tc / convection / precip / synoptic / pbl / chem / climate / ensemble, each with
its own stated evidence. `post/recipes/` holds one module per type (mapfig.py is the shared
cartopy map-figure writer every type reuses), each producing figures from established tools —
wrf-python getvar/interplevel (including `uhel` for 2-5 km updraft helicity, exactly the layer
decision 0057 asks for, with no bespoke formula needed), MetPy (`wtune post sounding`, reused),
`wtune post tc`'s own MET-backed pipeline (reused, plus one new self-written figure this repo has
no tool for: a 10 m tangential-wind radius-time plot, `post/recipes/tc_extra.py`), CDO
(`ymonmean`/`yseasmean`), and ERA5 via ARCO (`post/recipes/era5.py`, reusing
`tools/era5_intermediate.py`'s own zarr reader). A ParaView cloud-isosurface-over-terrain render
for tc/convection (`post/xwrf_convert.py` + `post/paraview/render_cloud.py`, run under
`pvbatch --force-offscreen-rendering`) is included but not exercised against a real binary this
round (ParaView not installed locally) — same disclosed-unverified status MET's own config
templates already carry. Every observation-dependent figure with no obs wired up this round
(radar, Stage IV/MRMS/IMERG, chem stations, ensemble verification, the ERA5 side of the climate
seasonal bias) is reported as an explicit "missing: dataset, reason" entry in the HTML report,
never silently dropped. Every figure is captioned from `src/post/recipes/catalog.json` (zh+en:
question/how-to-read/common-misreadings) and stamped with run/domain/valid time and the excluded
spin-up length.

**Disclosed deviation:** the synoptic ERA5 z500 diff figure's RMSE/bias is computed directly in
numpy after a scipy regrid, not through MET Grid-Stat as decision 0057 names — MET is not
installed locally this round; see `post/recipes/synoptic.py`'s module docstring and the README.

## 0.3.0 — 2026-09-29

`wtune post` (decision 0056): a post-processing/analysis package, advisory-only, no autotuning.
`inventory` (domains/times/grid/variables/missing-frame check, netCDF4 only), `extract`
(wrf-python getvar diagnostics -> points CSV or a zlib-compressed box NetCDF, parallelized over
frames with ProcessPoolExecutor), `plot` (filled-contour maps, LANDMASK or cartopy coastline),
`sounding` (MetPy Skew-T with surface-parcel CAPE/CIN), `tc` (wind-structure and finest-domain
intensity metrics ported from the p2-nest reference scripts, against HURDAT2 + optional EBTRK),
`compare` (bitwise/max|delta|/RMS between two runs, to verify a STRICT change did not move the
answer), `report` (self-contained HTML, images base64-embedded). All heavy imports
(wrf-python/matplotlib/metpy/cartopy) are lazy, so `wtune --help` and `wtune post inventory`/
`compare` never need them. `envs/wtune/pixi.toml` gained matplotlib, metpy, nco, cdo, cartopy.

**2026-09-29 update (same-day, decision 0056):** every `wtune post` subcommand now wraps an
established tool instead of re-implementing it, on user instruction ("post-processing must be
built on established tools such as ParaView or other common WRF visualization/analysis tools; do
not write our own unless necessary") -- the self-written substitute each one replaced is deleted,
not kept alongside it. `compare` now runs `diffwrf` (WRF's own netCDF differ) per common frame
instead of a self-written numpy diff. `inventory` reads grid/variables from `ncdump -h`, cross-
checked against `cdo sinfon`; only the missing-frame cadence check stays self-written (WRF/WTune-
specific, no tool audits it). `plot` now defaults to `--basemap cartopy` (`wrf.get_cartopy`/
`wrf.latlon_coords`, Natural Earth coastlines/borders via `WTUNE_CARTOPY_DATA`), falling back to a
LANDMASK contour -- and printing that it did -- only when the Natural Earth cache is unavailable,
never on a cartopy import failure. `tc` keeps TempestExtremes for tracking (Vmax/MSLP now read
straight from its own track) and `post/tc_metrics.py`'s quadrant-radii math (the one metric no
tool derives from a WRF wind field), writes the result as a new ATCF a-deck (`post/atcf.py`, the
one justified new module -- no tool converts a TempestExtremes track to ATCF; format verified
against a real NHC archive file), resolves a b-deck (`--bdeck FILE`, NHC archive download, or
HURDAT2 as a last resort, converted through the same ATCF writer), and compares them with new MET
`tc_pairs`/`tc_stat` (track/intensity/wind-radii error) and `tc_rmw` (azimuthal-mean radius-height
structure on a `cdo remapbil`-regridded frame) wrappers (`post/met_tc.py`) -- replacing the
self-written track-error/intensity/azimuthal-profile/RMW code (most of the old `tc_metrics.py`)
outright. `post/tools.py` centralizes tool location (`WRF_TREE`/`PATH`/`WTUNE_POSTPROC_VIEW`) and
raises a clear error naming the missing tool + `bin/install-postproc.sh` instead of a silent
fallback. MET is not installed yet (spack job 1575 was still building `envs/postproc` at the time
of this update), so `post/met/*.conf.tmpl`'s exact config syntax is unverified against a real
binary -- see README section 7 and each template's header comment.

Policy enforcement for decision 0056 ("STRICT auto-applies, anything that changes the solution is
advisory only"): `wtune run` without `--cfl-rescue` now stops on a CFL/NaN failure and writes the
recommended `time_step` as `segment.advice.namelist.input` next to the failed segment instead of
silently doing nothing more than failing; `--cfl-rescue` remains the one user-pre-authorized
exception. `wtune_compliance.assert_strict_only` guards `wtune tune`'s final recommended
configuration (`wtune_tune.build_report`), raising `ComplianceError` if any changed namelist key
classifies as anything other than STRICT (measurement-only probe windows are exempt).

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
