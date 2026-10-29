---
name: wtune-operate
description: Install WTune, set up a project, prepare inputs, tune, run and monitor WRF, and use every other WTune command (predict, advise, calc, physics, compliance, convert, metdata, scenario, calibrate, post tools). Use when the user asks to install or deploy WTune, start a WTune project, run or tune a WRF case, check why a run is slow or stuck, or asks what a WTune command does.
---

# wtune-operate

WTune is a shell tool: commands, files and exit codes. This skill covers installing it and every
command an agent may run. Events and observations are in `wtune-event`; figures, preview and
assessment are in `wtune-post`; the agent's overall role is in `AGENT.md` at the repository
root. `wtune <cmd> --help` is the authority for flags; this file says which command to use when.

Rules that hold for every command:

- Never edit WRF source, the physics of the namelist, or WRF output. WTune changes only layout,
  launcher, env and I/O options.
- Commands that submit Slurm jobs (`tune`, `run`, `submit`, `prep wps|real`, `predict-rebuild`,
  `post ... --srun`) cost allocation. Say the size (nodes x hours) before starting one.
- Commands that need internet (`prep wps` download, `prep geog fetch|srtm-fetch`,
  `metdata fetch`, `post fetch ...`, `scenario generate`) run on the login node. Compute nodes
  often have none (f1-arm has none).
- Long commands run in the background with a log: `nohup wtune run ... > run.log 2>&1 &`.
  Poll; do not block one call for hours.
- Stop WTune work with `wtune stop`, never with `kill` or `scancel` on jobs you did not start.

## 1. Install

| layer | what | command |
|---|---|---|
| core | pixi envs `envs/wtune` (python, catboost, netCDF4...) and `envs/adios2` | `curl -fsSL https://pixi.sh/install.sh \| bash`; `pixi install --frozen --manifest-path envs/wtune/pixi.toml`; the same for `envs/adios2/pixi.toml`; `bin/wtune --version` |
| WRF/WPS | built by the user, WTune never builds it | find and set the paths as in "Paths" below |
| postproc (optional) | MET 12.1.1 and nccmp (spack env `envs/postproc`), ParaView (`envs/paraview`), Natural Earth | `sbatch bin/install-postproc.sbatch` (on a compute node), `bin/install-postproc.sh --check`, `bin/fetch-natural-earth.sh`; then set `WTUNE_POSTPROC_VIEW`, `WTUNE_PARAVIEW_BIN`, `WTUNE_CARTOPY_DATA` |

Then `wtune doctor` until no line says `FAIL`. A missing optional tool shows up later as a
skipped figure with the reason, not as an error.

### Paths: set them up yourself

`wtune doctor --json` checks every path a job uses and gives each `FAIL`/`WARN` a `fix`: the
site file, the partition, the env file every job sources, the WRF tree (`main/wrf.exe`,
`main/real.exe`), the libraries `wrf.exe` cannot load after sourcing the env file, the WRF_TREE
of the env file against machine.yaml's `wrf_tree` (tune jobs use the first, `wtune run` the
second), WPS, the geog data, the site's adios2 python (`PIXI_ENV`) and Natural Earth. Do not ask
the user for a path WTune can find:

1. `wtune doctor --json`. Done when `ok` is true and every `WARN` left is a feature the user does
   not need.
2. `wtune doctor --find --json` searches `$HOME`; add the cluster's other file systems with
   `--find-root /work/$USER --find-root /opt ...` (at most `--find-seconds`, default 120). It
   lists `wrf_tree`, `wps_dir`, `geog`, `env_files` (existing WTune env files) and, for each
   library `wrf.exe` cannot load, the folders that hold it.
3. Pick from what it found:
   - WRF tree: `main/wrf.exe` and `main/real.exe` exist, `configure.wrf` has
     `DMPARALLEL = 1` (dmpar), and `ADIOS2` appears in it when the runs use io_form 14 (WTune's
     default). Several valid trees: take the one the user named, else the newest, and say which.
   - Libraries: the folder whose path names the same compiler and MPI as the WRF build (spack
     views and install prefixes do); if WRF was built under `module load`, put the same
     `module load` lines in the env file instead.
4. Write the env file (one per machine, e.g. `~/wtune-<site>/env.sh`, sourced by every job and at
   each login):

   ```bash
   # module load ...                        # only if WRF was built with modules
   export WRF_TREE=<tree>
   export WPS_DIR=<wps>
   export LD_LIBRARY_PATH=<lib folders>:${LD_LIBRARY_PATH:-}
   export WTUNE_PIXI_ENV=<wtune>/envs/adios2
   export WTUNE_CARTOPY_DATA=<wtune>/envs/cartopy-data   # after bin/fetch-natural-earth.sh
   export PATH=<wtune>/bin:$PATH
   ```

5. Record it: `wtune project set --user --env-source <env file> --wrf-tree <tree> --geog <geog>`
   (the same tree as in the env file), then `source <env file>`.
6. Scheduler values: partition from `sinfo -h -o %P`; cores per node from
   `sinfo -h -p <partition> -o %c`; an account from `sacctmgr -nP show assoc user=$USER
   format=account` when the site needs one; set them with `wtune project set --user
   --partition P --cores-per-node N --account A`.
7. Back to step 1.

Ask the user only when: no `wrf.exe` exists anywhere (WTune does not build WRF:
`docs/guide/build-wrf-f1-arm.md` is one recipe); `sacctmgr` lists several accounts; or the
cluster has no site file in `campaign/sites/` (copy the closest one, set `CORES_PER_NODE`,
`site_env` and `site_launch`, and have the user review it before the first job).

Site and machine:

- The site file `campaign/sites/<site>.sh` (scc, f1, f1arm) holds the launcher; pick it with
  `site:` in machine.yaml, `--site`, or `WTUNE_SITE`.
- Per user: `~/.config/wtune/machine.yaml` (write it with `wtune project set --user ...`).
  Per project: `machine:` in project.yaml. Environment variables beat both
  (`docs/reference/env.md`).
- `wtune project set` machine flags: `--partition`, `--account`, `--max-nodes`,
  `--cores-per-node`, `--hardware-cores-per-node`, `--numa-per-node`, `--exclude-nodes`,
  `--login-nodes`, `--node-memory-gib`, `--max-hours`, `--max-omp`, `--env-source`,
  `--wrf-tree`, `--geog`, `--site`, `--hardware-id`, `--calibration`, `--measured NAME=NUMBER`,
  `--copy-probe-inputs on|off`, `--node-preflight on|off`.

Guides: `docs/tutorial/install.md`; f1-arm (env.sh, machine.yaml, deploy, postproc):
`docs/guide/f1-arm.md`; building WRF there: `docs/guide/build-wrf-f1-arm.md`; every variable:
`docs/reference/env.md`. Change `WRF_TREE` in the env file and `wrf_tree` in machine.yaml
together: tune jobs source the env file.

A new cluster also needs a calibration: `wtune calibrate plan --site S --case-dir C --out D`
writes sbatch scripts at several rank counts (you submit them); `wtune calibrate fit --runs D
--out cal.json` fits it; point machine.yaml at it with `--calibration`.

## 2. Project

```bash
wtune project init ~/projects/mycase --case /path/to/case_dir [--storm AL122005 --hurdat2 F] [--exclude-nodes n1]
cd ~/projects/mycase
wtune doctor
wtune project info          # resolved layout, machine and post settings
```

- `--case` is read only: the namelist is copied to `case/`, the rest linked.
- Init writes `AGENTS.md` (pointer to these instructions); `wtune project set --agents-md`
  rewrites it for an older project; `--register-case` registers `case/` for tune in a project made
  before registration existed.
- Storm settings: `project init --storm --hurdat2` (NHC); for another agency's best track
  `wtune project set --storm WP122017 --bdeck <ATCF b-deck>`.
- Post settings: `--post-types`, `--post-every`, `--post-fetch`, `--post-focus
  LAT0,LAT1,LON0,LON1`, `--cross-section KIND=LAT0,LON0,LAT1,LON1`.
- Layout: `case/`, `input/` (forcing, obs), `tune/<id>/`, `runs/<id>/`, `post/<id>/`,
  `analysis/`, `manifest.jsonl` (every command that changed the project).

## 3. Inputs (when the case has no wrfinput/wrfbdy yet)

| need | command |
|---|---|
| forcing GRIB + geogrid/ungrib/metgrid | `wtune prep wps [--forcing NAME]` (GFS/FNL/CFSR chosen by date unless pinned; downloads on the login node, then one job for geogrid, ungrib and metgrid) |
| real.exe, link wrfinput/wrfbdy into `case/` | `wtune prep real [--replace]` |
| which GRIB source covers a window | `wtune metdata find --case-dir C`; `check` (HEAD requests); `fetch --cache-dir D` |
| static geography | `wtune prep geog list`; `fetch <dataset> --yes`; `bounds` (domain bounds and SRTM tiles); `srtm-fetch --bounds-from namelist.wps --out D --yes`; `import-dem <tif...> --name N`; `verify <dir>` |
| a fictional date (scenario forcing) | `wtune scenario list`; `scenario generate --generator G --start --end --bbox --out`; `scenario inspect <file>` |
| netCDF <-> ADIOS2 BP | `wtune convert inputs <case>` (cached, with manifest); `convert nc2bp` / `bp2nc` (`--check` round trip) |

## 4. Before running: check, predict, advise

| question | command |
|---|---|
| will this run_dir fail (time windows, iofields, physics data files, decomposition, disk)? | `wtune check <run_dir> [--ranks N] [--json]` |
| namelist values sane? rules of thumb | `wtune calc check namelist.input`; `wtune calc <dt\|cfl\|radt\|nest\|domain\|vert\|physics\|geog\|decomp\|io\|run\|units\|nudge\|scales\|les\|acoustic\|bdy\|movenest\|longrun\|spinup>` |
| how long will it take, per stage? | `wtune predict <case> --nodes N [--nx --ny --omp] [--include-io] [--json]` |
| a better prediction from measured rebuilt segments | `wtune predict-rebuild <case> --nodes N --nx --ny [--dry-run]` (jobs; needs `WPS_DIR`) |
| which WRF modes/workflows fit this case (WRFDA, nudging, moving nest, tc.exe, chem, hydro, surge) | `wtune advise <case> [--need chem\|hydro\|surge] [--json]` |
| physics diagnosis, numerics advice, scheme menu | `wtune physics --namelist F [--menu] [--no-probe]` |
| may this knob change the answer? | `wtune compliance list`, `compliance classify NAME...`, `compliance validate` |
| tune search space | `wtune config init <case>`, `config show F`, `config validate F` |

`advise`, `calc`, `physics` and `compliance` only advise; they never change files.

## 5. Tune, run, monitor

One command does all three: `wtune submit --dry-run` prints the steps, then
`wtune submit --detach` runs tune -> run -> post recipe (`-O0..3`, default 2; `--skip-tune`
uses the newest tune report; `--no-post`; `--resume N`). The log is `submit-<id>.log`;
`tune/<id>`, `runs/<id>`, `post/<id>` share one id.

Step by step:

1. `wtune tune -O2 [--minutes M] [--nodes search|N|auto]` writes `tune/<id>/tune_report.json`
   with the WINNER layout (nodes, ranks x omp, nx x ny, tiles, s/step). `-O0` submits no jobs
   (prediction only); `-O3` is the full search. `--minutes` is a hard cap.
2. `wtune run` (uses the newest tune report; or `--nodes N --nx --ny`) runs WRF in segments
   with restart hand-offs. Useful flags: `--resume N` (continue after a failed segment),
   `--cfl-rescue` (opt in: rerun a CFL/NaN-failed segment once at 0.75 dt, marked PHYSICS-GRAY; without it a CFL failure stops and writes the advised time_step),
   `--segment-minutes`, `--stop-at`, `--io adios2` (default) or `netcdf` (legacy),
   `--compress-history`, `--thompson-cache`, `--dry-run`.
3. `wtune watch runs/<id> --once --json` reads progress, remaining time, hangs, a nest that
   stopped, failures. Read only; poll it.
4. `wtune stop [dir] [--dry-run]` stops a running tune or run like Ctrl-C (cancels its own jobs,
   writes its report), or cancels the jobs a dead one left.

Report from a run: segment table in `runs/<id>/`, wall time vs `predict`, and any CFL rescue.

## 6. Post tools besides the recipe

The recipe, preview, brief, custom and check are in `wtune-post`. The single tools:

| need | command |
|---|---|
| domains, times, variables, missing frames | `wtune post inventory <run>` |
| a diagnostic at points (CSV) or in a box (NetCDF) | `wtune post extract <run> --vars V --out F [--points\|--box] [--times]` |
| one map | `wtune post plot <run> --field F [--level] [--times] --out D` |
| one skew-T | `wtune post sounding <run> --point LAT,LON --time T --out D` |
| TC track (TempestExtremes) and MET a-deck/b-deck | `wtune post tc <runs...> [--storm --hurdat2\|--bdeck]` |
| did a STRICT change move the answer? | `wtune post compare <run_a> <run_b> [--vars] [--json]` |
| observations | `wtune post fetch --list`; `post fetch <bdeck\|era5\|era5_monthly\|gpcp_monthly\|igra\|mrms\|nexrad\|stage4\|stations>` (login node) |
| 3D viewers | `wtune post cf3d <run>` (CF NetCDF for ParaView/Met.3D); `post vapor-bundle <run>` (VAPOR, on a workstation) |
| an HTML page from a folder of PNG/CSV | `wtune post report <dir>` |
| wrf-python diagnostics / tracking without a project | `wtune analyze diag --wrfout F --vars V --out D`; `analyze tc-track --wrfout F --hurdat2 H --storm S --out D` |

Any `post` command takes `--srun [--srun-minutes N]` to run as a Slurm step on a compute node.

## Internal commands

`screen`, `selftest` and `machine-profile` are developer tools (the analytic screen, the CONUS
calibration self-test, the summary of `machine/run_machine_profile.sh`). Do not use them for a
user's case unless asked.

## When something fails

1. Read the last `error:` lines of the command or `wtune watch --once`.
2. `wtune doctor` for environment problems; `wtune check <run_dir>` for a run_dir.
3. Report the cause and the exact command to the user. Do not retry a job-submitting command
   with changed resources without asking.
