# wtune run: design notes

The behaviour rules of `wtune run` (the text that used to head `src/wtune/run/`; the code now
lives in the `src/wtune/run/` package, one topic per module).

```text
wtune/run/ -- segmented WRF executor with restart hand-offs (`wtune run`).

    wtune run CASE_DIR [--nodes N --nx X --ny Y] [--numtiles T] [--omp O]
        [--wall-limit HH:MM:SS | --segment-minutes M] [--stop-at YYYY-MM-DD_HH:MM:SS]
        [--prediction pred_rebuild.json] [--tune-report tune_report.json]
        [--restart-io-form 14|2] [--io adios2|netcdf] [--cfl-rescue] [--resume N]
        [--checkpoint-minutes M|auto] [--sidecar] --out OUT [--dry-run]

The layout (nodes, nproc_x/y, numtiles, omp) is the CLI's; flags not given come from
--tune-report's winner (tune_report_layout), and the node/nx/ny triple must come from one of them.
CASE_DIR is never modified and --out may not be inside it. The run [start, end] is the case's own
(run_* wins for d01, wtune.predict.get_d01_steps), cut at --stop-at when that is earlier (it must
fall on a whole d01 time step). The run is split into segments; segment k runs in OUT/seg_<k> as
its own Slurm job and hands off to k+1 through the restart files it writes at its end.

Segment planning (plan_run; boundaries are d01 step indices)
    ONE restart quantum L for the whole run (segments never set override_restart_timers, so the
    restart file carries every alarm's phase AND segment 0's restart interval to later segments --
    scc R1: with the override, d01's 60-min history alarm re-anchored at a 00:30 boundary and the
    01:00 frame was never written). L is whole minutes on the d01 step grid (quantum = lcm(dt,
    60 s); a nest step divides its parent's, so every domain's step divides it); every segment is
    one L block [jL, (j+1)L] from the run start, the last possibly shorter, and restart_interval =
    L in every segment (a segment ending on the grid, the last included, writes a restart).
    --segment-minutes M   L = M.
    --wall-limit W        L = the largest quantum multiple for which EVERY L-grid segment's
                          predicted wall <= 0.8 W (choose_restart_quantum), where predicted wall =
                          compute over the interval + init + restart read (k > 0) + restart write
                          (end on the grid) + history frames written inside the interval.
    neither               one segment.
    Restart cadence (restart_setting): the case's own restart_interval (WRF's _d/_h/_m/_s sum) is
    never switched off when the case asked for restart output inside the run: one segment keeps
    the case's restart keys untouched when that interval is < the run length (a 13-month case
    still writes its checkpoints), else restart_interval = RESTART_OFF_MIN; several segments keep
    it when it divides L (every boundary is then a restart time anyway), else restart_interval =
    L. Checkpoints (--checkpoint-minutes, for --resume): when the case's interval puts no
    restart strictly inside a segment, restart_interval = M -- auto (only with --resume > 0):
    the smallest multiple of the segment-cut unit (input record times, see I/O) >= run length
    / 8, which with segments must also divide L and be < L; an explicit M wins over the case
    and must be a multiple of that unit (and divide L); 0 writes none. The restart writes at
    the cadence are counted in every predicted wall, listed in the io_form 14 adios2.xml and
    get pre-created output dirs; the plan line prints the effective cadence.
    Compute over [a, b] = C(b) - C(a), C a prefix sum of tools/coldstart_sample.
    predicted_compute_total's per-step terms (the same arithmetic wtune.probe_plan.combine uses
    for the whole run; C(N) is asserted equal to its total). Its
    inputs come from --prediction's pred_rebuild.json (result.points at/s_normal, result.r_night/
    r_day, result.init_s, io.restart.per_write_s, io.history[].median_s) or from each
    --tune-report ranking row (points at/s_normal/r/n_rad; r_night/r_day pooled exactly like
    combine via wtune.probe_plan._resolve_r_rad_from_points). A -O report with no ranking points
    (0 prediction points) but a measured winner: its median s/step as a constant rate, replanned
    from observed steps like the no-prediction case. With neither: the uncalibrated
    cell-step model (wtune.tune._fallback_s_per_step) x FALLBACK_RADIATION_FACTOR. Without a
    measured `io` block, restart and history I/O scale from measured scc anchors by grid points x
    levels (fallback_overheads; conus25 restart per io_form, dorian2019 history). After each
    segment its observed overhead (job elapsed - sum d01) -- and, without a prediction, its mean
    d01 step -- replace the predictions in later segments' --time; never the L grid.

Segment namelist (render_segment_namelist; tools/coldstart_sample set_list & co. do the editing)
    start_* / end_* of EVERY active domain = the segment's start / end (yagi truth 1038: a nest
    whose end_* was not updated stopped early); run_days/hours/minutes = 0, run_seconds = segment
    length; nproc_x/nproc_y/numtiles of the segment's config; restart_interval = L minutes in
    every segment (restart_interval_d/_h/_s zeroed, _m set equal when present) unless
    restart_setting keeps the case's own (then every restart_interval* key is left as the case
    has it); io_form_restart per --restart-io-form; k > 0: restart=.true.;
    override_restart_timers is never set (a case value .true. is set .false.: the tuner's
    restart-start probes want the override, a chain must not have it). write_hist_at_0h_rst is
    never set (its default .false. avoids a duplicate history frame at the boundary); every
    other history setting is untouched. adios2_numaggregators, when the case sets it, follows
    the segment's rank count (configs._apply_adios2: it must equal the rank count).

I/O (--io, default adios2; 2026-10-03 user decision: netCDF is no longer used for WRF I/O)
    Every segment namelist gets io_form_history/restart/input/boundary = 14 and io_form = 14 for
    every aux input stream the case reads (configs.adios2_input_streams: auxinput4 when
    sst_update=1, gfdda/sgfdda when grid_fdda/grid_sfdda > 0, any auxinputN the namelist names
    with io_form_auxinputN > 0), io_form_auxhistN = 14 for every aux output the case writes
    (io_form_auxhistN > 0; `wtune convert bp2nc` turns one back into netCDF),
    adios2_numaggregators = the segment's rank count. WRF's
    io_adios2 opens the inname exactly (no suffix added), so each stream's inname becomes
    "<template>.bp" (input_inname="wrfinput_d<domain>.bp", bdy_inname="wrfbdy_d<domain>.bp",
    auxinput4_inname="wrflowinp_d<domain>.bp", ... -- campaign/configs._apply_adios2's
    convention) and each segment dir links wrfinput_d01.bp, wrfbdy_d01.bp, wrflowinp_d01.bp, ...
    to: CASE_DIR/<name>.bp if the case already has it, else the netCDF CASE_DIR/<name> converted
    once by wtune.bp.convert_inputs into <project>/input/bp/ (no project: OUT/_bp_inputs/),
    before segment 0 is submitted. adios2.xml = the case's own + wtune.bp.adios2_xml() + the BP4
    restart entries below. An aux stream whose file is missing keeps the case's io_form (a
    note). --io netcdf is the LEGACY escape hatch (WARNING): the case's io_form_* stay as they
    are, restart per --restart-io-form, as before 2026-10-03.
    Per-record inputs: WRF 4.8's ADIOS2 reader never selects a BP step, so a multi-record
    wrfbdy/wrflowinp/wrffdda BP yields record 1 for every read; convert_inputs writes those one BP
    per record (<cache>/<name>/<name>_<date>.bp). Each segment dir links every record file
    under its own name (wtune.bp.bp_input_links) and the namelist gets
    wtune.bp.per_record_namelist_edits (multi_bdy_files=.true. in &bdy_control, dated
    bdy_inname/auxinput4_inname/gfdda_inname, frames_per_auxinput4/10 = 1). WRF names each file
    by the clock time of the read, so a resumed segment must start on a record time: every cut
    is a multiple of lcm(whole-minute step quantum, interval_seconds, auxinput4_interval,
    gfdda_interval_m) from the run start (input_record_alignment; --segment-minutes is rounded
    DOWN to one with a note, shorter than one is an error; --wall-limit searches only those).
    Every segment start's record file is checked before segment 0 is submitted. Each attempt
    links a fresh copy made here, on the submitting host (seg dir/_bp_inputs, reflink; removed
    after success): NFS open stall, wiki 2026-10-03 adios2 gate. wrfinput is not copied (read
    once at init, outside the timed steps).

Segment run dir (setup_segment_dir): symlinks to every regular file of CASE_DIR except namelist.*,
rsl.*, wrfout*, wrfrst*, wrf.out (plus ADIOS2 *.bp input directories), every file the namelist
names (iofields_filename, sst_update's auxinput4 wrflowinp, fdda, auxinput*, tslist -- configs.
stage_namelist_files; also from a subdirectory of CASE_DIR), WRF_TREE/run/* not already present,
the previous segment's wrfrst_d<NN>_<start> for k > 0, a fresh namelist.input, the iofields
file(s) as a regular file with adjacent same-prefix lines merged when that changes them
(wtune.iofields: same masks, WRF re-parses the file at every I/O call), and for io_form 14
an adios2.xml selecting the BP4 engine for every restart file the segment reads or writes
(campaign/strict_campaign.sbatch write_adios2_restart_xml, same content; a case adios2.xml is
copied with these entries appended instead of being linked). History output stays in each
OUT/seg_<k>.

Segment job (render_segment_sbatch): exclusive nodes, --exclude=head[,WTUNE_SBATCH_EXCLUDE],
--time = 1.5 x predicted wall + 10 min rounded up to 5 min, capped at --wall-limit; the shared
signal-trap + node_hygiene reap+census preamble (coldstart_sample.render_signal_trap_and_hygiene_
preamble, dispatched by campaign/sites/scc.sh site_hygiene); OMP env as strict_campaign.sbatch
configure_omp_env does on scc; mpirun ppr/PE binding as sites/scc.sh site_launch; last line
'SEGMENT rc=<rc> wall=<s> ok=<SUCCESS COMPLETE WRF count>'.

Controller (run_controller): submit, wait (sacct), inspect with wtune.watch.snapshot. A segment is
done when watch says 'done' and (not last) every domain's wrfrst at its end exists. Otherwise, with
--cfl-rescue and a CFL line or NaN in the rsl, the SAME segment is re-run once per run from the
same restart with time_step = the largest integer in [0.5 dt, 0.75 dt] dividing the segment length
and L
(and the d01 history interval and radt when possible); the run is then PHYSICS-GRAY and later segments go
back to the case's time_step. Anything else stops with an error -- never a silent retry.
OUT/run_report.json is rewritten after every segment attempt.

Auto-resume (--resume N, default 1 per run; 0 disables): an infrastructure failure (sacct
NODE_FAIL/TIMEOUT/PREEMPTED/BOOT_FAIL, or a job that ended with no CFL/NaN/segfault/forrtl/FATAL
in any rsl; never CANCELLED) is resumed in OUT/seg_<k>_r<n> from the failed attempt's latest
valid restart (wtune.resume) to the segment end, same layout and time step.
--sidecar (wtune.sidecar): while a job runs, each valid restart is recorded in the report and
finished history frames are converted to netCDF in the post cache, on the job's idle cores.
```
