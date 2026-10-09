"""Functions cut from src/wtune_tune/ when --rank-by predicted was retired (2026-10-10),
verbatim as at 1fa9fe8. One file, not wired to anything, not importable as is; the branches
removed from inside functions that stayed are in that commit's history."""


# --- src/wtune_tune/cli.py ---------------------------------------------------------------------

def _run_predicted_mode(ctx: Ctx, args: argparse.Namespace, o: _RunOptions, ns: _NodeChoice,
                        s_est: float, node_health: dict | None, out_dir: str) -> dict:
    """--rank-by predicted on the RUNNING-time budget with a hard wall cap: run_predicted, then
    print the report (+ advice) and write tune_report.json. Returns the report."""
    ctx.budget_mode = "running"
    ctx.stage2_budget_s = o.stage2_minutes * 60.0 if o.stage2_minutes is not None else None
    max_wall_min = getattr(args, "max_wall_minutes", None)
    ctx.max_wall_explicit = max_wall_min is not None
    ctx.max_wall_s = (max_wall_min if max_wall_min is not None
                      else MAX_WALL_FACTOR * o.minutes) * 60.0
    print("predicted mode: budget %.1f min of RUNNING time, hard wall-clock cap %.1f min"
          % (o.minutes, ctx.max_wall_s / 60.0))
    report = _predicted.run_predicted(ctx, ns.nodes, ns.nodes_note, s_est)
    if node_health:
        report["node_health"] = node_health
    _final_cancel_safety_net(ctx)
    text = _predicted.render_predicted_report_text(report) + _attach_advice(report)
    print("")
    print(text)
    report["winner"] = final_winner(report)
    write_report_json(report, out_dir)
    return report



# --- src/wtune_tune/budget.py ------------------------------------------------------------------

def describe_cost_model(m: dict) -> str:
    return ("per-run %.0fs = startup %.0fs + restart read %.0fs + %d steps x %.3fs x rad factor "
            "%.3f (r_rad %.2f every %s steps); spin-up %.0fs (%d steps, rad factor %.3f)"
            % (m["per_run_s"], m["startup_s"], m["restart_read_s"], m["steps"], m["s_step"],
               m["rad_factor_run"], m["r_rad"], m["stepra"] or "-", m["spin_s"], m["spin_steps"],
               m["rad_factor_spin"]))


def estimate_stage2_s(n_points: int, n_configs: int, steps: int, spinup_min: float, dt: float,
                      s_step: float, slots: int, startup_s: float, restart_read_s: float = 30.0,
                      stepra: int | None = None) -> float:
    """Expected RUNNING time of stage 2: one point job (spin-up + n_configs runs) per point,
    `slots` at a time, so ceil(n_points/slots) waves."""
    m = ranking_cost_model(startup_s, restart_read_s, steps, spinup_min * 60.0 / dt, stepra,
                           s_step)
    waves = int(math.ceil(float(n_points) / max(1, int(slots))))
    return waves * (m["spin_s"] + n_configs * m["per_run_s"])


def _reserve_s_step(ctx: Ctx) -> float | None:
    """s/step of the stage-2 reserve: median of stage 1's per-job anchor normalizers, else the
    prior."""
    vals = [j["s_anchor_job"] for j in ctx.stage1_jobs.values() if j.get("s_anchor_job")]
    return statistics.median(vals) if vals else ctx.s_prior


def _stage2_estimate(ctx: Ctx, n_configs: int, s_step: float) -> float:
    r = ctx.reserve
    return estimate_stage2_s(r["n_points"], n_configs, r["steps"], r["spinup_min"], r["dt"],
                             s_step, stage2_concurrency(ctx), ctx.startup_s, _restart_read_s(ctx),
                             stepra=r.get("stepra"))


def _reserve_k(ctx: Ctx, s_step: float) -> int:
    """Candidates the stage-2 reserve holds room for: TOP_K_STAGE2 once a stage-1 job reported;
    before that the largest k that still leaves one minimal stage-1 job (else the full reserve
    blocks stage 1 at concurrency 1; fit_stage2 trims stage 2 later)."""
    if ctx.stage1_jobs:
        return TOP_K_STAGE2
    # one minimal stage-1 job: pstar = spin-up + 3 ranking runs; t0 = 3 plain t0-STEPS runs
    min_job = _job_time_est(ctx, 3, 1, s_step)["total_run_s"]
    for k in range(TOP_K_STAGE2, 0, -1):
        if ctx.elapsed() + min_job + ctx.stage1_extra_reserve_s \
                + _stage2_estimate(ctx, k + 2, s_step) <= ctx.budget_s:
            return k
    return 1


def stage2_reserve_s(ctx: Ctx) -> float:
    """Running time stage 1 leaves for stage 2: _reserve_k candidates + ANCHOR/ANCHOREND at every
    point (0.0 outside predicted-mode stage 1)."""
    r = ctx.reserve
    if not _stage1_active(ctx) or not r:
        return 0.0
    s_step = _reserve_s_step(ctx)
    if not s_step:
        return 0.0
    return _stage2_estimate(ctx, _reserve_k(ctx, s_step) + 2, s_step)


def stage1_reserve_s(ctx: Ctx) -> float:
    """What stage-1 decisions hold back from --minutes: the stage-2 reserve (+ extra) when
    reserve_in_stage1 (pstar); 0.0 in stage1=t0, where stage 2 is a separate announced cost."""
    if not _stage1_active(ctx) or not ctx.reserve_in_stage1:
        return 0.0
    return stage2_reserve_s(ctx) + ctx.stage1_extra_reserve_s


def _stage1_active(ctx: Ctx) -> bool:
    """True during predicted-mode stage 1, in either stage1 mode."""
    return bool(ctx.ranking or ctx.stage1)


def _stage1_anchor(ctx: Ctx) -> dict | None:
    if ctx.ranking:
        return ctx.ranking["anchor"]
    return (ctx.stage1 or {}).get("anchor")


def describe_stage2_reserve(ctx: Ctx) -> str:
    """One log line: how the stage-2 reserve was sized (to judge it against reality)."""
    r = ctx.reserve
    s_step = _reserve_s_step(ctx)
    if not _stage1_active(ctx) or not r or not s_step:
        return "stage-2 reserve: none"
    m = ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), r["steps"],
                           r["spinup_min"] * 60.0 / r["dt"], r.get("stepra"), s_step)
    conc = stage2_concurrency(ctx)
    waves = int(math.ceil(float(r["n_points"]) / conc))
    return ("stage-2 %s %s = %d wave(s) [%d point job(s), %d concurrent: observed max %d "
            "stage-1 job(s) running at once, cap floor(max_nodes/nodes)=%d] x (spin-up + %d "
            "runs); %s"
            % ("reserve" if ctx.reserve_in_stage1 else
               "estimate (NOT held back from --minutes, stage1=t0)",
               _fmt_hms(stage2_reserve_s(ctx)), waves, r["n_points"], conc,
               ctx.stage1_max_running, compute_slots(ctx.max_nodes, ctx.nodes),
               _reserve_k(ctx, s_step) + 2, describe_cost_model(m)))


def _stage1_may_start(ctx: Ctx, job_run_s: float) -> tuple[bool, str | None]:
    """(ok, reason): a stage-1 job may start while elapsed + its run time + the stage-2 reserve
    fits the budget; always ok outside predicted mode."""
    if not _stage1_active(ctx):
        return True, None
    reserve = stage1_reserve_s(ctx)
    if ctx.elapsed() + job_run_s + reserve > ctx.budget_s:
        return False, ("elapsed %s + this job %s + stage-2 reserve %s > budget %s"
                       % (_fmt_hms(ctx.elapsed()), _fmt_hms(job_run_s), _fmt_hms(reserve),
                          _fmt_hms(ctx.budget_s)))
    return True, None


def _stage1_cutoff(ctx: Ctx) -> Callable[[], bool] | None:
    """pending_cutoff of stage-1 waits: true once elapsed + the stage-2 reserve reaches the
    budget; None outside predicted mode."""
    if not _stage1_active(ctx):
        return None
    return lambda: ctx.elapsed() + stage1_reserve_s(ctx) >= ctx.budget_s


def _ledger_case(ctx: Ctx) -> str:
    """Case id of the ledgers / RUNDIRs: the rebuilt point's in pstar, else ctx.case."""
    return ctx.ranking["case_id"] if ctx.ranking else ctx.case


def _ranking_wrap(ctx: Ctx, configs: list[dict]) -> list[dict]:
    """wrap_with_anchors in predicted-mode stage 1, else `configs` itself (same list)."""
    anchor = _stage1_anchor(ctx) if _stage1_active(ctx) else None
    return wrap_with_anchors(configs, anchor) if anchor else configs


def _padded_limit(total_run_s: float) -> tuple[float, str]:
    """(job_hours, time_limit): estimate_run_time's padding (x1.5 + 10 min, capped)."""
    padded = (total_run_s / 3600.0) * sweep.TIME_SAFETY_FACTOR + sweep.TIME_PAD_MIN / 60.0
    capped = min(padded, sweep.TIME_CAP_HOURS)
    return capped, sweep.campaign_experiment._format_hms(capped * 60.0)


def _job_time_est(ctx: Ctx, n_or_pairs: int | list, reps: int, s_est: float | None,
                  ranking: dict | None = None) -> dict:
    """estimate_run_time for ctx; with a ranking context the runs follow ranking_cost_model and
    the shared ANCHOR spin-up is added (raw total and padded --time, own SPINUP_TIMEOUT_S)."""
    ranking = ctx.ranking if ranking is None else ranking
    if isinstance(n_or_pairs, int):
        est = estimate_run_time(n_or_pairs, reps, ctx.steps if not ranking else ranking["steps"],
                                s_est=s_est, startup_s=ctx.startup_s)
    else:
        est = estimate_run_time(n_or_pairs, reps, ctx.steps if not ranking else ranking["steps"],
                                startup_s=ctx.startup_s)
    if not ranking:
        return est
    dt = float(ranking.get("dt") or 0) or None
    spin_steps = (ranking["spinup_minutes"] * 60.0 / dt) if dt else 0.0
    steps = ranking["steps"]
    pairs = ([(None, s_est)] * n_or_pairs if isinstance(n_or_pairs, int)
             else [(c, s) for c, s in n_or_pairs])
    runs = [ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), steps, spin_steps,
                               ranking.get("stepra"), s)["per_run_s"] for _c, s in pairs]
    m = ranking_cost_model(ctx.startup_s, _restart_read_s(ctx), steps, spin_steps,
                           ranking.get("stepra"), s_est if s_est else (pairs[0][1] if pairs else 0))
    total = reps * sum(runs) + m["spin_s"]
    est = dict(est, total_run_s=total, cost_model=m,
               timeout_s=max(sweep.TIMEOUT_MIN_S,
                             int(math.ceil(sweep.TIMEOUT_MULTIPLIER * max(runs or [0])))),
               spinup_timeout_s=max(sweep.TIMEOUT_MIN_S,
                                    int(math.ceil(sweep.TIMEOUT_MULTIPLIER * m["spin_s"]))))
    est["job_hours"], est["time_limit"] = _padded_limit(est["total_run_s"])
    return est



# --- src/wtune_tune/jobs.py --------------------------------------------------------------------

def cancel_unstarted(ctx: Ctx, job_ids: list, stage: str) -> None:
    """Stage-1 cutoff: scancel jobs that never started; their candidates go to ctx.not_measured."""
    print("%s: stage-1 cutoff reached -- cancelling %d job(s) that never started: %s "
          "(their candidates are not measured)" % (stage, len(job_ids), ",".join(job_ids)))
    ctx.scancel_runner(job_ids)
    ctx.cancelled_jobs.extend(job_ids)
    for j in job_ids:
        cids = [c for c in ctx.job_cids.get(j, []) if c not in ("ANCHOR", ANCHOR_END_ID)]
        ctx.not_measured.append(dict(job_id=j, cids=cids, stage=stage,
                                     reason="job still PENDING at the stage-1 cutoff"))


def _record_stage1_gates(ctx: Ctx, job_ids: list, rows: list[dict], label: str) -> None:
    """Predicted-mode stage 1: record (and warn about, never raise) every job's gates, plus its
    s_anchor_job = mean(ANCHOR, ANCHOREND) normalizer and whether foreign load or a missed restart
    excludes its candidates (ctx.ranking_gates, ctx.stage1_jobs). No-op outside stage 1."""
    if not _stage1_active(ctx):
        return
    require_restart = ctx.ranking is not None     # stage1=t0 jobs start at t0, not a restart
    for jid in job_ids:
        mine = [r for r in rows if (parse_run_id(r.get("run_id")) or {}).get("job") == str(jid)]
        cids = [parse_run_id(r["run_id"])["cid"] for r in mine]
        s_by_cid = {}
        for r in mine:
            s = sweep._to_float(r.get("s_per_step_median"))
            if s:
                s_by_cid[parse_run_id(r["run_id"])["cid"]] = s
        rec = evaluate_job_gates(jid, mine, s_by_cid, cids, require_restart=require_restart)
        reasons = []
        for r in mine:
            kv = _notes_kv(r.get("notes"))
            cid = parse_run_id(r["run_id"])["cid"]
            if kv.get("foreign_load") == "1":
                reasons.append("%s foreign_load=1 (%s)" % (cid, kv.get("foreign", "?")))
            if require_restart and kv.get("restart_used") != "1":
                reasons.append("%s restart_used=%s" % (cid, kv.get("restart_used", "missing")))
        a, e = s_by_cid.get("ANCHOR"), s_by_cid.get(ANCHOR_END_ID)
        s_anchor_job = ((a + e) / 2.0) if (a and e) else a
        if not s_anchor_job:
            reasons.append("no ANCHOR measurement to normalize by")
        rec.update(stage="stage1", label=label, s_anchor_job=s_anchor_job,
                   excluded=bool(reasons), exclude_reason="; ".join(reasons) or None)
        ctx.ranking_gates.append(rec)
        ctx.stage1_jobs[str(jid)] = dict(s_anchor_job=s_anchor_job, excluded=bool(reasons),
                                         reason=rec["exclude_reason"], label=label)
        if reasons:
            print("WARNING: %s job %s EXCLUDED from the stage-1 ranking: %s"
                  % (label, jid, rec["exclude_reason"]))
        elif not rec["ok"]:
            print("WARNING: %s job %s failed ranking gates (kept, anchor-normalized): %s"
                  % (label, jid, "; ".join(rec["failures"])))



# --- src/wtune_tune/rounds.py ------------------------------------------------------------------

def select_confirm_configs(ctx: Ctx, nodes: int, all_results: dict,
                           anchor_cid: str | None) -> list[dict]:
    """The configs of select_confirm_entries."""
    return [e["config"] for e in select_confirm_entries(ctx, nodes, all_results, anchor_cid)]


def fit_stage1_job(ctx: Ctx, candidates: list[dict],
                   s_est: float) -> tuple[list[dict] | None, list[dict], dict | None]:
    """Predicted mode: the longest prefix of `candidates` (sampling priority order) whose job --
    spin-up + ANCHOR + prefix + ANCHOREND -- still fits elapsed + job + stage-2 reserve <= budget.
    Returns (wrapped, dropped, time_est), or (None, candidates, estimate of the minimum)."""
    cands = [c for c in candidates if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    est = None
    for n in range(len(cands), 0, -1):
        wrapped = wrap_with_anchors(cands[:n], _stage1_anchor(ctx))
        est = _job_time_est(ctx, len(wrapped), 1, s_est)
        ok, _why = _stage1_may_start(ctx, est["total_run_s"])
        if ok:
            return wrapped, cands[n:], est
    return None, cands, est


def _submit_ranking_wave(ctx: Ctx, configs: list[dict], slots: int, s_est: float,
                         name_prefix: str) -> Wave:
    """Predicted-mode stage-1 wave: candidates dealt over min(slots, #candidates) slots, each
    with its own ANCHOR/ANCHOREND and SHRUNK to fit (fit_stage1_job; trimmed ones recorded); the
    plan is logged before anything is submitted. Same return as _submit_round0_wave."""
    cands = [c for c in configs if c.get("id") not in ("ANCHOR", ANCHOR_END_ID)]
    n_slots = max(1, min(int(slots), len(cands)))
    plan = []
    for i, chunk in enumerate(chunk_round_robin(cands, n_slots)):
        wrapped, dropped, est = fit_stage1_job(ctx, chunk, s_est)
        plan.append((i, wrapped, dropped, est))
    print("%s plan: %d slot(s) (max %d concurrent = floor(max_nodes/nodes)), candidates per slot "
          "%s, expected %s per job; elapsed %s, %s"
          % (name_prefix, sum(1 for p in plan if p[1]), slots,
             [len(p[1]) - 2 if p[1] else 0 for p in plan],
             ", ".join(_fmt_hms(p[3]["total_run_s"]) if p[1] else "skip" for p in plan),
             _fmt_hms(ctx.elapsed()), describe_stage2_reserve(ctx)))
    if plan and plan[0][3] and plan[0][3].get("cost_model"):
        print("%s cost model: %s" % (name_prefix, describe_cost_model(plan[0][3]["cost_model"])))
    job_ids, configs_by_cid, job_configs = [], {}, {}
    for i, wrapped, dropped, est in plan:
        for c in dropped:
            ctx.stage1_trimmed.append(dict(cid=c["id"], stage=name_prefix,
                                           reason="trimmed to fit the budget + stage-2 reserve"))
        if wrapped is None:
            print("%s slot %d: SKIP -- even ANCHOR + 1 candidate + ANCHOREND (%s) does not fit: %s"
                  % (name_prefix, i, _fmt_hms(est["total_run_s"]) if est else "-",
                     _stage1_may_start(ctx, est["total_run_s"])[1] if est else "no candidate"))
            continue
        if dropped:
            print("%s slot %d: trimmed %d candidate(s) to fit: %s"
                  % (name_prefix, i, len(dropped), ", ".join(c["id"] for c in dropped)))
        job_id, _path, _te = submit_configs_job(ctx, wrapped, reps=1,
                                                  name_suffix="%s-s%d" % (name_prefix, i),
                                                  s_est=s_est, time_est=est)
        job_ids.append(job_id)
        job_configs[job_id] = wrapped
        for c in wrapped:
            configs_by_cid[c["id"]] = c
        print("%s slot %d: submitted job %s (%d configs, expected %s)"
              % (name_prefix, i, job_id, len(wrapped), _fmt_hms(est["total_run_s"])))
    states = _jobs.wait_for_jobs(ctx, job_ids, name_prefix, pending_cutoff=_stage1_cutoff(ctx))
    rows = ctx.ledger_reader(_ledger_case(ctx), job_ids)
    _record_stage1_gates(ctx, job_ids, rows, name_prefix)
    return job_ids, configs_by_cid, rows, job_configs, states


def measure_ranking_overhead(ctx: Ctx, job_configs: dict, states: dict,
                             wave_results: dict) -> tuple[float | None, int]:
    """Predicted-mode measure_startup_s: per-run overhead = (elapsed - compute) / (runs + 1), the
    compute including the ANCHOR spin-up and radiation factors; it replaces startup_s + restart
    read together. (None, 0) without a usable COMPLETED job."""
    r = ctx.ranking
    spin_steps = r["spinup_minutes"] * 60.0 / float(r["dt"])
    f_run, f_spin = rad_factor(r["steps"], r.get("stepra")), rad_factor(spin_steps, r.get("stepra"))
    tot_e = tot_c = 0.0
    n = 0
    for jid, configs in job_configs.items():
        state, elapsed_str = states.get(jid, ("UNKNOWN", ""))
        el = _parse_slurm_elapsed(elapsed_str)
        a = wave_results.get("ANCHOR")
        if state != "COMPLETED" or el is None or not a:
            continue
        comp, cnt = spin_steps * a["median"] * f_spin, 0
        for c in configs:
            w = wave_results.get(c["id"])
            if w:
                comp += r["steps"] * w["median"] * f_run
                cnt += 1
        if cnt:
            tot_e += el
            tot_c += comp
            n += cnt + 1
    if not n:
        return None, 0
    return max(0.0, (tot_e - tot_c) / n), n



# --- src/wtune_tune/prediction.py --------------------------------------------------------------

def build_ranking_sbatch_command(case: str, nodes: int, cores_per_node: int, time_limit: str,
                                 timeout_s: int, reps: int, steps: int, configs_path: str,
                                 out_dir: str, spinup_minutes: Any, env_source: str | None = None,
                                 sbatch_script: str | None = None,
                                 campaign_dir: str | None = None,
                                 extra_export: list | None = None) -> str:
    """sbatch command of a RANKING_START=spinup trajectory job: scc_sweep.build_sbatch_command's
    shape + SPINUP_MINUTES (':'-separated -- a ',' would split --export; ValueError), CAMPAIGN
    when given, then extra_export [(KEY, VALUE)]."""
    env_source = env_source or sweep.DEFAULT_ENV_SOURCE
    sbatch_script = sbatch_script or sweep.DEFAULT_SBATCH_SCRIPT
    total_cores = nodes * cores_per_node
    export = ("ALL,SITE=scc,WTUNE_TOTAL_CORES=%d,CASE=%s,MODE=timing,REPS=%d,STEPS=%d,"
              "TIMEOUT_S=%d,CONFIG_FILE=%s,RANKING_START=spinup,SPINUP_MINUTES=%s"
              % (total_cores, case, reps, steps, timeout_s, configs_path, spinup_minutes))
    if "," in str(spinup_minutes):
        raise ValueError("SPINUP_MINUTES %r contains ',' which sbatch --export would split; "
                         "use ':'" % (spinup_minutes,))
    campaign_dir = campaign_dir or sweep.campaign_dir_for(sbatch_script)
    if campaign_dir:
        export += ",CAMPAIGN=%s" % campaign_dir
    export += "".join(",%s=%s" % kv for kv in extra_export or [])
    sbatch = ("sbatch -p compute -N %d --ntasks-per-node=%d --exclusive %s "
              "--time=%s --job-name=c3-traj-%s-n%d --output=%s/c3traj-%%j.out --export=%s %s"
              % (nodes, cores_per_node, sweep.exclude_arg(), time_limit, case, nodes, out_dir, export,
                 sweep._expand(sbatch_script)))
    return "source %s; %s" % (sweep._expand(env_source), sbatch)


def run_trajectory_points(ctx: Ctx, points: list, best_config: dict, nodes: int, run_start: Any,
                          stepra: int, out_dir: str, sbatch_script: str | None = None,
                          campaign_dir: str | None = None) -> tuple[list, str] | list:
    """Measure the trajectory/restart points in ONE job of the winner's layout (REPS = points,
    SPINUP_MINUTES = each point's t0-relative minute; rep r -> points[r-1]). Returns
    ([PointMeasurement] in point order, job id); RuntimeError on any infrastructure failure."""
    if not points:
        return []
    minute_list = [int(round((p.at - run_start).total_seconds() / 60.0)) for p in points]
    reps = len(points)
    cfg = dict(id="ANCHOR", ranks=best_config["ranks"], omp=best_config["omp"],
               nx=best_config["nx"], ny=best_config["ny"], numtiles=best_config["numtiles"],
               dist=best_config["dist"], blocktime=best_config.get("blocktime", 0),
               patch_x=None, patch_y=None, numtiles_x=0, numtiles_y=0, tile_sz_x=0,
               tile_sz_y=0, tile_strategy=0, config_source="anchor")
    os.makedirs(out_dir, exist_ok=True)
    configs_path = os.path.join(out_dir, "configs_trajectory.json")
    with open(configs_path, "w") as f:
        json.dump({"configs": [cfg]}, f)
    steps = points[0].steps
    # watchdog: 20 s per simulated anchor spin-up minute, floor 1800 s (job 1211 postmortem)
    timeout_s = max(1800, max(minute_list) * 20)
    time_min = max(45, timeout_s // 60 + 15)
    time_limit = "%02d:%02d:00" % (time_min // 60, time_min % 60)
    cmd_str = build_ranking_sbatch_command(
        ctx.case, nodes, ctx.cores_per_node, time_limit, timeout_s, reps, steps,
        os.path.abspath(configs_path), out_dir, ":".join(str(m) for m in minute_list),
        env_source=ctx.env_source, sbatch_script=sbatch_script, campaign_dir=campaign_dir,
        extra_export=_jobs.fresh_input_export(ctx, ctx.case, out_dir,
                                              2 * reps + 1))   # spin-up + run per point
    job_id = sweep.submit_job(cmd_str, sbatch_runner=ctx.sbatch_runner)
    ctx.owned_jobs.append(job_id)
    print("C3 trajectory job submitted: %s (targets=%s minutes)" % (job_id, minute_list))
    states = _jobs.wait_for_jobs(ctx, [job_id], "c3-trajectory")
    state = states.get(job_id, ("UNKNOWN", ""))[0]
    if state != "COMPLETED":
        raise RuntimeError("C3 trajectory job %s ended in state %s (expected COMPLETED) -- "
                            "infrastructure failure, stop and diagnose read-only" % (job_id, state))
    _n, _run_s, dt, _note = wtune_predict.get_d01_steps(
        wtune_predict.parse_namelist(ctx.namelist_path))
    measurements = []
    for i, p in enumerate(points):
        rep = i + 1
        rundir = os.path.join(_RANKING_RUNDIR_ROOT, "%s_ANCHOR_r%d_%s" % (ctx.case, rep, job_id))
        steps_path = os.path.join(rundir, "steps.txt")
        if not os.path.exists(steps_path) or os.path.getsize(steps_path) == 0:
            raise RuntimeError("C3 trajectory job %s: %s missing/empty for point %s (rep %d)"
                                % (job_id, steps_path, p.at.isoformat(), rep))
        g0 = wtune_probe_plan.first_global_step_from_rsl(
            os.path.join(rundir, "rsl.error.0000"), run_start, dt)
        measurements.append(wtune_probe_plan.measurement_from_steps_file(
            steps_path, stepra, p.at, p.reason, p.source, first_global_step=g0))
    return measurements, job_id


def run_c3_prediction(ctx: Ctx, best_config: dict, nodes: int, out_dir: str,
                      sbatch_script: str | None = None, campaign_dir: str | None = None,
                      grib_dir: str | None = None) -> tuple:
    """Plan the winner's points, measure trajectory/restart points (one job) and rebuild points
    (one job per point) -- one allocation per source type (decision 0054) -- and combine() them.
    Returns (result, points, notes, job_ids); raises on any infrastructure failure."""
    nl = wtune_predict.parse_namelist(ctx.namelist_path)
    run_start, run_end, dt, stepra = _run_window(nl)
    spinup_s = float(ctx.probe_cfg.get("spinup_min") or 0) * 60.0
    case_dir = ctx.case_json["prep_dir"]
    cost_series = _cost_series_for_ctx(ctx, run_start, run_end, case_dir, grib_dir)
    points, notes = wtune_probe_plan.plan_points(
        nl, run_start, run_end, dt, stepra, spinup_s, ctx.probe_cfg, case_dir=case_dir,
        indicators=cost_series)

    traj_points = [p for p in points if p.source in ("trajectory", "restart")]
    rebuild_points = [p for p in points if p.source == "rebuild"]

    job_ids = []
    measurements: list = []
    if traj_points:
        out_traj = os.path.join(out_dir, "c3_trajectory")
        m, jid = run_trajectory_points(ctx, traj_points, best_config, nodes, run_start, stepra,
                                        out_traj, sbatch_script=sbatch_script,
                                        campaign_dir=campaign_dir)
        measurements.extend(m)
        job_ids.append(jid)
    if rebuild_points:
        out_rebuild = os.path.join(out_dir, "c3_rebuild")
        spinup_min = ctx.probe_cfg.get("spinup_min") or 60
        m, jid = run_rebuild_points(ctx, rebuild_points, best_config, nodes, case_dir, out_rebuild,
                                     run_start, spinup_min=spinup_min, grib_dir=grib_dir)
        measurements.extend(m)
        job_ids.append(jid)

    calibration = _drift_calibration(ctx)
    sample_points, _geom_src = wtune_predict.read_wrfinput_sample_points(case_dir)
    result = wtune_probe_plan.combine(
        measurements, run_start, run_end, dt, stepra, calibration=calibration,
        sample_points=sample_points, case_dir=case_dir, nl=nl, ranks=best_config.get("ranks"),
        nodes=nodes, cost_series=cost_series)
    return result, points, notes, job_ids


def _match_points(res_points: list | None, wrapped: list[dict]) -> list[dict]:
    """plan_points_for_case's points (indices, objects with .at, datetimes or dicts) mapped onto
    the planned points `wrapped` [{index, point}]."""
    out = []
    for item in res_points or []:
        at = item if isinstance(item, datetime.datetime) else None
        idx = item if isinstance(item, int) and not isinstance(item, bool) else None
        if isinstance(item, dict):
            idx, at = item.get("index"), item.get("at")
        elif hasattr(item, "at"):
            at = item.at
        for w in wrapped:
            if (idx is not None and w["index"] == idx) or (at is not None and w["point"].at == at):
                out.append(w)
                break
    return out
