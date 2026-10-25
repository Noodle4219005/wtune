"""The segment job script: exclusive nodes, the shared signal-trap + node-hygiene preamble
(coldstart_sample, dispatched by campaign/sites/scc.sh), OMP env and mpirun binding as on scc,
the 'SEGMENT rc=<rc> wall=<s> ok=<SUCCESS COMPLETE WRF count>' record, then the Thompson table
cache."""
from __future__ import annotations

import os
import shlex

import wtune.machine
from ._deps import _CAMPAIGN_DIR, coldstart, sweep
from .layout import Config


def _header_lines(seg_dir: str, nodes: int, time_limit: str, job_name: str, env_source: str,
                  campaign_dir: str) -> list[str]:
    return ["#!/usr/bin/env bash",
            "#SBATCH --job-name=%s" % job_name,
            ("#SBATCH -p %s -N %d --exclusive --ntasks-per-node=%d --time=%s %s"
             % (wtune.machine.partition(), nodes, wtune.machine.cores_per_node(), time_limit,
                sweep.scheduler_args())).rstrip(),
            "#SBATCH --signal=B:USR1@90",
            "#SBATCH --no-requeue",        # a requeued job would rerun beside the resume
            "#SBATCH --output=%s/segment-%%j.out" % seg_dir,
            "set -uo pipefail",
            "SEG=%s" % shlex.quote(seg_dir),
            "SITE=%s" % wtune.machine.site(),
            "CAMPAIGN=${CAMPAIGN:-%s}" % shlex.quote(os.path.abspath(campaign_dir)),
            "source \"$CAMPAIGN/sites/$SITE.sh\"   # site_env / site_launch / site_hygiene / W",
            "source \"%s\"" % env_source,
            "ulimit -s unlimited",
            # the probes' environment: on f1-arm site_env pins Open MPI's TCP to IPoIB; without
            # it a 5-node c34 segment ran 6.4x slower than tune measured (2026-10-21)
            "export WTUNE_ENV_SOURCE=\"%s\"" % env_source,
            "site_env",
            "W=${WRF_TREE:-$W}"]


def _launch_lines(k: int, config: Config) -> list[str]:
    """The site file's own launcher (site_launch), the one the probes were measured with."""
    ppn = -(-config.ranks // config.nodes)
    return ["CUR_WINDOW=%d" % k,
            "T0=$(date +%s)",
            "site_launch %d %d %d %s" % (config.ranks, ppn, config.omp, config.dist),
            "\"${SITE_LAUNCH_CMD[@]}\" &",
            "WRF_PID=$!",
            "wait \"$WRF_PID\"; rc=$?",
            "WRF_PID=\"\"; CUR_WINDOW=\"\"",
            "wall=$(( $(date +%s) - T0 ))",
            "ok=$(grep -c 'SUCCESS COMPLETE WRF' rsl.error.0000 2>/dev/null); ok=${ok:-0}",
            "echo \"SEGMENT rc=$rc wall=$wall ok=$ok\" | tee -a \"$STATUS\""]


def render_segment_sbatch(seg_dir: str, k: int, config: Config, time_limit: str, job_name: str,
                          env_source: str | None = None,
                          campaign_dir: str = _CAMPAIGN_DIR, thompson_cache: bool = False) -> str:
    config = Config.coerce(config)
    lines = _header_lines(os.path.abspath(seg_dir), config.nodes, time_limit, job_name,
                          env_source or wtune.machine.env_source(), campaign_dir)
    lines += coldstart.render_signal_trap_and_hygiene_preamble(
        config.nodes, extra_lines_after_alloc_nodes=["STATUS=\"$SEG/segment_status.txt\"",
                                                     ": > \"$STATUS\""],
        record_label="SEGMENT")
    lines.append("cd \"$SEG\"")
    if thompson_cache:
        # --thompson-cache: Thompson computes its qr_acr_q*/freezeH2O tables at init when absent
        # (~70 s on scc, under 1 s on f1-arm); they depend only on the build: the probes'
        # per-build cache. nullglob drops only patterns, so freezeH2O.dat is checked: a link
        # into an empty cache made WRF fail writing the table (f1-arm matthew 2026-10-25)
        lines += [coldstart.table_cache_line("$W"),
                  "shopt -s nullglob; for f in \"$TBL_CACHE\"/qr_acr_q*.dat \"$TBL_CACHE\"/freezeH2O.dat; "
                  "do [ -e \"$f\" ] || continue; [ -e \"$(basename \"$f\")\" ] || ln -s \"$f\" .; done; "
                  "shopt -u nullglob"]
    lines += ["for f in \"$W\"/run/*; do",
              "  b=$(basename \"$f\")",
              "  case \"$b\" in namelist.input|namelist.output|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;",
              "    *) [ -e \"$b\" ] || ln -sf \"$f\" \"$b\" ;; esac",
              "done",
              "export OMP_NUM_THREADS=%d" % config.omp,
              "unset KMP_BLOCKTIME"]
    lines.append("export OMP_PLACES=cores OMP_PROC_BIND=close" if config.omp > 1
                 else "unset OMP_PLACES OMP_PROC_BIND")
    # tables this segment computed go to the cache, only from a run that completed, via a
    # temporary name so a concurrent segment never links a partial file
    harvest = [] if not thompson_cache else ["if [ \"$rc\" -eq 0 ] && [ \"$ok\" -gt 0 ]; then",
               "  shopt -s nullglob; for f in qr_acr_q*.dat freezeH2O.dat; do",
               "    [ -L \"$f\" ] || [ -e \"$TBL_CACHE/$f\" ] || { mkdir -p \"$TBL_CACHE\" && "
               "cp -f \"$f\" \"$TBL_CACHE/.$f.$$\" && mv -f \"$TBL_CACHE/.$f.$$\" \"$TBL_CACHE/$f\"; }",
               "  done; shopt -u nullglob",
               "fi"]
    return "\n".join(lines + _launch_lines(k, config) + harvest) + "\n"
