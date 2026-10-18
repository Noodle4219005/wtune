"""The segment job script: exclusive nodes, the shared signal-trap + node-hygiene preamble
(coldstart_sample, dispatched by campaign/sites/scc.sh), OMP env and mpirun binding as on scc,
last line 'SEGMENT rc=<rc> wall=<s> ok=<SUCCESS COMPLETE WRF count>'."""
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
            "source \"$CAMPAIGN/sites/$SITE.sh\"   # site_launch / site_hygiene / W",
            "source \"%s\"" % env_source,
            "W=${WRF_TREE:-$W}",
            "ulimit -s unlimited"]


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
                          campaign_dir: str = _CAMPAIGN_DIR) -> str:
    config = Config.coerce(config)
    lines = _header_lines(os.path.abspath(seg_dir), config.nodes, time_limit, job_name,
                          env_source or wtune.machine.env_source(), campaign_dir)
    lines += coldstart.render_signal_trap_and_hygiene_preamble(
        config.nodes, extra_lines_after_alloc_nodes=["STATUS=\"$SEG/segment_status.txt\"",
                                                     ": > \"$STATUS\""],
        record_label="SEGMENT")
    lines += ["cd \"$SEG\"",
              "for f in \"$W\"/run/*; do",
              "  b=$(basename \"$f\")",
              "  case \"$b\" in namelist.input|namelist.output|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;",
              "    *) [ -e \"$b\" ] || ln -sf \"$f\" \"$b\" ;; esac",
              "done",
              "export OMP_NUM_THREADS=%d" % config.omp,
              "unset KMP_BLOCKTIME"]
    lines.append("export OMP_PLACES=cores OMP_PROC_BIND=close" if config.omp > 1
                 else "unset OMP_PLACES OMP_PROC_BIND")
    return "\n".join(lines + _launch_lines(k, config)) + "\n"
