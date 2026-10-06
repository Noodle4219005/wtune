"""The modules wtune_run builds on, imported once. src/, src/tools and src/predictor are
top-level import roots, as for every other src/ module."""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
for _p in (os.path.join(_HERE, "tools"), os.path.join(_HERE, "predictor"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import coldstart_sample as coldstart   # noqa: E402  tools/: namelist editing, predicted compute,
                                        # the shared sbatch signal-trap/hygiene preamble
import scc_sweep as sweep              # noqa: E402  tools/: sacct_states/submit_job/exclude_arg
import wtune_extensions                # noqa: E402  chem/hydro/WRFDA refusals and notes
import wtune_io_model as iom           # noqa: E402  per-format I/O anchors (history, restart)
import wtune_predict as wp             # noqa: E402  namelist parsing, calibration, sample points
import wtune_probe_plan as pp          # noqa: E402  _resolve_r_rad_from_points (combine's r rule)
import wtune_resume as resume          # noqa: E402  failure class, valid restarts (--resume)
import wtune_watch                     # noqa: E402  snapshot(rundir): done/failed/CFL/NaN

try:
    import wtune_tune                  # _fallback_s_per_step, _parse_slurm_elapsed
except ImportError:                    # pragma: no cover
    wtune_tune = None

__all__ = ["_HERE", "_REPO_ROOT", "_CAMPAIGN_DIR", "coldstart", "sweep", "wtune_extensions",
           "iom", "wp", "pp", "resume", "wtune_watch", "wtune_tune"]
