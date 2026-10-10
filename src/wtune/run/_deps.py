"""The modules wtune_run builds on, imported once. src/, src/wtune/tools and src/wtune/predictor are
top-level import roots, as for every other src/ module."""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # src/
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from wtune.tools import coldstart_sample as coldstart   # noqa: E402  tools/: namelist editing, predicted compute,
                                        # the shared sbatch signal-trap/hygiene preamble
from wtune.tools import scc_sweep as sweep              # noqa: E402  tools/: sacct_states/submit_job/exclude_arg
from wtune import extensions as wtune_extensions                # noqa: E402  chem/hydro/WRFDA refusals and notes
from wtune import io_model as iom           # noqa: E402  per-format I/O anchors (history, restart)
from wtune import predict as wp             # noqa: E402  namelist parsing, calibration, sample points
from wtune import probe_plan as pp          # noqa: E402  _resolve_r_rad_from_points (combine's r rule)
from wtune import resume as resume          # noqa: E402  failure class, valid restarts (--resume)
from wtune import watch as wtune_watch                     # noqa: E402  snapshot(rundir): done/failed/CFL/NaN

try:
    from wtune import tune as wtune_tune                  # _fallback_s_per_step, _parse_slurm_elapsed
except ImportError:                    # pragma: no cover
    wtune_tune = None

__all__ = ["_HERE", "_REPO_ROOT", "_CAMPAIGN_DIR", "coldstart", "sweep", "wtune_extensions",
           "iom", "wp", "pp", "resume", "wtune_watch", "wtune_tune"]
