"""The modules wtune_tune builds on, imported once: required ones directly, the prediction /
physics stack best-effort (None when not importable; callers check before use).

src/, src/tools and src/predictor are top-level import roots, as for every other src/ module.
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # src/
_TOOLS_DIR = os.path.join(_HERE, "tools")
_PREDICTOR_DIR = os.path.join(_HERE, "predictor")
for _p in (_TOOLS_DIR, _PREDICTOR_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scc_sweep as sweep        # noqa: E402  tools/: sbatch builder, sacct/submit runners, anchors
import wtune_case                # noqa: E402  build_profile: the pre-measurement PRIOR
import wtune_compliance as wc    # noqa: E402  decision 0056: assert_strict_only on the final report
import wtune_extensions          # noqa: E402  chem/hydro/WRFDA gates and OMP caps
import wtune_nodes               # noqa: E402  node-count choice from measured s/step
import wtune_config              # noqa: E402  --config YAML (PyYAML imported lazily)
import targets as pred_targets   # noqa: E402  predictor/: NOISE, the reproducibility floor

try:
    import wtune_predict          # full-run prediction, namelist parsing
except ImportError:
    wtune_predict = None

try:
    import wtune_probe_plan       # multi-point probe planning / combine (needs wtune_predict)
except ImportError:
    wtune_probe_plan = None

try:
    import wtune_io_model         # run-length I/O terms of the whole-run prediction
except ImportError:
    wtune_io_model = None

try:
    import coldstart_sample as coldstart   # tools/: ERA5 rebuild fetch/stage/score
except ImportError:
    coldstart = None

try:
    import wtune_physics_advice   # phase-3 candidates, speed-up, fidelity gate, advice table
except ImportError:
    wtune_physics_advice = None

try:
    import wtune_cost_drivers     # plan_points_for_case: the points a case type needs + their band
except ImportError:
    wtune_cost_drivers = None

campaign_configs = sweep.campaign_configs
decomp_space = sweep.decomp_space

__all__ = ["_HERE", "sweep", "wtune_case", "wc", "wtune_extensions", "wtune_nodes", "wtune_config",
           "pred_targets", "wtune_predict", "wtune_probe_plan", "wtune_io_model", "coldstart",
           "wtune_physics_advice", "wtune_cost_drivers", "campaign_configs", "decomp_space"]
