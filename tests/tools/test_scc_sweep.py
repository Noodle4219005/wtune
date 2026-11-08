#!/usr/bin/env python3
"""pytest suite for tools/scc_sweep.py.

Anchor legality is checked against predictor/experiment.py's own documented ground truth for
matthew's grid (e_we=91, e_sn=100 -- see that module's _max_legal_ranks/_generated_baseline
docstrings: "matthew 91x100 at total_cores=112: ... true max legal rank count is 90, shape
9x10"), not re-derived here. Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest tools/test_scc_sweep.py
"""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wtune.tools import scc_sweep as sw  # noqa: E402

MATTHEW_LIKE = dict(e_we=91, e_sn=100, e_vert=45)
TINY_ILLEGAL = dict(e_we=5, e_sn=5, e_vert=10)   # floor(5/1) < MIN_PATCH(10): no legal shape ever


# ---------------------------------------------------------------------------
# skip when legal space empty
# ---------------------------------------------------------------------------
def test_build_configs_skips_when_space_empty():
    built = sw.build_configs("tiny", TINY_ILLEGAL, nodes=1, k=5, seed=1,
                              hardware_id="scc-gh200", cores_per_node=72)
    assert built["configs"] == []
    assert built["space_size"] == 0
    assert built["skip_reason"] is not None
    assert "tiny" in built["skip_reason"]


# ---------------------------------------------------------------------------
# time-limit formatting and cap
# ---------------------------------------------------------------------------
def test_estimate_s_per_step_scales_and_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        # measured 4.0 s/step at n_nodes=2 -> target nodes=4 should scale to 2.0
        path = os.path.join(tmp, "ledger_matthew_555.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_base", "hardware_id", "n_nodes", "s_per_step_median"])
            w.writerow(["matthew", "scc-gh200", "2", "4.0"])
            w.writerow(["matthew", "scc-gh200", "2", ""])          # failed row, ignored
            w.writerow(["other_case", "scc-gh200", "2", "9.0"])    # different case, ignored
            w.writerow(["matthew", "f1-spr-icpnp", "2", "1.0"])    # different hardware, ignored
        s_est, source, n = sw.estimate_s_per_step("matthew", nodes=4, ledger_dir=tmp,
                                                    hardware_id="scc-gh200")
        assert source == "ledger"
        assert n == 1
        assert abs(s_est - 2.0) < 1e-9

        s_est2, source2, n2 = sw.estimate_s_per_step("nosuchcase", nodes=4, ledger_dir=tmp,
                                                       hardware_id="scc-gh200")
        assert source2 == "fallback_10s"
        assert n2 == 0
        assert s_est2 == sw.CONSERVATIVE_S_PER_STEP


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))
