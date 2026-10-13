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
# anchor legality
# ---------------------------------------------------------------------------
def test_anchor_matches_documented_ground_truth_112_cores():
    # predictor/experiment.py's own documented example (_max_legal_ranks/_generated_baseline
    # docstrings): matthew's 91x100 grid at total_cores=112 -> max legal ranks is 90, shape 9x10.
    a = sw.build_anchor(MATTHEW_LIKE, total_cores=112, nodes=1, hardware_id="scc-gh200")
    assert a is not None
    assert (a["ranks"], a["nx"], a["ny"]) == (90, 9, 10)
    assert a["omp"] == 1
    assert a["numtiles"] == 1
    assert a["dist"] == "block"          # overridden from _generated_baseline's own 'cyclic'
    assert a["blocktime"] == "0"


def test_anchor_realistic_scc_single_node_72_cores():
    # N=1 node on scc (72 cores/node): 91x100 divides exactly (8x9=72), no downward scan needed.
    a = sw.build_anchor(MATTHEW_LIKE, total_cores=72, nodes=1, hardware_id="scc-gh200")
    assert a is not None
    assert a["ranks"] == 72
    assert (a["nx"], a["ny"]) in ((8, 9), (9, 8))
    assert a["dist"] == "block"
    assert a["numtiles"] == 1


def test_anchor_none_when_totally_illegal():
    assert sw.build_anchor(TINY_ILLEGAL, total_cores=72, nodes=1, hardware_id="scc-gh200") is None


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


def test_build_configs_nonempty_case_has_anchor_plus_sampled():
    built = sw.build_configs("matthew", MATTHEW_LIKE, nodes=1, k=5, seed=1,
                              hardware_id="scc-gh200", cores_per_node=72)
    assert built["skip_reason"] is None
    assert built["configs"][0]["id"] == "ANCHOR"
    assert built["n_anchor"] == 1
    assert 1 <= len(built["configs"]) <= 6   # anchor + up to 5 sampled (fewer if deduped)


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


def test_estimate_s_per_step_expands_unexpanded_ledger_dir(monkeypatch):
    # Regression for the bug where ledger_dir was never expanded inside estimate_s_per_step:
    # a caller passing the literal default DEFAULT_LEDGER_DIR="$HOME/wtune-runs/campaign" (or any
    # other unexpanded $VAR-style path) silently matched zero files via glob.glob and fell back to
    # the constant for every case, even with measured rows on disk.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ledger_isc21_777.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_base", "hardware_id", "n_nodes", "s_per_step_median"])
            w.writerow(["isc21", "scc-gh200", "2", "16.0"])
        monkeypatch.setenv("WTUNE_TEST_LEDGER_DIR", tmp)
        # nodes=4 == 2*n_measured -> ideal scaling halves 16.0 to 8.0
        s_est, source, n = sw.estimate_s_per_step(
            "isc21", nodes=4, ledger_dir="$WTUNE_TEST_LEDGER_DIR/", hardware_id="scc-gh200")
        assert source == "ledger"
        assert n == 1
        assert abs(s_est - 8.0) < 1e-9


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))


def test_sacct_states_never_reads_an_old_job_of_the_same_number():
    # f1-arm 2026-10-14: job ids started over, and `sacct -j 14841` answered with another
    # user's FAILED job of 2024 until slurmdbd had the new one
    seen = []

    def runner(cmd):
        seen.append(cmd)
        return "14841|PENDING|00:00:00\n"
    assert sw.sacct_states(["14841"], sacct_runner=runner) == {"14841": ("PENDING", "00:00:00")}
    assert seen[0][seen[0].index("-S") + 1] == "now-30days"
