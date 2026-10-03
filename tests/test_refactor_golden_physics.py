"""Golden snapshots of predictor/physics.predict_physics_shortlist, pinned before the round-2
refactor split it. Inputs are test_physics.py's synthetic ledger (imported, not copied); floats
are rounded to 10 significant digits (the snapshot pins logic, not the last ulp of a solve).

Regenerate (only when a behaviour change is intended) with:
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_physics.py
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor"))
import physics  # noqa: E402
import test_physics as tp  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fixtures", "golden", "physics")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"


def _normalize(obj):
    if isinstance(obj, float):
        return float("%.10g" % obj) if obj == obj else "nan"
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def _check(name, result):
    text = json.dumps(_normalize(result), indent=1, default=repr) + "\n"
    path = os.path.join(GOLDEN_DIR, name + ".json")
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    with open(path) as fh:
        expected = fh.read()
    assert text == expected, "shortlist differs from golden %s" % path


@pytest.fixture(scope="module")
def ledger_rows():
    rows = tp.make_physics_ledger()
    physics.attach_change_context(rows)
    physics.add_perf_targets(rows)
    physics.add_fidelity_targets(rows)
    return rows


def _karina_p00(rows):
    return next(r for r in rows if r["case_base"] == "karina"
                and physics._template_of(r) == "P00" and "_twin" not in r["run_id"]
                and r["rep_index"] == 1)


CANDIDATES = [
    dict(template="P01", family="radiation", applicable=True,
         changes={"swint_opt": {"from": None, "to": 1}}),
    dict(template="P06", family="radiation", applicable=True, generic_rule_used=True,
         changes={"ra_lw_physics": {"from": 4, "to": 1}, "ra_sw_physics": {"from": 4, "to": 1}}),
    dict(template="P04", family="microphysics", applicable=True,
         changes={"mp_physics": {"from": 8, "to": 6}}),
    dict(template="P03", family="dynamics", applicable=True,
         changes={"time_step": {"from": 15, "to": 18}}),
    dict(template="P05", family="pbl", applicable=True,
         changes={"bl_pbl_physics": {"from": 1, "to": 5}}),
    dict(template="P02", applicable=True, changes={"radt": {"from": 12, "to": 24}}),
    dict(template="P99", family="dynamics", applicable=False, skip_reason="not on this grid",
         changes={}),
    dict(template="PXX", family="unknown", applicable=True,
         changes={"aer_opt": {"from": 0, "to": 1}}),
]

VARIANTS = {
    "default": dict(probe_verdicts={}),
    "probe_mixed": dict(probe_verdicts={"mp_physics": "REJECT", "ra_lw_physics": "CONFIRM",
                                        "radt": "PASS->WRF", "time_step": "CONFIRM"}),
    "lenient": dict(probe_verdicts=None, max_infeasible_rate=1.0, max_dangerous_miss=1.0,
                    max_template_large_rate=1.01, min_speedup_pct=50.0),
    "strict_speed": dict(probe_verdicts={}, min_speedup_pct=-20.0),
    "allowed_within_floor": dict(probe_verdicts={}, allowed_labels=("WITHIN-FLOOR",)),
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_shortlist_golden(name, ledger_rows):
    out = physics.predict_physics_shortlist(ledger_rows, _karina_p00(ledger_rows), CANDIDATES,
                                            **VARIANTS[name])
    _check("shortlist_%s" % name, out)


def test_shortlist_other_hardware_golden(ledger_rows):
    case = dict(_karina_p00(ledger_rows), hardware_id="no-such-hardware")
    _check("shortlist_other_hardware",
           physics.predict_physics_shortlist(ledger_rows, case, CANDIDATES[:3]))


def test_shortlist_no_training_data_golden(ledger_rows):
    _check("shortlist_no_training_data",
           physics.predict_physics_shortlist([], _karina_p00(ledger_rows), CANDIDATES))


def test_shortlist_too_few_rows_golden(ledger_rows):
    few = [r for r in ledger_rows if r["case_base"] == "karina"][:3]
    _check("shortlist_too_few_rows",
           physics.predict_physics_shortlist(few, _karina_p00(ledger_rows), CANDIDATES[:2]))
