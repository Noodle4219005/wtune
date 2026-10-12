"""Golden snapshots of predictor/recommend.recommend and recommend.recommend_physics_gray, pinned
before the readability refactor split them into private helpers. The fixtures under
tests/fixtures/golden/recommend/ were written by the UNCHANGED code on the same synthetic ledgers
and namelists tests/predictor/test_recommend.py uses (imported from there, not copied, so the two
cannot drift). The comparison is on the JSON text with key order preserved: `--json` consumers
see key order, so a reordered result dict is a visible change too.

Regenerate (only when a behaviour change is intended) with:
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_recommend.py
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor"))

import recommend  # noqa: E402
import test_recommend as tr  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fixtures", "golden", "recommend")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"


def _normalize(obj):
    """JSON-safe copy with floats rounded to 10 significant digits (the snapshot pins logic, not
    the last ulp of a least-squares solve) and tuple/non-str keys made explicit."""
    if isinstance(obj, float):
        return float("%.10g" % obj) if obj == obj else "nan"
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else "%s:%r" % (type(k).__name__, k)): _normalize(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def _check_golden(name, result, tmp_path):
    """A str result (a formatted report) is snapshotted as text, anything else as JSON."""
    if isinstance(result, str):
        text, ext = result, ".txt"
    else:
        text, ext = json.dumps(_normalize(result), indent=1, default=repr), ".json"
    text = text.replace(str(tmp_path), "<TMP>")
    path = os.path.join(GOLDEN_DIR, name + ext)
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text + "\n")
    with open(path) as fh:
        expected = fh.read()
    assert text + "\n" == expected, "result differs from golden %s" % path


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


@pytest.fixture
def files(tmp_path):
    ledger = str(tmp_path / "runs.csv")
    tr.write_ledger_csv(tr.make_training_rows(), ledger)
    mixed = str(tmp_path / "runs_mixed_hw.csv")
    tr.write_ledger_csv(tr._case_rows("trainSame", tr.HARDWARE_ID_TRAIN, 901, 901, 45, 0.0)
                        + tr._case_rows("trainOther", tr.HARDWARE_ID_TRAIN, 850, 850, 40, 0.05)
                        + tr._case_rows("trainCross", "hw-cross-vendor", 905, 905, 45, 0.1),
                        mixed)
    phys = str(tmp_path / "physics_runs.csv")
    tr.write_ledger_csv(tr.make_physics_training_rows(), phys)
    acc = str(tmp_path / "physics_runs_accuracy.csv")
    tr.write_ledger_csv(tr.make_accuracy_training_rows(), acc)
    return dict(
        ledger=ledger, mixed=mixed, phys=phys, acc=acc,
        main=_write(tmp_path, "namelist.input", tr.MAIN_NAMELIST),
        small=_write(tmp_path, "namelist.input.small", tr.SMALL_NAMELIST),
        multi=_write(tmp_path, "namelist.input.multi", tr.MULTI_DOMAIN_NAMELIST),
        zero=_write(tmp_path, "namelist.input.zero", tr.ZERO_RUN_NAMELIST),
        neighbor=_write(tmp_path, "namelist.input.neighbor", tr.NEIGHBOR_NAMELIST),
        fine=_write(tmp_path, "namelist.input.finedx", tr.FINE_DX_NAMELIST),
        accuracy=_write(tmp_path, "namelist.input.accuracy", tr.ACCURACY_NAMELIST),
        probe=_write(tmp_path, "screen_synthtest.txt", tr.PROBE_SCREEN_TEXT),
    )


HW = tr.HARDWARE_ID_TRAIN

# name -> (namelist key, ledger key, kwargs) for recommend(); every branch of the warnings
# assembly plus the empty-candidate early return and the default (pairwise, falls back) model.
STRICT_CASES = {
    "main_ridge": ("main", "ledger", dict(case_name="synthtest", hardware_id=HW, model="ridge")),
    "main_default_model": ("main", "ledger", dict(case_name="synthtest", hardware_id=HW)),
    "main_point_no_intervals": ("main", "ledger", dict(
        case_name="synthtest", hardware_id=HW, model="catboost_point", no_intervals=True,
        top_k=3)),
    "no_hw_match": ("main", "ledger", dict(case_name="synthtest", hardware_id="hw-unknown",
                                           model="ridge")),
    "out_of_range": ("small", "ledger", dict(case_name="smallcase", hardware_id=HW,
                                             model="ridge")),
    "in_sample_by_name": ("main", "ledger", dict(case_name="trainA", hardware_id=HW,
                                                 model="ridge")),
    "holdout_same_case": ("main", "ledger", dict(case_name="trainA", hardware_id=HW,
                                                 model="ridge", holdout_case="trainA")),
    "holdout_zero_match": ("main", "ledger", dict(case_name="synthtest", hardware_id=HW,
                                                  model="ridge", holdout_case="nosuchcase")),
    "multi_domain_native": ("multi", "ledger", dict(case_name="multidom", hardware_id=HW,
                                                    model="ridge")),
    "multi_domain_max_dom_1": ("multi", "ledger", dict(case_name="multidom", hardware_id=HW,
                                                       model="ridge", max_dom=1)),
    "zero_run_minutes": ("zero", "ledger", dict(case_name="synthtest", hardware_id=HW,
                                                model="ridge")),
    "known_case_two_nodes": ("main", "ledger", dict(case_name="karina", hardware_id=HW,
                                                    model="ridge", nodes=2)),
    "mixed_hw_neighbors": ("neighbor", "mixed", dict(case_name="neighbortest", hardware_id=HW,
                                                     model="ridge")),
}


@pytest.mark.parametrize("name", sorted(STRICT_CASES))
def test_recommend_golden(name, files, tmp_path):
    nl, led, kwargs = STRICT_CASES[name]
    result = recommend.recommend(ledger_path=files[led], namelist_path=files[nl], **kwargs)
    _check_golden("recommend_" + name, result, tmp_path)


def test_recommend_unsupported_compliance_message(files):
    with pytest.raises(recommend.UnsupportedComplianceError) as exc:
        recommend.recommend(ledger_path=files["ledger"], namelist_path=files["main"],
                            case_name="synthtest", hardware_id=HW, compliance="PHYSICS-GRAY")
    assert str(exc.value) == (
        "--compliance=PHYSICS-GRAY recommendation needs the fidelity-risk model "
        "(predictor/validate.py fidelity_model_status/constrained_recommend) and is not "
        "implemented yet -- only STRICT is supported today; refusing to proceed rather than "
        "silently treating it as STRICT")


# name -> (namelist key, physics ledger keys, kwargs) for recommend_physics_gray().
PHYSICS_CASES = {
    "no_training_data": ("main", [], dict(case_name="synthtest", hardware_id=HW)),
    "synthetic_ledger": ("main", ["phys"], dict(case_name="synthtest", hardware_id=HW)),
    "cross_hardware": ("main", ["phys"], dict(case_name="synthtest",
                                              hardware_id="hw-never-trained-on")),
    "fine_dx": ("fine", [], dict(case_name="finecase", hardware_id=HW)),
    "in_sample": ("main", ["phys"], dict(case_name="phystrain0", hardware_id=HW)),
    "holdout": ("main", ["phys"], dict(case_name="phystrain0", hardware_id=HW,
                                       holdout_case="phystrain0")),
    "holdout_zero_match": ("main", ["phys"], dict(case_name="synthtest", hardware_id=HW,
                                                  holdout_case="nosuchcase")),
    "isc21_twin_override": ("main", ["phys"], dict(case_name="isc21", hardware_id=HW)),
    "explicit_decomp_labels_str": ("main", ["phys"], dict(
        case_name="synthtest", hardware_id=HW, decomp="C05",
        allowed_labels="WITHIN-FLOOR, SMALL-DRIFT", top_k=3)),
    "profile_shares": ("main", [], dict(
        case_name="synthtest", hardware_id=HW,
        profile_shares={"profile_radiation_share": 0.019, "profile_dynamics_share": 0.55})),
    "frozen_keys": ("main", ["phys"], dict(case_name="synthtest", hardware_id=HW,
                                           frozen_keys=["time_step"])),
    "accuracy": ("accuracy", ["acc"], dict(case_name="synthtest", hardware_id=HW,
                                           objective="accuracy")),
    "accuracy_no_data": ("accuracy", [], dict(case_name="synthtest", hardware_id=HW,
                                              objective="accuracy")),
    "lenient_gates": ("main", ["phys"], dict(
        case_name="synthtest", hardware_id=HW, max_infeasible_rate=1.0, max_dangerous_miss=1.0,
        min_speedup_pct=0.0, min_bucket_share=0.0)),
}


@pytest.mark.parametrize("name", sorted(PHYSICS_CASES))
def test_recommend_physics_gray_golden(name, files, tmp_path):
    nl, ledgers, kwargs = PHYSICS_CASES[name]
    result = recommend.recommend_physics_gray(
        ledger_path=files["ledger"], namelist_path=files[nl],
        physics_ledgers=[files[k] for k in ledgers], probe_screen=files["probe"], **kwargs)
    _check_golden("physics_gray_" + name, result, tmp_path)
    _check_golden("physics_gray_report_" + name, recommend.format_physics_report(result),
                  tmp_path)


def test_recommend_physics_gray_run_probe_golden(monkeypatch, files, tmp_path):
    monkeypatch.setattr(
        recommend, "run_probe_screen",
        lambda namelist_path, latlon=None, wps=None, wrfinput=None, **kw: tr.PROBE_SCREEN_TEXT)
    result = recommend.recommend_physics_gray(
        ledger_path=files["ledger"], namelist_path=files["main"], case_name="synthtest",
        hardware_id=HW, latlon="10.0,120.0", physics_ledgers=[files["phys"]])
    _check_golden("physics_gray_run_probe", result, tmp_path)


def test_recommend_physics_gray_observations_golden(monkeypatch, files, tmp_path):
    # No registered case declares observations today; stub both lookups (called through the
    # module globals) to pin the observed-skill warning and the obs_scores plumbing.
    monkeypatch.setattr(recommend, "case_observations",
                        lambda case_name, cases=None: {"kind": "station", "path": "/obs/x.csv"})
    monkeypatch.setattr(recommend, "observation_scores_for_case",
                        lambda case_name, wrfout, cases=None: {"P01": (0.25, "obs")})
    result = recommend.recommend_physics_gray(
        ledger_path=files["ledger"], namelist_path=files["accuracy"], case_name="synthtest",
        hardware_id=HW, physics_ledgers=[files["acc"]], probe_screen=files["probe"],
        objective="accuracy", obs_wrfout={"P01": "/runs/P01/wrfout_d01_*"})
    _check_golden("physics_gray_observations", result, tmp_path)


def test_recommend_physics_gray_errors(files):
    with pytest.raises(ValueError) as exc:
        recommend.recommend_physics_gray(
            ledger_path=files["ledger"], namelist_path=files["main"], case_name="synthtest",
            hardware_id=HW, probe_screen=files["probe"], objective="bogus")
    assert str(exc.value) == "unknown objective 'bogus' (choose from %s)" % ", ".join(
        recommend.OBJECTIVES)
    with pytest.raises(ValueError, match="SpeedyWeather probe"):
        recommend.recommend_physics_gray(
            ledger_path=files["ledger"], namelist_path=files["main"], case_name="synthtest",
            hardware_id=HW)
