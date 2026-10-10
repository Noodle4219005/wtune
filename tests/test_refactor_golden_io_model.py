"""Golden snapshot of wtune_io_model.run_length_io (full dict, key order included) and
breakdown_line on the namelists of test_wtune_io_model.py, pinned before the round-3 tidy-up
split run_length_io into per-term helpers.

Regenerate (only for an intended behaviour change):
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_io_model.py
"""
import json
import os

import test_wtune_io_model as tim
from wtune import io_model as iom

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "golden",
                      "io_model", "run_length_io.json")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
VARIANT = tim.SC26_P2.replace("io_form_history = 2", "io_form_history = 0").replace(
    "radt = 12,", "radt = 12, sst_update = 1,").replace(
    "history_interval = 60,", "history_interval = 60, auxinput4_interval = 180,")
SPEC = tim.CLIMATE_EA50 + " &bdy_control\n specified = .false.\n /\n"

CASES = {
    "climate": (tim.CLIMATE_EA50, {}), "climate-netcdf": (tim.CLIMATE_EA50, dict(io="netcdf")),
    "climate-iofields": (tim.CLIMATE_EA50, dict(iofields_calls=[500], call_s=0.001)),
    "climate-iofields-netcdf": (tim.CLIMATE_EA50, dict(io="netcdf", iofields_calls=[500])),
    "sc26": (tim.SC26_P2, {}), "sc26-netcdf": (tim.SC26_P2, dict(io="netcdf")),
    "sc26-run-s": (tim.SC26_P2, dict(run_s=7200)), "three": (tim.THREE_DOM, {}),
    "three-iofields": (tim.THREE_DOM, dict(iofields_calls=[10, 20, 0])),
    "three-netcdf": (tim.THREE_DOM, dict(io="14")), "three-io-nc": (tim.THREE_DOM, dict(io="nc")),
    "history-off-aux": (VARIANT, {}), "not-specified": (SPEC, {}),
}


def _collect(tmp_path):
    out = {}
    for cid, (text, kw) in CASES.items():
        nl = tim._nl(tmp_path, text, cid + ".input")
        r = iom.run_length_io(nl, **kw)
        out[cid] = {"keys": list(r), "values": r,
                    "line": iom.breakdown_line(1000.0, r, 60.0)}
    return json.loads(json.dumps(out))


def test_run_length_io_matches_golden(tmp_path):
    got = _collect(tmp_path)
    if REGEN:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as fh:
            json.dump(got, fh, indent=1)
            fh.write("\n")
    with open(GOLDEN) as fh:
        want = json.load(fh)
    assert got == want
