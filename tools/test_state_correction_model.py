#!/usr/bin/env python3
"""Offline tests for tools/state_correction_model.py. No network access. The pure-numpy pieces
(ridge/LOCO/kNN/destagger/RH) are exercised unconditionally; anything that round-trips through a
real netCDF file is gated on netCDF4 being importable in THIS interpreter and skipped with a
clear reason otherwise (the cluster env has netCDF4 1.7; this dev box may not).

Run with:
  python3 -m pytest -q tools/test_state_correction_model.py
"""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import state_correction_model as scm  # noqa: E402

try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    HAVE_NETCDF4 = False


# ---------------------------------------------------------------------------
# 1. destagger / restagger exact round trip
# ---------------------------------------------------------------------------

class TestStagger(unittest.TestCase):
    def test_round_trip_1d(self):
        mass = np.array([1.0, 2.0, 3.0, 4.0])
        stag = scm.restagger(mass, axis=0)
        self.assertEqual(stag.shape, (5,))
        back = scm.destagger(stag, axis=0)
        np.testing.assert_allclose(back, mass, atol=0, rtol=0)

    def test_round_trip_2d_axis1(self):
        mass = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        stag = scm.restagger(mass, axis=1)
        self.assertEqual(stag.shape, (2, 4))
        back = scm.destagger(stag, axis=1)
        np.testing.assert_allclose(back, mass, atol=0, rtol=0)

    def test_destagger_simple_average(self):
        stag = np.array([0.0, 2.0, 4.0])
        mass = scm.destagger(stag, axis=0)
        np.testing.assert_allclose(mass, [1.0, 3.0])

    def test_restagger_avg_edges_and_interior(self):
        mass = np.array([1.0, 2.0, 3.0])
        stag = scm.restagger_avg(mass, axis=0)
        # edges = nearest neighbor, interior = mean of neighbors
        np.testing.assert_allclose(stag, [1.0, 1.5, 2.5, 3.0])


# ---------------------------------------------------------------------------
# 2. RH computation vs an independently-written reference
# ---------------------------------------------------------------------------

class TestRH(unittest.TestCase):
    def test_matches_hand_computed(self):
        theta, qv, p = 300.0, 0.01, 100000.0
        temp = theta * (p / 1.0e5) ** 0.2854
        es = 611.2 * math.exp(17.67 * (temp - 273.15) / (temp - 29.65))
        qs = 0.622 * es / max(p - es, 1.0)
        expected_rh = qv / qs

        got_rh = scm.relative_humidity(theta, qv, p, clip=None)
        self.assertAlmostEqual(float(got_rh), expected_rh, places=10)
        got_temp = scm.temp_from_theta(theta, p)
        self.assertAlmostEqual(float(got_temp), temp, places=10)

    def test_clip_bounds(self):
        # absurdly moist column -> RH clipped to 1.2, never negative
        rh = scm.relative_humidity(200.0, 0.5, 100000.0, clip=1.2)
        self.assertLessEqual(float(rh), 1.2)
        rh2 = scm.relative_humidity(400.0, 1e-9, 100000.0, clip=1.2)
        self.assertGreaterEqual(float(rh2), 0.0)


# ---------------------------------------------------------------------------
# 3. LOCO harness never trains on the held-out case (spy)
# ---------------------------------------------------------------------------

class TestLOCOSpy(unittest.TestCase):
    def test_never_sees_holdout_group(self):
        rng = np.random.default_rng(0)
        n = 40
        X = rng.normal(size=(n, 2))
        y = np.arange(n, dtype=float)          # every y value uniquely identifies its row
        groups = np.array(["c%d" % (i % 4) for i in range(n)])
        y_to_group = {float(y[i]): groups[i] for i in range(n)}

        # spy fit_fn: records which groups its training labels came from; the real
        # loco_predict must never hand it a row from the case currently held out.
        seen_by_call = []

        def fit_spy(Xtr, ytr):
            seen_by_call.append({y_to_group[float(v)] for v in ytr})
            return {"mean": float(ytr.mean())}

        def predict_spy(model, Xte):
            return np.full(len(Xte), model["mean"])

        preds = scm.loco_predict(X, y, groups, fit_spy, predict_spy)

        self.assertEqual(len(preds), n)
        self.assertEqual(len(seen_by_call), len(set(groups.tolist())))
        for held, seen_groups in zip(sorted(set(groups.tolist())), seen_by_call):
            self.assertNotIn(held, seen_groups)


# ---------------------------------------------------------------------------
# 4. ridge LOCO recovers an exact constant offset (truth T = input T + 0.5)
# ---------------------------------------------------------------------------

class TestRidgeExactOffset(unittest.TestCase):
    def test_recovers_constant_shift(self):
        rng = np.random.default_rng(1)
        n_cases = 5
        rows_per_case = 12
        X = rng.normal(size=(n_cases * rows_per_case, 4))
        y = np.full(n_cases * rows_per_case, 0.5)   # truth - input == +0.5 EXACTLY, always
        groups = np.array(["case%d" % (i // rows_per_case) for i in range(len(y))])

        preds = scm.loco_predict(X, y, groups, scm.fit_ridge_nested, scm.ridge_predict_model)
        baseline_rmse = scm._rmse(np.zeros_like(y), y)
        model_rmse = scm._rmse(preds, y)

        self.assertAlmostEqual(baseline_rmse, 0.5, places=9)
        self.assertLess(model_rmse, 1e-8)
        np.testing.assert_allclose(preds, 0.5, atol=1e-8)
        self.assertTrue(scm.is_improved(model_rmse, baseline_rmse, n_cases=n_cases))


# ---------------------------------------------------------------------------
# 5. condensate two-part model: cloud present iff RH > 0.95
# ---------------------------------------------------------------------------

class TestCondensateTwoPart(unittest.TestCase):
    def test_classifier_beats_zero_baseline_csi(self):
        # Bimodal rh per case (dry cluster well below 0.95, moist cluster well above it) so the
        # 0/1-regressed ridge classifier has a clearly separable signal to threshold at 0.5 --
        # a uniform rh spread with only ~10%% positives lets a linear probability fit stay
        # under 0.5 everywhere and would make this test flaky, not the model being tested.
        rng = np.random.default_rng(2)
        n_cases = 5
        rows_per_case = 60
        rh_chunks, group_chunks = [], []
        for c in range(n_cases):
            half = rows_per_case // 2
            dry = rng.uniform(0.5, 0.8, size=rows_per_case - half)
            moist = rng.uniform(0.96, 1.0, size=half)
            rh_chunks.append(np.concatenate([dry, moist]))
            group_chunks.extend(["case%d" % c] * rows_per_case)
        rh = np.concatenate(rh_chunks)
        X = rh.reshape(-1, 1)
        present = rh > 0.95
        y_amount = np.where(present, 2e-4, 0.0)
        groups = np.array(group_chunks)
        n = len(rh)

        preds = scm.loco_predict(X, y_amount, groups, scm.fit_two_part, scm.two_part_predict)
        pred_present = preds > scm.CONDENSATE_THRESHOLD
        truth_present = y_amount > scm.CONDENSATE_THRESHOLD

        baseline_csi = scm._csi(np.zeros_like(truth_present, dtype=bool), truth_present)
        model_csi = scm._csi(pred_present, truth_present)

        self.assertEqual(baseline_csi, 0.0)
        self.assertGreater(model_csi, baseline_csi)
        self.assertTrue(scm.is_improved_condensate(model_csi, baseline_csi, n_cases=n_cases))


# ---------------------------------------------------------------------------
# 6. decision-rule edge cases
# ---------------------------------------------------------------------------

class TestDecisionRule(unittest.TestCase):
    def test_too_few_cases_never_improved(self):
        self.assertFalse(scm.is_improved(0.01, 1.0, n_cases=3))
        self.assertFalse(scm.is_improved_condensate(0.9, 0.0, n_cases=3))

    def test_zero_baseline_never_improved(self):
        # a perfect (zero-error) baseline leaves no headroom for a "20%% better" model --
        # must never be reported as improved, not even by a model that ties it.
        self.assertFalse(scm.is_improved(0.0, 0.0, n_cases=10))


# ---------------------------------------------------------------------------
# 7. netCDF4-gated: grid check, build-pairs, apply (skipped when netCDF4 unavailable)
# ---------------------------------------------------------------------------

NZ, NY, NX = 4, 5, 6
TIMES = "2026-01-01_00:00:00"


def _write_wrf_like(path, seed, dx=3000.0):
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w") as ds:
        ds.setncattr("DX", dx)
        ds.setncattr("DY", dx)
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", NY)
        ds.createDimension("south_north_stag", NY + 1)
        ds.createDimension("west_east", NX)
        ds.createDimension("west_east_stag", NX + 1)
        ds.createDimension("bottom_top", NZ)
        ds.createDimension("bottom_top_stag", NZ + 1)
        ds.createDimension("soil_layers_stag", 4)

        tv = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        tv[0] = np.array(list(TIMES), dtype="S1")

        lat0, lon0 = 35.0, -100.0
        lat = lat0 + 0.02 * np.arange(NY)[:, None] * np.ones((1, NX))
        lon = lon0 + 0.02 * np.arange(NX)[None, :] * np.ones((NY, 1))
        for name, dims, data in [
            ("XLAT", ("south_north", "west_east"), lat),
            ("XLONG", ("south_north", "west_east"), lon),
            ("HGT", ("south_north", "west_east"), np.full((NY, NX), 500.0)),
            ("LANDMASK", ("south_north", "west_east"), np.ones((NY, NX))),
            ("TSK", ("south_north", "west_east"), np.full((NY, NX), 290.0)),
            ("MU", ("south_north", "west_east"), np.full((NY, NX), 1000.0)),
            ("MUB", ("south_north", "west_east"), np.full((NY, NX), 90000.0)),
        ]:
            v = ds.createVariable(name, "f4", (("Time",) + dims))
            v[0] = data

        znu = ds.createVariable("ZNU", "f4", ("Time", "bottom_top"))
        znu[0] = np.linspace(0.95, 0.05, NZ)

        smois = ds.createVariable("SMOIS", "f4", ("Time", "soil_layers_stag", "south_north",
                                                    "west_east"))
        smois[0] = np.full((4, NY, NX), 0.25)

        # all-levels pressure kept comfortably above 850 hPa so every level lands in the
        # 'low' LEVEL_GROUPS bucket -- keeps the apply() test's expected mask deterministic.
        p = np.full((NZ, NY, NX), 90000.0) - 500.0 * np.arange(NZ)[:, None, None]
        pb = np.zeros((NZ, NY, NX))
        theta = 290.0 + 2.0 * np.arange(NZ)[:, None, None] + 0 * rng.normal(size=(NZ, NY, NX))
        qv = np.full((NZ, NY, NX), 0.006)
        ph = np.cumsum(np.full((NZ + 1, NY, NX), 300.0), axis=0)
        phb = np.zeros((NZ + 1, NY, NX))
        u = np.full((NZ, NY, NX + 1), 5.0)
        v = np.full((NZ, NY + 1, NX), -2.0)
        for name, dims, data in [
            ("P", ("bottom_top", "south_north", "west_east"), p),
            ("PB", ("bottom_top", "south_north", "west_east"), pb),
            ("T", ("bottom_top", "south_north", "west_east"), theta - 300.0),
            ("QVAPOR", ("bottom_top", "south_north", "west_east"), qv),
            ("PH", ("bottom_top_stag", "south_north", "west_east"), ph),
            ("PHB", ("bottom_top_stag", "south_north", "west_east"), phb),
            ("U", ("bottom_top", "south_north", "west_east_stag"), u),
            ("V", ("bottom_top", "south_north_stag", "west_east"), v),
            ("QCLOUD", ("bottom_top", "south_north", "west_east"), np.zeros((NZ, NY, NX))),
            ("QRAIN", ("bottom_top", "south_north", "west_east"), np.zeros((NZ, NY, NX))),
        ]:
            v_ = ds.createVariable(name, "f4", (("Time",) + dims))
            v_[0] = data


@unittest.skipUnless(HAVE_NETCDF4, "netCDF4 not installed in this interpreter")
class TestGridCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.a = os.path.join(self.tmp, "a.nc")
        self.b = os.path.join(self.tmp, "b.nc")
        _write_wrf_like(self.a, seed=1)
        _write_wrf_like(self.b, seed=2)

    def test_matching_grid_ok(self):
        ds_a = netCDF4.Dataset(self.a)
        ds_b = netCDF4.Dataset(self.b)
        try:
            scm.check_grid_match(ds_a, ds_b)  # must not raise
        finally:
            ds_a.close()
            ds_b.close()

    def test_mismatched_dx_raises(self):
        c = os.path.join(self.tmp, "c.nc")
        _write_wrf_like(c, seed=3, dx=9000.0)
        ds_a = netCDF4.Dataset(self.a)
        ds_c = netCDF4.Dataset(c)
        try:
            with self.assertRaises(scm.StateCorrectionError):
                scm.check_grid_match(ds_a, ds_c)
        finally:
            ds_a.close()
            ds_c.close()


# ---------------------------------------------------------------------------
# K5 (2026-09-24): --targets-from-report's "nothing improved -> write nothing" path never
# reaches netCDF4 at all (the early return is before `import netCDF4`), so it's exercised here
# unconditionally, not gated on HAVE_NETCDF4.
# ---------------------------------------------------------------------------

class TestApplyTargetsFromReportNoNetcdf4Needed(unittest.TestCase):
    def _report(self, **improved_flags):
        n_feat = len(scm.FEATURE_NAMES)
        model = {"kind": "ridge", "w": [1.0] + [0.0] * n_feat,
                 "mu": [0.0] * n_feat, "sd": [1.0] * n_feat, "alpha": 1.0}
        return {"n_cases": 5, "n_rows": 100,
                "targets": {k: {"kind": "residual", "model": model, "improved": v}
                           for k, v in improved_flags.items()}}

    def test_no_improved_targets_writes_nothing_and_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            report_path = os.path.join(d, "report.json")
            with open(report_path, "w") as f:
                json.dump(self._report(dT__low=False, dQV__mid=False), f)
            wrfinput_in = os.path.join(d, "wrfinput_d01")   # never opened -- doesn't even exist
            out = os.path.join(d, "wrfinput_d01.ml")
            result = scm.apply_correction(report_path, wrfinput_in, out,
                                          targets_from_report=True)
            self.assertIsNone(result)
            self.assertFalse(os.path.exists(out))

    def test_some_improved_still_requires_the_refusal_checks(self):
        # the --out == --wrfinput / --out-already-exists refusals must still fire even when
        # targets_from_report is set and the eventual netCDF4 work would never be reached.
        with tempfile.TemporaryDirectory() as d:
            report_path = os.path.join(d, "report.json")
            with open(report_path, "w") as f:
                json.dump(self._report(dT__low=True), f)
            wrfinput_in = os.path.join(d, "wrfinput_d01")
            with self.assertRaises(scm.StateCorrectionError):
                scm.apply_correction(report_path, wrfinput_in, wrfinput_in,
                                     targets_from_report=True)

    def test_cli_targets_from_report_no_improved_exits_0(self):
        with tempfile.TemporaryDirectory() as d:
            report_path = os.path.join(d, "report.json")
            with open(report_path, "w") as f:
                json.dump(self._report(dT__low=False), f)
            wrfinput_in = os.path.join(d, "wrfinput_d01")
            out = os.path.join(d, "wrfinput_d01.ml")
            rc = scm.main(["apply", "--model", report_path, "--wrfinput", wrfinput_in,
                          "--out", out, "--targets-from-report"])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(out))


@unittest.skipUnless(HAVE_NETCDF4, "netCDF4 not installed in this interpreter")
class TestApply(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wrfinput = os.path.join(self.tmp, "wrfinput_d01")
        _write_wrf_like(self.wrfinput, seed=5)

        # a trivial ridge model that predicts a KNOWN constant dT shift (+1.25 K) everywhere,
        # nothing for any other target -- lets the test assert an exact, predictable change.
        n_feat = len(scm.FEATURE_NAMES)
        model = {"kind": "ridge", "w": [1.25] + [0.0] * n_feat,
                 "mu": [0.0] * n_feat, "sd": [1.0] * n_feat, "alpha": 1.0}
        self.report = {"n_cases": 5, "n_rows": 100,
                        "targets": {"dT__low": {"kind": "residual", "model": model}}}
        self.report_path = os.path.join(self.tmp, "report.json")
        with open(self.report_path, "w") as f:
            json.dump(self.report, f)

    def test_refuses_same_path(self):
        with self.assertRaises(scm.StateCorrectionError):
            scm.apply_correction(self.report_path, self.wrfinput, self.wrfinput)

    def test_refuses_existing_out(self):
        out = os.path.join(self.tmp, "out.nc")
        with open(out, "w") as f:
            f.write("x")
        with self.assertRaises(scm.StateCorrectionError):
            scm.apply_correction(self.report_path, self.wrfinput, out)

    def test_input_untouched_and_unrelated_vars_preserved(self):
        before = open(self.wrfinput, "rb").read()
        ds_in = netCDF4.Dataset(self.wrfinput)
        hgt_before = np.array(ds_in.variables["HGT"][:])
        tsk_before = np.array(ds_in.variables["TSK"][:])
        landmask_before = np.array(ds_in.variables["LANDMASK"][:])
        t_before = np.array(ds_in.variables["T"][:])
        ds_in.close()

        out = os.path.join(self.tmp, "corrected.nc")
        scm.apply_correction(self.report_path, self.wrfinput, out)

        after = open(self.wrfinput, "rb").read()
        self.assertEqual(before, after, "apply_correction must never modify the input file")

        ds_out = netCDF4.Dataset(out)
        try:
            np.testing.assert_array_equal(np.array(ds_out.variables["HGT"][:]), hgt_before)
            np.testing.assert_array_equal(np.array(ds_out.variables["TSK"][:]), tsk_before)
            np.testing.assert_array_equal(np.array(ds_out.variables["LANDMASK"][:]),
                                           landmask_before)
            t_after = np.array(ds_out.variables["T"][:])
            np.testing.assert_allclose(t_after, t_before + 1.25, atol=1e-4)
        finally:
            ds_out.close()

    def test_targets_from_report_skips_non_improved_targets(self):
        # K5: a target with a fitted model but improved=False must be SKIPPED entirely under
        # --targets-from-report, even though a plain (non-filtered) apply would have used it.
        n_feat = len(scm.FEATURE_NAMES)
        decoy_model = {"kind": "ridge", "w": [999.0] + [0.0] * n_feat,   # would be obviously
                       "mu": [0.0] * n_feat, "sd": [1.0] * n_feat,       # wrong if applied
                       "alpha": 1.0}
        report = {
            "n_cases": 5, "n_rows": 100,
            "targets": {
                "dT__low": {"kind": "residual",
                           "model": {"kind": "ridge", "w": [1.25] + [0.0] * n_feat,
                                    "mu": [0.0] * n_feat, "sd": [1.0] * n_feat, "alpha": 1.0},
                           "improved": True},
                "dQV__low": {"kind": "residual", "model": decoy_model, "improved": False},
            },
        }
        report_path = os.path.join(self.tmp, "report_mixed.json")
        with open(report_path, "w") as f:
            json.dump(report, f)

        ds_in = netCDF4.Dataset(self.wrfinput)
        qv_before = np.array(ds_in.variables["QVAPOR"][:])
        t_before = np.array(ds_in.variables["T"][:])
        ds_in.close()

        out = os.path.join(self.tmp, "filtered.nc")
        scm.apply_correction(report_path, self.wrfinput, out, targets_from_report=True)

        ds_out = netCDF4.Dataset(out)
        try:
            t_after = np.array(ds_out.variables["T"][:])
            qv_after = np.array(ds_out.variables["QVAPOR"][:])
            np.testing.assert_allclose(t_after, t_before + 1.25, atol=1e-4)
            np.testing.assert_allclose(qv_after, qv_before, atol=1e-9)   # dQV__low NOT applied
            self.assertEqual(ds_out.getncattr("STATE_CORRECTION_TARGETS"), "dT__low")
        finally:
            ds_out.close()


@unittest.skipUnless(HAVE_NETCDF4, "netCDF4 not installed in this interpreter")
class TestBuildPairs(unittest.TestCase):
    def test_small_end_to_end(self):
        tmp = tempfile.mkdtemp()
        rst = os.path.join(tmp, "wrfrst_d01_2026-01-01_00:00:00")
        wrfinput = os.path.join(tmp, "wrfinput_d01")
        _write_wrf_like(rst, seed=10)
        _write_wrf_like(wrfinput, seed=11)
        out = os.path.join(tmp, "pairs.npz")
        stats = scm.build_pairs([("caseA", rst, wrfinput)], out, n_columns=5, seed=0)
        self.assertEqual(stats["n_pairs"], 1)
        self.assertEqual(stats["n_rows"], 5 * NZ)

        d = np.load(out)
        self.assertEqual(d["X"].shape, (5 * NZ, len(scm.FEATURE_NAMES)))
        self.assertEqual(len(set(d["case_id"].tolist())), 1)


if __name__ == "__main__":
    unittest.main()
