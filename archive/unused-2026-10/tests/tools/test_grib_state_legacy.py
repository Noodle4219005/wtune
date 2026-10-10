# Tests of the archived grib_state trajectory code. Not runnable as is.


# from tests/test_review_fixes_core.py


def test_grib_state_trajectory_passes_grib_dir_to_select_source(monkeypatch):
    gs = importlib.import_module("grib_state")
    seen = {}

    class _Stop(Exception):
        pass

    def fake_select(valid_dt, now=None, override=None, grib_dir=None):
        seen.update(override=override, grib_dir=grib_dir)
        raise _Stop()

    monkeypatch.setattr(gs, "load_case", lambda *a, **k: ({}, None, None))
    monkeypatch.setattr(gs, "target_grid", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(gs, "resolve_times", lambda *a, **k: [datetime.datetime(2010, 6, 1)])
    monkeypatch.setattr(gs.md, "select_source", fake_select)
    with pytest.raises(_Stop):
        gs.trajectory("unused", source="auto", grib_dir="/data/narr")
    assert seen == {"override": "auto", "grib_dir": "/data/narr"}
# from tests/tools/test_grib_state.py


class TestPatchStats(unittest.TestCase):
    def test_patch_slices_cover_extent_exactly(self):
        slices = gs.patch_slices(23, 4)   # 23 % 4 = 3 remainder
        self.assertEqual(slices[0][0], 0)
        self.assertEqual(slices[-1][1], 23)
        widths = [b - a for a, b in slices]
        self.assertEqual(sum(widths), 23)
        self.assertEqual(sorted(widths), [5, 6, 6, 6])   # divmod(23,4) = (5, 3) -> three 6s, one 5

    def test_patch_slices_matches_rsl_patch_width_for_rank(self):
        import decomp_space as ds
        for r in range(4):
            w = gs.patch_slices(23, 4)[r]
            self.assertEqual(w[1] - w[0], ds.rsl_patch_width_for_rank(23, 4, r))

    def test_patch_stats_uniform_field_zero_spread(self):
        field = np.ones((20, 20)) * 7.0
        stats = gs.patch_stats(field, 4, 4)
        self.assertAlmostEqual(stats['patch_spread'], 0.0)
        self.assertAlmostEqual(stats['patch_max'], 7.0)
        self.assertAlmostEqual(stats['patch_min'], 7.0)

    def test_patch_stats_gradient_field_nonzero_spread(self):
        field = np.tile(np.arange(20.0), (20, 1))   # increases left -> right
        stats = gs.patch_stats(field, 4, 1)
        self.assertGreater(stats['patch_spread'], 0.0)
        self.assertLess(stats['patch_min'], stats['patch_max'])

    def test_patch_stats_legality_matches_decomp_space(self):
        import decomp_space as ds
        field = np.ones((100, 91))
        stats = gs.patch_stats(field, 4, 4)
        min_w, max_w = ds.rsl_patch_extents(91, 4)
        min_h, max_h = ds.rsl_patch_extents(100, 4)
        self.assertEqual(stats['min_patch_x'], min_w)
        self.assertEqual(stats['max_patch_x'], max_w)
        self.assertEqual(stats['min_patch_y'], min_h)
        self.assertEqual(stats['max_patch_y'], max_h)
        self.assertEqual(stats['legal'], min_w >= ds.MIN_PATCH and min_h >= ds.MIN_PATCH)

    def test_patch_stats_patch_mean_matches_unweighted_mean_of_patch_means(self):
        # Same semantics as tools/state_timeline.py's max_mean_over_patches: mean over the
        # per-patch means themselves, NOT a size-weighted mean over all columns.
        field = np.tile(np.arange(20.0), (20, 1))
        stats = gs.patch_stats(field, 4, 1)
        self.assertAlmostEqual(stats['patch_mean'], float(np.mean(stats['patch_means'])))

    def test_patch_stats_uniform_field_patch_mean_equals_value(self):
        field = np.ones((20, 20)) * 3.0
        stats = gs.patch_stats(field, 4, 4)
        self.assertAlmostEqual(stats['patch_mean'], 3.0)


class TestImbalanceRatio(unittest.TestCase):
    def test_ratio_of_max_over_mean(self):
        self.assertAlmostEqual(gs.imbalance_ratio(4.0, 2.0), 2.0)

    def test_none_when_mean_is_none(self):
        self.assertIsNone(gs.imbalance_ratio(4.0, None))

    def test_none_when_max_is_none(self):
        self.assertIsNone(gs.imbalance_ratio(None, 2.0))

    def test_none_when_mean_is_zero(self):
        self.assertIsNone(gs.imbalance_ratio(0.0, 0.0))

    def test_uniform_field_ratio_is_one(self):
        field = np.ones((20, 20))
        stats = gs.patch_stats(field, 4, 4)
        self.assertAlmostEqual(gs.imbalance_ratio(stats['patch_max'], stats['patch_mean']), 1.0)


# from tests/tools/test_grib_state.py, class TestFeatureMath
    def test_precip_rate_from_apcp_diff(self):
        a = np.array([0.0, 3.0])
        b = np.array([3.0, 9.0])
        rate = gs.precip_rate_from_apcp_diff(a, b, 3.0)
        np.testing.assert_allclose(rate, [1.0, 2.0])

    def test_cloud_frac_tcdc_threshold(self):
        tcdc = np.array([10.0, 49.0, 50.0, 51.0, 90.0])   # strictly > 50 -> 2 of 5
        self.assertAlmostEqual(gs.cloud_frac_tcdc(tcdc), 2.0 / 5.0)

    def test_cloud_frac_olr_threshold(self):
        olr = np.array([150.0, 199.0, 200.0, 201.0, 300.0])  # strictly < 200 -> 2 of 5
        self.assertAlmostEqual(gs.cloud_frac_olr(olr), 2.0 / 5.0)
