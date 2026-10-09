#!/usr/bin/env python3
"""Tests for tools/bp_compression_check.py.

The parsing is tested against REAL captured `bpls -l -a --hidden_attrs` output
from a conus12 wrfout frame (job 1082176, arm A_default frame 1), vendored at
tools/testdata/bpls_l_conus12_wrfout_frame1.txt. The BP directories themselves
are 180-600 MB, far too big to keep as fixtures, so the stored-bytes side is
tested against synthetic directories whose data.* files have known sizes.

Real measured numbers from job 1082176 that pin the constants below:
  logical (parsed from this listing) 595,501,316 B over 203 variables
  stored, compression off            595,875,392 B   ratio 0.999x
  stored, adios2_blosc_compressor lz4 181,090,208 B  ratio 3.29x
  stored, adios2_blosc_compressor zstd 166,682,400 B ratio 3.57x
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bp_compression_check as bpc  # noqa: E402

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")
REAL_LISTING = os.path.join(TESTDATA, "bpls_l_conus12_wrfout_frame1.txt")

# Measured on conus12 job 1082176 (425x300x50, 112 ranks, 1 history frame).
REAL_LOGICAL_BYTES = 595501316
REAL_N_VARIABLES = 203
STORED_UNCOMPRESSED = 595875392
STORED_LZ4 = 181090208
STORED_ZSTD = 166682400


def _make_bp_dir(tmp, data_sizes, extra=()):
    """Build a fake BP directory with data.N files of the given sizes."""
    os.makedirs(tmp)
    for i, size in enumerate(data_sizes):
        with open(os.path.join(tmp, "data.%d" % i), "wb") as f:
            f.write(b"\0" * size)
    for name, size in extra:
        with open(os.path.join(tmp, name), "wb") as f:
            f.write(b"\0" * size)
    return tmp


class TestParseRealListing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REAL_LISTING) as f:
            cls.text = f.read()
        cls.variables = bpc.parse_bpls_listing(cls.text)

    def test_variable_count_matches_the_real_output(self):
        self.assertEqual(len(self.variables), REAL_N_VARIABLES)

    def test_logical_bytes_match_the_measured_total(self):
        total, skipped = bpc.logical_bytes(self.variables)
        self.assertEqual(total, REAL_LOGICAL_BYTES)
        self.assertEqual(skipped, [])

    def test_logical_total_is_within_a_tenth_of_a_percent_of_uncompressed_stored(self):
        # This is the whole basis of the check: on an uncompressed BP the parsed
        # logical size and the bytes on disk agree to BP header overhead only.
        total, _ = bpc.logical_bytes(self.variables)
        self.assertLess(abs(total - STORED_UNCOMPRESSED) / float(STORED_UNCOMPRESSED),
                        0.001)

    def test_attribute_lines_are_not_counted_as_variables(self):
        names = [n for n, _, _ in self.variables]
        self.assertNotIn("ACGRDFLX/units", names)
        self.assertNotIn("ACGRDFLX/Dims", names)
        self.assertIn("ACGRDFLX", names)
        self.assertFalse([n for n in names if "/" in n])

    def test_known_variables_parse_with_the_right_type_and_shape(self):
        by_name = dict((n, (t, s)) for n, t, s in self.variables)
        self.assertEqual(by_name["T"], ("float", [49, 299, 424]))
        self.assertEqual(by_name["U"], ("float", [49, 299, 425]))
        self.assertEqual(by_name["QVAPOR"], ("float", [49, 299, 424]))
        self.assertEqual(by_name["ACGRDFLX"], ("float", [299, 424]))

    def test_a_single_variables_size_is_shape_times_itemsize(self):
        total, _ = bpc.logical_bytes([("T", "float", [49, 299, 424])])
        self.assertEqual(total, 49 * 299 * 424 * 4)


class TestParseEdgeCases(unittest.TestCase):
    def test_every_fixed_width_type_is_sized(self):
        vs = [("a", "double", [2]), ("b", "int32_t", [2]), ("c", "int64_t", [2]),
              ("d", "char", [8]), ("e", "uint16_t", [4])]
        total, skipped = bpc.logical_bytes(vs)
        self.assertEqual(skipped, [])
        self.assertEqual(total, 2 * 8 + 2 * 4 + 2 * 8 + 8 * 1 + 4 * 2)

    def test_unsized_types_are_reported_not_silently_dropped(self):
        total, skipped = bpc.logical_bytes([("s", "string", [3]), ("t", "float", [2])])
        self.assertEqual(total, 8)
        self.assertEqual(skipped, [("s", "string")])

    def test_multi_step_shape_includes_the_leading_step_dimension(self):
        vs = bpc.parse_bpls_listing("  float    T    {3, 49, 299, 424} = -1 / 1\n")
        self.assertEqual(vs, [("T", "float", [3, 49, 299, 424])])
        total, _ = bpc.logical_bytes(vs)
        self.assertEqual(total, 3 * 49 * 299 * 424 * 4)

    def test_lines_without_braces_are_ignored(self):
        self.assertEqual(bpc.parse_bpls_listing("File info:\n  of variables: 203\n"), [])

    def test_empty_input_parses_to_nothing(self):
        self.assertEqual(bpc.parse_bpls_listing(""), [])
        self.assertEqual(bpc.logical_bytes([]), (0, []))


class TestStoredBytes(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sums_only_data_subfiles(self):
        d = _make_bp_dir(os.path.join(self.tmp, "x.bp"), [100, 200, 300],
                         extra=[("md.0", 4096), ("md.idx", 128), ("mmd.0", 64)])
        self.assertEqual(bpc.stored_bytes(d), 600)

    def test_metadata_size_does_not_move_the_answer(self):
        # md.0 grows by ~365 KB when the operator IS declared (measured:
        # 14,057,352 vs 13,692,680 on job 1082176). Counting it would
        # contaminate the ratio in the wrong direction, so it must not be.
        a = _make_bp_dir(os.path.join(self.tmp, "a.bp"), [1000], extra=[("md.0", 13692680)])
        b = _make_bp_dir(os.path.join(self.tmp, "b.bp"), [1000], extra=[("md.0", 14057352)])
        self.assertEqual(bpc.stored_bytes(a), bpc.stored_bytes(b))


class TestVerdict(unittest.TestCase):
    """The ratio thresholding, exercised on the three real measured arms."""

    def _verdict(self, stored, min_ratio=bpc.DEFAULT_MIN_RATIO):
        ratio = float(REAL_LOGICAL_BYTES) / float(stored)
        return ratio, ratio >= min_ratio

    def test_uncompressed_arm_is_reported_uncompressed(self):
        ratio, compressed = self._verdict(STORED_UNCOMPRESSED)
        self.assertFalse(compressed)
        self.assertAlmostEqual(ratio, 0.9994, delta=0.0005)

    def test_lz4_arm_is_reported_compressed(self):
        ratio, compressed = self._verdict(STORED_LZ4)
        self.assertTrue(compressed)
        self.assertAlmostEqual(ratio, 3.2884, delta=0.0005)

    def test_zstd_arm_is_reported_compressed(self):
        ratio, compressed = self._verdict(STORED_ZSTD)
        self.assertTrue(compressed)
        self.assertAlmostEqual(ratio, 3.5727, delta=0.0005)

    def test_zstd_beats_lz4_on_size(self):
        self.assertLess(STORED_ZSTD, STORED_LZ4)

    def test_default_threshold_sits_well_clear_of_both_sides(self):
        uncompressed = float(REAL_LOGICAL_BYTES) / STORED_UNCOMPRESSED
        weakest_codec = float(REAL_LOGICAL_BYTES) / STORED_LZ4
        self.assertLess(uncompressed, bpc.DEFAULT_MIN_RATIO)
        self.assertGreater(weakest_codec, bpc.DEFAULT_MIN_RATIO * 2)

    def test_min_ratio_can_demand_more_than_the_floor(self):
        _, compressed = self._verdict(STORED_LZ4, min_ratio=4.0)
        self.assertFalse(compressed)


class TestCheckBpDir(unittest.TestCase):
    """End-to-end with bpls stubbed out, so the test needs no BP file."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        with open(REAL_LISTING) as f:
            self.listing = f.read()
        self._real_run_bpls = bpc.run_bpls
        bpc.run_bpls = lambda bp_dir, bpls=None: self.listing

    def tearDown(self):
        import shutil
        bpc.run_bpls = self._real_run_bpls
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_compressed_directory(self):
        d = _make_bp_dir(os.path.join(self.tmp, "c.bp"), [1000])
        r = bpc.check_bp_dir(d)
        # 595,501,316 / 1000 is hugely > 1.05
        self.assertTrue(r["compressed"])
        self.assertIsNone(r["error"])
        self.assertEqual(r["logical_bytes"], REAL_LOGICAL_BYTES)
        self.assertEqual(r["stored_bytes"], 1000)
        self.assertEqual(r["n_variables"], REAL_N_VARIABLES)

    def test_uncompressed_directory(self):
        d = os.path.join(self.tmp, "u.bp")
        os.makedirs(d)
        # one data file of exactly the measured uncompressed size, sparsely
        with open(os.path.join(d, "data.0"), "wb") as f:
            f.truncate(STORED_UNCOMPRESSED)
        r = bpc.check_bp_dir(d)
        self.assertFalse(r["compressed"])
        self.assertIsNone(r["error"])
        self.assertAlmostEqual(r["ratio"], 0.9994, delta=0.0005)

    def test_missing_directory_is_an_error_not_a_crash(self):
        r = bpc.check_bp_dir(os.path.join(self.tmp, "nope.bp"))
        self.assertEqual(r["error"], "not a directory")
        self.assertFalse(r["compressed"])

    def test_directory_with_no_data_subfiles_is_an_error(self):
        d = _make_bp_dir(os.path.join(self.tmp, "e.bp"), [], extra=[("md.0", 10)])
        r = bpc.check_bp_dir(d)
        self.assertEqual(r["error"], "no data.* subfiles found")
        self.assertFalse(r["compressed"])

    def test_main_exit_status_is_nonzero_when_any_bp_is_uncompressed(self):
        good = _make_bp_dir(os.path.join(self.tmp, "g.bp"), [1000])
        bad = os.path.join(self.tmp, "b.bp")
        os.makedirs(bad)
        with open(os.path.join(bad, "data.0"), "wb") as f:
            f.truncate(STORED_UNCOMPRESSED)
        self.assertEqual(bpc.main([good]), 0)
        self.assertEqual(bpc.main([bad]), 1)
        self.assertEqual(bpc.main([good, bad]), 1)

    def test_trailing_slash_is_tolerated(self):
        d = _make_bp_dir(os.path.join(self.tmp, "t.bp"), [1000])
        self.assertEqual(bpc.main([d + "/"]), 0)


if __name__ == "__main__":
    unittest.main()
