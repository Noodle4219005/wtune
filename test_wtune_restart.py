import json
import datetime
import unittest

import wtune_restart as wr


class RestartWindowTests(unittest.TestCase):
    def test_windows_are_half_open_and_progress_utc_day_of_year(self):
        windows = wr.build_windows(25, 90, start_utc="2024-12-31T23:30:00Z")
        self.assertEqual([(w["start_hour"], w["end_hour"]) for w in windows],
                         [(0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0),
                          (6.0, 7.5), (7.5, 9.0), (9.0, 10.5), (10.5, 12.0),
                          (12.0, 13.5), (13.5, 15.0), (15.0, 16.5),
                          (16.5, 18.0), (18.0, 19.5), (19.5, 21.0),
                          (21.0, 22.5), (22.5, 24.0), (24.0, 25.0)])
        self.assertEqual(windows[0]["day_of_year"], 366)
        self.assertEqual(windows[-1]["day_of_year"], 1)

    def test_window_and_run_overrides_and_validation(self):
        self.assertEqual(len(wr.build_windows(10, 60, window_hours=4)), 3)
        self.assertEqual(len(wr.build_windows(10, 60, window_hours=4, run_hours_override=5)), 2)
        with self.assertRaises(ValueError):
            wr.build_windows(10, 0)


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.topology = {"nodes": [{"name": "n1", "numa_domains": [
            {"id": 0, "cpus": [0, 1, 2, 3]}, {"id": 1, "cpus": [4, 5]}]}],
                         "allowed_team_sizes": [1, 2, 4]}

    def test_validate_topology_and_candidates_are_legal_and_deterministic(self):
        self.assertEqual(wr.validate_topology(self.topology), self.topology)
        candidates = wr.enumerate_candidates(rank_count=4, patch_x=40, patch_y=20,
                                             local_patch_y=20, topology=self.topology)
        self.assertEqual([(c["px"], c["py"]) for c in candidates], [(2, 2), (4, 1)])
        self.assertTrue(all(c["rank_count"] == 4 for c in candidates))
        with self.assertRaises(ValueError):
            wr.validate_topology({"nodes": []})

    def test_allocation_prioritizes_load_without_claiming_speedup(self):
        alloc = wr.allocate_teams([10.0, 2.0, 1.0], [1, 2], topology=self.topology)
        self.assertEqual([a["team_size"] for a in alloc], [2, 2, 2])
        self.assertTrue(all(a["predicted_speedup"] == 0.0 for a in alloc))
        self.assertEqual(alloc[0]["numa_domain"], 0)

    def test_topology_rejects_bad_ids_and_overlapping_cpus(self):
        for bad in ({"id": "0", "cpus": [0]}, {"id": 0, "cpus": [0, 0]}):
            topology = {"nodes": [{"name": "n", "numa_domains": [bad]}],
                        "allowed_team_sizes": [1]}
            with self.assertRaises(ValueError):
                wr.validate_topology(topology)
        topology = {"nodes": [{"name": "n", "numa_domains": [
            {"id": 0, "cpus": [0, 1]}, {"id": 0, "cpus": [2]}]}],
                    "allowed_team_sizes": [1]}
        with self.assertRaises(ValueError):
            wr.validate_topology(topology)

    def test_candidates_and_allocation_use_per_domain_capacity_and_cpu_maps(self):
        topology = {"nodes": [{"name": "n", "numa_domains": [
            {"id": 0, "cpus": [0, 1]}, {"id": 1, "cpus": [2, 3]}]}],
                    "allowed_team_sizes": [1, 2, 3]}
        self.assertEqual(wr.enumerate_candidates(2, 40, 20, topology, local_patch_y=20)[0]["team_size"], 1)
        with self.assertRaises(ValueError):
            wr.allocate_teams([1, 2], [3], topology)
        alloc = wr.allocate_teams([10, 2], [1, 2], topology)
        self.assertEqual(sum(len(a["cpus"]) for a in alloc), 4)
        self.assertEqual(len(set(cpu for a in alloc for cpu in a["cpus"])), 4)

    def test_score_fields_and_local_extent(self):
        candidates = wr.enumerate_candidates(4, 40, 20, self.topology,
                                             local_patch_y=20,
                                             predicted_loads=[4, 1, 1, 1])
        self.assertTrue(all("structural_score" in c and "criticality_score" in c
                            and "score" in c for c in candidates))
        with self.assertRaises(ValueError):
            wr.enumerate_candidates(4, 40, 20, self.topology)


class RenderingTests(unittest.TestCase):
    def test_rendering_is_deterministic_and_changed_verdicts_are_unverified(self):
        spec = {"config": {"max_dom": 1, "strategy": 0, "numtiles": 99},
                "windows": [{"id": 0}], "candidates": [{"rank_count": 4}],
                "local_patch_y": 4,
                "proxy_cost": None, "transition_cost": "UNKNOWN", "restart_cost": None}
        manifest = wr.render_manifest(spec)
        self.assertFalse(manifest["actionable"])
        self.assertEqual(manifest["strict_bitwise"], "UNVERIFIED")
        self.assertEqual(manifest["scientific_tolerance"], "UNVERIFIED")
        self.assertEqual(manifest["config"]["numtiles"], 4)
        self.assertEqual(manifest, wr.render_manifest(json.loads(json.dumps(spec))))
        self.assertNotIn("PENDING", wr.render_map(spec))
        wrapper = wr.render_wrapper(spec)
        self.assertTrue(wrapper.startswith("#!/bin/sh\n"))
        self.assertIn("PMI_RANK", wrapper)
        self.assertIn("OMP_NUM_THREADS", wrapper)
        self.assertIn("exec", wrapper)
        self.assertNotIn("'config'", wrapper)

    def test_canonical_rendering_and_utc_offset(self):
        value = datetime.datetime(2024, 12, 31, 23, 30,
            tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
        self.assertEqual(wr.build_windows(1, 60, start_utc=value)[0]["day_of_year"], 366)
        a = {"b": 1, "a": 2}
        self.assertEqual(wr.render_map(a), wr.render_map({"a": 2, "b": 1}))

    def test_unsupported_tile_controls_are_rejected(self):
        with self.assertRaises(ValueError):
            wr.normalize_tile_config({"strategy": 3})
        with self.assertRaises(ValueError):
            wr.normalize_tile_config({"strategy": 0, "numtiles_x": 2})


if __name__ == "__main__":
    unittest.main()
