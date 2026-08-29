import json
import base64
import datetime
import copy
import os
import subprocess
import tempfile
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

    def test_topology_ids_and_cpus_are_node_local_and_input_is_immutable(self):
        topology = {"nodes": [
            {"name": "n1", "numa_domains": [{"id": 0, "cpus": [0, 1]}]},
            {"name": "n2", "numa_domains": [{"id": 0, "cpus": [0, 1]}]},
        ], "allowed_team_sizes": [1, 2]}
        original = copy.deepcopy(topology)
        validated = wr.validate_topology(topology)
        self.assertEqual(topology, original)
        self.assertEqual(validated, topology)
        self.assertIsNot(validated, topology)

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

    def test_allocation_rejects_team_size_not_allowed_by_topology(self):
        with self.assertRaises(ValueError):
            wr.allocate_teams([1], [3], self.topology)

    def test_allocation_resource_keys_include_node_and_domain(self):
        topology = {"nodes": [
            {"name": "n1", "numa_domains": [{"id": 0, "cpus": [0]}]},
            {"name": "n2", "numa_domains": [{"id": 0, "cpus": [0]}]},
        ], "allowed_team_sizes": [1]}
        alloc = wr.allocate_teams([2], [1], topology)
        self.assertEqual(alloc[0]["resource_key"], "n1:0")

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
        self.assertIn("WTUNE_PAYLOAD", wrapper)
        self.assertIn("python3 -c", wrapper)
        self.assertIn('"$@"', wrapper)
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

    def test_all_renderers_reject_unsupported_tiles_without_scalar_numtiles(self):
        for spec in ({"config": {"strategy": 3}},
                     {"config": {"numtiles_x": 2}}):
            for render in (wr.render_manifest, wr.render_map, wr.render_wrapper):
                with self.assertRaises(ValueError):
                    render(spec)

    def test_renderers_preserve_explicit_proxy_transition_restart_fields(self):
        spec = {"config": {}, "proxy_cost": 1, "transition_cost": "x",
                "restart_cost": {"seconds": 2}}
        self.assertEqual(wr.render_manifest(spec)["transition_cost"], "x")
        self.assertIn('"proxy_cost":1', wr.render_map(spec))
        wrapper = wr.render_wrapper(spec)
        payload = wrapper.split("WTUNE_PAYLOAD='")[1].split("'")[0]
        self.assertEqual(json.loads(base64.b64decode(payload).decode("ascii"))["restart_cost"], {"seconds": 2})

    def test_wrapper_executes_fake_command_using_json_allocation(self):
        spec = {"config": {}, "allocations": [{"rank": 0, "team_size": 2,
                "node": "n1", "numa_domain": 0, "cpus": [4, 5]}]}
        wrapper = wr.render_wrapper(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wrapper.sh")
            out = os.path.join(tmp, "env.txt")
            command = os.path.join(tmp, "fake.sh")
            with open(path, "w") as handle:
                handle.write(wrapper)
            with open(command, "w") as handle:
                handle.write("#!/bin/sh\nprintf '%%s|%%s|%%s|%%s' \"$OMP_NUM_THREADS\" \"$OMP_PLACES\" \"$OMP_PROC_BIND\" \"$1\" > \"%s\"\n" % out)
            os.chmod(path, 0o755)
            os.chmod(command, 0o755)
            env = os.environ.copy()
            env.pop("OMPI_COMM_WORLD_RANK", None)
            env.pop("PMIX_RANK", None)
            env["PMI_RANK"] = "0"
            subprocess.check_call([path, command, "ran"], env=env)
            with open(out) as handle:
                self.assertEqual(handle.read(), "2|{4,5}|close|ran")

    def test_wrapper_preserves_piped_stdin_for_known_rank(self):
        spec = {"config": {}, "allocations": [{"rank": 0, "team_size": 1,
                "node": "n1", "numa_domain": 0, "cpus": [0]}]}
        wrapper = wr.render_wrapper(spec)
        payload = b"WRF stdin\\x00must survive\\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wrapper.sh")
            with open(path, "w") as handle:
                handle.write(wrapper)
            os.chmod(path, 0o755)
            env = os.environ.copy()
            for key in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK"):
                env.pop(key, None)
            env["PMI_RANK"] = "0"
            completed = subprocess.run([path, "/bin/cat"], input=payload, env=env,
                                       stdout=subprocess.PIPE, check=True)
            self.assertEqual(completed.stdout, payload)

    def test_wrapper_rejects_missing_or_unknown_rank_without_running_target(self):
        spec = {"config": {}, "allocations": [{"rank": 3, "team_size": 1,
                "node": "n1", "numa_domain": 0, "cpus": [7]}]}
        wrapper = wr.render_wrapper(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wrapper.sh")
            marker = os.path.join(tmp, "ran")
            command = os.path.join(tmp, "fake.sh")
            with open(path, "w") as handle:
                handle.write(wrapper)
            with open(command, "w") as handle:
                handle.write("#!/bin/sh\ntouch %s\n" % marker)
            os.chmod(path, 0o755)
            os.chmod(command, 0o755)
            for rank_env in ({}, {"PMI_RANK": "99"}):
                env = os.environ.copy()
                for key in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "PMIX_RANK"):
                    env.pop(key, None)
                env.update(rank_env)
                completed = subprocess.run([path, command], env=env,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(os.path.exists(marker))

    def test_wrapper_rejects_duplicate_rank_allocations_without_running_target(self):
        spec = {"config": {}, "allocations": [
            {"rank": 0, "team_size": 1, "node": "n1", "numa_domain": 0,
             "cpus": [4]},
            {"rank": 0, "team_size": 1, "node": "n1", "numa_domain": 0,
             "cpus": [5]},
        ]}
        wrapper = wr.render_wrapper(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wrapper.sh")
            marker = os.path.join(tmp, "ran")
            with open(path, "w") as handle:
                handle.write(wrapper)
            os.chmod(path, 0o755)
            env = os.environ.copy()
            for key in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "PMIX_RANK"):
                env.pop(key, None)
            env["PMI_RANK"] = "0"
            completed = subprocess.run([path, "/bin/sh", "-c", "touch \"$0\"", marker],
                                       env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(os.path.exists(marker))

    def test_wrapper_rejects_duplicate_cpu_ids_without_running_target(self):
        spec = {"config": {}, "allocations": [{"rank": 0, "team_size": 2,
                "node": "n1", "numa_domain": 0, "cpus": [4, 4]}]}
        wrapper = wr.render_wrapper(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wrapper.sh")
            marker = os.path.join(tmp, "ran")
            with open(path, "w") as handle:
                handle.write(wrapper)
            os.chmod(path, 0o755)
            env = os.environ.copy()
            for key in ("PMI_RANK", "OMPI_COMM_WORLD_RANK", "PMIX_RANK"):
                env.pop(key, None)
            env["PMI_RANK"] = "0"
            completed = subprocess.run([path, "/bin/sh", "-c", "touch \"$0\"", marker],
                                       env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(os.path.exists(marker))

if __name__ == "__main__":
    unittest.main()
