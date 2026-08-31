import unittest

from wtune_migration import (MigrationCoordinator, MigrationError, MigrationManifest,
                             MigrationPhase, RestartableField, verify_bitwise_payloads)


def manifest(payload=b"abc", **kwargs):
    values = dict(domain_clock={}, alarms=[], nesting={"max_dom": 1}, halo={},
                  cartesian_communicator={}, rank_coordinates=[], runtime_cache={},
                  derived_communication={})
    values.update(kwargs)
    values["fields"] = [RestartableField("T", [0, 1], "cell", "float32", "owner", payload)]
    return MigrationManifest(**values)


class MigrationTests(unittest.TestCase):
    def test_happy_path_round_trip_and_epoch(self):
        source, target = manifest(), manifest()
        self.assertEqual(MigrationManifest.from_dict(source.to_dict()).canonical_bytes(), source.canonical_bytes())
        c = MigrationCoordinator(source); c.poll(); c.prepare(target); c.transfer(target); c.commit()
        self.assertEqual(c.phase, MigrationPhase.COMMIT); self.assertIs(c.manifest, target)
        self.assertEqual(target.epoch, 1); self.assertIsNone(c._pending); c.finalize()

    def test_invalid_transition_and_validation_edges(self):
        c = MigrationCoordinator(manifest())
        for call in (c.commit, c.finalize):
            with self.assertRaises(MigrationError): call()
        with self.assertRaises(MigrationError): c.transfer(manifest())
        for bad in (manifest(nesting={"max_dom": 2}), manifest(scope="nested"),
                    manifest(field_inventory_complete=False)):
            bad_coordinator = MigrationCoordinator(manifest()); bad_coordinator.poll()
            with self.assertRaises(MigrationError): bad_coordinator.prepare(bad)
        with self.assertRaises(MigrationError): MigrationManifest.from_dict({})
        with self.assertRaises(MigrationError): RestartableField("x", [], "s", "d", "o", b"")
        with self.assertRaises(MigrationError): RestartableField("x", [0], "s", "d", "o", "bytes")

    def test_payload_mismatch_and_extra_missing_fields(self):
        with self.assertRaises(MigrationError): verify_bitwise_payloads(manifest(), manifest(b"abd"))
        extra = manifest(); extra.fields.append(RestartableField("U", [0, 1], "cell", "float32", "owner", b"x"))
        with self.assertRaises(MigrationError): verify_bitwise_payloads(manifest(), extra)
        c = MigrationCoordinator(manifest()); c.poll(); c.prepare(manifest()); c.transfer(c._pending)
        c._pending.fields[0].payload = b"abd"
        with self.assertRaises(MigrationError): c.commit()
        self.assertEqual(c.phase, MigrationPhase.ABORT); self.assertEqual(c.manifest.fields[0].payload, b"abc")

    def test_abort_is_fail_closed_and_idempotent(self):
        c = MigrationCoordinator(manifest()); target = manifest(); c.poll(); c.prepare(target)
        c.abort("cancelled"); c.abort("again")
        self.assertEqual(c.phase, MigrationPhase.ABORT); self.assertIs(c.manifest, c.manifest)
        with self.assertRaises(MigrationError): c.finalize()


if __name__ == "__main__":
    unittest.main()
