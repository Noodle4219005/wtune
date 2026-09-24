import unittest

from wtune_migration import MigrationError, MigrationManifest, RestartableField
from wtune_runtime_migration import (ProviderError, RectangleOwnership,
                                     RectanglePatch, ReservedPoolProvider,
                                     SingleDomainMigrationEngine)


def manifest(payload=b"state", **kwargs):
    values = {
        "domain_clock": {"step": 1}, "alarms": [],
        "nesting": {"max_dom": 1}, "halo": {"width": 1},
        "cartesian_communicator": {"dims": [1]}, "rank_coordinates": [],
        "runtime_cache": {}, "derived_communication": {},
    }
    values.update(kwargs)
    return MigrationManifest(
        [RestartableField("temperature", [0, 1], "none", "bytes",
                          "single-domain", payload)], **values)


class RuntimeMigrationTests(unittest.TestCase):
    def patches(self):
        return RectangleOwnership([
            RectanglePatch("west", (0, 0, 1, 2), b"ab", 0),
            RectanglePatch("east", (1, 0, 2, 2), b"cd", 1),
        ], (0, 1))

    def test_provider_grow_shrink_and_combined_replacement(self):
        provider = ReservedPoolProvider((0, 1, 2, 3), active=(0, 1))
        grow = provider.prepare_resize(3)
        self.assertEqual((0, 1, 2), grow.new_ranks)
        self.assertEqual("sparse-p2p", grow.transport)
        self.assertNotEqual(grow.old_communicator, grow.new_communicator)
        provider.commit_resize()
        shrink = provider.prepare_resize((1, 2))
        self.assertEqual((1, 2), shrink.new_ranks)
        provider.commit_resize()
        replacement = provider.prepare_resize((0, 3))
        self.assertEqual((0, 3), replacement.new_ranks)
        self.assertGreater(replacement.generation, shrink.generation)

    def test_provider_pending_does_not_publish_and_abort_is_idempotent(self):
        provider = ReservedPoolProvider((0, 1, 2), active=(0,))
        pending = provider.prepare_resize(2)
        self.assertEqual((0,), provider.active_ranks)
        self.assertIs(provider.pending_transition, pending)
        aborted_generation = pending.generation
        provider.abort_resize(pending)
        provider.abort_resize(pending)
        self.assertEqual((0,), provider.active_ranks)
        self.assertIsNone(provider.pending_transition)
        next_pending = provider.prepare_resize(2)
        self.assertGreater(next_pending.generation, aborted_generation)

    def test_rectangle_zero_and_one_cell_and_exactly_once_reassignment(self):
        ownership = RectangleOwnership([
            RectanglePatch("empty", (2, 2, 2, 2), b"", 0),
            RectanglePatch("cell", (2, 2, 3, 3), b"x", 0),
        ], (0,))
        moved = ownership.reassign((4, 5))
        self.assertTrue(moved.verify_exactly_once())
        self.assertEqual((2, 2, 3, 3), moved.patches[0].bounds)
        self.assertEqual(1, moved.payload_bytes)
        self.assertEqual({"empty", "cell"}, {p.patch_id for p in moved.patches})

    def test_engine_trace_hooks_result_and_bitwise_migration(self):
        events, contexts = [], []
        hooks = {name: (lambda context, name=name: (events.append(name), contexts.append(context)))
                 for name in ("drain", "freeze", "communicator", "pack",
                              "transfer", "unpack", "mesh", "patch", "halo", "peer")}
        engine = SingleDomainMigrationEngine(manifest(), ReservedPoolProvider((0, 1, 2), (0, 1)),
                                             self.patches(), hooks)
        result = engine.migrate(manifest(), 3)
        self.assertEqual(("drain", "freeze", "bridge", "pack", "transfer",
                          "unpack", "rebuild", "verify", "commit", "finalize"),
                         result.trace)
        self.assertEqual(("drain", "freeze", "communicator", "pack", "transfer",
                          "unpack", "communicator", "mesh", "patch", "halo", "peer"),
                         tuple(events))
        self.assertEqual("sparse-p2p", result.transport)
        self.assertEqual(4, result.payload_bytes)
        self.assertEqual(1, result.epoch)
        self.assertTrue(all({"source_manifest", "target_manifest", "transition", "ownership"}
                            .issubset(context) for context in contexts))

    def test_bitwise_nested_and_incomplete_targets_fail_closed(self):
        provider = ReservedPoolProvider((0, 1), (0,))
        engine = SingleDomainMigrationEngine(manifest(), provider, self.patches())
        for target in (manifest(b"changed"),
                       manifest(nesting={"max_dom": 2}),
                       manifest(field_inventory_complete=False)):
            before = (engine.manifest, engine.ownership, provider.active_ranks)
            expected = MigrationError if target.fields[0].payload != b"state" else ValueError
            with self.assertRaises(expected):
                engine.migrate(target, 2)
            self.assertIs(engine.manifest, before[0])
            self.assertIs(engine.ownership, before[1])
            self.assertEqual(provider.active_ranks, before[2])

    def test_hook_and_provider_failures_rollback_and_pending_clears(self):
        provider = ReservedPoolProvider((0, 1), (0,))
        def fail(_):
            raise RuntimeError("hook failed")
        engine = SingleDomainMigrationEngine(manifest(), provider, self.patches(), {"mesh": fail})
        before_manifest, before_ownership = engine.manifest, engine.ownership
        with self.assertRaises(RuntimeError):
            engine.migrate(manifest(), 2)
        self.assertIs(engine.manifest, before_manifest)
        self.assertIs(engine.ownership, before_ownership)
        self.assertIsNone(provider.pending_transition)
        self.assertEqual((0,), provider.active_ranks)
        provider = ReservedPoolProvider((0, 1), (0,))
        provider.fail_next = True
        engine = SingleDomainMigrationEngine(manifest(), provider, self.patches())
        before_manifest, before_ownership = engine.manifest, engine.ownership
        with self.assertRaises(ProviderError):
            engine.migrate(manifest(), 2)
        self.assertIs(engine.manifest, before_manifest)
        self.assertIs(engine.ownership, before_ownership)
        self.assertEqual((0,), provider.active_ranks)


if __name__ == "__main__":
    unittest.main()
