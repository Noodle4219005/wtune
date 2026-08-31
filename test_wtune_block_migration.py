import unittest

from wtune_block_migration import (
    BlockChunk,
    BlockMigrationEngine,
    BlockOwnership,
    OwnershipRegistry,
    StaleEpochError,
    plan_all_to_all_comparison,
    plan_sparse_transfers,
)


def chunk(block_id, size, owner, bounds=None, generation=0):
    return BlockChunk(block_id, bounds or (0, 0, 1, size), bytes([block_id]), owner,
                      generation)


class BlockMigrationTests(unittest.TestCase):
    def test_weighted_reassignment_is_deterministic_and_exactly_once(self):
        source = BlockOwnership((chunk(1, 1, 0), chunk(2, 5, 1), chunk(3, 3, 0),
                                 chunk(4, 5, 1)), (0, 1), epoch=4)
        result = source.reassign((2, 7), generation=9)
        self.assertEqual([item.owner for item in result.blocks], [2, 7, 2, 7])
        self.assertEqual([item.generation for item in result.blocks], [9] * 4)
        self.assertTrue(result.verify_exactly_once())
        self.assertEqual(result.payload_bytes, source.payload_bytes)

    def test_zero_area_and_one_cell_chunks_coexist(self):
        ownership = BlockOwnership((
            BlockChunk("empty", (2, 2, 2, 9), b"", 0),
            BlockChunk("cell", (0, 0, 1, 1), b"x", 1),
        ), (0, 1))
        self.assertTrue(ownership.verify_exactly_once())
        self.assertEqual(ownership.payload_bytes, 1)

    def test_registry_is_next_epoch_only_and_cache_fail_closed(self):
        old = BlockOwnership((chunk(1, 1, 0),), (0,), epoch=2)
        registry = OwnershipRegistry(old)
        self.assertEqual(registry.epoch, 2)
        with self.assertRaises(StaleEpochError):
            registry.publish(BlockOwnership((chunk(1, 1, 0),), (0,), epoch=2))
        with self.assertRaises(StaleEpochError):
            registry.publish(BlockOwnership((chunk(1, 1, 0),), (0,), epoch=4))
        with self.assertRaises(StaleEpochError):
            registry.validate_cache(1, 0, 1)
        with self.assertRaises(StaleEpochError):
            registry.validate_cache("missing", 0, 2)
        with self.assertRaises(StaleEpochError):
            registry.validate_cache(1, 9, 2)
        self.assertIs(registry.ownership, old)

        new = BlockOwnership((chunk(1, 1, 1),), (1,), epoch=3)
        registry.publish(new)
        self.assertIs(registry.ownership, new)
        self.assertTrue(registry.validate_cache(1, 1, 3))

    def test_sparse_and_dense_comparison_have_distinct_labels(self):
        source = BlockOwnership((chunk(1, 1, 0), chunk(2, 2, 1)), (0, 1), epoch=1)
        target = BlockOwnership((chunk(1, 1, 1), chunk(2, 2, 1)), (0, 1), epoch=2)
        sparse = plan_sparse_transfers(source, target)
        dense = plan_all_to_all_comparison(source, target)
        self.assertEqual(sparse.transport, "sparse-p2p")
        self.assertEqual(dense.transport, "all-to-all-comparison")
        self.assertEqual(len(sparse.transfers), 1)
        self.assertEqual(sparse.transfers[0].payload, b"\x01")
        self.assertEqual(len(dense.transfers), 2)

    def test_planners_reject_id_and_payload_mismatches(self):
        source = BlockOwnership((chunk(1, 1, 0),), (0,))
        with self.assertRaises(ValueError):
            plan_sparse_transfers(source, BlockOwnership((chunk(2, 1, 1),), (1,), epoch=1))
        with self.assertRaises(ValueError):
            plan_sparse_transfers(source, BlockOwnership((BlockChunk(1, (0, 0, 1, 1), b"z", 1),), (1,), epoch=1))

    def test_engine_publishes_sparse_result_and_rolls_back_on_failure(self):
        source = BlockOwnership((chunk(1, 1, 0), chunk(2, 2, 1)), (0, 1), epoch=0)
        registry = OwnershipRegistry(source)
        engine = BlockMigrationEngine(registry)
        target = BlockOwnership((chunk(1, 1, 1), chunk(2, 2, 1)), (0, 1), epoch=1)
        result = engine.migrate(target)
        self.assertEqual(result.transport, "sparse-p2p")
        self.assertIs(registry.ownership, target)
        self.assertEqual(result.payload_bytes, source.payload_bytes)

        before = registry.ownership
        with self.assertRaises(StaleEpochError):
            engine.migrate(BlockOwnership((chunk(1, 1, 0), chunk(2, 2, 1)), (0, 1), epoch=1))
        self.assertIs(registry.ownership, before)


if __name__ == "__main__":
    unittest.main()
