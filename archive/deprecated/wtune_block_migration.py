"""Dependency-free block ownership and migration planning contract.

This module plans and verifies migrations; it does not execute MPI or WRF.
"""

from wtune_runtime_migration import ProviderError


class StaleEpochError(ProviderError):
    """Raised when a cache or ownership publication is not current."""


def _rank_sequence(value, label):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("%s must be a non-empty sequence" % label)
    result = tuple(value)
    if any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
           for rank in result):
        raise ValueError("%s must contain non-negative integer ranks" % label)
    if len(set(result)) != len(result):
        raise ValueError("%s ranks must be unique" % label)
    return result


def _block_key(block_id):
    return (type(block_id).__name__, repr(block_id))


class BlockChunk(object):
    def __init__(self, block_id, bounds, payload, owner, generation=0):
        try:
            hash(block_id)
        except TypeError:
            raise ValueError("block_id must be hashable")
        if (not isinstance(bounds, tuple) or len(bounds) != 4 or
                any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in bounds) or bounds[2] < bounds[0] or
                bounds[3] < bounds[1]):
            raise ValueError("bounds must be non-negative half-open four-int tuple")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("payload must be bytes-like")
        if (not isinstance(owner, int) or isinstance(owner, bool) or owner < 0 or
                not isinstance(generation, int) or isinstance(generation, bool) or
                generation < 0):
            raise ValueError("owner and generation must be non-negative integers")
        self.block_id = block_id
        self.bounds = tuple(bounds)
        self.payload = bytes(payload)
        self.owner = owner
        self.generation = generation


class BlockOwnership(object):
    def __init__(self, blocks, active_ranks, epoch=0):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.blocks = tuple(blocks)
        self.active_ranks = _rank_sequence(active_ranks, "active_ranks")
        if any(not isinstance(block, BlockChunk) for block in self.blocks):
            raise ValueError("blocks must contain BlockChunk values")
        ids = [block.block_id for block in self.blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("block IDs must be unique")
        if any(block.owner not in self.active_ranks for block in self.blocks):
            raise ValueError("block owner must be active")
        self.epoch = epoch

    @property
    def payload_bytes(self):
        return sum(len(block.payload) for block in self.blocks)

    def verify_exactly_once(self):
        ids = [block.block_id for block in self.blocks]
        return (len(ids) == len(set(ids)) and
                all(block.owner in self.active_ranks for block in self.blocks))

    def reassign(self, active_ranks, generation=None):
        ranks = _rank_sequence(active_ranks, "active_ranks")
        generation = self.epoch if generation is None else generation
        if (not isinstance(generation, int) or isinstance(generation, bool) or
                generation < 0):
            raise ValueError("generation must be a non-negative integer")
        loads = dict((rank, 0) for rank in ranks)
        assignments = {}
        ordered = sorted(self.blocks,
                         key=lambda block: (-len(block.payload), _block_key(block.block_id)))
        for block in ordered:
            owner = min(ranks, key=lambda rank: (loads[rank], rank))
            assignments[block.block_id] = owner
            loads[owner] += len(block.payload)
        reassigned = tuple(BlockChunk(block.block_id, block.bounds, block.payload,
                                      assignments[block.block_id], generation)
                           for block in self.blocks)
        return BlockOwnership(reassigned, ranks, self.epoch)


class OwnershipRegistry(object):
    def __init__(self, ownership):
        if not isinstance(ownership, BlockOwnership):
            raise ValueError("ownership must be BlockOwnership")
        self._ownership = ownership

    @property
    def epoch(self):
        return self._ownership.epoch

    @property
    def ownership(self):
        return self._ownership

    def publish(self, ownership):
        if (not isinstance(ownership, BlockOwnership) or
                ownership.epoch != self.epoch + 1):
            raise StaleEpochError("publication must be exactly the next epoch")
        _validated_pairs(self.ownership, ownership)
        self._ownership = ownership
        return ownership

    def validate_cache(self, block_id, owner, epoch):
        if epoch != self.epoch:
            raise StaleEpochError("cache epoch is stale")
        matches = [block for block in self.ownership.blocks if block.block_id == block_id]
        if len(matches) != 1 or matches[0].owner != owner:
            raise StaleEpochError("cache block or owner is stale")
        return True


class TransferRecord(object):
    def __init__(self, block_id, source_rank, target_rank, payload):
        self.block_id = block_id
        self.source_rank = source_rank
        self.target_rank = target_rank
        self.payload = payload


class TransferPlan(object):
    def __init__(self, transfers, transport):
        self.transfers = tuple(transfers)
        self.transport = transport


def _validated_pairs(source, target):
    if not isinstance(source, BlockOwnership) or not isinstance(target, BlockOwnership):
        raise ValueError("source and target must be BlockOwnership")
    source_ids = [block.block_id for block in source.blocks]
    target_ids = [block.block_id for block in target.blocks]
    if len(source_ids) != len(set(source_ids)) or len(target_ids) != len(set(target_ids)):
        raise ValueError("block IDs must be unique")
    source_by_id = dict((block.block_id, block) for block in source.blocks)
    target_by_id = dict((block.block_id, block) for block in target.blocks)
    if set(source_by_id) != set(target_by_id):
        raise ValueError("source and target block IDs must match")
    for block_id in source_by_id:
        if source_by_id[block_id].payload != target_by_id[block_id].payload:
            raise ValueError("block payloads must match bitwise")
    return source_by_id, target_by_id


def _plan(source, target, transport, changed_only):
    source_by_id, target_by_id = _validated_pairs(source, target)
    records = []
    for block_id in sorted(source_by_id, key=_block_key):
        old, new = source_by_id[block_id], target_by_id[block_id]
        if changed_only and old.owner == new.owner:
            continue
        records.append(TransferRecord(block_id, old.owner, new.owner, old.payload))
    return TransferPlan(records, transport)


def plan_sparse_transfers(source, target):
    return _plan(source, target, "sparse-p2p", True)


def plan_all_to_all_comparison(source, target):
    return _plan(source, target, "all-to-all-comparison", False)


class BlockMigrationResult(object):
    def __init__(self, transfers, ownership, epoch, payload_bytes, transport):
        self.transfers = tuple(transfers)
        self.ownership = ownership
        self.epoch = epoch
        self.payload_bytes = payload_bytes
        self.transport = transport


class BlockMigrationEngine(object):
    def __init__(self, registry):
        if not isinstance(registry, OwnershipRegistry):
            raise ValueError("registry must be OwnershipRegistry")
        self.registry = registry

    def migrate(self, target_ownership):
        if (not isinstance(target_ownership, BlockOwnership) or
                target_ownership.epoch != self.registry.epoch + 1):
            raise StaleEpochError("target ownership must be the next epoch")
        plan = plan_sparse_transfers(self.registry.ownership, target_ownership)
        self.registry.publish(target_ownership)
        return BlockMigrationResult(plan.transfers, target_ownership,
                                    target_ownership.epoch,
                                    target_ownership.payload_bytes,
                                    plan.transport)
