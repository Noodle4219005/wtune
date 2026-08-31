"""Dependency-free safe-point runtime migration simulation; not MPI execution."""

from wtune_migration import MigrationManifest, verify_bitwise_payloads


class ProviderError(Exception):
    pass


class CommunicatorTransition(object):
    def __init__(self, old_ranks, new_ranks, old_communicator,
                 new_communicator, generation, transport="sparse-p2p"):
        self.old_ranks = tuple(old_ranks)
        self.new_ranks = tuple(new_ranks)
        self.old_communicator = old_communicator
        self.new_communicator = new_communicator
        self.old_communicator_id = old_communicator
        self.new_communicator_id = new_communicator
        self.generation = generation
        self.transport = transport


class ReservedPoolProvider(object):
    def __init__(self, pool, active=None):
        self.pool = self._ranks(pool, "pool", allow_empty=False)
        self.active_ranks = self.pool if active is None else self._ranks(active, "active", allow_empty=False)
        if not set(self.active_ranks).issubset(set(self.pool)):
            raise ProviderError("active ranks must be in pool")
        self._generation = 0
        self._pending = None
        self.fail_next = False

    @staticmethod
    def _ranks(value, label, allow_empty):
        if not isinstance(value, (list, tuple)) or (not allow_empty and not value):
            raise ProviderError(label + " must be a non-empty ordered sequence")
        result = tuple(value)
        if any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 for rank in result):
            raise ProviderError(label + " entries must be non-negative integer ranks")
        if len(set(result)) != len(result):
            raise ProviderError(label + " ranks must be unique")
        return result

    @property
    def pending_transition(self):
        return self._pending

    def prepare_resize(self, target_ranks):
        if self._pending is not None:
            raise ProviderError("resize already pending")
        if isinstance(target_ranks, int) and not isinstance(target_ranks, bool):
            if target_ranks <= 0 or target_ranks > len(self.pool):
                raise ProviderError("target count is outside reserved pool")
            new_ranks = self.pool[:target_ranks]
        else:
            new_ranks = self._ranks(target_ranks, "target", allow_empty=False)
            if not set(new_ranks).issubset(set(self.pool)):
                raise ProviderError("target ranks must be in pool")
        # Reserve the epoch at prepare time so aborted transitions cannot
        # cause a later communicator epoch to be reused.
        old_generation = self._generation
        self._generation += 1
        generation = self._generation
        self._pending = CommunicatorTransition(
            self.active_ranks, new_ranks,
            "communicator-%d" % old_generation,
            "communicator-%d" % generation, generation)
        return self._pending

    def commit_resize(self, transition=None):
        if self.fail_next:
            self.fail_next = False
            raise ProviderError("provider commit failure")
        if self._pending is None or (transition is not None and transition is not self._pending):
            raise ProviderError("no exact pending transition")
        pending = self._pending
        self.active_ranks = pending.new_ranks
        self._pending = None
        return pending

    def abort_resize(self, transition=None):
        if transition is not None and self._pending is not None and transition is not self._pending:
            raise ProviderError("transition is not pending")
        self._pending = None


class RectanglePatch(object):
    def __init__(self, patch_id, bounds, payload, owner):
        if not isinstance(patch_id, str) or not patch_id:
            raise ValueError("patch_id must be non-empty")
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or
                any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in bounds) or
                bounds[2] < bounds[0] or bounds[3] < bounds[1]):
            raise ValueError("bounds must be non-negative half-open rectangle")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("payload must be bytes-like")
        if not isinstance(owner, int) or isinstance(owner, bool) or owner < 0:
            raise ValueError("owner must be a non-negative rank")
        self.patch_id, self.bounds = patch_id, tuple(bounds)
        self.payload, self.owner = bytes(payload), owner


class RectangleOwnership(object):
    def __init__(self, patches, active_ranks):
        self.patches = tuple(patches)
        self.active_ranks = ReservedPoolProvider._ranks(active_ranks, "active", False)
        ids = set()
        for patch in self.patches:
            if not isinstance(patch, RectanglePatch):
                raise ValueError("patches must contain RectanglePatch values")
            if patch.patch_id in ids:
                raise ValueError("duplicate patch ID")
            if patch.owner not in self.active_ranks:
                raise ValueError("patch owner is not active")
            ids.add(patch.patch_id)

    @property
    def payload_bytes(self):
        return sum(len(patch.payload) for patch in self.patches)

    def verify_exactly_once(self):
        ids = [patch.patch_id for patch in self.patches]
        return (len(ids) == len(set(ids)) and
                all(patch.owner in self.active_ranks for patch in self.patches))

    def reassign(self, active_ranks):
        ranks = ReservedPoolProvider._ranks(active_ranks, "active", False)
        ordered = sorted(self.patches, key=lambda patch: patch.patch_id)
        reassigned = [RectanglePatch(p.patch_id, p.bounds, p.payload,
                                     ranks[index % len(ranks)])
                      for index, p in enumerate(ordered)]
        return RectangleOwnership(reassigned, ranks)


class MigrationResult(object):
    def __init__(self, transition, ownership, epoch, payload_bytes,
                 transport, trace):
        self.transition = transition
        self.ownership = ownership
        self.epoch = epoch
        self.payload_bytes = payload_bytes
        self.transport = transport
        self.trace = tuple(trace)


class SingleDomainMigrationEngine(object):
    _trace = ("drain", "freeze", "bridge", "pack", "transfer", "unpack",
              "rebuild", "verify", "commit", "finalize")
    _rebuild_hooks = ("communicator", "mesh", "patch", "halo", "peer")

    def __init__(self, manifest, provider, ownership, hooks=None):
        self._validate_manifest(manifest)
        if not isinstance(provider, ReservedPoolProvider):
            raise ProviderError("reserved pool provider required")
        if not isinstance(ownership, RectangleOwnership):
            raise ValueError("rectangle ownership required")
        self._manifest, self._ownership, self.provider = manifest, ownership, provider
        self.hooks = {} if hooks is None else dict(hooks)
        self._last_trace = ()

    @staticmethod
    def _validate_manifest(value):
        if not isinstance(value, MigrationManifest):
            raise ValueError("manifest required")
        if value.scope != "single-domain" or value.nesting.get("max_dom") != 1:
            raise ValueError("nested-domain metadata is not allowed")
        if not value.field_inventory_complete:
            raise ValueError("field inventory is incomplete")

    @property
    def manifest(self):
        return self._manifest

    @property
    def ownership(self):
        return self._ownership

    @property
    def last_trace(self):
        return self._last_trace

    def _hook(self, name, context):
        callback = self.hooks.get(name)
        if callback is not None:
            callback(context)

    def migrate(self, target_manifest, target_ranks):
        self._validate_manifest(target_manifest)
        original_manifest, original_ownership = self._manifest, self._ownership
        transition = None
        trace = []
        try:
            verify_bitwise_payloads(original_manifest, target_manifest)
            target_ownership = original_ownership.reassign(target_ranks if not isinstance(target_ranks, int)
                                                           else tuple(self.provider.pool[:target_ranks]))
            transition = self.provider.prepare_resize(target_ranks)
            context = {"source_manifest": original_manifest,
                       "target_manifest": target_manifest,
                       "source": original_manifest, "target": target_manifest,
                       "transition": transition, "ownership": target_ownership}
            for step in self._trace:
                trace.append(step)
                if step in ("drain", "freeze"):
                    self._hook(step, context)
                elif step == "bridge":
                    self._hook("communicator", context)
                elif step in ("pack", "transfer", "unpack"):
                    self._hook(step, context)
                elif step == "rebuild":
                    for hook in self._rebuild_hooks:
                        self._hook(hook, context)
                elif step == "verify":
                    if not target_ownership.verify_exactly_once():
                        raise ValueError("ownership verification failed")
                elif step == "commit":
                    self.provider.commit_resize(transition)
                    target_manifest.epoch = original_manifest.epoch + 1
                    self._manifest, self._ownership = target_manifest, target_ownership
                elif step == "finalize":
                    pass
            self._last_trace = tuple(trace)
            return MigrationResult(transition, self._ownership, self._manifest.epoch,
                                   self._ownership.payload_bytes, transition.transport, trace)
        except Exception:
            self.provider.abort_resize()
            self._manifest, self._ownership = original_manifest, original_ownership
            raise
