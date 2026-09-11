import unittest
import inspect

from wtune_dynamic_provider import DynamicProcessProvider
from wtune_runtime_migration import (CommunicatorTransition, ProviderError,
                                     ReservedPoolProvider)


class RecordingBackend(object):
    def __init__(self):
        self.calls = []
        self.next_communicator = "comm-1"
        self.fail = set()
        self.admitted = None

    def admit_processes(self, target_ranks, mpi_abi):
        self.calls.append(("admit", target_ranks, mpi_abi))
        if "admit" in self.fail:
            raise RuntimeError("admit failed")
        self.admitted = tuple(range(target_ranks)) if isinstance(target_ranks, int) else tuple(target_ranks)
        return self.admitted

    def create_communicator(self, old_communicator, new_ranks, generation, mpi_abi):
        self.calls.append(("create", old_communicator, tuple(new_ranks), generation, mpi_abi))
        if "create" in self.fail:
            raise RuntimeError("create failed")
        if "same" in self.fail:
            return old_communicator
        return self.next_communicator

    def publish_communicator(self, new_communicator):
        self.calls.append(("publish", new_communicator))
        if "publish" in self.fail:
            raise RuntimeError("publish failed")

    def release_processes(self, ranks):
        self.calls.append(("release", tuple(ranks)))
        if "release" in self.fail:
            raise RuntimeError("release failed")

    def retire_communicator(self, communicator):
        self.calls.append(("retire", communicator))
        if "retire" in self.fail:
            raise RuntimeError("retire failed")


class DynamicProcessProviderTests(unittest.TestCase):
    ABI = "intel-mpi-2021.11"

    def test_provider_contract_and_exact_backend_call_order(self):
        backend = RecordingBackend()
        provider = DynamicProcessProvider(backend, (0, 1), self.ABI)
        transition = provider.prepare_resize((0, 1, 2))
        self.assertIsInstance(transition, CommunicatorTransition)
        self.assertEqual((0, 1), provider.active_ranks)
        self.assertIs(provider.pending_transition, transition)
        self.assertNotIn(("publish", transition.new_communicator), backend.calls)
        self.assertEqual(
            [("admit", (0, 1, 2), self.ABI),
             ("create", "communicator-0", (0, 1, 2), 1, self.ABI)],
            backend.calls)
        self.assertEqual("sparse-p2p", transition.transport)
        self.assertNotEqual(transition.old_communicator, transition.new_communicator)

    def test_grow_shrink_and_combined_replacement_commit_once(self):
        backend = RecordingBackend()
        provider = DynamicProcessProvider(backend, (0, 1), self.ABI)
        grow = provider.prepare_resize(3)
        provider.commit_resize(grow)
        self.assertEqual((0, 1, 2), provider.active_ranks)
        backend.next_communicator = "comm-2"
        shrink = provider.prepare_resize((1, 2))
        provider.commit_resize()
        self.assertEqual((1, 2), provider.active_ranks)
        backend.next_communicator = "comm-3"
        replacement = provider.prepare_resize((0, 3))
        provider.commit_resize(replacement)
        self.assertEqual((0, 3), provider.active_ranks)
        self.assertGreater(shrink.generation, grow.generation)
        self.assertGreater(replacement.generation, shrink.generation)
        self.assertEqual(3, len([call for call in backend.calls if call[0] == "publish"]))

    def test_prepare_failure_rolls_back_admission_and_new_communicator(self):
        for failure in ("create", "same"):
            backend = RecordingBackend()
            backend.fail.add(failure)
            provider = DynamicProcessProvider(backend, (0,), self.ABI)
            with self.assertRaises(ProviderError):
                provider.prepare_resize((0, 1))
            self.assertEqual((0,), provider.active_ranks)
            self.assertIsNone(provider.pending_transition)
            self.assertEqual(("release", (0, 1)), backend.calls[-1])
            self.assertNotIn(("retire", "communicator-0"), backend.calls)

        backend = RecordingBackend()
        backend.fail.add("create")
        provider = DynamicProcessProvider(backend, (0,), self.ABI)
        with self.assertRaises(ProviderError):
            provider.prepare_resize((0, 1))
        backend.fail.clear()
        next_transition = provider.prepare_resize((0, 1))
        self.assertGreater(next_transition.generation, 1)

    def test_commit_failure_and_abort_cleanup_are_idempotent(self):
        backend = RecordingBackend()
        backend.fail.add("publish")
        provider = DynamicProcessProvider(backend, (0,), self.ABI)
        pending = provider.prepare_resize((0, 1))
        with self.assertRaises(ProviderError):
            provider.commit_resize(pending)
        self.assertEqual((0,), provider.active_ranks)
        self.assertIsNone(provider.pending_transition)
        self.assertEqual(("retire", "comm-1"), backend.calls[-2])
        self.assertEqual(("release", (0, 1)), backend.calls[-1])
        backend.fail.clear()
        pending = provider.prepare_resize((0, 1))
        provider.abort_resize(pending)
        provider.abort_resize(pending)
        self.assertIsNone(provider.pending_transition)
        self.assertEqual(("retire", "comm-1"), backend.calls[-2])
        self.assertEqual(("release", (0, 1)), backend.calls[-1])
        self.assertGreater(pending.generation, 1)

    def test_cleanup_failure_is_surface_and_abort_retries_pending_cleanup(self):
        backend = RecordingBackend()
        provider = DynamicProcessProvider(backend, (0,), self.ABI)
        pending = provider.prepare_resize((0, 1))
        backend.fail.clear()
        backend.fail.add("retire")
        with self.assertRaises(ProviderError) as error:
            provider.abort_resize(pending)
        self.assertIn("cleanup", str(error.exception))
        self.assertIs(provider.pending_transition, pending)
        backend.fail.clear()
        provider.abort_resize(pending)
        self.assertIsNone(provider.pending_transition)
        self.assertEqual("retire", backend.calls[-1][0])
        self.assertEqual(1, len([call for call in backend.calls if call[0] == "release"]))

    def test_commit_cleanup_failure_is_surface_and_pending_is_retained(self):
        backend = RecordingBackend()
        provider = DynamicProcessProvider(backend, (0,), self.ABI)
        pending = provider.prepare_resize((0, 1))
        backend.fail.update(("publish", "retire", "release"))
        with self.assertRaises(ProviderError) as error:
            provider.commit_resize(pending)
        self.assertIn("cleanup", str(error.exception))
        self.assertIs(provider.pending_transition, pending)
        self.assertEqual((0,), provider.active_ranks)
        self.assertNotIn(("publish", pending.new_communicator), backend.calls[0:2])

    def test_provider_contract_signature_parity(self):
        for name in ("prepare_resize", "commit_resize", "abort_resize"):
            dynamic = inspect.signature(getattr(DynamicProcessProvider, name))
            reserved = inspect.signature(getattr(ReservedPoolProvider, name))
            self.assertEqual(dynamic, reserved)

    def test_distinct_communicator_uses_identity_not_only_equality(self):
        class EqualCommunicator(object):
            def __eq__(self, other):
                return True

        backend = RecordingBackend()
        equal_but_distinct = EqualCommunicator()
        backend.next_communicator = equal_but_distinct
        provider = DynamicProcessProvider(backend, (0,), self.ABI)
        transition = provider.prepare_resize((0, 1))
        self.assertIsNot(transition.new_communicator, transition.old_communicator)
        provider.abort_resize(transition)
        self.assertIn(("retire", equal_but_distinct), backend.calls)

    def test_rejects_invalid_backend_abi_and_transition_inputs(self):
        for backend in (object(),):
            with self.assertRaises(ProviderError):
                DynamicProcessProvider(backend, (0,), self.ABI)
        mismatched = RecordingBackend()
        mismatched.mpi_abi = "other-mpi"
        with self.assertRaises(ProviderError):
            DynamicProcessProvider(mismatched, (0,), self.ABI)
        backend = RecordingBackend()
        provider = DynamicProcessProvider(backend, (0,), self.ABI)
        for target in (0, -1, True, (), (0, 0), (-1,), (False,)):
            with self.assertRaises(ProviderError):
                provider.prepare_resize(target)
        pending = provider.prepare_resize((0, 1))
        with self.assertRaises(ProviderError):
            provider.commit_resize(CommunicatorTransition((0,), (0,), "x", "y", 99))
        self.assertNotIn(("publish", pending.new_communicator), backend.calls)
        provider.abort_resize(pending)

    def test_admission_failure_leaves_state_unchanged(self):
        backend = RecordingBackend()
        backend.fail.add("admit")
        provider = DynamicProcessProvider(backend, (0,), self.ABI)
        with self.assertRaises(ProviderError):
            provider.prepare_resize(2)
        self.assertEqual((0,), provider.active_ranks)
        self.assertIsNone(provider.pending_transition)
        self.assertEqual([("admit", 2, self.ABI)], backend.calls)

    def test_prepare_cleanup_failure_is_surface(self):
        backend = RecordingBackend()
        backend.fail.update(("create", "release"))
        provider = DynamicProcessProvider(backend, (0,), self.ABI)
        with self.assertRaises(ProviderError) as error:
            provider.prepare_resize((0, 1))
        self.assertIn("cleanup incomplete", str(error.exception))
        self.assertEqual((0,), provider.active_ranks)
        self.assertIsNone(provider.pending_transition)


if __name__ == "__main__":
    unittest.main()
