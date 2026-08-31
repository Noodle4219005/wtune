"""Injected, fail-closed process admission for communicator replacement."""

from wtune_runtime_migration import CommunicatorTransition, ProviderError


class DynamicProcessProvider(object):
    """Provider adapter; the injected backend is not a live MPI resize hook."""

    _BACKEND_METHODS = (
        "admit_processes", "create_communicator", "publish_communicator",
        "release_processes", "retire_communicator")

    def __init__(self, backend, active, mpi_abi):
        self._require_backend(backend)
        if not isinstance(mpi_abi, str) or not mpi_abi:
            raise ProviderError("mpi_abi must be a non-empty string")
        advertised_abi = getattr(backend, "mpi_abi", mpi_abi)
        if advertised_abi != mpi_abi:
            raise ProviderError("backend MPI ABI does not match configured ABI")
        self.backend = backend
        self.mpi_abi = mpi_abi
        self.active_ranks = self._ranks(active, "active")
        self._communicator = "communicator-0"
        self._generation = 0
        self._pending = None

    @classmethod
    def _require_backend(cls, backend):
        if backend is None or any(not callable(getattr(backend, name, None))
                                 for name in cls._BACKEND_METHODS):
            raise ProviderError("backend does not implement the provider ABI")

    @staticmethod
    def _ranks(value, label):
        if isinstance(value, (str, bytes)) or value is None:
            raise ProviderError(label + " must be a non-empty ordered sequence")
        try:
            result = tuple(value)
        except TypeError:
            raise ProviderError(label + " must be a non-empty ordered sequence")
        if not result:
            raise ProviderError(label + " must be a non-empty ordered sequence")
        if any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
               for rank in result):
            raise ProviderError(label + " entries must be non-negative integer ranks")
        if len(set(result)) != len(result):
            raise ProviderError(label + " ranks must be unique")
        return result

    @property
    def pending_transition(self):
        return self._pending

    def _target(self, target_ranks):
        if isinstance(target_ranks, int) and not isinstance(target_ranks, bool):
            if target_ranks <= 0:
                raise ProviderError("target count must be positive")
            return target_ranks, None
        requested = self._ranks(target_ranks, "target")
        return requested, requested

    def _cleanup(self, ranks, communicator):
        errors = []
        if communicator is not None and communicator != self._communicator:
            try:
                self.backend.retire_communicator(communicator)
            except Exception as error:
                errors.append(error)
        if ranks:
            try:
                self.backend.release_processes(ranks)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ProviderError("backend cleanup failed")

    def prepare_resize(self, target_ranks):
        if self._pending is not None:
            raise ProviderError("resize already pending")
        requested, exact = self._target(target_ranks)
        generation = self._generation + 1
        self._generation = generation
        admitted = ()
        new_communicator = None
        try:
            admitted_value = self.backend.admit_processes(requested, self.mpi_abi)
            admitted = self._ranks(admitted_value, "admitted")
            if exact is not None and admitted != exact:
                raise ProviderError("backend admitted ranks differ from target")
            new_communicator = self.backend.create_communicator(
                self._communicator, admitted, generation, self.mpi_abi)
            if new_communicator is None or new_communicator == self._communicator:
                raise ProviderError("backend did not create a new communicator")
            transition = CommunicatorTransition(
                self.active_ranks, admitted, self._communicator,
                new_communicator, generation, transport="sparse-p2p")
            self._pending = transition
            return transition
        except ProviderError:
            try:
                self._cleanup(admitted, new_communicator)
            except ProviderError:
                pass
            raise
        except Exception as error:
            try:
                self._cleanup(admitted, new_communicator)
            except ProviderError:
                pass
            raise ProviderError("backend preparation failed") from error

    def commit_resize(self, transition=None):
        if self._pending is None or (transition is not None and
                                     transition is not self._pending):
            raise ProviderError("no exact pending transition")
        pending = self._pending
        try:
            self.backend.publish_communicator(pending.new_communicator)
        except Exception as error:
            self._pending = None
            try:
                self._cleanup(pending.new_ranks, pending.new_communicator)
            except ProviderError:
                pass
            raise ProviderError("backend publication failed") from error
        self.active_ranks = pending.new_ranks
        self._communicator = pending.new_communicator
        self._pending = None
        return pending

    def abort_resize(self, transition=None):
        if self._pending is None:
            return None
        if transition is not None and transition is not self._pending:
            raise ProviderError("transition is not pending")
        pending = self._pending
        self._pending = None
        self._cleanup(pending.new_ranks, pending.new_communicator)
        return pending
