"""Dual-mode profiler backend: PollingProfiler (confirmed usable on this cluster today)
and PrivilegedProfiler (interface-compatible stub, not implemented/tested -- see
privileged.py). capability.select_backend() picks between them; everything downstream
(state_model, policy, engine) depends only on profiler.base.Profiler.
"""
from .base import Profiler, ThreadRunState, ThreadSample

__all__ = ["Profiler", "ThreadRunState", "ThreadSample"]
