"""Out-of-process WRF runtime controller (observe -> policy -> actuate), phase 1.

Never links into, wraps, or otherwise modifies the WRF process. Every module here only
reads /proc metadata and calls the standard affinity syscalls (sched_getaffinity /
sched_setaffinity) on TIDs it does not own by ptrace or memory access -- see
knowledge/2026-09-13-f1-no-unprivileged-ebpf-or-sched-tracepoints-proc-polling-and-perf-stat-still-work.md
in the wiki for the permission survey this design is built from, and
/home/u5727520/.claude/plans/mossy-honking-torvalds.md for the approved phase-1 plan.
"""
