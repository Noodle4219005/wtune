#ifndef WT_MPI_REMOTE_H
#define WT_MPI_REMOTE_H

#include "wt_ledger.h"
#include "wt_mpi_channel.h"
#include "wt_policy.h"

#include <stddef.h>
#include <stdint.h>

/* Async owner/worker protocol over one fixed-peer MPI channel. It provides
 * process-lifetime at-most-once ledger behavior only; it does not claim
 * exactly-once execution or recovery across MPI/process failure. Every entry
 * is serialized on the channel creator thread and performs at most one
 * nonblocking channel progress pass. */
enum wt_mpi_remote_status {
  WT_MPI_REMOTE_OK = 0,
  WT_MPI_REMOTE_ARGUMENT,
  WT_MPI_REMOTE_ALLOCATION,
  WT_MPI_REMOTE_POLICY_REFUSED,
  WT_MPI_REMOTE_BAD_PACKET,
  WT_MPI_REMOTE_DUPLICATE,
  WT_MPI_REMOTE_ENDPOINT_MISMATCH,
  WT_MPI_REMOTE_LEDGER,
  WT_MPI_REMOTE_CALLBACK_FAILED,
  WT_MPI_REMOTE_BUSY,
  WT_MPI_REMOTE_TRANSPORT,
  WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY,
  WT_MPI_REMOTE_UNKNOWN_DELIVERY,
  WT_MPI_REMOTE_DEADLINE_EXPIRED_RESULT_KNOWN,
  WT_MPI_REMOTE_POISONED,
  WT_MPI_REMOTE_ADMISSION_RELEASE_FAILED
};

enum wt_mpi_remote_event_kind {
  WT_MPI_REMOTE_EVENT_NONE = 0,
  WT_MPI_REMOTE_EVENT_OWNER_COMMITTED = 1U << 0,
  WT_MPI_REMOTE_EVENT_WORKER_EXECUTED = 1U << 1,
  WT_MPI_REMOTE_EVENT_WORKER_CACHED_RESULT = 1U << 2,
  WT_MPI_REMOTE_EVENT_SEND_COMPLETE = 1U << 3,
  WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY = 1U << 4
};

struct wt_mpi_remote_event {
  unsigned kinds;
  enum wt_mpi_channel_status channel_status;
  enum wt_ledger_status ledger_status;
  enum wt_policy_status policy_status;
};

/* Trusted configuration binds an opaque logical rank/node identity to the
 * channel's actual fixed MPI peer. Received wire fields never create or change
 * this authority. */
struct wt_mpi_remote_endpoint {
  int mpi_peer;
  uint32_t rank_id;
  uint32_t node_id;
};

struct wt_mpi_remote_packet {
  const uint8_t *request;
  size_t request_len;
  size_t result_limit;
};

struct wt_mpi_remote_owner;
struct wt_mpi_remote_worker;

/* These constructors are local and allocate only adapter state. channel,
 * ledger, pool and contract must already be valid and serialized by the
 * caller. The ledger, pool, contract and trusted endpoint configuration are
 * borrowed logically: the pointed-to ledger/pool must outlive the adapter and
 * their configuration and the contract must not be mutated while it lives.
 * The trusted endpoint value itself is copied. On success ownership of *channel
 * moves atomically into the adapter
 * and *channel becomes NULL; on failure *channel is unchanged. */
enum wt_mpi_remote_status wt_mpi_remote_owner_create_take(
    struct wt_mpi_channel **channel,
    const struct wt_mpi_remote_endpoint *endpoint,
    struct wt_ledger *ledger, struct wt_policy_pool *pool,
    const struct wt_ledger_contract *contract,
    struct wt_mpi_remote_owner **out);

enum wt_mpi_remote_status wt_mpi_remote_worker_create_take(
    struct wt_mpi_channel **channel,
    const struct wt_mpi_remote_endpoint *endpoint,
    struct wt_ledger *ledger, const struct wt_ledger_contract *contract,
    wt_ledger_run_callback run, void *run_context,
    struct wt_mpi_remote_worker **out);

/* begin stores the absolute CLOCK_MONOTONIC deadline; later progress calls
 * cannot extend it. Request bytes are ledger- and channel-owned before begin
 * returns. A duplicate ledger request publishes no MPI message. */
enum wt_mpi_remote_status wt_mpi_remote_owner_begin(
    struct wt_mpi_remote_owner *owner,
    const struct wt_policy_config *policy,
    const struct wt_policy_request *request,
    const struct wt_policy_candidate *candidates, size_t candidate_count,
    const struct wt_mpi_remote_packet *packet,
    int64_t deadline_monotonic_ns,
    wt_ledger_commit_callback commit, void *commit_context,
    struct wt_policy_choice *choice);

enum wt_mpi_remote_status wt_mpi_remote_owner_progress(
    struct wt_mpi_remote_owner *owner, struct wt_mpi_remote_event *event);

/* abandon never cancels MPI. Once work may have been published it permanently
 * fails the ledger entry, reports unknown delivery, retains admission, and
 * requires progress/reap followed by collective shutdown. It never replays. */
enum wt_mpi_remote_status wt_mpi_remote_owner_abandon(
    struct wt_mpi_remote_owner *owner, struct wt_mpi_remote_event *event);

/* A worker is deliberately not re-armed after one result send completes; the
 * service loop explicitly arms each receive window so a healthy idle endpoint
 * can become quiescent for shutdown. */
enum wt_mpi_remote_status wt_mpi_remote_worker_arm(
    struct wt_mpi_remote_worker *worker, int64_t deadline_monotonic_ns);
enum wt_mpi_remote_status wt_mpi_remote_worker_progress(
    struct wt_mpi_remote_worker *worker, struct wt_mpi_remote_event *event);

/* ready calls are purely local. destroy_collective must be entered by every
 * participant in matching order after external global readiness is established;
 * it must not return early from role-local failed state before the channel's
 * collective agreement. Timeout-held admission is released only after channel
 * destruction succeeds globally. MPI_Finalize follows successful shutdown. */
enum wt_mpi_remote_status wt_mpi_remote_owner_shutdown_ready(
    const struct wt_mpi_remote_owner *owner);
enum wt_mpi_remote_status wt_mpi_remote_worker_shutdown_ready(
    const struct wt_mpi_remote_worker *worker);
enum wt_mpi_remote_status wt_mpi_remote_owner_destroy_collective(
    struct wt_mpi_remote_owner **owner);
/* If collective destruction closed the channel but admission release failed,
 * the owner and private ticket remain intact. This local call explicitly
 * retries that release once and frees only on success; it performs no MPI. */
enum wt_mpi_remote_status wt_mpi_remote_owner_finish_closed(
    struct wt_mpi_remote_owner **owner);
enum wt_mpi_remote_status wt_mpi_remote_worker_destroy_collective(
    struct wt_mpi_remote_worker **worker);

#endif
