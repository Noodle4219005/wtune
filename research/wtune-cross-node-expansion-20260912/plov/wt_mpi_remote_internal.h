#ifndef WT_MPI_REMOTE_INTERNAL_H
#define WT_MPI_REMOTE_INTERNAL_H

#include "wt_mpi_remote.h"

/* Private state is intentionally single-operation and queue-free. A live
 * ticket flag, rather than an exposed ticket, is the authority for each
 * one-time ledger/admission transition. */
enum wt_mpi_remote_owner_state {
  WT_MPI_REMOTE_OWNER_IDLE = 0,
  WT_MPI_REMOTE_OWNER_ACTIVE,
  WT_MPI_REMOTE_OWNER_COMMITTED_DRAINING,
  WT_MPI_REMOTE_OWNER_FAILED_DRAINING,
  WT_MPI_REMOTE_OWNER_POISONED,
  WT_MPI_REMOTE_OWNER_CHANNEL_CLOSED
};

enum wt_mpi_remote_worker_state {
  WT_MPI_REMOTE_WORKER_IDLE = 0,
  WT_MPI_REMOTE_WORKER_RECEIVING,
  WT_MPI_REMOTE_WORKER_SENDING,
  WT_MPI_REMOTE_WORKER_FAILED_DRAINING,
  WT_MPI_REMOTE_WORKER_POISONED,
  WT_MPI_REMOTE_WORKER_CHANNEL_CLOSED
};

struct wt_mpi_remote_owner {
  struct wt_mpi_channel *channel;
  struct wt_ledger *ledger;
  struct wt_policy_pool *pool;
  const struct wt_ledger_contract *contract;
  struct wt_mpi_remote_endpoint endpoint;
  struct wt_ledger_ticket ledger_ticket;
  struct wt_policy_ticket admission_ticket;
  struct wt_work_id request_id;
  uint8_t request_fingerprint[WT_WIRE_FINGERPRINT_BYTES];
  wt_ledger_commit_callback commit;
  void *commit_context;
  size_t channel_capacity;
  size_t result_limit;
  int64_t deadline_monotonic_ns;
  enum wt_mpi_remote_owner_state state;
  enum wt_mpi_remote_status terminal_status;
  unsigned ledger_ticket_live;
  unsigned admission_ticket_live;
  unsigned remote_terminal_known;
  unsigned published_maybe;
  unsigned send_complete;
  unsigned callback_active;
};

struct wt_mpi_remote_worker {
  struct wt_mpi_channel *channel;
  struct wt_ledger *ledger;
  const struct wt_ledger_contract *contract;
  struct wt_mpi_remote_endpoint endpoint;
  struct wt_ledger_ticket ledger_ticket;
  wt_ledger_run_callback run;
  void *run_context;
  size_t channel_capacity;
  int64_t deadline_monotonic_ns;
  enum wt_mpi_remote_worker_state state;
  enum wt_mpi_remote_status terminal_status;
  unsigned ledger_ticket_live;
  unsigned published_maybe;
  unsigned callback_active;
};

/* Common helpers never infer endpoint authority from frame fields. */
void wt_mpi_remote_private_clear_event(struct wt_mpi_remote_event *event);
int wt_mpi_remote_private_decode_trusted(
    const struct wt_ledger_contract *contract,
    const uint8_t *bytes, size_t length, uint8_t kind, uint32_t phase,
    struct wt_wire_frame *out);
int wt_mpi_remote_private_choice_matches(
    const struct wt_mpi_remote_endpoint *endpoint,
    const struct wt_policy_choice *choice);
enum wt_mpi_remote_status wt_mpi_remote_private_deadline_expired(
    int64_t deadline_monotonic_ns, int *out_expired);

#endif
