#ifndef WT_REMOTE_H
#define WT_REMOTE_H

#include "wt_ledger.h"
#include "wt_policy.h"
#include "wt_stream.h"

/* Generic synchronous local-process transport over a trusted POSIX stream.
 * This is neither an MPI/WRF adapter nor cross-node performance evidence.
 * A caller owns every context, scratch buffer, descriptor and ledger, and
 * serializes each context. No function retries or invokes a local fallback. */
enum wt_remote_status {
  WT_REMOTE_OK = 0, WT_REMOTE_ARGUMENT, WT_REMOTE_POLICY_REFUSED,
  WT_REMOTE_BAD_PACKET, WT_REMOTE_DUPLICATE, WT_REMOTE_IO,
  WT_REMOTE_LEDGER, WT_REMOTE_CALLBACK_FAILED, WT_REMOTE_BUSY,
  WT_REMOTE_COMMITTED_RELEASE_FAILED, WT_REMOTE_ENDPOINT_MISMATCH
};

struct wt_remote_packet {
  const uint8_t *request;
  size_t request_len;
  size_t result_limit;
};

struct wt_remote_owner {
  struct wt_stream *stream;
  struct wt_ledger *ledger;
  const struct wt_ledger_contract *contract;
  uint8_t *receive_scratch;
  size_t receive_capacity;
  uint32_t callback_active;
  uint32_t endpoint_rank_id;
  uint32_t endpoint_node_id;
};

struct wt_remote_worker {
  struct wt_stream *stream;
  struct wt_ledger *ledger;
  const struct wt_ledger_contract *contract;
  uint8_t *receive_scratch;
  size_t receive_capacity;
  uint32_t callback_active;
};

enum wt_remote_status wt_remote_owner_exchange(struct wt_remote_owner *owner,
    const struct wt_policy_config *policy, const struct wt_policy_request *request,
    const struct wt_policy_candidate *candidates, size_t candidate_count,
    struct wt_policy_pool *pool, const struct wt_remote_packet *packet,
    wt_ledger_commit_callback commit, void *commit_context,
    struct wt_policy_choice *choice);

enum wt_remote_status wt_remote_worker_serve_once(struct wt_remote_worker *worker,
    wt_ledger_run_callback run, void *run_context);

#endif
