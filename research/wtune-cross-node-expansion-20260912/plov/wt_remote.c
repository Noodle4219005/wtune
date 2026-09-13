#include "wt_remote.h"

#include <limits.h>
#include <string.h>

static int trusted_id(const struct wt_ledger_contract *contract,
    const struct wt_wire_frame *frame, uint32_t phase) {
  return memcmp(frame->id.run_uuid, contract->run_uuid, sizeof(contract->run_uuid)) == 0 &&
      frame->id.owner_rank == contract->owner_rank &&
      frame->id.owner_generation == contract->owner_generation &&
      frame->id.phase == phase;
}

static int decode_frame(const struct wt_ledger_contract *contract,
    const uint8_t *bytes, size_t length, uint8_t kind, uint32_t phase,
    struct wt_wire_frame *out) {
  return contract != NULL && wt_wire_decode(bytes, length, &contract->limits,
      contract->schema_fingerprint, &contract->abi, out) == WT_WIRE_OK &&
      out->kind == kind && trusted_id(contract, out, phase);
}

static void fail_owner(struct wt_remote_owner *owner,
    const struct wt_ledger_ticket *ticket, int poison) {
  (void)wt_ledger_fail(owner->ledger, ticket);
  if (poison) owner->stream->poisoned = 1;
}

static enum wt_remote_status release_or(struct wt_policy_pool *pool,
    const struct wt_policy_ticket *ticket, enum wt_remote_status status) {
  return wt_policy_pool_release(pool, ticket) == WT_POLICY_OK ? status : WT_REMOTE_LEDGER;
}

enum wt_remote_status wt_remote_owner_exchange(struct wt_remote_owner *owner,
    const struct wt_policy_config *policy, const struct wt_policy_request *request,
    const struct wt_policy_candidate *candidates, size_t candidate_count,
    struct wt_policy_pool *pool, const struct wt_remote_packet *packet,
    wt_ledger_commit_callback commit, void *commit_context,
    struct wt_policy_choice *choice) {
  struct wt_wire_frame request_frame;
  struct wt_wire_frame result_frame;
  struct wt_policy_ticket reservation;
  struct wt_ledger_ticket ledger_ticket;
  size_t result_len;
  int ledger_status;
  if (owner == NULL || policy == NULL || request == NULL || pool == NULL ||
      packet == NULL || choice == NULL || owner->stream == NULL || owner->ledger == NULL ||
      owner->contract == NULL || owner->receive_scratch == NULL || commit == NULL ||
      packet->request == NULL || packet->request_len == 0 || packet->result_limit == 0 ||
      owner->callback_active) return owner != NULL && owner->callback_active ?
          WT_REMOTE_BUSY : WT_REMOTE_ARGUMENT;
  if (!decode_frame(owner->contract, packet->request, packet->request_len,
      WT_WIRE_REQUEST, request->phase_id, &request_frame)) return WT_REMOTE_BAD_PACKET;
  if (request->origin.rank_id != request_frame.id.owner_rank ||
      memcmp(policy->schema_fingerprint, owner->contract->schema_fingerprint,
      sizeof(policy->schema_fingerprint)) != 0 ||
      packet->request_len > SIZE_MAX - packet->result_limit ||
      packet->request_len + packet->result_limit > request->worstcase_bytes ||
      owner->receive_capacity < packet->result_limit) return WT_REMOTE_BAD_PACKET;
  if (wt_policy_choose(policy, request, candidates, candidate_count, choice) != WT_POLICY_OK)
    return WT_REMOTE_POLICY_REFUSED;
  if (choice->peer_rank_id != owner->endpoint_rank_id ||
      choice->peer_node_id != owner->endpoint_node_id) return WT_REMOTE_ENDPOINT_MISMATCH;
  if (wt_policy_pool_reserve(pool, request->required_worker_slots,
      request->worstcase_bytes, &reservation) != WT_POLICY_OK) return WT_REMOTE_POLICY_REFUSED;
  ledger_status = wt_ledger_owner_register_request(owner->ledger, packet->request,
      packet->request_len, packet->result_limit, &ledger_ticket);
  if (ledger_status != WT_LEDGER_OK) {
    (void)wt_policy_pool_release(pool, &reservation);
    return ledger_status == WT_LEDGER_DUPLICATE ? WT_REMOTE_DUPLICATE : WT_REMOTE_LEDGER;
  }
  if (wt_stream_send(owner->stream, packet->request, packet->request_len) != WT_STREAM_OK) {
    fail_owner(owner, &ledger_ticket, 1);
    return release_or(pool, &reservation, WT_REMOTE_IO);
  }
  if (wt_stream_receive(owner->stream, owner->receive_scratch, owner->receive_capacity,
      &result_len) != WT_STREAM_OK) {
    fail_owner(owner, &ledger_ticket, 1);
    return release_or(pool, &reservation, WT_REMOTE_IO);
  }
  if (result_len > packet->result_limit || !decode_frame(owner->contract,
      owner->receive_scratch, result_len, WT_WIRE_RESULT, request_frame.id.phase,
      &result_frame) || !wt_wire_id_equal(&request_frame.id, &result_frame.id) ||
      memcmp(request_frame.request_fingerprint, result_frame.request_fingerprint,
      sizeof(request_frame.request_fingerprint)) != 0) {
    fail_owner(owner, &ledger_ticket, 1);
    return release_or(pool, &reservation, WT_REMOTE_BAD_PACKET);
  }
  owner->callback_active = 1;
  ledger_status = wt_ledger_owner_apply_result(owner->ledger, &ledger_ticket,
      owner->receive_scratch, result_len, commit, commit_context);
  owner->callback_active = 0;
  if (ledger_status != WT_LEDGER_OK) {
    fail_owner(owner, &ledger_ticket, 1);
    return release_or(pool, &reservation, ledger_status == WT_LEDGER_DUPLICATE ?
        WT_REMOTE_DUPLICATE : WT_REMOTE_CALLBACK_FAILED);
  }
  return wt_policy_pool_release(pool, &reservation) == WT_POLICY_OK ? WT_REMOTE_OK :
      WT_REMOTE_COMMITTED_RELEASE_FAILED;
}

enum wt_remote_status wt_remote_worker_serve_once(struct wt_remote_worker *worker,
    wt_ledger_run_callback run, void *run_context) {
  struct wt_wire_frame request_frame;
  struct wt_wire_frame result_frame;
  struct wt_ledger_ticket ticket;
  const uint8_t *result;
  size_t request_len;
  size_t result_len;
  int ledger_status;
  if (worker == NULL || worker->stream == NULL || worker->ledger == NULL ||
      worker->contract == NULL || worker->receive_scratch == NULL || run == NULL)
    return WT_REMOTE_ARGUMENT;
  if (worker->callback_active) return WT_REMOTE_BUSY;
  if (wt_stream_receive(worker->stream, worker->receive_scratch, worker->receive_capacity,
      &request_len) != WT_STREAM_OK) return WT_REMOTE_IO;
  /* phase is opaque at this layer; identity ownership is still mandatory. */
  if (wt_wire_decode(worker->receive_scratch, request_len, &worker->contract->limits,
      worker->contract->schema_fingerprint, &worker->contract->abi, &request_frame) != WT_WIRE_OK ||
      request_frame.kind != WT_WIRE_REQUEST || !trusted_id(worker->contract,
      &request_frame, request_frame.id.phase)) {
    worker->stream->poisoned = 1;
    return WT_REMOTE_BAD_PACKET;
  }
  ledger_status = wt_ledger_worker_accept(worker->ledger, worker->receive_scratch,
      request_len, worker->contract->limits.max_frame, &ticket);
  if (ledger_status != WT_LEDGER_OK && ledger_status != WT_LEDGER_DUPLICATE) {
    worker->stream->poisoned = 1;
    return WT_REMOTE_LEDGER;
  }
  worker->callback_active = 1;
  ledger_status = wt_ledger_worker_run(worker->ledger, &ticket, run, run_context);
  worker->callback_active = 0;
  if (ledger_status != WT_LEDGER_OK && ledger_status != WT_LEDGER_DUPLICATE) {
    worker->stream->poisoned = 1;
    return WT_REMOTE_CALLBACK_FAILED;
  }
  if (wt_ledger_result_bytes(worker->ledger, &ticket, &result, &result_len) != WT_LEDGER_OK ||
      !decode_frame(worker->contract, result, result_len, WT_WIRE_RESULT,
      request_frame.id.phase, &result_frame) ||
      !wt_wire_id_equal(&request_frame.id, &result_frame.id)) {
    worker->stream->poisoned = 1;
    return WT_REMOTE_LEDGER;
  }
  return wt_stream_send(worker->stream, result, result_len) == WT_STREAM_OK ? WT_REMOTE_OK :
      WT_REMOTE_IO;
}
