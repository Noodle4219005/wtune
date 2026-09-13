#include "wt_mpi_remote_internal.h"

#include <stdlib.h>
#include <string.h>

static enum wt_mpi_remote_status from_channel(
    enum wt_mpi_channel_status status) {
  if (status == WT_MPI_CHANNEL_OK) return WT_MPI_REMOTE_OK;
  if (status == WT_MPI_CHANNEL_ARGUMENT) return WT_MPI_REMOTE_ARGUMENT;
  if (status == WT_MPI_CHANNEL_BUSY) return WT_MPI_REMOTE_BUSY;
  if (status == WT_MPI_CHANNEL_TIMEOUT)
    return WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY;
  if (status == WT_MPI_CHANNEL_POISONED) return WT_MPI_REMOTE_POISONED;
  return WT_MPI_REMOTE_TRANSPORT;
}

static enum wt_mpi_remote_status release_admission(
    struct wt_mpi_remote_owner *owner, struct wt_mpi_remote_event *event) {
  enum wt_policy_status status;
  if (!owner->admission_ticket_live) return WT_MPI_REMOTE_OK;
  status = wt_policy_pool_release(owner->pool, &owner->admission_ticket);
  if (event != NULL) event->policy_status = status;
  if (status != WT_POLICY_OK) return WT_MPI_REMOTE_ADMISSION_RELEASE_FAILED;
  owner->admission_ticket_live = 0;
  return WT_MPI_REMOTE_OK;
}

static void fail_live_ledger(struct wt_mpi_remote_owner *owner,
    struct wt_mpi_remote_event *event) {
  int status;
  if (!owner->ledger_ticket_live) return;
  status = wt_ledger_fail(owner->ledger, &owner->ledger_ticket);
  if (event != NULL) event->ledger_status = status;
  owner->ledger_ticket_live = 0;
}

struct commit_trampoline_context {
  struct wt_mpi_remote_owner *owner;
  enum wt_mpi_remote_status clock_status;
  int expired;
};

static int commit_with_deadline(void *opaque,
    const struct wt_wire_frame *result) {
  struct commit_trampoline_context *context = opaque;
  context->clock_status = wt_mpi_remote_private_deadline_expired(
      context->owner->deadline_monotonic_ns, &context->expired);
  if (context->clock_status != WT_MPI_REMOTE_OK || context->expired) return 1;
  return context->owner->commit(context->owner->commit_context, result);
}

static enum wt_mpi_remote_status fail_before_publish(
    struct wt_mpi_remote_owner *owner, enum wt_mpi_remote_status status) {
  enum wt_policy_status release_status;
  fail_live_ledger(owner, NULL);
  release_status = wt_policy_pool_release(owner->pool, &owner->admission_ticket);
  if (release_status != WT_POLICY_OK)
    return WT_MPI_REMOTE_ADMISSION_RELEASE_FAILED;
  owner->admission_ticket_live = 0;
  owner->state = WT_MPI_REMOTE_OWNER_IDLE;
  return status;
}

enum wt_mpi_remote_status wt_mpi_remote_owner_create_take(
    struct wt_mpi_channel **channel,
    const struct wt_mpi_remote_endpoint *endpoint,
    struct wt_ledger *ledger, struct wt_policy_pool *pool,
    const struct wt_ledger_contract *contract,
    struct wt_mpi_remote_owner **out) {
  struct wt_mpi_remote_owner *owner;
  enum wt_mpi_channel_status channel_status;
  size_t capacity;
  int peer;
  if (out != NULL) *out = NULL;
  if (channel == NULL || *channel == NULL || endpoint == NULL || ledger == NULL ||
      pool == NULL || contract == NULL || out == NULL)
    return WT_MPI_REMOTE_ARGUMENT;
  channel_status = wt_mpi_channel_get_peer(*channel, &peer);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  channel_status = wt_mpi_channel_get_capacity(*channel, &capacity);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  if (peer != endpoint->mpi_peer) return WT_MPI_REMOTE_ENDPOINT_MISMATCH;
  if (contract->limits.max_frame == 0 || contract->limits.max_frame > capacity)
    return WT_MPI_REMOTE_BAD_PACKET;
  owner = calloc(1, sizeof(*owner));
  if (owner == NULL) return WT_MPI_REMOTE_ALLOCATION;
  owner->channel = *channel;
  owner->ledger = ledger;
  owner->pool = pool;
  owner->contract = contract;
  owner->endpoint = *endpoint;
  owner->channel_capacity = capacity;
  owner->state = WT_MPI_REMOTE_OWNER_IDLE;
  owner->terminal_status = WT_MPI_REMOTE_OK;
  *channel = NULL;
  *out = owner;
  return WT_MPI_REMOTE_OK;
}

enum wt_mpi_remote_status wt_mpi_remote_owner_begin(
    struct wt_mpi_remote_owner *owner,
    const struct wt_policy_config *policy,
    const struct wt_policy_request *request,
    const struct wt_policy_candidate *candidates, size_t candidate_count,
    const struct wt_mpi_remote_packet *packet,
    int64_t deadline_monotonic_ns,
    wt_ledger_commit_callback commit, void *commit_context,
    struct wt_policy_choice *choice) {
  struct wt_wire_frame frame;
  enum wt_policy_status policy_status;
  enum wt_mpi_channel_status channel_status;
  size_t capacity;
  int ledger_status;
  if (owner == NULL || policy == NULL || request == NULL || packet == NULL ||
      choice == NULL || commit == NULL || deadline_monotonic_ns < 0)
    return WT_MPI_REMOTE_ARGUMENT;
  channel_status = wt_mpi_channel_get_capacity(owner->channel, &capacity);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  if (owner->state == WT_MPI_REMOTE_OWNER_POISONED ||
      owner->state == WT_MPI_REMOTE_OWNER_FAILED_DRAINING)
    return WT_MPI_REMOTE_POISONED;
  if (owner->state != WT_MPI_REMOTE_OWNER_IDLE) return WT_MPI_REMOTE_BUSY;
  if (packet->request == NULL || packet->request_len == 0 ||
      packet->request_len > capacity || packet->result_limit > capacity ||
      packet->request_len > owner->contract->limits.max_frame ||
      packet->result_limit > owner->contract->limits.max_frame ||
      packet->result_limit < WT_WIRE_HEADER_BYTES ||
      packet->request_len > UINT64_MAX - packet->result_limit ||
      request->worstcase_bytes < packet->request_len + packet->result_limit)
    return WT_MPI_REMOTE_BAD_PACKET;
  if (!wt_mpi_remote_private_decode_trusted(owner->contract, packet->request,
      packet->request_len, WT_WIRE_REQUEST, request->phase_id, &frame) ||
      frame.id.owner_rank != request->origin.rank_id)
    return WT_MPI_REMOTE_BAD_PACKET;
  policy_status = wt_policy_choose(policy, request, candidates, candidate_count,
      choice);
  if (policy_status != WT_POLICY_OK) return WT_MPI_REMOTE_POLICY_REFUSED;
  if (!wt_mpi_remote_private_choice_matches(&owner->endpoint, choice))
    return WT_MPI_REMOTE_ENDPOINT_MISMATCH;
  policy_status = wt_policy_pool_reserve(owner->pool,
      request->required_worker_slots, request->worstcase_bytes,
      &owner->admission_ticket);
  if (policy_status != WT_POLICY_OK) return WT_MPI_REMOTE_POLICY_REFUSED;
  owner->admission_ticket_live = 1;
  ledger_status = wt_ledger_owner_register_request(owner->ledger, packet->request,
      packet->request_len, packet->result_limit, &owner->ledger_ticket);
  if (ledger_status != WT_LEDGER_OK) {
    enum wt_mpi_remote_status status = ledger_status == WT_LEDGER_DUPLICATE ?
        WT_MPI_REMOTE_DUPLICATE : WT_MPI_REMOTE_LEDGER;
    enum wt_policy_status released = wt_policy_pool_release(owner->pool,
        &owner->admission_ticket);
    if (released != WT_POLICY_OK)
      return WT_MPI_REMOTE_ADMISSION_RELEASE_FAILED;
    owner->admission_ticket_live = 0;
    return status;
  }
  owner->ledger_ticket_live = 1;
  owner->request_id = frame.id;
  memcpy(owner->request_fingerprint, frame.request_fingerprint,
      sizeof(owner->request_fingerprint));
  owner->commit = commit;
  owner->commit_context = commit_context;
  owner->result_limit = packet->result_limit;
  owner->deadline_monotonic_ns = deadline_monotonic_ns;
  owner->remote_terminal_known = 0;
  owner->published_maybe = 0;
  owner->send_complete = 0;
  owner->state = WT_MPI_REMOTE_OWNER_ACTIVE;
  channel_status = wt_mpi_channel_begin_recv(owner->channel);
  if (channel_status != WT_MPI_CHANNEL_OK)
    return fail_before_publish(owner, from_channel(channel_status));
  owner->published_maybe = 1;
  channel_status = wt_mpi_channel_begin_send(owner->channel, packet->request,
      packet->request_len);
  if (channel_status != WT_MPI_CHANNEL_OK) {
    fail_live_ledger(owner, NULL);
    owner->state = WT_MPI_REMOTE_OWNER_FAILED_DRAINING;
    owner->terminal_status = WT_MPI_REMOTE_UNKNOWN_DELIVERY;
    return owner->terminal_status;
  }
  return WT_MPI_REMOTE_OK;
}

static enum wt_mpi_remote_status owner_transport_failure(
    struct wt_mpi_remote_owner *owner, enum wt_mpi_channel_status status,
    struct wt_mpi_channel_event *channel_event,
    struct wt_mpi_remote_event *event) {
  if (owner->published_maybe && !owner->remote_terminal_known)
    event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
  if ((channel_event->kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) &&
      wt_mpi_channel_release_receive(owner->channel) != WT_MPI_CHANNEL_OK) {
    owner->state = WT_MPI_REMOTE_OWNER_POISONED;
    owner->terminal_status = WT_MPI_REMOTE_TRANSPORT;
    return owner->terminal_status;
  }
  fail_live_ledger(owner, event);
  if (status == WT_MPI_CHANNEL_TIMEOUT ||
      (channel_event->kinds & WT_MPI_CHANNEL_EVENT_UNKNOWN_DELIVERY)) {
    event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
    owner->terminal_status = WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY;
  } else {
    owner->terminal_status = from_channel(status);
  }
  owner->state = WT_MPI_REMOTE_OWNER_POISONED;
  return owner->terminal_status;
}

enum wt_mpi_remote_status wt_mpi_remote_owner_progress(
    struct wt_mpi_remote_owner *owner, struct wt_mpi_remote_event *event) {
  struct wt_mpi_channel_event channel_event;
  struct wt_wire_frame result;
  struct commit_trampoline_context trampoline;
  enum wt_mpi_channel_status channel_status;
  enum wt_mpi_remote_status release_status;
  int ledger_status;
  int expired;
  size_t capacity;
  if (owner == NULL || event == NULL) return WT_MPI_REMOTE_ARGUMENT;
  wt_mpi_remote_private_clear_event(event);
  channel_status = wt_mpi_channel_get_capacity(owner->channel, &capacity);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  if (owner->callback_active) return WT_MPI_REMOTE_BUSY;
  if (owner->state == WT_MPI_REMOTE_OWNER_IDLE) return WT_MPI_REMOTE_OK;
  if (owner->state == WT_MPI_REMOTE_OWNER_CHANNEL_CLOSED)
    return WT_MPI_REMOTE_ARGUMENT;
  channel_status = wt_mpi_channel_poll(owner->channel,
      owner->deadline_monotonic_ns, &channel_event);
  event->channel_status = channel_status;
  if (channel_event.kinds & WT_MPI_CHANNEL_EVENT_SEND_COMPLETE) {
    owner->send_complete = 1;
    event->kinds |= WT_MPI_REMOTE_EVENT_SEND_COMPLETE;
  }
  if (channel_status != WT_MPI_CHANNEL_OK)
    return owner_transport_failure(owner, channel_status, &channel_event, event);
  if (owner->state == WT_MPI_REMOTE_OWNER_POISONED ||
      owner->state == WT_MPI_REMOTE_OWNER_FAILED_DRAINING) {
    if ((channel_event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) &&
        wt_mpi_channel_release_receive(owner->channel) != WT_MPI_CHANNEL_OK)
      return WT_MPI_REMOTE_TRANSPORT;
    return owner->terminal_status;
  }
  if (owner->state == WT_MPI_REMOTE_OWNER_ACTIVE) {
    enum wt_mpi_remote_status deadline_status =
        wt_mpi_remote_private_deadline_expired(
            owner->deadline_monotonic_ns, &expired);
    if (deadline_status != WT_MPI_REMOTE_OK) {
      if ((channel_event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) &&
          wt_mpi_channel_release_receive(owner->channel) != WT_MPI_CHANNEL_OK)
        deadline_status = WT_MPI_REMOTE_TRANSPORT;
      fail_live_ledger(owner, event);
      if (owner->published_maybe)
        event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
      owner->state = WT_MPI_REMOTE_OWNER_POISONED;
      owner->terminal_status = deadline_status;
      return deadline_status;
    }
    if (expired) {
      if ((channel_event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) &&
          wt_mpi_channel_release_receive(owner->channel) != WT_MPI_CHANNEL_OK) {
        event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
        owner->state = WT_MPI_REMOTE_OWNER_POISONED;
        owner->terminal_status = WT_MPI_REMOTE_TRANSPORT;
        return owner->terminal_status;
      }
      fail_live_ledger(owner, event);
      event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
      owner->state = WT_MPI_REMOTE_OWNER_POISONED;
      owner->terminal_status = WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY;
      return owner->terminal_status;
    }
  }
  if ((channel_event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) != 0) {
    int trusted = channel_event.received_length <= owner->result_limit &&
        wt_mpi_remote_private_decode_trusted(owner->contract,
        channel_event.received_bytes, channel_event.received_length,
        WT_WIRE_RESULT, owner->request_id.phase, &result) &&
        wt_wire_id_equal(&owner->request_id, &result.id) &&
        memcmp(owner->request_fingerprint, result.request_fingerprint,
        sizeof(owner->request_fingerprint)) == 0;
    if (!trusted) {
      if (owner->published_maybe)
        event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
      if (wt_mpi_channel_release_receive(owner->channel) != WT_MPI_CHANNEL_OK) {
        owner->state = WT_MPI_REMOTE_OWNER_POISONED;
        owner->terminal_status = WT_MPI_REMOTE_TRANSPORT;
        return owner->terminal_status;
      }
      fail_live_ledger(owner, event);
      owner->state = WT_MPI_REMOTE_OWNER_POISONED;
      owner->terminal_status = WT_MPI_REMOTE_BAD_PACKET;
      return owner->terminal_status;
    }
    owner->remote_terminal_known = 1;
    owner->published_maybe = 0;
    trampoline = (struct commit_trampoline_context){
        owner, WT_MPI_REMOTE_OK, 0};
    owner->callback_active = 1;
    ledger_status = wt_ledger_owner_apply_result(owner->ledger,
        &owner->ledger_ticket, channel_event.received_bytes,
        channel_event.received_length, commit_with_deadline, &trampoline);
    owner->callback_active = 0;
    if (wt_mpi_channel_release_receive(owner->channel) != WT_MPI_CHANNEL_OK) {
      owner->state = WT_MPI_REMOTE_OWNER_POISONED;
      owner->terminal_status = WT_MPI_REMOTE_TRANSPORT;
      return owner->terminal_status;
    }
    event->ledger_status = (enum wt_ledger_status)ledger_status;
    owner->ledger_ticket_live = 0;
    release_status = release_admission(owner, event);
    if (release_status != WT_MPI_REMOTE_OK) {
      owner->state = WT_MPI_REMOTE_OWNER_POISONED;
      owner->terminal_status = release_status;
      return release_status;
    }
    if (ledger_status != WT_LEDGER_OK) {
      owner->state = WT_MPI_REMOTE_OWNER_POISONED;
      if (trampoline.clock_status != WT_MPI_REMOTE_OK)
        owner->terminal_status = trampoline.clock_status;
      else if (trampoline.expired)
        owner->terminal_status = WT_MPI_REMOTE_DEADLINE_EXPIRED_RESULT_KNOWN;
      else
        owner->terminal_status = ledger_status == WT_LEDGER_ECALLBACK ?
            WT_MPI_REMOTE_CALLBACK_FAILED : WT_MPI_REMOTE_LEDGER;
      return owner->terminal_status;
    }
    event->kinds |= WT_MPI_REMOTE_EVENT_OWNER_COMMITTED;
    owner->state = owner->send_complete ? WT_MPI_REMOTE_OWNER_IDLE :
        WT_MPI_REMOTE_OWNER_COMMITTED_DRAINING;
  }
  if (owner->state == WT_MPI_REMOTE_OWNER_COMMITTED_DRAINING &&
      owner->send_complete)
    owner->state = WT_MPI_REMOTE_OWNER_IDLE;
  return WT_MPI_REMOTE_OK;
}

enum wt_mpi_remote_status wt_mpi_remote_owner_abandon(
    struct wt_mpi_remote_owner *owner, struct wt_mpi_remote_event *event) {
  struct wt_mpi_channel_event channel_event;
  enum wt_mpi_channel_status channel_status;
  size_t capacity;
  if (owner == NULL || event == NULL) return WT_MPI_REMOTE_ARGUMENT;
  wt_mpi_remote_private_clear_event(event);
  channel_status = wt_mpi_channel_get_capacity(owner->channel, &capacity);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  if (capacity != owner->channel_capacity) return WT_MPI_REMOTE_TRANSPORT;
  if (owner->callback_active) return WT_MPI_REMOTE_BUSY;
  if (owner->state != WT_MPI_REMOTE_OWNER_ACTIVE)
    return owner->state == WT_MPI_REMOTE_OWNER_POISONED ?
        WT_MPI_REMOTE_POISONED : WT_MPI_REMOTE_BUSY;
  fail_live_ledger(owner, event);
  channel_status = wt_mpi_channel_poll(owner->channel, 0, &channel_event);
  event->channel_status = channel_status;
  if (channel_event.kinds & WT_MPI_CHANNEL_EVENT_SEND_COMPLETE)
    owner->send_complete = 1;
  if ((channel_event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) &&
      wt_mpi_channel_release_receive(owner->channel) != WT_MPI_CHANNEL_OK) {
    owner->state = WT_MPI_REMOTE_OWNER_POISONED;
    owner->terminal_status = WT_MPI_REMOTE_TRANSPORT;
    return owner->terminal_status;
  }
  event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
  owner->state = WT_MPI_REMOTE_OWNER_POISONED;
  owner->terminal_status = WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY;
  return owner->terminal_status;
}

enum wt_mpi_remote_status wt_mpi_remote_owner_shutdown_ready(
    const struct wt_mpi_remote_owner *owner) {
  enum wt_mpi_channel_status status;
  if (owner == NULL) return WT_MPI_REMOTE_ARGUMENT;
  if (owner->callback_active) return WT_MPI_REMOTE_BUSY;
  status = wt_mpi_channel_destroy_ready(owner->channel);
  if (status != WT_MPI_CHANNEL_OK) return from_channel(status);
  return owner->state == WT_MPI_REMOTE_OWNER_IDLE ||
      owner->state == WT_MPI_REMOTE_OWNER_POISONED ||
      owner->state == WT_MPI_REMOTE_OWNER_FAILED_DRAINING ?
      WT_MPI_REMOTE_OK : WT_MPI_REMOTE_BUSY;
}

enum wt_mpi_remote_status wt_mpi_remote_owner_destroy_collective(
    struct wt_mpi_remote_owner **owner_pointer) {
  struct wt_mpi_remote_owner *owner;
  enum wt_mpi_channel_status channel_status;
  enum wt_mpi_remote_status status;
  if (owner_pointer == NULL || *owner_pointer == NULL)
    return WT_MPI_REMOTE_ARGUMENT;
  owner = *owner_pointer;
  if (owner->callback_active) return WT_MPI_REMOTE_BUSY;
  channel_status = wt_mpi_channel_destroy_collective(&owner->channel);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  owner->state = WT_MPI_REMOTE_OWNER_CHANNEL_CLOSED;
  status = wt_mpi_remote_owner_finish_closed(owner_pointer);
  return status;
}

enum wt_mpi_remote_status wt_mpi_remote_owner_finish_closed(
    struct wt_mpi_remote_owner **owner_pointer) {
  struct wt_mpi_remote_owner *owner;
  enum wt_mpi_remote_status status;
  if (owner_pointer == NULL || *owner_pointer == NULL)
    return WT_MPI_REMOTE_ARGUMENT;
  owner = *owner_pointer;
  if (owner->callback_active) return WT_MPI_REMOTE_BUSY;
  if (owner->state != WT_MPI_REMOTE_OWNER_CHANNEL_CLOSED)
    return WT_MPI_REMOTE_BUSY;
  status = release_admission(owner, NULL);
  if (status != WT_MPI_REMOTE_OK) return status;
  free(owner);
  *owner_pointer = NULL;
  return WT_MPI_REMOTE_OK;
}
