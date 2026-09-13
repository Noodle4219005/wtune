#include "wt_mpi_remote_internal.h"

#include <stdlib.h>

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

static enum wt_mpi_remote_status poison_worker(
    struct wt_mpi_remote_worker *worker, enum wt_mpi_remote_status status) {
  worker->state = WT_MPI_REMOTE_WORKER_POISONED;
  worker->terminal_status = status;
  return status;
}

enum wt_mpi_remote_status wt_mpi_remote_worker_create_take(
    struct wt_mpi_channel **channel,
    const struct wt_mpi_remote_endpoint *endpoint,
    struct wt_ledger *ledger, const struct wt_ledger_contract *contract,
    wt_ledger_run_callback run, void *run_context,
    struct wt_mpi_remote_worker **out) {
  struct wt_mpi_remote_worker *worker;
  enum wt_mpi_channel_status channel_status;
  size_t capacity;
  int peer;
  if (out != NULL) *out = NULL;
  if (channel == NULL || *channel == NULL || endpoint == NULL || ledger == NULL ||
      contract == NULL || run == NULL || out == NULL)
    return WT_MPI_REMOTE_ARGUMENT;
  channel_status = wt_mpi_channel_get_peer(*channel, &peer);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  channel_status = wt_mpi_channel_get_capacity(*channel, &capacity);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  if (peer != endpoint->mpi_peer || endpoint->rank_id != contract->owner_rank)
    return WT_MPI_REMOTE_ENDPOINT_MISMATCH;
  if (contract->limits.max_frame == 0 || contract->limits.max_frame > capacity)
    return WT_MPI_REMOTE_BAD_PACKET;
  worker = calloc(1, sizeof(*worker));
  if (worker == NULL) return WT_MPI_REMOTE_ALLOCATION;
  worker->channel = *channel;
  worker->ledger = ledger;
  worker->contract = contract;
  worker->endpoint = *endpoint;
  worker->run = run;
  worker->run_context = run_context;
  worker->channel_capacity = capacity;
  worker->state = WT_MPI_REMOTE_WORKER_IDLE;
  worker->terminal_status = WT_MPI_REMOTE_OK;
  *channel = NULL;
  *out = worker;
  return WT_MPI_REMOTE_OK;
}

enum wt_mpi_remote_status wt_mpi_remote_worker_arm(
    struct wt_mpi_remote_worker *worker, int64_t deadline_monotonic_ns) {
  enum wt_mpi_channel_status status;
  size_t capacity;
  if (worker == NULL || deadline_monotonic_ns < 0)
    return WT_MPI_REMOTE_ARGUMENT;
  status = wt_mpi_channel_get_capacity(worker->channel, &capacity);
  if (status != WT_MPI_CHANNEL_OK) return from_channel(status);
  if (worker->callback_active) return WT_MPI_REMOTE_BUSY;
  if (worker->state == WT_MPI_REMOTE_WORKER_POISONED ||
      worker->state == WT_MPI_REMOTE_WORKER_FAILED_DRAINING)
    return WT_MPI_REMOTE_POISONED;
  if (worker->state != WT_MPI_REMOTE_WORKER_IDLE) return WT_MPI_REMOTE_BUSY;
  status = wt_mpi_channel_begin_recv(worker->channel);
  if (status != WT_MPI_CHANNEL_OK) return from_channel(status);
  worker->deadline_monotonic_ns = deadline_monotonic_ns;
  worker->published_maybe = 0;
  worker->state = WT_MPI_REMOTE_WORKER_RECEIVING;
  return WT_MPI_REMOTE_OK;
}

static enum wt_mpi_remote_status worker_transport_failure(
    struct wt_mpi_remote_worker *worker, enum wt_mpi_channel_status status,
    const struct wt_mpi_channel_event *channel_event,
    struct wt_mpi_remote_event *event) {
  if (worker->published_maybe)
    event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
  if ((channel_event->kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) &&
      wt_mpi_channel_release_receive(worker->channel) != WT_MPI_CHANNEL_OK)
    return poison_worker(worker, WT_MPI_REMOTE_TRANSPORT);
  if (status == WT_MPI_CHANNEL_TIMEOUT ||
      (channel_event->kinds & WT_MPI_CHANNEL_EVENT_UNKNOWN_DELIVERY)) {
    event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
    return poison_worker(worker, WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
  }
  return poison_worker(worker, from_channel(status));
}

enum wt_mpi_remote_status wt_mpi_remote_worker_progress(
    struct wt_mpi_remote_worker *worker, struct wt_mpi_remote_event *event) {
  struct wt_mpi_channel_event channel_event;
  struct wt_wire_frame request;
  enum wt_mpi_channel_status channel_status;
  const uint8_t *result_bytes;
  size_t result_len;
  size_t capacity;
  int expired;
  int ledger_status;
  enum wt_mpi_remote_status deadline_status;
  if (worker == NULL || event == NULL) return WT_MPI_REMOTE_ARGUMENT;
  wt_mpi_remote_private_clear_event(event);
  channel_status = wt_mpi_channel_get_capacity(worker->channel, &capacity);
  if (channel_status != WT_MPI_CHANNEL_OK) return from_channel(channel_status);
  if (worker->callback_active) return WT_MPI_REMOTE_BUSY;
  if (worker->state == WT_MPI_REMOTE_WORKER_IDLE) return WT_MPI_REMOTE_OK;
  if (worker->state == WT_MPI_REMOTE_WORKER_CHANNEL_CLOSED)
    return WT_MPI_REMOTE_ARGUMENT;
  channel_status = wt_mpi_channel_poll(worker->channel,
      worker->deadline_monotonic_ns, &channel_event);
  event->channel_status = channel_status;
  if (channel_event.kinds & WT_MPI_CHANNEL_EVENT_SEND_COMPLETE) {
    event->kinds |= WT_MPI_REMOTE_EVENT_SEND_COMPLETE;
    if (worker->state == WT_MPI_REMOTE_WORKER_SENDING)
      worker->state = WT_MPI_REMOTE_WORKER_IDLE;
  }
  if (channel_status != WT_MPI_CHANNEL_OK)
    return worker_transport_failure(worker, channel_status, &channel_event, event);
  if (worker->state == WT_MPI_REMOTE_WORKER_POISONED ||
      worker->state == WT_MPI_REMOTE_WORKER_FAILED_DRAINING) {
    if ((channel_event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) &&
        wt_mpi_channel_release_receive(worker->channel) != WT_MPI_CHANNEL_OK)
      return poison_worker(worker, WT_MPI_REMOTE_TRANSPORT);
    return worker->terminal_status;
  }
  if ((channel_event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) == 0)
    return WT_MPI_REMOTE_OK;
  deadline_status = wt_mpi_remote_private_deadline_expired(
      worker->deadline_monotonic_ns, &expired);
  if (deadline_status != WT_MPI_REMOTE_OK || expired) {
    if (wt_mpi_channel_release_receive(worker->channel) != WT_MPI_CHANNEL_OK)
      return poison_worker(worker, WT_MPI_REMOTE_TRANSPORT);
    if (expired) event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
    return poison_worker(worker, expired ?
        WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY : deadline_status);
  }
  if (worker->state != WT_MPI_REMOTE_WORKER_RECEIVING ||
      wt_wire_decode(channel_event.received_bytes, channel_event.received_length,
      &worker->contract->limits, worker->contract->schema_fingerprint,
      &worker->contract->abi, &request) != WT_WIRE_OK ||
      !wt_mpi_remote_private_decode_trusted(worker->contract,
      channel_event.received_bytes, channel_event.received_length,
      WT_WIRE_REQUEST, request.id.phase, &request)) {
    if (wt_mpi_channel_release_receive(worker->channel) != WT_MPI_CHANNEL_OK)
      return poison_worker(worker, WT_MPI_REMOTE_TRANSPORT);
    return poison_worker(worker, WT_MPI_REMOTE_BAD_PACKET);
  }
  ledger_status = wt_ledger_worker_accept(worker->ledger,
      channel_event.received_bytes, channel_event.received_length,
      worker->contract->limits.max_frame, &worker->ledger_ticket);
  if (wt_mpi_channel_release_receive(worker->channel) != WT_MPI_CHANNEL_OK)
    return poison_worker(worker, WT_MPI_REMOTE_TRANSPORT);
  if (ledger_status != WT_LEDGER_OK && ledger_status != WT_LEDGER_DUPLICATE) {
    event->ledger_status = (enum wt_ledger_status)ledger_status;
    return poison_worker(worker, WT_MPI_REMOTE_LEDGER);
  }
  worker->ledger_ticket_live = 1;
  worker->callback_active = 1;
  ledger_status = wt_ledger_worker_run(worker->ledger, &worker->ledger_ticket,
      worker->run, worker->run_context);
  worker->callback_active = 0;
  event->ledger_status = (enum wt_ledger_status)ledger_status;
  if (ledger_status == WT_LEDGER_OK)
    event->kinds |= WT_MPI_REMOTE_EVENT_WORKER_EXECUTED;
  else if (ledger_status == WT_LEDGER_DUPLICATE)
    event->kinds |= WT_MPI_REMOTE_EVENT_WORKER_CACHED_RESULT;
  else {
    worker->ledger_ticket_live = 0;
    return poison_worker(worker, ledger_status == WT_LEDGER_ECALLBACK ?
        WT_MPI_REMOTE_CALLBACK_FAILED : WT_MPI_REMOTE_LEDGER);
  }
  deadline_status = wt_mpi_remote_private_deadline_expired(
      worker->deadline_monotonic_ns, &expired);
  if (deadline_status != WT_MPI_REMOTE_OK || expired) {
    worker->ledger_ticket_live = 0;
    if (expired) event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
    return poison_worker(worker, expired ?
        WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY : deadline_status);
  }
  ledger_status = wt_ledger_result_bytes(worker->ledger, &worker->ledger_ticket,
      &result_bytes, &result_len);
  if (ledger_status != WT_LEDGER_OK || result_len == 0 ||
      result_len > worker->channel_capacity) {
    event->ledger_status = (enum wt_ledger_status)ledger_status;
    worker->ledger_ticket_live = 0;
    return poison_worker(worker, WT_MPI_REMOTE_LEDGER);
  }
  worker->published_maybe = 1;
  channel_status = wt_mpi_channel_begin_send(worker->channel, result_bytes,
      result_len);
  worker->ledger_ticket_live = 0;
  if (channel_status != WT_MPI_CHANNEL_OK) {
    event->kinds |= WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY;
    return poison_worker(worker, WT_MPI_REMOTE_UNKNOWN_DELIVERY);
  }
  worker->state = WT_MPI_REMOTE_WORKER_SENDING;
  return WT_MPI_REMOTE_OK;
}

enum wt_mpi_remote_status wt_mpi_remote_worker_shutdown_ready(
    const struct wt_mpi_remote_worker *worker) {
  enum wt_mpi_channel_status status;
  if (worker == NULL) return WT_MPI_REMOTE_ARGUMENT;
  if (worker->callback_active) return WT_MPI_REMOTE_BUSY;
  status = wt_mpi_channel_destroy_ready(worker->channel);
  if (status != WT_MPI_CHANNEL_OK) return from_channel(status);
  return worker->state == WT_MPI_REMOTE_WORKER_IDLE ||
      worker->state == WT_MPI_REMOTE_WORKER_POISONED ||
      worker->state == WT_MPI_REMOTE_WORKER_FAILED_DRAINING ?
      WT_MPI_REMOTE_OK : WT_MPI_REMOTE_BUSY;
}

enum wt_mpi_remote_status wt_mpi_remote_worker_destroy_collective(
    struct wt_mpi_remote_worker **worker_pointer) {
  struct wt_mpi_remote_worker *worker;
  enum wt_mpi_channel_status status;
  if (worker_pointer == NULL || *worker_pointer == NULL)
    return WT_MPI_REMOTE_ARGUMENT;
  worker = *worker_pointer;
  if (worker->callback_active) return WT_MPI_REMOTE_BUSY;
  status = wt_mpi_channel_destroy_collective(&worker->channel);
  if (status != WT_MPI_CHANNEL_OK) return from_channel(status);
  worker->state = WT_MPI_REMOTE_WORKER_CHANNEL_CLOSED;
  free(worker);
  *worker_pointer = NULL;
  return WT_MPI_REMOTE_OK;
}
