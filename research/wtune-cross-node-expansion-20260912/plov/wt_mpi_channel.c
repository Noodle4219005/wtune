#define _POSIX_C_SOURCE 200809L

#include "wt_mpi_channel.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct wt_mpi_channel {
  MPI_Comm communicator;
  MPI_Request send_request;
  MPI_Request receive_request;
  unsigned char *send_buffer;
  unsigned char *receive_buffer;
  size_t max_frame;
  size_t receive_length;
  const void *creator_thread_token;
  int peer;
  int send_active;
  int receive_armed;
  int receive_request_active;
  int receive_ready;
  int draining_oversize;
  int poisoned;
};

static _Thread_local unsigned char wt_mpi_channel_thread_token;

static enum wt_mpi_channel_status check_thread(
    const struct wt_mpi_channel *channel) {
  if (channel == NULL) return WT_MPI_CHANNEL_ARGUMENT;
  if (channel->creator_thread_token != &wt_mpi_channel_thread_token)
    return WT_MPI_CHANNEL_THREAD_AFFINITY;
  return WT_MPI_CHANNEL_OK;
}

static enum wt_mpi_channel_status poison(struct wt_mpi_channel *channel,
    enum wt_mpi_channel_status status) {
  channel->poisoned = 1;
  if (channel->receive_armed && !channel->receive_request_active)
    channel->receive_armed = 0;
  return status;
}

static int64_t now_monotonic_ns(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0 ||
      now.tv_nsec < 0 || now.tv_nsec >= 1000000000L ||
      (uint64_t)now.tv_sec >
          ((uint64_t)INT64_MAX - (uint64_t)now.tv_nsec) /
              UINT64_C(1000000000))
    return -1;
  return (int64_t)((uint64_t)now.tv_sec * UINT64_C(1000000000) +
      (uint64_t)now.tv_nsec);
}

static void discard_allocations(struct wt_mpi_channel *channel) {
  if (channel == NULL) return;
  free(channel->receive_buffer);
  free(channel->send_buffer);
  free(channel);
}

enum wt_mpi_channel_status wt_mpi_channel_get_peer(
    const struct wt_mpi_channel *channel, int *out_peer) {
  enum wt_mpi_channel_status status = check_thread(channel);
  if (status != WT_MPI_CHANNEL_OK) return status;
  if (out_peer == NULL) return WT_MPI_CHANNEL_ARGUMENT;
  *out_peer = channel->peer;
  return WT_MPI_CHANNEL_OK;
}

enum wt_mpi_channel_status wt_mpi_channel_get_capacity(
    const struct wt_mpi_channel *channel, size_t *out_capacity) {
  enum wt_mpi_channel_status status = check_thread(channel);
  if (status != WT_MPI_CHANNEL_OK) return status;
  if (out_capacity == NULL) return WT_MPI_CHANNEL_ARGUMENT;
  *out_capacity = channel->max_frame;
  return WT_MPI_CHANNEL_OK;
}

enum wt_mpi_channel_status wt_mpi_channel_create(MPI_Comm parent, int peer,
    size_t max_frame, struct wt_mpi_channel **out) {
  struct wt_mpi_channel *channel;
  MPI_Comm communicator = MPI_COMM_NULL;
  int initialized = 0;
  int finalized = 0;
  int provided = MPI_THREAD_SINGLE;
  int is_main = 0;
  int communicator_size = 0;
  int local_validation = WT_MPI_CHANNEL_OK;
  int collective_validation = WT_MPI_CHANNEL_OK;
  int local_allocation_ok;
  int all_allocations_ok = 0;
  int mpi_status;

  if (out != NULL) *out = NULL;
  mpi_status = MPI_Initialized(&initialized);
  if (mpi_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;
  if (!initialized) return WT_MPI_CHANNEL_MPI_UNAVAILABLE;
  mpi_status = MPI_Finalized(&finalized);
  if (mpi_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;
  if (finalized) return WT_MPI_CHANNEL_MPI_UNAVAILABLE;
  mpi_status = MPI_Query_thread(&provided);
  if (mpi_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;
  if (provided < MPI_THREAD_FUNNELED) return WT_MPI_CHANNEL_THREAD_LEVEL;
  mpi_status = MPI_Is_thread_main(&is_main);
  if (mpi_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;
  if (!is_main) return WT_MPI_CHANNEL_THREAD_AFFINITY;
  /* A common valid parent is a collective precondition: there is no safe
   * communicator on which ranks could agree about a rank-local null parent. */
  if (parent == MPI_COMM_NULL) return WT_MPI_CHANNEL_ARGUMENT;

  mpi_status = MPI_Comm_dup(parent, &communicator);
  if (mpi_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;
  mpi_status = MPI_Comm_set_errhandler(communicator, MPI_ERRORS_RETURN);
  if (mpi_status != MPI_SUCCESS) {
    int free_status = MPI_Comm_free(&communicator);
    if (free_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;
    return WT_MPI_CHANNEL_MPI_ERROR;
  }
  mpi_status = MPI_Comm_size(communicator, &communicator_size);
  if (mpi_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;

  if (out == NULL || max_frame == 0)
    local_validation = WT_MPI_CHANNEL_ARGUMENT;
  if (peer < 0 || peer >= communicator_size)
    local_validation = WT_MPI_CHANNEL_PEER;
  if (max_frame > INT_MAX)
    local_validation = WT_MPI_CHANNEL_LIMIT;
  mpi_status = MPI_Allreduce(&local_validation, &collective_validation, 1,
      MPI_INT, MPI_MAX, communicator);
  if (mpi_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;
  if (collective_validation != WT_MPI_CHANNEL_OK) {
    int free_status = MPI_Comm_free(&communicator);
    if (free_status != MPI_SUCCESS) return WT_MPI_CHANNEL_MPI_ERROR;
    return (enum wt_mpi_channel_status)collective_validation;
  }

  channel = calloc(1, sizeof(*channel));
  if (channel != NULL) {
    channel->send_buffer = malloc(max_frame);
    channel->receive_buffer = malloc(max_frame);
  }
  local_allocation_ok = channel != NULL && channel->send_buffer != NULL &&
      channel->receive_buffer != NULL;
  mpi_status = MPI_Allreduce(&local_allocation_ok, &all_allocations_ok, 1,
      MPI_INT, MPI_MIN, communicator);
  if (mpi_status != MPI_SUCCESS || !all_allocations_ok) {
    int free_status;
    discard_allocations(channel);
    free_status = MPI_Comm_free(&communicator);
    if (free_status != MPI_SUCCESS || mpi_status != MPI_SUCCESS)
      return WT_MPI_CHANNEL_MPI_ERROR;
    return WT_MPI_CHANNEL_ALLOCATION;
  }

  channel->communicator = communicator;
  channel->send_request = MPI_REQUEST_NULL;
  channel->receive_request = MPI_REQUEST_NULL;
  channel->max_frame = max_frame;
  channel->peer = peer;
  channel->creator_thread_token = &wt_mpi_channel_thread_token;
  *out = channel;
  return WT_MPI_CHANNEL_OK;
}

enum wt_mpi_channel_status wt_mpi_channel_begin_send(
    struct wt_mpi_channel *channel, const void *bytes, size_t length) {
  enum wt_mpi_channel_status status = check_thread(channel);
  int mpi_status;
  if (status != WT_MPI_CHANNEL_OK) return status;
  if (channel->poisoned) return WT_MPI_CHANNEL_POISONED;
  if (bytes == NULL || length == 0) return WT_MPI_CHANNEL_ARGUMENT;
  if (length > channel->max_frame || length > INT_MAX) return WT_MPI_CHANNEL_LIMIT;
  if (channel->send_active) return WT_MPI_CHANNEL_BUSY;
  memcpy(channel->send_buffer, bytes, length);
  channel->send_request = MPI_REQUEST_NULL;
  channel->send_active = 1;
  mpi_status = MPI_Isend(channel->send_buffer, (int)length, MPI_BYTE,
      channel->peer, 0, channel->communicator, &channel->send_request);
  if (mpi_status != MPI_SUCCESS)
    return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  return WT_MPI_CHANNEL_OK;
}

enum wt_mpi_channel_status wt_mpi_channel_begin_recv(
    struct wt_mpi_channel *channel) {
  enum wt_mpi_channel_status status = check_thread(channel);
  if (status != WT_MPI_CHANNEL_OK) return status;
  if (channel->poisoned) return WT_MPI_CHANNEL_POISONED;
  if (channel->receive_armed || channel->receive_ready)
    return WT_MPI_CHANNEL_BUSY;
  channel->receive_armed = 1;
  return WT_MPI_CHANNEL_OK;
}

static enum wt_mpi_channel_status progress_send(struct wt_mpi_channel *channel,
    struct wt_mpi_channel_event *event) {
  int complete = 0;
  int mpi_status;
  if (!channel->send_active) return WT_MPI_CHANNEL_OK;
  mpi_status = MPI_Test(&channel->send_request, &complete, MPI_STATUS_IGNORE);
  if (mpi_status != MPI_SUCCESS) {
    event->mpi_error = mpi_status;
    if (channel->send_request == MPI_REQUEST_NULL) channel->send_active = 0;
    return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  }
  if (complete) {
    channel->send_active = 0;
    event->kinds |= WT_MPI_CHANNEL_EVENT_SEND_COMPLETE;
  }
  return WT_MPI_CHANNEL_OK;
}

static enum wt_mpi_channel_status finish_receive_request(
    struct wt_mpi_channel *channel, struct wt_mpi_channel_event *event) {
  MPI_Status receive_status;
  int complete = 0;
  int received_count = 0;
  int mpi_status;
  if (!channel->receive_request_active) return WT_MPI_CHANNEL_OK;
  mpi_status = MPI_Test(&channel->receive_request, &complete, &receive_status);
  if (mpi_status != MPI_SUCCESS) {
    event->mpi_error = mpi_status;
    if (channel->receive_request == MPI_REQUEST_NULL) {
      channel->receive_request_active = 0;
      channel->receive_armed = 0;
      channel->draining_oversize = 0;
    }
    return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  }
  if (!complete) return WT_MPI_CHANNEL_OK;
  channel->receive_request_active = 0;
  channel->receive_armed = 0;
  if (channel->draining_oversize) {
    channel->draining_oversize = 0;
    return poison(channel, WT_MPI_CHANNEL_LIMIT);
  }
  mpi_status = MPI_Get_count(&receive_status, MPI_BYTE, &received_count);
  if (mpi_status != MPI_SUCCESS || received_count < 0 ||
      (size_t)received_count != channel->receive_length) {
    event->mpi_error = mpi_status;
    return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  }
  channel->receive_ready = 1;
  event->kinds |= WT_MPI_CHANNEL_EVENT_RECEIVE_READY;
  event->received_bytes = channel->receive_buffer;
  event->received_length = channel->receive_length;
  return WT_MPI_CHANNEL_OK;
}

static enum wt_mpi_channel_status probe_receive(struct wt_mpi_channel *channel,
    struct wt_mpi_channel_event *event) {
  MPI_Status probe_status;
  int available = 0;
  int received_count = 0;
  int mpi_status;
  if (!channel->receive_armed || channel->receive_request_active ||
      channel->receive_ready) return WT_MPI_CHANNEL_OK;
  mpi_status = MPI_Iprobe(channel->peer, 0, channel->communicator, &available,
      &probe_status);
  if (mpi_status != MPI_SUCCESS) {
    event->mpi_error = mpi_status;
    return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  }
  if (!available) return WT_MPI_CHANNEL_OK;
  mpi_status = MPI_Get_count(&probe_status, MPI_BYTE, &received_count);
  if (mpi_status != MPI_SUCCESS || received_count == MPI_UNDEFINED ||
      received_count < 0) {
    event->mpi_error = mpi_status;
    return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  }
  channel->receive_request = MPI_REQUEST_NULL;
  channel->receive_request_active = 1;
  if ((size_t)received_count > channel->max_frame) {
    channel->poisoned = 1;
    channel->draining_oversize = 1;
    mpi_status = MPI_Irecv(channel->receive_buffer, (int)channel->max_frame,
        MPI_BYTE, channel->peer, 0, channel->communicator,
        &channel->receive_request);
    if (mpi_status != MPI_SUCCESS) {
      event->mpi_error = mpi_status;
      if (channel->receive_request == MPI_REQUEST_NULL) {
        channel->receive_request_active = 0;
        channel->receive_armed = 0;
        channel->draining_oversize = 0;
      }
      return WT_MPI_CHANNEL_MPI_ERROR;
    }
    return WT_MPI_CHANNEL_LIMIT;
  }
  channel->receive_length = (size_t)received_count;
  mpi_status = MPI_Irecv(channel->receive_buffer, received_count, MPI_BYTE,
      channel->peer, 0, channel->communicator, &channel->receive_request);
  if (mpi_status != MPI_SUCCESS) {
    event->mpi_error = mpi_status;
    if (channel->receive_request == MPI_REQUEST_NULL) {
      channel->receive_request_active = 0;
      channel->receive_armed = 0;
    }
    return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  }
  return WT_MPI_CHANNEL_OK;
}

enum wt_mpi_channel_status wt_mpi_channel_poll(
    struct wt_mpi_channel *channel, int64_t deadline_monotonic_ns,
    struct wt_mpi_channel_event *event) {
  enum wt_mpi_channel_status status = check_thread(channel);
  enum wt_mpi_channel_status progress_status;
  int64_t now;
  if (status != WT_MPI_CHANNEL_OK) return status;
  if (event == NULL || deadline_monotonic_ns < 0) return WT_MPI_CHANNEL_ARGUMENT;
  *event = (struct wt_mpi_channel_event){0, NULL, 0, MPI_SUCCESS};

  progress_status = progress_send(channel, event);
  if (progress_status == WT_MPI_CHANNEL_MPI_ERROR) return progress_status;
  progress_status = finish_receive_request(channel, event);
  if (progress_status == WT_MPI_CHANNEL_MPI_ERROR) return progress_status;
  if (progress_status == WT_MPI_CHANNEL_LIMIT) return progress_status;
  progress_status = probe_receive(channel, event);
  if (progress_status != WT_MPI_CHANNEL_OK) return progress_status;

  if (!channel->poisoned && (channel->send_active || channel->receive_armed)) {
    now = now_monotonic_ns();
    if (now < 0) return poison(channel, WT_MPI_CHANNEL_CLOCK_ERROR);
    if (now >= deadline_monotonic_ns) {
      event->kinds |= WT_MPI_CHANNEL_EVENT_UNKNOWN_DELIVERY;
      return poison(channel, WT_MPI_CHANNEL_TIMEOUT);
    }
  }
  return channel->poisoned ? WT_MPI_CHANNEL_POISONED : WT_MPI_CHANNEL_OK;
}

enum wt_mpi_channel_status wt_mpi_channel_release_receive(
    struct wt_mpi_channel *channel) {
  enum wt_mpi_channel_status status = check_thread(channel);
  if (status != WT_MPI_CHANNEL_OK) return status;
  if (!channel->receive_ready) return WT_MPI_CHANNEL_ARGUMENT;
  channel->receive_ready = 0;
  channel->receive_length = 0;
  return WT_MPI_CHANNEL_OK;
}

enum wt_mpi_channel_status wt_mpi_channel_destroy_ready(
    const struct wt_mpi_channel *channel) {
  enum wt_mpi_channel_status status = check_thread(channel);
  if (status != WT_MPI_CHANNEL_OK) return status;
  return channel->send_active || channel->receive_armed ||
      channel->receive_request_active || channel->receive_ready ?
      WT_MPI_CHANNEL_BUSY : WT_MPI_CHANNEL_OK;
}

enum wt_mpi_channel_status wt_mpi_channel_destroy_collective(
    struct wt_mpi_channel **channel_pointer) {
  struct wt_mpi_channel *channel;
  enum wt_mpi_channel_status status;
  int local_ready;
  int all_ready = 0;
  int mpi_status;
  if (channel_pointer == NULL || *channel_pointer == NULL)
    return WT_MPI_CHANNEL_ARGUMENT;
  channel = *channel_pointer;
  status = check_thread(channel);
  if (status != WT_MPI_CHANNEL_OK) return status;
  local_ready = wt_mpi_channel_destroy_ready(channel) == WT_MPI_CHANNEL_OK;
  mpi_status = MPI_Allreduce(&local_ready, &all_ready, 1, MPI_INT, MPI_MIN,
      channel->communicator);
  if (mpi_status != MPI_SUCCESS) return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  if (!all_ready) return WT_MPI_CHANNEL_BUSY;
  mpi_status = MPI_Comm_free(&channel->communicator);
  if (mpi_status != MPI_SUCCESS) return poison(channel, WT_MPI_CHANNEL_MPI_ERROR);
  discard_allocations(channel);
  *channel_pointer = NULL;
  return WT_MPI_CHANNEL_OK;
}
