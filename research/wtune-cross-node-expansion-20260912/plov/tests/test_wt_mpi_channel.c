#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

#include "../wt_mpi_channel.h"

#define CHECK(expr) do { \
  if (!(expr)) { \
    fprintf(stderr, "rank %d line %d: %s\n", rank, __LINE__, #expr); \
    MPI_Abort(MPI_COMM_WORLD, 2); \
    return 2; \
  } \
} while (0)

struct foreign_call {
  struct wt_mpi_channel *channel;
  enum wt_mpi_channel_status status;
  int peer;
  size_t capacity;
};

static int64_t monotonic_ns(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
  return (int64_t)now.tv_sec * INT64_C(1000000000) + now.tv_nsec;
}

static void pause_one_ms(void) {
  const struct timespec delay = {0, 1000000};
  (void)nanosleep(&delay, NULL);
}

static void *foreign_begin_recv(void *opaque) {
  struct foreign_call *call = opaque;
  call->status = wt_mpi_channel_begin_recv(call->channel);
  return NULL;
}

static void *foreign_get_peer(void *opaque) {
  struct foreign_call *call = opaque;
  call->status = wt_mpi_channel_get_peer(call->channel, &call->peer);
  return NULL;
}

static void *foreign_get_capacity(void *opaque) {
  struct foreign_call *call = opaque;
  call->status = wt_mpi_channel_get_capacity(call->channel, &call->capacity);
  return NULL;
}

static int poll_until_exchange(struct wt_mpi_channel *channel,
    const unsigned char *expected, size_t expected_len, int rank) {
  int saw_send = 0;
  int saw_receive = 0;
  const int64_t deadline = monotonic_ns() + INT64_C(5000000000);
  while (!saw_send || !saw_receive) {
    struct wt_mpi_channel_event event;
    enum wt_mpi_channel_status status = wt_mpi_channel_poll(channel, deadline, &event);
    CHECK(status == WT_MPI_CHANNEL_OK);
    if ((event.kinds & WT_MPI_CHANNEL_EVENT_SEND_COMPLETE) != 0) saw_send = 1;
    if ((event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) != 0) {
      CHECK(event.received_length == expected_len);
      CHECK(memcmp(event.received_bytes, expected, expected_len) == 0);
      saw_receive = 1;
    }
    if (!saw_send || !saw_receive) pause_one_ms();
  }
  CHECK(wt_mpi_channel_release_receive(channel) == WT_MPI_CHANNEL_OK);
  return 0;
}

static int test_validation_and_exchange(int rank, int peer) {
  struct wt_mpi_channel *channel = NULL;
  struct foreign_call foreign;
  pthread_t thread;
  unsigned char source[5];
  const unsigned char rank0_bytes[5] = {1, 3, 5, 7, 9};
  const unsigned char rank1_bytes[5] = {2, 4, 6, 8, 10};
  const unsigned char *expected = rank == 0 ? rank1_bytes : rank0_bytes;

  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, 0, &channel) ==
      WT_MPI_CHANNEL_ARGUMENT);
  CHECK(channel == NULL);
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, 1024, &channel) ==
      WT_MPI_CHANNEL_OK);
  CHECK(channel != NULL);

  CHECK(wt_mpi_channel_get_peer(channel, &foreign.peer) == WT_MPI_CHANNEL_OK);
  CHECK(foreign.peer == peer);
  {
    size_t capacity = 0;
    CHECK(wt_mpi_channel_get_capacity(channel, &capacity) == WT_MPI_CHANNEL_OK);
    CHECK(capacity == 1024);
    CHECK(wt_mpi_channel_get_capacity(channel, NULL) == WT_MPI_CHANNEL_ARGUMENT);
  }
  CHECK(wt_mpi_channel_get_peer(channel, NULL) == WT_MPI_CHANNEL_ARGUMENT);
  foreign = (struct foreign_call){channel, WT_MPI_CHANNEL_OK, -1, 77};
  CHECK(pthread_create(&thread, NULL, foreign_get_peer, &foreign) == 0);
  CHECK(pthread_join(thread, NULL) == 0);
  CHECK(foreign.status == WT_MPI_CHANNEL_THREAD_AFFINITY);
  CHECK(foreign.peer == -1);

  CHECK(pthread_create(&thread, NULL, foreign_get_capacity, &foreign) == 0);
  CHECK(pthread_join(thread, NULL) == 0);
  CHECK(foreign.status == WT_MPI_CHANNEL_THREAD_AFFINITY);
  CHECK(foreign.capacity == 77);

  foreign = (struct foreign_call){channel, WT_MPI_CHANNEL_OK, -1, 77};
  CHECK(pthread_create(&thread, NULL, foreign_begin_recv, &foreign) == 0);
  CHECK(pthread_join(thread, NULL) == 0);
  CHECK(foreign.status == WT_MPI_CHANNEL_THREAD_AFFINITY);

  CHECK(wt_mpi_channel_begin_send(channel, rank0_bytes, 0) ==
      WT_MPI_CHANNEL_ARGUMENT);
  CHECK(wt_mpi_channel_begin_send(channel, rank0_bytes, 1025) ==
      WT_MPI_CHANNEL_LIMIT);

  memcpy(source, rank == 0 ? rank0_bytes : rank1_bytes, sizeof(source));
  CHECK(wt_mpi_channel_begin_send(channel, source, sizeof(source)) ==
      WT_MPI_CHANNEL_OK);
  memset(source, 0xee, sizeof(source));
  CHECK(wt_mpi_channel_begin_send(channel, source, sizeof(source)) ==
      WT_MPI_CHANNEL_BUSY);
  CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_BUSY);
  CHECK(wt_mpi_channel_destroy_ready(channel) == WT_MPI_CHANNEL_BUSY);
  CHECK(poll_until_exchange(channel, expected, sizeof(source), rank) == 0);
  CHECK(wt_mpi_channel_destroy_ready(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_destroy_collective(&channel) == WT_MPI_CHANNEL_OK);
  CHECK(channel == NULL);
  return 0;
}

static int test_armed_receive_timeout(int rank, int peer) {
  struct wt_mpi_channel *channel = NULL;
  struct wt_mpi_channel_event event;
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, 128, &channel) ==
      WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_poll(channel, monotonic_ns() - 1, &event) ==
      WT_MPI_CHANNEL_TIMEOUT);
  CHECK((event.kinds & WT_MPI_CHANNEL_EVENT_UNKNOWN_DELIVERY) != 0);
  CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_POISONED);
  CHECK(wt_mpi_channel_begin_send(channel, "x", 1) == WT_MPI_CHANNEL_POISONED);
  CHECK(wt_mpi_channel_destroy_ready(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_destroy_collective(&channel) == WT_MPI_CHANNEL_OK);
  return 0;
}

static int test_collective_destroy_agrees_on_busy(int rank, int peer) {
  struct wt_mpi_channel *channel = NULL;
  struct wt_mpi_channel_event event;
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, 128, &channel) ==
      WT_MPI_CHANNEL_OK);
  if (rank == 0)
    CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_destroy_collective(&channel) == WT_MPI_CHANNEL_BUSY);
  CHECK(channel != NULL);
  if (rank == 0) {
    CHECK(wt_mpi_channel_poll(channel, monotonic_ns() - 1, &event) ==
        WT_MPI_CHANNEL_TIMEOUT);
  }
  CHECK(wt_mpi_channel_destroy_ready(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_destroy_collective(&channel) == WT_MPI_CHANNEL_OK);
  return 0;
}

static int test_pending_send_timeout_and_reap(int rank, int peer) {
  const size_t frame_size = 4U * 1024U * 1024U;
  struct wt_mpi_channel *channel = NULL;
  struct wt_mpi_channel_event event;
  unsigned char *payload = malloc(frame_size);
  int received = 0;
  int send_reaped = 0;
  const int64_t finish = monotonic_ns() + INT64_C(10000000000);
  CHECK(payload != NULL);
  memset(payload, 0x5c, frame_size);
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, frame_size, &channel) ==
      WT_MPI_CHANNEL_OK);

  if (rank == 0) {
    CHECK(wt_mpi_channel_begin_send(channel, payload, frame_size) ==
        WT_MPI_CHANNEL_OK);
    memset(payload, 0x33, frame_size);
    CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_OK);
    CHECK(wt_mpi_channel_poll(channel, finish, &event) == WT_MPI_CHANNEL_OK);
    CHECK(event.kinds == WT_MPI_CHANNEL_EVENT_NONE);
    CHECK(wt_mpi_channel_poll(channel, monotonic_ns() - 1, &event) ==
        WT_MPI_CHANNEL_TIMEOUT);
    CHECK((event.kinds & WT_MPI_CHANNEL_EVENT_UNKNOWN_DELIVERY) != 0);
    CHECK(wt_mpi_channel_begin_send(channel, payload, 1) ==
        WT_MPI_CHANNEL_POISONED);
  } else {
    const struct timespec delay = {0, 50000000};
    (void)nanosleep(&delay, NULL);
    CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_OK);
  }

  while (monotonic_ns() < finish && ((rank == 0 && !send_reaped) ||
      (rank == 1 && !received))) {
    enum wt_mpi_channel_status status = wt_mpi_channel_poll(channel, finish, &event);
    if (rank == 0) {
      CHECK(status == WT_MPI_CHANNEL_POISONED);
      if ((event.kinds & WT_MPI_CHANNEL_EVENT_SEND_COMPLETE) != 0) send_reaped = 1;
    } else {
      CHECK(status == WT_MPI_CHANNEL_OK);
      if ((event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) != 0) {
        CHECK(event.received_length == frame_size);
        CHECK(((const unsigned char *)event.received_bytes)[0] == 0x5c);
        CHECK(((const unsigned char *)event.received_bytes)[frame_size - 1] == 0x5c);
        received = 1;
      }
    }
    if ((rank == 0 && !send_reaped) || (rank == 1 && !received)) pause_one_ms();
  }
  CHECK(rank == 0 ? send_reaped : received);
  if (rank == 1) CHECK(wt_mpi_channel_release_receive(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_destroy_ready(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_destroy_collective(&channel) == WT_MPI_CHANNEL_OK);
  free(payload);
  return 0;
}

static int test_oversize_peer_frame_is_drained(int rank, int peer) {
  const size_t local_limit = rank == 0 ? 256 : 32;
  unsigned char payload[64];
  struct wt_mpi_channel *channel = NULL;
  struct wt_mpi_channel_event event;
  int terminal = 0;
  const int64_t deadline = monotonic_ns() + INT64_C(5000000000);
  memset(payload, 0xa7, sizeof(payload));
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, local_limit, &channel) ==
      WT_MPI_CHANNEL_OK);
  if (rank == 0) {
    CHECK(wt_mpi_channel_begin_send(channel, payload, sizeof(payload)) ==
        WT_MPI_CHANNEL_OK);
  } else {
    CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_OK);
  }
  while (!terminal && monotonic_ns() < deadline) {
    enum wt_mpi_channel_status status = wt_mpi_channel_poll(channel, deadline, &event);
    if (rank == 0) {
      CHECK(status == WT_MPI_CHANNEL_OK);
      terminal = (event.kinds & WT_MPI_CHANNEL_EVENT_SEND_COMPLETE) != 0;
    } else {
      CHECK(status == WT_MPI_CHANNEL_OK || status == WT_MPI_CHANNEL_LIMIT ||
          status == WT_MPI_CHANNEL_POISONED || status == WT_MPI_CHANNEL_MPI_ERROR);
      terminal = wt_mpi_channel_destroy_ready(channel) == WT_MPI_CHANNEL_OK;
    }
    if (!terminal) pause_one_ms();
  }
  CHECK(terminal);
  if (rank == 1) CHECK(wt_mpi_channel_begin_recv(channel) == WT_MPI_CHANNEL_POISONED);
  CHECK(wt_mpi_channel_destroy_ready(channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_mpi_channel_destroy_collective(&channel) == WT_MPI_CHANNEL_OK);
  return 0;
}

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  int rank = -1;
  int size = 0;
  int peer;
  struct wt_mpi_channel *invalid_channel = NULL;
  (void)argv;
  if (wt_mpi_channel_create(MPI_COMM_WORLD, 0, 16, &invalid_channel) !=
      WT_MPI_CHANNEL_MPI_UNAVAILABLE || invalid_channel != NULL) {
    fprintf(stderr, "create accepted an uninitialized MPI runtime\n");
    return 1;
  }
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
    return 1;
  if (provided < MPI_THREAD_FUNNELED || MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS || size != 2) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  peer = 1 - rank;
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, rank == 0 ? peer : size, 16,
      &invalid_channel) ==
      WT_MPI_CHANNEL_PEER);
  CHECK(invalid_channel == NULL);
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer,
      rank == 0 ? 16 : (size_t)INT_MAX + 1,
      &invalid_channel) == WT_MPI_CHANNEL_LIMIT);
  CHECK(invalid_channel == NULL);
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, 16,
      rank == 0 ? NULL : &invalid_channel) == WT_MPI_CHANNEL_ARGUMENT);
  CHECK(invalid_channel == NULL);
  CHECK(test_validation_and_exchange(rank, peer) == 0);
  CHECK(test_armed_receive_timeout(rank, peer) == 0);
  CHECK(test_collective_destroy_agrees_on_busy(rank, peer) == 0);
  CHECK(test_pending_send_timeout_and_reap(rank, peer) == 0);
  CHECK(test_oversize_peer_frame_is_drained(rank, peer) == 0);
  CHECK(MPI_Finalize() == MPI_SUCCESS);
  if (wt_mpi_channel_create(MPI_COMM_WORLD, peer, 16, &invalid_channel) !=
      WT_MPI_CHANNEL_MPI_UNAVAILABLE || invalid_channel != NULL) {
    fprintf(stderr, "create accepted a finalized MPI runtime\n");
    return 1;
  }
  return 0;
}
