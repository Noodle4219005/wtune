#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

#include "../wt_mpi_remote_internal.h"

#define CHECK(expr) do { \
  if (!(expr)) { \
    fprintf(stderr, "rank %d line %d: %s\n", rank, __LINE__, #expr); \
    MPI_Abort(MPI_COMM_WORLD, 4); \
    return 4; \
  } \
} while (0)

struct callback_context {
  const struct wt_ledger_contract *contract;
  struct wt_mpi_remote_owner **owner;
  struct wt_mpi_remote_worker **worker;
  enum wt_mpi_remote_status abandon_status;
  enum wt_mpi_remote_status ready_status;
  enum wt_mpi_remote_status destroy_status;
  unsigned sleep_ms;
  int calls;
};

struct fixture {
  struct wt_ledger_contract contract;
  struct wt_mpi_channel *channel;
  struct wt_ledger *ledger;
  struct wt_policy_pool *pool;
  struct wt_mpi_remote_owner *owner;
  struct wt_mpi_remote_worker *worker;
};

static int64_t now_ns(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
  return (int64_t)now.tv_sec * INT64_C(1000000000) + now.tv_nsec;
}

static void sleep_ms(unsigned milliseconds) {
  struct timespec delay = {(time_t)(milliseconds / 1000),
      (long)(milliseconds % 1000) * 1000000L};
  (void)nanosleep(&delay, NULL);
}

static struct wt_ledger_contract contract_for(size_t capacity) {
  struct wt_ledger_contract contract = {0};
  contract.run_uuid[0] = 0x71;
  contract.owner_rank = 0;
  contract.owner_generation = 9;
  memset(contract.schema_fingerprint, 0x2c,
      sizeof(contract.schema_fingerprint));
  contract.abi = (struct wt_payload_abi){4, 8, 4,
      WT_FLOAT_IEEE754_BINARY64, 1, 2};
  contract.limits = (struct wt_wire_limits){capacity,
      capacity - WT_WIRE_HEADER_BYTES, 2};
  return contract;
}

static int encode_frame(const struct wt_ledger_contract *contract, uint8_t kind,
    uint32_t phase, const uint8_t *payload, size_t payload_len,
    uint8_t *out, size_t capacity, size_t *out_len) {
  struct wt_wire_frame frame = {0};
  frame.kind = kind;
  frame.id.run_uuid[0] = 0x71;
  frame.id.epoch = 1;
  frame.id.domain = 2;
  frame.id.phase = phase;
  frame.id.call_seq = 3;
  frame.id.owner_rank = 0;
  frame.id.owner_generation = 9;
  frame.id.gi1 = frame.id.gj1 = 1;
  memcpy(frame.schema_fingerprint, contract->schema_fingerprint,
      sizeof(frame.schema_fingerprint));
  memset(frame.request_fingerprint, 0x6a,
      sizeof(frame.request_fingerprint));
  frame.abi = contract->abi;
  frame.payload = payload;
  frame.payload_len = payload_len;
  return wt_wire_encode(&frame, &contract->limits, out, capacity, out_len);
}

static int run_result(void *opaque, const uint8_t *request, size_t request_len,
    uint8_t *result, size_t capacity, size_t *result_len) {
  struct callback_context *context = opaque;
  struct wt_wire_frame frame;
  uint8_t payload = 0x5d;
  ++context->calls;
  if (context->worker != NULL) {
    context->ready_status = wt_mpi_remote_worker_shutdown_ready(*context->worker);
    context->destroy_status = wt_mpi_remote_worker_destroy_collective(
        context->worker);
  }
  if (context->sleep_ms != 0) sleep_ms(context->sleep_ms);
  if (wt_wire_decode(request, request_len, &context->contract->limits,
      context->contract->schema_fingerprint, &context->contract->abi,
      &frame) != WT_WIRE_OK) return 1;
  return encode_frame(context->contract, WT_WIRE_RESULT, frame.id.phase,
      &payload, 1, result, capacity, result_len) == WT_WIRE_OK ? 0 : 1;
}

static int commit_result(void *opaque, const struct wt_wire_frame *result) {
  struct callback_context *context = opaque;
  (void)result;
  ++context->calls;
  if (context->owner != NULL) {
    struct wt_mpi_remote_event event;
    context->abandon_status = wt_mpi_remote_owner_abandon(
        *context->owner, &event);
    context->ready_status = wt_mpi_remote_owner_shutdown_ready(*context->owner);
    context->destroy_status = wt_mpi_remote_owner_destroy_collective(
        context->owner);
  }
  return 0;
}

static struct wt_policy_candidate candidate_for(
    const struct wt_ledger_contract *contract) {
  struct wt_policy_candidate candidate = {0};
  candidate.peer = (struct wt_policy_peer){1, 11, 0, 1, {0},
      WT_POLICY_CAP_CROSS_NODE, 1, 1024};
  memcpy(candidate.peer.schema_fingerprint, contract->schema_fingerprint,
      sizeof(candidate.peer.schema_fingerprint));
  candidate.estimate = (struct wt_policy_estimate){1, 1, 1, 1, 1, 1,
      1000, 200, 200};
  return candidate;
}

static int setup(struct fixture *fixture, int rank, int peer,
    struct callback_context *context) {
  struct wt_mpi_remote_endpoint endpoint = {peer, rank == 0 ? 1U : 0U,
      rank == 0 ? 11U : 10U};
  fixture->contract = contract_for(512);
  fixture->channel = NULL;
  fixture->ledger = NULL;
  fixture->pool = NULL;
  fixture->owner = NULL;
  fixture->worker = NULL;
  context->contract = &fixture->contract;
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, 512,
      &fixture->channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_ledger_create(rank == 0 ? WT_LEDGER_OWNER : WT_LEDGER_WORKER,
      &fixture->contract, 4, 8192, &fixture->ledger) == WT_LEDGER_OK);
  if (rank == 0) {
    const struct wt_policy_pool_config config = {1, 1, 1024};
    CHECK(wt_policy_pool_create(&config, &fixture->pool) == WT_POLICY_OK);
    CHECK(wt_mpi_remote_owner_create_take(&fixture->channel, &endpoint,
        fixture->ledger, fixture->pool, &fixture->contract, &fixture->owner) ==
        WT_MPI_REMOTE_OK);
  } else {
    CHECK(wt_mpi_remote_worker_create_take(&fixture->channel, &endpoint,
        fixture->ledger, &fixture->contract, run_result, context,
        &fixture->worker) == WT_MPI_REMOTE_OK);
  }
  return 0;
}

static int begin(struct fixture *fixture, const uint8_t *frame, size_t length,
    uint32_t phase, int64_t deadline, struct callback_context *context) {
  struct wt_policy_config policy = {1, WT_POLICY_SCOPE_CROSS_NODE, {0}, 1, 0};
  struct wt_policy_request request = {phase, {0, 10, 0, 1},
      WT_POLICY_SCOPE_CROSS_NODE, 1, length + 512, 10000};
  struct wt_policy_candidate candidate = candidate_for(&fixture->contract);
  struct wt_mpi_remote_packet packet = {frame, length, 512};
  struct wt_policy_choice choice;
  memcpy(policy.schema_fingerprint, fixture->contract.schema_fingerprint,
      sizeof(policy.schema_fingerprint));
  return wt_mpi_remote_owner_begin(fixture->owner, &policy, &request,
      &candidate, 1, &packet, deadline, commit_result, context, &choice);
}

static int destroy(struct fixture *fixture, int rank) {
  if (rank == 0) {
    CHECK(wt_mpi_remote_owner_shutdown_ready(fixture->owner) == WT_MPI_REMOTE_OK);
    CHECK(wt_mpi_remote_owner_destroy_collective(&fixture->owner) ==
        WT_MPI_REMOTE_OK);
  } else {
    CHECK(wt_mpi_remote_worker_shutdown_ready(fixture->worker) == WT_MPI_REMOTE_OK);
    CHECK(wt_mpi_remote_worker_destroy_collective(&fixture->worker) ==
        WT_MPI_REMOTE_OK);
  }
  CHECK(wt_ledger_destroy(&fixture->ledger) == WT_LEDGER_OK);
  if (rank == 0) wt_policy_pool_destroy(fixture->pool);
  return 0;
}

static int test_owner_late_ready(int rank, int peer) {
  struct fixture fixture;
  struct callback_context context = {0};
  struct wt_mpi_remote_event event;
  uint8_t frame[512], payload = 1;
  size_t length;
  int worker_sent = 0;
  int64_t deadline;
  CHECK(setup(&fixture, rank, peer, &context) == 0);
  CHECK(encode_frame(&fixture.contract, WT_WIRE_REQUEST, 51, &payload, 1,
      frame, sizeof(frame), &length) == WT_WIRE_OK);
  deadline = now_ns() + INT64_C(500000000);
  if (rank == 1)
    CHECK(wt_mpi_remote_worker_arm(fixture.worker,
        now_ns() + INT64_C(5000000000)) == WT_MPI_REMOTE_OK);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(begin(&fixture, frame, length, 51, deadline, &context) ==
        WT_MPI_REMOTE_OK);
  if (rank == 1) {
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (!worker_sent && now_ns() < stop) {
      CHECK(wt_mpi_remote_worker_progress(fixture.worker, &event) ==
          WT_MPI_REMOTE_OK);
      worker_sent = (event.kinds & WT_MPI_REMOTE_EVENT_WORKER_EXECUTED) != 0;
      if (!worker_sent) sleep_ms(1);
    }
    CHECK(worker_sent);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0) {
    CHECK(wt_mpi_remote_owner_progress(fixture.owner, &event) == WT_MPI_REMOTE_OK);
    CHECK((event.kinds & WT_MPI_REMOTE_EVENT_OWNER_COMMITTED) == 0);
    while (now_ns() < deadline + INT64_C(20000000)) sleep_ms(1);
    CHECK(wt_mpi_remote_owner_progress(fixture.owner, &event) ==
        WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
    CHECK((event.kinds & WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY) != 0);
    CHECK(context.calls == 0);
  } else {
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (wt_mpi_remote_worker_shutdown_ready(fixture.worker) ==
        WT_MPI_REMOTE_BUSY && now_ns() < stop) {
      CHECK(wt_mpi_remote_worker_progress(fixture.worker, &event) ==
          WT_MPI_REMOTE_OK);
      sleep_ms(1);
    }
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0) {
    struct wt_policy_ticket saved = fixture.owner->admission_ticket;
    struct wt_policy_pool_stats stats;
    fixture.owner->admission_ticket.pool_identity ^= UINT64_C(1);
    CHECK(wt_mpi_remote_owner_destroy_collective(&fixture.owner) ==
        WT_MPI_REMOTE_ADMISSION_RELEASE_FAILED);
    CHECK(fixture.owner != NULL && fixture.owner->channel == NULL &&
        fixture.owner->state == WT_MPI_REMOTE_OWNER_CHANNEL_CLOSED);
    CHECK(wt_mpi_remote_owner_finish_closed(&fixture.owner) ==
        WT_MPI_REMOTE_ADMISSION_RELEASE_FAILED);
    fixture.owner->admission_ticket = saved;
    CHECK(wt_mpi_remote_owner_finish_closed(&fixture.owner) == WT_MPI_REMOTE_OK);
    CHECK(wt_policy_pool_get_stats(fixture.pool, &stats) == WT_POLICY_OK);
    CHECK(stats.inflight == 0);
  } else {
    CHECK(wt_mpi_remote_worker_destroy_collective(&fixture.worker) ==
        WT_MPI_REMOTE_OK);
  }
  CHECK(wt_ledger_destroy(&fixture.ledger) == WT_LEDGER_OK);
  if (rank == 0) wt_policy_pool_destroy(fixture.pool);
  return 0;
}

static int test_callback_guards_and_worker_deadline(int rank, int peer) {
  struct fixture fixture;
  struct callback_context context = {0};
  struct wt_mpi_remote_event event;
  uint8_t frame[512], payload = 2;
  size_t length;
  int done = 0;
  enum wt_mpi_remote_status status = WT_MPI_REMOTE_OK;
  CHECK(setup(&fixture, rank, peer, &context) == 0);
  CHECK(encode_frame(&fixture.contract, WT_WIRE_REQUEST, 52, &payload, 1,
      frame, sizeof(frame), &length) == WT_WIRE_OK);
  if (rank == 0) context.owner = &fixture.owner;
  if (rank == 1)
    CHECK(wt_mpi_remote_worker_arm(fixture.worker,
        now_ns() + INT64_C(5000000000)) == WT_MPI_REMOTE_OK);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(begin(&fixture, frame, length, 52,
        now_ns() + INT64_C(5000000000), &context) == WT_MPI_REMOTE_OK);
  {
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (!done && now_ns() < stop) {
      status = rank == 0 ? wt_mpi_remote_owner_progress(fixture.owner, &event) :
          wt_mpi_remote_worker_progress(fixture.worker, &event);
      CHECK(status == WT_MPI_REMOTE_OK);
      done = rank == 0 ?
          (event.kinds & WT_MPI_REMOTE_EVENT_OWNER_COMMITTED) != 0 :
          (event.kinds & WT_MPI_REMOTE_EVENT_SEND_COMPLETE) != 0;
      if (!done) sleep_ms(1);
    }
  }
  CHECK(done);
  if (rank == 0) {
    CHECK(context.calls == 1);
    CHECK(context.abandon_status == WT_MPI_REMOTE_BUSY);
    CHECK(context.ready_status == WT_MPI_REMOTE_BUSY);
    CHECK(context.destroy_status == WT_MPI_REMOTE_BUSY);
    CHECK(fixture.owner != NULL);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  CHECK(destroy(&fixture, rank) == 0);

  context = (struct callback_context){0};
  CHECK(setup(&fixture, rank, peer, &context) == 0);
  CHECK(encode_frame(&fixture.contract, WT_WIRE_REQUEST, 53, &payload, 1,
      frame, sizeof(frame), &length) == WT_WIRE_OK);
  if (rank == 1) {
    context.worker = &fixture.worker;
    context.sleep_ms = 500;
    CHECK(wt_mpi_remote_worker_arm(fixture.worker,
        now_ns() + INT64_C(300000000)) == WT_MPI_REMOTE_OK);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(begin(&fixture, frame, length, 53,
        now_ns() + INT64_C(5000000000), &context) == WT_MPI_REMOTE_OK);
  {
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (status == WT_MPI_REMOTE_OK && now_ns() < stop) {
      status = rank == 0 ? wt_mpi_remote_owner_progress(fixture.owner, &event) :
          wt_mpi_remote_worker_progress(fixture.worker, &event);
      if (rank == 0 &&
          (event.kinds & WT_MPI_REMOTE_EVENT_SEND_COMPLETE) != 0) break;
      if (status == WT_MPI_REMOTE_OK) sleep_ms(1);
    }
  }
  if (rank == 1) {
    CHECK(status == WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
    CHECK(context.calls == 1);
    CHECK(context.ready_status == WT_MPI_REMOTE_BUSY);
    CHECK(context.destroy_status == WT_MPI_REMOTE_BUSY);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0) {
    CHECK(wt_mpi_remote_owner_abandon(fixture.owner, &event) ==
        WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
    CHECK(context.calls == 0);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  CHECK(destroy(&fixture, rank) == 0);
  return 0;
}

static int test_bad_result_is_unknown(int rank, int peer) {
  struct fixture fixture;
  struct callback_context context = {0};
  struct wt_mpi_remote_event remote_event;
  struct wt_mpi_channel_event channel_event;
  uint8_t request[512], result[512], payload = 3;
  size_t request_len, result_len;
  int done = 0;
  CHECK(setup(&fixture, rank, peer, &context) == 0);
  CHECK(encode_frame(&fixture.contract, WT_WIRE_REQUEST, 54, &payload, 1,
      request, sizeof(request), &request_len) == WT_WIRE_OK);
  CHECK(encode_frame(&fixture.contract, WT_WIRE_RESULT, 54, &payload, 1,
      result, sizeof(result), &result_len) == WT_WIRE_OK);
  result[result_len - 1] ^= 1;
  if (rank == 1)
    CHECK(wt_mpi_channel_begin_recv(fixture.worker->channel) == WT_MPI_CHANNEL_OK);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(begin(&fixture, request, request_len, 54,
        now_ns() + INT64_C(5000000000), &context) == WT_MPI_REMOTE_OK);
  {
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (!done && now_ns() < stop) {
      if (rank == 0) {
        enum wt_mpi_remote_status status =
            wt_mpi_remote_owner_progress(fixture.owner, &remote_event);
        if (status != WT_MPI_REMOTE_OK) {
          CHECK(status == WT_MPI_REMOTE_BAD_PACKET);
          CHECK((remote_event.kinds & WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY) != 0);
          CHECK(context.calls == 0);
          done = 1;
        }
      } else {
        CHECK(wt_mpi_channel_poll(fixture.worker->channel,
            now_ns() + INT64_C(5000000000), &channel_event) == WT_MPI_CHANNEL_OK);
        if (channel_event.kinds & WT_MPI_CHANNEL_EVENT_RECEIVE_READY) {
          CHECK(wt_mpi_channel_release_receive(fixture.worker->channel) ==
              WT_MPI_CHANNEL_OK);
          CHECK(wt_mpi_channel_begin_send(fixture.worker->channel, result,
              result_len) == WT_MPI_CHANNEL_OK);
        }
        done = (channel_event.kinds & WT_MPI_CHANNEL_EVENT_SEND_COMPLETE) != 0;
      }
      if (!done) sleep_ms(1);
    }
  }
  CHECK(done);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  CHECK(destroy(&fixture, rank) == 0);
  return 0;
}

static int test_worker_late_ready_skips_callback(int rank, int peer) {
  struct fixture fixture;
  struct callback_context context = {0};
  struct wt_mpi_remote_event event;
  uint8_t frame[512], payload = 4;
  size_t length;
  int sent = 0;
  int64_t deadline = now_ns() + INT64_C(500000000);
  CHECK(setup(&fixture, rank, peer, &context) == 0);
  CHECK(encode_frame(&fixture.contract, WT_WIRE_REQUEST, 55, &payload, 1,
      frame, sizeof(frame), &length) == WT_WIRE_OK);
  if (rank == 1)
    CHECK(wt_mpi_remote_worker_arm(fixture.worker, deadline) == WT_MPI_REMOTE_OK);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(begin(&fixture, frame, length, 55,
        now_ns() + INT64_C(5000000000), &context) == WT_MPI_REMOTE_OK);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 1) {
    CHECK(wt_mpi_remote_worker_progress(fixture.worker, &event) ==
        WT_MPI_REMOTE_OK);
    CHECK((event.kinds & WT_MPI_REMOTE_EVENT_WORKER_EXECUTED) == 0);
    while (now_ns() < deadline + INT64_C(20000000)) sleep_ms(1);
    CHECK(wt_mpi_remote_worker_progress(fixture.worker, &event) ==
        WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
    CHECK(context.calls == 0);
    {
      const int64_t stop = now_ns() + INT64_C(5000000000);
      while (wt_mpi_remote_worker_shutdown_ready(fixture.worker) ==
          WT_MPI_REMOTE_BUSY && now_ns() < stop) {
        enum wt_mpi_remote_status status =
            wt_mpi_remote_worker_progress(fixture.worker, &event);
        CHECK(status == WT_MPI_REMOTE_POISONED ||
            status == WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
        sleep_ms(1);
      }
      CHECK(now_ns() < stop);
    }
  } else {
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (!sent && now_ns() < stop) {
      CHECK(wt_mpi_remote_owner_progress(fixture.owner, &event) ==
          WT_MPI_REMOTE_OK);
      sent = (event.kinds & WT_MPI_REMOTE_EVENT_SEND_COMPLETE) != 0;
      if (!sent) sleep_ms(1);
    }
    CHECK(sent);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(wt_mpi_remote_owner_abandon(fixture.owner, &event) ==
        WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  CHECK(destroy(&fixture, rank) == 0);
  return 0;
}

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  int rank = -1;
  int size = 0;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
    return 1;
  if (provided < MPI_THREAD_FUNNELED ||
      MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS || size != 2) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  CHECK(test_owner_late_ready(rank, 1 - rank) == 0);
  CHECK(test_callback_guards_and_worker_deadline(rank, 1 - rank) == 0);
  CHECK(test_bad_result_is_unknown(rank, 1 - rank) == 0);
  CHECK(test_worker_late_ready_skips_callback(rank, 1 - rank) == 0);
  CHECK(MPI_Finalize() == MPI_SUCCESS);
  return 0;
}
