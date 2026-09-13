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
    MPI_Abort(MPI_COMM_WORLD, 5); \
    return 5; \
  } \
} while (0)

static int fake_clock_calls;
static int64_t fake_deadline;

int64_t wt_mpi_remote_test_monotonic_now_ns(void) {
  ++fake_clock_calls;
  return fake_clock_calls < 3 ? fake_deadline - 1 : fake_deadline;
}

struct callback_context {
  const struct wt_ledger_contract *contract;
  int run_calls;
  int commit_calls;
};

static void pause_one_ms(void) {
  const struct timespec delay = {0, 1000000};
  (void)nanosleep(&delay, NULL);
}

static int64_t real_now_ns(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
  return (int64_t)now.tv_sec * INT64_C(1000000000) + now.tv_nsec;
}

static struct wt_ledger_contract make_contract(void) {
  struct wt_ledger_contract contract = {0};
  contract.run_uuid[0] = 0x81;
  contract.owner_rank = 0;
  contract.owner_generation = 3;
  memset(contract.schema_fingerprint, 0x4a,
      sizeof(contract.schema_fingerprint));
  contract.abi = (struct wt_payload_abi){4, 8, 4,
      WT_FLOAT_IEEE754_BINARY64, 1, 2};
  contract.limits = (struct wt_wire_limits){512,
      512 - WT_WIRE_HEADER_BYTES, 2};
  return contract;
}

static int encode(const struct wt_ledger_contract *contract, uint8_t kind,
    uint8_t payload, uint8_t *out, size_t *out_len) {
  struct wt_wire_frame frame = {0};
  frame.kind = kind;
  frame.id.run_uuid[0] = 0x81;
  frame.id.epoch = 1;
  frame.id.domain = 2;
  frame.id.phase = 91;
  frame.id.call_seq = 4;
  frame.id.owner_rank = 0;
  frame.id.owner_generation = 3;
  frame.id.gi1 = frame.id.gj1 = 1;
  memcpy(frame.schema_fingerprint, contract->schema_fingerprint,
      sizeof(frame.schema_fingerprint));
  memset(frame.request_fingerprint, 0x7b,
      sizeof(frame.request_fingerprint));
  frame.abi = contract->abi;
  frame.payload = &payload;
  frame.payload_len = 1;
  return wt_wire_encode(&frame, &contract->limits, out, 512, out_len);
}

static int run_callback(void *opaque, const uint8_t *request,
    size_t request_len, uint8_t *result, size_t capacity, size_t *result_len) {
  struct callback_context *context = opaque;
  struct wt_wire_frame frame;
  (void)capacity;
  ++context->run_calls;
  if (wt_wire_decode(request, request_len, &context->contract->limits,
      context->contract->schema_fingerprint, &context->contract->abi,
      &frame) != WT_WIRE_OK) return 1;
  return encode(context->contract, WT_WIRE_RESULT, 0x42, result, result_len) ==
      WT_WIRE_OK ? 0 : 1;
}

static int commit_callback(void *opaque, const struct wt_wire_frame *result) {
  struct callback_context *context = opaque;
  (void)result;
  ++context->commit_calls;
  return 0;
}

int main(int argc, char **argv) {
  struct wt_ledger_contract contract;
  struct wt_mpi_channel *channel = NULL;
  struct wt_ledger *ledger = NULL;
  struct wt_policy_pool *pool = NULL;
  struct wt_mpi_remote_owner *owner = NULL;
  struct wt_mpi_remote_worker *worker = NULL;
  struct callback_context context = {0};
  struct wt_mpi_remote_event event;
  uint8_t request[512];
  size_t request_len;
  int provided = MPI_THREAD_SINGLE;
  int rank = -1, size = 0, done = 0;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
    return 1;
  CHECK(provided >= MPI_THREAD_FUNNELED);
  CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
  CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS && size == 2);
  contract = make_contract();
  context.contract = &contract;
  fake_deadline = real_now_ns() + INT64_C(5000000000);
  CHECK(encode(&contract, WT_WIRE_REQUEST, 0x11, request, &request_len) ==
      WT_WIRE_OK);
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, 1 - rank, 512, &channel) ==
      WT_MPI_CHANNEL_OK);
  CHECK(wt_ledger_create(rank == 0 ? WT_LEDGER_OWNER : WT_LEDGER_WORKER,
      &contract, 2, 4096, &ledger) == WT_LEDGER_OK);
  if (rank == 0) {
    const struct wt_policy_pool_config pool_config = {1, 1, 1024};
    const struct wt_mpi_remote_endpoint endpoint = {1, 1, 11};
    struct wt_policy_config policy = {1, WT_POLICY_SCOPE_CROSS_NODE, {0}, 1, 0};
    struct wt_policy_request policy_request = {91, {0, 10, 0, 1},
        WT_POLICY_SCOPE_CROSS_NODE, 1, request_len + 512, 10000};
    struct wt_policy_candidate candidate = {{1, 11, 0, 1, {0},
        WT_POLICY_CAP_CROSS_NODE, 1, 1024}, {1, 1, 1, 1, 1, 1, 1000, 200, 200}};
    struct wt_mpi_remote_packet packet = {request, request_len, 512};
    struct wt_policy_choice choice;
    memcpy(policy.schema_fingerprint, contract.schema_fingerprint,
        sizeof(policy.schema_fingerprint));
    memcpy(candidate.peer.schema_fingerprint, contract.schema_fingerprint,
        sizeof(candidate.peer.schema_fingerprint));
    CHECK(wt_policy_pool_create(&pool_config, &pool) == WT_POLICY_OK);
    CHECK(wt_mpi_remote_owner_create_take(&channel, &endpoint, ledger, pool,
        &contract, &owner) == WT_MPI_REMOTE_OK);
    CHECK(wt_mpi_remote_owner_begin(owner, &policy, &policy_request, &candidate,
        1, &packet, fake_deadline, commit_callback, &context, &choice) ==
        WT_MPI_REMOTE_OK);
  } else {
    const struct wt_mpi_remote_endpoint endpoint = {0, 0, 10};
    CHECK(wt_mpi_remote_worker_create_take(&channel, &endpoint, ledger,
        &contract, run_callback, &context, &worker) == WT_MPI_REMOTE_OK);
    CHECK(wt_mpi_remote_worker_arm(worker,
        real_now_ns() + INT64_C(5000000000)) == WT_MPI_REMOTE_OK);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 1) {
    const int64_t stop = real_now_ns() + INT64_C(5000000000);
    while (!done && real_now_ns() < stop) {
      CHECK(wt_mpi_remote_worker_progress(worker, &event) == WT_MPI_REMOTE_OK);
      done = (event.kinds & WT_MPI_REMOTE_EVENT_WORKER_EXECUTED) != 0;
      if (!done) pause_one_ms();
    }
    CHECK(done);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0) {
    struct wt_policy_pool_stats stats;
    struct wt_ledger_ticket ticket = owner->ledger_ticket;
    enum wt_ledger_state ledger_state;
    CHECK(wt_mpi_remote_owner_progress(owner, &event) == WT_MPI_REMOTE_OK);
    CHECK((event.kinds & WT_MPI_REMOTE_EVENT_OWNER_COMMITTED) == 0);
    CHECK(wt_mpi_remote_owner_progress(owner, &event) ==
        WT_MPI_REMOTE_DEADLINE_EXPIRED_RESULT_KNOWN);
    CHECK(context.commit_calls == 0);
    CHECK((event.kinds & WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY) == 0);
    CHECK(wt_policy_pool_get_stats(pool, &stats) == WT_POLICY_OK);
    CHECK(stats.inflight == 0);
    CHECK(wt_ledger_get_state(ledger, &ticket, &ledger_state) == WT_LEDGER_OK);
    CHECK(ledger_state == WT_LEDGER_FAILED);
  } else {
    const int64_t stop = real_now_ns() + INT64_C(5000000000);
    while (wt_mpi_remote_worker_shutdown_ready(worker) == WT_MPI_REMOTE_BUSY &&
        real_now_ns() < stop) {
      CHECK(wt_mpi_remote_worker_progress(worker, &event) == WT_MPI_REMOTE_OK);
      pause_one_ms();
    }
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(wt_mpi_remote_owner_destroy_collective(&owner) == WT_MPI_REMOTE_OK);
  else
    CHECK(wt_mpi_remote_worker_destroy_collective(&worker) == WT_MPI_REMOTE_OK);
  CHECK(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
  if (rank == 0) wt_policy_pool_destroy(pool);
  CHECK(MPI_Finalize() == MPI_SUCCESS);
  return 0;
}
