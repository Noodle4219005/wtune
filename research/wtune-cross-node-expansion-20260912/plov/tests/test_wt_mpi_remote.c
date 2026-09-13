#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

#include "../wt_mpi_remote.h"

#define CHECK(expr) do { \
  if (!(expr)) { \
    fprintf(stderr, "rank %d line %d: %s\n", rank, __LINE__, #expr); \
    MPI_Abort(MPI_COMM_WORLD, 3); \
    return 3; \
  } \
} while (0)

struct run_context {
  const struct wt_ledger_contract *contract;
  size_t result_payload_len;
  int calls;
  int fail;
};

struct commit_context {
  int calls;
  uint8_t value;
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

static void pause_one_ms(void) {
  const struct timespec delay = {0, 1000000};
  (void)nanosleep(&delay, NULL);
}

static struct wt_ledger_contract make_contract(size_t max_frame) {
  struct wt_ledger_contract contract = {0};
  contract.run_uuid[0] = 0x91;
  contract.owner_rank = 0;
  contract.owner_generation = 7;
  memset(contract.schema_fingerprint, 0x4d,
      sizeof(contract.schema_fingerprint));
  contract.abi = (struct wt_payload_abi){4, 8, 4,
      WT_FLOAT_IEEE754_BINARY64, 1, 2};
  contract.limits = (struct wt_wire_limits){max_frame,
      max_frame - WT_WIRE_HEADER_BYTES, 2};
  return contract;
}

static int make_frame(const struct wt_ledger_contract *contract, uint8_t kind,
    uint32_t phase, uint32_t sequence, uint8_t fingerprint,
    const uint8_t *payload, size_t payload_len, uint8_t *out, size_t capacity,
    size_t *out_len) {
  struct wt_wire_frame frame = {0};
  frame.kind = kind;
  frame.id.run_uuid[0] = 0x91;
  frame.id.epoch = 2;
  frame.id.domain = 3;
  frame.id.phase = phase;
  frame.id.call_seq = sequence;
  frame.id.chunk = 1;
  frame.id.owner_rank = 0;
  frame.id.owner_generation = 7;
  frame.id.gi0 = 4;
  frame.id.gi1 = 4;
  frame.id.gj0 = 5;
  frame.id.gj1 = 5;
  memcpy(frame.schema_fingerprint, contract->schema_fingerprint,
      sizeof(frame.schema_fingerprint));
  memset(frame.request_fingerprint, fingerprint,
      sizeof(frame.request_fingerprint));
  frame.abi = contract->abi;
  frame.payload = payload;
  frame.payload_len = payload_len;
  return wt_wire_encode(&frame, &contract->limits, out, capacity, out_len);
}

static int run_callback(void *opaque, const uint8_t *request, size_t request_len,
    uint8_t *result, size_t result_capacity, size_t *result_len) {
  struct run_context *context = opaque;
  struct wt_wire_frame input;
  uint8_t *payload;
  int status;
  ++context->calls;
  if (context->fail) return 1;
  status = wt_wire_decode(request, request_len, &context->contract->limits,
      context->contract->schema_fingerprint, &context->contract->abi, &input);
  if (status != WT_WIRE_OK || input.payload_len != 1 ||
      context->result_payload_len > result_capacity - WT_WIRE_HEADER_BYTES)
    return 1;
  payload = malloc(context->result_payload_len);
  if (payload == NULL) return 1;
  memset(payload, 0xa5, context->result_payload_len);
  payload[0] = (uint8_t)(input.payload[0] + 1);
  status = make_frame(context->contract, WT_WIRE_RESULT, input.id.phase,
      input.id.call_seq, input.request_fingerprint[0], payload,
      context->result_payload_len, result, result_capacity, result_len);
  free(payload);
  return status == WT_WIRE_OK ? 0 : 1;
}

static int commit_callback(void *opaque, const struct wt_wire_frame *result) {
  struct commit_context *context = opaque;
  if (result->payload_len == 0) return 1;
  ++context->calls;
  context->value = result->payload[0];
  return 0;
}

static struct wt_policy_config policy_for(
    const struct wt_ledger_contract *contract) {
  struct wt_policy_config policy = {0};
  policy.enabled = 1;
  policy.allowed_scopes = WT_POLICY_SCOPE_CROSS_NODE;
  memcpy(policy.schema_fingerprint, contract->schema_fingerprint,
      sizeof(policy.schema_fingerprint));
  policy.minimum_samples = 1;
  return policy;
}

static struct wt_policy_request policy_request(uint32_t phase,
    size_t worstcase_bytes) {
  struct wt_policy_request request = {0};
  request.phase_id = phase;
  request.origin = (struct wt_policy_topology){0, 10, 0, 1};
  request.scope = WT_POLICY_SCOPE_CROSS_NODE;
  request.required_worker_slots = 1;
  request.worstcase_bytes = worstcase_bytes;
  request.local_ns = 10000;
  return request;
}

static struct wt_policy_candidate candidate_for(uint32_t rank_id,
    const struct wt_ledger_contract *contract) {
  struct wt_policy_candidate candidate = {0};
  candidate.peer = (struct wt_policy_peer){rank_id, 11, 0, 1, {0},
      WT_POLICY_CAP_CROSS_NODE, 1, contract->limits.max_frame * 2};
  memcpy(candidate.peer.schema_fingerprint, contract->schema_fingerprint,
      sizeof(candidate.peer.schema_fingerprint));
  candidate.estimate = (struct wt_policy_estimate){1, 1, 1, 1, 1, 1,
      1000, 200, 200};
  return candidate;
}

static int setup_fixture(struct fixture *fixture, int rank, int peer,
    size_t max_frame, struct run_context *run) {
  struct wt_mpi_remote_endpoint endpoint;
  fixture->contract = make_contract(max_frame);
  fixture->channel = NULL;
  fixture->ledger = NULL;
  fixture->pool = NULL;
  fixture->owner = NULL;
  fixture->worker = NULL;
  CHECK(wt_mpi_channel_create(MPI_COMM_WORLD, peer, max_frame,
      &fixture->channel) == WT_MPI_CHANNEL_OK);
  CHECK(wt_ledger_create(rank == 0 ? WT_LEDGER_OWNER : WT_LEDGER_WORKER,
      &fixture->contract, 8, max_frame * 16, &fixture->ledger) == WT_LEDGER_OK);
  if (rank == 0) {
    const struct wt_policy_pool_config pool_config = {1, 1, max_frame * 2};
    struct wt_mpi_remote_endpoint bad_endpoint;
    endpoint = (struct wt_mpi_remote_endpoint){peer, 1, 11};
    CHECK(wt_policy_pool_create(&pool_config, &fixture->pool) == WT_POLICY_OK);
    bad_endpoint = endpoint;
    bad_endpoint.mpi_peer = rank;
    CHECK(wt_mpi_remote_owner_create_take(&fixture->channel, &bad_endpoint,
        fixture->ledger, fixture->pool, &fixture->contract, &fixture->owner) ==
        WT_MPI_REMOTE_ENDPOINT_MISMATCH);
    CHECK(fixture->channel != NULL && fixture->owner == NULL);
    CHECK(wt_mpi_remote_owner_create_take(&fixture->channel, &endpoint,
        fixture->ledger, fixture->pool, &fixture->contract, &fixture->owner) ==
        WT_MPI_REMOTE_OK);
  } else {
    struct wt_mpi_remote_endpoint bad_endpoint;
    endpoint = (struct wt_mpi_remote_endpoint){peer, 0, 10};
    bad_endpoint = endpoint;
    bad_endpoint.rank_id = 99;
    CHECK(wt_mpi_remote_worker_create_take(&fixture->channel, &bad_endpoint,
        fixture->ledger, &fixture->contract, run_callback, run,
        &fixture->worker) == WT_MPI_REMOTE_ENDPOINT_MISMATCH);
    CHECK(fixture->channel != NULL && fixture->worker == NULL);
    CHECK(wt_mpi_remote_worker_create_take(&fixture->channel, &endpoint,
        fixture->ledger, &fixture->contract, run_callback, run,
        &fixture->worker) == WT_MPI_REMOTE_OK);
  }
  CHECK(fixture->channel == NULL);
  return 0;
}

static int destroy_fixture(struct fixture *fixture, int rank) {
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
  return 0;
}

static int begin_owner(struct fixture *fixture, const uint8_t *bytes,
    size_t length, size_t result_limit, uint32_t phase, int64_t deadline,
    struct commit_context *commit, struct wt_policy_candidate *candidate) {
  struct wt_policy_config policy = policy_for(&fixture->contract);
  struct wt_policy_request request = policy_request(phase,
      length + result_limit);
  struct wt_mpi_remote_packet packet = {bytes, length, result_limit};
  struct wt_policy_choice choice;
  return wt_mpi_remote_owner_begin(fixture->owner, &policy, &request,
      candidate, 1, &packet, deadline, commit_callback, commit, &choice);
}

static int drive_success(struct fixture *fixture, int rank, int *saw_commit) {
  const int64_t stop = now_ns() + INT64_C(5000000000);
  while (now_ns() < stop) {
    struct wt_mpi_remote_event event;
    enum wt_mpi_remote_status status = rank == 0 ?
        wt_mpi_remote_owner_progress(fixture->owner, &event) :
        wt_mpi_remote_worker_progress(fixture->worker, &event);
    CHECK(status == WT_MPI_REMOTE_OK);
    if (rank == 0 && (event.kinds & WT_MPI_REMOTE_EVENT_OWNER_COMMITTED))
      *saw_commit = 1;
    if ((rank == 0 && *saw_commit &&
        wt_mpi_remote_owner_shutdown_ready(fixture->owner) == WT_MPI_REMOTE_OK) ||
        (rank == 1 && (event.kinds & WT_MPI_REMOTE_EVENT_SEND_COMPLETE) &&
        wt_mpi_remote_worker_shutdown_ready(fixture->worker) == WT_MPI_REMOTE_OK))
      return 0;
    pause_one_ms();
  }
  return 1;
}

static int test_healthy_and_preflight(int rank, int peer) {
  struct fixture fixture;
  struct run_context run;
  struct commit_context commit = {0, 0};
  struct wt_policy_candidate candidate;
  struct wt_policy_pool_stats stats;
  uint8_t frame[512];
  size_t frame_len;
  int saw_commit;
  run = (struct run_context){&fixture.contract, 1, 0, 0};
  CHECK(setup_fixture(&fixture, rank, peer, sizeof(frame), &run) == 0);
  run.contract = &fixture.contract;
  candidate = candidate_for(1, &fixture.contract);
  for (uint32_t sequence = 1; sequence <= 2; ++sequence) {
    uint8_t payload = (uint8_t)(10 + sequence);
    uint32_t phase = sequence == 1 ? UINT32_C(0x10203040) : UINT32_C(0xa1b2c3d4);
    CHECK(make_frame(&fixture.contract, WT_WIRE_REQUEST, phase, sequence,
        (uint8_t)(0x30 + sequence), &payload, 1, frame, sizeof(frame),
        &frame_len) == WT_WIRE_OK);
    if (rank == 1)
      CHECK(wt_mpi_remote_worker_arm(fixture.worker,
          now_ns() + INT64_C(5000000000)) == WT_MPI_REMOTE_OK);
    CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
    if (rank == 0)
      CHECK(begin_owner(&fixture, frame, frame_len, sizeof(frame), phase,
          now_ns() + INT64_C(5000000000), &commit, &candidate) ==
          WT_MPI_REMOTE_OK);
    saw_commit = 0;
    CHECK(drive_success(&fixture, rank, &saw_commit) == 0);
    CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  }
  if (rank == 0) {
    struct wt_policy_config wrong_schema = policy_for(&fixture.contract);
    struct wt_policy_config valid_policy = policy_for(&fixture.contract);
    struct wt_policy_request request = policy_request(UINT32_C(0xa1b2c3d4),
        frame_len + sizeof(frame));
    struct wt_mpi_remote_packet packet = {frame, frame_len, sizeof(frame)};
    struct wt_policy_choice choice;
    CHECK(commit.calls == 2 && commit.value == 13);
    CHECK(begin_owner(&fixture, frame, frame_len, sizeof(frame),
        UINT32_C(0xa1b2c3d4), now_ns() + INT64_C(1000000000), &commit,
        &candidate) == WT_MPI_REMOTE_DUPLICATE);
    {
      uint8_t conflict_payload = 99;
      CHECK(make_frame(&fixture.contract, WT_WIRE_REQUEST,
          UINT32_C(0xa1b2c3d4), 2, 0x32, &conflict_payload, 1, frame,
          sizeof(frame), &frame_len) == WT_WIRE_OK);
      CHECK(begin_owner(&fixture, frame, frame_len, sizeof(frame),
          UINT32_C(0xa1b2c3d4), now_ns() + INT64_C(1000000000), &commit,
          &candidate) == WT_MPI_REMOTE_LEDGER);
      conflict_payload = 12;
      CHECK(make_frame(&fixture.contract, WT_WIRE_REQUEST,
          UINT32_C(0xa1b2c3d4), 2, 0x32, &conflict_payload, 1, frame,
          sizeof(frame), &frame_len) == WT_WIRE_OK);
    }
    candidate.peer.rank_id = 99;
    CHECK(begin_owner(&fixture, frame, frame_len, sizeof(frame),
        UINT32_C(0xa1b2c3d4), now_ns() + INT64_C(1000000000), &commit,
        &candidate) == WT_MPI_REMOTE_ENDPOINT_MISMATCH);
    candidate = candidate_for(1, &fixture.contract);
    wrong_schema.schema_fingerprint[0] ^= 1;
    CHECK(wt_mpi_remote_owner_begin(fixture.owner, &wrong_schema, &request,
        &candidate, 1, &packet, now_ns() + INT64_C(1000000000),
        commit_callback, &commit, &choice) == WT_MPI_REMOTE_POLICY_REFUSED);
    packet.request_len = sizeof(frame) + 1;
    CHECK(wt_mpi_remote_owner_begin(fixture.owner,
        &(struct wt_policy_config){0}, &request, &candidate, 1, &packet,
        now_ns() + INT64_C(1000000000), commit_callback, &commit, &choice) ==
        WT_MPI_REMOTE_BAD_PACKET);
    packet.request_len = frame_len;
    packet.result_limit = sizeof(frame) + 1;
    CHECK(wt_mpi_remote_owner_begin(fixture.owner, &valid_policy,
        &request, &candidate, 1, &packet, now_ns() + INT64_C(1000000000),
        commit_callback, &commit, &choice) == WT_MPI_REMOTE_BAD_PACKET);
    CHECK(wt_policy_pool_get_stats(fixture.pool, &stats) == WT_POLICY_OK);
    CHECK(stats.inflight == 0 && stats.bytes_inflight == 0);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  CHECK(destroy_fixture(&fixture, rank) == 0);
  if (rank == 0) wt_policy_pool_destroy(fixture.pool);
  return 0;
}

static int test_abandon_discards_late_result(int rank, int peer) {
  struct fixture fixture;
  struct run_context run;
  struct commit_context commit = {0, 0};
  struct wt_policy_candidate candidate;
  struct wt_policy_pool_stats stats;
  uint8_t frame[512], payload = 9;
  size_t frame_len;
  int done = 0;
  run = (struct run_context){&fixture.contract, 1, 0, 0};
  CHECK(setup_fixture(&fixture, rank, peer, sizeof(frame), &run) == 0);
  run.contract = &fixture.contract;
  candidate = candidate_for(1, &fixture.contract);
  CHECK(make_frame(&fixture.contract, WT_WIRE_REQUEST, 77, 1, 0x44,
      &payload, 1, frame, sizeof(frame), &frame_len) == WT_WIRE_OK);
  if (rank == 1)
    CHECK(wt_mpi_remote_worker_arm(fixture.worker,
        now_ns() + INT64_C(5000000000)) == WT_MPI_REMOTE_OK);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(begin_owner(&fixture, frame, frame_len, sizeof(frame), 77,
        now_ns() + INT64_C(5000000000), &commit, &candidate) == WT_MPI_REMOTE_OK);
  if (rank == 1) {
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (!done && now_ns() < stop) {
      struct wt_mpi_remote_event event;
      CHECK(wt_mpi_remote_worker_progress(fixture.worker, &event) == WT_MPI_REMOTE_OK);
      done = (event.kinds & WT_MPI_REMOTE_EVENT_SEND_COMPLETE) != 0;
      if (!done) pause_one_ms();
    }
    CHECK(done);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0) {
    struct wt_mpi_remote_event event;
    CHECK(wt_mpi_remote_owner_abandon(fixture.owner, &event) ==
        WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
    CHECK((event.kinds & WT_MPI_REMOTE_EVENT_UNKNOWN_DELIVERY) != 0);
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (wt_mpi_remote_owner_shutdown_ready(fixture.owner) ==
        WT_MPI_REMOTE_BUSY && now_ns() < stop) {
      CHECK(wt_mpi_remote_owner_progress(fixture.owner, &event) ==
          WT_MPI_REMOTE_POISONED);
      pause_one_ms();
    }
    CHECK(now_ns() < stop);
    CHECK(commit.calls == 0);
    CHECK(wt_policy_pool_get_stats(fixture.pool, &stats) == WT_POLICY_OK);
    CHECK(stats.inflight == 1);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  CHECK(destroy_fixture(&fixture, rank) == 0);
  if (rank == 0) {
    CHECK(wt_policy_pool_get_stats(fixture.pool, &stats) == WT_POLICY_OK);
    CHECK(stats.inflight == 0);
    wt_policy_pool_destroy(fixture.pool);
  }
  return 0;
}

static int test_worker_callback_failure(int rank, int peer) {
  struct fixture fixture;
  struct run_context run;
  struct commit_context commit = {0, 0};
  struct wt_policy_candidate candidate;
  struct wt_policy_pool_stats stats;
  uint8_t frame[512], payload = 3;
  size_t frame_len;
  run = (struct run_context){&fixture.contract, 1, 0, 1};
  CHECK(setup_fixture(&fixture, rank, peer, sizeof(frame), &run) == 0);
  run.contract = &fixture.contract;
  candidate = candidate_for(1, &fixture.contract);
  CHECK(make_frame(&fixture.contract, WT_WIRE_REQUEST, 88, 1, 0x55,
      &payload, 1, frame, sizeof(frame), &frame_len) == WT_WIRE_OK);
  if (rank == 1)
    CHECK(wt_mpi_remote_worker_arm(fixture.worker,
        now_ns() + INT64_C(1000000000)) == WT_MPI_REMOTE_OK);
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  if (rank == 0)
    CHECK(begin_owner(&fixture, frame, frame_len, sizeof(frame), 88,
        now_ns() + INT64_C(100000000), &commit, &candidate) == WT_MPI_REMOTE_OK);
  if (rank == 1) {
    enum wt_mpi_remote_status status = WT_MPI_REMOTE_OK;
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (status == WT_MPI_REMOTE_OK && now_ns() < stop) {
      struct wt_mpi_remote_event event;
      status = wt_mpi_remote_worker_progress(fixture.worker, &event);
      if (status == WT_MPI_REMOTE_OK) pause_one_ms();
    }
    CHECK(status == WT_MPI_REMOTE_CALLBACK_FAILED);
    CHECK(run.calls == 1);
    CHECK(wt_mpi_remote_worker_arm(fixture.worker,
        now_ns() + INT64_C(1000000000)) == WT_MPI_REMOTE_POISONED);
  } else {
    enum wt_mpi_remote_status status = WT_MPI_REMOTE_OK;
    const int64_t stop = now_ns() + INT64_C(5000000000);
    while (status == WT_MPI_REMOTE_OK && now_ns() < stop) {
      struct wt_mpi_remote_event event;
      status = wt_mpi_remote_owner_progress(fixture.owner, &event);
      if (status == WT_MPI_REMOTE_OK) pause_one_ms();
    }
    CHECK(status == WT_MPI_REMOTE_TIMEOUT_UNKNOWN_DELIVERY);
    CHECK(commit.calls == 0);
    CHECK(begin_owner(&fixture, frame, frame_len, sizeof(frame), 88,
        now_ns() + INT64_C(1000000000), &commit, &candidate) ==
        WT_MPI_REMOTE_POISONED);
  }
  CHECK(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  CHECK(destroy_fixture(&fixture, rank) == 0);
  if (rank == 0) {
    CHECK(wt_policy_pool_get_stats(fixture.pool, &stats) == WT_POLICY_OK);
    CHECK(stats.inflight == 0);
    wt_policy_pool_destroy(fixture.pool);
  }
  return 0;
}

int main(int argc, char **argv) {
  int provided = MPI_THREAD_SINGLE;
  int rank = -1;
  int size = 0;
  int peer;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
    return 1;
  if (provided < MPI_THREAD_FUNNELED || MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS || size != 2) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  peer = 1 - rank;
  CHECK(test_healthy_and_preflight(rank, peer) == 0);
  CHECK(test_abandon_discards_late_result(rank, peer) == 0);
  CHECK(test_worker_callback_failure(rank, peer) == 0);
  CHECK(MPI_Finalize() == MPI_SUCCESS);
  return 0;
}
