#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../wt_policy.h"

#define CHECK(expr) do { if (!(expr)) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; \
} } while (0)

static struct wt_policy_estimate estimate(double remote) {
  struct wt_policy_estimate e = {0};
  e.sample_count = 4;
  e.pack_ns = 2.0;
  e.validate_commit_ns = 3.0;
  e.queue_ns = 4.0;
  e.remote_exec_ns = remote;
  e.link_latency_ns = 1.0;
  e.link_throughput_bytes_per_ns = 10.0;
  e.outbound_bytes = 10;
  e.inbound_bytes = 10;
  return e;
}

static struct wt_policy_config config(void) {
  struct wt_policy_config c = {0};
  c.enabled = 1;
  c.allowed_scopes = WT_POLICY_SCOPE_SAME_NODE | WT_POLICY_SCOPE_CROSS_NODE;
  memset(c.schema_fingerprint, 0x5a, sizeof(c.schema_fingerprint));
  c.minimum_samples = 3;
  c.safety_margin_ns = 5.0;
  return c;
}

static struct wt_policy_request request(uint32_t scope) {
  struct wt_policy_request r = {0};
  r.phase_id = UINT32_C(0xfedcba98);
  r.origin.rank_id = 9001;
  r.origin.node_id = 42;
  r.origin.numa_id = 300;
  r.origin.topology_known = 1;
  r.scope = scope;
  r.required_worker_slots = 1;
  r.worstcase_bytes = 100;
  r.local_ns = 100.0;
  return r;
}

static struct wt_policy_candidate candidate(uint32_t rank, uint32_t node,
    uint32_t caps, double remote) {
  struct wt_policy_candidate c = {0};
  c.peer.rank_id = rank;
  c.peer.node_id = node;
  c.peer.numa_id = node + 1000;
  c.peer.topology_known = 1;
  memcpy(c.peer.schema_fingerprint, config().schema_fingerprint,
      sizeof(c.peer.schema_fingerprint));
  c.peer.capabilities = caps;
  c.peer.worker_slots = 2;
  c.peer.live_byte_budget = 1000;
  c.estimate = estimate(remote);
  return c;
}

static int test_scope_and_speed(void) {
  struct wt_policy_config c = config();
  struct wt_policy_request r = request(WT_POLICY_SCOPE_SAME_NODE);
  struct wt_policy_candidate peers[] = {
    candidate(101, 42, WT_POLICY_CAP_SAME_NODE, 40.0),
    candidate(7, 99, WT_POLICY_CAP_CROSS_NODE, 15.0),
  };
  struct wt_policy_choice out;
  CHECK(wt_policy_choose(&c, &r, peers, 2, &out) == WT_POLICY_OK);
  CHECK(out.peer_rank_id == 101 && out.scope == WT_POLICY_SCOPE_SAME_NODE);
  r.scope = WT_POLICY_SCOPE_CROSS_NODE;
  CHECK(wt_policy_choose(&c, &r, peers, 2, &out) == WT_POLICY_OK);
  CHECK(out.peer_rank_id == 7 && out.end_to_end_gain_ns > c.safety_margin_ns);
  peers[0].estimate.remote_exec_ns = 10.0;
  r.scope = WT_POLICY_SCOPE_SAME_NODE;
  CHECK(wt_policy_choose(&c, &r, peers, 2, &out) == WT_POLICY_OK);
  CHECK(out.peer_rank_id == 101);
  return 0;
}

static int test_cost_oracle(void) {
  struct wt_policy_config c = config();
  struct wt_policy_request r = request(WT_POLICY_SCOPE_SAME_NODE);
  struct wt_policy_candidate peer = candidate(101, 42, WT_POLICY_CAP_SAME_NODE, 40.0);
  struct wt_policy_choice out;
  /* 2 pack + 3 commit + 4 queue + 40 remote + (1 + 10/10) out
   * + (1 + 10/10) in = 53; local baseline 100 leaves gain 47. */
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);
  CHECK(out.end_to_end_ns == 53.0 && out.end_to_end_gain_ns == 47.0);
  peer.estimate.pack_ns = 9.0;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);
  CHECK(out.end_to_end_ns == 60.0);
  peer.estimate = estimate(40.0);
  peer.estimate.validate_commit_ns = 14.0;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);
  CHECK(out.end_to_end_ns == 64.0);
  peer.estimate = estimate(40.0);
  peer.estimate.queue_ns = 17.0;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);
  CHECK(out.end_to_end_ns == 66.0);
  peer.estimate = estimate(57.0);
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);
  CHECK(out.end_to_end_ns == 70.0);
  peer.estimate = estimate(40.0);
  peer.estimate.link_latency_ns = 2.0;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);
  CHECK(out.end_to_end_ns == 55.0);
  peer.estimate = estimate(40.0);
  peer.estimate.outbound_bytes = 20;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);
  CHECK(out.end_to_end_ns == 54.0);
  peer.estimate = estimate(40.0);
  peer.estimate.inbound_bytes = 20;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);
  CHECK(out.end_to_end_ns == 54.0);
  peer = candidate(101, 42, WT_POLICY_CAP_SAME_NODE, 82.0);
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_NO_POSITIVE_GAIN);
  return 0;
}

static int test_rejections(void) {
  struct wt_policy_config c = config();
  struct wt_policy_request r = request(WT_POLICY_SCOPE_SAME_NODE);
  struct wt_policy_candidate good = candidate(17, 42,
      WT_POLICY_CAP_SAME_NODE, 10.0);
  struct wt_policy_choice out;
  CHECK(wt_policy_choose(&c, &r, NULL, 0, &out) == WT_POLICY_NO_ELIGIBLE_PEER);
  memset(c.schema_fingerprint, 0, sizeof(c.schema_fingerprint));
  memset(good.peer.schema_fingerprint, 0, sizeof(good.peer.schema_fingerprint));
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_SCHEMA_MISMATCH);
  c = config();
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_SCHEMA_MISMATCH);
  memcpy(good.peer.schema_fingerprint, c.schema_fingerprint,
      sizeof(good.peer.schema_fingerprint));
  good.peer.schema_fingerprint[31] ^= 1;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_SCHEMA_MISMATCH);
  memcpy(good.peer.schema_fingerprint, c.schema_fingerprint,
      sizeof(good.peer.schema_fingerprint));
  good.peer.capabilities = 0;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_UNSUPPORTED_SCOPE);
  good.peer.capabilities = WT_POLICY_CAP_SAME_NODE;
  good.estimate.remote_exec_ns = 100.0;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_NO_POSITIVE_GAIN);
  good.estimate = estimate(10.0);
  good.estimate.sample_count = 2;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_NO_SAMPLES);
  good.estimate.sample_count = 4;
  r.local_ns = NAN;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_BAD_ESTIMATE);
  r.local_ns = INFINITY;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_BAD_ESTIMATE);
  r.local_ns = -1.0;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_BAD_ESTIMATE);
  r.local_ns = 100.0;
  good.estimate = estimate(10.0);
  good.estimate.outbound_bytes = UINT64_MAX;
  good.estimate.link_throughput_bytes_per_ns = 1e-300;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_BAD_ESTIMATE);
  good.estimate = estimate(10.0);
  r.origin.topology_known = 0;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_UNKNOWN_TOPOLOGY);
  r.origin.topology_known = 1;
  r.required_worker_slots = 0;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_BAD_ARGUMENT);
  r.required_worker_slots = 1;
  r.worstcase_bytes = 0;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_BAD_ARGUMENT);
  r.worstcase_bytes = 100;
  struct wt_policy_candidate duplicate[2] = {good, good};
  CHECK(wt_policy_choose(&c, &r, duplicate, 2, &out) == WT_POLICY_INVALID_TOPOLOGY);
  c.enabled = 0;
  CHECK(wt_policy_choose(&c, &r, &good, 1, &out) == WT_POLICY_DISABLED);
  return 0;
}

static int test_credits_and_pool(void) {
  struct wt_policy_config c = config();
  struct wt_policy_request r = request(WT_POLICY_SCOPE_SAME_NODE);
  struct wt_policy_candidate peer = candidate(17, 42,
      WT_POLICY_CAP_SAME_NODE, 10.0);
  struct wt_policy_choice out;
  peer.peer.worker_slots = 0;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_INSUFFICIENT_CREDITS);
  peer.peer.worker_slots = 1;
  peer.peer.live_byte_budget = 99;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_INSUFFICIENT_CREDITS);
  peer.peer.live_byte_budget = 100;
  CHECK(wt_policy_choose(&c, &r, &peer, 1, &out) == WT_POLICY_OK);

  struct wt_policy_pool_config pc = {2, 2, 100};
  struct wt_policy_pool *pool = NULL;
  struct wt_policy_pool *foreign_pool = NULL;
  struct wt_policy_ticket a, b;
  struct wt_policy_pool_stats stats;
  CHECK(wt_policy_pool_create(&pc, &pool) == WT_POLICY_OK);
  CHECK(wt_policy_pool_create(&pc, &foreign_pool) == WT_POLICY_OK);
  CHECK(wt_policy_pool_reserve(pool, 0, 1, &a) == WT_POLICY_BAD_ARGUMENT);
  CHECK(wt_policy_pool_reserve(pool, 1, 0, &a) == WT_POLICY_BAD_ARGUMENT);
  CHECK(wt_policy_pool_reserve(pool, 1, 60, &a) == WT_POLICY_OK);
  CHECK(wt_policy_pool_get_stats(pool, &stats) == WT_POLICY_OK);
  CHECK(stats.inflight == 1 && stats.bytes_inflight == 60 && stats.worker_slots_inflight == 1);
  CHECK(wt_policy_pool_reserve(pool, 1, 41, &b) == WT_POLICY_INSUFFICIENT_CREDITS);
  CHECK(wt_policy_pool_reserve(pool, 1, 40, &b) == WT_POLICY_OK);
  CHECK(wt_policy_pool_reserve(pool, 1, 0, &b) == WT_POLICY_BAD_ARGUMENT);
  struct wt_policy_ticket forged = b;
  forged.worstcase_bytes = 0;
  CHECK(wt_policy_pool_release(pool, &forged) != WT_POLICY_OK);
  forged = b;
  forged.worker_slots = 0;
  CHECK(wt_policy_pool_release(pool, &forged) != WT_POLICY_OK);
  forged = b;
  forged.slot = 0;
  CHECK(wt_policy_pool_release(pool, &forged) != WT_POLICY_OK);
  forged = b;
  forged.generation = 0;
  CHECK(wt_policy_pool_release(pool, &forged) != WT_POLICY_OK);
  CHECK(wt_policy_pool_release(foreign_pool, &b) == WT_POLICY_STALE_TICKET);
  CHECK(wt_policy_pool_get_stats(pool, &stats) == WT_POLICY_OK);
  CHECK(stats.inflight == 2 && stats.bytes_inflight == 100 && stats.worker_slots_inflight == 2);
  CHECK(wt_policy_pool_release(pool, &a) == WT_POLICY_OK);
  CHECK(wt_policy_pool_release(pool, &a) == WT_POLICY_STALE_TICKET);
  CHECK(wt_policy_pool_release(pool, &b) == WT_POLICY_OK);
  for (uint32_t i = 0; i < 32; ++i) {
    CHECK(wt_policy_pool_reserve(pool, 1, 100, &a) == WT_POLICY_OK);
    CHECK(wt_policy_pool_release(pool, &a) == WT_POLICY_OK);
  }
  CHECK(wt_policy_pool_get_stats(pool, &stats) == WT_POLICY_OK);
  CHECK(stats.inflight == 0 && stats.bytes_inflight == 0 && stats.worker_slots_inflight == 0);
  wt_policy_pool_destroy(pool);
  wt_policy_pool_destroy(foreign_pool);
  return 0;
}

int main(void) {
  return test_scope_and_speed() || test_cost_oracle() || test_rejections() ||
      test_credits_and_pool();
}
