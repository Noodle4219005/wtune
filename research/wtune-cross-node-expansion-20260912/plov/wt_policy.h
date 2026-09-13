#ifndef WT_POLICY_H
#define WT_POLICY_H

#include <stddef.h>
#include <stdint.h>

/* Portable, single-caller policy core.  It neither discovers topology nor
 * allocates payloads: every topology, measurement, capacity and worst-case
 * byte value is a caller-supplied contract.  All identifiers are opaque. */
enum wt_policy_status {
  WT_POLICY_OK = 0,
  WT_POLICY_DISABLED,
  WT_POLICY_BAD_ARGUMENT,
  WT_POLICY_BAD_ESTIMATE,
  WT_POLICY_NO_SAMPLES,
  WT_POLICY_SCHEMA_MISMATCH,
  WT_POLICY_UNSUPPORTED_SCOPE,
  WT_POLICY_INSUFFICIENT_CREDITS,
  WT_POLICY_UNKNOWN_TOPOLOGY,
  WT_POLICY_INVALID_TOPOLOGY,
  WT_POLICY_NO_POSITIVE_GAIN,
  WT_POLICY_NO_ELIGIBLE_PEER,
  WT_POLICY_STALE_TICKET,
  WT_POLICY_NOMEM,
  WT_POLICY_OVERFLOW
};

enum wt_policy_scope {
  WT_POLICY_SCOPE_SAME_NODE = UINT32_C(1),
  WT_POLICY_SCOPE_CROSS_NODE = UINT32_C(2)
};

enum wt_policy_capability {
  WT_POLICY_CAP_SAME_NODE = UINT32_C(1),
  WT_POLICY_CAP_CROSS_NODE = UINT32_C(2)
};

struct wt_policy_topology {
  uint32_t rank_id;
  uint32_t node_id;
  uint32_t numa_id;
  uint32_t topology_known;
};

struct wt_policy_peer {
  uint32_t rank_id;
  uint32_t node_id;
  uint32_t numa_id;
  uint32_t topology_known;
  uint8_t schema_fingerprint[32];
  uint32_t capabilities;
  uint32_t worker_slots;
  uint64_t live_byte_budget;
};

/* Each time is in ns and throughput is bytes/ns.  outbound_bytes and
 * inbound_bytes are independently charged, each with link latency. */
struct wt_policy_estimate {
  uint32_t sample_count;
  double pack_ns;
  double validate_commit_ns;
  double queue_ns;
  double remote_exec_ns;
  double link_latency_ns;
  double link_throughput_bytes_per_ns;
  uint64_t outbound_bytes;
  uint64_t inbound_bytes;
};

struct wt_policy_candidate {
  struct wt_policy_peer peer;
  struct wt_policy_estimate estimate;
};

struct wt_policy_config {
  uint32_t enabled;
  uint32_t allowed_scopes;
  uint8_t schema_fingerprint[32];
  uint32_t minimum_samples;
  double safety_margin_ns;
};

struct wt_policy_request {
  uint32_t phase_id;
  struct wt_policy_topology origin;
  uint32_t scope;
  uint32_t required_worker_slots;
  uint64_t worstcase_bytes;
  double local_ns;
};

struct wt_policy_choice {
  uint32_t phase_id;
  uint32_t peer_rank_id;
  uint32_t peer_node_id;
  uint32_t peer_numa_id;
  uint32_t scope;
  enum wt_policy_status reason;
  double end_to_end_ns;
  double end_to_end_gain_ns;
};

/* Chooses the greatest strictly positive gain above safety_margin_ns.  Ties
 * break by peer rank, node, then NUMA ID, independent of input order. */
enum wt_policy_status wt_policy_choose(const struct wt_policy_config *config,
    const struct wt_policy_request *request,
    const struct wt_policy_candidate *candidates, size_t candidate_count,
    struct wt_policy_choice *out);

/* byte_budget is the explicit sum of all worst-case snapshots, clones,
 * results, guards and caller metadata.  It is accounting capacity, not an
 * allocator estimate.  Caller serializes all operations on a pool. */
struct wt_policy_pool_config {
  uint32_t worker_slots;
  uint32_t max_inflight;
  uint64_t byte_budget;
};

struct wt_policy_ticket {
  uint32_t slot;
  uint32_t worker_slots;
  uint64_t pool_identity;
  uint64_t generation;
  uint64_t worstcase_bytes;
};

struct wt_policy_pool_stats {
  uint32_t inflight;
  uint32_t worker_slots_inflight;
  uint64_t bytes_inflight;
};

struct wt_policy_pool;

enum wt_policy_status wt_policy_pool_create(
    const struct wt_policy_pool_config *config, struct wt_policy_pool **out);
enum wt_policy_status wt_policy_pool_reserve(struct wt_policy_pool *pool,
    uint32_t worker_slots, uint64_t worstcase_bytes,
    struct wt_policy_ticket *out);
enum wt_policy_status wt_policy_pool_release(struct wt_policy_pool *pool,
    const struct wt_policy_ticket *ticket);
enum wt_policy_status wt_policy_pool_get_stats(const struct wt_policy_pool *pool,
    struct wt_policy_pool_stats *out);
void wt_policy_pool_destroy(struct wt_policy_pool *pool);

#endif
