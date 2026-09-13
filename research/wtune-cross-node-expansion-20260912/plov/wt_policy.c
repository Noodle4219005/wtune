#include "wt_policy.h"

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

struct wt_policy_reservation {
  uint64_t generation;
  uint64_t worstcase_bytes;
  uint32_t worker_slots;
};

struct wt_policy_pool {
  struct wt_policy_pool_config config;
  uint64_t *owners;
  struct wt_policy_reservation *reservations;
  uint64_t identity;
  uint64_t next_generation;
  struct wt_policy_pool_stats stats;
};

static _Atomic(uint64_t) next_pool_identity = UINT64_C(1);

static int known_schema(const uint8_t fingerprint[32]) {
  size_t i;
  for (i = 0; i < 32; ++i) {
    if (fingerprint[i] != 0) return 1;
  }
  return 0;
}

static enum wt_policy_status claim_pool_identity(uint64_t *out) {
  uint64_t expected = atomic_load_explicit(&next_pool_identity,
      memory_order_relaxed);
  for (;;) {
    if (expected == 0 || expected == UINT64_MAX) return WT_POLICY_OVERFLOW;
    if (atomic_compare_exchange_weak_explicit(&next_pool_identity, &expected,
        expected + 1, memory_order_relaxed, memory_order_relaxed)) {
      *out = expected;
      return WT_POLICY_OK;
    }
  }
}

static int finite_nonnegative(double value) {
  return isfinite(value) && value >= 0.0;
}

static int valid_estimate(const struct wt_policy_estimate *estimate) {
  return estimate != NULL && finite_nonnegative(estimate->pack_ns) &&
      finite_nonnegative(estimate->validate_commit_ns) &&
      finite_nonnegative(estimate->queue_ns) &&
      finite_nonnegative(estimate->remote_exec_ns) &&
      finite_nonnegative(estimate->link_latency_ns) &&
      isfinite(estimate->link_throughput_bytes_per_ns) &&
      estimate->link_throughput_bytes_per_ns > 0.0;
}

static int add_time(double *total, double part) {
  *total += part;
  return isfinite(*total);
}

static int candidate_total(const struct wt_policy_estimate *estimate,
    double *out_total) {
  double outbound;
  double inbound;
  double total = 0.0;
  if (!valid_estimate(estimate)) {
    return 0;
  }
  outbound = estimate->link_latency_ns +
      ((double)estimate->outbound_bytes / estimate->link_throughput_bytes_per_ns);
  inbound = estimate->link_latency_ns +
      ((double)estimate->inbound_bytes / estimate->link_throughput_bytes_per_ns);
  if (!isfinite(outbound) || !isfinite(inbound) ||
      !add_time(&total, estimate->pack_ns) ||
      !add_time(&total, estimate->validate_commit_ns) ||
      !add_time(&total, estimate->queue_ns) ||
      !add_time(&total, estimate->remote_exec_ns) ||
      !add_time(&total, outbound) || !add_time(&total, inbound)) {
    return 0;
  }
  *out_total = total;
  return 1;
}

static enum wt_policy_status initialize_choice(
    const struct wt_policy_config *config, const struct wt_policy_request *request,
    struct wt_policy_choice *out) {
  if (config == NULL || request == NULL || out == NULL ||
      !finite_nonnegative(config->safety_margin_ns) ||
      config->minimum_samples == 0 ||
      request->required_worker_slots == 0 || request->worstcase_bytes == 0 ||
      (request->scope != WT_POLICY_SCOPE_SAME_NODE &&
       request->scope != WT_POLICY_SCOPE_CROSS_NODE)) {
    return WT_POLICY_BAD_ARGUMENT;
  }
  if (!finite_nonnegative(request->local_ns)) return WT_POLICY_BAD_ESTIMATE;
  *out = (struct wt_policy_choice){0};
  out->phase_id = request->phase_id;
  out->scope = request->scope;
  if (!known_schema(config->schema_fingerprint)) {
    out->reason = WT_POLICY_SCHEMA_MISMATCH;
    return WT_POLICY_SCHEMA_MISMATCH;
  }
  if (!config->enabled) {
    out->reason = WT_POLICY_DISABLED;
    return WT_POLICY_DISABLED;
  }
  if ((config->allowed_scopes & request->scope) == 0) {
    out->reason = WT_POLICY_UNSUPPORTED_SCOPE;
    return WT_POLICY_UNSUPPORTED_SCOPE;
  }
  if (!request->origin.topology_known) {
    out->reason = WT_POLICY_UNKNOWN_TOPOLOGY;
    return WT_POLICY_UNKNOWN_TOPOLOGY;
  }
  return WT_POLICY_OK;
}

static int duplicate_ranks(const struct wt_policy_candidate *candidates,
    size_t candidate_count) {
  size_t i;
  size_t j;
  for (i = 0; i < candidate_count; ++i) {
    for (j = i + 1; j < candidate_count; ++j) {
      if (candidates[i].peer.rank_id == candidates[j].peer.rank_id) {
        return 1;
      }
    }
  }
  return 0;
}

static int scope_matches(const struct wt_policy_request *request,
    const struct wt_policy_peer *peer) {
  if (request->scope == WT_POLICY_SCOPE_SAME_NODE) {
    return request->origin.node_id == peer->node_id;
  }
  return request->origin.node_id != peer->node_id;
}

static int wins_tie(const struct wt_policy_peer *candidate,
    const struct wt_policy_choice *best) {
  if (candidate->rank_id != best->peer_rank_id) {
    return candidate->rank_id < best->peer_rank_id;
  }
  if (candidate->node_id != best->peer_node_id) {
    return candidate->node_id < best->peer_node_id;
  }
  return candidate->numa_id < best->peer_numa_id;
}

enum wt_policy_status wt_policy_choose(const struct wt_policy_config *config,
    const struct wt_policy_request *request,
    const struct wt_policy_candidate *candidates, size_t candidate_count,
    struct wt_policy_choice *out) {
  enum wt_policy_status status = initialize_choice(config, request, out);
  size_t i;
  int found = 0;
  int saw_unknown = 0;
  int saw_schema = 0;
  int saw_support = 0;
  int saw_credits = 0;
  int saw_samples = 0;
  int saw_gain = 0;
  if (status != WT_POLICY_OK) {
    return status;
  }
  if (candidate_count == 0) {
    out->reason = WT_POLICY_NO_ELIGIBLE_PEER;
    return WT_POLICY_NO_ELIGIBLE_PEER;
  }
  if (candidates == NULL) {
    out->reason = WT_POLICY_BAD_ARGUMENT;
    return WT_POLICY_BAD_ARGUMENT;
  }
  if (duplicate_ranks(candidates, candidate_count)) {
    out->reason = WT_POLICY_INVALID_TOPOLOGY;
    return WT_POLICY_INVALID_TOPOLOGY;
  }
  for (i = 0; i < candidate_count; ++i) {
    const struct wt_policy_candidate *candidate = &candidates[i];
    uint32_t needed_cap = request->scope == WT_POLICY_SCOPE_SAME_NODE ?
        WT_POLICY_CAP_SAME_NODE : WT_POLICY_CAP_CROSS_NODE;
    double total;
    double gain;
    if (!candidate->peer.topology_known) {
      saw_unknown = 1;
      continue;
    }
    if (!scope_matches(request, &candidate->peer)) {
      continue;
    }
    if (candidate->peer.rank_id == request->origin.rank_id) {
      continue;
    }
    if (!known_schema(candidate->peer.schema_fingerprint) ||
        memcmp(candidate->peer.schema_fingerprint, config->schema_fingerprint,
        sizeof(config->schema_fingerprint)) != 0) {
      saw_schema = 1;
      continue;
    }
    if ((candidate->peer.capabilities & needed_cap) == 0) {
      saw_support = 1;
      continue;
    }
    if (candidate->peer.worker_slots < request->required_worker_slots ||
        candidate->peer.live_byte_budget < request->worstcase_bytes) {
      saw_credits = 1;
      continue;
    }
    if (candidate->estimate.sample_count < config->minimum_samples) {
      saw_samples = 1;
      continue;
    }
    if (!candidate_total(&candidate->estimate, &total)) {
      out->reason = WT_POLICY_BAD_ESTIMATE;
      return WT_POLICY_BAD_ESTIMATE;
    }
    gain = request->local_ns - total;
    if (!isfinite(gain)) {
      out->reason = WT_POLICY_BAD_ESTIMATE;
      return WT_POLICY_BAD_ESTIMATE;
    }
    if (gain <= config->safety_margin_ns) {
      saw_gain = 1;
      continue;
    }
    if (!found || gain > out->end_to_end_gain_ns ||
        (gain == out->end_to_end_gain_ns && wins_tie(&candidate->peer, out))) {
      found = 1;
      out->peer_rank_id = candidate->peer.rank_id;
      out->peer_node_id = candidate->peer.node_id;
      out->peer_numa_id = candidate->peer.numa_id;
      out->end_to_end_ns = total;
      out->end_to_end_gain_ns = gain;
    }
  }
  if (found) {
    out->reason = WT_POLICY_OK;
    return WT_POLICY_OK;
  }
  if (saw_unknown) status = WT_POLICY_UNKNOWN_TOPOLOGY;
  else if (saw_schema) status = WT_POLICY_SCHEMA_MISMATCH;
  else if (saw_support) status = WT_POLICY_UNSUPPORTED_SCOPE;
  else if (saw_credits) status = WT_POLICY_INSUFFICIENT_CREDITS;
  else if (saw_samples) status = WT_POLICY_NO_SAMPLES;
  else if (saw_gain) status = WT_POLICY_NO_POSITIVE_GAIN;
  else status = WT_POLICY_NO_ELIGIBLE_PEER;
  out->reason = status;
  return status;
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *out) {
  if (UINT64_MAX - left < right) return 0;
  *out = left + right;
  return 1;
}

enum wt_policy_status wt_policy_pool_create(
    const struct wt_policy_pool_config *config, struct wt_policy_pool **out) {
  struct wt_policy_pool *pool;
  size_t slot_count;
  if (out == NULL || config == NULL || config->worker_slots == 0 ||
      config->max_inflight == 0) {
    return WT_POLICY_BAD_ARGUMENT;
  }
  slot_count = (size_t)config->worker_slots;
  if (slot_count > SIZE_MAX / sizeof(*pool->owners)) return WT_POLICY_OVERFLOW;
  *out = NULL;
  pool = calloc(1, sizeof(*pool));
  if (pool == NULL) return WT_POLICY_NOMEM;
  pool->owners = calloc(slot_count, sizeof(*pool->owners));
  if (pool->owners == NULL) {
    free(pool);
    return WT_POLICY_NOMEM;
  }
  pool->reservations = calloc(slot_count, sizeof(*pool->reservations));
  if (pool->reservations == NULL) {
    free(pool->owners);
    free(pool);
    return WT_POLICY_NOMEM;
  }
  pool->config = *config;
  if (claim_pool_identity(&pool->identity) != WT_POLICY_OK) {
    free(pool->reservations);
    free(pool->owners);
    free(pool);
    return WT_POLICY_OVERFLOW;
  }
  pool->next_generation = 1;
  *out = pool;
  return WT_POLICY_OK;
}

enum wt_policy_status wt_policy_pool_reserve(struct wt_policy_pool *pool,
    uint32_t worker_slots, uint64_t worstcase_bytes,
    struct wt_policy_ticket *out) {
  uint64_t bytes_after;
  uint32_t slot;
  uint32_t i;
  if (pool == NULL || out == NULL || worker_slots == 0 || worstcase_bytes == 0 ||
      worker_slots > pool->config.worker_slots) return WT_POLICY_BAD_ARGUMENT;
  if (pool->stats.inflight >= pool->config.max_inflight ||
      !add_u64(pool->stats.bytes_inflight, worstcase_bytes, &bytes_after) ||
      bytes_after > pool->config.byte_budget ||
      pool->config.worker_slots - pool->stats.worker_slots_inflight < worker_slots) {
    return WT_POLICY_INSUFFICIENT_CREDITS;
  }
  if (pool->next_generation == UINT64_MAX) return WT_POLICY_OVERFLOW;
  for (slot = 0; slot <= pool->config.worker_slots - worker_slots; ++slot) {
    for (i = 0; i < worker_slots && pool->owners[slot + i] == 0; ++i) { }
    if (i == worker_slots) break;
  }
  if (slot > pool->config.worker_slots - worker_slots) return WT_POLICY_INSUFFICIENT_CREDITS;
  for (i = 0; i < worker_slots; ++i) pool->owners[slot + i] = pool->next_generation;
  pool->reservations[slot] = (struct wt_policy_reservation){
      pool->next_generation, worstcase_bytes, worker_slots};
  *out = (struct wt_policy_ticket){
      .slot = slot,
      .worker_slots = worker_slots,
      .pool_identity = pool->identity,
      .generation = pool->next_generation,
      .worstcase_bytes = worstcase_bytes};
  ++pool->next_generation;
  ++pool->stats.inflight;
  pool->stats.worker_slots_inflight += worker_slots;
  pool->stats.bytes_inflight = bytes_after;
  return WT_POLICY_OK;
}

enum wt_policy_status wt_policy_pool_release(struct wt_policy_pool *pool,
    const struct wt_policy_ticket *ticket) {
  uint32_t i;
  if (pool == NULL || ticket == NULL || ticket->worker_slots == 0 ||
      ticket->slot >= pool->config.worker_slots ||
      ticket->worker_slots > pool->config.worker_slots - ticket->slot) {
    return WT_POLICY_BAD_ARGUMENT;
  }
  if (ticket->pool_identity != pool->identity || ticket->generation == 0 ||
      pool->stats.inflight == 0 || pool->stats.worker_slots_inflight < ticket->worker_slots ||
      pool->stats.bytes_inflight < ticket->worstcase_bytes) return WT_POLICY_STALE_TICKET;
  if (pool->reservations[ticket->slot].generation != ticket->generation ||
      pool->reservations[ticket->slot].worker_slots != ticket->worker_slots ||
      pool->reservations[ticket->slot].worstcase_bytes != ticket->worstcase_bytes) {
    return WT_POLICY_STALE_TICKET;
  }
  for (i = 0; i < ticket->worker_slots; ++i) {
    if (pool->owners[ticket->slot + i] != ticket->generation) return WT_POLICY_STALE_TICKET;
  }
  for (i = 0; i < ticket->worker_slots; ++i) pool->owners[ticket->slot + i] = 0;
  pool->reservations[ticket->slot] = (struct wt_policy_reservation){0};
  --pool->stats.inflight;
  pool->stats.worker_slots_inflight -= ticket->worker_slots;
  pool->stats.bytes_inflight -= ticket->worstcase_bytes;
  return WT_POLICY_OK;
}

enum wt_policy_status wt_policy_pool_get_stats(const struct wt_policy_pool *pool,
    struct wt_policy_pool_stats *out) {
  if (pool == NULL || out == NULL) return WT_POLICY_BAD_ARGUMENT;
  *out = pool->stats;
  return WT_POLICY_OK;
}

void wt_policy_pool_destroy(struct wt_policy_pool *pool) {
  if (pool != NULL) {
    free(pool->owners);
    free(pool->reservations);
    free(pool);
  }
}
