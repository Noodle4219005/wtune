#ifndef WT_NODE_LOCAL_H
#define WT_NODE_LOCAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WT_NL_VERSION 1u

enum wt_nl_status {
    WT_NL_OK = 0,
    WT_NL_IDLE = 1,
    WT_NL_PENDING = 2,
    WT_NL_BUSY = 3,
    WT_NL_INVALID = -1,
    WT_NL_OVERFLOW = -2,
    WT_NL_SYSTEM = -3,
    WT_NL_UNSUPPORTED = -4,
    WT_NL_STALE = -5,
    WT_NL_OWNER = -6,
    WT_NL_FAILED = -7,
    WT_NL_REJECTED = -8
};

enum wt_nl_slot_state {
    WT_NL_SLOT_EMPTY = 0,
    WT_NL_SLOT_PUBLISHED = 1,
    WT_NL_SLOT_CLAIMED_SELF = 2,
    WT_NL_SLOT_CLAIMED_HELPER = 3,
    WT_NL_SLOT_DONE = 4,
    WT_NL_SLOT_VALIDATING = 5,
    WT_NL_SLOT_COMMITTED = 6,
    WT_NL_SLOT_FAILED = 7,
    WT_NL_SLOT_RESERVED = 8,
    WT_NL_SLOT_FAILING = 9
};

enum wt_nl_execute_result {
    WT_NL_EXECUTE_OK = 0,
    WT_NL_EXECUTE_REJECTED = 1,
    WT_NL_EXECUTE_FATAL = -1
};

/* Signature identity is deliberately separate from owner identity.  Peers may
 * execute a request only when every signature field is equal.  Geometry is an
 * adapter-owned byte string and may differ when the adapter explicitly accepts
 * it in its compatibility callback. */
struct wt_nl_signature {
    uint64_t phase;
    uint64_t profile;
    uint64_t schema;
    uint64_t grid;
    uint64_t epoch;
    uint64_t branch;
    uint64_t init_generation;
};

struct wt_nl_descriptor {
    struct wt_nl_signature signature;
    uint64_t call_sequence;
    uint32_t geometry_bytes;
    uint32_t reserved;
    size_t request_bytes;
    size_t result_bytes;
};

struct wt_nl_config {
    uint32_t max_ranks;
    uint32_t max_threads;
    size_t slot_payload_bytes;
};

struct wt_nl_ticket {
    uint64_t work_id;
    uint32_t slot;
    uint32_t owner_rank;
    uint32_t owner_thread;
};

struct wt_nl_counters {
    uint64_t staged;
    uint64_t self_executed;
    uint64_t helper_executed;
    uint64_t committed;
    uint64_t failed;
};

/* Public handle contains no pointer into an owner's private address space.  It
 * may be copied only within one process; call detach exactly once for an
 * fd-backed handle.  Before detach or storage reuse, the caller must join or
 * otherwise quiesce every thread/process that can be inside a service callback.
 * Poison prevents commit and replay; it does not cancel a running callback. */
struct wt_nl_service {
    void *mapping;
    size_t mapping_bytes;
    int mapped_by_service;
};

typedef int (*wt_nl_geometry_compatible_fn)(
    void *context, const struct wt_nl_descriptor *descriptor,
    const void *geometry, size_t geometry_bytes);

typedef int (*wt_nl_execute_fn)(
    void *context, const struct wt_nl_descriptor *descriptor,
    const void *geometry, size_t geometry_bytes,
    const void *request, size_t request_bytes,
    void *result, size_t result_capacity, size_t *result_written);
/* The execute callback returns WT_NL_EXECUTE_OK, a positive value when it
 * rejected before entering the native helper, or a negative value after native
 * entry.  Both failures poison the work and prohibit replay; collect reports
 * WT_NL_REJECTED for the former and WT_NL_FAILED for the latter. */

/* validate must complete every adapter-specific precommit check and must not
 * write owner data.  commit is called once only after core and adapter checks
 * pass; it is intentionally void because an owner write cannot be rolled back. */
typedef int (*wt_nl_owner_validate_fn)(
    void *context, const struct wt_nl_descriptor *descriptor,
    const void *geometry, size_t geometry_bytes,
    const void *request, size_t request_bytes,
    const void *result, size_t result_bytes);
/* Validation returns 1 only when every adapter check passes; 0 rejects. */
typedef void (*wt_nl_owner_commit_fn)(
    void *context, const struct wt_nl_descriptor *descriptor,
    const void *geometry, size_t geometry_bytes,
    const void *request, size_t request_bytes,
    const void *result, size_t result_bytes);

enum wt_nl_status wt_nl_required_bytes(
    const struct wt_nl_config *config, size_t *out);
/* init_mapping/init_fd have one initializer: no peer may attach until they
 * return WT_NL_OK.  The release/acquire READY publication then makes the fixed
 * layout visible to all peers. */
enum wt_nl_status wt_nl_init_mapping(
    void *mapping, size_t mapping_bytes, const struct wt_nl_config *config,
    struct wt_nl_service *out);
enum wt_nl_status wt_nl_attach_mapping(
    void *mapping, size_t mapping_bytes, struct wt_nl_service *out);
enum wt_nl_status wt_nl_init_fd(
    int fd, const struct wt_nl_config *config, struct wt_nl_service *out);
enum wt_nl_status wt_nl_attach_fd(
    int fd, size_t mapping_bytes, struct wt_nl_service *out);
void wt_nl_detach(struct wt_nl_service *service);

enum wt_nl_status wt_nl_slot_index(
    const struct wt_nl_service *service, uint32_t rank_slot,
    uint32_t thread_slot, uint32_t *slot);

enum wt_nl_status wt_nl_publish(
    struct wt_nl_service *service, uint32_t owner_rank,
    uint32_t owner_thread, const struct wt_nl_descriptor *descriptor,
    const void *geometry, const void *request, struct wt_nl_ticket *ticket);

/* Scans the bounded slot array once.  Slots owned by helper_rank are skipped,
 * so helper_executed always denotes another local rank, not another thread. */
enum wt_nl_status wt_nl_try_execute(
    struct wt_nl_service *service, uint32_t helper_rank,
    uint32_t helper_thread, const struct wt_nl_signature *required_signature,
    wt_nl_geometry_compatible_fn compatible, wt_nl_execute_fn execute,
    void *context, uint64_t executor_tag, struct wt_nl_ticket *executed);

/* The adapter must serialize publish and all owner_* / finish calls for one
 * owner slot.  Helpers may concurrently claim that slot.  This single-owner
 * rule keeps ticket validation and slot recycling in one ordered control flow. */
enum wt_nl_status wt_nl_owner_execute(
    struct wt_nl_service *service, uint32_t caller_rank,
    uint32_t caller_thread, const struct wt_nl_ticket *ticket,
    wt_nl_execute_fn execute, void *context, uint64_t executor_tag);
enum wt_nl_status wt_nl_owner_collect(
    struct wt_nl_service *service, uint32_t caller_rank,
    uint32_t caller_thread, const struct wt_nl_ticket *ticket,
    wt_nl_owner_validate_fn validate, wt_nl_owner_commit_fn commit,
    void *context);
/* Returns the recorded executor tag only for the current owner's exact ticket
 * after execution completed (DONE) or commit completed (COMMITTED). The owner
 * must serialize this query with its other calls for the slot. */
enum wt_nl_status wt_nl_ticket_executor(
    const struct wt_nl_service *service, uint32_t caller_rank,
    uint32_t caller_thread, const struct wt_nl_ticket *ticket,
    uint64_t *executor_tag);
enum wt_nl_status wt_nl_owner_poison(
    struct wt_nl_service *service, uint32_t caller_rank,
    uint32_t caller_thread, const struct wt_nl_ticket *ticket);
enum wt_nl_status wt_nl_finish(
    struct wt_nl_service *service, uint32_t caller_rank,
    uint32_t caller_thread, const struct wt_nl_ticket *ticket);
enum wt_nl_status wt_nl_get_counters(
    const struct wt_nl_service *service, struct wt_nl_counters *out);

#ifdef __cplusplus
}
#endif
#endif
