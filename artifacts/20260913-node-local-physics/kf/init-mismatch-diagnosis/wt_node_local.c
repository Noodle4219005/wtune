#define _POSIX_C_SOURCE 200809L
#include "wt_node_local.h"

#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define WT_NL_MAGIC UINT64_C(0x57544e4f44454c31)
#define WT_NL_ALIGN ((size_t)64)
#define WT_NL_GUARD_BYTES ((size_t)32)
#define WT_NL_GUARD_VALUE 0xa5
#define WT_NL_INIT_READY 2u

struct wt_nl_shared {
    alignas(64) _Atomic uint32_t init_state;
    uint32_t version;
    uint64_t magic;
    uint32_t max_ranks, max_threads, slot_count, reserved;
    size_t mapping_bytes, header_bytes, slot_stride, slot_payload_bytes;
    _Atomic uint64_t next_work_id;
    _Atomic uint64_t staged, self_executed, helper_executed, committed, failed;
};

struct wt_nl_slot {
    alignas(64) _Atomic uint32_t state;
    uint32_t owner_rank, owner_thread, reserved;
    _Atomic uint64_t work_id;
    struct wt_nl_descriptor descriptor;
    uint64_t immutable_hash;
    _Atomic uint64_t executor_tag;
    _Atomic size_t result_written;
    _Atomic int failure_kind;
    unsigned char payload[];
};

struct wt_nl_parts {
    unsigned char *front_guard, *geometry, *request, *result, *back_guard;
};

static int add_size(size_t a, size_t b, size_t *out)
{
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int align_size(size_t value, size_t *out)
{
    size_t extra = WT_NL_ALIGN - 1;
    if (value > SIZE_MAX - extra) return 0;
    *out = (value + extra) & ~extra;
    return 1;
}

static int dimensions(const struct wt_nl_config *config, size_t *header,
                      size_t *stride, uint32_t *slots, size_t *total)
{
    size_t count, raw_stride, body;
    if (!config || !header || !stride || !slots || !total ||
        !config->max_ranks || !config->max_threads ||
        !config->slot_payload_bytes) return 0;
    if (!mul_size(config->max_ranks, config->max_threads, &count) ||
        count > UINT32_MAX || !align_size(sizeof(struct wt_nl_shared), header) ||
        !add_size(sizeof(struct wt_nl_slot), config->slot_payload_bytes, &raw_stride) ||
        !align_size(raw_stride, stride) || !mul_size(count, *stride, &body) ||
        !add_size(*header, body, total)) return 0;
    *slots = (uint32_t)count;
    return 1;
}

static struct wt_nl_shared *shared_of(const struct wt_nl_service *service)
{
    if (!service || !service->mapping ||
        service->mapping_bytes < sizeof(struct wt_nl_shared) ||
        (uintptr_t)service->mapping % alignof(struct wt_nl_shared)) return NULL;
    return (struct wt_nl_shared *)service->mapping;
}

static struct wt_nl_slot *slot_at(const struct wt_nl_shared *shared,
                                  uint32_t index)
{
    unsigned char *base = (unsigned char *)(uintptr_t)shared;
    return (struct wt_nl_slot *)(void *)(base + shared->header_bytes +
                                         (size_t)index * shared->slot_stride);
}

static int shared_valid(const struct wt_nl_service *service,
                        struct wt_nl_shared **out)
{
    struct wt_nl_shared *shared = shared_of(service);
    struct wt_nl_config config;
    size_t header, stride, expected;
    uint32_t slots;
    if (!shared || atomic_load_explicit(&shared->init_state, memory_order_acquire) !=
                       WT_NL_INIT_READY || shared->magic != WT_NL_MAGIC ||
        shared->version != WT_NL_VERSION) return 0;
    config.max_ranks = shared->max_ranks;
    config.max_threads = shared->max_threads;
    config.slot_payload_bytes = shared->slot_payload_bytes;
    if (!dimensions(&config, &header, &stride, &slots, &expected) ||
        slots != shared->slot_count || header != shared->header_bytes ||
        stride != shared->slot_stride || expected != shared->mapping_bytes ||
        expected > service->mapping_bytes)
        return 0;
    if (out) *out = shared;
    return 1;
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t bytes)
{
    const unsigned char *p = data;
    while (bytes--) { hash ^= *p++; hash *= UINT64_C(1099511628211); }
    return hash;
}

static int parts_for(const struct wt_nl_shared *shared, struct wt_nl_slot *slot,
                     struct wt_nl_parts *parts)
{
    size_t used = WT_NL_GUARD_BYTES;
    if (slot->descriptor.reserved || !slot->descriptor.geometry_bytes ||
        !slot->descriptor.request_bytes || !slot->descriptor.result_bytes ||
        !add_size(used, slot->descriptor.geometry_bytes, &used) ||
        !add_size(used, slot->descriptor.request_bytes, &used) ||
        !add_size(used, slot->descriptor.result_bytes, &used) ||
        !add_size(used, WT_NL_GUARD_BYTES, &used) ||
        used > shared->slot_payload_bytes) return 0;
    parts->front_guard = slot->payload;
    parts->geometry = parts->front_guard + WT_NL_GUARD_BYTES;
    parts->request = parts->geometry + slot->descriptor.geometry_bytes;
    parts->result = parts->request + slot->descriptor.request_bytes;
    parts->back_guard = parts->result + slot->descriptor.result_bytes;
    return 1;
}

static uint64_t immutable_hash(const struct wt_nl_slot *slot,
                               const struct wt_nl_parts *parts)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = hash_bytes(hash, &slot->owner_rank, sizeof slot->owner_rank);
    hash = hash_bytes(hash, &slot->owner_thread, sizeof slot->owner_thread);
    {
        uint64_t work_id = atomic_load(&slot->work_id);
        hash = hash_bytes(hash, &work_id, sizeof work_id);
    }
    hash = hash_bytes(hash, &slot->descriptor, sizeof slot->descriptor);
    hash = hash_bytes(hash, parts->geometry, slot->descriptor.geometry_bytes);
    return hash_bytes(hash, parts->request, slot->descriptor.request_bytes);
}

static int guard_ok(const unsigned char *guard)
{
    size_t i;
    for (i = 0; i < WT_NL_GUARD_BYTES; ++i)
        if (guard[i] != WT_NL_GUARD_VALUE) return 0;
    return 1;
}

static int slot_integrity(const struct wt_nl_shared *shared,
                          struct wt_nl_slot *slot, struct wt_nl_parts *parts)
{
    return parts_for(shared, slot, parts) && guard_ok(parts->front_guard) &&
           guard_ok(parts->back_guard) &&
           immutable_hash(slot, parts) == slot->immutable_hash;
}

static int signatures_equal(const struct wt_nl_signature *a,
                            const struct wt_nl_signature *b)
{
    return a && b && !memcmp(a, b, sizeof *a);
}

static enum wt_nl_status next_work_id(struct wt_nl_shared *shared, uint64_t *out)
{
    uint64_t current = atomic_load(&shared->next_work_id);
    if (!current || current == UINT64_MAX) return WT_NL_OVERFLOW;
    if (!atomic_compare_exchange_strong(&shared->next_work_id, &current,
                                        current + 1)) return WT_NL_BUSY;
    *out = current;
    return WT_NL_OK;
}

static enum wt_nl_status owner_location(const struct wt_nl_shared *shared,
    const struct wt_nl_ticket *ticket, uint32_t rank, uint32_t thread,
    struct wt_nl_slot **out)
{
    uint32_t expected;
    if (!ticket || rank >= shared->max_ranks || thread >= shared->max_threads ||
        ticket->owner_rank != rank || ticket->owner_thread != thread ||
        ticket->slot >= shared->slot_count) return WT_NL_OWNER;
    expected = rank * shared->max_threads + thread;
    if (ticket->slot != expected) return WT_NL_OWNER;
    if (out) *out = slot_at(shared, ticket->slot);
    if (atomic_load_explicit(&slot_at(shared, ticket->slot)->work_id,
                             memory_order_acquire) != ticket->work_id)
        return WT_NL_STALE;
    return WT_NL_OK;
}

static int ticket_matches(const struct wt_nl_slot *slot,
                          const struct wt_nl_ticket *ticket)
{
    return atomic_load(&slot->work_id) == ticket->work_id &&
           slot->owner_rank == ticket->owner_rank &&
           slot->owner_thread == ticket->owner_thread;
}

static enum wt_nl_status fail_from(struct wt_nl_shared *shared,
                                   struct wt_nl_slot *slot, uint32_t expected,
                                   int failure_kind)
{
    if (atomic_compare_exchange_strong(&slot->state, &expected,
                                       WT_NL_SLOT_FAILING)) {
        atomic_store(&slot->failure_kind, failure_kind);
        atomic_fetch_add(&shared->failed, 1);
        atomic_store_explicit(&slot->state, WT_NL_SLOT_FAILED,
                              memory_order_release);
        return failure_kind > 0 ? WT_NL_REJECTED : WT_NL_FAILED;
    }
    return atomic_load_explicit(&slot->state, memory_order_acquire) ==
                   WT_NL_SLOT_FAILED ? WT_NL_FAILED : WT_NL_BUSY;
}

static enum wt_nl_status execute_claimed(struct wt_nl_shared *shared,
    struct wt_nl_slot *slot, uint32_t claimed_state, wt_nl_execute_fn execute,
    void *context, uint64_t executor_tag, int helper)
{
    struct wt_nl_parts parts;
    size_t written = 0;
    int result;
    if (!execute || !slot_integrity(shared, slot, &parts))
        return fail_from(shared, slot, claimed_state, WT_NL_EXECUTE_FATAL);
    atomic_store(&slot->executor_tag, executor_tag);
    result = execute(context, &slot->descriptor, parts.geometry,
                     slot->descriptor.geometry_bytes, parts.request,
                     slot->descriptor.request_bytes, parts.result,
                     slot->descriptor.result_bytes, &written);
    atomic_store(&slot->result_written, written);
    if (result || written != slot->descriptor.result_bytes ||
        !slot_integrity(shared, slot, &parts))
        return fail_from(shared, slot, claimed_state,
                         result > 0 ? result : WT_NL_EXECUTE_FATAL);
    if (!atomic_compare_exchange_strong(&slot->state, &claimed_state,
                                        WT_NL_SLOT_DONE))
        return WT_NL_FAILED;
    if (helper) atomic_fetch_add(&shared->helper_executed, 1);
    else atomic_fetch_add(&shared->self_executed, 1);
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_required_bytes(const struct wt_nl_config *config,
                                       size_t *out)
{
    size_t header, stride, total;
    uint32_t slots;
    if (!out) return WT_NL_INVALID;
    if (!dimensions(config, &header, &stride, &slots, &total))
        return WT_NL_OVERFLOW;
    *out = total;
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_init_mapping(void *mapping, size_t mapping_bytes,
    const struct wt_nl_config *config, struct wt_nl_service *out)
{
    struct wt_nl_shared *shared;
    size_t header, stride, total;
    uint32_t slots, i;
    if (!mapping || !out || (uintptr_t)mapping % alignof(struct wt_nl_shared) ||
        !dimensions(config, &header, &stride, &slots, &total) ||
        mapping_bytes < total) return WT_NL_INVALID;
    memset(mapping, 0, total);
    shared = mapping;
    atomic_init(&shared->init_state, 0);
    shared->version = WT_NL_VERSION;
    shared->magic = WT_NL_MAGIC;
    shared->max_ranks = config->max_ranks;
    shared->max_threads = config->max_threads;
    shared->slot_count = slots;
    shared->mapping_bytes = total;
    shared->header_bytes = header;
    shared->slot_stride = stride;
    shared->slot_payload_bytes = config->slot_payload_bytes;
    atomic_init(&shared->next_work_id, 1);
    atomic_init(&shared->staged, 0); atomic_init(&shared->self_executed, 0);
    atomic_init(&shared->helper_executed, 0); atomic_init(&shared->committed, 0);
    atomic_init(&shared->failed, 0);
    for (i = 0; i < slots; ++i) {
        struct wt_nl_slot *slot = slot_at(shared, i);
        atomic_init(&slot->state, WT_NL_SLOT_EMPTY);
        atomic_init(&slot->work_id, 0);
        atomic_init(&slot->executor_tag, 0);
        atomic_init(&slot->result_written, 0);
        atomic_init(&slot->failure_kind, 0);
    }
    if (!atomic_is_lock_free(&shared->next_work_id) ||
        !atomic_is_lock_free(&shared->init_state) ||
        !atomic_is_lock_free(&shared->staged) ||
        !atomic_is_lock_free(&slot_at(shared, 0)->state) ||
        !atomic_is_lock_free(&slot_at(shared, 0)->work_id) ||
        !atomic_is_lock_free(&slot_at(shared, 0)->executor_tag) ||
        !atomic_is_lock_free(&slot_at(shared, 0)->result_written) ||
        !atomic_is_lock_free(&slot_at(shared, 0)->failure_kind))
        return WT_NL_UNSUPPORTED;
    out->mapping = mapping;
    out->mapping_bytes = total;
    out->mapped_by_service = 0;
    atomic_store_explicit(&shared->init_state, WT_NL_INIT_READY,
                          memory_order_release);
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_attach_mapping(void *mapping, size_t mapping_bytes,
                                       struct wt_nl_service *out)
{
    struct wt_nl_service candidate = {mapping, mapping_bytes, 0};
    if (!out || !shared_valid(&candidate, NULL)) return WT_NL_INVALID;
    *out = candidate;
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_init_fd(int fd, const struct wt_nl_config *config,
                                struct wt_nl_service *out)
{
    size_t bytes;
    void *mapping;
    enum wt_nl_status status = wt_nl_required_bytes(config, &bytes);
    if (status != WT_NL_OK || fd < 0 || !out || bytes > (size_t)INT64_MAX)
        return WT_NL_INVALID;
    if (ftruncate(fd, (off_t)bytes)) return WT_NL_SYSTEM;
    mapping = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) return WT_NL_SYSTEM;
    status = wt_nl_init_mapping(mapping, bytes, config, out);
    if (status != WT_NL_OK) { munmap(mapping, bytes); return status; }
    out->mapped_by_service = 1;
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_attach_fd(int fd, size_t mapping_bytes,
                                  struct wt_nl_service *out)
{
    struct stat information;
    void *mapping;
    enum wt_nl_status status;
    if (fd < 0 || !mapping_bytes || !out || fstat(fd, &information) ||
        information.st_size < 0 ||
        (uintmax_t)information.st_size < (uintmax_t)mapping_bytes)
        return WT_NL_INVALID;
    mapping = mmap(NULL, mapping_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) return WT_NL_SYSTEM;
    status = wt_nl_attach_mapping(mapping, mapping_bytes, out);
    if (status != WT_NL_OK) { munmap(mapping, mapping_bytes); return status; }
    out->mapped_by_service = 1;
    return WT_NL_OK;
}

void wt_nl_detach(struct wt_nl_service *service)
{
    if (!service) return;
    if (service->mapped_by_service && service->mapping)
        (void)munmap(service->mapping, service->mapping_bytes);
    memset(service, 0, sizeof *service);
}

enum wt_nl_status wt_nl_slot_index(const struct wt_nl_service *service,
    uint32_t rank_slot, uint32_t thread_slot, uint32_t *slot)
{
    struct wt_nl_shared *shared;
    if (!slot || !shared_valid(service, &shared) ||
        rank_slot >= shared->max_ranks || thread_slot >= shared->max_threads)
        return WT_NL_INVALID;
    *slot = rank_slot * shared->max_threads + thread_slot;
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_publish(struct wt_nl_service *service,
    uint32_t owner_rank, uint32_t owner_thread,
    const struct wt_nl_descriptor *descriptor, const void *geometry,
    const void *request, struct wt_nl_ticket *ticket)
{
    struct wt_nl_shared *shared;
    struct wt_nl_slot *slot;
    struct wt_nl_parts parts;
    uint32_t index, expected = WT_NL_SLOT_EMPTY;
    uint64_t work_id;
    enum wt_nl_status status;
    if (!ticket || !descriptor || !geometry || !request || descriptor->reserved ||
        !descriptor->signature.phase || !descriptor->signature.profile ||
        !descriptor->signature.schema || !descriptor->signature.epoch ||
        !descriptor->signature.init_generation || !descriptor->call_sequence ||
        !shared_valid(service, &shared) ||
        wt_nl_slot_index(service, owner_rank, owner_thread, &index) != WT_NL_OK)
        return WT_NL_INVALID;
    slot = slot_at(shared, index);
    if (!atomic_compare_exchange_strong(&slot->state, &expected,
                                        WT_NL_SLOT_RESERVED)) return WT_NL_BUSY;
    slot->owner_rank = owner_rank; slot->owner_thread = owner_thread;
    slot->descriptor = *descriptor;
    status = next_work_id(shared, &work_id);
    if (!parts_for(shared, slot, &parts) || status != WT_NL_OK) {
        slot->owner_rank = slot->owner_thread = slot->reserved = 0;
        slot->work_id = 0;
        memset(&slot->descriptor, 0, sizeof slot->descriptor);
        slot->immutable_hash = 0;
        atomic_store(&slot->executor_tag, 0);
        atomic_store(&slot->result_written, 0);
        atomic_store(&slot->failure_kind, 0);
        atomic_store_explicit(&slot->state, WT_NL_SLOT_EMPTY, memory_order_release);
        return status == WT_NL_OK ? WT_NL_OVERFLOW : status;
    }
    atomic_store(&slot->work_id, work_id);
    memset(parts.front_guard, WT_NL_GUARD_VALUE, WT_NL_GUARD_BYTES);
    memcpy(parts.geometry, geometry, descriptor->geometry_bytes);
    memcpy(parts.request, request, descriptor->request_bytes);
    memset(parts.result, 0, descriptor->result_bytes);
    memset(parts.back_guard, WT_NL_GUARD_VALUE, WT_NL_GUARD_BYTES);
    atomic_store(&slot->executor_tag, 0); atomic_store(&slot->result_written, 0);
    atomic_store(&slot->failure_kind, 0);
    slot->immutable_hash = immutable_hash(slot, &parts);
    ticket->work_id = work_id; ticket->slot = index;
    ticket->owner_rank = owner_rank; ticket->owner_thread = owner_thread;
    atomic_fetch_add(&shared->staged, 1);
    atomic_store_explicit(&slot->state, WT_NL_SLOT_PUBLISHED, memory_order_release);
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_try_execute(struct wt_nl_service *service,
    uint32_t helper_rank, uint32_t helper_thread,
    const struct wt_nl_signature *required_signature,
    wt_nl_geometry_compatible_fn compatible, wt_nl_execute_fn execute,
    void *context, uint64_t executor_tag, struct wt_nl_ticket *executed)
{
    struct wt_nl_shared *shared;
    uint32_t i;
    if (!required_signature || !compatible || !execute || !executed ||
        !shared_valid(service, &shared) || helper_rank >= shared->max_ranks ||
        helper_thread >= shared->max_threads) return WT_NL_INVALID;
    for (i = 0; i < shared->slot_count; ++i) {
        struct wt_nl_slot *slot = slot_at(shared, i);
        uint32_t expected = WT_NL_SLOT_PUBLISHED;
        struct wt_nl_parts parts;
        uint32_t owner_slot_rank = i / shared->max_threads;
        if (owner_slot_rank == helper_rank ||
            !atomic_compare_exchange_strong_explicit(
                &slot->state, &expected, WT_NL_SLOT_CLAIMED_HELPER,
                memory_order_acquire, memory_order_relaxed)) continue;
        if (!slot_integrity(shared, slot, &parts))
            return fail_from(shared, slot, WT_NL_SLOT_CLAIMED_HELPER,
                             WT_NL_EXECUTE_FATAL);
        if (!signatures_equal(&slot->descriptor.signature, required_signature) ||
            !compatible(context, &slot->descriptor, parts.geometry,
                        slot->descriptor.geometry_bytes)) {
            expected = WT_NL_SLOT_CLAIMED_HELPER;
            if (!atomic_compare_exchange_strong_explicit(
                    &slot->state, &expected, WT_NL_SLOT_PUBLISHED,
                    memory_order_release, memory_order_relaxed))
                return WT_NL_FAILED;
            continue;
        }
        executed->work_id = atomic_load(&slot->work_id); executed->slot = i;
        executed->owner_rank = slot->owner_rank;
        executed->owner_thread = slot->owner_thread;
        return execute_claimed(shared, slot, WT_NL_SLOT_CLAIMED_HELPER,
                               execute, context, executor_tag, 1);
    }
    return WT_NL_IDLE;
}

enum wt_nl_status wt_nl_owner_execute(struct wt_nl_service *service,
    uint32_t caller_rank, uint32_t caller_thread,
    const struct wt_nl_ticket *ticket, wt_nl_execute_fn execute,
    void *context, uint64_t executor_tag)
{
    struct wt_nl_shared *shared;
    struct wt_nl_slot *slot;
    uint32_t expected = WT_NL_SLOT_PUBLISHED;
    enum wt_nl_status status;
    if (!shared_valid(service, &shared)) return WT_NL_INVALID;
    status = owner_location(shared, ticket, caller_rank, caller_thread, &slot);
    if (status != WT_NL_OK) return status;
    if (!atomic_compare_exchange_strong_explicit(
            &slot->state, &expected, WT_NL_SLOT_CLAIMED_SELF,
            memory_order_acquire, memory_order_relaxed)) {
        if (expected == WT_NL_SLOT_CLAIMED_HELPER) return WT_NL_PENDING;
        if (expected == WT_NL_SLOT_DONE) return WT_NL_BUSY;
        if (expected == WT_NL_SLOT_FAILED)
            return ticket_matches(slot, ticket) ? WT_NL_FAILED : WT_NL_STALE;
        return expected == WT_NL_SLOT_FAILED ? WT_NL_FAILED : WT_NL_BUSY;
    }
    if (!ticket_matches(slot, ticket)) {
        expected = WT_NL_SLOT_CLAIMED_SELF;
        (void)atomic_compare_exchange_strong_explicit(
            &slot->state, &expected, WT_NL_SLOT_PUBLISHED,
            memory_order_release, memory_order_relaxed);
        return WT_NL_STALE;
    }
    return execute_claimed(shared, slot, WT_NL_SLOT_CLAIMED_SELF, execute,
                           context, executor_tag, 0);
}

enum wt_nl_status wt_nl_owner_collect(struct wt_nl_service *service,
    uint32_t caller_rank, uint32_t caller_thread,
    const struct wt_nl_ticket *ticket, wt_nl_owner_validate_fn validate,
    wt_nl_owner_commit_fn commit, void *context)
{
    struct wt_nl_shared *shared;
    struct wt_nl_slot *slot;
    struct wt_nl_parts parts;
    uint32_t expected = WT_NL_SLOT_DONE;
    enum wt_nl_status status;
    if (!validate || !commit || !shared_valid(service, &shared)) return WT_NL_INVALID;
    status = owner_location(shared, ticket, caller_rank, caller_thread, &slot);
    if (status != WT_NL_OK) return status;
    if (!atomic_compare_exchange_strong_explicit(
            &slot->state, &expected, WT_NL_SLOT_VALIDATING,
            memory_order_acquire, memory_order_relaxed)) {
        if (expected == WT_NL_SLOT_FAILED) {
            if (!ticket_matches(slot, ticket)) return WT_NL_STALE;
            return atomic_load(&slot->failure_kind) > 0 ? WT_NL_REJECTED :
                                                         WT_NL_FAILED;
        }
        if (expected == WT_NL_SLOT_PUBLISHED ||
            expected == WT_NL_SLOT_CLAIMED_SELF ||
            expected == WT_NL_SLOT_CLAIMED_HELPER) return WT_NL_PENDING;
        return WT_NL_BUSY;
    }
    if (!ticket_matches(slot, ticket)) {
        expected = WT_NL_SLOT_VALIDATING;
        (void)atomic_compare_exchange_strong_explicit(
            &slot->state, &expected, WT_NL_SLOT_DONE,
            memory_order_release, memory_order_relaxed);
        return WT_NL_STALE;
    }
    if (!slot_integrity(shared, slot, &parts) ||
        atomic_load(&slot->result_written) != slot->descriptor.result_bytes ||
        !validate(context, &slot->descriptor, parts.geometry,
                  slot->descriptor.geometry_bytes, parts.request,
                  slot->descriptor.request_bytes, parts.result,
                  slot->descriptor.result_bytes))
        return fail_from(shared, slot, WT_NL_SLOT_VALIDATING,
                         WT_NL_EXECUTE_FATAL);
    commit(context, &slot->descriptor, parts.geometry,
           slot->descriptor.geometry_bytes, parts.request,
           slot->descriptor.request_bytes, parts.result,
           slot->descriptor.result_bytes);
    atomic_fetch_add(&shared->committed, 1);
    atomic_store_explicit(&slot->state, WT_NL_SLOT_COMMITTED,
                          memory_order_release);
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_ticket_executor(const struct wt_nl_service *service,
    uint32_t caller_rank, uint32_t caller_thread,
    const struct wt_nl_ticket *ticket, uint64_t *executor_tag)
{
    struct wt_nl_shared *shared;
    struct wt_nl_slot *slot;
    uint32_t state;
    enum wt_nl_status status;
    if (!executor_tag || !shared_valid(service,&shared)) return WT_NL_INVALID;
    status=owner_location(shared,ticket,caller_rank,caller_thread,&slot);
    if (status!=WT_NL_OK) return status;
    state=atomic_load_explicit(&slot->state,memory_order_acquire);
    if (!ticket_matches(slot,ticket)) return WT_NL_STALE;
    if (state==WT_NL_SLOT_DONE || state==WT_NL_SLOT_COMMITTED) {
        *executor_tag=atomic_load(&slot->executor_tag); return WT_NL_OK;
    }
    if (state==WT_NL_SLOT_PUBLISHED || state==WT_NL_SLOT_CLAIMED_SELF ||
        state==WT_NL_SLOT_CLAIMED_HELPER) return WT_NL_PENDING;
    if (state==WT_NL_SLOT_FAILED) return WT_NL_FAILED;
    return WT_NL_BUSY;
}

enum wt_nl_status wt_nl_owner_poison(struct wt_nl_service *service,
    uint32_t caller_rank, uint32_t caller_thread,
    const struct wt_nl_ticket *ticket)
{
    struct wt_nl_shared *shared;
    struct wt_nl_slot *slot;
    uint32_t state, claimed;
    enum wt_nl_status status;
    if (!shared_valid(service, &shared)) return WT_NL_INVALID;
    status = owner_location(shared, ticket, caller_rank, caller_thread, &slot);
    if (status != WT_NL_OK) return status;
    state = atomic_load_explicit(&slot->state, memory_order_acquire);
    if (state != WT_NL_SLOT_PUBLISHED && state != WT_NL_SLOT_CLAIMED_SELF &&
        state != WT_NL_SLOT_CLAIMED_HELPER) {
        if (state == WT_NL_SLOT_FAILED)
            return ticket_matches(slot, ticket) ? WT_NL_FAILED : WT_NL_STALE;
        return WT_NL_BUSY;
    }
    claimed = state;
    if (!atomic_compare_exchange_strong_explicit(
            &slot->state, &state, WT_NL_SLOT_FAILING,
            memory_order_acquire, memory_order_relaxed)) return WT_NL_BUSY;
    if (!ticket_matches(slot, ticket)) {
        state = WT_NL_SLOT_FAILING;
        (void)atomic_compare_exchange_strong_explicit(
            &slot->state, &state, claimed,
            memory_order_release, memory_order_relaxed);
        return WT_NL_STALE;
    }
    atomic_store(&slot->failure_kind, WT_NL_EXECUTE_FATAL);
    atomic_fetch_add(&shared->failed, 1);
    atomic_store_explicit(&slot->state, WT_NL_SLOT_FAILED,
                          memory_order_release);
    return WT_NL_FAILED;
}

enum wt_nl_status wt_nl_finish(struct wt_nl_service *service,
    uint32_t caller_rank, uint32_t caller_thread,
    const struct wt_nl_ticket *ticket)
{
    struct wt_nl_shared *shared;
    struct wt_nl_slot *slot;
    uint32_t expected = WT_NL_SLOT_COMMITTED;
    enum wt_nl_status status;
    if (!shared_valid(service, &shared)) return WT_NL_INVALID;
    status = owner_location(shared, ticket, caller_rank, caller_thread, &slot);
    if (status != WT_NL_OK) return status;
    if (!atomic_compare_exchange_strong_explicit(
            &slot->state, &expected, WT_NL_SLOT_RESERVED,
            memory_order_acquire, memory_order_relaxed))
        return expected == WT_NL_SLOT_FAILED ? WT_NL_FAILED : WT_NL_BUSY;
    if (!ticket_matches(slot, ticket)) {
        expected = WT_NL_SLOT_RESERVED;
        (void)atomic_compare_exchange_strong_explicit(
            &slot->state, &expected, WT_NL_SLOT_COMMITTED,
            memory_order_release, memory_order_relaxed);
        return WT_NL_STALE;
    }
    atomic_store(&slot->work_id, 0);
    slot->owner_rank = slot->owner_thread = 0;
    memset(&slot->descriptor, 0, sizeof slot->descriptor);
    slot->immutable_hash = 0; atomic_store(&slot->executor_tag, 0);
    atomic_store(&slot->result_written, 0); atomic_store(&slot->failure_kind, 0);
    atomic_store_explicit(&slot->state, WT_NL_SLOT_EMPTY, memory_order_release);
    return WT_NL_OK;
}

enum wt_nl_status wt_nl_get_counters(const struct wt_nl_service *service,
                                     struct wt_nl_counters *out)
{
    struct wt_nl_shared *shared;
    if (!out || !shared_valid(service, &shared)) return WT_NL_INVALID;
    out->staged = atomic_load(&shared->staged);
    out->self_executed = atomic_load(&shared->self_executed);
    out->helper_executed = atomic_load(&shared->helper_executed);
    out->committed = atomic_load(&shared->committed);
    out->failed = atomic_load(&shared->failed);
    return WT_NL_OK;
}
