#include "wt_ledger_internal.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static _Atomic uint64_t next_ledger_identity = 1u;

static int allocate_identity(uint64_t *identity) {
    uint64_t current = atomic_load_explicit(&next_ledger_identity,
                                            memory_order_relaxed);
    while (current != UINT64_MAX) {
        if (atomic_compare_exchange_weak_explicit(
                &next_ledger_identity, &current, current + 1u,
                memory_order_relaxed, memory_order_relaxed)) {
            *identity = current;
            return WT_LEDGER_OK;
        }
    }
    return WT_LEDGER_EFULL;
}

static void release_entries(struct wt_ledger *ledger) {
    size_t i;
    for (i = 0; i < ledger->max_entries; ++i) {
        free(ledger->entries[i].request);
        free(ledger->entries[i].result);
        memset(&ledger->entries[i], 0, sizeof(ledger->entries[i]));
    }
    memset(ledger->closed, 0,
           ledger->max_entries * sizeof(ledger->closed[0]));
    ledger->used_entries = 0u;
    ledger->closed_count = 0u;
    ledger->live_bytes = 0u;
}

static int accept_request(struct wt_ledger *ledger, uint8_t role,
                          const uint8_t *request, size_t request_len,
                          size_t expected_result_limit,
                          struct wt_ledger_ticket *ticket) {
    struct wt_wire_frame frame;
    struct wt_ledger_entry *entry;
    uint8_t *request_copy;
    uint8_t *result_buffer;
    size_t slot;
    int status;
    if (ledger == NULL || request == NULL || ticket == NULL ||
        ledger->role != role || expected_result_limit < WT_WIRE_HEADER_BYTES ||
        (uint64_t)expected_result_limit > ledger->contract.limits.max_frame) {
        return WT_LEDGER_EINVAL;
    }
    if (ledger->callback_depth != 0u) {
        return WT_LEDGER_EBUSY;
    }
    status = wt_ledger_private_decode_frame(ledger, request, request_len, WT_WIRE_REQUEST, &frame);
    if (status != WT_LEDGER_OK) {
        return status;
    }
    if (wt_ledger_private_is_closed(ledger, frame.id.domain, frame.id.epoch)) {
        return WT_LEDGER_ESTALE;
    }
    entry = wt_ledger_private_find_key(ledger, &frame.id, &slot);
    if (entry != NULL) {
        if (entry->request_len == request_len &&
            entry->result_capacity == expected_result_limit &&
            memcmp(entry->request, request, request_len) == 0) {
            wt_ledger_private_make_ticket(ledger, slot, ticket);
            return WT_LEDGER_DUPLICATE;
        }
        if (entry->state != WT_LEDGER_COMMITTING &&
            entry->state != WT_LEDGER_COMMITTED) {
            entry->state = WT_LEDGER_FAILED;
        }
        return WT_LEDGER_ECONFLICT;
    }
    if (ledger->used_entries == ledger->max_entries) {
        return WT_LEDGER_EFULL;
    }
    if (request_len > ledger->max_live_bytes - ledger->live_bytes ||
        expected_result_limit >
            ledger->max_live_bytes - ledger->live_bytes - request_len) {
        return WT_LEDGER_EBUDGET;
    }
    for (slot = 0; slot < ledger->max_entries; ++slot) {
        if (!ledger->entries[slot].used) {
            break;
        }
    }
    request_copy = malloc(request_len);
    if (request_copy == NULL) {
        return WT_LEDGER_ENOMEM;
    }
    result_buffer = malloc(expected_result_limit);
    if (result_buffer == NULL) {
        free(request_copy);
        return WT_LEDGER_ENOMEM;
    }
    memcpy(request_copy, request, request_len);
    entry = &ledger->entries[slot];
    memset(entry, 0, sizeof(*entry));
    entry->used = 1;
    entry->state = WT_LEDGER_ACCEPTED;
    entry->id = frame.id;
    memcpy(entry->request_fingerprint, frame.request_fingerprint,
           sizeof(entry->request_fingerprint));
    entry->request = request_copy;
    entry->request_len = request_len;
    entry->result = result_buffer;
    entry->result_capacity = expected_result_limit;
    entry->nonce = ledger->next_nonce++;
    if (ledger->next_nonce == 0u) {
        ledger->next_nonce = 1u;
    }
    ++ledger->used_entries;
    ledger->live_bytes += request_len + expected_result_limit;
    wt_ledger_private_make_ticket(ledger, slot, ticket);
    return WT_LEDGER_OK;
}

int wt_ledger_create(enum wt_ledger_role role,
                     const struct wt_ledger_contract *contract,
                     size_t max_entries, size_t max_live_bytes,
                     struct wt_ledger **out) {
    struct wt_ledger *ledger;
    uint64_t identity;
    int status;
    if (out == NULL || (role != WT_LEDGER_OWNER && role != WT_LEDGER_WORKER) ||
        !wt_ledger_private_valid_contract(contract) || max_entries == 0u ||
        max_entries > UINT32_MAX ||
        max_entries > SIZE_MAX / sizeof(struct wt_ledger_entry) ||
        max_entries > SIZE_MAX / sizeof(struct wt_closed_epoch)) {
        return WT_LEDGER_EINVAL;
    }
    status = allocate_identity(&identity);
    if (status != WT_LEDGER_OK) {
        return status;
    }
    ledger = calloc(1u, sizeof(*ledger));
    if (ledger == NULL) {
        return WT_LEDGER_ENOMEM;
    }
    ledger->entries = calloc(max_entries, sizeof(ledger->entries[0]));
    ledger->closed = calloc(max_entries, sizeof(ledger->closed[0]));
    if (ledger->entries == NULL || ledger->closed == NULL) {
        free(ledger->entries);
        free(ledger->closed);
        free(ledger);
        return WT_LEDGER_ENOMEM;
    }
    ledger->role = role;
    ledger->contract = *contract;
    ledger->max_entries = max_entries;
    ledger->max_live_bytes = max_live_bytes;
    ledger->generation = 1u;
    ledger->next_nonce = 1u;
    ledger->identity = identity;
    *out = ledger;
    return WT_LEDGER_OK;
}

int wt_ledger_owner_register_request(struct wt_ledger *ledger,
                                     const uint8_t *request,
                                     size_t request_len,
                                     size_t expected_result_limit,
                                     struct wt_ledger_ticket *ticket) {
    return accept_request(ledger, WT_LEDGER_OWNER, request, request_len,
                          expected_result_limit, ticket);
}

int wt_ledger_worker_accept(struct wt_ledger *ledger,
                            const uint8_t *request, size_t request_len,
                            size_t expected_result_limit,
                            struct wt_ledger_ticket *ticket) {
    return accept_request(ledger, WT_LEDGER_WORKER, request, request_len,
                          expected_result_limit, ticket);
}

int wt_ledger_worker_run(struct wt_ledger *ledger,
                         const struct wt_ledger_ticket *ticket,
                         wt_ledger_run_callback callback, void *context) {
    struct wt_ledger_entry *entry;
    struct wt_wire_frame frame;
    size_t result_len = 0u;
    int callback_status;
    int status;
    if (ledger == NULL || callback == NULL || ledger->role != WT_LEDGER_WORKER) {
        return WT_LEDGER_EINVAL;
    }
    if (ledger->callback_depth != 0u) {
        return WT_LEDGER_EBUSY;
    }
    status = wt_ledger_private_ticket_entry(ledger, ticket, &entry);
    if (status != WT_LEDGER_OK) {
        return status;
    }
    if (entry->state == WT_LEDGER_DONE) {
        return WT_LEDGER_DUPLICATE;
    }
    if (entry->state != WT_LEDGER_ACCEPTED) {
        return WT_LEDGER_ESTATE;
    }
    entry->state = WT_LEDGER_RUNNING;
    ++ledger->callback_depth;
    callback_status = callback(context, entry->request, entry->request_len,
                               entry->result, entry->result_capacity,
                               &result_len);
    --ledger->callback_depth;
    if (entry->state != WT_LEDGER_RUNNING) {
        return WT_LEDGER_ESTATE;
    }
    if (callback_status != 0) {
        entry->state = WT_LEDGER_FAILED;
        return WT_LEDGER_ECALLBACK;
    }
    if (result_len > entry->result_capacity ||
        wt_ledger_private_decode_frame(ledger, entry->result, result_len, WT_WIRE_RESULT,
                     &frame) != WT_LEDGER_OK ||
        !wt_wire_id_equal(&entry->id, &frame.id) ||
        memcmp(entry->request_fingerprint, frame.request_fingerprint,
               sizeof(entry->request_fingerprint)) != 0) {
        entry->state = WT_LEDGER_FAILED;
        return WT_LEDGER_EWIRE;
    }
    entry->result_len = result_len;
    entry->state = WT_LEDGER_DONE;
    return WT_LEDGER_OK;
}

int wt_ledger_owner_apply_result(struct wt_ledger *ledger,
                                 const struct wt_ledger_ticket *ticket,
                                 const uint8_t *result, size_t result_len,
                                 wt_ledger_commit_callback callback,
                                 void *context) {
    struct wt_ledger_entry *entry;
    struct wt_wire_frame frame;
    int callback_status;
    int status;
    if (ledger == NULL || result == NULL || callback == NULL ||
        ledger->role != WT_LEDGER_OWNER) {
        return WT_LEDGER_EINVAL;
    }
    if (ledger->callback_depth != 0u) {
        return WT_LEDGER_EBUSY;
    }
    status = wt_ledger_private_ticket_entry(ledger, ticket, &entry);
    if (status != WT_LEDGER_OK) {
        return status;
    }
    if (result_len > entry->result_capacity ||
        wt_ledger_private_decode_frame(ledger, result, result_len, WT_WIRE_RESULT, &frame) !=
            WT_LEDGER_OK ||
        !wt_wire_id_equal(&entry->id, &frame.id) ||
        memcmp(entry->request_fingerprint, frame.request_fingerprint,
               sizeof(entry->request_fingerprint)) != 0) {
        return WT_LEDGER_EWIRE;
    }
    if (entry->state == WT_LEDGER_COMMITTED) {
        return entry->result_len == result_len &&
                       memcmp(entry->result, result, result_len) == 0
                   ? WT_LEDGER_DUPLICATE
                   : WT_LEDGER_ECONFLICT;
    }
    if (entry->state != WT_LEDGER_ACCEPTED) {
        return WT_LEDGER_ESTATE;
    }
    memcpy(entry->result, result, result_len);
    entry->result_len = result_len;
    if (wt_ledger_private_decode_frame(ledger, entry->result, entry->result_len,
                     WT_WIRE_RESULT, &frame) != WT_LEDGER_OK) {
        entry->state = WT_LEDGER_FAILED;
        return WT_LEDGER_EWIRE;
    }
    entry->state = WT_LEDGER_COMMITTING;
    ++ledger->callback_depth;
    callback_status = callback(context, &frame);
    --ledger->callback_depth;
    if (entry->state != WT_LEDGER_COMMITTING || callback_status != 0) {
        entry->state = WT_LEDGER_FAILED;
        return WT_LEDGER_ECALLBACK;
    }
    entry->state = WT_LEDGER_COMMITTED;
    return WT_LEDGER_OK;
}

int wt_ledger_fail(struct wt_ledger *ledger,
                   const struct wt_ledger_ticket *ticket) {
    struct wt_ledger_entry *entry;
    int status;
    if (ledger == NULL) {
        return WT_LEDGER_EINVAL;
    }
    if (ledger->callback_depth != 0u) {
        return WT_LEDGER_EBUSY;
    }
    status = wt_ledger_private_ticket_entry(ledger, ticket, &entry);
    if (status != WT_LEDGER_OK) {
        return status;
    }
    if (entry->state == WT_LEDGER_FAILED) {
        return WT_LEDGER_DUPLICATE;
    }
    if (entry->state != WT_LEDGER_ACCEPTED) {
        return WT_LEDGER_ESTATE;
    }
    entry->state = WT_LEDGER_FAILED;
    return WT_LEDGER_OK;
}

int wt_ledger_get_state(const struct wt_ledger *ledger,
                        const struct wt_ledger_ticket *ticket,
                        enum wt_ledger_state *state) {
    struct wt_ledger_entry *entry;
    int status;
    if (state == NULL) {
        return WT_LEDGER_EINVAL;
    }
    status = wt_ledger_private_ticket_entry(ledger, ticket, &entry);
    if (status == WT_LEDGER_OK) {
        *state = entry->state;
    }
    return status;
}

int wt_ledger_request_bytes(const struct wt_ledger *ledger,
                            const struct wt_ledger_ticket *ticket,
                            const uint8_t **request, size_t *request_len) {
    struct wt_ledger_entry *entry;
    int status;
    if (request == NULL || request_len == NULL) {
        return WT_LEDGER_EINVAL;
    }
    status = wt_ledger_private_ticket_entry(ledger, ticket, &entry);
    if (status == WT_LEDGER_OK) {
        *request = entry->request;
        *request_len = entry->request_len;
    }
    return status;
}

int wt_ledger_result_bytes(const struct wt_ledger *ledger,
                           const struct wt_ledger_ticket *ticket,
                           const uint8_t **result, size_t *result_len) {
    struct wt_ledger_entry *entry;
    int status;
    if (result == NULL || result_len == NULL) {
        return WT_LEDGER_EINVAL;
    }
    status = wt_ledger_private_ticket_entry(ledger, ticket, &entry);
    if (status != WT_LEDGER_OK) {
        return status;
    }
    if (entry->state != WT_LEDGER_DONE &&
        entry->state != WT_LEDGER_COMMITTED) {
        return WT_LEDGER_ESTATE;
    }
    *result = entry->result;
    *result_len = entry->result_len;
    return WT_LEDGER_OK;
}

int wt_ledger_close_epoch(struct wt_ledger *ledger, uint32_t domain,
                          uint64_t epoch) {
    size_t i;
    if (ledger == NULL) {
        return WT_LEDGER_EINVAL;
    }
    if (ledger->callback_depth != 0u) {
        return WT_LEDGER_EBUSY;
    }
    if (wt_ledger_private_is_closed(ledger, domain, epoch)) {
        return WT_LEDGER_DUPLICATE;
    }
    if (ledger->closed_count == ledger->max_entries) {
        return WT_LEDGER_EFULL;
    }
    for (i = 0; i < ledger->max_entries; ++i) {
        enum wt_ledger_state state = ledger->entries[i].state;
        if (ledger->entries[i].used && ledger->entries[i].id.domain == domain &&
            ledger->entries[i].id.epoch == epoch && state != WT_LEDGER_DONE &&
            state != WT_LEDGER_COMMITTED && state != WT_LEDGER_FAILED) {
            return WT_LEDGER_EBUSY;
        }
    }
    ledger->closed[ledger->closed_count].used = 1;
    ledger->closed[ledger->closed_count].domain = domain;
    ledger->closed[ledger->closed_count].epoch = epoch;
    ++ledger->closed_count;
    return WT_LEDGER_OK;
}

int wt_ledger_reset(struct wt_ledger *ledger,
                    const struct wt_ledger_contract *new_contract) {
    size_t i;
    if (ledger == NULL || !wt_ledger_private_valid_contract(new_contract) ||
        wt_ledger_private_same_identity(&ledger->contract, new_contract)) {
        return WT_LEDGER_EINVAL;
    }
    if (ledger->callback_depth != 0u) {
        return WT_LEDGER_EBUSY;
    }
    if (ledger->generation == UINT64_MAX) {
        return WT_LEDGER_EFULL;
    }
    for (i = 0; i < ledger->max_entries; ++i) {
        enum wt_ledger_state state = ledger->entries[i].state;
        if (ledger->entries[i].used && state != WT_LEDGER_DONE &&
            state != WT_LEDGER_COMMITTED && state != WT_LEDGER_FAILED) {
            return WT_LEDGER_EBUSY;
        }
    }
    release_entries(ledger);
    ledger->contract = *new_contract;
    ++ledger->generation;
    return WT_LEDGER_OK;
}

int wt_ledger_destroy(struct wt_ledger **ledger_pointer) {
    struct wt_ledger *ledger;
    if (ledger_pointer == NULL || *ledger_pointer == NULL) {
        return WT_LEDGER_EINVAL;
    }
    ledger = *ledger_pointer;
    if (ledger->callback_depth != 0u) {
        return WT_LEDGER_EBUSY;
    }
    release_entries(ledger);
    free(ledger->entries);
    free(ledger->closed);
    free(ledger);
    *ledger_pointer = NULL;
    return WT_LEDGER_OK;
}

#ifdef WT_LEDGER_TESTING
void wt_ledger_private_test_set_next_identity(uint64_t value) {
    atomic_store_explicit(&next_ledger_identity, value, memory_order_relaxed);
}

void wt_ledger_private_test_set_generation(struct wt_ledger *ledger,
                                           uint64_t value) {
    ledger->generation = value;
}
#endif
