#include "wt_ledger_internal.h"

#include <stdint.h>
#include <string.h>

static int any_nonzero(const uint8_t *bytes, size_t length) {
    size_t i;
    for (i = 0; i < length; ++i) {
        if (bytes[i] != 0u) {
            return 1;
        }
    }
    return 0;
}

static int valid_abi(const struct wt_payload_abi *abi) {
    uint64_t max_true;
    if (abi == NULL || (abi->int_bytes != 1u && abi->int_bytes != 2u &&
                        abi->int_bytes != 4u && abi->int_bytes != 8u) ||
        (abi->logical_bytes != 1u && abi->logical_bytes != 2u &&
         abi->logical_bytes != 4u && abi->logical_bytes != 8u) ||
        abi->logical_true_bits == 0u) {
        return 0;
    }
    max_true = abi->logical_bytes == 8u
                   ? UINT64_MAX
                   : (UINT64_C(1) << (8u * abi->logical_bytes)) - 1u;
    return abi->logical_true_bits <= max_true &&
           ((abi->float_format == WT_FLOAT_IEEE754_BINARY32 &&
             abi->real_bytes == 4u) ||
            (abi->float_format == WT_FLOAT_IEEE754_BINARY64 &&
             abi->real_bytes == 8u));
}

int wt_ledger_private_valid_contract(
    const struct wt_ledger_contract *contract) {
    return contract != NULL && valid_abi(&contract->abi) &&
           any_nonzero(contract->run_uuid, sizeof(contract->run_uuid)) &&
           any_nonzero(contract->schema_fingerprint,
                       sizeof(contract->schema_fingerprint)) &&
           contract->abi.nargs <= contract->limits.max_nargs &&
           contract->limits.max_frame >= WT_WIRE_HEADER_BYTES &&
           contract->limits.max_frame <= (uint64_t)SIZE_MAX &&
           contract->limits.max_payload <= (uint64_t)SIZE_MAX;
}

int wt_ledger_private_same_identity(
    const struct wt_ledger_contract *a,
    const struct wt_ledger_contract *b) {
    return memcmp(a->run_uuid, b->run_uuid, sizeof(a->run_uuid)) == 0 &&
           a->owner_rank == b->owner_rank &&
           a->owner_generation == b->owner_generation;
}

static int frame_in_contract(const struct wt_ledger *ledger,
                             const struct wt_wire_frame *frame,
                             uint8_t kind) {
    return frame->kind == kind &&
           memcmp(frame->id.run_uuid, ledger->contract.run_uuid,
                  sizeof(frame->id.run_uuid)) == 0 &&
           frame->id.owner_rank == ledger->contract.owner_rank &&
           frame->id.owner_generation == ledger->contract.owner_generation;
}

int wt_ledger_private_decode_frame(const struct wt_ledger *ledger,
                                   const uint8_t *bytes, size_t length,
                                   uint8_t kind,
                                   struct wt_wire_frame *frame) {
    if (wt_wire_decode(bytes, length, &ledger->contract.limits,
                       ledger->contract.schema_fingerprint,
                       &ledger->contract.abi, frame) != WT_WIRE_OK ||
        !frame_in_contract(ledger, frame, kind)) {
        return WT_LEDGER_EWIRE;
    }
    return WT_LEDGER_OK;
}

int wt_ledger_private_is_closed(const struct wt_ledger *ledger,
                                uint32_t domain, uint64_t epoch) {
    size_t i;
    for (i = 0; i < ledger->closed_count; ++i) {
        if (ledger->closed[i].used && ledger->closed[i].domain == domain &&
            ledger->closed[i].epoch == epoch) {
            return 1;
        }
    }
    return 0;
}

static int same_logical_key(const struct wt_work_id *a,
                            const struct wt_work_id *b) {
    return memcmp(a->run_uuid, b->run_uuid, sizeof(a->run_uuid)) == 0 &&
           a->epoch == b->epoch && a->domain == b->domain &&
           a->phase == b->phase && a->call_seq == b->call_seq &&
           a->chunk == b->chunk && a->owner_rank == b->owner_rank &&
           a->owner_generation == b->owner_generation;
}

struct wt_ledger_entry *wt_ledger_private_find_key(
    struct wt_ledger *ledger, const struct wt_work_id *id, size_t *slot) {
    size_t i;
    for (i = 0; i < ledger->max_entries; ++i) {
        if (ledger->entries[i].used &&
            same_logical_key(&ledger->entries[i].id, id)) {
            if (slot != NULL) {
                *slot = i;
            }
            return &ledger->entries[i];
        }
    }
    return NULL;
}

void wt_ledger_private_make_ticket(const struct wt_ledger *ledger,
                                   size_t slot,
                                   struct wt_ledger_ticket *ticket) {
    ticket->ledger_identity = ledger->identity;
    ticket->generation = ledger->generation;
    ticket->nonce = ledger->entries[slot].nonce;
    ticket->slot = (uint32_t)slot;
}

int wt_ledger_private_ticket_entry(const struct wt_ledger *ledger,
                                   const struct wt_ledger_ticket *ticket,
                                   struct wt_ledger_entry **entry) {
    if (ledger == NULL || ticket == NULL || entry == NULL) {
        return WT_LEDGER_EINVAL;
    }
    if (ticket->generation != ledger->generation) {
        return WT_LEDGER_ESTALE;
    }
    if (ticket->ledger_identity != ledger->identity ||
        (size_t)ticket->slot >= ledger->max_entries ||
        !ledger->entries[ticket->slot].used ||
        ledger->entries[ticket->slot].nonce != ticket->nonce) {
        return WT_LEDGER_EFOREIGN;
    }
    *entry = &ledger->entries[ticket->slot];
    return WT_LEDGER_OK;
}
