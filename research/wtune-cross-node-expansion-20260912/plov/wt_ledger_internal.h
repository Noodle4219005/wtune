#ifndef WT_LEDGER_INTERNAL_H
#define WT_LEDGER_INTERNAL_H

#include "wt_ledger.h"

struct wt_ledger_entry {
    int used;
    enum wt_ledger_state state;
    struct wt_work_id id;
    uint8_t request_fingerprint[WT_WIRE_FINGERPRINT_BYTES];
    uint8_t *request;
    size_t request_len;
    uint8_t *result;
    size_t result_capacity;
    size_t result_len;
    uint64_t nonce;
};

struct wt_closed_epoch {
    int used;
    uint32_t domain;
    uint64_t epoch;
};

struct wt_ledger {
    enum wt_ledger_role role;
    struct wt_ledger_contract contract;
    size_t max_entries;
    size_t used_entries;
    size_t max_live_bytes;
    size_t live_bytes;
    struct wt_ledger_entry *entries;
    struct wt_closed_epoch *closed;
    size_t closed_count;
    uint64_t generation;
    uint64_t next_nonce;
    unsigned int callback_depth;
    uint64_t identity;
};

int wt_ledger_private_valid_contract(
    const struct wt_ledger_contract *contract);
int wt_ledger_private_same_identity(
    const struct wt_ledger_contract *a,
    const struct wt_ledger_contract *b);
int wt_ledger_private_decode_frame(const struct wt_ledger *ledger,
                                   const uint8_t *bytes, size_t length,
                                   uint8_t kind,
                                   struct wt_wire_frame *frame);
int wt_ledger_private_is_closed(const struct wt_ledger *ledger,
                                uint32_t domain, uint64_t epoch);
struct wt_ledger_entry *wt_ledger_private_find_key(
    struct wt_ledger *ledger, const struct wt_work_id *id, size_t *slot);
void wt_ledger_private_make_ticket(const struct wt_ledger *ledger,
                                   size_t slot,
                                   struct wt_ledger_ticket *ticket);
int wt_ledger_private_ticket_entry(const struct wt_ledger *ledger,
                                   const struct wt_ledger_ticket *ticket,
                                   struct wt_ledger_entry **entry);

#endif
