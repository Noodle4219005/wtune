#ifndef WT_LEDGER_H
#define WT_LEDGER_H

#include "wt_wire.h"

#include <stddef.h>
#include <stdint.h>

/*
 * WT Ledger is an in-memory, bounded request/result ledger.  Callers must
 * serialize access to one ledger.  It prevents a callback from being invoked
 * twice while that ledger lives, but does not promise exactly-once execution
 * across process failure.  Entries are never evicted or recycled.
 *
 * The logical key is run/owner/generation/domain/phase/epoch/call_seq/chunk.
 * Region bounds remain immutable request metadata: changing only a bound is a
 * conflict, while a result must still reproduce the complete registered ID.
 * Adapters must guarantee unique output ownership; this ledger has no write-set
 * information and therefore does not reject overlapping halo read regions.
 */

enum wt_ledger_status {
    WT_LEDGER_OK = 0,
    WT_LEDGER_DUPLICATE = 1,
    WT_LEDGER_EINVAL = 2,
    WT_LEDGER_EWIRE = 3,
    WT_LEDGER_ECONFLICT = 4,
    WT_LEDGER_ESTATE = 5,
    WT_LEDGER_EFOREIGN = 6,
    WT_LEDGER_ESTALE = 7,
    WT_LEDGER_EBUDGET = 8,
    WT_LEDGER_EFULL = 9,
    WT_LEDGER_ENOMEM = 10,
    WT_LEDGER_ECALLBACK = 11,
    WT_LEDGER_EBUSY = 12
};

enum wt_ledger_role {
    WT_LEDGER_OWNER = 1,
    WT_LEDGER_WORKER = 2
};

enum wt_ledger_state {
    WT_LEDGER_ACCEPTED = 1,
    WT_LEDGER_RUNNING = 2,
    WT_LEDGER_DONE = 3,
    WT_LEDGER_COMMITTING = 4,
    WT_LEDGER_COMMITTED = 5,
    WT_LEDGER_FAILED = 6
};

struct wt_ledger;

struct wt_ledger_contract {
    uint8_t run_uuid[16];
    uint32_t owner_rank;
    uint32_t owner_generation;
    uint8_t schema_fingerprint[WT_WIRE_FINGERPRINT_BYTES];
    struct wt_payload_abi abi;
    struct wt_wire_limits limits;
};

/*
 * Opaque capability.  Every field is checked against private ledger data.
 * Ledger identities are process-unique and creation returns WT_LEDGER_EFULL
 * instead of wrapping the identity counter.
 */
struct wt_ledger_ticket {
    uint64_t ledger_identity;
    uint64_t generation;
    uint64_t nonce;
    uint32_t slot;
};

/*
 * result points to ledger-owned storage of exactly result_capacity bytes.
 * Set result_len to the complete encoded WT_WIRE_RESULT length.  A nonzero
 * return or invalid result permanently fails the request.
 */
typedef int (*wt_ledger_run_callback)(void *context,
                                      const uint8_t *request,
                                      size_t request_len,
                                      uint8_t *result,
                                      size_t result_capacity,
                                      size_t *result_len);

/*
 * The decoded result is backed by a private ledger copy.  Returning zero
 * asserts that the trusted callback applied the result atomically.  A nonzero
 * return permanently fails the request and is never retried by this ledger.
 */
typedef int (*wt_ledger_commit_callback)(void *context,
                                         const struct wt_wire_frame *result);

/*
 * max_entries separately bounds the two fixed metadata tables allocated at
 * creation.  max_live_bytes bounds private encoded request/result buffers;
 * each acceptance checks its complete reservation before allocating either.
 * A request_fingerprint may be all zero because it is caller-supplied metadata;
 * complete encoded request bytes remain the duplicate authority.
 */
int wt_ledger_create(enum wt_ledger_role role,
                     const struct wt_ledger_contract *contract,
                     size_t max_entries, size_t max_live_bytes,
                     struct wt_ledger **out);

/* Register before publishing request bytes. */
int wt_ledger_owner_register_request(struct wt_ledger *ledger,
                                     const uint8_t *request,
                                     size_t request_len,
                                     size_t expected_result_limit,
                                     struct wt_ledger_ticket *ticket);

int wt_ledger_worker_accept(struct wt_ledger *ledger,
                            const uint8_t *request, size_t request_len,
                            size_t expected_result_limit,
                            struct wt_ledger_ticket *ticket);

int wt_ledger_worker_run(struct wt_ledger *ledger,
                         const struct wt_ledger_ticket *ticket,
                         wt_ledger_run_callback callback, void *context);

int wt_ledger_owner_apply_result(struct wt_ledger *ledger,
                                 const struct wt_ledger_ticket *ticket,
                                 const uint8_t *result, size_t result_len,
                                 wt_ledger_commit_callback callback,
                                 void *context);

/* Explicitly mark accepted work failed; there is no automatic worker retry. */
int wt_ledger_fail(struct wt_ledger *ledger,
                   const struct wt_ledger_ticket *ticket);

int wt_ledger_get_state(const struct wt_ledger *ledger,
                        const struct wt_ledger_ticket *ticket,
                        enum wt_ledger_state *state);

/* Borrowed private bytes remain valid until reset or destruction. */
int wt_ledger_request_bytes(const struct wt_ledger *ledger,
                            const struct wt_ledger_ticket *ticket,
                            const uint8_t **request, size_t *request_len);
int wt_ledger_result_bytes(const struct wt_ledger *ledger,
                           const struct wt_ledger_ticket *ticket,
                           const uint8_t **result, size_t *result_len);

/* Closing is permanent until a valid reset and also rejects old duplicates. */
int wt_ledger_close_epoch(struct wt_ledger *ledger, uint32_t domain,
                          uint64_t epoch);

/*
 * Requires quiescence and a run UUID/rank/generation tuple different from the
 * current tuple.  The caller must supply a globally unique new identity; reset
 * invalidates all tickets and clears retained entries and closed epochs.
 * An exhausted ticket generation returns WT_LEDGER_EFULL without clearing.
 */
int wt_ledger_reset(struct wt_ledger *ledger,
                    const struct wt_ledger_contract *new_contract);

/* On success frees the ledger and sets *ledger to NULL; busy callbacks refuse. */
int wt_ledger_destroy(struct wt_ledger **ledger);

#endif
