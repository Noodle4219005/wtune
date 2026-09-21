#ifndef WT_MYNN_H
#define WT_MYNN_H

#include "wt_mynn_layout.h"

typedef struct wt_mynn_tx wt_mynn_tx;
enum wt_mynn_status {
    WT_MYNN_OK = 0, WT_MYNN_ARGUMENT, WT_MYNN_LAYOUT, WT_MYNN_BUDGET,
    WT_MYNN_ALIAS, WT_MYNN_ALLOCATION, WT_MYNN_STATE, WT_MYNN_STALE,
    WT_MYNN_REBIND, WT_MYNN_GUARD, WT_MYNN_READONLY, WT_MYNN_OVERFLOW
};
enum wt_mynn_state {
    WT_MYNN_PREPARED, WT_MYNN_ENTERED, WT_MYNN_EXECUTED,
    WT_MYNN_COMMITTED, WT_MYNN_FAILED
};

/* Owner-local prototype, NOT native MYNN admission. No fallback exists here.
 * The caller supplies the separately attested layout ABI and truthful capacities.
 * Owner thread must remain alive; the transaction and all owner spans, argv and
 * capacity tables remain valid and exclusively owned until destroy. Concurrent
 * access, asynchronous callbacks, longjmp and cancellation across execute are
 * unsupported. The callback may access only its supplied argument views, must
 * return normally, and must not retain the temporary argv table.
 *
 * prepare clears *out on ordinary failure; structural reentry preserves *out
 * so the caller can still destroy the poisoned outer transaction. *out must not alias an input/owner. It clones
 * all present views, including INTENT(OUT), scalars and inactive 4-byte views.
 * Strictly overlapping equal-permission views share a region. Mixed writable /
 * read-only overlaps are refused. Merely adjacent intervals remain separate.
 * A positive total_budget caps all owned heap bytes: transaction metadata plus
 * two region copies and two 32-byte guards per region. No unbudgeted heap work;
 * fixed bounded preparation/callback stack scratch is not heap allocation.
 * Native numerical/alias/ABI correctness still requires separate attestation.
 *
 * Successful path: PREPARED -> ENTERED -> EXECUTED -> COMMITTED. Owner staleness
 * before entry fails with zero callback calls. After ENTERED, ANY error must be
 * treated as fatal by the caller; never retry, execute native fallback, or replay.
 * All guards/read-only copies/owner snapshots are checked before any commit write.
 * Commit writes each writable union once and does not provide concurrent atomicity.
 *
 * Same-thread invalid transitions poison FAILED, except terminal COMMITTED stays
 * terminal. Structural reentry (including prepare of another transaction) poisons
 * the current ENTERED transaction before reading new arguments or allocating.
 * Reentrant destroy during ENTERED poisons but does not free; once the
 * outer callback returns, destroy may release private storage. Foreign-thread
 * calls return STATE without reading/writing mutable transaction state. Their
 * get_state result is FAILED (an error result, not a mutation of the transaction).
 * No API accepts a pointer after successful destroy.
 */
enum wt_mynn_status wt_mynn_prepare(void *const argv[WT_MYNN_NARG],
    const size_t capacities[WT_MYNN_NARG], const struct wt_mynn_layout_config *config,
    size_t total_budget, wt_mynn_tx **out);
enum wt_mynn_status wt_mynn_execute_once(wt_mynn_tx *, void (*real)(void **));
enum wt_mynn_status wt_mynn_commit(wt_mynn_tx *);
enum wt_mynn_status wt_mynn_destroy(wt_mynn_tx *);
enum wt_mynn_state wt_mynn_get_state(const wt_mynn_tx *);

#endif
