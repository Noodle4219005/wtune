#ifndef WT_SFCLAY_BOOTSTRAP_H
#define WT_SFCLAY_BOOTSTRAP_H

#include <stddef.h>
#include <stdio.h>
#include "wt_sfclay_live.h"

enum wt_sfclay_bootstrap_reason {
    WT_SFCLAY_BOOT_NOT_STARTED = 0,
    WT_SFCLAY_BOOT_DISABLED,
    WT_SFCLAY_BOOT_INVALID_SELECTOR,
    WT_SFCLAY_BOOT_INVALID_BUDGET,
    WT_SFCLAY_BOOT_INVALID_PROFILE,
    WT_SFCLAY_BOOT_SOURCE_ABI,
    WT_SFCLAY_BOOT_CALLER_CHAIN,
    WT_SFCLAY_BOOT_INITIALIZER_RELOCATION,
    WT_SFCLAY_BOOT_NATIVE_REJECTED,
    WT_SFCLAY_BOOT_CONFIGURED
};

struct wt_sfclay_bootstrap_snapshot {
    int started;
    int requested;
    int selector;
    int configured;
    int startup_state;
    int source_abi_verified;
    int caller_chain_verified;
    int initializer_relocation_verified;
    size_t max_spans;
    size_t max_nodes;
    size_t byte_budget;
    enum wt_sfclay_bootstrap_reason reason;
    struct wt_sfclay_live_counters counters;
};

/* Call once from a serialized startup path; there is deliberately no
 * constructor.  requested <= 0 records a disabled decision.  The first call,
 * including a refusal or failed native installation, is terminal; later calls
 * return that original configure result and cannot change the selector or
 * budgets.  Environment variables tune only the three positive decimal
 * budgets and never attest a proof.
 *
 * state is 0 for disabled/refused/not-started, -1 for a configured runtime
 * with no observed staged call, and 1 only after the live staged counter is
 * positive.  get_state snapshots the actual live counters at call time.
 */
int wt_sfclay_bootstrap_start(int requested, int grid_selector);
int wt_sfclay_bootstrap_state(void);
int wt_sfclay_bootstrap_get_state(
    struct wt_sfclay_bootstrap_snapshot *out);
const char *wt_sfclay_bootstrap_reason_name(
    enum wt_sfclay_bootstrap_reason reason);
void wt_sfclay_bootstrap_dump(FILE *stream);

#endif
