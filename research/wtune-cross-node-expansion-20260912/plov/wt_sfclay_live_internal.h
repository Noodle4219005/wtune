#ifndef WT_SFCLAY_LIVE_INTERNAL_H
#define WT_SFCLAY_LIVE_INTERNAL_H
#include "wt_sfclay_live.h"
#include <pthread.h>

struct wt_sf_live_tls {
    uint64_t outer, surface, generation;
    /* Assigned only by successful configure; never transferred or scope-cleared. */
    uint64_t owner_nonce;
    void *grid, *config;
    struct wt_sfclay_native_result result;
    enum wt_sfclay_live_reason reason;
    int outer_ok, surface_ok;
    unsigned dispatch_depth, control_depth, init_depth, reader_depth;
    unsigned cancel_depth;
    int saved_cancel;
};
extern _Thread_local struct wt_sf_live_tls wt_sf_tls;
extern pthread_mutex_t wt_sf_lifecycle_mutex;
extern unsigned wt_sf_active;
_Noreturn void wt_sf_die(void);
void wt_sf_pcheck(int status);
void wt_sf_activity_enter(void);
void wt_sf_activity_leave(void);
/* Reader lease is logical: its mutex is released before invoking any callback.
 * READY and generation are observed together under that mutex. */
int wt_sf_reader_get(uint64_t *generation);
int wt_sf_reader_current(uint64_t generation);
void wt_sf_reader_put(void);
uint64_t wt_sfclay_init_fingerprint(void);
#endif
