#include "wt_sfclay_live_internal.h"
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>

#ifdef WT_SFCLAY_NODE_LOCAL
#ifndef WT_SFCLAY_PSIMTB_F77
#define WT_SFCLAY_PSIMTB_F77 module_sf_sfclay_mp_psimtb_
#endif
#ifndef WT_SFCLAY_PSIHTB_F77
#define WT_SFCLAY_PSIHTB_F77 module_sf_sfclay_mp_psihtb_
#endif
extern float WT_SFCLAY_PSIMTB_F77[1001];
extern float WT_SFCLAY_PSIHTB_F77[1001];
#endif

_Thread_local struct wt_sf_live_tls wt_sf_tls;
pthread_mutex_t wt_sf_lifecycle_mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned wt_sf_active;
static pthread_mutex_t lease_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t lease_changed = PTHREAD_COND_INITIALIZER;
static unsigned readers, waiting_writers;
static int writer_active, ready;
static uint64_t init_generation;
static _Atomic uint64_t init_fingerprint;

uint64_t wt_sfclay_init_fingerprint(void)
{ return atomic_load_explicit(&init_fingerprint,memory_order_acquire); }

#ifdef WT_SFCLAY_NODE_LOCAL
static uint64_t table_fingerprint(void)
{
    const unsigned char *a=(const unsigned char *)WT_SFCLAY_PSIMTB_F77;
    const unsigned char *b=(const unsigned char *)WT_SFCLAY_PSIHTB_F77;
    uint64_t hash=UINT64_C(1469598103934665603);
    size_t i;
    for (i=0;i<1001*sizeof(float);++i) {
        hash^=a[i]; hash*=UINT64_C(1099511628211);
    }
    for (i=0;i<1001*sizeof(float);++i) {
        hash^=b[i]; hash*=UINT64_C(1099511628211);
    }
    return hash ? hash : 1;
}
#endif

void wt_sf_pcheck(int status)
{
    if (status) wt_sf_die();
}

void wt_sf_activity_enter(void)
{
    if (wt_sf_tls.cancel_depth == UINT_MAX) wt_sf_die();
    if (!wt_sf_tls.cancel_depth)
        wt_sf_pcheck(pthread_setcancelstate(PTHREAD_CANCEL_DISABLE,
                                            &wt_sf_tls.saved_cancel));
    ++wt_sf_tls.cancel_depth;
    wt_sf_pcheck(pthread_mutex_lock(&wt_sf_lifecycle_mutex));
    if (wt_sf_active == UINT_MAX) wt_sf_die();
    ++wt_sf_active;
    wt_sf_pcheck(pthread_mutex_unlock(&wt_sf_lifecycle_mutex));
}

void wt_sf_activity_leave(void)
{
    if (!wt_sf_tls.cancel_depth) wt_sf_die();
    wt_sf_pcheck(pthread_mutex_lock(&wt_sf_lifecycle_mutex));
    if (!wt_sf_active) wt_sf_die();
    --wt_sf_active;
    wt_sf_pcheck(pthread_mutex_unlock(&wt_sf_lifecycle_mutex));
    if (!--wt_sf_tls.cancel_depth) {
        int saved = wt_sf_tls.saved_cancel;
        wt_sf_tls.saved_cancel = 0;
        wt_sf_pcheck(pthread_setcancelstate(saved, NULL));
    }
}

int wt_sf_reader_get(uint64_t *generation)
{
    int initialized;
    if (!wt_sf_tls.cancel_depth || wt_sf_tls.reader_depth ||
        wt_sf_tls.init_depth) wt_sf_die();
    wt_sf_pcheck(pthread_mutex_lock(&lease_mutex));
    /* Writer preference blocks new native fallbacks as well as new stages. */
    while (writer_active || waiting_writers)
        wt_sf_pcheck(pthread_cond_wait(&lease_changed, &lease_mutex));
    if (readers == UINT_MAX) wt_sf_die();
    ++readers;
    wt_sf_tls.reader_depth = 1;
    initialized = ready;
    *generation = init_generation;
    wt_sf_pcheck(pthread_mutex_unlock(&lease_mutex));
    return initialized;
}

int wt_sf_reader_current(uint64_t generation)
{
    int current;
    wt_sf_pcheck(pthread_mutex_lock(&lease_mutex));
    current = wt_sf_tls.reader_depth == 1 && readers && !writer_active &&
              ready && generation && generation == init_generation;
    wt_sf_pcheck(pthread_mutex_unlock(&lease_mutex));
    return current;
}

void wt_sf_reader_put(void)
{
    wt_sf_pcheck(pthread_mutex_lock(&lease_mutex));
    if (wt_sf_tls.reader_depth != 1 || !readers || writer_active) wt_sf_die();
    --readers;
    wt_sf_tls.reader_depth = 0;
    if (!readers) wt_sf_pcheck(pthread_cond_broadcast(&lease_changed));
    wt_sf_pcheck(pthread_mutex_unlock(&lease_mutex));
}

void wt_sfclay_init_dispatch(int *allowed, void (*real)(int *))
{
    if (!allowed || !real || wt_sf_tls.control_depth ||
        wt_sf_tls.dispatch_depth || wt_sf_tls.reader_depth ||
        wt_sf_tls.init_depth) wt_sf_die();
    wt_sf_activity_enter();
    wt_sf_tls.init_depth = 1;
    wt_sf_pcheck(pthread_mutex_lock(&lease_mutex));
    if (waiting_writers == UINT_MAX) wt_sf_die();
    ++waiting_writers;
#ifdef WT_SFCLAY_NODE_LOCAL
    /* A node-live reader may be the current native callback. Waiting here
     * would deadlock that callback; reinitialization during a live lease is a
     * structural failure instead. */
    if (readers) wt_sf_die();
#endif
    while (writer_active || readers)
        wt_sf_pcheck(pthread_cond_wait(&lease_changed, &lease_mutex));
    --waiting_writers;
    if (init_generation == UINT64_MAX) wt_sf_die();
    writer_active = 1;
    ready = 0;
    wt_sf_pcheck(pthread_mutex_unlock(&lease_mutex));

    /* Forward the original Fortran LOGICAL address, once, on any thread. */
    real(allowed);

    wt_sf_pcheck(pthread_mutex_lock(&lease_mutex));
    ++init_generation;
#ifdef WT_SFCLAY_NODE_LOCAL
    atomic_store_explicit(&init_fingerprint,table_fingerprint(),
                          memory_order_release);
#else
    atomic_store_explicit(&init_fingerprint,init_generation,
                          memory_order_release);
#endif
    ready = 1;
    writer_active = 0;
    wt_sf_pcheck(pthread_cond_broadcast(&lease_changed));
    wt_sf_pcheck(pthread_mutex_unlock(&lease_mutex));
    wt_sf_tls.init_depth = 0;
    wt_sf_activity_leave();
}
