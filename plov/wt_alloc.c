/* wt_alloc.c — record every ifort ALLOCATE so a stager can ask "how many
 * bytes does this pointer really own?" without touching heap metadata.
 *
 * WRF allocates every grid% field through the ifort RTL entry
 *   int for_allocate_handle(size_t nbytes, void **base, int flags, void *handle)
 * (inc/allocs_*.o: rdi=nbytes, rsi=&descriptor.base, edx=flags, rcx=handle;
 * 510 call sites in allocs_8.o alone).  Inactive Registry-package fields are
 * allocated with nbytes=4 (ALLOCATE(grid%f(1,1,1))), active ones with the
 * exact ims:ime,kms:kme,jms:jme product times 4.  Linked with
 * -Wl,--wrap=for_allocate_handle this wrapper sees them all.
 *
 * glibc malloc_usable_size() is NOT usable for this: ifort's pointer is not a
 * malloc chunk start and job 1070388 died with SIGSEGV inside it on all 224
 * ranks.  This table answers exactly, or says "unknown" (0). */
#include <stddef.h>
#include <stdint.h>

extern int __real_for_allocate_handle(size_t, void **, int, void *);

#define WT_ALLOC_BITS 18                /* 262144 slots x 16 B = 4 MiB per rank */
#define WT_ALLOC_N (1u << WT_ALLOC_BITS)

static struct { const void *p; size_t n; } wt_alloc_tab[WT_ALLOC_N];
static volatile int wt_alloc_lock;
unsigned long wt_alloc_calls, wt_alloc_dropped;   /* dropped = table full */

static unsigned wt_alloc_hash(const void *p)
{
    uint64_t x = (uint64_t)(uintptr_t)p >> 4;
    x *= 0x9E3779B97F4A7C15ull;
    return (unsigned)(x >> (64 - WT_ALLOC_BITS));
}

static void wt_alloc_insert(const void *p, size_t n)
{
    unsigned h = wt_alloc_hash(p), i;
    while (__sync_lock_test_and_set(&wt_alloc_lock, 1)) ;
    for (i = 0; i < WT_ALLOC_N; i++) {
        unsigned s = (h + i) & (WT_ALLOC_N - 1);
        if (!wt_alloc_tab[s].p || wt_alloc_tab[s].p == p) {   /* new or reused address */
            wt_alloc_tab[s].p = p; wt_alloc_tab[s].n = n; break;
        }
    }
    if (i == WT_ALLOC_N) wt_alloc_dropped++;
    __sync_lock_release(&wt_alloc_lock);
}

/* Bytes recorded for the allocation whose data starts exactly at p; 0 if p is
 * not a recorded ALLOCATE base (stack automatic, slab of a 4D array, ...). */
size_t wt_alloc_size(const void *p)
{
    unsigned h = wt_alloc_hash(p), i;
    size_t n = 0;
    if (!p) return 0;
    while (__sync_lock_test_and_set(&wt_alloc_lock, 1)) ;
    for (i = 0; i < WT_ALLOC_N; i++) {
        unsigned s = (h + i) & (WT_ALLOC_N - 1);
        if (!wt_alloc_tab[s].p) break;
        if (wt_alloc_tab[s].p == p) { n = wt_alloc_tab[s].n; break; }
    }
    __sync_lock_release(&wt_alloc_lock);
    return n;
}

int __wrap_for_allocate_handle(size_t nbytes, void **base, int flags, void *handle)
{
    int rc = __real_for_allocate_handle(nbytes, base, flags, handle);
    wt_alloc_calls++;
    if (rc == 0 && base && *base) wt_alloc_insert(*base, nbytes);
    return rc;
}
