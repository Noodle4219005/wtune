#include "wt_sfclay.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(float) == 4, "SFCLAY audit requires four-byte REAL");
_Static_assert(sizeof(int) == 4, "SFCLAY audit requires four-byte INTEGER");

enum { GUARD = 32, TX_PREPARED = 1, TX_USED = 2 };
struct item { void *owner, *clone, *base, *snapshot; size_t bytes; };
struct wt_sfclay_tx { struct item item[WT_SFCLAY_NARG]; void *argv[WT_SFCLAY_NARG]; int state; };

static int add(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}
static int mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}
static int scalar(const void *p, int *v) {
    if (!p) return 0;
    memcpy(v, p, sizeof *v);
    return 1;
}
static int extent(int low, int high, size_t *out) {
    int64_t n = (int64_t)high - low + 1;
    if (n <= 0 || (uint64_t)n > SIZE_MAX) return 0;
    *out = (size_t)n;
    return 1;
}
static int layout(void *const *argv, size_t *n3, size_t *n2) {
    int ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte;
    size_t ni, nj, nk, n;
    if (!scalar(argv[64],&ims)||!scalar(argv[65],&ime)||!scalar(argv[66],&jms)||
        !scalar(argv[67],&jme)||!scalar(argv[68],&kms)||!scalar(argv[69],&kme)||
        !scalar(argv[70],&its)||!scalar(argv[71],&ite)||!scalar(argv[72],&jts)||
        !scalar(argv[73],&jte)||!scalar(argv[74],&kts)||!scalar(argv[75],&kte)) return WT_SFCLAY_BAD_ARGUMENT;
    /* SFCLAY requires level 1 in memory; its tile may legitimately begin above
     * level 1 because the callee's surface access is not tile-k bounded. */
    if (!extent(ims,ime,&ni)||!extent(jms,jme,&nj)||!extent(kms,kme,&nk) ||
        its<ims||ite>ime||its>ite||jts<jms||jte>jme||jts>jte||kts>kte||kts<kms||kte>kme||kms>1||kme<1) return WT_SFCLAY_BOUNDS;
    if (!mul(ni,nk,&n)||!mul(n,nj,&n)||!mul(n,4,n3)||!mul(ni,nj,&n)||!mul(n,4,n2)) return WT_SFCLAY_OVERFLOW;
    return WT_SFCLAY_OK;
}
static int interval(const void *p, size_t n, uintptr_t *lo, uintptr_t *hi) {
    uintptr_t v = (uintptr_t)p;
    if (!p || n == 0 || v > UINTPTR_MAX - n) return 0;
    *lo=v; *hi=v+n; return 1;
}
static int overlaps(const struct item *a, const struct item *b) {
    uintptr_t al,ah,bl,bh;
    return interval(a->owner,a->bytes,&al,&ah) && interval(b->owner,b->bytes,&bl,&bh) && al<bh && bl<ah;
}
static int guards(const struct item *x) {
    unsigned char *b=x->base;
    for (size_t i=0;i<GUARD;i++) if (b[i]!=0xa5 || b[GUARD+x->bytes+i]!=0x5a) return 0;
    return 1;
}
static int preflight(void *const *argv, const size_t *cap) {
    for (size_t i=0;i<WT_SFCLAY_NARG;i++) {
        uintptr_t lo, hi;
        if (!argv[i]) {
            if (i>=81 && i<84) continue;
            return WT_SFCLAY_BAD_ARGUMENT;
        }
        if (cap[i] < 4) return WT_SFCLAY_CAPACITY;
        if (!interval(argv[i], 4, &lo, &hi)) return WT_SFCLAY_OVERFLOW;
    }
    return WT_SFCLAY_OK;
}

int wt_sfclay_prepare(void *const argv[WT_SFCLAY_NARG],
                      const size_t cap[WT_SFCLAY_NARG], size_t budget,
                      wt_sfclay_tx **out)
{
    wt_sfclay_tx *tx; struct item item[WT_SFCLAY_NARG] = {{0}};
    size_t n3,n2,total=sizeof *tx; int status;
    if (!out) return WT_SFCLAY_BAD_ARGUMENT;
    *out=NULL;
    if (!argv || !cap) return WT_SFCLAY_BAD_ARGUMENT;
    status=preflight(argv,cap); if (status != WT_SFCLAY_OK) return status;
    status=layout(argv,&n3,&n2); if (status != WT_SFCLAY_OK) return status;
    for (size_t i=0;i<WT_SFCLAY_NARG;i++) {
        struct item *x=&item[i]; size_t alloc, copies;
        if (!argv[i]) continue;
        x->owner=argv[i]; x->bytes=wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_3D?n3:
                                  wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_2D?n2:4;
        if (cap[i]<x->bytes) return WT_SFCLAY_CAPACITY;
        { uintptr_t lo,hi; if (!interval(x->owner,x->bytes,&lo,&hi)) return WT_SFCLAY_OVERFLOW; }
        if (!add(x->bytes,GUARD*2,&alloc) || !add(alloc,x->bytes,&copies) ||
            !add(total,copies,&total)) return WT_SFCLAY_OVERFLOW;
        if (total > budget) return WT_SFCLAY_BUDGET;
    }
    for (size_t i=0;i<WT_SFCLAY_NARG;i++) for (size_t j=i+1;j<WT_SFCLAY_NARG;j++)
        if (item[i].owner && item[j].owner && overlaps(&item[i],&item[j]) &&
            (wt_sfclay_schema_access[i]==WT_SFCLAY_WRITABLE || wt_sfclay_schema_access[j]==WT_SFCLAY_WRITABLE)) {
            return WT_SFCLAY_OVERLAP;
        }
    tx=calloc(1,sizeof *tx); if (!tx) return WT_SFCLAY_BUDGET;
    for (size_t i=0;i<WT_SFCLAY_NARG;i++) {
        struct item *x=&tx->item[i]; size_t alloc;
        if (!item[i].owner) continue;
        *x=item[i]; add(x->bytes,GUARD*2,&alloc);
        x->base=malloc(alloc); x->snapshot=malloc(x->bytes);
        if (!x->base || !x->snapshot) { wt_sfclay_destroy(tx); return WT_SFCLAY_BUDGET; }
        memset(x->base,0xa5,GUARD); memset((unsigned char*)x->base+GUARD+x->bytes,0x5a,GUARD);
        x->clone=(unsigned char*)x->base+GUARD; memcpy(x->clone,x->owner,x->bytes); memcpy(x->snapshot,x->owner,x->bytes);
    }
    for (size_t i=0;i<WT_SFCLAY_NARG;i++) tx->argv[i]=tx->item[i].clone;
    tx->state=TX_PREPARED; *out=tx; return WT_SFCLAY_OK;
}

void **wt_sfclay_cloned_argv(wt_sfclay_tx *tx) {
    if (!tx || tx->state!=TX_PREPARED) return NULL;
    return tx->argv;
}
int wt_sfclay_commit(wt_sfclay_tx *tx) {
    if (!tx || tx->state!=TX_PREPARED) return WT_SFCLAY_STATE;
    tx->state=TX_USED;
    for (size_t i=0;i<WT_SFCLAY_NARG;i++) { struct item *x=&tx->item[i];
        if (!x->owner) continue;
        if (!guards(x) || memcmp(x->owner,x->snapshot,x->bytes) ||
            (wt_sfclay_schema_access[i]==WT_SFCLAY_READONLY && memcmp(x->clone,x->snapshot,x->bytes))) return WT_SFCLAY_STALE;
    }
    for (size_t i=0;i<WT_SFCLAY_NARG;i++) { struct item *x=&tx->item[i];
        if (x->owner && wt_sfclay_schema_access[i]==WT_SFCLAY_WRITABLE) memcpy(x->owner,x->clone,x->bytes);
    }
    return WT_SFCLAY_OK;
}
void wt_sfclay_destroy(wt_sfclay_tx *tx) {
    if (!tx) return;
    for (size_t i=0;i<WT_SFCLAY_NARG;i++) { free(tx->item[i].base); free(tx->item[i].snapshot); }
    free(tx);
}
