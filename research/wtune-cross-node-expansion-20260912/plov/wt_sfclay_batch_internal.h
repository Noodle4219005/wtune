#ifndef WT_SFCLAY_BATCH_INTERNAL_H
#define WT_SFCLAY_BATCH_INTERNAL_H

#include "wt_sfclay_batch.h"

#define SFB_GUARD 64u
#define SFB_EDGE_GUARD 16u
#define SFB_FILL 0xa7

enum {
    SFB_IMS = 64, SFB_IME, SFB_JMS, SFB_JME, SFB_KMS, SFB_KME,
    SFB_ITS, SFB_ITE, SFB_JTS, SFB_JTE, SFB_KTS, SFB_KTE
};

struct sfb_header {
    uint64_t schema, bytes;
    uint32_t state, reserved;
    int32_t scalar[WT_SFCLAY_NARG];
    unsigned char alignment[WT_SFCLAY_NARG];
    unsigned char present[WT_SFCLAY_NARG];
};

struct sfb_field {
    size_t full, packed, offset, first;
    int writable;
};

struct sfb_layout {
    struct sfb_field field[WT_SFCLAY_NARG];
    size_t bytes, heap;
};

int sfb_add(size_t, size_t, size_t *);
int sfb_mul(size_t, size_t, size_t *);
int sfb_interval(const void *, size_t, uintptr_t *);
enum wt_sfclay_batch_status sfb_layout(const struct sfb_header *,
                                       struct sfb_layout *);
/* direction 0: full -> packed, 1: packed -> full, 2: compare. */
int sfb_transfer(const struct sfb_field *, void *full, void *packed,
                 int direction);
int sfb_writes_only(const struct sfb_header *, unsigned,
                    const struct sfb_field *, const unsigned char *full,
                    const unsigned char *before);

#endif
