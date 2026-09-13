#ifndef WT_KF_BATCH_INTERNAL_H
#define WT_KF_BATCH_INTERNAL_H
#include "wt_kf_batch.h"
#define KFB_GUARD 64u
#define KFB_FILL 0xa7
struct kfb_header {
    uint64_t schema, bytes;
    uint32_t state, reserved;
    int32_t scalar[WT_KF_NARG];
    unsigned char alignment[WT_KF_NARG];
    unsigned char present[WT_KF_NARG];
};
struct kfb_field {
    size_t full, packed, offset, row, rows, first, stride, plane;
    int writable;
};
struct kfb_layout {
    struct kfb_field field[WT_KF_NARG];
    size_t bytes, heap;
};
int kfb_add(size_t, size_t, size_t *);
int kfb_mul(size_t, size_t, size_t *);
int kfb_interval(const void *, size_t, uintptr_t *);
enum wt_kf_status kfb_layout(const struct kfb_header *, struct kfb_layout *);
/* direction: 0 full -> packed, 1 packed -> full, 2 compare */
int kfb_transfer(const struct kfb_header *, unsigned,
    const struct kfb_field *, void *full, void *packed, int direction);
int kfb_writes_only(const struct kfb_header *, unsigned,
    const struct kfb_field *, const unsigned char *full,
    const unsigned char *before);
#endif
