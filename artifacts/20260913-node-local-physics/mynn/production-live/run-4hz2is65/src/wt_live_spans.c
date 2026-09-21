#include "wt_live_spans.h"
#include <stdlib.h>

struct span { uintptr_t first, end; size_t width; int type, rank; };
struct wt_live_spans {
    struct span *entries;
    size_t count, limit;
    uint64_t generation;
    enum { ENDED, COLLECTING, FROZEN, INVALID } state;
};

wt_live_spans *wt_live_spans_create(size_t max_spans)
{
    if (!max_spans || max_spans > SIZE_MAX / sizeof(struct span)) return NULL;
    wt_live_spans *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->entries = calloc(max_spans, sizeof *s->entries);
    if (!s->entries) { free(s); return NULL; }
    s->limit = max_spans;
    return s;
}

void wt_live_spans_destroy(wt_live_spans *s)
{
    if (s) { free(s->entries); free(s); }
}

uint64_t wt_live_spans_begin(wt_live_spans *s)
{
    if (!s) return 0;
    s->count = 0;
    s->state = INVALID;
    if (s->generation == UINT64_MAX) return 0;
    ++s->generation;
    s->state = COLLECTING;
    return s->generation;
}

void wt_live_spans_fail(void *context)
{
    wt_live_spans *s = context;
    if (s) { s->count = 0; s->state = INVALID; }
}

int wt_live_spans_add(void *context, void *base, size_t bytes,
                     int type, size_t width, int rank)
{
    wt_live_spans *s = context;
    uintptr_t first = (uintptr_t)base;
    if (!s) return WT_SPAN_INVALID;
    if (s->state != COLLECTING || !base || !bytes || !width ||
        type < WT_SPAN_REAL || type > WT_SPAN_LOGICAL ||
        rank < 0 || rank > 7 || (type == WT_SPAN_LOGICAL && rank > 2) ||
        bytes % width || first % width || bytes > UINTPTR_MAX - first)
        goto invalid;
    uintptr_t end = first + bytes;
    for (size_t i = 0; i < s->count; ++i) {
        const struct span *p = &s->entries[i];
        if (first < p->end && p->first < end &&
            (type != p->type || width != p->width)) goto invalid;
        if (first == p->first && end == p->end &&
            type == p->type && width == p->width) return WT_SPAN_OK;
    }
    if (s->count == s->limit) goto invalid;
    s->entries[s->count++] = (struct span){first, end, width, type, rank};
    return WT_SPAN_OK;
invalid:
    wt_live_spans_fail(s);
    return WT_SPAN_INVALID;
}

int wt_live_spans_finish(void *context)
{
    wt_live_spans *s = context;
    if (!s || s->state != COLLECTING) return WT_SPAN_INVALID;
    s->state = FROZEN;
    return WT_SPAN_OK;
}

void wt_live_spans_end(wt_live_spans *s)
{
    if (s) { s->count = 0; s->state = ENDED; }
}

int wt_live_spans_is_current(const wt_live_spans *s, uint64_t generation)
{
    return s && s->state == FROZEN && generation &&
           generation == s->generation;
}

int wt_live_spans_lookup(const wt_live_spans *s, uint64_t generation,
                        const void *pointer, int type, size_t width,
                        size_t required, size_t *remaining)
{
    if (remaining) *remaining = 0;
    if (!s || !pointer || !remaining || !width || !required ||
        required % width || s->state != FROZEN || !generation ||
        generation != s->generation) return WT_SPAN_UNKNOWN;
    uintptr_t address = (uintptr_t)pointer;
    for (size_t i = 0; i < s->count; ++i) {
        const struct span *p = &s->entries[i];
        if (p->type != type || p->width != width || address < p->first ||
            address >= p->end || (address - p->first) % width) continue;
        size_t available = p->end - address;
        if (available >= required && available > *remaining) *remaining = available;
    }
    return *remaining ? WT_SPAN_OK : WT_SPAN_UNKNOWN;
}
