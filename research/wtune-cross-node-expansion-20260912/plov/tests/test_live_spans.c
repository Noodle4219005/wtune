#include "wt_live_spans.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#ifndef WT_TEST_C_ONLY
extern int fixture_grid(void);
#endif

int main(void)
{
    float backing[24] = {0};
    size_t remaining = 999;
    wt_live_spans *s = wt_live_spans_create(2);
    assert(s);
    assert(wt_live_spans_create(SIZE_MAX) == NULL);
    uint64_t gen = wt_live_spans_begin(s);
    assert(gen);
    assert(!wt_live_spans_is_current(NULL, gen));
    assert(!wt_live_spans_is_current(s, 0));
    assert(!wt_live_spans_is_current(s, gen));
    assert(wt_live_spans_add(s, backing, sizeof backing, WT_SPAN_REAL, 4, 4) == WT_SPAN_OK);
    assert(wt_live_spans_lookup(s, gen, backing, WT_SPAN_REAL, 4, 4, &remaining) == WT_SPAN_UNKNOWN);
    assert(remaining == 0);
    assert(wt_live_spans_add(s, backing, sizeof backing, WT_SPAN_REAL, 4, 4) == WT_SPAN_OK);
    assert(wt_live_spans_finish(s) == WT_SPAN_OK);
    assert(wt_live_spans_is_current(s, gen));
    assert(wt_live_spans_lookup(s, gen, backing + 6, WT_SPAN_REAL, 4, 72, &remaining) == WT_SPAN_OK);
    assert(remaining == 72);
    assert(wt_live_spans_lookup(s, gen, backing + 6, WT_SPAN_REAL, 4, 76, &remaining) == WT_SPAN_UNKNOWN);
    assert(remaining == 0);
    assert(wt_live_spans_lookup(s, gen, backing, WT_SPAN_INTEGER, 4, 4, &remaining) == WT_SPAN_UNKNOWN);
    assert(wt_live_spans_lookup(s, gen, backing, WT_SPAN_REAL, 8, 8, &remaining) == WT_SPAN_UNKNOWN);
    assert(wt_live_spans_lookup(s, gen, (char *)backing + 1, WT_SPAN_REAL, 4, 4, &remaining) == WT_SPAN_UNKNOWN);
    assert(wt_live_spans_lookup(s, gen, backing + 24, WT_SPAN_REAL, 4, 4, &remaining) == WT_SPAN_UNKNOWN);
    wt_live_spans_end(s);
    assert(!wt_live_spans_is_current(s, gen));
    assert(wt_live_spans_lookup(s, gen, backing, WT_SPAN_REAL, 4, 4, &remaining) == WT_SPAN_UNKNOWN);
    uint64_t next = wt_live_spans_begin(s);
    assert(next != gen);
    assert(!wt_live_spans_is_current(s, gen));
    assert(!wt_live_spans_is_current(s, next));
    assert(wt_live_spans_add(s, backing, sizeof backing, WT_SPAN_REAL, 4, 1) == WT_SPAN_OK);
    assert(wt_live_spans_finish(s) == WT_SPAN_OK);
    assert(!wt_live_spans_is_current(s, gen));
    assert(wt_live_spans_is_current(s, next));
    assert(wt_live_spans_lookup(s, gen, backing, WT_SPAN_REAL, 4, 4, &remaining) == WT_SPAN_UNKNOWN);
    assert(wt_live_spans_lookup(s, next, backing, WT_SPAN_REAL, 4, 4, &remaining) == WT_SPAN_OK);
    wt_live_spans_end(s);
    assert(wt_live_spans_begin(s));
    assert(wt_live_spans_add(s, backing, sizeof backing, WT_SPAN_REAL, 4, 1) == WT_SPAN_OK);
    assert(wt_live_spans_add(s, backing, sizeof backing, WT_SPAN_INTEGER, 4, 1) == WT_SPAN_INVALID);
    assert(wt_live_spans_finish(s) == WT_SPAN_INVALID);
    assert(wt_live_spans_begin(s));
    assert(wt_live_spans_add(s, (void *)(UINTPTR_MAX - 3), 8, WT_SPAN_REAL, 4, 1) == WT_SPAN_INVALID);
    assert(wt_live_spans_begin(s));
    assert(wt_live_spans_add(s, backing, 3, WT_SPAN_REAL, 4, 1) == WT_SPAN_INVALID);
    assert(wt_live_spans_begin(s));
    assert(wt_live_spans_add(s, backing, 4, WT_SPAN_REAL, 4, 1) == WT_SPAN_OK);
    assert(wt_live_spans_add(s, backing + 1, 4, WT_SPAN_REAL, 4, 1) == WT_SPAN_OK);
    assert(wt_live_spans_add(s, backing + 2, 4, WT_SPAN_REAL, 4, 1) == WT_SPAN_INVALID);
    assert(wt_live_spans_finish(s) == WT_SPAN_INVALID);
    wt_live_spans_destroy(s);
    s = wt_live_spans_create(3);
    assert(s);
    gen = wt_live_spans_begin(s);
    assert(wt_live_spans_add(s, backing, 48, WT_SPAN_REAL, 4, 1) == WT_SPAN_OK);
    assert(wt_live_spans_add(s, backing + 6, 72, WT_SPAN_REAL, 4, 1) == WT_SPAN_OK);
    assert(wt_live_spans_finish(s) == WT_SPAN_OK);
    assert(wt_live_spans_lookup(s, gen, backing + 6, WT_SPAN_REAL, 4, 72, &remaining) == WT_SPAN_OK);
    assert(remaining == 72);
    wt_live_spans_fail(s);
    assert(!wt_live_spans_is_current(s, gen));
    assert(wt_live_spans_lookup(s, gen, backing, WT_SPAN_REAL, 4, 4, &remaining) == WT_SPAN_UNKNOWN);
    assert(wt_live_spans_begin(s));
    assert(wt_live_spans_add(s, backing, 48, WT_SPAN_REAL, 4, 1) == WT_SPAN_OK);
    assert(wt_live_spans_add(s, backing + 6, 72, WT_SPAN_DOUBLE, 8, 1) == WT_SPAN_INVALID);
    assert(wt_live_spans_begin(s));
    assert(wt_live_spans_add(s, NULL, 4, WT_SPAN_REAL, 4, 0) == WT_SPAN_INVALID);
    assert(wt_live_spans_begin(s));
    assert(wt_live_spans_add(s, backing, 0, WT_SPAN_REAL, 4, 0) == WT_SPAN_INVALID);
    assert(wt_live_spans_begin(s));
    assert(wt_live_spans_add(s, backing, 4, WT_SPAN_LOGICAL, 4, 3) == WT_SPAN_INVALID);
    gen = wt_live_spans_begin(s);
    assert(wt_live_spans_finish(s) == WT_SPAN_OK);
    assert(wt_live_spans_is_current(s, gen));
    assert(wt_live_spans_add(s, backing, 4, WT_SPAN_REAL, 4, 0) == WT_SPAN_INVALID);
    assert(wt_live_spans_lookup(s, gen, backing, WT_SPAN_REAL, 4, 4, &remaining) == WT_SPAN_UNKNOWN);
    wt_live_spans_destroy(s);
#ifndef WT_TEST_C_ONLY
    assert(fixture_grid() == 0);
#endif
    puts("live span checks passed");
    return 0;
}
