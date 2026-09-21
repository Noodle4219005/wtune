#ifndef WT_LIVE_SPANS_H
#define WT_LIVE_SPANS_H
#include <stddef.h>
#include <stdint.h>

typedef struct wt_live_spans wt_live_spans;
enum { WT_SPAN_OK = 0, WT_SPAN_UNKNOWN = 1, WT_SPAN_INVALID = 2 };
enum { WT_SPAN_REAL = 1, WT_SPAN_DOUBLE = 2,
       WT_SPAN_INTEGER = 3, WT_SPAN_LOGICAL = 4 };

/* Synchronous, exclusive outer-driver scope only: no concurrent mutation,
 * free, reallocation, or domain destruction from begin through end. This is
 * live descriptor evidence, never an allocation-history or ownership oracle.
 * Each begin invalidates all old tokens; end invalidates the current token.
 * The context must outlive every call; destroy also destroys token identity.
 * Do not retain a context/token pair across destroy and create.
 * No function below reads or dereferences the registered data pointers. */
wt_live_spans *wt_live_spans_create(size_t max_spans);
void wt_live_spans_destroy(wt_live_spans *s);
uint64_t wt_live_spans_begin(wt_live_spans *s);
int wt_live_spans_add(void *context, void *base, size_t bytes,
                     int type, size_t width, int rank);
void wt_live_spans_fail(void *context);
int wt_live_spans_finish(void *context);
void wt_live_spans_end(wt_live_spans *s);
/* True only for the current frozen scope, including a valid empty capture. */
int wt_live_spans_is_current(const wt_live_spans *s, uint64_t generation);
int wt_live_spans_lookup(const wt_live_spans *s, uint64_t generation,
                        const void *pointer, int type, size_t width,
                        size_t required, size_t *remaining);

/* wt_grid_spans is a plain Fortran external subroutine, NOT BIND(C).
 * Compile it against the frozen WRF build's existing module_domain_type.mod
 * with the matching compiler, kind options, and ABI. The default gfortran
 * fixture ABI is wt_grid_spans_(void *domain, void **context,
 *                              const size_t *max_nodes, int *status).
 * This signature is not a portable native-WRF ABI claim.
 * begin -> wt_grid_spans -> finish iff status==OK -> lookup -> end.
 * Missing/unsupported/unassociated fields have UNKNOWN capacity. A malformed
 * selected array or incomplete traversal invalidates the entire collection. */
#endif
