/* wtmpi -- how much of a WRF timestep is spent waiting, per rank.
 *
 * This is the last unmeasured piece of the step. The instrumented compute critical path is
 * 61.0 s against a 104.2 s solve; the 43.2 s difference has never been split into "waiting"
 * and "computing something with no timer on it". Core lending only has something to lend if
 * that gap is mostly waiting.
 *
 * Plain PMPI, not -Wl,--wrap: defining MPI_Waitall here makes the linker resolve WRF's
 * reference to this object (it is named explicitly on the link line, so it beats the library),
 * and PMPI_Waitall still reaches the real one. WRF's halo exchange is C -- external/RSL_LITE
 * issues MPI_Isend/Irecv/Waitall directly -- so the C bindings are the ones that matter.
 * Fortran-side collectives go through mpi_*_ symbols and are NOT counted here; the gate below
 * is what says whether that omission matters.
 */
#include <mpi.h>
#include <stdint.h>

void wt_add_wait(uint64_t cycles, int calls);
void wt_add_kind(int kind, uint64_t cycles);
void wt_lewi_lend_begin(void);
void wt_lewi_lend_end(void);

static inline uint64_t tsc_now(void)
{
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

/* kind: 0 Wait 1 Waitall 2 Waitany 3 Barrier 4 Allreduce 5 Bcast 6 Sendrecv.
 * Separating them answers whether WRF synchronises GLOBALLY between phases (Barrier/Allreduce,
 * which couple all 448 ranks) or only with NEIGHBOURS (Waitall on the halo's point-to-point
 * requests). The aggregate count cannot tell those apart. */
#define WRAP(name, kind, proto, args)              \
    int name proto {                               \
        uint64_t t0 = tsc_now();                   \
        wt_lewi_lend_begin();                      \
        int rc = P##name args;                     \
        wt_lewi_lend_end();                        \
        uint64_t wt_dt_ = tsc_now() - t0;          \
        wt_add_wait(wt_dt_, 1);                    \
        wt_add_kind(kind, wt_dt_);                 \
        return rc;                                 \
    }

WRAP(MPI_Wait, 0, (MPI_Request *r, MPI_Status *s), (r, s))
WRAP(MPI_Waitall, 1, (int c, MPI_Request *r, MPI_Status *s), (c, r, s))
WRAP(MPI_Waitany, 2, (int c, MPI_Request *r, int *i, MPI_Status *s), (c, r, i, s))
WRAP(MPI_Barrier, 3, (MPI_Comm c), (c))
WRAP(MPI_Allreduce, 4, (const void *sb, void *rb, int n, MPI_Datatype d, MPI_Op o, MPI_Comm c),
                    (sb, rb, n, d, o, c))
WRAP(MPI_Bcast, 5, (void *b, int n, MPI_Datatype d, int r, MPI_Comm c), (b, n, d, r, c))
WRAP(MPI_Sendrecv, 6, (const void *sb, int sn, MPI_Datatype sd, int dst, int st,
                    void *rb, int rn, MPI_Datatype rd, int src, int rt,
                    MPI_Comm c, MPI_Status *s),
                   (sb, sn, sd, dst, st, rb, rn, rd, src, rt, c, s))
