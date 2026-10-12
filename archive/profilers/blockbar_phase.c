/* PASS 2 profiler: per-rank x per-SEGMENT compute/wait timing, no WRF source change.

   WRF's phases are separated by halo exchanges, so the interval between two consecutive
   intercepted MPI calls IS one compute phase. The call sequence per step is deterministic
   and identical on every rank, so segment index k denotes the same phase on all ranks.
   That gives the dense per-rank x per-phase matrix that answers:

     "is a rank that is IDLE in segment k BUSY in some other segment k' of the SAME step?"
        yes -> within-step phase anticorrelation -> deferred settlement (wait-late) can pool
        no  -> the rank is light everywhere -> only moving real work helps

   Records a bounded window [SKIP, SKIP+MAXSEG) to keep the gather small, then rank 0
   writes rank,seg,comp_s,wait_s.

   env: BLOCKBAR_PHASE_OUT (default ./phase_seg.csv), BLOCKBAR_PHASE_SKIP, BLOCKBAR_PHASE_MAX
   Uses blocking PMPI directly (no nanosleep poll) to minimise perturbation of the thing
   being measured.                                                                        */
#define _GNU_SOURCE
#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int inited = 0, grank = 0, gn = 0;
static long seg = 0, SKIP = 2000, MAXSEG = 1200;
static double t_last = 0.0;
static float *comp = NULL, *wait = NULL;
static unsigned char *kind = NULL;   /* 0=Wait 1=Waitall 2=Barrier 3=Allreduce */
static float *tend = NULL;           /* segment end time relative to a barrier-synced t0 */
static double t0 = 0.0;

static double now(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void initcfg(void) {
  if (inited) return; inited = 1;
  char *e;
  if ((e = getenv("BLOCKBAR_PHASE_SKIP"))) SKIP = atol(e);
  if ((e = getenv("BLOCKBAR_PHASE_MAX")))  MAXSEG = atol(e);
  PMPI_Comm_rank(MPI_COMM_WORLD, &grank);
  PMPI_Comm_size(MPI_COMM_WORLD, &gn);
  comp = calloc(MAXSEG, sizeof(float));
  wait = calloc(MAXSEG, sizeof(float));
  kind = calloc(MAXSEG, 1);
  tend = calloc(MAXSEG, sizeof(float));
  PMPI_Barrier(MPI_COMM_WORLD);   /* common time origin so windows can be time-aligned */
  t0 = now();
  t_last = now();
  if (grank == 0)
    fprintf(stderr, "[blockbar_phase] active ranks=%d skip=%ld max=%ld\n", gn, SKIP, MAXSEG);
}
/* record the compute interval that ENDED at this MPI call, and the wait the call itself cost */
static inline long seg_begin(double *s) { initcfg(); *s = now(); return seg; }
static inline void seg_end2(long k, double s, unsigned char kd) {
  double e = now();
  long i = k - SKIP;
  if (i >= 0 && i < MAXSEG) { comp[i] = (float)(s - t_last); wait[i] = (float)(e - s);
                              kind[i] = kd; tend[i] = (float)(e - t0); }
  t_last = e; seg++;
}

int MPI_Wait(MPI_Request *rq, MPI_Status *st) {
  double s; long k = seg_begin(&s); int r = PMPI_Wait(rq, st); seg_end2(k, s, 0); return r;
}
int MPI_Waitall(int n, MPI_Request rq[], MPI_Status st[]) {
  double s; long k = seg_begin(&s); int r = PMPI_Waitall(n, rq, st); seg_end2(k, s, 1); return r;
}
int MPI_Barrier(MPI_Comm c) {
  double s; long k = seg_begin(&s); int r = PMPI_Barrier(c); seg_end2(k, s, 2); return r;
}
int MPI_Allreduce(const void *sb, void *rb, int c, MPI_Datatype dt, MPI_Op o, MPI_Comm m) {
  double s; long k = seg_begin(&s); int r = PMPI_Allreduce(sb, rb, c, dt, o, m); seg_end2(k, s, 3); return r;
}

int MPI_Finalize(void) {
  initcfg();
  float *ac = NULL, *aw = NULL, *at = NULL; unsigned char *ak = NULL;
  if (grank == 0) { ac = malloc((size_t)gn * MAXSEG * sizeof(float));
                    aw = malloc((size_t)gn * MAXSEG * sizeof(float));
                    ak = malloc((size_t)gn * MAXSEG);
                    at = malloc((size_t)gn * MAXSEG * sizeof(float)); }
  PMPI_Gather(comp, MAXSEG, MPI_FLOAT, ac, MAXSEG, MPI_FLOAT, 0, MPI_COMM_WORLD);
  PMPI_Gather(wait, MAXSEG, MPI_FLOAT, aw, MAXSEG, MPI_FLOAT, 0, MPI_COMM_WORLD);
  PMPI_Gather(kind, MAXSEG, MPI_BYTE, ak, MAXSEG, MPI_BYTE, 0, MPI_COMM_WORLD);
  PMPI_Gather(tend, MAXSEG, MPI_FLOAT, at, MAXSEG, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if (grank == 0) {
    const char *o = getenv("BLOCKBAR_PHASE_OUT");
    FILE *f = fopen(o ? o : "phase_seg.csv", "w");
    if (f) {
      fprintf(f, "rank,seg,comp_s,wait_s,kind,tend_s\n");
      for (int r = 0; r < gn; r++)
        for (long i = 0; i < MAXSEG; i++) {
          float c = ac[(size_t)r * MAXSEG + i], w = aw[(size_t)r * MAXSEG + i];
          if (c != 0.0f || w != 0.0f)
            fprintf(f, "%d,%ld,%.6f,%.6f,%d,%.6f\n", r, SKIP + i, c, w,
                    (int)ak[(size_t)r*MAXSEG+i], at[(size_t)r*MAXSEG+i]);
        }
      fclose(f);
      fprintf(stderr, "[blockbar_phase] wrote %s (%d ranks x %ld segs)\n", o ? o : "phase_seg.csv", gn, MAXSEG);
    }
  }
  return PMPI_Finalize();
}
