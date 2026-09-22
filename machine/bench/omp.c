/* omp.c -- OpenMP parallel-region fork/join and barrier overhead, at
 * thread counts 1,2,4,8,...,all cores. Both are time-boxed (repeat until
 * a wall-clock budget is spent, not a fixed rep count) so the bench stays
 * fast at low thread counts and doesn't undersample at high thread
 * counts where OS scheduling noise is larger.
 *
 * Emits exactly one JSON line to stdout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int parse_thread_list(const char *s, int *out, int max_n) {
    if (!s) return 0;
    int n = 0;
    char buf[256];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    char *tok = strtok(buf, ",");
    while (tok && n < max_n) { out[n++] = atoi(tok); tok = strtok(NULL, ","); }
    return n;
}

static int default_thread_list(int *out, int max_n) {
    int ncores = omp_get_max_threads();
    int n = 0;
    for (int t = 1; t <= ncores && n < max_n; t *= 2) out[n++] = t;
    if (n == 0 || out[n - 1] != ncores) { if (n < max_n) out[n++] = ncores; }
    return n;
}

/* Time repeated empty parallel regions for `budget_s` seconds; return
 * microseconds per fork/join and the rep count used. */
static double bench_forkjoin(int threads, double budget_s, long *reps_out) {
    long reps = 0;
    volatile long sink = 0; /* touched inside the region: blocks the
                                compiler from proving the region is a
                                provable no-op and eliding it entirely */
    double t0 = now_s();
    double elapsed = 0.0;
    while (elapsed < budget_s) {
        #pragma omp parallel num_threads(threads)
        {
            sink += omp_get_thread_num();
        }
        reps++;
        if ((reps & 0xFF) == 0) elapsed = now_s() - t0;
    }
    elapsed = now_s() - t0;
    *reps_out = reps;
    if (sink == -1) fprintf(stderr, "omp: impossible sink\n"); /* consume sink */
    return (elapsed / (double)reps) * 1e6;
}

/* Time a fixed number of barriers inside one persistent parallel region
 * for `budget_s` seconds; return microseconds per barrier. */
static double bench_barrier(int threads, double budget_s, long *reps_out) {
    long reps = 0;
    long sink = 0;
    double t0 = now_s();
    double shared_elapsed = 0.0; /* declared outside the parallel region:
                                     shared by default, so thread 0's write
                                     is visible to every thread after the
                                     barrier below (a thread-private copy
                                     here would spin the other threads
                                     forever -- caught by local testing). */
    #pragma omp parallel num_threads(threads) shared(reps, t0, shared_elapsed) reduction(+:sink)
    {
        long local_reps = 0;
        while (shared_elapsed < budget_s) {
            #pragma omp barrier
            sink += 1;
            local_reps++;
            if ((local_reps & 0xFFF) == 0) {
                #pragma omp masked
                shared_elapsed = now_s() - t0;
                #pragma omp barrier /* all threads see the updated elapsed */
            }
        }
        #pragma omp masked
        reps = local_reps;
    }
    double elapsed = now_s() - t0;
    *reps_out = reps;
    if (sink == -1) fprintf(stderr, "omp: impossible sink\n"); /* consume sink */
    return reps > 0 ? (elapsed / (double)reps) * 1e6 : 0.0;
}

int main(int argc, char **argv) {
    double budget_s = 0.2;
    const char *threads_arg = NULL;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--seconds")) budget_s = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--threads-list")) threads_arg = argv[i + 1];
    }
    int tlist[64];
    int nthr = threads_arg ? parse_thread_list(threads_arg, tlist, 64)
                            : default_thread_list(tlist, 64);
    int ncores = omp_get_max_threads();

    char fj[2048] = ""; int fj_off = 0;
    char br[2048] = ""; int br_off = 0;
    for (int i = 0; i < nthr; i++) {
        long reps;
        double us = bench_forkjoin(tlist[i], budget_s, &reps);
        fj_off += snprintf(fj + fj_off, sizeof(fj) - fj_off,
            "%s{\"threads\":%d,\"us_per_region\":%.4f,\"reps\":%ld}", i ? "," : "", tlist[i], us, reps);
    }
    for (int i = 0; i < nthr; i++) {
        long reps;
        double us = bench_barrier(tlist[i], budget_s, &reps);
        br_off += snprintf(br + br_off, sizeof(br) - br_off,
            "%s{\"threads\":%d,\"us_per_barrier\":%.4f,\"reps\":%ld}", i ? "," : "", tlist[i], us, reps);
    }

    printf("{\"bench\":\"omp\",\"ncores\":%d,\"forkjoin\":[%s],\"barrier\":[%s]}\n", ncores, fj, br);
    return 0;
}
