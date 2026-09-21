/* elastic_bench.c -- mechanism probe for external elastic-core lending.
 *
 * Mimics one WRF physics call: a static-scheduled OpenMP loop over NCHUNKS equal chunks
 * (~6 ms each at 1 thread), repeated NREGIONS times. Nothing in here touches affinity;
 * an external process (expander.py) widens/narrows this process's thread masks while a
 * region is running. Every chunk records which thread ran it and on which CPU it started
 * and ended, so iteration ownership and execution placement can be told apart.
 *
 * Status protocol (for the external process): the file given as argv[3] is rewritten
 * with "start <k>\n" just before region k and "end <k>\n" just after it.
 *
 * usage: elastic_bench NCHUNKS NREGIONS STATUSFILE [ITERS]
 */
#define _GNU_SOURCE
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

#define MAXT 64
#define MAXC 256

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double cpu_s(void){ struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static double work(long iters){
    double a = 1.0;
    for (long i = 0; i < iters; i++) { a = a * 1.0000001 + 1e-9; __asm__ __volatile__("" : "+x"(a)); }
    return a;
}

static void mask_str(char *buf, size_t n, const cpu_set_t *m){
    size_t p = 0; int first = 1;
    buf[0] = 0;
    for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, m)) {
        int w = snprintf(buf + p, n - p, first ? "%d" : ",%d", c); if (w < 0 || (size_t)w >= n - p) break; p += w; first = 0;
    }
}

static void status(const char *path, const char *what, int k){
    FILE *f = fopen(path, "w"); if (!f) return;
    fprintf(f, "%s %d\n", what, k); fclose(f);
}

struct rec { int thread, cpu0, cpu1; double cpu_ms, end_off_ms; };

int main(int argc, char **argv){
    if (argc < 4) { fprintf(stderr, "usage: %s NCHUNKS NREGIONS STATUSFILE [ITERS]\n", argv[0]); return 2; }
    int nchunks = atoi(argv[1]), nregions = atoi(argv[2]);
    const char *sf = argv[3];
    long iters = argc > 4 ? atol(argv[4]) : 0;
    int idle_ms = argc > 5 ? atoi(argv[5]) : 0;   /* sleep between regions: a donor that waits */
    if (nchunks > MAXC) nchunks = MAXC;

    /* calibrate ~6 ms per chunk on one thread if not given */
    if (iters <= 0) {
        long probe = 2000000; double t0 = now_s(); work(probe); double dt = now_s() - t0;
        iters = (long)(probe * (0.006 / dt));
    }

    int nt = omp_get_max_threads();
    pid_t tids[MAXT] = {0};
    #pragma omp parallel
    { tids[omp_get_thread_num()] = syscall(SYS_gettid); }
    printf("pid %d nthreads %d iters %ld nchunks %d\n", getpid(), nt, iters, nchunks);
    printf("tids"); for (int i = 0; i < nt; i++) printf(" %d", tids[i]); printf("\n");
    fflush(stdout);

    static struct rec rec[MAXC];
    static char masks[MAXT][256];

    for (int k = 0; k < nregions; k++) {
        /* each thread reads its own affinity mask at region entry: shows whether the
         * runtime re-pins between regions after an external change */
        #pragma omp parallel
        { cpu_set_t m; CPU_ZERO(&m); sched_getaffinity(0, sizeof m, &m); mask_str(masks[omp_get_thread_num()], 256, &m); }

        status(sf, "start", k);
        double t0 = now_s();
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < nchunks; c++) {
            int cpu0 = sched_getcpu(); double c0 = cpu_s();
            volatile double r = work(iters); (void)r;
            rec[c].thread = omp_get_thread_num(); rec[c].cpu0 = cpu0; rec[c].cpu1 = sched_getcpu();
            rec[c].cpu_ms = (cpu_s() - c0) * 1e3; rec[c].end_off_ms = (now_s() - t0) * 1e3;
        }
        double wall = (now_s() - t0) * 1e3;
        status(sf, "end", k);

        printf("region %d wall_ms %.3f masks", k, wall);
        for (int i = 0; i < nt; i++) printf(" t%d=[%s]", i, masks[i]);
        printf("\n");
        for (int c = 0; c < nchunks; c++)
            printf("  chunk %d thread %d cpu %d->%d cpu_ms %.2f end_ms %.2f\n",
                   c, rec[c].thread, rec[c].cpu0, rec[c].cpu1, rec[c].cpu_ms, rec[c].end_off_ms);
        fflush(stdout);
        if (idle_ms > 0) usleep((useconds_t)idle_ms * 1000);
    }

    /* per-thread scheduler counters at exit */
    for (int i = 0; i < nt; i++) {
        char p[128], line[256]; long vol = -1, nonvol = -1, mig = -1;
        snprintf(p, sizeof p, "/proc/self/task/%d/status", tids[i]);
        FILE *f = fopen(p, "r");
        if (f) { while (fgets(line, sizeof line, f)) { sscanf(line, "voluntary_ctxt_switches: %ld", &vol); sscanf(line, "nonvoluntary_ctxt_switches: %ld", &nonvol); } fclose(f); }
        snprintf(p, sizeof p, "/proc/self/task/%d/sched", tids[i]);
        f = fopen(p, "r");
        if (f) { while (fgets(line, sizeof line, f)) { char *q = strstr(line, "nr_migrations"); if (q) { q = strchr(q, ':'); if (q) mig = atol(q + 1); } } fclose(f); }
        printf("thread %d tid %d vol_ctxsw %ld nonvol_ctxsw %ld nr_migrations %ld\n", i, tids[i], vol, nonvol, mig);
    }
    return 0;
}
