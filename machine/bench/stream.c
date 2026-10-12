/* stream.c -- OpenMP STREAM triad: a[i] = b[i] + scalar*c[i]. Array size
 * defaults to >= 4x the detected LLC (so the working set cannot fit in
 * any level of cache), overridable via --llc-bytes for machines without a
 * readable LLC size. Bandwidth is measured at 1,2,4,8,...,ncores threads
 * so the profile captures bandwidth-vs-active-cores, not just a single
 * peak number -- the GH200-vs-F1-Grace gap this project cares about shows
 * up as a difference in this curve's shape, not just its ceiling.
 *
 * Emits exactly one JSON line to stdout.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <omp.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Best-effort LLC size (bytes) from sysfs: highest-numbered cache index
 * under cpu0 with type Unified. Returns 0 if unreadable. */
static long detect_llc_bytes(void) {
    long best = 0;
    for (int idx = 0; idx < 8; idx++) {
        char p[256];
        snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            long v = atol(buf);
            char unit = 0;
            for (char *c = buf; *c; c++) if (*c == 'K' || *c == 'M') { unit = *c; break; }
            if (unit == 'K') v *= 1024L;
            else if (unit == 'M') v *= 1024L * 1024L;
            if (v > best) best = v;
        }
        fclose(f);
    }
    return best;
}

static int parse_thread_list(const char *s, int *out, int max_n) {
    if (!s) return 0;
    int n = 0;
    char buf[256];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    char *tok = strtok(buf, ",");
    while (tok && n < max_n) {
        out[n++] = atoi(tok);
        tok = strtok(NULL, ",");
    }
    return n;
}

static int default_thread_list(int *out, int max_n) {
    int ncores = omp_get_max_threads();
    int n = 0;
    for (int t = 1; t <= ncores && n < max_n; t *= 2) out[n++] = t;
    if (n == 0 || out[n - 1] != ncores) { if (n < max_n) out[n++] = ncores; }
    return n;
}

int main(int argc, char **argv) {
    long llc_bytes = 0;
    int iters = 20;
    const char *threads_arg = NULL;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--llc-bytes")) llc_bytes = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--threads-list")) threads_arg = argv[i + 1];
    }
    if (llc_bytes <= 0) llc_bytes = detect_llc_bytes();
    if (llc_bytes <= 0) llc_bytes = 32L * 1024 * 1024; /* fallback assumption */

    long need_bytes = 4L * llc_bytes;
    size_t n_elems = (size_t)(need_bytes / (3 * (long)sizeof(double)));
    if (n_elems < 1000000) n_elems = 1000000;

    double *a = malloc(n_elems * sizeof(double));
    double *b = malloc(n_elems * sizeof(double));
    double *c = malloc(n_elems * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "stream: alloc failed for %zu elems\n", n_elems); return 1; }
    #pragma omp parallel for
    for (size_t i = 0; i < n_elems; i++) { b[i] = 1.0; c[i] = 2.0; a[i] = 0.0; }

    int tlist[64];
    int nthr = threads_arg ? parse_thread_list(threads_arg, tlist, 64)
                            : default_thread_list(tlist, 64);

    printf("{\"bench\":\"stream\",\"array_elements\":%zu,\"array_bytes_total\":%zu,"
           "\"llc_bytes_assumed\":%ld,\"iters\":%d,\"results\":[",
           n_elems, n_elems * 3 * sizeof(double), llc_bytes, iters);

    const double scalar = 3.0;
    for (int ti = 0; ti < nthr; ti++) {
        int t = tlist[ti];
        omp_set_num_threads(t);
        /* one warmup pass, not timed */
        #pragma omp parallel for
        for (size_t i = 0; i < n_elems; i++) a[i] = b[i] + scalar * c[i];

        double best_time = -1.0;
        double t0 = now_s();
        for (int r = 0; r < iters; r++) {
            #pragma omp parallel for
            for (size_t i = 0; i < n_elems; i++) a[i] = b[i] + scalar * c[i];
        }
        double total = now_s() - t0;
        best_time = total / iters;

        double bytes_per_iter = (double)n_elems * 3.0 * sizeof(double);
        double gbps = bytes_per_iter / best_time / 1e9;

        /* Validation, standard STREAM style: block DCE and catch a broken
         * kernel (checked value should equal b[0]+scalar*c[0]). */
        double expect = 1.0 + scalar * 2.0;
        if (a[0] < expect - 1e-9 || a[0] > expect + 1e-9) {
            fprintf(stderr, "stream: validation failed at threads=%d (got %.6f want %.6f)\n",
                    t, a[0], expect);
        }

        printf("%s{\"threads\":%d,\"gbps\":%.3f,\"time_s\":%.6f}",
               ti ? "," : "", t, gbps, best_time);
    }
    printf("]}\n");

    free(a); free(b); free(c);
    return 0;
}
