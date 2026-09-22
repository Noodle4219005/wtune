/* freq.c -- sustained per-core clock estimate via a loop-carried dependent
 * integer add chain, timed with clock_gettime. Integer ADD has 1-cycle
 * latency on essentially every mainstream core (x86, Neoverse-V2 included),
 * so iterations/second of a purely register-resident dependency chain is
 * (to first order) the core's *sustained* clock -- not its rated/boost
 * clock, and not a proxy like "core count". Run once pinned to one core,
 * then again with one thread pinned to every allowed core simultaneously,
 * to capture the clock the core actually sustains under full-package load
 * (thermal/power throttling, turbo drop-off).
 *
 * Emits exactly one JSON line to stdout.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <dirent.h>
#include <stdint.h>
#include <omp.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* One opaque ADD forced into a real instruction via inline asm: with a
 * plain "x = x + 1" in a fixed-trip loop, GCC's induction-variable pass
 * strength-reduces the whole loop to a single "x += n" (observed: reports
 * tens of thousands of "GHz"). The "+r" asm constraint makes x an opaque
 * register value the compiler cannot analyze through, so the add chain
 * must actually execute once per iteration -- the loop-carried dependency
 * (each add reads the previous add's result) is what pins iterations to
 * cycles. Loop control (compare+branch) adds a small, roughly constant
 * per-iteration overhead on top of the ADD itself, so the resulting
 * "ghz" is a conservative (slightly low) estimate of sustained clock,
 * not an exact one. */
static inline uint64_t add_chain_step(uint64_t x) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile ("add $1, %0" : "+r"(x));
#elif defined(__aarch64__)
    __asm__ volatile ("add %0, %0, #1" : "+r"(x));
#else
    x = x + 1; /* fallback: closed-form risk on unlisted architectures */
#endif
    return x;
}

/* Dependent add chain timed in batches so clock_gettime overhead is
 * amortized. Returns elapsed seconds; writes iteration count and final
 * value (consumed by the caller to block dead-code elimination). */
static double run_chain(double duration_s, uint64_t *out_iters, uint64_t *out_x) {
    uint64_t x = 1;
    uint64_t iters = 0;
    const uint64_t batch = 200000000ULL;
    double t0 = now_s();
    double elapsed = 0.0;
    while (elapsed < duration_s) {
        for (uint64_t i = 0; i < batch; i++) {
            x = add_chain_step(x);
        }
        iters += batch;
        elapsed = now_s() - t0;
    }
    *out_iters = iters;
    *out_x = x;
    return elapsed;
}

/* CPUs this process is actually allowed to run on (respects a Slurm/cgroup
 * cpuset), not a bare 0..nproc-1 assumption. */
static int get_allowed_cpus(int *cpus, int max_cpus) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return 0;
    int n = 0;
    for (int c = 0; c < CPU_SETSIZE && n < max_cpus; c++) {
        if (CPU_ISSET(c, &set)) cpus[n++] = c;
    }
    return n;
}

/* Read scaling_cur_freq (kHz) under each cpuN/cpufreq sysfs directory,
 * for every online CPU, if the sysfs knob exists at all. */
static int read_scaling_freqs(long *mhz_out, int max_n) {
    DIR *d = opendir("/sys/devices/system/cpu");
    if (!d) return -1;
    struct dirent *ent;
    int n = 0;
    int any_path_checked = 0;
    while ((ent = readdir(d)) != NULL && n < max_n) {
        int cpuid;
        if (sscanf(ent->d_name, "cpu%d", &cpuid) != 1) continue;
        char path[320];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/%s/cpufreq/scaling_cur_freq", ent->d_name);
        any_path_checked = 1;
        FILE *f = fopen(path, "r");
        if (!f) continue;
        long khz = 0;
        if (fscanf(f, "%ld", &khz) == 1) {
            mhz_out[n++] = khz / 1000;
        }
        fclose(f);
    }
    closedir(d);
    if (!any_path_checked || n == 0) return -1;
    return n;
}

int main(int argc, char **argv) {
    double seconds = 1.0;
    int threads_req = -1;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--threads")) threads_req = atoi(argv[i + 1]);
    }

    int cpus[4096];
    int ncpus = get_allowed_cpus(cpus, 4096);
    if (ncpus <= 0) { fprintf(stderr, "freq: sched_getaffinity failed\n"); return 1; }
    int nthreads = (threads_req > 0 && threads_req <= ncpus) ? threads_req : ncpus;

    /* Phase 1: single core, pinned to the first allowed CPU. */
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpus[0], &one);
    sched_setaffinity(0, sizeof(one), &one);
    uint64_t iters1 = 0, x1 = 0;
    double t1 = run_chain(seconds, &iters1, &x1);
    double ghz_1core = (t1 > 0.0) ? (double)iters1 / t1 / 1e9 : 0.0;

    /* Phase 2: one thread pinned to every allowed core, simultaneously. */
    double *per_core_ghz = calloc(nthreads, sizeof(double));
    uint64_t checksum = 0;
    #pragma omp parallel num_threads(nthreads) reduction(+:checksum)
    {
        int tid = omp_get_thread_num();
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(cpus[tid % ncpus], &s);
        sched_setaffinity(0, sizeof(s), &s);
        #pragma omp barrier
        uint64_t iters = 0, xv = 0;
        double el = run_chain(seconds, &iters, &xv);
        per_core_ghz[tid] = (el > 0.0) ? (double)iters / el / 1e9 : 0.0;
        checksum += xv;
    }
    /* Consume checksum so the whole dependency chain cannot be DCE'd. */
    if (checksum == 0xFFFFFFFFFFFFFFFFULL) fprintf(stderr, "freq: impossible checksum\n");

    double sum = 0, mn = per_core_ghz[0], mx = per_core_ghz[0];
    for (int i = 0; i < nthreads; i++) {
        sum += per_core_ghz[i];
        if (per_core_ghz[i] < mn) mn = per_core_ghz[i];
        if (per_core_ghz[i] > mx) mx = per_core_ghz[i];
    }
    double mean = sum / nthreads;

    long scaling_mhz[4096];
    int nscaling = read_scaling_freqs(scaling_mhz, 4096);

    printf("{\"bench\":\"freq\",\"ncores_allowed\":%d,\"nthreads\":%d,"
           "\"duration_s_requested\":%.3f,\"duration_s_1core\":%.3f,"
           "\"cycles_per_op_assumed\":1,"
           "\"ghz_1core\":%.4f,"
           "\"ghz_allcores_mean\":%.4f,\"ghz_allcores_min\":%.4f,\"ghz_allcores_max\":%.4f,"
           "\"per_core_ghz_allcores\":[",
           ncpus, nthreads, seconds, t1, ghz_1core, mean, mn, mx);
    for (int i = 0; i < nthreads; i++) printf("%s%.4f", i ? "," : "", per_core_ghz[i]);
    printf("],\"scaling_cur_freq_mhz\":");
    if (nscaling > 0) {
        printf("[");
        for (int i = 0; i < nscaling; i++) printf("%s%ld", i ? "," : "", scaling_mhz[i]);
        printf("]");
    } else {
        printf("null");
    }
    printf("}\n");

    free(per_core_ghz);
    return 0;
}
