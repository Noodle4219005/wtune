/* tsample -- external per-thread sampler for WRF ranks (verification only, no actuation).
 *
 * Waits until rsl.out.0000 shows the first timestep, then every --interval-us reads, for every
 * thread of the selected local ranks, /proc/<pid>/task/<tid>/stat (state, last CPU) and
 * /proc/<pid>/task/<tid>/schedstat (sum_exec_runtime in ns, maintained even with
 * kernel.sched_schedstats=0). The task directory is re-listed every sample so thread churn
 * (TIDs that appear or vanish) is visible. Stops after the 40th timestep or when the ranks exit.
 *
 * usage: tsample --ranks R1,R2,... --rsl PATH --out FILE [--exe wrf.exe] [--interval-us 5000]
 *                [--max-s 900] [--last-step 40]
 * stdout: META rank=<r> pid=<p> home=<cpu list> threads=<n> lines, then a TSAMPLE summary.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#define MAXR 64

static volatile sig_atomic_t stop;
static void on_sig(int s) { (void)s; stop = 1; }

static const char *EXE = "wrf.exe", *RSL, *OUT;
static int WANT[MAXR], NWANT, INTERVAL_US = 5000, MAX_S = 900, LAST_STEP = 40;
static pid_t PID[MAXR];
static int RANK[MAXR], NPID;

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static int read_small(const char *p, char *b, int n) {
    int fd = open(p, O_RDONLY);
    if (fd < 0) return -1;
    int k = (int)read(fd, b, (size_t)n - 1);
    close(fd);
    if (k <= 0) return -1;
    b[k] = 0;
    return k;
}

static int comm_is(pid_t pid) {
    char p[64], b[64], want[16];
    snprintf(p, sizeof p, "/proc/%d/comm", pid);
    if (read_small(p, b, sizeof b) < 0) return 0;
    char *nl = strchr(b, '\n');
    if (nl) *nl = 0;
    snprintf(want, sizeof want, "%s", EXE);
    return !strcmp(b, want);
}

static int env_rank(pid_t pid) {
    static char buf[1 << 20];
    char p[64];
    snprintf(p, sizeof p, "/proc/%d/environ", pid);
    int fd = open(p, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t k, n = 0;
    while (n < (ssize_t)sizeof buf - 1 && (k = read(fd, buf + n, sizeof buf - 1 - n)) > 0) n += k;
    close(fd);
    buf[n] = 0;
    for (ssize_t i = 0; i < n;) {
        char *s = buf + i;
        size_t l = strnlen(s, (size_t)(n - i));
        if (l > 13 && !strncmp(s, "SLURM_PROCID=", 13)) return atoi(s + 13);
        i += (ssize_t)l + 1;
    }
    return -1;
}

static void discover(void) {
    DIR *d = opendir("/proc");
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] < '1' || e->d_name[0] > '9') continue;
        pid_t pid = (pid_t)atoi(e->d_name);
        char p[64];
        struct stat st;
        snprintf(p, sizeof p, "/proc/%d", pid);
        if (stat(p, &st) || st.st_uid != getuid() || !comm_is(pid)) continue;
        int r = env_rank(pid), w = 0, known = 0;
        for (int i = 0; i < NWANT; i++) if (WANT[i] == r) w = 1;
        for (int i = 0; i < NPID; i++) if (PID[i] == pid) known = 1;
        if (w && !known && NPID < MAXR) { PID[NPID] = pid; RANK[NPID] = r; NPID++; }
    }
    closedir(d);
}

static int count_timing(void) {
    FILE *f = fopen(RSL, "r");
    if (!f) return 0;
    char line[512];
    int n = 0;
    while (fgets(line, sizeof line, f)) if (strstr(line, "Timing for main")) n++;
    fclose(f);
    return n;
}

int main(int argc, char **argv) {
    static struct option lo[] = {{"exe", 1, 0, 'e'}, {"ranks", 1, 0, 'r'}, {"rsl", 1, 0, 'l'}, {"out", 1, 0, 'o'},
                                 {"interval-us", 1, 0, 'i'}, {"max-s", 1, 0, 'x'}, {"last-step", 1, 0, 's'}, {0, 0, 0, 0}};
    int ch;
    while ((ch = getopt_long(argc, argv, "", lo, NULL)) != -1) {
        if (ch == 'e') EXE = optarg;
        else if (ch == 'l') RSL = optarg;
        else if (ch == 'o') OUT = optarg;
        else if (ch == 'i') INTERVAL_US = atoi(optarg);
        else if (ch == 'x') MAX_S = atoi(optarg);
        else if (ch == 's') LAST_STEP = atoi(optarg);
        else if (ch == 'r') {
            for (char *save, *t = strtok_r(optarg, ",", &save); t && NWANT < MAXR; t = strtok_r(NULL, ",", &save))
                WANT[NWANT++] = atoi(t);
        } else return 2;
    }
    if (!RSL || !OUT || !NWANT) { fprintf(stderr, "need --ranks --rsl --out\n"); return 2; }
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    char host[256] = "?";
    gethostname(host, sizeof host);
    uint64_t deadline = now_ns() + (uint64_t)MAX_S * 1000000000ull;

    while (!stop && now_ns() < deadline && count_timing() < 1) usleep(50 * 1000);
    discover();
    if (NPID == 0) { printf("TSAMPLE host=%s no selected ranks on this node\n", host); return 0; }
    for (int i = 0; i < NPID; i++) {
        cpu_set_t m;
        CPU_ZERO(&m);
        sched_getaffinity(PID[i], sizeof m, &m);
        char list[512] = "";
        int first = 1;
        for (int c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, &m)) {
                char x[16];
                snprintf(x, sizeof x, first ? "%d" : ",%d", c);
                strncat(list, x, sizeof list - strlen(list) - 1);
                first = 0;
            }
        char tp[64];
        snprintf(tp, sizeof tp, "/proc/%d/task", PID[i]);
        int nt = 0;
        DIR *d = opendir(tp);
        if (d) { struct dirent *e; while ((e = readdir(d))) if (e->d_name[0] != '.') nt++; closedir(d); }
        printf("META rank=%d pid=%d home=%s threads=%d\n", RANK[i], PID[i], list, nt);
    }

    FILE *f = fopen(OUT, "w");
    if (!f) { perror(OUT); return 3; }
    setvbuf(f, NULL, _IOFBF, 1 << 20);
    fprintf(f, "t_ms,step,rank,pid,tid,state,cpu,exec_ns\n");
    uint64_t t0 = now_ns(), next = t0, last_rsl = 0;
    int step = count_timing();
    long samples = 0, lines = 0;
    while (!stop && now_ns() < deadline) {
        uint64_t t = now_ns();
        if (t - last_rsl > 100000000ull) { step = count_timing(); last_rsl = t; }
        int alive = 0;
        for (int i = 0; i < NPID; i++) {
            char tp[64];
            snprintf(tp, sizeof tp, "/proc/%d/task", PID[i]);
            DIR *d = opendir(tp);
            if (!d) continue;
            alive++;
            struct dirent *e;
            while ((e = readdir(d))) {
                if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
                char p[128], b[1024], sb[128];
                snprintf(p, sizeof p, "/proc/%d/task/%s/stat", PID[i], e->d_name);
                if (read_small(p, b, sizeof b) < 0) continue;
                snprintf(p, sizeof p, "/proc/%d/task/%s/schedstat", PID[i], e->d_name);
                if (read_small(p, sb, sizeof sb) < 0) continue;
                char *s = strrchr(b, ')');
                if (!s || !s[1] || !s[2]) continue;
                s += 2;
                char state = *s;
                int field = 0, cpu = -1;
                for (char *q = s; *q; q++)
                    if (*q == ' ' && ++field == 36) { cpu = atoi(q + 1); break; }
                fprintf(f, "%.3f,%d,%d,%d,%s,%c,%d,%llu\n", (t - t0) / 1e6, step, RANK[i], PID[i], e->d_name, state, cpu,
                        strtoull(sb, NULL, 10));
                lines++;
            }
            closedir(d);
        }
        samples++;
        if (!alive || step >= LAST_STEP) break;
        next += (uint64_t)INTERVAL_US * 1000ull;
        uint64_t now = now_ns();
        if (next <= now) next = now + (uint64_t)INTERVAL_US * 1000ull;
        struct timespec ts = {(time_t)(next / 1000000000ull), (long)(next % 1000000000ull)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
    }
    fclose(f);
    printf("TSAMPLE host=%s ranks=%d samples=%ld lines=%ld span_s=%.1f last_step=%d\n", host, NPID, samples, lines,
           (now_ns() - t0) / 1e9, step);
    return 0;
}
