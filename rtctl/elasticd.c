/* elasticd -- external elastic-core controller for pristine WRF (no wrap, no LD_PRELOAD).
 *
 * One instance per node, launched as an overlapping Slurm step. It finds the node's WRF ranks in
 * /proc (comm + SLURM_PROCID from environ), identifies each rank's OpenMP team by CPU time, picks
 * hot ranks and donors per NUMA domain from an offline per-rank profile, and moves hot-rank workers
 * onto donor cores with sched_setaffinity to a single CPU. It never widens a mask: this kernel's
 * sched_migration_cost_ns=5 ms leaves a widened thread where it is for hundreds of ms
 * (knowledge/2026-09-13-widening-an-affinity-mask-...). It never touches WRF memory, arguments or
 * control flow, and restores every mask it recorded at discovery when it exits.
 *
 * modes: observe  discovery + per-rank team CPU accounting, no actuation
 *        static   each borrower worker pinned to its donor's last home core for the whole run
 *        dynamic  lend while the donor's workers have been asleep >= --lend-after-ms and the hot
 *                 team has been in a region >= --hot-run-ms; return the poll a donor worker is
 *                 runnable, the donor master sits on the lent core, or the hot region ends
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
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

#define MAXR 128
#define MAXT 32
#define MAXCPU 1024
#define MAXPROF 65536

typedef struct {
    pid_t tid;
    int fd;
    char state;
    int cpu;
    unsigned long long ticks, ticks0;
    cpu_set_t orig;
    int orig_ok;
} thr_t;

typedef struct {
    int rank;
    pid_t pid;
    int nt;
    thr_t t[MAXT];
    int team[MAXT];     /* indices into t; team[0] is the master (tid == pid) */
    int nteam;
    int home[8];
    int nhome;
    int dom;
    double score;
    int role;           /* 0 neutral, 1 hot, 2 donor */
    int busy, alive;
    uint64_t since;     /* last busy/idle transition */
    double master_s, workers_s;
} rank_t;

typedef struct {
    int h, d, ti;       /* hot rank index, donor rank index, borrower thread index in hot rank */
    int lent;           /* cpu, or -1 when home */
    uint64_t t_lend, lent_ns;
    long nlend, ncollide, nhotend, nerr;
} slot_t;

static rank_t R[MAXR];
static int NR;
static slot_t S[MAXR * 4];
static int NS;
static double PROF[MAXPROF];
static char PROF_OK[MAXPROF];
static int CPU2NODE[MAXCPU];
static long HZ;
static volatile sig_atomic_t stop;
static long ERRNO_HIST[256];
#define NOTE_ERR() (ERRNO_HIST[errno & 255]++)

static struct {
    const char *exe, *mode, *profile, *col, *rsl, *plan;
    int omp, rpn, start_step, start_delay_ms, poll_us, dph, single_domain, max_s, max_hot, pin_donor_master;
    double hot_ratio, lend_after_ms, hot_run_ms, hot_abs;
} O = {"wrf.exe", "observe", NULL, NULL, NULL, NULL, 4, 56, 2, 0, 1000, 2, 0, 3600, 64, 0, 1.5, 3.0, 3.0, 0.0};

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void on_sig(int s) { (void)s; stop = 1; }

static void load_profile(void) {
    if (!O.profile) return;
    FILE *f = fopen(O.profile, "r");
    if (!f) { perror(O.profile); exit(2); }
    char line[8192];
    int rc = -1, sc = -1, i = 0;
    if (!fgets(line, sizeof line, f)) { fprintf(stderr, "empty profile\n"); exit(2); }
    for (char *save, *tok = strtok_r(line, ",\r\n", &save); tok; tok = strtok_r(NULL, ",\r\n", &save), i++) {
        if (!strcmp(tok, "rank")) rc = i;
        if (O.col && !strcmp(tok, O.col)) sc = i;
    }
    if (rc < 0) { fprintf(stderr, "profile has no rank column\n"); exit(2); }
    if (sc < 0) { if (O.col) { fprintf(stderr, "profile has no column %s\n", O.col); exit(2); } sc = rc == 0 ? 1 : 0; }
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        int r = -1; double s = 0; i = 0;
        for (char *save, *tok = strtok_r(line, ",\r\n", &save); tok; tok = strtok_r(NULL, ",\r\n", &save), i++) {
            if (i == rc) r = atoi(tok);
            if (i == sc) s = atof(tok);
        }
        if (r >= 0 && r < MAXPROF) { PROF[r] = s; PROF_OK[r] = 1; n++; }
    }
    fclose(f);
    printf("ELASTICD profile %s rows=%d\n", O.profile, n);
}

static void load_numa(void) {
    for (int n = 0; n < 64; n++) {
        char p[128], b[4096];
        snprintf(p, sizeof p, "/sys/devices/system/node/node%d/cpulist", n);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        if (fgets(b, sizeof b, f)) {
            char *s = b;
            while (*s && *s != '\n') {
                long a = strtol(s, &s, 10), e = a;
                if (*s == '-') e = strtol(s + 1, &s, 10);
                for (long c = a; c <= e && c < MAXCPU; c++) CPU2NODE[c] = n;
                if (*s == ',') s++; else break;
            }
        }
        fclose(f);
    }
}

static int comm_is(pid_t pid) {
    char p[64], b[64], want[16];
    snprintf(p, sizeof p, "/proc/%d/comm", pid);
    int fd = open(p, O_RDONLY);
    if (fd < 0) return 0;
    ssize_t k = read(fd, b, sizeof b - 1);
    close(fd);
    if (k <= 0) return 0;
    b[k] = 0;
    char *nl = strchr(b, '\n');
    if (nl) *nl = 0;
    snprintf(want, sizeof want, "%s", O.exe);   /* kernel comm is truncated to 15 chars */
    return !strcmp(b, want);
}

static int env_rank(pid_t pid) {
    static char buf[1 << 20];
    char p[64];
    snprintf(p, sizeof p, "/proc/%d/environ", pid);
    int fd = open(p, O_RDONLY);
    if (fd < 0) return -2;
    ssize_t k, n = 0;
    while (n < (ssize_t)sizeof buf - 1 && (k = read(fd, buf + n, sizeof buf - 1 - n)) > 0) n += k;
    close(fd);
    buf[n] = 0;
    int r = -1;
    for (ssize_t i = 0; i < n;) {
        char *s = buf + i;
        size_t l = strnlen(s, (size_t)(n - i));
        if (l > 13 && !strncmp(s, "SLURM_PROCID=", 13)) r = atoi(s + 13);
        else if (r < 0 && l > 9 && !strncmp(s, "PMI_RANK=", 9)) r = atoi(s + 9);
        i += (ssize_t)l + 1;
    }
    return r;
}

static int discover(void) {
    DIR *d = opendir("/proc");
    struct dirent *e;
    int n = 0;
    uid_t me = getuid();
    while ((e = readdir(d))) {
        if (e->d_name[0] < '1' || e->d_name[0] > '9') continue;
        pid_t pid = (pid_t)atoi(e->d_name);
        char p[64];
        struct stat st;
        snprintf(p, sizeof p, "/proc/%d", pid);
        if (stat(p, &st) || st.st_uid != me || !comm_is(pid)) continue;
        int r = env_rank(pid);
        if (r == -2) { printf("ELASTICD FATAL environ of pid %d unreadable (%s)\n", pid, strerror(errno)); exit(3); }
        if (r < 0 || n >= MAXR) continue;
        memset(&R[n], 0, sizeof R[n]);
        R[n].rank = r; R[n].pid = pid; R[n].alive = 1;
        n++;
    }
    closedir(d);
    return n;
}

static void list_threads(rank_t *r) {
    char p[64];
    snprintf(p, sizeof p, "/proc/%d/task", r->pid);
    DIR *d = opendir(p);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        pid_t tid = (pid_t)atoi(e->d_name);
        int k;
        for (k = 0; k < r->nt; k++) if (r->t[k].tid == tid) break;
        if (k < r->nt || r->nt >= MAXT) continue;
        thr_t *t = &r->t[r->nt++];
        memset(t, 0, sizeof *t);
        t->tid = tid;
        char sp[96];
        snprintf(sp, sizeof sp, "/proc/%d/task/%d/stat", r->pid, tid);
        t->fd = open(sp, O_RDONLY);
        CPU_ZERO(&t->orig);
        t->orig_ok = sched_getaffinity(tid, sizeof t->orig, &t->orig) == 0;
    }
    closedir(d);
}

static int read_thr(thr_t *t) {
    char b[1024];
    if (t->fd < 0) return -1;
    ssize_t k = pread(t->fd, b, sizeof b - 1, 0);
    if (k <= 0) return -1;
    b[k] = 0;
    char *s = strrchr(b, ')');
    if (!s || !s[1] || !s[2]) return -1;
    s += 2;                               /* token 0 = state; token i = proc(5) field i+3 */
    t->state = *s;
    int field = 0;
    unsigned long long ut = 0, stt = 0;
    for (char *q = s; *q; q++) {
        if (*q != ' ') continue;
        field++;
        if (field == 11) ut = strtoull(q + 1, NULL, 10);
        else if (field == 12) stt = strtoull(q + 1, NULL, 10);
        else if (field == 36) { t->cpu = atoi(q + 1); break; }
    }
    t->ticks = ut + stt;
    return 0;
}

static void account(void) {
    for (int i = 0; i < NR; i++) {
        rank_t *r = &R[i];
        double m = 0, w = 0;
        int ok = 0;
        for (int j = 0; j < r->nteam; j++) {
            thr_t *t = &r->t[r->team[j]];
            if (read_thr(t)) continue;
            ok = 1;
            if (j == 0) m = (double)t->ticks / HZ; else w += (double)t->ticks / HZ;
        }
        if (ok) { r->master_s = m; r->workers_s = w; }
    }
}

static void identify_teams(void) {
    for (int i = 0; i < NR; i++) {
        list_threads(&R[i]);
        for (int k = 0; k < R[i].nt; k++) { read_thr(&R[i].t[k]); R[i].t[k].ticks0 = R[i].t[k].ticks; }
    }
    usleep(1000 * 1000);
    for (int i = 0; i < NR; i++) {
        rank_t *r = &R[i];
        list_threads(r);
        int idx[MAXT], m = 0, master = -1;
        for (int k = 0; k < r->nt; k++) {
            read_thr(&r->t[k]);
            if (r->t[k].tid == r->pid) master = k; else idx[m++] = k;
        }
        for (int a = 0; a < m; a++)
            for (int b = a + 1; b < m; b++)
                if (r->t[idx[b]].ticks - r->t[idx[b]].ticks0 > r->t[idx[a]].ticks - r->t[idx[a]].ticks0) {
                    int x = idx[a]; idx[a] = idx[b]; idx[b] = x;
                }
        r->nteam = 0;
        if (master >= 0) r->team[r->nteam++] = master;
        for (int j = 0; j < m && r->nteam < O.omp; j++)
            if (r->t[idx[j]].ticks - r->t[idx[j]].ticks0 >= 3) r->team[r->nteam++] = idx[j];
        r->nhome = 0;
        if (master >= 0)
            for (int c = 0; c < MAXCPU && r->nhome < 8; c++)
                if (CPU_ISSET(c, &r->t[master].orig)) r->home[r->nhome++] = c;
        r->dom = (O.single_domain || r->nhome == 0) ? 0 : CPU2NODE[r->home[0]];
        r->score = (r->rank < MAXPROF && PROF_OK[r->rank]) ? PROF[r->rank] : 0;
    }
}

static int cmp_score_desc(const void *a, const void *b) {
    double x = R[*(const int *)a].score, y = R[*(const int *)b].score;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void assign(void) {
    int seen[64] = {0};
    for (int i = 0; i < NR; i++) {
        int d = R[i].dom;
        if (d < 0 || d >= 64 || seen[d]) continue;
        seen[d] = 1;
        int ids[MAXR], n = 0;
        for (int j = 0; j < NR; j++) if (R[j].dom == d) ids[n++] = j;
        qsort(ids, (size_t)n, sizeof ids[0], cmp_score_desc);
        double med = R[ids[n / 2]].score;
        int maxhot = n / (1 + O.dph);
        if (maxhot > O.max_hot) maxhot = O.max_hot;
        int nh = 0;
        while (nh < maxhot && R[ids[nh]].score > 0 &&
               R[ids[nh]].score >= (O.hot_abs > 0 ? O.hot_abs : O.hot_ratio * med) &&
               R[ids[nh]].nteam - 1 >= O.dph && R[ids[nh]].nhome >= 1) nh++;
        int don[MAXR], ndn = 0;
        for (int j = n - 1; j >= nh && ndn < nh * O.dph; j--)
            if (R[ids[j]].nhome >= 2) don[ndn++] = ids[j];
        if (ndn < nh * O.dph) nh = ndn / O.dph;
        for (int h = 0; h < nh; h++) R[ids[h]].role = 1;
        for (int j = 0; j < nh * O.dph; j++) R[don[j]].role = 2;
        for (int k = 0; k < O.dph; k++)
            for (int h = 0; h < nh; h++) {
                slot_t *s = &S[NS++];
                memset(s, 0, sizeof *s);
                s->h = ids[h];
                s->d = don[k * nh + h];
                s->ti = R[ids[h]].team[R[ids[h]].nteam - 1 - k];
                s->lent = -1;
            }
        printf("ELASTICD dom=%d ranks=%d median_score=%.2f hot=%d donors=%d\n", d, n, med, nh, nh * O.dph);
    }
    for (int k = 0; k < NS; k++) {
        slot_t *s = &S[k];
        rank_t *h = &R[s->h], *d = &R[s->d];
        printf("PLAN slot=%d hot=%d(%.2f,home %d-%d) tid=%d donor=%d(%.2f,home %d-%d)\n", k, h->rank, h->score,
               h->home[0], h->home[h->nhome - 1], h->t[s->ti].tid, d->rank, d->score, d->home[0], d->home[d->nhome - 1]);
    }
}

static void assign_plan(void) {
    FILE *f = fopen(O.plan, "r");
    if (!f) { perror(O.plan); exit(2); }
    int hr, dr, nhot_slots[MAXR] = {0}, skipped = 0, xdom = 0;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%d %d", &hr, &dr) != 2) continue;
        int h = -1, d = -1;
        for (int i = 0; i < NR; i++) { if (R[i].rank == hr) h = i; if (R[i].rank == dr) d = i; }
        if (h < 0 || d < 0) continue;              /* pair lives on another node */
        if (R[h].nteam - 1 - nhot_slots[h] < 1 || R[d].nhome < 2 || NS >= MAXR * 4) { skipped++; continue; }
        if (R[h].dom != R[d].dom) xdom++;
        slot_t *s = &S[NS++];
        memset(s, 0, sizeof *s);
        s->h = h; s->d = d; s->lent = -1;
        s->ti = R[h].team[R[h].nteam - 1 - nhot_slots[h]];
        nhot_slots[h]++;
        R[h].role = 1; R[d].role = 2;
    }
    fclose(f);
    printf("ELASTICD plan %s slots=%d skipped=%d cross_domain=%d\n", O.plan, NS, skipped, xdom);
    for (int k = 0; k < NS; k++) {
        slot_t *s = &S[k];
        rank_t *h = &R[s->h], *d = &R[s->d];
        printf("PLAN slot=%d hot=%d(nteam %d,home %d-%d) tid=%d donor=%d(nteam %d,home %d-%d)\n", k, h->rank, h->nteam,
               h->home[0], h->home[h->nhome - 1], h->t[s->ti].tid, d->rank, d->nteam, d->home[0], d->home[d->nhome - 1]);
    }
}

static void self_pin(void) {
    int best = -1;
    for (int i = 0; i < NR; i++)
        if (R[i].role == 0 && R[i].nhome > 0 && (best < 0 || R[i].score < R[best].score)) best = i;
    if (best < 0)
        for (int i = 0; i < NR; i++)
            if (R[i].role == 2 && R[i].nhome > 0 && (best < 0 || R[i].score < R[best].score)) best = i;
    if (best < 0) { printf("ELASTICD self_pin none\n"); return; }
    cpu_set_t m;
    CPU_ZERO(&m);
    CPU_SET(R[best].home[0], &m);
    int rc = sched_setaffinity(0, sizeof m, &m);
    printf("ELASTICD self_pin cpu=%d (rank %d role %d) rc=%d\n", R[best].home[0], R[best].rank, R[best].role, rc);
}

static int pin1(pid_t tid, int cpu) {
    cpu_set_t m;
    CPU_ZERO(&m);
    CPU_SET(cpu, &m);
    return sched_setaffinity(tid, sizeof m, &m);
}

static void restore_all(void) {
    int n = 0, err = 0;
    for (int i = 0; i < NR; i++)
        for (int k = 0; k < R[i].nt; k++)
            if (R[i].t[k].orig_ok) {
                if (sched_setaffinity(R[i].t[k].tid, sizeof R[i].t[k].orig, &R[i].t[k].orig)) { if (errno != ESRCH) err++; }
                else n++;
            }
    printf("ELASTICD restore ok=%d err=%d\n", n, err);
}

static int any_alive(void) {
    for (int i = 0; i < NR; i++) {
        char p[64];
        snprintf(p, sizeof p, "/proc/%d", R[i].pid);
        if (access(p, F_OK) == 0) return 1;
    }
    return 0;
}

static int count_timing(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[512];
    int n = 0;
    while (fgets(line, sizeof line, f)) if (strstr(line, "Timing for main")) n++;
    fclose(f);
    return n;
}

static void run_static(uint64_t deadline) {
    uint64_t t0 = now_ns();
    for (int k = 0; k < NS; k++) {
        slot_t *s = &S[k];
        rank_t *d = &R[s->d];
        int core = d->home[d->nhome - 1];
        if (pin1(R[s->h].t[s->ti].tid, core)) { s->nerr++; NOTE_ERR(); }
        else { s->lent = core; s->t_lend = t0; s->nlend = 1; }
    }
    while (!stop && now_ns() < deadline && any_alive()) { account(); usleep(200 * 1000); }
    uint64_t t1 = now_ns();
    for (int k = 0; k < NS; k++) if (S[k].lent >= 0) { S[k].lent_ns += t1 - S[k].t_lend; S[k].lent = -1; }
}

static void run_dynamic(uint64_t deadline, long *polls, uint64_t *cost_sum, uint64_t *cost_max) {
    uint64_t per = (uint64_t)O.poll_us * 1000ull;
    uint64_t lend_after = (uint64_t)(O.lend_after_ms * 1e6), hot_run = (uint64_t)(O.hot_run_ms * 1e6);
    int inv[MAXR], ni = 0;
    for (int i = 0; i < NR; i++) if (R[i].role) inv[ni++] = i;
    uint64_t next = now_ns(), last_acct = 0;
    while (!stop) {
        uint64_t t0 = now_ns();
        if (t0 > deadline) break;
        int alive = 0;
        for (int a = 0; a < ni; a++) {
            rank_t *r = &R[inv[a]];
            int busy = 0, ok = 0;
            for (int j = 0; j < r->nteam; j++) {
                thr_t *t = &r->t[r->team[j]];
                if (read_thr(t) == 0) ok = 1;
                if (j > 0 && t->state == 'R') busy = 1;
            }
            r->alive = ok;
            if (!ok) continue;
            alive++;
            if (busy != r->busy || r->since == 0) { r->busy = busy; r->since = t0; }
        }
        if (!alive) break;
        for (int k = 0; k < NS; k++) {
            slot_t *s = &S[k];
            rank_t *h = &R[s->h], *d = &R[s->d];
            thr_t *bt = &h->t[s->ti], *dm = &d->t[d->team[0]];
            if (!h->alive || !d->alive) continue;
            if (s->lent < 0) {
                if (h->busy && t0 - h->since >= hot_run && !d->busy && t0 - d->since >= lend_after) {
                    int core = d->home[d->nhome - 1];
                    if (dm->state == 'R' && dm->cpu == core && d->nhome > 1) core = d->home[0];
                    if (pin1(bt->tid, core)) { s->nerr++; NOTE_ERR(); }
                    else { s->lent = core; s->t_lend = t0; s->nlend++; }
                }
            } else {
                int why = 0;
                if (d->busy || (dm->state == 'R' && dm->cpu == s->lent)) why = 1;
                else if (!h->busy) why = 2;
                if (why) {
                    if (sched_setaffinity(bt->tid, sizeof bt->orig, &bt->orig)) { s->nerr++; NOTE_ERR(); }
                    s->lent_ns += t0 - s->t_lend;
                    s->lent = -1;
                    if (why == 1) s->ncollide++; else s->nhotend++;
                }
            }
        }
        uint64_t c = now_ns() - t0;
        *cost_sum += c;
        if (c > *cost_max) *cost_max = c;
        (*polls)++;
        if (t0 - last_acct > 1000000000ull) { account(); last_acct = t0; }
        next += per;
        uint64_t now = now_ns();
        if (next <= now) next = now + per;
        struct timespec ts = {(time_t)(next / 1000000000ull), (long)(next % 1000000000ull)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
    }
    uint64_t t1 = now_ns();
    for (int k = 0; k < NS; k++)
        if (S[k].lent >= 0) {
            if (sched_setaffinity(R[S[k].h].t[S[k].ti].tid, sizeof R[S[k].h].t[S[k].ti].orig, &R[S[k].h].t[S[k].ti].orig)) S[k].nerr++;
            S[k].lent_ns += t1 - S[k].t_lend;
            S[k].lent = -1;
        }
}

int main(int argc, char **argv) {
    static struct option lo[] = {
        {"exe", 1, 0, 'e'}, {"mode", 1, 0, 'm'}, {"profile", 1, 0, 'p'}, {"col", 1, 0, 'c'}, {"rsl", 1, 0, 'r'},
        {"omp", 1, 0, 'o'}, {"rpn", 1, 0, 'n'}, {"start-step", 1, 0, 's'}, {"start-delay-ms", 1, 0, 'S'},
        {"poll-us", 1, 0, 'u'}, {"donors-per-hot", 1, 0, 'k'}, {"single-domain", 0, 0, '1'}, {"max-s", 1, 0, 'x'},
        {"max-hot", 1, 0, 'H'}, {"hot-ratio", 1, 0, 'h'}, {"lend-after-ms", 1, 0, 'l'}, {"hot-run-ms", 1, 0, 'b'},
        {"pin-donor-master", 0, 0, 'M'}, {"hot-abs", 1, 0, 'A'}, {"plan", 1, 0, 'P'}, {0, 0, 0, 0}};
    int ch;
    while ((ch = getopt_long(argc, argv, "", lo, NULL)) != -1) {
        switch (ch) {
        case 'e': O.exe = optarg; break;
        case 'm': O.mode = optarg; break;
        case 'p': O.profile = optarg; break;
        case 'c': O.col = optarg; break;
        case 'r': O.rsl = optarg; break;
        case 'o': O.omp = atoi(optarg); break;
        case 'n': O.rpn = atoi(optarg); break;
        case 's': O.start_step = atoi(optarg); break;
        case 'S': O.start_delay_ms = atoi(optarg); break;
        case 'u': O.poll_us = atoi(optarg); break;
        case 'k': O.dph = atoi(optarg); break;
        case '1': O.single_domain = 1; break;
        case 'x': O.max_s = atoi(optarg); break;
        case 'H': O.max_hot = atoi(optarg); break;
        case 'h': O.hot_ratio = atof(optarg); break;
        case 'l': O.lend_after_ms = atof(optarg); break;
        case 'b': O.hot_run_ms = atof(optarg); break;
        case 'M': O.pin_donor_master = 1; break;
        case 'A': O.hot_abs = atof(optarg); break;
        case 'P': O.plan = optarg; break;
        default: fprintf(stderr, "bad option\n"); return 2;
        }
    }
    int mode = !strcmp(O.mode, "observe") ? 0 : !strcmp(O.mode, "static") ? 1 : !strcmp(O.mode, "dynamic") ? 2 : -1;
    if (mode < 0) { fprintf(stderr, "mode must be observe|static|dynamic\n"); return 2; }
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);
    HZ = sysconf(_SC_CLK_TCK);
    char host[256] = "?";
    gethostname(host, sizeof host);
    uint64_t start = now_ns(), deadline = start + (uint64_t)O.max_s * 1000000000ull;
    load_profile();
    load_numa();
    printf("ELASTICD host=%s mode=%s exe=%s omp=%d rpn=%d dph=%d poll_us=%d hot_ratio=%.2f hot_abs=%.2f pin_donor_master=%d lend_after_ms=%.1f hot_run_ms=%.1f\n",
           host, O.mode, O.exe, O.omp, O.rpn, O.dph, O.poll_us, O.hot_ratio, O.hot_abs, O.pin_donor_master, O.lend_after_ms, O.hot_run_ms);

    while (!stop && now_ns() < deadline) {
        NR = discover();
        if (NR >= O.rpn) break;
        usleep(200 * 1000);
    }
    if (NR < O.rpn) { printf("ELASTICD FATAL found %d of %d ranks, no actuation\n", NR, O.rpn); return 4; }
    if (O.rsl) {
        while (!stop && now_ns() < deadline && count_timing(O.rsl) < O.start_step) usleep(200 * 1000);
    } else if (O.start_delay_ms > 0) {
        usleep((useconds_t)O.start_delay_ms * 1000);
    }
    printf("ELASTICD discovered %d ranks after %.1f s; identifying teams\n", NR, (now_ns() - start) / 1e9);
    identify_teams();
    int short_team = 0;
    for (int i = 0; i < NR; i++) if (R[i].nteam < O.omp) short_team++;
    printf("ELASTICD teams identified: short_teams=%d (expected size %d)\n", short_team, O.omp);
    if (mode) { if (O.plan) assign_plan(); else assign(); }
    self_pin();
    if (mode && O.pin_donor_master) {
        int n = 0, err = 0;
        for (int i = 0; i < NR; i++)
            if (R[i].role == 2 && R[i].nteam > 0 && R[i].nhome >= 2) {
                if (pin1(R[i].t[R[i].team[0]].tid, R[i].home[0])) err++; else n++;
            }
        printf("ELASTICD pin_donor_master ok=%d err=%d\n", n, err);
    }

    long polls = 0;
    uint64_t cost_sum = 0, cost_max = 0, t_act = now_ns();
    if (mode == 1) run_static(deadline);
    else if (mode == 2) run_dynamic(deadline, &polls, &cost_sum, &cost_max);
    else while (!stop && now_ns() < deadline && any_alive()) { account(); usleep(500 * 1000); }
    double act_s = (now_ns() - t_act) / 1e9;

    long tl = 0, tc = 0, th = 0, te = 0;
    double tls = 0;
    for (int k = 0; k < NS; k++) {
        slot_t *s = &S[k];
        double ls = s->lent_ns / 1e9;
        printf("SLOT %d hot=%d donor=%d nlend=%ld ncollide=%ld nhotend=%ld lent_s=%.3f lent_frac=%.3f mean_borrow_ms=%.2f nerr=%ld\n",
               k, R[s->h].rank, R[s->d].rank, s->nlend, s->ncollide, s->nhotend, ls, act_s > 0 ? ls / act_s : 0,
               s->nlend ? 1e3 * ls / s->nlend : 0, s->nerr);
        tl += s->nlend; tc += s->ncollide; th += s->nhotend; te += s->nerr; tls += ls;
    }
    printf("TOTAL host=%s mode=%s act_s=%.1f slots=%d lends=%ld collides=%ld hotends=%ld repins=%ld lent_s=%.2f errs=%ld polls=%ld poll_cost_mean_us=%.1f poll_cost_max_us=%.1f\n",
           host, O.mode, act_s, NS, tl, tc, th, 2 * (tc + th) + (mode == 1 ? NS : 0), tls, te, polls,
           polls ? cost_sum / 1e3 / polls : 0, cost_max / 1e3);
    for (int i = 0; i < NR; i++)
        printf("RANKCPU host=%s rank=%d role=%d score=%.2f nteam=%d home=%d-%d master_s=%.2f workers_s=%.2f\n", host, R[i].rank,
               R[i].role, R[i].score, R[i].nteam, R[i].nhome ? R[i].home[0] : -1, R[i].nhome ? R[i].home[R[i].nhome - 1] : -1,
               R[i].master_s, R[i].workers_s);
    printf("ERRNO host=%s", host);
    for (int e = 0; e < 256; e++) if (ERRNO_HIST[e]) printf(" %s=%ld", strerror(e), ERRNO_HIST[e]);
    printf("\n");
    restore_all();
    return 0;
}
