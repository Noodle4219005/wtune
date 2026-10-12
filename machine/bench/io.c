/* io.c -- write bandwidth to a target directory: (a) a single writer doing
 * one large sequential file (proxy for a rank-0-serial netCDF write) and
 * (b) N concurrent writers each to their own file (proxy for
 * per-rank/split-file I/O). Both include fsync in the timed region, since
 * that's when the write actually has to leave the page cache.
 *
 * Emits exactly one JSON line to stdout.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Write `bytes` total to `path` in `block_bytes` chunks, fsync, close.
 * Returns elapsed seconds, or -1.0 on error. */
static double write_file(const char *path, long bytes, long block_bytes, char *buf) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1.0;
    double t0 = now_s();
    long remaining = bytes;
    while (remaining > 0) {
        long chunk = remaining < block_bytes ? remaining : block_bytes;
        ssize_t w = write(fd, buf, chunk);
        if (w < 0) { close(fd); return -1.0; }
        remaining -= w;
    }
    fsync(fd);
    double elapsed = now_s() - t0;
    close(fd);
    return elapsed;
}

int main(int argc, char **argv) {
    const char *dir = ".";
    long size_mb = 100;
    int nwriters = 4;
    long block_kb = 1024;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--dir")) dir = argv[i + 1];
        else if (!strcmp(argv[i], "--size-mb")) size_mb = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--writers")) nwriters = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--block-kb")) block_kb = atol(argv[i + 1]);
    }
    long block_bytes = block_kb * 1024;
    long bytes = size_mb * 1024L * 1024L;
    char *buf = malloc(block_bytes);
    memset(buf, 0x42, block_bytes);

    /* (a) single writer, large sequential file */
    char path1[512];
    snprintf(path1, sizeof(path1), "%s/machine_io_single_%d.tmp", dir, (int)getpid());
    double t_single = write_file(path1, bytes, block_bytes, buf);
    unlink(path1);
    double gbps_single = (t_single > 0) ? (double)bytes / t_single / 1e9 : -1.0;

    /* (b) N concurrent writers, each its own file, each `bytes` long.
     * fork()+pipe: each child times its own write+fsync and reports back
     * (elapsed_s, ok) over its pipe; the parent's aggregate bandwidth is
     * total bytes moved by all writers divided by the SLOWEST writer's
     * time, i.e. the wall time the whole batch actually took. */
    int pipes[256][2];
    if (nwriters > 256) nwriters = 256;
    for (int w = 0; w < nwriters; w++) {
        if (pipe(pipes[w]) != 0) { fprintf(stderr, "io: pipe failed\n"); return 1; }
    }
    double t0 = now_s();
    for (int w = 0; w < nwriters; w++) {
        pid_t pid = fork();
        if (pid == 0) {
            /* child */
            for (int k = 0; k < nwriters; k++) if (k != w) { close(pipes[k][0]); close(pipes[k][1]); }
            close(pipes[w][0]);
            char path[512];
            snprintf(path, sizeof(path), "%s/machine_io_conc_%d_%d.tmp", dir, (int)getpid(), w);
            double el = write_file(path, bytes, block_bytes, buf);
            unlink(path);
            ssize_t wr = write(pipes[w][1], &el, sizeof(el));
            (void)wr;
            close(pipes[w][1]);
            _exit(el >= 0 ? 0 : 1);
        }
        close(pipes[w][1]);
    }
    double max_child_elapsed = 0.0;
    int ok_writers = 0;
    for (int w = 0; w < nwriters; w++) {
        double el = -1.0;
        ssize_t r = read(pipes[w][0], &el, sizeof(el));
        close(pipes[w][0]);
        int status;
        wait(&status);
        if (r == (ssize_t)sizeof(el) && el >= 0) {
            ok_writers++;
            if (el > max_child_elapsed) max_child_elapsed = el;
        }
    }
    double wall_elapsed = now_s() - t0;
    double aggregate_gbps = (max_child_elapsed > 0 && ok_writers > 0)
        ? ((double)ok_writers * (double)bytes) / max_child_elapsed / 1e9 : -1.0;

    printf("{\"bench\":\"io\",\"dir\":\"%s\",\"size_mb\":%ld,\"block_kb\":%ld,"
           "\"single_writer\":{\"bytes\":%ld,\"seconds\":%.4f,\"gbps\":%.4f},"
           "\"concurrent_writers\":{\"n\":%d,\"ok\":%d,\"bytes_each\":%ld,"
           "\"wall_seconds\":%.4f,\"slowest_writer_seconds\":%.4f,\"aggregate_gbps\":%.4f}}\n",
           dir, size_mb, block_kb, bytes, t_single, gbps_single,
           nwriters, ok_writers, bytes, wall_elapsed, max_child_elapsed, aggregate_gbps);

    free(buf);
    return 0;
}
