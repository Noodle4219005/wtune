/* halo.c -- 2-D Cartesian nearest-neighbour halo exchange sized like WRF's
 * dyn_em halo (width 5, e_vert levels, a few fields per direction), at
 * patch sizes representative of WRF decompositions (20..200 cells/side),
 * plus a plain point-to-point ping-pong latency/bandwidth sweep.
 *
 * This binary does NOT self-classify intra- vs inter-node: the driver
 * (run_machine_profile.sh) launches it twice with different rank
 * placement (--map-by) and passes --tag intra / --tag inter so the two
 * JSON lines can be told apart after the fact. That is far more robust
 * than trying to infer placement from MPI_Get_processor_name, which
 * degenerates at odd rank counts / non-square process grids.
 *
 * Emits exactly one JSON line to stdout, from rank 0 only.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int parse_int_list(const char *s, int *out, int max_n) {
    if (!s) return 0;
    int n = 0;
    char buf[256];
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    char *tok = strtok(buf, ",");
    while (tok && n < max_n) { out[n++] = atoi(tok); tok = strtok(NULL, ","); }
    return n;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char *tag = "run";
    int halo = 5, evert = 33, nfields = 4, reps = 20;
    int patches[32]; int npatches = 0;
    long pp_max_bytes = 1L << 20;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--tag")) tag = argv[i + 1];
        else if (!strcmp(argv[i], "--halo")) halo = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--evert")) evert = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--nfields")) nfields = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--patches")) npatches = parse_int_list(argv[i + 1], patches, 32);
        else if (!strcmp(argv[i], "--pingpong-max-bytes")) pp_max_bytes = atol(argv[i + 1]);
    }
    if (npatches == 0) { int def[4] = {20, 50, 100, 200}; memcpy(patches, def, sizeof(def)); npatches = 4; }

    int dims[2] = {0, 0};
    MPI_Dims_create(nranks, 2, dims);
    int periods[2] = {1, 1}; /* periodic: every rank does the same 4-neighbour work,
                                 no special-casing of domain-edge ranks needed */
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);
    int west, east, south, north;
    MPI_Cart_shift(cart, 0, 1, &west, &east);
    MPI_Cart_shift(cart, 1, 1, &south, &north);

    char exch_json[8192] = "";
    int exch_off = 0;

    for (int pi = 0; pi < npatches; pi++) {
        int side = patches[pi];
        /* East/west face buffer: halo columns x evert x side, times nfields.
         * North/south face buffer: side x evert x halo, times nfields. */
        size_t ew_elems = (size_t)halo * evert * side * nfields;
        size_t ns_elems = (size_t)side * evert * halo * nfields;
        double *sendW = malloc(ew_elems * sizeof(double)), *recvW = malloc(ew_elems * sizeof(double));
        double *sendE = malloc(ew_elems * sizeof(double)), *recvE = malloc(ew_elems * sizeof(double));
        double *sendS = malloc(ns_elems * sizeof(double)), *recvS = malloc(ns_elems * sizeof(double));
        double *sendN = malloc(ns_elems * sizeof(double)), *recvN = malloc(ns_elems * sizeof(double));
        for (size_t i = 0; i < ew_elems; i++) sendW[i] = sendE[i] = 1.0;
        for (size_t i = 0; i < ns_elems; i++) sendS[i] = sendN[i] = 1.0;

        MPI_Barrier(cart);
        double t0 = MPI_Wtime();
        for (int r = 0; r < reps; r++) {
            MPI_Request req[8];
            MPI_Irecv(recvW, ew_elems, MPI_DOUBLE, west, 0, cart, &req[0]);
            MPI_Irecv(recvE, ew_elems, MPI_DOUBLE, east, 1, cart, &req[1]);
            MPI_Irecv(recvS, ns_elems, MPI_DOUBLE, south, 2, cart, &req[2]);
            MPI_Irecv(recvN, ns_elems, MPI_DOUBLE, north, 3, cart, &req[3]);
            MPI_Isend(sendE, ew_elems, MPI_DOUBLE, east, 0, cart, &req[4]);
            MPI_Isend(sendW, ew_elems, MPI_DOUBLE, west, 1, cart, &req[5]);
            MPI_Isend(sendN, ns_elems, MPI_DOUBLE, north, 2, cart, &req[6]);
            MPI_Isend(sendS, ns_elems, MPI_DOUBLE, south, 3, cart, &req[7]);
            MPI_Waitall(8, req, MPI_STATUSES_IGNORE);
        }
        double local_elapsed = MPI_Wtime() - t0;
        double max_elapsed;
        MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

        if (rank == 0) {
            double per_exchange_s = max_elapsed / reps;
            double bytes_per_rank_per_exchange =
                2.0 * ew_elems * sizeof(double) + 2.0 * ns_elems * sizeof(double); /* sent+recv, 4 neighbours */
            double us = per_exchange_s * 1e6;
            double gbps = bytes_per_rank_per_exchange / per_exchange_s / 1e9;
            exch_off += snprintf(exch_json + exch_off, sizeof(exch_json) - exch_off,
                "%s{\"patch\":%d,\"msg_bytes_ew\":%zu,\"msg_bytes_ns\":%zu,"
                "\"us_per_exchange\":%.3f,\"gbps\":%.3f}",
                pi ? "," : "", side, ew_elems * sizeof(double), ns_elems * sizeof(double), us, gbps);
        }
        free(sendW); free(recvW); free(sendE); free(recvE);
        free(sendS); free(recvS); free(sendN); free(recvN);
    }

    /* Plain ping-pong between rank 0 and rank 1 on MPI_COMM_WORLD (not the
     * cartesian comm): pure latency/bandwidth curve, independent of halo
     * shape. */
    char pp_json[4096] = "";
    int pp_off = 0;
    int did_pingpong = 0;
    if (nranks >= 2) {
        did_pingpong = 1;
        int first = 1;
        for (long nbytes = 8; nbytes <= pp_max_bytes; nbytes *= 8) {
            char *buf = malloc(nbytes);
            memset(buf, 1, nbytes);
            const int pp_reps = 50;
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            if (rank == 0) {
                for (int r = 0; r < pp_reps; r++) {
                    MPI_Send(buf, nbytes, MPI_BYTE, 1, 10, MPI_COMM_WORLD);
                    MPI_Recv(buf, nbytes, MPI_BYTE, 1, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            } else if (rank == 1) {
                for (int r = 0; r < pp_reps; r++) {
                    MPI_Recv(buf, nbytes, MPI_BYTE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf, nbytes, MPI_BYTE, 0, 11, MPI_COMM_WORLD);
                }
            }
            double elapsed = MPI_Wtime() - t0;
            if (rank == 0) {
                double rtt_s = elapsed / pp_reps;
                double us = rtt_s * 1e6 / 2.0; /* one-way */
                double gbps = (double)nbytes / (rtt_s / 2.0) / 1e9;
                pp_off += snprintf(pp_json + pp_off, sizeof(pp_json) - pp_off,
                    "%s{\"bytes\":%ld,\"us\":%.3f,\"gbps\":%.4f}", first ? "" : ",", nbytes, us, gbps);
                first = 0;
            }
            free(buf);
        }
    }

    if (rank == 0) {
        if (did_pingpong) {
            printf("{\"bench\":\"halo\",\"tag\":\"%s\",\"nranks\":%d,\"px\":%d,\"py\":%d,"
                   "\"halo_width\":%d,\"e_vert\":%d,\"nfields\":%d,"
                   "\"exchange\":[%s],\"pingpong\":[%s]}\n",
                   tag, nranks, dims[0], dims[1], halo, evert, nfields, exch_json, pp_json);
        } else {
            printf("{\"bench\":\"halo\",\"tag\":\"%s\",\"nranks\":%d,\"px\":%d,\"py\":%d,"
                   "\"halo_width\":%d,\"e_vert\":%d,\"nfields\":%d,"
                   "\"exchange\":[%s],\"pingpong\":null}\n",
                   tag, nranks, dims[0], dims[1], halo, evert, nfields, exch_json);
        }
    }

    MPI_Finalize();
    return 0;
}
