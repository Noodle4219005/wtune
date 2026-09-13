#define _POSIX_C_SOURCE 200809L

#include "wt_mpi_remote_internal.h"
#include <limits.h>
#include <string.h>
#include <time.h>

void wt_mpi_remote_private_clear_event(struct wt_mpi_remote_event *event)
{
    if (event) *event = (struct wt_mpi_remote_event){
        WT_MPI_REMOTE_EVENT_NONE, WT_MPI_CHANNEL_OK, WT_LEDGER_OK, WT_POLICY_OK};
}

int wt_mpi_remote_private_decode_trusted(
    const struct wt_ledger_contract *contract,
    const uint8_t *bytes, size_t length, uint8_t kind, uint32_t phase,
    struct wt_wire_frame *out)
{
    if (!contract || !out ||
        wt_wire_decode(bytes, length, &contract->limits,
                       contract->schema_fingerprint, &contract->abi, out) != WT_WIRE_OK)
        return 0;
    return out->kind == kind && out->id.phase == phase &&
        out->id.owner_rank == contract->owner_rank &&
        out->id.owner_generation == contract->owner_generation &&
        memcmp(out->id.run_uuid, contract->run_uuid, sizeof contract->run_uuid) == 0;
}

int wt_mpi_remote_private_choice_matches(
    const struct wt_mpi_remote_endpoint *endpoint,
    const struct wt_policy_choice *choice)
{
    return endpoint && choice && choice->peer_rank_id == endpoint->rank_id &&
        choice->peer_node_id == endpoint->node_id;
}

enum wt_mpi_remote_status wt_mpi_remote_private_deadline_expired(
    int64_t deadline_monotonic_ns, int *out_expired)
{
#ifdef WT_MPI_REMOTE_TEST_CLOCK
    extern int64_t wt_mpi_remote_test_monotonic_now_ns(void);
    int64_t injected_now;
#else
    struct timespec now;
    uint64_t now_ns;
#endif
    if (deadline_monotonic_ns < 0 || out_expired == NULL)
        return WT_MPI_REMOTE_ARGUMENT;
#ifdef WT_MPI_REMOTE_TEST_CLOCK
    injected_now = wt_mpi_remote_test_monotonic_now_ns();
    if (injected_now < 0)
        return WT_MPI_REMOTE_TRANSPORT;
    *out_expired = injected_now >= deadline_monotonic_ns;
#else
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0 ||
        now.tv_nsec < 0 || now.tv_nsec >= 1000000000L ||
        (uint64_t)now.tv_sec >
            ((uint64_t)INT64_MAX - (uint64_t)now.tv_nsec) / UINT64_C(1000000000))
        return WT_MPI_REMOTE_TRANSPORT;
    now_ns = (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
    *out_expired = now_ns >= (uint64_t)deadline_monotonic_ns;
#endif
    return WT_MPI_REMOTE_OK;
}
