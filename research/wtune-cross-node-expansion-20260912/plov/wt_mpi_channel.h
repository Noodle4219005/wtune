#ifndef WT_MPI_CHANNEL_H
#define WT_MPI_CHANNEL_H

#include <stddef.h>
#include <stdint.h>

#include <mpi.h>

/* Standalone, point-to-point MPI transport. The creating thread is the only
 * thread allowed to call this API. Calls never wait, retry, cancel, free a
 * request, or abort the MPI job. The caller supplies an absolute
 * CLOCK_MONOTONIC deadline in nanoseconds to poll(). */
enum wt_mpi_channel_status {
  WT_MPI_CHANNEL_OK = 0,
  WT_MPI_CHANNEL_ARGUMENT,
  WT_MPI_CHANNEL_MPI_UNAVAILABLE,
  WT_MPI_CHANNEL_THREAD_LEVEL,
  WT_MPI_CHANNEL_THREAD_AFFINITY,
  WT_MPI_CHANNEL_PEER,
  WT_MPI_CHANNEL_LIMIT,
  WT_MPI_CHANNEL_ALLOCATION,
  WT_MPI_CHANNEL_BUSY,
  WT_MPI_CHANNEL_TIMEOUT,
  WT_MPI_CHANNEL_POISONED,
  WT_MPI_CHANNEL_MPI_ERROR,
  WT_MPI_CHANNEL_CLOCK_ERROR
};

enum wt_mpi_channel_event_kind {
  WT_MPI_CHANNEL_EVENT_NONE = 0,
  WT_MPI_CHANNEL_EVENT_SEND_COMPLETE = 1U << 0,
  WT_MPI_CHANNEL_EVENT_RECEIVE_READY = 1U << 1,
  /* A local deadline expired while communication was active. Any send that
   * remains pending has unknown delivery; poll() must reap it later. */
  WT_MPI_CHANNEL_EVENT_UNKNOWN_DELIVERY = 1U << 2
};

struct wt_mpi_channel_event {
  unsigned kinds;
  const void *received_bytes;
  size_t received_length;
  int mpi_error;
};

struct wt_mpi_channel;

/* Purely local creator-thread queries; they perform no MPI call and change
 * their output only on success. */
enum wt_mpi_channel_status wt_mpi_channel_get_peer(
    const struct wt_mpi_channel *channel, int *out_peer);
enum wt_mpi_channel_status wt_mpi_channel_get_capacity(
    const struct wt_mpi_channel *channel, size_t *out_capacity);

/* Collective over parent. Every participant must supply the same valid parent
 * and must enter with MPI initialized, not finalized, at least
 * MPI_THREAD_FUNNELED, and on MPI's main thread. Rank-local out, peer and
 * max_frame validation is collectively agreed, so all participants return the
 * same validation failure. The channel owns a duplicated communicator fixed
 * to tag 0 and exactly two max_frame-byte buffers. */
enum wt_mpi_channel_status wt_mpi_channel_create(MPI_Comm parent, int peer,
    size_t max_frame, struct wt_mpi_channel **out);

/* Copies bytes before starting MPI_Isend. Empty and oversized frames are
 * rejected. At most one send may be active. */
enum wt_mpi_channel_status wt_mpi_channel_begin_send(
    struct wt_mpi_channel *channel, const void *bytes, size_t length);

/* Arms exact-peer/tag probing. MPI_Iprobe is first called by poll(). At most
 * one receive may be armed or ready. */
enum wt_mpi_channel_status wt_mpi_channel_begin_recv(
    struct wt_mpi_channel *channel);

/* Performs one nonblocking progress pass. A deadline poisons the channel and
 * reports UNKNOWN_DELIVERY without discarding requests or owned buffers.
 * Polling a poisoned channel remains legal solely to reap pending requests.
 * MPI errors also poison it; MPI continuation after such an error is not
 * guaranteed by this component. */
enum wt_mpi_channel_status wt_mpi_channel_poll(
    struct wt_mpi_channel *channel, int64_t deadline_monotonic_ns,
    struct wt_mpi_channel_event *event);

/* The receive pointer remains owned by the channel and valid until release. */
enum wt_mpi_channel_status wt_mpi_channel_release_receive(
    struct wt_mpi_channel *channel);

/* Purely local: BUSY means an operation or unreleased receive remains. */
enum wt_mpi_channel_status wt_mpi_channel_destroy_ready(
    const struct wt_mpi_channel *channel);

/* Collective over the private communicator. The caller must establish global
 * quiescence first (destroy_ready()==OK at every participant). An all-reduce
 * prevents ranks from branching differently into MPI_Comm_free if that
 * precondition is accidentally violated. MPI_Finalize belongs after this
 * succeeds. Unreapable peer loss requires an explicit caller/job abort. */
enum wt_mpi_channel_status wt_mpi_channel_destroy_collective(
    struct wt_mpi_channel **channel);

#endif
