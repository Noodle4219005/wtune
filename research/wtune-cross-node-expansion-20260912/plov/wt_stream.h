#ifndef WT_STREAM_H
#define WT_STREAM_H
#include <stddef.h>
#include <stdint.h>

/* POSIX stream-socket transport; not an MPI or WRF adapter. The caller owns
 * the descriptor and serializes access, including close/flags changes.
 * Descriptor must already be nonblocking. Requires per-call MSG_NOSIGNAL or
 * per-socket SO_NOSIGPIPE; other platforms return UNSUPPORTED at init.
 * No process-global signal changes.
 * After an I/O error, timeout or malformed length, discard the connection:
 * its framing position is unknown and no retry is performed. */
enum wt_stream_status {
    WT_STREAM_OK=0, WT_STREAM_ARGUMENT, WT_STREAM_LIMIT, WT_STREAM_TIMEOUT,
    WT_STREAM_CLOSED, WT_STREAM_IO, WT_STREAM_POISONED, WT_STREAM_UNSUPPORTED
};
struct wt_stream {
    int fd, poisoned;
    size_t max_frame;
    uint32_t timeout_ms;
};
int wt_stream_init(struct wt_stream *, int fd, size_t max_frame, uint32_t timeout_ms);
int wt_stream_send(struct wt_stream *, const void *frame, size_t length);
/* Caller provides actual capacity. Buffer may contain partial bytes on error;
 * out_length changes only on success. A frame still requires wt_wire_decode
 * and ledger validation before it can be used. Empty frames are permitted. */
int wt_stream_receive(struct wt_stream *, void *buffer, size_t capacity,
                      size_t *out_length);
#endif
