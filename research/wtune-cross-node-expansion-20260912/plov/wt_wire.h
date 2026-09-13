#ifndef WT_WIRE_H
#define WT_WIRE_H

#include <stddef.h>
#include <stdint.h>

/*
 * WT Wire is a portable control-plane frame format.  All integers on the wire
 * are little-endian and this interface never serializes a native C/Fortran
 * struct or pointer.  Adapters must turn Fortran arguments into a canonical
 * byte payload before calling wt_wire_encode(); multi-byte payload scalars are
 * little-endian too.
 *
 * CRC-32 detects accidental corruption only; it provides no authentication.
 * schema_fingerprint and request_fingerprint are caller-supplied digests.
 * Receivers must compare complete canonical requests in their ledger before
 * treating duplicate request fingerprints as equivalent.
 */

#define WT_WIRE_VERSION 1u
#define WT_WIRE_HEADER_BYTES 176u
#define WT_WIRE_FINGERPRINT_BYTES 32u

enum wt_wire_status {
    WT_WIRE_OK = 0,
    WT_WIRE_EINVAL = 1,
    WT_WIRE_EBOUNDS = 2,
    WT_WIRE_EFORMAT = 3,
    WT_WIRE_ECHECKSUM = 4,
    WT_WIRE_EMISMATCH = 5
};

enum wt_wire_kind {
    WT_WIRE_REQUEST = 1,
    WT_WIRE_RESULT = 2,
    WT_WIRE_ERROR = 3,
    WT_WIRE_ACK = 4
};

enum wt_float_format {
    WT_FLOAT_IEEE754_BINARY32 = 1,
    WT_FLOAT_IEEE754_BINARY64 = 2
};

struct wt_work_id {
    uint8_t run_uuid[16];
    uint64_t epoch;
    uint32_t domain;
    uint32_t phase;
    uint32_t call_seq;
    uint32_t chunk;
    uint32_t owner_rank;
    uint32_t owner_generation;
    int32_t gi0;
    int32_t gi1;
    int32_t gj0;
    int32_t gj1;
};

struct wt_payload_abi {
    uint8_t int_bytes;
    uint8_t real_bytes;
    uint8_t logical_bytes;
    uint8_t float_format;
    uint64_t logical_true_bits;
    uint32_t nargs;
};

struct wt_wire_limits {
    uint64_t max_frame;
    uint64_t max_payload;
    uint32_t max_nargs;
};

struct wt_wire_frame {
    uint8_t kind;
    struct wt_work_id id;
    uint8_t schema_fingerprint[WT_WIRE_FINGERPRINT_BYTES];
    uint8_t request_fingerprint[WT_WIRE_FINGERPRINT_BYTES];
    struct wt_payload_abi abi;
    const uint8_t *payload; /* borrowed on decode; lifetime is input buffer */
    uint64_t payload_len;
};

/* Compares every work-ID field, including all 16 UUID bytes. */
int wt_wire_id_equal(const struct wt_work_id *left, const struct wt_work_id *right);

/*
 * Writes one complete frame into out.  payload may be empty; a non-empty
 * payload must point to canonical bytes.  out_len is changed only on success.
 */
int wt_wire_encode(const struct wt_wire_frame *frame,
                   const struct wt_wire_limits *limits,
                   uint8_t *out, size_t out_capacity, size_t *out_len);

/*
 * Validates one exact frame and returns a borrowed view in out.  expected_schema
 * and expected_abi are mandatory exact-match contracts.  out is unchanged when
 * this function returns an error.
 */
int wt_wire_decode(const uint8_t *input, size_t input_len,
                   const struct wt_wire_limits *limits,
                   const uint8_t expected_schema[WT_WIRE_FINGERPRINT_BYTES],
                   const struct wt_payload_abi *expected_abi,
                   struct wt_wire_frame *out);

#endif
