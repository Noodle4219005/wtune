#include "wt_wire.h"

#include <limits.h>
#include <string.h>

enum {
    WT_OFF_MAGIC = 0,
    WT_OFF_VERSION = 4,
    WT_OFF_KIND = 6,
    WT_OFF_RESERVED0 = 7,
    WT_OFF_HEADER_LEN = 8,
    WT_OFF_RESERVED1 = 10,
    WT_OFF_TOTAL_LEN = 12,
    WT_OFF_PAYLOAD_LEN = 20,
    WT_OFF_UUID = 28,
    WT_OFF_EPOCH = 44,
    WT_OFF_DOMAIN = 52,
    WT_OFF_PHASE = 56,
    WT_OFF_CALL_SEQ = 60,
    WT_OFF_CHUNK = 64,
    WT_OFF_OWNER_RANK = 68,
    WT_OFF_OWNER_GENERATION = 72,
    WT_OFF_GI0 = 76,
    WT_OFF_GI1 = 80,
    WT_OFF_GJ0 = 84,
    WT_OFF_GJ1 = 88,
    WT_OFF_SCHEMA = 92,
    WT_OFF_REQUEST = 124,
    WT_OFF_INT_BYTES = 156,
    WT_OFF_REAL_BYTES = 157,
    WT_OFF_LOGICAL_BYTES = 158,
    WT_OFF_FLOAT_FORMAT = 159,
    WT_OFF_LOGICAL_TRUE = 160,
    WT_OFF_NARGS = 168,
    WT_OFF_CHECKSUM = 172
};

static void put_u16(uint8_t *out, uint16_t value) {
    out[0] = (uint8_t)value;
    out[1] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *out, uint32_t value) {
    unsigned int index;
    for (index = 0; index < 4u; ++index) {
        out[index] = (uint8_t)(value >> (8u * index));
    }
}

static void put_u64(uint8_t *out, uint64_t value) {
    unsigned int index;
    for (index = 0; index < 8u; ++index) {
        out[index] = (uint8_t)(value >> (8u * index));
    }
}

static uint16_t get_u16(const uint8_t *input) {
    return (uint16_t)((uint16_t)input[0] | ((uint16_t)input[1] << 8));
}

static uint32_t get_u32(const uint8_t *input) {
    uint32_t value = 0;
    unsigned int index;
    for (index = 0; index < 4u; ++index) {
        value |= (uint32_t)input[index] << (8u * index);
    }
    return value;
}

static uint64_t get_u64(const uint8_t *input) {
    uint64_t value = 0;
    unsigned int index;
    for (index = 0; index < 8u; ++index) {
        value |= (uint64_t)input[index] << (8u * index);
    }
    return value;
}

static uint32_t i32_bits(int32_t value) {
    if (value >= 0) {
        return (uint32_t)value;
    }
    return UINT32_MAX - (uint32_t)(-(value + 1));
}

static int32_t get_i32(const uint8_t *input) {
    uint32_t value = get_u32(input);
    if (value <= (uint32_t)INT32_MAX) {
        return (int32_t)value;
    }
    return -(int32_t)(UINT32_MAX - value) - 1;
}

static uint32_t crc32_update(uint32_t crc, const uint8_t *bytes, size_t length) {
    size_t index;
    for (index = 0; index < length; ++index) {
        unsigned int bit;
        crc ^= bytes[index];
        for (bit = 0; bit < 8u; ++bit) {
            crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & (uint32_t)-(int32_t)(crc & 1u));
        }
    }
    return crc;
}

static uint32_t frame_crc32(const uint8_t header[WT_WIRE_HEADER_BYTES],
                            const uint8_t *payload, size_t payload_len) {
    static const uint8_t zero_checksum[4] = { 0, 0, 0, 0 };
    uint32_t crc = UINT32_C(0xffffffff);
    crc = crc32_update(crc, header, WT_OFF_CHECKSUM);
    crc = crc32_update(crc, zero_checksum, sizeof(zero_checksum));
    if (payload_len != 0u) {
        crc = crc32_update(crc, payload, payload_len);
    }
    return crc ^ UINT32_C(0xffffffff);
}

static int valid_kind(uint8_t kind) {
    return kind >= WT_WIRE_REQUEST && kind <= WT_WIRE_ACK;
}

static int valid_abi(const struct wt_payload_abi *abi) {
    if (abi == NULL || (abi->int_bytes != 1u && abi->int_bytes != 2u &&
                        abi->int_bytes != 4u && abi->int_bytes != 8u) ||
        (abi->logical_bytes != 1u && abi->logical_bytes != 2u &&
         abi->logical_bytes != 4u && abi->logical_bytes != 8u) ||
        abi->logical_true_bits == 0u) {
        return 0;
    }
    if (abi->logical_bytes == 1u && abi->logical_true_bits > UINT8_MAX) {
        return 0;
    }
    if (abi->logical_bytes == 2u && abi->logical_true_bits > UINT16_MAX) {
        return 0;
    }
    if (abi->logical_bytes == 4u && abi->logical_true_bits > UINT32_MAX) {
        return 0;
    }
    return (abi->float_format == WT_FLOAT_IEEE754_BINARY32 && abi->real_bytes == 4u) ||
           (abi->float_format == WT_FLOAT_IEEE754_BINARY64 && abi->real_bytes == 8u);
}

static int abi_equal(const struct wt_payload_abi *left, const struct wt_payload_abi *right) {
    return left->int_bytes == right->int_bytes && left->real_bytes == right->real_bytes &&
           left->logical_bytes == right->logical_bytes && left->float_format == right->float_format &&
           left->logical_true_bits == right->logical_true_bits && left->nargs == right->nargs;
}

static int valid_limits(const struct wt_wire_limits *limits) {
    return limits != NULL && limits->max_frame >= WT_WIRE_HEADER_BYTES;
}

static int valid_id(const struct wt_work_id *id) {
    return id != NULL && id->gi0 <= id->gi1 && id->gj0 <= id->gj1;
}

static int valid_frame(const struct wt_wire_frame *frame,
                       const struct wt_wire_limits *limits, uint64_t *total_len) {
    if (frame == NULL || !valid_limits(limits) || !valid_kind(frame->kind) ||
        !valid_id(&frame->id) || !valid_abi(&frame->abi) ||
        frame->abi.nargs > limits->max_nargs || frame->payload_len > limits->max_payload ||
        (frame->payload_len != 0u && frame->payload == NULL) ||
        frame->payload_len > UINT64_MAX - WT_WIRE_HEADER_BYTES) {
        return 0;
    }
    *total_len = WT_WIRE_HEADER_BYTES + frame->payload_len;
    return *total_len <= limits->max_frame;
}

int wt_wire_id_equal(const struct wt_work_id *left, const struct wt_work_id *right) {
    return left != NULL && right != NULL &&
           memcmp(left->run_uuid, right->run_uuid, sizeof(left->run_uuid)) == 0 &&
           left->epoch == right->epoch && left->domain == right->domain &&
           left->phase == right->phase && left->call_seq == right->call_seq &&
           left->chunk == right->chunk && left->owner_rank == right->owner_rank &&
           left->owner_generation == right->owner_generation && left->gi0 == right->gi0 &&
           left->gi1 == right->gi1 && left->gj0 == right->gj0 && left->gj1 == right->gj1;
}

int wt_wire_encode(const struct wt_wire_frame *frame,
                   const struct wt_wire_limits *limits,
                   uint8_t *out, size_t out_capacity, size_t *out_len) {
    uint8_t header[WT_WIRE_HEADER_BYTES] = { 0 };
    uint64_t total_len;
    uint32_t checksum;
    if (out == NULL || out_len == NULL || frame == NULL || limits == NULL) {
        return WT_WIRE_EINVAL;
    }
    if (!valid_kind(frame->kind) || !valid_id(&frame->id) || !valid_abi(&frame->abi)) {
        return WT_WIRE_EFORMAT;
    }
    if (!valid_frame(frame, limits, &total_len) || total_len > (uint64_t)SIZE_MAX ||
        out_capacity < (size_t)total_len) {
        return WT_WIRE_EBOUNDS;
    }
    header[WT_OFF_MAGIC] = 'W';
    header[WT_OFF_MAGIC + 1] = 'T';
    header[WT_OFF_MAGIC + 2] = 'W';
    header[WT_OFF_MAGIC + 3] = 'R';
    put_u16(header + WT_OFF_VERSION, WT_WIRE_VERSION);
    header[WT_OFF_KIND] = frame->kind;
    put_u16(header + WT_OFF_HEADER_LEN, WT_WIRE_HEADER_BYTES);
    put_u64(header + WT_OFF_TOTAL_LEN, total_len);
    put_u64(header + WT_OFF_PAYLOAD_LEN, frame->payload_len);
    memcpy(header + WT_OFF_UUID, frame->id.run_uuid, sizeof(frame->id.run_uuid));
    put_u64(header + WT_OFF_EPOCH, frame->id.epoch);
    put_u32(header + WT_OFF_DOMAIN, frame->id.domain);
    put_u32(header + WT_OFF_PHASE, frame->id.phase);
    put_u32(header + WT_OFF_CALL_SEQ, frame->id.call_seq);
    put_u32(header + WT_OFF_CHUNK, frame->id.chunk);
    put_u32(header + WT_OFF_OWNER_RANK, frame->id.owner_rank);
    put_u32(header + WT_OFF_OWNER_GENERATION, frame->id.owner_generation);
    put_u32(header + WT_OFF_GI0, i32_bits(frame->id.gi0));
    put_u32(header + WT_OFF_GI1, i32_bits(frame->id.gi1));
    put_u32(header + WT_OFF_GJ0, i32_bits(frame->id.gj0));
    put_u32(header + WT_OFF_GJ1, i32_bits(frame->id.gj1));
    memcpy(header + WT_OFF_SCHEMA, frame->schema_fingerprint, WT_WIRE_FINGERPRINT_BYTES);
    memcpy(header + WT_OFF_REQUEST, frame->request_fingerprint, WT_WIRE_FINGERPRINT_BYTES);
    header[WT_OFF_INT_BYTES] = frame->abi.int_bytes;
    header[WT_OFF_REAL_BYTES] = frame->abi.real_bytes;
    header[WT_OFF_LOGICAL_BYTES] = frame->abi.logical_bytes;
    header[WT_OFF_FLOAT_FORMAT] = frame->abi.float_format;
    put_u64(header + WT_OFF_LOGICAL_TRUE, frame->abi.logical_true_bits);
    put_u32(header + WT_OFF_NARGS, frame->abi.nargs);
    checksum = frame_crc32(header, frame->payload, (size_t)frame->payload_len);
    put_u32(header + WT_OFF_CHECKSUM, checksum);
    if (frame->payload_len != 0u) {
        memmove(out + WT_WIRE_HEADER_BYTES, frame->payload, (size_t)frame->payload_len);
    }
    memcpy(out, header, sizeof(header));
    *out_len = (size_t)total_len;
    return WT_WIRE_OK;
}

int wt_wire_decode(const uint8_t *input, size_t input_len,
                   const struct wt_wire_limits *limits,
                   const uint8_t expected_schema[WT_WIRE_FINGERPRINT_BYTES],
                   const struct wt_payload_abi *expected_abi,
                   struct wt_wire_frame *out) {
    struct wt_wire_frame candidate;
    uint64_t total_len;
    uint64_t payload_len;
    if (input == NULL || out == NULL || expected_schema == NULL || !valid_limits(limits) ||
        !valid_abi(expected_abi) || input_len < WT_WIRE_HEADER_BYTES ||
        (uint64_t)input_len > limits->max_frame) {
        return WT_WIRE_EBOUNDS;
    }
    if (input[0] != 'W' || input[1] != 'T' || input[2] != 'W' || input[3] != 'R' ||
        get_u16(input + WT_OFF_VERSION) != WT_WIRE_VERSION || !valid_kind(input[WT_OFF_KIND]) ||
        input[WT_OFF_RESERVED0] != 0u || get_u16(input + WT_OFF_RESERVED1) != 0u ||
        get_u16(input + WT_OFF_HEADER_LEN) != WT_WIRE_HEADER_BYTES) {
        return WT_WIRE_EFORMAT;
    }
    total_len = get_u64(input + WT_OFF_TOTAL_LEN);
    payload_len = get_u64(input + WT_OFF_PAYLOAD_LEN);
    if (total_len < WT_WIRE_HEADER_BYTES || total_len != (uint64_t)input_len ||
        payload_len != total_len - WT_WIRE_HEADER_BYTES || payload_len > limits->max_payload ||
        total_len > limits->max_frame) {
        return WT_WIRE_EBOUNDS;
    }
    if (get_u32(input + WT_OFF_CHECKSUM) !=
        frame_crc32(input, input + WT_WIRE_HEADER_BYTES, (size_t)payload_len)) {
        return WT_WIRE_ECHECKSUM;
    }
    memset(&candidate, 0, sizeof(candidate));
    candidate.kind = input[WT_OFF_KIND];
    memcpy(candidate.id.run_uuid, input + WT_OFF_UUID, sizeof(candidate.id.run_uuid));
    candidate.id.epoch = get_u64(input + WT_OFF_EPOCH);
    candidate.id.domain = get_u32(input + WT_OFF_DOMAIN);
    candidate.id.phase = get_u32(input + WT_OFF_PHASE);
    candidate.id.call_seq = get_u32(input + WT_OFF_CALL_SEQ);
    candidate.id.chunk = get_u32(input + WT_OFF_CHUNK);
    candidate.id.owner_rank = get_u32(input + WT_OFF_OWNER_RANK);
    candidate.id.owner_generation = get_u32(input + WT_OFF_OWNER_GENERATION);
    candidate.id.gi0 = get_i32(input + WT_OFF_GI0);
    candidate.id.gi1 = get_i32(input + WT_OFF_GI1);
    candidate.id.gj0 = get_i32(input + WT_OFF_GJ0);
    candidate.id.gj1 = get_i32(input + WT_OFF_GJ1);
    memcpy(candidate.schema_fingerprint, input + WT_OFF_SCHEMA, WT_WIRE_FINGERPRINT_BYTES);
    memcpy(candidate.request_fingerprint, input + WT_OFF_REQUEST, WT_WIRE_FINGERPRINT_BYTES);
    candidate.abi.int_bytes = input[WT_OFF_INT_BYTES];
    candidate.abi.real_bytes = input[WT_OFF_REAL_BYTES];
    candidate.abi.logical_bytes = input[WT_OFF_LOGICAL_BYTES];
    candidate.abi.float_format = input[WT_OFF_FLOAT_FORMAT];
    candidate.abi.logical_true_bits = get_u64(input + WT_OFF_LOGICAL_TRUE);
    candidate.abi.nargs = get_u32(input + WT_OFF_NARGS);
    if (!valid_id(&candidate.id) || !valid_abi(&candidate.abi)) {
        return WT_WIRE_EFORMAT;
    }
    if (candidate.abi.nargs > limits->max_nargs) {
        return WT_WIRE_EBOUNDS;
    }
    if (memcmp(candidate.schema_fingerprint, expected_schema, WT_WIRE_FINGERPRINT_BYTES) != 0 ||
        !abi_equal(&candidate.abi, expected_abi)) {
        return WT_WIRE_EMISMATCH;
    }
    candidate.payload = input + WT_WIRE_HEADER_BYTES;
    candidate.payload_len = payload_len;
    *out = candidate;
    return WT_WIRE_OK;
}
