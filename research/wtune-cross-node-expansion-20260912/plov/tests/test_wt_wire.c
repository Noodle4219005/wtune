#include "../wt_wire.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static struct wt_wire_limits limits(void) {
    struct wt_wire_limits value = { 4096u, 1024u, 16u };
    return value;
}

static struct wt_payload_abi abi(void) {
    struct wt_payload_abi value = { 4u, 4u, 4u, WT_FLOAT_IEEE754_BINARY32,
                                    UINT64_C(1), 3u };
    return value;
}

static struct wt_wire_frame frame(const uint8_t *payload, uint64_t payload_len) {
    struct wt_wire_frame value;
    uint32_t n;
    memset(&value, 0, sizeof(value));
    value.kind = WT_WIRE_REQUEST;
    for (n = 0; n < 16u; ++n) {
        value.id.run_uuid[n] = (uint8_t)n;
    }
    value.id.epoch = UINT64_C(0x0102030405060708);
    value.id.domain = 7u;
    value.id.phase = 8u;
    value.id.call_seq = 9u;
    value.id.chunk = 10u;
    value.id.owner_rank = 11u;
    value.id.owner_generation = 12u;
    value.id.gi0 = INT32_MIN;
    value.id.gi1 = INT32_MAX;
    value.id.gj0 = -9;
    value.id.gj1 = 17;
    memset(value.schema_fingerprint, 0xa5, sizeof(value.schema_fingerprint));
    memset(value.request_fingerprint, 0x5a, sizeof(value.request_fingerprint));
    value.abi = abi();
    value.payload = payload;
    value.payload_len = payload_len;
    return value;
}

static uint32_t test_crc32(const uint8_t *bytes, size_t length) {
    uint32_t crc = UINT32_C(0xffffffff);
    size_t index;
    for (index = 0; index < length; ++index) {
        uint8_t byte = (index >= 172u && index < 176u) ? 0u : bytes[index];
        unsigned int bit;
        crc ^= byte;
        for (bit = 0; bit < 8u; ++bit) {
            crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & (uint32_t)-(int32_t)(crc & 1u));
        }
    }
    return crc ^ UINT32_C(0xffffffff);
}

static void refresh_checksum(uint8_t *bytes, size_t length) {
    uint32_t crc = test_crc32(bytes, length);
    bytes[172] = (uint8_t)crc;
    bytes[173] = (uint8_t)(crc >> 8);
    bytes[174] = (uint8_t)(crc >> 16);
    bytes[175] = (uint8_t)(crc >> 24);
}

static void test_roundtrip_and_golden_prefix(void) {
    const uint8_t payload[] = { 0xde, 0xad, 0xbe };
    const uint8_t golden[] = {
        0x57, 0x54, 0x57, 0x52, 0x01, 0x00, 0x01, 0x00,
        0xb0, 0x00, 0x00, 0x00, 0xb3, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
        0x0c, 0x0d, 0x0e, 0x0f, 0x08, 0x07, 0x06, 0x05,
        0x04, 0x03, 0x02, 0x01
    };
    uint8_t bytes[512];
    size_t encoded = 0;
    struct wt_wire_frame input = frame(payload, sizeof(payload));
    struct wt_wire_frame decoded;
    struct wt_wire_limits lim = limits();
    struct wt_payload_abi expected_abi = abi();

    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    assert(encoded == WT_WIRE_HEADER_BYTES + sizeof(payload));
    assert(memcmp(bytes, golden, sizeof(golden)) == 0);
    assert(wt_wire_decode(bytes, encoded, &lim, input.schema_fingerprint,
                          &expected_abi, &decoded) == WT_WIRE_OK);
    assert(decoded.payload == bytes + WT_WIRE_HEADER_BYTES);
    assert(decoded.payload_len == sizeof(payload));
    assert(memcmp(decoded.payload, payload, sizeof(payload)) == 0);
    assert(memcmp(input.id.run_uuid, decoded.id.run_uuid, sizeof(input.id.run_uuid)) == 0);
    assert(input.id.epoch == decoded.id.epoch);
    assert(input.id.domain == decoded.id.domain);
    assert(input.id.phase == decoded.id.phase);
    assert(input.id.call_seq == decoded.id.call_seq);
    assert(input.id.chunk == decoded.id.chunk);
    assert(input.id.owner_rank == decoded.id.owner_rank);
    assert(input.id.owner_generation == decoded.id.owner_generation);
    assert(input.id.gi0 == decoded.id.gi0);
    assert(input.id.gi1 == decoded.id.gi1);
    assert(input.id.gj0 == decoded.id.gj0);
    assert(input.id.gj1 == decoded.id.gj1);
    assert(wt_wire_id_equal(&input.id, &decoded.id));
    assert(!wt_wire_id_equal(&input.id, &(struct wt_work_id){0}));
}

static void assert_rejected_without_output_update(uint8_t *bytes, size_t length,
                                                   const struct wt_wire_frame *input) {
    struct wt_wire_frame output;
    struct wt_wire_frame sentinel;
    struct wt_wire_limits lim = limits();
    struct wt_payload_abi expected_abi = abi();
    memset(&sentinel, 0x3c, sizeof(sentinel));
    output = sentinel;
    assert(wt_wire_decode(bytes, length, &lim, input->schema_fingerprint,
                          &expected_abi, &output) != WT_WIRE_OK);
    assert(memcmp(&output, &sentinel, sizeof(output)) == 0);
}

static void test_rejects_malformed_frames(void) {
    const uint8_t payload[] = { 1u, 2u, 3u, 4u };
    uint8_t bytes[512];
    size_t encoded = 0;
    struct wt_wire_frame input = frame(payload, sizeof(payload));
    struct wt_wire_limits lim = limits();
    struct wt_payload_abi expected_abi = abi();

    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    assert_rejected_without_output_update(bytes, encoded - 1u, &input);
    bytes[6] = 99u;
    assert_rejected_without_output_update(bytes, encoded, &input);
    bytes[6] = WT_WIRE_REQUEST;
    bytes[7] = 1u;
    assert_rejected_without_output_update(bytes, encoded, &input);
    bytes[7] = 0u;
    bytes[172] ^= 0x01u;
    assert_rejected_without_output_update(bytes, encoded, &input);
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    bytes[36] ^= 0x01u;
    assert_rejected_without_output_update(bytes, encoded, &input);
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    bytes[WT_WIRE_HEADER_BYTES] ^= 0x01u;
    assert_rejected_without_output_update(bytes, encoded, &input);
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    bytes[124] ^= 0x01u;
    assert(wt_wire_decode(bytes, encoded, &lim, input.schema_fingerprint,
                          &expected_abi, &(struct wt_wire_frame){0}) == WT_WIRE_ECHECKSUM);
}

static void test_limits_schema_and_abi(void) {
    const uint8_t payload[] = { 1u, 2u, 3u, 4u };
    uint8_t bytes[512];
    uint8_t wrong_schema[WT_WIRE_FINGERPRINT_BYTES] = { 0 };
    size_t encoded = 0;
    struct wt_wire_frame input = frame(payload, sizeof(payload));
    struct wt_wire_limits lim = limits();
    struct wt_wire_limits tiny_frame = { WT_WIRE_HEADER_BYTES + 3u, 1024u, 16u };
    struct wt_wire_limits tiny_payload = { 4096u, 3u, 16u };
    struct wt_wire_limits tiny_nargs = { 4096u, 1024u, 2u };
    struct wt_payload_abi wrong_abi = abi();
    struct wt_wire_frame decoded;

    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    assert(wt_wire_decode(bytes, encoded, &lim, wrong_schema, &input.abi, &decoded) == WT_WIRE_EMISMATCH);
    wrong_abi.real_bytes = 8u;
    wrong_abi.float_format = WT_FLOAT_IEEE754_BINARY64;
    assert(wt_wire_decode(bytes, encoded, &lim, input.schema_fingerprint,
                          &wrong_abi, &decoded) == WT_WIRE_EMISMATCH);
    assert(wt_wire_decode(bytes, encoded, &tiny_frame, input.schema_fingerprint,
                          &input.abi, &decoded) == WT_WIRE_EBOUNDS);
    assert(wt_wire_decode(bytes, encoded, &tiny_payload, input.schema_fingerprint,
                          &input.abi, &decoded) == WT_WIRE_EBOUNDS);
    assert(wt_wire_decode(bytes, encoded, &tiny_nargs, input.schema_fingerprint,
                          &input.abi, &decoded) == WT_WIRE_EBOUNDS);
    memset(bytes + 12, 0xff, 8u);
    assert(wt_wire_decode(bytes, encoded, &lim, input.schema_fingerprint,
                          &input.abi, &decoded) == WT_WIRE_EBOUNDS);
    input.id.gi0 = 10;
    input.id.gi1 = 9;
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_EFORMAT);
    input = frame(payload, sizeof(payload));
    input.abi.real_bytes = 8u;
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_EFORMAT);
    input = frame(payload, sizeof(payload));
    input.abi.logical_true_bits = 0u;
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_EFORMAT);
    input = frame(payload, sizeof(payload));
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    bytes[157] = 8u;
    refresh_checksum(bytes, encoded);
    assert(wt_wire_decode(bytes, encoded, &lim, input.schema_fingerprint,
                          &input.abi, &decoded) == WT_WIRE_EFORMAT);
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    memset(bytes + 160, 0, 8u);
    refresh_checksum(bytes, encoded);
    assert(wt_wire_decode(bytes, encoded, &lim, input.schema_fingerprint,
                          &input.abi, &decoded) == WT_WIRE_EFORMAT);
    assert(wt_wire_encode(&input, &lim, bytes, sizeof(bytes), &encoded) == WT_WIRE_OK);
    bytes[76] = 1u;
    bytes[77] = 0u;
    bytes[78] = 0u;
    bytes[79] = 0u;
    memset(bytes + 80, 0, 4u);
    refresh_checksum(bytes, encoded);
    assert(wt_wire_decode(bytes, encoded, &lim, input.schema_fingerprint,
                          &input.abi, &decoded) == WT_WIRE_EFORMAT);
}

int main(void) {
    test_roundtrip_and_golden_prefix();
    test_rejects_malformed_frames();
    test_limits_schema_and_abi();
    puts("PASS test_wt_wire");
    return 0;
}
