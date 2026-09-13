#include "wt_ledger.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { BUFFER_BYTES = 512 };

#ifdef WT_LEDGER_TESTING
void wt_ledger_private_test_set_next_identity(uint64_t value);
void wt_ledger_private_test_set_generation(struct wt_ledger *ledger,
                                           uint64_t value);
#endif

struct fixture {
    struct wt_ledger_contract contract;
    struct wt_work_id id;
    uint8_t request[BUFFER_BYTES];
    size_t request_len;
    uint8_t result[BUFFER_BYTES];
    size_t result_len;
};

struct run_context {
    const struct fixture *fixture;
    int calls;
    int fail;
    struct wt_ledger *ledger;
    struct wt_ledger_ticket ticket;
    const struct fixture *reentrant_accept_fixture;
    int reentrant_run_status;
    int reentrant_fail_status;
    int reentrant_accept_status;
    int reentrant_destroy_status;
};

struct commit_context {
    int calls;
    int fail;
    uint8_t bytes[64];
    size_t length;
    struct wt_ledger *ledger;
    struct wt_ledger_ticket ticket;
    const uint8_t *encoded_result;
    size_t encoded_result_len;
    int reentrant_apply_status;
    int reentrant_destroy_status;
};

static void fill_bytes(uint8_t *bytes, size_t length, uint8_t seed) {
    size_t i;
    for (i = 0; i < length; ++i) {
        bytes[i] = (uint8_t)(seed + (uint8_t)i);
    }
}

static int ticket_equal(const struct wt_ledger_ticket *a,
                        const struct wt_ledger_ticket *b) {
    return a->ledger_identity == b->ledger_identity &&
           a->generation == b->generation && a->nonce == b->nonce &&
           a->slot == b->slot;
}

static void init_fixture(struct fixture *f, uint32_t call_seq,
                         const uint8_t *request_payload, size_t request_payload_len,
                         const uint8_t *result_payload, size_t result_payload_len) {
    struct wt_wire_frame frame;
    memset(f, 0, sizeof(*f));
    fill_bytes(f->contract.run_uuid, sizeof(f->contract.run_uuid), 0x10u);
    f->contract.owner_rank = 7u;
    f->contract.owner_generation = 11u;
    fill_bytes(f->contract.schema_fingerprint,
               sizeof(f->contract.schema_fingerprint), 0x40u);
    f->contract.abi.int_bytes = 4u;
    f->contract.abi.real_bytes = 8u;
    f->contract.abi.logical_bytes = 4u;
    f->contract.abi.float_format = WT_FLOAT_IEEE754_BINARY64;
    f->contract.abi.logical_true_bits = 1u;
    f->contract.abi.nargs = 3u;
    f->contract.limits.max_frame = BUFFER_BYTES;
    f->contract.limits.max_payload = BUFFER_BYTES - WT_WIRE_HEADER_BYTES;
    f->contract.limits.max_nargs = 8u;
    memcpy(f->id.run_uuid, f->contract.run_uuid, sizeof(f->id.run_uuid));
    f->id.epoch = 23u;
    f->id.domain = 2u;
    f->id.phase = 5u;
    f->id.call_seq = call_seq;
    f->id.chunk = 1u;
    f->id.owner_rank = f->contract.owner_rank;
    f->id.owner_generation = f->contract.owner_generation;
    f->id.gi0 = 1;
    f->id.gi1 = 4;
    f->id.gj0 = 2;
    f->id.gj1 = 6;
    memset(&frame, 0, sizeof(frame));
    frame.kind = WT_WIRE_REQUEST;
    frame.id = f->id;
    memcpy(frame.schema_fingerprint, f->contract.schema_fingerprint,
           sizeof(frame.schema_fingerprint));
    fill_bytes(frame.request_fingerprint, sizeof(frame.request_fingerprint),
               (uint8_t)(0x60u + call_seq));
    frame.abi = f->contract.abi;
    frame.payload = request_payload;
    frame.payload_len = request_payload_len;
    assert(wt_wire_encode(&frame, &f->contract.limits, f->request,
                          sizeof(f->request), &f->request_len) == WT_WIRE_OK);
    frame.kind = WT_WIRE_RESULT;
    frame.payload = result_payload;
    frame.payload_len = result_payload_len;
    assert(wt_wire_encode(&frame, &f->contract.limits, f->result,
                          sizeof(f->result), &f->result_len) == WT_WIRE_OK);
}

static int copy_result(void *opaque, const uint8_t *request,
                       size_t request_len, uint8_t *result,
                       size_t result_capacity, size_t *result_len) {
    struct run_context *ctx = opaque;
    ++ctx->calls;
    assert(request_len == ctx->fixture->request_len);
    assert(memcmp(request, ctx->fixture->request, request_len) == 0);
    if (ctx->fail) {
        return -9;
    }
    assert(result_capacity >= ctx->fixture->result_len);
    memcpy(result, ctx->fixture->result, ctx->fixture->result_len);
    *result_len = ctx->fixture->result_len;
    return 0;
}

static int reentrant_result(void *opaque, const uint8_t *request,
                            size_t request_len, uint8_t *result,
                            size_t result_capacity, size_t *result_len) {
    struct run_context *ctx = opaque;
    (void)request;
    (void)request_len;
    ++ctx->calls;
    ctx->reentrant_run_status = wt_ledger_worker_run(
        ctx->ledger, &ctx->ticket, copy_result, ctx);
    ctx->reentrant_fail_status = wt_ledger_fail(ctx->ledger, &ctx->ticket);
    ctx->reentrant_accept_status = wt_ledger_worker_accept(
        ctx->ledger, ctx->reentrant_accept_fixture->request,
        ctx->reentrant_accept_fixture->request_len,
        ctx->reentrant_accept_fixture->result_len, &ctx->ticket);
    ctx->reentrant_destroy_status = wt_ledger_destroy(&ctx->ledger);
    assert(result_capacity >= ctx->fixture->result_len);
    memcpy(result, ctx->fixture->result, ctx->fixture->result_len);
    *result_len = ctx->fixture->result_len;
    return 0;
}

static int commit_result(void *opaque, const struct wt_wire_frame *result) {
    struct commit_context *ctx = opaque;
    ++ctx->calls;
    assert(result->kind == WT_WIRE_RESULT);
    assert(result->payload_len <= sizeof(ctx->bytes));
    memcpy(ctx->bytes, result->payload, (size_t)result->payload_len);
    ctx->length = (size_t)result->payload_len;
    return ctx->fail ? -7 : 0;
}

static int reentrant_commit(void *opaque, const struct wt_wire_frame *result) {
    struct commit_context *ctx = opaque;
    ++ctx->calls;
    ctx->reentrant_apply_status = wt_ledger_owner_apply_result(
        ctx->ledger, &ctx->ticket, ctx->encoded_result,
        ctx->encoded_result_len, commit_result, ctx);
    ctx->reentrant_destroy_status = wt_ledger_destroy(&ctx->ledger);
    assert(result->payload_len <= sizeof(ctx->bytes));
    memcpy(ctx->bytes, result->payload, (size_t)result->payload_len);
    ctx->length = (size_t)result->payload_len;
    return 0;
}

static struct wt_ledger *make_ledger(enum wt_ledger_role role,
                                     const struct fixture *f,
                                     size_t max_entries,
                                     size_t max_live_bytes) {
    struct wt_ledger *ledger = NULL;
    assert(wt_ledger_create(role, &f->contract, max_entries,
                            max_live_bytes, &ledger) == WT_LEDGER_OK);
    assert(ledger != NULL);
    return ledger;
}

static void test_worker_duplicate_and_cached_result(void) {
    static const uint8_t request_payload[] = {1u, 2u, 3u};
    static const uint8_t result_payload[] = {9u, 8u, 7u, 6u};
    struct fixture f;
    struct wt_ledger *ledger;
    struct wt_ledger_ticket first;
    struct wt_ledger_ticket duplicate;
    struct run_context ctx;
    const uint8_t *cached = NULL;
    size_t cached_len = 0u;
    init_fixture(&f, 1u, request_payload, sizeof(request_payload),
                 result_payload, sizeof(result_payload));
    ledger = make_ledger(WT_LEDGER_WORKER, &f, 4u, 2048u);
    assert(wt_ledger_worker_accept(ledger, f.request, f.request_len,
                                   f.result_len, &first) == WT_LEDGER_OK);
    assert(wt_ledger_worker_accept(ledger, f.request, f.request_len,
                                   f.result_len, &duplicate) ==
           WT_LEDGER_DUPLICATE);
    assert(ticket_equal(&first, &duplicate));
    memset(&ctx, 0, sizeof(ctx));
    ctx.fixture = &f;
    assert(wt_ledger_worker_run(ledger, &first, copy_result, &ctx) ==
           WT_LEDGER_OK);
    assert(ctx.calls == 1);
    assert(wt_ledger_worker_run(ledger, &duplicate, copy_result, &ctx) ==
           WT_LEDGER_DUPLICATE);
    assert(ctx.calls == 1);
    assert(wt_ledger_result_bytes(ledger, &first, &cached, &cached_len) ==
           WT_LEDGER_OK);
    assert(cached_len == f.result_len);
    assert(memcmp(cached, f.result, cached_len) == 0);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
    assert(ledger == NULL);
}

static void test_owner_commit_once_and_duplicate_result(void) {
    static const uint8_t request_payload[] = {4u, 5u};
    static const uint8_t result_payload[] = {0xa1u, 0xb2u, 0xc3u};
    struct fixture f;
    struct wt_ledger *ledger;
    struct wt_ledger_ticket ticket;
    struct commit_context ctx;
    enum wt_ledger_state state = WT_LEDGER_ACCEPTED;
    init_fixture(&f, 2u, request_payload, sizeof(request_payload),
                 result_payload, sizeof(result_payload));
    ledger = make_ledger(WT_LEDGER_OWNER, &f, 2u, 1024u);
    assert(wt_ledger_owner_register_request(ledger, f.request, f.request_len,
                                            f.result_len, &ticket) ==
           WT_LEDGER_OK);
    memset(&ctx, 0, sizeof(ctx));
    assert(wt_ledger_owner_apply_result(ledger, &ticket, f.result,
                                        f.result_len, commit_result, &ctx) ==
           WT_LEDGER_OK);
    assert(ctx.calls == 1);
    assert(ctx.length == sizeof(result_payload));
    assert(memcmp(ctx.bytes, result_payload, sizeof(result_payload)) == 0);
    assert(wt_ledger_owner_apply_result(ledger, &ticket, f.result,
                                        f.result_len, commit_result, &ctx) ==
           WT_LEDGER_DUPLICATE);
    assert(ctx.calls == 1);
    assert(wt_ledger_get_state(ledger, &ticket, &state) == WT_LEDGER_OK);
    assert(state == WT_LEDGER_COMMITTED);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
}

static void test_same_id_different_request_is_terminal_conflict(void) {
    static const uint8_t p1[] = {1u};
    static const uint8_t p2[] = {2u};
    static const uint8_t result_payload[] = {3u};
    struct fixture first;
    struct fixture different;
    struct wt_ledger *ledger;
    struct wt_ledger_ticket ticket;
    struct run_context ctx;
    enum wt_ledger_state state = WT_LEDGER_ACCEPTED;
    init_fixture(&first, 3u, p1, sizeof(p1), result_payload,
                 sizeof(result_payload));
    init_fixture(&different, 3u, p2, sizeof(p2), result_payload,
                 sizeof(result_payload));
    /* Re-encode with the first request's digest but different canonical bytes. */
    {
        struct wt_wire_frame frame;
        assert(wt_wire_decode(first.request, first.request_len,
                              &first.contract.limits,
                              first.contract.schema_fingerprint,
                              &first.contract.abi, &frame) == WT_WIRE_OK);
        frame.payload = p2;
        frame.payload_len = sizeof(p2);
        assert(wt_wire_encode(&frame, &different.contract.limits,
                              different.request, sizeof(different.request),
                              &different.request_len) == WT_WIRE_OK);
    }
    ledger = make_ledger(WT_LEDGER_WORKER, &first, 2u, 1024u);
    assert(wt_ledger_worker_accept(ledger, first.request, first.request_len,
                                   first.result_len, &ticket) == WT_LEDGER_OK);
    assert(wt_ledger_worker_accept(ledger, different.request,
                                   different.request_len, different.result_len,
                                   &ticket) == WT_LEDGER_ECONFLICT);
    assert(wt_ledger_get_state(ledger, &ticket, &state) == WT_LEDGER_OK);
    assert(state == WT_LEDGER_FAILED);
    memset(&ctx, 0, sizeof(ctx));
    ctx.fixture = &first;
    assert(wt_ledger_worker_run(ledger, &ticket, copy_result, &ctx) ==
           WT_LEDGER_ESTATE);
    assert(ctx.calls == 0);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
}

static void test_changed_region_metadata_conflicts_on_same_logical_key(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u};
    struct fixture original;
    struct fixture changed;
    struct wt_wire_frame frame;
    struct wt_ledger *ledger;
    struct wt_ledger_ticket ticket;
    struct run_context run;
    enum wt_ledger_state state;
    init_fixture(&original, 12u, p, sizeof(p), r, sizeof(r));
    changed = original;
    assert(wt_wire_decode(original.request, original.request_len,
                          &original.contract.limits,
                          original.contract.schema_fingerprint,
                          &original.contract.abi, &frame) == WT_WIRE_OK);
    frame.id.gi0 = 3;
    frame.id.gi1 = 8;
    assert(wt_wire_encode(&frame, &changed.contract.limits, changed.request,
                          sizeof(changed.request), &changed.request_len) ==
           WT_WIRE_OK);
    ledger = make_ledger(WT_LEDGER_WORKER, &original, 2u, 2048u);
    assert(wt_ledger_worker_accept(ledger, original.request,
                                   original.request_len, original.result_len,
                                   &ticket) == WT_LEDGER_OK);
    assert(wt_ledger_worker_accept(ledger, changed.request,
                                   changed.request_len, changed.result_len,
                                   &ticket) == WT_LEDGER_ECONFLICT);
    assert(wt_ledger_get_state(ledger, &ticket, &state) == WT_LEDGER_OK);
    assert(state == WT_LEDGER_FAILED);
    memset(&run, 0, sizeof(run));
    run.fixture = &original;
    assert(wt_ledger_worker_run(ledger, &ticket, copy_result, &run) ==
           WT_LEDGER_ESTATE);
    assert(run.calls == 0);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
}

static void test_rejects_uninitialized_contract_sentinels(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u};
    struct fixture f;
    struct wt_ledger *ledger = (struct wt_ledger *)(uintptr_t)1u;
    init_fixture(&f, 10u, p, sizeof(p), r, sizeof(r));
    memset(f.contract.run_uuid, 0, sizeof(f.contract.run_uuid));
    assert(wt_ledger_create(WT_LEDGER_WORKER, &f.contract, 1u, 1024u,
                            &ledger) == WT_LEDGER_EINVAL);
    assert(ledger == (struct wt_ledger *)(uintptr_t)1u);
    init_fixture(&f, 10u, p, sizeof(p), r, sizeof(r));
    memset(f.contract.schema_fingerprint, 0,
           sizeof(f.contract.schema_fingerprint));
    assert(wt_ledger_create(WT_LEDGER_WORKER, &f.contract, 1u, 1024u,
                            &ledger) == WT_LEDGER_EINVAL);
    assert(ledger == (struct wt_ledger *)(uintptr_t)1u);
}

static void test_destroyed_ledger_ticket_never_matches_recreated_ledger(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u};
    struct fixture f;
    struct wt_ledger *first;
    struct wt_ledger *replacement;
    struct wt_ledger_ticket old_ticket;
    struct wt_ledger_ticket new_ticket;
    enum wt_ledger_state state;
    init_fixture(&f, 11u, p, sizeof(p), r, sizeof(r));
    first = make_ledger(WT_LEDGER_WORKER, &f, 1u, 1024u);
    assert(wt_ledger_worker_accept(first, f.request, f.request_len,
                                   f.result_len, &old_ticket) == WT_LEDGER_OK);
    assert(wt_ledger_destroy(&first) == WT_LEDGER_OK);
    replacement = make_ledger(WT_LEDGER_WORKER, &f, 1u, 1024u);
    assert(wt_ledger_worker_accept(replacement, f.request, f.request_len,
                                   f.result_len, &new_ticket) == WT_LEDGER_OK);
    assert(old_ticket.ledger_identity != new_ticket.ledger_identity);
    assert(wt_ledger_get_state(replacement, &old_ticket, &state) ==
           WT_LEDGER_EFOREIGN);
    assert(wt_ledger_destroy(&replacement) == WT_LEDGER_OK);
}

static void test_rejects_bad_wire_before_callbacks(void) {
    static const uint8_t request_payload[] = {1u, 2u};
    static const uint8_t result_payload[] = {3u, 4u};
    struct fixture f;
    struct fixture wrong;
    struct wt_ledger *worker;
    struct wt_ledger *owner;
    struct wt_ledger_ticket ticket;
    struct commit_context ctx;
    uint8_t corrupt[BUFFER_BYTES];
    init_fixture(&f, 4u, request_payload, sizeof(request_payload),
                 result_payload, sizeof(result_payload));
    worker = make_ledger(WT_LEDGER_WORKER, &f, 3u, 2048u);
    assert(wt_ledger_worker_accept(worker, f.request, f.request_len - 1u,
                                   f.result_len, &ticket) == WT_LEDGER_EWIRE);
    memcpy(corrupt, f.request, f.request_len);
    corrupt[f.request_len - 1u] ^= 1u;
    assert(wt_ledger_worker_accept(worker, corrupt, f.request_len,
                                   f.result_len, &ticket) == WT_LEDGER_EWIRE);
    wrong = f;
    wrong.contract.schema_fingerprint[0] ^= 1u;
    {
        struct wt_ledger *wrong_schema = make_ledger(
            WT_LEDGER_WORKER, &wrong, 1u, 1024u);
        assert(wt_ledger_worker_accept(wrong_schema, f.request, f.request_len,
                                       f.result_len, &ticket) ==
               WT_LEDGER_EWIRE);
        assert(wt_ledger_destroy(&wrong_schema) == WT_LEDGER_OK);
    }
    assert(wt_ledger_destroy(&worker) == WT_LEDGER_OK);
    owner = make_ledger(WT_LEDGER_OWNER, &f, 2u, 1024u);
    assert(wt_ledger_owner_register_request(owner, f.request, f.request_len,
                                            f.result_len, &ticket) ==
           WT_LEDGER_OK);
    memset(&ctx, 0, sizeof(ctx));
    memcpy(corrupt, f.result, f.result_len);
    corrupt[44] ^= 1u;
    assert(wt_ledger_owner_apply_result(owner, &ticket, corrupt, f.result_len,
                                        commit_result, &ctx) ==
           WT_LEDGER_EWIRE);
    assert(ctx.calls == 0);
    assert(wt_ledger_destroy(&owner) == WT_LEDGER_OK);
}

static void test_budget_entry_limit_and_out_of_order(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u};
    struct fixture a;
    struct fixture b;
    struct wt_ledger *ledger;
    struct wt_ledger_ticket ta;
    struct wt_ledger_ticket tb;
    struct run_context ca;
    struct run_context cb;
    init_fixture(&a, 5u, p, sizeof(p), r, sizeof(r));
    init_fixture(&b, 6u, p, sizeof(p), r, sizeof(r));
    ledger = make_ledger(WT_LEDGER_WORKER, &a, 2u,
                         a.request_len + a.result_len);
    assert(wt_ledger_worker_accept(ledger, a.request, a.request_len,
                                   a.result_len, &ta) == WT_LEDGER_OK);
    assert(wt_ledger_worker_accept(ledger, b.request, b.request_len,
                                   b.result_len, &tb) == WT_LEDGER_EBUDGET);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
    ledger = make_ledger(WT_LEDGER_WORKER, &a, 1u, 4096u);
    assert(wt_ledger_worker_accept(ledger, a.request, a.request_len,
                                   a.result_len, &ta) == WT_LEDGER_OK);
    assert(wt_ledger_worker_accept(ledger, b.request, b.request_len,
                                   b.result_len, &tb) == WT_LEDGER_EFULL);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
    ledger = make_ledger(WT_LEDGER_WORKER, &a, 2u, 4096u);
    assert(wt_ledger_worker_accept(ledger, a.request, a.request_len,
                                   a.result_len, &ta) == WT_LEDGER_OK);
    assert(wt_ledger_worker_accept(ledger, b.request, b.request_len,
                                   b.result_len, &tb) == WT_LEDGER_OK);
    memset(&ca, 0, sizeof(ca));
    memset(&cb, 0, sizeof(cb));
    ca.fixture = &a;
    cb.fixture = &b;
    assert(wt_ledger_worker_run(ledger, &tb, copy_result, &cb) == WT_LEDGER_OK);
    assert(wt_ledger_worker_run(ledger, &ta, copy_result, &ca) == WT_LEDGER_OK);
    assert(ca.calls == 1 && cb.calls == 1);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
}

static void test_result_identity_and_fingerprint_precede_commit(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u, 3u};
    struct fixture f;
    struct wt_wire_frame frame;
    struct wt_ledger *owner;
    struct wt_ledger_ticket ticket;
    struct commit_context commit;
    uint8_t wrong[BUFFER_BYTES];
    size_t wrong_len = 0u;
    init_fixture(&f, 14u, p, sizeof(p), r, sizeof(r));
    owner = make_ledger(WT_LEDGER_OWNER, &f, 1u, 1024u);
    assert(wt_ledger_owner_register_request(owner, f.request, f.request_len,
                                            f.result_len, &ticket) ==
           WT_LEDGER_OK);
    assert(wt_wire_decode(f.result, f.result_len, &f.contract.limits,
                          f.contract.schema_fingerprint, &f.contract.abi,
                          &frame) == WT_WIRE_OK);
    frame.request_fingerprint[0] ^= 1u;
    assert(wt_wire_encode(&frame, &f.contract.limits, wrong, sizeof(wrong),
                          &wrong_len) == WT_WIRE_OK);
    memset(&commit, 0, sizeof(commit));
    assert(wt_ledger_owner_apply_result(owner, &ticket, wrong, wrong_len,
                                        commit_result, &commit) ==
           WT_LEDGER_EWIRE);
    assert(commit.calls == 0);
    assert(wt_ledger_owner_apply_result(owner, &ticket, f.result, f.result_len,
                                        commit_result, &commit) ==
           WT_LEDGER_OK);
    assert(commit.calls == 1);
    assert(wt_ledger_destroy(&owner) == WT_LEDGER_OK);
}

static void test_callback_failures_are_terminal(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u};
    struct fixture f;
    struct wt_ledger *worker;
    struct wt_ledger *owner;
    struct wt_ledger_ticket ticket;
    struct run_context run;
    struct commit_context commit;
    init_fixture(&f, 7u, p, sizeof(p), r, sizeof(r));
    worker = make_ledger(WT_LEDGER_WORKER, &f, 1u, 1024u);
    assert(wt_ledger_worker_accept(worker, f.request, f.request_len,
                                   f.result_len, &ticket) == WT_LEDGER_OK);
    memset(&run, 0, sizeof(run));
    run.fixture = &f;
    run.fail = 1;
    assert(wt_ledger_worker_run(worker, &ticket, copy_result, &run) ==
           WT_LEDGER_ECALLBACK);
    assert(wt_ledger_worker_run(worker, &ticket, copy_result, &run) ==
           WT_LEDGER_ESTATE);
    assert(run.calls == 1);
    assert(wt_ledger_destroy(&worker) == WT_LEDGER_OK);
    owner = make_ledger(WT_LEDGER_OWNER, &f, 1u, 1024u);
    assert(wt_ledger_owner_register_request(owner, f.request, f.request_len,
                                            f.result_len, &ticket) ==
           WT_LEDGER_OK);
    memset(&commit, 0, sizeof(commit));
    commit.fail = 1;
    assert(wt_ledger_owner_apply_result(owner, &ticket, f.result, f.result_len,
                                        commit_result, &commit) ==
           WT_LEDGER_ECALLBACK);
    assert(wt_ledger_owner_apply_result(owner, &ticket, f.result, f.result_len,
                                        commit_result, &commit) ==
           WT_LEDGER_ESTATE);
    assert(commit.calls == 1);
    assert(wt_ledger_destroy(&owner) == WT_LEDGER_OK);
}

static void test_reentrant_calls_cannot_repeat_or_destroy(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t conflict_payload[] = {9u};
    static const uint8_t r[] = {2u, 3u};
    struct fixture f;
    struct fixture conflict;
    struct wt_ledger *worker;
    struct wt_ledger *owner;
    struct wt_ledger_ticket ticket;
    struct run_context run;
    struct commit_context commit;
    init_fixture(&f, 8u, p, sizeof(p), r, sizeof(r));
    init_fixture(&conflict, 8u, conflict_payload, sizeof(conflict_payload),
                 r, sizeof(r));
    worker = make_ledger(WT_LEDGER_WORKER, &f, 1u, 1024u);
    assert(wt_ledger_worker_accept(worker, f.request, f.request_len,
                                   f.result_len, &ticket) == WT_LEDGER_OK);
    memset(&run, 0, sizeof(run));
    run.fixture = &f;
    run.ledger = worker;
    run.ticket = ticket;
    run.reentrant_accept_fixture = &conflict;
    assert(wt_ledger_worker_run(worker, &ticket, reentrant_result, &run) ==
           WT_LEDGER_OK);
    assert(run.calls == 1);
    assert(run.reentrant_run_status == WT_LEDGER_EBUSY);
    assert(run.reentrant_fail_status == WT_LEDGER_EBUSY);
    assert(run.reentrant_accept_status == WT_LEDGER_EBUSY);
    assert(run.reentrant_destroy_status == WT_LEDGER_EBUSY);
    assert(wt_ledger_destroy(&worker) == WT_LEDGER_OK);
    owner = make_ledger(WT_LEDGER_OWNER, &f, 1u, 1024u);
    assert(wt_ledger_owner_register_request(owner, f.request, f.request_len,
                                            f.result_len, &ticket) ==
           WT_LEDGER_OK);
    memset(&commit, 0, sizeof(commit));
    commit.ledger = owner;
    commit.ticket = ticket;
    commit.encoded_result = f.result;
    commit.encoded_result_len = f.result_len;
    assert(wt_ledger_owner_apply_result(owner, &ticket, f.result, f.result_len,
                                        reentrant_commit, &commit) ==
           WT_LEDGER_OK);
    assert(commit.calls == 1);
    assert(commit.reentrant_apply_status == WT_LEDGER_EBUSY);
    assert(commit.reentrant_destroy_status == WT_LEDGER_EBUSY);
    assert(wt_ledger_destroy(&owner) == WT_LEDGER_OK);
}

static void test_identity_and_generation_never_wrap(void) {
#ifdef WT_LEDGER_TESTING
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u};
    struct fixture f;
    struct wt_ledger_contract next_contract;
    struct wt_ledger *ledger = (struct wt_ledger *)(uintptr_t)1u;
    struct wt_ledger_ticket ticket;
    init_fixture(&f, 15u, p, sizeof(p), r, sizeof(r));
    wt_ledger_private_test_set_next_identity(UINT64_MAX);
    assert(wt_ledger_create(WT_LEDGER_WORKER, &f.contract, 1u, 1024u,
                            &ledger) == WT_LEDGER_EFULL);
    assert(ledger == (struct wt_ledger *)(uintptr_t)1u);
    wt_ledger_private_test_set_next_identity(UINT64_MAX - 1u);
    ledger = NULL;
    assert(wt_ledger_create(WT_LEDGER_WORKER, &f.contract, 1u, 1024u,
                            &ledger) == WT_LEDGER_OK);
    assert(wt_ledger_worker_accept(ledger, f.request, f.request_len,
                                   f.result_len, &ticket) == WT_LEDGER_OK);
    assert(wt_ledger_fail(ledger, &ticket) == WT_LEDGER_OK);
    wt_ledger_private_test_set_generation(ledger, UINT64_MAX);
    next_contract = f.contract;
    ++next_contract.owner_generation;
    assert(wt_ledger_reset(ledger, &next_contract) == WT_LEDGER_EFULL);
    assert(wt_ledger_worker_accept(ledger, f.request, f.request_len,
                                   f.result_len, &ticket) ==
           WT_LEDGER_DUPLICATE);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
    wt_ledger_private_test_set_next_identity(UINT64_C(1000000));
#endif
}

static void test_tickets_failure_close_and_reset(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u};
    struct fixture f;
    struct fixture next;
    struct wt_ledger *ledger;
    struct wt_ledger *foreign;
    struct wt_ledger_ticket ticket;
    struct wt_ledger_ticket mutated;
    struct wt_ledger_ticket next_ticket;
    enum wt_ledger_state state;
    init_fixture(&f, 9u, p, sizeof(p), r, sizeof(r));
    ledger = make_ledger(WT_LEDGER_WORKER, &f, 3u, 2048u);
    foreign = make_ledger(WT_LEDGER_WORKER, &f, 1u, 1024u);
    assert(wt_ledger_worker_accept(ledger, f.request, f.request_len,
                                   f.result_len, &ticket) == WT_LEDGER_OK);
    mutated = ticket;
    ++mutated.nonce;
    assert(wt_ledger_get_state(ledger, &mutated, &state) ==
           WT_LEDGER_EFOREIGN);
    assert(wt_ledger_get_state(foreign, &ticket, &state) ==
           WT_LEDGER_EFOREIGN);
    assert(wt_ledger_fail(ledger, &ticket) == WT_LEDGER_OK);
    assert(wt_ledger_fail(ledger, &ticket) == WT_LEDGER_DUPLICATE);
    assert(wt_ledger_close_epoch(ledger, f.id.domain, f.id.epoch) ==
           WT_LEDGER_OK);
    assert(wt_ledger_worker_accept(ledger, f.request, f.request_len,
                                   f.result_len, &mutated) ==
           WT_LEDGER_ESTALE);
    assert(wt_ledger_close_epoch(ledger, f.id.domain, f.id.epoch) ==
           WT_LEDGER_DUPLICATE);
    assert(wt_ledger_reset(ledger, &f.contract) == WT_LEDGER_EINVAL);
    next = f;
    next.contract.owner_generation = f.contract.owner_generation + 1u;
    next.id.owner_generation = next.contract.owner_generation;
    memcpy(next.id.run_uuid, next.contract.run_uuid, sizeof(next.id.run_uuid));
    {
        struct wt_wire_frame frame;
        assert(wt_wire_decode(f.request, f.request_len, &f.contract.limits,
                              f.contract.schema_fingerprint, &f.contract.abi,
                              &frame) == WT_WIRE_OK);
        frame.id = next.id;
        assert(wt_wire_encode(&frame, &next.contract.limits, next.request,
                              sizeof(next.request), &next.request_len) ==
               WT_WIRE_OK);
    }
    assert(wt_ledger_reset(ledger, &next.contract) == WT_LEDGER_OK);
    assert(wt_ledger_get_state(ledger, &ticket, &state) == WT_LEDGER_ESTALE);
    assert(wt_ledger_worker_accept(ledger, f.request, f.request_len,
                                   f.result_len, &mutated) ==
           WT_LEDGER_EWIRE);
    assert(wt_ledger_worker_accept(ledger, next.request, next.request_len,
                                   next.result_len, &next_ticket) ==
           WT_LEDGER_OK);
    assert(wt_ledger_destroy(&foreign) == WT_LEDGER_OK);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
}

static void test_reset_requires_all_requests_terminal(void) {
    static const uint8_t p[] = {1u};
    static const uint8_t r[] = {2u};
    struct fixture f;
    struct wt_ledger_contract next_contract;
    struct wt_ledger *ledger;
    struct wt_ledger_ticket ticket;
    init_fixture(&f, 13u, p, sizeof(p), r, sizeof(r));
    ledger = make_ledger(WT_LEDGER_WORKER, &f, 1u, 1024u);
    assert(wt_ledger_worker_accept(ledger, f.request, f.request_len,
                                   f.result_len, &ticket) == WT_LEDGER_OK);
    next_contract = f.contract;
    ++next_contract.owner_generation;
    assert(wt_ledger_reset(ledger, &next_contract) == WT_LEDGER_EBUSY);
    assert(wt_ledger_fail(ledger, &ticket) == WT_LEDGER_OK);
    assert(wt_ledger_reset(ledger, &next_contract) == WT_LEDGER_OK);
    assert(wt_ledger_destroy(&ledger) == WT_LEDGER_OK);
}

int main(void) {
    test_worker_duplicate_and_cached_result();
    test_owner_commit_once_and_duplicate_result();
    test_same_id_different_request_is_terminal_conflict();
    test_changed_region_metadata_conflicts_on_same_logical_key();
    test_rejects_bad_wire_before_callbacks();
    test_budget_entry_limit_and_out_of_order();
    test_result_identity_and_fingerprint_precede_commit();
    test_callback_failures_are_terminal();
    test_reentrant_calls_cannot_repeat_or_destroy();
    test_identity_and_generation_never_wrap();
    test_tickets_failure_close_and_reset();
    test_reset_requires_all_requests_terminal();
    test_rejects_uninitialized_contract_sentinels();
    test_destroyed_ledger_ticket_never_matches_recreated_ledger();
    puts("wt_ledger tests passed");
    return 0;
}
