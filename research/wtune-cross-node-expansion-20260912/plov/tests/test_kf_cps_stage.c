#include "wt_cumulus.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define STORAGE 160
#define CHECK(condition) do { ++checks; if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)

struct fixture {
    void *args[WT_KF_NARG];
    size_t capacity[WT_KF_NARG], length[WT_KF_NARG];
    int32_t scalar[WT_KF_NARG];
    uint32_t inactive[5];
    unsigned char arrays[WT_KF_NARG][STORAGE];
};
enum mode { NORMAL, BAD_INPUT, BAD_GUARD, CALLBACK_FAIL, REBIND,
            REENTER_EXECUTE, REENTER_COMMIT, REENTER_DESTROY,
            BAD_INACTIVE };
struct execution {
    struct fixture *fixture;
    struct wt_kf_txn *transaction;
    enum mode mode;
    unsigned char *saved;
    enum wt_kf_status inner_status;
    int32_t seen_edrates;
    unsigned bad_inactive_index;
    int calls;
};
static int checks;
static const unsigned inactive_fields[5] = {
    WT_KF_UDR_KF, WT_KF_DDR_KF, WT_KF_UER_KF, WT_KF_DER_KF,
    WT_KF_TIMEC_KF
};

static void initialize(struct fixture *fixture)
{
    unsigned i;
    memset(fixture, 0, sizeof(*fixture));
    for (i = 0; i < WT_KF_NARG; ++i) {
        memset(fixture->arrays[i], (int)i + 1, STORAGE);
        if (wt_kf_shape[i] == WT_KF_SCALAR) {
            fixture->args[i] = &fixture->scalar[i];
            fixture->capacity[i] = fixture->length[i] = 4;
        } else {
            fixture->args[i] = fixture->arrays[i];
            fixture->capacity[i] = STORAGE; /* Actual storage, never invented. */
            fixture->length[i] = wt_kf_shape[i] == WT_KF_3D ? 8 : 4;
        }
    }
    for (i = WT_KF_IDS; i <= WT_KF_KTE; ++i) fixture->scalar[i] = 1;
    fixture->scalar[WT_KF_KDE] = fixture->scalar[WT_KF_KME] = 2;
    fixture->scalar[WT_KF_TRIGGER] = 1;
}

static void use_inactive_backing(struct fixture *fixture)
{
    unsigned i;
    for (i = 0; i < 5; ++i) {
        fixture->inactive[i] = UINT32_C(0x10203040) + i;
        fixture->args[inactive_fields[i]] = &fixture->inactive[i];
        fixture->capacity[inactive_fields[i]] = 4;
        fixture->length[inactive_fields[i]] = 4;
    }
}

/* Physical tile stays at (1,1), with real negative-index memory halos. */
static void negative_halos(struct fixture *fixture)
{
    unsigned i;
    fixture->scalar[WT_KF_IMS] = fixture->scalar[WT_KF_JMS] = -2;
    for (i = 0; i < WT_KF_NARG; ++i)
        fixture->length[i] = wt_kf_shape[i] == WT_KF_3D ? 128 :
                             wt_kf_shape[i] == WT_KF_2D ? 64 : 4;
}

static int same_owner(const struct fixture *a, const struct fixture *b)
{
    /* Also compare every byte outside the formal extents as an owner canary. */
    return !memcmp(a->scalar, b->scalar, sizeof(a->scalar)) &&
           !memcmp(a->inactive, b->inactive, sizeof(a->inactive)) &&
           !memcmp(a->arrays, b->arrays, sizeof(a->arrays));
}

static int fake_cps(void **args, void *context)
{
    struct execution *execution = context;
    unsigned i;
    size_t j;
    ++execution->calls;
    memcpy(&execution->seen_edrates, args[WT_KF_KF_EDRATES],
           sizeof(execution->seen_edrates));
    if (execution->mode == REENTER_EXECUTE) {
        execution->inner_status = wt_kf_execute(execution->transaction, fake_cps, context);
        return 0;
    }
    if (execution->mode == REENTER_COMMIT) {
        execution->inner_status = wt_kf_commit(execution->transaction);
        return 0;
    }
    if (execution->mode == REENTER_DESTROY) {
        wt_kf_destroy(execution->transaction);
        return 0;
    }
    if (execution->mode == REBIND) {
        args[WT_KF_RHO] = (void *)(uintptr_t)1;
        return 0;
    }
    for (i = 0; i < WT_KF_NARG; ++i) {
        int inactive = *(int32_t *)args[WT_KF_KF_EDRATES] != 1 &&
          (i == WT_KF_UDR_KF || i == WT_KF_DDR_KF || i == WT_KF_UER_KF ||
           i == WT_KF_DER_KF || i == WT_KF_TIMEC_KF);
        if (args[i] && wt_kf_intent[i] != WT_KF_IN && !inactive)
            for (j = 0; j < execution->fixture->length[i]; ++j)
                ((unsigned char *)args[i])[j] ^= (unsigned char)(0x5a + i + j);
    }
    if (execution->mode == BAD_INPUT) {
        execution->saved = args[WT_KF_RHO];
        execution->saved[0] ^= 1;
    }
    if (execution->mode == BAD_GUARD) {
        execution->saved = args[WT_KF_TIMEC_KF];
        execution->saved[-1] = 0; /* Inside allocation's explicit guard area. */
    }
    if (execution->mode == BAD_INACTIVE) {
        execution->saved = args[execution->bad_inactive_index];
        execution->saved[0] ^= 1;
    }
    return execution->mode == CALLBACK_FAIL ? 7 : 0;
}

static int rejected(struct fixture *f, struct wt_kf_config *config,
                    enum wt_kf_status expected)
{
    struct fixture before = *f;
    struct wt_kf_txn *transaction = NULL;
    CHECK(wt_kf_prepare(f->args, f->capacity, config, &transaction) == expected);
    CHECK(transaction == NULL);
    CHECK(same_owner(f, &before));
    return 0;
}

static int equivalence(int with_halos, int absent_shall)
{
    struct fixture owner, direct, before;
    struct wt_kf_config config = {1, 1, 0, 1024 * 1024};
    struct wt_kf_txn *transaction = NULL;
    struct execution staged = {0}, oracle = {0};
    initialize(&owner);
    initialize(&direct);
    if (with_halos) { negative_halos(&owner); negative_halos(&direct); }
    if (absent_shall) owner.args[WT_KF_SHALL] = direct.args[WT_KF_SHALL] = NULL;
    before = owner;
    staged.fixture = &owner;
    oracle.fixture = &direct;
    CHECK(fake_cps(direct.args, &oracle) == 0);
    CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
    CHECK(same_owner(&owner, &before));
    staged.transaction = transaction;
    CHECK(wt_kf_execute(transaction, fake_cps, &staged) == WT_KF_OK);
    CHECK(staged.calls == 1);
    CHECK(same_owner(&owner, &before));
    CHECK(wt_kf_execute(transaction, fake_cps, &staged) == WT_KF_BAD_STATE);
    CHECK(staged.calls == 1);
    CHECK(wt_kf_commit(transaction) == WT_KF_OK);
    CHECK(same_owner(&owner, &direct));
    CHECK(wt_kf_commit(transaction) == WT_KF_BAD_STATE);
    CHECK(same_owner(&owner, &direct));
    wt_kf_destroy(transaction);
    return 0;
}

static int failure_paths(void)
{
    struct fixture owner, before;
    struct wt_kf_config config = {1, 1, 0, 1024 * 1024};
    struct wt_kf_txn *transaction;
    enum mode mode;
    for (mode = BAD_INPUT; mode <= REENTER_DESTROY; ++mode) {
        struct execution execution = {0};
        enum wt_kf_status expected_execute =
            mode == BAD_INPUT || mode == BAD_GUARD ? WT_KF_OK : WT_KF_BAD_STATE;
        initialize(&owner);
        before = owner;
        execution.fixture = &owner;
        execution.mode = mode;
        CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
        execution.transaction = transaction;
        CHECK(wt_kf_execute(transaction, fake_cps, &execution) == expected_execute);
        CHECK(execution.calls == 1);
        CHECK(same_owner(&owner, &before));
        CHECK(wt_kf_commit(transaction) ==
              (mode == BAD_INPUT ? WT_KF_STALE : mode == BAD_GUARD ? WT_KF_GUARD : WT_KF_BAD_STATE));
        CHECK(same_owner(&owner, &before));
        /* Repairing the evidence must not revive a failed transaction. */
        if (mode == BAD_INPUT) execution.saved[0] ^= 1;
        if (mode == BAD_GUARD) execution.saved[-1] = 0xa5;
        CHECK(wt_kf_commit(transaction) == WT_KF_BAD_STATE);
        CHECK(wt_kf_execute(transaction, fake_cps, &execution) == WT_KF_BAD_STATE);
        CHECK(execution.calls == 1);
        if (mode == REENTER_EXECUTE || mode == REENTER_COMMIT)
            CHECK(execution.inner_status == WT_KF_BAD_STATE);
        CHECK(same_owner(&owner, &before));
        wt_kf_destroy(transaction);
    }
    {
        struct execution execution = {0};
        initialize(&owner);
        before = owner;
        execution.fixture = &owner;
        CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
        CHECK(wt_kf_execute(transaction, fake_cps, &execution) == WT_KF_OK);
        owner.arrays[WT_KF_RHO][0] ^= 1;
        CHECK(wt_kf_commit(transaction) == WT_KF_STALE);
        owner.arrays[WT_KF_RHO][0] ^= 1;
        CHECK(same_owner(&owner, &before));
        CHECK(wt_kf_commit(transaction) == WT_KF_BAD_STATE);
        wt_kf_destroy(transaction);
    }
    initialize(&owner);
    before = owner;
    CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
    wt_kf_destroy(transaction);
    CHECK(same_owner(&owner, &before));
    return 0;
}

static int inactive_contract(void)
{
    struct fixture owner, direct, before;
    struct wt_kf_config config = {1, 1, 0, 1024 * 1024};
    struct wt_kf_txn *transaction;
    struct execution execution = {0}, oracle = {0};

    initialize(&owner);
    initialize(&direct);
    use_inactive_backing(&owner);
    use_inactive_backing(&direct);
    before = owner;
    execution.fixture = &owner;
    oracle.fixture = &direct;
    CHECK(fake_cps(direct.args, &oracle) == 0);
    CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
    execution.transaction = transaction;
    CHECK(wt_kf_execute(transaction, fake_cps, &execution) == WT_KF_OK);
    CHECK(execution.seen_edrates == 0);
    CHECK(wt_kf_commit(transaction) == WT_KF_OK);
    CHECK(same_owner(&owner, &direct));
    CHECK(!memcmp(owner.inactive, before.inactive, sizeof(owner.inactive)));
    wt_kf_destroy(transaction);

    {
        unsigned i;
        for (i = 0; i < 5; ++i) {
            initialize(&owner);
            use_inactive_backing(&owner);
            before = owner;
            execution = (struct execution){0};
            execution.fixture = &owner;
            execution.mode = BAD_INACTIVE;
            execution.bad_inactive_index = inactive_fields[i];
            CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
            execution.transaction = transaction;
            CHECK(wt_kf_execute(transaction, fake_cps, &execution) == WT_KF_OK);
            CHECK(wt_kf_commit(transaction) == WT_KF_STALE);
            CHECK(same_owner(&owner, &before));
            CHECK(wt_kf_commit(transaction) == WT_KF_BAD_STATE);
            wt_kf_destroy(transaction);
        }
    }

    initialize(&owner);
    use_inactive_backing(&owner);
    execution = (struct execution){0};
    execution.fixture = &owner;
    CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
    execution.transaction = transaction;
    owner.scalar[WT_KF_KF_EDRATES] = 1;
    before = owner;
    CHECK(wt_kf_execute(transaction, fake_cps, &execution) == WT_KF_OK);
    CHECK(execution.seen_edrates == 0);
    CHECK(wt_kf_commit(transaction) == WT_KF_STALE);
    CHECK(same_owner(&owner, &before));
    wt_kf_destroy(transaction);

    initialize(&owner);
    owner.scalar[WT_KF_KF_EDRATES] = 1;
    execution = (struct execution){0};
    execution.fixture = &owner;
    CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
    execution.transaction = transaction;
    owner.scalar[WT_KF_KF_EDRATES] = 0;
    before = owner;
    CHECK(wt_kf_execute(transaction, fake_cps, &execution) == WT_KF_OK);
    CHECK(execution.seen_edrates == 1);
    CHECK(wt_kf_commit(transaction) == WT_KF_STALE);
    CHECK(same_owner(&owner, &before));
    wt_kf_destroy(transaction);
    return 0;
}

static int active_contract(void)
{
    struct fixture owner, direct;
    struct wt_kf_config config = {1, 1, 0, 1024 * 1024};
    struct wt_kf_txn *transaction;
    struct execution execution = {0}, oracle = {0};

    initialize(&owner);
    owner.scalar[WT_KF_KF_EDRATES] = 1;
    use_inactive_backing(&owner);
    CHECK(rejected(&owner, &config, WT_KF_CAPACITY) == 0);

    initialize(&owner);
    initialize(&direct);
    owner.scalar[WT_KF_KF_EDRATES] = direct.scalar[WT_KF_KF_EDRATES] = 1;
    execution.fixture = &owner;
    oracle.fixture = &direct;
    CHECK(fake_cps(direct.args, &oracle) == 0);
    CHECK(wt_kf_prepare(owner.args, owner.capacity, &config, &transaction) == WT_KF_OK);
    execution.transaction = transaction;
    CHECK(wt_kf_execute(transaction, fake_cps, &execution) == WT_KF_OK);
    CHECK(execution.seen_edrates == 1);
    CHECK(wt_kf_commit(transaction) == WT_KF_OK);
    CHECK(same_owner(&owner, &direct));
    wt_kf_destroy(transaction);
    return 0;
}

static int rejected_inputs(void)
{
    struct fixture f;
    struct wt_kf_config config = {1, 1, 0, 1024 * 1024};
    size_t payload = 0;
    unsigned i;
    initialize(&f);
    config.enabled = 0; CHECK(!rejected(&f, &config, WT_KF_DISABLED)); config.enabled = 1;
    config.num_tiles = 2; CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT)); config.num_tiles = 1;
    config.in_parallel = 1; CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT)); config.in_parallel = 0;
    f.scalar[WT_KF_TRIGGER] = 2; CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT));
    initialize(&f);
    f.scalar[WT_KF_ADAPT_STEP_FLAG] = 1; CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT));
    initialize(&f);
    f.args[WT_KF_ADAPT_STEP_FLAG] = NULL; CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT));
    for (i = WT_KF_F_QV; i <= WT_KF_RQVFTEN; ++i) {
        initialize(&f);
        f.args[i] = NULL;
        CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT));
    }
    initialize(&f);
    f.capacity[WT_KF_DT] = 3; CHECK(!rejected(&f, &config, WT_KF_CAPACITY));
    initialize(&f);
    f.capacity[WT_KF_RHO] = 7; CHECK(!rejected(&f, &config, WT_KF_CAPACITY));
    initialize(&f);
    f.args[WT_KF_U] = f.args[WT_KF_V]; CHECK(!rejected(&f, &config, WT_KF_OVERLAP));
    initialize(&f);
    f.scalar[WT_KF_IME] = 0; CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT));
    initialize(&f);
    f.scalar[WT_KF_KTE] = 2; CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT));
    initialize(&f);
    f.scalar[WT_KF_KMS] = f.scalar[WT_KF_KTS] = f.scalar[WT_KF_KTE] = 2;
    f.scalar[WT_KF_KME] = 3;
    CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT)); /* KTS alone, otherwise valid. */
    initialize(&f);
    f.scalar[WT_KF_IMS] = f.scalar[WT_KF_JMS] = INT32_MIN;
    f.scalar[WT_KF_IME] = f.scalar[WT_KF_JME] = INT32_MAX;
    CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT));
    initialize(&f);
    f.args[WT_KF_IDS] = (void *)(UINTPTR_MAX - 2);
    CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT)); /* Reject before scalar read. */
    initialize(&f);
    f.args[WT_KF_RHO] = (void *)(UINTPTR_MAX - 7);
    CHECK(!rejected(&f, &config, WT_KF_BAD_ARGUMENT)); /* Full array end overflow. */
    initialize(&f);
    for (i = 0; i < WT_KF_NARG; ++i) payload += f.length[i];
    config.byte_budget = payload; CHECK(!rejected(&f, &config, WT_KF_BUDGET));
    config.byte_budget = 2 * payload + 32 * WT_KF_NARG;
    CHECK(!rejected(&f, &config, WT_KF_BUDGET)); /* Metadata must count too. */
    return 0;
}

int main(void)
{
    CHECK(WT_KF_NARG == 79 && WT_KF_IDS == 0 && WT_KF_KTE == 17 &&
          WT_KF_RAINCV == 25 && WT_KF_F_QV == 57 &&
          WT_KF_RQVFTEN == 68 && WT_KF_KF_EDRATES == 78);
    CHECK(!equivalence(0, 0));
    CHECK(!equivalence(1, 0));
    CHECK(!equivalence(0, 1));
    CHECK(!failure_paths());
    CHECK(!inactive_contract());
    CHECK(!active_contract());
    CHECK(!rejected_inputs());
    printf("KF CPS local transaction fixture: %d checks PASS\n", checks);
    return 0;
}
