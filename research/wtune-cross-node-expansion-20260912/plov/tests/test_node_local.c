#define _POSIX_C_SOURCE 200809L
#include "wt_node_local.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) fail(#x, __LINE__); } while (0)
#define PAYLOAD_BYTES 512u

struct request { uint64_t value, salt; };
struct geometry { uint32_t first, count, stride, reserved; };
struct result { uint64_t value, salt, pid, thread; };

struct callback_context {
    _Atomic int calls;
    uint64_t thread;
    int mode;
    int started_fd;
    int release_fd;
};

struct commit_context {
    int validate_calls;
    int commit_calls;
    struct result result;
};

struct helper_thread {
    struct wt_nl_service *service;
    struct wt_nl_signature signature;
    struct callback_context callback;
    enum wt_nl_status status;
    struct wt_nl_ticket ticket;
    uint32_t rank, thread;
};

struct test_shared_header {
    alignas(64) _Atomic uint32_t init_state;
    uint32_t version;
    uint64_t magic;
    uint32_t max_ranks, max_threads, slot_count, reserved;
    size_t mapping_bytes, header_bytes, slot_stride, slot_payload_bytes;
    _Atomic uint64_t next_work_id;
    _Atomic uint64_t staged, self_executed, helper_executed, committed, failed;
};

enum callback_mode {
    CALLBACK_NORMAL,
    CALLBACK_REJECT,
    CALLBACK_FATAL,
    CALLBACK_SHORT,
    CALLBACK_REQUEST_MUTATION,
    CALLBACK_DESCRIPTOR_MUTATION,
    CALLBACK_GUARD_MUTATION,
    CALLBACK_BLOCK
};

static void fail(const char *expression, int line)
{
    fprintf(stderr, "FAIL line %d: %s\n", line, expression);
    exit(2);
}

static struct wt_nl_signature signature(uint64_t phase, uint64_t epoch)
{
    struct wt_nl_signature value = {phase, 2, 3, 4, epoch, 6, 7};
    return value;
}

static struct wt_nl_descriptor descriptor(uint64_t phase, uint64_t epoch,
                                           uint64_t sequence)
{
    struct wt_nl_descriptor value;
    memset(&value, 0, sizeof value);
    value.signature = signature(phase, epoch);
    value.call_sequence = sequence;
    value.geometry_bytes = sizeof(struct geometry);
    value.request_bytes = sizeof(struct request);
    value.result_bytes = sizeof(struct result);
    return value;
}

static int compatible(void *opaque, const struct wt_nl_descriptor *desc,
                      const void *geometry_data, size_t geometry_bytes)
{
    struct geometry geometry;
    (void)opaque;
    if (geometry_bytes != sizeof geometry) return 0;
    memcpy(&geometry, geometry_data, sizeof geometry);
    return desc->geometry_bytes == sizeof geometry && geometry.count == 4 &&
           geometry.stride == 1;
}

static int execute(void *opaque, const struct wt_nl_descriptor *desc,
                   const void *geometry_data, size_t geometry_bytes,
                   const void *request_data, size_t request_bytes,
                   void *result_data, size_t result_capacity,
                   size_t *result_written)
{
    struct callback_context *context = opaque;
    struct request request;
    struct result result;
    char byte = 's';
    (void)geometry_data;
    (void)geometry_bytes;
    CHECK(request_bytes == sizeof request);
    CHECK(result_capacity == sizeof result);
    memcpy(&request, request_data, sizeof request);
    atomic_fetch_add(&context->calls, 1);
    if (context->mode == CALLBACK_REJECT) return WT_NL_EXECUTE_REJECTED;
    if (context->mode == CALLBACK_FATAL) return WT_NL_EXECUTE_FATAL;
    if (context->mode == CALLBACK_REQUEST_MUTATION)
        ((unsigned char *)(uintptr_t)request_data)[0] ^= 1;
    if (context->mode == CALLBACK_DESCRIPTOR_MUTATION)
        ((struct wt_nl_descriptor *)(uintptr_t)desc)->call_sequence++;
    result.value = request.value * 3;
    result.salt = request.salt;
    result.pid = (uint64_t)getpid();
    result.thread = context->thread;
    memcpy(result_data, &result, sizeof result);
    *result_written = context->mode == CALLBACK_SHORT ? sizeof result - 1 :
                                                        sizeof result;
    if (context->mode == CALLBACK_GUARD_MUTATION)
        ((unsigned char *)result_data)[result_capacity] = 0;
    if (context->mode == CALLBACK_BLOCK) {
        CHECK(write(context->started_fd, &byte, 1) == 1);
        CHECK(read(context->release_fd, &byte, 1) == 1);
    }
    return WT_NL_EXECUTE_OK;
}

static int validate(void *opaque, const struct wt_nl_descriptor *desc,
                    const void *geometry_data, size_t geometry_bytes,
                    const void *request_data, size_t request_bytes,
                    const void *result_data, size_t result_bytes)
{
    struct commit_context *context = opaque;
    struct request request;
    struct result result;
    (void)desc;
    (void)geometry_data;
    context->validate_calls++;
    if (request_bytes != sizeof request || result_bytes != sizeof result)
        return 0;
    memcpy(&request, request_data, sizeof request);
    memcpy(&result, result_data, sizeof result);
    return geometry_bytes == sizeof(struct geometry) &&
           result.value == request.value * 3 && result.salt == request.salt;
}

static void commit(void *opaque, const struct wt_nl_descriptor *desc,
                   const void *geometry_data, size_t geometry_bytes,
                   const void *request_data, size_t request_bytes,
                   const void *result_data, size_t result_bytes)
{
    struct commit_context *context = opaque;
    (void)desc; (void)geometry_data; (void)geometry_bytes;
    (void)request_data; (void)request_bytes;
    CHECK(result_bytes == sizeof context->result);
    context->commit_calls++;
    memcpy(&context->result, result_data, result_bytes);
}

static int make_service(const char *directory, const char *name, int *fd,
                        size_t *bytes, struct wt_nl_service *service)
{
    struct wt_nl_config config = {2, 2, PAYLOAD_BYTES};
    char path[1024];
    CHECK(snprintf(path, sizeof path, "%s/%s", directory, name) > 0);
    *fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    CHECK(*fd >= 0);
    CHECK(wt_nl_required_bytes(&config, bytes) == WT_NL_OK);
    CHECK(wt_nl_init_fd(*fd, &config, service) == WT_NL_OK);
    return 0;
}

static struct wt_nl_ticket publish(struct wt_nl_service *service,
                                   uint32_t thread, uint64_t phase,
                                   uint64_t epoch, uint64_t sequence)
{
    struct wt_nl_descriptor desc = descriptor(phase, epoch, sequence);
    struct geometry geometry = {8, 4, 1, 0};
    struct request request = {sequence + 10, sequence + 20};
    struct wt_nl_ticket ticket;
    CHECK(wt_nl_publish(service, 0, thread, &desc, &geometry, &request,
                        &ticket) == WT_NL_OK);
    return ticket;
}

static void *helper_main(void *opaque)
{
    struct helper_thread *thread = opaque;
    thread->callback.thread = thread->thread;
    thread->status = wt_nl_try_execute(
        thread->service, thread->rank, thread->thread, &thread->signature,
        compatible, execute, &thread->callback,
        ((uint64_t)thread->rank << 32) | thread->thread, &thread->ticket);
    return NULL;
}

static void two_thread_try(struct wt_nl_service *service, uint32_t rank,
                           struct wt_nl_signature sig,
                           struct helper_thread threads[2])
{
    pthread_t ids[2];
    unsigned i;
    memset(threads, 0, 2 * sizeof *threads);
    for (i = 0; i < 2; ++i) {
        threads[i].service = service;
        threads[i].signature = sig;
        threads[i].rank = rank;
        threads[i].thread = i;
        CHECK(pthread_create(&ids[i], NULL, helper_main, &threads[i]) == 0);
    }
    for (i = 0; i < 2; ++i) CHECK(pthread_join(ids[i], NULL) == 0);
}

struct child_report { int ok, idle, calls, winning_thread; pid_t pid; };

static void test_cross_process(const char *directory)
{
    struct wt_nl_service service, attached;
    struct helper_thread same_rank[2], helpers[2];
    struct commit_context owner = {0};
    struct wt_nl_counters counters;
    struct wt_nl_ticket ticket;
    struct child_report report;
    uint64_t executor_tag = 0;
    int fd, pipefd[2], status;
    size_t bytes;
    pid_t child;

    make_service(directory, "cross-process.shm", &fd, &bytes, &service);
    ticket = publish(&service, 0, 11, 12, 1);
    CHECK(wt_nl_ticket_executor(&service,0,0,&ticket,&executor_tag)==
          WT_NL_PENDING);
    CHECK(wt_nl_ticket_executor(&service,1,0,&ticket,&executor_tag)==
          WT_NL_OWNER);
    two_thread_try(&service, 0, signature(11, 12), same_rank);
    CHECK(same_rank[0].status == WT_NL_IDLE &&
          same_rank[1].status == WT_NL_IDLE);
    CHECK(atomic_load(&same_rank[0].callback.calls) == 0 &&
          atomic_load(&same_rank[1].callback.calls) == 0);
    CHECK(wt_nl_owner_collect(&service, 1, 0, &ticket, validate, commit,
                              &owner) == WT_NL_OWNER);
    CHECK(wt_nl_owner_collect(&service, 0, 1, &ticket, validate, commit,
                              &owner) == WT_NL_OWNER);
    CHECK(pipe(pipefd) == 0);
    child = fork();
    CHECK(child >= 0);
    if (!child) {
        unsigned i;
        struct child_report child_report = {0};
        close(pipefd[0]);
        wt_nl_detach(&service);
        CHECK(wt_nl_attach_fd(fd, bytes, &attached) == WT_NL_OK);
        two_thread_try(&attached, 1, signature(11, 12), helpers);
        for (i = 0; i < 2; ++i) {
            child_report.ok += helpers[i].status == WT_NL_OK;
            child_report.idle += helpers[i].status == WT_NL_IDLE;
            child_report.calls += atomic_load(&helpers[i].callback.calls);
            if (helpers[i].status == WT_NL_OK)
                child_report.winning_thread = (int)i;
        }
        child_report.pid = getpid();
        CHECK(write(pipefd[1], &child_report, sizeof child_report) ==
              (ssize_t)sizeof child_report);
        wt_nl_detach(&attached);
        close(fd); close(pipefd[1]);
        _exit(0);
    }
    close(pipefd[1]);
    CHECK(read(pipefd[0], &report, sizeof report) == (ssize_t)sizeof report);
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
          WEXITSTATUS(status) == 0);
    CHECK(report.ok == 1 && report.idle == 1 && report.calls == 1);
    CHECK(wt_nl_ticket_executor(&service,0,0,&ticket,&executor_tag)==WT_NL_OK);
    CHECK(executor_tag==(UINT64_C(1)<<32 |
                         (uint64_t)report.winning_thread));
    CHECK(wt_nl_owner_collect(&service, 0, 0, &ticket, validate, commit,
                              &owner) == WT_NL_OK);
    CHECK(wt_nl_ticket_executor(&service,0,0,&ticket,&executor_tag)==WT_NL_OK);
    CHECK(executor_tag==(UINT64_C(1)<<32 |
                         (uint64_t)report.winning_thread));
    CHECK(owner.validate_calls == 1 && owner.commit_calls == 1);
    CHECK(owner.result.pid == (uint64_t)report.pid &&
          owner.result.pid != (uint64_t)getpid());
    {
        struct result exact = {33, 21, (uint64_t)report.pid,
                               (uint64_t)report.winning_thread};
        CHECK(!memcmp(&owner.result, &exact, sizeof exact));
    }
    CHECK(wt_nl_owner_collect(&service, 0, 0, &ticket, validate, commit,
                              &owner) == WT_NL_BUSY);
    CHECK(wt_nl_get_counters(&service, &counters) == WT_NL_OK);
    CHECK(counters.staged == 1 && counters.helper_executed == 1 &&
          counters.self_executed == 0 && counters.committed == 1 &&
          counters.failed == 0);
    CHECK(wt_nl_finish(&service, 0, 0, &ticket) == WT_NL_OK);
    CHECK(wt_nl_ticket_executor(&service,0,0,&ticket,&executor_tag)==
          WT_NL_STALE);
    close(pipefd[0]); wt_nl_detach(&service); close(fd);
}

static void test_mapping_validation(const char *directory)
{
    struct wt_nl_config config = {2, 2, PAYLOAD_BYTES};
    struct wt_nl_service service, attached;
    struct test_shared_header *header;
    struct wt_nl_descriptor oversized = descriptor(1, 1, 1);
    struct geometry geometry = {8, 4, 1, 0};
    struct request request = {1, 2};
    struct wt_nl_ticket ticket;
    void *mapping = NULL;
    char path[1024];
    size_t bytes;
    int short_fd;

    CHECK(wt_nl_required_bytes(&config, &bytes) == WT_NL_OK);
    CHECK(posix_memalign(&mapping, 64, bytes) == 0);
    CHECK(wt_nl_init_mapping(mapping, bytes, &config, &service) == WT_NL_OK);
    CHECK(wt_nl_attach_mapping((unsigned char *)mapping + 1, bytes - 1,
                               &attached) == WT_NL_INVALID);
    header = mapping;
    header->slot_stride += 64;
    CHECK(wt_nl_attach_mapping(mapping, bytes, &attached) == WT_NL_INVALID);
    header->slot_stride -= 64;
    oversized.result_bytes = PAYLOAD_BYTES;
    CHECK(wt_nl_publish(&service, 0, 0, &oversized, &geometry, &request,
                        &ticket) == WT_NL_OVERFLOW);
    ticket = publish(&service, 0, 1, 1, 1);
    CHECK(wt_nl_owner_poison(&service, 0, 0, &ticket) == WT_NL_FAILED);
    oversized=descriptor(1,1,2);
    atomic_store(&header->next_work_id,0);
    CHECK(wt_nl_publish(&service,0,1,&oversized,&geometry,&request,&ticket)==WT_NL_OVERFLOW);
    CHECK(atomic_load(&header->next_work_id)==0);
    atomic_store(&header->next_work_id,UINT64_MAX-1);
    CHECK(wt_nl_publish(&service,0,1,&oversized,&geometry,&request,&ticket)==WT_NL_OK);
    CHECK(ticket.work_id==UINT64_MAX-1);
    CHECK(wt_nl_owner_poison(&service,0,1,&ticket)==WT_NL_FAILED);
    CHECK(wt_nl_publish(&service,1,0,&oversized,&geometry,&request,&ticket)==WT_NL_OVERFLOW);
    CHECK(atomic_load(&header->next_work_id)==UINT64_MAX);
    wt_nl_detach(&service);
    free(mapping);

    CHECK(snprintf(path, sizeof path, "%s/short-fd.shm", directory) > 0);
    short_fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    CHECK(short_fd >= 0 && ftruncate(short_fd, 1) == 0);
    CHECK(wt_nl_attach_fd(short_fd, bytes, &attached) == WT_NL_INVALID);
    close(short_fd);
}

static void test_signature_self_and_stale(const char *directory)
{
    struct wt_nl_service service;
    struct helper_thread helper = {0};
    struct callback_context self = {0};
    struct commit_context owner = {0};
    struct wt_nl_ticket first, second;
    struct wt_nl_counters counters;
    int fd;
    size_t bytes;

    make_service(directory, "self-stale.shm", &fd, &bytes, &service);
    first = publish(&service, 0, 20, 30, 1);
    helper.service = &service; helper.rank = 1; helper.thread = 0;
    helper.signature = signature(21, 30);
    helper_main(&helper);
    CHECK(helper.status == WT_NL_IDLE && atomic_load(&helper.callback.calls) == 0);
    helper.signature = signature(20, 31);
    helper_main(&helper);
    CHECK(helper.status == WT_NL_IDLE && atomic_load(&helper.callback.calls) == 0);
    CHECK(wt_nl_owner_execute(&service, 0, 0, &first, execute, &self, 9) ==
          WT_NL_OK);
    CHECK(wt_nl_owner_collect(&service, 0, 0, &first, validate, commit,
                              &owner) == WT_NL_OK);
    CHECK(wt_nl_finish(&service, 0, 0, &first) == WT_NL_OK);
    second = publish(&service, 0, 20, 30, 2);
    CHECK(wt_nl_owner_poison(&service, 0, 0, &first) == WT_NL_STALE);
    CHECK(wt_nl_owner_execute(&service, 0, 0, &first, execute, &self, 10) ==
          WT_NL_STALE);
    CHECK(wt_nl_owner_execute(&service, 0, 0, &second, execute, &self, 11) ==
          WT_NL_OK);
    CHECK(wt_nl_owner_collect(&service, 0, 0, &second, validate, commit,
                              &owner) == WT_NL_OK);
    CHECK(wt_nl_finish(&service, 0, 0, &second) == WT_NL_OK);
    CHECK(owner.commit_calls == 2 && atomic_load(&self.calls) == 2);
    CHECK(wt_nl_get_counters(&service, &counters) == WT_NL_OK);
    CHECK(counters.staged == 2 && counters.self_executed == 2 &&
          counters.committed == 2 && counters.failed == 0);
    wt_nl_detach(&service); close(fd);
}

static void test_execute_failure(const char *directory, int mode,
                                 const char *name, enum wt_nl_status expected)
{
    struct wt_nl_service service;
    struct callback_context callback = {0};
    struct commit_context owner = {0};
    struct wt_nl_counters counters;
    struct wt_nl_ticket ticket;
    int fd;
    size_t bytes;

    make_service(directory, name, &fd, &bytes, &service);
    ticket = publish(&service, 0, 40, 50, 1);
    callback.mode = mode;
    CHECK(wt_nl_owner_execute(&service, 0, 0, &ticket, execute, &callback, 1) ==
          expected);
    CHECK(wt_nl_owner_collect(&service, 0, 0, &ticket, validate, commit,
                              &owner) == expected);
    CHECK(wt_nl_owner_execute(&service, 0, 0, &ticket, execute, &callback, 2) ==
          WT_NL_FAILED);
    CHECK(atomic_load(&callback.calls) == 1 && owner.commit_calls == 0);
    CHECK(wt_nl_get_counters(&service, &counters) == WT_NL_OK);
    CHECK(counters.staged == 1 && counters.self_executed == 0 &&
          counters.helper_executed == 0 && counters.committed == 0 &&
          counters.failed == 1);
    CHECK(wt_nl_finish(&service, 0, 0, &ticket) == WT_NL_FAILED);
    wt_nl_detach(&service); close(fd);
}

static void test_poison_running(const char *directory)
{
    struct wt_nl_service service, attached;
    struct helper_thread helper = {0};
    struct commit_context owner = {0};
    struct wt_nl_ticket ticket;
    int fd, started[2], release[2], report[2], status, child_status;
    size_t bytes;
    pid_t child;
    char byte;
    uint64_t executor_tag=0;

    make_service(directory, "poison-running.shm", &fd, &bytes, &service);
    ticket = publish(&service, 0, 60, 70, 1);
    CHECK(pipe(started) == 0 && pipe(release) == 0 && pipe(report) == 0);
    child = fork(); CHECK(child >= 0);
    if (!child) {
        close(started[0]); close(release[1]); close(report[0]);
        wt_nl_detach(&service);
        CHECK(wt_nl_attach_fd(fd, bytes, &attached) == WT_NL_OK);
        helper.service = &attached; helper.rank = 1; helper.thread = 0;
        helper.signature = signature(60, 70);
        helper.callback.mode = CALLBACK_BLOCK;
        helper.callback.started_fd = started[1];
        helper.callback.release_fd = release[0];
        helper_main(&helper);
        status = helper.status;
        CHECK(write(report[1], &status, sizeof status) == (ssize_t)sizeof status);
        wt_nl_detach(&attached);
        _exit(0);
    }
    close(started[1]); close(release[0]); close(report[1]);
    CHECK(read(started[0], &byte, 1) == 1);
    CHECK(wt_nl_ticket_executor(&service,0,0,&ticket,&executor_tag)==
          WT_NL_PENDING);
    CHECK(wt_nl_owner_poison(&service, 0, 0, &ticket) == WT_NL_FAILED);
    CHECK(write(release[1], "r", 1) == 1);
    CHECK(read(report[0], &child_status, sizeof child_status) ==
          (ssize_t)sizeof child_status);
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
          WEXITSTATUS(status) == 0);
    CHECK(child_status == WT_NL_FAILED);
    CHECK(wt_nl_owner_collect(&service, 0, 0, &ticket, validate, commit,
                              &owner) == WT_NL_FAILED);
    CHECK(owner.commit_calls == 0);
    wt_nl_detach(&service); close(fd);
    close(started[0]); close(release[1]); close(report[0]);
}

static void test_repeat_claim(const char *directory)
{
    struct wt_nl_service service;
    struct commit_context owner = {0};
    struct wt_nl_ticket ticket, previous = {0};
    struct helper_thread helpers[2];
    int fd;
    size_t bytes;
    unsigned iteration, i, ok, calls;

    make_service(directory, "repeat.shm", &fd, &bytes, &service);
    for (iteration = 1; iteration <= 200; ++iteration) {
        ticket = publish(&service, 0, 80, 90, iteration);
        if (iteration > 1)
            CHECK(wt_nl_owner_poison(&service, 0, 0, &previous) == WT_NL_STALE);
        two_thread_try(&service, 1, signature(80, 90), helpers);
        ok = calls = 0;
        for (i = 0; i < 2; ++i) {
            ok += helpers[i].status == WT_NL_OK;
            calls += (unsigned)atomic_load(&helpers[i].callback.calls);
        }
        CHECK(ok == 1 && calls == 1);
        CHECK(wt_nl_owner_collect(&service, 0, 0, &ticket, validate, commit,
                                  &owner) == WT_NL_OK);
        CHECK(wt_nl_finish(&service, 0, 0, &ticket) == WT_NL_OK);
        previous = ticket;
    }
    CHECK(owner.commit_calls == 200);
    wt_nl_detach(&service); close(fd);
}

#define PUBLISHERS 16u
#define PUBLISH_ROUNDS 100u
struct publisher_context {
    struct wt_nl_service *service;
    pthread_barrier_t *barrier;
    unsigned index;
    uint64_t ids[PUBLISH_ROUNDS];
};
static void *publisher_main(void *opaque)
{
    struct publisher_context *p=opaque;
    struct callback_context callback={0};
    struct commit_context owner={0};
    uint32_t rank=p->index/4, thread=p->index%4;
    for (unsigned round=0;round<PUBLISH_ROUNDS;++round) {
        struct wt_nl_descriptor d=descriptor(101,102,round+1);
        struct geometry g={8,4,1,0}; struct request r={round+10,round+20};
        struct wt_nl_ticket ticket;
        int status=pthread_barrier_wait(p->barrier);
        CHECK(status==0 || status==PTHREAD_BARRIER_SERIAL_THREAD);
        /* All publishers own disjoint slots; ID contention must not reject. */
        CHECK(wt_nl_publish(p->service,rank,thread,&d,&g,&r,&ticket)==WT_NL_OK);
        p->ids[round]=ticket.work_id;
        CHECK(wt_nl_owner_execute(p->service,rank,thread,&ticket,execute,&callback,
                                  ((uint64_t)rank<<32)|thread)==WT_NL_OK);
        CHECK(wt_nl_owner_collect(p->service,rank,thread,&ticket,validate,commit,&owner)==WT_NL_OK);
        CHECK(wt_nl_finish(p->service,rank,thread,&ticket)==WT_NL_OK);
    }
    CHECK(owner.commit_calls==PUBLISH_ROUNDS);
    CHECK(atomic_load(&callback.calls)==PUBLISH_ROUNDS);
    return NULL;
}
static void test_publish_contention(void)
{
    struct wt_nl_config config={4,4,PAYLOAD_BYTES};
    struct wt_nl_service service;
    struct publisher_context publishers[PUBLISHERS];
    pthread_t ids[PUBLISHERS]; pthread_barrier_t barrier;
    unsigned char seen[PUBLISHERS*PUBLISH_ROUNDS+1]={0};
    void *mapping=NULL; size_t bytes;
    CHECK(wt_nl_required_bytes(&config,&bytes)==WT_NL_OK);
    CHECK(posix_memalign(&mapping,64,bytes)==0);
    CHECK(wt_nl_init_mapping(mapping,bytes,&config,&service)==WT_NL_OK);
    CHECK(pthread_barrier_init(&barrier,NULL,PUBLISHERS)==0);
    for (unsigned i=0;i<PUBLISHERS;++i) {
        publishers[i]=(struct publisher_context){.service=&service,.barrier=&barrier,.index=i};
        CHECK(pthread_create(&ids[i],NULL,publisher_main,&publishers[i])==0);
    }
    for (unsigned i=0;i<PUBLISHERS;++i) CHECK(pthread_join(ids[i],NULL)==0);
    for (unsigned i=0;i<PUBLISHERS;++i) for (unsigned j=0;j<PUBLISH_ROUNDS;++j) {
        uint64_t id=publishers[i].ids[j];
        CHECK(id>0 && id<=PUBLISHERS*PUBLISH_ROUNDS && !seen[id]); seen[id]=1;
    }
    CHECK(pthread_barrier_destroy(&barrier)==0);
    wt_nl_detach(&service); free(mapping);
    puts("disjoint publisher contention: 1600 unique IDs, exact native/commit counts");
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    test_publish_contention();
    test_mapping_validation(argv[1]);
    test_cross_process(argv[1]);
    test_signature_self_and_stale(argv[1]);
    test_execute_failure(argv[1], CALLBACK_REJECT, "reject.shm", WT_NL_REJECTED);
    test_execute_failure(argv[1], CALLBACK_FATAL, "fatal.shm", WT_NL_FAILED);
    test_execute_failure(argv[1], CALLBACK_SHORT, "short.shm", WT_NL_FAILED);
    test_execute_failure(argv[1], CALLBACK_REQUEST_MUTATION, "request.shm",
                         WT_NL_FAILED);
    test_execute_failure(argv[1], CALLBACK_DESCRIPTOR_MUTATION, "descriptor.shm",
                         WT_NL_FAILED);
    test_execute_failure(argv[1], CALLBACK_GUARD_MUTATION, "guard.shm",
                         WT_NL_FAILED);
    test_poison_running(argv[1]);
    test_repeat_claim(argv[1]);
    puts("node-local service tests passed");
    return 0;
}
