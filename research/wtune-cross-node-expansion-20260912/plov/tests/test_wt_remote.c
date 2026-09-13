#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../wt_remote.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%d: %s\n", __LINE__, #x); return 1; } } while (0)
static int worker_calls;

static struct wt_ledger_contract contract(void) {
  struct wt_ledger_contract c = {0};
  c.run_uuid[0] = 7; c.owner_rank = 901; c.owner_generation = 3;
  memset(c.schema_fingerprint, 0x5a, sizeof(c.schema_fingerprint));
  c.abi = (struct wt_payload_abi){4, 8, 4, WT_FLOAT_IEEE754_BINARY64, 1, 2};
  c.limits = (struct wt_wire_limits){512, 336, 2};
  return c;
}

static int make_request(const struct wt_ledger_contract *c, uint32_t phase,
    uint8_t *out, size_t *out_len) {
  uint8_t payload[] = {9, 6};
  struct wt_wire_frame f = {0};
  f.kind = WT_WIRE_REQUEST; f.id.run_uuid[0] = 7; f.id.epoch = 1;
  f.id.domain = 4; f.id.phase = phase; f.id.call_seq = 2; f.id.owner_rank = 901;
  f.id.owner_generation = 3; f.id.gi1 = 1; f.id.gj1 = 1;
  memcpy(f.schema_fingerprint, c->schema_fingerprint, 32);
  memset(f.request_fingerprint, 0x33, 32); f.abi = c->abi;
  f.payload = payload; f.payload_len = sizeof(payload);
  return wt_wire_encode(&f, &c->limits, out, 512, out_len);
}

static int worker_sum(void *ctx, const uint8_t *request, size_t request_len,
    uint8_t *result, size_t result_capacity, size_t *result_len) {
  const struct wt_ledger_contract *c = ctx; struct wt_wire_frame in, out = {0};
  uint8_t sum[1];
  ++worker_calls;
  if (wt_wire_decode(request, request_len, &c->limits, c->schema_fingerprint, &c->abi, &in) != WT_WIRE_OK) return 1;
  sum[0] = (uint8_t)(in.payload[0] + in.payload[1]);
  out.kind = WT_WIRE_RESULT; out.id = in.id; memcpy(out.schema_fingerprint, in.schema_fingerprint, 32);
  memcpy(out.request_fingerprint, in.request_fingerprint, 32); out.abi = in.abi; out.payload = sum; out.payload_len = 1;
  return wt_wire_encode(&out, &c->limits, result, result_capacity, result_len);
}

static int commit_sum(void *ctx, const struct wt_wire_frame *result) {
  uint8_t *state = ctx; if (result->payload_len != 1) return 1; *state = result->payload[0]; return 0;
}
static int worker_fail(void *ctx, const uint8_t *request, size_t request_len,
    uint8_t *result, size_t result_capacity, size_t *result_len) {
  (void)ctx; (void)request; (void)request_len; (void)result; (void)result_capacity; (void)result_len;
  ++worker_calls; return 1;
}

static struct wt_policy_config policy(const struct wt_ledger_contract *c) {
  struct wt_policy_config p = {0}; p.enabled = 1; p.allowed_scopes = WT_POLICY_SCOPE_SAME_NODE;
  memcpy(p.schema_fingerprint, c->schema_fingerprint, 32); p.minimum_samples = 1; return p;
}

static struct wt_policy_request policy_request(uint32_t phase) {
  struct wt_policy_request r = {0}; r.phase_id = phase; r.origin = (struct wt_policy_topology){901, 77, 2, 1};
  r.scope = WT_POLICY_SCOPE_SAME_NODE; r.required_worker_slots = 1; r.worstcase_bytes = 1024; r.local_ns = 100;
  return r;
}

static struct wt_policy_candidate peer(const struct wt_ledger_contract *c) {
  struct wt_policy_candidate x = {0}; x.peer = (struct wt_policy_peer){17,77,9,1,{0},WT_POLICY_CAP_SAME_NODE,1,1024};
  memcpy(x.peer.schema_fingerprint, c->schema_fingerprint, 32); x.estimate = (struct wt_policy_estimate){1,1,1,1,1,1,10,1,1}; return x;
}

static int test_worker_callback_failure(void) {
  int fds[2], flags, status; pid_t child; uint8_t request[512], scratch[512]; size_t request_len;
  struct wt_ledger_contract c = contract(); CHECK(socketpair(AF_UNIX,SOCK_STREAM,0,fds)==0);
  flags=fcntl(fds[0],F_GETFL); CHECK(flags>=0 && fcntl(fds[0],F_SETFL,flags|O_NONBLOCK)==0);
  flags=fcntl(fds[1],F_GETFL); CHECK(flags>=0 && fcntl(fds[1],F_SETFL,flags|O_NONBLOCK)==0);
  CHECK(make_request(&c,UINT32_C(0xdecafbad),request,&request_len)==WT_WIRE_OK); child=fork(); CHECK(child>=0);
  if (child==0) { struct wt_stream s; struct wt_ledger *l=NULL; struct wt_remote_worker w; close(fds[0]);
    if(wt_stream_init(&s,fds[1],512,1000)||wt_ledger_create(WT_LEDGER_WORKER,&c,2,1024,&l)) _exit(2);
    worker_calls=0; w=(struct wt_remote_worker){&s,l,&c,scratch,sizeof(scratch),0}; status=wt_remote_worker_serve_once(&w,worker_fail,NULL);
    wt_ledger_destroy(&l); close(fds[1]); _exit(status==WT_REMOTE_CALLBACK_FAILED && s.poisoned && worker_calls==1 ? 0 : 3); }
  close(fds[1]); { struct wt_stream s; CHECK(wt_stream_init(&s,fds[0],512,1000)==WT_STREAM_OK); CHECK(wt_stream_send(&s,request,request_len)==WT_STREAM_OK); }
  CHECK(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0); close(fds[0]); return 0;
}

static int test_owner_eof_after_publish(void) {
  int fds[2], flags, status; pid_t child; uint8_t request[512], scratch[512], state=0;
  size_t request_len, received; struct wt_ledger_contract c=contract(); struct wt_stream stream;
  struct wt_ledger *ledger=NULL; struct wt_policy_pool *pool=NULL; struct wt_remote_owner owner;
  struct wt_policy_config p=policy(&c); struct wt_policy_request r=policy_request(UINT32_C(0x0badf00d));
  struct wt_policy_candidate candidate=peer(&c); struct wt_remote_packet packet; struct wt_policy_choice choice;
  struct wt_ledger_ticket duplicate; struct wt_policy_pool_stats stats; enum wt_ledger_state ledger_state;
  CHECK(socketpair(AF_UNIX,SOCK_STREAM,0,fds)==0);
  flags=fcntl(fds[0],F_GETFL); CHECK(flags>=0 && fcntl(fds[0],F_SETFL,flags|O_NONBLOCK)==0);
  flags=fcntl(fds[1],F_GETFL); CHECK(flags>=0 && fcntl(fds[1],F_SETFL,flags|O_NONBLOCK)==0);
  CHECK(make_request(&c,r.phase_id,request,&request_len)==WT_WIRE_OK); child=fork(); CHECK(child>=0);
  if(child==0) { struct wt_stream s; close(fds[0]); if(wt_stream_init(&s,fds[1],512,1000)) _exit(2);
    status=wt_stream_receive(&s,scratch,sizeof(scratch),&received); close(fds[1]); _exit(status==WT_STREAM_OK && received==request_len ? 0 : 3); }
  close(fds[1]); CHECK(wt_stream_init(&stream,fds[0],512,1000)==WT_STREAM_OK);
  CHECK(wt_ledger_create(WT_LEDGER_OWNER,&c,2,2048,&ledger)==WT_LEDGER_OK); { struct wt_policy_pool_config pc={1,1,1024}; CHECK(wt_policy_pool_create(&pc,&pool)==WT_POLICY_OK); }
  owner=(struct wt_remote_owner){&stream,ledger,&c,scratch,sizeof(scratch),0,17,77}; packet=(struct wt_remote_packet){request,request_len,512};
  CHECK(wt_remote_owner_exchange(&owner,&p,&r,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_IO);
  CHECK(state==0 && stream.poisoned && wt_policy_pool_get_stats(pool,&stats)==WT_POLICY_OK && stats.inflight==0 && stats.bytes_inflight==0);
  CHECK(wt_ledger_owner_register_request(ledger,request,request_len,512,&duplicate)==WT_LEDGER_DUPLICATE);
  CHECK(wt_ledger_get_state(ledger,&duplicate,&ledger_state)==WT_LEDGER_OK && ledger_state==WT_LEDGER_FAILED);
  CHECK(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
  wt_policy_pool_destroy(pool); CHECK(wt_ledger_destroy(&ledger)==WT_LEDGER_OK); close(fds[0]); return 0;
}

static int test_owner_rejects_wrong_result_fingerprint(void) {
  int fds[2], flags, status; pid_t child; uint8_t request[512], scratch[512], state=0; size_t request_len;
  struct wt_ledger_contract c=contract(); struct wt_stream stream; struct wt_ledger *ledger=NULL; struct wt_policy_pool *pool=NULL;
  struct wt_remote_owner owner; struct wt_policy_config p=policy(&c); struct wt_policy_request r=policy_request(UINT32_C(0x0f1e2d3c)); struct wt_policy_candidate candidate=peer(&c); struct wt_remote_packet packet; struct wt_policy_choice choice; struct wt_ledger_ticket duplicate; struct wt_policy_pool_stats stats; enum wt_ledger_state ledger_state;
  CHECK(socketpair(AF_UNIX,SOCK_STREAM,0,fds)==0); flags=fcntl(fds[0],F_GETFL); CHECK(flags>=0 && fcntl(fds[0],F_SETFL,flags|O_NONBLOCK)==0); flags=fcntl(fds[1],F_GETFL); CHECK(flags>=0 && fcntl(fds[1],F_SETFL,flags|O_NONBLOCK)==0); CHECK(make_request(&c,r.phase_id,request,&request_len)==WT_WIRE_OK); child=fork(); CHECK(child>=0);
  if(child==0) { struct wt_stream s; struct wt_wire_frame in,out={0}; uint8_t reply[512],payload[1]={15}; size_t got,reply_len; close(fds[0]); if(wt_stream_init(&s,fds[1],512,1000)||wt_stream_receive(&s,scratch,sizeof(scratch),&got)||wt_wire_decode(scratch,got,&c.limits,c.schema_fingerprint,&c.abi,&in)!=WT_WIRE_OK) _exit(2); out.kind=WT_WIRE_RESULT; out.id=in.id; memcpy(out.schema_fingerprint,in.schema_fingerprint,32); memset(out.request_fingerprint,0x44,32); out.abi=in.abi; out.payload=payload; out.payload_len=1; status=wt_wire_encode(&out,&c.limits,reply,sizeof(reply),&reply_len); if(!status) status=wt_stream_send(&s,reply,reply_len); close(fds[1]); _exit(status==WT_STREAM_OK?0:3); }
  close(fds[1]); CHECK(wt_stream_init(&stream,fds[0],512,1000)==WT_STREAM_OK); CHECK(wt_ledger_create(WT_LEDGER_OWNER,&c,2,2048,&ledger)==WT_LEDGER_OK); { struct wt_policy_pool_config pc={1,1,1024}; CHECK(wt_policy_pool_create(&pc,&pool)==WT_POLICY_OK); } owner=(struct wt_remote_owner){&stream,ledger,&c,scratch,sizeof(scratch),0,17,77}; packet=(struct wt_remote_packet){request,request_len,512};
  CHECK(wt_remote_owner_exchange(&owner,&p,&r,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_BAD_PACKET); CHECK(state==0 && stream.poisoned && wt_policy_pool_get_stats(pool,&stats)==WT_POLICY_OK && stats.inflight==0 && stats.bytes_inflight==0); CHECK(wt_ledger_owner_register_request(ledger,request,request_len,512,&duplicate)==WT_LEDGER_DUPLICATE); CHECK(wt_ledger_get_state(ledger,&duplicate,&ledger_state)==WT_LEDGER_OK && ledger_state==WT_LEDGER_FAILED); CHECK(waitpid(child,&status,0)==child && WIFEXITED(status)&&WEXITSTATUS(status)==0); wt_policy_pool_destroy(pool); CHECK(wt_ledger_destroy(&ledger)==WT_LEDGER_OK); close(fds[0]); return 0;
}

int main(void) {
  if (test_owner_rejects_wrong_result_fingerprint()) return 1;
  if (test_owner_eof_after_publish()) return 1;
  if (test_worker_callback_failure()) return 1;
  int fds[2], flags, status; pid_t child; uint8_t request[512], scratch[512], state = 0; size_t request_len;
  struct wt_ledger_contract c = contract(); struct wt_stream owner_stream; struct wt_ledger *owner_ledger = NULL; struct wt_policy_pool *pool = NULL;
  struct wt_remote_owner owner; struct wt_remote_packet packet; struct wt_policy_config p = policy(&c); struct wt_policy_request pr = policy_request(UINT32_C(0xa1b2c3d4)); struct wt_policy_candidate candidate = peer(&c); struct wt_policy_choice choice;
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
  flags = fcntl(fds[0], F_GETFL); CHECK(flags >= 0 && fcntl(fds[0], F_SETFL, flags | O_NONBLOCK) == 0);
  flags = fcntl(fds[1], F_GETFL); CHECK(flags >= 0 && fcntl(fds[1], F_SETFL, flags | O_NONBLOCK) == 0);
  CHECK(make_request(&c, pr.phase_id, request, &request_len) == WT_WIRE_OK);
  child = fork(); CHECK(child >= 0);
  if (child == 0) { struct wt_stream s; struct wt_ledger *l = NULL; struct wt_remote_worker w;
    close(fds[0]); if (wt_stream_init(&s, fds[1], 512, 1000) || wt_ledger_create(WT_LEDGER_WORKER,&c,4,2048,&l)) _exit(2);
    w = (struct wt_remote_worker){&s,l,&c,scratch,sizeof(scratch),0}; status = wt_remote_worker_serve_once(&w, worker_sum, &c);
    if (status == WT_REMOTE_OK) status = wt_remote_worker_serve_once(&w, worker_sum, &c);
    if (status == WT_REMOTE_OK) status = wt_remote_worker_serve_once(&w, worker_sum, &c);
    wt_ledger_destroy(&l); close(fds[1]); _exit(status == WT_REMOTE_OK && worker_calls == 2 ? 0 : 3); }
  close(fds[1]); CHECK(wt_stream_init(&owner_stream, fds[0], 512, 1000) == WT_STREAM_OK); CHECK(wt_ledger_create(WT_LEDGER_OWNER,&c,4,2048,&owner_ledger) == WT_LEDGER_OK); { struct wt_policy_pool_config pc={1,1,1024}; CHECK(wt_policy_pool_create(&pc,&pool)==WT_POLICY_OK); }
  owner=(struct wt_remote_owner){&owner_stream,owner_ledger,&c,scratch,sizeof(scratch),0,17,77}; packet=(struct wt_remote_packet){request,request_len,512};
  CHECK(wt_remote_owner_exchange(&owner,&p,&pr,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_OK); CHECK(state==15);
  pr = policy_request(UINT32_C(0x10203040));
  CHECK(make_request(&c, pr.phase_id, request, &request_len) == WT_WIRE_OK);
  packet = (struct wt_remote_packet){request,request_len,512};
  CHECK(wt_remote_owner_exchange(&owner,&p,&pr,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_OK); CHECK(state==15);
  CHECK(wt_stream_send(&owner_stream, request, request_len) == WT_STREAM_OK);
  { size_t duplicate_len; CHECK(wt_stream_receive(&owner_stream,scratch,sizeof(scratch),&duplicate_len)==WT_STREAM_OK && duplicate_len > WT_WIRE_HEADER_BYTES); }
  CHECK(wt_remote_owner_exchange(&owner,&p,&pr,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_DUPLICATE);
  CHECK(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
  {
    int eof_fds[2]; int eof_flags; struct wt_stream eof_stream;
    struct wt_ledger *eof_ledger = NULL; struct wt_remote_worker eof_worker;
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, eof_fds) == 0);
    eof_flags = fcntl(eof_fds[0], F_GETFL);
    CHECK(eof_flags >= 0 && fcntl(eof_fds[0], F_SETFL, eof_flags | O_NONBLOCK) == 0);
    close(eof_fds[1]);
    CHECK(wt_stream_init(&eof_stream, eof_fds[0], 512, 1000) == WT_STREAM_OK);
    CHECK(wt_ledger_create(WT_LEDGER_WORKER, &c, 2, 1024, &eof_ledger) == WT_LEDGER_OK);
    eof_worker = (struct wt_remote_worker){&eof_stream,eof_ledger,&c,scratch,sizeof(scratch),0};
    CHECK(wt_remote_worker_serve_once(&eof_worker, worker_sum, &c) == WT_REMOTE_IO);
    CHECK(wt_ledger_destroy(&eof_ledger) == WT_LEDGER_OK); close(eof_fds[0]);
  }
  {
    struct wt_policy_candidate fastest[2] = {candidate, candidate};
    struct wt_policy_pool_stats pool_stats;
    fastest[1].peer.rank_id = 99; fastest[1].estimate.remote_exec_ns = 0;
    pr = policy_request(UINT32_C(0x55667788)); state = 0;
    CHECK(make_request(&c, pr.phase_id, request, &request_len) == WT_WIRE_OK);
    packet = (struct wt_remote_packet){request,request_len,512};
    CHECK(wt_remote_owner_exchange(&owner,&p,&pr,fastest,2,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_ENDPOINT_MISMATCH);
    CHECK(state == 0 && wt_policy_pool_get_stats(pool,&pool_stats)==WT_POLICY_OK &&
        pool_stats.inflight == 0 && pool_stats.bytes_inflight == 0);
    pr = policy_request(UINT32_C(0x12345678));
    CHECK(make_request(&c, pr.phase_id, request, &request_len) == WT_WIRE_OK);
    packet = (struct wt_remote_packet){request,request_len,512};
    pr.origin.rank_id = 1;
    CHECK(wt_remote_owner_exchange(&owner,&p,&pr,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_BAD_PACKET);
    pr = policy_request(UINT32_C(0x12345678)); p.schema_fingerprint[0] ^= 1;
    CHECK(wt_remote_owner_exchange(&owner,&p,&pr,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_BAD_PACKET);
    p = policy(&c); pr.worstcase_bytes = request_len + 511;
    CHECK(wt_remote_owner_exchange(&owner,&p,&pr,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_BAD_PACKET);
    pr = policy_request(UINT32_C(0x12345678)); owner.receive_capacity = 511;
    CHECK(wt_remote_owner_exchange(&owner,&p,&pr,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_BAD_PACKET);
    owner.receive_capacity = sizeof(scratch);
    CHECK(state == 0 && wt_policy_pool_get_stats(pool,&pool_stats)==WT_POLICY_OK &&
        pool_stats.inflight == 0 && pool_stats.bytes_inflight == 0);
  }
  p.enabled=0; state=0; CHECK(wt_remote_owner_exchange(&owner,&p,&pr,&candidate,1,pool,&packet,commit_sum,&state,&choice)==WT_REMOTE_POLICY_REFUSED && state==0);
  wt_policy_pool_destroy(pool); wt_ledger_destroy(&owner_ledger); close(fds[0]); return 0;
}
