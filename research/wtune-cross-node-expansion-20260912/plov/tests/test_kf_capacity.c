#include "wt_kf_capacity.h"
#include "wt_kf_cps_schema.h"
#include "wt_live_spans.h"
#include <assert.h>
#include <stdint.h>
#include <string.h>

#define STORE 128u
struct fixture {
  void *argv[WT_KF_NARG]; int32_t scalar[WT_KF_NARG];
  unsigned char store[WT_KF_NARG][STORE]; wt_live_spans *spans;
  uint64_t generation; struct wt_kf_capacity provider;
};
static int type_of(unsigned i) {
  /* Independent audit oracle: kf_cps_schema real_2d lists NCA and SHALL;
     logical_2d lists only CU_ACT_FLAG. */
  return i == WT_KF_CU_ACT_FLAG ? WT_SPAN_LOGICAL : WT_SPAN_REAL;
}
static size_t bytes_of(unsigned i) { return wt_kf_shape[i] == WT_KF_3D ? 48 : 24; }
static int automatic_proof(unsigned i) {
  return i == WT_KF_TH || i == WT_KF_PI || i == WT_KF_DZ8W ||
    i == WT_KF_CU_ACT_FLAG;
}
static void init(struct fixture *f, int active)
{
  unsigned i; memset(f, 0, sizeof *f);
  for (i=0;i<WT_KF_NARG;i++) f->argv[i] = wt_kf_shape[i] == WT_KF_SCALAR ?
    (void *)&f->scalar[i] : f->store[i];
  f->scalar[WT_KF_IMS]=-1; f->scalar[WT_KF_IME]=1;
  f->scalar[WT_KF_JMS]=2; f->scalar[WT_KF_JME]=3;
  f->scalar[WT_KF_KMS]=1; f->scalar[WT_KF_KME]=2;
  f->scalar[WT_KF_KF_EDRATES]=active ? 1 : 0;
  f->spans=wt_live_spans_create(WT_KF_NARG); assert(f->spans);
  f->generation=wt_live_spans_begin(f->spans); assert(f->generation);
  for (i=0;i<WT_KF_NARG;i++) if (wt_kf_shape[i] != WT_KF_SCALAR &&
      (active || i < WT_KF_UDR_KF || i > WT_KF_TIMEC_KF))
    assert(wt_live_spans_add(f->spans, f->argv[i], bytes_of(i),type_of(i),4,
      wt_kf_shape[i]) == WT_SPAN_OK);
  assert(wt_live_spans_finish(f->spans)==WT_SPAN_OK);
}
static void done(struct fixture *f) { wt_live_spans_destroy(f->spans); }
static void checked_init(struct fixture *f, int authorized) {
  assert(wt_kf_capacity_init(&f->provider, f->argv, f->spans, f->generation,
    authorized));
}
static void omit_automatic_proof(struct fixture *f)
{
  unsigned i; int active = f->scalar[WT_KF_KF_EDRATES] == 1;
  wt_live_spans_end(f->spans); f->generation=wt_live_spans_begin(f->spans);
  for (i=0;i<WT_KF_NARG;i++) if (wt_kf_shape[i]!=WT_KF_SCALAR &&
      !automatic_proof(i) && (active || i<WT_KF_UDR_KF || i>WT_KF_TIMEC_KF))
    assert(wt_live_spans_add(f->spans,f->argv[i],bytes_of(i),type_of(i),4,
      wt_kf_shape[i])==WT_SPAN_OK);
  assert(wt_live_spans_finish(f->spans)==WT_SPAN_OK);
}
static void test_current_and_interior(void)
{
  struct fixture f; size_t bytes=0; init(&f,0);
  /* Registered span can begin before the exact current interior pointer. */
  f.argv[WT_KF_RAINCV]=(unsigned char *)f.argv[WT_KF_RAINCV]+4;
  wt_live_spans_end(f.spans); f.generation=wt_live_spans_begin(f.spans);
  assert(wt_live_spans_add(f.spans, f.store[WT_KF_RAINCV], STORE, WT_SPAN_REAL,4,2)==WT_SPAN_OK);
  /* Rebuild all other live records after begin. */
  for (unsigned i=0;i<WT_KF_NARG;i++) if (wt_kf_shape[i]!=WT_KF_SCALAR && i!=WT_KF_RAINCV &&
      (i<WT_KF_UDR_KF || i>WT_KF_TIMEC_KF)) assert(wt_live_spans_add(f.spans,f.argv[i],bytes_of(i),type_of(i),4,wt_kf_shape[i])==WT_SPAN_OK);
  assert(wt_live_spans_finish(f.spans)==WT_SPAN_OK); checked_init(&f,1);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_DT,f.argv[WT_KF_DT],&bytes)&&bytes==4);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_PRATEC,f.argv[WT_KF_PRATEC],&bytes)&&bytes==24);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_RHO,f.argv[WT_KF_RHO],&bytes)&&bytes==48);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_RAINCV,f.argv[WT_KF_RAINCV],&bytes)&&bytes==24);
  assert(!wt_kf_capacity_lookup(&f.provider,WT_KF_RAINCV,f.store[WT_KF_RAINCV],&bytes));
  wt_live_spans_end(f.spans);
  assert(!wt_kf_capacity_lookup(&f.provider,WT_KF_DT,f.argv[WT_KF_DT],&bytes)); done(&f);
}
static void test_refusals(void)
{
  struct fixture f; size_t bytes=77; init(&f,0);
  assert(!wt_kf_capacity_init(&f.provider,f.argv,f.spans,f.generation,0));
  assert(!f.provider.ready);
  checked_init(&f,1); f.argv[WT_KF_DT]=0;
  assert(!wt_kf_capacity_init(&f.provider,f.argv,f.spans,f.generation,1));
  assert(!f.provider.ready && !f.provider.expected[WT_KF_IDS]);
  f.scalar[WT_KF_IME]=f.scalar[WT_KF_IMS]-1;
  assert(!wt_kf_capacity_init(&f.provider,f.argv,f.spans,f.generation,1)); done(&f);
  init(&f,0); f.argv[WT_KF_SHALL]=0; checked_init(&f,1);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_SHALL,0,&bytes)&&bytes==0); done(&f);
  init(&f,0); f.argv[WT_KF_RHO]=(void *)(uintptr_t)1;
  assert(!wt_kf_capacity_init(&f.provider,f.argv,f.spans,f.generation,1));
  assert(!f.provider.ready); done(&f);
  init(&f,0); f.scalar[WT_KF_IMS]=INT32_MIN; f.scalar[WT_KF_IME]=INT32_MAX;
  f.scalar[WT_KF_JMS]=INT32_MIN; f.scalar[WT_KF_JME]=INT32_MAX;
  f.scalar[WT_KF_KMS]=INT32_MIN; f.scalar[WT_KF_KME]=INT32_MAX;
  assert(!wt_kf_capacity_init(&f.provider,f.argv,f.spans,f.generation,1)); done(&f);
}
static void test_inactive_and_active(void)
{
  struct fixture f; size_t bytes; init(&f,0); checked_init(&f,1);
  for (unsigned i=WT_KF_UDR_KF;i<=WT_KF_TIMEC_KF;i++) {
    assert(wt_kf_capacity_lookup(&f.provider,i,f.argv[i],&bytes)&&bytes==4);
  }
  done(&f);
  init(&f,1); checked_init(&f,1);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_UDR_KF,f.argv[WT_KF_UDR_KF],&bytes)&&bytes==48); done(&f);
  init(&f,1); wt_live_spans_end(f.spans); f.generation=wt_live_spans_begin(f.spans);
  assert(wt_live_spans_finish(f.spans)==WT_SPAN_OK);
  assert(!wt_kf_capacity_init(&f.provider,f.argv,f.spans,f.generation,1)); done(&f);
}
static void test_schema_types(void)
{
  struct fixture f; init(&f, 0);
  assert(type_of(WT_KF_NCA) == WT_SPAN_REAL);
  assert(type_of(WT_KF_SHALL) == WT_SPAN_REAL);
  assert(type_of(WT_KF_CU_ACT_FLAG) == WT_SPAN_LOGICAL);
  checked_init(&f, 1); done(&f);
  init(&f, 0); wt_live_spans_end(f.spans); f.generation=wt_live_spans_begin(f.spans);
  for (unsigned i=0;i<WT_KF_NARG;i++) if (wt_kf_shape[i]!=WT_KF_SCALAR &&
      (i<WT_KF_UDR_KF || i>WT_KF_TIMEC_KF)) {
    int wrong = i == WT_KF_NCA || i == WT_KF_SHALL ? WT_SPAN_LOGICAL : type_of(i);
    assert(wt_live_spans_add(f.spans,f.argv[i],bytes_of(i),wrong,4,wt_kf_shape[i])==WT_SPAN_OK);
  }
  assert(wt_live_spans_finish(f.spans)==WT_SPAN_OK);
  assert(!wt_kf_capacity_init(&f.provider,f.argv,f.spans,f.generation,1)); done(&f);
}
static void test_attested_automatic_proof(void)
{
  struct fixture f; size_t bytes;
  assert(sizeof(float)==4 && sizeof(int32_t)==4 && sizeof(uint32_t)==4);
  init(&f,0); omit_automatic_proof(&f);
  assert(!wt_kf_capacity_init(&f.provider,f.argv,f.spans,f.generation,1));
  assert(wt_kf_capacity_init_attested(&f.provider,f.argv,f.spans,f.generation,1,1));
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_TH,f.argv[WT_KF_TH],&bytes)&&bytes==48);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_PI,f.argv[WT_KF_PI],&bytes)&&bytes==48);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_DZ8W,f.argv[WT_KF_DZ8W],&bytes)&&bytes==48);
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_CU_ACT_FLAG,f.argv[WT_KF_CU_ACT_FLAG],&bytes)&&bytes==24);
  wt_live_spans_end(f.spans);
  assert(!wt_kf_capacity_lookup(&f.provider,WT_KF_TH,f.argv[WT_KF_TH],&bytes));
  done(&f);
  init(&f,1); omit_automatic_proof(&f);
  assert(wt_kf_capacity_init_attested(&f.provider,f.argv,f.spans,f.generation,1,1));
  assert(wt_kf_capacity_lookup(&f.provider,WT_KF_UDR_KF,f.argv[WT_KF_UDR_KF],&bytes)&&bytes==48);
  done(&f);
  init(&f,0); omit_automatic_proof(&f); f.scalar[WT_KF_IME]=f.scalar[WT_KF_IMS]-1;
  assert(!wt_kf_capacity_init_attested(&f.provider,f.argv,f.spans,f.generation,1,1)); done(&f);
  init(&f,0); omit_automatic_proof(&f); f.scalar[WT_KF_IMS]=INT32_MIN; f.scalar[WT_KF_IME]=INT32_MAX;
  f.scalar[WT_KF_JMS]=INT32_MIN; f.scalar[WT_KF_JME]=INT32_MAX;
  f.scalar[WT_KF_KMS]=INT32_MIN; f.scalar[WT_KF_KME]=INT32_MAX;
  assert(!wt_kf_capacity_init_attested(&f.provider,f.argv,f.spans,f.generation,1,1)); done(&f);
}
int main(void) { test_current_and_interior(); test_refusals(); test_inactive_and_active(); test_schema_types(); test_attested_automatic_proof(); return 0; }
