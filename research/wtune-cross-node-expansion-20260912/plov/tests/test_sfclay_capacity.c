#include "wt_sfclay_capacity.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NI = 2, NJ = 2, NK = 2, N2 = NI*NJ*4, N3 = NI*NJ*NK*4 };
struct fixture { void *argv[WT_SFCLAY_NARG], *base[WT_SFCLAY_NARG]; size_t bytes[WT_SFCLAY_NARG]; };
static void fail(const char *s) { fputs(s, stderr); exit(2); }
static void yes(int value, const char *s) { if (!value) fail(s); }
static void put_i(void *p, int32_t n) { memcpy(p, &n, 4); }
static void *const *unreadable_argv(void) { volatile uintptr_t address=1; return (void *const *)address; }
static size_t need(unsigned i) { return wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_3D ? N3 : wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_2D ? N2 : 4; }
static void make(struct fixture *f) {
  unsigned i; memset(f,0,sizeof *f);
  for (i=0;i<WT_SFCLAY_NARG;i++) { f->bytes[i]=need(i); f->base[i]=calloc(1,f->bytes[i]+16); yes(f->base[i] != NULL,"allocation\n"); f->argv[i]=(char *)f->base[i]+16; }
  put_i(f->argv[64],-1); put_i(f->argv[65],0); put_i(f->argv[66],4); put_i(f->argv[67],5); put_i(f->argv[68],1); put_i(f->argv[69],2);
  for (i=81;i<84;i++) { free(f->base[i]); f->base[i]=f->argv[i]=NULL; f->bytes[i]=0; }
}
static void drop(struct fixture *f) { unsigned i; for(i=0;i<WT_SFCLAY_NARG;i++) free(f->base[i]); }
static uint64_t captured(const struct fixture *f, wt_live_spans *s, int omit, int wrong, int short_span) {
  unsigned i; uint64_t token=wt_live_spans_begin(s); if (!token) return 0;
  for(i=0;i<WT_SFCLAY_NARG;i++) if (wt_sfclay_schema_shape[i] != WT_SFCLAY_SHAPE_SCALAR && f->argv[i] && (int)i != omit) {
    size_t n=f->bytes[i]+16; if ((int)i==short_span) n -= 4;
    if (wt_live_spans_add(s,f->base[i],n,(int)i==wrong?WT_SPAN_INTEGER:WT_SPAN_REAL,4,wt_sfclay_schema_shape[i]) != WT_SPAN_OK) return 0;
  }
  return wt_live_spans_finish(s) == WT_SPAN_OK ? token : 0;
}
static wt_live_spans *spans(void) { wt_live_spans *s=wt_live_spans_create(128); yes(s != NULL,"spans\n"); return s; }
static void valid_and_stale(void) {
  struct fixture f; struct wt_sfclay_capacity p; wt_live_spans *s=spans(); uint64_t g; size_t n;
  make(&f); g=captured(&f,s,-1,-1,-1); yes(g,"capture\n");
  yes(!wt_sfclay_capacity_init(&p,unreadable_argv(),s,g,0),"source gate before argv read\n");
  yes(!wt_sfclay_capacity_init(&p,f.argv,s,g,0),"source abi gate\n");
  yes(wt_sfclay_capacity_init(&p,f.argv,s,g,1),"valid strict\n");
  yes(wt_sfclay_capacity_lookup(&p,3,f.argv[3],&n)&&n==N3,"interior QV\n");
  yes(wt_sfclay_capacity_lookup(&p,6,f.argv[6],&n)&&n==4,"scalar\n");
  yes(wt_sfclay_capacity_lookup(&p,81,NULL,&n)&&n==0,"null optional\n");
  yes(!wt_sfclay_capacity_lookup(&p,3,(char *)f.argv[3]+4,&n),"wrong identity\n");
  wt_live_spans_end(s); yes(!wt_sfclay_capacity_lookup(&p,6,f.argv[6],&n),"stale scalar\n");
  wt_live_spans_destroy(s); drop(&f);
}
static void rejected_spans_and_bounds(void) {
  struct fixture f; struct wt_sfclay_capacity p; wt_live_spans *s; uint64_t g;
  make(&f); s=spans(); g=captured(&f,s,-1,3,-1); yes(g&&!wt_sfclay_capacity_init(&p,f.argv,s,g,1),"wrong typed span\n"); wt_live_spans_destroy(s);
  s=spans(); g=captured(&f,s,-1,-1,3); yes(g&&!wt_sfclay_capacity_init(&p,f.argv,s,g,1),"short QV\n"); wt_live_spans_destroy(s);
  s=spans(); g=captured(&f,s,-1,-1,-1); put_i(f.argv[64],INT32_MIN); put_i(f.argv[65],INT32_MAX);
  put_i(f.argv[66],INT32_MIN); put_i(f.argv[67],INT32_MAX); put_i(f.argv[68],INT32_MIN); put_i(f.argv[69],INT32_MAX);
  yes(g&&!wt_sfclay_capacity_init(&p,f.argv,s,g,1),"dimension overflow\n"); wt_live_spans_destroy(s); drop(&f);
}
static void caller_chain_whitelist(void) {
  struct fixture f; struct wt_sfclay_capacity p; wt_live_spans *s; uint64_t g; size_t n;
  make(&f); s=spans(); g=captured(&f,s,0,-1,-1); yes(g&&!wt_sfclay_capacity_init(&p,f.argv,s,g,1),"strict needs U3D span\n");
  yes(wt_sfclay_capacity_init_attested(&p,f.argv,s,g,1,1),"attested U3D whitelist\n");
  yes(wt_sfclay_capacity_lookup(&p,0,f.argv[0],&n)&&n==N3,"whitelist capacity\n"); wt_live_spans_destroy(s);
  s=spans(); g=captured(&f,s,3,-1,-1); yes(g&&!wt_sfclay_capacity_init_attested(&p,f.argv,s,g,1,1),"QV remains registry-only\n");
  wt_live_spans_destroy(s); drop(&f);
}
int main(void) { valid_and_stale(); rejected_spans_and_bounds(); caller_chain_whitelist(); puts("SFCLAY capacity checks passed"); return 0; }
