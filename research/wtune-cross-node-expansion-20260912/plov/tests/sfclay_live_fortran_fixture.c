#include "wt_sfclay_live.h"
#include <pthread.h>
#include "wt_sfclay_schema.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void wt_sfclay_bridge_real(void **);
extern void wt_sfclay_bridge_call(void **);
extern size_t wt_sfclay_bridge_wrapped_calls(void);
extern void __module_sf_sfclay_MOD_sfclayinit(int *);

enum { N3 = 32, N2 = 16 };
struct context { void *arg[WT_SFCLAY_NARG], *initial[WT_SFCLAY_NARG]; size_t cap[WT_SFCLAY_NARG]; };
static unsigned initializer_dispatches, initializer_calls, captures, queries;
static int *expected_logical;
static void (*native_initializer)(int *);
static int mock_config;
extern void __real_wt_sfclay_init_dispatch(int *, void (*)(int *));

static void die(const char *message) { fputs(message, stderr); exit(2); }
static void check(int condition, const char *message) { if (!condition) die(message); }
static int slot(const char *name) {
  int i; for (i = 0; i < WT_SFCLAY_NARG; ++i) if (!strcmp(name, wt_sfclay_schema_name[i])) return i;
  die("schema name missing\n"); return -1;
}
static float *fp(struct context *c, const char *name) { return c->arg[slot(name)]; }
static int *ip(struct context *c, const char *name) { return c->arg[slot(name)]; }
static void setf(struct context *c, const char *name, float value) {
  size_t i, n = c->cap[slot(name)] / sizeof(float); float *p = fp(c, name);
  for (i = 0; i < n; ++i) p[i] = value;
}
static void seti(struct context *c, const char *name, int value) { *ip(c, name) = value; }
static size_t bytes_for(int i) {
  return wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_3D ? N3 * sizeof(float) :
         wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_2D ? N2 * sizeof(float) : sizeof(int);
}
static void init_context(struct context *c, int stable, int water, int isftc, int iz0, int scm, int controls) {
  int i;
  memset(c, 0, sizeof *c);
  for (i = 0; i < WT_SFCLAY_NARG; ++i) {
    c->cap[i] = bytes_for(i); c->arg[i] = calloc(1, c->cap[i]); check(c->arg[i] != NULL, "allocation\n");
    c->initial[i] = malloc(c->cap[i]); check(c->initial[i] != NULL, "snapshot allocation\n");
    if (wt_sfclay_schema_shape[i] != WT_SFCLAY_SHAPE_SCALAR) setf(c, wt_sfclay_schema_name[i], 0.2f);
  }
  setf(c,"U3D",5.f); setf(c,"V3D",1.f); setf(c,"T3D",300.f); setf(c,"QV3D",.010f);
  setf(c,"P3D",90000.f); setf(c,"dz8w",50.f); setf(c,"PSFC",90000.f); setf(c,"PBLH",500.f);
  setf(c,"MAVAIL",.8f); setf(c,"ZNT",.05f); setf(c,"UST",.30f); setf(c,"TSK",stable ? 290.f : 310.f);
  setf(c,"XLAND",water ? 2.f : 1.f); setf(c,"DX",1000.f); setf(c,"CHS",.01f); setf(c,"CHS2",.01f);
  setf(c,"CQS2",.01f); setf(c,"CPM",1004.f); setf(c,"QSFC",.01f); setf(c,"QGH",.01f);
  setf(c,"CP",1004.f); setf(c,"G",9.81f); setf(c,"ROVCP",.286f); setf(c,"R",287.f); setf(c,"XLV",2.5e6f);
  setf(c,"SVP1",.6112f); setf(c,"SVP2",17.67f); setf(c,"SVP3",29.65f); setf(c,"SVPT0",273.15f);
  setf(c,"EP1",.608f); setf(c,"EP2",.622f); setf(c,"KARMAN",.4f); setf(c,"EOMEG",7.292e-5f);
  setf(c,"STBOLT",5.67e-8f); setf(c,"P1000mb",100000.f);
  seti(c,"ISFFLX",1); seti(c,"ids",-1); seti(c,"ide",2); seti(c,"jds",3); seti(c,"jde",4);
  seti(c,"kds",1); seti(c,"kde",2); seti(c,"ims",-1); seti(c,"ime",2); seti(c,"jms",3); seti(c,"jme",4);
  seti(c,"kms",1); seti(c,"kme",2); seti(c,"its",0); seti(c,"ite",0); seti(c,"jts",3); seti(c,"jte",3);
  seti(c,"kts",2); seti(c,"kte",2); seti(c,"ISFTCFLX",isftc); seti(c,"IZ0TLND",iz0); seti(c,"SCM_FORCE_FLUX",scm);
  if (!controls) for (i = 81; i < 84; ++i) { free(c->arg[i]); free(c->initial[i]); c->arg[i]=c->initial[i]=NULL; c->cap[i]=0; }
  for (i = 0; i < WT_SFCLAY_NARG; ++i) if (c->arg[i]) memcpy(c->initial[i],c->arg[i],c->cap[i]);
}
static void destroy_context(struct context *c) { int i; for (i=0;i<WT_SFCLAY_NARG;++i) { free(c->arg[i]); free(c->initial[i]); } }
/* Instrument the production init wrapper, retaining live lease and real Fortran. */
static void observe_initializer(int *allowed) {
  int old;
  check(allowed == expected_logical, "initializer pointer changed\n");
  check(!pthread_setcancelstate(PTHREAD_CANCEL_DISABLE,&old), "cancel query\n");
  check(old == PTHREAD_CANCEL_DISABLE, "initializer cancellation enabled\n");
  ++initializer_calls;
  native_initializer(allowed);
}
void __wrap_wt_sfclay_init_dispatch(int *allowed, void (*real)(int *)) {
  ++initializer_dispatches;
  check(allowed == expected_logical, "init wrapper pointer changed\n");
  native_initializer=real;
  __real_wt_sfclay_init_dispatch(allowed,observe_initializer);
}
/* Mock grid/config metadata: no native WRF descriptor or caller-chain ABI claim. */
static int capture(void *grid, wt_live_spans *spans, size_t max_nodes, int *tiles) {
  struct context *c=grid;
  check(max_nodes==128, "capture limit\n");
  ++captures; *tiles=1;
  for (int i=0;i<WT_SFCLAY_NARG;++i) {
    if (wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_SCALAR) continue;
    int rank=wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_3D?3:2;
    if (wt_live_spans_add(spans,c->arg[i],c->cap[i],WT_SPAN_REAL,4,rank)!=WT_SPAN_OK) return 0;
  }
  return 1;
}
static int query(void *grid, void *cfg, struct wt_sfclay_native_result *out) {
  check(grid && cfg==&mock_config, "mock native query inputs\n");
  ++queries;
  *out=(struct wt_sfclay_native_result){1,7,0,7,WT_SFCLAY_NATIVE_OK};
  return 1;
}
static int serial(void) { return 0; }
static void live_call(struct context *c) {
  uint64_t outer=wt_sfclay_first_rk_begin(c,&mock_config);
  uint64_t surface=wt_sfclay_surface_begin();
  wt_sfclay_bridge_call(c->arg);
  wt_sfclay_surface_end(surface);
  wt_sfclay_first_rk_end(outer);
}
static void tile(struct context *c, int i, int j) { seti(c,"its",i); seti(c,"ite",i); seti(c,"jts",j); seti(c,"jte",j); }
static void call_one_tile(struct context *c, int staged) {
  unsigned char scalar_before[WT_SFCLAY_NARG][sizeof(int)];
  for (int i=0;i<WT_SFCLAY_NARG;++i)
    if (c->arg[i] && wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_SCALAR)
      memcpy(scalar_before[i],c->arg[i],sizeof(int));
  if (staged) live_call(c); else wt_sfclay_bridge_real(c->arg);
  for (int i=0;i<WT_SFCLAY_NARG;++i)
    if (c->arg[i] && wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_SCALAR &&
        wt_sfclay_schema_access[i]==WT_SFCLAY_READONLY)
      check(!memcmp(scalar_before[i],c->arg[i],sizeof(int)), "readonly scalar changed\n");
}
/* Each call is a separate admitted num_tiles=1 scope, with disjoint tile cells. */
static void call_two_tiles(struct context *c, int staged) {
  tile(c,0,3); call_one_tile(c,staged);
  tile(c,1,4); call_one_tile(c,staged);
}
static void finite_writable(const struct context *c) {
  int i; size_t j;
  for (i=0;i<WT_SFCLAY_NARG;++i) if (c->arg[i] && wt_sfclay_schema_access[i] == WT_SFCLAY_WRITABLE &&
      wt_sfclay_schema_shape[i] != WT_SFCLAY_SHAPE_SCALAR)
    for (j=0;j<c->cap[i]/sizeof(float);++j) check(isfinite(((float *)c->arg[i])[j]),"nonfinite writable\n");
}
static void run_case(int stable, int water, int isftc, int iz0, int scm, int controls) {
  struct context direct, staged; int i; size_t before = wt_sfclay_bridge_wrapped_calls();
  struct wt_sfclay_live_counters before_live, after_live;
  wt_sfclay_live_get_counters(&before_live);
  init_context(&direct,stable,water,isftc,iz0,scm,controls); init_context(&staged,stable,water,isftc,iz0,scm,controls);
  call_two_tiles(&direct,0); call_two_tiles(&staged,1);
  wt_sfclay_live_get_counters(&after_live);
  check(after_live.staged==before_live.staged+2 && after_live.requested==before_live.requested+2 && !after_live.fallback, "live stage counters\n");
  check(wt_sfclay_bridge_wrapped_calls() == before + 2, "wrap composition\n");
  for (i=0;i<WT_SFCLAY_NARG;++i) if (direct.arg[i]) {
    check(!memcmp(direct.arg[i],staged.arg[i],direct.cap[i]),"direct/staged identity\n");
    if (wt_sfclay_schema_access[i] == WT_SFCLAY_READONLY && wt_sfclay_schema_shape[i] != WT_SFCLAY_SHAPE_SCALAR)
      check(!memcmp(direct.arg[i],direct.initial[i],direct.cap[i]),"readonly changed\n");
    if (wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_2D && wt_sfclay_schema_access[i] == WT_SFCLAY_WRITABLE) {
      size_t j; for (j=0;j<direct.cap[i]/sizeof(float);++j) if (j != 1 && j != 6)
        check(!memcmp((float *)direct.arg[i]+j,(float *)direct.initial[i]+j,sizeof(float)),"outside tile changed\n");
    }
  }
  check(memcmp(fp(&direct,"CHS") + 1, (float *)direct.initial[slot("CHS")] + 1, sizeof(float)),"no writable mechanism\n");
  finite_writable(&direct); finite_writable(&staged); destroy_context(&direct); destroy_context(&staged);
}
int main(void) {
  int true_value=1, stable, water, isftc, iz0, scm;
  struct wt_sfclay_live_config config={1,1,1,1,128,128,1u<<20,capture,serial,query};
  struct wt_sfclay_live_counters counters;
  expected_logical=&true_value;
  __module_sf_sfclay_MOD_sfclayinit(&true_value);
  check(initializer_calls==1 && initializer_dispatches==1, "initializer exactly once before configure\n");
  check(wt_sfclay_live_configure(&config), "configure\n");
  for (stable=0;stable<2;++stable) for (water=0;water<2;++water) for (isftc=0;isftc<3;++isftc)
    for (iz0=0;iz0<3;++iz0) for (scm=0;scm<2;++scm) run_case(stable,water,isftc,iz0,scm,1);
  run_case(0,0,0,0,0,0);
  wt_sfclay_live_get_counters(&counters);
  check(counters.staged==146 && counters.requested==146 && !counters.fallback, "146 live stages\n");
  check(captures==146 && queries==438 && initializer_calls==1 && initializer_dispatches==1, "callback counts\n");
  puts("SFCLAY live Fortran: 73 cases, 146 stages, 1 initializer, bitwise backing arrays PASS"); return 0;
}

