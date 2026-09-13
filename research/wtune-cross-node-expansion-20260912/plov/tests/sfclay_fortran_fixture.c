#include "wt_sfclay.h"
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
static const size_t *dispatch_cap;
static unsigned dispatches;

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
static void dispatch(void **argv, size_t nargs, void (*real)(void **)) {
  wt_sfclay_tx *tx = NULL;
  check(nargs == WT_SFCLAY_NARG, "wrapped nargs\n");
  check(wt_sfclay_prepare(argv, dispatch_cap, 1u << 20, &tx) == WT_SFCLAY_OK, "wrapped prepare\n");
  ++dispatches; real(wt_sfclay_cloned_argv(tx));
  check(wt_sfclay_commit(tx) == WT_SFCLAY_OK, "wrapped commit\n");
  wt_sfclay_destroy(tx);
}
void wt_sfclay_bridge_dispatch(void **argv, size_t nargs, void (*real)(void **)) { dispatch(argv,nargs,real); }
static void tile(struct context *c, int i, int j) { seti(c,"its",i); seti(c,"ite",i); seti(c,"jts",j); seti(c,"jte",j); }
static void call_two_tiles(struct context *c, int staged) {
  tile(c,0,3); if (staged) wt_sfclay_bridge_call(c->arg); else wt_sfclay_bridge_real(c->arg);
  tile(c,1,4); if (staged) wt_sfclay_bridge_call(c->arg); else wt_sfclay_bridge_real(c->arg);
}
static void finite_writable(const struct context *c) {
  int i; size_t j;
  for (i=0;i<WT_SFCLAY_NARG;++i) if (c->arg[i] && wt_sfclay_schema_access[i] == WT_SFCLAY_WRITABLE &&
      wt_sfclay_schema_shape[i] != WT_SFCLAY_SHAPE_SCALAR)
    for (j=0;j<c->cap[i]/sizeof(float);++j) check(isfinite(((float *)c->arg[i])[j]),"nonfinite writable\n");
}
static void run_case(int stable, int water, int isftc, int iz0, int scm, int controls) {
  struct context direct, staged; int i; size_t before = wt_sfclay_bridge_wrapped_calls();
  init_context(&direct,stable,water,isftc,iz0,scm,controls); init_context(&staged,stable,water,isftc,iz0,scm,controls);
  call_two_tiles(&direct,0); dispatch_cap=staged.cap; dispatches=0; call_two_tiles(&staged,1);
  check(dispatches == 2 && wt_sfclay_bridge_wrapped_calls() == before + 2, "wrap composition\n");
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
  finite_writable(&direct); destroy_context(&direct); destroy_context(&staged);
}
int main(void) {
  int true_value=1, stable, water, isftc, iz0, scm;
  __module_sf_sfclay_MOD_sfclayinit(&true_value);
  for (stable=0;stable<2;++stable) for (water=0;water<2;++water) for (isftc=0;isftc<3;++isftc)
    for (iz0=0;iz0<3;++iz0) for (scm=0;scm<2;++scm) run_case(stable,water,isftc,iz0,scm,1);
  run_case(0,0,0,0,0,0);
  puts("SFCLAY Fortran transactional wrap checks passed"); return 0;
}
