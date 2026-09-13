#include "wt_rk_layout.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OK(x) do { if (!(x)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)
static uint32_t scalars[WT_RK_NARG], logical_false;
static float payload[71][4096];
static void *argvv[WT_RK_NARG]; static size_t caps[WT_RK_NARG];
static struct wt_rk_native_result before = {0,3,0,2,2,0,0,0,1,0,0,0,0,0,0,0,0}, after;
static struct wt_rk_layout_config cfg;
static void setint(unsigned n, int x) { memcpy(&scalars[n], &x, 4); }
static int zeroed(const struct wt_rk_layout *layout) { struct wt_rk_layout zero={0}; return !memcmp(layout,&zero,sizeof zero); }
static void fixture(void) {
 unsigned i; memset(&cfg,0,sizeof cfg); cfg.enabled=cfg.source_abi_verified=cfg.caller_closure_verified=1;
 after=before; cfg.config_read_lease=cfg.exclusive_owner=1; cfg.logical_true_bits=1; cfg.config_identity=&cfg; cfg.query_before=&before; cfg.query_after=&after;
 for(i=0;i<WT_RK_NARG;i++){ argvv[i]=payload[i]; caps[i]=sizeof payload[i]; }
 argvv[WT_RK_CONFIG_FLAGS]=&cfg; caps[WT_RK_CONFIG_FLAGS]=0;
 for(i=0;i<WT_RK_NARG;i++) if(wt_rk_args[i].shape==WT_RK_SHAPE_SCALAR && i!=WT_RK_CONFIG_FLAGS){ argvv[i]=&scalars[i]; caps[i]=4; }
 argvv[WT_RK_TENDDEC]=argvv[WT_RK_MOIST_STEP]=argvv[WT_RK_MIX2_OFF]=argvv[WT_RK_MIX6_OFF]=&logical_false;
 setint(WT_RK_SCS,1); setint(WT_RK_SCE,1); setint(WT_RK_RK_STEP,2);
 setint(WT_RK_IDS,1); setint(WT_RK_IDE,8); setint(WT_RK_JDS,1); setint(WT_RK_JDE,8); setint(WT_RK_KDS,1); setint(WT_RK_KDE,6);
 setint(WT_RK_IMS,1); setint(WT_RK_IME,8); setint(WT_RK_JMS,1); setint(WT_RK_JME,8); setint(WT_RK_KMS,1); setint(WT_RK_KME,6);
 setint(WT_RK_ITS,2); setint(WT_RK_ITE,6); setint(WT_RK_JTS,2); setint(WT_RK_JTE,6); setint(WT_RK_KTS,2); setint(WT_RK_KTE,6);
}
int main(void) {
 struct wt_rk_layout out; enum wt_rk_layout_status s; fixture();
 s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_OK && out.ready && out.required_bytes[WT_RK_RU]==1536);
 OK(out.writable[WT_RK_WWE] && out.writable[WT_RK_WWI] && out.writable[WT_RK_RQVFTEN] && !out.writable[WT_RK_WW]);
 cfg.source_abi_verified=0; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_ABI && !out.ready); cfg.source_abi_verified=1;
 caps[WT_RK_RU]=0; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_CAPACITY && !out.ready); caps[WT_RK_RU]=sizeof payload[0];
 argvv[WT_RK_CONFIG_FLAGS]=NULL; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_ARGUMENT && !out.ready); argvv[WT_RK_CONFIG_FLAGS]=&cfg;
 argvv[WT_RK_CONFIG_FLAGS]=(void *)(uintptr_t)1; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_ARGUMENT && !out.ready); argvv[WT_RK_CONFIG_FLAGS]=&cfg;
 cfg.config_identity=(void *)(uintptr_t)1; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_ARGUMENT && !out.ready); cfg.config_identity=&cfg;
 setint(WT_RK_SCE,2); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_BRANCH && !out.ready); setint(WT_RK_SCE,1);
 setint(WT_RK_ITE,7); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_BOUNDS && !out.ready); setint(WT_RK_ITE,6);
 before.v_sca_adv_order=3; after.v_sca_adv_order=3; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_BRANCH); before.v_sca_adv_order=after.v_sca_adv_order=2;
 { struct wt_rk_native_result changed=before; changed.time_step=2; cfg.query_after=&changed; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_QUERY); } cfg.query_after=&after;
 cfg.query_after=cfg.query_before; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_QUERY && zeroed(&out)); cfg.query_after=&after;
 setint(WT_RK_IDS,INT_MAX); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK((s==WT_RK_LAYOUT_BOUNDS||s==WT_RK_LAYOUT_OVERFLOW) && !out.ready);
 fixture(); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_OK && out.required_bytes[WT_RK_C1H]==24);
 caps[WT_RK_C1H]=6; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_CAPACITY && !out.ready);
 fixture(); setint(WT_RK_IDE,INT_MIN+1); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK((s==WT_RK_LAYOUT_BOUNDS||s==WT_RK_LAYOUT_OVERFLOW) && !out.ready);
 fixture(); setint(WT_RK_ITE,INT_MAX); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK((s==WT_RK_LAYOUT_BOUNDS||s==WT_RK_LAYOUT_OVERFLOW) && !out.ready);
 fixture(); setint(WT_RK_ITS,7); setint(WT_RK_ITE,6); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_BOUNDS && !out.ready);
 fixture(); setint(WT_RK_JTS,7); setint(WT_RK_JTE,6); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_BOUNDS && !out.ready);
 fixture(); setint(WT_RK_KTS,6); setint(WT_RK_KTE,5); s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_BOUNDS && !out.ready);
 fixture(); cfg.byte_budget=1u<<20; cfg.allocation_descriptor_present=2; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_CONFIG && zeroed(&out));
 fixture(); before.specified=2; after.specified=2; s=wt_rk_layout_build(argvv,caps,&cfg,&out); OK(s==WT_RK_LAYOUT_QUERY && zeroed(&out)); before.specified=after.specified=0;
 puts("RK layout gates, full backing, opacity, and zeroed failures PASS"); return 0;
}
