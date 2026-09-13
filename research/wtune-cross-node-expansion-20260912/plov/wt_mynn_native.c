#include "wt_mynn_native.h"
#include <stdint.h>
#include <string.h>

extern void wt_grid_capture_(void *,void **,const size_t *,int *,int *);
extern void wt_mynn_native_scope_(void *,void *,int *,int *,int *);
extern void wt_mynn_static_bits_(uint32_t *,int *);
static int capture(void *grid,wt_live_spans *spans,size_t max_nodes,int *tiles)
{
    void *opaque=spans; int status=WT_SPAN_INVALID;
    wt_grid_capture_(grid,&opaque,&max_nodes,tiles,&status);
    return status==WT_SPAN_OK && *tiles>0;
}
static int scope(void *grid,void *cfg,uint64_t *id,uint64_t *epoch)
{
    int i=0,e=0,status=WT_SPAN_INVALID;
    wt_mynn_native_scope_(grid,cfg,&i,&e,&status);
    if(status!=WT_SPAN_OK || i<=0 || e<=0) return 0;
    *id=(uint64_t)i; *epoch=(uint64_t)e; return 1;
}
static uint64_t fingerprint(void)
{
    uint32_t bits[86]; int status=WT_SPAN_INVALID;
    uint64_t h=UINT64_C(14695981039346656037);
    const unsigned char *p=(const unsigned char *)bits;
    wt_mynn_static_bits_(bits,&status);
    if(status!=WT_SPAN_OK) return 0;
    for(size_t i=0;i<sizeof(bits);++i) { h^=p[i]; h*=UINT64_C(1099511628211); }
    return h;
}
int wt_mynn_native_configure(struct wt_mynn_node_live_config *c)
{
    if(!c) return 0;
    c->capture=capture; c->scope=scope; c->fingerprint=fingerprint;
    return wt_mynn_node_live_configure(c);
}
static int32_t scalar(void *const *a,unsigned i)
{ int32_t n; memcpy(&n,a[i],4); return n; }
static int condition(unsigned code,void *const *a,unsigned index)
{
    switch(code) {
    case WT_MYNN_COND_NEVER:return 0;
    case WT_MYNN_COND_ALWAYS:return 1;
    case WT_MYNN_COND_PRESENT:return a[index]!=NULL;
    case WT_MYNN_COND_QC:return scalar(a,WT_MYNN_FLAG_QC)!=0;
    case WT_MYNN_COND_QI:return scalar(a,WT_MYNN_FLAG_QI)!=0;
    case WT_MYNN_COND_QS:return scalar(a,WT_MYNN_FLAG_QS)!=0;
    case WT_MYNN_COND_QNC:return scalar(a,WT_MYNN_FLAG_QNC)!=0;
    case WT_MYNN_COND_QNI:return scalar(a,WT_MYNN_FLAG_QNI)!=0;
    case WT_MYNN_COND_QNWFA:return scalar(a,WT_MYNN_FLAG_QNWFA)!=0;
    case WT_MYNN_COND_QNIFA:return scalar(a,WT_MYNN_FLAG_QNIFA)!=0;
    case WT_MYNN_COND_QNBCA:return scalar(a,WT_MYNN_FLAG_QNBCA)!=0;
    case WT_MYNN_COND_PLUME:return scalar(a,WT_MYNN_BL_MYNN_EDMF)>0;
    case WT_MYNN_COND_OUTPUT:return scalar(a,WT_MYNN_BL_MYNN_OUTPUT)>0;
    case WT_MYNN_COND_SPP:return scalar(a,WT_MYNN_SPP_PBL)>0;
    case WT_MYNN_COND_BUDGET:return scalar(a,WT_MYNN_TKE_BUDGET)==1;
    default:return -1;
    }
}
int wt_mynn_native_capacities(void *const a[WT_MYNN_NARG],
    const wt_live_spans *spans,uint64_t generation,const struct wt_mynn_node_live_config *c,
    size_t cap[WT_MYNN_NARG])
{
    size_t dims[3],plane,volume;
    if(!a || !c || !cap || !c->abi_verified || !c->caller_verified ||
       !wt_live_spans_is_current(spans,generation)) return 0;
    memset(cap,0,WT_MYNN_NARG*sizeof(*cap));
    for(unsigned i=0;i<WT_MYNN_NARG;++i) {
        if(!a[i]) { if(!wt_mynn_args[i].optional) return 0; continue; }
        if(((uintptr_t)a[i]&3) || (uintptr_t)a[i]>UINTPTR_MAX-4) return 0;
        cap[i]=4;
    }
    for(unsigned d=0;d<3;++d) {
        unsigned lower=d==0 ? WT_MYNN_IMS : d==1 ? WT_MYNN_JMS : WT_MYNN_KMS;
        int64_t n=(int64_t)scalar(a,lower+1)-scalar(a,lower)+1;
        if(n<=0 || (uint64_t)n>SIZE_MAX) return 0;
        dims[d]=(size_t)n;
    }
    if(dims[0]>SIZE_MAX/4 || dims[1]>SIZE_MAX/(dims[0]*4)) return 0;
    plane=dims[0]*dims[1]*4;
    if(dims[2]>SIZE_MAX/plane) return 0;
    volume=plane*dims[2];
    for(unsigned i=0;i<WT_MYNN_NARG;++i) if(a[i] && wt_mynn_args[i].rank) {
        int active=condition(wt_mynn_args[i].access,a,i),write=condition(wt_mynn_args[i].write,a,i);
        size_t needed,remaining;
        if(active<0 || write<0) return 0;
        needed=active || write ? (wt_mynn_args[i].rank==2 ? plane : volume) : 4;
        if((uintptr_t)a[i]>UINTPTR_MAX-needed) return 0;
        /* firstRK:103-107 and its pbl call:594/596/599 prove these three
         * full-formal scratch actuals; they are not domain fieldlist members. */
        if(needed>4 && i!=WT_MYNN_TH && i!=WT_MYNN_DZ && i!=WT_MYNN_EXNER &&
           wt_live_spans_lookup(spans,generation,a[i],i==WT_MYNN_KPBL ? WT_SPAN_INTEGER : WT_SPAN_REAL,
                                4,needed,&remaining)!=WT_SPAN_OK) return 0;
        cap[i]=needed;
        /* Scalar aliasing is permitted only among the typed readonly scalars.
         * Never hide a scalar/array alias by cloning the scalar argument. */
        for(unsigned j=0;j<WT_MYNN_NARG;++j) if(a[j] && !wt_mynn_args[j].rank &&
            (uintptr_t)a[j]<(uintptr_t)a[i]+needed && (uintptr_t)a[i]<(uintptr_t)a[j]+4) return 0;
    }
    return 1;
}
