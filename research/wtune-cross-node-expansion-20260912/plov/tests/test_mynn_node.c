#define _POSIX_C_SOURCE 200809L
#include "wt_mynn_node.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define BUDGET ((size_t)1<<20)
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"failed %s:%d: %s\\n",__FILE__,__LINE__,#x); exit(2); } } while (0)
struct fixture { void *argv[WT_MYNN_NARG]; size_t cap[WT_MYNN_NARG]; unsigned char *before[WT_MYNN_NARG]; unsigned shared[WT_MYNN_NARG]; };
struct wire_header { uint64_t schema; uint32_t state, logical_true_bits; size_t bytes; int32_t j; unsigned char present[WT_MYNN_NARG], active[WT_MYNN_NARG], writable[WT_MYNN_NARG], alignment[WT_MYNN_NARG]; int32_t scalar[WT_MYNN_NARG]; size_t offset[WT_MYNN_NARG], packed[WT_MYNN_NARG], full[WT_MYNN_NARG]; };
struct callback { unsigned char marker; int mode, calls; };
static size_t ni(void) { return 4; } static size_t nj(void) { return 2; } static size_t nk(void) { return 3; }
static size_t full(unsigned a) { return !wt_mynn_args[a].rank ? 4 : wt_mynn_args[a].rank==2 ? ni()*nj()*4 : ni()*nj()*nk()*4; }
static void put(struct fixture *f,unsigned a,int32_t x) { memcpy(f->argv[a],&x,4); }
static void setup(struct fixture *f,int row)
{
    int32_t bound[18]={1,5,1,3,1,4,1,4,1,2,1,3,2,3,row,row,1,2}; unsigned a;
    memset(f,0,sizeof *f);
    for(a=0;a<WT_MYNN_NARG;++a) { f->cap[a]=full(a); f->argv[a]=malloc(f->cap[a]); f->before[a]=malloc(f->cap[a]); CHECK(f->argv[a]&&f->before[a]); memset(f->argv[a],wt_mynn_args[a].rank ? 0x20+a : 0,f->cap[a]); }
    for(a=0;a<18;++a) put(f,a,bound[a]);
    put(f,WT_MYNN_INITFLAG,0); put(f,WT_MYNN_TKE_BUDGET,0); put(f,WT_MYNN_SPP_PBL,0);
    for(a=0;a<WT_MYNN_NARG;++a) memcpy(f->before[a],f->argv[a],f->cap[a]);
}
static void share_arrays(struct fixture *dst,struct fixture *src)
{ unsigned a; for(a=0;a<WT_MYNN_NARG;++a) if(wt_mynn_args[a].rank) { free(dst->argv[a]); free(dst->before[a]); dst->argv[a]=src->argv[a]; dst->before[a]=src->before[a]; dst->cap[a]=src->cap[a]; dst->shared[a]=1; } }
static void destroy(struct fixture *f)
{ unsigned a; for(a=0;a<WT_MYNN_NARG;++a) if(!f->shared[a]) { free(f->argv[a]); free(f->before[a]); } }
static size_t row_offset(void **argv,unsigned a)
{ int32_t jms,jts; memcpy(&jms,argv[WT_MYNN_JMS],4); memcpy(&jts,argv[WT_MYNN_JTS],4); return (size_t)(jts-jms)*(wt_mynn_args[a].rank==2 ? ni()*4 : ni()*nk()*4); }
static size_t active_offset(void **argv,unsigned a)
{ int32_t ims,its; memcpy(&ims,argv[WT_MYNN_IMS],4); memcpy(&its,argv[WT_MYNN_ITS],4); return row_offset(argv,a)+(size_t)(its-ims)*4; }
static int native(void *opaque,void *argv[WT_MYNN_NARG])
{
    struct callback *c=opaque; unsigned a; size_t i,off;
    ++c->calls;
    if(c->mode==3) return 1;
    if(c->mode==1) { ((unsigned char *)argv[WT_MYNN_XLAND])[active_offset(argv,WT_MYNN_XLAND)]^=1; return 0; }
    if(c->mode==2) { ((unsigned char *)argv[WT_MYNN_PBLH])[full(WT_MYNN_PBLH)]^=1; return 0; }
    { int32_t jms,jme,jts,jte; memcpy(&jms,argv[WT_MYNN_JMS],4); memcpy(&jme,argv[WT_MYNN_JME],4); memcpy(&jts,argv[WT_MYNN_JTS],4); memcpy(&jte,argv[WT_MYNN_JTE],4); if(jms!=1||jme!=2||jts!=jte) return 1; }
    for(a=0;a<WT_MYNN_NARG;++a) if(wt_mynn_args[a].rank && wt_mynn_args[a].write==WT_MYNN_COND_ALWAYS) {
        off=active_offset(argv,a); for(i=0;i<(wt_mynn_args[a].rank==2?1:nk());++i) memset((unsigned char *)argv[a]+off+i*ni()*4,c->marker,2*4);
    }
    return 0;
}
static const struct wt_mynn_layout_config layout_config={1,1,1,BUDGET};
static void unchanged_except_rows(const struct fixture *f,unsigned rows,unsigned char marker)
{
    unsigned a; size_t i;
    for(a=0;a<WT_MYNN_NARG;++a) if(wt_mynn_args[a].rank) for(i=0;i<f->cap[a];++i) {
        size_t stride=wt_mynn_args[a].rank==2?ni()*4:ni()*nk()*4, row=i/stride, local=i%stride;
        int write=wt_mynn_args[a].write==WT_MYNN_COND_ALWAYS && row<2 && (rows&(1u<<row)) && local%(ni()*4)>=4 && local%(ni()*4)<12;
        if(write) CHECK(((unsigned char *)f->argv[a])[i]==marker); else CHECK(((unsigned char *)f->argv[a])[i]==f->before[a][i]);
    }
}
static void direct_batch(void)
{
    struct fixture f; struct wt_mynn_batch *b=NULL; struct callback cb={0x71,0,0}; const void *q; void *r; size_t n; int invoked=-1;
    setup(&f,1); CHECK(wt_mynn_batch_prepare(f.argv,f.cap,&layout_config,BUDGET,&b)==WT_MYNN_OK); q=wt_mynn_batch_request(b,&n); CHECK(q&&wt_mynn_batch_geometry(b,(int32_t[18]){0})); r=malloc(n); CHECK(r);
    CHECK(wt_mynn_batch_execute(q,n,r,n,BUDGET,native,&cb,&invoked)==WT_MYNN_OK&&invoked==1&&cb.calls==1);
    CHECK(wt_mynn_batch_commit(b,r,n)==WT_MYNN_OK); unchanged_except_rows(&f,1,cb.marker); free(r); wt_mynn_batch_destroy(b); destroy(&f);
}
static void batch_failures(void)
{
    struct fixture f; struct wt_mynn_batch *b; struct callback cb={0x11,0,0}; const void *q; void *r; size_t n; int invoked;
    setup(&f,1); put(&f,WT_MYNN_TKE_BUDGET,1); CHECK(wt_mynn_batch_prepare(f.argv,f.cap,&layout_config,BUDGET,&b)==WT_MYNN_LAYOUT); put(&f,WT_MYNN_TKE_BUDGET,0); put(&f,WT_MYNN_SPP_PBL,1); CHECK(wt_mynn_batch_prepare(f.argv,f.cap,&layout_config,BUDGET,&b)==WT_MYNN_LAYOUT); put(&f,WT_MYNN_SPP_PBL,0); put(&f,WT_MYNN_INITFLAG,1); CHECK(wt_mynn_batch_prepare(f.argv,f.cap,&layout_config,BUDGET,&b)==WT_MYNN_LAYOUT); put(&f,WT_MYNN_INITFLAG,0);
    CHECK(wt_mynn_batch_prepare(f.argv,f.cap,&layout_config,BUDGET,&b)==WT_MYNN_OK); q=wt_mynn_batch_request(b,&n); r=malloc(n); CHECK(r); { unsigned char *bad=malloc(n); CHECK(bad); memcpy(bad,q,n); ((struct wire_header *)bad)->alignment[WT_MYNN_XLAND]=255; CHECK(wt_mynn_batch_execute(bad,n,r,n,BUDGET,native,&cb,&invoked)==WT_MYNN_ARGUMENT&&invoked==0); free(bad); } cb.mode=1; CHECK(wt_mynn_batch_execute(q,n,r,n,BUDGET,native,&cb,&invoked)==WT_MYNN_READONLY&&invoked==1); CHECK(wt_mynn_batch_commit(b,r,n)!=WT_MYNN_OK); free(r); wt_mynn_batch_destroy(b);
    CHECK(wt_mynn_batch_prepare(f.argv,f.cap,&layout_config,BUDGET,&b)==WT_MYNN_OK); q=wt_mynn_batch_request(b,&n); r=malloc(n); cb.mode=2; CHECK(wt_mynn_batch_execute(q,n,r,n,BUDGET,native,&cb,&invoked)==WT_MYNN_GUARD&&invoked==1); free(r); wt_mynn_batch_destroy(b); unchanged_except_rows(&f,0,0); destroy(&f);
}
static struct wt_nl_signature signature(void) { struct wt_nl_signature s={WT_MYNN_NODE_PHASE,19,WT_MYNN_BATCH_SCHEMA,4,7,WT_MYNN_BATCH_SCHEMA,23}; return s; }
static struct wt_mynn_node_context context(struct wt_nl_service *s,unsigned rank,unsigned thread,struct callback *cb) { struct wt_mynn_node_context c={s,{0},rank,thread,BUDGET,native,cb}; c.signature=signature(); return c; }
static int service(struct wt_nl_service *s)
{ char path[]="/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/mynn/node-service-XXXXXX"; struct wt_nl_config c={2,2,65536}; int fd=mkstemp(path); CHECK(fd>=0); unlink(path); CHECK(wt_nl_init_fd(fd,&c,s)==WT_NL_OK); return fd; }
static void forked_rows(void)
{
    struct wt_nl_service s; struct fixture a,b; struct callback owner={0x44,0,0}; struct wt_mynn_node_context oc,hc; struct wt_mynn_node_call ca={0},cb={0}; struct wt_nl_ticket ticket; enum wt_mynn_status bs; struct wt_nl_counters counts; int fd=service(&s),pipefd[2],status; pid_t child;
    setup(&a,1); setup(&b,2); share_arrays(&b,&a); oc=context(&s,0,0,&owner);
    CHECK(wt_mynn_node_publish(&oc,a.argv,a.cap,&layout_config,1,&ca,&bs)==WT_NL_OK); oc.thread=1; CHECK(wt_mynn_node_publish(&oc,b.argv,b.cap,&layout_config,2,&cb,&bs)==WT_NL_OK); CHECK(pipe(pipefd)==0); child=fork(); CHECK(child>=0);
    if(!child) { char x; size_t mapped=s.mapping_bytes; struct wt_nl_service attached; struct callback helper={0x44,0,0}; close(pipefd[1]); wt_nl_detach(&s); if(wt_nl_attach_fd(fd,mapped,&attached)!=WT_NL_OK) _exit(11); hc=context(&attached,1,0,&helper); if(read(pipefd[0],&x,1)!=1) _exit(12); if(wt_mynn_node_progress(&hc,99,&ticket)!=WT_NL_OK) _exit(13); if(wt_mynn_node_progress(&hc,99,&ticket)!=WT_NL_OK) _exit(14); if(helper.calls!=2) _exit(15); wt_nl_detach(&attached); _exit(0); }
    close(pipefd[0]); CHECK(write(pipefd[1],"x",1)==1); close(pipefd[1]); CHECK(waitpid(child,&status,0)==child); if(!WIFEXITED(status)||WEXITSTATUS(status)) { fprintf(stderr,"child=%d\\n",WIFEXITED(status)?WEXITSTATUS(status):-1); exit(2); } oc.thread=0; CHECK(wt_mynn_node_collect(&oc,&ca)==WT_NL_OK&&wt_mynn_node_finish(&oc,&ca)==WT_NL_OK); oc.thread=1; CHECK(wt_mynn_node_collect(&oc,&cb)==WT_NL_OK&&wt_mynn_node_finish(&oc,&cb)==WT_NL_OK); CHECK(wt_nl_get_counters(&s,&counts)==WT_NL_OK&&counts.helper_executed==2&&counts.committed==2); unchanged_except_rows(&a,3,0x44); wt_mynn_node_destroy(&ca); wt_mynn_node_destroy(&cb); wt_nl_detach(&s); close(fd); destroy(&b); destroy(&a);
}
int main(void) { direct_batch(); batch_failures(); forked_rows(); puts("PASS MYNN batch/node strict row transport, guards, and forked helper"); return 0; }
