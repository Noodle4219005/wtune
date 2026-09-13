#include "wt_sfclay_capacity.h"

#include <stdint.h>
#include <string.h>

static int read_i32(const void *p, int32_t *out) { if (!p || !out) return 0; memcpy(out,p,4); return 1; }
static int extent(int32_t low, int32_t high, size_t *out) {
  int64_t n=(int64_t)high-(int64_t)low+1; if(n<=0 || (uint64_t)n>SIZE_MAX) return 0; *out=(size_t)n; return 1;
}
static int multiply(size_t a,size_t b,size_t *out) { if(!a||!b||a>SIZE_MAX/b)return 0; *out=a*b; return 1; }
static int shapes(void *const *a,size_t *two,size_t *three) {
  int32_t ims,ime,jms,jme,kms,kme; size_t ni,nj,nk,n;
  if(!read_i32(a[WT_SF_IMS],&ims)||!read_i32(a[WT_SF_IME],&ime)||!read_i32(a[WT_SF_JMS],&jms)||!read_i32(a[WT_SF_JME],&jme)||
     !read_i32(a[WT_SF_KMS],&kms)||!read_i32(a[WT_SF_KME],&kme)||!extent(ims,ime,&ni)||!extent(jms,jme,&nj)||!extent(kms,kme,&nk)||
     !multiply(ni,nj,&n)||!multiply(n,4,two)||!multiply(n,nk,&n)||!multiply(n,4,three)) return 0;
  return 1;
}
static int caller_owned(unsigned i) {
  switch(i) { case WT_SF_U3D: case WT_SF_V3D: case WT_SF_DZ8W: case WT_SF_CHS: case WT_SF_CHS2: case WT_SF_CQS2: case WT_SF_CPM: case WT_SF_PSIM: case WT_SF_PSIH: case WT_SF_QGH: case WT_SF_GZ1OZ0: case WT_SF_LAKEMASK: return 1; default:return 0; }
}
static int live(const wt_live_spans *s,uint64_t g,void *arg,size_t required) {
  size_t remaining=0;
  return arg && wt_live_spans_lookup(s,g,arg,WT_SPAN_REAL,4,required,&remaining)==WT_SPAN_OK;
}
void wt_sfclay_capacity_reset(struct wt_sfclay_capacity *p) { if(p) memset(p,0,sizeof *p); }
int wt_sfclay_capacity_init(struct wt_sfclay_capacity *p,void *const a[WT_SFCLAY_NARG],const wt_live_spans *s,uint64_t g,int verified) {
  return wt_sfclay_capacity_init_attested(p,a,s,g,verified,0);
}
int wt_sfclay_capacity_init_attested(struct wt_sfclay_capacity *p,void *const a[WT_SFCLAY_NARG],const wt_live_spans *s,uint64_t g,int verified,int chain) {
  size_t two,three,need; unsigned i;
  if(!p) return 0;
  wt_sfclay_capacity_reset(p);
  /* Do not inspect argv, including bound scalars, before proof and epoch gates. */
  if(!a||!s||verified!=1||(chain!=0&&chain!=1)||!wt_live_spans_is_current(s,g)) return 0;
  for(i=0;i<WT_SFCLAY_NARG;i++) {
    p->expected[i]=a[i];
    if(!a[i]) { if(i>=WT_SF_ISFTCFLX) { p->bytes[i]=0; continue; } goto fail; }
    if(wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_SCALAR) { p->bytes[i]=4; continue; }
  }
  if(!shapes(a,&two,&three)) goto fail;
  for(i=0;i<WT_SFCLAY_NARG;i++) if(wt_sfclay_schema_shape[i]!=WT_SFCLAY_SHAPE_SCALAR) {
    need=wt_sfclay_schema_shape[i]==WT_SFCLAY_SHAPE_3D?three:two;
    if(!(chain && caller_owned(i)) && !live(s,g,a[i],need)) goto fail;
    p->bytes[i]=need;
  }
  p->spans=s; p->generation=g; p->ready=1; return 1;
fail: wt_sfclay_capacity_reset(p); return 0;
}
int wt_sfclay_capacity_lookup(void *opaque,unsigned i,void *arg,size_t *bytes) {
  struct wt_sfclay_capacity *p=opaque; if(bytes)*bytes=0;
  if(!p||!bytes||!p->ready||i>=WT_SFCLAY_NARG||p->expected[i]!=arg||!wt_live_spans_is_current(p->spans,p->generation)) return 0;
  *bytes=p->bytes[i]; return 1;
}
