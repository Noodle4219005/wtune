#include <assert.h>
extern void wt_mynn_native_scope_(void*,void*,int*,int*,int*);
int main(void) { int g[2]={4,27},cfg=5,id=0,e=0,s=2;
wt_mynn_native_scope_(g,&cfg,&id,&e,&s);
#ifdef WIDE
assert(s==2 && id==0 && e==0);
#else
assert(s==0 && id==4 && e==27);cfg=1;wt_mynn_native_scope_(g,&cfg,&id,&e,&s);assert(s==2);
cfg=5;g[1]=0;wt_mynn_native_scope_(g,&cfg,&id,&e,&s);assert(s==2);
g[0]=-1;g[1]=27;wt_mynn_native_scope_(g,&cfg,&id,&e,&s);assert(s==2);
#endif
return 0; }
