
#include "wt_sfclay.h"
#include "wt_sfclay_schema.h"
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *m) { fputs(m, stderr); exit(2); }
static void yes(int ok, const char *m) { if (!ok) fail(m); }
static void puti(void *p, int v) { memcpy(p, &v, 4); }
static void setup(void **a, size_t *c, void **owned) {
  int i;
  const size_t n3 = 3u * 2u * 2u * 4u, n2 = 3u * 2u * 4u;
  for (i=0;i<WT_SFCLAY_NARG;i++) {
    size_t n = wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_3D ? n3 :
               wt_sfclay_schema_shape[i] == WT_SFCLAY_SHAPE_2D ? n2 : 4u;
    owned[i] = calloc(1,n); if (!owned[i]) fail("alloc\n");
    a[i]=owned[i]; c[i]=n; memset(a[i], i+1, n);
  }
  puti(a[58],1); puti(a[59],3); puti(a[60],1); puti(a[61],2); puti(a[62],1); puti(a[63],2);
  puti(a[64],1); puti(a[65],3); puti(a[66],1); puti(a[67],2); puti(a[68],1); puti(a[69],2);
  puti(a[70],1); puti(a[71],2); puti(a[72],1); puti(a[73],2); puti(a[74],1); puti(a[75],2);
  free(owned[81]); free(owned[82]); free(owned[83]); owned[81]=owned[82]=owned[83]=NULL;
  a[81]=a[82]=a[83]=NULL; c[81]=c[82]=c[83]=0;
}
static void cleanup(void **o) { for (int i=0;i<WT_SFCLAY_NARG;i++) free(o[i]); }
int main(void) {
  void *a[WT_SFCLAY_NARG], *o[WT_SFCLAY_NARG], **clone; size_t c[WT_SFCLAY_NARG];
  wt_sfclay_tx *tx = NULL; unsigned char first;
  setup(a,c,o);
  a[64]=(void *)(uintptr_t)1; c[64]=0;
  yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_CAPACITY,"bound capacity preflight\n");
  a[64]=o[64]; c[64]=4;
  a[0]=(void *)(UINTPTR_MAX-1); c[0]=4;
  yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OVERFLOW,"owner address preflight\n");
  a[0]=o[0]; c[0]=48;
  puti(a[64],INT_MIN); puti(a[65],INT_MAX); puti(a[66],INT_MIN); puti(a[67],INT_MAX);
  puti(a[68],INT_MIN); puti(a[69],INT_MAX);
  yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OVERFLOW,"shape overflow\n");
  puti(a[64],1); puti(a[65],3); puti(a[66],1); puti(a[67],2); puti(a[68],1); puti(a[69],2);
  yes(wt_sfclay_prepare(a,c,1,&tx)==WT_SFCLAY_BUDGET,"budget preflight\n");
  a[6]=a[7]; c[6]=c[7]=4; yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OK,"readonly alias\n"); wt_sfclay_destroy(tx); a[6]=o[6]; c[6]=4;
  yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OK && tx, "prepare\n");
  clone=wt_sfclay_cloned_argv(tx); yes(clone && clone[12]!=a[12], "clone\n");
  ((unsigned char*)clone[12])[0]=77; yes(wt_sfclay_commit(tx)==WT_SFCLAY_OK, "commit\n");
  yes(((unsigned char*)a[12])[0]==77 && ((unsigned char*)a[12])[1]==13, "writable preserve\n");
  yes(wt_sfclay_commit(tx)==WT_SFCLAY_STATE, "double commit\n"); wt_sfclay_destroy(tx);
  tx=NULL; yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OK,"serialized tile prepare\n");
  clone=wt_sfclay_cloned_argv(tx); ((unsigned char*)clone[12])[1]=66;
  yes(wt_sfclay_commit(tx)==WT_SFCLAY_OK && ((unsigned char*)a[12])[0]==77 && ((unsigned char*)a[12])[1]==66,
      "serialized tile preserve\n"); wt_sfclay_destroy(tx);
  first=((unsigned char*)a[0])[0]; tx=NULL; yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OK,"prepare readonly\n");
  clone=wt_sfclay_cloned_argv(tx); ((unsigned char*)clone[0])[0]^=1;
  yes(wt_sfclay_commit(tx)==WT_SFCLAY_STALE,"readonly guard\n");
  yes(((unsigned char*)a[0])[0]==first,"readonly owner\n"); wt_sfclay_destroy(tx);
  tx=NULL; yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OK,"canary prepare\n");
  clone=wt_sfclay_cloned_argv(tx); ((unsigned char*)clone[0])[-1]=0;
  yes(wt_sfclay_commit(tx)==WT_SFCLAY_STALE,"canary guard\n"); wt_sfclay_destroy(tx);
  tx=NULL; yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OK,"owner snapshot prepare\n");
  ((unsigned char*)a[12])[0]^=1; yes(wt_sfclay_commit(tx)==WT_SFCLAY_STALE,"owner snapshot\n"); wt_sfclay_destroy(tx);
  tx=NULL; a[76]=NULL; c[76]=0; yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_BAD_ARGUMENT,"required optional\n"); a[76]=o[76]; c[76]=24;
  a[12]=a[0]; c[12]=48; yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_OVERLAP,"writable overlap\n"); a[12]=o[12]; c[12]=24;
  puti(a[68],2); puti(a[69],3); puti(a[74],2); puti(a[75],3);
  yes(wt_sfclay_prepare(a,c,1u<<20,&tx)==WT_SFCLAY_BOUNDS,"k lower bound\n");
  cleanup(o); puts("SFCLAY transaction checks passed"); return 0;
}
