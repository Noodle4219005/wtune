#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"CHECK %d: %s\n",__LINE__,#x); exit(2); } } while(0)
static int event[8],n; static void *want[84];
uint64_t wt_sfclay_first_rk_begin(void *g,void *c) { CHECK(g==want[0] && c==want[1]); event[n++]=1; return 11; }
void wt_sfclay_first_rk_end(uint64_t t) { CHECK(t==11); event[n++]=5; }
uint64_t wt_mynn_first_rk_begin(void *g,void *c) { CHECK(g==want[0] && c==want[1]); event[n++]=2; return 22; }
void wt_mynn_first_rk_end(uint64_t t) { CHECK(t==22); event[n++]=4; }
void module_first_rk_step_part1_mp_first_rk_step_part1_(void *a0, void *a1, void *a2, void *a3, void *a4, void *a5, void *a6, void *a7, void *a8, void *a9, void *a10, void *a11, void *a12, void *a13, void *a14, void *a15, void *a16, void *a17, void *a18, void *a19, void *a20, void *a21, void *a22, void *a23, void *a24, void *a25, void *a26, void *a27, void *a28, void *a29, void *a30, void *a31, void *a32, void *a33, void *a34, void *a35, void *a36, void *a37, void *a38, void *a39, void *a40, void *a41, void *a42, void *a43, void *a44, void *a45, void *a46, void *a47, void *a48, void *a49, void *a50, void *a51, void *a52, void *a53, void *a54, void *a55, void *a56, void *a57, void *a58, void *a59, void *a60, void *a61, void *a62, void *a63, void *a64, void *a65, void *a66, void *a67, void *a68, void *a69, void *a70, void *a71, void *a72, void *a73, void *a74, void *a75, void *a76, void *a77, void *a78, void *a79, void *a80, void *a81, void *a82, void *a83) {
  void *got[84]={a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a80, a81, a82, a83}; event[n++]=3;
  for(unsigned i=0;i<84;++i) CHECK(got[i]==want[i]);
}
extern void wt_sfclay_scope_test_call(void **);
int main(void) {
  int storage[84]; static const int expected[]={1, 3, 5};
  for(unsigned i=0;i<84;++i) want[i]=&storage[i];
  want[80]=want[81]=want[82]=want[83]=0;
  wt_sfclay_scope_test_call(want);
  CHECK(n==sizeof(expected)/sizeof(expected[0]));
  for(unsigned i=0;i<(unsigned)n;++i) CHECK(event[i]==expected[i]);
  puts("shared physics first-RK scope forwarding passed"); return 0;
}
