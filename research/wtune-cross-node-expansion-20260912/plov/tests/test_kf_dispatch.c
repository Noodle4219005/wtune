#include "wt_kf_dispatch.h"
#include <assert.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <pthread.h>

#define STORAGE 256
static void *argvs[WT_KF_NARG]; static size_t capacities[WT_KF_NARG];
static int32_t scalars[WT_KF_NARG]; static unsigned char data[WT_KF_NARG][STORAGE];
static int calls, fatal_calls, reenter, corrupt; static struct wt_kf_dispatch_context *active;
static _Atomic int threaded_calls;
#ifndef WT_KF_NO_OMP_HOOK
int wt_kf_omp_in_parallel(void) { return 0; }
#endif
static int omp_no(void *x) { (void)x; return 0; }
static int omp_yes(void *x) { (void)x; return 1; }
static void native_count(void **a) { (void)a; ++threaded_calls; }
static void *omp_thread(void *opaque) { struct wt_kf_dispatch_context *c = opaque; wt_kf_dispatch(c,argvs,WT_KF_NARG,native_count); return 0; }
static void fatal(void *x, enum wt_kf_dispatch_reason why) { (void)x; (void)why; ++fatal_calls; }
static int capacity(void *x, unsigned i, void *arg, size_t *bytes)
{ (void)x; (void)arg; *bytes = capacities[i]; return 1; }
static void init(void)
{ unsigned i; memset(scalars,0,sizeof(scalars)); memset(data,0,sizeof(data)); calls=fatal_calls=0;
  for(i=0;i<WT_KF_NARG;i++) { argvs[i]=data[i]; capacities[i]=STORAGE; }
  for(i=0;i<19;i++) { argvs[i]=&scalars[i]; capacities[i]=4; scalars[i]=1; }
  scalars[WT_KF_KDE]=scalars[WT_KF_KME]=2; scalars[WT_KF_TRIGGER]=1;
}
static void native(void **a)
{ ++calls; ((unsigned char *)a[WT_KF_RAINCV])[0]++; if(corrupt) ((unsigned char *)argvs[WT_KF_RAINCV])[0]++;
  if(reenter) { reenter=0; wt_kf_dispatch(active,a,WT_KF_NARG,native); } }
static void child_fatal(int mode)
{ struct rlimit limit = {0,0}; struct wt_kf_dispatch_context c={0}; setrlimit(RLIMIT_CORE,&limit); init(); c.enabled=c.abi_verified=1; c.transaction.enabled=1; c.transaction.num_tiles=1; c.transaction.byte_budget=100000; c.capacity=capacity;c.omp_query=omp_no;c.fatal=fatal; active=&c; corrupt=mode==1;reenter=mode==2; wt_kf_dispatch(&c,argvs,WT_KF_NARG,native); _exit(0); }
int main(void)
{ struct wt_kf_dispatch_context c={0}; pid_t p; int status; init();
  wt_kf_dispatch(&c,argvs,WT_KF_NARG,native); assert(calls==1 && c.counters.fallback[WT_KF_DISPATCH_DISABLED]==1);
  init(); c=(struct wt_kf_dispatch_context){0}; c.enabled=c.abi_verified=1;c.transaction.enabled=1;c.transaction.num_tiles=1;c.transaction.byte_budget=100000;c.omp_query=omp_no;
  wt_kf_dispatch(&c,argvs,WT_KF_NARG,native); assert(calls==1 && c.counters.fallback[WT_KF_DISPATCH_CAPACITY]==1);
  init(); c.capacity=capacity; wt_kf_dispatch(&c,argvs,WT_KF_NARG,native); assert(calls==1 && c.counters.staged==1 && data[WT_KF_RAINCV][0]==1);
  init(); scalars[WT_KF_TRIGGER]=2; wt_kf_dispatch(&c,argvs,WT_KF_NARG,native); assert(calls==1 && c.counters.fallback[WT_KF_DISPATCH_PREPARE]==1);
  { pthread_t one,two; init(); threaded_calls=0; c=(struct wt_kf_dispatch_context){0}; c.enabled=c.abi_verified=1; c.omp_query=omp_yes; assert(!pthread_create(&one,0,omp_thread,&c)); assert(!pthread_create(&two,0,omp_thread,&c)); pthread_join(one,0); pthread_join(two,0); assert(threaded_calls==2 && c.counters.fallback[WT_KF_DISPATCH_OMP]==2); }
  p=fork(); if(!p) child_fatal(1); waitpid(p,&status,0); assert(WIFSIGNALED(status)&&WTERMSIG(status)==SIGABRT);
  p=fork(); if(!p) child_fatal(2); waitpid(p,&status,0); assert(WIFSIGNALED(status)&&WTERMSIG(status)==SIGABRT);
  init(); c=(struct wt_kf_dispatch_context){0};
  c.enabled=c.abi_verified=1;c.transaction.enabled=1;c.transaction.num_tiles=1;
  c.transaction.byte_budget=100000;c.capacity=capacity;
  wt_kf_dispatch(&c,argvs,WT_KF_NARG,native);
#ifdef WT_KF_NO_OMP_HOOK
  assert(calls==1 && c.counters.staged==0 && c.counters.fallback[WT_KF_DISPATCH_OMP]==1);
#else
  assert(calls==1 && c.counters.staged==1);
#endif
  return 0;
}
