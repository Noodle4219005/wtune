/* Audited frozen module_cu_kfeta.f90:3087, source SHA256
 * 9533f8600ad060a2e8aa491ed9b7fece633c3d1e2f445e001208cdceed6f1c05.
 * Ordered init35: RTHCUTEN,RQVCUTEN,RQCCUTEN,RQRCUTEN,RQICUTEN,RQSCUTEN,
 * NCA,W0AVG,P_QI,P_QS,SVP1,SVP2,SVP3,SVPT0,P_FIRST_SCALAR,restart,
 * allowed_to_read,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,
 * its,ite,jts,jte,kts,kte. All by-reference; no CHARACTER/hidden lengths. */
#include "wt_kf_node_live.h"
#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <pthread.h>
#ifndef WT_KF_BUILD_INIT_TABLES_VERIFIED
#define WT_KF_BUILD_INIT_TABLES_VERIFIED 0
#endif
static pthread_mutex_t lease_mutex=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t lease_changed=PTHREAD_COND_INITIALIZER;
static unsigned readers,waiting_writers;
static int writer;
static _Atomic uint64_t fingerprint;
static _Thread_local unsigned read_depth,init_depth;
static void pcheck(int status) { if (status) abort(); }
uint64_t wt_kf_init_reader_enter(void)
{
    uint64_t hash;
    if (init_depth || read_depth==UINT_MAX) abort();
    pcheck(pthread_mutex_lock(&lease_mutex));
    while (writer || waiting_writers) pcheck(pthread_cond_wait(&lease_changed,&lease_mutex));
    if (readers==UINT_MAX) abort();
    ++readers; ++read_depth;
    hash=atomic_load_explicit(&fingerprint,memory_order_acquire);
    pcheck(pthread_mutex_unlock(&lease_mutex));
    return hash;
}
void wt_kf_init_reader_leave(void)
{
    pcheck(pthread_mutex_lock(&lease_mutex));
    if (!read_depth || !readers || writer) abort();
    --read_depth; --readers;
    pcheck(pthread_mutex_unlock(&lease_mutex));
}
uint64_t wt_kf_init_fingerprint(void)
{ return atomic_load_explicit(&fingerprint,memory_order_acquire); }
#if WT_KF_BUILD_INIT_TABLES_VERIFIED
/* Intel object's global COMMON symbol sizes were verified with nm. */
extern unsigned char module_cu_kfeta_mp_ttab_[220000],module_cu_kfeta_mp_qstab_[220000];
extern unsigned char module_cu_kfeta_mp_the0k_[880],module_cu_kfeta_mp_alu_[800];
extern unsigned char module_cu_kfeta_mp_rdpr_[4],module_cu_kfeta_mp_rdthk_[4],module_cu_kfeta_mp_plutop_[4];
static uint64_t tables_hash(void)
{
    const unsigned char *p[]={module_cu_kfeta_mp_ttab_,module_cu_kfeta_mp_qstab_,
        module_cu_kfeta_mp_the0k_,module_cu_kfeta_mp_alu_,module_cu_kfeta_mp_rdpr_,
        module_cu_kfeta_mp_rdthk_,module_cu_kfeta_mp_plutop_};
    const size_t n[]={220000,220000,880,800,4,4,4};
    uint64_t h=UINT64_C(14695981039346656037);
    for (unsigned a=0;a<7;++a) for (size_t i=0;i<n[a];++i) { h^=p[a][i]; h*=UINT64_C(1099511628211); }
    return h ? h : 1;
}
#else
static uint64_t tables_hash(void) { return 0; }
#endif
extern void __real_module_cu_kfeta_mp_kf_eta_init_(void *a0, void *a1, void *a2, void *a3, void *a4, void *a5, void *a6, void *a7, void *a8, void *a9, void *a10, void *a11, void *a12, void *a13, void *a14, void *a15, void *a16, void *a17, void *a18, void *a19, void *a20, void *a21, void *a22, void *a23, void *a24, void *a25, void *a26, void *a27, void *a28, void *a29, void *a30, void *a31, void *a32, void *a33, void *a34);
void __wrap_module_cu_kfeta_mp_kf_eta_init_(void *a0, void *a1, void *a2, void *a3, void *a4, void *a5, void *a6, void *a7, void *a8, void *a9, void *a10, void *a11, void *a12, void *a13, void *a14, void *a15, void *a16, void *a17, void *a18, void *a19, void *a20, void *a21, void *a22, void *a23, void *a24, void *a25, void *a26, void *a27, void *a28, void *a29, void *a30, void *a31, void *a32, void *a33, void *a34)
{
    if (read_depth || init_depth) abort();
    pcheck(pthread_mutex_lock(&lease_mutex));
    if (readers || waiting_writers==UINT_MAX) abort();
    ++waiting_writers;
    while (writer) pcheck(pthread_cond_wait(&lease_changed,&lease_mutex));
    --waiting_writers; writer=1; init_depth=1;
    pcheck(pthread_mutex_unlock(&lease_mutex));
    atomic_store_explicit(&fingerprint,0,memory_order_release);
    __real_module_cu_kfeta_mp_kf_eta_init_(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34);
    atomic_store_explicit(&fingerprint,tables_hash(),memory_order_release);
    pcheck(pthread_mutex_lock(&lease_mutex));
    writer=0; init_depth=0;
    pcheck(pthread_cond_broadcast(&lease_changed));
    pcheck(pthread_mutex_unlock(&lease_mutex));
}
