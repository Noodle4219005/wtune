#ifndef WT_KF_CAPACITY_H
#define WT_KF_CAPACITY_H
#include <stddef.h>
#include <stdint.h>
#include "wt_kf_cps_schema.h"
#include "wt_live_spans.h"

/* Source ABI authorization covers the audited CPS, current cumulus-driver
 * actuals (including PRATEC/scalars/inactive fields), not merely the CPS hash.
 * Initialisation requires exclusive owner/lifetime control. It retains no owner
 * storage: only argument identities and byte requirements are cached. */
struct wt_kf_capacity {
  const wt_live_spans *spans;
  uint64_t generation;
  void *expected[WT_KF_NARG];
  size_t bytes[WT_KF_NARG];
  int ready;
};
int wt_kf_capacity_init(struct wt_kf_capacity *, void *const argv[WT_KF_NARG],
                        const wt_live_spans *, uint64_t generation,
                        int source_abi_verified);
/* Default init is strict. This opt-in admits only TH, PI, DZ8W and
 * CU_ACT_FLAG without a live span, after caller_chain_verified independently
 * attests solve_em.o, first-rk.o, cumulus-driver.o, KF-CPS.o, module_domain.o
 * and its bounds-helper implementation, model_data_order.inc, compiler ABI
 * widths/options, and every wrapped CPS relocation on the audited path.
 * A CPS hash or environment flag alone is insufficient. Attestation is
 * falsified by array sections, extra callers, bounds drift, or async use;
 * lifetime must extend through synchronous return. All other arrays still
 * require the frozen live registry generation. */
int wt_kf_capacity_init_attested(struct wt_kf_capacity *,
  void *const argv[WT_KF_NARG], const wt_live_spans *, uint64_t generation,
  int source_abi_verified, int caller_chain_verified);
int wt_kf_capacity_lookup(void *opaque, unsigned index, void *arg, size_t *bytes);
void wt_kf_capacity_reset(struct wt_kf_capacity *);
#endif
