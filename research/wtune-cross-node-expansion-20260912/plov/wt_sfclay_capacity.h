#ifndef WT_SFCLAY_CAPACITY_H
#define WT_SFCLAY_CAPACITY_H

#include <stddef.h>
#include <stdint.h>
#include "wt_live_spans.h"
#include "wt_sfclay_schema.h"

#define WT_SFCLAY_PROVIDER_PROOF_SHA256 "108cb46a7528fa06b035f98f035219c6e90e2a66bb7d91c500f60b2db3c5d7cc"
enum {
  WT_SF_U3D=0, WT_SF_V3D=1, WT_SF_DZ8W=5, WT_SF_CHS=12, WT_SF_CHS2=13,
  WT_SF_CQS2=14, WT_SF_CPM=15, WT_SF_PSIM=23, WT_SF_PSIH=24, WT_SF_QGH=34,
  WT_SF_GZ1OZ0=42, WT_SF_LAKEMASK=57, WT_SF_IMS=64, WT_SF_IME=65,
  WT_SF_JMS=66, WT_SF_JME=67, WT_SF_KMS=68, WT_SF_KME=69,
  WT_SF_ISFTCFLX=81, WT_SF_IZ0TLND=82, WT_SF_SCM_FORCE_FLUX=83
};

/* source_abi_verified attests the audited SFCLAY source, driver actuals and
 * width/default-kind contract.  caller_chain_verified is false by default.
 * If true, it additionally attests the complete synchronous surface/solve_em
 * path described by WT_SFCLAY_PROVIDER_PROOF_SHA256; it admits only U3D, V3D,
 * DZ8W, CHS, CHS2, CQS2, CPM, QGH, LAKEMASK, PSIM, PSIH and GZ1OZ0.
 * Neither flag is an environment setting or an allocation claim.  The caller
 * retains owner lifetime and exclusive access through native invocation. */
struct wt_sfclay_capacity {
  const wt_live_spans *spans;
  uint64_t generation;
  void *expected[WT_SFCLAY_NARG];
  size_t bytes[WT_SFCLAY_NARG];
  int ready;
};

int wt_sfclay_capacity_init(struct wt_sfclay_capacity *,
  void *const argv[WT_SFCLAY_NARG], const wt_live_spans *, uint64_t generation,
  int source_abi_verified);
int wt_sfclay_capacity_init_attested(struct wt_sfclay_capacity *,
  void *const argv[WT_SFCLAY_NARG], const wt_live_spans *, uint64_t generation,
  int source_abi_verified, int caller_chain_verified);
int wt_sfclay_capacity_lookup(void *opaque, unsigned index, void *arg, size_t *bytes);
void wt_sfclay_capacity_reset(struct wt_sfclay_capacity *);
#endif
