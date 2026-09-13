#include "wt_kf_capacity.h"
#include <limits.h>
#include <stdint.h>
#include <string.h>

static int span_type(unsigned i)
{
  if (i == WT_KF_CU_ACT_FLAG) return WT_SPAN_LOGICAL;
  return WT_SPAN_REAL;
}
static int caller_proven_automatic(unsigned i)
{
  return i == WT_KF_TH || i == WT_KF_PI || i == WT_KF_DZ8W ||
    i == WT_KF_CU_ACT_FLAG;
}
static int read_i32(const void *p, int32_t *out)
{
  if (!p || !out) return 0;
  memcpy(out, p, sizeof *out);
  return 1;
}
static int extent(int32_t low, int32_t high, size_t *out)
{
  int64_t n = (int64_t)high - (int64_t)low + 1;
  if (n <= 0 || (uint64_t)n > SIZE_MAX) return 0;
  *out = (size_t)n;
  return 1;
}
static int multiply(size_t a, size_t b, size_t *out)
{
  if (!a || !b || a > SIZE_MAX / b) return 0;
  *out = a * b;
  return 1;
}
static int required_bytes(unsigned shape, size_t ni, size_t nj, size_t nk,
                          size_t *out)
{
  size_t elements;
  if (!multiply(ni, nj, &elements)) return 0;
  if (shape == WT_KF_3D && !multiply(elements, nk, &elements)) return 0;
  return multiply(elements, 4, out);
}
static int checked_shape(void *const argv[WT_KF_NARG], size_t *two, size_t *three)
{
  int32_t ims, ime, jms, jme, kms, kme;
  size_t ni, nj, nk;
  if (!read_i32(argv[WT_KF_IMS], &ims) || !read_i32(argv[WT_KF_IME], &ime) ||
      !read_i32(argv[WT_KF_JMS], &jms) || !read_i32(argv[WT_KF_JME], &jme) ||
      !read_i32(argv[WT_KF_KMS], &kms) || !read_i32(argv[WT_KF_KME], &kme) ||
      !extent(ims, ime, &ni) || !extent(jms, jme, &nj) || !extent(kms, kme, &nk))
    return 0;
  return required_bytes(WT_KF_2D, ni, nj, nk, two) &&
         required_bytes(WT_KF_3D, ni, nj, nk, three);
}
static int live(const wt_live_spans *spans, uint64_t generation, void *arg,
                unsigned index, size_t needed)
{
  size_t remaining = 0;
  return arg && wt_live_spans_lookup(spans, generation, arg, span_type(index),
    4, needed, &remaining) == WT_SPAN_OK;
}
void wt_kf_capacity_reset(struct wt_kf_capacity *provider)
{
  if (provider) memset(provider, 0, sizeof *provider);
}
int wt_kf_capacity_init_attested(struct wt_kf_capacity *provider,
                                 void *const argv[WT_KF_NARG],
                                 const wt_live_spans *spans,
                                 uint64_t generation,
                                 int source_abi_verified,
                                 int caller_chain_verified)
{
  size_t two, three, needed;
  int32_t edrates;
  unsigned i;
  if (!provider) return 0;
  wt_kf_capacity_reset(provider);
  if (!argv || !spans || !source_abi_verified ||
      !wt_live_spans_is_current(spans, generation)) return 0;
  for (i = 0; i < WT_KF_NARG; ++i) if (wt_kf_shape[i] == WT_KF_SCALAR) {
    if (!argv[i]) goto fail;
    provider->expected[i] = argv[i]; provider->bytes[i] = 4;
  }
  if (!checked_shape(argv, &two, &three) ||
      !read_i32(argv[WT_KF_KF_EDRATES], &edrates)) goto fail;
  for (i = 0; i < WT_KF_NARG; ++i) {
    if (wt_kf_shape[i] == WT_KF_SCALAR) continue;
    provider->expected[i] = argv[i];
    if (i == WT_KF_SHALL && !argv[i]) { provider->bytes[i] = 0; continue; }
    if (!argv[i]) goto fail;
    needed = wt_kf_shape[i] == WT_KF_3D ? three : two;
    if (i == WT_KF_PRATEC) { provider->bytes[i] = two; continue; }
    if (caller_chain_verified && caller_proven_automatic(i)) {
      provider->bytes[i] = needed; continue;
    }
    if (i >= WT_KF_UDR_KF && i <= WT_KF_TIMEC_KF && edrates != 1) {
      provider->bytes[i] = 4; continue;
    }
    if (!live(spans, generation, argv[i], i, needed)) goto fail;
    provider->bytes[i] = needed;
  }
  provider->spans = spans; provider->generation = generation; provider->ready = 1;
  return 1;
fail:
  wt_kf_capacity_reset(provider);
  return 0;
}
int wt_kf_capacity_init(struct wt_kf_capacity *provider,
                        void *const argv[WT_KF_NARG],
                        const wt_live_spans *spans, uint64_t generation,
                        int source_abi_verified)
{
  return wt_kf_capacity_init_attested(provider, argv, spans, generation,
    source_abi_verified, 0);
}
int wt_kf_capacity_lookup(void *opaque, unsigned index, void *arg, size_t *bytes)
{
  struct wt_kf_capacity *provider = opaque;
  if (bytes) *bytes = 0;
  if (!provider || !bytes || !provider->ready || index >= WT_KF_NARG ||
      !wt_live_spans_is_current(provider->spans, provider->generation) ||
      provider->expected[index] != arg) return 0;
  *bytes = provider->bytes[index];
  return 1;
}
