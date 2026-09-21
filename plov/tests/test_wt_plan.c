/* Unit test for wt_plan.h (predictive PLOV node-local plan). Exit 0 = all assertions hold.
 * build: cc -O2 -Wall -Wextra -I.. test_wt_plan.c -o test_wt_plan */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../wt_plan.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void show(const char *name, const wt_plan *p)
{
    printf("%s: n=%d valid=%d mean=%.3f slots=%d\n", name, p->n, p->valid, p->mean_ms, p->nslot);
    for (int k = 0; k < p->nslot; k++)
        printf("  owner %d -> donor %d  chunks %d  %.3f ms\n", p->slot[k].owner, p->slot[k].donor, p->slot[k].chunks, p->slot[k].ms);
}

static void invariants(const char *name, const wt_plan *p, int n, const double *pred, const int *nj,
                       int rows, int maxc, double maxpct, double ovh)
{
    for (int i = 0; i < n; i++) {
        double chunk = (pred[i] > 0 && nj[i] > 0) ? pred[i] * rows / nj[i] : 0;
        int cap_pct = (pred[i] > 0 && nj[i] > 0) ? (int)(maxpct * nj[i]) / rows : 0;
        CHECK(p->chunks_out[i] <= maxc && p->chunks_out[i] <= cap_pct, "%s rank %d chunks_out %d caps %d/%d", name, i, p->chunks_out[i], maxc, cap_pct);
        CHECK(p->assigned_out_ms[i] <= p->excess_ms[i] + chunk + 1e-9, "%s rank %d out %.3f > excess %.3f + chunk", name, i, p->assigned_out_ms[i], p->excess_ms[i]);
        CHECK(p->assigned_in_ms[i] <= p->slack_ms[i] + 0.5 * 50.0 + ovh + 1e-9 || p->assigned_in_ms[i] == 0, "%s rank %d in %.3f vs slack %.3f", name, i, p->assigned_in_ms[i], p->slack_ms[i]);
        CHECK(!(p->chunks_out[i] > 0 && p->chunks_in[i] > 0), "%s rank %d both publishes and receives", name, i);
        if (pred[i] <= 0) CHECK(p->chunks_out[i] == 0 && p->chunks_in[i] == 0, "%s invalid rank %d participates", name, i);
    }
    int out = 0, in = 0;
    for (int i = 0; i < n; i++) { out += p->chunks_out[i]; in += p->chunks_in[i]; }
    CHECK(out == in, "%s chunks out %d != in %d", name, out, in);
}

int main(void)
{
    wt_plan a, b;
    /* realistic-looking node: 2 hot owners, 4 cold donors, 33 j-rows each */
    double pred1[] = {40, 30, 20, 20, 15, 15};
    int nj1[] = {33, 33, 33, 33, 33, 33};
    wt_plan_build(&a, 6, pred1, nj1, 4, 6, 0.75, 1.0);
    show("six", &a);
    invariants("six", &a, 6, pred1, nj1, 4, 6, 0.75, 1.0);
    CHECK(a.nslot > 0, "six: expected some slots");
    CHECK(fabs(a.mean_ms - 23.3333333) < 1e-5, "six: mean %.6f", a.mean_ms);
    CHECK(a.chunks_out[0] > 0 && a.chunks_out[0] >= a.chunks_out[1], "six: hottest rank should give most");
    wt_plan_build(&b, 6, pred1, nj1, 4, 6, 0.75, 1.0);
    CHECK(memcmp(&a, &b, sizeof a) == 0, "six: plan not deterministic");

    /* flat node: nothing to move */
    double flat[] = {20, 20, 20, 20};
    int nj4[] = {33, 33, 33, 33};
    wt_plan_build(&a, 4, flat, nj4, 4, 6, 0.75, 1.0);
    CHECK(a.nslot == 0, "flat: %d slots", a.nslot);

    /* single valid rank and invalid predictions */
    double one[] = {30, 0, -1};
    int nj3[] = {33, 33, 33};
    wt_plan_build(&a, 3, one, nj3, 4, 6, 0.75, 1.0);
    CHECK(a.nslot == 0 && a.valid == 1, "one: slots %d valid %d", a.nslot, a.valid);

    /* invalid rank among valid ones must not participate */
    double mixed[] = {45, 0, 15, 15, 15};
    int nj5[] = {33, 33, 33, 33, 33};
    wt_plan_build(&a, 5, mixed, nj5, 4, 6, 0.75, 1.0);
    show("mixed", &a);
    invariants("mixed", &a, 5, mixed, nj5, 4, 6, 0.75, 1.0);

    /* caps: tiny patch limits chunks by max_pct */
    double capd[] = {80, 10, 10, 10};
    int njsmall[] = {9, 33, 33, 33};
    wt_plan_build(&a, 4, capd, njsmall, 4, 6, 0.75, 0.0);
    show("cap", &a);
    invariants("cap", &a, 4, capd, njsmall, 4, 6, 0.75, 0.0);
    CHECK(a.chunks_out[0] <= 1, "cap: owner with 9 rows may publish at most 1 chunk, got %d", a.chunks_out[0]);

    /* a large node: 56 ranks, heavy tail */
    double big[56];
    int njb[56];
    for (int i = 0; i < 56; i++) { big[i] = 10.0 + (i < 6 ? 30.0 - 3 * i : 0.2 * i); njb[i] = 33; }
    wt_plan_build(&a, 56, big, njb, 4, 6, 0.75, 1.0);
    printf("big: slots=%d mean=%.3f\n", a.nslot, a.mean_ms);
    invariants("big", &a, 56, big, njb, 4, 6, 0.75, 1.0);

    printf(fails ? "test_wt_plan: %d FAILURES\n" : "test_wt_plan: all checks passed\n", fails);
    return fails != 0;
}
