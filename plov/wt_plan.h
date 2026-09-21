/* wt_plan.h -- deterministic node-local owner->donor plan for predictive PLOV (header-only, no I/O).
 *
 * Input is one node's ranks: the predicted full micro cost of each rank (per-rank EWMA, alpha 0.7,
 * frozen 2026-09-14) and its patch j-row count. Work is assumed to scale with rows inside a patch
 * (tile-level cost map, job 1074165: 13 tiles within ~6 points of uniform), so a chunk of
 * rows_per_chunk rows of rank i is predicted to cost pred_ms[i] * rows_per_chunk / nj[i].
 *
 *   mean      = mean of valid predictions
 *   excess[i] = max(pred[i] - mean, 0)     slack[i] = max(mean - pred[i], 0)
 *
 * Greedy: repeatedly take the owner with the largest remaining excess and the donor with the
 * largest remaining slack, move as many whole chunks as fit both (a donor is also charged ovh_ms per
 * chunk for the second driver entry), respecting max_chunks and max_pct * nj per owner. Ties break on
 * the lower local index, so the plan is a pure function of its inputs. A plan only decides how much
 * to publish and who is expected to take it; correctness never depends on it being followed.
 */
#ifndef WT_PLAN_H
#define WT_PLAN_H

#include <string.h>

#define WT_PLAN_MAXR 128
#define WT_PLAN_MAXS 512

typedef struct { int owner, donor, chunks; double ms; } wt_plan_slot;

typedef struct {
    int n, valid, nslot;
    double mean_ms;
    double excess_ms[WT_PLAN_MAXR], slack_ms[WT_PLAN_MAXR];
    int chunks_out[WT_PLAN_MAXR], chunks_in[WT_PLAN_MAXR];
    double assigned_out_ms[WT_PLAN_MAXR], assigned_in_ms[WT_PLAN_MAXR];
    wt_plan_slot slot[WT_PLAN_MAXS];
} wt_plan;

static void wt_plan_build(wt_plan *p, int n, const double *pred_ms, const int *nj,
                          int rows_per_chunk, int max_chunks, double max_pct, double ovh_ms)
{
    double e[WT_PLAN_MAXR], s[WT_PLAN_MAXR], chunk_ms[WT_PLAN_MAXR], sum = 0;
    int cap[WT_PLAN_MAXR];
    memset(p, 0, sizeof *p);
    if (n > WT_PLAN_MAXR) n = WT_PLAN_MAXR;
    p->n = n;
    for (int i = 0; i < n; i++)
        if (pred_ms[i] > 0 && nj[i] > 0) { sum += pred_ms[i]; p->valid++; }
    if (p->valid < 2 || rows_per_chunk < 1 || max_chunks < 1) return;
    p->mean_ms = sum / p->valid;
    for (int i = 0; i < n; i++) {
        int ok = pred_ms[i] > 0 && nj[i] > 0;
        double x = ok ? pred_ms[i] - p->mean_ms : 0;
        p->excess_ms[i] = e[i] = x > 0 ? x : 0;
        p->slack_ms[i] = s[i] = x < 0 ? -x : 0;
        chunk_ms[i] = ok ? pred_ms[i] * rows_per_chunk / nj[i] : 0;
        int by_pct = ok ? (int)(max_pct * nj[i]) / rows_per_chunk : 0;
        cap[i] = by_pct < max_chunks ? by_pct : max_chunks;
    }
    while (p->nslot < WT_PLAN_MAXS) {
        int o = -1, d = -1;
        for (int i = 0; i < n; i++)
            if (chunk_ms[i] > 0 && p->chunks_out[i] < cap[i] && e[i] >= 0.5 * chunk_ms[i] && (o < 0 || e[i] > e[o]))
                o = i;
        if (o < 0) break;
        for (int i = 0; i < n; i++)
            if (i != o && s[i] >= 0.5 * chunk_ms[o] + ovh_ms && (d < 0 || s[i] > s[d]))
                d = i;
        if (d < 0) break;
        int c_e = (int)(e[o] / chunk_ms[o] + 0.5);
        int c_s = (int)(s[d] / (chunk_ms[o] + ovh_ms) + 0.5);
        int c = c_e < c_s ? c_e : c_s;
        if (c > cap[o] - p->chunks_out[o]) c = cap[o] - p->chunks_out[o];
        if (c < 1) c = 1;
        double ms = c * chunk_ms[o];
        p->slot[p->nslot].owner = o;
        p->slot[p->nslot].donor = d;
        p->slot[p->nslot].chunks = c;
        p->slot[p->nslot].ms = ms;
        p->nslot++;
        p->chunks_out[o] += c;
        p->chunks_in[d] += c;
        p->assigned_out_ms[o] += ms;
        p->assigned_in_ms[d] += ms + c * ovh_ms;
        e[o] = e[o] > ms ? e[o] - ms : 0;
        s[d] = s[d] > ms + c * ovh_ms ? s[d] - ms - c * ovh_ms : 0;
    }
}

#endif
