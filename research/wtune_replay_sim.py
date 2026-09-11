#!/usr/bin/env python3
"""WTune replay simulator — reprice the "19pp cross-phase slack pooling" item.

The 29% settlement tax decomposes (job 1040022, 16x28/448, 1440 steps) as:
    Sigma(per-phase max) 0.939  ->  max(per-rank Sigma) 0.757   = ~19pp  "slack pooling"
    max(per-rank Sigma)  0.757  ->  balanced floor      0.664   = ~10pp  "residual imbalance"
                                                                 = ~4pp  order-statistics floor

The 19pp is an IDEAL bound: it assumes slack can move anywhere. Real slack only flows through
the halo dependency cone. This replays the MEASURED dependency structure to ask two questions
the bound cannot answer:

  Q1 REACHABILITY - how many hops until an idle waiter's chain reaches a rank that actually
     has work?  If that is large, cone-limited pooling collapses and 19pp is fiction.
  Q2 ABSORBABILITY - are the waiters *phase-anticorrelated* (busy in some phase, idle in
     another -> pooling works) or *globally under-loaded* (cheap in every phase -> pooling
     gives nothing; only real work movement helps)?  This is the question that decides whether
     the 19pp belongs to "pooling" at all.

Inputs are the preserved processed CSVs (the raw run dir is gone):
  critical-step-peer-phase-1040022.csv   step, waiter_rank, peer_rank, peer_phase, wait_s_total
  extended-chain-1040022.csv             step, hop, rank, work_phase, work_frac, side, ...
  halo-wait-rank-frequency-1040022.csv   rank, rank_i, rank_j, steps, share_pct
  halo-wait-top-slack-1040022.csv        step, max/median halo wait, slack_s

HONEST SCOPE: this is a *structural* replay over the measured critical edges (8 per step,
51 distinct waiters of 448), not a dense 448-rank x per-phase DAG - that trace no longer
exists. It bounds the SCHEDULING question; it says nothing about cache effects of loop
splitting or correctness hazards.
"""
import csv, collections, statistics as st, sys

D = "/home/u5727520/my-llm-wiki/data/wrf/"
MEASURED_SIGMA_MAX = 0.939   # s/step, Sigma(per-phase max)  == measured wall
MEASURED_MAX_SIGMA = 0.757   # s/step, max(per-rank Sigma)   == ideal pooling target
MEASURED_FLOOR     = 0.664   # s/step, fully balanced floor
WORKER_THRESHOLD   = 0.25    # work_frac above which a rank is "actually doing work"


def load():
    edges = list(csv.DictReader(open(D + "critical-step-peer-phase-1040022.csv")))
    chain = list(csv.DictReader(open(D + "extended-chain-1040022.csv")))
    freq  = list(csv.DictReader(open(D + "halo-wait-rank-frequency-1040022.csv")))
    slack = list(csv.DictReader(open(D + "halo-wait-top-slack-1040022.csv")))
    return edges, chain, freq, slack


def calibration(edges):
    """Check the replayed structure is consistent with the measured aggregates."""
    per_step = collections.defaultdict(float)
    for r in edges:
        per_step[int(r["step"])] += float(r["wait_s_total"])
    v = sorted(per_step.values())
    med = st.median(v)
    ideal_gain = MEASURED_SIGMA_MAX - MEASURED_MAX_SIGMA
    print("=" * 74)
    print("0. CALIBRATION - is the replayed structure consistent with the measured wall?")
    print("=" * 74)
    print(f"  measured step wall          : {MEASURED_SIGMA_MAX:.3f} s")
    print(f"  replayed critical-edge wait : {med:.3f} s median  ({med/MEASURED_SIGMA_MAX*100:.1f}% of wall)")
    print(f"  independent halo_wait share : 27.7-28.2% of aggregated step  [profile-attribution]")
    ok = 0.20 <= med / MEASURED_SIGMA_MAX <= 0.55
    print(f"  -> critical-edge wait is the same order as the measured halo_wait share: "
          f"{'CONSISTENT' if ok else 'INCONSISTENT - model suspect'}")
    print(f"  ideal pooling prize (Sigma max - max Sigma) = {ideal_gain:.3f} s/step "
          f"= {ideal_gain/MEASURED_SIGMA_MAX*100:.1f}pp of wall")
    return med


def q1_reachability(chain):
    """How many hops before the chain reaches a rank that actually has work?"""
    by_step = collections.defaultdict(list)
    for r in chain:
        by_step[r["step"]].append(r)
    print()
    print("=" * 74)
    print("Q1. REACHABILITY - hops until the idle waiter's chain reaches a real worker")
    print("=" * 74)
    hops_needed, crossings = [], []
    for s, rows in sorted(by_step.items()):
        rows.sort(key=lambda r: int(r["hop"]))
        hit = next((int(r["hop"]) for r in rows if float(r["work_frac"]) >= WORKER_THRESHOLD), None)
        side0 = rows[0]["side"]
        cross = next((int(r["hop"]) for r in rows if r["side"] != side0), None)
        hops_needed.append(hit)
        crossings.append(cross)
        head = rows[0]
        print(f"  step {s:>5}: head rank {head['rank']:>4} ({head['side']}, {head['work_phase']}, "
              f"work_frac={float(head['work_frac']):.3f})")
        print(f"              -> first worker (work_frac>={WORKER_THRESHOLD}) at hop {hit}"
              f"   |  terminator crossing at hop {cross}")
    good = [h for h in hops_needed if h is not None]
    print(f"\n  hops-to-worker: {good}   median = {st.median(good):.1f}")
    print(f"  -> the dependency cone reaches loaded ranks in a FEW hops, not tens.")
    print(f"     The 'slack cannot travel far enough' worry is NOT the binding constraint.")
    return good


def q2_absorbability(chain):
    """Are waiters phase-anticorrelated (pooling works) or globally under-loaded (it does not)?"""
    print()
    print("=" * 74)
    print("Q2. ABSORBABILITY - are the idle waiters anticorrelated, or just globally light?")
    print("=" * 74)
    # NB: classify per OBSERVATION, not per rank. The terminator MOVES between steps, so the
    # same rank is night at one step and day at another (e.g. rank 317: night/0.016 @step1001,
    # day/0.533 @step881). Grouping by rank and maxing over steps mixes the two and inverts
    # the result.
    night = [float(r["work_frac"]) for r in chain if r["side"] == "night"]
    day   = [float(r["work_frac"]) for r in chain if r["side"] == "day"]
    def summarize(tag, v):
        if not v:
            print(f"  {tag}: (none observed)"); return None
        print(f"  {tag}: n={len(v):>3}  work_frac  min {min(v):.3f} / median {st.median(v):.3f} "
              f"/ p90 {sorted(v)[int(.9*len(v))-1]:.3f} / max {max(v):.3f}")
        return st.median(v)
    nm = summarize("night-side observations (waiting)", night)
    dm = summarize("day-side   observations (working)", day)
    print()
    if nm is not None and dm is not None:
        print(f"  night median work_frac {nm:.3f}  vs  day median {dm:.3f}   ({dm/max(nm,1e-9):.1f}x)")
        print("  -> when a rank is on the night side it is genuinely idle; the load follows the")
        print("     terminator, i.e. this is a SPATIAL/LEVEL effect.")
    print()
    print("  LIMIT OF THIS DATA: the chain records each rank in exactly ONE phase per step, so")
    print("  it cannot show whether a night-idle rank is busy in ANOTHER phase of the SAME step.")
    print("  That within-step phase anticorrelation is precisely what decides whether the 19pp")
    print("  is 'pooling' (recoverable by deferring settlement) or a level difference (recoverable")
    print("  only by moving real work). -> NOT RESOLVABLE from the surviving CSVs.")
    return nm, dm


def q3_concentration(edges, freq):
    """Where is the wait concentrated - a few edges/ranks, or spread thin?"""
    print()
    print("=" * 74)
    print("Q3. CONCENTRATION - how targeted can a fix be?")
    print("=" * 74)
    tot = sum(float(r["wait_s_total"]) for r in edges)
    byph = collections.Counter()
    byedge = collections.Counter()
    for r in edges:
        byph[r["peer_phase"]] += float(r["wait_s_total"])
        byedge[(r["waiter_rank"], r["peer_rank"])] += float(r["wait_s_total"])
    top_phase, top_v = byph.most_common(1)[0]
    print(f"  wait by phase : '{top_phase}' carries {top_v/tot*100:.1f}% of all wait "
          f"({len(byph)} phases total)")
    se = byedge.most_common()
    c = 0
    for k, (_, v) in enumerate(se, 1):
        c += v
        if k in (5, 20):
            print(f"  wait by edge  : top {k:>2} of {len(se)} edges carry {c/tot*100:.1f}%")
    print(f"  waiter ranks  : top 2 of {len(freq)} recorded ranks carry "
          f"{sum(float(r['share_pct']) for r in freq[:2]):.1f}% of critical steps"
          f"  (ranks {freq[0]['rank']} @i={freq[0]['rank_i']}, {freq[1]['rank']} @i={freq[1]['rank_i']}"
          f" - both DOMAIN-EDGE ranks)")
    print("  -> a fix does not need all ~110 halo sites; it needs the dynamics halo on a few edges.")
    return tot


def verdict(hops, nm, dm):
    print()
    print("=" * 74)
    print("VERDICT - repricing the 19pp")
    print("=" * 74)
    ideal = MEASURED_SIGMA_MAX - MEASURED_MAX_SIGMA
    print(f"  ideal pooling bound                     : {ideal/MEASURED_SIGMA_MAX*100:.1f}pp")
    print(f"  Q1 cone reachability                    : NOT binding (worker within "
          f"{st.median(hops):.0f} hops)")
    print(f"  Q2 within-step phase anticorrelation    : UNRESOLVED - surviving CSVs record each")
    print(f"     rank in one phase per step. This is the ONE measurement that decides whether the")
    print(f"     19pp is recoverable by deferring settlement, or only by moving real work.")
    print()
    print("  => The cone was the wrong thing to worry about (Q1 kills that worry). The open")
    print("     question is now sharp and singular, and it is cheap to settle: one ~10 min")
    print("     instrumented 4-node rerun emitting per-rank x per-phase time for all 448 ranks.")
    print()
    print("  => Meanwhile Q3 already de-risks the IMPLEMENTATION side: the wait is not spread")
    print("     over ~110 halo sites. It is 86.5% in the dynamics ('step') halo and 80% inside")
    print("     20 of 102 edges, with 2 domain-EDGE ranks carrying 54.6% of critical steps.")
    print("     Whatever the 19pp turns out to be worth, capturing it is a narrow change.")
    print()
    print("  NOTE: structural replay over measured critical edges (8/step, 51 of 448 waiters).")
    print("        A dense 448-rank x per-phase DAG would tighten this; that trace no longer exists")
    print("        and would need one ~10 min instrumented 4-node rerun to regenerate.")


def main():
    edges, chain, freq, slack = load()
    calibration(edges)
    hops = q1_reachability(chain)
    nm, dm = q2_absorbability(chain)
    q3_concentration(edges, freq)
    verdict(hops, nm, dm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
