#!/usr/bin/env python3
"""predictor/stability.py -- how much stability margin does this case have left?

The tuner has been discovering infeasible settings the expensive way: P03 (time_step x1.2) crashed
conus25/conus12 and timed out maria1km, and D01 (time_step_sound=3) hung conus12 for the full 900 s
timeout, twice. Each of those costs a 4-node slot for the whole timeout. Most of that is avoidable
from the namelist alone, because the acoustic sub-step count and its CFL number are pure arithmetic
on (dt, dx, dy).

What this module DOES cover, from WRF source:

  `num_sound_steps` when time_step_sound == 0 (the default = "auto"), replicated exactly from
  dyn_em/solve_em.F:451-464:

      num_sound_steps = max( 2 * ( INT(300 * dt / min(dx,dy) - 0.01) + 1 ), 4 )

  ...which, as WRF's own comment says, "returns even numbers only". All thirteen registered cases
  land on 4. The acoustic time step is then dts = dt / num_sound_steps, and the acoustic Courant
  number is c_s * dts / min(dx,dy) with c_s ~ 340 m/s (dry sound speed; WRF's split-explicit scheme
  integrates the acoustic modes on this sub-step, which is the entire reason the sub-step exists).

  Measured margins at each case's own settings, which is the point of the module: SC26-P1's d01 sits
  at 0.094 -- 9% of the way to the classical limit -- because its dt (10 s) is ~5x more conservative
  than the usual 6*dx_km rule at 9 km, while karina sits at 0.425 and conus12 at 0.510. A tuner that
  cannot see that difference has to find it by crashing.

What this module does NOT cover, stated plainly so nobody reads more into it:

  * The P03 failures are NOT explained by the acoustic number. conus25 at time_step 15->18 lands at
    acoustic CFL 0.408 (the auto formula raises num_sound_steps to 6 at the larger dt), which is
    LOWER than karina's 0.425 at its own working settings -- and karina survived P03 while conus25
    crashed. The binding constraint there is advective/vertical, i.e. it depends on the flow, not
    only on the grid. `advective_cfl` is provided for when a wind estimate exists (from a probe or
    from wrfinput), and `assess` reports it as unknown when no wind is supplied. Do not present an
    acoustic-only verdict as a stability guarantee.
  * Odd `time_step_sound` values are flagged unsupported rather than scored: WRF's own auto path
    only ever produces even counts, and the one odd value we measured (3, on conus12) hung before
    the first step in both replicates (job 1080698) while the same setting ran fine on karina.
"""
from __future__ import annotations

import math

# Dry-air sound speed used for the acoustic Courant number. WRF's split-explicit scheme integrates
# acoustic modes on the sub-step; 340 m/s is the standard reference value for this estimate.
SOUND_SPEED = 340.0

# Classical stability limit for the acoustic sub-step is 1; WRF is run well inside it. These are
# advisory bands for a recommender, not WRF-enforced thresholds.
CFL_SAFE = 0.60          # plenty of headroom: an aggressive setting is worth proposing
CFL_TIGHT = 0.85         # close to the limit: propose nothing that raises it


def auto_sound_steps(dt, dx, dy=None):
    """WRF's own `num_sound_steps` for time_step_sound=0, dyn_em/solve_em.F:451-464. Returns even
    numbers only, minimum 4 -- the `- 0.01` and the INT truncation are both WRF's, not a rounding
    convenience: at exactly 300*dt/spacing == 1.0 (SC26-P1's d02) the truncation is what keeps the
    result at 4 rather than 6."""
    spacing = min(dx, dy) if dy else dx
    return max(2 * (int(300.0 * dt / spacing - 0.01) + 1), 4)


def sound_steps(dt, dx, dy=None, time_step_sound=0):
    """The count WRF will actually use: the namelist value when nonzero, else the auto formula."""
    return auto_sound_steps(dt, dx, dy) if not time_step_sound else int(time_step_sound)


def acoustic_cfl(dt, dx, dy=None, time_step_sound=0, sound_speed=SOUND_SPEED):
    """c_s * (dt / num_sound_steps) / min(dx,dy)."""
    nss = sound_steps(dt, dx, dy, time_step_sound)
    spacing = min(dx, dy) if dy else dx
    return sound_speed * (float(dt) / nss) / spacing


def advective_cfl(dt, dx, wind_max, dy=None):
    """u_max * dt / min(dx,dy). Requires a wind estimate; None when one is not available -- this is
    the term that decides whether a time_step increase survives, and it cannot be had from the
    namelist alone."""
    if wind_max is None:
        return None
    spacing = min(dx, dy) if dy else dx
    return float(wind_max) * float(dt) / spacing


def safe_sound_steps(dt, dx, dy=None, cfl_limit=CFL_SAFE, sound_speed=SOUND_SPEED):
    """The smallest EVEN sub-step count whose acoustic CFL stays <= cfl_limit, never below 2.
    Even-only because WRF's auto path only produces even counts and the one odd value measured hung
    (conus12 with time_step_sound=3, job 1080698)."""
    spacing = min(dx, dy) if dy else dx
    n = 2
    while n < 64:
        if sound_speed * (float(dt) / n) / spacing <= cfl_limit:
            return n
        n += 2
    return n


def assess(dt, dx, dy=None, time_step_sound=0, wind_max=None, cfl_limit=CFL_SAFE):
    """One domain's stability picture: what WRF will use, where it sits, and what could be proposed.

    Returns a dict with `sound_steps`, `acoustic_cfl`, `advective_cfl` (None without a wind
    estimate), `headroom` (cfl_limit / acoustic_cfl), `safe_sound_steps`, `sound_steps_saving`
    (fraction of acoustic sub-step work a tuner could remove, 0 when there is nothing to take), and
    `notes` listing every reason the verdict is incomplete."""
    nss = sound_steps(dt, dx, dy, time_step_sound)
    ac = acoustic_cfl(dt, dx, dy, time_step_sound)
    proposed = safe_sound_steps(dt, dx, dy, cfl_limit)
    notes = []
    if time_step_sound and int(time_step_sound) % 2:
        notes.append("time_step_sound=%s is ODD: WRF's auto path never produces an odd count, and "
                     "the one odd value measured (3 on conus12) hung before step 1 -- unsupported"
                     % time_step_sound)
    adv = advective_cfl(dt, dx, wind_max, dy)
    if adv is None:
        notes.append("no wind estimate: the advective/vertical limit is UNKNOWN, and that is the "
                     "term that decided the measured time_step failures (P03 crashed conus25/conus12 "
                     "at acoustic CFL values LOWER than karina's, which survived)")
    elif adv > 1.0:
        notes.append("advective CFL %.2f exceeds 1 -- expect a CFL abort" % adv)
    if ac >= CFL_TIGHT:
        notes.append("acoustic CFL %.2f is within %.2f of the classical limit: do not propose "
                     "anything that raises it" % (ac, 1.0 - ac))
    saving = max(0.0, 1.0 - proposed / float(nss))
    return dict(sound_steps=nss, acoustic_cfl=ac, advective_cfl=adv,
                headroom=cfl_limit / ac if ac else math.inf,
                safe_sound_steps=proposed, sound_steps_saving=saving, notes=notes)


def assess_case(case, wind_max=None, cfl_limit=CFL_SAFE):
    """Per-domain assessment for a campaign/cases.json entry. A nest inherits its parent's dt
    divided by parent_time_step_ratio (SC26-P1 sets that ratio to 1, so both its domains run at
    dt=10 and the 3 km nest carries 3x the parent's acoustic Courant number) -- when the ratio is
    absent the parent dt is used for every domain and a note says so."""
    dt = case.get("dt")
    dxs = case.get("dx_list") or [case.get("dx")]
    dys = case.get("dy_list") or [case.get("dy")] * len(dxs)
    ratios = case.get("parent_time_step_ratio_list")
    tss = case.get("time_step_sound_list") or [case.get("time_step_sound", 0)] * len(dxs)
    out = []
    dt_d = float(dt)
    for i, dx in enumerate(dxs):
        if i and ratios and i < len(ratios):
            dt_d = dt_d / float(ratios[i])
        dy = dys[i] if i < len(dys) and dys[i] else dx
        a = assess(dt_d, dx, dy, tss[i] if i < len(tss) else 0, wind_max, cfl_limit)
        a["domain"] = i + 1
        a["dt"] = dt_d
        a["dx"] = dx
        if i and not ratios:
            a["notes"].append("no parent_time_step_ratio recorded for this case: assumed the nest "
                              "runs at the parent dt (true for SC26-P1, false for the usual 3:1)")
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# The per-domain form of a time_step_sound template (2026-09-17).
#
# campaign/configs.py's 'set' op writes ONE value to every active domain, which is right for every
# key WRF requires to be identical across domains and wrong for the acoustic sub-step count: the
# margin is a property of (dt, dx) and a nest does not share its parent's. SC26-P1 is the measured
# example -- parent_time_step_ratio=1, so d01 (9 km) and d02 (3 km) run at the same dt=10 and the
# nest carries exactly 3x the parent's acoustic Courant number (0.094 vs 0.283). One value for both
# either wastes the parent's margin or overspends the nest's.
#
# The values are COMPUTED from safe_sound_steps per domain, never hard-coded: at cfl_limit=0.50
# SC26-P1 resolves to [2, 4]; at the module default CFL_SAFE=0.60 it resolves to [2, 2] (d02's
# n=2 lands at 0.567, still under 0.60) and the variant is SKIPPED as uniform. The threshold is the
# caller's knob, and the skip rule is what keeps a uniform answer out of the per-domain form.
#
# Direction: safe_sound_steps returns the SMALLEST safe even count, so this variant does LESS
# acoustic work than the auto 4 -- it is a COST lever, the opposite of A02's own accuracy direction
# (A02 raises the count to 6). The returned variant therefore carries direction='cost', or
# --objective accuracy would mistake a coarser acoustic integration for a converged reference.
# ---------------------------------------------------------------------------
def effective_max_dom(case, max_dom=None):
    """How many domains this case actually runs: an explicit `max_dom` wins, then a
    namelist_overrides.max_dom (lafe trims 3 -> 1), then the case's own max_dom, then 1. Mirrors
    campaign/configs.py's _case_effective_max_dom precedence."""
    if max_dom is not None:
        return int(max_dom)
    overrides = case.get("namelist_overrides") or {}
    if "max_dom" in overrides:
        try:
            return int(overrides["max_dom"])
        except (TypeError, ValueError):
            pass
    try:
        return int(case.get("max_dom") or 1)
    except (TypeError, ValueError):
        return 1


def per_domain_sound_steps(case, max_dom=None, cfl_limit=CFL_SAFE):
    """(values, None) -- safe_sound_steps for every ACTIVE domain of `case` -- or (None, reason).

    Needs the per-domain grid, i.e. `dx_list` (plus `dy_list`/`parent_time_step_ratio_list` when
    they differ) on the case entry, the shape assess_case already reads. NOTE: no
    campaign/cases.json entry carries those keys today -- they record a scalar `dx` and put the
    nest geometry in `notes` -- so a registered nested case returns the reason string rather than a
    silently parent-only answer."""
    n = effective_max_dom(case, max_dom)
    doms = assess_case(case, cfl_limit=cfl_limit)
    if len(doms) < n:
        return None, ("case declares max_dom=%d but carries per-domain grid information for only "
                      "%d domain(s): add dx_list (and dy_list/parent_time_step_ratio_list when "
                      "they differ) to the case entry -- campaign/cases.json records only a scalar "
                      "dx today" % (n, len(doms)))
    return [d["safe_sound_steps"] for d in doms[:n]], None


def per_domain_variant(template, case, max_dom=None, cfl_limit=CFL_SAFE, key="time_step_sound"):
    """(variant_template, values) or (None, reason): the optional per-domain form of an acoustic
    sub-step template (A02), with its single 'set' edit on `key` rewritten to the
    `values_by_domain` form campaign/configs.py understands.

    SKIPPED, with a reason instead of a template, when the case runs one domain (the scalar form
    already says everything), when every domain resolves to the same count (same reason), or when
    the case carries no per-domain grid. The variant's `direction` is 'cost': fewer sub-steps is
    less acoustic work, the mirror of A02's own accuracy direction."""
    n = effective_max_dom(case, max_dom)
    if n < 2:
        return None, ("single-domain case (max_dom=%d): a per-domain variant cannot say anything "
                      "the scalar value does not" % n)
    edits = list((template or {}).get("edits") or [])
    targets = [i for i, e in enumerate(edits)
               if e.get("op") == "set" and e.get("key") == key]
    if len(targets) != 1:
        return None, ("template carries %d 'set' edit(s) on %r -- the per-domain variant is defined "
                      "for exactly one" % (len(targets), key))
    values, reason = per_domain_sound_steps(case, n, cfl_limit)
    if values is None:
        return None, reason
    if len(set(values)) == 1:
        return None, ("every active domain resolves to %s=%d at cfl_limit=%.2f: the scalar form "
                      "already expresses it, so there is no per-domain variant to run"
                      % (key, values[0], cfl_limit))
    new_edits = [dict(e) for e in edits]
    edit = new_edits[targets[0]]
    edit.pop("value", None)
    edit["per_domain"] = True
    edit["values_by_domain"] = list(values)
    variant = dict(template)
    variant["edits"] = new_edits
    variant["direction"] = "cost"
    variant["description"] = (
        "PER-DOMAIN VARIANT (predictor/stability.py per_domain_variant, cfl_limit=%.2f): %s = %s, "
        "the smallest EVEN acoustic sub-step count per domain whose acoustic CFL stays <= %.2f "
        "(computed from each domain's own dt/dx, never hard-coded). This is a COST lever -- fewer "
        "sub-steps than the auto 4 -- and therefore the OPPOSITE direction to the template it is "
        "derived from. Base template: %s"
        % (cfl_limit, key, ", ".join(str(v) for v in values), cfl_limit,
           (template or {}).get("description", "")))
    return variant, values


def _cli(argv=None):
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "campaign"))
    import configs as campaign_configs

    ap = argparse.ArgumentParser(prog="stability.py")
    ap.add_argument("case", nargs="?", help="a campaign/cases.json case id (default: every case)")
    ap.add_argument("--wind-max", type=float, default=None)
    ap.add_argument("--cfl-limit", type=float, default=CFL_SAFE)
    a = ap.parse_args(argv)
    ids = [a.case] if a.case else sorted(campaign_configs.CASES)
    print("%-11s %3s %7s %6s %4s %8s %6s %8s" %
          ("case", "dom", "dx", "dt", "nss", "acou_cfl", "safe", "saving"))
    for cid in ids:
        case = campaign_configs.CASES.get(cid)
        if case is None:
            print("unknown case %r" % cid)
            continue
        for d in assess_case(case, a.wind_max, a.cfl_limit):
            print("%-11s %3d %7.0f %6.1f %4d %8.3f %6d %7.0f%%" %
                  (cid, d["domain"], d["dx"], d["dt"], d["sound_steps"], d["acoustic_cfl"],
                   d["safe_sound_steps"], 100 * d["sound_steps_saving"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
