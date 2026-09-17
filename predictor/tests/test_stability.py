"""Retro-validation of predictor/stability.py against what the cluster actually did.

Every number here is either read out of WRF's own source (dyn_em/solve_em.F:451-464) or measured by
a job whose id is cited. The point of the module is to predict feasibility BEFORE burning a 900 s
timeout on it, so the tests are written as "would this have told us what we had to learn the hard
way?".
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stability as st  # noqa: E402


# (case, dt, dx, dy) for every registered case, transcribed from campaign/cases.json
CASES = [
    ("karina", 15, 3000, 3000), ("conus25", 15, 2500, 2500), ("conus12", 72, 12000, 12000),
    ("maria1km", 25, 5000, 5000), ("lafe", 10, 2500, 2500), ("isc21", 15, 3000, 3000),
    ("zhengzhou", 36, 9000, 9000), ("bigchill", 15, 3000, 3000), ("elevconv", 15, 3000, 3000),
    ("marshall", 6, 1000, 1000), ("matthew", 150, 27000, 27000), ("severe2023", 180, 30000, 30000),
]


def test_auto_formula_gives_four_for_every_registered_case():
    """WRF's auto path returns even numbers only, minimum 4, and all twelve registered cases land
    on exactly 4 -- which is why 'reduce the sub-step count' is a real lever nowhere measured yet."""
    for name, dt, dx, dy in CASES:
        assert st.auto_sound_steps(dt, dx, dy) == 4, name


def test_auto_formula_matches_wrf_source_where_truncation_decides():
    # SC26-P1 d02: 300*10/3000 == 1.0 exactly; WRF's "- 0.01" then INT() keeps this at 4, not 6.
    assert st.auto_sound_steps(10, 3000) == 4
    # a genuinely large dt/dx ratio must climb, still even
    assert st.auto_sound_steps(90, 3000) == 18
    assert st.auto_sound_steps(60, 3000) == 12
    assert st.auto_sound_steps(300, 3000) == 60


def test_sc26_p1_has_by_far_the_most_acoustic_headroom():
    """The reason to look at P1's dynamics at all: dt=10 s at 9 km is ~5x more conservative than the
    usual 6*dx_km rule, so d01 sits at 9% of the classical limit while karina sits at 43%."""
    p1_d01 = st.acoustic_cfl(10, 9000)
    p1_d02 = st.acoustic_cfl(10, 3000)          # parent_time_step_ratio = 1, so the nest shares dt
    karina = st.acoustic_cfl(15, 3000)
    conus12 = st.acoustic_cfl(72, 12000)
    assert p1_d01 == pytest.approx(0.0944, abs=1e-3)
    assert p1_d02 == pytest.approx(0.2833, abs=1e-3)
    assert karina == pytest.approx(0.425, abs=1e-3)
    assert conus12 == pytest.approx(0.51, abs=1e-3)
    assert p1_d01 < karina / 4


def test_p1_can_halve_its_sub_steps_and_karina_cannot():
    """safe_sound_steps is the actionable output: 2 for P1's parent (CFL 0.19), no reduction for
    karina (halving would put it at 0.85)."""
    assert st.safe_sound_steps(10, 9000) == 2
    assert st.acoustic_cfl(10, 9000, time_step_sound=2) == pytest.approx(0.189, abs=1e-3)
    assert st.safe_sound_steps(15, 3000) == 4      # karina stays where it is
    a = st.assess(10, 9000)
    assert a["sound_steps_saving"] == pytest.approx(0.5)
    assert st.assess(15, 3000)["sound_steps_saving"] == 0.0


def test_odd_sound_step_counts_are_flagged_unsupported():
    """conus12 with time_step_sound=3 hung before step 1 in both replicates (job 1080698) while the
    same value ran on karina; WRF's auto path never produces an odd count."""
    notes = st.assess(72, 12000, time_step_sound=3)["notes"]
    assert any("ODD" in n for n in notes)
    assert not any("ODD" in n for n in st.assess(72, 12000, time_step_sound=4)["notes"])


def test_acoustic_alone_does_not_explain_the_measured_time_step_failures():
    """Honesty check, and the reason `assess` always notes the missing wind estimate: P03 raised
    time_step x1.2; conus25 CRASHED at an acoustic CFL lower than karina's, and karina SURVIVED.
    Anything that reported an acoustic-only verdict as a stability guarantee would have been wrong."""
    conus25_after = st.acoustic_cfl(18, 2500)     # crashed (job 1077169)
    karina_working = st.acoustic_cfl(15, 3000)    # survived P03 (job 1077168)
    assert conus25_after < karina_working
    assert any("UNKNOWN" in n for n in st.assess(18, 2500)["notes"])


def test_advective_cfl_needs_a_wind_estimate_and_flags_the_limit():
    assert st.advective_cfl(10, 9000, None) is None
    assert st.advective_cfl(10, 9000, 40) == pytest.approx(0.0444, abs=1e-4)
    notes = st.assess(72, 12000, wind_max=200)["notes"]     # 200 m/s * 72 / 12000 = 1.2
    assert any("exceeds 1" in n for n in notes)


def test_assess_case_uses_the_parent_time_step_ratio_for_nests():
    p1 = dict(dt=10, dx=9000, dy=9000, dx_list=[9000, 3000], dy_list=[9000, 3000],
              parent_time_step_ratio_list=[1, 1])
    doms = st.assess_case(p1)
    assert [d["dt"] for d in doms] == [10, 10]
    assert doms[1]["acoustic_cfl"] > doms[0]["acoustic_cfl"] * 2.9   # 3x finer mesh, same dt

    conventional = dict(p1, parent_time_step_ratio_list=[1, 3])
    doms = st.assess_case(conventional)
    assert doms[1]["dt"] == pytest.approx(10 / 3.0)

    single = dict(dt=15, dx=3000, dy=3000)
    assert len(st.assess_case(single)) == 1


# ---------------------------------------------------------------------------
# per_domain_variant (2026-09-17): the optional per-domain form of A02's time_step_sound edit.
# Values are COMPUTED from each domain's own (dt, dx); nothing here hard-codes SC26-P1's answer.
# ---------------------------------------------------------------------------
SC26_P1 = dict(dt=10, dx=9000, dy=9000, max_dom=2, dx_list=[9000, 3000], dy_list=[9000, 3000],
               parent_time_step_ratio_list=[1, 1])


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _a02():
    import json
    path = os.path.join(_REPO_ROOT, "campaign", "physics_templates.json")
    with open(path) as fh:
        return json.load(fh)["templates"]["A02"]


def test_per_domain_sound_steps_is_computed_per_domain_not_assumed():
    """SC26-P1 runs both domains at dt=10 (parent_time_step_ratio=1), so the 3x mesh refinement is
    the ONLY thing separating them -- and it is enough to separate the safe sub-step count once the
    limit is tight enough to bind on the nest. Both numbers come out of safe_sound_steps."""
    tight, reason = st.per_domain_sound_steps(SC26_P1, cfl_limit=0.50)
    assert reason is None
    assert tight == [st.safe_sound_steps(10, 9000, cfl_limit=0.50),
                     st.safe_sound_steps(10, 3000, cfl_limit=0.50)]
    assert tight == [2, 4]


def test_per_domain_variant_rewrites_a02_as_values_by_domain_in_the_cost_direction():
    variant, values = st.per_domain_variant(_a02(), SC26_P1, cfl_limit=0.50)
    assert variant is not None
    assert values == [2, 4]
    edit = variant["edits"][0]
    assert edit["key"] == "time_step_sound"
    assert edit["values_by_domain"] == [2, 4]
    assert "value" not in edit           # the scalar form is replaced, not shadowed
    assert edit["per_domain"] is True
    # fewer sub-steps than the auto 4 is LESS acoustic work: the opposite of A02's own direction
    assert _a02()["direction"] == "accuracy"
    assert variant["direction"] == "cost"
    assert _a02()["edits"][0]["value"] == 6      # the source template is not mutated


def test_per_domain_variant_output_actually_applies_through_configs():
    sys.path.insert(0, os.path.join(_REPO_ROOT, "campaign"))
    import configs as campaign_configs
    variant, _values = st.per_domain_variant(_a02(), SC26_P1, cfl_limit=0.50)
    nl = "&domains\n time_step = 10, max_dom = 2,\n/\n&dynamics\n diff_opt = 1, 1,\n/\n"
    out, summary = campaign_configs.apply_physics_template(nl, variant, max_dom=2)
    assert "time_step_sound = 2, 4," in out
    assert summary[0]["new"] == ["2", "4"]


def test_per_domain_variant_is_skipped_when_every_domain_resolves_the_same():
    """At the module default CFL_SAFE=0.60 SC26-P1's nest still fits in 2 sub-steps (CFL 0.567), so
    both domains resolve to 2 and there is nothing per-domain to say. The threshold is the caller's
    knob; the skip rule is what keeps a uniform answer out of the per-domain form."""
    variant, reason = st.per_domain_variant(_a02(), SC26_P1, cfl_limit=st.CFL_SAFE)
    assert variant is None
    assert "time_step_sound=2" in reason and "already expresses it" in reason


def test_per_domain_variant_is_skipped_for_a_single_domain_case():
    karina = dict(dt=15, dx=3000, dy=3000, max_dom=1)
    variant, reason = st.per_domain_variant(_a02(), karina)
    assert variant is None
    assert "single-domain" in reason


def test_per_domain_variant_reports_a_case_with_no_per_domain_grid():
    """No campaign/cases.json entry carries dx_list today -- they record a scalar dx and put the
    nest geometry in notes -- so a registered nested case must say so instead of silently
    answering for the parent only."""
    scalar_only = dict(dt=10, dx=9000, dy=9000, max_dom=2)
    variant, reason = st.per_domain_variant(_a02(), scalar_only, cfl_limit=0.50)
    assert variant is None
    assert "dx_list" in reason


def test_effective_max_dom_follows_the_namelist_override():
    assert st.effective_max_dom(dict(max_dom=3)) == 3
    assert st.effective_max_dom(dict(max_dom=3, namelist_overrides={"max_dom": 1})) == 1
    assert st.effective_max_dom(dict(max_dom=3), max_dom=2) == 2
    assert st.effective_max_dom({}) == 1
