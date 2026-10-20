"""wtune event: template, check (vocabulary, verbatim quotes, storm ids, sha) and the
observation plan (module / agent_fetch / unsupported / local / model_only)."""
import os
import re
from datetime import datetime

import pytest
import yaml

from wtune import event

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.join(os.path.dirname(HERE), ".claude", "skills", "wtune-event", "SKILL.md")
EVENTS = os.path.join(HERE, "fixtures", "events")

TXT = ("2013 年 6 月加拿大 Alberta 的洪水，暖濕氣流撞上洛磯山脈東坡，兩天下了 200 mm 以上。\n"
       "想看兩天累積雨量的分布對不對，還有山坡上的上升運動剖面。雨量站資料在 /data/ab_gauges.csv。\n")


def _skill_example():
    text = open(SKILL).read()
    return yaml.safe_load(re.search(r"```yaml\n(.*?)```", text, re.S).group(1))


@pytest.fixture
def case(tmp_path):
    p = tmp_path / "event.txt"
    p.write_text(TXT)
    doc = _skill_example()
    doc["source_sha256"] = event.sha256_file(str(p))
    return p, doc


def test_skill_example_passes_check(case):
    p, doc = case
    err, warn = event.check(doc, txt=TXT, txt_sha=event.sha256_file(str(p)))
    assert err == [] and warn == []


def test_template_has_sha_and_schema(case):
    p, _doc = case
    t = event.template(str(p))
    assert t["schema"] == event.SCHEMA and t["source_sha256"] == event.sha256_file(str(p))


def test_quote_not_in_text_is_an_error(case):
    _p, doc = case
    doc["verify"].append({"value": "wind10", "from": "地面風"})
    err, _ = event.check(doc, txt=TXT)
    assert any("not in event.txt" in e for e in err)


def test_unknown_vocabulary_is_an_error(case):
    _p, doc = case
    doc["types"].append({"value": "hailstorm", "from": "洪水"})
    doc["verify"].append({"value": "rain", "from": "雨量"})
    err, _ = event.check(doc, txt=TXT)
    assert any(e.startswith("types[2]") for e in err)
    assert any(e.startswith("verify[1]") for e in err)


def test_storm_id_must_be_quoted():
    txt = "2016 年 10 月 Matthew 颶風北上。看路徑。"
    doc = {"schema": event.SCHEMA, "event": {"value": "Matthew", "from": "Matthew 颶風"},
           "types": [{"value": "tropical_cyclone", "from": "颶風"}],
           "storms": [{"name": "Matthew", "year": 2016, "id": "AL142016", "from": "Matthew 颶風"}],
           "verify": [{"value": "track", "from": "看路徑"}]}
    err, _ = event.check(doc, txt=txt)
    assert any("id 'AL142016' is not in its quote" in e for e in err)
    doc["storms"][0]["id"] = None
    assert event.check(doc, txt=txt)[0] == []


def test_tc_without_storm_needs_a_question():
    txt = "跑一個颱風看看"
    doc = {"schema": event.SCHEMA, "event": {"value": "a typhoon", "from": "颱風"},
           "types": [{"value": "tropical_cyclone", "from": "颱風"}]}
    assert any(e.startswith("storms:") for e in event.check(doc, txt=txt)[0])
    doc["questions"] = ["哪一個颱風、哪一年？"]
    assert event.check(doc, txt=txt)[0] == []


def test_changed_text_fails_sha(case):
    p, doc = case
    p.write_text(TXT + "也看地面風。\n")
    err, _ = event.check(doc, txt=p.read_text(), txt_sha=event.sha256_file(str(p)))
    assert any(e.startswith("source_sha256") for e in err)


def test_local_path_must_be_inside_its_quote(case):
    _p, doc = case
    doc["local_obs"][0]["path"] = "/data/other.csv"
    assert any("not inside its quote" in e for e in event.check(doc, txt=TXT)[0])


def _status(rows, need):
    return {r["need"]: r["status"] for r in rows}[need]


def test_plan_outside_conus_needs_agent_fetch(case):
    _p, doc = case
    rows = event.plan(doc, datetime(2013, 6, 19), datetime(2013, 6, 22), (45, 55, -120, -108))
    assert _status(rows, "precip_total") == "agent_fetch"
    assert _status(rows, "/data/ab_gauges.csv") == "local"
    assert any(r["status"] == "model_only" for r in rows)


def test_plan_conus_uses_modules():
    doc = {"types": [{"value": "severe_wind"}],
           "verify": [{"value": "reflectivity"}, {"value": "precip_24h"}, {"value": "t2"},
                      {"value": "gust"}, {"value": "aod"}]}
    rows = event.plan(doc, datetime(2020, 8, 10), datetime(2020, 8, 11), (38, 45, -100, -85))
    assert _status(rows, "reflectivity") == "module"
    assert _status(rows, "precip_24h") == "module"
    assert _status(rows, "t2") == "module"
    assert _status(rows, "gust") == "agent_fetch"
    assert _status(rows, "aod") == "unsupported"


def test_plan_best_track_by_basin():
    doc = {"types": [{"value": "tropical_cyclone"}], "verify": [{"value": "track"}],
           "storms": [{"name": "Matthew", "id": "AL142016"}]}
    assert _status(event.plan(doc), "track") == "module"
    doc["storms"] = [{"name": "Morakot", "id": "WP092009"}]
    assert _status(event.plan(doc), "track") == "agent_fetch"


def test_plan_idealized_has_no_observations():
    doc = {"types": [{"value": "idealized"}], "verify": [{"value": "reflectivity"}]}
    assert _status(event.plan(doc), "reflectivity") == "unsupported"


def test_every_verify_kind_is_a_known_kind():
    assert set(event.VERIFY.values()) <= set(event.OBS_KINDS)


@pytest.mark.parametrize("folder", ["events", "events-holdout"])
def test_expected_fixture_uses_the_vocabulary(folder):
    base = os.path.join(HERE, "fixtures", folder)
    exp = yaml.safe_load(open(os.path.join(base, "expected.yaml")))
    for case_name, e in exp.items():
        assert os.path.exists(os.path.join(base, case_name, "event.txt")), case_name
        for key, table in (("types", event.TYPES), ("verify", event.VERIFY),
                           ("obs_requests", event.OBS_KINDS)):
            spec = e.get(key) or {"required": [], "allowed": []}
            names = {v for x in spec["required"] for v in x.split("|")}
            assert names | set(spec.get("allowed") or []) <= set(table), (case_name, key)


def test_cli_check_exit_codes(case, capsys):
    p, doc = case
    y = p.parent / "event.yaml"
    y.write_text(yaml.safe_dump(doc, allow_unicode=True))
    assert event.main(["check", str(y)]) == 0
    doc["verify"][0]["from"] = "沒有這句"
    y.write_text(yaml.safe_dump(doc, allow_unicode=True))
    assert event.main(["check", str(y)]) == 1
    assert "FAIL" in capsys.readouterr().out
