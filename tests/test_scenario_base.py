"""Tests for the scenario generator registry and license gate (scenario.base), and for
scenario.config.parse_scenario."""

import pytest

import wtune.scenario.base as base
from wtune.scenario.base import LicenseError, ScenarioError, get_generator, list_generators
from wtune.scenario.config import parse_scenario


def test_list_generators_contains_every_registered_name():
    names = {row["name"] for row in list_generators()}
    for expected in ("analog", "aifs", "aurora", "neuralgcm", "wrf-coarse", "graphcast",
                      "pangu", "analog-bdy"):
        assert expected in names


def test_unknown_generator_raises():
    with pytest.raises(ScenarioError):
        get_generator("does-not-exist")


def test_noncommercial_generator_refused_by_default():
    with pytest.raises(LicenseError) as exc_info:
        get_generator("graphcast")
    msg = str(exc_info.value)
    assert "allow_noncommercial" in msg
    assert "CC BY-NC-SA" in msg


def test_noncommercial_generator_refused_pangu():
    with pytest.raises(LicenseError) as exc_info:
        get_generator("pangu")
    msg = str(exc_info.value)
    assert "allow_noncommercial" in msg
    assert "CC BY-NC-SA" in msg


def test_noncommercial_generator_allowed_when_flag_set():
    # graphcast/pangu are stubs (implemented=False): the gate passes, generate() itself raises
    # NotImplementedError -- distinct from the license gate.
    gen = get_generator("graphcast", allow_noncommercial=True)
    assert gen.name == "graphcast"
    with pytest.raises(NotImplementedError):
        gen.capabilities()


def test_commercial_generator_never_needs_the_flag():
    gen = get_generator("analog")
    assert gen.name == "analog"


def test_config_gates_generator():
    with pytest.raises(ScenarioError):
        parse_scenario({"generator": "graphcast"})
    cfg = parse_scenario({"generator": "graphcast", "allow_noncommercial": True})
    assert cfg.generator == "graphcast"
    assert cfg.allow_noncommercial is True


def test_config_gates_ensemble_members():
    with pytest.raises(ScenarioError):
        parse_scenario({"generator": "analog", "ensemble": ["analog", "pangu"]})
    cfg = parse_scenario({"generator": "analog", "ensemble": ["analog", "pangu"],
                           "allow_noncommercial": True})
    assert cfg.ensemble == ("analog", "pangu")


def test_config_unknown_key_raises():
    with pytest.raises(ScenarioError) as exc_info:
        parse_scenario({"generator": "analog", "bogus_key": 1})
    assert "bogus_key" in str(exc_info.value)


def test_config_defaults():
    cfg = parse_scenario({})
    assert cfg.when == "auto"
    assert cfg.generator == "aifs"   # condition 5 winner (decision 0054)
    assert cfg.ic == "analog"
    assert cfg.ensemble is None
    assert cfg.allow_noncommercial is False
    assert cfg.gpu == "auto"


def test_config_invalid_choice_raises():
    with pytest.raises(ScenarioError):
        parse_scenario({"when": "sometimes"})
    with pytest.raises(ScenarioError):
        parse_scenario({"ic": "bogus"})
    with pytest.raises(ScenarioError):
        parse_scenario({"gpu": "yes"})
