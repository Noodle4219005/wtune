"""Tests for the scenario generator registry and license gate (scenario.base), and for
scenario.config.parse_scenario."""

import pytest

from wtune.scenario.base import ScenarioError, get_generator, list_generators
from wtune.scenario.config import parse_scenario


def test_list_generators_contains_every_registered_name():
    names = {row["name"] for row in list_generators()}
    for expected in ("analog", "aifs", "aurora", "neuralgcm", "wrf-coarse", "graphcast",
                      "pangu", "analog-bdy"):
        assert expected in names


def test_unknown_generator_raises():
    with pytest.raises(ScenarioError):
        get_generator("does-not-exist")


def test_noncommercial_generator_allowed_when_flag_set():
    # graphcast/pangu are stubs (implemented=False): the gate passes, generate() itself raises
    # NotImplementedError -- distinct from the license gate.
    gen = get_generator("graphcast", allow_noncommercial=True)
    assert gen.name == "graphcast"
    with pytest.raises(NotImplementedError):
        gen.capabilities()


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


def test_config_invalid_choice_raises():
    with pytest.raises(ScenarioError):
        parse_scenario({"when": "sometimes"})
    with pytest.raises(ScenarioError):
        parse_scenario({"ic": "bogus"})
    with pytest.raises(ScenarioError):
        parse_scenario({"gpu": "yes"})
