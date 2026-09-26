"""scenario.config -- parse_scenario(dict) -> ScenarioConfig, for embedding a scenario block in
a wtune.yaml-style config WITHOUT wiring into src/wtune_config.py itself (that file is owned by
another agent in this round; this module is deliberately standalone).

Applies the license gate (base.get_generator) to `generator` AND every name in `ensemble`, so a
noncommercial generator can't slip in through the ensemble list."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .base import ScenarioError, get_generator

_VALID_KEYS = {"when", "generator", "ic", "ensemble", "allow_noncommercial", "gpu"}
_VALID_WHEN = {"auto", "always", "never"}
_VALID_IC = {"analog", "analog-blend"}
_VALID_GPU = {"auto", "true", "false"}


@dataclass(frozen=True)
class ScenarioConfig:
    when: str = "auto"
    generator: str = "analog"
    ic: str = "analog"
    ensemble: Optional[Tuple[str, ...]] = None
    allow_noncommercial: bool = False
    gpu: str = "auto"


def _check_choice(value, valid, key):
    if value not in valid:
        raise ScenarioError(
            "scenario config key %r: %r is not one of %s" % (key, value, sorted(valid)))


def parse_scenario(d: dict) -> ScenarioConfig:
    """Validates `d` (a plain dict, e.g. a YAML `scenario:` block) and applies the license gate
    to `generator` and every `ensemble` member. Raises ScenarioError on an unknown key, an
    invalid choice, or a gated (noncommercial, not explicitly allowed) generator."""
    unknown = set(d) - _VALID_KEYS
    if unknown:
        raise ScenarioError(
            "unknown scenario config key(s): %s (valid keys: %s)"
            % (", ".join(sorted(unknown)), ", ".join(sorted(_VALID_KEYS))))

    when = d.get("when", "auto")
    _check_choice(when, _VALID_WHEN, "when")

    generator = d.get("generator", "analog")
    ic = d.get("ic", "analog")
    _check_choice(ic, _VALID_IC, "ic")

    ensemble = d.get("ensemble")
    if ensemble is not None:
        ensemble = tuple(ensemble)

    allow_noncommercial = bool(d.get("allow_noncommercial", False))

    gpu = d.get("gpu", "auto")
    _check_choice(gpu, _VALID_GPU, "gpu")

    get_generator(generator, allow_noncommercial=allow_noncommercial)
    if ensemble:
        for name in ensemble:
            get_generator(name, allow_noncommercial=allow_noncommercial)

    return ScenarioConfig(when=when, generator=generator, ic=ic, ensemble=ensemble,
                           allow_noncommercial=allow_noncommercial, gpu=gpu)
