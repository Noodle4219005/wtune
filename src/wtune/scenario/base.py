"""scenario.base -- the one interface every scenario generator implements: ScenarioRequest in,
ForcingSet out, gated by a small license/GPU-aware registry.

ScenarioGenerator.generate(req) produces WPS intermediate files (see wps_intermediate.py) for a
requested bbox/time window, whatever the request's `ic_source` says the initial condition should
be built from. Concrete generators live under scenario/generators/; this module only defines the
shapes and the registry (register/get_generator/list_generators) they plug into.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ScenarioError(Exception):
    """Base error for the scenario package (bad config, bad request, gated generator, ...)."""


class LicenseError(ScenarioError):
    """Raised by get_generator() when a generator's license marks it noncommercial and the
    caller did not pass allow_noncommercial=True. The message always names the license string
    and the allow_noncommercial switch so a caller can decide whether to override it."""


@dataclass(frozen=True)
class ScenarioRequest:
    """One request to a ScenarioGenerator. `start`/`end` are the FICTIONAL (possibly relabelled)
    valid-time window the generator must produce forcing for -- see generators/analog.py's module
    docstring for what "fictional/relabelled" means in this repo. `analog_date`, when given,
    overrides whatever real-date selection a generator would otherwise do (e.g. analog's own
    climatological/scored selector)."""

    bbox: tuple  # (lat_min, lat_max, lon_min, lon_max), degrees
    start: datetime.datetime
    end: datetime.datetime
    interval_h: int = 6
    ic_source: str = "analog"          # "analog" | "analog-blend"
    members: int = 1
    out_dir: Path = None
    case_dir: Optional[Path] = None
    analog_date: Optional[datetime.datetime] = None

    def __post_init__(self):
        if self.ic_source not in ("analog", "analog-blend"):
            raise ScenarioError("ScenarioRequest.ic_source must be 'analog' or 'analog-blend', "
                                 "got %r" % (self.ic_source,))
        if self.out_dir is None:
            raise ScenarioError("ScenarioRequest.out_dir is required")
        object.__setattr__(self, "out_dir", Path(self.out_dir))
        if self.case_dir is not None:
            object.__setattr__(self, "case_dir", Path(self.case_dir))


@dataclass(frozen=True)
class Caps:
    """What a generator can actually produce, independent of any one request."""

    fields: tuple             # WPS field names this generator produces, e.g. ("TT","UU",...)
    levels_pa: tuple          # pressure levels in Pa this generator produces on
    has_soil: bool
    has_sst: bool
    has_hydrometeors: bool
    min_interval_h: int
    max_lead_h: int
    needs_gpu: bool
    license: str
    noncommercial: bool


def valid_times(req: ScenarioRequest):
    """Every valid time in [req.start, req.end] at req.interval_h spacing, inclusive of both
    ends when the span divides evenly."""
    out = []
    t = req.start
    step = datetime.timedelta(hours=req.interval_h)
    while t <= req.end:
        out.append(t)
        t += step
    return out


def request_to_dict(req: ScenarioRequest) -> dict:
    """JSON-safe dict of a ScenarioRequest (Path -> str, datetime -> ISO-8601) for embedding in
    ForcingSet.provenance."""
    d = dataclasses.asdict(req)
    for key in ("out_dir", "case_dir"):
        if d.get(key) is not None:
            d[key] = str(d[key])
    for key in ("start", "end", "analog_date"):
        if d.get(key) is not None:
            d[key] = d[key].isoformat()
    return d


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError("not JSON serializable: %r" % (type(obj),))


@dataclass
class ForcingSet:
    """The output of one ScenarioGenerator.generate() call: the WPS intermediate files produced,
    what fields/times they cover, what's missing and where it would come from, and enough
    provenance to reproduce or audit the run later."""

    files: list
    prefix: str
    valid_times: list
    fields_present: list = field(default_factory=list)
    fields_missing: list = field(default_factory=list)
    fill_rules: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_json(self, path) -> None:
        """Writes this ForcingSet as a manifest (forcing.json). datetime/Path values anywhere in
        `provenance` are serialized as ISO-8601 strings / plain path strings -- from_json() does
        NOT reconstruct datetime objects inside `provenance` (only `valid_times` and `files` are
        typed on the way back in); treat provenance as a JSON-safe audit record, not live data."""
        payload = {
            "prefix": self.prefix,
            "files": [str(p) for p in self.files],
            "valid_times": [t.isoformat() for t in self.valid_times],
            "fields_present": list(self.fields_present),
            "fields_missing": list(self.fields_missing),
            "fill_rules": dict(self.fill_rules),
            "provenance": self.provenance,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default, sort_keys=True)

    @classmethod
    def from_json(cls, path) -> "ForcingSet":
        with open(path) as f:
            payload = json.load(f)
        return cls(
            files=[Path(p) for p in payload["files"]],
            prefix=payload["prefix"],
            valid_times=[datetime.datetime.fromisoformat(t) for t in payload["valid_times"]],
            fields_present=list(payload.get("fields_present", [])),
            fields_missing=list(payload.get("fields_missing", [])),
            fill_rules=dict(payload.get("fill_rules", {})),
            provenance=payload.get("provenance", {}),
        )


class ScenarioGenerator(ABC):
    """One pluggable forcing generator. `name` is the registry key; `license`/`noncommercial`/
    `needs_gpu` are class attributes so the registry and the license gate can inspect them
    WITHOUT instantiating the generator (a not-yet-implemented generator's __init__ may not even
    be safe to call). `implemented=False` on a subclass marks it as a registered stub whose
    generate() raises NotImplementedError -- used for the "planned" generators the roadmap names
    but this round does not build."""

    name: str = ""
    license: str = ""
    noncommercial: bool = False
    needs_gpu: bool = False
    implemented: bool = True

    @abstractmethod
    def capabilities(self) -> Caps:
        raise NotImplementedError

    @abstractmethod
    def generate(self, req: ScenarioRequest) -> ForcingSet:
        raise NotImplementedError


_REGISTRY: dict = {}
_generators_loaded = False


def _load_generators() -> None:
    """Imports scenario.generators (which imports every concrete generator module, running each
    module's @register decorator) exactly once. Lazy and function-local so that importing
    scenario.base alone never has to import numpy/netCDF4/case_rebuild -- only get_generator()/
    list_generators() need the concrete generators loaded."""
    global _generators_loaded
    if _generators_loaded:
        return
    _generators_loaded = True
    from . import generators  # noqa: F401  (side effect: registers analog, aifs, ...)


def register(cls):
    """Class decorator: adds `cls` to the generator registry under `cls.name`."""
    if not getattr(cls, "name", ""):
        raise ScenarioError("%r must set a non-empty class attribute `name` to register" % (cls,))
    _REGISTRY[cls.name] = cls
    return cls


def get_generator(name: str, allow_noncommercial: bool = False) -> ScenarioGenerator:
    """Instantiates the registered generator `name`, after the license gate. Raises
    LicenseError (a ScenarioError) when the generator is noncommercial-licensed and
    allow_noncommercial was not passed True; the message names the license and the switch."""
    _load_generators()
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ScenarioError(
            "unknown scenario generator %r (registered: %s)"
            % (name, ", ".join(sorted(_REGISTRY)) or "none")) from None
    if cls.noncommercial and not allow_noncommercial:
        raise LicenseError(
            "scenario generator %r is licensed %s (noncommercial) -- pass "
            "allow_noncommercial=True (CLI: --allow-noncommercial) to use it anyway"
            % (name, cls.license))
    return cls()


def list_generators() -> list:
    """[{'name','license','noncommercial','needs_gpu','implemented'}, ...] for every registered
    generator, sorted by name -- independent of the license gate (listing is always allowed)."""
    _load_generators()
    return [
        {
            "name": cls.name,
            "license": cls.license,
            "noncommercial": cls.noncommercial,
            "needs_gpu": cls.needs_gpu,
            "implemented": bool(getattr(cls, "implemented", True)),
        }
        for cls in sorted(_REGISTRY.values(), key=lambda c: c.name)
    ]


def _register_stub(name, license_str, noncommercial, needs_gpu, note):
    """Builds and registers a not-yet-implemented generator stub so `list_generators()` and the
    license gate work for it before it has a real implementation. `capabilities`/`generate` MUST
    be supplied in the class namespace at type() creation time -- ABCMeta freezes
    __abstractmethods__ from the class body at creation, so assigning them onto the class
    afterward (cls.generate = ...) leaves the class abstract and uninstantiable."""

    def capabilities(self):
        raise NotImplementedError("scenario generator %r is not implemented yet: %s" % (name, note))

    def generate(self, req):
        raise NotImplementedError("scenario generator %r is not implemented yet: %s" % (name, note))

    _Stub = type("Stub_%s" % name.replace("-", "_"), (ScenarioGenerator,), {
        "name": name, "license": license_str, "noncommercial": noncommercial,
        "needs_gpu": needs_gpu, "implemented": False,
        "capabilities": capabilities, "generate": generate,
    })
    register(_Stub)
    return _Stub


# Planned-but-not-yet-implemented generators (design item 1): registered as stubs so `wtune
# scenario list` and the license gate work for them ahead of a real implementation.
_register_stub("aurora", "MIT", False, True,
               "Microsoft Aurora foundation model adapter -- planned, not yet built")
_register_stub("neuralgcm", "CC BY-SA 4.0", False, True,
               "Google NeuralGCM adapter -- planned, not yet built")
_register_stub("wrf-coarse", "WRF (public domain)", False, False,
               "coarse-resolution WRF-as-forcing generator -- planned, not yet built")
_register_stub("graphcast", "CC BY-NC-SA 4.0", True, True,
               "DeepMind GraphCast adapter -- planned, not yet built")
_register_stub("pangu", "CC BY-NC-SA 4.0", True, True,
               "Huawei Pangu-Weather adapter -- planned, not yet built")
_register_stub("analog-bdy", "depends on the forcing source used (see the `analog` generator)",
               False, False,
               "boundary-only variant of `analog` (lateral bdy forcing without a full IC) -- "
               "planned, not yet built")
