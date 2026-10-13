"""Shared by every wtune.run layer: constants, RunError, time and fraction helpers, and Record,
the base of the run's dataclasses."""
from __future__ import annotations

import dataclasses
import fractions
import math
import re
from typing import Any, TypeVar

WRF_FMT = "%Y-%m-%d_%H:%M:%S"
ADIOS2_IO_FORM = 14                    # WRF io_form for ADIOS2 (the default for every stream)
_TIME_PARTS = ("year", "month", "day", "hour", "minute", "second")
_PER_RECORD_KIND_RE = re.compile(r"^(wrfbdy|wrflowinp|wrffdda)_d(\d\d)$")


class RunError(Exception):
    pass


class _Unset:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()                  # a Record field that is absent from its JSON form
R = TypeVar("R", bound="Record")


def to_plain(value: Any) -> Any:
    """`value` with every Record in it replaced by its to_dict() (what json.dump is given)."""
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    return value


class Record:
    """Base of the run's dataclasses. to_dict() is the JSON form: the set fields, in field
    order. A record still reads like the dict it replaced (rec["key"], rec.get(), dict(rec),
    rec == {...}), so callers written against the dicts keep working."""

    def keys(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self) if getattr(self, f.name) is not UNSET]

    def to_dict(self) -> dict:
        return {k: to_plain(getattr(self, k)) for k in self.keys()}

    def __getitem__(self, key: str) -> Any:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if key in self.keys() else default

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Record):
            other = other.to_dict()
        return self.to_dict() == other if isinstance(other, dict) else NotImplemented

    @classmethod
    def coerce(cls: type[R], value: Any) -> R:
        """`value` as this record: itself, or built from the dict form a caller still passes."""
        return value if isinstance(value, cls) else cls(**value)


def _fmt(t) -> str:
    return t.strftime(WRF_FMT)


def _hms(s: float) -> str:
    s = int(round(s))
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def parse_hms(text: str) -> int:
    """'HH:MM:SS' (hours may exceed 24) or 'D-HH:MM:SS' -> seconds."""
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    if len(parts) != 3:
        raise ValueError("expected HH:MM:SS, got %r" % text)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _frac(x) -> fractions.Fraction:
    return fractions.Fraction(x).limit_denominator(10000)


def _frac_lcm(a, b) -> fractions.Fraction:
    a, b = _frac(a), _frac(b)
    den = a.denominator * b.denominator // math.gcd(a.denominator, b.denominator)
    na, nb = int(a * den), int(b * den)
    return fractions.Fraction(na * nb // math.gcd(na, nb), den)
