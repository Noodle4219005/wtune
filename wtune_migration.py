"""Deterministic WTune migration protocol; this is not MPI runtime execution."""
import base64
import hashlib
import json


class MigrationError(Exception):
    pass


class MigrationPhase(object):
    INIT = "init"
    POLL = "poll"
    PREPARE = "prepare"
    TRANSFER = "transfer"
    COMMIT = "commit"
    ABORT = "abort"
    FINALIZE = "finalize"


def _mapping(value, label):
    if not isinstance(value, dict):
        raise MigrationError(label + " must be a mapping")


class RestartableField(object):
    def __init__(self, name, bounds, stagger, datatype, ownership, payload):
        if not isinstance(name, str) or not name:
            raise MigrationError("field name must be a non-empty string")
        if not isinstance(bounds, (dict, list, tuple)) or not bounds:
            raise MigrationError("field bounds must be a non-empty mapping or sequence")
        for label, value in (("stagger", stagger), ("datatype", datatype), ("ownership", ownership)):
            if not isinstance(value, str) or not value:
                raise MigrationError("field " + label + " must be a non-empty string")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise MigrationError("field payload must be bytes-like")
        self.name, self.bounds = name, bounds
        self.stagger, self.datatype, self.ownership = stagger, datatype, ownership
        self.payload = bytes(payload)

    def to_dict(self):
        return {"name": self.name, "bounds": self.bounds, "stagger": self.stagger,
                "datatype": self.datatype, "ownership": self.ownership,
                "payload": base64.b64encode(self.payload).decode("ascii")}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or "payload" not in value:
            raise MigrationError("invalid field descriptor")
        try:
            payload = base64.b64decode(value["payload"].encode("ascii"), validate=True)
        except (ValueError, TypeError, AttributeError):
            raise MigrationError("invalid encoded field payload")
        return cls(value.get("name"), value.get("bounds"), value.get("stagger"),
                   value.get("datatype"), value.get("ownership"), payload)


class MigrationManifest(object):
    _sections = ("domain_clock", "alarms", "nesting", "halo", "cartesian_communicator",
                 "rank_coordinates", "runtime_cache", "derived_communication")

    def __init__(self, fields, domain_clock, alarms, nesting, halo,
                 cartesian_communicator, rank_coordinates, runtime_cache,
                 derived_communication, epoch=0, scope="single-domain",
                 field_inventory_complete=True):
        if not isinstance(fields, (list, tuple)):
            raise MigrationError("fields must be a list")
        self.fields = list(fields)
        names = set()
        for field in self.fields:
            if not isinstance(field, RestartableField) or field.name in names:
                raise MigrationError("fields must contain unique RestartableField descriptors")
            names.add(field.name)
        section_values = {"domain_clock": domain_clock, "alarms": alarms, "nesting": nesting,
                          "halo": halo, "cartesian_communicator": cartesian_communicator,
                          "rank_coordinates": rank_coordinates, "runtime_cache": runtime_cache,
                          "derived_communication": derived_communication}
        for name, value in section_values.items():
            if name in ("alarms", "rank_coordinates"):
                if not isinstance(value, list):
                    raise MigrationError(name + " must be a list")
            else:
                _mapping(value, name)
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise MigrationError("epoch must be a non-negative integer")
        if not isinstance(scope, str) or not scope:
            raise MigrationError("scope must be a non-empty string")
        self.domain_clock, self.alarms, self.nesting, self.halo = domain_clock, alarms, nesting, halo
        self.cartesian_communicator, self.rank_coordinates = cartesian_communicator, rank_coordinates
        self.runtime_cache, self.derived_communication = runtime_cache, derived_communication
        self.epoch, self.scope, self.field_inventory_complete = epoch, scope, bool(field_inventory_complete)

    def to_dict(self):
        return {"fields": [f.to_dict() for f in self.fields],
                "domain_clock": self.domain_clock, "alarms": self.alarms, "nesting": self.nesting,
                "halo": self.halo, "cartesian_communicator": self.cartesian_communicator,
                "rank_coordinates": self.rank_coordinates, "runtime_cache": self.runtime_cache,
                "derived_communication": self.derived_communication, "epoch": self.epoch,
                "scope": self.scope, "field_inventory_complete": self.field_inventory_complete}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise MigrationError("manifest must be a mapping")
        try:
            fields = [RestartableField.from_dict(x) for x in value["fields"]]
            return cls(fields, *[value[x] for x in cls._sections], value.get("epoch", 0),
                       value.get("scope", "single-domain"), value.get("field_inventory_complete", True))
        except KeyError as exc:
            raise MigrationError("missing manifest state: " + str(exc))

    def canonical_bytes(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_commit(manifest):
    if not manifest.field_inventory_complete:
        raise MigrationError("field inventory is incomplete")
    if manifest.scope != "single-domain" or manifest.nesting.get("max_dom") != 1:
        raise MigrationError("nested-domain metadata is not allowed")


def verify_bitwise_payloads(source, target):
    if not isinstance(source, MigrationManifest) or not isinstance(target, MigrationManifest):
        raise MigrationError("source and target must be manifests")
    left, right = {f.name: f for f in source.fields}, {f.name: f for f in target.fields}
    if set(left) != set(right):
        raise MigrationError("field names differ")
    for name in sorted(left):
        if hashlib.sha256(left[name].payload).digest() != hashlib.sha256(right[name].payload).digest() or left[name].payload != right[name].payload:
            raise MigrationError("payload mismatch for field " + name)
    return True


class MigrationCoordinator(object):
    def __init__(self, manifest):
        if not isinstance(manifest, MigrationManifest):
            raise MigrationError("manifest required")
        self.manifest, self._phase, self._pending = manifest, MigrationPhase.INIT, None

    @property
    def phase(self): return self._phase

    def _advance(self, expected, phase):
        if self._phase != expected: raise MigrationError("invalid transition from " + self._phase)
        self._phase = phase

    def poll(self): self._advance(MigrationPhase.INIT, MigrationPhase.POLL)
    def prepare(self, target_manifest):
        self._advance(MigrationPhase.POLL, MigrationPhase.PREPARE)
        _validate_commit(target_manifest); self._pending = target_manifest
    def transfer(self, target_manifest):
        self._advance(MigrationPhase.PREPARE, MigrationPhase.TRANSFER)
        if target_manifest is not self._pending: raise MigrationError("target differs from prepared manifest")
    def commit(self):
        self._advance(MigrationPhase.TRANSFER, MigrationPhase.COMMIT)
        try:
            verify_bitwise_payloads(self.manifest, self._pending); _validate_commit(self._pending)
            self._pending.epoch = self.manifest.epoch + 1
            self.manifest, self._pending = self._pending, None
        except Exception as exc:
            self._pending, self._phase = None, MigrationPhase.ABORT
            if isinstance(exc, MigrationError): raise
            raise MigrationError(str(exc))

    def abort(self, reason):
        self._pending, self._phase = None, MigrationPhase.ABORT
        return reason

    def finalize(self): self._advance(MigrationPhase.COMMIT, MigrationPhase.FINALIZE)
