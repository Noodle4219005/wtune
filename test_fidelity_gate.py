import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np

import fidelity_gate as fg


MODULE = Path(fg.__file__).resolve()

NT, NY, NX, NZ = 3, 8, 10, 4
TIMES = ["2026-01-01_00:00:00", "2026-01-01_01:00:00", "2026-01-01_02:00:00"]

FIELD_DIMS = {
    "T2": ("Time", "south_north", "west_east"),
    "Q2": ("Time", "south_north", "west_east"),
    "U10": ("Time", "south_north", "west_east"),
    "V10": ("Time", "south_north", "west_east"),
    "PSFC": ("Time", "south_north", "west_east"),
    "T": ("Time", "bottom_top", "south_north", "west_east"),
    "QVAPOR": ("Time", "bottom_top", "south_north", "west_east"),
    "U": ("Time", "bottom_top", "south_north", "west_east_stag"),
    "V": ("Time", "bottom_top", "south_north_stag", "west_east"),
    "W": ("Time", "bottom_top_stag", "south_north", "west_east"),
    "RAINNC": ("Time", "south_north", "west_east"),
    "SWDOWN": ("Time", "south_north", "west_east"),
    "QSNOW": ("Time", "south_north", "west_east"),
}


def _pattern(ny, nx, amp, phase=0.0):
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return amp * (np.sin(yy / 2.0 + phase) + np.cos(xx / 3.0 + phase))


def _broadcast(pattern, shape):
    return np.broadcast_to(pattern, shape).astype(np.float32).copy()


def _base_fields(ny=NY, nx=NX, nz=NZ, nt=NT):
    fields = {}
    fields["T2"] = 280.0 + _broadcast(_pattern(ny, nx, 5.0), (nt, ny, nx))
    fields["Q2"] = 0.008 + _broadcast(_pattern(ny, nx, 0.001, 1.0), (nt, ny, nx))
    fields["U10"] = _broadcast(_pattern(ny, nx, 3.0, 2.0), (nt, ny, nx))
    fields["V10"] = _broadcast(_pattern(ny, nx, 2.0, 0.5), (nt, ny, nx))
    fields["PSFC"] = 101000.0 + _broadcast(_pattern(ny, nx, 200.0, 1.5), (nt, ny, nx))
    fields["T"] = 260.0 + _broadcast(_pattern(ny, nx, 8.0), (nt, nz, ny, nx))
    fields["QVAPOR"] = 0.003 + _broadcast(_pattern(ny, nx, 0.0005, 0.7), (nt, nz, ny, nx))
    fields["U"] = _broadcast(_pattern(ny, nx + 1, 4.0, 0.3), (nt, nz, ny, nx + 1))
    fields["V"] = _broadcast(_pattern(ny + 1, nx, 3.0, 0.9), (nt, nz, ny + 1, nx))
    fields["W"] = _broadcast(_pattern(ny, nx, 0.1, 0.4), (nt, nz + 1, ny, nx))
    accum = np.arange(nt, dtype=np.float32).reshape(nt, 1, 1) * 2.5
    fields["RAINNC"] = accum + _broadcast(_pattern(ny, nx, 0.3, 0.2), (nt, ny, nx))
    fields["SWDOWN"] = 400.0 + _broadcast(_pattern(ny, nx, 50.0, 1.1), (nt, ny, nx))
    return fields


def _write(path, fields, times=TIMES, drop_fields=(), ny=NY, nx=NX, nz=NZ):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", len(times))
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", ny)
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("west_east", nx)
        ds.createDimension("west_east_stag", nx + 1)
        ds.createDimension("bottom_top", nz)
        ds.createDimension("bottom_top_stag", nz + 1)

        times_var = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times_var[:] = np.array([list(t) for t in times], dtype="S1")

        for name, dims in FIELD_DIMS.items():
            if name in drop_fields or name not in fields:
                continue
            data = fields[name]
            if data.shape[0] != len(times):
                data = data[: len(times)]
            var = ds.createVariable(name, "f4", dims)
            var[:] = data


def _shift(fields, frac, names=None):
    """Uniform additive shift of frac * std(field) to each named field (default: all).
    Because the shift is a spatial constant, rms_diff(ref, shifted) == frac * std(ref)
    exactly (up to float32 rounding), so nd == frac and bias(field) == frac * std(field)."""
    names = names or list(fields)
    out = dict(fields)
    for name in names:
        arr = fields[name]
        out[name] = arr + np.float32(frac * float(np.std(arr)))
    return out


def _scaled_noise(fields, base_noise, scale, names=None):
    """base_noise: dict name -> array (same shape as fields[name]) of small deterministic
    'noise'. Returns fields + scale * base_noise for the given names (default: all)."""
    names = names or list(fields)
    out = dict(fields)
    for name in names:
        out[name] = fields[name] + np.float32(scale) * base_noise[name]
    return out


def _make_noise(fields, frac=1e-4):
    """Deterministic small-amplitude 'round-off-like' perturbation field per variable, built
    from a different smooth pattern phase (uncorrelated with the base field) and scaled to
    frac * std(field) -- relative to each field's own magnitude, like _shift -- so it stays
    well above float32 rounding noise regardless of whether the field is ~0.003 or ~101000."""
    noise = {}
    for name, arr in fields.items():
        shape = arr.shape
        ny, nx = shape[-2], shape[-1]
        pattern = _pattern(ny, nx, 1.0, phase=3.7)
        scale = frac * float(np.std(arr))
        noise[name] = np.float32(scale) * _broadcast(pattern, shape)
    return noise


def _run(*args):
    return subprocess.run(
        [sys.executable, str(MODULE), *map(str, args)],
        cwd=MODULE.parent,
        text=True,
        capture_output=True,
    )


def _gate(tmp_path, ref_fields, twin_fields, cand_fields, family="dynamics",
          bdy_width=0, thresholds=None, prefix="out", **write_kwargs):
    ref = tmp_path / "ref.nc"
    twin = tmp_path / "twin.nc"
    cand = tmp_path / "cand.nc"
    _write(ref, ref_fields, **write_kwargs)
    _write(twin, twin_fields, **write_kwargs)
    _write(cand, cand_fields, **write_kwargs)
    out_prefix = tmp_path / prefix
    args = [
        "--reference", ref, "--twin", twin, "--candidate", cand,
        "--family", family, "--bdy-width", bdy_width, "--out-prefix", out_prefix,
    ]
    if thresholds is not None:
        args += ["--thresholds", json.dumps(thresholds)]
    result = _run(*args)
    payload = json.loads((tmp_path / (prefix + ".json")).read_text())
    return result, payload, out_prefix


def test_within_floor(tmp_path):
    ref = _base_fields()
    # frac=1e-2 (not the usual 1e-4 "floor" scale) so the perturbation clears float32
    # rounding noise at every field's own magnitude (PSFC ~1e5 has the worst headroom)
    # and the 1.0x/1.5x ratio survives the netCDF float32 round-trip to << 1%.
    noise = _make_noise(ref, frac=1e-2)
    twin = _scaled_noise(ref, noise, 1.0)
    cand = _scaled_noise(ref, noise, 1.5)  # same noise pattern, 1.5x -> floor_ratio ~= 1.5

    result, payload, _ = _gate(tmp_path, ref, twin, cand)
    assert result.returncode == 0
    assert payload["label"] == "WITHIN-FLOOR"
    t2 = payload["per_field_summary_at_last_time"]["T2"]
    assert t2["status"] == "ok"
    assert abs(t2["floor_ratio"] - 1.5) < 0.05
    assert t2["floor_zero"] is False


def test_small_drift(tmp_path):
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)
    cand = _shift(ref, frac=0.01)  # nd == 0.01 for every field, well under 0.10

    result, payload, _ = _gate(tmp_path, ref, twin, cand)
    assert result.returncode == 0
    assert payload["label"] == "SMALL-DRIFT"
    t2 = payload["per_field_summary_at_last_time"]["T2"]
    assert abs(t2["nd"] - 0.01) < 1e-3
    assert abs(t2["bias"]) < 0.1


def test_large_drift_t2_bias(tmp_path):
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)
    # Only T2 gets a large uniform shift (0.5 std -> both nd>0.10 and |bias(T2)|>0.1K);
    # every other core field stays at reference (tiny floor-scale noise only).
    cand = _shift(ref, frac=0.5, names=["T2"])
    cand = _scaled_noise(cand, noise, 1.0, names=[n for n in ref if n != "T2"])

    result, payload, _ = _gate(tmp_path, ref, twin, cand)
    assert result.returncode == 0
    assert payload["label"] == "LARGE-DRIFT"
    assert "T2" in payload["deciding"]
    t2 = payload["per_field_summary_at_last_time"]["T2"]
    assert abs(t2["bias"]) > 0.1
    assert t2["nd"] > 0.10


def test_invalid_missing_core_field(tmp_path):
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)
    cand = dict(ref)

    ref_path = tmp_path / "ref.nc"
    twin_path = tmp_path / "twin.nc"
    cand_path = tmp_path / "cand.nc"
    _write(ref_path, ref)
    _write(twin_path, twin)
    _write(cand_path, cand, drop_fields={"W"})  # W is a required core field

    out_prefix = tmp_path / "out"
    result = _run(
        "--reference", ref_path, "--twin", twin_path, "--candidate", cand_path,
        "--family", "dynamics", "--bdy-width", "0", "--out-prefix", out_prefix,
    )
    payload = json.loads((out_prefix.with_suffix(".json")).read_text())
    assert result.returncode == 0  # a computed INVALID label is not a tool error
    assert payload["label"] == "INVALID"
    assert "W" in payload["deciding"]


def test_invalid_no_common_times(tmp_path):
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)
    cand = dict(ref)

    ref_path = tmp_path / "ref.nc"
    twin_path = tmp_path / "twin.nc"
    cand_path = tmp_path / "cand.nc"
    other_times = ["2027-06-01_00:00:00", "2027-06-01_01:00:00", "2027-06-01_02:00:00"]
    _write(ref_path, ref)
    _write(twin_path, twin)
    _write(cand_path, cand, times=other_times)

    out_prefix = tmp_path / "out"
    result = _run(
        "--reference", ref_path, "--twin", twin_path, "--candidate", cand_path,
        "--family", "dynamics", "--bdy-width", "0", "--out-prefix", out_prefix,
    )
    payload = json.loads((out_prefix.with_suffix(".json")).read_text())
    assert result.returncode == 0
    assert payload["label"] == "INVALID"
    assert "no common valid times" in payload["deciding"]


def test_time_tolerance_matches_post_init_time(tmp_path):
    # A candidate whose time_step differs from the reference's writes history at times
    # that no longer coincide exactly (2026-09-15 fix: karina/isc21 P03, a 1.2x time_step
    # multiplier, used to match ONLY at t=0 under exact-string matching and were labelled
    # WITHIN-FLOOR with nd=0/floor_ratio=0 -- a false "no change"). t=0 (initialization)
    # still lines up exactly; every later time is offset by a fixed 90s, comfortably
    # inside the default 180s --time-tol-s, so the fix must still reach a real
    # post-initialization comparison instead of only t=0.
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-2)
    twin = _scaled_noise(ref, noise, 1.0)
    cand = _scaled_noise(ref, noise, 1.5)  # same pattern as test_within_floor: floor_ratio ~= 1.5

    def _plus_90s(t):
        dt = datetime.strptime(t, "%Y-%m-%d_%H:%M:%S") + timedelta(seconds=90)
        return dt.strftime("%Y-%m-%d_%H:%M:%S")

    shifted_times = [TIMES[0]] + [_plus_90s(t) for t in TIMES[1:]]

    ref_path = tmp_path / "ref.nc"
    twin_path = tmp_path / "twin.nc"
    cand_path = tmp_path / "cand.nc"
    _write(ref_path, ref)
    _write(twin_path, twin)
    _write(cand_path, cand, times=shifted_times)

    out_prefix = tmp_path / "out"
    result = _run(
        "--reference", ref_path, "--twin", twin_path, "--candidate", cand_path,
        "--family", "dynamics", "--bdy-width", "0", "--out-prefix", out_prefix,
    )
    payload = json.loads(out_prefix.with_suffix(".json").read_text())
    assert result.returncode == 0
    # Under the OLD exact-string matching this candidate would share NO time with the
    # reference after t=0 (common_times == [TIMES[0]] only) -> last_time == TIMES[0] and
    # the comparison would be a false no-op. The fix must reach the actual last time.
    assert payload["last_time"] == TIMES[-1]
    assert payload["last_time_match"]["candidate"] == shifted_times[-1]
    assert abs(payload["last_time_match"]["candidate_offset_s"] - 90.0) < 1e-6
    assert payload["last_time_match"]["twin_offset_s"] == 0.0
    assert payload["label"] == "WITHIN-FLOOR"


def test_time_only_initial_overlap_is_invalid(tmp_path):
    # A candidate that only has the initialization time in common with the reference
    # (e.g. every later output drifted outside --time-tol-s, or the run only produced
    # t=0) must not be silently compared at t=0 only -- that reads as "no change" when
    # nothing was actually compared post-initialization.
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)
    cand_trunc = {name: arr[:1] for name, arr in ref.items()}

    ref_path = tmp_path / "ref.nc"
    twin_path = tmp_path / "twin.nc"
    cand_path = tmp_path / "cand.nc"
    _write(ref_path, ref)
    _write(twin_path, twin)
    _write(cand_path, cand_trunc, times=TIMES[:1])

    out_prefix = tmp_path / "out"
    result = _run(
        "--reference", ref_path, "--twin", twin_path, "--candidate", cand_path,
        "--family", "dynamics", "--bdy-width", "0", "--out-prefix", out_prefix,
    )
    payload = json.loads(out_prefix.with_suffix(".json").read_text())
    assert result.returncode == 0
    assert payload["label"] == "INVALID"
    assert payload["deciding"] == "no matched valid time after initialization"
    assert payload["common_times"] == TIMES[:1]  # the t=0 match was found, just unusable


def test_low_variance_field_nd_undefined_not_large_drift(tmp_path):
    # maria1km P02 (nighttime): nd(SWDOWN)=1.4e6 because the reference SWDOWN has ~zero
    # spatial variance at night, so even a tiny absolute difference normalizes to a huge
    # relative one. Here the reference SWDOWN is exactly zero (its twin -- built from the
    # same std-scaled noise helpers -- is then bit-identical, floor_zero); the candidate
    # gets a physically negligible absolute offset. nd must be left undefined, and that
    # alone must never force the comparison to LARGE-DRIFT.
    ref = _base_fields()
    ref["SWDOWN"] = np.zeros_like(ref["SWDOWN"])

    noise = _make_noise(ref, frac=1e-4)  # scaled by each field's own std -> 0 for SWDOWN
    twin = _scaled_noise(ref, noise, 1.0)
    cand = _shift(ref, frac=0.01)  # SMALL-DRIFT-scale on every field (0 for SWDOWN too)
    cand["SWDOWN"] = ref["SWDOWN"] + np.float32(1e-3)  # tiny absolute offset, not std-scaled

    result, payload, _ = _gate(tmp_path, ref, twin, cand, family="radiation")
    assert result.returncode == 0
    swdown = payload["per_field_summary_at_last_time"]["SWDOWN"]
    assert swdown["low_variance"] is True
    assert swdown["nd"] is None
    assert swdown["nd_status"] == "undefined_low_variance"
    assert swdown["floor_zero"] is True
    assert abs(swdown["bias"]) < fg.DEFAULT_THRESHOLDS["bias_swdown_max"]
    assert payload["label"] == "SMALL-DRIFT"


def test_boundary_exclusion(tmp_path):
    ny, nx, ring = 10, 12, 2
    ref = _base_fields(ny=ny, nx=nx)
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)

    cand = dict(ref)
    t2 = ref["T2"].copy()
    mask = np.ones((ny, nx), dtype=bool)
    mask[ring: ny - ring, ring: nx - ring] = False  # True on the boundary ring only
    t2[:, mask] += np.float32(50.0)  # huge perturbation confined to the boundary ring
    cand["T2"] = t2

    result_excl, payload_excl, _ = _gate(
        tmp_path, ref, twin, cand, bdy_width=ring, prefix="excl", ny=ny, nx=nx,
    )
    assert result_excl.returncode == 0
    assert payload_excl["label"] == "WITHIN-FLOOR"

    result_incl, payload_incl, _ = _gate(
        tmp_path, ref, twin, cand, bdy_width=0, prefix="incl", ny=ny, nx=nx,
    )
    assert result_incl.returncode == 0
    assert payload_incl["label"] == "LARGE-DRIFT"


def test_floor_zero(tmp_path):
    # 2026-09-15 fix: maria1km/lafe twins came back bit-identical to the reference (twin
    # decomposition doesn't perturb them at all), so the OLD floor_ratio = rms_diff /
    # max(floor, tiny) divided by ~0 and printed an astronomically large, uninformative
    # number (1e9-1e10) for every field. floor_ratio is now left null (undefined) instead
    # -- WITHIN-FLOOR for a floor_zero field requires rms_diff == 0.0 exactly; any nonzero
    # diff falls back to the nd/bias criteria rather than a meaningless giant ratio.
    ref = _base_fields()
    twin = dict(ref)  # bit-identical twin -> floor == 0 for every field

    result_id, payload_id, _ = _gate(tmp_path, ref, twin, dict(ref), prefix="ident")
    assert result_id.returncode == 0
    assert payload_id["label"] == "WITHIN-FLOOR"
    t2_id = payload_id["per_field_summary_at_last_time"]["T2"]
    assert t2_id["floor_zero"] is True
    assert t2_id["rms_diff"] == 0.0
    assert t2_id["floor_ratio"] is None  # undefined, not the old tiny-fallback 0.0

    noise = _make_noise(ref, frac=1e-3)
    cand = _scaled_noise(ref, noise, 1.0)
    result_tiny, payload_tiny, _ = _gate(tmp_path, ref, twin, cand, prefix="tinydiff")
    assert result_tiny.returncode == 0
    t2_tiny = payload_tiny["per_field_summary_at_last_time"]["T2"]
    assert t2_tiny["floor_zero"] is True
    assert t2_tiny["rms_diff"] > 0.0
    assert t2_tiny["floor_ratio"] is None  # undefined, not the old giant tiny-fallback ratio
    assert t2_tiny["nd"] is not None and 0.0 < t2_tiny["nd"] < 0.01  # frac=1e-3 noise scale, well under nd_max
    assert payload_tiny["label"] == "SMALL-DRIFT"  # nd/bias are tiny; decided without floor_ratio


def test_threshold_override(tmp_path):
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)
    # Shift only PSFC (leave T2 untouched) so the label is decided purely by the nd
    # criterion, not incidentally by the independent |bias(T2)| check.
    cand = _shift(ref, frac=0.2, names=["PSFC"])  # nd(PSFC) == 0.2 -> LARGE-DRIFT under default nd_max=0.10

    default_result, default_payload, _ = _gate(tmp_path, ref, twin, cand, prefix="default")
    assert default_payload["label"] == "LARGE-DRIFT"

    loose_result, loose_payload, _ = _gate(
        tmp_path, ref, twin, cand, thresholds={"nd_max": 0.5}, prefix="loose",
    )
    assert loose_result.returncode == 0
    assert loose_payload["label"] == "SMALL-DRIFT"
    assert loose_payload["thresholds"]["nd_max"] == 0.5


def _sparse_qsnow_fields(twin_frac, cand_frac):
    """A QSNOW-like field with a single sparse nonzero cell (tiny std(ref)) plus a
    reference/twin/candidate triple where QSNOW's twin/candidate diffs from reference are
    UNIFORM shifts of frac * std(QSNOW) -- so nd(QSNOW) == frac and floor_ratio(QSNOW) ==
    cand_frac / twin_frac EXACTLY (a uniform shift's rms over any slice equals the shift
    itself), independent of std(QSNOW)'s actual value.

    Every OTHER field gets floor-scale noise for the twin and a real (if small) uniform
    shift for the candidate -- their own nd stays comfortably under nd_max but their
    floor_ratio is pushed well past floor_ratio_max, so the aggregate WITHIN-FLOOR
    short-circuit (worst floor_ratio over ALL key fields) does not mask the
    QSNOW-specific per-field behaviour under test."""
    ref = _base_fields()
    qsnow = np.zeros((NT, NY, NX), dtype=np.float32)
    qsnow[:, 0, 0] = 1e-3  # single sparse nonzero cell -> tiny std(ref), like a real QSNOW field
    ref["QSNOW"] = qsnow
    std_ref = float(np.std(ref["QSNOW"]))

    twin = dict(ref)
    cand = dict(ref)
    twin["QSNOW"] = ref["QSNOW"] + np.float32(twin_frac * std_ref)
    cand["QSNOW"] = ref["QSNOW"] + np.float32(cand_frac * std_ref)

    other = [n for n in ref if n != "QSNOW"]
    other_noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(twin, other_noise, 1.0, names=other)
    cand = _shift(cand, frac=0.005, names=other)
    return ref, twin, cand


def test_sparse_field_high_nd_but_floor_ratio_ok_stays_small_drift(tmp_path):
    # Mirrors measured karina QSNOW: nd=0.117 (> nd_max 0.10) but floor_ratio=1 (the
    # candidate's QSNOW diff from reference is the SAME size as the twin's own chaos
    # floor) -- under the OLD rule (max nd over all fields <= nd_max) this alone would
    # have flipped the whole comparison to LARGE-DRIFT. It must not: QSNOW is
    # individually indistinguishable from the floor, so it passes the per-field test.
    ref, twin, cand = _sparse_qsnow_fields(twin_frac=0.117, cand_frac=0.117)

    result, payload, _ = _gate(tmp_path, ref, twin, cand, family="microphysics")
    assert result.returncode == 0
    qsnow = payload["per_field_summary_at_last_time"]["QSNOW"]
    assert qsnow["nd"] > 0.10
    assert abs(qsnow["floor_ratio"] - 1.0) < 0.05
    assert payload["label"] == "SMALL-DRIFT"


def test_sparse_field_high_nd_and_high_floor_ratio_is_large_drift(tmp_path):
    # Same sparse QSNOW field, but now the candidate's diff is 5x the twin's own floor --
    # both nd AND floor_ratio exceed their thresholds, so this field must fail (a high nd
    # alone must not be forgiven when the field is also far from the chaos floor).
    ref, twin, cand = _sparse_qsnow_fields(twin_frac=0.05, cand_frac=0.5)

    result, payload, _ = _gate(tmp_path, ref, twin, cand, family="microphysics")
    assert result.returncode == 0
    qsnow = payload["per_field_summary_at_last_time"]["QSNOW"]
    assert qsnow["nd"] > 0.10
    assert qsnow["floor_ratio"] > 2.0
    assert payload["label"] == "LARGE-DRIFT"
    assert "QSNOW" in payload["deciding"]


def test_multi_file_per_role_matched_by_times(tmp_path):
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)
    cand = _scaled_noise(ref, noise, 1.2)

    def _split(fields):
        parts = []
        for i in range(NT):
            parts.append({name: arr[i: i + 1] for name, arr in fields.items()})
        return parts

    ref_parts = _split(ref)
    ref_paths = []
    for i, part in enumerate(ref_parts):
        p = tmp_path / f"ref_{i}.nc"
        _write(p, part, times=[TIMES[i]])
        ref_paths.append(p)

    twin_path = tmp_path / "twin.nc"
    _write(twin_path, twin)

    # candidate only has the first two times -> common_times should stop there
    cand_path = tmp_path / "cand.nc"
    cand_trunc = {name: arr[:2] for name, arr in cand.items()}
    _write(cand_path, cand_trunc, times=TIMES[:2])

    out_prefix = tmp_path / "out"
    result = _run(
        "--reference", *ref_paths, "--twin", twin_path, "--candidate", cand_path,
        "--family", "dynamics", "--bdy-width", "0", "--out-prefix", out_prefix,
    )
    payload = json.loads(out_prefix.with_suffix(".json").read_text())
    assert result.returncode == 0
    assert payload["common_times"] == TIMES[:2]
    assert payload["last_time"] == TIMES[1]
    assert payload["label"] == "WITHIN-FLOOR"


def test_family_field_selection_and_csv_rows(tmp_path):
    ref = _base_fields()
    noise = _make_noise(ref, frac=1e-4)
    twin = _scaled_noise(ref, noise, 1.0)
    cand = _scaled_noise(ref, noise, 1.0)

    result, payload, out_prefix = _gate(tmp_path, ref, twin, cand, family="radiation", prefix="rad")
    assert result.returncode == 0
    assert set(payload["fields_requested"]) == set(fg.CORE_FIELDS) | {"SWDOWN", "GLW", "OLR", "TSK"}
    # GLW/OLR/TSK are not present in our synthetic files -> recorded missing, not fatal
    assert "GLW" in payload["fields_missing"]
    csv_text = Path(str(out_prefix) + ".csv").read_text().strip().splitlines()
    n_fields = len(payload["fields_requested"])
    assert len(csv_text) - 1 == n_fields * NT  # header + one row per field x time


def test_every_template_family_is_a_legal_gate_family():
    """The gate's --family choices must cover every family physics_templates.json actually uses, or
    a clean run silently loses its verdict: D04 (family 'diagnostics') ran six times across two
    cases (jobs 1081089/1081090, 1081237/1081238) and each time the gate exited with
    "--family: invalid choice: 'diagnostics'" AFTER the WRF runs had been paid for. Same for
    'nesting' (D06, zhengzhou job 1080703)."""
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "campaign", "physics_templates.json")) as fh:
        templates = json.load(fh)["templates"]
    used = {t.get("family") for t in templates.values()} - {None, "none"}
    missing = sorted(f for f in used if f not in fg.FAMILIES)
    assert not missing, "families used by templates but rejected by the gate: %s" % missing


def test_core_only_families_have_no_extra_fields():
    for fam in ("dynamics", "diagnostics", "nesting"):
        assert fg.FAMILY_FIELDS[fam] == [], fam
