"""Tests for scenario.generators.analog -- with mocked download/ungrib (no network, no real WPS
install). Verifies: (1) analog date selection matches case_rebuild's own selector bit-for-bit;
(2) generate() produces the requested valid_times, relabelled from the analog date; (3) the
ForcingSet manifest round-trips through JSON."""

import datetime

import numpy as np
import pytest

import case_rebuild as cr
import scenario.wps_intermediate as wi
from scenario.base import ForcingSet, ScenarioRequest, get_generator
from scenario.generators.analog import select_analog_date


def test_select_analog_date_matches_case_rebuild_selector():
    # climatological_analog() (T14, 2026-09-26) is the SAME function case_rebuild.cmd_analog's
    # own no-wrfinput branch calls -- select_analog_date() must call it, not re-derive the rule
    # (the old rule, max(candidate_dates(...)), was a bug: the window's last day, not same-DOY).
    start = datetime.datetime(2031, 7, 15, 12)
    got = select_analog_date(start, years=(2021, 2025), doy_window=45, hour="same")
    expected = cr.climatological_analog(start, (2021, 2025), hour="same")
    assert got == expected
    assert got == datetime.datetime(2025, 7, 15, 12)  # same DOY, most recent year -- not Aug/Sep


def test_select_analog_date_wraps_year_boundary():
    start = datetime.datetime(2031, 1, 10, 6)
    got = select_analog_date(start, years=(2021, 2022), doy_window=45, hour="same")
    expected = cr.climatological_analog(start, (2021, 2022), hour="same")
    assert got == expected
    assert got == datetime.datetime(2022, 1, 10, 6)


def _fake_ungrib_writer(nx=2, ny=2):
    def run_ungrib(grib_paths, analog_times, bbox, out_dir, prefix):
        paths = []
        for i, t in enumerate(analog_times):
            slab = np.full((ny, nx), float(i), dtype=np.float32)
            rec = wi.Record(hdate=t.strftime("%Y-%m-%d_%H:%M:%S"), xfcst=0.0,
                             map_source="FAKE-GFS", field="TT", units="K", desc="Temperature",
                             xlvl=wi.SURFACE_PA, nx=nx, ny=ny, iproj=0, startloc="SWCORNER",
                             startlat=bbox[0], startlon=bbox[2], deltalat=1.0, deltalon=1.0,
                             earth_radius=wi.EARTH_RADIUS_KM, is_wind_earth_rel=True, slab=slab)
            path = out_dir / ("analog_raw_%d" % i)
            wi.write_fields(path, [rec])
            paths.append(path)
        return paths
    return run_ungrib


def _fake_download(analog_times, bbox, out_dir):
    return ["fake-grib-%d" % i for i in range(len(analog_times))]


def test_generate_with_mocked_download_and_ungrib(tmp_path):
    gen = get_generator("analog")
    req = ScenarioRequest(
        bbox=(30.0, 40.0, -110.0, -100.0),
        start=datetime.datetime(2031, 7, 15, 0),
        end=datetime.datetime(2031, 7, 15, 12),
        interval_h=6,
        out_dir=tmp_path / "out",
    )
    forcing = gen.generate(req, download=_fake_download, run_ungrib=_fake_ungrib_writer())

    expected_times = [datetime.datetime(2031, 7, 15, 0), datetime.datetime(2031, 7, 15, 6),
                       datetime.datetime(2031, 7, 15, 12)]
    assert forcing.valid_times == expected_times
    assert len(forcing.files) == 3
    assert forcing.fill_rules == {"hydrometeors": "spin-up"}
    assert forcing.fields_missing == []

    # every output file's HDATE is the FICTIONAL time, not the analog time.
    for f, t in zip(forcing.files, expected_times):
        recs = wi.read_fields(f)
        assert recs[0].hdate == t.strftime("%Y-%m-%d_%H:%M:%S")

    manifest_path = tmp_path / "forcing.json"
    forcing.to_json(manifest_path)
    back = ForcingSet.from_json(manifest_path)
    assert back.valid_times == forcing.valid_times
    assert [str(p) for p in back.files] == [str(p) for p in forcing.files]
    assert back.fields_present == forcing.fields_present
    assert back.fields_missing == forcing.fields_missing
    assert back.fill_rules == forcing.fill_rules
    assert back.provenance["generator"] == "analog"


def test_relabelled_slab_bytes_match_unrelabelled_ungrib_output(tmp_path):
    """Item 2 (2026-09-26 review): this generator's relabel-to-fictional-date behaviour changes
    ONLY the HDATE header -- SLAB (data) bytes must be byte-for-byte identical to what
    run_ungrib produced at the (unrelabelled) analog date. case_rebuild's own flow never
    relabels at all (it runs WRF directly at the analog date); this generator's relabel is an
    explicit architect decision, not a reproduction of case_rebuild's semantics -- this test
    pins down what "relabel" is allowed to change."""
    gen = get_generator("analog")
    req = ScenarioRequest(
        bbox=(30.0, 40.0, -110.0, -100.0),
        start=datetime.datetime(2031, 7, 15, 0),
        end=datetime.datetime(2031, 7, 15, 0),
        interval_h=6,
        out_dir=tmp_path / "out",
    )
    forcing = gen.generate(req, download=_fake_download, run_ungrib=_fake_ungrib_writer())

    # the raw (unrelabelled) ungrib output the fake writer left behind under out_dir.
    raw_path = req.out_dir / "analog_raw_0"
    raw_recs = wi.read_fields(raw_path)
    relabelled_recs = wi.read_fields(forcing.files[0])

    assert len(raw_recs) == len(relabelled_recs) == 1
    assert raw_recs[0].hdate != relabelled_recs[0].hdate  # HDATE DID change (analog -> fictional)
    assert relabelled_recs[0].hdate == "2031-07-15_00:00:00"
    assert np.array_equal(raw_recs[0].slab, relabelled_recs[0].slab)  # SLAB bytes UNCHANGED

    # every other header field is also unchanged by relabel -- only HDATE differs.
    for attr in ("field", "units", "desc", "xlvl", "nx", "ny", "startlat", "startlon",
                 "deltalat", "deltalon", "earth_radius", "is_wind_earth_rel", "map_source"):
        assert getattr(raw_recs[0], attr) == getattr(relabelled_recs[0], attr)


def test_generate_respects_analog_date_override(tmp_path):
    gen = get_generator("analog")
    override = datetime.datetime(2022, 3, 1, 0)
    req = ScenarioRequest(
        bbox=(30.0, 40.0, -110.0, -100.0),
        start=datetime.datetime(2031, 7, 15, 0),
        end=datetime.datetime(2031, 7, 15, 0),
        interval_h=6,
        out_dir=tmp_path / "out",
        analog_date=override,
    )
    forcing = gen.generate(req, download=_fake_download, run_ungrib=_fake_ungrib_writer())
    assert forcing.provenance["analog_date"] == override
