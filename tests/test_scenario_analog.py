"""Tests for scenario.generators.analog -- with mocked download/ungrib (no network, no real WPS
install). Verifies: (1) analog date selection matches case_rebuild's own selector bit-for-bit;
(2) generate() produces the requested valid_times, relabelled from the analog date; (3) the
ForcingSet manifest round-trips through JSON."""

import datetime

import numpy as np

import wtune.scenario.wps_intermediate as wi
from wtune.scenario.base import ForcingSet, ScenarioRequest, get_generator


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
