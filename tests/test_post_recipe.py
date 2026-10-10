"""Tests for post/recipe.py: the fig()/missing() record shape, catalog.json completeness
against every figure id the dispatch table can actually produce, and build_recipe_report's own
wiring (via post/report.py) with synthetic figure records -- no real wrfout/namelist data
needed (that path is exercised by tests/test_post_caseinfo.py's R1/R4 tests plus the manual
end-to-end run recorded in the delegation report).
"""
import json
import os
import re

import pytest

from wtune.post import recipe, report


def test_fig_and_missing_record_shapes():
    f = recipe.fig("common_domain_map", "/tmp/x.png", "run0", domain="d01",
                   valid_time="2005-08-27_00:00:00", extra={"note": "n"})
    assert f == dict(id="common_domain_map", missing=False, path="/tmp/x.png", run="run0",
                     domain="d01", valid_time="2005-08-27_00:00:00", note="n")

    m = recipe.missing("precip_fss", "Stage IV", "not fetched this round")
    assert m == dict(id="precip_fss", missing=True, dataset="Stage IV",
                     reason="not fetched this round", run=None)


def test_catalog_loads_and_every_entry_has_both_languages():
    catalog = recipe.load_catalog()
    assert "_meta" in catalog
    for fig_id, entry in catalog.items():
        if fig_id == "_meta":
            continue
        for lang in ("zh", "en"):
            assert lang in entry, "%s missing %s" % (fig_id, lang)
            for field in ("question", "how_to_read", "common_misreadings"):
                assert entry[lang].get(field), "%s.%s.%s is empty" % (fig_id, lang, field)


# Every id that appears as a Python string literal in post/recipes/*.py's fig()/missing() calls
# -- a cheap static completeness check that the catalog covers what the code can actually emit,
# without importing every module (several need wrf-python/cartopy/xwrf at import time only
# inside functions, which is fine, but this test should stay import-free and fast).
_RECIPES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "wtune", "post", "recipes")
_ID_CALL_RE = re.compile(r'(?:fig|missing)\(\s*"([a-zA-Z0-9_]+)"')


def _ids_referenced_in_recipes():
    ids = set()
    for name in os.listdir(_RECIPES_DIR):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(_RECIPES_DIR, name)) as fh:
            text = fh.read()
        ids |= set(_ID_CALL_RE.findall(text))
    return ids


def test_catalog_covers_every_id_referenced_in_recipes():
    catalog = recipe.load_catalog()
    referenced = _ids_referenced_in_recipes()
    missing_from_catalog = referenced - set(catalog.keys())
    assert not missing_from_catalog, ("catalog.json is missing caption entries for: %s"
                                      % sorted(missing_from_catalog))


def _synthetic_case_info():
    return dict(types=["common", "tc"],
               evidence={"common": "always produced", "tc": "3 fixes inside d01",
                        "convection": "no domain with dx<=4 km", "precip": "always produced",
                        "synoptic": "below thresholds", "pbl": "n/a", "chem": "n/a",
                        "climate": "n/a", "ensemble": "n/a"},
               domains=[], max_dom=1, run_length_h=48.0, d01_extent_km=(2028.0, 1548.0))


def test_build_recipe_report_groups_by_type_and_renders_missing(tmp_path):
    import base64
    png_path = tmp_path / "fig1.png"
    png_path.write_bytes(base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))

    figures = [
        recipe.fig("common_domain_map", str(png_path), "run0", domain="all",
                  valid_time="2005-08-27_00:00:00"),
        recipe.missing("tc_figure", "MET tc_pairs", "tc_pairs not found on PATH"),
    ]
    out_path = report.build_recipe_report(str(tmp_path), _synthetic_case_info(), figures,
                                          [str(tmp_path)], ["run0"], 6.0, recipe.load_catalog(),
                                          title="unit test report")
    assert os.path.exists(out_path)
    text = open(out_path).read()
    assert "unit test report" in text
    assert "common, tc" in text
    assert "data:image/png;base64," in text
    assert "missing: tc_figure" in text
    assert "tc_pairs not found on PATH" in text
    assert "spin-up excluded: 6 h" in text
    assert "http://" not in text and "https://" not in text
