"""Tests for post/recipe.py: the fig()/missing() record shape, catalog.json completeness
against every figure id the dispatch table can actually produce, and build_recipe_report's own
wiring (via post/report.py) with synthetic figure records -- no real wrfout/namelist data
needed (that path is exercised by tests/test_post_caseinfo.py's R1/R4 tests plus the manual
end-to-end run recorded in the delegation report).
"""
import os


from wtune.post import recipe, report


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
