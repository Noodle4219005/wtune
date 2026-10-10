"""Tests for post/report.py: self-contained HTML with images embedded, no external refs."""
import csv
import os

import wtune.post.report as report


def _write_png(path):
    # smallest valid 1x1 PNG
    import base64
    data = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    with open(path, "wb") as fh:
        fh.write(data)


def test_report_embeds_image_and_csv_with_no_external_refs(tmp_path):
    _write_png(tmp_path / "fig1.png")
    with open(tmp_path / "table1.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b"])
        w.writerow(["1", "2"])

    out_path = report.build_report(str(tmp_path), title="Test Report")
    assert os.path.exists(out_path)
    text = open(out_path).read()
    assert "Test Report" in text
    assert "data:image/png;base64," in text
    assert "<table" in text and "<td>1</td>" in text
    assert "http://" not in text and "https://" not in text
    assert "src='fig1.png'" not in text


def test_report_empty_dir_notes_nothing_found(tmp_path):
    out_path = report.build_report(str(tmp_path))
    text = open(out_path).read()
    assert "no PNG/SVG figures" in text
