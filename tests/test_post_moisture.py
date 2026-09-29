"""Test for post/recipes/moisture.py's _pw_change_figure(): the 6 h PW change must pick the
frame nearest 6 h before the LAST frame, not simply the second-to-last frame on disk -- a real
bug caught on the scc P7 validation run (2026-09-29): m1's wrfout is hourly, so "the previous
frame" was only 1 h earlier and silently produced a wrongly-labelled 'moisture_pw_chg6h' entry
that was actually a 1 h change.
"""
from post.recipes import moisture


def test_pw_change_figure_picks_frame_nearest_6h_before_last(monkeypatch):
    hourly_times = ["2005-08-27_%02d:00:00" % h for h in range(10)]  # 00:00 .. 09:00
    monkeypatch.setattr(moisture, "list_frames",
                        lambda run_dir, domain=1: [(t, "/fake/%s" % t) for t in hourly_times])

    captured = {}

    def fake_change_map(run_dir, run_label, field, domain, out_dir, t_prev, t_curr, t0, **kw):
        captured.update(t_prev=t_prev, t_curr=t_curr, t0=t0)
        return {"id": kw.get("fig_id"), "missing": False}

    monkeypatch.setattr("post.changemap.change_map", fake_change_map)

    moisture._pw_change_figure("/fake/run", "run0", 2, "/tmp/out")

    assert captured["t_curr"] == "2005-08-27_09:00:00"
    assert captured["t_prev"] == "2005-08-27_03:00:00"   # 09:00 - 6h
    assert captured["t0"] == "2005-08-27_00:00:00"


def test_pw_change_figure_returns_none_with_fewer_than_two_frames(monkeypatch):
    monkeypatch.setattr(moisture, "list_frames", lambda run_dir, domain=1: [("t0", "/fake")])
    assert moisture._pw_change_figure("/fake/run", "run0", 1, "/tmp/out") is None
