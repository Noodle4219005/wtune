import datetime
import threading
import time

import pytest

from wtune.tools import coldstart_sample as cs


class _Source:
    dataset = "gfs_aws"
    forcing_interval_h = 3

    def urls_for(self, vt):
        name = "gfs.%s" % vt.strftime("%Y%m%d%H")
        return [(name, ["http://example.invalid/" + name])]


def _window():
    t = datetime.datetime(2022, 4, 4, 18)
    return t, t + datetime.timedelta(hours=21)             # 8 valid times


def _fetch(tmp_path, download_fn, **kw):
    t, t_end = _window()
    return cs._download_forcing_for_window(_Source(), t, t_end, str(tmp_path),
                                           download_fn=download_fn, sleep_fn=lambda s: None, **kw)


def test_files_are_fetched_concurrently_and_all_land(tmp_path, monkeypatch):
    monkeypatch.setenv("WTUNE_FETCH_WORKERS", "8")
    threads, lock = set(), threading.Lock()

    def slow(url, dest):
        with lock:
            threads.add(threading.get_ident())
        time.sleep(0.2)
        open(dest, "w").write("x")

    t0 = time.time()
    paths = _fetch(tmp_path, slow)
    assert len(paths) == 8 and len(threads) > 1
    assert time.time() - t0 < 1.0                          # serial would be 8 x 0.2 s


def test_workers_1_is_serial_and_missing_keeps_window_order(tmp_path, monkeypatch):
    monkeypatch.setenv("WTUNE_FETCH_WORKERS", "1")
    order = []

    def flaky(url, dest):
        order.append(url)
        if url.endswith(("2022040418", "2022040500")):
            raise OSError("404")
        open(dest, "w").write("x")

    with pytest.raises(cs.ColdstartError) as e:
        _fetch(tmp_path, flaky, attempts=1)
    assert "gfs.2022040418, gfs.2022040500" in str(e.value)
    assert order == sorted(order)                          # serial: one file after the other


def test_missing_order_is_the_same_when_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("WTUNE_FETCH_WORKERS", "8")

    def flaky(url, dest):
        if url.endswith(("2022040418", "2022040500")):
            raise OSError("404")
        open(dest, "w").write("x")

    with pytest.raises(cs.ColdstartError) as e:
        _fetch(tmp_path, flaky, attempts=1)
    assert "gfs.2022040418, gfs.2022040500" in str(e.value)
