import json
import os

from wtune.post import timeseries

CATALOG = os.path.join(os.path.dirname(timeseries.__file__), "recipes", "catalog.json")


def test_every_time_series_field_has_a_caption_entry():
    with open(CATALOG, encoding="utf-8") as fh:
        cat = json.load(fh)
    missing = ["timeseries_%s" % f for f in timeseries.TIME_SERIES_FIELDS
               if "timeseries_%s" % f not in cat]
    assert not missing, missing
    for f in timeseries.TIME_SERIES_FIELDS:
        for lang in ("zh", "en"):
            assert {"question", "how_to_read", "common_misreadings"} <= set(cat["timeseries_%s" % f][lang])
