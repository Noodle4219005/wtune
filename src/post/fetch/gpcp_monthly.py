"""post/fetch/gpcp_monthly.py -- GPCP v2.3 monthly precipitation (2.5 deg, mm/day) from NOAA NCEI
(one small NetCDF per month), the precipitation reference of the climate recipe.

Why GPCP and not ERA5 tp: ERA5 precipitation is a short-range FORECAST accumulation (model
physics, known wet biases over the tropical oceans and orography), and aggregating a monthly
mean from the ARCO store needs every hourly chunk. GPCP merges gauges and satellite estimates,
is public with a stable per-month URL, and is the standard observational reference for monthly
regional precipitation. Its 2.5 deg grid is coarse against a 50 km model: the recipe compares
box means and coarse-grid maps and says so on the figure.

Source: https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-monthly/access/<YYYY>/
gpcp_v02r03_monthly_dYYYYMM_c<processing date>.nc (the file name carries a processing date, so
the year directory is listed to find each month). Files are stored unchanged under
<project>/input/obs/gpcp_monthly/. Variable `precip` in mm/day.
"""
from __future__ import annotations

import os
import re

from post import sources
from post.fetch.era5_monthly import month_starts

NAME = "gpcp_monthly"
BASE = "https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-monthly/access"
HELP = "GPCP v2.3 monthly precipitation, 2.5 deg (NOAA NCEI) -> <project>/input/obs/gpcp_monthly/"
PROVENANCE = {
    "dataset": "GPCP Version 2.3 Monthly Precipitation Analysis (v02r03)",
    "provider": "NOAA NCEI / GPCP (Global Precipitation Climatology Project)",
    "url": BASE + "/",
    "license": "US Government work / public (NOAA NCEI, CC0)",
    "citation": "Adler, R.F., et al., 2018: The Global Precipitation Climatology Project (GPCP) "
                "Monthly Analysis (New Version 2.3) and a Review of 2017 Global Precipitation. "
                "Atmosphere, 9(4), 138, doi:10.3390/atmos9040138.",
    "coverage_time": "1979-01 to about two months before present",
    "coverage_space": "global, 2.5 deg",
    "resolution": "monthly mean, mm/day",
    "access": "public, no credentials",
}


def add_arguments(parser):
    sources.add_window_arguments(parser)
    parser.add_argument("--skip-days", type=float, default=30.0, dest="skip_days",
                        help="skip months starting before window start + this (spin-up; default 30)")


def list_month_files(year_listing_html):
    """{YYYYMM: file name} from an NCEI directory listing (newest processing date wins)."""
    out = {}
    for name in sorted(set(re.findall(r"gpcp_v02r03_monthly_d\d{6}_c\d{8}\.nc", year_listing_html))):
        out[name.split("_d")[1][:6]] = name      # sorted: a later processing date replaces an earlier
    return out


def run(args):
    try:
        start, end, bbox = sources.resolve_window(args)
    except ValueError as e:
        print("gpcp_monthly: %s" % e)
        return []
    months = month_starts(start, end, args.skip_days)
    if not months:
        print("gpcp_monthly: no complete calendar month in the window after the spin-up skip")
        return []
    out_dir = sources.dataset_dir(args.project, NAME)
    known = (sources.read_provenance(args.project, NAME) or {}).get("files", {})
    listings, written, files, urls = {}, [], {}, []
    for ms in months:
        y = ms.year
        if y not in listings:
            body = sources.http_get("%s/%d/" % (BASE, y))
            listings[y] = list_month_files(body.decode("utf-8", "replace")) if body else {}
        name = listings[y].get(ms.strftime("%Y%m"))
        if name is None:
            print("gpcp_monthly: no GPCP file for %s (not yet published or outside 1979-present)" % ms.strftime("%Y-%m"))
            continue
        url = "%s/%d/%s" % (BASE, y, name)
        sha, _ = sources.download(url, os.path.join(out_dir, name), known.get(name))
        if sha is None:
            print("gpcp_monthly: %s not available" % url)
            continue
        written.append(os.path.join(out_dir, name))
        files[name] = sha
        urls.append(url)
    if not written:
        return []
    sources.write_provenance(args.project, NAME, PROVENANCE, urls, files,
                             window=sources.window_dict(start, end), bbox=list(bbox))
    return written
