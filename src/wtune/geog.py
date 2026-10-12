"""wtune/geog.py -- OPTIONAL high-resolution WPS_GEOG datasets and external DEM import
(decision 0063).  `wtune prep geog <list|fetch|import-dem|verify|bounds|srtm-fetch>`.

Nothing here runs unless the user types the command: wtune tune / advise / calc never import
this module (tests/test_wtune_geog.py::test_g3_* enforce that statically).  Only data and a
COPY of GEOGRID.TBL are written; WRF/WPS sources and the official GEOGRID.TBL are never touched.

Facts this module relies on (checked in WPS 4.6 geogrid/src/source_data_module.F):
  * GEOGRID.TBL accepts `rel_path = <res>:<dir>/` (joined to geog_data_path) and
    `abs_path = <res>:</absolute/dir/>` (line ~495), both limited to 128 characters.
  * geog_data_res = 'a+b+default' selects, PER FIELD, the FIRST key that has a path in the table;
    exactly ONE source dataset is read per field.  '+default' is a table-lookup fallback for
    fields lacking the key, NOT a per-grid-point fallback to the default dataset: points the
    imported DEM does not cover read as missing_value and HGT_M then gets fill_missing=0.
    Hence import-dem has --fill-from (void fill from an existing WPS dataset) and
    --require-bounds (refuse a DEM that does not cover the domain).
  * Index keywords/format: WPS User Guide, "Writing Static Data to the Geogrid Binary Format"
    (https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/wps.html#writing-static-data-to-the-geogrid-binary-format).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tarfile
from datetime import datetime, timezone

import numpy as np

DEFAULT_GEOG = "/nfs/data/wrf/WPS_GEOG/"
FALLBACK_DEST = "~/wps-geog-extra"
MISSING_I16 = -32768
URL_BASE = "https://www2.mmm.ucar.edu/wrf/src/wps_files/"
URL_PAGE = "https://www2.mmm.ucar.edu/wrf/users/download/get_sources_wps_geog.html"
SKADI_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/%s/%s.hgt.gz"

# GEOGRID.TBL resolution key -> official tarball.  Names/links read from URL_PAGE and sizes from
# HEAD requests on 2026-10-01; the tarballs' top-level directory is the dataset directory
# (verified by streaming tar listings of modis_landuse_20class_15s and geog_older_than_2000).
CATALOG = {
    "modis_15s":     dict(tar="modis_landuse_20class_15s.tar.bz2", dir="modis_landuse_20class_15s", bytes=65224208),
    "modis_15s_lake": dict(tar="modis_landuse_20class_15s_with_lakes.tar.gz", dir="modis_landuse_20class_15s_with_lakes", bytes=83851162),
    "modis_30s_lake": dict(tar="modis_landuse_20class_30s_with_lakes.tar.bz2", dir="modis_landuse_20class_30s_with_lakes", bytes=32532921),
    "usgs_lakes":    dict(tar="landuse_30s_with_lakes.tar.bz2", dir="landuse_30s_with_lakes", bytes=21037100),
    "nlcd2006_9s":   dict(tar="nlcd2006_ll_9s.tar.bz2", dir="nlcd2006_ll_9s", bytes=537939081),
    "nlcd2011_9s":   dict(tar="nlcd2011_ll_9s.tar.bz2", dir="nlcd2011_ll_9s", bytes=586571892),
    "gmted2010_30s": dict(tar="topo_gmted2010_30s.tar.bz2", dir="topo_gmted2010_30s", bytes=182721518),
}
NOT_DISTRIBUTED = {
    "usgs_30s": "no stand-alone tarball on %s (dir landuse_30s/ is not in any listed archive; "
                "usgs_lakes = landuse_30s_with_lakes is)" % URL_PAGE,
}


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------

def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return "%.1f %s" % (n, u) if u != "B" else "%d B" % n
        n /= 1024.0


def _geog_path(arg=None):
    return os.path.expanduser(arg or os.environ.get("WPS_GEOG") or DEFAULT_GEOG)


def _wps_dir(arg=None):
    cands = [arg, os.environ.get("WPS_DIR"), "~/wps-build/WPS-4.6.0"]
    for c in cands:
        if c and os.path.isfile(os.path.join(os.path.expanduser(c), "geogrid", "GEOGRID.TBL")):
            return os.path.expanduser(c)
    return os.path.expanduser(arg or os.environ.get("WPS_DIR") or "~/wps-build/WPS-4.6.0")


def _tbl_path(wps_dir):
    return os.path.join(wps_dir, "geogrid", "GEOGRID.TBL")


def _fmt(x):
    return "%.12g" % x


# --------------------------------------------------------------------------------------------
# GEOGRID.TBL parsing and the table COPY
# --------------------------------------------------------------------------------------------

_PATH_RE = re.compile(r"^\s*(rel_path|abs_path)\s*=\s*(\S+)\s*$")
_NAME_RE = re.compile(r"^\s*name\s*=\s*(\S+)")


def split_tbl(text):
    """[(name|None, [lines])]: one entry per `====` separated field block (lines keep no newline)."""
    blocks, cur = [], [None, []]
    for ln in text.splitlines():
        if ln.strip().startswith("====="):
            blocks.append(cur)
            cur = [None, [ln]]
            continue
        m = _NAME_RE.match(ln)
        if m and cur[0] is None:
            cur[0] = m.group(1)
        cur[1].append(ln)
    blocks.append(cur)
    return [(n, ls) for n, ls in blocks]


def tbl_entries(text):
    """[(field, res, path, kind)] for every rel_path/abs_path line; kind 'rel'|'abs'."""
    out = []
    for name, lines in split_tbl(text):
        for ln in lines:
            m = _PATH_RE.match(ln)
            if m:
                res, _, p = m.group(2).partition(":") if ":" in m.group(2) else ("default", "", m.group(2))
                out.append((name, res, p, "rel" if m.group(1) == "rel_path" else "abs"))
    return out


def tbl_copy_text(base_text, field, res, path, absolute, interp="average_gcell(4.0)+four_pt+average_4pt"):
    """GEOGRID.TBL text with `interp_option = <res>:<interp>` and `rel_path|abs_path = <res>:<path>`
    added to `field`'s block (after the last such line).  Raises ValueError when the field is
    missing or already defines `res`."""
    out, found = [], False
    for name, lines in split_tbl(base_text):
        if name != field:
            out.extend(lines)
            continue
        found = True
        for ln in lines:
            if re.match(r"^\s*(interp_option|rel_path|abs_path)\s*=\s*%s\s*:" % re.escape(res), ln):
                raise ValueError("field %s already defines resolution %r" % (field, res))
        lines = list(lines)

        def insert_after_last(prefixes, new):
            idx = max((i for i, ln in enumerate(lines)
                       if re.match(r"^\s*(%s)\s*=" % "|".join(prefixes), ln)), default=None)
            if idx is None:
                idx = len(lines) - 1
                while idx > 0 and not lines[idx].strip():
                    idx -= 1
            indent = re.match(r"\s*", lines[idx]).group(0) or "        "
            lines.insert(idx + 1, "%s%s" % (indent, new))

        insert_after_last(("interp_option",), "interp_option = %s:%s" % (res, interp))
        insert_after_last(("rel_path", "abs_path"),
                          "%s = %s:%s" % ("abs_path" if absolute else "rel_path", res, path))
        out.extend(lines)
    if not found:
        raise ValueError("field %s not found in GEOGRID.TBL" % field)
    return "\n".join(out) + "\n"


def tbl_abs_copy_text(base_text, dir_to_abs):
    """GEOGRID.TBL text where every rel_path whose directory is a key of dir_to_abs becomes an
    abs_path to that absolute directory (used when a fetched dataset lives outside geog_data_path)."""
    out = []
    for ln in base_text.splitlines():
        m = _PATH_RE.match(ln)
        if m and m.group(1) == "rel_path" and ":" in m.group(2):
            res, _, p = m.group(2).partition(":")
            if p in dir_to_abs:
                indent = re.match(r"\s*", ln).group(0)
                ln = "%sabs_path = %s:%s" % (indent, res, dir_to_abs[p])
        out.append(ln)
    return "\n".join(out) + "\n"


def write_tbl_copy(text, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "GEOGRID.TBL")
    with open(p, "w") as fh:
        fh.write(text)
    return p


# --------------------------------------------------------------------------------------------
# WPS index files and tiles
# --------------------------------------------------------------------------------------------

def parse_index(path):
    """{key: value} of a geogrid `index` file (quotes stripped, `#` comments dropped)."""
    d = {}
    with open(path) as fh:
        for ln in fh:
            ln = ln.split("#", 1)[0].strip()
            if "=" in ln:
                k, v = ln.split("=", 1)
                d[k.strip()] = v.strip().strip("\"'")
    return d


def index_resolution(idx):
    """(dx, dy, human text) of an index dict."""
    try:
        dx, dy = float(idx["dx"]), float(idx["dy"])
    except (KeyError, ValueError):
        return None, None, "?"
    if idx.get("projection", "") == "regular_ll":
        return dx, dy, "%.4g deg = %.4g arcsec = ~%.0f m" % (dx, dx * 3600, dx * 111320)
    return dx, dy, "%.4g m" % dx


def _tile_name(x0, x1, y0, y1, digits=5):
    f = "%%0%dd" % digits
    return (f + "-" + f + "." + f + "-" + f) % (x0, x1, y0, y1)


_TILE_RE = re.compile(r"^(\d+)-(\d+)\.(\d+)-(\d+)$")


class WpsTiles:
    """Reader for one WPS binary dataset directory (continuous or categorical, 1 level)."""

    def __init__(self, path):
        self.path = path
        self.idx = parse_index(os.path.join(path, "index"))
        i = self.idx
        self.wordsize = int(i["wordsize"])
        self.tx, self.ty = int(i["tile_x"]), int(i["tile_y"])
        self.signed = i.get("signed", "no") == "yes"
        self.scale = float(i.get("scale_factor", 1.0))
        self.missing = float(i["missing_value"]) if "missing_value" in i else None
        self.bdr = int(i.get("tile_bdr", 0))
        self.top_bottom = i.get("row_order", "bottom_top") == "top_bottom"
        self.dtype = np.dtype((">" if i.get("endian", "big") == "big" else "<")
                              + ("i" if self.signed else "u") + str(self.wordsize))
        self.digits = int(i.get("filename_digits", 5))
        self.dx, self.dy = float(i["dx"]), float(i["dy"])
        self.kx, self.ky = float(i.get("known_x", 1)), float(i.get("known_y", 1))
        self.klat, self.klon = float(i["known_lat"]), float(i["known_lon"])
        self._cache = {}

    def tile_files(self):
        out = []
        for p in sorted(os.listdir(self.path)):
            m = _TILE_RE.match(p)
            if m:
                out.append((p, tuple(int(g) for g in m.groups())))
        return out

    def tile(self, x0, y0):
        """Raw (not scaled) tile array [row S->N, col W->E] starting at 1-based (x0, y0), or None."""
        key = (x0, y0)
        if key not in self._cache:
            p = os.path.join(self.path, _tile_name(x0, x0 + self.tx - 1, y0, y0 + self.ty - 1, self.digits))
            if not os.path.exists(p):
                self._cache[key] = None
            else:
                a = np.fromfile(p, dtype=self.dtype).reshape(self.ty + 2 * self.bdr, self.tx + 2 * self.bdr)
                a = a[self.bdr:self.bdr + self.ty, self.bdr:self.bdr + self.tx]
                self._cache[key] = a[::-1] if self.top_bottom else a
        return self._cache[key]

    def assemble(self):
        """(raw array [S->N, W->E], i0, j0): every present tile placed on the 1-based index grid;
        absent areas hold `missing` (0 if the index has none)."""
        files = self.tile_files()
        if not files:
            raise ValueError("%s: no tiles" % self.path)
        x0 = min(f[1][0] for f in files); x1 = max(f[1][1] for f in files)
        y0 = min(f[1][2] for f in files); y1 = max(f[1][3] for f in files)
        fill = int(self.missing) if self.missing is not None else 0
        arr = np.full((y1 - y0 + 1, x1 - x0 + 1), fill, dtype=self.dtype.newbyteorder("="))
        for _, (a, b, c, d) in files:
            t = self.tile(a, c)
            arr[c - y0:d - y0 + 1, a - x0:b - x0 + 1] = t
        return arr, x0, y0

    def sample_nearest(self, lat, lon):
        """Scaled nearest-cell values at lat/lon arrays; NaN where there is no tile or the cell
        equals missing_value."""
        lat = np.asarray(lat, float); lon = np.asarray(lon, float)
        i = np.rint((lon - self.klon) / self.dx + self.kx).astype(np.int64)
        j = np.rint((lat - self.klat) / self.dy + self.ky).astype(np.int64)
        out = np.full(lat.shape, np.nan)
        tx0 = (i - 1) // self.tx * self.tx + 1
        ty0 = (j - 1) // self.ty * self.ty + 1
        for a, b in set(zip(tx0.ravel().tolist(), ty0.ravel().tolist())):
            t = self.tile(a, b)
            if t is None:
                continue
            sel = (tx0 == a) & (ty0 == b)
            v = t[j[sel] - b, i[sel] - a].astype(float)
            if self.missing is not None:
                v[v == self.missing] = np.nan
            out[sel] = v * self.scale
        return out


# --------------------------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------------------------

def inventory(geog_path):
    """[{dir, index_path, resolution_text, dx, dy, type, projection, units, description}] for every
    directory under geog_path (depth <= 2) that holds an `index` file."""
    out = []
    geog_path = os.path.expanduser(geog_path)
    if not os.path.isdir(geog_path):
        return out
    for depth_glob in ("*/index", "*/*/index"):
        for ip in sorted(glob.glob(os.path.join(geog_path, depth_glob))):
            rel = os.path.relpath(os.path.dirname(ip), geog_path)
            try:
                idx = parse_index(ip)
            except OSError:
                continue
            dx, dy, txt = index_resolution(idx)
            out.append(dict(dir=rel, dx=dx, dy=dy, resolution=txt, type=idx.get("type", "?"),
                            projection=idx.get("projection", "?"), units=idx.get("units", ""),
                            description=idx.get("description", "")))
    return out


def build_list(geog_path, wps_dir):
    tbl = _tbl_path(wps_dir)
    entries = []
    tbl_error = None
    try:
        with open(tbl) as fh:
            entries = tbl_entries(fh.read())
    except OSError as e:
        tbl_error = "cannot read %s: %s" % (tbl, e)
    by_dir = {}
    for field, res, p, kind in entries:
        by_dir.setdefault(p.rstrip("/"), []).append("%s:%s" % (field, res))
    inv = inventory(geog_path)
    have = {d["dir"] for d in inv}
    for d in inv:
        d["tbl_uses"] = by_dir.get(d["dir"], [])
    missing = []
    for p, uses in sorted(by_dir.items()):
        if p.startswith("/") or p in have:
            continue
        keys = sorted({u.split(":", 1)[1] for u in uses})
        fetch = [k for k in keys if k in CATALOG and CATALOG[k]["dir"] == p]
        missing.append(dict(dir=p, tbl_uses=uses, fetch_keys=fetch,
                            fetch_bytes=CATALOG[fetch[0]]["bytes"] if fetch else None,
                            note=next((NOT_DISTRIBUTED[k] for k in keys if k in NOT_DISTRIBUTED), "")))
    return dict(geog_path=geog_path, wps_dir=wps_dir, tbl=tbl, tbl_error=tbl_error,
                present=inv, missing=missing)


def cmd_list(a):
    r = build_list(_geog_path(a.geog_path), _wps_dir(a.wps_dir))
    if a.json:
        print(json.dumps(r, indent=2))
        return 0
    print("WPS_GEOG : %s%s" % (r["geog_path"], "" if os.path.isdir(r["geog_path"]) else "  (not a directory)"))
    print("GEOGRID.TBL: %s%s" % (r["tbl"], "  (" + r["tbl_error"] + ")" if r["tbl_error"] else ""))
    print("\nPRESENT (%d datasets with an index file)" % len(r["present"]))
    for d in r["present"]:
        uses = ",".join(d["tbl_uses"]) or "(not referenced by this GEOGRID.TBL)"
        print("  %-48s %-30s %-11s %s" % (d["dir"], d["resolution"], d["type"], uses))
    print("\nKNOWN TO GEOGRID.TBL BUT MISSING (%d)" % len(r["missing"]))
    for m in r["missing"]:
        how = ("`wtune prep geog fetch %s` (%s download)" % (m["fetch_keys"][0], _human(m["fetch_bytes"]))
               if m["fetch_keys"] else (m["note"] or "no official single-file download known to wtune"))
        print("  %-48s %s\n  %-48s -> %s" % (m["dir"], ",".join(m["tbl_uses"][:4]) + (",..." if len(m["tbl_uses"]) > 4 else ""), "", how))
    return 0


# --------------------------------------------------------------------------------------------
# fetch (official WPS datasets)
# --------------------------------------------------------------------------------------------

def _head_size(url, timeout=30):
    import urllib.request
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "wtune-geog"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def _download(url, dest):
    from wtune.post import sources
    return sources.download(url, dest)


def _confirm(prompt, yes):
    if yes:
        return True
    if not sys.stdin.isatty():
        print("%s\nnot confirmed (no --yes and stdin is not a terminal); nothing downloaded." % prompt)
        return False
    return input(prompt + " [y/N] ").strip().lower() in ("y", "yes")


def _safe_extract(tar_path, dest):
    """Extract tar_path into dest, dropping a leading 'WPS_GEOG/' and refusing absolute/.. members."""
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            name = m.name
            if name.startswith("WPS_GEOG/"):
                name = name[len("WPS_GEOG/"):]
            if not name or name.startswith("/") or ".." in name.split("/"):
                continue
            if not (m.isfile() or m.isdir()):
                continue
            m.name = name
            tf.extract(m, dest, filter="data") if sys.version_info >= (3, 12) else tf.extract(m, dest)
            n += 1
    return n


def fetch_plan(key, geog_path, dest=None):
    if key in NOT_DISTRIBUTED:
        raise SystemExit("%s: %s" % (key, NOT_DISTRIBUTED[key]))
    ent = CATALOG.get(key)
    if ent is None:
        for k, v in CATALOG.items():
            if v["dir"] == key:
                ent, key = v, k
        if ent is None:
            raise SystemExit("unknown dataset %r; known: %s" % (key, ", ".join(sorted(CATALOG))))
    explicit = dest is not None
    target = os.path.expanduser(dest or geog_path)
    fallback = False
    if not explicit and not (os.path.isdir(target) and os.access(target, os.W_OK)):
        target, fallback = os.path.expanduser(FALLBACK_DEST), True
    return dict(key=key, tar=ent["tar"], dir=ent["dir"], url=URL_BASE + ent["tar"], dest=target,
                fallback=fallback, approx_bytes=ent["bytes"])


def cmd_fetch(a):
    geog = _geog_path(a.geog_path)
    plan = fetch_plan(a.dataset, geog, a.dest)
    try:
        size = _head_size(plan["url"])
    except Exception as e:  # noqa: BLE001
        size = None
        print("note: HEAD %s failed (%s); using the catalog size" % (plan["url"], e))
    size = size or plan["approx_bytes"]
    final = os.path.join(plan["dest"], plan["dir"])
    print("dataset   : %s -> %s/" % (plan["key"], final))
    print("source    : %s  (WPS geog downloads, %s)" % (plan["url"], URL_PAGE))
    print("download  : %s compressed (extracted size is larger; check free space on %s)" % (_human(size), plan["dest"]))
    if plan["fallback"]:
        print("\n%s is not writable, so the dataset goes to %s.\n"
              "geogrid reads ONE geog_data_path: rel_path entries of GEOGRID.TBL are joined to it.  Options:\n"
              "  (a) point geog_data_path at %s and symlink the existing datasets into it\n"
              "      (re-run with --link-existing, or: ln -s %s*/ %s), keep the official GEOGRID.TBL; or\n"
              "  (b) keep geog_data_path and use the GEOGRID.TBL COPY written next to the dataset, in which this\n"
              "      dataset's rel_path lines became `abs_path = <res>:<absolute dir>/` (supported by geogrid,\n"
              "      source_data_module.F; 128-character limit), via opt_geogrid_tbl_path."
              % (geog, plan["dest"], plan["dest"], geog, plan["dest"] + "/"))
    if os.path.exists(os.path.join(final, "index")) and not a.force:
        print("\nalready present: %s/index exists (use --force to re-download)" % final)
        return 0
    if not _confirm("\nDownload and extract %s?" % plan["key"], a.yes):
        return 2
    os.makedirs(plan["dest"], exist_ok=True)
    cache = os.path.join(plan["dest"], ".wtune-download")
    tar_path = os.path.join(cache, plan["tar"])
    sha, _ = _download(plan["url"], tar_path)
    if sha is None:
        print("server says the file does not exist: %s" % plan["url"])
        return 1
    n = _safe_extract(tar_path, plan["dest"])
    if not os.path.exists(os.path.join(final, "index")):
        print("extraction wrote %d members but %s/index is missing" % (n, final))
        return 1
    prov = dict(dataset="WPS geog %s (%s)" % (plan["key"], plan["dir"]),
                provider="NCAR/MMM WRF Preprocessing System geographical static data",
                url=plan["url"], license="public WPS static data; see %s" % URL_PAGE,
                citation="Skamarock et al., WRF-ARW; WPS_GEOG static data downloads, NCAR MMM",
                coverage_time="static", coverage_space="global (dataset dependent)",
                resolution=index_resolution(parse_index(os.path.join(final, "index")))[2],
                access="HTTPS download, no key",
                urls=[plan["url"]], files={plan["tar"]: sha}, fetched_utc=_utc_now(),
                window=None, bbox=None)
    with open(os.path.join(final, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    if not a.keep_archive:
        os.remove(tar_path)
    print("\nextracted %d members -> %s (provenance.json written)" % (n, final))
    tbl_dir = None
    if plan["fallback"]:
        if a.link_existing:
            _link_existing(geog, plan["dest"])
        try:
            with open(_tbl_path(_wps_dir(a.wps_dir))) as fh:
                base = fh.read()
            tbl_dir = os.path.join(plan["dest"], "GEOGRID.TBL.d")
            write_tbl_copy(tbl_abs_copy_text(base, {plan["dir"] + "/": final.rstrip("/") + "/"}), tbl_dir)
        except OSError as e:
            print("could not write the GEOGRID.TBL copy: %s" % e)
    print_namelist_hint(plan["key"], plan["dest"] if plan["fallback"] else geog, tbl_dir,
                        link_note=plan["fallback"] and not a.link_existing)
    return 0


def _link_existing(geog, dest):
    n = 0
    for p in sorted(glob.glob(os.path.join(geog, "*/"))):
        link = os.path.join(dest, os.path.basename(p.rstrip("/")))
        if not os.path.lexists(link):
            os.symlink(p.rstrip("/"), link)
            n += 1
    print("symlinked %d existing dataset dirs from %s into %s" % (n, geog, dest))


def print_namelist_hint(res, geog_data_path, tbl_dir, link_note=False):
    print("\nnamelist.wps (&geogrid), one value per domain:")
    print("  geog_data_res = '%s+default', '%s+default'," % (res, res))
    print("  geog_data_path = '%s'," % geog_data_path.rstrip("/") + "/")
    if tbl_dir:
        print("  opt_geogrid_tbl_path = '%s/'," % tbl_dir.rstrip("/"))
    if link_note:
        print("  # geog_data_path above only works after the existing datasets are linked into it (--link-existing).")
    print("'+default' only supplies the table lookup for fields that have no '%s' entry; every field uses ONE source." % res)


# --------------------------------------------------------------------------------------------
# DEM rasters: GeoTIFF (rasterio, else Pillow), SRTM .hgt(.gz)
# --------------------------------------------------------------------------------------------

class Raster:
    """arr[row S->N, col W->E] float32 (NaN = nodata); lat0/lon0 = centre of arr[0,0]; dx/dy deg > 0."""

    def __init__(self, arr, lat0, lon0, dx, dy):
        self.arr, self.lat0, self.lon0, self.dx, self.dy = arr, float(lat0), float(lon0), float(dx), float(dy)

    @property
    def shape(self):
        return self.arr.shape

    def bounds_edges(self):
        ny, nx = self.arr.shape
        return (self.lat0 - self.dy / 2, self.lat0 + (ny - 0.5) * self.dy,
                self.lon0 - self.dx / 2, self.lon0 + (nx - 0.5) * self.dx)


def _read_hgt(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as fh:
        raw = fh.read()
    n = int(round(math.sqrt(len(raw) // 2)))
    if n * n * 2 != len(raw):
        raise ValueError("%s: not a square .hgt (%d bytes)" % (path, len(raw)))
    base = os.path.basename(path)
    m = re.match(r"([NS])(\d{2})([EW])(\d{3})", base.upper())
    if not m:
        raise ValueError("%s: cannot read the tile corner from the file name (e.g. N37W122.hgt)" % path)
    lat0 = (1 if m.group(1) == "N" else -1) * int(m.group(2))
    lon0 = (1 if m.group(3) == "E" else -1) * int(m.group(4))
    a = np.frombuffer(raw, dtype=">i2").reshape(n, n)[::-1].astype(np.float32)
    a[a == MISSING_I16] = np.nan
    d = 1.0 / (n - 1)
    return Raster(a, lat0, lon0, d, d)


def _read_tiff_pillow(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    tags = im.tag_v2
    arr = np.array(im)
    if arr.ndim != 2:
        raise ValueError("%s: only single-band GeoTIFF is supported" % path)
    scale, tie = tags.get(33550), tags.get(33922)
    if scale is None or tie is None or tags.get(34264):
        raise ValueError("%s: no ModelPixelScale/ModelTiepoint (rotated or non-GeoTIFF rasters are unsupported)" % path)
    point = False
    gk = tags.get(34735)
    if gk:
        gk = list(gk)
        for k in range(4, len(gk) - 3, 4):
            if gk[k] == 1025:
                point = gk[k + 3] == 2
            if gk[k] == 1024 and gk[k + 3] != 2:
                raise ValueError("%s: raster is not in a geographic (lat/lon) CRS" % path)
    nodata = tags.get(42113)
    nodata = float(str(nodata).strip("\x00 ")) if nodata not in (None, "") else None
    return _raster_from_top_down(arr, tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1],
                                 scale[0], scale[1], point, nodata)


def _raster_from_top_down(arr, lon_ul, lat_ul, sx, sy, point, nodata):
    a = arr.astype(np.float32)
    if nodata is not None:
        a[arr == nodata] = np.nan
    elif np.issubdtype(arr.dtype, np.integer):
        a[arr == MISSING_I16] = np.nan
    a = a[::-1]
    ny = a.shape[0]
    off = 0.0 if point else 0.5
    return Raster(a, lat_ul - (ny - 1 + off) * sy, lon_ul + off * sx, sx, sy)


def _read_tiff_rasterio(path):
    import rasterio  # noqa: PLC0415 -- optional dependency
    with rasterio.open(path) as ds:
        if ds.crs is not None and not ds.crs.is_geographic:
            raise ValueError("%s: raster is not in a geographic (lat/lon) CRS" % path)
        t = ds.transform
        if t.b or t.d or t.e > 0:
            raise ValueError("%s: rotated or south-up raster is unsupported" % path)
        point = (ds.tags().get("AREA_OR_POINT", "Area") == "Point")
        return _raster_from_top_down(ds.read(1), t.c, t.f, t.a, -t.e, point, ds.nodata)


def read_raster(path):
    if re.search(r"\.hgt(\.gz)?$", path, re.I):
        return _read_hgt(path)
    try:
        import rasterio  # noqa: F401, PLC0415
    except ImportError:
        return _read_tiff_pillow(path)
    return _read_tiff_rasterio(path)


def mosaic(rasters):
    """One Raster from same-spacing rasters on a common lattice (e.g. SRTM tiles, which overlap by one
    row/column: later rasters overwrite, valid data is never replaced by NaN)."""
    if len(rasters) == 1:
        return rasters[0]
    dx, dy = rasters[0].dx, rasters[0].dy
    for r in rasters:
        if abs(r.dx - dx) > 1e-9 * dx or abs(r.dy - dy) > 1e-9 * dy:
            raise ValueError("rasters have different pixel sizes; resample them to one grid first")
    lat0 = min(r.lat0 for r in rasters); lon0 = min(r.lon0 for r in rasters)
    offs = []
    for r in rasters:
        j, i = (r.lat0 - lat0) / dy, (r.lon0 - lon0) / dx
        if abs(j - round(j)) > 1e-3 or abs(i - round(i)) > 1e-3:
            raise ValueError("rasters are not on a common pixel lattice")
        offs.append((int(round(j)), int(round(i))))
    ny = max(j + r.shape[0] for (j, _), r in zip(offs, rasters))
    nx = max(i + r.shape[1] for (_, i), r in zip(offs, rasters))
    out = np.full((ny, nx), np.nan, np.float32)
    for (j, i), r in zip(offs, rasters):
        view = out[j:j + r.shape[0], i:i + r.shape[1]]
        ok = ~np.isnan(r.arr)
        view[ok] = r.arr[ok]
    return Raster(out, lat0, lon0, dx, dy)


# --------------------------------------------------------------------------------------------
# import-dem
# --------------------------------------------------------------------------------------------

def _check_name(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", name):
        raise SystemExit("--name must match [a-z0-9_]+ (it becomes a GEOGRID.TBL resolution key): %r" % name)
    if name in CATALOG or name in NOT_DISTRIBUTED or name in ("default", "lowres"):
        raise SystemExit("--name %r collides with an official GEOGRID.TBL resolution key" % name)


def _fill_voids(r, base):
    ny, nx = r.shape
    jj, ii = np.nonzero(np.isnan(r.arr))
    if not len(jj):
        return 0
    lat = r.lat0 + jj * r.dy
    lon = r.lon0 + ii * r.dx
    v = base.sample_nearest(lat, lon)
    ok = ~np.isnan(v)
    r.arr[jj[ok], ii[ok]] = v[ok].astype(np.float32)
    return int(ok.sum())


def write_wps_dataset(r, out_dir, name, tile=1200, units="meters MSL", description=None, scale_factor=1.0):
    """Quantise r to big-endian int16 tiles + `index` in out_dir.  Returns the index dict written."""
    ny, nx = r.shape
    q = np.rint(r.arr / scale_factor)
    valid = ~np.isnan(q)
    if valid.any() and (q[valid].min() < -32767 or q[valid].max() > 32767):
        raise ValueError("elevation range %.0f..%.0f does not fit int16 at scale_factor %g"
                         % (q[valid].min() * scale_factor, q[valid].max() * scale_factor, scale_factor))
    out = np.where(valid, q, MISSING_I16).astype(">i2")
    os.makedirs(out_dir, exist_ok=True)
    digits = max(5, len(str(max(nx, ny) + tile)))
    ntiles = 0
    for y0 in range(0, ny, tile):
        for x0 in range(0, nx, tile):
            sub = out[y0:y0 + tile, x0:x0 + tile]
            if (sub == MISSING_I16).all():
                continue
            t = np.full((tile, tile), MISSING_I16, dtype=">i2")
            t[:sub.shape[0], :sub.shape[1]] = sub
            fn = _tile_name(x0 + 1, x0 + tile, y0 + 1, y0 + tile, digits)
            t.tofile(os.path.join(out_dir, fn))      # row 0 = southernmost (row_order=bottom_top)
            ntiles += 1
    index = [("type", "continuous"), ("signed", "yes"), ("projection", "regular_ll"),
             ("dx", _fmt(r.dx)), ("dy", _fmt(r.dy)), ("known_x", "1"), ("known_y", "1"),
             ("known_lat", _fmt(r.lat0)), ("known_lon", _fmt(r.lon0)), ("wordsize", "2"),
             ("tile_x", str(tile)), ("tile_y", str(tile)), ("tile_z", "1"),
             ("units", '"%s"' % units),
             ("description", '"%s"' % (description or "User DEM %s imported by wtune prep geog" % name)),
             ("missing_value", str(MISSING_I16))]
    if scale_factor != 1.0:
        index.append(("scale_factor", _fmt(scale_factor)))
    if digits != 5:
        index.append(("filename_digits", str(digits)))
    with open(os.path.join(out_dir, "index"), "w") as fh:
        fh.write("\n".join("%s=%s" % kv for kv in index) + "\n")
    return dict(index), ntiles


def namelist_text(name, geog_data_path, tbl_dir, ndom=2):
    return ("&geogrid\n  geog_data_res = %s,\n  geog_data_path = '%s',\n  opt_geogrid_tbl_path = '%s/',\n/\n"
            % (", ".join(["'%s+default'" % name] * ndom), geog_data_path.rstrip("/") + "/", tbl_dir.rstrip("/")))


def import_dem(inputs, name, dest=None, field="HGT_M", tbl_out=None, geog_path=None, wps_dir=None,
               tile=1200, fill_from=None, require_bounds=None, units="meters MSL", description=None):
    """Returns a result dict (dataset dir, tbl copy, bounds, namelist text, counts)."""
    _check_name(name)
    r = mosaic([read_raster(p) for p in inputs])
    s, n, w, e = r.bounds_edges()
    if w < -180 or e > 180:
        raise SystemExit("longitudes %.4f..%.4f outside -180..180 are unsupported" % (w, e))
    if require_bounds:
        rs, rn, rw, re_ = require_bounds
        if rs < s or rn > n or rw < w or re_ > e:
            raise SystemExit("DEM covers S=%.5f N=%.5f W=%.5f E=%.5f but the domain needs S=%.5f N=%.5f W=%.5f E=%.5f: "
                             "outside the DEM geogrid reads missing_value and HGT_M becomes 0 (fill_missing=0); "
                             "'+default' does not fall back per grid point" % (s, n, w, e, rs, rn, rw, re_))
    nvoid0 = int(np.isnan(r.arr).sum())
    nfilled = 0
    if fill_from and nvoid0:
        nfilled = _fill_voids(r, WpsTiles(os.path.expanduser(fill_from)))
    dest_root = os.path.expanduser(dest or FALLBACK_DEST)
    ds_dir = os.path.join(dest_root, name)
    if os.path.exists(os.path.join(ds_dir, "index")):
        shutil.rmtree(ds_dir)
    idx, ntiles = write_wps_dataset(r, ds_dir, name, tile=tile, units=units, description=description)
    geog = _geog_path(geog_path)
    wps = _wps_dir(wps_dir)
    with open(_tbl_path(wps)) as fh:
        base_text = fh.read()
    geog_n = os.path.normpath(geog) + os.sep
    inside = os.path.normpath(ds_dir).startswith(geog_n)
    path_value = (os.path.relpath(ds_dir, geog).rstrip("/") + "/") if inside else os.path.abspath(ds_dir).rstrip("/") + "/"
    warn = []
    if len("%s:%s" % (name, path_value)) > 127 or (inside and len(geog_n + path_value) > 127):
        warn.append("GEOGRID.TBL path lines are limited to 128 characters; use a shorter --dest")
    text = tbl_copy_text(base_text, field, name, path_value, absolute=not inside)
    tbl_dir = os.path.expanduser(tbl_out) if tbl_out else os.path.join(dest_root, name + ".geogrid")
    tbl_file = write_tbl_copy(text, tbl_dir)
    prov = dict(dataset="User DEM %s" % name, provider="user supplied (%s)" % ", ".join(os.path.basename(p) for p in inputs),
                url="-", license="per source file; user responsibility", citation="-",
                coverage_time="static", coverage_space="S=%.6f N=%.6f W=%.6f E=%.6f" % (s, n, w, e),
                resolution="dx=%s deg dy=%s deg" % (_fmt(r.dx), _fmt(r.dy)), access="local file",
                urls=[], files={os.path.abspath(p): _sha256(p) for p in inputs}, fetched_utc=_utc_now(),
                window=None, bbox=[s, n, w, e],
                **{"import": dict(sources=[os.path.abspath(p) for p in inputs], field=field, shape=list(r.shape),
                                  tile=tile, fill_from=os.path.abspath(os.path.expanduser(fill_from)) if fill_from else None,
                                  voids_before_fill=nvoid0, voids_filled=nfilled, tbl=os.path.abspath(tbl_file))})
    with open(os.path.join(ds_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    return dict(dataset_dir=ds_dir, tbl=tbl_file, tbl_dir=tbl_dir, bounds=dict(south=s, north=n, west=w, east=e),
                shape=list(r.shape), ntiles=ntiles, voids=nvoid0, filled=nfilled, index=idx, warnings=warn,
                path_line=("rel_path" if inside else "abs_path") + " = %s:%s" % (name, path_value),
                namelist=namelist_text(name, geog, tbl_dir))


def cmd_import(a):
    rb = None
    if a.require_bounds:
        rb = tuple(float(x) for x in a.require_bounds)
    if a.require_bounds_from:
        rb = domain_bounds(a.require_bounds_from, a.domain, pad=a.pad)
    r = import_dem(a.inputs, a.name, a.dest, a.field, a.tbl_out, a.geog_path, a.wps_dir, a.tile,
                   a.fill_from, rb, description=a.description)
    b = r["bounds"]
    print("dataset   : %s (%d tiles, %dx%d points, dx=%s)" % (r["dataset_dir"], r["ntiles"], r["shape"][1], r["shape"][0], r["index"]["dx"]))
    print("coverage  : S=%.5f N=%.5f W=%.5f E=%.5f (pixel edges); geogrid reads missing_value outside it" % (
        b["south"], b["north"], b["west"], b["east"]))
    if r["voids"]:
        print("voids     : %d nodata pixels, %d filled from --fill-from%s" % (
            r["voids"], r["filled"], "" if a.fill_from else " (none: no --fill-from; they stay missing -> HGT_M 0)"))
    print("GEOGRID.TBL copy: %s   (official table untouched; added %s and interp_option under %s)" % (r["tbl"], r["path_line"], a.field))
    for w in r["warnings"]:
        print("WARNING: " + w)
    print("\nnamelist.wps:\n" + r["namelist"])
    print("Only %s has a '%s' entry; every other field resolves through '+default'.  Per-point fallback to the default\n"
          "dataset does NOT exist in geogrid (one source per field): keep the domain inside the coverage above." % (a.field, a.name))
    return 0


# --------------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------------

def verify_dataset(path, source=None):
    """Decode every tile of the WPS dataset at `path` and compare with the source rasters (default:
    the ones recorded in provenance.json by import-dem).  Returns dict(ok, max_abs_diff, ...)."""
    wt = WpsTiles(path)
    res = dict(path=path, tiles=len(wt.tile_files()), ok=True, problems=[])
    arr, i0, j0 = wt.assemble()
    dec = arr.astype(np.float64) * wt.scale
    miss = (arr == int(wt.missing)) if wt.missing is not None else np.zeros(arr.shape, bool)
    res["decoded_shape"] = list(arr.shape)
    for k, want in (("type", "continuous"), ("projection", "regular_ll"), ("wordsize", "2")):
        if wt.idx.get(k) != want:
            res["problems"].append("index %s=%r (expected %r)" % (k, wt.idx.get(k), want))
    srcs = [source] if source else None
    if srcs is None:
        try:
            with open(os.path.join(path, "provenance.json")) as fh:
                srcs = json.load(fh)["import"]["sources"]
        except (OSError, KeyError, ValueError):
            srcs = None
    if not srcs:
        res["note"] = "no source raster recorded or given: index/tile decode check only"
    else:
        r = mosaic([read_raster(p) for p in srcs])
        ny, nx = r.shape
        gj = int(round((r.lat0 - wt.klat) / wt.dy)) + int(round(wt.ky)) - j0
        gi = int(round((r.lon0 - wt.klon) / wt.dx)) + int(round(wt.kx)) - i0
        if gj < 0 or gi < 0 or gj + ny > arr.shape[0] or gi + nx > arr.shape[1]:
            res["problems"].append("source raster lies outside the decoded tile area")
        else:
            d, m = dec[gj:gj + ny, gi:gi + nx], miss[gj:gj + ny, gi:gi + nx]
            ok = ~np.isnan(r.arr)
            both = ok & ~m
            res["max_abs_diff"] = float(np.abs(d[both] - r.arr[both]).max()) if both.any() else 0.0
            res["tolerance"] = 0.5 * wt.scale + 1e-6
            res["valid_pixels"] = int(ok.sum())
            res["valid_decoded_as_missing"] = int((ok & m).sum())
            res["nodata_decoded_valid"] = int((~ok & ~m).sum())      # > 0 only after --fill-from
            if res["max_abs_diff"] > res["tolerance"]:
                res["problems"].append("max abs diff %.4g > %.4g" % (res["max_abs_diff"], res["tolerance"]))
            if res["valid_decoded_as_missing"]:
                res["problems"].append("%d valid source pixels decoded as missing_value" % res["valid_decoded_as_missing"])
    res["ok"] = not res["problems"]
    return res


def cmd_verify(a):
    r = verify_dataset(a.dir, a.source)
    print(json.dumps(r, indent=2))
    print("VERIFY %s" % ("PASS" if r["ok"] else "FAIL: " + "; ".join(r["problems"])))
    return 0 if r["ok"] else 1


# --------------------------------------------------------------------------------------------
# domain bounds (for --require-bounds-from and SRTM tile selection)
# --------------------------------------------------------------------------------------------

def _nl_values(text, key):
    m = re.search(r"^\s*%s\s*=\s*([^\n/&]+)" % re.escape(key), text, re.M | re.I)
    if not m:
        return None
    vals = [v.strip().strip("'\"") for v in m.group(1).replace("\n", " ").split(",") if v.strip()]
    return vals


def _bounds_from_nc(path):
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        for la, lo in (("XLAT_C", "XLONG_C"), ("XLAT_M", "XLONG_M"), ("XLAT", "XLONG")):
            if la in ds.variables:
                lat = np.asarray(ds.variables[la][0]); lon = np.asarray(ds.variables[lo][0])
                return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())
    raise ValueError("%s has no XLAT/XLONG" % path)


def _bounds_from_namelist_wps(path, domain):
    import pyproj
    t = open(path).read()
    g = lambda k, cast=str: [cast(v) for v in (_nl_values(t, k) or [])]  # noqa: E731
    proj = (g("map_proj") or ["lambert"])[0]
    dx = g("dx", float)[0]; dy = (g("dy", float) or [dx])[0]
    ref_lat, ref_lon = g("ref_lat", float)[0], g("ref_lon", float)[0]
    t1 = (g("truelat1", float) or [ref_lat])[0]; t2 = (g("truelat2", float) or [t1])[0]
    sl = (g("stand_lon", float) or [ref_lon])[0]
    R = 6370000.0
    if proj == "lambert":
        crs = pyproj.CRS.from_proj4("+proj=lcc +lat_1=%f +lat_2=%f +lat_0=%f +lon_0=%f +a=%f +b=%f +units=m" % (t1, t2, ref_lat, sl, R, R))
    elif proj == "mercator":
        crs = pyproj.CRS.from_proj4("+proj=merc +lat_ts=%f +lon_0=%f +a=%f +b=%f +units=m" % (t1, sl, R, R))
    elif proj == "polar":
        crs = pyproj.CRS.from_proj4("+proj=stere +lat_0=%f +lat_ts=%f +lon_0=%f +a=%f +b=%f +units=m" % (90 if ref_lat >= 0 else -90, t1, sl, R, R))
    else:
        raise ValueError("map_proj %r unsupported for namelist.wps bounds; give geo_em.dNN.nc or --bbox" % proj)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(ref_lon, ref_lat)
    we, sn = g("e_we", int), g("e_sn", int)
    ratio, ips, jps = g("parent_grid_ratio", int), g("i_parent_start", int), g("j_parent_start", int)
    par = g("parent_id", int) or [1] * len(we)
    x0 = cx - (we[0] - 1) * dx / 2; y0 = cy - (sn[0] - 1) * dy / 2
    org = {1: (x0, y0, dx, dy)}
    for d in range(2, domain + 1):
        px, py, pdx, pdy = org[par[d - 1]]
        org[d] = (px + (ips[d - 1] - 1) * pdx, py + (jps[d - 1] - 1) * pdy, pdx / ratio[d - 1], pdy / ratio[d - 1])
    x0, y0, ddx, ddy = org[domain]
    xs = np.linspace(x0, x0 + (we[domain - 1] - 1) * ddx, 41)
    ys = np.linspace(y0, y0 + (sn[domain - 1] - 1) * ddy, 41)
    bx = np.concatenate([xs, xs, np.full_like(ys, xs[0]), np.full_like(ys, xs[-1])])
    by = np.concatenate([np.full_like(xs, ys[0]), np.full_like(xs, ys[-1]), ys, ys])
    lon, lat = inv.transform(bx, by)
    return float(np.min(lat)), float(np.max(lat)), float(np.min(lon)), float(np.max(lon))


def domain_bounds(path, domain=2, pad=0.0):
    """(S, N, W, E) in degrees of domain `domain` from a geo_em/wrfinput/wrfout NetCDF, a namelist.wps,
    or a directory holding one of them (a WRF run dir with only namelist.input works through its
    wrfinput_dNN).  `pad` degrees are added on every side."""
    p = os.path.expanduser(path)
    if os.path.isdir(p):
        cands = []
        for pat in ("geo_em.d%02d.nc", "wrfinput_d%02d", "wrfout_d%02d_*", "wrfbdy_d%02d"):
            cands += sorted(glob.glob(os.path.join(p, pat % domain)))
        cands += [q for q in (os.path.join(p, "namelist.wps"),) if os.path.exists(q)]
        if not cands:
            raise ValueError("%s: no geo_em.d%02d.nc / wrfinput_d%02d / wrfout_d%02d_* / namelist.wps (namelist.input alone "
                             "has no map projection)" % (p, domain, domain, domain))
        p = cands[0]
    if os.path.basename(p).startswith("namelist"):
        if "input" in os.path.basename(p):
            d = os.path.dirname(os.path.abspath(p))
            return domain_bounds(d, domain, pad)
        b = _bounds_from_namelist_wps(p, domain)
    else:
        b = _bounds_from_nc(p)
    return (b[0] - pad, b[1] + pad, b[2] - pad, b[3] + pad)


def srtm_tiles(bbox):
    s, n, w, e = bbox
    out = []
    for la in range(int(math.floor(s)), int(math.floor(n)) + 1):
        for lo in range(int(math.floor(w)), int(math.floor(e)) + 1):
            nm = "%s%02d%s%03d" % ("N" if la >= 0 else "S", abs(la), "E" if lo >= 0 else "W", abs(lo))
            out.append((nm, SKADI_URL % (nm[:3], nm)))
    return out


def cmd_bounds(a):
    b = domain_bounds(a.path, a.domain, a.pad) if not a.bbox else tuple(float(x) for x in a.bbox)
    print("S=%.5f N=%.5f W=%.5f E=%.5f   (pad %.3f deg)" % (b + (a.pad,)))
    t = srtm_tiles(b)
    print("SRTM 1-arcsec tiles (AWS Terrain Tiles, skadi): %d" % len(t))
    for nm, u in t:
        print("  " + u)
    return 0


def cmd_srtm_fetch(a):
    b = domain_bounds(a.bounds_from, a.domain, a.pad) if a.bounds_from else tuple(float(x) for x in a.bbox)
    tiles = srtm_tiles(b)
    out = os.path.expanduser(a.out)
    total = 0
    sizes = []
    for nm, u in tiles:
        try:
            sz = _head_size(u)
        except Exception:  # noqa: BLE001
            sz = None
        sizes.append(sz)
        total += sz or 0
    print("bbox S=%.5f N=%.5f W=%.5f E=%.5f -> %d tiles %s; total %s%s -> %s"
          % (b + (len(tiles), ",".join(t[0] for t in tiles), _human(total),
                  " (some sizes unknown)" if None in sizes else "", out)))
    print("source: AWS Open Data 'Terrain Tiles' (skadi, SRTM-derived 1 arcsec, no API key): https://registry.opendata.aws/terrain-tiles/")
    if not _confirm("Download these tiles?", a.yes):
        return 2
    paths = []
    for nm, u in tiles:
        dest = os.path.join(out, nm + ".hgt.gz")
        sha, _ = _download(u, dest)
        if sha is None:
            print("no tile %s (ocean/outside coverage); skipped" % nm)
            continue
        paths.append(dest)
    print("\n".join(paths))
    print("next: wtune prep geog import-dem %s --name <id> [--fill-from <WPS_GEOG>/topo_gmted2010_30s]" % " ".join(paths))
    return 0


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(prog="wtune prep geog", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--geog-path", help="WPS_GEOG directory (default $WPS_GEOG, else %s)" % DEFAULT_GEOG)
        p.add_argument("--wps-dir", help="WPS directory holding geogrid/GEOGRID.TBL (default $WPS_DIR)")

    p = sub.add_parser("list", help="datasets present under WPS_GEOG vs. known to GEOGRID.TBL")
    common(p); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("fetch", help="download an official WPS geog dataset (size shown, --yes to proceed)")
    p.add_argument("dataset", help="GEOGRID.TBL key or directory: %s" % ", ".join(sorted(CATALOG)))
    common(p)
    p.add_argument("--dest", help="parent directory (default WPS_GEOG; ~/wps-geog-extra when not writable)")
    p.add_argument("--yes", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--keep-archive", action="store_true")
    p.add_argument("--link-existing", action="store_true", help="when falling back, symlink existing WPS_GEOG datasets into the new dir")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("import-dem", help="GeoTIFF / SRTM .hgt(.gz) -> WPS binary dataset + GEOGRID.TBL copy")
    p.add_argument("inputs", nargs="+", help="GeoTIFF(s) or .hgt(.gz) tiles on one lattice (mosaicked)")
    p.add_argument("--name", required=True, help="resolution key, [a-z0-9_]+")
    p.add_argument("--dest", help="parent dir of <name>/ (default ~/wps-geog-extra)")
    p.add_argument("--field", default="HGT_M")
    p.add_argument("--tbl-out", help="directory for the GEOGRID.TBL copy (default <dest>/<name>.geogrid)")
    p.add_argument("--tile", type=int, default=1200)
    p.add_argument("--description")
    p.add_argument("--fill-from", help="existing WPS dataset dir (e.g. <WPS_GEOG>/topo_gmted2010_30s) used for nodata inside the DEM")
    p.add_argument("--require-bounds", nargs=4, metavar=("S", "N", "W", "E"))
    p.add_argument("--require-bounds-from", metavar="PATH", help="geo_em/wrfinput/namelist.wps or a run dir; refuse a DEM that does not cover it")
    p.add_argument("--domain", type=int, default=2); p.add_argument("--pad", type=float, default=0.05)
    common(p)
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("verify", help="decode a WPS dataset and compare with its source raster")
    p.add_argument("dir"); p.add_argument("--source")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("bounds", help="domain lat/lon bounds and the SRTM tiles that cover them")
    p.add_argument("path", nargs="?", default="."); p.add_argument("--domain", type=int, default=2)
    p.add_argument("--pad", type=float, default=0.05); p.add_argument("--bbox", nargs=4, metavar=("S", "N", "W", "E"))
    p.set_defaults(fn=cmd_bounds)

    p = sub.add_parser("srtm-fetch", help="download SRTM 1-arcsec tiles (AWS Terrain Tiles, no key) for a domain/bbox")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bounds-from", metavar="PATH"); g.add_argument("--bbox", nargs=4, metavar=("S", "N", "W", "E"))
    p.add_argument("--domain", type=int, default=2); p.add_argument("--pad", type=float, default=0.05)
    p.add_argument("--out", default="~/wps-geog-extra/srtm"); p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_srtm_fetch)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "geog":
        argv = argv[1:]
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a) or 0
    except (ValueError, OSError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
