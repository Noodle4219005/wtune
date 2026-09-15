#!/usr/bin/env python3
"""Rank every decomposition measured for one case across campaign ledgers, and emit a
confirmation CONFIG_FILE for the top-k.

Configs are grouped by what was actually run (ranks, omp, nproc_x/y, numtiles, distribution,
KMP_BLOCKTIME, tile geometry from the notes column), not by config id, so a template (C02), a D3
proposal (R01D0003) and its confirmation rerun (CFR01D0003) of the same decomposition pool together.

Usage:
  decomp_summary.py --case CASE --ledger L1 [L2 ...] [--top K --confirm-out cfg.json] [--tsv out.tsv]
"""
import argparse
import csv
import json
import statistics
import sys

TILE_KEYS = ("numtiles_x", "numtiles_y", "tile_sz_x", "tile_sz_y", "tile_strategy")


def _notes_fields(notes):
    out = {}
    for tok in (notes or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def row_identity(row):
    notes = _notes_fields(row.get("notes"))
    blocktime = str(row.get("kmp_blocktime") or "")
    blocktime = "0" if blocktime in ("0", "0.0") else "default"
    return (_int(row["n_ranks"]), _int(row["omp_threads"]), _int(row["nproc_x"]),
            _int(row["nproc_y"]), _int(row["numtiles"], 1), row.get("distribution") or "",
            blocktime) + tuple(_int(notes.get(k)) for k in TILE_KEYS)


def config_id(row):
    cid = _notes_fields(row.get("notes")).get("config") or ""
    return cid[2:] if cid.startswith("CF") else cid


def summarize(rows, case):
    groups = {}
    for r in rows:
        if r.get("case_base") != case or r.get("compliance_class") != "STRICT":
            continue
        try:
            s = float(r["s_per_step_median"])
        except (KeyError, TypeError, ValueError):
            continue
        g = groups.setdefault(row_identity(r), dict(ids=set(), s=[], e_we=_int(r.get("e_we")),
                                                    e_sn=_int(r.get("e_sn"))))
        g["ids"].add(config_id(r))
        g["s"].append(s)
    out = []
    for ident, g in groups.items():
        out.append(dict(identity=ident, ids=sorted(i for i in g["ids"] if i),
                        e_we=g["e_we"], e_sn=g["e_sn"],
                        n=len(g["s"]), s_median=statistics.median(g["s"]),
                        s_min=min(g["s"]), s_max=max(g["s"])))
    out.sort(key=lambda d: d["s_median"])
    return out


def confirm_configs(ranked, top):
    cfgs = []
    for d in ranked[:top]:
        ranks, omp, nx, ny, numtiles, dist, blocktime, tx, ty, sx, sy, strat = d["identity"]
        base = d["ids"][0] if d["ids"] else "X%d" % len(cfgs)
        # patch_x/y must be non-empty: strict_campaign.sbatch reads this list with IFS=$'\t' read,
        # which collapses empty fields and shifts every later column
        cfgs.append(dict(id="CF" + base, ranks=ranks, omp=omp, nx=nx, ny=ny, numtiles=numtiles,
                         patch_x=d["e_we"] // nx if nx else 0, patch_y=d["e_sn"] // ny if ny else 0,
                         numtiles_x=tx, numtiles_y=ty, tile_sz_x=sx, tile_sz_y=sy,
                         tile_strategy=strat, tile_shape="2d" if (tx and ty) else "1d_y",
                         dist=dist, blocktime=blocktime, config_source="generated"))
    return cfgs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--ledger", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--confirm-out")
    ap.add_argument("--tsv")
    a = ap.parse_args(argv)
    rows = []
    for p in a.ledger:
        with open(p) as f:
            rows.extend(csv.DictReader(f))
    ranked = summarize(rows, a.case)
    if not ranked:
        print("no successful STRICT rows for case %s" % a.case)
        return 1
    best = ranked[0]["s_median"]
    lines = ["rank\tids\tranks\tomp\tnx\tny\tnumtiles\tdist\tblocktime\ttiles\tn\ts_median\tvs_best_pct"]
    for i, d in enumerate(ranked, 1):
        r, o, nx, ny, nt, dist, bt = d["identity"][:7]
        lines.append("%d\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%d\t%.5f\t%+.2f" % (
            i, "+".join(d["ids"]), r, o, nx, ny, nt, dist, bt,
            ",".join(str(v) for v in d["identity"][7:]), d["n"], d["s_median"],
            100 * (d["s_median"] / best - 1)))
    print("\n".join(lines))
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write("\n".join(lines) + "\n")
    if a.confirm_out:
        with open(a.confirm_out, "w") as f:
            json.dump(dict(case_id=a.case, configs=confirm_configs(ranked, a.top)), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
