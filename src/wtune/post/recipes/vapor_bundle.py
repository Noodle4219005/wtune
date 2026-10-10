"""post/recipes/vapor_bundle.py -- the always-on VAPOR bundle data product of `wtune post recipe`
(post/vapor_bundle.py): one ready-to-copy <out>/vapor/d0N/ per domain, listed in the report's
"common" section as a data-file entry (bundle dir, file count, total size, hard-linked vs
copied, links to README.md and vapor_render.py), not a figure.

Same domains (d01 + the finest nest, post/recipes/cf3d.cf_domains) and the same times (the
--every/--times selection, else the last frame) as the CF pressure-level product, so the two
3-D outputs of one report always describe the same frames. Building the bundle needs nothing
beyond the stdlib -- VAPOR itself runs later on the user's workstation -- so the failures here
are a domain without frames or a filesystem error, each a missing entry. This path NEVER copies
(allow_copy stays False): when the frames cannot be hard-linked (the recipe output on another
filesystem) the missing entry carries the exact `wtune post vapor-bundle ... --allow-copy`
command and the size it would copy -- an always-on product must not silently duplicate wrfout
into someone's quota.
"""
from __future__ import annotations

CATALOG_ID = "common_vapor_bundle"


def figures(run_dir, run_label, out_dir, case_info, every=None, times=None):
    from wtune.post import vapor_bundle
    from wtune.post.recipe import fig, missing
    from wtune.post.recipes.cf3d import cf_domains

    out = []
    for r in vapor_bundle.export(run_dir, out_dir, cf_domains(case_info), every=every,
                                 times=times, run_label=run_label):
        if r.get("error"):
            out.append(missing(CATALOG_ID, "wrfout d%02d" % r["domain"], r["error"],
                               run=run_label))
            continue
        out.append(fig(CATALOG_ID, r["dir"], run_label, domain="d%02d" % r["domain"],
                       valid_time=", ".join(f["valid_time"] for f in r["frames"]),
                       extra=dict(data_file=True, size_bytes=r["total_bytes"],
                                  links=[r["readme"], r["script"]],
                                  note="%d wrfout file(s), %s" % (r["n_files"],
                                                                  r["link_note"]))))
    return out
