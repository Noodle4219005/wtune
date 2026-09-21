"""Reproduce the compact inventory from the preserved read-only audit."""
import csv
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
fields = "role path size_bytes sha256 archive_member archive_sha256 archive_match cps_reference_count nm_returncode".split()
rows = []
for item in data["member_comparisons"]:
    rows.append(dict(role="audited_object", **item["loose"],
                     archive_member=item["archive_member"],
                     archive_sha256=item["archive_member_sha256"],
                     archive_match=item["match"]))
for item in data["source_and_module_inputs"]:
    rows.append(dict(role="source_or_module", **item))
for item in data["explicit_link_inputs"]:
    rows.append(dict(role="explicit_link_input", path=item["path"],
                     size_bytes=item["size_bytes"], sha256=item["sha256"],
                     cps_reference_count=len(item["cps_references"]),
                     nm_returncode=item["nm_returncode"]))
with open(sys.argv[2], "x", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
