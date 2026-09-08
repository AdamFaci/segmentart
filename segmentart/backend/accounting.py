# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Group-level accounting (pipeline P3) — no UI dependency.

Quantifies the gain of "one description per group rather than one per object":
how many group-level description units cover how many objects.
"""


def group_accounting(objects, group_meta=None):
    """
    `objects`: a list of dicts {image, object_id, label, group, attributes,
    region_desc}. `group_meta`: {group_id: {"name", "description"}}.

    Returns (summary, per_group_table):
      - n_objects, n_groups, n_groups_described
      - descriptions_per_object = G / N  (lower = more leverage)
      - objects_per_group_mean = N / G
      - reduction_vs_per_object_pct = 100·(1 − G/N)  (description units avoided)
      - counts of the descriptions still made at object level (attributes,
        regions)
    """
    group_meta = group_meta or {}
    grouped = [o for o in objects if o.get("group") is not None]
    n = len(grouped)

    groups = {}
    for o in grouped:
        groups.setdefault(o["group"], []).append(o)
    g = len(groups)

    rows = []
    n_described = 0
    n_obj_attr = 0
    n_obj_region = 0
    for gid, members in sorted(groups.items(), key=lambda kv: kv[0]):
        meta = group_meta.get(gid, {})
        named = bool(meta.get("name"))
        described = bool(meta.get("description") or meta.get("name"))
        n_described += int(described)
        imgs = {m.get("image") for m in members}
        obj_attr = sum(1 for m in members if m.get("attributes"))
        obj_region = sum(1 for m in members if m.get("region_desc"))
        n_obj_attr += obj_attr
        n_obj_region += obj_region
        rows.append({
            "group_id": gid, "n_members": len(members), "n_images": len(imgs),
            "named": named, "described": described,
            "labels": ",".join(sorted({m.get("label", "") for m in members})),
            "members_with_attributes": obj_attr,
            "members_with_region_desc": obj_region,
        })

    summary = {
        "n_objects": n,
        "n_groups": g,
        "n_groups_described": n_described,
        "descriptions_per_object": round(g / n, 4) if n else None,
        "objects_per_group_mean": round(n / g, 3) if g else None,
        "reduction_vs_per_object_pct": round(100 * (1 - g / n), 1) if n else None,
        "individual_region_descriptions": n_obj_region,
        "individual_attribute_sets": n_obj_attr,
    }
    return summary, rows
