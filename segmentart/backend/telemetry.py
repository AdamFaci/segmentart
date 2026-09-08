# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet
"""Telemetry of actions and timing — no UI dependency.

Designed as a non-blocking OPTION: when `enabled` is False, every method is a
no-op. Events are accumulated in memory and can be dumped as JSONL (one line per
event).

The intended unit of analysis for the method study is the (image × pipeline)
pair. Every event therefore carries `image` and `pipeline`, which makes it
possible to aggregate time, interaction count and quality per condition.

Canonical event types (`event` field)
-------------------------------------
    category_start   {ann_id, label}            start of an annotation
    edit_start       {ann_id}                   resuming a validated annotation
    box_validate     {ann_id}                   box validated → points
    polygon          {ann_id, n_polygons, n_vertices}   P0 (manual) outline
    sam2_run         {ann_id, has_box, n_pos, n_neg, n_candidates, best_score, elapsed_s}
    opencv_fill      {ann_id, method}           morphological post-processing
    validate         {ann_id, label, has_box, has_mask, n_pos, n_neg}
    delete           {ann_id}                   deletion

`reduce_to_instances` turns this stream into one row per instance (image,
pipeline, instance, idle-gated active time, interaction count), ready to be
joined with the IoU and with `n_shapes` to produce the CSV consumed by the
analysis scripts.
"""

import json
import time

# Pipelines of the comparative study
PIPELINES = {
    "P0": "Manual (polygons)",
    "P1": "SAM2-assisted (box/points)",
    "P2": "SAM2 + OpenCV post-processing",
    "P3": "SAM2 + grouping (group-level annotation)",
}


class Telemetry:
    """Annotation event log, which can be switched on and off at runtime."""

    def __init__(self, enabled=False, path=None):
        self.enabled = enabled
        self.path = path
        self.events = []
        self._active = {}          # (image, pipeline) -> start t of the active segment

    # ── Life cycle of a timed segment ──────────────────────────────────────────
    def start_segment(self, image, pipeline):
        """Start (or restart) the timer of an (image, pipeline) segment."""
        if not self.enabled:
            return
        self._active[(image, pipeline)] = time.perf_counter()

    def end_segment(self, image, pipeline, **extra):
        """Close a segment and log its active duration."""
        if not self.enabled:
            return
        t0 = self._active.pop((image, pipeline), None)
        if t0 is None:
            return
        self.log("segment", image=image, pipeline=pipeline,
                 duration_s=round(time.perf_counter() - t0, 3), **extra)

    # ── Logging of individual events ───────────────────────────────────────────
    def log(self, event, image=None, pipeline=None, **fields):
        """Log an event (click, prompt, SAM call, correction, …)."""
        if not self.enabled:
            return
        rec = {"t": time.time(), "event": event}
        if image is not None:
            rec["image"] = image
        if pipeline is not None:
            rec["pipeline"] = pipeline
        rec.update(fields)
        self.events.append(rec)

    # ── Simple aggregates ──────────────────────────────────────────────────────
    def counts_by(self, key):
        """Count the events by the value of a field (e.g. 'event')."""
        out = {}
        for e in self.events:
            v = e.get(key)
            out[v] = out.get(v, 0) + 1
        return out

    def summary(self):
        """Summary per (image, pipeline): total duration and interaction count."""
        agg = {}
        for e in self.events:
            k = (e.get("image"), e.get("pipeline"))
            d = agg.setdefault(k, {"duration_s": 0.0, "events": 0})
            d["events"] += 1
            if e.get("event") == "segment":
                d["duration_s"] += e.get("duration_s", 0.0)
        return agg

    # ── Persistence ────────────────────────────────────────────────────────────
    def to_jsonl(self):
        return "\n".join(json.dumps(e, ensure_ascii=False) for e in self.events)

    def flush(self, path=None):
        """Write the events as JSONL. Returns the number of events written."""
        path = path or self.path
        if not path:
            return 0
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_jsonl())
        return len(self.events)

    def reset(self):
        self.events.clear()
        self._active.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Reduction of the event stream → one row per instance
# ═══════════════════════════════════════════════════════════════════════════════

def reduce_to_instances(events, idle_cap_s=120.0):
    """
    Aggregate an event stream (see the canonical types) into one row per instance.

    Active time: the sum of the intervals between consecutive events charged to
    the active instance; an interval longer than `idle_cap_s` is treated as
    inactivity and ignored (idle-gating).

    Interactions:
      - P0 / polygon: number of vertices;
      - assisted (P1–P3): box (0/1) + points + SAM calls + post-processing steps.

    Returns a list of dicts: image, pipeline, instance, label, time_s,
    n_interactions, n_sam, n_opencv, has_mask, order(None).
    """
    evs = sorted(events, key=lambda e: e.get("t", 0))
    inst = {}
    active = None
    last_t = None

    def acc(aid):
        return inst.setdefault(aid, dict(
            instance=aid, image=None, pipeline=None, label=None,
            time_s=0.0, n_sam=0, n_opencv=0, n_vertices=0, n_clicks=0,
            has_mask=False, has_box=False, n_pos=0, n_neg=0, order=None,
            _has_clicks=False))

    for e in evs:
        t = e.get("t")
        et = e.get("event")
        if active is not None and last_t is not None and t is not None:
            dt = t - last_t
            if 0 <= dt <= idle_cap_s:
                acc(active)["time_s"] += dt
        last_t = t

        aid = e.get("ann_id")
        if aid is None:
            continue
        a = acc(aid)
        if e.get("image") and a["image"] is None:
            a["image"] = e.get("image")
        if e.get("pipeline") and a["pipeline"] is None:
            a["pipeline"] = e.get("pipeline")

        if et in ("category_start", "edit_start"):
            if e.get("label"):
                a["label"] = e.get("label")
            active = aid
        elif et == "polygon":
            a["n_vertices"] = max(a["n_vertices"], int(e.get("n_vertices", 0)))
            active = aid
        elif et == "sam2_run":
            a["n_sam"] += 1
            active = aid
        elif et == "opencv_fill":
            a["n_opencv"] += 1
            active = aid
        elif et == "box_validate":
            a["has_box"] = True
            active = aid
        elif et == "validate":
            if e.get("label"):
                a["label"] = e.get("label")
            a["has_mask"] = bool(e.get("has_mask"))
            a["has_box"] = bool(e.get("has_box", a["has_box"]))
            a["n_pos"] = int(e.get("n_pos", a["n_pos"]))
            a["n_neg"] = int(e.get("n_neg", a["n_neg"]))
            if "n_clicks" in e:
                a["n_clicks"] = int(e.get("n_clicks", 0))
                a["_has_clicks"] = True
            active = None
        elif et == "delete":
            inst.pop(aid, None)
            if active == aid:
                active = None

    rows = []
    for a in inst.values():
        if a["_has_clicks"]:
            n_int = a["n_clicks"] + a["n_sam"] + a["n_opencv"]
        elif a["pipeline"] == "P0" or a["n_vertices"] > 0:
            n_int = a["n_vertices"]
        else:
            n_int = int(a["has_box"]) + a["n_pos"] + a["n_neg"] + a["n_sam"] + a["n_opencv"]
        rows.append({
            "image": a["image"], "pipeline": a["pipeline"], "instance": a["instance"],
            "label": a["label"], "time_s": round(a["time_s"], 3),
            "n_interactions": n_int, "n_clicks": a["n_clicks"],
            "n_sam": a["n_sam"], "n_opencv": a["n_opencv"],
            "has_mask": a["has_mask"], "order": a["order"],
        })
    return rows


def instances_to_csv(rows):
    """Serialize the per-instance rows as CSV (stable columns)."""
    import csv
    import io
    cols = ["image", "pipeline", "instance", "label", "time_s",
            "n_interactions", "n_clicks", "n_sam", "n_opencv", "has_mask", "order"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()
