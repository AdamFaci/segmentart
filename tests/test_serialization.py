# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Export and import: schema keys, the three standards, and the native round-trip."""

import json

import numpy as np
import pytest

from segmentart.backend import masks, serialization


@pytest.fixture
def image():
    return {"name": "sample.jpg", "array": np.full((60, 80, 3), 200, dtype=np.uint8)}


@pytest.fixture
def annotation():
    m = np.zeros((60, 80), dtype=bool)
    m[10:40, 20:60] = True
    return {
        "id": 1,
        "label": "lion",
        "box": (20, 10, 40, 30),
        "mask": masks.encode_rle(m),
        "attributes": ["red"],
        "group": 0,
        "region_desc": "a lion facing left",
        "points_pos": [(30, 20)],
        "points_neg": [],
    }


def _as_dict(payload):
    return json.loads(payload) if isinstance(payload, str) else payload


def test_native_round_trip_preserves_label_box_and_mask(image, annotation):
    exported = _as_dict(serialization.to_json([image], [[annotation]]))
    target = [[]]
    added, skipped = serialization.import_annotations(
        exported, [image], target,
        lambda label: {"id": 99, "label": label, "box": None, "mask": None,
                       "attributes": []},
    )
    assert (added, skipped) == (1, 0)

    restored = target[0][0]
    assert restored["label"] == "lion"
    assert tuple(restored["box"]) == (20, 10, 40, 30)
    assert np.array_equal(masks.ensure_dense(restored["mask"]),
                          masks.ensure_dense(annotation["mask"]))


def test_import_skips_annotations_for_unknown_images(image, annotation):
    exported = _as_dict(serialization.to_json([image], [[annotation]]))
    other = {"name": "different.jpg", "array": image["array"]}
    added, skipped = serialization.import_annotations(
        exported, [other], [[]], lambda label: {"id": 1, "label": label},
    )
    assert (added, skipped) == (0, 1)


@pytest.mark.parametrize(
    "standard, expected_keys",
    [
        ("VisualGenome", {"images", "relationships", "groups"}),
        ("COCO", {"images", "annotations", "categories"}),
        ("Custom", {"images", "relationships", "groups"}),
    ],
)
def test_export_standards_have_their_top_level_keys(image, annotation, standard,
                                                    expected_keys):
    out = _as_dict(serialization.export_standard(
        standard, [image], [[annotation]], ["lion"], ["material"], [], {}))
    assert set(out) == expected_keys


def test_visual_genome_object_carries_the_documented_fields(image, annotation):
    out = _as_dict(serialization.export_standard(
        "VisualGenome", [image], [[annotation]], ["lion"], ["material"], [], {}))
    obj = out["images"][0]["objects"][0]
    assert {"object_id", "name", "box", "attributes", "group",
            "mask_area", "segmentation"} <= set(obj)
    assert obj["name"] == "lion"


def test_custom_standard_adds_the_user_fields(image, annotation):
    out = _as_dict(serialization.export_standard(
        "Custom", [image], [[annotation]], ["lion"], ["material"], [], {}))
    assert "fields" in out["images"][0]["objects"][0]


def test_export_contains_no_non_english_keys(image, annotation):
    """The published schema is English; guard against a French key creeping back."""
    for standard in ("VisualGenome", "COCO", "Custom"):
        blob = json.dumps(_as_dict(serialization.export_standard(
            standard, [image], [[annotation]], ["lion"], ["material"], [], {})),
            default=str)
        for stale in ("aire_px", "couleur", "taille", "forme", "masque"):
            assert stale not in blob, f"{stale!r} found in the {standard} export"
