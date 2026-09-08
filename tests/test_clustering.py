# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Rule-based attributes: keys, vocabulary and group characterisation."""

import numpy as np
import pytest

from segmentart.backend import clustering, masks

ATTRIBUTE_KEYS = {"color", "size", "position", "shape"}


def _annotation(box, mask=None):
    return {"id": 1, "label": "motif", "box": box, "mask": mask, "attributes": []}


@pytest.fixture
def red_image():
    arr = np.full((300, 300, 3), 220, dtype=np.uint8)
    arr[100:200, 100:200] = (200, 40, 40)
    return arr


def test_attribute_keys_are_the_documented_ones(red_image):
    attrs = clustering.rule_attributes(_annotation((100, 100, 100, 100)), red_image)
    assert set(attrs) <= ATTRIBUTE_KEYS


def test_dominant_color_is_detected(red_image):
    attrs = clustering.rule_attributes(_annotation((100, 100, 100, 100)), red_image)
    assert attrs["color"] == "red"


def test_size_buckets(red_image):
    small = clustering.rule_attributes(_annotation((10, 10, 20, 20)), red_image)
    large = clustering.rule_attributes(_annotation((0, 0, 300, 300)), red_image)
    assert small["size"] == "small"
    assert large["size"] == "large"


@pytest.mark.parametrize(
    "box, expected",
    [
        ((130, 10, 40, 40), "top"),
        ((130, 250, 40, 40), "bottom"),
        ((130, 130, 40, 40), "center"),
    ],
)
def test_vertical_position(red_image, box, expected):
    """`position` is the vertical band, or "vertical-horizontal" when they differ."""
    position = clustering.rule_attributes(_annotation(box), red_image)["position"]
    assert position.split("-")[0] == expected


@pytest.mark.parametrize(
    "box, expected",
    [
        ((10, 130, 40, 40), "left"),
        ((250, 130, 40, 40), "right"),
    ],
)
def test_horizontal_position(red_image, box, expected):
    position = clustering.rule_attributes(_annotation(box), red_image)["position"]
    assert position.split("-")[-1] == expected


def test_position_collapses_when_both_bands_agree(red_image):
    """A centred box reads "center", not "center-center"."""
    assert clustering.rule_attributes(
        _annotation((130, 130, 40, 40)), red_image)["position"] == "center"


@pytest.mark.parametrize(
    "box, expected",
    [
        ((100, 140, 120, 40), "wide"),
        ((140, 100, 40, 120), "tall"),
        ((130, 130, 50, 50), "square"),
    ],
)
def test_shape_from_aspect_ratio(red_image, box, expected):
    assert clustering.rule_attributes(_annotation(box), red_image)["shape"] == expected


def test_attribute_vocabulary_is_english(red_image):
    attrs = clustering.rule_attributes(_annotation((100, 100, 100, 100)), red_image)
    stale = {"rouge", "vert", "bleu", "petit", "moyen", "grand", "carre", "carré",
             "haut", "bas", "gauche", "droite", "centre", "large"}
    assert not (set(attrs.values()) & stale)


def test_mask_restricts_the_color_sample(red_image):
    """With a mask covering only the red square, the colour must still read red."""
    m = np.zeros((300, 300), dtype=bool)
    m[100:200, 100:200] = True
    ann = _annotation((50, 50, 200, 200), masks.encode_rle(m))
    assert clustering.rule_attributes(ann, red_image)["color"] == "red"


def test_group_characterization_splits_shared_and_differing(red_image):
    a = _annotation((100, 100, 100, 100))
    b = _annotation((110, 105, 100, 100))
    shared, diffs = clustering.group_characterization([(a, red_image), (b, red_image)])
    assert isinstance(shared, dict) and isinstance(diffs, dict)
    assert set(shared) | set(diffs) <= ATTRIBUTE_KEYS
    assert not (set(shared) & set(diffs))


def test_group_characterization_on_a_single_member(red_image):
    a = _annotation((100, 100, 100, 100))
    shared, diffs = clustering.group_characterization([(a, red_image)])
    assert diffs == {}
    assert set(shared) <= ATTRIBUTE_KEYS
