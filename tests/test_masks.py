# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Adam Faci, Léa Maronet

"""Mask representation: RLE round-trips, area, shape and IoU."""

import numpy as np
import pytest

from segmentart.backend import masks


@pytest.fixture
def rect():
    m = np.zeros((60, 80), dtype=bool)
    m[10:40, 20:60] = True
    return m


def test_rle_round_trip_is_lossless(rect):
    rle = masks.encode_rle(rect)
    assert masks.is_rle(rle)
    assert np.array_equal(masks.decode_rle(rle), rect)


def test_ensure_dense_accepts_both_representations(rect):
    assert np.array_equal(masks.ensure_dense(rect), rect)
    assert np.array_equal(masks.ensure_dense(masks.encode_rle(rect)), rect)


def test_rle_round_trip_on_edge_cases():
    for m in (np.zeros((8, 8), dtype=bool),          # empty
              np.ones((8, 8), dtype=bool),           # full
              np.eye(16, dtype=bool)):               # fragmented
        assert np.array_equal(masks.decode_rle(masks.encode_rle(m)), m)


def test_area_matches_dense_count(rect):
    assert masks.mask_area(rect) == int(rect.sum())
    assert masks.mask_area(masks.encode_rle(rect)) == int(rect.sum())


def test_shape_is_preserved_by_encoding(rect):
    assert masks.mask_shape(rect) == rect.shape
    assert masks.mask_shape(masks.encode_rle(rect)) == rect.shape


def test_iou_bounds(rect):
    assert masks.mask_iou(rect, rect) == pytest.approx(1.0)
    assert masks.mask_iou(rect, np.zeros_like(rect)) == pytest.approx(0.0)


def test_iou_is_symmetric_and_between_zero_and_one(rect):
    other = np.zeros_like(rect)
    other[25:55, 40:75] = True
    a, b = masks.mask_iou(rect, other), masks.mask_iou(other, rect)
    assert a == pytest.approx(b)
    assert 0.0 < a < 1.0


def test_to_compact_produces_an_rle(rect):
    assert masks.is_rle(masks.to_compact(rect))
