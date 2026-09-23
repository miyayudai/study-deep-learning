"""
tests/test_ch10_object_detectio.py
==================================
Unit tests for Section 10.4 Object Detection
Bishop & Bishop (2024), Deep Learning: Foundations and Concepts.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from common.object_detection import (
    BoundingBox,
    NonMaxSuppression,
    analyze_sliding_window_efficiency,
    generate_figure_10_19,
    generate_figure_10_20,
    generate_figure_10_21,
    generate_figure_10_22,
    generate_figure_10_23,
    generate_figure_10_24,
    generate_figure_10_25,
)


class TestBoundingBoxAndIoU:
    def test_coordinate_conversions(self):
        corners = np.array([10.0, 20.0, 30.0, 50.0])  # x1, y1, x2, y2
        center = BoundingBox.corners_to_center(corners)
        np.testing.assert_allclose(center, [20.0, 35.0, 20.0, 30.0])  # xc, yc, w, h
        recovered = BoundingBox.center_to_corners(center)
        np.testing.assert_allclose(recovered, corners)

    def test_iou_identical_boxes(self):
        box = np.array([0.0, 0.0, 10.0, 10.0])
        assert BoundingBox.compute_iou(box, box) == pytest.approx(1.0)

    def test_iou_disjoint_boxes(self):
        b1 = np.array([0.0, 0.0, 2.0, 2.0])
        b2 = np.array([5.0, 5.0, 7.0, 7.0])
        assert BoundingBox.compute_iou(b1, b2) == 0.0

    def test_iou_analytical_overlap(self):
        # b1: [0, 0, 2, 2] -> Area = 4
        # b2: [1, 0, 3, 2] -> Area = 4
        # Intersection: [1, 0, 2, 2] -> Area = 2
        # Union: 4 + 4 - 2 = 6
        # IoU = 2 / 6 = 1/3
        b1 = np.array([0.0, 0.0, 2.0, 2.0])
        b2 = np.array([1.0, 0.0, 3.0, 2.0])
        assert BoundingBox.compute_iou(b1, b2) == pytest.approx(1.0 / 3.0)

    def test_iou_nested_boxes(self):
        # b1: [0, 0, 4, 4] -> Area = 16
        # b2: [1, 1, 3, 3] -> Area = 4
        # Intersection: 4, Union: 16 -> IoU = 4/16 = 0.25
        b1 = np.array([0.0, 0.0, 4.0, 4.0])
        b2 = np.array([1.0, 1.0, 3.0, 3.0])
        assert BoundingBox.compute_iou(b1, b2) == pytest.approx(0.25)


class TestNonMaxSuppression:
    def test_nms_figure_10_25_scenario(self):
        # Scenario from Figure 10.25:
        # Box 0: Left winner [1.5, 2.0, 4.7, 5.8], score 0.95
        # Box 1: Left overlap [1.1, 1.8, 4.5, 5.4], score 0.81 (IoU > 0.5 with Box 0)
        # Box 2: Left overlap [1.9, 2.4, 4.9, 5.9], score 0.75 (IoU > 0.5 with Box 0)
        # Box 3: Right winner [6.2, 2.2, 9.2, 5.8], score 0.91 (Disjoint from Box 0)
        # Box 4: Weak false detection, score 0.40 (Below 0.7 threshold)
        boxes = np.array([
            [1.5, 2.0, 4.7, 5.8],
            [1.1, 1.8, 4.5, 5.4],
            [1.9, 2.4, 4.9, 5.9],
            [6.2, 2.2, 9.2, 5.8],
            [7.0, 1.0, 8.5, 2.5],
        ])
        scores = np.array([0.95, 0.81, 0.75, 0.91, 0.40])

        kept = NonMaxSuppression.apply(boxes, scores, score_threshold=0.7, iou_threshold=0.5)

        # Expected: Box 0 (0.95) and Box 3 (0.91) are kept. Box 1 and 2 suppressed by IoU. Box 4 suppressed by score.
        assert kept == [0, 3]

    def test_nms_all_low_scores(self):
        boxes = np.array([[0, 0, 1, 1]])
        scores = np.array([0.2])
        kept = NonMaxSuppression.apply(boxes, scores, score_threshold=0.7)
        assert kept == []


class TestSlidingWindowEfficiency:
    def test_exercise_10_12_accounting(self):
        stats = analyze_sliding_window_efficiency()
        # Single 6x6 pass
        assert stats["single_pass_total"] == 148
        # 9 windows naive pass
        assert stats["naive_total_multiplications"] == 1332
        # Expanded conv 8x8 pass
        assert stats["conv_expanded_total_multiplications"] == 340
        # Speedup ratio ~3.92x
        assert stats["speedup_ratio"] == pytest.approx(1332.0 / 340.0)


class TestFigureGenerators:
    def test_generate_all_section_10_4_figures(self, tmp_path):
        figs = [
            generate_figure_10_19(save_dir=str(tmp_path)),
            generate_figure_10_20(save_dir=str(tmp_path)),
            generate_figure_10_21(save_dir=str(tmp_path)),
            generate_figure_10_22(save_dir=str(tmp_path)),
            generate_figure_10_23(save_dir=str(tmp_path)),
            generate_figure_10_24(save_dir=str(tmp_path)),
            generate_figure_10_25(save_dir=str(tmp_path)),
        ]
        for idx, fig in enumerate(figs, start=19):
            assert isinstance(fig, plt.Figure)
            assert (tmp_path / f"Figure_10_{idx}.png").exists()
            plt.close(fig)
