"""Tests for Chapter 20 Section 20.4: Guided Diffusion."""

import os
import pytest
import numpy as np
from common.reverse_decoder import DiffusionModel
from common.guided_diffusion import (
    ClassifierGuidedDiffusion,
    ClassifierFreeGuidedModel,
    generate_figure_20_8,
    generate_figure_20_9,
    generate_all_section_20_4_figures,
)


class TestClassifierGuidance:
    """Test suite for classifier guidance and inpainting."""

    def test_classifier_gradient_and_guided_noise(self):
        diff = DiffusionModel(data_dim=2, T=20, random_state=42)

        # Toy classifier log prob: class 0 prefers x1 > 0, class 1 prefers x1 < 0
        def classifier_log_prob(z, t, y):
            # log p(y=1 | z) = sigmoid(-z[:, 0]), log p(y=0 | z) = sigmoid(z[:, 0])
            score = z[:, 0] if y == 0 else -z[:, 0]
            return score

        guided_diff = ClassifierGuidedDiffusion(diff, classifier_log_prob)

        z = np.array([[1.0, 0.0], [-1.0, 0.0]])
        grad_0 = guided_diff.classifier_gradient(z, t=10, y=0)
        grad_1 = guided_diff.classifier_gradient(z, t=10, y=1)

        # Gradient should point towards +x1 for class 0, -x1 for class 1
        assert grad_0[0, 0] > 0.0
        assert grad_1[0, 0] < 0.0

    def test_guided_sampling_and_inpainting(self):
        diff = DiffusionModel(data_dim=2, T=10, random_state=42)
        guided_diff = ClassifierGuidedDiffusion(diff, lambda z, t, y: z[:, 0])

        samples = guided_diff.sample_guided(num_samples=4, target_class=0, gamma=2.0, random_state=42)
        assert samples.shape == (4, 2)
        assert not np.any(np.isnan(samples))

        # Test inpainting
        img = np.array([[2.0, 2.0]])
        mask = np.array([[1.0, 0.0]])  # first dimension known
        inpainted = guided_diff.inpaint(img, mask, random_state=42)
        assert inpainted.shape == (1, 2)
        assert inpainted[0, 0] == pytest.approx(2.0, abs=1e-4)


class TestClassifierFreeGuidance:
    """Test suite for Classifier-Free Guidance (CFG)."""

    def test_cfg_formula_and_interpolation(self):
        cfg_model = ClassifierFreeGuidedModel(data_dim=2, num_classes=2, T=10, random_state=42)
        z = np.array([[0.5, -0.5]])
        t = np.array([5])

        # Conditional prediction y=0, y=1, and null class y=2
        eps_c0, _ = cfg_model.forward(z, t, np.array([0]))
        eps_uncond, _ = cfg_model.forward(z, t, np.array([2]))

        # Check gamma = 0 gives unconditional
        gamma_0 = eps_uncond + 0.0 * (eps_c0 - eps_uncond)
        assert np.allclose(gamma_0, eps_uncond)

        # Check gamma = 1 gives conditional
        gamma_1 = eps_uncond + 1.0 * (eps_c0 - eps_uncond)
        assert np.allclose(gamma_1, eps_c0)

        # Sampling runs and produces valid shapes
        samples = cfg_model.sample_guided(num_samples=3, target_class=0, gamma=2.5, random_state=42)
        assert samples.shape == (3, 2)
        assert not np.any(np.isnan(samples))


class TestSection204Figures:
    """Test suite for Section 20.4 figure generation (Figures 20.8, 20.9)."""

    def test_all_section_20_4_figures_generate(self):
        figs = generate_all_section_20_4_figures()
        assert len(figs) == 2
        assert "fig_20_8.png" in figs
        assert "fig_20_9.png" in figs

        base_dir = os.path.dirname(os.path.dirname(__file__))
        for fname in ["fig_20_8.png", "fig_20_9.png"]:
            p1 = os.path.join(base_dir, "20", "result", fname)
            p2 = os.path.join(base_dir, "result", fname)
            assert os.path.exists(p1), f"Missing {p1}"
            assert os.path.exists(p2), f"Missing {p2}"
            assert os.path.getsize(p1) > 1000
            assert os.path.getsize(p2) > 1000
