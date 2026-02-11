"""Tests for the diffusion sampler and diffusion prover post-processing.

These tests verify the Phase 0 deliverables without requiring a GPU or
the actual LLaDA model weights — they use mocks for the sampler.
"""

from unittest.mock import MagicMock, patch

import torch

from lean_dojo_v2.diffusion.config import DiffusionConfig
from lean_dojo_v2.prover.diffusion_prover import _postprocess_tactics


class TestDiffusionConfig:
    """Test DiffusionConfig dataclass construction."""

    def test_default_config(self):
        cfg = DiffusionConfig()
        assert cfg.model_name == "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
        assert cfg.mask_id == 156895
        assert cfg.steps == 128
        assert cfg.gen_length == 128
        assert cfg.block_length == 32
        assert cfg.temperature == 0.0
        assert cfg.cfg_scale == 0.0
        assert cfg.remasking == "low_confidence"
        assert cfg.dtype == torch.bfloat16
        assert cfg.max_tactic_tokens == 64
        assert cfg.max_proof_tokens == 512

    def test_custom_config(self):
        cfg = DiffusionConfig(
            model_name="custom/model",
            steps=64,
            temperature=0.5,
            device="cpu",
        )
        assert cfg.model_name == "custom/model"
        assert cfg.steps == 64
        assert cfg.temperature == 0.5
        assert cfg.device == "cpu"
        # Other fields keep defaults
        assert cfg.mask_id == 156895


class TestPostprocessTactics:
    """Test the tactic post-processing logic shared with HFProver."""

    def test_single_clean_tactic(self):
        assert _postprocess_tactics(["exact h"]) == ["exact h"]

    def test_strips_whitespace(self):
        assert _postprocess_tactics(["  exact h  "]) == ["exact h"]

    def test_takes_first_line(self):
        result = _postprocess_tactics(["exact h\napply f\nsorry"])
        assert result == ["exact h"]

    def test_splits_on_semicolon_marker(self):
        result = _postprocess_tactics(["exact h<;>apply f"])
        assert result == ["exact h"]

    def test_skips_sorry(self):
        result = _postprocess_tactics(["sorry"])
        assert result == []

    def test_skips_admit(self):
        result = _postprocess_tactics(["admit"])
        assert result == []

    def test_skips_empty(self):
        result = _postprocess_tactics([""])
        assert result == []
        result = _postprocess_tactics(["   "])
        assert result == []

    def test_multiple_samples_mixed(self):
        raw = [
            "exact h",
            "sorry",
            "  apply f\nextra stuff  ",
            "",
            "simp<;>done",
            "admit",
        ]
        result = _postprocess_tactics(raw)
        assert result == ["exact h", "apply f", "simp"]

    def test_newline_then_sorry(self):
        """First line is valid even if later lines contain sorry."""
        result = _postprocess_tactics(["rfl\nsorry"])
        assert result == ["rfl"]


class TestDiffusionProverWithMock:
    """Test DiffusionProver using a mocked DiffusionSampler."""

    def test_next_tactic_returns_string(self):
        """next_tactic should return a valid tactic string when sampler produces output."""
        from lean_dojo_v2.prover.diffusion_prover import DiffusionProver

        with patch.object(DiffusionProver, "__init__", lambda self, **kw: None):
            prover = DiffusionProver()
            prover.theorem = MagicMock()
            prover.num_samples = 3
            prover.sampler = MagicMock()
            prover.sampler.sample_tactic.return_value = [
                "exact h",
                "sorry",
                "apply f",
            ]

            state = MagicMock()
            state.__str__ = lambda self: "a : Nat\n⊢ a = a"

            tactic = prover.next_tactic(state, goal_id=0)
            assert tactic in ["exact h", "apply f"]

    def test_next_tactic_returns_none_when_no_theorem(self):
        """next_tactic should return None when theorem is not set."""
        from lean_dojo_v2.prover.diffusion_prover import DiffusionProver

        with patch.object(DiffusionProver, "__init__", lambda self, **kw: None):
            prover = DiffusionProver()
            prover.theorem = None

            state = MagicMock()
            result = prover.next_tactic(state, goal_id=0)
            assert result is None

    def test_next_tactic_returns_none_when_all_sorry(self):
        """next_tactic should return None when all samples are sorry/empty."""
        from lean_dojo_v2.prover.diffusion_prover import DiffusionProver

        with patch.object(DiffusionProver, "__init__", lambda self, **kw: None):
            prover = DiffusionProver()
            prover.theorem = MagicMock()
            prover.num_samples = 2
            prover.sampler = MagicMock()
            prover.sampler.sample_tactic.return_value = ["sorry", ""]

            state = MagicMock()
            state.__str__ = lambda self: "⊢ True"

            tactic = prover.next_tactic(state, goal_id=0)
            assert tactic is None
