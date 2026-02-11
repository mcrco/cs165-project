"""Diffusion-based prover for Lean theorem proving.

Uses a discrete masked-diffusion language model (LLaDA-style) for
next-tactic generation and whole-proof generation.
"""

import random
from typing import Optional

from loguru import logger
from pantograph.expr import GoalState, Tactic

from lean_dojo_v2.database.models.theorems import Theorem
from lean_dojo_v2.diffusion.config import DiffusionConfig
from lean_dojo_v2.diffusion.sampler import DiffusionSampler
from lean_dojo_v2.prover.base_prover import BaseProver


class DiffusionProver(BaseProver):
    """Prover that uses a discrete masked-diffusion LM for tactic generation.

    Mirrors the interface of HFProver but replaces autoregressive
    generation with the LLaDA masked-diffusion denoising loop.
    """

    def __init__(
        self,
        model_name: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        device: str = "auto",
        num_samples: int = 5,
        **sampler_kwargs,
    ):
        """Initialize the diffusion prover.

        Args:
            model_name: HuggingFace model identifier for the diffusion LM.
            device: Device to load the model on ('auto', 'cuda', 'cpu').
            num_samples: Number of tactic candidates to sample per call.
            **sampler_kwargs: Additional kwargs forwarded to DiffusionConfig
                (e.g. steps, temperature, block_length).
        """
        super().__init__()
        self.num_samples = num_samples

        config = DiffusionConfig(
            model_name=model_name,
            device=device,
            **sampler_kwargs,
        )
        self.sampler = DiffusionSampler(config)

    def next_tactic(
        self,
        state: GoalState,
        goal_id: int,
    ) -> Optional[Tactic]:
        """Generate the next tactic using the diffusion model.

        Builds the same prompt template as HFProver, samples multiple
        candidates via diffusion, post-processes, and returns one tactic.
        """
        if not hasattr(self, "theorem") or self.theorem is None:
            return None

        prompt = (
            "### System:\n"
            "You are a Lean 4 tactic generator. Given a goal state, "
            "output exactly ONE Lean tactic that advances or solves the goal.\n"
            "Rules:\n"
            "- Output only the tactic text; no prose, quotes, or code fences.\n"
            "- Single line only; no `by` blocks.\n"
            "- Never use `sorry` or `admit`.\n"
            "### User:\n"
            "{goal_str}\n\n"
            "### Assistant:\n"
        ).format(goal_str=str(state))

        raw_samples = self.sampler.sample_tactic(prompt, n=self.num_samples)

        tactics = _postprocess_tactics(raw_samples)

        if not tactics:
            logger.debug("Diffusion sampler produced no valid tactics")
            return None

        return random.choice(tactics)

    def generate_whole_proof(self, theorem: Theorem) -> str:
        """Generate a complete proof for the given theorem.

        Mirrors HFProver.generate_whole_proof: builds a theorem-based
        prompt and samples a long completion.
        """
        self.theorem = theorem

        prompt = (
            "### System:\n"
            "Given a theorem statement, "
            "output the complete proof of the theorem in Lean 4 code.\n"
            "Only output the proof, no explanation, no comments, no theorem, nothing else."
            "### User:\n"
            "{theorem_str}\n\n"
            "### Assistant:\n"
        ).format(theorem_str=str(self.theorem))

        raw_samples = self.sampler.sample_proof(prompt, n=1)

        proof = raw_samples[0].strip().replace("<;> ", "")
        return proof


def _postprocess_tactics(raw_samples: list[str]) -> list[str]:
    """Post-process raw diffusion outputs into valid single-line tactics.

    Applies the same filters as HFProver.next_tactic:
    - Strip whitespace
    - Take first line only (split on newline)
    - Split on '<;>' and take first segment
    - Skip empty strings and 'sorry'
    """
    tactics: list[str] = []
    for text in raw_samples:
        tactic = text.strip()
        tactic = tactic.split("\n")[0].split("<;>")[0].strip()
        if tactic and tactic != "sorry" and tactic != "admit":
            tactics.append(tactic)
    return tactics
