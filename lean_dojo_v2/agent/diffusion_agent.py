"""Diffusion-based agent for Lean theorem proving.

Mirrors ExternalAgent but uses DiffusionProver (local LLaDA-style
masked-diffusion model) instead of an external API prover.
"""

from lean_dojo_v2.agent.base_agent import BaseAgent
from lean_dojo_v2.prover.diffusion_prover import DiffusionProver


class DiffusionAgent(BaseAgent):
    """Agent that uses a diffusion language model for theorem proving.

    Drop-in replacement for ExternalAgent / HFAgent that uses
    DiffusionProver under the hood.
    """

    def __init__(
        self,
        model_name: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        device: str = "auto",
        num_samples: int = 5,
        **sampler_kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.num_samples = num_samples
        self.sampler_kwargs = sampler_kwargs

    def _get_build_deps(self) -> bool:
        """DiffusionAgent doesn't build dependencies by default."""
        return False

    def _setup_prover(self):
        """Set up the DiffusionProver."""
        self.prover = DiffusionProver(
            model_name=self.model_name,
            device=self.device,
            num_samples=self.num_samples,
            **self.sampler_kwargs,
        )
