"""Discrete masked-diffusion sampler for LLaDA-style models.

Implements the iterative denoising loop from the LLaDA model card:
corrupt text → mask tokens, then iteratively denoise via Gumbel-noise
sampling with confidence-based or random remasking.
"""

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from .config import DiffusionConfig


class DiffusionSampler:
    """Sampler for discrete masked-diffusion language models.

    Exposes the D0.3 interface:
        - sample_tactic(prompt, n, **kwargs) -> list[str]
        - sample_proof(prompt, n, **kwargs) -> list[str]

    Internally wraps the LLaDA denoising generate loop.
    """

    def __init__(self, config: DiffusionConfig):
        self.config = config

        # Resolve device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Loading diffusion model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=config.dtype,
        ).to(self.device)
        self.model.eval()
        logger.info(
            f"Diffusion model loaded on {self.device} (dtype={config.dtype})"
        )

    def sample_tactic(
        self, prompt: str, n: int = 1, **kwargs
    ) -> list[str]:
        """Sample `n` tactic strings from the diffusion model.

        Args:
            prompt: The formatted prompt string (system + goal state).
            n: Number of independent samples to generate.
            **kwargs: Override any DiffusionConfig field for this call
                (e.g. temperature=0.5, steps=64).

        Returns:
            List of `n` decoded tactic strings (raw, before post-processing).
        """
        gen_length = kwargs.pop("gen_length", self.config.max_tactic_tokens)
        return self._sample(prompt, n, gen_length=gen_length, **kwargs)

    def sample_proof(
        self, prompt: str, n: int = 1, **kwargs
    ) -> list[str]:
        """Sample `n` whole-proof strings from the diffusion model.

        Args:
            prompt: The formatted prompt string (system + theorem statement).
            n: Number of independent samples to generate.
            **kwargs: Override any DiffusionConfig field for this call.

        Returns:
            List of `n` decoded proof strings (raw, before post-processing).
        """
        gen_length = kwargs.pop("gen_length", self.config.max_proof_tokens)
        return self._sample(prompt, n, gen_length=gen_length, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample(
        self, prompt: str, n: int, gen_length: int, **kwargs
    ) -> list[str]:
        """Run the denoising loop `n` times and return decoded strings."""
        input_ids = self.tokenizer(prompt)["input_ids"]
        prompt_tensor = (
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        )

        results: list[str] = []
        for _ in range(n):
            output_ids = self._generate(
                prompt_tensor,
                gen_length=gen_length,
                steps=kwargs.get("steps", self.config.steps),
                block_length=kwargs.get("block_length", self.config.block_length),
                temperature=kwargs.get("temperature", self.config.temperature),
                cfg_scale=kwargs.get("cfg_scale", self.config.cfg_scale),
                remasking=kwargs.get("remasking", self.config.remasking),
            )
            # Decode only the generated part (after the prompt)
            gen_tokens = output_ids[:, prompt_tensor.shape[1] :]
            text = self.tokenizer.batch_decode(
                gen_tokens, skip_special_tokens=True
            )[0]
            results.append(text)

        return results

    @torch.no_grad()
    def _generate(
        self,
        prompt: torch.Tensor,
        gen_length: int = 128,
        steps: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
    ) -> torch.Tensor:
        """LLaDA-style iterative masked-diffusion decoding.

        Adapted from the official HuggingFace model card for
        inclusionAI/LLaDA-MoE-7B-A1B-Instruct.

        Args:
            prompt: Tokenized prompt tensor of shape (1, prompt_len).
            gen_length: Number of new tokens to generate.
            steps: Total denoising steps across all blocks.
            block_length: Block size for semi-autoregressive generation.
            temperature: Gumbel noise temperature.
            cfg_scale: Classifier-free guidance scale.
            remasking: 'low_confidence' or 'random'.

        Returns:
            Full token tensor of shape (1, prompt_len + gen_length).
        """
        mask_id = self.config.mask_id

        # Ensure gen_length is divisible by block_length
        if gen_length % block_length != 0:
            # Round up to nearest multiple
            gen_length = ((gen_length // block_length) + 1) * block_length

        # Initialize output: prompt tokens + masked generation region
        x = torch.full(
            (1, prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()
        prompt_index = x != mask_id

        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            steps = num_blocks * (steps // num_blocks + 1)
        steps_per_block = steps // num_blocks

        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, block_start:block_end] == mask_id
            num_transfer_tokens = self._get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = self._add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # (1, seq_len)

                if remasking == "low_confidence":
                    p = F.softmax(logits.float(), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")

                # Only consider tokens in current and past blocks
                x0_p[:, block_end:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=int(num_transfer_tokens[j, i].item())
                    )
                    transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]

        return x

    @staticmethod
    def _add_gumbel_noise(
        logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Add Gumbel noise to logits for stochastic sampling."""
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def _get_num_transfer_tokens(
        mask_index: torch.Tensor, steps: int
    ) -> torch.Tensor:
        """Compute how many tokens to unmask at each step.

        Distributes masked tokens evenly across steps, with remainder
        going to the first few steps.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = (
            torch.zeros(
                mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
            )
            + base
        )
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1
        return num_transfer_tokens
