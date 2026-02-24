"""
Usage: python examples/diffusion_sft.py

Example script for training a diffusion language model with supervised fine-tuning
for next tactic prediction on a GitHub repository.
"""

from lean_dojo_v2.agent.diffusion_agent import DiffusionAgent
from lean_dojo_v2.trainer.diffusion_sft_trainer import DiffusionSFTTrainer

url = "https://github.com/durant42040/lean4-example"
commit = "b14fef0ceca29a65bc3122bf730406b33c7effe5"

trainer = DiffusionSFTTrainer(
    model_name="inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    output_dir="outputs-llada-sft",
    epochs_per_repo=1,
    batch_size=2,
    lr=2e-5,
)

agent = DiffusionAgent(trainer=trainer)
agent.setup_github_repository(url=url, commit=commit)
agent.train()
agent.prove()
