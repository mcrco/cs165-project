from typing import Optional

from lean_dojo_v2.lean_agent.config import ProverConfig, TrainingConfig
from lean_dojo_v2.prover import DiffusionProver
from lean_dojo_v2.trainer import DiffusionSFTTrainer

from .base_agent import BaseAgent


class DiffusionAgent(BaseAgent):
    def __init__(
        self,
        trainer: Optional[DiffusionSFTTrainer] = None,
        database_path: str = "dynamic_database.json",
        training_config: Optional[TrainingConfig] = None,
        prover_config: Optional[ProverConfig] = None,
    ):
        super().__init__(database_path)
        self.config = training_config or TrainingConfig()
        self.prover_config = prover_config or ProverConfig()
        self.trainer = trainer
        self.use_lora = False
        if self.trainer:
            self.output_dir = self.trainer.output_dir
            self.use_lora = self.trainer.lora_config is not None

    def _get_build_deps(self) -> bool:
        """DiffusionAgent doesn't build dependencies by default."""
        return False

    def _setup_prover(self):
        """Set up the DiffusionProver for DiffusionAgent."""
        if self.trainer:
            self.prover = DiffusionProver(
                ckpt_path=self.output_dir, use_lora=self.use_lora
            )
        else:
            # Use DiffusionProver's default base model when no fine-tuned ckpt is given.
            self.prover = DiffusionProver()


def main():
    """
    Main function to run DiffusionAgent.
    """
    url = "https://github.com/durant42040/lean4-example"
    commit = "b14fef0ceca29a65bc3122bf730406b33c7effe5"

    agent = DiffusionAgent()
    agent.setup_github_repository(url=url, commit=commit)
    agent.prove()


if __name__ == "__main__":
    main()
