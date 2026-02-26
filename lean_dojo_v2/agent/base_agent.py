import os
import subprocess
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
import sys

from loguru import logger
from pantograph import Server

from lean_dojo_v2.database import DynamicDatabase
from lean_dojo_v2.database.models import Repository
from lean_dojo_v2.lean_dojo.data_extraction.lean import LeanGitRepo
from lean_dojo_v2.lean_dojo.data_extraction.trace import get_traced_repo_path
from lean_dojo_v2.utils.constants import DATA_DIR, RAID_DIR


class _TeeWriter:
    """Write stream output to multiple sinks."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class BaseAgent(ABC):
    """Abstract base class for theorem proving agents.

    This class defines the common interface and functionality for different types
    of theorem proving agents, such as HFAgent and LeanAgent.
    """

    def __init__(self, database_path: str = "dynamic_database.json"):
        self.database = DynamicDatabase(json_path=database_path)
        self.data_path = Path(os.path.join(RAID_DIR, DATA_DIR, "merged"))
        self.repos = []

    @abstractmethod
    def _get_build_deps(self) -> bool:
        """Get whether to build dependencies. Must be implemented by subclasses."""
        pass

    def trace_repository(self, url: str, commit: str, build_deps: bool) -> Repository:
        """Trace a repository and return a Repository object."""
        return self.database.trace_repository(url, commit, build_deps)

    def add_repository(self, repo: Repository):
        """Add a repository to the database."""
        self.database.add_repository(repo)
        self.repos.append(LeanGitRepo(repo.url, repo.commit))

    def train(self):
        """Train the model on the repository.

        Raises:
            ValueError: If no repository is loaded
        """
        sorted_repos = self.database.sort_repositories_by_difficulty()

        if len(sorted_repos) == 0:
            raise ValueError(
                "No repository loaded. Call setup_github_repository() or setup_local_repository() first."
            )

        self.trainer.train(
            repos=sorted_repos, database=self.database, data_path=self.data_path
        )

    def evaluate(self):
        """Evaluate the trained model."""
        self.trainer.evaluate()

    def setup_github_repository(self, url: str, commit: str):
        """Set up a GitHub repository for processing."""
        traced_repo = self.trace_repository(
            url, commit, build_deps=self._get_build_deps()
        )
        if traced_repo:
            self.add_repository(traced_repo)
        else:
            raise ValueError(f"Failed to setup github repository {url}")

    def setup_local_repository(self, path: str):
        """Set up a local repository for processing."""
        repo = LeanGitRepo.from_path(path)
        traced_repo = self.trace_repository(
            repo.url, repo.commit, build_deps=self._get_build_deps()
        )
        if traced_repo:
            self.add_repository(traced_repo)
        else:
            raise ValueError(f"Failed to setup local repository {path}")

    def initialize_prover(self):
        """Initialize the theorem prover.

        Returns:
            List of sorry theorems to prove
        """
        sorry_theorems = []
        for repo in self.repos:
            repository = self.database.get_repository(repo.url, repo.commit)
            for theorem in repository.sorry_theorems_unproved:
                sorry_theorems.append((theorem, repo))

        self._setup_prover()

        return sorry_theorems

    @abstractmethod
    def _setup_prover(self):
        """Set up the prover agent. Must be implemented by subclasses."""
        pass

    def prove(self, whole_proof: bool = False):
        """Prove sorry theorems."""
        sorry_theorems = self.initialize_prover()

        if not sorry_theorems:
            print("No sorry theorems found to prove.")
            return

        print(f"Found {len(sorry_theorems)} sorry theorems to prove")
        for theorem, repo in sorry_theorems:
            self.prove_theorem(theorem, repo, whole_proof)

    def prove_theorem(self, theorem, repo, whole_proof: bool = False):
        """Processes a single theorem."""
        if whole_proof:
            proof = self.prover.generate_whole_proof(theorem)
            print(proof)
            return

        traced_repo_path = get_traced_repo_path(repo, build_deps=self._get_build_deps())

        server_kwargs = {
            "imports": ["Init", str(theorem.file_path).replace(".lean", "")],
            "project_path": traced_repo_path,
        }

        # Optional pantograph runtime override for environments where the packaged
        # pantograph-repl expects a different Lean stdlib than the traced repo's
        # own toolchain.
        stdlib_lean_path = os.environ.get("PANTOGRAPH_STDLIB_LEAN_PATH")
        toolchain_bin = os.environ.get("PANTOGRAPH_TOOLCHAIN_BIN")
        if stdlib_lean_path:
            if toolchain_bin:
                env = os.environ.copy()
                env["PATH"] = f"{toolchain_bin}:{env.get('PATH', '')}"
                subprocess.run(
                    ["lake", "build"],
                    cwd=traced_repo_path,
                    env=env,
                    check=True,
                )

            project_lean_path = os.path.join(
                traced_repo_path, ".lake", "build", "lib", "lean"
            )
            server_kwargs["lean_path"] = f"{project_lean_path}:{stdlib_lean_path}"

        server = Server(**server_kwargs)

        proof_attempts_log = os.environ.get("PROOF_ATTEMPTS_LOG")
        if not proof_attempts_log:
            proof_attempts_log = os.path.join(RAID_DIR, "logs", "proof_attempts.log")
        proof_attempts_log_path = Path(proof_attempts_log)
        proof_attempts_log_path.parent.mkdir(parents=True, exist_ok=True)

        with proof_attempts_log_path.open("a", encoding="utf-8") as attempts_file:
            tee = _TeeWriter(sys.stdout, attempts_file)
            with redirect_stdout(tee):
                print(
                    f"\n=== THEOREM START [{datetime.now().isoformat(timespec='seconds')}] "
                    f"{theorem.full_name} ==="
                )
                print(f"Proving {theorem.full_name}")
                result, used_tactics = self.prover.search(
                    server=server, theorem=theorem, verbose=True
                )
                print(result)
                if result.success:
                    for tactic in used_tactics:
                        print(tactic)
                else:
                    logger.info(f"No proof found for {theorem.full_name}")
                print(f"=== THEOREM END {theorem.full_name} ===")

        return result.success
