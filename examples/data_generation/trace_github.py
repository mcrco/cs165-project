"""
Example script for generating dataset from a Lean 4 GitHub repository. 
The data is saved at <RAID_DIR>/<DATA_DIR>/<repo.name>_<repo.commit>.
e.g. LeanDojo-v2/raid/data/lean4-example_005de00d03f1aaa32cb2923d5e3cbaf0b954a192

Usage: python examples/data_generation/trace_github.py
"""

from lean_dojo_v2.database import DynamicDatabase

url = "https://github.com/durant42040/MyNat.git"
commit = "9806cd09fb59de11258c7d63336fbbc8799af253"

database = DynamicDatabase()

database.trace_repository(
    url=url,
    commit=commit,
    build_deps=True,
)
