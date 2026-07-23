"""Prepare repositories for SWE-bench instances.

Handles locating cached repos, cloning missing ones, checking out required
commits, creating isolated workspaces, and resetting working trees so that
one task's modifications cannot contaminate another.

Lifecycle:
    Cached Repository
          │
          ▼
    Select Repository
          │
          ▼
    Checkout Required Base Commit
          │
          ▼
    Create Isolated Workspace
          │
          ▼
    Agent Receives Workspace
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def setup_repos(
    repo_name: str,
    base_commit: str,
    seed_repo_dir: str = "seed_repos",
    cache_dir: str | None = None,
) -> Path:
    """Set up a repository for a single SWE-bench instance.

    Parameters
    ----------
    repo_name :
        Name of the repository (e.g., ``django``).
    base_commit :
        The commit hash to checkout after cloning.
    seed_repo_dir :
        Directory where cloned repos are stored. Defaults to
        ``seed_repos/`` relative to this script's location.
    cache_dir :
        Optional override for the seed repo directory.

    Returns
    -------
    pathlib.Path
        The path to the isolated workspace for this instance.
    """
    if cache_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        seed_repo_dir = str(project_root / seed_repo_dir)

    repo_path = Path(seed_repo_dir) / repo_name

    # If repo doesn't exist, clone it from GitHub
    if not repo_path.exists():
        print(f"[setup_repos] Cloning {repo_name} into {repo_path}")
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo_name}.git", str(repo_path)],
            capture_output=True, check=False,
        )

    # Checkout the required base commit
    print(f"[setup_repos] Checking out {base_commit} in {repo_path}")
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=str(repo_path),
        capture_output=True, check=False,
    )

    # Create an isolated workspace for this instance
    workspace_dir = repo_path / f"workspace_{base_commit[:8]}"

    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)

    subprocess.run(
        ["git", "worktree", "add", "-f", str(workspace_dir), base_commit],
        cwd=str(repo_path),
        capture_output=True, check=False,
    )

    return Path(workspace_dir)


def reset_workspace(repo_path: str | Path, workspace_id: str) -> None:
    """Reset a workspace to its clean state.

    Runs ``git clean -fdx`` and ``git checkout -- .`` to ensure the working
    tree is pristine before an agent begins editing.
    """
    subprocess.run(
        ["git", "clean", "-fdx"],
        cwd=str(repo_path),
        capture_output=True, check=False,
    )
    subprocess.run(
        ["git", "checkout", "--", "."],
        cwd=str(repo_path),
        capture_output=True, check=False,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python setup_repos.py <repo_name> <base_commit>")
        sys.exit(1)

    repo_name = sys.argv[1]
    base_commit = sys.argv[2]
    workspace = setup_repos(repo_name, base_commit)
    print(f"\n[setup_repos] Workspace ready at: {workspace}")
