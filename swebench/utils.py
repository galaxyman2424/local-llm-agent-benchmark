"""Shared SWE-bench utility functions.

Provides centralized helpers for dataset loading, repository setup, patch
application, git operations, and subprocess management. All higher-level
scripts (download_dataset, setup_repos, run_instance, evaluate) import from here
instead of duplicating logic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: str | None = None) -> list[dict]:
    """Load the SWE-bench Lite dataset from a JSONL file.

    Parameters
    ----------
    dataset_path :
        Absolute or relative path to the dataset file. Defaults to the
        cached location under ``seed_repos/swe_bench_lite/dataset.jsonl``.

    Returns
    -------
    list[dict]
        Each entry is a task instance with keys like ``instance_id``,
        ``repo``, ``base_commit``, etc.
    """
    if dataset_path is None:
        dataset_path = Path("seed_repos/swe_bench_lite/dataset.jsonl")

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run swebench/download_dataset.py first."
        )

    instances: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def load_instance(instance_id: str, dataset_path: str | None = None) -> dict:
    """Load a single instance by its ID from the cached dataset.

    Parameters
    ----------
    instance_id :
        The unique identifier (e.g., ``django__django-12345``).
    dataset_path :
        Optional override for the dataset file path.

    Returns
    -------
    dict
        The parsed task description for that instance.
    """
    instances = load_dataset(dataset_path)
    for inst in instances:
        if inst.get("instance_id") == instance_id:
            return inst
    raise KeyError(
        f"Instance '{instance_id}' not found in dataset."
    )


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------

def find_repository(repo_name: str, seed_repo_dir: str = "seed_repos") -> Path:
    """Locate a cached repository directory.

    Returns the path to ``<seed_repo_dir>/<repo_name>/`` if it exists;
    otherwise returns a placeholder path so callers can decide whether to
    clone or fail fast.
    """
    return Path(seed_repo_dir) / repo_name


def create_workspace(repo_path: str | Path, workspace_id: str) -> Path:
    """Create an isolated workspace for a single SWE-bench instance.

    Each workspace is a fresh checkout of the repository at its required
    base commit so that one task's modifications cannot contaminate another.

    Parameters
    ----------
    repo_path :
        Path to the cloned repository.
    workspace_id :
        Unique identifier for this workspace (typically the instance ID).

    Returns
    -------
    pathlib.Path
        The path to the isolated workspace directory.
    """
    workspace_dir = Path(repo_path) / f"workspace_{workspace_id}"

    # Clean up any previous attempt so we start fresh every time
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)

    workspace_dir.mkdir(parents=True, exist_ok=True)

    return workspace_dir


def reset_repository(repo_path: str | Path) -> None:
    """Reset a repository to a clean state.

    Runs ``git clean -fdx`` and ``git checkout -- .`` to ensure the working
    tree is pristine before an agent begins editing.
    """
    subprocess.run(
        ["git", "clean", "-fdx"],
        cwd=str(repo_path),
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ["git", "checkout", "--", "."],
        cwd=str(repo_path),
        capture_output=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_git_diff(repo_path: str | Path) -> str:
    """Return the uncommitted diff of a repository.

    Useful for capturing whatever patch the agent produced before evaluation.
    """
    result = subprocess.run(
        ["git", "diff"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def apply_patch(repo_path: str | Path, patch_content: str) -> None:
    """Apply a unified-diff patch to a repository.

    Parameters
    ----------
    repo_path :
        The repository directory to modify.
    patch_content :
        A standard unified diff string (the output of ``git diff``).
    """
    with open(patch_content, "w") as f:
        f.write(patch_content)

    result = subprocess.run(
        ["git", "apply", "--reject"],
        cwd=str(repo_path),
        input=patch_content,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to apply patch:\n{result.stderr}")


def checkout_commit(repo_path: str | Path, commit_hash: str) -> None:
    """Checkout a specific commit in the given repository.

    Used after cloning (or re-cloning) to land exactly at the required base
    commit for an instance.
    """
    subprocess.run(
        ["git", "checkout", commit_hash],
        cwd=str(repo_path),
        capture_output=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def run_command(cmd: str | list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    """Run an arbitrary shell command with a timeout and return the result."""
    if isinstance(cmd, str):
        cmd = ["sh", "-c", cmd]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def save_logs(log_path: str | Path, log_entries: list[dict]) -> None:
    """Persist a list of log entries to disk as JSON.

    Parameters
    ----------
    log_path :
        Destination file path (created if missing).
    log_entries :
        Each entry should be a dict with keys like ``timestamp``, ``level``,
        ``message`` etc.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log_entries, f, indent=2)

