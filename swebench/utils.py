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
import sys
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


# Alias kept for callers (e.g. experiments/run_experiment.py) that import
# `find_repo_path` by that name.
find_repo_path = find_repository


def create_workspace(repo_path: str | Path, workspace_id: str, base_commit: str | None = None) -> Path:
    """Create an isolated workspace for a single SWE-bench instance.

    Each workspace is a real, git-backed checkout of the repository (via
    ``git worktree``) at its required base commit, so that one task's
    modifications cannot contaminate another and so ``get_git_diff()`` /
    the agent's file tools have actual files to work with.

    Parameters
    ----------
    repo_path :
        Path to the cloned repository.
    workspace_id :
        Unique identifier for this workspace (typically the instance ID).
    base_commit :
        Commit to check the worktree out at. If omitted, checks out
        whatever HEAD currently points to in ``repo_path``.

    Returns
    -------
    pathlib.Path
        The path to the isolated workspace directory.
    """
    repo_path = Path(repo_path)
    # Place workspaces as siblings of repo_path (not nested inside it) so
    # `git worktree` doesn't get confused about a worktree living inside
    # the main tree it belongs to.
    workspace_dir = repo_path.parent / f"{repo_path.name}__workspace_{workspace_id}"

    if workspace_dir.exists():
        # Clean up any previous attempt so we start fresh every time --
        # both the worktree registration (so git doesn't think it's still
        # in use) and the directory itself.
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace_dir)],
            cwd=str(repo_path), capture_output=True, text=True, check=False,
        )
        shutil.rmtree(workspace_dir, ignore_errors=True)

    cmd = ["git", "worktree", "add", "--force", "--detach", str(workspace_dir)]
    if base_commit:
        cmd.append(base_commit)
    result = subprocess.run(
        cmd, cwd=str(repo_path), capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create workspace via git worktree for {workspace_id}: "
            f"{result.stderr.strip()}"
        )

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





# ---------------------------------------------------------------------------
# Per-repo test environment
# ---------------------------------------------------------------------------
#
# setup_repo() (in benchmarks/swebench.py) only clones and checks out a
# commit -- it never installs the repo's own dependencies. Without that,
# every single test run fails immediately with import errors or pytest
# version mismatches, regardless of what the agent actually did. This
# creates one virtualenv per repo (not per instance -- repos are cloned
# once and reused across their instances, so this follows the same pattern
# and pays the setup cost once per repo, not once per instance).
#
# KNOWN LIMITATION: different instances of the SAME repo can be pinned to
# different dependency versions across their history (this is exactly why
# the official SWE-bench harness uses one Docker image PER INSTANCE, not
# per repo). A single shared venv per repo is a practical middle ground --
# it fixes the common case (missing test-only dependencies like
# `hypothesis`, wildly incompatible pytest majors) but won't perfectly
# reproduce every instance's exact original environment. If you need exact
# parity, use the official Docker-based harness (see
# docs/DOWNLOADING_SWEBENCH_LITE.md, Option 3).

def ensure_repo_environment(
    repo_path: str | Path,
    install_path: str | Path | None = None,
    timeout: float = 600.0,
) -> str:
    """Create (or reuse) a per-repo virtualenv with the package installed
    editable, plus pytest and commonly-missing test dependencies.

    Parameters
    ----------
    repo_path :
        Path used to locate/create the venv itself (``<repo_path>/
        .swebench_venv``). Pass the shared cloned repo here (not a
        per-instance workspace) so the (often slow) dependency install is
        only paid once per repo and reused across that repo's instances.
    install_path :
        Directory to editable-install the package FROM. Defaults to
        ``repo_path``. Pass a per-instance workspace directory here (e.g.
        a git worktree with the agent's edits) so the venv's editable
        install points at THAT instance's files rather than a stale copy
        in the shared repo -- re-running ``pip install -e <install_path>``
        against an already-populated venv is fast (it only needs to update
        the editable pointer, not reinstall every dependency).
    timeout :
        Per-pip-install-step timeout in seconds (dependency installs for
        large scientific packages like astropy/scikit-learn can be slow).

    Returns
    -------
    str
        Path to that venv's ``python`` executable. Callers should use this
        (via ``[python_bin, "-m", "pytest", ...]``) instead of the ambient
        ``python``/``pytest`` when running tests for this repo.
    """
    repo_path = Path(repo_path)
    install_path = Path(install_path) if install_path else repo_path
    venv_dir = repo_path / ".swebench_venv"
    venv_python = venv_dir / "bin" / "python"

    venv_already_existed = venv_python.exists()

    if not venv_already_existed:
        print(f"[env] Creating virtualenv for {repo_path.name} at {venv_dir}")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0 or not venv_python.exists():
            print(f"[env] Warning: failed to create venv for {repo_path.name} "
                  f"({result.stderr.strip()[:200]}); falling back to ambient Python")
            return sys.executable

    def _pip(*args: str, cwd: Path) -> None:
        try:
            proc = subprocess.run(
                [str(venv_python), "-m", "pip", *args, "-q"],
                cwd=str(cwd), capture_output=True, text=True,
                timeout=timeout, check=False,
            )
            if proc.returncode != 0:
                print(f"[env] pip {' '.join(args)} for {cwd.name} "
                      f"exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        except subprocess.TimeoutExpired:
            print(f"[env] pip {' '.join(args)} for {cwd.name} timed out after {timeout}s")

    if not venv_already_existed:
        _pip("install", "--upgrade", "pip", cwd=repo_path)
        # A reasonably old, broadly-compatible pytest -- SWE-bench Lite
        # repos mostly predate pytest 8's stricter warning-filter parsing,
        # which is exactly the crash seen in practice against newer
        # pytest. Also install `hypothesis`, a very common SWE-bench test
        # dependency (astropy, sympy, etc.) not always pulled in by a
        # plain `pip install -e .`.
        _pip("install", "pytest<8", "hypothesis", cwd=repo_path)

    # Always (re-)do the editable install against install_path, even on a
    # reused venv -- this is what makes a per-instance workspace's edits
    # actually importable, and is cheap once dependencies are already in
    # place.
    _pip("install", "-e", ".", cwd=install_path)

    return str(venv_python)


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


# ---------------------------------------------------------------------------
# FAIL_TO_PASS / PASS_TO_PASS evaluation
# ---------------------------------------------------------------------------
#
# This is a *lightweight* reimplementation of the official SWE-bench
# evaluation methodology -- it checks the same FAIL_TO_PASS/PASS_TO_PASS
# test lists the official harness does, but runs them directly with pytest
# in-place rather than inside the official per-instance Docker images with
# pinned dependency versions. That means:
#   - It works for repos whose test suite pytest can actually collect
#     (most of the SWE-bench Lite repos: requests, pylint, sympy, etc.)
#   - It will NOT work well for repos with bespoke test runners (e.g.
#     Django's `runtests.py`) or that need a specific conda/virtualenv per
#     instance -- the official harness handles this with Docker; this
#     project intentionally doesn't (see docs/DOWNLOADING_SWEBENCH_LITE.md,
#     "Option 3", for the official Docker-based harness if you need exact
#     parity).

def parse_test_id_list(raw: Any) -> list[str]:
    """Parse a FAIL_TO_PASS / PASS_TO_PASS dataset field into test node ids.

    In the raw dataset these fields are usually a JSON-encoded string, e.g.
    ``'["tests/test_foo.py::test_bar"]'``, but callers may also pass an
    already-parsed list (some dataset loaders decode it automatically).
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t) for t in parsed]
        except json.JSONDecodeError:
            pass
        # Not JSON -- treat the whole string as a single test id.
        return [raw]
    return []


def run_test_ids(
    repo_path: str | Path,
    test_ids: list[str],
    timeout: float = 60.0,
    python_bin: str = "python",
) -> dict[str, bool | None]:
    """Run each test node id individually with pytest and record pass/fail.

    Running tests one at a time (rather than parsing one combined run's
    output) is slower but far more robust across the different test
    reporters used by SWE-bench repos -- it only relies on pytest's own
    return code, never on scraping stdout.

    Parameters
    ----------
    python_bin :
        Python executable to invoke pytest with. Pass the path returned by
        :func:`ensure_repo_environment` so tests run against the repo's own
        installed dependencies -- the ambient Python typically doesn't have
        the target package (or its test-only deps like `hypothesis`)
        installed at all, which causes every test to fail immediately with
        import errors regardless of what the agent did.

    Returns
    -------
    dict[str, bool | None]
        Maps each test id to ``True`` (passed), ``False`` (failed/errored),
        or ``None`` (timed out or could not be run at all, e.g. pytest
        couldn't collect it).
    """
    results: dict[str, bool | None] = {}
    for test_id in test_ids:
        try:
            proc = subprocess.run(
                [python_bin, "-m", "pytest", "-q", test_id],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            results[test_id] = proc.returncode == 0
        except subprocess.TimeoutExpired:
            results[test_id] = None
        except Exception:
            results[test_id] = None
    return results


def evaluate_fail_to_pass(
    repo_path: str | Path,
    instance: dict[str, Any],
    timeout: float = 60.0,
    python_bin: str = "python",
) -> dict[str, Any]:
    """Run the FAIL_TO_PASS / PASS_TO_PASS check for one instance.

    This is the core of "did the agent actually resolve the issue" per the
    official SWE-bench methodology: apply the instance's ``test_patch``
    (which brings in the reference tests -- the agent never sees this, only
    the ``problem_statement``), then run each FAIL_TO_PASS and PASS_TO_PASS
    test individually. An instance only counts as resolved if *every*
    FAIL_TO_PASS test now passes (the bug is actually fixed) AND *every*
    PASS_TO_PASS test still passes (no regressions introduced).

    Parameters
    ----------
    repo_path :
        Path to the repository/workspace containing the agent's changes.
    instance :
        The dataset instance dict; must contain (or omit) ``FAIL_TO_PASS``,
        ``PASS_TO_PASS``, and optionally ``test_patch``.
    timeout :
        Per-test timeout in seconds.
    python_bin :
        Python executable to run tests with -- see :func:`run_test_ids`.
        Pass the result of :func:`ensure_repo_environment` (repo_path) here.

    Returns
    -------
    dict
        ``status``: one of ``"resolved"``, ``"not_resolved"``,
        ``"patch_failure"`` (the test_patch itself couldn't be applied --
        an environment/setup problem, not the agent's fault), or
        ``"timeout"`` (at least one test run timed out).
        Also includes ``fail_to_pass_results`` / ``pass_to_pass_results``
        (dicts of test_id -> True/False/None) and the corresponding
        ``*_count`` (how many passed) / ``*_total`` (how many were checked).
    """
    fail_to_pass = parse_test_id_list(instance.get("FAIL_TO_PASS"))
    pass_to_pass = parse_test_id_list(instance.get("PASS_TO_PASS"))

    test_patch = instance.get("test_patch", "")
    if test_patch:
        try:
            apply_patch(repo_path, test_patch)
        except Exception as e:
            return {
                "status": "patch_failure",
                "error": f"Failed to apply test_patch: {e}",
                "fail_to_pass_results": {},
                "pass_to_pass_results": {},
                "fail_to_pass_count": 0,
                "pass_to_pass_count": 0,
                "fail_to_pass_total": len(fail_to_pass),
                "pass_to_pass_total": len(pass_to_pass),
            }

    fail_to_pass_results = run_test_ids(repo_path, fail_to_pass, timeout=timeout, python_bin=python_bin)
    pass_to_pass_results = run_test_ids(repo_path, pass_to_pass, timeout=timeout, python_bin=python_bin)

    fail_to_pass_count = sum(1 for v in fail_to_pass_results.values() if v is True)
    pass_to_pass_count = sum(1 for v in pass_to_pass_results.values() if v is True)

    all_results = {**fail_to_pass_results, **pass_to_pass_results}
    timed_out = any(v is None for v in all_results.values())

    if timed_out:
        status = "timeout"
    elif not fail_to_pass and not pass_to_pass:
        # No gold test lists to check against at all (e.g. a synthetic
        # local task) -- there's nothing to confirm resolution against.
        status = "not_resolved"
    elif fail_to_pass_count == len(fail_to_pass) and pass_to_pass_count == len(pass_to_pass):
        status = "resolved"
    else:
        status = "not_resolved"

    return {
        "status": status,
        "fail_to_pass_results": fail_to_pass_results,
        "pass_to_pass_results": pass_to_pass_results,
        "fail_to_pass_count": fail_to_pass_count,
        "pass_to_pass_count": pass_to_pass_count,
        "fail_to_pass_total": len(fail_to_pass),
        "pass_to_pass_total": len(pass_to_pass),
    }

