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

def _read_pyproject_build_requires(path: str | Path) -> list[str]:
    """Read a package's own declared build requirements from its
    ``pyproject.toml`` ``[build-system] requires`` list.

    This is far more reliable than guessing a generic set of build
    prerequisites -- e.g. astropy's pyproject.toml declares
    ``extension-helpers``, a specific pinned ``cython`` version, and
    ``oldest-supported-numpy``, none of which a generic
    numpy/Cython/setuptools/wheel guess would include.

    Returns an empty list if there's no pyproject.toml, no
    ``[build-system]`` section, or it can't be parsed.
    """
    pyproject = Path(path) / "pyproject.toml"
    if not pyproject.exists():
        return []

    text = pyproject.read_text()
    try:
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(text)
        return list(data.get("build-system", {}).get("requires", []))
    except Exception:
        pass

    # Fallback: no TOML parser available -- regex out a simple single
    # `requires = [...]` block, which covers the common case.
    import re
    match = re.search(r"requires\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not match:
        return []
    return re.findall(r'["\']([^"\']+)["\']', match.group(1))


def _find_build_python() -> str:
    """Find the best available Python interpreter for building old packages.

    Many SWE-bench-era packages (astropy and similar C-extension packages
    circa 2020-2023) cannot build on Python 3.12+ at all, regardless of
    setuptools pinning: ``distutils`` was removed from the stdlib in 3.12,
    and setuptools versions old enough to still support the deprecated
    APIs those packages call (e.g. ``setuptools.dep_util``) predate
    setuptools's own Python-3.12 compatibility work (they hit
    ``pkgutil.ImpImporter``, also removed in 3.12) -- there is no
    overlapping version that satisfies both constraints.

    If the interpreter currently running this code is already 3.11 or
    older, just use it. Otherwise search for an older interpreter on PATH
    (newest-first: 3.11, 3.10, 3.9, 3.8) and use that instead if found.
    Falls back to ``sys.executable`` with a loud warning if nothing older
    is available -- the resulting venv will likely fail to build older
    C-extension packages, but is still useful for pure-Python repos.
    """
    if sys.version_info[:2] <= (3, 11):
        return sys.executable

    for minor in (11, 10, 9, 8):
        candidate = shutil.which(f"python3.{minor}")
        if candidate:
            return candidate

    print(
        "[env] WARNING: only Python "
        f"{sys.version_info[0]}.{sys.version_info[1]} is available, and no "
        "older interpreter (python3.11/3.10/3.9/3.8) was found on PATH. "
        "Older C-extension packages (astropy, and similar packages from "
        "~2020-2023) are known to fail building on Python 3.12+ regardless "
        "of setuptools pinning -- distutils was removed from the stdlib "
        "in 3.12. Install an older Python for full compatibility, e.g.:\n"
        "    sudo apt install python3.10 python3.10-venv\n"
        "and re-run -- this function will pick it up automatically."
    )
    return sys.executable


def ensure_repo_environment(
    repo_path: str | Path,
    install_path: str | Path | None = None,
    timeout: float = 900.0,
) -> tuple[str, bool]:
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
        in the shared repo -- re-running the install against an
        already-populated venv is fast (it only needs to update the
        editable pointer, not reinstall every dependency).
    timeout :
        Per-pip-install-step timeout in seconds. Building packages with C
        extensions (astropy, scikit-learn, ...) from scratch can genuinely
        take several minutes; raised from 600s to 900s.

    Returns
    -------
    tuple[str, bool]
        ``(python_bin, install_ok)`` -- the venv's python executable, and
        whether the package itself installed successfully. Callers should
        check ``install_ok`` and record it distinctly from "agent didn't
        fix the bug" -- a failed environment setup means the FAIL_TO_PASS/
        PASS_TO_PASS check was never meaningful for this instance,
        regardless of what the agent did.

    Full pip output for every step is written to
    ``<repo_path>/.swebench_venv/pip_install.log`` (appended across calls)
    so failures are actually diagnosable -- only a short summary is
    printed to the console.
    """
    repo_path = Path(repo_path)
    install_path = Path(install_path) if install_path else repo_path
    venv_dir = repo_path / ".swebench_venv"
    venv_python = venv_dir / "bin" / "python"
    log_path = venv_dir / "pip_install.log"

    build_python = _find_build_python()
    venv_already_existed = venv_python.exists()

    if venv_already_existed:
        # Detect staleness: if a different (e.g. now-installed older)
        # Python is preferred than whatever this venv was actually built
        # with, rebuild rather than silently reusing it forever -- a stale
        # venv skips every install step below (gated on "not
        # venv_already_existed"), so it would otherwise never pick up
        # dependency/pytest-version/setuptools-pin fixes made after it was
        # first created.
        def _venv_py_version(python_exe: str) -> str:
            try:
                proc = subprocess.run(
                    [python_exe, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                return proc.stdout.strip() or "?"
            except Exception:
                return "?"

        existing_version = _venv_py_version(str(venv_python))
        target_version = _venv_py_version(build_python)

        if target_version != "?" and existing_version != target_version:
            print(f"[env] Existing venv for {repo_path.name} was built with Python "
                  f"{existing_version}, but Python {target_version} is now preferred "
                  f"(e.g. an older interpreter was installed after this venv was "
                  f"first created) -- rebuilding from scratch so all fixes apply.")
            shutil.rmtree(venv_dir, ignore_errors=True)
            venv_already_existed = False
        else:
            print(f"[env] Reusing existing venv for {repo_path.name} (Python {existing_version})")

    if not venv_already_existed:
        print(f"[env] Creating virtualenv for {repo_path.name} at {venv_dir} (using {build_python})")
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [build_python, "-m", "venv", str(venv_dir)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0 or not venv_python.exists():
            print(f"[env] Warning: failed to create venv for {repo_path.name} "
                  f"({result.stderr.strip()[:200]}); falling back to ambient Python")
            return sys.executable, False

    def _log(heading: str, proc: subprocess.CompletedProcess) -> None:
        with open(log_path, "a") as f:
            f.write(f"\n{'='*70}\n{heading}\nreturncode={proc.returncode}\n")
            f.write(f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")

    last_failure_output = ""

    def _pip(*args: str, cwd: Path, extra_env: dict | None = None) -> bool:
        """Run a pip command; returns True on success (returncode 0)."""
        nonlocal last_failure_output
        env = {**os.environ, **(extra_env or {})}
        try:
            proc = subprocess.run(
                [str(venv_python), "-m", "pip", *args],
                cwd=str(cwd), capture_output=True, text=True,
                timeout=timeout, check=False, env=env,
            )
            _log(f"pip {' '.join(args)} (cwd={cwd})", proc)
            if proc.returncode != 0:
                last_failure_output = proc.stderr + proc.stdout
                # Print a genuinely useful chunk, not 300 chars -- and
                # always point at the full log for the rest.
                tail = proc.stderr.strip()[-1500:] or proc.stdout.strip()[-1500:]
                print(f"[env] pip {' '.join(args)} for {cwd.name} exited "
                      f"{proc.returncode}. Last part of output:\n{tail}\n"
                      f"[env] Full output logged to {log_path}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print(f"[env] pip {' '.join(args)} for {cwd.name} timed out after {timeout}s")
            return False

    if not venv_already_existed:
        _pip("install", "--upgrade", "pip", "-q", cwd=repo_path)

        build_requires = _read_pyproject_build_requires(install_path)
        if build_requires:
            print(f"[env] Installing {install_path.name}'s declared build requirements: {build_requires}")
            _pip("install", "-q", *build_requires, cwd=repo_path)
        else:
            # No pyproject.toml (or no [build-system] requires) to go on --
            # fall back to a generic guess for packages with C extensions.
            _pip("install", "-q", "numpy", "Cython", "setuptools", "wheel", cwd=repo_path)

        # Pin AFTER the declared requirements so this wins: pyproject.toml
        # build-system.requires commonly just says "setuptools" with no
        # upper bound, which resolves to the latest version -- but many
        # SWE-bench-era packages call setup.py APIs (e.g.
        # setuptools.dep_util) that were removed in modern setuptools.
        # This is the actual cause of a second, different build failure
        # seen in practice after fixing the first (missing declared build
        # deps) one.
        #
        # IMPORTANT: pin to an EXACT version, not just an upper bound like
        # "<71" -- setuptools 70.3.0 (which satisfies "<71") already lacks
        # dep_util entirely, so that bound doesn't actually get you a
        # working version. 65.5.0 is confirmed (via direct testing) to
        # still contain the dep_util module, and satisfies
        # extension-helpers' own "setuptools>=64" requirement. On Python
        # 3.12+ this version still fails for an unrelated reason
        # (dep_util's own top-level code references pkgutil.ImpImporter,
        # removed in 3.12) -- but on Python 3.10/3.11 (what
        # _find_build_python should have selected), it works cleanly.
        _pip("install", "-q", "setuptools==65.5.0", "wheel", cwd=repo_path)

        # A reasonably old, broadly-compatible pytest -- SWE-bench Lite
        # repos mostly predate pytest 8's stricter warning-filter parsing,
        # which is exactly the crash seen in practice against newer
        # pytest. Also install `hypothesis`, a very common SWE-bench test
        # dependency (astropy, sympy, etc.) not always pulled in by a
        # plain package install.
        # `hypothesis` and `pytest-remotedata` are both very common
        # SWE-bench-era test dependencies (astropy, sympy, and others) not
        # always pulled in by a plain package install. pytest-remotedata
        # specifically gates tests marked @pytest.mark.remote_data (tests
        # that would otherwise need network access) -- without it,
        # collecting any test file that imports it directly raises
        # ModuleNotFoundError and aborts collection of the whole run.
        _pip("install", "-q", "pytest<8", "hypothesis", "pytest-remotedata", cwd=repo_path)

    # Fetch git submodules -- older astropy (and other packages of that
    # era) vendored C libraries (cfitsio, wcslib, expat, erfa) as git
    # submodules, which a plain `git clone` never populates. Best-effort:
    # harmless no-op for repos without submodules.
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=str(install_path), capture_output=True, text=True,
        timeout=timeout, check=False,
    )

    # Install the package itself editable, from install_path, using
    # --no-build-isolation so it actually respects everything we just
    # installed (numpy/Cython/the declared build requirements/our
    # setuptools pin). We deliberately do NOT fall back to a normal
    # isolated build on failure: an isolated build creates its own fresh
    # temporary environment from pyproject.toml's own declared
    # requirements (usually an unbounded "setuptools"), which reintroduces
    # exactly the dep_util-style failure we just fixed -- so a retry there
    # can never help for the packages this pin exists for, and just
    # produces a second, confusingly-different-looking failure.
    install_ok = _pip("install", "-q", "--no-build-isolation", "-e", ".", cwd=install_path)

    if not install_ok:
        hint = _diagnose_build_failure(last_failure_output, venv_python)
        print(f"[env] WARNING: could not install {install_path.name}'s own package. "
              f"Tests will very likely fail regardless of what the agent did -- "
              f"see {log_path} for the full error.")
        if hint:
            print(f"[env] {hint}")

    return str(venv_python), install_ok


def _diagnose_build_failure(output: str, venv_python: Path) -> str | None:
    """Recognize common C-extension build failures and return an
    actionable one-line hint, or None if nothing recognized.

    These are system-package problems pip cannot fix on its own --
    catching the common ones here saves a round trip through a confusing
    raw traceback.
    """
    # Figure out which Python version the venv actually uses, so the hint
    # names the right -dev package rather than guessing.
    import re
    match = re.search(r"python3\.(\d+)", str(venv_python))
    py_suffix = f"python3.{match.group(1)}" if match else "python3"

    if "Python.h: No such file or directory" in output:
        return (
            f"Missing Python development headers. Install them with:\n"
            f"    sudo apt install {py_suffix}-dev\n"
            f"then delete this repo's .swebench_venv and re-run."
        )
    if re.search(r"fatal error: [\w./]+\.h: No such file or directory", output):
        missing_header = re.search(r"fatal error: ([\w./]+\.h): No such file or directory", output)
        header_name = missing_header.group(1) if missing_header else "a header file"
        return (
            f"Missing a system header ({header_name}) needed to compile a C "
            f"extension. This is usually a missing system dev package (e.g. "
            f"`sudo apt install {py_suffix}-dev build-essential libfreetype6-dev "
            f"libpng-dev` depending on which library it belongs to) -- check "
            f"the full pip_install.log for which library's headers are missing."
        )
    if "gcc" in output.lower() and "command not found" in output.lower():
        return "No C compiler found. Install one with: sudo apt install build-essential"
    if "No module named 'Cython'" in output or "No module named 'cython'" in output:
        return "Cython wasn't actually available at build time -- this usually means the pyproject.toml build requirements install (above) failed silently; check pip_install.log for that step."

    return None


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
# Cheap-subset filtering (pure-Python repos, no C-extension build pain)
# ---------------------------------------------------------------------------
#
# SWE-bench Lite spans 12 repos. Several of them (astropy, matplotlib,
# scikit-learn, pandas-adjacent xarray/seaborn) pull in numpy/scipy/pandas/
# matplotlib and sometimes need an old setuptools or submodule fetching just
# to build -- see ensure_repo_environment() below, most of whose complexity
# exists purely to survive that. None of that build pain has anything to do
# with whether the Reasoner/Actioner split actually works, so for early
# iteration it's worth being able to filter it out while still running the
# real dataset and the real FAIL_TO_PASS/PASS_TO_PASS scoring.
PURE_PYTHON_REPOS: tuple[str, ...] = (
    "django/django",
    "pallets/flask",
    "psf/requests",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "sphinx-doc/sphinx",
    "sympy/sympy",
)


def filter_pure_python_instances(instances: list[dict]) -> list[dict]:
    """Keep only instances whose ``repo`` is in :data:`PURE_PYTHON_REPOS`.

    A cheap way to get a faster-iterating subset of SWE-bench Lite (no
    numpy/scipy/matplotlib/pandas C-extension builds) while still running
    real instances with real FAIL_TO_PASS/PASS_TO_PASS scoring, instead of
    switching to a synthetic or non-SWE-bench task.
    """
    return [inst for inst in instances if inst.get("repo") in PURE_PYTHON_REPOS]


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
    test_patch_already_applied: bool = False,
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
    test_patch_already_applied :
        Set to ``True`` when the caller already applied
        ``instance["test_patch"]`` to ``repo_path`` earlier in the run --
        e.g. before handing the workspace to the agent, so the agent's own
        ``run_tests`` calls can target real FAIL_TO_PASS node ids instead of
        a generic command (see ``Agent.solve``'s ``fail_to_pass_tests``
        parameter). Applying the same unified diff twice would fail with
        "patch does not apply", which would misreport a genuine non-issue
        as ``patch_failure``; this flag skips the redundant re-application.

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
    if test_patch and not test_patch_already_applied:
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