#!/usr/bin/env python3
"""Install a repo's dependencies inside its Docker image.

Runs once, at `docker build` time, inside the repo-specific image (see
build_repo_image() in docker_utils.py). Mirrors ensure_repo_environment()'s
actual install sequence from swebench/utils.py -- literally importing its
helper functions rather than re-implementing them, so this can't silently
drift out of sync if that function's pins/logic change later.

Unlike ensure_repo_environment(), there's no venv here: the container
itself is the isolated environment, so everything installs straight into
the image's system Python.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# utils.py is COPY'd in next to this script (see build_repo_image()).
sys.path.insert(0, str(Path(__file__).parent))
from utils import _diagnose_build_failure, _read_pyproject_build_requires  # noqa: E402

REPO_DIR = Path("/workspace/repo")


_last_failure_output = ""


def _pip(*args: str) -> bool:
    global _last_failure_output
    proc = subprocess.run(
        [sys.executable, "-m", "pip", *args],
        cwd=str(REPO_DIR),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        _last_failure_output = proc.stderr + proc.stdout
        print(f"[container-env] pip {' '.join(args)} failed:\n{proc.stderr[-2000:]}")
        return False
    return True


def main() -> None:
    # 1. Package's own declared build requirements (numpy/Cython pins etc,
    #    e.g. astropy's pyproject.toml) -- falls back to a generic guess if
    #    there's no pyproject.toml build-system section to read.
    build_requires = _read_pyproject_build_requires(REPO_DIR)
    if build_requires:
        print(f"[container-env] Installing declared build requirements: {build_requires}")
        _pip("install", "-q", *build_requires)
    else:
        _pip("install", "-q", "numpy", "Cython", "setuptools", "wheel")

    # 2. The setuptools pin ensure_repo_environment() found necessary for
    #    packages that call setup.py APIs removed from modern setuptools
    #    (see the long comment on this exact line in utils.py -- 65.5.0 is
    #    the version confirmed to still contain dep_util).
    _pip("install", "-q", "setuptools==65.5.0", "wheel")

    # 3. Common SWE-bench-era test dependencies not always pulled in by a
    #    plain package install.
    _pip("install", "-q", "pytest<8", "hypothesis", "pytest-remotedata")

    # 4. Git submodules (older astropy-era vendored C libraries).
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=str(REPO_DIR), capture_output=True, text=True, check=False,
    )

    # 5. The package itself, editable, --no-build-isolation so it respects
    #    everything installed above instead of building in a fresh
    #    pyproject.toml-only environment (which would reintroduce the
    #    exact setuptools problem step 2 just fixed).
    install_ok = _pip("install", "-q", "--no-build-isolation", "-e", ".")

    if not install_ok:
        hint = _diagnose_build_failure(_last_failure_output, Path(sys.executable))
        print("[container-env] WARNING: package install failed -- tests will "
              "very likely fail regardless of what the agent does.")
        if hint:
            print(f"[container-env] {hint}")
        sys.exit(1)


if __name__ == "__main__":
    main()
