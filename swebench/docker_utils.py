"""Docker-backed repo environments for SWE-bench Lite instances.

Replacement for the environment-setup half of swebench/utils.py's
ensure_repo_environment() -- instead of a shared venv on the host (picking
whatever <=3.11 interpreter is available, per _find_build_python()), each
repo gets its own Docker image built on Python 3.11, with dependencies
installed by container_env_setup.py -- which imports and reuses
_read_pyproject_build_requires()/_diagnose_build_failure() from utils.py
directly, so the actual pip sequence (build requires, the setuptools==65.5.0
pin, pytest<8/hypothesis/pytest-remotedata, --no-build-isolation editable
install) stays identical to the host-side path instead of being
re-transcribed and risking drift.

One image is built per REPO (not per instance), matching the existing
"pay the install cost once, reuse across instances" pattern in
ensure_repo_environment() -- images are cached by tag and only rebuilt if
missing.

WHAT THIS DOES NOT DO: doesn't touch agent.py, reasoner.py, or
ollama_client.py -- the reasoner_failed issue is unrelated to this.

INTEGRATION POINTS (not made blind, since I don't have run_instance.py or
Actioner.execute() -- happy to wire these in directly if you share those):
  - run_instance.py: replace the ensure_repo_environment(...) call with
    ensure_repo_environment_docker(...) below.
  - Actioner.execute()'s run_command/run_tests branches: replace
    subprocess.run(cmd, cwd=workspace_dir, ...) with
    exec_in_container(container_name, cmd).
  - evaluate.py's run_test_ids(..., python_bin=...): pass python_bin="python3"
    and route the subprocess.run call through exec_in_container instead.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

DOCKER_BASE_IMAGE = "swebench-base:latest"  # built via build_base_image.sh
_THIS_DIR = Path(__file__).parent  # .../docker_setup/swebench/


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"[docker] $ {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def image_tag_for_repo(repo_name: str) -> str:
    safe = repo_name.replace("/", "__").lower()
    return f"swebench-repo-{safe}:latest"


def build_repo_image(repo_name: str, repo_source_dir: str | Path, force_rebuild: bool = False) -> str:
    """Build (or reuse) a Docker image for one repo.

    repo_source_dir: the shared cloned repo (same directory
    ensure_repo_environment() would build a venv next to) -- NOT a
    per-instance workspace. This mirrors "build once per repo, reuse across
    instances of that repo."

    force_rebuild: set True to rebuild even if a cached image with this tag
    already exists -- e.g. after the repo checkout has changed, the same
    case that makes ensure_repo_environment() detect staleness and rebuild
    its venv.
    """
    tag = image_tag_for_repo(repo_name)
    repo_source_dir = Path(repo_source_dir)

    if not force_rebuild:
        existing = _run(["docker", "images", "-q", tag])
        if existing.stdout.strip():
            print(f"[docker] Reusing cached image {tag}")
            return tag

    # Build context needs the repo itself, plus utils.py and
    # container_env_setup.py so the install step can import the real
    # helper functions instead of duplicating their logic. Docker can't
    # COPY from outside the build context, so stage them into a temp
    # subdirectory of the repo checkout, build from there, then clean up.
    staging_utils = repo_source_dir / "_docker_stage_utils.py"
    staging_setup = repo_source_dir / "_docker_stage_container_env_setup.py"
    shutil.copy(_THIS_DIR / "utils.py", staging_utils)
    shutil.copy(_THIS_DIR.parent / "docker" / "container_env_setup.py", staging_setup)

    dockerfile = f"""
FROM {DOCKER_BASE_IMAGE}
COPY . /workspace/repo
COPY _docker_stage_utils.py /workspace/utils.py
COPY _docker_stage_container_env_setup.py /workspace/container_env_setup.py
WORKDIR /workspace/repo
RUN python3 /workspace/container_env_setup.py
"""
    dockerfile_path = repo_source_dir / "Dockerfile.generated"
    dockerfile_path.write_text(dockerfile)

    try:
        result = _run(["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(repo_source_dir)])
    finally:
        # Don't leave build-staging artifacts sitting in the actual repo
        # checkout -- ensure_repo_environment()/reset_repository() and
        # friends assume that directory only ever contains the repo itself.
        staging_utils.unlink(missing_ok=True)
        staging_setup.unlink(missing_ok=True)
        dockerfile_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build Docker image for {repo_name}:\n{result.stderr[-3000:]}")
    print(f"[docker] Built image {tag}")
    return tag


def start_instance_container(image_tag: str, instance_workspace_dir: str | Path, container_name: str) -> str:
    """Start a long-lived container for one instance, with its workspace
    (the git worktree from create_workspace()) mounted so file edits made
    through Actioner tools are visible both inside and outside the
    container.
    """
    _run(["docker", "rm", "-f", container_name])  # clean up any stale container with this name

    result = _run([
        "docker", "run", "-d",
        "--name", container_name,
        "-v", f"{Path(instance_workspace_dir).resolve()}:/workspace/repo",
        "-w", "/workspace/repo",
        image_tag,
        "sleep", "infinity",  # keep alive for repeated `docker exec`
    ])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container {container_name}:\n{result.stderr}")
    return container_name


def exec_in_container(container_name: str, command: str, timeout_seconds: float = 120.0) -> subprocess.CompletedProcess:
    """Drop-in replacement for subprocess.run(cmd, cwd=workspace_dir, ...)
    in Actioner.execute()'s run_command/run_tests branches, and for
    run_test_ids()'s per-test-id subprocess.run call in utils.py.
    """
    try:
        return subprocess.run(
            ["docker", "exec", container_name, "bash", "-lc", command],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Command timed out in container {container_name}: {command}") from e


def stop_instance_container(container_name: str) -> None:
    _run(["docker", "rm", "-f", container_name])


def ensure_repo_environment_docker(
    repo_name: str,
    repo_source_dir: str | Path,
    instance_id: str,
    instance_workspace_dir: str | Path,
) -> tuple[str, str]:
    """Replacement for ensure_repo_environment() at its call site in
    run_instance.py.

    Returns (python_bin, container_name):
      - python_bin: always "python3" here (the container's own interpreter
        -- pass this to run_test_ids/evaluate_fail_to_pass's python_bin
        param, same as the venv_python string it currently returns).
      - container_name: pass to exec_in_container() for run_command/
        run_tests calls instead of calling subprocess.run() on the host.

    Unlike ensure_repo_environment(), this doesn't return install_ok --
    check the "errors" printed by container_env_setup.py during
    build_repo_image() (surfaced via the RuntimeError it raises on build
    failure) instead.
    """
    image_tag = build_repo_image(repo_name, repo_source_dir)
    container_name = f"swebench-{instance_id.replace('/', '_')}"
    start_instance_container(image_tag, instance_workspace_dir, container_name)
    return "python3", container_name
