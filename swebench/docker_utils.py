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
from typing import Any

from .utils import apply_patch, parse_test_id_list

DOCKER_BASE_IMAGE = "swebench-base:latest"  # built via build_base_image.sh
_THIS_DIR = Path(__file__).parent  # .../docker_setup/swebench/


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"[docker] $ {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def image_tag_for_repo(repo_name: str, python_version: str) -> str:
    safe_name = repo_name.replace("/", "__")
    return f"swebench-repo-{safe_name}:py{python_version}"

def build_repo_image(
    repo_name: str,
    repo_source_dir: str | Path,
    force_rebuild: bool = False,
    python_version: str | None = None,
) -> str:
    """Build (or reuse) a Docker image for one repo.
    ...
    python_version: e.g. "3.9" -- selects the base image's Python so old
    C-extension repos build against a compatible interpreter/toolchain
    (see get_python_version() in repo_python_versions.py). Falls back to
    DOCKER_BASE_IMAGE if not provided, for backward compatibility.
    """
    if python_version is None:
        from swebench.repo_python_versions import get_python_version
        python_version = get_python_version(repo_name)

    tag = image_tag_for_repo(repo_name, python_version)  # see below -- must include version
    repo_source_dir = Path(repo_source_dir)

    if not force_rebuild:
        existing = _run(["docker", "images", "-q", tag])
        if existing.stdout.strip():
            print(f"[docker] Reusing cached image {tag}")
            return tag

    staging_utils = repo_source_dir / "_docker_stage_utils.py"
    staging_setup = repo_source_dir / "_docker_stage_container_env_setup.py"
    shutil.copy(_THIS_DIR / "utils.py", staging_utils)
    shutil.copy(_THIS_DIR.parent / "docker" / "container_env_setup.py", staging_setup)

    base_image = f"python:{python_version}-slim-bullseye"

    dockerfile = f"""
FROM {base_image}
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential git \\
    && rm -rf /var/lib/apt/lists/*
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
        stdout_lines = result.stdout.splitlines() if result.stdout else []
        stderr_lines = result.stderr.splitlines() if result.stderr else []
        
        combined_output = (
            "\n".join(f"[STDOUT] {line}" for line in stdout_lines)
            + "\n[STDERR]\n"
            + "\n".join("  " + line for line in stderr_lines)
        )[:6000]

        raise RuntimeError(f"Failed to build Docker image for {repo_name}:\n{combined_output}")
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


def run_test_ids_in_container(
    container_name: str,
    test_ids: list[str],
    timeout: float = 60.0,
    python_bin: str = "python3",
) -> dict[str, bool | None]:
    """Container-routed equivalent of utils.py's run_test_ids() -- same
    per-test-id, return-code-only approach, just executed via `docker exec`
    instead of a host subprocess.run().
    """
    results: dict[str, bool | None] = {}
    for test_id in test_ids:
        cmd = f"{python_bin} -m pytest -q {shlex.quote(test_id)}"
        try:
            proc = exec_in_container(container_name, cmd, timeout_seconds=timeout)
            results[test_id] = proc.returncode == 0
        except RuntimeError:
            results[test_id] = None
    return results


def evaluate_fail_to_pass_in_container(
    container_name: str,
    instance: dict[str, Any],
    workspace_dir: str | Path,
    timeout: float = 60.0,
    python_bin: str = "python3",
    test_patch_already_applied: bool = False,
) -> dict[str, Any]:
    """Container-routed equivalent of utils.py's evaluate_fail_to_pass().

    apply_patch() itself stays host-side and unmodified: it's a `git apply`
    against workspace_dir, which is the same directory bind-mounted into
    the container, so the patch is visible on both sides regardless of
    where the git command runs.
    """
    fail_to_pass = parse_test_id_list(instance.get("FAIL_TO_PASS"))
    pass_to_pass = parse_test_id_list(instance.get("PASS_TO_PASS"))

    test_patch = instance.get("test_patch", "")
    if test_patch and not test_patch_already_applied:
        try:
            apply_patch(workspace_dir, test_patch)
        except Exception as e:
            return {
                "status": "patch_failure",
                "error": f"Failed to apply test_patch: {e}",
                "fail_to_pass_results": {}, "pass_to_pass_results": {},
                "fail_to_pass_count": 0, "pass_to_pass_count": 0,
                "fail_to_pass_total": len(fail_to_pass), "pass_to_pass_total": len(pass_to_pass),
            }

    fail_to_pass_results = run_test_ids_in_container(container_name, fail_to_pass, timeout, python_bin)
    pass_to_pass_results = run_test_ids_in_container(container_name, pass_to_pass, timeout, python_bin)

    fail_to_pass_count = sum(1 for v in fail_to_pass_results.values() if v is True)
    pass_to_pass_count = sum(1 for v in pass_to_pass_results.values() if v is True)

    all_results = {**fail_to_pass_results, **pass_to_pass_results}
    timed_out = any(v is None for v in all_results.values())

    if timed_out:
        status = "timeout"
    elif not fail_to_pass and not pass_to_pass:
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