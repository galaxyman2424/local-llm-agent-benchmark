"""Download and cache the SWE-bench Lite dataset.

This script is run once to acquire the benchmark metadata and store it
locally so subsequent runs don't need to re-download anything. It does NOT
execute agents or evaluate tasks — that's handled by other scripts in this
directory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def download_dataset(
    cache_dir: str | None = None,
    dataset_url: str | None = None,
) -> Path:
    """Download the SWE-bench Lite dataset and save it locally.

    Parameters
    ----------
    cache_dir :
        Directory where the cached dataset will be stored. Defaults to a
        subdirectory under ``seed_repos/swe_bench_lite/`` in the current
        project root.
    dataset_url :
        URL to fetch the dataset from. If not provided, defaults to the
        official SWE-bench Lite JSONL endpoint or raises an error if no
        source is configured.

    Returns
    -------
    pathlib.Path
        The path to the saved ``dataset.jsonl`` file.
    """
    # Determine cache directory relative to this script's location so that
    # running from any working directory still caches in the right place.
    if cache_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        seed_repos = project_root / "seed_repos"
        swe_bench_lite_dir = seed_repos / "swe_bench_lite"
        cache_dir = str(swe_bench_lite_dir)

    # Ensure the directory structure exists (repos/ subdirectory is created
    # so that setup_repos.py can write cloned repositories later).
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    repos_dir = Path(cache_dir) / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(cache_dir) / "dataset.jsonl"

    if dataset_path.exists():
        print(f"[download_dataset] Dataset already cached at {dataset_path}")
        return dataset_path

    # In a real implementation we'd fetch from the network. For now we raise
    # since this is a scaffolding script — replace with actual download logic
    # when integrating with the SWE-bench API or GitHub releases.
    if not dataset_url:
        raise RuntimeError(
            "No dataset URL provided and local file not found. "
            "Provide dataset_url or pre-populate the cache."
        )

    print(f"[download_dataset] Fetching dataset from {dataset_url}")
    import subprocess
    result = subprocess.run(
        ["curl", "-sL", dataset_url, "--output", str(dataset_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download dataset: {result.stderr}")

    print(f"[download_dataset] Saved to {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    args = sys.argv[1:]
    cache_dir = None
    url = None
    for a in args:
        if a.startswith("--cache-dir="):
            cache_dir = a.split("=", 1)[1]
        elif a.startswith("--url="):
            url = a.split("=", 1)[1]

    result_path = download_dataset(cache_dir=cache_dir, dataset_url=url)
    print(f"\n[download_dataset] Done. Dataset at: {result_path}")
