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
    hf_dataset_id: str = "princeton-nlp/SWE-bench_Lite",
    hf_split: str = "test",
) -> Path:
    """Download the SWE-bench Lite dataset and save it locally.

    Parameters
    ----------
    cache_dir :
        Directory where the cached dataset will be stored. Defaults to a
        subdirectory under ``seed_repos/swe_bench_lite/`` in the current
        project root.
    dataset_url :
        Direct URL to a JSONL file. If provided, this takes priority over
        the Hugging Face path (useful for mirrors or pre-converted files).
    hf_dataset_id :
        Hugging Face dataset id to fall back to when no ``dataset_url`` is
        given. Defaults to the official SWE-bench Lite dataset. See
        ``docs/DOWNLOADING_SWEBENCH_LITE.md`` for details and alternatives.
    hf_split :
        Dataset split to load from Hugging Face (SWE-bench Lite only has
        a "test" split).

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

    if dataset_url:
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

    # Default path: pull the official dataset from Hugging Face. Requires
    # `pip install datasets`. See docs/DOWNLOADING_SWEBENCH_LITE.md for
    # alternative methods (huggingface_hub parquet download, official
    # swebench harness) if this dependency isn't available.
    print(f"[download_dataset] No --url given; fetching '{hf_dataset_id}' "
          f"(split={hf_split}) from Hugging Face")
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as e:
        raise RuntimeError(
            "No dataset_url provided and the 'datasets' package isn't "
            "installed. Install it with `pip install datasets "
            "--break-system-packages`, or see "
            "docs/DOWNLOADING_SWEBENCH_LITE.md for alternative download "
            "methods."
        ) from e

    ds = hf_load_dataset(hf_dataset_id, split=hf_split)
    with open(dataset_path, "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")

    print(f"[download_dataset] Wrote {len(ds)} instances to {dataset_path}")
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
