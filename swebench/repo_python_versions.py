# swebench/repo_python_versions.py
"""
Per-repo (or per-instance) Python version mapping for Docker base images.

SWE-bench Lite repos were originally tested against a range of Python
versions depending on repo era. A single shared base image can't satisfy
all of them -- astropy__astropy needs pre-3.11 (longintrepr.h moved
internal in 3.11); other repos may have their own constraints in the
opposite direction (needing a newer Python for a language feature, or a
newer setuptools that dropped old APIs).

Populate this from whatever the dataset actually provides first --
the datasets `princeton-nlp/SWE-bench_Lite` and similar carry a
`version` field per instance, and the upstream SWE-bench harness ships
a MAP_REPO_VERSION_TO_SPECS-style table with the exact Python pin per
repo/version. Prefer pulling from that if you have the harness installed
or the dataset loaded, rather than hand-maintaining this table --
treat the dict below as a fallback / starting point, not a verified
source of truth.
"""

from __future__ import annotations

# repo_name -> default python version, used when no more specific
# instance-level override applies. Verify/replace with authoritative
# values from the dataset or upstream harness before relying on this.
DEFAULT_PYTHON_VERSION_BY_REPO: dict[str, str] = {
    "astropy/astropy": "3.9",
    "psf/requests": "3.9",
    # ... fill in for the rest of your repo set
}

FALLBACK_PYTHON_VERSION = "3.9"


def get_python_version(repo_name: str, instance_id: str | None = None,
                        dataset_row: dict | None = None) -> str:
    """
    Resolve the Python version to use for this repo's Docker image.

    Priority:
      1. Explicit field on the dataset row, if you have one loaded
         (e.g. row.get("python_version") or row.get("version")-derived).
      2. Per-repo default table above.
      3. Global fallback.
    """
    if dataset_row is not None:
        pv = dataset_row.get("python_version")
        if pv:
            return str(pv)

    return DEFAULT_PYTHON_VERSION_BY_REPO.get(repo_name, FALLBACK_PYTHON_VERSION)