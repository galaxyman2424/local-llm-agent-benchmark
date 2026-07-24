# Downloading SWE-bench Lite

This project's `swebench/download_dataset.py`, `benchmarks/swebench.py`, and
`benchmarks/swebench_lite.py` all expect a local `dataset.jsonl` (or
`instances.jsonl`) file containing SWE-bench Lite task instances. This guide
covers how to actually get that file, since the dataset isn't hosted as a
plain JSONL download anywhere — it lives on Hugging Face as a dataset with
parquet files.

The official dataset is **`princeton-nlp/SWE-bench_Lite`**, 300 Issue/PR
pairs from 11 popular Python repositories:
https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite

There are three practical ways to get it locally. Option 1 is the easiest
and is what this project's scripts expect.

---

## Do you need a Python environment for this?

Yes. `datasets` (the Hugging Face library) is not part of Python's standard
library, so you need a working Python installation with `pip` to install it.
Two common situations:

- **Plain Python already installed** (most systems): just run
  `pip install datasets`. If you hit an "externally-managed-environment"
  error (common on newer Debian/Ubuntu, e.g. 23.04+), either add
  `--break-system-packages` to the pip command, or use a virtual
  environment instead (recommended, keeps things isolated):
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate      # on Windows: .venv\Scripts\activate
  pip install datasets pytest
  ```
- **No Python at all yet**: install Python 3.9+ from
  [python.org](https://www.python.org/downloads/) (or via your OS package
  manager / `pyenv`/`conda`), then follow the venv steps above.

You'll also need `pytest` installed in that same environment once you get to
actually running the agent against instances, since the FAIL_TO_PASS/
PASS_TO_PASS evaluation (see `swebench/utils.py::evaluate_fail_to_pass`)
runs tests via `python -m pytest` directly.

---

## Option 1: `datasets` library → JSONL (recommended)

This is the most reliable method since it doesn't depend on guessing an
exact parquet filename/URL that could change.

```bash
pip install datasets --break-system-packages   # if not already installed
```

```python
# save as: swebench/fetch_swebench_lite.py  (or run inline)
from datasets import load_dataset
import json
from pathlib import Path

out_dir = Path("seed_repos/swe_bench_lite")
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

out_path = out_dir / "dataset.jsonl"
with open(out_path, "w") as f:
    for row in ds:
        f.write(json.dumps(row) + "\n")

print(f"Wrote {len(ds)} instances to {out_path}")
```

Run it:

```bash
python swebench/fetch_swebench_lite.py
```

This produces `seed_repos/swe_bench_lite/dataset.jsonl`, exactly where
`swebench/utils.py::load_dataset()` and `benchmarks/swebench.py::list_tasks()`
look for it by default.

Each row/instance includes (per the official schema):

| Field | Description |
|---|---|
| `instance_id` | e.g. `django__django-11099` |
| `repo` | `owner/repo` on GitHub, e.g. `django/django` |
| `base_commit` | commit hash to check out before the agent runs |
| `problem_statement` | the GitHub issue text — this is the task |
| `patch` | the **gold** patch that resolved the issue (don't show this to the agent — it's the answer key, used only for evaluation) |
| `test_patch` | patch that adds/modifies the relevant tests |
| `FAIL_TO_PASS` | test names that should go from failing → passing |
| `PASS_TO_PASS` | test names that must keep passing (regression check) |

Note this project's code currently reads a `task_description` field (falling
back to `problem_statement` after the fixes applied to `agent.py`/
`benchmarks/swebench.py`/`benchmarks/swebench_lite.py`); the real dataset
only has `problem_statement`, so the fallback is what actually gets used —
you don't need to rename any fields.

---

## Option 2: `huggingface_hub` direct parquet download

If you don't want the full `datasets` dependency:

```bash
pip install huggingface_hub pandas pyarrow --break-system-packages
```

```python
from huggingface_hub import hf_hub_download
import pandas as pd
import json
from pathlib import Path

# List exact files first if this path 404s — HF auto-converts parquet
# filenames occasionally; check https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/tree/main/data
local_file = hf_hub_download(
    repo_id="princeton-nlp/SWE-bench_Lite",
    filename="data/test-00000-of-00001.parquet",
    repo_type="dataset",
)

df = pd.read_parquet(local_file)
out_path = Path("seed_repos/swe_bench_lite/dataset.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_json(out_path, orient="records", lines=True)
print(f"Wrote {len(df)} instances to {out_path}")
```

## Option 3: Official SWE-bench harness

The SWE-bench authors maintain a full harness (dataset loading, Docker-based
execution environments, and the official evaluation logic) at
https://github.com/SWE-bench/SWE-bench.

```bash
pip install swebench --break-system-packages
```

```python
from swebench.harness.utils import load_swebench_dataset
instances = load_swebench_dataset("princeton-nlp/SWE-bench_Lite", split="test")
```

Use this route if you eventually want the "real" evaluation semantics
(Docker containers per instance, official FAIL_TO_PASS/PASS_TO_PASS
checking) rather than this project's lightweight `pytest`-in-place approach.
It's heavier to set up (requires Docker) but is the ground-truth reference
implementation this project is inspired by.

---

## Wiring it into this project

Once `seed_repos/swe_bench_lite/dataset.jsonl` exists:

```bash
# Sanity check it loads
python -c "from swebench.utils import load_dataset; d = load_dataset('seed_repos/swe_bench_lite/dataset.jsonl'); print(len(d), 'instances loaded')"

# You'll also need pytest to run FAIL_TO_PASS/PASS_TO_PASS evaluation
pip install pytest --break-system-packages

# Clone + checkout the repos for instances as you run them
# (benchmarks/swebench.py::setup_repo does this automatically per-instance
# now, cloning from https://github.com/<repo>.git and checking out base_commit)

# Run a quick single-instance smoke test
python swebench/run_instance.py --instance django__django-11099 --config configs/qwen_ornith.yaml

# Or run a small batch through the experiment runner
python experiments/run_experiment.py --config configs/qwen_ornith.yaml --limit 3
```

**Disk space note:** cloning the full repos for all 11 projects (Django,
Flask, scikit-learn, sympy, etc.) across all 300 instances can add up to
several GB, especially if the benchmark checks out many different commits
per repo. `setup_repo()` clones each repo once and re-checks-out commits
in place, so you pay the clone cost once per repo rather than once per
instance.
