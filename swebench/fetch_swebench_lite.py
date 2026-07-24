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