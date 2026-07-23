# Progress Tracker — Local LLM Agent Benchmark

## Completed Sections ✅

### 1. `agents/__init__.py` ✅ (Section 2)
- Exports: `Agent`, `Reasoner`, `Actioner` from their respective modules.

### 2. `agents/reasoner.py` ✅ (Section 3a)
- **Class**: `Reasoner`
- **Methods**:
  - `__init__(self, model_id: str = "qwen3.5:9b")`: Initializes Ollama client with specified model.
  - `solve(self, reasoning_prompt: str) -> str`: Sends a single-turn request to the LLM and returns its response text.

### 3. `agents/actioner.py` ✅ (Section 3b)
- **Class**: `Actioner`
- **Methods**:
  - `__init__(self, model_id: str = "ornith:9b")`: Initializes Ollama client with specified model.
  - `solve(self, tool_call_str: str) -> dict`: Sends a single-turn request to the LLM and parses the response as a JSON dict containing `"tool"` and `"args"`.

### 4. `agents/agent.py` ✅ (Section 3c)
- **Class**: `Agent`
- **Methods**:
  - `__init__(self, reasoner: Reasoner, actioner: Actioner, max_iterations: int = 50)`: Initializes with reasoner and actioner instances.
  - `solve(self, repo_path: str, problem_statement: str) -> dict`: Iteratively calls reasoner → actioner until the agent reaches its maximum iterations or produces a final patch.

### 5. `agents/ollama_client.py` ✅ (Section 3d)
- **Class**: `OllamaClient`
- **Methods**:
  - `__init__(self, model: str = "qwen2.5:7b")`: Initializes with the Ollama model name.
  - `send_request(self, prompt: str, temperature: float = 0.1) -> dict`: Sends a single-turn request to Ollama and returns the response as a dictionary.

### 6. `benchmarks/swebench.py` ✅ (Section 4a)
- **Class**: `SWEbenchBenchmark`
- **Methods**:
  - `__init__(self, seed_repo_dir: str = "seed_repos", results_dir: str = "results")`: Initializes with directories for seed repos and results storage.
  - `list_tasks(self, dataset_path: str | None = None) -> list[dict]`: Loads tasks from the SWE-bench lite JSONL file.
  - `setup_repo(self, repo_name: str, base_commit: str) -> Path`: Sets up a seed repo at the specified commit (placeholder for real implementation).
  - `run_instance(self, task: dict, agent, reasoner_model: str | None = None, actioner_model: str | None = None) -> InstanceRecord`: Runs one SWE-bench instance through the full pipeline.
  - `_evaluate(self, record: InstanceRecord) -> str`: Evaluates an instance by checking if test results indicate success (placeholder for real evaluation).
  - `save_raw(self, results: list[InstanceRecord]) -> Path`: Persists raw results to `results/raw/`.
  - `save_processed(self, results: list[InstanceRecord]) -> Path`: Aggregates and saves processed summary to `results/processed/`.

### 7. `benchmarks/swebench_lite.py` ✅ (Section 4b)
- **Class**: `SWEbenchLiteBenchmark`
- **Methods**:
  - `__init__(self, seed_repo_dir: str = "seed_repos", results_dir: str = "results")`: Initializes with directories for seed repos and results storage.
  - `list_tasks(self) -> list[dict]`: Loads tasks from the SWE-bench lite JSONL file.

### 8. `configs/qwen_ornith.yaml` ✅ (Section 5a)
- **Config**: Defines agent configuration with reasoner model (`qwen3.5:9b`) and actioner model (`ornith:9b`).

### 9. `configs/single_model.yaml` ✅ (Section 5b)
- **Config**: Defines a single-model agent configuration where both reasoner and actioner use the same model.

---

## Completed Sections ✅

### 10. `experiments/run_experiment.py` ✅ (Section 8.1)
- **Purpose**: Run a complete experiment based on a YAML configuration file.
- **Key features**:
  - Loads configuration from YAML files (e.g., `configs/qwen_ornith.yaml`).
  - Initializes Reasoner, Actioner, and Agent controller from config.
  - Initializes SWE-bench benchmark.
  - Selects instances with optional `--limit` flag for incremental testing.
  - Runs the full benchmark pipeline on selected instances.
  - Collects results and saves raw results to `results/raw/`.
  - Saves aggregated summary (passed, failed, error counts) to `results/processed/`.
- **Usage**:
  ```bash
  # Test with a single instance first
  python experiments/run_experiment.py --config configs/qwen_ornith.yaml --limit 1
  
  # Scale up incrementally
  python experiments/run_experiment.py --config configs/qwen_ornith.yaml --limit 5
  
  # Full benchmark (after stability confirmed)
  python experiments/run_experiment.py --config configs/qwen_ornith.yaml
  ```
