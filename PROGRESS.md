# Progress Tracker — SWE-bench Lite Local LLM Agent Benchmark

**Created:** 2026-07-22  
**Last Updated:** 2026-07-22  
**Source Plan:** [ACTIONPLAN.md](./ACTIONPLAN.md)

---

## Status Legend
| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented (code exists on disk) |
| 📝 | Specified in plan but not yet coded |
| 🔲 | Not started / planned for future |
| ⚠️  | Partially implemented or empty stub |

---

## 1. Project Setup & Documentation

### 1.1 README.md
- **Status:** ✅ Implemented (with content)
- **Notes:** Contains architecture overview, quick start guide, configuration parameters table, data flow diagrams, troubleshooting section, and future work list.

### 1.2 ACTIONPLAN.md
- **Status:** ✅ Implemented (with full specification)
- **Notes:** Complete build plan covering all sections below.

---

## 2. Core Agent Components (`agents/`)

| File | Status | Notes |
|------|--------|-------|
| `agents/__init__.py` | ⚠️ Empty stub | Exists but empty |
| `agents/ollama_client.py` | ⚠️ Empty stub | Specified in plan (Section 4.1). Should implement OllamaClient class for model communication. |
| `agents/reasoner.py` | ⚠️ Empty stub | Specified in plan (Section 4.2). Should implement Reasoner class for task analysis and planning. |
| `agents/actioner.py` | ⚠️ Empty stub | Specified in plan (Section 4.3). Should implement Actioner class with file system tool layer (read, write, edit, delete, list_dir, search_code, run_command, run_tests, get_git_diff). |
| `agents/agent.py` | ⚠️ Empty stub | Specified in plan (Section 4.4). Should implement Agent controller that coordinates reasoner + actioner loop with iteration tracking and configurable max_iterations. |

**Summary:** All 5 agent files exist as empty Python modules, but no implementation has been written yet. The architecture is fully specified in the ACTIONPLAN.

---

## 3. Benchmark Infrastructure (`swebench/`)

| File | Status | Notes |
|------|--------|-------|
| `swebench/__init__.py` | 🔲 Not started | Specified in plan (Section 5) |
| `swebench/download_dataset.py` | 🔲 Not started | Should download and cache SWE-bench Lite dataset to `seed_repos/swe_bench_lite/dataset.json`. |
| `swebench/setup_repos.py` | 🔲 Not started | Should handle repo cloning, checkout of base commits, workspace isolation. |
| `swebench/run_instance.py` | 🔲 Not started | Should run a single SWE-bench instance end-to-end. CLI interface: `python swebench/run_instance.py --instance <id> --config <path>` |
| `swebench/evaluate.py` | 🔲 Not started | Should evaluate agent results (resolved/not resolved/test failure, FAIL_TO_PASS/PASS_TO_PASS tracking). |
| `swebench/utils.py` | 🔲 Not started | Shared utilities: load dataset, find repo, run subprocess, apply patch, reset repo. |

**Summary:** The entire SWE-bench benchmark infrastructure is specified but not yet implemented. This is a significant chunk of work.

---

## 4. Configurations (`configs/`)

| File | Status | Notes |
|------|--------|-------|
| `configs/__init__.py` | ⚠️ Empty stub | Exists but empty |
| `configs/agent_config.py` | ⚠️ Empty stub | Exists but empty. Plan specifies this should hold tunable parameters (reasoner_model, actioner_model, max_iterations, ollama_base_url). |
| `configs/qwen_ornith.yaml` | 🔲 Not started | Primary experiment: Qwen 3.5 9B reasoner + Ornith 9B actioner. |
| `configs/qwen_deepseek.yaml` | 🔲 Not started | Comparison experiment: Qwen + DeepSeek. |
| `configs/single_model.yaml` | 🔲 Not started | Baseline: single model for both reasoning and action. |

**Summary:** Only the Python config module exists as empty stubs. The YAML configuration files have not been created yet.

---

## 5. Benchmark Abstraction (`benchmarks/`)

| File | Status | Notes |
|------|--------|-------|
| `benchmarks/__init__.py` | ⚠️ Empty stub | Exists but empty |
| `benchmarks/swebench.py` | ⚠️ Empty stub | Specified in plan (Section 6.1). Should expose SWE-bench Lite as a benchmark object with `.run()`, instance filtering, and limits support. |

**Summary:** Benchmark abstraction is specified but not yet implemented. This is the layer that connects the Agent to the SWE-bench dataset.

---

## 6. Experiments (`experiments/`)

| File | Status | Notes |
|------|--------|-------|
| `experiments/__init__.py` | 🔲 Not started | Specified in plan (Section 8) |
| `experiments/run_experiment.py` | 🔲 Not started | Should run a complete experiment from config. CLI: `python experiments/run_experiment.py --config configs/qwen_ornith.yaml [--limit N]`. |

**Summary:** No experiment framework exists yet. This is the entry point for running benchmarks end-to-end.

---

## 7. Analysis & Visualization (`analysis/`)

| File | Status | Notes |
|------|--------|-------|
| `analysis/analyze_results.py` | 🔲 Not started | Should compute resolution rate, avg runtime, iterations, tool calls, failure categories, comparison tables. |
| `analysis/visualize_results.py` | 🔲 Not started | Specified in plan (Section 10.2). Should create charts/graphs from processed results. |

**Summary:** Analysis tools are listed as future work and not yet started.

---

## 8. Results Storage (`results/`)

| Directory/File | Status | Notes |
|----------------|--------|-------|
| `results/raw/` | 🔲 Not started | Should store per-task results (result.json, agent.log, test.log, patch.diff) organized by config and instance. |
| `results/processed/` | 🔲 Not started | Should store aggregated summaries (qwen_ornith_summary.json, etc.). |

**Summary:** Results directories exist but are empty. No result schema or saving logic has been implemented yet.

---

## 9. Seed Repositories (`seed_repos/`)

| Item | Status | Notes |
|------|--------|-------|
| `seed_repos/swe_bench_lite/` | 🔲 Not started | Plan states this should be cached/downloaded once and reused. |
| `seed_repos/swe_bench_lite/dataset.json` | 🔲 Not started | Dataset file to be populated by download_dataset.py. |
| `seed_repos/swe_bench_lite/repos/` | 🔲 Not started | Repository cache directory for individual repos. |

**Summary:** Seed repo infrastructure not yet set up. This is expected to happen after the dataset module (Section 5.1) is implemented.

---

## Progress Summary

| Category | Files Specified | Files Implemented | % Complete |
|----------|----------------|-------------------|------------|
| Documentation | 2 | 2 | ✅ 100% |
| Agent Components | 4 (plus __init__) | 5 empty stubs | 📝 ~0% implementation |
| Benchmark Infra (`swebench/`) | 6 | 0 | 🔲 0% |
| Configurations | 3 YAML + 2 Python | 2 empty stubs | 📝 ~17% |
| Benchmarks Abstraction | 2 (plus __init__) | 2 empty stubs | 📝 ~0% implementation |
| Experiments | 2 | 0 | 🔲 0% |
| Analysis & Visualization | 2 | 0 | 🔲 0% |
| Results Storage | N/A (dirs) | 0 | 🔲 0% |

**Overall Implementation: ~5-10%** — The project skeleton and documentation are in place, but the core logic for all agent components, benchmark infrastructure, experiment runner, analysis tools, and configuration files has not yet been written.

---

## Recommended Next Steps (Priority Order)

1. **Implement `agents/ollama_client.py`** — Foundation for all LLM communication
2. **Define result schema & create `results/raw/`, `results/processed/` directories** — Needed before anything else can record results
3. **Implement `swebench/download_dataset.py` and `swebench/utils.py`** — Get the dataset flowing
4. **Implement `agents/reasoner.py`** — Core reasoning component
5. **Implement `agents/actioner.py`** — Core action-taking component with tool layer
6. **Implement `configs/qwen_ornith.yaml`, `qwen_deepseek.yaml`, `single_model.yaml`** — Configuration files
7. **Implement `benchmarks/swebench.py`** — Benchmark abstraction layer
8. **Implement `agents/agent.py`** — Agent controller that ties reasoner + actioner together
9. **Implement `swebench/setup_repos.py` and `run_instance.py`** — Repo management and instance execution
10. **Implement `experiments/run_experiment.py`** — End-to-end experiment runner
11. **Implement `analysis/analyze_results.py` and `visualize_results.py`** — Post-hoc analysis
