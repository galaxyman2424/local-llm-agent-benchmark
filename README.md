# Local LLM Agent Benchmark

A benchmark framework for evaluating local language models as autonomous software agents, inspired by SWE-bench methodology. This project tests whether smaller, locally-run models can solve real-world coding tasks through iterative reasoning and action-taking cycles.

## Architecture Overview

The system follows a clean separation of concerns:

```
┌─────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    Reasoner (LLM)   │────▶│ Actioner (Executor)│◀──│    Ollama Client  │
│  "What should I do?" │     │ "How do I do it?" │     │  Local Model API  │
└─────────────────────┘     └──────────────────┘     └──────────────────┘
       ▲                              │
       │         ┌───────────────────┘
       │         ▼
┌─────────────────────┐     ┌──────────────────┐
│   Agent (Orchestrator)│◀──│  SWE-bench Benchmark│
│ Loops reasoner +      │     │ Dataset / Setup /  │
│ actioner until done   │     │ Evaluation        │
└─────────────────────┘     └──────────────────┘
```

## Core Principles

1. **Separate Agent from Benchmark**: The agent architecture is independent of SWE-bench specifics, allowing it to be tested on other benchmarks later.

2. **Separate Reasoning from Action**: The reasoner determines what should change; the actioner executes those changes in the filesystem and shell.

3. **Preserve Reproducibility**: Every instance records its full execution trace for offline analysis.

## Quick Start

### Prerequisites

- Python 3.10+
- Ollama installed with at least `qwen2.5:7b` model loaded
- A seed repository (e.g., SWE-bench lite) cloned into `seed_repos/swe_bench_lite/`

### Installation

```bash
cd local-llm-agent-benchmark
pip install -r requirements.txt  # if needed in the future
```

### Running a Benchmark Run

```python
from configs.agent_config import AgentConfig
from agents.ollama_client import OllamaClient
from agents.reasoner import Reasoner
from agents.actioner import Actioner
from agents.agent import Agent
from benchmarks.swebench import SWEbenchBenchmark

# Configure the agent pipeline
config = AgentConfig(
    reasoner_model="qwen2.5:7b",
    max_iterations=50,
)

# Initialize components
client = OllamaClient(base_url=config.ollama_base_url)
reasoner = Reasoner(model_name=config.reasoner_model)
actioner = Actioner()
agent = Agent(reasoner, actioner, max_iterations=config.max_iterations)

# Run the benchmark
benchmark = SWEbenchBenchmark(
    seed_repo_dir="seed_repos",
    results_dir="results",
)

tasks = benchmark.list_tasks()
results = []
for task in tasks:
    result = benchmark.run_instance(task, agent)
    results.append(result)

# Save raw and processed results
benchmark.save_raw(results)
benchmark.save_processed(results)
```

## Project Structure

```
local-llm-agent-benchmark/
├── agents/                  # Core agent components
│   ├── __init__.py         # Package exports
│   ├── ollama_client.py    # Ollama HTTP API wrapper
│   ├── reasoner.py         # Task interpretation & planning
│   ├── actioner.py         # Filesystem & shell execution
│   └── agent.py            # Orchestration loop
├── benchmarks/             # Benchmark definitions & runners
│   ├── __init__.py         # Package exports
│   └── swebench.py         # SWE-bench implementation
├── configs/                # Configuration classes
│   ├── __init__.py         # Package exports
│   └── agent_config.py     # Tunable parameters
├── analysis/               # Analysis scripts (future)
├── experiments/            # Experiment tracking (future)
├── results/                # Output storage
│   ├── raw/                # Per-task execution traces
│   └── processed/          # Aggregated summaries
├── seed_repos/             # Seed repositories for cloning
│   └── swe_bench_lite/     # SWE-bench lite dataset + repos
├── swebench/               # Original SWE-bench reference (future)
├── ACTIONPLAN.md           # Implementation plan
└── TEST.md                 # Test status document
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reasoner_model` | `"qwen2.5:7b"` | Model used for planning/reasoning |
| `actioner_model` | (same as reasoner) | Model used for action decisions |
| `max_iterations` | 50 | Maximum reasoning-action cycles per task |
| `ollama_base_url` | `"http://localhost:11434"` | Ollama server address |
| `timeout_seconds` | 60 | Per-request timeout in seconds |

## Data Flow

### Reasoner → Actioner Loop

Each iteration follows this pattern:

1. **Context Assembly**: The agent collects current state (file contents, error messages, git diff) and passes it to the reasoner.
2. **Reasoning**: The LLM analyzes context and outputs a JSON plan with action types (`read`, `edit`, `execute`, `test`).
3. **Execution**: The actioner carries out each step — reading files, writing patches, running commands.
4. **State Update**: Results are fed back into the context for the next iteration.

### Evaluation Flow

After the agent runs out of iterations:

1. Tests are executed on the modified repository.
2. Final patch is compared against expected solution (pass@1).
3. Raw results are saved to `results/raw/results.jsonl`.
4. Processed summary with pass rate is saved to `results/processed/summary.json`.

## Extending This Framework

### Adding a New Benchmark

Create a new module in `benchmarks/` that inherits from or mirrors the SWEbenchBenchmark interface:

```python
from benchmarks.swebench import InstanceRecord

class MyBenchmark:
    def list_tasks(self) -> list[dict]: ...
    def setup_repo(self, repo_name, base_commit) -> Path: ...
    def run_instance(self, task, agent): ...
    def _evaluate(self, record) -> str: ...
    def save_raw(self, results): ...
    def save_processed(self, results): ...
```

### Adding a New Agent Component

Extend the existing components or create new ones in `agents/`. The key interfaces are:

- **OllamaClient**: Handles all LLM communication.
- **Reasoner**: Takes context dict → returns list of PlanSteps.
- **Actioner**: Provides methods for file read/write, command execution, patch application.
- **Agent**: Orchestrates the loop with configurable max iterations.

## Troubleshooting

### Ollama not responding

Ensure Ollama is running:
```bash
ollama serve  # or check your system's Ollama service
```

Verify the model is loaded:
```bash
curl http://localhost:11434/api/tags
```

If the model isn't downloaded, pull it first:
```bash
ollama pull qwen2.5:7b
```

### No seed repository found

Clone SWE-bench lite into `seed_repos/swe_bench_lite/`:
```bash
git clone <swe-bench-lite-url> seed_repos/swe_bench_lite/
```

## Future Work

- [ ] Integrate with real SWE-bench evaluation harness
- [ ] Add support for multiple model backends (OpenAI API, vLLM, etc.)
- [ ] Implement proper patch-based reasoning with unified diffs
- [ ] Add experiment tracking and comparison across configurations
- [ ] Create visualization tools for execution traces
- [ ] Build analysis scripts for statistical evaluation

## License

This project is for research purposes. See individual files for specific licensing terms where applicable.


Don't guess blind — use the logging that's already there. reasoner.py already prints prompt_eval_count and eval_count on every call (lines ~254-257, ~271-282). Run one task with these new values and watch those numbers to confirm you're nowhere near num_ctx before committing to a benchmark run.
The actioner's prompt is tiny by comparison (just the schema block + the reasoner's chosen action — large values get stripped via _extract_large_values before it even sees them), so ornith:9b's 128k window is mostly headroom there; no analogous constant to raise.
Watch VRAM. Both models loaded simultaneously with KV cache sized for 128k each (vs. today's 16k) is a real jump — keep ollama ps / nvidia-smi open on your first run in case you need to scale back.


1. Find out where it's actually failing (do this first)

Right now nothing aggregates why a run didn't resolve — just pass/fail. Add a quick counter across a batch of previous_actions records (or grep your logs) for:

"Rejected invalid action" (Actioner produced schema-invalid JSON)
"SEARCH/REPLACE failed. Text not found" (from agents/actioner.py's execute)
"Reasoner reply contained no JSON object" / repaired-JSON events
Reasoner saying "done" before any run_tests action ever happened

My strong guess, given this architecture (small local models, JSON-mode, two-model handoff): replace_in_file's exact-substring match is your biggest silent killer. A 7-9B model asked to reproduce an exact multi-line search string (correct whitespace, correct indentation, matching the actual file content it may have only seen truncated) will very often be a few characters off, and content.replace(search_text, replace_text) in actioner.py fails with zero partial credit — the whole edit is silently discarded and the loop burns an iteration.

2. Make editing more forgiving (highest leverage fix)

Two options, in order of effort:

Cheap: normalize whitespace before comparing — strip trailing whitespace per line, or do a fuzzy/near-match fallback (e.g. try matching ignoring leading indentation, then re-indent the replacement) before giving up.

Better: have the Actioner emit the search text as line-anchored (e.g. "replace lines containing X through Y") rather than an exact block, or switch to unified-diff application (git apply) for edits instead of raw substring replace — diffs tolerate context better and you already have apply_patch in swebench/utils.py you could reuse for this instead of only using it for test_patch.

Better option (diff-based) — same call site, but instead of string .replace(), build a unified diff from search_text/replace_text and call the existing apply_patch from swebench/utils.py:

python
elif tool_name in ("replace_in_file", "edit_file"):
    path = self._resolve_path(params.get("path", ""))
    from swebench.utils import apply_patch
    import difflib
    with open(path, 'r') as f:
        content = f.read()
    search_text = params.get("search", "")
    replace_text = params.get("replace", "")
    ...
    # build a diff between content-with-search and content-with-replace,
    # then apply_patch(self.workspace_dir, diff_text) with git apply's
    # fuzzier context matching instead of exact substring replace

This is more invasive (you'd restructure the whole branch), so I'd start with the cheap fuzzy fallback above and only move to diff-based if that's still not tolerant enough.


3. Give the Reasoner a starting foothold instead of total blindness

Right now Reasoner.plan's first call has previous_actions = [] and current_state with no file tree, no README, nothing — it's guessing what to search_code for based purely on the issue text. For real repos (django, astropy, sympy) that's a lot of wasted iterations just orienting. Consider seeding current_state in Agent.solve with a shallow list_directory result or git ls-files | head -50 before the loop starts, so iteration 1 isn't spent discovering the repo has a src/ layout.

4. Check iteration budget vs. actual task complexity

configs/single_model.yaml and qwen_deepseek.yaml set max_iterations: 10 — that's search → read → edit → run_tests → (probably fail once) → re-read → re-edit → run_tests, which is tight for a 9B model that isn't going to nail it in one edit. qwen_ornith.yaml gives 50, which is more realistic. If your low-iteration configs are the ones you're checking pass rates on, bump them — 10 is likely starving the agent, not testing the model fairly.

5. Watch for premature "done"

STOP_ACTIONS = ("done", "submit_solution") lets the Reasoner bail without ever running tests. If your failure logs show "done" frequently with no preceding run_tests in that instance's history, the Reasoner is hallucinating completion. Fix: in the Reasoner's prompt, explicitly disallow choosing "done" unless run_tests appears with returncode == 0 somewhere in previous_actions — that's a prompt-level guard you can add in reasoner.py's plan() prompt text, no architecture change needed.

6. num_predict=512 in the Actioner may be truncating write_to_file

agents/actioner.py's plan_action hardcodes num_predict=512 for every action, including write_to_file with a full new file's content. If any instance needs a new file longer than ~512 tokens, that call will truncate mid-content (the repair-JSON fallback will "succeed" but hand back a broken/incomplete file). Consider scaling num_predict up specifically when reasoner_plan.get("next_action") == "write_to_file".