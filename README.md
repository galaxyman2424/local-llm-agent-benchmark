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
