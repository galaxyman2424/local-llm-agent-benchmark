# SWE-bench Lite Local LLM Agent Benchmark — Build Plan

## 1. Project Purpose

This project is designed to evaluate local Large Language Model (LLM) agents on software engineering tasks using SWE-bench Lite.

The primary experimental architecture separates reasoning from action:

```text
                         SWE-bench Issue
                               │
                               ▼
                     ┌───────────────────┐
                     │     Reasoner      │
                     │    Qwen 3.5 9B   │
                     └─────────┬─────────┘
                               │
                         Analysis / Plan
                               │
                               ▼
                     ┌───────────────────┐
                     │  Agent Controller │
                     └─────────┬─────────┘
                               │
                         Action Request
                               │
                               ▼
                     ┌───────────────────┐
                     │     Actioner      │
                     │     Ornith 9B     │
                     └─────────┬─────────┘
                               │
                         Tool Execution
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
         Read Files       Edit Files       Run Tests
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                       Repository Changes
                               │
                               ▼
                          Test Results
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
               PASS                         FAIL
                 │                           │
                 ▼                           ▼
               Finish                Return to Reasoner
                                             │
                                             ▼
                                       New Plan / Fix
```

The primary experiment is:

```text
Reasoner: Qwen 3.5 9B
Actioner: Ornith 9B
Benchmark: SWE-bench Lite
```

The architecture should also support alternative configurations so that the performance of the two-model system can be compared against other approaches.

---

# 2. Complete Project Structure

The target project structure is:

```text
modelResearch/local-llm-agent-benchmark/
│
├── README.md
├── ACTIONPLAN.md
├── requirements.txt
│
├── swebench/
│   ├── __init__.py
│   ├── download_dataset.py
│   ├── setup_repos.py
│   ├── run_instance.py
│   ├── evaluate.py
│   └── utils.py
│
├── agents/
│   ├── __init__.py
│   ├── ollama_client.py
│   ├── reasoner.py
│   ├── actioner.py
│   └── agent.py
│
├── configs/
│   ├── qwen_ornith.yaml
│   ├── qwen_deepseek.yaml
│   └── single_model.yaml
│
├── benchmarks/
│   ├── __init__.py
│   └── swebench_lite.py
│
├── experiments/
│   ├── __init__.py
│   └── run_experiment.py
│
├── results/
│   ├── raw/
│   └── processed/
│
├── analysis/
│   ├── analyze_results.py
│   └── visualize_results.py
│
└── seed_repos/
    └── swe_bench_lite/
        ├── dataset.json
        └── repos/
```

The `seed_repos/` directory contains the locally cached SWE-bench Lite data and repositories.

The dataset and repositories should be downloaded or cloned once and then reused whenever possible.

Do not repeatedly download the same repositories for every benchmark instance.

---

# 3. Implementation Principles

The implementation must follow these principles.

## 3.1 Separate the Agent From the Benchmark

The agent should not contain SWE-bench-specific logic.

The agent should be capable of receiving:

```text
Repository
Task / Issue
Available Tools
```

and attempting to solve the task.

The SWE-bench benchmark should be responsible for:

```text
Dataset
Repository setup
Base commit
Test execution
Evaluation
Result storage
```

This separation allows the same agent architecture to be tested on other benchmarks later.

---

## 3.2 Separate Reasoning From Action

The Reasoner should determine:

```text
What is the problem?
What files are relevant?
What is the likely cause?
What should be changed?
What should be tested?
How should a failure be debugged?
```

The Actioner should determine:

```text
How do I inspect the file?
How do I edit the file?
How do I execute the command?
How do I run the tests?
```

The Reasoner should not directly modify the repository unless explicitly required by the architecture.

The Actioner should execute the Reasoner's plan.

---

## 3.3 Preserve Reproducibility

Every SWE-bench instance must begin from the correct `base_commit`.

The benchmark should record:

```text
Instance ID
Repository
Base Commit
Agent Configuration
Reasoner Model
Actioner Model
Start Time
End Time
Runtime
Number of Iterations
Number of Tool Calls
Test Results
Final Patch
Final Status
```

The raw result of each task should never be overwritten.

---

# 4. `agents/` Directory

The `agents/` directory contains the local LLM agent implementation.

---

## 4.1 `agents/ollama_client.py`

### Purpose

This file provides the low-level interface between Python and the local Ollama server.

It should be responsible for:

* Connecting to Ollama.
* Sending prompts.
* Sending chat messages.
* Receiving model responses.
* Supporting streaming if needed.
* Handling request timeouts.
* Handling connection errors.
* Recording latency.
* Recording token usage when available.
* Returning structured responses.

The client should not know anything about SWE-bench.

It should not know:

```text
Repository structure
Test patches
SWE-bench instances
```

Its only responsibility is communicating with local models.

Conceptually:

```python
class OllamaClient:

    def __init__(self, model, base_url):
        pass

    def generate(self, prompt):
        pass

    def chat(self, messages):
        pass
```

The exact implementation may differ.

---

## 4.2 `agents/reasoner.py`

### Purpose

This file implements the Reasoner model.

The primary Reasoner is:

```text
Qwen 3.5 9B
```

The Reasoner should analyze the SWE-bench issue and create a plan for the Actioner.

The Reasoner should be capable of:

1. Understanding the problem statement.
2. Identifying relevant repository areas.
3. Inspecting repository information through available tools or context.
4. Forming a hypothesis about the bug.
5. Creating an implementation plan.
6. Identifying files that should be modified.
7. Identifying tests that should be run.
8. Analyzing failed tests.
9. Creating a revised plan after a failed attempt.

The Reasoner's output should be structured whenever possible.

Example:

```json
{
    "diagnosis": "The filtering condition is incorrect when ...",
    "files_to_inspect": [
        "django/db/models/query.py"
    ],
    "plan": [
        "Inspect the existing filtering implementation",
        "Trace the call path",
        "Identify the incorrect condition",
        "Modify the condition",
        "Run the relevant regression test"
    ],
    "tests_to_run": [
        "path/to/test.py::TestCase::test_name"
    ]
}
```

The exact schema may be adjusted during implementation.

---

## 4.3 `agents/actioner.py`

### Purpose

This file implements the Actioner model.

The primary Actioner is:

```text
Ornith 9B
```

The Actioner is responsible for carrying out the Reasoner's plan.

The Actioner should have access to tools for:

```text
Read file
Write file
Edit file
Delete file
List directory
Search repository
Search code
Run terminal commands
Run tests
Inspect command output
Inspect git diff
```

The Actioner should be able to interact with the repository through a controlled tool layer.

Conceptually:

```text
Ornith 9B
    │
    ├── read_file()
    ├── write_file()
    ├── edit_file()
    ├── delete_file()
    ├── list_directory()
    ├── search_code()
    ├── run_command()
    ├── run_tests()
    └── get_git_diff()
```

The Actioner should report what it did.

Example:

```json
{
    "action": "edit_file",
    "file": "django/db/models/query.py",
    "success": true,
    "description": "Updated filtering condition"
}
```

The Actioner should not independently replace the Reasoner's overall strategy unless it detects that the plan is impossible or incorrect.

---

## 4.4 `agents/agent.py`

### Purpose

This file is the central agent controller.

It coordinates:

```text
Reasoner
Actioner
Tools
Tests
Repository
```

The primary execution loop should be:

```text
Receive task
    │
    ▼
Reasoner analyzes task
    │
    ▼
Reasoner creates plan
    │
    ▼
Actioner executes plan
    │
    ▼
Run tests
    │
    ├── PASS ──► Finish
    │
    └── FAIL
          │
          ▼
    Send failure information
    to Reasoner
          │
          ▼
    Reasoner creates revised plan
          │
          ▼
    Actioner executes fix
          │
          └──────────────► Repeat
```

The controller should track:

```text
Iteration count
Reasoner calls
Actioner calls
Tool calls
Test runs
Elapsed time
Token usage
Final status
```

The maximum number of iterations must be configurable.

Example:

```yaml
agent:
  max_iterations: 10
```

The agent must terminate when the maximum number of iterations is reached.

---

# 5. `swebench/` Directory

This directory manages the SWE-bench Lite benchmark infrastructure.

The SWE-bench infrastructure should be independent of the specific Reasoner and Actioner models.

---

## 5.1 `swebench/download_dataset.py`

### Purpose

Download and cache the SWE-bench Lite dataset.

This should generally be run once.

It should:

1. Download the benchmark metadata.
2. Save the dataset locally.
3. Create the required directory structure.
4. Avoid unnecessary repeated downloads.

Expected location:

```text
seed_repos/
└── swe_bench_lite/
    └── dataset.json
```

This script should not execute agents.

This script should not evaluate tasks.

Its responsibility is dataset acquisition and caching.

---

## 5.2 `swebench/setup_repos.py`

### Purpose

Prepare repositories for SWE-bench instances.

It should be responsible for:

* Locating cached repositories.
* Cloning missing repositories.
* Checking out required commits.
* Creating isolated workspaces.
* Resetting repositories.
* Ensuring clean working trees.

The basic lifecycle is:

```text
Cached Repository
       │
       ▼
Select Repository
       │
       ▼
Checkout Required Base Commit
       │
       ▼
Create Isolated Workspace
       │
       ▼
Agent Receives Workspace
```

Every task must begin from the exact base commit specified by the SWE-bench instance.

Do not allow one task's modifications to contaminate another task.

---

## 5.3 `swebench/run_instance.py`

### Purpose

Run a single SWE-bench Lite instance.

Example usage:

```bash
python swebench/run_instance.py \
    --instance django__django-12345 \
    --config configs/qwen_ornith.yaml
```

The script should:

1. Load the SWE-bench instance.
2. Prepare the repository.
3. Checkout the required base commit.
4. Initialize the agent.
5. Provide the problem statement to the agent.
6. Allow the agent to modify the repository.
7. Run the evaluation.
8. Capture the final patch.
9. Save the result.

The flow is:

```text
SWE-bench Instance
        │
        ▼
Repository Setup
        │
        ▼
Agent Initialization
        │
        ▼
Problem Statement
        │
        ▼
Agent Execution
        │
        ▼
Repository Changes
        │
        ▼
Evaluation
        │
        ▼
Result
```

---

## 5.4 `swebench/evaluate.py`

### Purpose

Evaluate whether the agent successfully solved the SWE-bench issue.

The implementation should use the official SWE-bench evaluation methodology where possible.

The evaluator should:

1. Capture the agent's final repository state.
2. Generate the final patch.
3. Apply or evaluate the patch using the appropriate SWE-bench process.
4. Run the required tests.
5. Determine whether the issue was resolved.
6. Record test results.

The evaluation should distinguish between:

```text
Resolved
Not Resolved
Test Failure
Patch Failure
Timeout
Agent Error
Environment Error
```

The evaluator should separately track relevant:

```text
FAIL_TO_PASS
PASS_TO_PASS
```

tests where applicable.

---

## 5.5 `swebench/utils.py`

### Purpose

Provide shared SWE-bench utility functions.

Potential utilities include:

```text
Load dataset
Load instance
Find repository
Create workspace
Run subprocess
Run command
Apply patch
Get git diff
Reset repository
Save logs
```

Common functionality should be centralized here instead of duplicated throughout the benchmark code.

---

# 6. `benchmarks/` Directory

This directory provides a standard benchmark interface.

---

## 6.1 `benchmarks/swebench_lite.py`

### Purpose

Represent SWE-bench Lite as a benchmark that can be called by the experiment framework.

It should manage:

```text
Dataset
Instance Selection
Agent Execution
Evaluation
Result Collection
```

Conceptually:

```python
benchmark = SWEBenchLite(...)

benchmark.run(
    agent=agent,
    instances=instances
)
```

The benchmark interface should support:

```text
Run one instance
Run a subset of instances
Run the entire benchmark
```

It should also support instance filtering and limits.

Examples:

```text
--instance
--limit
--repository
```

---

# 7. `configs/` Directory

The configuration files define experiments.

---

## 7.1 `configs/qwen_ornith.yaml`

This is the primary experiment.

Configuration:

```text
Reasoner:
    Qwen 3.5 9B

Actioner:
    Ornith 9B
```

Example configuration:

```yaml
name: qwen_ornith

reasoner:
  model: qwen3.5:9b

actioner:
  model: ornith:9b

agent:
  max_iterations: 10
  timeout: 600

benchmark:
  name: swebench_lite
```

The exact Ollama model names must match the models installed locally.

Check with:

```bash
ollama ls
```

---

## 7.2 `configs/qwen_deepseek.yaml`

This configuration provides a comparison experiment.

```text
Reasoner:
    Qwen

Actioner:
    DeepSeek
```

The purpose is to determine whether Ornith performs better as an Actioner than another model.

---

## 7.3 `configs/single_model.yaml`

This configuration provides a single-model baseline.

The baseline could use one model for both reasoning and action.

Example:

```text
Single Model
    │
    ├── Reasoning
    └── Actions
```

This baseline is important.

Without a baseline, it is difficult to determine whether separating reasoning and action actually improves performance.

---

# 8. `experiments/` Directory

---

## 8.1 `experiments/run_experiment.py`

### Purpose

Run a complete experiment based on a configuration file.

Example:

```bash
python experiments/run_experiment.py \
    --config configs/qwen_ornith.yaml
```

The script should:

1. Load the configuration.
2. Initialize the Reasoner.
3. Initialize the Actioner.
4. Initialize the Agent Controller.
5. Initialize the SWE-bench benchmark.
6. Select instances.
7. Run the benchmark.
8. Collect results.
9. Save raw results.

It should support small test runs before full benchmark runs.

For example:

```bash
python experiments/run_experiment.py \
    --config configs/qwen_ornith.yaml \
    --limit 1
```

Then:

```bash
python experiments/run_experiment.py \
    --config configs/qwen_ornith.yaml \
    --limit 5
```

Then:

```bash
python experiments/run_experiment.py \
    --config configs/qwen_ornith.yaml \
    --limit 10
```

Only after the system is stable should the full benchmark be run.

---

# 9. `results/` Directory

Results should be separated into raw and processed data.

---

## 9.1 `results/raw/`

This directory stores the raw results from individual tasks.

Example:

```text
results/
└── raw/
    └── qwen_ornith/
        ├── astropy__12345/
        │   ├── result.json
        │   ├── agent.log
        │   ├── test.log
        │   └── patch.diff
        │
        └── django__12345/
            ├── result.json
            ├── agent.log
            ├── test.log
            └── patch.diff
```

Raw results must not be overwritten.

They represent the experimental record.

A result should contain information such as:

```json
{
    "instance_id": "django__django-12345",
    "configuration": "qwen_ornith",
    "reasoner": "qwen3.5:9b",
    "actioner": "ornith:9b",
    "success": true,
    "iterations": 3,
    "time_seconds": 412.5,
    "tests_passed": true,
    "patch_generated": true
}
```

The exact schema can be expanded as the project develops.

---

## 9.2 `results/processed/`

This directory stores aggregated results.

Example:

```text
results/
└── processed/
    ├── qwen_ornith_summary.json
    ├── qwen_deepseek_summary.json
    └── single_model_summary.json
```

Processed results may include:

```text
Total Instances
Resolved Instances
Resolution Rate
Average Runtime
Average Iterations
Average Tool Calls
Timeout Rate
Failure Categories
```

---

# 10. `analysis/` Directory

This directory analyzes benchmark results.

---

## 10.1 `analysis/analyze_results.py`

### Purpose

Read raw results and produce aggregated statistics.

It should calculate metrics including:

```text
Resolution Rate
Average Runtime
Average Iterations
Average Tool Calls
Test Pass Rate
Timeout Rate
Failure Rate
Failure Categories
```

It should support comparisons such as:

```text
                    Resolved    Rate
Qwen + Ornith          21       7.0%
Qwen + DeepSeek        24       8.0%
Single Model           15       5.0%
```

Additional analysis should eventually include:

```text
Performance by Repository
Performance by Issue Type
Performance by Difficulty
Performance by Number of Iterations
```

---

## 10.2 `analysis/visualize_results.py`

### Purpose

Create visualizations from processed benchmark results.

Potential visualizations:

```text
Resolution Rate by Configuration
Resolution Rate by Repository
Average Runtime
Average Iterations
Failure Categories
Tool Calls
```

The visualization script should read processed results rather than directly modifying raw results.

---

# 11. `README.md`

The project README should explain:

1. Project purpose.
2. System requirements.
3. Hardware requirements.
4. Python environment setup.
5. Ollama installation.
6. Model installation.
7. SWE-bench Lite dataset setup.
8. Repository setup.
9. Running one instance.
10. Running a small experiment.
11. Running the full benchmark.
12. Analyzing results.
13. Reproducing experiments.
14. Project architecture.

The README should be written so that another developer can reproduce the experiment.

---

# 12. `requirements.txt`

This file should contain Python dependencies required by the project.

Dependencies should be added only when they are actually required.

Potential dependency categories include:

```text
Ollama API client
YAML configuration
Dataset handling
SWE-bench evaluation tooling
Data analysis
Visualization
```

Avoid adding unnecessary packages.

---

# 13. Recommended Implementation Order

Do not implement the entire system at once.

Build and test incrementally.

The recommended order is:

```text
ACTION 01
Create the project directory structure.

ACTION 02
Audit the existing SWE-bench download/setup code.
Do not recreate functionality that already exists.

ACTION 03
Create agents/ollama_client.py.

ACTION 04
Test communication with Ollama.

ACTION 05
Create agents/reasoner.py.

ACTION 06
Test Qwen as the Reasoner independently.

ACTION 07
Create agents/actioner.py.

ACTION 08
Test Ornith as the Actioner independently.

ACTION 09
Create agents/agent.py.

ACTION 10
Test the Reasoner → Actioner workflow on a simple repository task.

ACTION 11
Create swebench/utils.py.

ACTION 12
Create swebench/download_dataset.py if the existing implementation does not already provide this functionality.

ACTION 13
Create swebench/setup_repos.py.

ACTION 14
Verify that a SWE-bench repository can be restored to the correct base commit.

ACTION 15
Create swebench/evaluate.py.

ACTION 16
Create swebench/run_instance.py.

ACTION 17
Run one SWE-bench Lite instance manually.

ACTION 18
Debug the complete single-instance workflow.

ACTION 19
Create benchmarks/swebench_lite.py.

ACTION 20
Create configs/qwen_ornith.yaml.

ACTION 21
Create configs/qwen_deepseek.yaml.

ACTION 22
Create configs/single_model.yaml.

ACTION 23
Create experiments/run_experiment.py.

ACTION 24
Run exactly one benchmark instance through the experiment runner.

ACTION 25
Run five benchmark instances.

ACTION 26
Run ten benchmark instances.

ACTION 27
Create analysis/analyze_results.py.

ACTION 28
Create analysis/visualize_results.py.

ACTION 29
Run comparison experiments.

ACTION 30
Run the full benchmark only after the pipeline is stable.
```

---

# 14. Critical Development Rule

Do not build the entire benchmark infrastructure before verifying the core agent loop.

The first working end-to-end pipeline should be:

```text
SWE-bench Instance
        │
        ▼
Repository Setup
        │
        ▼
Qwen Reasoner
        │
        ▼
Reasoning Plan
        │
        ▼
Ornith Actioner
        │
        ▼
Tool Execution
        │
        ▼
Repository Modification
        │
        ▼
Test Execution
        │
        ▼
Pass / Fail
```

Once this works on one task, add:

```text
Multiple Instances
        │
        ▼
Result Collection
        │
        ▼
Experiment Configurations
        │
        ▼
Statistical Analysis
        │
        ▼
Model Comparisons
```

Do not run the entire SWE-bench Lite benchmark until the single-instance workflow is reliable.

---

# 15. Primary Research Question

The project should ultimately allow the following question to be investigated:

> Does separating software engineering reasoning from software engineering action improve the performance of local LLM agents on SWE-bench Lite?

The primary architecture is:

```text
Qwen 3.5 9B
Reasoning
    +
Ornith 9B
Action
```

The architecture should be compared against:

```text
Single Model Baseline
```

and:

```text
Qwen + Alternative Actioner
```

The most important metric is:

```text
SWE-bench Lite Resolution Rate
```

Additional metrics should include:

```text
Average Runtime
Average Iterations
Average Tool Calls
Token Usage
Timeout Rate
Failure Categories
```

The final system should produce reproducible results that can be compared across model configurations.

---

# 16. Final Architecture

The finished system should operate approximately as follows:

```text
                         EXPERIMENT RUNNER
                                │
                                ▼
                         CONFIGURATION
                                │
                                ▼
                         SWE-bench Lite
                                │
                                ▼
                          INSTANCE SETUP
                                │
                                ▼
                       BASE COMMIT CHECKOUT
                                │
                                ▼
                         AGENT CONTROLLER
                                │
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           │
             QWEN REASONER                    │
                  │                           │
                  ▼                           │
             ACTION PLAN                      │
                  │                           │
                  ▼                           │
            ORNITH ACTIONER                   │
                  │                           │
                  ▼                           │
             TOOL EXECUTION                   │
                  │                           │
        ┌─────────┼─────────┐                 │
        │         │         │                 │
        ▼         ▼         ▼                 │
      Files     Search    Terminal            │
        │         │         │                 │
        └─────────┼─────────┘                 │
                  │                           │
                  ▼                           │
             CODE CHANGES                     │
                  │                           │
                  ▼                           │
             TEST EXECUTION                   │
                  │                           │
             ┌────┴────┐                      │
             │         │                      │
           PASS       FAIL                    │
             │         │                      │
             ▼         └──────────────────────┘
          SUCCESS           │
                            ▼
                       QWEN REASONER
                       DEBUG ANALYSIS
                            │
                            ▼
                       ORNITH ACTIONER
                            │
                            ▼
                       RETRY / FIX
                            │
                            ▼
                         TESTS
                            │
                            ▼
                        RESULTS
                            │
                            ▼
                      RAW RESULTS
                            │
                            ▼
                   PROCESSED RESULTS
                            │
                            ▼
                       ANALYSIS
                            │
                            ▼
                      COMPARISON
```

This architecture should be implemented incrementally, with each component tested before moving to the next component.
