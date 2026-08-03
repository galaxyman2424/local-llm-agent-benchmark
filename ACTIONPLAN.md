The real reason models are failing

Once I looked at the actual history, the pattern is clear and it's an environment mismatch, not a model-capability problem:

Your final grading (benchmarks/swebench.py line ~132, swebench/utils.py::ensure_repo_environment) builds a per-repo virtualenv at <repo_path>/.swebench_venv and correctly uses that python_bin to run the official tests.
But the Actioner's run_command and run_tests tools never get that python_bin — Actioner.__init__ only takes model_id, workspace_dir, timeout_seconds, num_ctx. Both tools just do subprocess.run(command, shell=True, cwd=self.workspace_dir), so any python/pytest the model types resolves to the ambient system interpreter, not the venv with astropy's built C extensions.
In the trace I pulled, the model correctly diagnosed the astropy separability_matrix bug, read the right file, and tried to verify its hypothesis with python -c "..." — and got ImportError: cannot import name '_compiler' (astropy needs pip install -e . / extension build) every single time. It retried a near-identical command, the oscillation/repeat detector correctly saw a stuck loop, and killed the run — before the model ever got a chance to run real tests or write a fix.

So the model isn't failing at reasoning — it's failing because it can't get truthful feedback from its own tools.

Fix: thread the repo's python_bin through to the Actioner and use it for run_command/run_tests (e.g. prepend .swebench_venv/bin to PATH for the subprocess, or explicitly rewrite bare python/pytest invocations to the venv's binaries). This needs to happen in both benchmarks/swebench.py (pass python_bin into Actioner(...)) and agents/actioner.py (accept it, use it).

Two smaller bugs worth fixing while you're in there
agents/agent.py line ~186 — is_valid, error = validate_action(action) is computed but never checked. Invalid actions from the Actioner sail through to execute() uncontested (there's a redundant safety check inside execute() itself, but the intended "validate before executing" step in the main loop is dead code).



What a minimal implementation would look like:

A new agents/memory.py with a simple in-memory vector store scoped to one Agent.solve() call (no need for cross-task persistence — each SWE-bench instance is independent by design per CHANGES.md items 3.3/25-27).
On each iteration, embed: (a) the full tool result before truncating it, (b) chunks of any file read/searched.
When building the Reasoner's prompt, replace _format_previous_actions(last_n=5) with a similarity query against task + current_state, returning top-k relevant chunks instead of "whatever happened most recently."
This only touches Reasoner.plan's prompt assembly — Actioner.execute stays deterministic and untouched, so it doesn't violate the reasoning/action separation you've already built.

Costs to weigh honestly:

You need an actual embedding model pulled in Ollama (not currently in models.txt) — e.g. nomic-embed-text.
Extra latency per iteration (embedding calls), which eats into reasoner.timeout_seconds/agent.timeout budgets that are already tight (120s per call, 600s total in most configs).
Retrieval quality now depends on chunking strategy — worth prototyping on 1-2 instances before trusting it project-wide, consistent with the incremental philosophy already in ACTIONPLAN.md.

So: yes, feasible, and arguably more principled than the current "truncate to N chars" approach — but I'd treat it as a follow-on improvement after the loop is proven stable, not a prerequisite.

2. Clearer goal/exit condition — there's a real gap here worth fixing

Look at the actual flow: evaluate_fail_to_pass (in swebench/utils.py) applies test_patch and checks FAIL_TO_PASS/PASS_TO_PASS after agent.solve() finishes — the agent's own run_tests calls during the loop just run a generic test_command (e.g. "pytest"), with no reference tests applied yet. So the agent's internal "tests passed, I'm done" signal (Agent.solve's test_passed check) can be completely disconnected from what's actually being scored.

The fix isn't "add vague motivational framing" — it's giving the Reasoner the actual target: FAIL_TO_PASS test IDs are just names, not the solution (the gold patch is what stays hidden), so there's no reason to withhold them. Concretely:

Thread instance["FAIL_TO_PASS"] into current_state at the start of Agent.solve / run_experiment.py, the same way test_command already gets set.
Have the Reasoner's prompt say explicitly: "you are done only once these specific tests pass" instead of the current generic test_command.
Optionally have run_tests target those specific node IDs (there's already run_test_ids in swebench/utils.py doing exactly this for evaluation — it's just not wired into the live loop).