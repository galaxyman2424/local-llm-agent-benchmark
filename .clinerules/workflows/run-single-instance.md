Run one SWE-bench Lite instance end-to-end and report a clear diagnosis.

Ask the user (if not already given in their message):
1. Which `instance_id` to run (e.g. `django__django-11099`).
2. Which config to use (default `configs/qwen_ornith.yaml`).

Steps:
1. Confirm `seed_repos/swe_bench_lite/dataset.jsonl` exists. If not, run
   `python swebench/fetch_swebench_lite.py` (or point the user to
   `docs/DOWNLOADING_SWEBENCH_LITE.md`) before continuing.
2. Run:
   ```
   python swebench/run_instance.py --instance <instance_id> --config <config_path>
   ```
3. Read the printed `status`, `runtime`, and `test_results` JSON.
4. Diagnose based on status, per `03-project-conventions.md`'s status
   vocabulary:
   - `environment_error` → open `<repo_path>/.swebench_venv/pip_install.log`,
     summarize the last real failure (missing system header, missing
     Cython, wrong Python version — see
     `swebench/utils.py::_diagnose_build_failure` for known patterns), and
     suggest the concrete fix (e.g. `sudo apt install pythonX.Y-dev`).
   - `resolved` → report which FAIL_TO_PASS tests now pass.
   - `not_resolved` → show which FAIL_TO_PASS tests still fail and which
     PASS_TO_PASS tests regressed, from `fail_to_pass_results` /
     `pass_to_pass_results`.
   - `timeout` → note this is an evaluation-run timeout (see
     `evaluate_fail_to_pass`'s per-test timeout), not the agent's own
     `agent.timeout` — check both if unclear.
5. Do NOT modify `seed_repos/**/repos/**` by hand to "fix" a failure —
   that's the isolated workspace/venv, not project source.
6. Summarize in plain language: did the agent's patch resolve the issue,
   and if not, what's the next concrete thing to try (different model,
   longer timeout, more max_iterations)?
