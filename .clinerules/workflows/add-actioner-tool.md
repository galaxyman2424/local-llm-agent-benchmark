Add a new tool the Actioner can execute, following this project's
single-source-of-truth schema contract (see `01-architecture.md`).

Ask the user for: the tool name, its required/optional parameters, and a
one-line description of what it does — if not already given.

Steps:
1. Add an entry to `TOOL_SCHEMAS` in `agents/tool_schemas.py`:
   ```python
   "new_tool_name": {
       "required": [...],
       "optional": [...],
       "description": "...",
   },
   ```
   This single dict is what both `schema_prompt_block()` (shown to the
   Reasoner and Actioner in their prompts) and `validate_action()` use — do
   not hard-code the tool's parameters anywhere else.
2. Implement the deterministic branch in `Actioner.execute`
   (`agents/actioner.py`). Rules:
   - No LLM calls inside this method.
   - Any path parameter must go through `self._resolve_path(...)` — never
     join paths manually.
   - Wrap the real work in the existing try/except so a runtime error comes
     back as `{"tool": tool_name, "error": str(e)}` rather than crashing the
     loop.
   - Return `{"tool": tool_name, "result": ...}` on success, matching the
     shape other tools use (so `Agent.solve`'s history/state logic and
     `Reasoner._format_previous_actions` keep working unmodified).
3. If the tool has a non-obvious best practice (e.g. "prefer this over
   `run_command` for X"), add ONE short guidance bullet to the "Guidance"
   list in `Reasoner.plan`'s prompt in `agents/reasoner.py` — don't duplicate
   the full schema there, `schema_prompt_block()` already renders it.
4. Test the tool directly before running it through the full Agent loop:
   ```python
   from agents import Actioner
   a = Actioner(workspace_dir="/path/to/a/scratch/repo")
   print(a.execute({"tool": "new_tool_name", "parameters": {...}}))
   ```
5. Confirm `validate_action` rejects a call missing a required parameter,
   and that `Agent.solve` logs (not crashes on) an invalid action — per the
   existing behavior in the loop's step 3.
