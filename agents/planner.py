"""Planner for task analysis and planning.

The Planner is responsible ONLY for deciding WHAT should happen next. It
never executes tools itself -- it hands a structured plan to the Actioner,
which turns that plan into exactly one concrete tool call
(``Actioner.plan_action``), which the deterministic ``Actioner.execute``
then runs. Enforcing this separation is why ``Planner.plan`` returns a
``next_action`` / ``parameters`` / ``expected_outcome`` triple rather than
anything that looks like an already-executed result.
"""

from __future__ import annotations

from typing import Any

from .tool_schemas import schema_prompt_block
from .json_utils import extract_json_object, repair_truncated_json

# Truncate any single previous tool result embedded back into the prompt so
# a large file read/test output doesn't blow the context window. Full,
# untruncated output is still kept in `previous_actions`/benchmark logs --
# only what gets fed back into the LLM prompt is capped here.
MAX_OUTPUT_CHARS = 32000

# How many times the same (tool, parameters) pair can repeat in a row
# before we inject explicit "you are stuck" feedback into the prompt.
REPEATED_ACTION_THRESHOLD = 2

# Ollama's default num_ctx (often 2048-4096) is frequently too small once a
# prompt embeds tool schemas + several previous tool results + repo state --
# the model then hits the context ceiling mid-generation and the reply
# truncates well before num_predict would ever kick in. Set this generously
# for every planning call.
DEFAULT_NUM_CTX = 131072

# Fields whose value is literal file/code content that gets written
# straight to disk by the Actioner (write_to_file's `content`,
# replace_in_file's `search`/`replace`). repair_truncated_json() can turn a
# reply that was cut off mid-generation into syntactically valid JSON by
# just closing the open string wherever generation happened to stop -- that
# makes the JSON parse, but if the cut-off point was inside one of these
# fields, the "repaired" value is a silently truncated/corrupted piece of
# code, not a safe approximation. A truncated `expected_outcome` or search
# `query` is harmless prose; a truncated `replace` is a broken file waiting
# to happen. So a repair that touches any of these fields is never trusted.
RISKY_CONTENT_FIELDS = ("content", "search", "replace")


class Planner:
    """Reasoning component that analyzes tasks and produces action plans."""

    def __init__(
        self,
        *,
        model_id: str = "qwen3.5:9b",
        timeout_seconds: float = 240.0,
        num_ctx: int = DEFAULT_NUM_CTX,
    ):
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.num_ctx = num_ctx

        # Populated by every _call_model() call with the exact messages sent
        # and the raw assistant reply received. The Agent reads these after
        # plan() returns so that, when the Actioner is running the SAME
        # model, it can continue this exact conversation (see
        # Actioner.plan_action's `conversation` parameter) instead of
        # starting a brand new, disconnected prompt -- letting the model
        # pick up its own train of thought and letting Ollama reuse the KV
        # cache for the shared prefix instead of reprocessing it.
        self.last_messages: list[dict[str, str]] | None = None
        self.last_reply: str | None = None

    def get_last_conversation(self) -> list[dict[str, str]] | None:
        """Return the full conversation (prompt + assistant reply) from the
        most recent successful _call_model() call, suitable for handing to
        another component that wants to continue the same chat -- or
        ``None`` if no call has completed yet.
        """
        if self.last_messages is None or self.last_reply is None:
            return None
        return self.last_messages + [{"role": "assistant", "content": self.last_reply}]

    def analyze(self, task_description: str, state_context: dict[str, Any]) -> dict[str, Any]:
        """Analyze the current repository state and generate a high-level plan.

        This is a coarse, non-binding summary (used for logging/telemetry);
        the actual step-by-step decisions come from :meth:`plan`.
        """
        prompt = f"""You are an expert software engineer. Analyze this task and the current state to determine what needs to be done.

TASK DESCRIPTION:
{task_description}

CURRENT STATE CONTEXT:
{_truncate(state_context)}

Provide your analysis as a structured JSON response with these fields:
- "understanding": A clear summary of what the task requires
- "plan": An ordered list of specific actions needed (file reads, edits, test runs)
- "expected_outcome": What should be true after all actions complete"""

        analysis = {
            "understanding": f"Task requires {task_description[:100]}...",
            "plan": [],
            "expected_outcome": "Repository should be in correct state with all changes applied.",
        }

        result = self._call_model(short_prompt, num_predict=1024, temperature=temperature)
        if isinstance(result, dict):
            analysis.update(result)

        return analysis

    def plan(
        self,
        task: str,
        current_state: dict[str, Any],
        previous_actions: list[dict],
        avoid_action: tuple[str, str] | None = None,
        stagnation_hint: str | None = None,
        keep_alive: str | int | None = 0,
    ) -> dict[str, Any] | None:

        """Generate the next single action to take.

        Parameters
        ----------
        task
            The problem description from the SWE-bench instance.
        current_state
            Current repository/agent state (workspace, test command, last
            result, etc.).
        previous_actions
            Full history of ``{"iteration", "planner_plan", "action",
            "result"}`` records taken so far this run.
        avoid_action
            If set, ``(tool, json_stringified_params)`` of an action the
            Agent detected repeating -- injected as a hard constraint and
            paired with a higher sampling temperature.
        keep_alive
            Forwarded to Ollama's ``keep_alive``. Defaults to ``0``
            (unload immediately after this call), matching the historical
            behavior. The Agent passes a non-zero/negative value here when
            the Actioner is configured with the SAME model_id, so the
            weights stay resident in VRAM instead of being unloaded only
            to be reloaded a moment later for ``Actioner.plan_action``.

        Returns
        -------
        dict | None
            ``{"next_action": ..., "parameters": ..., "expected_outcome": ...}``,
            or ``None`` if the model could not be reached or produced no
            usable plan after a retry.
        """
        loop_warning = _loop_warning(previous_actions)

        fail_to_pass_tests = (
            current_state.get("fail_to_pass_tests", []) if isinstance(current_state, dict) else []
        )
        pass_to_pass_tests = (
            current_state.get("pass_to_pass_tests", []) if isinstance(current_state, dict) else []
        )

        if fail_to_pass_tests:
            # We know the instance's actual gold test ids -- these are just
            # test NAMES, not the solution (the gold `patch` stays hidden),
            # so there's no reason to withhold them. Use the Agent's own
            # authoritative live check (re-run after every run_tests call --
            # see Agent.solve's fail_to_pass_tests handling) instead of the
            # weaker "did the last run_tests return 0" heuristic, since the
            # model could have run an unrelated command and gotten a
            # misleading returncode 0.
            tests_confirmed_passing = bool(
                isinstance(current_state, dict) and current_state.get("target_tests_passing")
            )
            goal_block = f"""
GOAL / EXIT CONDITION (this is exactly what will be scored -- not a vague
"make the tests pass"):
The fix is complete once ALL of the following specific tests pass:
FAIL_TO_PASS (currently failing; must start passing):
{chr(10).join(f"  - {t}" for t in fail_to_pass_tests)}
PASS_TO_PASS (already passing; must keep passing -- do not regress these):
{chr(10).join(f"  - {t}" for t in pass_to_pass_tests) if pass_to_pass_tests else "  (none tracked)"}
Use run_tests to check your progress against these specific tests (e.g.
"pytest {fail_to_pass_tests[0]}") rather than a generic full-suite command.
"""
            done_rule = (
                "You MAY choose \"done\" now: the FAIL_TO_PASS and "
                "PASS_TO_PASS tests listed above were just directly "
                "re-checked and ALL of them currently pass."
                if tests_confirmed_passing else
                "You MUST NOT choose \"done\" yet. The FAIL_TO_PASS/"
                "PASS_TO_PASS tests listed above have not all been "
                "confirmed passing. If you believe the fix is complete, "
                "your next_action MUST be \"run_tests\" to confirm it, not "
                "\"done\" -- a generic run_tests call is re-checked "
                "automatically against those specific tests."
            )
        else:
            # No gold test list for this task (e.g. a synthetic/local
            # instance) -- fall back to the previous, looser heuristic.
            tests_confirmed_passing = _tests_passed(previous_actions)
            goal_block = ""
            done_rule = (
                "You MAY choose \"done\" now, since a run_tests action with "
                "returncode == 0 already appears in the previous actions above."
                if tests_confirmed_passing else
                "You MUST NOT choose \"done\" yet. No run_tests action with "
                "returncode == 0 has occurred so far. If you believe the fix is "
                "complete, your next_action MUST be \"run_tests\" to confirm it, "
                "not \"done\"."
            )

        file_cache = current_state.get("file_cache", {})

        avoid_block = ""
        if avoid_action is not None:
            avoid_tool, avoid_params = avoid_action
            avoid_block = f"""
HARD CONSTRAINT: You are FORBIDDEN from choosing this exact action again
right now -- you already did this and it made no further progress:
    tool={avoid_tool} parameters={avoid_params}
You MUST pick a genuinely different tool call (or "done"). Re-reading a
file you already have the full contents of is NOT allowed.
"""

        stagnation_block = f"\n{stagnation_hint}\n" if stagnation_hint else "" 

        files_read = current_state.get("files_read", {}) if isinstance(current_state, dict) else {}
        state_for_prompt = (
            {k: v for k, v in current_state.items() if k != "files_read"}
            if isinstance(current_state, dict) else current_state
        )

        prompt = f"""You are the reasoning component of a software engineering agent.
{avoid_block}
{stagnation_block}


Your job is to select exactly ONE next action. You do NOT execute tools
yourself -- a separate component (the Actioner) will translate your choice
into a concrete tool call.

TASK:
{task}
{goal_block}
{done_rule}

CURRENT STATE:
{_truncate(state_for_prompt)}

FILES ALREADY READ THIS TASK (do not re-read the same range again; use
different start_line/end_line if you need a part you haven't seen):
{_format_files_read(files_read)}

RECENT PREVIOUS ACTIONS AND THEIR RESULTS (most recent last):
{_truncate(_format_previous_actions(previous_actions))}
{loop_warning}

You may also choose the special action "done" (with empty parameters) if
you believe the task is already complete (e.g. the tests already pass and
no further changes are needed).

Guidance:
- Prefer to search and read before you write or edit.
- Use search_code with concrete identifiers extracted from the task
  (function/class names, file names, error messages) -- never paste the
  entire task description as a search query.
- For files you expect to be large, or after a search_code hit gives you a
  line number, use read_file's start_line/end_line to read just the
  relevant section instead of the whole file.
- Only choose write_to_file when a brand new file is genuinely required;
  prefer replace_in_file for editing existing files.
- Never invent a project, path, or workspace unrelated to this task.
- For files you expect to be large, or after a search_code hit gives you a
  line number, use read_file's start_line/end_line to read just the
  relevant section instead of the whole file.
- When editing, prefer replace_lines over replace_in_file if you already
  know the exact line numbers (e.g. from a recent read_file or
  search_code call) -- it avoids needing to retype the original text
  exactly. Include expected_line_count (end_line - start_line + 1) as a
  safety check. Use replace_in_file instead only when you don't have
  reliable line numbers but do have distinctive exact text to match.
- Check the FILES ALREADY READ manifest before editing: if a file shows as
  modified since your last read (or marked STALE), re-read the relevant
  section first -- its line numbers or content may have shifted, especially
  after a replace_lines edit that changed the file's total line count.
Return ONLY a JSON object with exactly these fields:

{{
"next_action": "tool_name_or_done",
"parameters": {{}},
"expected_outcome": "short description"
}}

Rules:
- Select exactly one next action.
- Do not output Markdown.
- Do not output explanations outside the JSON.
- Keep the response concise.
- Do not repeat the task description.
"""
        print("=" * 20, "PLANNER PROMPT", "=" * 20)
        print(prompt)
        print("=" * 60)

        temperature = 0.6 if avoid_action is not None else 0.1

        result = self._call_model(prompt, temperature=temperature, keep_alive=keep_alive)
        if result is None:
            print("[Planner.plan] Empty/invalid model reply, retrying once with a shorter prompt.")
            short_prompt = f"""You are the reasoning component of a software engineering agent.
Select exactly ONE next action as JSON: {{"next_action": "tool_name_or_done", "parameters": {{}}, "expected_outcome": "..."}}

TASK (truncated): {task[:400]}
{avoid_block}
Available tools:
{schema_prompt_block()}

Return only the JSON object, nothing else."""
            result = self._call_model(
                short_prompt, num_predict=1024, temperature=temperature, keep_alive=keep_alive
            )

        if not isinstance(result, dict):
            print("[Planner.plan] No usable plan after retry; giving up for this iteration.")
            return None

        return {
            "next_action": result.get("next_action", result.get("action", "search_code")),
            "parameters": result.get("parameters", result.get("params", {})),
            "expected_outcome": result.get("expected_outcome", result.get("outcome", "")),
        }


    def _call_model(
        self,
        prompt: str,
        *,
        num_predict: int = 4096,
        temperature: float = 0.1,
        keep_alive: str | int | None = 0,
    ) -> Any:
        """Call the reasoning model and parse its JSON reply into a dict.

        Returns ``None`` (rather than raising) if the model is unreachable,
        returns empty content, or the reply isn't valid JSON even after a
        repair attempt, so callers can retry or fall back to a heuristic.
        Only ``message.content`` is ever parsed as the action JSON --
        ``message.thinking`` (the model's hidden reasoning trace, when
        thinking mode is on) is never treated as the final answer.

        Records the exact messages sent and the raw reply on
        ``self.last_messages`` / ``self.last_reply`` regardless of outcome,
        so ``get_last_conversation()`` reflects the most recent attempt --
        overwritten if the caller (``plan``) retries with a shorter prompt.
        """
        import json as _json

        from .ollama_client import OllamaClient

        messages = [{"role": "user", "content": prompt}]
        self.last_messages = messages
        self.last_reply = None

        client = OllamaClient(model=self.model_id, timeout_seconds=self.timeout_seconds)
        try:
            response = client.chat(
                messages,
                temperature=temperature,
                json_mode=True,
                num_predict=num_predict,
                num_ctx=self.num_ctx,
                keep_alive=keep_alive,
                think=False,
            )
        except RuntimeError as e:
            print(f"[Planner] Ollama request failed: {e}")
            return None

        message = response.get("message", {})
        text = message.get("content", "")
        self.last_reply = text

        if not text.strip():
            print("[Planner] Ollama returned an empty content response.")
            print("[Planner] Done reason: {}".format(response.get("done_reason", "unknown")))
            #print("[Planner] Thinking length: {}".format(len(message.get("thinking", "") or "")))
            #print("[Planner] prompt_eval_count={} eval_count={} (if eval_count is near num_predict "
                  #"or prompt_eval_count+eval_count is near num_ctx={}, the context window is too "
                  #"small for this prompt).".format(
                      #response.get("prompt_eval_count", "?"), response.get("eval_count", "?"), self.num_ctx))
            return None

        candidate = extract_json_object(text)

        if candidate is None:
            # Likely a context-window truncation (see DEFAULT_NUM_CTX note
            # above) rather than a genuinely malformed reply -- try to
            # repair it (close the unterminated string/braces) before
            # giving up entirely.
            repaired = repair_truncated_json(text)
            if repaired is not None:
                try:
                    parsed = _json.loads(repaired)
                except _json.JSONDecodeError:
                    parsed = None

                if parsed is not None:
                    if _repair_touches_risky_field(parsed):
                        # The truncation happened somewhere inside content/
                        # search/replace -- repair_truncated_json only closed
                        # the open string, it did not recover the missing
                        # tail. Writing this to disk could silently corrupt a
                        # real file (e.g. an unterminated string literal) in
                        # a way the model itself has no way to notice later.
                        # Treat this exactly like an unparseable reply so the
                        # caller's existing short-prompt retry kicks in,
                        # instead of returning a plan we can't trust.
                        print("[Planner] Model reply was truncated mid-JSON while "
                              "generating a file-content field (content/search/"
                              "replace) -- refusing to trust the repaired value "
                              "since it may be silently truncated code. Treating "
                              "this call as failed so it gets retried.")
                        return None

                    print("[Planner] Model reply was truncated mid-JSON (likely num_ctx too small "
                          f"for this prompt: prompt_eval_count={response.get('prompt_eval_count', '?')} "
                          f"eval_count={response.get('eval_count', '?')} num_ctx={self.num_ctx}); "
                          "repaired it and continuing.")
                    return parsed

            print("[Planner] Model reply contained no JSON object (and could not be repaired)")
            print(f"[Planner] done_reason={response.get('done_reason', '?')} "
                  f"prompt_eval_count={response.get('prompt_eval_count', '?')} "
                  f"eval_count={response.get('eval_count', '?')} num_ctx={self.num_ctx}")
            print("[Planner] Raw model response:")
            #print(repr(text[:2000]))
            return None
        try:
            return _json.loads(candidate)
        except _json.JSONDecodeError as e:
            print(f"[Planner] Model reply was not valid JSON: {e}")
            return None


def _truncate(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text) - MAX_OUTPUT_CHARS} more chars]"
    return text


def _format_previous_actions(
    previous_actions: list[dict],
    *,
    last_n: int = 5,
) -> str:
    """Render recent previous actions without duplicating cached file contents."""
    if not previous_actions:
        return "(none yet)"

    recent = previous_actions[-last_n:]
    lines = []

    for record in recent:
        action = record.get("action", {})
        result = record.get("result", {})

        tool = action.get("tool")
        parameters = action.get("parameters", {})
        result_text = str(result.get("result", result.get("error", "")))

        if tool == "read_file" and "error" not in result:
            path = parameters.get("path", "(unknown)")
            result_display = (
                "[file contents cached under path '{}'; "
                "see CACHED FILE CONTENTS above]"
            ).format(path)
        else:
            result_display = result_text[:4000]

        lines.append(
            "- tool={} params={} -> success={} result={}".format(
                tool,
                parameters,
                "error" not in result,
                result_display,
            )
        )

    return "\n".join(lines)


def _loop_warning(previous_actions: list[dict]) -> str:
    """Detect immediate repetition of the same (tool, parameters) action and
    inject explicit feedback telling the Planner it must change course.
    """
    if len(previous_actions) < REPEATED_ACTION_THRESHOLD:
        return ""

    def _key(record: dict) -> tuple:
        action = record.get("action", {})
        return (action.get("tool"), _json_stable(action.get("parameters", {})))

    tail = previous_actions[-REPEATED_ACTION_THRESHOLD:]
    keys = [_key(r) for r in tail]
    if len(set(keys)) != 1:
        return ""

    last = tail[-1]
    action = last.get("action", {})
    result = last.get("result", {})
    return f"""
WARNING: The previous action did not make progress and was repeated
{REPEATED_ACTION_THRESHOLD} times in a row without changing the outcome.

Previous action:
{action}

Previous result:
{str(result)[:1000]}

You MUST choose a different action or inspect a more specific file/query
than before. Do not repeat the same tool call with the same parameters.
"""

def _tests_passed(previous_actions: list[dict]) -> bool:
    """Whether the MOST RECENT run_tests action passed -- not whether any
    run_tests action ever passed at some earlier point. A later edit could
    have broken something after an earlier passing run, so scanning for
    "any pass in history" would wrongly let the Planner declare "done"
    on a since-regressed fix.
    """
    for record in reversed(previous_actions):
        action = record.get("action", {})
        if action.get("tool") != "run_tests":
            continue
        result = record.get("result", {})
        return result.get("returncode") == 0
    return False

def _json_stable(value: Any) -> str:
    import json as _json
    try:
        return _json.dumps(value, sort_keys=True)
    except TypeError:
        return repr(value)

def _format_file_cache(
    file_cache: dict[str, str],
    *,
    max_file_chars: int = 32000,
    max_total_chars: int = 120000,
) -> str:
    """Render cached file contents for the Planner prompt.

    The Planner gets actual file contents instead of only a list of paths.
    Individual files and the total cache are capped to avoid exhausting the
    model's context window. Partial reads are rendered as line-range chunks.
    """
    if not file_cache:
        return "(no files cached yet)"

    sections = []
    total_chars = 0

    for path, entry in file_cache.items():
        if isinstance(entry, str):
            content_text = entry
            header = "FILE: {}\n".format(path)
        elif isinstance(entry, dict) and entry.get("type") == "full":
            content_text = str(entry.get("content", ""))
            header = "FILE: {}\n".format(path)
        elif isinstance(entry, dict) and entry.get("type") == "chunked":
            chunks = entry.get("chunks", [])
            rendered_chunks = []
            for chunk in chunks:
                start = chunk.get("start_line")
                end = chunk.get("end_line")
                if start is not None or end is not None:
                    prefix = f"[lines {start or 1}-{end or '?'}]"
                else:
                    prefix = "[chunk]"
                rendered_chunks.append(
                    f"{prefix}\n{chunk.get('content', '')}"
                )
            content_text = "\n\n".join(rendered_chunks)
            header = "FILE: {} (partial chunks)\n".format(path)
        else:
            content_text = str(entry)
            header = "FILE: {}\n".format(path)

        if len(content_text) > max_file_chars:
            content_text = (
                content_text[:max_file_chars]
                + "\n... [file content truncated for prompt]"
            )

        section = "{}{}\n".format(header, content_text)

        if total_chars + len(section) > max_total_chars:
            remaining = max_total_chars - total_chars

            if remaining > 100:
                sections.append(
                    section[:remaining]
                    + "\n... [remaining cached files omitted]"
                )

            break

        sections.append(section)
        total_chars += len(section)

    return "\n".join(sections)

def _state_without_file_cache(current_state: dict[str, Any]) -> dict[str, Any]:
    state = dict(current_state)
    state.pop("file_cache", None)
    return state

def _format_files_read(files_read: dict) -> str:
    """Render the persistent per-file manifest -- small and NOT subject to
    MAX_OUTPUT_CHARS truncation, so it stays visible every iteration even
    once earlier read_file/edit results have aged out of previous_actions.
    """
    if not files_read:
        return "(none yet)"
    lines = []
    for path, info in files_read.items():
        if info.get("deleted_at_iteration"):
            lines.append(f"- {path}: DELETED at iteration {info['deleted_at_iteration']}")
            continue

        parts = []
        if "last_read_iteration" in info:
            parts.append(
                f"read at iteration {info['last_read_iteration']} "
                f"(lines {info.get('last_range_read', '?')} of {info.get('lines', '?')} total)"
            )
        if "last_modified_iteration" in info:
            parts.append(
                f"MODIFIED at iteration {info['last_modified_iteration']} "
                f"via {info.get('last_modified_tool', '?')}"
            )
        if info.get("stale_since_last_read"):
            parts.append(
                "** your earlier read is now STALE -- re-read before editing "
                "again or relying on its line numbers/content **"
            )

        lines.append(f"- {path}: " + "; ".join(parts))
    return "\n".join(lines)

def _repair_touches_risky_field(parsed: Any) -> bool:
    """True if a (successfully re-parsed) repaired plan sets any of the
    file-content fields in RISKY_CONTENT_FIELDS.

    Used only on the repair path in ``Planner._call_model`` -- a plan that
    parsed cleanly on the first try (no repair needed) is never subject to
    this check, since there was nothing to silently truncate.
    """
    if not isinstance(parsed, dict):
        return False
    params = parsed.get("parameters")
    if not isinstance(params, dict):
        return False
    return any(field in params for field in RISKY_CONTENT_FIELDS)