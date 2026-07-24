"""Qwen-based Reasoner for task analysis and planning.

The Reasoner is responsible for:
1. Analyzing the current state of a repository
2. Understanding what needs to be changed
3. Planning a sequence of tool calls to accomplish the goal
4. Generating structured action plans with expected outcomes
"""

from typing import Any


class Reasoner:
    """Reasoning component that analyzes tasks and produces action plans."""

    def __init__(self, *, model_id: str = "qwen3.5:9b", timeout_seconds: float = 120.0):
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds

    def analyze(self, task_description: str, state_context: dict[str, Any]) -> dict[str, Any]:
        """Analyze the current repository state and generate a plan of actions."""
        prompt = f"""You are an expert software engineer. Analyze this task and the current state to determine what needs to be done.

TASK DESCRIPTION:
{task_description}

CURRENT STATE CONTEXT:
{state_context}

Provide your analysis as a structured JSON response with these fields:
- "understanding": A clear summary of what the task requires
- "plan": An ordered list of specific actions needed (file reads, edits, test runs)
- "expected_outcome": What should be true after all actions complete"""

        # In production, this would call the Ollama API with Qwen 3.5 as the reasoner model
        # For now, return a structured response based on the task description
        analysis = {
            "understanding": f"Task requires {task_description[:100]}...",
            "plan": [],
            "expected_outcome": "Repository should be in correct state with all changes applied."
        }

        try:
            result = self._call_model(prompt)
            if isinstance(result, dict):
                analysis.update(result)
        except Exception as e:
            print(f"[Reasoner.analyze] Falling back to basic analysis: {e}")

        return analysis

    def plan(self, task_description: str, current_state: dict[str, Any],
             previous_actions: list[dict]) -> dict[str, Any]:
        """Generate a concrete action plan based on the task and current state.

        Parameters
        ----------
        task_description : str
            The problem description from SWE-bench instance.
        current_state : dict
            Current repository state including file contents, test results, etc.
        previous_actions : list[dict]
            List of actions already taken in this iteration.

        Returns
        -------
        dict
            Structured plan with action steps and expected outcomes.
        """
        prompt = f"""You are the reasoning component of a software engineering agent.

        Your job is to select exactly ONE next tool action.

        TASK:
        {task_description}

        CURRENT STATE:
        {current_state}

        PREVIOUS ACTIONS:
        {previous_actions}

        Available tools:
        - read_file
        - write_to_file
        - replace_in_file
        - edit_file
        - delete_file
        - list_directory
        - search_code
        - run_command
        - run_tests
        - get_git_diff

        Return ONLY a JSON object with exactly these fields:

        {{
        "next_action": "tool_name",
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
        try:
            result = self._call_model(prompt)
            if isinstance(result, dict):
                return {
                    "next_action": result.get("action", "read_file"),
                    "parameters": result.get("params", result.get("parameters", {})),
                    "expected_outcome": result.get("outcome", result.get("expected_outcome", "")),
                }
        except Exception as e:
            print(f"[Reasoner.plan] Falling back to heuristic plan: {e}")
            return None

        # Fallback plan based on task description
        return {
            "next_action": "search_code",
            "parameters": {"query": f"Related to {task_description[:50]}"},
            "expected_outcome": "Found relevant code context for the issue."
        }

    def _call_model(self, prompt: str) -> Any:
        """Call the reasoning model and parse its JSON reply into a dict.

        Returns ``None`` (rather than raising) if the model is unreachable or
        the reply isn't valid JSON, so callers can fall back to a heuristic.
        """
        import json as _json

        from agents.ollama_client import OllamaClient

        client = OllamaClient(model=self.model_id, timeout_seconds=self.timeout_seconds)
        # json_mode grammar-constrains Ollama's decoding to valid JSON syntax
        # -- without it, smaller local models frequently produce malformed
        # JSON (unterminated strings, missing commas, stray prose before/
        # after the object). num_predict is set generously so a full plan
        # isn't silently truncated mid-string by a low default token cap.
        response = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            json_mode=True,
            num_predict=4096,
            keep_alive=0,
        )

        message = response.get("message", {})
        text = message.get("content", "")

        if not text.strip():
            print("[Reasoner] Ollama returned an empty content response.")
            print("[Reasoner] Done reason: {}".format(
                response.get("done_reason", "unknown")
            ))
            print("[Reasoner] Thinking length: {}".format(
                len(message.get("thinking", ""))
            ))
            print("[Reasoner] Full response: {}".format(repr(response)))
            return None

        candidate = _extract_json_object(text)
        
        if candidate is None:
            print("[Reasoner] Model reply contained no JSON object")
            print("[Reasoner] Raw model response:")
            print(repr(text))
            return None
        try:
            return _json.loads(candidate)
        except _json.JSONDecodeError as e:
            print(f"[Reasoner] Model reply was not valid JSON: {e}")
            return None


def _extract_json_object(text: str) -> str | None:
    """Extract the first balanced ``{...}`` object from free-form text.

    Unlike a greedy ``\\{.*\\}`` regex (which spans from the first '{' to
    the very LAST '}' anywhere in the text -- splicing unrelated content
    together whenever a model emits more than one brace-looking chunk, e.g.
    a JSON object followed by an explanation containing more braces), this
    scans character-by-character and tracks brace depth (respecting quoted
    strings, so braces inside string literals don't confuse the count) to
    find exactly the first complete, balanced object.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
