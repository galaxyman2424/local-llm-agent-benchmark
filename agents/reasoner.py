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

    def __init__(self, *, model_id: str = "qwen3.5:9b"):
        self.model_id = model_id

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

        # Simple heuristic parsing for now - in production, use the LLM response
        try:
            import json
            result = self._call_model(prompt)
            if isinstance(result, dict):
                analysis.update(result)
        except Exception as e:
            pass  # Fall back to basic analysis

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
        prompt = f"""Analyze the following task and current state to determine what needs to be done next.

TASK: {task_description}

CURRENT STATE:
{current_state}

PREVIOUS ACTIONS TAKEN:
{previous_actions}

Return a structured plan as JSON with:
- "next_action": The immediate next step (which tool to call)
- "parameters": Arguments for that action
- "expected_outcome": What should happen after this action"""

        try:
            result = self._call_model(prompt)
            if isinstance(result, dict):
                return {
                    "next_action": result.get("action", "read_file"),
                    "parameters": result.get("params", {}),
                    "expected_outcome": result.get("outcome", ""),
                }
        except Exception:
            pass

        # Fallback plan based on task description
        return {
            "next_action": "search_code",
            "parameters": {"query": f"Related to {task_description[:50]}"},
            "expected_outcome": "Found relevant code context for the issue."
        }

    def _call_model(self, prompt: str) -> Any:
        """Call the reasoning model (Qwen 3.5)."""
        try:
            from agents.ollama_client import OllamaClient
            client = OllamaClient(model=self.model_id)
            return client.chat([{"role": "user", "content": prompt}])["message"]["content"]
        except Exception as e:
            print(f"[Reasoner] Error calling model: {e}")
            return None
