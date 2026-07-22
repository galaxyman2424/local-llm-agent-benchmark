"""Configuration for agent models and benchmark settings."""

from __future__ import annotations


class AgentConfig:
    """Holds all tunable parameters for the agent pipeline.

    This configuration is consumed by both the :class:`agents.agent.Agent` and
    the :class:`benchmarks.swebench.SWEbenchBenchmark`. Keeping it in one place
    makes it easy to swap models, change iteration limits, or tweak timeouts
    without touching agent/benchmark code.
    """

    def __init__(
        self,
        reasoner_model: str = "qwen2.5:7b",
        actioner_model: str | None = None,  # None means use the same model
        max_iterations: int = 50,
        ollama_base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
    ):
        self.reasoner_model = reasoner_model
        if actioner_model is None:
            self.actioner_model = reasoner_model
        else:
            self.actioner_model = actioner_model
        self.max_iterations = max_iterations
        self.ollama_base_url = ollama_base_url
        self.timeout_seconds = timeout_seconds
