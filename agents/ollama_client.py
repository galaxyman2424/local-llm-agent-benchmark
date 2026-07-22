"""Ollama client for communicating with the Ollama API."""

import json
from typing import Any, Generator


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API.

    Supports both streaming and non-streaming chat completions.
    All methods raise ``RuntimeError`` on network or server errors.
    """

    BASE_URL = "http://localhost:11434"

    def __init__(self, model: str = "qwen2.5-coder:3b-instruct", *, base_url: str | None = None):
        self.model = model
        self.base_url = base_url or self.BASE_URL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue a synchronous JSON request to the Ollama API."""
        import urllib.request
        url = f"{OllamaClient.BASE_URL}{path}" if not path.startswith("http") else path
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Send a chat completion request (streaming or not).

        Parameters
        ----------
        messages
            List of ``{"role": "...", "content": "..."}`` dicts.
        stream
            If *True*, return a generator yielding partial response chunks.
        temperature
            Sampling temperature forwarded to Ollama.

        Returns
        -------
        dict
            Parsed JSON body from the API (non-streaming) or the first
            chunk when streaming is enabled.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }

        if stream:
            import urllib.request
            url = f"{self.base_url}/api/chat"
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req)
            reader = resp.read()
            while reader:
                chunk = reader.decode(errors="replace")
                if chunk.strip():
                    try:
                        yield json.loads(chunk)
                    except json.JSONDecodeError:
                        pass
                reader = resp.read()
            return  # pragma: no cover - generator doesn't return normally
        else:
            result = self._request("POST", "/api/chat", payload)
            if "error" in result:
                raise RuntimeError(result["error"])
            return result

    def embed(self, prompt: str) -> dict[str, Any]:
        """Embed a single prompt. Returns ``{"embedding": [...]}``."""
        return self._request("POST", "/api/embeddings", {"prompt": prompt})
