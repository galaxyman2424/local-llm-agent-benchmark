"""Ollama client for communicating with the Ollama API."""

import json
from typing import Any, Generator


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API.

    Supports both streaming and non-streaming chat completions.
    All methods raise ``RuntimeError`` on network or server errors.
    """

    BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b-instruct",
        *,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url or self.BASE_URL
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue a synchronous JSON request to the Ollama API.

        Uses ``self.base_url`` (not the class-level default) so a client
        constructed with a custom ``base_url`` actually talks to that host.
        """
        import urllib.error
        import urllib.request

        url = path if path.startswith("http") else f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to reach Ollama at {url}: {e}") from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float = 0.7,
        json_mode: bool = False,
        num_predict: int | None = None,
    ):
        """Send a chat completion request.

        Parameters
        ----------
        messages
            List of ``{"role": "...", "content": "..."}`` dicts.
        stream
            If *True*, returns a generator yielding partial response chunks
            (see :meth:`_chat_stream`). If *False* (default), returns the
            full parsed JSON response as a dict.
        temperature
            Sampling temperature forwarded to Ollama.
        json_mode
            If *True*, sets Ollama's ``format: "json"`` request option,
            which grammar-constrains decoding so the model can only emit
            syntactically valid JSON. This does NOT guarantee the JSON
            matches whatever schema you asked for in the prompt, but it
            eliminates malformed-JSON parse failures (unterminated
            strings, missing commas, stray prose) that are common with
            smaller local models doing free-form JSON.
        num_predict
            Optional cap (or floor) on generated tokens, forwarded as
            Ollama's ``num_predict`` option. Left unset, Ollama's own
            default is used, which can be too small for a full JSON plan
            and silently truncate the response mid-string -- if you're
            seeing "Unterminated string" JSON errors, try setting this
            higher (e.g. 512-1024).

        Returns
        -------
        dict | Generator[dict, None, None]
            Parsed JSON body from the API (non-streaming), or a generator of
            chunks (streaming).
        """
        options: dict[str, Any] = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"

        if stream:
            # Returning the generator itself (rather than yielding here)
            # keeps `chat` a normal function: calling chat(stream=False)
            # executes immediately instead of silently becoming a generator
            # function that never runs until iterated.
            return self._chat_stream(payload)

        result = self._request("POST", "/api/chat", payload)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def _chat_stream(self, payload: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
        """Yield successive JSON chunks from a streaming /api/chat response."""
        import urllib.request

        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            for line in resp:
                line = line.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def generate(self, prompt: str, *, temperature: float = 0.1) -> str:
        """Single-turn text completion convenience wrapper around chat()."""
        result = self.chat([{"role": "user", "content": prompt}], temperature=temperature)
        return result.get("message", {}).get("content", "")

    def embed(self, prompt: str) -> dict[str, Any]:
        """Embed a single prompt. Returns ``{"embedding": [...]}``."""
        return self._request("POST", "/api/embeddings", {"prompt": prompt})
