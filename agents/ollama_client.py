"""Ollama client for communicating with the Ollama API."""

import json
from typing import Any, Generator


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API.

    Supports both streaming and non-streaming chat completions.
    All methods raise ``RuntimeError`` on network or server errors.

    This is the SINGLE client implementation used by both the Reasoner and
    the Actioner -- do not duplicate this class elsewhere. Both components
    should import it the same way::

        from .ollama_client import OllamaClient
    """

    BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b-instruct",
        *,
        base_url: str | None = None,
        timeout_seconds: float = 300.0,
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
        import http.client
        import urllib.error
        import urllib.request

        url = path if path.startswith("http") else f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                # A malformed/truncated body (e.g. a connection that dropped
                # mid-response) isn't a URLError -- it's a ValueError from
                # json.loads -- so it must be caught and wrapped here too,
                # or it propagates uncaught past every layer above that
                # only expects RuntimeError (see Reasoner._call_model).
                snippet = raw[:200].decode("utf-8", errors="replace") if raw else "<empty body>"
                raise RuntimeError(
                    f"Ollama returned a non-JSON/truncated response from {url}: {e} "
                    f"(body started with: {snippet!r})"
                ) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            # TimeoutError/ConnectionError/http.client.HTTPException cover
            # failures that can occur mid-read (e.g. a slow cold model load
            # exceeding the timeout, or a dropped connection while streaming
            # the response) -- these are NOT urllib.error.URLError and were
            # previously left uncaught, so they escaped every downstream
            # "except RuntimeError" handler and surfaced as unexplained,
            # undiagnosed errors with no instance record at all.
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
        num_ctx: int | None = None,
        keep_alive: str | int | None = None,
        think: bool | None = None,
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
        num_ctx
            Total context window size (prompt tokens + generated tokens),
            forwarded as Ollama's ``num_ctx`` option. THIS, not
            ``num_predict``, is usually the real cause of mid-JSON
            truncation: Ollama's default ``num_ctx`` is small (often
            2048-4096 depending on the model/Modelfile), and a prompt that
            embeds tool schemas, prior tool results, and repository state
            can consume nearly all of that budget -- leaving almost no
            room to generate before hitting the context ceiling, so
            output stops mid-string well before ``num_predict`` would
            ever kick in, and the exact cutoff point shifts between
            retries as the prompt content shifts slightly. Set this
            generously (e.g. 8192-16384) for any planning/action call
            whose prompt embeds tool results or file contents.
        keep_alive
            Forwarded directly as Ollama's top-level ``keep_alive`` field
            (e.g. ``0`` to unload the model immediately after the request,
            or a duration string like ``"5m"`` to keep it warm). Useful
            when grid-searching many models back to back on limited VRAM.
        think
            When the target model supports Ollama's "thinking" mode,
            explicitly forces it on/off. For structured JSON planning
            calls, pass ``think=False`` so the model doesn't spend its
            entire ``num_predict`` budget on a hidden reasoning trace and
            return empty ``content`` (``done_reason: "length"``). Left as
            ``None``, Ollama's own per-model default is used.

        Returns
        -------
        dict | Generator[dict, None, None]
            Parsed JSON body from the API (non-streaming), or a generator of
            chunks (streaming).
        """
        options: dict[str, Any] = {
            "temperature": temperature,
        }

        if num_predict is not None:
            options["num_predict"] = num_predict

        if num_ctx is not None:
            options["num_ctx"] = num_ctx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }

        if json_mode:
            payload["format"] = "json"

        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        if think is not None:
            payload["think"] = think

        if not stream:
            result = self._request("POST", "/api/chat", payload)

            if "error" in result:
                raise RuntimeError(result["error"])

            return result

        return self._chat_stream(payload)

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