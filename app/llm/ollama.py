from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator

import httpx


class OllamaClient:
    """Resilient async Ollama HTTP client for Curly.

    Public interface intentionally matches Curly's existing usage:
      - OllamaClient(base_url=..., model=...)
      - await generate(messages, response_format=None)
      - async for token in generate_stream(messages)

    The client is conservative with resources, serializes model requests,
    retries transient 5xx failures, attempts to unload a broken runner before
    retrying, and falls back from streaming to a normal request when needed.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.1:8b",
        timeout: float = 120.0,
        connect_timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max(0, int(max_retries))

        self.timeout = httpx.Timeout(
            timeout,
            connect=connect_timeout,
            read=timeout,
            write=timeout,
            pool=connect_timeout,
        )

        # One model-generation request at a time prevents overlapping loads /
        # generations from competing for GPU/shared memory.
        self._request_lock = asyncio.Lock()

        # Keep inference bounded for a voice assistant. These are API options,
        # not Modelfile edits. Environment variables allow tuning later.
        self.num_ctx = self._env_int("CURLY_OLLAMA_NUM_CTX", 2048)
        self.num_predict = self._env_int("CURLY_OLLAMA_NUM_PREDICT", 256)
        self.temperature = self._env_float("CURLY_OLLAMA_TEMPERATURE", 0.35)
        self.keep_alive = os.getenv("CURLY_OLLAMA_KEEP_ALIVE", "5m")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
            return value if value > 0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
            return value if value >= 0 else default
        except (TypeError, ValueError):
            return default

    def _clean_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages into the exact shape Ollama chat expects."""
        cleaned: list[dict[str, Any]] = []

        for item in messages:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role", "")).strip().lower()
            content = item.get("content", "")

            if role not in {"system", "user", "assistant", "tool"}:
                continue

            if content is None:
                content = ""

            if not isinstance(content, str):
                content = str(content)

            cleaned.append({
                "role": role,
                "content": content,
            })

        if not cleaned:
            raise ValueError("Ollama request contains no valid messages")

        return cleaned

    def _payload(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        response_format: Any | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._clean_messages(messages),
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
            },
        }

        if response_format is not None:
            payload["format"] = response_format

        return payload

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        body = response.text.strip()
        if not body:
            return f"HTTP {response.status_code} {response.reason_phrase}"

        try:
            parsed = response.json()
            if isinstance(parsed, dict) and parsed.get("error"):
                return str(parsed["error"])
        except (ValueError, json.JSONDecodeError):
            pass

        return body[:4000]

    async def _unload_model(self) -> None:
        """Ask Ollama to unload the model after a runner/server-side failure."""
        payload = {
            "model": self.model,
            "keep_alive": 0,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                if response.status_code not in {200, 404}:
                    print(
                        f"[OLLAMA] Model unload returned HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
        except Exception as exc:
            print(f"[OLLAMA] Model unload attempt failed: {exc}")

    async def _request_json(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )

                if response.status_code >= 500:
                    detail = self._error_detail(response)
                    raise RuntimeError(
                        f"Ollama HTTP {response.status_code}: {detail}"
                    )

                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("Ollama returned a non-object JSON response")

                if data.get("error"):
                    raise RuntimeError(str(data["error"]))

                return data

            except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                print(
                    f"[OLLAMA] request attempt {attempt + 1}/{self.max_retries + 1} failed: {exc}"
                )

                if attempt >= self.max_retries:
                    break

                # Give the runner a chance to settle. On the first retry, unload
                # the model so a fresh runner/context can be created.
                await asyncio.sleep(0.75 * (attempt + 1))
                await self._unload_model()
                await asyncio.sleep(0.5)

        raise RuntimeError(
            f"Ollama request failed after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

    async def generate(
        self,
        messages: list[dict[str, Any]],
        response_format: Any | None = None,
    ) -> str:
        payload = self._payload(
            messages,
            stream=False,
            response_format=response_format,
        )

        async with self._request_lock:
            started = time.perf_counter()
            data = await self._request_json(payload)

        message = data.get("message", {})
        content = ""
        if isinstance(message, dict):
            content = message.get("content", "") or ""

        # Support older / alternate Ollama response shapes.
        if not content:
            content = data.get("response", "") or ""

        result = str(content).strip()
        if not result:
            raise RuntimeError("Ollama returned an empty response")

        print(
            f"[OLLAMA] generate complete | {time.perf_counter() - started:.2f}s | "
            f"chars={len(result)}"
        )
        return result

    async def _stream_once(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Yield content tokens from one streaming request."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                if response.status_code >= 500:
                    detail = self._error_detail(response)
                    raise RuntimeError(
                        f"Ollama HTTP {response.status_code}: {detail}"
                    )

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Invalid JSON chunk from Ollama: {line[:500]}"
                        ) from exc

                    if isinstance(item, dict) and item.get("error"):
                        raise RuntimeError(str(item["error"]))

                    message = item.get("message")
                    if isinstance(message, dict):
                        content = message.get("content") or ""
                    else:
                        content = item.get("response") or ""

                    if content:
                        yield str(content)

                    if item.get("done"):
                        return

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, stream=True)

        # Serialize the entire model request. A streaming request holds the
        # model runner, so allowing another generation concurrently is exactly
        # what we do not want on a constrained Windows machine.
        async with self._request_lock:
            started = time.perf_counter()
            yielded_any = False
            last_error: Exception | None = None

            for attempt in range(self.max_retries + 1):
                try:
                    async for token in self._stream_once(payload):
                        if token:
                            yielded_any = True
                            yield token
                    print(
                        f"[OLLAMA] stream complete | {time.perf_counter() - started:.2f}s"
                    )
                    return

                except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
                    last_error = exc
                    print(
                        f"[OLLAMA] stream attempt {attempt + 1}/{self.max_retries + 1} failed: {exc}"
                    )

                    # Retrying a partially streamed response would duplicate text.
                    # Only retry when no content reached the caller yet.
                    if yielded_any or attempt >= self.max_retries:
                        break

                    await asyncio.sleep(0.75 * (attempt + 1))
                    await self._unload_model()
                    await asyncio.sleep(0.5)

            # Final reliability fallback: if the stream failed before sending
            # any content, issue one normal request. This gives Curly a usable
            # answer even when streaming is the unstable part of the stack.
            if not yielded_any:
                print("[OLLAMA] Streaming failed before first token; using non-stream fallback")
                fallback_payload = self._payload(messages, stream=False)

                try:
                    data = await self._request_json(fallback_payload)
                    message = data.get("message", {})
                    content = ""
                    if isinstance(message, dict):
                        content = message.get("content", "") or ""
                    if not content:
                        content = data.get("response", "") or ""

                    content = str(content).strip()
                    if content:
                        yield content
                        return

                    raise RuntimeError("Ollama fallback returned an empty response")

                except Exception as fallback_error:
                    raise RuntimeError(
                        "Ollama streaming and non-streaming fallback both failed: "
                        f"stream={last_error}; fallback={fallback_error}"
                    ) from fallback_error

            raise RuntimeError(
                f"Ollama streaming failed after {self.max_retries + 1} attempts: {last_error}"
            ) from last_error

    async def close(self) -> None:
        """Close the Ollama client.

        The current implementation creates short-lived httpx clients for each
        request, so there is no persistent HTTP connection to close here.
        This method exists for lifecycle compatibility with the FastAPI app
        and tests.
        """
        return None

    async def health(self) -> dict[str, Any]:
        """Check server reachability and model availability."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

        models = data.get("models", []) if isinstance(data, dict) else []
        model_names = {
            item.get("name")
            for item in models
            if isinstance(item, dict) and item.get("name")
        }

        return {
            "server": True,
            "model": self.model,
            "model_available": self.model in model_names,
            "models": sorted(model_names),
        }
