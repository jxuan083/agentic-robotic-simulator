"""Model service clients: any OpenAI-compatible chat completions endpoint, and a replay client for tests.

  llm = OpenAICompatLLM.from_cfg(load("configs/chat.yaml")["llm"])
  out = llm.complete(messages, schema)        # -> LLMReply(data=dict, raw=str, latency_s, model)

No third-party dependency (urllib). The model is asked for one JSON object; with structured_output 'json_schema' the
server constrains the reply to the schema (vLLM structured outputs), otherwise the first JSON object in the reply is
parsed. Either way the caller validates the result before acting on it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class LLMError(Exception):
    """The model's reply could not be used (no JSON object, unexpected response shape)."""


class ServiceError(LLMError):
    """The model service failed: not reachable, timed out, or answered with an HTTP error."""


@dataclass
class LLMReply:
    data: dict
    raw: str
    latency_s: float
    model: str


def first_json_object(text: str) -> dict:
    """The first complete {...} in text (models sometimes wrap JSON in prose or code fences)."""
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    raise LLMError(f"no JSON object in the model reply: {text[:200]!r}")


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 400, timeout_s: float = 30.0, enable_thinking: bool | None = False,
                 structured_output: str | None = "json_schema"):
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key = model, api_key
        self.temperature, self.max_tokens, self.timeout_s = temperature, max_tokens, timeout_s
        self.enable_thinking, self.structured_output = enable_thinking, structured_output

    @classmethod
    def from_cfg(cls, c: dict) -> "OpenAICompatLLM":
        key = os.environ.get(c["api_key_env"]) if c.get("api_key_env") else None
        return cls(c["base_url"], c["model"], key, float(c.get("temperature", 0.0)), int(c.get("max_tokens", 400)),
                   float(c.get("timeout_s", 30)), c.get("enable_thinking", False), c.get("structured_output"))

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise ServiceError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ServiceError(f"model service not reachable at {self.base_url}: {e}") from None

    def models(self) -> list:
        req = urllib.request.Request(self.base_url + "/models")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return [m["id"] for m in json.loads(r.read())["data"]]
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
            raise ServiceError(f"model service not reachable at {self.base_url}: {e}") from None

    def complete(self, messages: list, schema: dict | None = None) -> LLMReply:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        if schema is not None and self.structured_output == "json_schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "intent", "schema": schema, "strict": True}}
        t0 = time.monotonic()
        out = self._post("/chat/completions", body)
        dt = time.monotonic() - t0
        try:
            text = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response: {str(out)[:300]}") from None
        return LLMReply(first_json_object(text), text, dt, out.get("model", self.model))


class ReplayLLM:
    """Fixed replies for tests and reproducible replays: a list consumed in order, or a dict keyed by the user's
    text (the last user message)."""

    def __init__(self, replies, model: str = "replay"):
        self.replies, self.model = replies, model
        self.i = 0
        self.seen: list = []

    def complete(self, messages: list, schema: dict | None = None) -> LLMReply:
        text = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        text = text.rsplit("使用者：", 1)[-1]          # the chat prompt puts the task state before the user's words
        self.seen.append(text)
        if isinstance(self.replies, dict):
            data = self.replies[text]
        else:
            data = self.replies[self.i]
            self.i += 1
        return LLMReply(dict(data), json.dumps(data, ensure_ascii=False), 0.0, self.model)
