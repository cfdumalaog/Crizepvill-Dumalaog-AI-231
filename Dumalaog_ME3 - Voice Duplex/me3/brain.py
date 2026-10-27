"""Qwen/OpenAI-compatible brain with local function-tool execution."""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable

import requests

from .tools import dispatch, tool_definitions

log = logging.getLogger("me3.brain")


class ToolBrain:
    """Synchronous Qwen chat loop that executes ME3 tools between turns.

    The vLLM endpoint is OpenAI-compatible.  Qwen may return several tool
    calls in one assistant message; each is executed locally, then the full
    tool result set is sent back for the final spoken answer.
    """

    name = "qwen-tools"

    def __init__(self, cfg: dict, system_prompt: str,
                 on_tool: Callable | None = None,
                 dispatch_fn: Callable | None = None):
        self.cfg = cfg
        self.base_url = cfg.get("base_url", "").rstrip("/")
        self.model = cfg.get("model", "")
        self.api_key_env = cfg.get("api_key_env", "")
        self.temperature = cfg.get("temperature", 0.4)
        # 256 (the old default) is too small for multi-file web-app builds:
        # a single write_file call's content alone can exceed it, which makes
        # Qwen truncate mid-JSON and the tool loop dies with
        # invalid_tool_arguments.  4096 covers realistic multi-tool turns.
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.timeout_s = float(cfg.get("timeout_s", 60))
        self.max_tool_steps = max(1, min(int(cfg.get("max_tool_steps", 8)), 16))
        self.api_key = os.environ.get(self.api_key_env, "")
        self.on_tool = on_tool
        # The default is the real ME3 tool dispatcher.  Local UI/demo surfaces
        # can inject a narrower dispatcher so provider side effects stay off
        # while the model and project tools are exercised normally.
        self.dispatch_fn = dispatch_fn or dispatch
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is required for the Qwen tool brain")

    def _request(self, messages: list[dict], include_tools: bool = True) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if include_tools:
            body["tools"] = tool_definitions()
            body["tool_choice"] = "auto"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError("Qwen returned no chat choice")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError("Qwen returned an invalid message")
        usage = payload.get("usage") or {}
        return message, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }

    @staticmethod
    def _tool_calls(message: dict) -> list[dict]:
        calls = message.get("tool_calls") or []
        return calls if isinstance(calls, list) else []

    def chat(self, history: list[dict], user_text: str) -> str:
        self.metrics = {
            "t0": time.perf_counter(),
            "first_tool_ms": None,
            "tool_calls": [],
            "n_llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        system = self.cfg.get("system_prompt") or ""
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_text})

        for step in range(self.max_tool_steps + 1):
            message, usage = self._request(messages, include_tools=True)
            self.metrics["n_llm_calls"] += 1
            self.metrics["prompt_tokens"] += usage["prompt_tokens"]
            self.metrics["completion_tokens"] += usage["completion_tokens"]
            calls = self._tool_calls(message)
            content = message.get("content")
            assistant_message = {"role": "assistant", "content": content or ""}
            if calls:
                # Preserve the provider's exact tool-call structure for the
                # next request; this is required by the OpenAI chat contract.
                assistant_message["tool_calls"] = calls
            messages.append(assistant_message)
            if not calls:
                if isinstance(content, str) and content.strip():
                    return content.strip()
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
                raise RuntimeError("Qwen returned neither tool calls nor spoken text")
            if step >= self.max_tool_steps:
                return "I reached the tool-call limit before finishing that task."

            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = function.get("name", "")
                raw_args = function.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments must be an object")
                except Exception as exc:  # noqa: BLE001
                    result = {"ok": False, "error": f"invalid_tool_arguments: {exc}"}
                else:
                    if self.metrics["first_tool_ms"] is None:
                        self.metrics["first_tool_ms"] = (
                            time.perf_counter() - self.metrics["t0"]) * 1e3
                    if self.on_tool:
                        self.on_tool("start", name, args)
                    log.info("tool call: %s", name)
                    t_call = time.perf_counter()
                    result = self.dispatch_fn(name, args)
                    tool_ms = (time.perf_counter() - t_call) * 1e3
                    self.metrics["tool_calls"].append({
                        "name": name, "ms": round(tool_ms),
                        "ok": (bool(result.get("ok"))
                               if isinstance(result, dict) else True),
                    })
                    if self.on_tool:
                        self.on_tool("finish", name, result)
                    log.info("tool result: %s ok=%s", name,
                             result.get("ok") if isinstance(result, dict) else True)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", f"call-{step}-{name}"),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False)[:50000],
                })
        raise RuntimeError("tool loop ended unexpectedly")


def build_system_prompt(base: str) -> str:
    return base + """

You can use the ME3 project tools to complete real tasks. When the user asks to
build or modify a web app, create the files under agent_workspace/<app-name>,
then run a direct verification command or background build and inspect its
output before claiming success. Use tools instead of merely describing code.
Prefer small, complete static HTML/CSS/JavaScript apps when no framework is
requested. Keep spoken replies brief, but summarize the actual files, commands,
and verification result. Never expose or request API-key contents, never write
credentials, and do not claim a task succeeded when a tool returned an error.
"""

