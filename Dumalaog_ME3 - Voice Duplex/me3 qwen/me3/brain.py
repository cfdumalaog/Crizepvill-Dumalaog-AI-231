"""Provider-neutral OpenAI-compatible brain with local function-tool execution."""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable

import requests

from .tools import dispatch, tool_definitions

log = logging.getLogger("me3.brain")


def resolve_llm_config(cfg: dict, selection: str = "auto") -> tuple[str | None, dict | None]:
    """Resolve a user-facing brain choice to a configured provider.

    ``remote`` remains the backwards-compatible name for Qwen/vLLM.  ``luna``
    selects the official OpenAI GPT-5.6 Luna backend.  Auto mode preserves the
    established Qwen-first behavior and uses Luna only when Qwen is not
    configured and ``OPENAI_API_KEY`` is available.
    """
    llm = cfg.get("llm", {}) or {}
    aliases = {
        "remote": "qwen",
        "qwen": "qwen",
        "openai": "luna",
        "luna": "luna",
    }
    if selection == "auto":
        candidates = ("qwen", "luna")
    else:
        provider = aliases.get(selection)
        if provider is None:
            raise ValueError(f"unknown brain selection: {selection}")
        candidates = (provider,)

    configs = {
        "qwen": dict(llm.get("remote", {}) or {}),
        "luna": dict(llm.get("openai", {}) or {}),
    }
    for provider in candidates:
        provider_cfg = configs[provider]
        key_env = provider_cfg.get("api_key_env", "")
        if selection != "auto" or os.environ.get(key_env, "").strip():
            if provider_cfg:
                provider_cfg["provider"] = provider
                return provider, provider_cfg
    return None, None


class ToolBrain:
    """Synchronous OpenAI-compatible tool loop for Qwen or GPT-5.6 Luna."""

    def __init__(self, cfg: dict, system_prompt: str,
                 on_tool: Callable | None = None,
                 dispatch_fn: Callable | None = None,
                 on_progress: Callable | None = None):
        self.cfg = cfg
        self.base_url = cfg.get("base_url", "").rstrip("/")
        self.model = cfg.get("model", "")
        self.provider = cfg.get("provider", "openai-compatible")
        self.name = cfg.get("name") or f"{self.provider}-tools"
        self.api_key_env = cfg.get("api_key_env", "")
        self.temperature = cfg.get("temperature")
        # Qwen3 exposes a provider-specific switch for the hidden reasoning
        # channel.  Voice turns must stay in fast mode: reasoning tokens add
        # latency and are never appropriate for the spoken response.
        self.fast_mode = bool(cfg.get("fast_mode", False))
        self.thinking = cfg.get("thinking")
        # Hidden reasoning is deliberately disabled for the functional voice
        # path. The UI may show safe phase summaries, never chain-of-thought.
        self.stream = bool(cfg.get("stream", False))
        # 256 (the old default) is too small for multi-file web-app builds:
        # a single write_file call's content alone can exceed it, which makes
        # Qwen truncate mid-JSON and the tool loop dies with
        # invalid_tool_arguments.  4096 covers realistic multi-tool turns.
        self.max_tokens = cfg.get("max_tokens")
        self.max_completion_tokens = cfg.get("max_completion_tokens")
        self.timeout_s = float(cfg.get("timeout_s", 60))
        # Allow realistic multi-file/web-app turns. If the model still needs
        # more rounds, chat() executes this final batch and asks for a safe
        # no-tool summary instead of speaking an internal limit error.
        self.max_tool_steps = max(1, min(int(cfg.get("max_tool_steps", 24)), 32))
        self.history_messages = max(
            8, min(int(cfg.get("history_messages", 16)), 32)
        )
        self.allow_tools = bool(cfg.get("allow_tools", True))
        self.api_key = os.environ.get(self.api_key_env, "")
        self.on_tool = on_tool
        # Progress is deliberately a safe summary callback.  It may describe
        # the current phase/tool, but it must never expose provider hidden
        # chain-of-thought or prompt contents.
        self.on_progress = on_progress
        # The default is the real ME3 tool dispatcher.  Local UI/demo surfaces
        # can inject a narrower dispatcher so provider side effects stay off
        # while the model and project tools are exercised normally.
        self.dispatch_fn = dispatch_fn or dispatch
        if not self.api_key:
            raise RuntimeError(
                f"{self.api_key_env} is required for the {self.model} tool brain")

    def _progress(self, stage: str, task: str) -> None:
        if self.on_progress:
            self.on_progress(stage, task)

    def _request(self, messages: list[dict], include_tools: bool = True) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.stream:
            body["stream"] = True
        if self.provider == "qwen" and (self.fast_mode or self.thinking is False):
            # vLLM's OpenAI-compatible Qwen3 template consumes this field.
            # Do not send OpenAI's reasoning_effort field to Qwen; it is not a
            # portable OpenAI-compatible parameter and may be rejected.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.max_completion_tokens is not None:
            body["max_completion_tokens"] = int(self.max_completion_tokens)
        elif self.max_tokens is not None:
            body["max_tokens"] = int(self.max_tokens)
        if include_tools and self.allow_tools:
            body["tools"] = tool_definitions()
            body["tool_choice"] = "auto"
        # The VPN path to the Qwen endpoint intermittently stalls on long
        # multi-tool turns past the request timeout (observed: 2 of 4 long
        # turns on 2026-10-31). One retry on pure transport failures keeps
        # the voice-to-tool path reliable; any other error surfaces at once.
        for _attempt in (1, 2):
            try:
                self._progress("thinking", f"Waiting for {self.name} model inference")
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json=body,
                    timeout=self.timeout_s,
                    stream=self.stream,
                )
                break
            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError):
                if _attempt == 2:
                    raise
                log.warning(
                    "%s: chat request stalled (no response within %.0f s); "
                    "retrying once", self.name, self.timeout_s,
                )
                time.sleep(2.0)
        response.raise_for_status()
        if self.stream:
            return self._read_stream_response(response)
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"{self.provider} returned no chat choice")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError(f"{self.provider} returned an invalid message")
        usage = payload.get("usage") or {}
        return message, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }

    def _read_stream_response(self, response) -> tuple[dict, dict]:
        """Assemble an OpenAI-compatible SSE response without exposing reasoning."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        usage: dict = {}
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
                line = str(line).strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("%s: ignored malformed stream event", self.name)
                    continue
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices or not isinstance(choices[0], dict):
                    continue
                delta = choices[0].get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                    self._progress("response_stream", "Receiving streamed response")
                if isinstance(delta.get("reasoning_content"), str):
                    reasoning_parts.append(delta["reasoning_content"])
                for fallback_index, call in enumerate(delta.get("tool_calls") or []):
                    if not isinstance(call, dict):
                        continue
                    index = int(call.get("index", fallback_index))
                    current = tool_calls.setdefault(index, {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })
                    if call.get("id"):
                        current["id"] = call["id"]
                    if call.get("type"):
                        current["type"] = call["type"]
                    function = call.get("function") or {}
                    if function.get("name"):
                        current["function"]["name"] += str(function["name"])
                    if function.get("arguments"):
                        current["function"]["arguments"] += str(function["arguments"])
        finally:
            close = getattr(response, "close", None)
            if close:
                close()
        message = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        if reasoning_parts:
            # Keep this only for diagnostics; chat() will never speak it.
            message["reasoning_content"] = "".join(reasoning_parts)
        return message, {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }

    @staticmethod
    def _tool_calls(message: dict) -> list[dict]:
        calls = message.get("tool_calls") or []
        return calls if isinstance(calls, list) else []

    def _summarize_without_tools(self, messages: list[dict]) -> str:
        """Finish a long tool turn without exposing the internal budget."""
        self._progress("response", "Summarizing completed task actions")
        summary_messages = list(messages)
        summary_messages.append({
            "role": "system",
            "content": (
                "Summarize the completed tool actions for the user. Do not call "
                "any tools. Use a formal, professional tone, state what was "
                "completed, and identify any remaining action in one to three "
                "short spoken sentences. Never mention a tool-call budget or "
                "internal limit."
            ),
        })
        message, usage = self._request(summary_messages, include_tools=False)
        self.metrics["n_llm_calls"] += 1
        self.metrics["prompt_tokens"] += usage["prompt_tokens"]
        self.metrics["completion_tokens"] += usage["completion_tokens"]
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return "I completed the available actions. I can continue from the current state if you require another step."

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
        messages.extend(history[-self.history_messages:])
        messages.append({"role": "user", "content": user_text})

        for step in range(self.max_tool_steps):
            message, usage = self._request(messages, include_tools=True)
            self.metrics["n_llm_calls"] += 1
            self.metrics["prompt_tokens"] += usage["prompt_tokens"]
            self.metrics["completion_tokens"] += usage["completion_tokens"]
            calls = self._tool_calls(message) if self.allow_tools else []
            if calls:
                self._progress("planning", f"Model selected {len(calls)} tool call(s)")
            else:
                self._progress("response", "Preparing the final response")
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
                if message.get("reasoning_content"):
                    self._progress("response", "Model returned no spoken response")
                raise RuntimeError(
                    f"{self.provider} returned neither tool calls nor spoken text")
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
                    self._progress("tool", f"Running tool: {name}")
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
                    self._progress(
                        "tool",
                        f"Finished tool: {name} ({'ok' if (result.get('ok') if isinstance(result, dict) else True) else 'error'})",
                    )
                    log.info("tool result: %s ok=%s", name,
                             result.get("ok") if isinstance(result, dict) else True)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", f"call-{step}-{name}"),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False)[:50000],
                })
        return self._summarize_without_tools(messages)


def build_tool_brain(cfg: dict, selection: str, system_prompt: str,
                     on_tool: Callable | None = None,
                     dispatch_fn: Callable | None = None,
                     on_progress: Callable | None = None,
                     allow_tools: bool = True) -> ToolBrain:
    """Build the selected tool brain while keeping provider secrets in env vars."""
    provider, provider_cfg = resolve_llm_config(cfg, selection)
    if provider_cfg is None:
        raise RuntimeError(
            "no inference provider is configured; set VLLM_API_KEY for Qwen "
            "or OPENAI_API_KEY for GPT-5.6 Luna")
    provider_cfg["system_prompt"] = system_prompt
    provider_cfg["name"] = (
        "qwen-tools" if provider == "qwen" else "gpt-5.6-luna-medium-tools")
    provider_cfg["allow_tools"] = bool(allow_tools)
    return ToolBrain(provider_cfg, system_prompt, on_tool=on_tool,
                     dispatch_fn=dispatch_fn, on_progress=on_progress)


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

