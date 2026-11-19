"""Provider-neutral OpenAI-compatible brain with local function-tool execution."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from .config import load_config, project_root
from .dev_workflow import is_software_request, verification_root
from .tools import dispatch, tool_definitions
from .secure_config import get as get_credential
from .provider_choice import openai_key

log = logging.getLogger("me3.brain")


def _ground_development_reply(content: str, path: str, *, tests_passed: bool,
                              serving_url: str | None,
                              github_published: bool) -> str:
    """Replace unsupported completion claims with the evidence we have."""
    low = str(content or "").lower()
    unsupported_tests = (not tests_passed and bool(re.search(
        r"\b(?:tests? (?:pass|passed|succeeded)|fully tested|fully working|"
        r"production.ready)\b", low)))
    unsupported_release = (not (serving_url or github_published) and bool(re.search(
        r"\b(?:deployed|published|live at)\b", low)))
    if unsupported_tests or unsupported_release:
        return (f"I changed the software at {path}, and its source checks passed. "
                "Project tests or live deployment are not verified yet.")
    return str(content or "").strip()


def _runtime_date_context() -> str:
    """Return fresh date context for every turn in the configured local zone."""
    try:
        now = datetime.now(ZoneInfo("Asia/Manila"))
    except Exception:  # noqa: BLE001 - keep the voice path alive on odd hosts
        now = datetime.now().astimezone()
    return (
        f"Current date: {now:%Y-%m-%d} ({now:%A}), "
        f"timezone: Asia/Manila ({now.tzname() or 'local'})."
    )


def _fdbench_mark(stage: str, **fields) -> None:
    """Emit an optional course-benchmark mark without coupling live runtime."""
    try:
        from fdbench_marks import mark
    except Exception:  # noqa: BLE001 - benchmark support is optional
        return
    mark(stage, **fields)


def _direct_me2_request(text: str) -> tuple[str, dict] | None:
    """Route clear ME2-style voice commands to their named tools.

    Qwen remains the general-purpose planner.  This small deterministic guard
    exists for short, unambiguous appliance/media commands where a repeated
    voice history can otherwise make a model answer with a generic greeting
    instead of emitting a function call.  It never runs when ``allow_tools``
    is false, so the clean benchmark path is unchanged.
    """
    raw = " ".join(str(text or "").split()).strip()
    low = raw.lower()
    if not raw or low in {"stop processing", "skip", "steer"}:
        return None

    # Music commands always go through the Spotify player adapter.  Browser
    # search URLs are not treated as completed playback.
    play = re.search(r"\b(?:play|start)\b", low)
    play_tail = low[play.end():].strip() if play else ""
    if play and (
        any(word in low for word in ("music", "song", "track", "album", "playlist", "spotify", "youtube"))
        or re.search(r"\bby\b", play_tail)
    ):
        query = raw[play.end():].strip(" .,!?:")
        query = re.sub(
            r"^(?:(?:some|the|a|an)\s+)?(?:music|songs?|tracks?)\b",
            "", query, flags=re.IGNORECASE,
        ).strip(" .,!?:")
        query = re.sub(
            r"\s+(?:on|in|using|from)\s+(?:spotify|youtube(?:\s+music)?)\s*$",
            "", query, flags=re.IGNORECASE,
        ).strip(" .,!?:")
        return "play_music", {"query": query or "music", "service": "spotify"}

    # Media controls are deliberately checked before generic search phrases.
    media = {
        "pause": "pause", "stop music": "stop", "stop the music": "stop",
        "skip": "skip", "next track": "skip", "next song": "skip",
        "volume up": "volume_up", "louder": "volume_up",
        "volume down": "volume_down", "quieter": "volume_down",
    }
    for marker, action in media.items():
        if marker in low:
            return "media_control", {"action": action}

    # Lights and dim/color lights share one tool and the same simulated bulb.
    if re.search(r"\blights?\b", low):
        action = None
        if re.search(r"\b(?:turn|switch)\s+on\b|\blights?\s+on\b", low):
            action = "on"
        elif re.search(r"\b(?:turn|switch)\s+off\b|\blights?\s+off\b", low):
            action = "off"
        elif any(word in low for word in ("dim", "brightness", "percent")):
            action = "dim"
        elif "color" in low or "colour" in low:
            action = "color"
        if action:
            value: int | str | None = None
            if action == "dim":
                match = re.search(r"(\d{1,3})\s*%?", low)
                if match:
                    value = int(match.group(1))
            elif action == "color":
                match = re.search(
                    r"\b(?:color|colour)\s+(?:to\s+)?([a-z]+)", low)
                value = match.group(1) if match else "warm white"
            return "control_lights", {"action": action, "room": "all", "value": value}

    # Thermostat/temperature commands.
    if "thermostat" in low or re.search(r"\btemperature\b", low):
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:degrees?|°c)?\b", low)
        if match:
            return "set_thermostat", {
                "temperature_c": float(match.group(1)), "room": "home"
            }

    # Timers and alarms use the existing validated local lifecycle tools.
    if "timer" in low:
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)",
            low,
        )
        if match:
            amount = float(match.group(1))
            unit = match.group(2)
            factor = 3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1
            return "set_timer", {"seconds": amount * factor, "label": "voice timer"}

    if "alarm" in low:
        match = re.search(
            r"\b(?:at|for)\s+(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)",
            raw, flags=re.IGNORECASE,
        )
        if match:
            return "set_alarm", {"time_text": match.group(1), "label": "voice alarm"}

    # A strict ISO-date form keeps calendar scheduling available when Qwen is
    # offline. Natural-language date interpretation remains with the model.
    meeting = re.match(
        r"^(?:please\s+)?(?:schedule|book|create|set up)\s+(?:a\s+)?meeting"
        r"(?:\s+(?:called|titled))?\s+(.+?)\s+(?:on|for)\s+"
        r"(\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})"
        r"(?:\s+(?:to|until)\s+(\d{1,2}:\d{2}))?$",
        raw, flags=re.IGNORECASE)
    if meeting:
        from datetime import datetime, timedelta
        title, date_text, start_text, end_text = meeting.groups()
        start = datetime.fromisoformat(f"{date_text}T{start_text}")
        end = (datetime.fromisoformat(f"{date_text}T{end_text}")
               if end_text else start + timedelta(minutes=30))
        return "schedule_meeting", {
            "title": title.strip(), "start": start.isoformat(),
            "end": end.isoformat(), "timezone": "Asia/Manila",
            "create_meet": True,
        }

    # Reminders and lists.
    if "reminder" in low or "remind me" in low:
        if re.search(r"\b(?:what are|list|show)\b.*\breminders?\b", low):
            return "list_reminders", {"include_completed": False}
        match = re.search(
            r"\bremind me to\s+(.+?)\s+in\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
            raw, flags=re.IGNORECASE)
        if match:
            amount = float(match.group(2))
            unit = match.group(3).lower()
            factor = (86400 if unit.startswith("d") else 3600 if unit.startswith("h")
                      else 60 if unit.startswith("m") else 1)
            return "create_reminder", {
                "text": match.group(1).strip(), "delay_seconds": amount * factor}
        match = re.search(
            r"\bremind me to\s+(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)\s*$",
            raw, flags=re.IGNORECASE)
        if match:
            return "create_reminder", {
                "text": match.group(1).strip(), "at_text": match.group(2).strip()}
        match = re.search(r"\bremind me to\s+(.+)$", raw, flags=re.IGNORECASE)
        if match:
            return "create_reminder", {"text": match.group(1).strip()}

    # Direct, complete communication commands execute through configured
    # provider/device adapters. Missing details are left for Qwen to clarify.
    if re.match(r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:call|phone|dial)\s+", low):
        match = re.match(
            r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:call|phone|dial)\s+(.+)$",
            raw, flags=re.IGNORECASE)
        if match:
            contact = re.sub(r"\s+(?:on|via|using)\s+.*$", "", match.group(1), flags=re.IGNORECASE)
            contact = re.sub(r"\s+please\s*$", "", contact, flags=re.IGNORECASE)
            return "call_contact", {"contact": contact.strip(), "confirm": True}
    if re.search(r"\b(?:send|text|message)\b", low):
        contact_match = re.search(
            r"\bto\s+(.+?)(?=\s+\b(?:saying|that says|with (?:the )?message|message is)\b|[,;:]|$)",
            raw, flags=re.IGNORECASE)
        body_match = re.search(
            r"\b(?:saying|that says|with (?:the )?message|message is)\s*[:,-]?\s*(.+)$",
            raw, flags=re.IGNORECASE)
        if contact_match and body_match:
            return "send_message", {
                "contact": contact_match.group(1).strip(),
                "body": body_match.group(1).strip(),
                "channel": "telegram", "confirm": True,
            }

    # Questions/search/weather.  Open-ended questions remain with Qwen unless
    # they match one of these clear, side-effect-free tool intents.
    if "weather" in low:
        match = re.search(r"\b(?:in|at|for)\s+(.+)$", raw, flags=re.IGNORECASE)
        return "get_weather", {"location": (match.group(1).strip() if match else "Manila")}
    if re.search(r"\b(?:what(?:'s| is) the time|what time is it|current time)\b", low):
        return "get_time", {}
    if re.search(r"\b(?:search|look up|find)\b", low):
        query = re.sub(r"^.*?\b(?:search|look up|find)\b\s*", "", raw, flags=re.IGNORECASE).strip(" .,!?:")
        if query:
            return "web_search", {"query": query, "max_results": 5}

    return None


def _github_confirmation_context(history: list[dict] | None) -> bool:
    """Return whether the latest assistant reply proposed a GitHub action."""
    assistant_text = next((str(item.get("content") or "").lower()
                           for item in reversed(history or [])
                           if item.get("role") == "assistant"), "")
    return (
        bool(re.search(r"\b(?:shall i|would you like|should i|please confirm|explicit confirmation)\b",
                       assistant_text))
        and bool(re.search(r"\b(?:publish|deploy|github|pages|website|site)\b",
                           assistant_text))
    )


def _github_approval_phrase(text: str) -> bool:
    """Recognize a short spoken approval without treating unrelated speech as approval."""
    # Voice ASR commonly preserves a comma or final period, for example
    # ``Yes, go ahead.``. Normalize punctuation before applying the narrow
    # approval grammar; the preceding GitHub confirmation-context check still
    # prevents an isolated approval from authorizing an old request.
    low = " ".join(re.sub(r"[^a-z0-9]+", " ",
                          str(text or "").lower()).split())
    accepted = {
        "yes", "yep", "yeah", "ok", "okay", "confirmed", "confirm",
        "approved", "go ahead", "yes go ahead", "yep go ahead",
        "yeah go ahead", "okay go ahead", "do it", "continue", "proceed",
        "publish", "publish it", "publish now", "publish the site",
        "publish the page", "deploy", "deploy it", "deploy now",
        "deploy the site", "deploy the page", "push", "push it", "push now",
        "go publish", "go publish it", "go deploy", "go deploy it",
        "go ahead and publish", "go ahead and publish it",
        "go ahead and deploy", "go ahead and deploy it",
        "proceed with publishing", "proceed with the deployment",
        "proceed with the deploy", "continue publishing",
        "continue with the deployment",
        "yes publish", "yes publish it", "yes deploy", "yes deploy it",
    }
    return low in accepted or bool(re.fullmatch(
        r"(?:yes|yeah|okay) (?:confirm|proceed|deploy|publish)(?: (?:yes|confirm|proceed|it|now)){0,3}",
        low,
    ))


def _github_site_from_context(history: list[dict] | None, current: str,
                              repo: str = "") -> str | None:
    """Find a validated agent_workspace site named in the recent conversation."""
    root = project_root().resolve() / "agent_workspace"
    context = " ".join(
        [str(current or "")] + [str(item.get("content") or "") for item in (history or [])[-12:]]
    )
    matches = re.findall(
        r"agent[_ ]workspace(?:[\\/][A-Za-z0-9_.-]+)+", context,
        flags=re.IGNORECASE,
    )
    candidates: list[Path] = []
    for raw in matches:
        clean = raw.rstrip(".,;:)]}\\\"'")
        relative = clean.replace("\\", "/").replace("agent workspace", "agent_workspace")
        candidate = (project_root() / relative).resolve()
        if not candidate.is_relative_to(root):
            continue
        if candidate.is_file() and candidate.name.lower() == "index.html":
            candidate = candidate.parent
        if candidate.is_dir() and (candidate / "index.html").is_file():
            candidates.append(candidate)
    if not candidates and (root / "index.html").is_file():
        candidates.append(root)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if repo:
        matching = [path for path in unique if path.name.casefold() == repo.casefold()]
        if len(matching) == 1:
            return matching[0].relative_to(project_root()).as_posix()
        named = (root / repo).resolve()
        if named.is_relative_to(root) and (named / "index.html").is_file():
            return named.relative_to(project_root()).as_posix()
    if len(unique) == 1:
        return unique[0].relative_to(project_root()).as_posix()
    return None


def _github_target_from_context(current: str, *, use_default: bool = True) -> dict[str, str] | None:
    """Use any explicitly named repository, or the convenience default.

    The configured target is only a fallback for requests that do not name an
    owner/repository. It is not an allowlist: the publisher validates and uses
    the repository named by the user, subject to the private credential's
    actual GitHub permissions.
    """
    text = str(current or "")
    match = re.search(
        r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
        text, flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b", text)
    if match and match.group(1).lower() not in {"agent_workspace", "github"}:
        owner, repo = match.groups()
        return {"owner": owner, "repo": repo, "branch": "main",
                "base_path": "", "pages_path": "/"}
    if not use_default:
        return None
    target = dict(load_config().get("github_pages", {}) or {})
    owner = str(target.get("owner") or "").strip()
    repo = str(target.get("repository") or "").strip()
    if not owner or not repo:
        return None
    return {
        "owner": owner,
        "repo": repo,
        "branch": str(target.get("branch") or "main"),
        "base_path": str(target.get("base_path") or ""),
        "pages_path": str(target.get("pages_path") or "/"),
    }


def _direct_github_pages_request(history: list[dict], text: str) -> tuple[str, dict] | None:
    """Route an unambiguous publish/approval turn to the real GitHub tool.

    This prevents the language model from merely narrating a GitHub action or
    losing a short follow-up such as ``Go ahead`` after it requested approval.
    """
    low = " ".join(str(text or "").lower().split())
    publish_intent = bool(re.search(
        r"\b(?:deploy|publish(?:ing|ed)?|push|release)\b", low))
    approval_intent = _github_approval_phrase(low) and _github_confirmation_context(history)
    if not publish_intent and not approval_intent:
        return None
    target = _github_target_from_context(text, use_default=False)
    if target is None:
        for item in reversed(history[-12:]):
            if item.get("role") == "assistant":
                proposed = _github_target_from_context(
                    str(item.get("content") or ""), use_default=False)
                if proposed:
                    target = proposed
                    break
    if target is None:
        target = _github_target_from_context(text)
    site_dir = _github_site_from_context(history, text, target["repo"] if target else "")
    if not site_dir or not target:
        return None
    return "deploy_github_pages", {
        "site_dir": site_dir,
        **target,
        "message": "Deploy website from ME3",
        "enable_pages": True,
        "create_if_missing": bool(re.search(
            r"\b(?:new repository|create (?:this |a |the )?repository)\b",
            " ".join(str(item.get("content") or "") for item in history[-12:]) + " " + text,
            flags=re.IGNORECASE,
        )),
    }


def _unverified_github_claim(text: str) -> bool:
    """Detect a model claim of a GitHub side effect without tool evidence."""
    low = " ".join(str(text or "").lower().split())
    if not re.search(r"\bgithub(?: pages|\.io)?\b|github\.com", low):
        return False
    if not re.search(
        r"\b(?:deployed|published|pushed|created (?:a )?(?:new )?repo|"
        r"repository (?:has been|was) created|live at)\b", low):
        return False
    return not bool(re.search(r"\b(?:not|never|cannot|can't|couldn't|unable|failed)\b", low))


_CURRENT_SENSITIVE_RE = re.compile(
    r"\b(?:current|currently|latest|recent|news|who won|election|"
    r"already happened|as of|this year|this month|today|now|president|"
    r"prime minister|ceo)\b",
    re.IGNORECASE,
)


def _web_search_query(history: list[dict] | None, current: str) -> str | None:
    """Choose when a turn must be grounded by the real web-search tool.

    This is intentionally conservative. Explicit search requests always run
    the tool. Questions involving current events or other time-sensitive facts
    also run it, including short follow-ups such as ``already happened`` that
    refer to the previous user question.
    """
    raw = " ".join(str(current or "").split()).strip()
    if not raw:
        return None
    low = raw.lower()
    explicit = bool(re.search(r"\b(?:search|look up|find|browse|web search)\b", low))
    question = bool("?" in raw or re.search(
        r"\b(?:who|what|when|where|which|how|is|are|did|does|tell me|check|verify)\b",
        low,
    ))
    current_sensitive = bool(_CURRENT_SENSITIVE_RE.search(low))

    # The runtime date is authoritative for a simple date question; do not
    # spend a network round searching for the date on the Pi.
    date_only = bool(re.search(
        r"\b(?:what(?:'s| is) (?:the )?(?:current )?date|today's date|"
        r"date today|what day is it)\b",
        low,
    )) and not bool(re.search(r"\b(?:news|election|latest|recent|who won)\b", low))
    if date_only and not explicit:
        return None

    if explicit:
        query = re.sub(
            r"^\s*(?:please\s+)?(?:search|look up|find|browse|web search)"
            r"\s*(?:the web\s*)?(?:for\s*)?",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip(" .,!?:")
        return (query or raw)[:500]

    if current_sensitive and question:
        return raw[:500]

    # Carry a short correction/follow-up back to the last user question that
    # was time-sensitive. This covers the observed ``already happened`` turn.
    if len(raw.split()) <= 12 and history:
        prior_users = [
            str(item.get("content") or "") for item in history
            if item.get("role") == "user" and str(item.get("content") or "").strip()
        ]
        if prior_users and _CURRENT_SENSITIVE_RE.search(prior_users[-1]):
            return f"{prior_users[-1]} {raw}"[:500]
    return None


def _direct_tool_reply(name: str, args: dict, result) -> str:
    """Turn a deterministic ME2 tool result into a short spoken response."""
    if not isinstance(result, dict):
        return "The requested command completed."
    if not result.get("ok"):
        if (name == "create_reminder"
                and result.get("error") == "provide delay_seconds or at_text"):
            return f"When should I remind you to {args.get('text', 'do that')}?"
        error = str(result.get("error") or "the tool reported an error").replace("_", " ")
        return f"I could not complete that command because {error}."
    if name == "play_music":
        artists = ", ".join(result.get("artists") or [])
        track = str(result.get("track") or args.get("query", "the requested music"))
        device = str(result.get("device") or "your Spotify player")
        return f"Started {track}{' by ' + artists if artists else ''} on {device}."
    if name == "media_control":
        return f"Spotify playback updated: {args.get('action', 'media control')}."
    if name == "control_lights":
        light = result.get("lights") or {}
        state = "on" if light.get("on") else "off"
        return f"The simulated light is {state} at {light.get('brightness', 100)} percent."
    if name == "set_thermostat":
        return f"The simulated thermostat is set to {result.get('temperature_c')} degrees Celsius."
    if name == "get_weather":
        return (f"The weather in {args.get('location', 'the requested location')} is "
                f"{result.get('description', 'available')} at {result.get('temperature_c', 'an unavailable')} degrees Celsius.")
    if name == "get_time":
        return f"The current local time is {result.get('time', 'available')}"
    if name == "web_search":
        rows = result.get("results") or []
        return f"I found {len(rows)} web result(s) for {args.get('query', 'your search')}."
    if name == "github_pages_health":
        if result.get("configured") and result.get("validated", True):
            return "The GitHub publishing credential is configured privately on the ME3 host."
        return "GitHub publishing is not configured on the ME3 host."
    if name == "deploy_github_pages":
        if result.get("ok"):
            return (f"The GitHub Pages site was deployed to {result.get('pages_url', 'the configured Pages URL')}. "
                    f"Commit {str(result.get('commit', 'completed'))[:12]} is recorded.")
        if result.get("partial_commit"):
            return ("The site files were committed to GitHub, but GitHub Pages could not be confirmed. "
                    "The repository commit was not rolled back.")
        if result.get("error") == "github_not_configured":
            return "The GitHub publishing credential is not configured privately on the ME3 host."
        return "I could not deploy the GitHub Pages site: " + str(result.get("error", "unknown error"))
    if name == "set_timer":
        return "The timer is set. ME3 will announce it when it finishes."
    if name == "set_alarm":
        return "The alarm is set. ME3 will announce it at the scheduled time."
    if name == "create_reminder":
        if result.get("calendar_synced") is False:
            return "I set a local reminder. It is not synced to Google Calendar yet."
        return "The reminder is set. ME3 will announce it when due."
    if name == "list_reminders":
        return f"There are {len(result.get('reminders') or [])} active reminder(s)."
    if name == "call_contact":
        if result.get("ok"):
            return f"Calling {result.get('contact', args.get('contact', 'the contact'))} now."
        return "I could not place the call: " + str(result.get("error") or "the Android phone did not confirm the call.")
    if name == "send_message":
        if result.get("ok"):
            return f"Telegram message sent to {args.get('contact', 'the recipient')}."
        return "I could not send the Telegram message: " + str(result.get("error") or "the provider did not confirm delivery.")
    if name == "schedule_meeting":
        if result.get("ok"):
            meet = result.get("meet_url")
            return "Meeting scheduled in Google Calendar." + (f" Google Meet link: {meet}." if meet else "")
        return "I could not schedule the meeting: " + str(result.get("error") or "the calendar did not confirm it.")
    return "The requested command completed successfully."


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
        "custom_vllm": "custom_vllm",
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
        "custom_vllm": dict(llm.get("remote", {}) or {}),
    }
    for provider in candidates:
        provider_cfg = configs[provider]
        key_env = provider_cfg.get("api_key_env", "")
        if selection != "auto" or (openai_key() if key_env == "OPENAI_API_KEY"
                                    else get_credential(key_env)):
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
        self.max_development_steps = max(
            self.max_tool_steps,
            min(int(cfg.get("max_development_steps", 48)), 64),
        )
        self.history_messages = max(
            8, min(int(cfg.get("history_messages", 16)), 32)
        )
        self.allow_tools = bool(cfg.get("allow_tools", True))
        self.api_key = str(cfg.get("api_key") or (
            openai_key() if self.api_key_env == "OPENAI_API_KEY"
            else get_credential(self.api_key_env)))
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
        _fdbench_mark(
            "llm_request",
            messages=len(messages),
            chars=sum(len(str(m.get("content") or "")) for m in messages),
        )
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
        first_token_marked = False
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
                    if not first_token_marked and delta.get("content"):
                        _fdbench_mark("llm_first_token")
                        first_token_marked = True
                    content_parts.append(delta["content"])
                    self._progress("response_stream", "Receiving streamed response")
                if isinstance(delta.get("reasoning_content"), str):
                    reasoning_parts.append(delta["reasoning_content"])
                for fallback_index, call in enumerate(delta.get("tool_calls") or []):
                    if not isinstance(call, dict):
                        continue
                    if not first_token_marked:
                        _fdbench_mark("llm_first_token")
                        first_token_marked = True
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

    @staticmethod
    def _explicit_action_request(tool_name: str, text: str,
                                 history: list[dict] | None = None) -> bool:
        low = " ".join(str(text or "").lower().split())
        explicit = False
        if tool_name in {"send_message", "send_telegram_message", "send_email"}:
            explicit = bool(re.match(
                r"^(?:please\s+)?(?:(?:can|could)\s+you\s+)?(?:send|text|message|email|telegram)\b", low))
        elif tool_name == "call_contact":
            explicit = bool(re.match(
                r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:call|phone|dial)\b", low))
        elif tool_name == "schedule_meeting":
            explicit = (bool(re.search(r"\b(?:schedule|book|set up|create)\b", low))
                        and bool(re.search(r"\b(?:meeting|meet|appointment)\b", low)))
        elif tool_name == "deploy_github_pages":
            explicit = bool(re.search(
                r"\b(?:deploy|publish(?:ing|ed)?|push|release)\b", low)) and bool(re.search(
                r"\b(?:github(?:\.io| pages)?|pages site|website|site)\b", low))
        if explicit:
            return True

        if (tool_name == "deploy_github_pages"
                and _github_approval_phrase(low)
                and _github_confirmation_context(history)):
            return True

        # Permit a short answer to an agent clarification only when the most
        # recent user turn clearly requested this same action and the agent
        # then asked for its missing details. This keeps multi-turn voice use
        # smooth without allowing an old request to authorize a later action.
        history = history or []
        user_indexes = [i for i, item in enumerate(history)
                        if item.get("role") == "user"]
        if not user_indexes or len(low) > 220:
            return False
        prior_index = user_indexes[-1]
        prior_text = str(history[prior_index].get("content") or "")
        prior_low = " ".join(prior_text.lower().split())
        assistant_followup = " ".join(
            str(item.get("content") or "") for item in history[prior_index + 1:]
            if item.get("role") == "assistant").lower()
        if tool_name in {"send_message", "send_telegram_message", "send_email"}:
            requested = bool(re.search(r"\b(?:send|text|message|email|telegram)\b", prior_low))
            asked = bool(re.search(r"\b(?:what|which|who).{0,100}\b(?:message|say|content|recipient|email|send)\b", assistant_followup))
        elif tool_name == "call_contact":
            requested = bool(re.match(r"^(?:please\s+)?(?:can you\s+|could you\s+)?(?:call|phone|dial)\b", prior_low))
            asked = bool(re.search(r"\b(?:who|which|what).{0,80}\b(?:call|contact|number)\b", assistant_followup))
        elif tool_name == "schedule_meeting":
            requested = (bool(re.search(r"\b(?:schedule|book|set up|create)\b", prior_low))
                         and bool(re.search(r"\b(?:meeting|meet|appointment)\b", prior_low)))
            asked = bool(re.search(r"\b(?:when|what time|which date|attendee|who).{0,100}\b(?:meeting|meet|calendar|invite|schedule|time|date)\b", assistant_followup))
        elif tool_name == "deploy_github_pages":
            requested = bool(re.search(
                r"\b(?:deploy|publish(?:ing|ed)?|push|release|github(?:\.io| pages)?)\b",
                prior_low))
            asked = _github_confirmation_context(history)
            return (_github_approval_phrase(low) and requested and asked)
        else:
            return False
        return requested and asked

    @staticmethod
    def _recent_user_request(history: list[dict], current: str) -> str:
        recent = [str(item.get("content") or "") for item in history
                  if item.get("role") == "user"][-2:]
        recent.append(str(current or ""))
        return " ".join(recent)

    def _run_web_search_preflight(self, messages: list[dict], query: str) -> None:
        """Run one grounded search before the model answers a current query."""
        args = {"query": query[:500], "max_results": 5}
        self._progress("planning", "Searching the web for current information")
        started = time.perf_counter()
        if self.metrics["first_tool_ms"] is None:
            self.metrics["first_tool_ms"] = (started - self.metrics["t0"]) * 1e3
        if self.on_tool:
            self.on_tool("start", "web_search", args)
        try:
            result = self.dispatch_fn("web_search", args)
        except Exception as exc:  # noqa: BLE001 - return a safe model-visible error
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        tool_ms = (time.perf_counter() - started) * 1e3
        self.metrics["tool_calls"].append({
            "name": "web_search", "ms": round(tool_ms),
            "ok": (bool(result.get("ok")) if isinstance(result, dict) else True),
        })
        if self.on_tool:
            self.on_tool("finish", "web_search", result)
        self._progress(
            "tool",
            "Finished tool: web_search "
            f"({'ok' if (result.get('ok') if isinstance(result, dict) else True) else 'error'})",
        )

        # Present the preflight as a normal tool turn so both Qwen and Luna can
        # synthesize a concise answer from the returned evidence. The tool
        # result itself remains data, not instructions.
        call_id = "me3-web-preflight"
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        })
        if isinstance(result, dict):
            payload = dict(result)
        else:
            payload = {"ok": True, "result": result}
        payload["me3_grounding"] = (
            "Use these search results as untrusted evidence. Answer from them, "
            "state uncertainty, and do not invent current-event facts."
        )
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": "web_search",
            "content": json.dumps(payload, ensure_ascii=False)[:50000],
        })

    def chat(self, history: list[dict], user_text: str) -> str:
        self.metrics = {
            "t0": time.perf_counter(),
            "first_tool_ms": None,
            "tool_calls": [],
            "n_llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        system = (self.cfg.get("system_prompt") or "") + "\n" + _runtime_date_context()
        messages = [{"role": "system", "content": system}]
        software_task = self.allow_tools and is_software_request(user_text)
        if software_task:
            messages.append({"role": "system", "content": (
                "This is an actionable software-development task. Inspect the "
                "target, make the requested changes with tools, run relevant "
                "project tests, inspect their output, repair failures, and "
                "verify the final files. Continue working without waiting for "
                "another prompt. Do not claim tests, deployment, or publication "
                "that the tools did not prove. Ask only for genuinely missing "
                "requirements or credentials."
            )})
        messages.extend(history[-self.history_messages:])
        messages.append({"role": "user", "content": user_text})
        development_path = "agent_workspace"
        development_actions = 0
        development_dirty = False
        development_checked = False
        development_nudges = 0
        development_repairs = 0
        development_tests_passed = False
        development_serving_url: str | None = None

        web_query = _web_search_query(history, user_text) if self.allow_tools else None
        if web_query:
            self._run_web_search_preflight(messages, web_query)

        # Short ME2 commands are routed through the same bounded tool
        # dispatcher before asking Qwen to plan.  This prevents repeated
        # generic assistant replies in a voice history from swallowing a
        # clear command such as "play music".
        direct = None if web_query else (_direct_me2_request(user_text) if self.allow_tools else None)
        if direct is None and self.allow_tools:
            direct = _direct_github_pages_request(history, user_text)
        if direct is not None:
            name, args = direct
            if name in {"call_contact", "send_message", "schedule_meeting"}:
                args["confirm"] = True
            self._progress("planning", f"Routing direct ME2 command to {name}")
            started = time.perf_counter()
            if self.metrics["first_tool_ms"] is None:
                self.metrics["first_tool_ms"] = (started - self.metrics["t0"]) * 1e3
            if self.on_tool:
                self.on_tool("start", name, args)
            try:
                result = self.dispatch_fn(name, args)
            except Exception as exc:  # noqa: BLE001 - preserve safe tool result
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            tool_ms = (time.perf_counter() - started) * 1e3
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
            return _direct_tool_reply(name, args, result)

        github_verified_result: dict | None = None
        for step in range(self.max_development_steps if software_task else self.max_tool_steps):
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
                if software_task and development_actions == 0:
                    if development_nudges == 0:
                        development_nudges += 1
                        self._progress("planning", "Continuing software task with tools")
                        messages.append({"role": "system", "content": (
                            "No development tool has run yet. Perform the "
                            "requested work now; a plan or promise alone is "
                            "not completion. If you cannot proceed, state the "
                            "specific blocker."
                        )})
                        continue
                    return "I have not changed or tested the software yet."
                if software_task and development_dirty:
                    args = {"path": development_path, "max_files": 300}
                    self._progress("verification", f"Checking software files: {development_path}")
                    if self.on_tool:
                        self.on_tool("start", "verify_project", args)
                    started = time.perf_counter()
                    try:
                        check = self.dispatch_fn("verify_project", args)
                    except Exception as exc:  # noqa: BLE001
                        check = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    self.metrics["tool_calls"].append({
                        "name": "verify_project",
                        "ms": round((time.perf_counter() - started) * 1e3),
                        "ok": bool(check.get("ok")) if isinstance(check, dict) else False,
                    })
                    if self.on_tool:
                        self.on_tool("finish", "verify_project", check)
                    call_id = f"me3-dev-verify-{step}"
                    messages.append({"role": "assistant", "content": "", "tool_calls": [{
                        "id": call_id, "type": "function", "function": {
                            "name": "verify_project",
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }]})
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "name": "verify_project",
                                     "content": json.dumps(check, ensure_ascii=False)[:12000]})
                    if isinstance(check, dict) and check.get("ok"):
                        development_dirty = False
                        development_checked = True
                        messages.append({"role": "system", "content": (
                            "The automatic source check passed. It covers syntax "
                            "and local static assets only. State project-specific "
                            "test results only if a test command actually ran. "
                            "Finish with a short, evidence-based result."
                        )})
                        continue
                    if development_repairs >= 2:
                        errors = check.get("errors") if isinstance(check, dict) else None
                        detail = "; ".join(str(x) for x in (errors or [])[:2])
                        return ("The software files were changed, but verification "
                                f"still fails: {detail or (check.get('error') if isinstance(check, dict) else 'unknown error')}.")
                    development_repairs += 1
                    messages.append({"role": "system", "content": (
                        "The automatic source check failed. Inspect the reported "
                        "errors, repair the files with tools, run relevant tests, "
                        "and verify again. Do not claim completion yet."
                    )})
                    continue
                if isinstance(content, str) and content.strip():
                    if github_verified_result is not None:
                        self._progress(
                            "response",
                            "Returning the verified GitHub publish result",
                        )
                        return _direct_tool_reply(
                            "deploy_github_pages", {}, github_verified_result)
                    if _unverified_github_claim(content):
                        self._progress(
                            "response",
                            "Blocked unverified GitHub completion claim",
                        )
                        return (
                            "I cannot confirm a GitHub change because the GitHub "
                            "publish tool did not complete. I can continue only "
                            "after the tool reports a verified commit and Pages URL."
                        )
                    if software_task and development_checked:
                        return _ground_development_reply(
                            content, development_path,
                            tests_passed=development_tests_passed,
                            serving_url=development_serving_url,
                            github_published=github_verified_result is not None,
                        )
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
                    if name in {"send_message", "send_telegram_message", "send_email",
                                "call_contact", "schedule_meeting"}:
                        args["confirm"] = True
                    if (name == "deploy_github_pages"
                            and not re.search(r"\b(?:deploy|publish|push|release)\b",
                                              user_text, flags=re.IGNORECASE)
                            and not (_github_approval_phrase(user_text)
                                     and _github_confirmation_context(history))):
                        result = {"ok": False, "error": "github_publish_request_required"}
                    else:
                        result = self.dispatch_fn(name, args)
                    if software_task and name in {
                            "write_file", "run_command", "start_background_task"}:
                        development_actions += 1
                        development_dirty = True
                        development_checked = False
                        if name == "write_file" and args.get("path"):
                            development_path = verification_root(str(args["path"]))
                            development_tests_passed = False
                        elif development_path == "agent_workspace" and args.get("cwd"):
                            development_path = verification_root(
                                str(args["cwd"]), is_directory=True)
                        if (name == "run_command" and isinstance(result, dict)
                                and result.get("ok") and re.search(
                                    r"\b(?:pytest|npm\s+test|yarn\s+test|pnpm\s+test|"
                                    r"node\s+--test|go\s+test|cargo\s+test|dotnet\s+test)\b",
                                    str(args.get("command") or ""), re.IGNORECASE)):
                            development_tests_passed = True
                    elif (software_task and name == "start_web_server"
                          and isinstance(result, dict) and result.get("ok")):
                        development_serving_url = str(result.get("url") or "") or None
                    elif software_task and name == "verify_project" and isinstance(result, dict):
                        development_checked = bool(result.get("ok"))
                        development_dirty = not development_checked
                        development_path = str(args.get("path") or development_path)
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
                    if (name == "deploy_github_pages"
                            and isinstance(result, dict)
                            and result.get("ok")
                            and result.get("commit")
                            and result.get("pages_url")):
                        # The provider's follow-up prose is not authoritative;
                        # retain the verified API result for the final reply.
                        github_verified_result = dict(result)
                    log.info("tool result: %s ok=%s", name,
                             result.get("ok") if isinstance(result, dict) else True)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", f"call-{step}-{name}"),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False)[:50000],
                })
        if github_verified_result is not None:
            self._progress("response", "Returning the verified GitHub publish result")
            return _direct_tool_reply(
                "deploy_github_pages", {}, github_verified_result)
        if software_task and development_actions:
            if development_dirty:
                return ("I changed the software files, but the task reached its "
                        "execution limit before I could verify them. The task "
                        "remains incomplete.")
            if development_checked:
                return ("The software files passed source checks. Project-specific "
                        "behavior still needs the test results from this task.")
        if software_task:
            return "I have not changed or tested the software yet."
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
            "or configure OpenAI through the private ME3 setup page")
    provider_cfg["system_prompt"] = system_prompt
    provider_cfg["name"] = (
        "qwen-tools" if provider == "qwen" else "gpt-4.1-mini-tools")
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
For GitHub, use `github_repositories` to discover repositories visible to the
private credential and `github_repository_access` to check a named target.
`deploy_github_pages` can publish a validated site to any requested
owner/repository that the credential can write, and create a missing repository
when the user requested a new one. A clear user request to publish is enough;
do not ask for a second confirmation or mention an approval keyword. The
configured target is only a convenience default, not an allowlist. Do not
claim that a repository was created, a commit was
pushed, or a site is live unless the corresponding GitHub tool returned
`ok=true` with its verification fields. A short reply
such as "yes", "go ahead", "proceed", or "publish it" refers to the latest
proposed GitHub action when the immediately preceding assistant turn described
that action. The runtime date is supplied separately on every turn;
use it for today's date. For current events, elections, latest or recent facts,
and any other date-sensitive claim, use the `web_search` tool before answering.
Do not rely on model memory for those claims. If search fails or does not provide
enough evidence, say that clearly instead of guessing.
"""

