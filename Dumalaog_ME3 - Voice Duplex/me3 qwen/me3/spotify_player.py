"""Spotify Web API player used for real voice-controlled playback.

The API controls the user's active Spotify Connect player.  It never reports
that playback started unless Spotify accepted the request and an active player
can be observed.  Tokens stay in the ME3 service process and are never logged.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

API = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"
_TOKEN_LOCK = threading.Lock()
_TOKEN: tuple[str, float] | None = None
_OAUTH_LOCK = threading.Lock()
_OAUTH_PENDING: dict | None = None
_SCOPES = "user-read-playback-state user-read-currently-playing user-modify-playback-state"


def _state_path() -> Path:
    configured = os.getenv("ME3_SPOTIFY_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        root = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "ME3"
    else:
        root = Path(os.getenv("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))) / "me3"
    return root / "spotify_oauth.json"


def _read_saved() -> dict:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_saved(data: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    if os.name != "nt":
        try:
            path.parent.chmod(0o700)
            temporary.chmod(0o600)
        except OSError:
            pass
    temporary.replace(path)
    if os.name != "nt":
        try:
            path.chmod(0o600)
        except OSError:
            pass


def client_id() -> str:
    return os.getenv("SPOTIFY_CLIENT_ID", "").strip() or str(_read_saved().get("client_id") or "")


def configure_client_id(value: str) -> dict:
    value = str(value or "").strip()
    if not (8 <= len(value) <= 128) or not value.isascii() or any(c.isspace() for c in value):
        return {"ok": False, "error": "invalid_spotify_client_id"}
    saved = _read_saved()
    saved["client_id"] = value
    _write_saved(saved)
    return {"ok": True, "configured": True}


def redirect_uri() -> str:
    return os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8100/api/spotify/callback").strip()


def begin_oauth() -> tuple[str | None, dict | None]:
    """Start Spotify's Authorization Code with PKCE flow without a client secret."""
    global _OAUTH_PENDING
    app_id = client_id()
    if not app_id:
        return None, {"ok": False, "error": "spotify_client_id_required"}
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    state = secrets.token_urlsafe(32)
    with _OAUTH_LOCK:
        _OAUTH_PENDING = {"state": state, "verifier": verifier,
                          "created": time.time()}
    params = {
        "client_id": app_id, "response_type": "code", "redirect_uri": redirect_uri(),
        "scope": _SCOPES, "state": state, "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    return "https://accounts.spotify.com/authorize?" + urlencode(params), None


def complete_oauth(code: str, state: str) -> dict:
    """Validate the one-time PKCE state, exchange the code, and persist refresh token."""
    global _OAUTH_PENDING, _TOKEN
    with _OAUTH_LOCK:
        pending = _OAUTH_PENDING
        _OAUTH_PENDING = None
    if (not pending or not secrets.compare_digest(str(state or ""), pending["state"])
            or time.time() - float(pending["created"]) > 600):
        return {"ok": False, "error": "spotify_oauth_state_invalid_or_expired"}
    app_id = client_id()
    try:
        response = requests.post(TOKEN_URL, data={
            "client_id": app_id, "grant_type": "authorization_code",
            "code": str(code or ""), "redirect_uri": redirect_uri(),
            "code_verifier": pending["verifier"],
        }, timeout=15)
        if response.status_code != 200:
            return {"ok": False, "error": "spotify_oauth_exchange_failed",
                    "status_code": response.status_code}
        data = response.json()
        refresh = str(data.get("refresh_token") or "")
        access = str(data.get("access_token") or "")
        if not refresh or not access:
            return {"ok": False, "error": "spotify_oauth_token_missing"}
        saved = _read_saved()
        saved.update({"client_id": app_id, "refresh_token": refresh,
                      "scopes": str(data.get("scope") or _SCOPES),
                      "connected_at": int(time.time())})
        _write_saved(saved)
        with _TOKEN_LOCK:
            _TOKEN = (access, time.monotonic() + max(60, int(data.get("expires_in", 3600))))
        return {"ok": True, "connected": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "spotify_oauth_request_failed",
                "error_type": type(exc).__name__}


def _credentials() -> tuple[str, str]:
    saved = _read_saved()
    app_id = client_id()
    refresh = os.getenv("SPOTIFY_REFRESH_TOKEN", "").strip() or str(saved.get("refresh_token") or "")
    return app_id, refresh


def health() -> dict:
    app_id, refresh = _credentials()
    missing = []
    if not app_id:
        missing.append("Spotify app Client ID")
    if not refresh:
        missing.append("one-time Spotify authorization")
    return {"provider": "spotify_web_api", "configured": not missing,
            "connected": bool(app_id and refresh), "missing": missing,
            "client_id_configured": bool(app_id),
            "authorized": bool(refresh),
            "requires_premium_for_playback": True,
            "redirect_uri": redirect_uri()}


def _access_token() -> tuple[str | None, dict | None]:
    global _TOKEN
    app_id, refresh = _credentials()
    if not app_id or not refresh:
        return None, {"ok": False, "error": "spotify_not_configured",
                      "missing": health()["missing"]}
    now = time.monotonic()
    with _TOKEN_LOCK:
        if _TOKEN and _TOKEN[1] - 30 > now:
            return _TOKEN[0], None
        try:
            response = requests.post(
                TOKEN_URL,
                data={"client_id": app_id, "grant_type": "refresh_token",
                      "refresh_token": refresh},
                timeout=12,
            )
            if response.status_code != 200:
                return None, {"ok": False, "error": "spotify_auth_failed",
                              "status_code": response.status_code}
            data = response.json()
            token = str(data.get("access_token") or "")
            if not token:
                return None, {"ok": False, "error": "spotify_auth_empty_token"}
            rotated = str(data.get("refresh_token") or "")
            if rotated:
                saved = _read_saved()
                saved.update({"client_id": app_id, "refresh_token": rotated})
                _write_saved(saved)
            _TOKEN = (token, now + max(60, int(data.get("expires_in", 3600))))
            return token, None
        except Exception as exc:  # noqa: BLE001 - never return provider bodies
            return None, {"ok": False, "error": "spotify_auth_request_failed",
                          "error_type": type(exc).__name__}


def _request(method: str, path: str, *, params: dict | None = None,
             body: dict | None = None) -> tuple[requests.Response | None, dict | None]:
    token, error = _access_token()
    if error:
        return None, error
    try:
        response = requests.request(
            method, API + path, headers={"Authorization": f"Bearer {token}"},
            params=params, json=body, timeout=12,
        )
        if response.status_code not in (200, 201, 202, 204):
            # Do not include Spotify's free-form body in voice replies or logs.
            return response, {"ok": False, "error": "spotify_request_rejected",
                              "status_code": response.status_code}
        return response, None
    except Exception as exc:  # noqa: BLE001
        return None, {"ok": False, "error": "spotify_request_failed",
                      "error_type": type(exc).__name__}


def _playback_snapshot() -> dict:
    response, error = _request("GET", "/me/player")
    if error:
        return error
    if response.status_code == 204:
        return {"ok": True, "active": False, "playing": False,
                "error": "spotify_player_inactive"}
    data = response.json()
    item = data.get("item") or {}
    device = data.get("device") or {}
    return {"ok": True, "active": bool(device.get("id")),
            "playing": bool(data.get("is_playing")),
            "device": device.get("name"),
            "track": item.get("name"),
            "artists": [a.get("name") for a in item.get("artists", [])
                        if a.get("name")],
            "uri": item.get("uri"),
            "progress_ms": data.get("progress_ms")}


def play(query: str, service: str = "spotify") -> dict:
    if service.lower() not in {"spotify", "spotify_connect"}:
        return {"ok": False, "error": "unsupported_music_provider",
                "supported_provider": "spotify"}
    query = str(query or "").strip()
    if not query:
        return {"ok": False, "error": "music_query_required"}
    response, error = _request(
        "GET", "/search", params={"q": query[:200], "type": "track", "limit": 1})
    if error:
        return error
    rows = ((response.json().get("tracks") or {}).get("items") or [])
    if not rows:
        return {"ok": False, "error": "spotify_no_track_match", "query": query}
    track = rows[0]
    uri = str(track.get("uri") or "")
    if not uri:
        return {"ok": False, "error": "spotify_track_uri_missing"}
    response, error = _request("PUT", "/me/player/play", body={"uris": [uri]})
    if error:
        if error.get("status_code") == 404:
            error["error"] = "spotify_player_inactive"
        return error
    # Spotify returns 204 after accepting playback.  Read back the active
    # player so the user hears a verified result instead of a local fake state.
    snapshot = _playback_snapshot()
    accepted = bool(snapshot.get("ok") and snapshot.get("playing"))
    return {"ok": accepted, "action": "play", "provider": "Spotify",
            "requested_query": query, "track": track.get("name"),
            "artists": [a.get("name") for a in track.get("artists", [])
                        if a.get("name")],
            "uri": uri, "device": snapshot.get("device"),
            "playback_verified": accepted,
            "error": None if accepted else snapshot.get("error", "playback_not_verified"),
            "note": "Open Spotify on the intended speaker once and keep it available as a Connect player."}


def control(action: str, value: int | None = None) -> dict:
    action = str(action or "").strip().lower().replace(" ", "_")
    mapping = {"pause": ("PUT", "/me/player/pause"),
               "stop": ("PUT", "/me/player/pause"),
               "skip": ("POST", "/me/player/next"),
               "next": ("POST", "/me/player/next"),
               "play": ("PUT", "/me/player/play")}
    if action in mapping:
        method, path = mapping[action]
        response, error = _request(method, path)
        if error:
            if error.get("status_code") == 404:
                error["error"] = "spotify_player_inactive"
            return error
        snapshot = _playback_snapshot()
        return {"ok": True, "action": action, "provider": "Spotify",
                "playback": snapshot}
    if action in {"volume_up", "volume_down", "set_volume"}:
        snapshot = _playback_snapshot()
        if not snapshot.get("ok") or not snapshot.get("active"):
            return {"ok": False, "error": "spotify_player_inactive"}
        volume = int(value if action == "set_volume" and value is not None
                     else 50)
        if action != "set_volume":
            response, error = _request("GET", "/me/player/devices")
            if error:
                return error
            devices = response.json().get("devices") or []
            active = next((d for d in devices if d.get("is_active")), {})
            volume = int(active.get("volume_percent") or 50)
            volume += 10 if action == "volume_up" else -10
        volume = max(0, min(100, volume))
        _, error = _request("PUT", "/me/player/volume", params={"volume_percent": volume})
        if error:
            return error
        return {"ok": True, "action": action, "volume": volume,
                "provider": "Spotify"}
    return {"ok": False, "error": "unsupported_media_action"}


def state() -> dict:
    return _playback_snapshot()
