# ME2 command parity in ME3

ME3 keeps the ME2 user-facing command set, but replaces the trained ME2
classifier with model inference plus explicit tools. The Qwen Instruct model
selects the tool from Filipino, English, or Taglish speech; the deterministic
adapter performs validation and records the action. The same registry is
available to the direct ToolBrain and the owned `me3-tasks` MCP server.

| ME2 command family | ME3 tool(s) | Current behavior | Live requirement |
|---|---|---|---|
| Play music | `play_music`, `music_state` | Sends a browser handoff event and search URL; Google maps to YouTube Music and Spotify is supported | One operator click in the signed-in browser; full programmatic control needs a provider player/OAuth bridge |
| Ask/search/weather | `web_search`, `fetch_url`, `get_weather`, `get_time` | Public web search, no-key current-weather lookup, and local time | Internet access for web/weather; verify results before acting |
| Lights on/off | `control_lights` | Validated local demo state | Home Assistant URL/token/entity mapping or another IoT adapter |
| Dim/color lights | `control_lights` | Percent/color validation and local state | Same physical-light connector |
| Set a timer | `set_timer`, `set_alarm` | Timer and next-occurrence alarm records | Platform notification scheduler for audible firing |
| Set an alarm | `set_alarm`, `list_alarms`, `cancel_alarm` | Alarm lifecycle is available to voice and MCP | Platform notification scheduler |
| Thermostat | `set_thermostat` | Celsius validation and local demo state | Home Assistant climate entity or vendor API |
| Media controls | `media_control` | Pause, stop, skip, play, volume up/down/set | Browser/player adapter for physical playback |
| Reminders and lists | `create_reminder`, `list_reminders`, `complete_reminder` | Reminder lifecycle and delay/time parsing | Platform notification scheduler for delivery |
| Calls and messaging | `call_contact`, `send_message`, `send_telegram_message` | Telegram Bot API text with confirmation; ordinary voice calls reported unsupported | Protected bot token and approved numeric chat-ID mapping; Telegram Bot API does not initiate ordinary voice calls |

## Safety and demo boundary

- Benchmark mode disables all tool calls, agent/harness dispatch, fillers, and
  backchannels; it remains direct-feed/injection-only as required by the class
  benchmark.
- Demo-state tools do not claim that a physical lamp, thermostat, player, or
  notification scheduler is connected. Their result explicitly reports
  `local_demo` until the corresponding connector is configured.
- External communication tools require explicit confirmation. Contact names
  resolve only through protected `ME3_CONTACT_<NAME>` or
  `ME3_TELEGRAM_CHAT_<NAME>` environment mappings.
- The Pi never reads or copies browser cookies. YouTube Music and Spotify use a
  visible browser handoff so the operator's existing login stays in that
  browser. Google Play Music itself is discontinued; use YouTube Music.
- Clear short commands are routed through a bounded voice-command guard before
  Qwen planning. This is still a tool call through the same dispatcher; it only
  prevents repeated conversation history from swallowing an unambiguous ME2
  command. Open-ended requests continue through normal Qwen tool inference.
