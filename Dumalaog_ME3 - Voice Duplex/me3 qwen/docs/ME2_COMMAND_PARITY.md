# ME2 command parity in ME3 Qwen

ME3 uses Qwen inference and deterministic tools instead of training the ME2
command classifier. A clear spoken request dispatches the matching tool
automatically. If required details or a provider connection are missing, ME3
states the specific blocker or asks for the missing detail.

| ME2 command | Tool and execution path | Current boundary / one-time setup |
|---|---|---|
| Play music | `play_music` searches Spotify and starts the track through Spotify Web API on the active Spotify Connect device. `music_state` reads back playback. | Spotify app ID supplied; save it in the Pi UI, authorize once, keep a Spotify Connect player active, and use Premium. The current Spotify dashboard session showed a Free account with Web API access disabled, so playback is not live-verified. No browser handoff is counted as playback. |
| Ask/search/weather/time | `web_search`, `fetch_url`, `get_weather`, `get_time` call their actual network or host-time adapters. | Requires network access for web/weather. |
| Lights on/off | `control_lights` updates the simulated bulb shown in the UI. | Demo-only. No physical light entity or Home Assistant service is connected. |
| Dim/color lights | `control_lights` updates simulated brightness/color and the same UI bulb. | Demo-only. |
| Set a timer | `set_timer` writes a durable Pi schedule; ME3 speaks when due and the connected web UI rings and shows a dismissible alert. The clock card shows the countdown. | The Pi service must be running. A browser must remain open with an enabled audio context for the laptop ring; scheduled state survives a Pi service restart. |
| Set an alarm | `set_alarm` writes the next Asia/Manila occurrence to the durable Pi schedule; voice and UI alert when due. | The Pi service must be running. Laptop sound requires the ME3 UI open and browser audio enabled. |
| Adjust thermostat | `set_thermostat` stores a simulated Celsius value. | Demo-only. No physical thermostat or climate service is connected. |
| Media control | `media_control` uses Spotify Web API for play/pause/stop/skip/volume on the active player. | Same Spotify OAuth, Premium, and active-player requirements as music playback. |
| Reminders and lists | `create_reminder` creates a Google Calendar event and a durable Pi voice/UI reminder. `list_reminders` and `complete_reminder` use the local reminder index; completion removes the associated Calendar event. | Connector and OAuth flow are implemented. Google OAuth client values and one-time authorization are still needed on the Pi. ME3 handles the due-time voice/ring alert; the Calendar event is the persistent calendar record. |
| Meetings and invites | `schedule_meeting` creates a Google Calendar event, attendee invitations, and optional Google Meet link. | Connector is implemented. Google OAuth client values, Calendar API enablement, and one-time authorization are still needed on the Pi. Clear requests execute; ME3 asks only for missing title/time/attendee information. |
| Calls | `call_contact` sends a call action to the connected ME3 Android companion, which uses the phone SIM/contacts and Android `CALL_PHONE` permission. | Install/open the updated companion and allow call permission; allow Contacts permission if using a saved contact name. The phone must be connected to the Pi service over LAN/VPN. Telegram Bot API cannot initiate voice calls. |
| Messages / SMS replacement | `send_message` sends a real Telegram text using Bot API to the configured numeric chat ID for `@mncfrndz` (Wifeeyy Dumalaog Fernandez+). | Create a Telegram bot with BotFather, save its token in the ME3 UI through the SSH tunnel, and have `@mncfrndz` press Start once. ME3 then discovers and stores the numeric chat mapping. Telegram cannot send a private message to an unstarted bot recipient or initiate voice calls; no Twilio or SMS provider is used. |

## Provider setup in the web UI

- Spotify: register the exact loopback redirect URI
  `http://127.0.0.1:8100/api/spotify/callback`, save the app, save the supplied
  Client ID in the Pi UI, then use **Connect Spotify**. OAuth uses PKCE and stores the
  refresh token privately on the Pi. The laptop SSH local-forward tunnel must
  be active during authorization; after authorization, Pi service restarts do
  not need the callback tunnel to refresh the token.
- Google Calendar: create a Google OAuth web client, enable Calendar API, and
  register `http://127.0.0.1:8100/api/google/callback`. Enter the client ID and
  secret in the UI through the SSH tunnel, then choose **Connect Calendar**.
  The Pi stores credentials in a mode-0600 file. Browser Gmail sign-in alone
  does not grant ME3 Calendar API access.
- Telegram: create the bot, paste its bot token in the UI through the SSH
  tunnel, then have the target open the bot and press **Start**. Use
  **Connect @mncfrndz** to discover and store the numeric chat mapping.
- Phone calls: open the ME3 Android companion in addition to the laptop UI;
  the browser audio hub keeps the laptop as its single audio sink and routes
  call actions to the Android device capability.

Secret-setting endpoints reject non-loopback requests. Use the SSH tunnel
(`127.0.0.1:8100` to Pi port `8126`) for provider credentials. The Pi service
continues to own the inference loop and tools; the laptop browser provides the
speaker/UI, and the Android companion provides the phone-call endpoint.

## Safety and demo boundary

- Benchmark mode still disables tools, agent/harness dispatch, fillers, and
  backchannels; class benchmark runs remain injected-audio measurements.
- A clear voice instruction is the authorization to execute that requested
  action. The agent does not require a second UI confirmation. It asks only for
  missing information and never reports provider success unless the adapter
  returns a success result.
- The Pi does not read browser cookies or use the laptop's signed-in sessions
  as API credentials.
- Current physical-device limitations are explicit: lights and thermostat
  remain simulated; Spotify requires API access, Premium, and a Connect player;
  Telegram requires a bot token and a target that started the bot; calls
  require a connected Android phone.
