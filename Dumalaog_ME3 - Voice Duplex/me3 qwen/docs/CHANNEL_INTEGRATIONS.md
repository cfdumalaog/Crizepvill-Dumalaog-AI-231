# ME3 communication channels

ME3 is now Telegram-first for outbound communication. The active external
messaging adapter is the Telegram Bot API `sendMessage` method. It requires a
bot token and numeric chat mapping. A clear direct user request authorizes the
send; ME3 asks only for missing recipient or message details.

Phone calls use the optional ME3 Android companion and the phone's own SIM; SMS
is out of scope. No Twilio or other phone-provider account is required. The
Telegram Bot API does not initiate ordinary Telegram voice calls.

## What is needed for real actions

Open the tunneled Pi UI at `http://127.0.0.1:8100/`. Enter provider values in
the setup cards there rather than sending secrets in chat. The browser request
travels through the laptop's SSH tunnel, and the Pi stores credentials in
private per-user files.

| Action | What must be ready | Where to complete it |
|---|---|---|
| Laptop alarm/timer sound | No provider account. Connect the browser speaker once and keep the browser tab open and laptop awake at the due time. The schedule persists on the Pi, but in-browser sound cannot wake a sleeping laptop or ring after the tab closes. | In ME3, connect the browser speaker, enable notifications if desired, and set an alarm/timer. Use the speaker-test button before relying on it. |
| Spotify playback | Spotify Web API app Client ID, Spotify Premium on the app owner's account, one-time OAuth authorization, and an online Spotify Connect player. You confirmed Web API is enabled, but the current Free account cannot control playback. The current PKCE flow does not require the app Client Secret. | After upgrading to Premium, enter/save the Client ID, then choose **Connect Spotify** in the tunneled Pi UI. |
| Local reminder | No Google credential is needed. It persists on the Pi and is announced by the assistant; laptop audio requires the connected browser and an awake laptop. | Set it by voice or with the Pi schedule controls in the UI. |
| Google Calendar reminder/meeting | Google OAuth Web client ID and secret, Calendar API enabled, callback registered, and account authorization. Calendar synchronization is separate from the local Pi scheduler. | Enter both client values and choose **Connect Calendar** in the tunneled Pi UI. |
| Telegram message to `@mncfrndz` | A bot token from `@BotFather`; the recipient must open that bot and send `/start` once. A personal Telegram login alone is not a bot API credential. | Save the token, then choose **Connect @mncfrndz** in the tunneled Pi UI to map the chat. |
| Phone call | Android companion installed/open on a phone with cellular calling; allow `CALL_PHONE` and, for contact names, `READ_CONTACTS`; keep the companion connected to the Pi. A live call has not yet been tested. | Install the project APK, open its ME3 UI, grant permissions, and keep it connected. |
| Carrier SMS | Not implemented in the current Android companion. Telegram Bot API sends Telegram chat text, not SMS. | No credential can enable carrier SMS yet; it needs a phone-side implementation and permission. |

Spotify callback: `http://127.0.0.1:8100/api/spotify/callback`<br>
Google callback: `http://127.0.0.1:8100/api/google/callback`

Spotify's [playback endpoint](https://developer.spotify.com/documentation/web-api/reference/start-a-users-playback)
requires Premium. Its [February 2026 Development Mode rules](https://developer.spotify.com/documentation/web-api/tutorials/february-2026-migration-guide)
also require an active Premium subscription on the app owner's account.

The optional Resend email adapter remains available as a separate transactional
email path. It is not needed for the Telegram demo and is also confirmation-
gated. If the project scope is strictly Telegram-only, leave the Resend values
unset.

## Telegram Bot API setup

1. In Telegram, open `@BotFather`, send `/newbot`, choose a bot name and unique
   username, and receive the bot token. Keep the token private; do not paste it
   into GitHub, the repository, the roadmap, or chat.
2. Open the new bot from the destination Telegram account and send `/start`.
   A bot cannot initiate a private conversation before the user starts it.
3. Open ME3 through the SSH tunnel at `http://127.0.0.1:8100/`, enter the bot
   token in the Telegram setup card, and choose **Connect @mncfrndz**. ME3 uses
   `getUpdates` to discover and save the numeric chat mapping privately on the
   Pi; no manual token/chat-ID shell commands are needed.
4. Confirm `channel_health` reports Telegram configured. A clear request such
   as "send Good morning to @mncfrndz on Telegram" executes `sendMessage`; ME3
   does not add a second confirmation prompt. Vague requests still require the
   missing recipient or message content.

The project contact helper accepts either a numeric chat ID or a named mapping
such as `ME3_TELEGRAM_CHAT_MONICA_FERNANDEZ`. A failed or unconfigured request
returns safe status metadata only; the bot token and Telegram response body are
never returned.

## Optional Resend email

Leave these unset for a Telegram-only demonstration. If email is later needed,
provision them through the same protected runtime mechanism:

```text
RESEND_API_KEY=<protected key>
RESEND_FROM_EMAIL=ME3 <verified sender@example.ph>
```

Email remains confirmation-gated and bounded. The tests use fake requests and do
not send email.

## What is and is not implemented

| Capability | Status |
|---|---|
| Telegram text via Bot API | Implemented; a clear user request authorizes sending; not live-tested until the Pi has a bot token and the target has started the bot |
| Telegram ordinary voice call | Not available through the Bot API |
| Phone call | Implemented through Android `ACTION_CALL`; requires the companion app, phone service connection, and runtime permission; live call not tested |
| SMS | Not implemented; Telegram Bot API does not send cellular SMS |
| Optional transactional email | Implemented through Resend; explicit user request required; not required for the demo |

All channel health checks are network-free. The local workstation UI blocks
external side effects; the Pi voice agent is the only runtime that may execute
an explicitly requested Telegram message.

For a call, install and open the Android companion, allow the phone-call
permission, and keep it connected to the ME3 Pi service. Contact-name calling
also requires Contacts permission; a numeric phone number avoids that lookup.
