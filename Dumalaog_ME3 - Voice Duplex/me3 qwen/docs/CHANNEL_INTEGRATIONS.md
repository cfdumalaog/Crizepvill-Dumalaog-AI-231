# ME3 communication channels

ME3 is now Telegram-first for outbound communication. The active external
messaging adapter is the Telegram Bot API `sendMessage` method. It is protected
by a bot token, an approved chat mapping, and an explicit confirmation gate.

Phone calls and SMS are intentionally out of scope. No phone-provider account,
phone number, or SMS provider is required by the active project. The Telegram Bot API
also does not initiate ordinary Telegram voice calls. Adding voice calls later
would require a separately reviewed MTProto/TDLib user-account bridge with its
own consent, privacy, and audit design; ME3 does not claim that capability now.

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
3. Retrieve the destination's numeric `chat_id` using a protected operator
   workflow (for example, a temporary `getUpdates` request on the Pi). Review
   the result, then remove any temporary output containing the token or update
   payload.
4. On the Pi, put the values in the protected ME3 environment, not in source:

   ```text
   TELEGRAM_BOT_TOKEN=<bot token>
   ME3_TELEGRAM_CHAT_MONICA_FERNANDEZ=<numeric chat id>
   ```

   Use mode `600` and owner `dalmacio` for the user environment, or the
   equivalent protected service environment. Never print the values.
5. Restart the Pi service and confirm `channel_health` reports Telegram as
   configured. Then ask ME3 to prepare the message. ME3 must read back the
   recipient and message and wait for an explicit confirmation before calling
   `sendMessage`.

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
| Telegram text via Bot API | Implemented, confirmation-gated, not live-tested until the operator supplies a bot token and chat ID |
| Telegram ordinary voice call | Not available through the Bot API |
| Phone call | Intentionally removed from the active project |
| SMS | Intentionally removed from the active project |
| Optional transactional email | Implemented through Resend, confirmation-gated, not required for the demo |

All channel health checks are network-free. The local workstation UI blocks
external side effects; the Pi voice agent is the only runtime that may execute a
confirmed Telegram request.
