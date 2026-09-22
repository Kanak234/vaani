# Meeting Takeover Mode

Vaani supports a session-scoped meeting takeover policy for situations where the
user cannot answer quickly enough.

## Modes

- **OFF** — Vaani never takes the conversational turn automatically.
- **ARMED** — Vaani watches for a pending question plus observable hesitation.
- **ACTIVE** — Vaani may continue answering successive turns until the user
  explicitly stops takeover.

Arming is a deliberate session action. This prevents an ordinary translation
session from unexpectedly generating original speech.

## Automatic handoff

A takeover is eligible when all of the following are true:

1. Takeover mode is armed.
2. The other participant has produced a pending question/turn.
3. The user has produced repeated short/hesitant attempts to answer.
4. The configured silence window has elapsed.

The controller does **not** diagnose panic, anxiety, or any medical condition.
It reacts only to observable conversation signals.

## Explicit handoff

The user can say a configured phrase such as `Vaani, take over`. This immediately
activates takeover when the mode is armed.

## Continuous operation

While ACTIVE, Vaani does not require a fresh approval before every sentence.
The conversation layer can generate and speak the next response until the user
stops takeover or the application hits a configured safety boundary.

## Stop invariant

`stop()` is authoritative. It returns the controller to OFF and clears the
pending conversational turn. The existing session emergency-stop path must also
flush already queued audio so a stop cannot leave synthesized speech playing.

## Language behavior

Takeover generation should use the same conversational language policy as the
meeting assistant: Hindi, English, and Hinglish are valid response languages.
Translation is separate from response generation; the existing translation
router remains responsible for translating user-authored speech.

## Architecture boundary

`src/vaani/session/takeover.py` decides **whether Vaani may take the floor**.
It does not generate text, access the microphone, synthesize audio, or select a
voice. Those responsibilities remain in the existing AI/session/voice layers.
This separation makes the takeover policy testable and keeps the emergency-stop
invariant centralized in the session state machine.
