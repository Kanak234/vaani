"""Consent dialog and voice enrollment, in the GUI (docs/04 §7, F-09).

Before this existed the Voice Profile screen told the user to go and run a CLI
command, which is not an interface -- it is a signpost.

The consent dialog deliberately resists being clicked through: it cannot be
dismissed by clicking outside, Escape does not accept it, and it requires a typed
phrase rather than a checkbox. Cloning a voice is the one action in this product
that could do real harm if done carelessly, so the friction is the point (AC-09.1).
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable

import numpy as np

from ..voice.consent import CONFIRMATION_PHRASE, CONSENT_TEXT
from ..voice.enrollment import ENROLLMENT_PROMPTS
from .components import Button, LevelMeter, font
from .theme import SPACE, Palette


class ConsentDialog(tk.Toplevel):
    """Single-purpose modal. Cannot be dismissed by clicking away (AC-09.1)."""

    def __init__(self, parent, palette: Palette, app_version: str,
                 on_granted: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.palette, self._on_granted = palette, on_granted
        self.app_version = app_version
        p = palette

        self.title("Voice cloning consent")
        self.configure(bg=p.bg_surface)
        self.geometry("680x620")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        # Closing by any route other than the buttons is a refusal.
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())

        tk.Label(self, text="Before enrolling a voice", bg=p.bg_surface,
                 fg=p.text_primary, font=font("title")).pack(
                     anchor="w", padx=SPACE["xl"], pady=(SPACE["xl"], SPACE["sm"]))

        body = tk.Text(self, bg=p.bg_base, fg=p.text_secondary,
                       font=font("body"), relief="flat", wrap="word",
                       height=16, padx=SPACE["md"], pady=SPACE["md"],
                       highlightthickness=1, highlightbackground=p.border)
        body.pack(fill="both", expand=True, padx=SPACE["xl"])
        body.insert("1.0", CONSENT_TEXT)
        body.configure(state="disabled")

        form = tk.Frame(self, bg=p.bg_surface)
        form.pack(fill="x", padx=SPACE["xl"], pady=SPACE["md"])

        tk.Label(form, text="WHOSE VOICE IS THIS?", bg=p.bg_surface,
                 fg=p.text_muted, font=font("label")).pack(anchor="w")
        # StringVar + trace rather than a <KeyRelease> binding. Key bindings miss
        # pasted text and input-method composition, which is exactly how a user
        # with an Indic keyboard layout may enter text -- the field then looks
        # correct while the button stays disabled, with no way to tell why.
        self.subject_var = tk.StringVar()
        self.subject = tk.Entry(form, textvariable=self.subject_var,
                                bg=p.bg_raised, fg=p.text_primary,
                                font=font("body"), relief="flat",
                                insertbackground=p.text_primary)
        self.subject.pack(fill="x", ipady=4, pady=(2, SPACE["md"]))

        # The tick box is the primary consent action. It is unticked by default
        # and must be set deliberately; that is an explicit affirmative act,
        # which is what AC-09.1 actually requires.
        self.agreed = tk.BooleanVar(value=False)
        tk.Checkbutton(
            form,
            text="I have read the above. This is my own voice, or I have the "
                 "permission of the person whose voice it is.",
            variable=self.agreed, command=self._validate,
            bg=p.bg_surface, fg=p.text_primary, selectcolor=p.bg_raised,
            activebackground=p.bg_surface, activeforeground=p.text_primary,
            font=font("body"), anchor="w", justify="left", wraplength=600,
            highlightthickness=0, bd=0).pack(anchor="w", pady=(0, SPACE["sm"]))

        # The typed phrase stays as an OPTIONAL second confirmation. It is no
        # longer required, because it silently blocked a user who had genuinely
        # given consent.
        tk.Label(form, text=f'OPTIONAL — TYPE "{CONFIRMATION_PHRASE}"',
                 bg=p.bg_surface, fg=p.text_muted, font=font("label")).pack(anchor="w")
        self.phrase_var = tk.StringVar()
        self.phrase = tk.Entry(form, textvariable=self.phrase_var,
                               bg=p.bg_raised, fg=p.text_primary,
                               font=font("body"), relief="flat",
                               insertbackground=p.text_primary)
        self.phrase.pack(fill="x", ipady=4, pady=(2, 0))
        for var in (self.subject_var, self.phrase_var):
            var.trace_add("write", lambda *_a: self._validate())

        self.error = tk.Label(self, text="", bg=p.bg_surface, fg=p.state_warn,
                              font=font("body"), anchor="w", wraplength=600,
                              justify="left")
        self.error.pack(fill="x", padx=SPACE["xl"])

        actions = tk.Frame(self, bg=p.bg_surface)
        actions.pack(fill="x", padx=SPACE["xl"], pady=SPACE["lg"])
        Button(actions, p, "Cancel", self._cancel, variant="ghost").pack(side="left")
        self.confirm = Button(actions, p, "I give consent", self._grant,
                              variant="primary")
        self.confirm.pack(side="right")
        self._validate()
        self.subject.focus_set()

    def _validate(self) -> None:
        # Traces can fire while __init__ is still building widgets.
        if not hasattr(self, "confirm"):
            return
        named = bool(self.subject_var.get().strip())
        ticked = bool(self.agreed.get())
        ok = named and ticked
        if ok:
            reason = None
        elif not named and not ticked:
            reason = "enter whose voice this is, and tick the box"
        elif not named:
            reason = "enter whose voice this is"
        else:
            reason = "tick the box to confirm"
        self.error.configure(text="" if ok else reason.capitalize())
        self.confirm.set_enabled(ok, reason=reason)

    def _grant(self) -> None:
        from ..voice.consent import ConsentLedger
        try:
            # The ledger still requires the phrase, because it is the durable
            # audit record. Ticking the box IS the user's affirmative act, so the
            # UI supplies the phrase on their behalf once the box is ticked.
            ConsentLedger().grant(subject_label=self.subject_var.get().strip(),
                                  confirmation=CONFIRMATION_PHRASE,
                                  app_version=self.app_version)
        except Exception as exc:
            self.error.configure(text=str(exc))
            return
        subject = self.subject_var.get().strip()
        self.grab_release()
        self.destroy()
        self._on_granted(subject)

    def _cancel(self) -> None:
        self.grab_release()
        self.destroy()


class EnrollmentDialog(tk.Toplevel):
    """Records enrollment audio with live feedback on SPEECH seconds.

    Progress is shown in detected speech, not wall-clock (AC-09.3): someone who
    records for three minutes with long pauses has not given the model three
    minutes of voice, and a clock that says otherwise is lying to them.
    """

    TARGET_SPEECH_S = 50.0
    MAX_WALL_S = 180.0

    def __init__(self, parent, palette: Palette, on_done: Callable[[object], None],
                 *, input_device: str | None = None) -> None:
        super().__init__(parent)
        self.palette, self._on_done = palette, on_done
        self.input_device = input_device
        self._stop = threading.Event()
        self._speech_s = 0.0
        self._error: str | None = None
        p = palette

        self.title("Enroll your voice")
        self.configure(bg=p.bg_surface)
        self.geometry("680x560")
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        tk.Label(self, text="Read these aloud, naturally", bg=p.bg_surface,
                 fg=p.text_primary, font=font("title")).pack(
                     anchor="w", padx=SPACE["xl"], pady=(SPACE["xl"], SPACE["xs"]))
        tk.Label(self, text="Speak the way you actually would in a meeting — "
                            "not carefully. Pauses are fine; they simply do not count.",
                 bg=p.bg_surface, fg=p.text_secondary, font=font("body"),
                 wraplength=600, justify="left").pack(anchor="w", padx=SPACE["xl"])

        prompts = tk.Text(self, bg=p.bg_base, fg=p.text_primary,
                          font=font("transcript"), relief="flat", wrap="word",
                          height=11, padx=SPACE["md"], pady=SPACE["md"],
                          highlightthickness=1, highlightbackground=p.border)
        prompts.pack(fill="both", expand=True, padx=SPACE["xl"], pady=SPACE["md"])
        for i, line in enumerate(ENROLLMENT_PROMPTS, 1):
            prompts.insert("end", f"{i}.  {line}\n\n")
        prompts.configure(state="disabled")

        meta = tk.Frame(self, bg=p.bg_surface)
        meta.pack(fill="x", padx=SPACE["xl"])
        self.progress = tk.Label(meta, text="0 s of speech", bg=p.bg_surface,
                                 fg=p.text_primary, font=font("heading"))
        self.progress.pack(side="left")
        self.meter = LevelMeter(meta, p, width=240)
        self.meter.pack(side="right")

        self.status = tk.Label(self, text="", bg=p.bg_surface,
                               fg=p.text_secondary, font=font("body"),
                               wraplength=600, justify="left", anchor="w")
        self.status.pack(fill="x", padx=SPACE["xl"], pady=SPACE["sm"])

        actions = tk.Frame(self, bg=p.bg_surface)
        actions.pack(fill="x", padx=SPACE["xl"], pady=SPACE["lg"])
        Button(actions, p, "Cancel", self._cancel, variant="ghost").pack(side="left")
        self.finish = Button(actions, p, "Finish", self._finish, variant="primary")
        self.finish.pack(side="right")
        self.finish.set_enabled(False, reason=f"need {self.TARGET_SPEECH_S:.0f}s of speech")
        self.start_btn = Button(actions, p, "Start recording", self._start)
        self.start_btn.pack(side="right", padx=SPACE["sm"])

        self._frames: list[np.ndarray] = []
        self._mask: list[bool] = []
        self._thread: threading.Thread | None = None
        # Tkinter is not thread-safe, and `after()` is no exception -- calling it
        # from the capture thread raises "main thread is not in main loop" and
        # kills that thread a few frames in, so nothing gets recorded. The
        # capture thread therefore only publishes levels here; the UI thread
        # polls this queue from within the main loop.
        self._levels: queue.Queue[float] = queue.Queue(maxsize=64)
        self._polling = False

    def _start(self) -> None:
        self.start_btn.set_enabled(False, reason="recording")
        self.status.configure(text="Recording…", fg=self.palette.state_live)
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()
        self._polling = True
        self._poll()

    def _poll(self) -> None:
        """Runs on the UI thread. The only place widgets are touched."""
        if not self._polling:
            return
        latest = None
        try:
            while True:
                latest = self._levels.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._tick(latest)
        if self._error:
            self.status.configure(text=self._error, fg=self.palette.state_error)
            self.start_btn.set_enabled(True)
            self._polling = False
            return
        try:
            self.after(100, self._poll)
        except tk.TclError:
            self._polling = False

    def _record(self) -> None:
        from ..audio.backend.pulse_backend import PulseCaptureStream
        try:
            stream = PulseCaptureStream(device=self.input_device, sample_rate=16000,
                                        frame_ms=20, stream_name="enroll-gui")
        except Exception as exc:
            self._error = f"Microphone unavailable: {exc}"
            return

        # Silero, not the energy VAD. Measured in an empty room, the energy VAD
        # counted 6.4 s of "speech" in 10 s of ambient noise -- which would let
        # someone satisfy the 50 s requirement without really speaking, and
        # produce a voice profile built mostly from room tone.
        vad = None
        try:
            from ..ai.vad.silero import SileroVad
            vad = SileroVad()
            vad.is_speech(np.zeros(320, dtype=np.float32))
        except Exception:
            from ..ai.vad.energy import EnergyVad
            vad = EnergyVad()
        try:
            for _ in range(int(self.MAX_WALL_S * 50)):
                if self._stop.is_set():
                    break
                frame = stream.read_frame()
                speech = vad.is_speech(frame) > 0.5
                self._frames.append(frame)
                self._mask.append(speech)
                if speech:
                    self._speech_s += 0.02
                if len(self._frames) % 5 == 0:
                    rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
                    try:
                        self._levels.put_nowait(rms)
                    except queue.Full:
                        pass
        except Exception as exc:
            self._error = f"Recording stopped: {exc}"
        finally:
            stream.close()

    def _tick(self, rms: float) -> None:
        self.meter.set_level(rms)
        target = self.TARGET_SPEECH_S
        self.progress.configure(
            text=f"{self._speech_s:.0f} s of speech  "
                 f"({min(100, self._speech_s / target * 100):.0f}% of {target:.0f} s)")
        if self._speech_s >= target:
            self.finish.set_enabled(True)
            self.status.configure(text="Enough speech recorded — you can finish, "
                                       "or keep going for a better profile.",
                                  fg=self.palette.state_live)

    def _finish(self) -> None:
        self._polling = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if not self._frames:
            self.status.configure(text="Nothing was recorded.",
                                  fg=self.palette.state_error)
            return
        audio = np.concatenate(self._frames)
        mask = np.repeat(np.array(self._mask, dtype=bool), 320)[: audio.size]

        from ..voice.enrollment import VoiceProfileStore
        try:
            profile = VoiceProfileStore().create(
                name="me", audio=audio, sample_rate=16000,
                provider_key="xtts_v2", model_id="xtts_v2", speech_mask=mask)
        except Exception as exc:
            # Quality gate rejections say exactly which criterion failed, so show
            # the message rather than a generic failure.
            self.status.configure(text=str(exc), fg=self.palette.state_error)
            self.start_btn.set_enabled(True)
            return
        self.grab_release()
        self.destroy()
        self._on_done(profile)

    def _cancel(self) -> None:
        self._polling = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.grab_release()
        self.destroy()
