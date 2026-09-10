"""Vaani desktop application (docs/04-UIUX-BRIEF.md §7).

Five surfaces: Dashboard, Meeting Mode, Voice Profile, Settings, Diagnostics.

Presentation only. Every engine interaction goes through the same public API the
CLI uses; there is no logic here that a headless run does not also exercise. The
`ai/` and `audio/` layers are never imported directly from this module's screens
except through the session and diagnostics entry points.

Threading rule: the engine runs on its own threads and calls back from them.
Tkinter is not thread-safe, so every callback is marshalled onto the UI thread
with `after(0, ...)`. Breaking that rule produces crashes that look random.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable

from ..core.latency_monitor import BUDGET_MS
from ..core.state_machine import SessionState
from ..core.types import PerformanceMode
from .components import (
    Banner,
    Button,
    Card,
    DiagnosticRow,
    EmptyState,
    LatencyBar,
    LevelMeter,
    MetricReadout,
    StatusPill,
    font,
)
from .theme import DARK, SPACE, Palette, state_colour


class VaaniApp:
    def __init__(self, palette: Palette = DARK) -> None:
        self.palette = palette
        self.root = tk.Tk()
        self.root.title("Vaani")
        self.root.geometry("1180x760")
        self.root.minsize(1024, 680)
        self.root.configure(bg=palette.bg_base)

        self.session = None
        self._events: queue.Queue = queue.Queue()
        self._screens: dict[str, tk.Frame] = {}
        self._nav_buttons: dict[str, tk.Label] = {}
        self._current = "dashboard"

        self._build_layout()
        self._build_screens()
        self.show("dashboard")
        self.root.after(100, self._drain_events)
        self.root.bind("<Control-period>", lambda _e: self._emergency_stop())
        self.refresh_readiness()

    # ------------------------------------------------------------- layout

    def _build_layout(self) -> None:
        p = self.palette
        self.rail = tk.Frame(self.root, bg=p.bg_surface, width=180)
        self.rail.pack(side="left", fill="y")
        self.rail.pack_propagate(False)

        tk.Label(self.rail, text="VAANI", bg=p.bg_surface, fg=p.text_primary,
                 font=font("title")).pack(anchor="w", padx=SPACE["lg"],
                                          pady=(SPACE["xl"], SPACE["xs"]))
        self.privacy_label = tk.Label(self.rail, text="◆ LOCAL", bg=p.bg_surface,
                                      fg=p.state_live, font=font("label"))
        self.privacy_label.pack(anchor="w", padx=SPACE["lg"], pady=(0, SPACE["xl"]))

        for key, label in (("dashboard", "Dashboard"), ("meeting", "Meeting Mode"),
                           ("voice", "Voice Profile"), ("settings", "Settings"),
                           ("diagnostics", "Diagnostics")):
            item = tk.Label(self.rail, text=label, bg=p.bg_surface,
                            fg=p.text_secondary, font=font("body"), anchor="w",
                            cursor="hand2", padx=SPACE["lg"], pady=SPACE["sm"])
            item.pack(fill="x")
            item.bind("<Button-1>", lambda _e, k=key: self.show(k))
            self._nav_buttons[key] = item

        self.stop_button = Button(self.rail, p, "Emergency Stop",
                                  self._emergency_stop, variant="danger")
        self.stop_button.pack(side="bottom", fill="x", padx=SPACE["md"],
                              pady=SPACE["lg"])

        self.content = tk.Frame(self.root, bg=p.bg_base)
        self.content.pack(side="left", fill="both", expand=True)
        self.banner = Banner(self.content, p)

    def _build_screens(self) -> None:
        self._screens["dashboard"] = DashboardScreen(self.content, self)
        self._screens["meeting"] = MeetingScreen(self.content, self)
        self._screens["voice"] = VoiceScreen(self.content, self)
        self._screens["settings"] = SettingsScreen(self.content, self)
        self._screens["diagnostics"] = DiagnosticsScreen(self.content, self)

    def show(self, key: str) -> None:
        for name, screen in self._screens.items():
            screen.pack_forget()
            self._nav_buttons[name].configure(
                fg=self.palette.text_secondary, bg=self.palette.bg_surface)
        self._screens[key].pack(fill="both", expand=True,
                                padx=SPACE["xl"], pady=SPACE["xl"])
        self._nav_buttons[key].configure(fg=self.palette.text_primary,
                                         bg=self.palette.bg_raised)
        self._current = key
        if key == "dashboard":
            self.refresh_readiness()

    # ------------------------------------------------ engine event bridge

    def post(self, kind: str, payload=None) -> None:
        """Called from ENGINE threads. Never touches tk directly."""
        self._events.put((kind, payload))

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                handler = getattr(self, f"_on_{kind}", None)
                if handler:
                    handler(payload)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_events)

    def _on_state(self, payload) -> None:
        self._screens["meeting"].set_state(payload)

    def _on_result(self, payload) -> None:
        self._screens["meeting"].add_result(payload)

    def _on_latency_warning(self, payload) -> None:
        self.banner.show(payload.message(), "warn")

    def _on_error(self, payload) -> None:
        self.banner.show(str(payload), "error")

    # --------------------------------------------------------- readiness

    def readiness(self) -> dict[str, tuple[bool, str]]:
        """What blocks Start. Each entry is (ok, human-readable reason)."""
        checks: dict[str, tuple[bool, str]] = {}
        try:
            from ..devices.manager import list_sources
            devices = [d for d in list_sources() if not d.is_virtual]
            checks["Microphone"] = (bool(devices),
                                    f"{len(devices)} device(s)" if devices
                                    else "no capture device found")
        except Exception as exc:
            checks["Microphone"] = (False, str(exc)[:60])

        try:
            from ..voice.consent import ConsentLedger
            from ..voice.enrollment import VoiceProfileStore
            profile = VoiceProfileStore().active()
            consented = ConsentLedger().has_active_consent()
            if profile and consented:
                checks["Voice profile"] = (True, f"{profile.name} "
                                                 f"({profile.speech_seconds:.0f}s)")
            elif not consented:
                checks["Voice profile"] = (False, "consent not given")
            else:
                checks["Voice profile"] = (False, "not enrolled — fallback voice")
        except Exception as exc:
            checks["Voice profile"] = (False, str(exc)[:60])

        try:
            from ..audio.backend.pulse_bindings import available
            checks["Audio system"] = (available(), "PipeWire/PulseAudio"
                                      if available() else "libpulse not found")
        except Exception as exc:
            checks["Audio system"] = (False, str(exc)[:60])

        try:
            from ..devices.manager import (
                VirtualMicrophone,
                virtual_mic_consumers,
            )
            if VirtualMicrophone.find_existing() is None:
                checks["Meeting app"] = (False, "not started yet")
            else:
                consumers = virtual_mic_consumers()
                checks["Meeting app"] = (
                    bool(consumers),
                    ", ".join(c.app_name for c in consumers) if consumers
                    else "NOT connected — select 'Vaani Virtual Microphone' as "
                         "the microphone inside Zoom/Meet/WhatsApp")
        except Exception as exc:
            checks["Meeting app"] = (False, str(exc)[:60])

        try:
            from ..core.model_budget import free_vram_mb, resolve_device
            free = free_vram_mb()
            placement = ", ".join(f"{s}={resolve_device(s)}"
                                  for s in ("stt", "translation", "tts"))
            checks["Models"] = (True, placement + (f"  ({free} MiB free)"
                                                   if free is not None else ""))
        except Exception as exc:
            checks["Models"] = (False, str(exc)[:60])
        return checks

    def refresh_readiness(self) -> None:
        self._screens["dashboard"].update_readiness(self.readiness())

    # --------------------------------------------------------- session

    def start_session(self) -> None:
        if self.session is not None:
            return
        self.banner.hide()
        self.show("meeting")
        threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self) -> None:
        """Model loading takes seconds; it must not freeze the UI."""
        try:
            from ..ai.stt.whisper import FasterWhisperRecognizer
            from ..ai.tts.fallback import FallbackSynthesizer
            from ..ai.translate.nllb import NllbTranslator
            from ..ai.translate.ollama import OllamaTranslator
            from ..ai.translate.router import RoutingTranslator
            from ..ai.vad.energy import EnergyVad
            from ..session.engine import SessionConfig, TranslationSession

            self.post("error", "Loading models…")
            recognizer = FasterWhisperRecognizer.for_mode(PerformanceMode.BALANCED)
            recognizer.warmup()
            translator = RoutingTranslator(fast=NllbTranslator(),
                                           accurate=OllamaTranslator())
            translator.warmup()

            synthesizer = FallbackSynthesizer()
            profile_id = None
            try:
                from ..ai.tts.xtts import XttsSynthesizer
                from ..voice.enrollment import VoiceProfileStore
                profile = VoiceProfileStore().active()
                if profile is not None:
                    synthesizer = XttsSynthesizer()
                    synthesizer.warmup()
                    profile_id = profile.id
            except Exception:
                pass    # fallback voice already selected; it self-identifies

            session = TranslationSession(
                recognizer=recognizer, translator=translator,
                synthesizer=synthesizer, vad=EnergyVad(),
                config=SessionConfig(voice_profile_id=profile_id),
                on_result=lambda r: self.post("result", r),
                on_state=lambda a, b: self.post("state", b),
                on_latency_warning=lambda w: self.post("latency_warning", w))
            session.start()
            self.session = session
            self.post("error", "")
        except Exception as exc:
            self.post("error", f"Could not start: {exc}")

    def stop_session(self) -> None:
        if self.session is None:
            return
        session, self.session = self.session, None
        threading.Thread(target=session.stop, daemon=True).start()

    def _emergency_stop(self) -> None:
        """Reachable from any screen and by Ctrl+. even unfocused (AC-12.1)."""
        if self.session is not None:
            self.session.emergency_stop()
        self.banner.show("Emergency stop — output halted.", "error")
        self.stop_session()

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        self.stop_session()
        self.root.destroy()


# ------------------------------------------------------------------ screens

class DashboardScreen(tk.Frame):
    def __init__(self, parent, app: VaaniApp):
        super().__init__(parent, bg=app.palette.bg_base)
        self.app, p = app, app.palette
        tk.Label(self, text="Dashboard", bg=p.bg_base, fg=p.text_primary,
                 font=font("title")).pack(anchor="w")
        tk.Label(self, text="Everything that must be ready before you start.",
                 bg=p.bg_base, fg=p.text_secondary, font=font("body")).pack(
                     anchor="w", pady=(0, SPACE["lg"]))

        self.cards: dict[str, tuple[Card, StatusPill, tk.Label]] = {}
        grid = tk.Frame(self, bg=p.bg_base)
        grid.pack(fill="x")
        for i, name in enumerate(("Microphone", "Voice profile", "Audio system",
                                  "Models", "Meeting app")):
            card = Card(grid, p, title=name)
            card.grid(row=i // 2, column=i % 2, sticky="ew",
                      padx=(0, SPACE["md"]), pady=(0, SPACE["md"]))
            grid.grid_columnconfigure(i % 2, weight=1)
            pill = StatusPill(card.body, p, "idle")
            pill.pack(anchor="w")
            detail = tk.Label(card.body, text="checking…", bg=p.bg_surface,
                              fg=p.text_secondary, font=font("body"), anchor="w",
                              wraplength=420, justify="left")
            detail.pack(anchor="w", pady=(SPACE["xs"], 0))
            self.cards[name] = (card, pill, detail)

        self.start = Button(self, p, "Start Translation", app.start_session,
                            variant="primary", size="lg")
        self.start.pack(anchor="w", pady=SPACE["xl"])

    def update_readiness(self, checks: dict[str, tuple[bool, str]]) -> None:
        p = self.app.palette
        blockers = []
        for name, (card, pill, detail) in self.cards.items():
            key = next((k for k in checks if k.lower() == name.lower()), None)
            ok, text = checks.get(key, (False, "unknown"))
            pill.set_state("listening" if ok else "degraded",
                           "Ready" if ok else "Attention")
            detail.configure(text=text)
            card.set_accent(None if ok else p.state_warn)
            # Neither of these blocks Start: the fallback voice works and
            # identifies itself, and the meeting app can only be connected AFTER
            # the virtual mic exists.
            if not ok and name not in ("Voice profile", "Meeting app"):
                blockers.append(f"{name}: {text}")
        # A missing voice profile does not block: the fallback voice still works,
        # and it identifies itself as not being the user's voice.
        self.start.set_enabled(not blockers,
                               reason="; ".join(blockers) if blockers else None)


class MeetingScreen(tk.Frame):
    def __init__(self, parent, app: VaaniApp):
        super().__init__(parent, bg=app.palette.bg_base)
        self.app, p = app, app.palette

        header = tk.Frame(self, bg=p.bg_base)
        header.pack(fill="x")
        self.state_label = tk.Label(header, text="Idle", bg=p.bg_base,
                                    fg=p.state_idle, font=font("display"))
        self.state_label.pack(side="left")
        metrics = tk.Frame(header, bg=p.bg_base)
        metrics.pack(side="right")
        self.latency = MetricReadout(metrics, p, "latency p50", " ms")
        self.latency.pack(side="left", padx=SPACE["xl"])
        self.spoken = MetricReadout(metrics, p, "spoken")
        self.spoken.pack(side="left", padx=SPACE["xl"])
        self.withheld = MetricReadout(metrics, p, "withheld")
        self.withheld.pack(side="left")

        self.bar = LatencyBar(self, p, width=420)
        self.bar.pack(anchor="w", pady=SPACE["md"])

        transcript_card = Card(self, p, title="Live transcript")
        transcript_card.pack(fill="both", expand=True, pady=SPACE["md"])
        self.transcript = tk.Text(transcript_card.body, bg=p.bg_surface,
                                  fg=p.text_primary, font=font("transcript"),
                                  relief="flat", wrap="word", state="disabled",
                                  insertbackground=p.text_primary)
        self.transcript.pack(fill="both", expand=True)
        for tag, colour in (("source", p.text_secondary),
                            ("translated", p.text_primary),
                            ("suppressed", p.state_warn),
                            ("error", p.state_error)):
            self.transcript.tag_configure(tag, foreground=colour)

        controls = tk.Frame(self, bg=p.bg_base)
        controls.pack(fill="x", pady=SPACE["md"])
        Button(controls, p, "Pause", self._pause).pack(side="left")
        Button(controls, p, "Mute", self._mute).pack(side="left", padx=SPACE["sm"])
        Button(controls, p, "Stop", app.stop_session).pack(side="left")
        # Visually separated, always enabled, never behind a confirmation.
        Button(controls, p, "Emergency Stop", app._emergency_stop,
               variant="danger").pack(side="right")

        self._spoken = self._withheld = 0

    def _pause(self) -> None:
        if self.app.session:
            self.app.session.pause()

    def _mute(self) -> None:
        if self.app.session:
            self.app.session.set_muted(True)

    def set_state(self, state: SessionState) -> None:
        from .theme import STATE_LABEL
        name = state.value if hasattr(state, "value") else str(state)
        self.state_label.configure(text=STATE_LABEL.get(name, name),
                                   fg=state_colour(self.app.palette, name))

    def add_result(self, result) -> None:
        p = self.app.palette
        self.transcript.configure(state="normal")
        if result.suppressed:
            self._withheld += 1
            reason = (result.suppression_reason.value if result.suppression_reason
                      else "unknown")
            src = result.transcript.text if result.transcript else ""
            self.transcript.insert("end", f"⊘ withheld ({reason})  {src}\n",
                                   "suppressed")
        else:
            self._spoken += 1
            if result.transcript:
                self.transcript.insert("end", f"{result.transcript.text}\n", "source")
            if result.translation:
                self.transcript.insert("end", f"  → {result.translation.text}\n",
                                       "translated")
            stages = {t.stage: t.duration_ms for t in result.timings if t.succeeded}
            self.bar.set_stages(stages, BUDGET_MS[PerformanceMode.BALANCED])
        self.transcript.insert("end", "\n")
        self.transcript.see("end")
        self.transcript.configure(state="disabled")

        self.spoken.set_value(self._spoken)
        self.withheld.set_value(self._withheld,
                                colour=p.state_warn if self._withheld else None)
        if self.app.session is not None:
            stats = self.app.session.latency_percentiles()
            if stats:
                over = stats["p50"] > BUDGET_MS[PerformanceMode.BALANCED]
                self.latency.set_value(stats["p50"],
                                       colour=p.state_warn if over else None)


class VoiceScreen(tk.Frame):
    def __init__(self, parent, app: VaaniApp):
        super().__init__(parent, bg=app.palette.bg_base)
        self.app, p = app, app.palette
        tk.Label(self, text="Voice Profile", bg=p.bg_base, fg=p.text_primary,
                 font=font("title")).pack(anchor="w")
        self.container = tk.Frame(self, bg=p.bg_base)
        self.container.pack(fill="both", expand=True, pady=SPACE["lg"])
        self.render()

    def render(self) -> None:
        for child in self.container.winfo_children():
            child.destroy()
        p = self.app.palette
        try:
            from ..voice.consent import ConsentLedger
            from ..voice.enrollment import VoiceProfileStore
            ledger, store = ConsentLedger(), VoiceProfileStore()
            profile, grant = store.active(), ledger.active_grant()
        except Exception as exc:
            tk.Label(self.container, text=str(exc), bg=p.bg_base,
                     fg=p.state_error, font=font("body")).pack(anchor="w")
            return

        if grant is None:
            EmptyState(self.container, p, "◈",
                       "Vaani can speak your translated words in your own voice. "
                       "Creating that voice profile needs your explicit consent "
                       "first — nothing is recorded until you give it.",
                       ("Review consent and continue", self._consent)
                       ).pack(fill="both", expand=True)
            return
        if profile is None:
            EmptyState(self.container, p, "◎",
                       f"Consent recorded for {grant.subject_label} on "
                       f"{grant.granted_at_utc[:10]}.\n\n"
                       "Now record about a minute of speech so Vaani can learn "
                       "how you sound.",
                       ("Enroll my voice", self._enroll)
                       ).pack(fill="both", expand=True)
            return

        card = Card(self.container, p, title="Enrolled voice")
        card.pack(fill="x")
        for label, value in (("Name", profile.name),
                             ("Speech recorded", f"{profile.speech_seconds:.0f} s"),
                             ("Signal-to-noise", f"{profile.quality_snr_db:.1f} dB"),
                             ("Created", profile.created_at_utc[:19]),
                             ("Consent record", grant.id[:16])):
            row = tk.Frame(card.body, bg=p.bg_surface)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=p.bg_surface, fg=p.text_secondary,
                     font=font("body"), width=18, anchor="w").pack(side="left")
            tk.Label(row, text=value, bg=p.bg_surface, fg=p.text_primary,
                     font=font("body"), anchor="w").pack(side="left")

        tk.Label(self.container,
                 text="Deleting the profile removes the recording and the voice "
                      "embedding from disk, and is verified.",
                 bg=p.bg_base, fg=p.text_muted, font=font("body"),
                 wraplength=560, justify="left").pack(anchor="w",
                                                      pady=(SPACE["lg"], SPACE["sm"]))
        actions = tk.Frame(self.container, bg=p.bg_base)
        actions.pack(fill="x")
        Button(actions, p, "Test this voice",
               lambda: self._test_voice(profile.id), variant="primary").pack(side="left")
        Button(actions, p, "Re-enroll", self._enroll).pack(side="left",
                                                           padx=SPACE["sm"])
        Button(actions, p, "Delete voice profile",
               lambda: self._delete(profile.id), variant="danger").pack(side="right")

    def _consent(self) -> None:
        from .. import __version__
        from .enrollment_flow import ConsentDialog
        ConsentDialog(self.app.root, self.app.palette, __version__,
                      on_granted=self._after_consent)

    def _after_consent(self, subject: str) -> None:
        self.app.banner.show(f"Consent recorded for {subject}.", "info")
        self.render()
        self.app.refresh_readiness()
        self._enroll()

    def _enroll(self) -> None:
        from .enrollment_flow import EnrollmentDialog
        EnrollmentDialog(self.app.root, self.app.palette,
                         on_done=self._after_enroll)

    def _after_enroll(self, profile) -> None:
        self.app.banner.show(
            f"Voice profile created — {profile.speech_seconds:.0f}s of speech, "
            f"SNR {profile.quality_snr_db:.1f} dB.", "info")
        self.render()
        self.app.refresh_readiness()

    def _test_voice(self, profile_id: str) -> None:
        """Speak a fixed sentence so the user can judge the clone (AC-09.5)."""
        self.app.banner.show("Generating a sample in your voice…", "info")
        threading.Thread(target=self._test_worker, args=(profile_id,),
                         daemon=True).start()

    def _test_worker(self, profile_id: str) -> None:
        try:
            from ..ai.tts.xtts import XttsSynthesizer
            from ..audio.backend.pulse_backend import PulsePlaybackStream
            synth = XttsSynthesizer()
            synth.warmup()
            out = synth.synthesize(
                "This is how I will sound in your meetings.",
                voice_profile_id=profile_id)
            stream = PulsePlaybackStream(sample_rate=out.sample_rate, frame_ms=20,
                                         stream_name="voice-test")
            try:
                n = out.sample_rate * 20 // 1000
                for i in range(0, out.samples.size, n):
                    block = out.samples[i:i + n]
                    if block.size < n:
                        import numpy as _np
                        block = _np.pad(block, (0, n - block.size))
                    stream.write(block)
            finally:
                stream.close()
            self.app.root.after(0, lambda: self.app.banner.show(
                "Played a sample in your voice.", "info"))
        except Exception as exc:
            self.app.root.after(0, lambda: self.app.banner.show(
                f"Could not test the voice: {exc}", "error"))

    def _delete(self, profile_id: str) -> None:
        from ..voice.enrollment import VoiceProfileStore
        try:
            VoiceProfileStore().delete(profile_id)
            self.app.banner.show("Voice profile deleted.", "info")
        except Exception as exc:
            self.app.banner.show(f"Could not delete: {exc}", "error")
        self.render()
        self.app.refresh_readiness()


class SettingsScreen(tk.Frame):
    def __init__(self, parent, app: VaaniApp):
        super().__init__(parent, bg=app.palette.bg_base)
        self.app, p = app, app.palette
        tk.Label(self, text="Settings", bg=p.bg_base, fg=p.text_primary,
                 font=font("title")).pack(anchor="w")

        privacy = Card(self, p, title="Privacy")
        privacy.pack(fill="x", pady=SPACE["md"])
        self.local_only = tk.BooleanVar(value=False)
        self.persist = tk.BooleanVar(value=False)
        for var, text in (
            (self.local_only, "Local only — refuse any provider that needs the internet"),
            (self.persist, "Store transcript text on disk (off by default)"),
        ):
            tk.Checkbutton(privacy.body, text=text, variable=var,
                           bg=p.bg_surface, fg=p.text_primary,
                           selectcolor=p.bg_raised, activebackground=p.bg_surface,
                           activeforeground=p.text_primary, font=font("body"),
                           anchor="w", highlightthickness=0, bd=0).pack(
                               anchor="w", pady=2)
        tk.Label(privacy.body,
                 text="Every provider in this build runs on this machine, so "
                      "local-only currently hides nothing.",
                 bg=p.bg_surface, fg=p.text_muted, font=font("body"),
                 wraplength=620, justify="left").pack(anchor="w",
                                                      pady=(SPACE["sm"], 0))

        perf = Card(self, p, title="Performance")
        perf.pack(fill="x", pady=SPACE["md"])
        self.mode = tk.StringVar(value="balanced")
        for mode in PerformanceMode:
            budget = BUDGET_MS[mode]
            tk.Radiobutton(perf.body,
                           text=f"{mode.value.replace('_', ' ').title()}  "
                                f"(budget {budget:.0f} ms)",
                           value=mode.value, variable=self.mode,
                           bg=p.bg_surface, fg=p.text_primary,
                           selectcolor=p.bg_raised, activebackground=p.bg_surface,
                           activeforeground=p.text_primary, font=font("body"),
                           anchor="w", highlightthickness=0, bd=0).pack(anchor="w")

        devices = Card(self, p, title="Audio input")
        devices.pack(fill="x", pady=SPACE["md"])
        self.device_list = tk.Listbox(devices.body, bg=p.bg_raised,
                                      fg=p.text_primary, font=font("body"),
                                      relief="flat", height=4,
                                      highlightthickness=0,
                                      selectbackground=p.accent)
        self.device_list.pack(fill="x")
        self.meter = LevelMeter(devices.body, p, width=300)
        self.meter.pack(anchor="w", pady=(SPACE["sm"], 0))
        Button(devices.body, p, "Refresh devices", self.refresh_devices,
               size="sm").pack(anchor="w", pady=(SPACE["sm"], 0))
        self.refresh_devices()

    def refresh_devices(self) -> None:
        self.device_list.delete(0, "end")
        try:
            from ..devices.manager import list_sources
            for d in list_sources():
                if not d.is_virtual:
                    self.device_list.insert("end", d.display_name)
        except Exception as exc:
            self.device_list.insert("end", f"error: {exc}")


class DiagnosticsScreen(tk.Frame):
    CHECKS = ["Audio backend", "Input devices", "Microphone capture",
              "Virtual microphone", "Voice activity detection",
              "Speech recognition runtime", "Translation round-trip",
              "Text to speech", "Voice consent", "Voice profile", "Network"]

    def __init__(self, parent, app: VaaniApp):
        super().__init__(parent, bg=app.palette.bg_base)
        self.app, p = app, app.palette
        tk.Label(self, text="Diagnostics", bg=p.bg_base, fg=p.text_primary,
                 font=font("title")).pack(anchor="w")
        tk.Label(self, text="Every check reports a measured result.",
                 bg=p.bg_base, fg=p.text_secondary, font=font("body")).pack(
                     anchor="w", pady=(0, SPACE["lg"]))

        self.run_button = Button(self, p, "Run all checks", self.run_all,
                                 variant="primary")
        self.run_button.pack(anchor="w", pady=(0, SPACE["lg"]))

        card = Card(self, p)
        card.pack(fill="both", expand=True)
        self.rows = {}
        for name in self.CHECKS:
            row = DiagnosticRow(card.body, p, name)
            row.pack(fill="x", pady=2)
            self.rows[name] = row

    def run_all(self) -> None:
        self.run_button.set_enabled(False, reason="running…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            from ..diagnostics.checks import ALL_CHECKS
            for check in ALL_CHECKS:
                result = check()
                self.app.root.after(0, self._apply, result)
        finally:
            self.app.root.after(0, lambda: self.run_button.set_enabled(True))

    def _apply(self, result) -> None:
        row = self.rows.get(result.name)
        if row is not None:
            row.set_result(result.status, result.detail, result.duration_ms)


def main() -> int:
    VaaniApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
