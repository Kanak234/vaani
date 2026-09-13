"""Integrated Windows desktop console for Vaani meeting takeover and screen analysis."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..core.types import PerformanceMode
from ..session.meeting_takeover_app import MeetingTakeoverApp
from ..vision.screen_recording import analyze_recording, _ollama_chat

MIC_DEFAULT = "Microphone (HP USB Sound Device)"
REMOTE_DEFAULT = "Speakers (HP USB Sound Device)"
OUTPUT_DEFAULT = "CABLE Input (VB-Audio Virtual Cable)"


class WindowsVaaniConsole(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Vaani — Windows Meeting Console")
        self.geometry("1180x820")
        self.minsize(980, 700)
        self._meeting = None
        self._analysis = None
        self._response = ""
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="Vaani", font=("Segoe UI", 24, "bold")).pack(anchor="w")
        ttk.Label(root, text="Windows Meeting Takeover + Screen Context", font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 14))

        devices = ttk.LabelFrame(root, text="Meeting audio", padding=12)
        devices.pack(fill="x", pady=(0, 12))
        self.mic = self._field(devices, "Your microphone", MIC_DEFAULT, 0)
        self.remote = self._field(devices, "Meeting speaker loopback", REMOTE_DEFAULT, 1)
        self.output = self._field(devices, "VB-CABLE playback", OUTPUT_DEFAULT, 2)
        self.model = self._field(devices, "Answer model", "qwen3:8b", 3)
        self.voice = tk.StringVar(value="fallback")
        ttk.Label(devices, text="Voice").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Combobox(devices, textvariable=self.voice, values=("fallback", "personal"), state="readonly", width=28).grid(row=4, column=1, sticky="ew", pady=5)
        devices.columnconfigure(1, weight=1)

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(0, 12))
        self.start_btn = ttk.Button(controls, text="ARM + START MEETING", command=self.start_meeting)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="STOP / EMERGENCY STOP", command=self.stop_meeting, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        self.status = tk.StringVar(value="OFF — not running")
        ttk.Label(controls, textvariable=self.status).pack(side="left", padx=12)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        meeting_tab = ttk.Frame(notebook, padding=12)
        screen_tab = ttk.Frame(notebook, padding=12)
        notebook.add(meeting_tab, text="Meeting Takeover")
        notebook.add(screen_tab, text="Screen Recording Analysis")

        ttk.Label(meeting_tab, text="Takeover is armed only while this session is running. STOP is authoritative.").pack(anchor="w")
        self.meeting_log = tk.Text(meeting_tab, height=20, wrap="word")
        self.meeting_log.pack(fill="both", expand=True, pady=(10, 0))
        self.meeting_log.insert("end", "Ready. Select the VB-CABLE playback endpoint as the meeting application's microphone.\n")

        file_row = ttk.Frame(screen_tab)
        file_row.pack(fill="x")
        self.recording = tk.StringVar()
        ttk.Entry(file_row, textvariable=self.recording).pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Choose recording…", command=self.choose_recording).pack(side="left", padx=8)
        self.vision_model = tk.StringVar(value="qwen3-vl:4b")
        ttk.Label(screen_tab, text="Local vision model").pack(anchor="w", pady=(12, 3))
        ttk.Entry(screen_tab, textvariable=self.vision_model).pack(fill="x")
        ttk.Label(screen_tab, text="Response instruction").pack(anchor="w", pady=(12, 3))
        self.response_instruction = tk.Text(screen_tab, height=4, wrap="word")
        self.response_instruction.pack(fill="x")
        self.response_instruction.insert("end", "Use the screen context to produce a concise, natural meeting response. Do not invent facts.")
        row = ttk.Frame(screen_tab)
        row.pack(fill="x", pady=10)
        ttk.Button(row, text="ANALYZE RECORDING", command=self.analyze_screen).pack(side="left")
        ttk.Button(row, text="GENERATE RESPONSE", command=self.generate_response).pack(side="left", padx=8)
        ttk.Button(row, text="SPEAK IN PERSONAL VOICE", command=self.speak_response).pack(side="left")
        self.screen_status = tk.StringVar(value="No recording analyzed")
        ttk.Label(screen_tab, textvariable=self.screen_status).pack(anchor="w")
        self.screen_text = tk.Text(screen_tab, wrap="word")
        self.screen_text.pack(fill="both", expand=True, pady=(8, 0))
        self.protocol("WM_DELETE_WINDOW", self._close)

    @staticmethod
    def _field(parent, label: str, value: str, row: int):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=5)
        var = tk.StringVar(value=value)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=5)
        return var

    def choose_recording(self) -> None:
        path = filedialog.askopenfilename(title="Choose a screen recording", filetypes=[("Video recordings", "*.mp4 *.mkv *.mov *.avi *.webm"), ("All files", "*.*")])
        if path:
            self.recording.set(path)

    def start_meeting(self) -> None:
        if self._meeting is not None:
            return
        self.status.set("Starting local STT/LLM/TTS + audio routing…")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self) -> None:
        try:
            app = MeetingTakeoverApp(input_device=self.mic.get(), remote_input_device=self.remote.get(), output_device=self.output.get(), llm_model=self.model.get().strip() or "qwen3:8b", voice=self.voice.get(), performance_mode=PerformanceMode.LOW_LATENCY)
            self._meeting = app
            app.start()
            self.after(0, lambda: self.status.set("ARMED + ACTIVE — listening for questions / hesitation"))
            self.after(0, lambda: self.meeting_log.insert("end", "\nMeeting takeover is running.\n"))
        except Exception as exc:
            self._meeting = None
            self.after(0, lambda: self.status.set("START FAILED"))
            self.after(0, lambda: messagebox.showerror("Vaani start failed", str(exc)))
            self.after(0, lambda: self.start_btn.configure(state="normal"))
            self.after(0, lambda: self.stop_btn.configure(state="disabled"))

    def stop_meeting(self) -> None:
        app = self._meeting
        self._meeting = None
        self.stop_btn.configure(state="disabled")
        self.status.set("Stopping…")
        if app is None:
            self.status.set("OFF — not running")
            self.start_btn.configure(state="normal")
            return
        threading.Thread(target=self._stop_worker, args=(app,), daemon=True).start()

    def _stop_worker(self, app) -> None:
        try:
            app.stop()
        finally:
            self.after(0, lambda: self.status.set("OFF — stopped"))
            self.after(0, lambda: self.start_btn.configure(state="normal"))

    def analyze_screen(self) -> None:
        path = self.recording.get().strip()
        if not path:
            messagebox.showwarning("Recording required", "Choose a screen recording first.")
            return
        self.screen_status.set("Analyzing sampled frames locally…")
        threading.Thread(target=self._analyze_worker, args=(path,), daemon=True).start()

    def _analyze_worker(self, path: str) -> None:
        try:
            result = analyze_recording(path, model=self.vision_model.get().strip() or "qwen3-vl:4b")
            self._analysis = result
            text = f"Recording: {Path(result.path).name}\nDuration: {result.duration_s:.1f}s | {result.width}x{result.height} | {result.fps:.1f} FPS | sampled frames: {result.frames_sampled}\n\nVISUAL CONTEXT\n{result.report}\n"
            self.after(0, lambda: self._set_screen_text(text))
        except Exception as exc:
            self.after(0, lambda: self.screen_status.set("ANALYSIS FAILED"))
            self.after(0, lambda: messagebox.showerror("Screen analysis failed", str(exc)))

    def _set_screen_text(self, text: str) -> None:
        self.screen_text.delete("1.0", "end")
        self.screen_text.insert("end", text)
        self.screen_status.set("Analysis complete — local only")

    def generate_response(self) -> None:
        if self._analysis is None:
            messagebox.showwarning("Analysis required", "Analyze a recording first.")
            return
        self.screen_status.set("Generating local response…")
        threading.Thread(target=self._response_worker, daemon=True).start()

    def _response_worker(self) -> None:
        try:
            instruction = self.response_instruction.get("1.0", "end").strip()
            prompt = ("You are Vaani, a local meeting assistant. Draft one concise spoken response based only on the visual context below. Preserve uncertainty and never invent facts, commitments, names, dates, or numbers. Output only the response.\n\n" f"Instruction: {instruction}\n\nVisual context:\n{self._analysis.report}")
            response = _ollama_chat(self.model.get().strip() or "qwen3:8b", prompt, [])
            if not response:
                raise RuntimeError("The local language model returned an empty response.")
            self._response = response
            self.after(0, lambda: self._append_screen("\n\nGENERATED RESPONSE\n" + response + "\n"))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Response generation failed", str(exc)))

    def _append_screen(self, text: str) -> None:
        self.screen_text.insert("end", text)
        self.screen_text.see("end")
        self.screen_status.set("Response ready — review before speaking")

    def speak_response(self) -> None:
        if not self._response.strip():
            messagebox.showwarning("Response required", "Generate a response first.")
            return
        if self.voice.get() != "personal":
            messagebox.showwarning("Personal voice disabled", "Set Voice to 'personal' to use the enrolled voice.")
            return
        threading.Thread(target=self._speak_worker, daemon=True).start()

    def _speak_worker(self) -> None:
        try:
            from ..ai.tts.xtts import XttsSynthesizer
            from ..voice.consent import ConsentLedger
            from ..voice.enrollment import VoiceProfileStore
            from ..audio.backend.windows_backend import WindowsPlaybackStream
            store, ledger = VoiceProfileStore(), ConsentLedger()
            profile = store.active()
            if profile is None or not ledger.has_active_consent():
                raise RuntimeError("An active enrolled voice profile and active voice consent are required.")
            synth = XttsSynthesizer(profile_store=store, consent_ledger=ledger)
            synth.warmup()
            audio = synth.synthesize(self._response, voice_profile_id=profile.id)
            with WindowsPlaybackStream(device=self.output.get(), sample_rate=16000, frame_ms=20) as player:
                for start in range(0, len(audio.samples), player.frame_samples):
                    player.write(audio.samples[start:start + player.frame_samples])
            self.after(0, lambda: self.screen_status.set("Response spoken through VB-CABLE"))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Voice synthesis failed", str(exc)))

    def _close(self) -> None:
        if self._meeting is not None:
            try:
                self._meeting.stop()
            except Exception:
                pass
            self._meeting = None
        self.destroy()


def main() -> int:
    if os.name != "nt":
        raise SystemExit("The Windows Vaani console is Windows-only.")
    WindowsVaaniConsole().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
