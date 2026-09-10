"""Speech recognition via faster-whisper (CTranslate2).

WHY faster-whisper rather than openai-whisper or transformers:
  * CTranslate2 gives int8/float16 quantisation, which is what makes a useful model
    fit in the ~1.6 GiB of VRAM actually free on the target machine.
  * It does not pull torch (~2.5 GB), which matters on a box with ~2.9 GiB free RAM.
  * Verified installable on CPython 3.14, where much of the ML ecosystem is not yet.

WHY Whisper at all for this problem: it is trained on genuinely multilingual audio
and -- crucially for Hinglish -- it transcribes code-mixed speech without being
forced into one language, because language is a decoder token rather than a
separate model. That is the single property this product needs most.

Language handling is the subtle part. Passing `language="hi"` pins the decoder and
tends to transliterate English words into Devanagari; passing `language=None` lets
Whisper detect, which is what code-mixed speech needs (AC-04.2). We therefore only
pin when the user has explicitly chosen a single-language mode.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ...core.errors import ErrorCode, Severity, VaaniError
from ...core.types import PerformanceMode, SourceLanguageMode, Transcript
from ..base import HealthState, SpeechRecognizer

#: Model per performance mode.
#:
#: Measured on the target machine (RTX 3050 Mobile, real 11 s speech):
#:   small  CPU  int8           RTF 0.262
#:   small  CUDA int8_float16   RTF 0.025
#:   medium CUDA int8_float16   RTF 0.061
#:
#: `medium` on GPU is four times faster than `small` on CPU, so when CUDA works
#: the accuracy/latency trade largely disappears and the better model is simply
#: the better choice. These are GPU-oriented defaults; `for_mode` steps down to
#: `small` on CPU, where medium costs ~2.6 s for a 4 s utterance.
_MODEL_FOR_MODE = {
    PerformanceMode.LOW_LATENCY: "small",
    PerformanceMode.BALANCED: "medium",
    PerformanceMode.QUALITY: "medium",
}


class FasterWhisperRecognizer(SpeechRecognizer):
    key = "faster_whisper"
    display_name = "Whisper (local)"
    requires_network = False  # after the first model download

    def __init__(self, *, model_size: str = "small", device: str = "auto",
                 compute_type: str | None = None, beam_size: int = 1,
                 download_root: str | None = None) -> None:
        super().__init__()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.download_root = download_root
        self._model: Any = None

    # ------------------------------------------------------------ lifecycle

    @classmethod
    def for_mode(cls, mode: PerformanceMode, **kw) -> FasterWhisperRecognizer:
        """Pick a model sized to the hardware that is actually usable.

        Beam size is 5 everywhere except low-latency mode. On this CPU beam=5 cost
        only ~1% over beam=1 (RTF 0.262 -> 0.265) and on GPU ~16% (0.025 -> 0.029),
        so the usual accuracy-for-latency trade is far cheaper than assumed --
        which is only knowable by measuring it.
        """
        from .cuda_setup import cuda_available
        gpu_ok, _ = cuda_available()
        size = _MODEL_FOR_MODE[mode]
        if not gpu_ok and size == "medium":
            # medium on CPU is RTF ~0.6+; it would blow every latency budget.
            size = "small"
        return cls(model_size=size,
                   beam_size=1 if mode is PerformanceMode.LOW_LATENCY else 5, **kw)

    def _resolve_placement(self) -> tuple[str, str]:
        """Pick device and quantisation from what this machine actually reports."""
        try:
            import ctranslate2
        except ImportError as exc:
            raise VaaniError(
                code=ErrorCode.MODEL_LOAD_FAILED,
                message="faster-whisper is not installed",
                severity=Severity.FATAL, cause=exc,
            ) from exc

        # Make pip-installed CUDA libraries loadable before we ask about devices.
        from .cuda_setup import ensure_cuda_libraries
        ensure_cuda_libraries()

        from ...core.model_budget import resolve_device
        device = resolve_device("stt", self.device)

        if self.compute_type:
            return device, self.compute_type
        supported = ctranslate2.get_supported_compute_types(device)
        # int8_float16 first: on a 4 GiB laptop GPU, memory is the binding
        # constraint, not arithmetic throughput.
        for candidate in ("int8_float16", "float16", "int8", "float32"):
            if candidate in supported:
                return device, candidate
        return device, "float32"

    def warmup(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        device, compute_type = self._resolve_placement()
        try:
            self._model = WhisperModel(
                self.model_size, device=device, compute_type=compute_type,
                download_root=self.download_root,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                raise VaaniError(
                    code=ErrorCode.INSUFFICIENT_MEMORY,
                    message=f"not enough GPU memory for Whisper '{self.model_size}'; "
                            "try a smaller model or CPU",
                    severity=Severity.FATAL,
                    detail={"model": self.model_size, "device": device},
                    cause=exc,
                ) from exc
            raise VaaniError(
                code=ErrorCode.MODEL_LOAD_FAILED,
                message=f"could not load Whisper '{self.model_size}': {exc}",
                severity=Severity.FATAL,
                detail={"model": self.model_size, "device": device},
                cause=exc,
            ) from exc

        self.device_used, self.compute_type_used = device, compute_type
        # A tiny decode so the first real utterance does not pay graph-build cost.
        try:
            list(self._model.transcribe(np.zeros(16000, dtype=np.float32),
                                        beam_size=1)[0])
        except Exception:
            pass  # warmup failure is not itself fatal
        self._set_health(HealthState.READY, f"{self.model_size} on {device}/{compute_type}")

    def shutdown(self) -> None:
        self._model = None
        super().shutdown()

    # ---------------------------------------------------------- recognition

    def transcribe(self, audio: np.ndarray, sample_rate: int, *,
                   language_hint: str | None = None,
                   context_prompt: str | None = None) -> Transcript:
        if self._model is None:
            self.warmup()
        if sample_rate != 16000:
            raise VaaniError(code=ErrorCode.CONFIG_INVALID,
                             message=f"Whisper needs 16 kHz audio, got {sample_rate}",
                             severity=Severity.SESSION)
        if audio.size == 0:
            raise VaaniError(code=ErrorCode.STT_EMPTY, message="empty audio",
                             stage="stt", provider_key=self.key)

        started = time.perf_counter()
        try:
            segments, info = self._model.transcribe(
                np.asarray(audio, dtype=np.float32),
                language=language_hint,          # None => detect (code-mix safe)
                beam_size=self.beam_size,
                task="transcribe",               # never "translate": Whisper's own
                                                 # translation bypasses our context
                                                 # engine and our confidence gate
                initial_prompt=context_prompt,
                condition_on_previous_text=False,  # stops runaway repetition loops
                vad_filter=False,                  # our segmenter already did this
                word_timestamps=False,
            )
            collected = list(segments)
        except Exception as exc:
            raise self._fail(ErrorCode.STT_FAILURE,
                             f"recognition failed: {exc}", cause=exc) from exc

        text = " ".join(s.text.strip() for s in collected).strip()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not text:
            raise VaaniError(code=ErrorCode.STT_EMPTY,
                             message="no speech recognised", stage="stt",
                             provider_key=self.key,
                             detail={"duration_ms": round(elapsed_ms, 1)})

        self._set_health(HealthState.READY)
        return Transcript(
            text=text,
            language_distribution=_language_distribution(text, info),
            confidence=_confidence(collected),
            is_final=True,
            duration_ms=elapsed_ms,
        )

    def supports_streaming(self) -> bool:
        return False


def _confidence(segments: list[Any]) -> float:
    """Duration-weighted mean of exp(avg_logprob).

    Weighting by duration matters: a stray one-word segment with a terrible score
    should not drag down a long, confidently-decoded sentence, and vice versa.
    """
    if not segments:
        return 0.0
    total_w = 0.0
    total = 0.0
    for s in segments:
        w = max(1e-3, float(getattr(s, "end", 0.0) - getattr(s, "start", 0.0)))
        lp = float(getattr(s, "avg_logprob", -1.0))
        total += math.exp(max(-10.0, lp)) * w
        total_w += w
    conf = total / total_w if total_w else 0.0

    # No-speech probability is a separate failure mode from low logprob: Whisper
    # can be confidently wrong on silence, so penalise it explicitly.
    ns = max((float(getattr(s, "no_speech_prob", 0.0)) for s in segments), default=0.0)
    return float(max(0.0, min(1.0, conf * (1.0 - 0.5 * ns))))


#: Below this, Whisper's language detection is not reliable enough to act on.
#: Measured: a one-word "hello" was detected as Arabic at p=0.6 and rendered in
#: Arabic script. Language ID needs more than a word to work with.
_MIN_RELIABLE_LANGID_CHARS = 12


def _language_distribution(text: str, info: Any) -> dict[str, float]:
    """Estimate the language mix of one utterance.

    Whisper reports a single detected language with a probability -- it has no
    notion of code-mixing. We combine that prior with the actual script ratio in
    the decoded text, which is directly observable and is what distinguishes
    "Actually मेरा point" from a monolingual sentence (AC-04.2).
    """
    devanagari = sum(1 for ch in text if "ऀ" <= ch <= "ॿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = devanagari + latin

    detected = str(getattr(info, "language", "") or "")
    detected_prob = float(getattr(info, "language_probability", 0.0) or 0.0)

    if total == 0:
        # Neither Devanagari nor Latin: some third script. Report it as detected
        # so the gate can suppress it rather than letting an untranslatable
        # language reach the translator.
        return {detected: detected_prob} if detected else {}

    # Short utterances: the script ratio is directly observable and trustworthy,
    # but the model's language guess is not. Prefer the evidence over the guess.
    if len(text) < _MIN_RELIABLE_LANGID_CHARS:
        script_hi = devanagari / total
        return {k: round(v, 3)
                for k, v in {"hi": script_hi, "en": 1.0 - script_hi}.items()
                if v > 0.01}

    script_hi = devanagari / total
    dist = {"hi": script_hi, "en": 1.0 - script_hi}

    # Romanised Hinglish ("mera point ye hai") is all-Latin script, so the script
    # ratio alone would call it English. Whisper's own detection is the only
    # signal that catches it, so let a confident "hi" pull the estimate back.
    if detected == "hi" and detected_prob > 0.7 and script_hi < 0.2:
        dist = {"hi": 0.5, "en": 0.5}

    return {k: round(v, 3) for k, v in dist.items() if v > 0.01}


def language_hint_for(mode: SourceLanguageMode) -> str | None:
    """Only pin the decoder when the user asked for a single language.

    AUTO and HINGLISH both return None so Whisper stays free to mix.
    """
    if mode is SourceLanguageMode.HINDI:
        return "hi"
    if mode is SourceLanguageMode.ENGLISH:
        return "en"
    return None
