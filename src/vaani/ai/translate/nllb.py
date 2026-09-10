"""Local translation with NLLB-200 (distilled 600M) via CTranslate2.

WHY NLLB-200 distilled 600M:
  * Already runs on the CTranslate2 runtime this project uses for STT, so it adds
    no new inference stack and inherits the same int8/float16 quantisation.
  * ~600 MB at int8 -- fits alongside Whisper in the memory this machine actually
    has free, which ruled out the larger 1.3B and 3.3B variants.
  * Genuinely offline once downloaded, which is what makes local-only mode real.

WHY NOT the alternatives:
  * IndicTrans2 is better on clean Indic->English but ships a bespoke tokenizer and
    preprocessing chain, and is larger. Worth revisiting if quality is the blocker.
  * A cloud LLM is materially better at code-mixed input and is the only option that
    genuinely uses conversational context -- see the honest limitations below.

HONEST LIMITATIONS -- read before trusting the output:

1. **Code-mixing is out of distribution.** NLLB was trained on monolingual sentence
   pairs. "Actually मेरा point ये है कि deadline tight है" is not Hindi and is not
   English; it is a register NLLB never saw. It is handled here by script-aware
   preprocessing (below), not by the model understanding Hinglish.
2. **Context is not used by the model.** NLLB is sentence-level with no mechanism
   to condition on prior turns. The `context` argument is accepted for interface
   compatibility and is honestly reported as unused via `used_context_turns=0`.
   This means "उसमें authentication वाला part बाकी है" will NOT resolve "उसमें"
   against an earlier turn -- the PRD's own motivating example (AC-06.2) is a case
   this engine cannot satisfy. A context-capable engine is required for that.
3. **Romanised Hindi is the weak spot.** "mera point ye hai" is Latin script but
   Hindi language; NLLB tagged as English will pass it through largely unchanged.
   The gate catches Devanagari leakage, but not this.

These are stated here rather than discovered later because the product's whole
safety posture is that it would rather stay silent than mislead.
"""
from __future__ import annotations

import re
import time
from typing import Any

from ...core.errors import ErrorCode, Severity, VaaniError
from ...core.types import Translation
from ..base import HealthState, TranslationEngine
from .passthrough import normalize_for_speech

#: NLLB uses FLORES-200 language codes, not ISO-639.
_LANG_CODES = {
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "ur": "urd_Arab",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
}

_DEFAULT_MODEL = "entai2965/nllb-200-distilled-600M-ctranslate2"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


class NllbTranslator(TranslationEngine):
    key = "nllb_600m"
    display_name = "NLLB-200 600M (local)"
    requires_network = False  # after the first download

    def __init__(self, *, model_id: str = _DEFAULT_MODEL, device: str = "auto",
                 compute_type: str | None = None, beam_size: int = 4,
                 max_input_tokens: int = 512) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.max_input_tokens = max_input_tokens
        self._translator: Any = None
        self._tokenizer: Any = None
        self.device_used = ""
        self.compute_type_used = ""

    # ------------------------------------------------------------ lifecycle

    def warmup(self) -> None:
        if self._translator is not None:
            return
        try:
            import ctranslate2
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise VaaniError(
                code=ErrorCode.MODEL_LOAD_FAILED,
                message="local translation needs `transformers` and `sentencepiece`",
                severity=Severity.FATAL, cause=exc,
            ) from exc

        # Reuse the CUDA preload the STT provider needs; harmless if already done.
        try:
            from ..stt.cuda_setup import ensure_cuda_libraries
            ensure_cuda_libraries()
        except Exception:
            pass

        try:
            path = snapshot_download(self.model_id)
        except Exception as exc:
            raise VaaniError(
                code=ErrorCode.NETWORK_UNAVAILABLE,
                message=f"could not download the translation model {self.model_id}",
                severity=Severity.FATAL, cause=exc,
            ) from exc

        # Resolve against free VRAM, not merely whether a GPU exists -- another
        # application may already have filled the card.
        from ...core.model_budget import resolve_device
        device = resolve_device("translation", self.device)
        compute_type = self.compute_type
        if not compute_type:
            supported = ctranslate2.get_supported_compute_types(device)
            compute_type = next((c for c in ("int8_float16", "int8", "float32")
                                 if c in supported), "float32")

        try:
            self._translator = ctranslate2.Translator(
                path, device=device, compute_type=compute_type)
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception as exc:
            msg = str(exc).lower()
            code = (ErrorCode.INSUFFICIENT_MEMORY if "memory" in msg
                    else ErrorCode.MODEL_LOAD_FAILED)
            raise VaaniError(
                code=code,
                message=f"could not load the translation model: {exc}",
                severity=Severity.FATAL,
                detail={"model": self.model_id, "device": device}, cause=exc,
            ) from exc

        self.device_used, self.compute_type_used = device, compute_type
        self._set_health(HealthState.READY, f"{device}/{compute_type}")

    def shutdown(self) -> None:
        self._translator = None
        self._tokenizer = None
        super().shutdown()

    # --------------------------------------------------------- translation

    def translate(self, text: str, *, source_language: str, target_language: str,
                  context: list[tuple[str, str]] | None = None) -> Translation:
        if self._translator is None:
            self.warmup()

        text = text.strip()
        if not text:
            raise VaaniError(code=ErrorCode.TRANSLATION_EMPTY, message="empty input",
                             stage="translate", provider_key=self.key)

        src = _LANG_CODES.get(source_language)
        tgt = _LANG_CODES.get(target_language)
        if src is None or tgt is None:
            raise VaaniError(
                code=ErrorCode.LANGID_UNSUPPORTED,
                message=f"unsupported language pair {source_language}->{target_language}",
                stage="translate", provider_key=self.key,
            )

        # Code-mix handling. NLLB expects one source language, so when the text
        # actually contains Devanagari we tag it Hindi regardless of what the
        # recogniser's dominant-language guess was: the script is direct evidence,
        # and mis-tagging mixed text as English makes NLLB pass it through
        # untranslated -- which the gate then correctly suppresses, losing the
        # utterance entirely.
        if _DEVANAGARI.search(text) and src != _LANG_CODES["hi"]:
            src = _LANG_CODES["hi"]

        started = time.perf_counter()
        try:
            self._tokenizer.src_lang = src
            tokens = self._tokenizer.convert_ids_to_tokens(
                self._tokenizer.encode(text, truncation=True,
                                       max_length=self.max_input_tokens))
            results = self._translator.translate_batch(
                [tokens],
                target_prefix=[[tgt]],
                beam_size=self.beam_size,
                # NLLB's failure mode on out-of-distribution input is looping;
                # these bound it rather than letting it fill the window.
                max_decoding_length=256,
                repetition_penalty=1.1,
                no_repeat_ngram_size=4,
                return_scores=True,
            )
        except Exception as exc:
            raise self._fail(ErrorCode.TRANSLATION_FAILURE,
                             f"translation failed: {exc}", cause=exc) from exc

        if not results or not results[0].hypotheses:
            raise VaaniError(code=ErrorCode.TRANSLATION_EMPTY,
                             message="translator returned nothing",
                             stage="translate", provider_key=self.key)

        hypothesis = results[0].hypotheses[0]
        # Drop the target-language token the prefix forced.
        if hypothesis and hypothesis[0] == tgt:
            hypothesis = hypothesis[1:]
        out = self._tokenizer.decode(
            self._tokenizer.convert_tokens_to_ids(hypothesis),
            skip_special_tokens=True)

        elapsed_ms = (time.perf_counter() - started) * 1000
        self._set_health(HealthState.READY)
        return Translation(
            text=normalize_for_speech(out),
            source_language=source_language,
            target_language=target_language,
            confidence=_confidence(results[0], len(hypothesis)),
            # Honestly zero: NLLB cannot condition on prior turns. Reporting the
            # window size here would misrepresent what the engine did.
            used_context_turns=0,
            provider_key=self.key,
        )

    def supports_code_mixed(self) -> bool:
        """False, and deliberately so.

        Script-aware source tagging makes mixed input *survivable*, not understood.
        Claiming True here would let the pipeline skip preprocessing this engine
        actually needs.
        """
        return False


def _confidence(result: Any, n_tokens: int) -> float:
    """Map CTranslate2's length-normalised log-likelihood onto [0, 1].

    The mapping is calibrated loosely rather than precisely: it exists to feed the
    confidence gate, and the gate's threshold is the thing that should be tuned
    against real output. exp() of the raw score would saturate near 1.0 for almost
    everything and make the gate useless.
    """
    scores = getattr(result, "scores", None)
    if not scores:
        return 0.5
    score = float(scores[0])
    # Typical range for reasonable NLLB output is about -0.1 to -1.5.
    conf = 1.0 + score / 2.0
    if n_tokens <= 2:
        # Very short outputs get extreme scores in both directions; damp them.
        conf = min(conf, 0.75)
    return float(max(0.0, min(1.0, conf)))
