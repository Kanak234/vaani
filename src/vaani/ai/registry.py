"""Provider registry and the local-only guarantee (AC-17.2).

The point of this module is one rule:

    In local-only mode, a provider whose `requires_network` is True is never
    handed to the pipeline.

That check lives HERE rather than inside each provider deliberately. If every
provider had to remember to check the privacy mode itself, the guarantee would be
only as strong as the least careful provider -- and a future contributor adding a
cloud engine would have no way to know they had just silently broken it. Enforcing
it at the single point where providers are handed out makes the guarantee
structural.

The mode is also fail-closed: an unknown provider is treated as networked.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.errors import ErrorCode, Severity, VaaniError
from .base import Provider, ProviderKind


@dataclass(slots=True)
class ProviderSpec:
    key: str
    kind: ProviderKind
    display_name: str
    requires_network: bool
    factory: Callable[..., Provider]
    #: Lower runs first when falling back.
    priority: int = 100
    enabled: bool = True


class ProviderRegistry:
    """Chooses providers, subject to the privacy mode."""

    def __init__(self, *, local_only: bool = False) -> None:
        self._specs: dict[ProviderKind, list[ProviderSpec]] = {}
        self.local_only = local_only

    def register(self, spec: ProviderSpec) -> None:
        self._specs.setdefault(spec.kind, []).append(spec)
        self._specs[spec.kind].sort(key=lambda s: s.priority)

    def available(self, kind: ProviderKind) -> list[ProviderSpec]:
        """Specs usable under the current privacy mode."""
        return [s for s in self._specs.get(kind, [])
                if s.enabled and not (self.local_only and s.requires_network)]

    def excluded(self, kind: ProviderKind) -> list[ProviderSpec]:
        """Specs hidden BECAUSE of local-only mode.

        Exposed so the UI can say "3 providers hidden by local-only mode" rather
        than silently offering a shorter list, which would look like a bug.
        """
        if not self.local_only:
            return []
        return [s for s in self._specs.get(kind, [])
                if s.enabled and s.requires_network]

    def get(self, kind: ProviderKind, key: str | None = None, **kwargs) -> Provider:
        """Instantiate a provider, refusing any network provider in local-only mode."""
        candidates = self._specs.get(kind, [])
        if key is not None:
            spec = next((s for s in candidates if s.key == key), None)
            if spec is None:
                raise VaaniError(
                    code=ErrorCode.PROVIDER_UNAVAILABLE,
                    message=f"no {kind.value} provider named {key!r}",
                    severity=Severity.FATAL,
                )
            # The check that matters. Explicitly asking for a cloud provider in
            # local-only mode is an error, not a silent downgrade -- the user
            # should learn their two settings contradict each other.
            if self.local_only and spec.requires_network:
                raise VaaniError(
                    code=ErrorCode.LOCAL_ONLY_VIOLATION,
                    message=f"provider {key!r} needs the internet, but local-only "
                            "mode is on; turn local-only off or pick a local provider",
                    severity=Severity.FATAL,
                    provider_key=key,
                    detail={"kind": kind.value},
                )
            return spec.factory(**kwargs)

        usable = self.available(kind)
        if not usable:
            hidden = len(self.excluded(kind))
            extra = (f" ({hidden} hidden by local-only mode)" if hidden else "")
            raise VaaniError(
                code=ErrorCode.PROVIDER_UNAVAILABLE,
                message=f"no usable {kind.value} provider{extra}",
                severity=Severity.FATAL,
                detail={"kind": kind.value, "hidden_by_local_only": hidden},
            )
        return usable[0].factory(**kwargs)

    def assert_local_only_holds(self) -> None:
        """Belt-and-braces check callable at session start.

        Cheap, and it turns a privacy guarantee from a code-review claim into
        something a test can assert.
        """
        if not self.local_only:
            return
        for kind, specs in self._specs.items():
            for spec in specs:
                if spec.enabled and spec.requires_network:
                    # Enabled network providers may EXIST; they must simply never
                    # be selected. available() already excludes them; this asserts
                    # that invariant rather than trusting it.
                    assert spec not in self.available(kind), (
                        f"local-only violated: {spec.key} is selectable")


def default_registry(*, local_only: bool = False) -> ProviderRegistry:
    """The providers this build ships with. All local today."""
    registry = ProviderRegistry(local_only=local_only)

    def _whisper(**kw):
        from .stt.whisper import FasterWhisperRecognizer
        return FasterWhisperRecognizer(**kw)

    def _nllb(**kw):
        from .translate.nllb import NllbTranslator
        return NllbTranslator(**kw)

    def _passthrough(**kw):
        from .translate.passthrough import PassthroughTranslator
        return PassthroughTranslator()

    def _ollama(**kw):
        from .translate.ollama import OllamaTranslator
        return OllamaTranslator(**kw)

    def _routing(**kw):
        from .translate.nllb import NllbTranslator
        from .translate.ollama import OllamaTranslator
        from .translate.router import RoutingTranslator
        return RoutingTranslator(
            fast=NllbTranslator(),
            accurate=OllamaTranslator(model=kw.get("llm_model", "qwen3-coder:latest")),
        )

    def _xtts(**kw):
        from .tts.xtts import XttsSynthesizer
        return XttsSynthesizer(**kw)

    def _fallback_tts(**kw):
        from .tts.fallback import FallbackSynthesizer
        return FallbackSynthesizer(**{k: v for k, v in kw.items()
                                      if k in ("sample_rate", "wpm")})

    registry.register(ProviderSpec(
        "faster_whisper", ProviderKind.STT, "Whisper (local)",
        requires_network=False, factory=_whisper, priority=10))
    # Routing first: it is the only configuration measured at 14/14 while
    # keeping p50 latency near the fast engine's (docs/07 §7).
    registry.register(ProviderSpec(
        "routing", ProviderKind.TRANSLATION,
        "Automatic (NMT + LLM for romanised Hinglish)",
        requires_network=False, factory=_routing, priority=5))
    registry.register(ProviderSpec(
        "nllb_600m", ProviderKind.TRANSLATION, "NLLB-200 600M (local)",
        requires_network=False, factory=_nllb, priority=10))
    # Ollama on loopback: no data leaves the machine, so this is a local provider.
    registry.register(ProviderSpec(
        "ollama_llm", ProviderKind.TRANSLATION, "Local LLM via Ollama",
        requires_network=False, factory=_ollama, priority=20))
    registry.register(ProviderSpec(
        "passthrough", ProviderKind.TRANSLATION, "Pass-through",
        requires_network=False, factory=_passthrough, priority=90))
    registry.register(ProviderSpec(
        "xtts_v2", ProviderKind.TTS, "XTTS-v2 (personal voice)",
        requires_network=False, factory=_xtts, priority=10))
    registry.register(ProviderSpec(
        "fallback_formant", ProviderKind.TTS, "Fallback voice",
        requires_network=False, factory=_fallback_tts, priority=90))
    return registry
