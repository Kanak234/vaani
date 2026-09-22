"""Provider registry: the local-only guarantee (AC-17.2).

These tests exist so the privacy claim is enforced by code rather than asserted in
a document.
"""
from __future__ import annotations

import pytest

from vaani.ai.base import ProviderKind
from vaani.ai.registry import ProviderRegistry, ProviderSpec, default_registry
from vaani.core.errors import ErrorCode, VaaniError


class FakeProvider:
    def __init__(self, **kw):
        self.kw = kw


def spec(key, *, network, kind=ProviderKind.TRANSLATION, priority=100):
    return ProviderSpec(key, kind, key, requires_network=network,
                        factory=lambda **kw: FakeProvider(**kw), priority=priority)


@pytest.fixture
def mixed():
    def build(local_only):
        r = ProviderRegistry(local_only=local_only)
        r.register(spec("cloud_llm", network=True, priority=10))
        r.register(spec("local_nmt", network=False, priority=20))
        return r
    return build


def test_cloud_provider_preferred_when_allowed(mixed):
    r = mixed(False)
    assert [s.key for s in r.available(ProviderKind.TRANSLATION)] == \
        ["cloud_llm", "local_nmt"]


def test_local_only_hides_network_providers(mixed):
    r = mixed(True)
    assert [s.key for s in r.available(ProviderKind.TRANSLATION)] == ["local_nmt"]


def test_local_only_auto_selection_picks_a_local_provider(mixed):
    """Even though the cloud provider has higher priority."""
    r = mixed(True)
    r.get(ProviderKind.TRANSLATION)     # must not raise
    assert r.available(ProviderKind.TRANSLATION)[0].key == "local_nmt"


def test_explicitly_requesting_a_cloud_provider_is_an_error_not_a_downgrade(mixed):
    """The user should learn their two settings contradict each other."""
    r = mixed(True)
    with pytest.raises(VaaniError) as e:
        r.get(ProviderKind.TRANSLATION, "cloud_llm")
    assert e.value.code is ErrorCode.LOCAL_ONLY_VIOLATION
    assert not e.value.retryable


def test_excluded_is_reported_so_the_ui_can_explain(mixed):
    r = mixed(True)
    assert [s.key for s in r.excluded(ProviderKind.TRANSLATION)] == ["cloud_llm"]
    assert mixed(False).excluded(ProviderKind.TRANSLATION) == []


def test_no_usable_provider_mentions_the_hidden_count():
    r = ProviderRegistry(local_only=True)
    r.register(spec("cloud_only", network=True))
    with pytest.raises(VaaniError) as e:
        r.get(ProviderKind.TRANSLATION)
    assert e.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    assert e.value.detail["hidden_by_local_only"] == 1


def test_unknown_provider_key_raises(mixed):
    with pytest.raises(VaaniError) as e:
        mixed(False).get(ProviderKind.TRANSLATION, "nope")
    assert e.value.code is ErrorCode.PROVIDER_UNAVAILABLE


def test_invariant_holds_under_local_only(mixed):
    mixed(True).assert_local_only_holds()


def test_disabled_providers_are_never_offered():
    r = ProviderRegistry()
    s = spec("off", network=False)
    s.enabled = False
    r.register(s)
    assert r.available(ProviderKind.TRANSLATION) == []


def test_shipped_registry_is_entirely_local():
    """Today every provider runs on-device; local-only must change nothing."""
    normal = default_registry(local_only=False)
    strict = default_registry(local_only=True)
    for kind in (ProviderKind.STT, ProviderKind.TRANSLATION, ProviderKind.TTS):
        assert [s.key for s in normal.available(kind)] == \
               [s.key for s in strict.available(kind)]
        assert strict.excluded(kind) == []


def test_shipped_registry_prefers_the_real_engines():
    r = default_registry()
    # Routing is the default because it is the only measured 14/14 configuration
    # while keeping p50 latency near the fast engine's (docs/07 §7).
    assert r.available(ProviderKind.TRANSLATION)[0].key == "routing"
    assert r.available(ProviderKind.TTS)[0].key == "xtts_v2"


def test_every_shipped_translation_provider_is_local():
    """Ollama runs on loopback, so even the LLM path keeps data on the machine."""
    r = default_registry()
    assert all(not s.requires_network
               for s in r.available(ProviderKind.TRANSLATION))


# ---------------------------------------------- start-up fallback (F-22)
#
# get_with_fallback() shipped with a docstring claiming "Used by: cli.py
# meeting/run start-up" and no caller and no tests. These cover the behaviour
# the meeting start-up path now depends on.

class WarmupProvider:
    """A provider that can fail at construction OR at warmup - the distinction
    that matters here, because construction proves nothing."""

    def __init__(self, *, fail_construct=False, fail_warmup=False, **kw):
        if fail_construct:
            raise ImportError("torch is not installed")
        self._fail_warmup = fail_warmup
        self.warmed = False

    def warmup(self):
        if self._fail_warmup:
            raise RuntimeError("could not load the model")
        self.warmed = True


def warm_spec(key, *, priority, **flags):
    return ProviderSpec(key, ProviderKind.TRANSLATION, key,
                        requires_network=False,
                        factory=lambda **kw: WarmupProvider(**flags, **kw),
                        priority=priority)


def test_fallback_returns_the_highest_priority_provider_that_works():
    r = ProviderRegistry(local_only=True)
    r.register(warm_spec("first", priority=10))
    r.register(warm_spec("second", priority=20))
    provider, failures = r.get_with_fallback(ProviderKind.TRANSLATION)
    assert isinstance(provider, WarmupProvider)
    assert failures == {}


def test_fallback_skips_a_provider_that_cannot_be_constructed():
    r = ProviderRegistry(local_only=True)
    r.register(warm_spec("broken", priority=10, fail_construct=True))
    r.register(warm_spec("working", priority=20))
    provider, failures = r.get_with_fallback(ProviderKind.TRANSLATION)
    assert provider.warmed is True
    # The caller must be able to say what it fell back FROM; a silent
    # downgrade looks like a bug.
    assert "broken" in failures
    assert "ImportError" in failures["broken"]


def test_a_provider_must_actually_warm_up_to_count():
    """Construction proves nothing: NllbTranslator constructs happily without
    torch and only fails when warmed. Discovering that mid-meeting is exactly
    the failure this prevents."""
    r = ProviderRegistry(local_only=True)
    r.register(warm_spec("constructs_but_dies", priority=10, fail_warmup=True))
    r.register(warm_spec("working", priority=20))
    provider, failures = r.get_with_fallback(ProviderKind.TRANSLATION)
    assert provider.warmed is True
    assert "constructs_but_dies" in failures


def test_warmup_can_be_skipped_for_speed():
    r = ProviderRegistry(local_only=True)
    r.register(warm_spec("lazy", priority=10, fail_warmup=True))
    provider, failures = r.get_with_fallback(ProviderKind.TRANSLATION, warmup=False)
    assert provider.warmed is False     # never warmed, so never failed
    assert failures == {}


def test_every_provider_failing_raises_with_all_the_reasons():
    r = ProviderRegistry(local_only=True)
    r.register(warm_spec("a", priority=10, fail_construct=True))
    r.register(warm_spec("b", priority=20, fail_warmup=True))
    with pytest.raises(VaaniError) as excinfo:
        r.get_with_fallback(ProviderKind.TRANSLATION)
    assert excinfo.value.code is ErrorCode.PROVIDER_UNAVAILABLE
    message = str(excinfo.value)
    assert "a" in message and "b" in message


def test_no_providers_at_all_names_local_only_as_the_cause():
    """Otherwise "no usable provider" is baffling when the user set
    local_only and every candidate was a cloud one."""
    r = ProviderRegistry(local_only=True)
    r.register(spec("cloud_only", network=True, priority=10))
    with pytest.raises(VaaniError) as excinfo:
        r.get_with_fallback(ProviderKind.TRANSLATION)
    assert "local-only" in str(excinfo.value)


def test_fallback_never_hands_out_a_network_provider_in_local_only_mode():
    """The privacy guarantee must hold on the fallback path too, not only in
    get()."""
    r = ProviderRegistry(local_only=True)
    r.register(spec("cloud", network=True, priority=10))
    r.register(warm_spec("local", priority=20))
    provider, _ = r.get_with_fallback(ProviderKind.TRANSLATION)
    assert isinstance(provider, WarmupProvider)     # the local one
