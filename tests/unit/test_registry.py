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
