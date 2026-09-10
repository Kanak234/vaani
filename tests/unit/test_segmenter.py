"""Segmenter behaviour tests.

These encode the decisions from the design docs as executable checks: false starts
must not open an utterance, onsets must not be lost, and long monologues must be
force-split rather than growing unbounded.
"""
from __future__ import annotations

import numpy as np
import pytest

from vaani.audio.segmenter import (
    SegmenterConfig,
    SegmenterState,
    UtteranceSegmenter,
)
from vaani.core.types import PerformanceMode

SR = 16000
FRAME_MS = 20
FRAME_N = SR * FRAME_MS // 1000


def frame(level: float = 0.1) -> np.ndarray:
    return np.full(FRAME_N, level, dtype=np.float32)


def drive(seg: UtteranceSegmenter, pattern: list[tuple[float, int]]):
    """Feed (speech_prob, n_frames) pairs; collect emitted segments."""
    out = []
    for prob, count in pattern:
        for _ in range(count):
            s = seg.push(frame(), prob)
            if s is not None:
                out.append(s)
    return out


def make(**kw) -> UtteranceSegmenter:
    cfg = SegmenterConfig(**kw) if kw else SegmenterConfig()
    return UtteranceSegmenter(config=cfg, sample_rate=SR, frame_ms=FRAME_MS)


def test_single_utterance_is_emitted_after_hangover():
    seg = make(hangover_ms=200, start_frames=2, min_utterance_ms=100)
    segments = drive(seg, [(0.0, 5), (0.9, 30), (0.0, 12)])
    assert len(segments) == 1
    assert segments[0].speech_ms >= 100
    assert not segments[0].truncated
    assert seg.state is SegmenterState.SILENCE


def test_no_segment_before_hangover_elapses():
    seg = make(hangover_ms=400, start_frames=2)
    # 5 frames of silence = 100 ms, well short of the 400 ms hangover.
    assert drive(seg, [(0.9, 30), (0.0, 5)]) == []
    assert seg.is_capturing


def test_false_start_does_not_open_an_utterance():
    """A single loud frame (cough, click) must not start a segment."""
    seg = make(start_frames=3)
    assert drive(seg, [(0.9, 1), (0.0, 20)]) == []
    assert seg.state is SegmenterState.SILENCE


def test_too_short_utterance_is_discarded():
    seg = make(hangover_ms=100, start_frames=2, min_utterance_ms=500)
    # ~120 ms of speech, under the 500 ms floor.
    assert drive(seg, [(0.9, 6), (0.0, 10)]) == []


def test_pre_roll_preserves_speech_onset():
    """The utterance must contain audio from before VAD fired (aspirated stops)."""
    seg = make(pre_roll_ms=200, start_frames=2, hangover_ms=200, min_utterance_ms=100)
    segments = drive(seg, [(0.0, 10), (0.9, 20), (0.0, 12)])
    assert len(segments) == 1
    # 20 speech frames = 400 ms; with pre-roll the segment must exceed that.
    assert segments[0].audio.size > int(SR * 0.4)


def test_long_monologue_is_force_split():
    seg = make(max_utterance_ms=600, start_frames=1, hangover_ms=1000)
    segments = drive(seg, [(0.9, 120)])
    assert len(segments) >= 2
    assert all(s.truncated for s in segments)


def test_brief_pause_does_not_split_a_sentence():
    """A 200 ms mid-sentence pause with a 500 ms hangover stays one utterance."""
    seg = make(hangover_ms=500, start_frames=2, min_utterance_ms=100)
    segments = drive(seg, [(0.9, 20), (0.0, 10), (0.9, 20), (0.0, 30)])
    assert len(segments) == 1


def test_flush_closes_in_progress_utterance():
    seg = make(hangover_ms=1000, start_frames=2, min_utterance_ms=100)
    drive(seg, [(0.9, 30)])
    assert seg.is_capturing
    s = seg.flush()
    assert s is not None and s.truncated
    assert not seg.is_capturing


def test_reset_discards_audio_without_emitting():
    seg = make(start_frames=2, min_utterance_ms=100)
    drive(seg, [(0.9, 30)])
    seg.reset()
    assert seg.state is SegmenterState.SILENCE
    assert seg.flush() is None


@pytest.mark.parametrize("mode,expected_max_hangover", [
    (PerformanceMode.LOW_LATENCY, 400),
    (PerformanceMode.BALANCED, 600),
    (PerformanceMode.QUALITY, 900),
])
def test_mode_presets_order_hangover_correctly(mode, expected_max_hangover):
    cfg = SegmenterConfig.for_mode(mode)
    assert cfg.hangover_ms <= expected_max_hangover


def test_hangover_increases_monotonically_with_quality():
    """Latency knob must move in the documented direction across modes."""
    lo = SegmenterConfig.for_mode(PerformanceMode.LOW_LATENCY).hangover_ms
    bal = SegmenterConfig.for_mode(PerformanceMode.BALANCED).hangover_ms
    hi = SegmenterConfig.for_mode(PerformanceMode.QUALITY).hangover_ms
    assert lo < bal < hi
