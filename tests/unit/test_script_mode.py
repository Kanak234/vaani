"""Tests for session/script_mode.py (Phase 5, offline script mode).

Two things are being defended here, and they are the two that make offline
different from live:

  1. NOTHING IS LOST. Not to a slow chunk, not to low confidence, not to one
     bad chunk, and not to a crash fifty minutes in.
  2. CHUNKS DO NOT CUT THROUGH WORDS. A fixed-window splitter corrupts one word
     per window and nothing downstream can tell.

Providers are stubbed: an hour of real Whisper is not something a suite can
afford, and none of the logic worth testing depends on the models.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from vaani.core.types import Transcript, Translation
from vaani.session.script_mode import (
    ScriptConfig, ScriptJob, ScriptResult, ScriptSegment,
    find_chunk_boundaries, rms_envelope,
)

SR = 16000


class StubRecognizer:
    def __init__(self, text="मैं कल आऊंगा", confidence=0.9, error=None):
        self.text = text
        self.confidence = confidence
        self.error = error
        self.calls = 0

    def transcribe(self, audio, sample_rate, *, language_hint=None,
                   context_prompt=None):
        self.calls += 1
        if self.error:
            raise self.error
        return Transcript(text=self.text, language_distribution={"hi": 1.0},
                          confidence=self.confidence)


class StubTranslator:
    def __init__(self, text="I will come tomorrow.", error=None):
        self.text = text
        self.error = error

    def translate(self, text, *, source_language, target_language, context=None):
        if self.error:
            raise self.error
        return Translation(text=self.text, source_language=source_language,
                           target_language=target_language, confidence=0.9)


def speech(seconds: float, *, level: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(SR * seconds)) * level).astype(np.float32)


def job(audio, **kw):
    return ScriptJob(audio=audio, sample_rate=SR,
                     recognizer=kw.pop("recognizer", StubRecognizer()),
                     translator=kw.pop("translator", StubTranslator()), **kw)


# ------------------------------------------------------------- chunking

def test_short_audio_is_one_chunk():
    assert find_chunk_boundaries(speech(5), SR) == [(0, SR * 5)]


def test_empty_audio_produces_no_chunks():
    assert find_chunk_boundaries(np.zeros(0, dtype=np.float32), SR) == []


def test_long_audio_is_split():
    bounds = find_chunk_boundaries(speech(90), SR, ScriptConfig(chunk_s=25))
    assert len(bounds) >= 3


def test_chunks_are_contiguous_and_cover_everything():
    """THE property that matters: no audio may be skipped between chunks, or
    the script quietly loses a sentence."""
    audio = speech(90)
    bounds = find_chunk_boundaries(audio, SR, ScriptConfig(chunk_s=25))
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(audio)
    for (_, end), (next_start, _) in zip(bounds, bounds[1:], strict=False):
        assert end == next_start


def test_the_cut_lands_in_the_quiet_part():
    """A fixed splitter would cut at exactly 25 s, mid-word. This one should
    find the silence planted just after it."""
    loud = speech(24, level=0.3)
    quiet = np.zeros(int(SR * 2.0), dtype=np.float32)
    more = speech(24, level=0.3)
    audio = np.concatenate([loud, quiet, more])

    bounds = find_chunk_boundaries(audio, SR,
                                   ScriptConfig(chunk_s=25, search_s=5))
    first_cut_s = bounds[0][1] / SR
    # The planted silence runs 24.0-26.0 s.
    assert 24.0 <= first_cut_s <= 26.0


def test_continuous_audio_degrades_to_a_fixed_cut():
    """With no pause to find, falling back to the target length is the honest
    behaviour -- there is no good cut point."""
    bounds = find_chunk_boundaries(speech(60, level=0.3), SR,
                                   ScriptConfig(chunk_s=25, search_s=5))
    assert len(bounds) >= 2
    first = bounds[0][1] / SR
    assert 20.0 <= first <= 30.0        # within the search window of target


def test_chunks_always_advance():
    """A splitter that returns a zero-length chunk loops forever."""
    for seconds in (26, 40, 51, 99):
        bounds = find_chunk_boundaries(speech(seconds), SR,
                                       ScriptConfig(chunk_s=25))
        for start, end in bounds:
            assert end > start


@pytest.mark.parametrize("chunk_s", [5.0, 10.0, 25.0])
def test_chunks_stay_near_the_target_length(chunk_s):
    """REGRESSION, and the reason `end > start` was too weak a test.

    The search for a quiet cut point could reach all the way back to
    start + 20 ms, so every chunk advanced by one window and 13 s of speech
    with a few pauses produced 147 chunks instead of 3. Each one is a separate
    Whisper call, so the run took 79 s for 13 s of audio.
    """
    seconds = 120.0
    # Speech with regular pauses - the shape that triggered the bug, because
    # a pause near the start of the window is the quietest point.
    blocks = []
    while len(blocks) * 3 < seconds:
        blocks.append(speech(2.0, level=0.3))
        blocks.append(np.zeros(int(SR * 1.0), dtype=np.float32))
    audio = np.concatenate(blocks)

    bounds = find_chunk_boundaries(audio, SR, ScriptConfig(chunk_s=chunk_s))
    expected = len(audio) / SR / chunk_s
    # Generous, but orders of magnitude tighter than the bug allowed.
    assert len(bounds) <= expected * 2 + 1
    for start, end in bounds[:-1]:      # the last chunk is whatever remains
        assert (end - start) / SR >= chunk_s * 0.5


def test_envelope_tracks_loudness():
    quiet = rms_envelope(np.zeros(SR, dtype=np.float32), SR)
    loud = rms_envelope(speech(1, level=0.5), SR)
    assert float(np.mean(quiet)) < float(np.mean(loud))


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        ScriptConfig(chunk_s=0)
    with pytest.raises(ValueError):
        ScriptConfig(search_s=-1)


# -------------------------------------------------------------- the run

def test_every_chunk_is_transcribed_and_translated():
    recognizer = StubRecognizer()
    result = job(speech(60), recognizer=recognizer).run()
    assert recognizer.calls == len(result.segments)
    assert all(s.english_text for s in result.segments)


def test_the_script_reads_as_paragraphs():
    result = job(speech(60)).run()
    text = result.script_text()
    assert "I will come tomorrow." in text
    assert "\n\n" in text               # one paragraph per chunk


def test_the_source_is_kept_for_checking_the_translation():
    result = job(speech(30)).run()
    assert "मैं कल आऊंगा" in result.source_text()


def test_silence_produces_no_paragraph_but_is_not_an_error():
    result = job(speech(30), recognizer=StubRecognizer(text="")).run()
    assert result.failed_chunks == 0
    assert result.script_text() == ""


def test_low_confidence_text_is_KEPT_and_marked_not_dropped():
    """The live pipeline suppresses uncertain text. Offline that is wrong: the
    user is going to read and edit this anyway, and a marked guess is more
    useful to them than a silent deletion."""
    result = job(speech(30), recognizer=StubRecognizer(confidence=0.3)).run()
    assert all(s.uncertain for s in result.segments)
    assert "[?]" in result.script_text()
    assert "I will come tomorrow." in result.script_text()


def test_confident_text_is_not_marked():
    result = job(speech(30), recognizer=StubRecognizer(confidence=0.95)).run()
    assert "[?]" not in result.script_text()


def test_a_failed_chunk_does_not_lose_the_rest():
    """One bad chunk must not cost the user the other fifty-nine minutes."""
    class FlakyRecognizer(StubRecognizer):
        def transcribe(self, audio, sample_rate, **kw):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("whisper choked")
            return Transcript(text="ठीक है", language_distribution={"hi": 1.0},
                              confidence=0.9)

    result = job(speech(90), recognizer=FlakyRecognizer()).run()
    assert result.failed_chunks == 1
    assert len(result.segments) >= 3
    # The hole is announced, not hidden.
    assert "failed" in result.script_text()


def test_a_translation_failure_keeps_the_source():
    """Only the translation failed. Keeping the Hindi lets the user fix that
    paragraph by hand instead of re-recording it."""
    result = job(speech(30),
                 translator=StubTranslator(error=RuntimeError("ollama down"))).run()
    assert result.segments[0].source_text == "मैं कल आऊंगा"
    assert result.segments[0].error is not None


def test_progress_is_reported():
    seen = []
    job(speech(90)).run(progress=lambda i, total, seg: seen.append((i, total)))
    assert seen
    assert seen[-1][0] == seen[-1][1]        # finished all of them


def test_a_broken_progress_display_does_not_stop_the_job():
    def boom(*_a):
        raise ValueError("terminal exploded")

    result = job(speech(60)).run(progress=boom)
    assert len(result.segments) >= 2


# ------------------------------------------------------------ checkpoints

def test_progress_is_checkpointed_after_every_chunk(tmp_path):
    """After EVERY chunk, not every N: a crash does not wait for a round
    number."""
    path = tmp_path / "job.json"
    seen_counts = []

    def watch(_i, _total, _seg):
        seen_counts.append(len(json.loads(path.read_text())["segments"]))

    job(speech(90), checkpoint_path=path).run(progress=watch)
    assert seen_counts == list(range(1, len(seen_counts) + 1))


def test_a_crashed_run_resumes_where_it_stopped(tmp_path):
    """THE reason this module checkpoints: the target desktop crashes under
    long sustained load, and losing fifty minutes means the user never trusts
    it enough to retry."""
    path = tmp_path / "job.json"
    audio = speech(90)

    class CrashingRecognizer(StubRecognizer):
        def transcribe(self, a, sr, **kw):
            self.calls += 1
            if self.calls == 3:
                raise KeyboardInterrupt("power cut")
            return Transcript(text="ठीक है", language_distribution={"hi": 1.0},
                              confidence=0.9)

    with pytest.raises(KeyboardInterrupt):
        job(audio, recognizer=CrashingRecognizer(), checkpoint_path=path).run()

    completed = len(json.loads(path.read_text())["segments"])
    assert completed == 2

    fresh = StubRecognizer()
    result = job(audio, recognizer=fresh, checkpoint_path=path).run()
    assert result.resumed_from == 2
    # Only the REMAINING chunks were re-transcribed.
    assert fresh.calls == len(result.segments) - 2


def test_a_finished_job_reruns_instantly(tmp_path):
    path = tmp_path / "job.json"
    audio = speech(60)
    job(audio, checkpoint_path=path).run()

    second = StubRecognizer()
    result = job(audio, recognizer=second, checkpoint_path=path).run()
    assert second.calls == 0            # nothing redone
    assert result.script_text()


def test_a_corrupt_checkpoint_starts_over_rather_than_failing(tmp_path):
    """A truncated write from the very crash this guards against must not be
    the reason the user cannot run the job at all."""
    path = tmp_path / "job.json"
    path.write_text('{"segments": [{"broken":')      # truncated mid-write
    result = job(speech(30), checkpoint_path=path).run()
    assert result.resumed_from == 0
    assert result.segments


def test_a_checkpoint_from_a_longer_recording_is_not_spliced_in(tmp_path):
    """Resuming a DIFFERENT recording's checkpoint would splice two scripts
    together and the user would never know."""
    path = tmp_path / "job.json"
    job(speech(120), checkpoint_path=path).run()
    long_count = len(json.loads(path.read_text())["segments"])
    assert long_count > 1

    result = job(speech(20), checkpoint_path=path).run()
    assert result.resumed_from == 0


def test_no_checkpoint_path_means_no_file_and_no_crash(tmp_path):
    result = job(speech(30)).run()
    assert result.segments
    assert not list(tmp_path.iterdir())


def test_an_unwritable_checkpoint_does_not_stop_the_run(tmp_path):
    """Losing a checkpoint is a setback, not a failure."""
    unwritable = tmp_path / "nope" / "deep"
    result = job(speech(30), checkpoint_path=unwritable / "x.json").run()
    assert result.segments


# ------------------------------------------------------------- reporting

def test_result_reports_audio_duration():
    result = job(speech(45)).run()
    assert 44.0 <= result.audio_seconds <= 46.0


def test_an_empty_result_renders_empty():
    assert ScriptResult().script_text() == ""


def test_segments_carry_their_timestamps():
    result = job(speech(60)).run()
    assert result.segments[0].start_s == 0.0
    assert result.segments[-1].end_s > 0
    for a, b in zip(result.segments, result.segments[1:], strict=False):
        assert b.start_s >= a.start_s


def test_segment_round_trips_through_json():
    """Checkpoint integrity depends on this."""
    segment = ScriptSegment(0, 0.0, 1.0, "स्रोत", "source", 0.8, True, None)
    assert ScriptSegment(**json.loads(json.dumps(segment.__dict__))) == segment
