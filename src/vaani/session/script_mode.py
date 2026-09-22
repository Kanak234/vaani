"""Offline script mode: a recording in, an English script out.

WHAT THIS IS FOR
----------------
Presentation prep, not live meetings. The user speaks Hindi/Hinglish into a
file -- or records one here -- and gets back an English script they can read,
edit and rehearse, optionally with an English audio track in their own cloned
voice.

WHY IT IS NOT JUST "run the meeting pipeline over a file"
---------------------------------------------------------
The live pipeline is built around a constraint this one does not have: it must
never fall behind. It drops utterances under load, suppresses anything it is
unsure about, and throws away audio the moment it is processed -- all correct
when a conversation is moving on without you, all wrong here. Offline, nobody
is waiting. Being slow is free; losing a sentence is not. So this module
inverts every one of those trade-offs:

  * nothing is dropped, ever -- a slow chunk just takes longer;
  * low-confidence text is KEPT and marked, because the user is going to read
    and edit the script anyway, and a marked guess is more useful to them than
    a silent deletion;
  * every chunk is checkpointed to disk as it completes.

WHY CHECKPOINTS ARE NOT OPTIONAL
--------------------------------
The desktop this targets (i9-14900KF) is known to crash under long sustained
load, and transcribing an hour of speech is exactly that. A run that loses
fifty minutes of work to a crash at minute fifty-one is worse than useless,
because the user will not trust it enough to try again. So progress is written
after every chunk and :meth:`ScriptJob.run` resumes from wherever it stopped,
including across a reboot.

WHY THE CHUNK BOUNDARIES FOLLOW SILENCE
---------------------------------------
Fixed 30-second windows are simpler but cut through the middle of words, and
Whisper cannot recover the half-word on either side -- so a naive chunker
quietly corrupts one word every thirty seconds. :func:`find_chunk_boundaries`
therefore targets a chunk length but cuts at the quietest point near it.

WHERE THE DATA FLOWS NEXT
-------------------------
:meth:`ScriptJob.run` returns a :class:`ScriptResult`. ``cli.py``'s `script`
command writes the .txt, and optionally synthesises the .wav through the same
XTTS voice the meeting uses.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: Whisper is trained on 30 s windows, so chunks near that length are the ones
#: it handles best. Shorter also means more frequent checkpoints, which is the
#: other thing this module cares about.
DEFAULT_CHUNK_S = 25.0

#: How far either side of the target the splitter may look for a quiet point.
#: Wide enough to find a real pause in normal speech, narrow enough that chunks
#: stay near the target length.
DEFAULT_SEARCH_S = 5.0


@dataclass(frozen=True, slots=True)
class ScriptConfig:
    chunk_s: float = DEFAULT_CHUNK_S
    search_s: float = DEFAULT_SEARCH_S
    #: Text below this STT confidence is kept but flagged. NOT dropped -- see
    #: the module docstring on why offline inverts the live trade-off.
    low_confidence_below: float = 0.6
    #: Marker wrapped around uncertain text in the written script, so the user
    #: can find every guess with a search rather than re-listening to the whole
    #: recording.
    uncertain_marker: str = "[?]"

    def __post_init__(self) -> None:
        if self.chunk_s <= 0:
            raise ValueError("chunk_s must be positive")
        if self.search_s < 0:
            raise ValueError("search_s must not be negative")


@dataclass
class ScriptSegment:
    """One transcribed and translated chunk."""
    index: int
    start_s: float
    end_s: float
    source_text: str
    english_text: str
    confidence: float = 1.0
    uncertain: bool = False
    error: str | None = None


@dataclass
class ScriptResult:
    segments: list[ScriptSegment] = field(default_factory=list)
    audio_seconds: float = 0.0
    elapsed_s: float = 0.0
    resumed_from: int = 0
    #: Chunks that failed outright. Kept as a count rather than silently
    #: skipped, because a script with holes must announce them.
    failed_chunks: int = 0

    def script_text(self, *, marker: str = "[?]") -> str:
        """The English script, as the user will read it.

        One paragraph per chunk rather than one run-on block: the chunks follow
        the speaker's own pauses, so they land close to where the speaker
        actually took a breath, which is where a reader wants a break too.
        """
        parts = []
        for segment in self.segments:
            if segment.error is not None:
                parts.append(f"{marker} [chunk {segment.index} failed: "
                             f"{segment.error}]")
                continue
            text = segment.english_text.strip()
            if not text:
                continue
            parts.append(f"{marker} {text}" if segment.uncertain else text)
        return "\n\n".join(parts)

    def source_text(self) -> str:
        """The original Hindi/Hinglish, for checking the translation against."""
        return "\n\n".join(s.source_text.strip() for s in self.segments
                           if s.source_text.strip())


def rms_envelope(audio: np.ndarray, sample_rate: int,
                 window_ms: float = 20.0) -> np.ndarray:
    """Per-window loudness. The basis for finding a pause to cut at."""
    window = max(1, int(sample_rate * window_ms / 1000.0))
    usable = (len(audio) // window) * window
    if usable == 0:
        return np.zeros(1, dtype=np.float32)
    frames = audio[:usable].reshape(-1, window).astype(np.float32)
    return np.sqrt(np.mean(frames * frames, axis=1))


def find_chunk_boundaries(audio: np.ndarray, sample_rate: int,
                          config: ScriptConfig | None = None) -> list[tuple[int, int]]:
    """Split audio into (start_sample, end_sample) chunks that cut at pauses.

    WHY NOT FIXED WINDOWS: a fixed 30 s cut lands mid-word roughly every time,
    and neither side can recover it -- the chunker would quietly corrupt one
    word every thirty seconds and nothing downstream could tell.

    The algorithm: walk forward by the target chunk length, then look within
    +/- search_s for the quietest 20 ms window and cut there. On audio with no
    pause at all (continuous speech, music) it degrades to the fixed cut, which
    is the honest fallback -- there is no good cut point to find.
    """
    config = config or ScriptConfig()
    total = len(audio)
    target = int(sample_rate * config.chunk_s)
    search = int(sample_rate * config.search_s)
    if total <= target:
        return [(0, total)] if total else []

    window = max(1, int(sample_rate * 0.02))
    envelope = rms_envelope(audio, sample_rate)

    boundaries: list[tuple[int, int]] = []
    start = 0
    while start < total:
        ideal = start + target
        if ideal >= total - window:
            boundaries.append((start, total))
            break

        # The search must never reach back near `start`, or the "quietest
        # point" can be found a few milliseconds in and the chunk makes almost
        # no progress. Observed: 13 s of speech with several pauses produced
        # 147 chunks instead of 3, because every cut landed at start+20 ms.
        # Clamping the backward reach to 40% of the target keeps a chunk at
        # least ~0.6x its nominal length.
        earliest = start + max(window, int(target * 0.6))
        low = max(earliest, ideal - search)
        high = min(total - window, ideal + search)
        if high <= low:
            cut = ideal
        else:
            # Index into the envelope, whose resolution is one window.
            lo_i, hi_i = low // window, high // window
            hi_i = min(hi_i, len(envelope) - 1)
            if hi_i <= lo_i:
                cut = ideal
            else:
                quietest = int(np.argmin(envelope[lo_i:hi_i])) + lo_i
                cut = quietest * window
        cut = max(start + window, min(cut, total))
        boundaries.append((start, cut))
        start = cut
    return boundaries


class ScriptJob:
    """A resumable transcribe-and-translate pass over one recording.

    Providers are injected rather than constructed here so the whole job can be
    driven by stubs in a test -- an hour of real Whisper is not something a
    test suite can afford, and the logic worth testing is the chunking,
    checkpointing and resumption, none of which depend on the models.
    """

    def __init__(self, *, audio: np.ndarray, sample_rate: int,
                 recognizer, translator,
                 config: ScriptConfig | None = None,
                 checkpoint_path: Path | None = None) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.recognizer = recognizer
        self.translator = translator
        self.config = config or ScriptConfig()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    # ----------------------------------------------------------- checkpoints

    def _load_checkpoint(self) -> list[ScriptSegment]:
        """Completed segments from a previous run, or [].

        Any problem reading it -- missing, truncated by a crash mid-write,
        written by an older version -- means "start over" rather than an error.
        A corrupt checkpoint must never be the reason the user cannot run the
        job at all; the worst case is redoing work, which is what they would
        have done anyway.
        """
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return []
        try:
            payload = json.loads(self.checkpoint_path.read_text())
            return [ScriptSegment(**entry) for entry in payload.get("segments", [])]
        except Exception:
            return []

    def _save_checkpoint(self, segments: list[ScriptSegment]) -> None:
        """Write progress. Atomic, because the crash this guards against can
        land in the middle of the write itself."""
        if self.checkpoint_path is None:
            return
        payload = {
            "version": 1,
            "sample_rate": self.sample_rate,
            "audio_samples": len(self.audio),
            "segments": [asdict(s) for s in segments],
        }
        temporary = self.checkpoint_path.with_suffix(
            self.checkpoint_path.suffix + ".tmp")
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, indent=2))
            temporary.replace(self.checkpoint_path)     # atomic on POSIX
        except Exception:
            # Losing a checkpoint is a setback, not a failure. The run
            # continues; the user just has less to resume from.
            pass

    # ------------------------------------------------------------------ run

    def run(self, *, progress=None) -> ScriptResult:
        """Process every chunk, resuming from a checkpoint when one exists.

        `progress(index, total, segment)` is called after each chunk so a CLI
        can show movement. An hour of audio is many minutes of work, and the
        Phase 2 lesson applies here too: a silent terminal is indistinguishable
        from a hang.
        """
        started = time.monotonic()
        boundaries = find_chunk_boundaries(self.audio, self.sample_rate,
                                           self.config)
        done = self._load_checkpoint()
        # Trust only a prefix of the checkpoint: resuming into the middle of a
        # DIFFERENT recording's checkpoint would splice two scripts together.
        if len(done) > len(boundaries):
            done = []
        segments = list(done)
        resumed_from = len(segments)

        for index in range(resumed_from, len(boundaries)):
            start, end = boundaries[index]
            segment = self._process_chunk(index, start, end)
            segments.append(segment)
            # After EVERY chunk, not every N: the whole point is surviving a
            # crash, and a crash does not wait for a round number.
            self._save_checkpoint(segments)
            if progress is not None:
                try:
                    progress(index + 1, len(boundaries), segment)
                except Exception:
                    pass        # a broken progress display must not stop the job

        return ScriptResult(
            segments=segments,
            audio_seconds=len(self.audio) / self.sample_rate,
            elapsed_s=time.monotonic() - started,
            resumed_from=resumed_from,
            failed_chunks=sum(1 for s in segments if s.error is not None),
        )

    def _process_chunk(self, index: int, start: int, end: int) -> ScriptSegment:
        """One chunk through STT and translation.

        A failure here is recorded on the segment rather than raised: one bad
        chunk must not cost the user the other fifty-nine minutes, and the
        script says plainly where the hole is.
        """
        start_s, end_s = start / self.sample_rate, end / self.sample_rate
        audio = self.audio[start:end]
        try:
            transcript = self.recognizer.transcribe(
                audio, self.sample_rate, language_hint=None, context_prompt=None)
        except Exception as exc:
            return ScriptSegment(index, start_s, end_s, "", "",
                                 error=f"transcription failed: {exc}")

        source = (transcript.text or "").strip()
        confidence = getattr(transcript, "confidence", 1.0)
        if not source:
            # Silence, or a chunk Whisper found nothing in. Not an error.
            return ScriptSegment(index, start_s, end_s, "", "",
                                 confidence=confidence)

        try:
            # The DETECTED language, not "auto". Passing "auto" made every
            # chunk fail with LANGID_UNSUPPORTED (auto->en) - caught only by
            # running the command end to end, because the stub translator in
            # the unit tests accepts any string. The live pipeline passes
            # transcript.dominant_language here too, so this now matches it.
            translation = self.translator.translate(
                source, source_language=transcript.dominant_language,
                target_language="en")
            english = (translation.text or "").strip()
        except Exception as exc:
            # The source survived; only the translation failed. Keep the Hindi
            # so the user can translate that paragraph by hand rather than
            # re-recording it.
            return ScriptSegment(index, start_s, end_s, source, "",
                                 confidence=confidence,
                                 error=f"translation failed: {exc}")

        return ScriptSegment(
            index, start_s, end_s, source, english, confidence=confidence,
            uncertain=confidence < self.config.low_confidence_below)
