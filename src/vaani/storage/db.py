"""Session, utterance and diagnostics persistence.

The privacy-critical behaviour lives in `record_utterance`: when the session was
started with `persist_transcript=False`, the text columns are **never written**.
The row still exists, carrying timings and outcome, so latency statistics and
suppression counts keep working with no content on disk.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..core.errors import ErrorCode, Severity, VaaniError
from ..core.types import UtteranceResult
from .schema import SCHEMA_VERSION, apply_migrations


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class RetentionPolicy:
    """Defaults from docs/05-DATA-MODEL.md §4."""

    transcript_days: int = 7
    session_days: int = 90
    metric_days: int = 90
    diagnostic_days: int = 30


class Database:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        except sqlite3.Error as exc:
            raise VaaniError(code=ErrorCode.STORAGE_FAILURE,
                             message=f"could not open the database: {exc}",
                             severity=Severity.FATAL, cause=exc) from exc
        self._conn.row_factory = sqlite3.Row
        # WAL so the UI can read while the audio thread writes.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        # Off by default in SQLite; without it the ON DELETE rules are decorative.
        self._conn.execute("PRAGMA foreign_keys = ON")
        apply_migrations(self._conn)
        if self.path.exists():
            os.chmod(self.path, 0o600)     # may contain transcripts

    @property
    def schema_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._conn:
                yield self._conn
        except sqlite3.Error as exc:
            raise VaaniError(code=ErrorCode.STORAGE_FAILURE,
                             message=f"database write failed: {exc}",
                             severity=Severity.SESSION, cause=exc) from exc

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------- sessions

    def start_session(self, *, mode: str, performance_mode: str = "balanced",
                      source_language_mode: str = "auto", target_language: str = "en",
                      voice_profile_id: int | None = None,
                      input_device_id: str | None = None,
                      output_device_id: str | None = None,
                      local_only: bool = False,
                      persist_transcript: bool = False) -> int:
        session_uuid = uuid.uuid4().hex
        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO sessions
                   (uuid, mode, voice_profile_id, source_language_mode,
                    target_language, performance_mode, input_device_id,
                    output_device_id, local_only, persist_transcript, started_at_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (session_uuid, mode, voice_profile_id, source_language_mode,
                 target_language, performance_mode, input_device_id,
                 output_device_id, int(local_only), int(persist_transcript), _now()))
            return int(cur.lastrowid)

    def end_session(self, session_id: int, *, reason: str = "user_stop") -> None:
        with self._tx() as conn:
            conn.execute("UPDATE sessions SET ended_at_utc=?, end_reason=? WHERE id=?",
                         (_now(), reason, session_id))

    def unfinished_sessions(self) -> list[sqlite3.Row]:
        """Sessions with no end time — i.e. the app crashed (AC-02.3 sibling)."""
        return list(self._conn.execute(
            "SELECT * FROM sessions WHERE ended_at_utc IS NULL ORDER BY started_at_utc"))

    def mark_crashed_sessions(self) -> int:
        """Close out sessions left open by a crash. Run at startup."""
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE sessions SET ended_at_utc=?, end_reason='crash' "
                "WHERE ended_at_utc IS NULL", (_now(),))
            return cur.rowcount

    def get_session(self, session_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM sessions WHERE id=?",
                                  (session_id,)).fetchone()

    def recent_sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM sessions ORDER BY started_at_utc DESC LIMIT ?", (limit,)))

    # ----------------------------------------------------------- utterances

    def record_utterance(self, session_id: int, result: UtteranceResult) -> int:
        """Persist one utterance.

        THE privacy rule: transcript and translation text are written only when
        the session opted in. Otherwise the columns stay NULL — the content is
        never on disk at all, rather than written now and deleted later.
        """
        session = self.get_session(session_id)
        if session is None:
            raise VaaniError(code=ErrorCode.STORAGE_FAILURE,
                             message=f"no session {session_id}",
                             severity=Severity.SESSION)
        persist = bool(session["persist_transcript"])

        transcript, translation = result.transcript, result.translation
        source_text = transcript.text if (persist and transcript) else None
        translated_text = translation.text if (persist and translation) else None

        with self._tx() as conn:
            cur = conn.execute(
                """INSERT INTO utterances
                   (session_id, seq, state, source_text, translated_text,
                    lang_distribution, stt_confidence, translation_confidence,
                    speech_ms, total_latency_ms, was_suppressed, error_code,
                    created_at_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (session_id, result.utterance.seq,
                 "suppressed" if result.suppressed else "complete",
                 source_text, translated_text,
                 json.dumps(transcript.language_distribution) if transcript else None,
                 transcript.confidence if transcript else None,
                 translation.confidence if translation else None,
                 int(result.utterance.speech_ms),
                 int(result.total_latency_ms) if result.total_latency_ms else None,
                 int(result.suppressed), result.error_code, _now()))
            utterance_id = int(cur.lastrowid)

            for timing in result.timings:
                conn.execute(
                    """INSERT INTO utterance_stages
                       (utterance_id, stage, provider_key, duration_ms, queued_ms,
                        succeeded, error_code)
                       VALUES (?,?,?,?,?,?,?)""",
                    (utterance_id, timing.stage, timing.provider_key,
                     timing.duration_ms, timing.queued_ms,
                     int(timing.succeeded), timing.error_code))

            if result.suppressed and result.suppression_reason is not None:
                conn.execute(
                    """INSERT INTO suppressions
                       (utterance_id, reason, created_at_utc) VALUES (?,?,?)""",
                    (utterance_id, result.suppression_reason.value, _now()))

            conn.execute(
                "UPDATE sessions SET utterance_count = utterance_count + 1, "
                "suppressed_count = suppressed_count + ? WHERE id = ?",
                (int(result.suppressed), session_id))
        return utterance_id

    def utterances_for(self, session_id: int) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM utterances WHERE session_id=? ORDER BY seq", (session_id,)))

    # -------------------------------------------------------------- metrics

    def stage_percentiles(self, stage: str, *, days: int = 30) -> dict[str, float]:
        """p50/p95 for one stage. Content-free, so always safe to compute."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            """SELECT s.duration_ms FROM utterance_stages s
               JOIN utterances u ON u.id = s.utterance_id
               WHERE s.stage = ? AND s.succeeded = 1 AND u.created_at_utc >= ?
               ORDER BY s.duration_ms""", (stage, cutoff)).fetchall()
        values = [r[0] for r in rows]
        if not values:
            return {}
        def pct(p: float) -> float:
            return round(values[min(len(values)-1, int(round((len(values)-1)*p)))], 1)
        return {"count": len(values), "p50": pct(0.5), "p95": pct(0.95),
                "min": round(values[0], 1), "max": round(values[-1], 1)}

    def suppression_summary(self, *, days: int = 30) -> dict[str, int]:
        """Why the system stayed silent. The question users ask most."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            "SELECT reason, COUNT(*) FROM suppressions WHERE created_at_utc >= ? "
            "GROUP BY reason ORDER BY COUNT(*) DESC", (cutoff,)).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------- settings

    def set_setting(self, key: str, value: Any) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO app_settings (key, value_json, updated_at_utc) "
                "VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "value_json=excluded.value_json, updated_at_utc=excluded.updated_at_utc",
                (key, json.dumps(value), _now()))

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value_json FROM app_settings WHERE key=?",
                                 (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ---------------------------------------------------------- diagnostics

    def record_diagnostics(self, results: list[Any]) -> str:
        run_uuid = uuid.uuid4().hex
        with self._tx() as conn:
            for r in results:
                conn.execute(
                    """INSERT INTO diagnostic_runs
                       (run_uuid, check_name, status, duration_ms, detail_json,
                        created_at_utc) VALUES (?,?,?,?,?,?)""",
                    (run_uuid, r.name, r.status, r.duration_ms,
                     json.dumps(_redact(getattr(r, "data", {}))), _now()))
        return run_uuid

    # ------------------------------------------------------------ retention

    def apply_retention(self, policy: RetentionPolicy | None = None) -> dict[str, int]:
        """Delete expired data. Run at startup and periodically.

        Transcript TEXT is cleared before whole rows are removed, so content
        disappears on the shorter clock even though metrics are kept longer.
        """
        policy = policy or RetentionPolicy()
        now = datetime.now(UTC)
        removed: dict[str, int] = {}
        with self._tx() as conn:
            cutoff = (now - timedelta(days=policy.transcript_days)).isoformat()
            cur = conn.execute(
                "UPDATE utterances SET source_text=NULL, translated_text=NULL "
                "WHERE created_at_utc < ? AND (source_text IS NOT NULL "
                "OR translated_text IS NOT NULL)", (cutoff,))
            removed["transcripts_cleared"] = cur.rowcount

            cutoff = (now - timedelta(days=policy.metric_days)).isoformat()
            cur = conn.execute("DELETE FROM utterances WHERE created_at_utc < ?", (cutoff,))
            removed["utterances_deleted"] = cur.rowcount

            cutoff = (now - timedelta(days=policy.session_days)).isoformat()
            cur = conn.execute(
                "DELETE FROM sessions WHERE started_at_utc < ? AND ended_at_utc IS NOT NULL",
                (cutoff,))
            removed["sessions_deleted"] = cur.rowcount

            cutoff = (now - timedelta(days=policy.diagnostic_days)).isoformat()
            cur = conn.execute("DELETE FROM diagnostic_runs WHERE created_at_utc < ?",
                               (cutoff,))
            removed["diagnostics_deleted"] = cur.rowcount
        return removed

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and everything derived from it (FK CASCADE)."""
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            return cur.rowcount > 0


_SECRET_HINTS = ("key", "token", "secret", "password", "credential", "auth")


def _redact(data: dict[str, Any]) -> dict[str, Any]:
    """Diagnostics are exportable for bug reports; never leak a secret in one."""
    return {k: ("<redacted>" if any(h in k.lower() for h in _SECRET_HINTS) else v)
            for k, v in (data or {}).items()}


def _default_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "vaani" / "vaani.db"
