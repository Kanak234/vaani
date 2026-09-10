"""Database layer, centred on the privacy rule.

The rule that matters: when a session did not opt into transcript persistence,
the text is NEVER WRITTEN — not written-then-deleted. These tests assert the
absence of content on disk, not merely that a cleanup ran.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from vaani.core.types import (
    StageTiming,
    SuppressionReason,
    Transcript,
    Translation,
    Utterance,
    UtteranceResult,
)
from vaani.storage.db import Database, RetentionPolicy
from vaani.storage.schema import SCHEMA_VERSION

SR = 16000


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


def make_result(seq=1, *, suppressed=False, text="मैं कल आऊंगा",
                translated="I will come tomorrow."):
    u = Utterance(session_id="s", seq=seq,
                  audio=np.zeros(SR, dtype=np.float32), sample_rate=SR,
                  speech_end_time=time.monotonic(), speech_ms=1000.0)
    r = UtteranceResult(utterance=u)
    r.transcript = Transcript(text=text, language_distribution={"hi": 1.0},
                              confidence=0.9)
    if not suppressed:
        r.translation = Translation(text=translated, source_language="hi",
                                    target_language="en", confidence=0.9)
    r.timings = [StageTiming("stt", 578.0, "faster_whisper"),
                 StageTiming("translate", 87.0, "nllb_600m")]
    r.total_latency_ms = 1103.0
    if suppressed:
        r.suppressed = True
        r.suppression_reason = SuppressionReason.LOW_STT_CONFIDENCE
    return r


# --- schema -----------------------------------------------------------------

def test_schema_is_created_and_versioned(db):
    assert db.schema_version == SCHEMA_VERSION


def test_migrations_are_idempotent(tmp_path):
    p = tmp_path / "x.db"
    Database(p).close()
    d = Database(p)
    assert d.schema_version == SCHEMA_VERSION
    d.close()


def test_foreign_keys_are_enforced(db):
    """Off by default in SQLite; without the pragma ON DELETE is decorative."""
    assert db._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_no_blob_columns_anywhere(db):
    """Audio must never be storable in this database."""
    tables = [r[0] for r in db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tables:
        for col in db._conn.execute(f"PRAGMA table_info({t})"):
            assert "BLOB" not in str(col[2]).upper(), f"{t}.{col[1]} is a BLOB"


def test_database_file_is_not_world_readable(db):
    import stat
    assert not (db.path.stat().st_mode & stat.S_IROTH)


# --- THE privacy rule -------------------------------------------------------

def test_transcript_text_is_never_written_when_persistence_is_off(db):
    sid = db.start_session(mode="meeting", persist_transcript=False)
    db.record_utterance(sid, make_result())
    row = db.utterances_for(sid)[0]
    assert row["source_text"] is None
    assert row["translated_text"] is None
    # But the metrics survive.
    assert row["total_latency_ms"] == 1103
    assert row["stt_confidence"] == pytest.approx(0.9)


def test_transcript_text_is_stored_when_opted_in(db):
    sid = db.start_session(mode="meeting", persist_transcript=True)
    db.record_utterance(sid, make_result())
    row = db.utterances_for(sid)[0]
    assert row["source_text"] == "मैं कल आऊंगा"
    assert row["translated_text"] == "I will come tomorrow."


def test_content_is_absent_from_the_raw_file_not_just_the_query(db):
    """Assert on the bytes on disk, not on what a SELECT returns."""
    sid = db.start_session(mode="meeting", persist_transcript=False)
    db.record_utterance(sid, make_result(text="SECRETPHRASE12345"))
    db._conn.commit()
    blob = db.path.read_bytes()
    wal = db.path.with_name(db.path.name + "-wal")
    if wal.exists():
        blob += wal.read_bytes()
    assert b"SECRETPHRASE12345" not in blob


def test_persistence_default_is_off(db):
    sid = db.start_session(mode="meeting")
    assert db.get_session(sid)["persist_transcript"] == 0


# --- sessions and utterances ------------------------------------------------

def test_session_counters_update(db):
    sid = db.start_session(mode="meeting")
    db.record_utterance(sid, make_result(seq=1))
    db.record_utterance(sid, make_result(seq=2, suppressed=True))
    s = db.get_session(sid)
    assert s["utterance_count"] == 2
    assert s["suppressed_count"] == 1


def test_stage_timings_are_recorded(db):
    sid = db.start_session(mode="meeting")
    uid = db.record_utterance(sid, make_result())
    rows = db._conn.execute(
        "SELECT stage, duration_ms, provider_key FROM utterance_stages "
        "WHERE utterance_id=? ORDER BY stage", (uid,)).fetchall()
    assert {r["stage"] for r in rows} == {"stt", "translate"}
    assert dict(rows[0])["provider_key"] in ("faster_whisper", "nllb_600m")


def test_suppression_reason_is_queryable(db):
    sid = db.start_session(mode="meeting")
    db.record_utterance(sid, make_result(suppressed=True))
    assert db.suppression_summary()["low_stt_confidence"] == 1


def test_duplicate_seq_in_one_session_is_rejected(db):
    import sqlite3
    from vaani.core.errors import VaaniError
    sid = db.start_session(mode="meeting")
    db.record_utterance(sid, make_result(seq=1))
    with pytest.raises((VaaniError, sqlite3.IntegrityError)):
        db.record_utterance(sid, make_result(seq=1))


def test_deleting_a_session_cascades(db):
    sid = db.start_session(mode="meeting")
    db.record_utterance(sid, make_result(suppressed=True))
    assert db.delete_session(sid)
    assert db.utterances_for(sid) == []
    assert db._conn.execute("SELECT COUNT(*) FROM utterance_stages").fetchone()[0] == 0
    assert db._conn.execute("SELECT COUNT(*) FROM suppressions").fetchone()[0] == 0


# --- crash recovery ---------------------------------------------------------

def test_unfinished_sessions_are_detected(db):
    sid = db.start_session(mode="meeting")
    assert [r["id"] for r in db.unfinished_sessions()] == [sid]
    db.end_session(sid)
    assert db.unfinished_sessions() == []


def test_crashed_sessions_are_closed_at_startup(db):
    db.start_session(mode="meeting")
    db.start_session(mode="meeting")
    assert db.mark_crashed_sessions() == 2
    assert db.unfinished_sessions() == []
    assert db.recent_sessions()[0]["end_reason"] == "crash"


# --- metrics ----------------------------------------------------------------

def test_stage_percentiles(db):
    sid = db.start_session(mode="meeting")
    for i in range(1, 11):
        db.record_utterance(sid, make_result(seq=i))
    p = db.stage_percentiles("stt")
    assert p["count"] == 10 and p["p50"] == pytest.approx(578.0)


def test_percentiles_empty_when_no_data(db):
    assert db.stage_percentiles("stt") == {}


# --- settings ---------------------------------------------------------------

def test_settings_roundtrip_and_upsert(db):
    db.set_setting("performance_mode", "quality")
    assert db.get_setting("performance_mode") == "quality"
    db.set_setting("performance_mode", "balanced")
    assert db.get_setting("performance_mode") == "balanced"
    assert db.get_setting("missing", "fallback") == "fallback"


def test_settings_store_structured_values(db):
    db.set_setting("gate", {"stt": 0.55, "translation": 0.5})
    assert db.get_setting("gate")["stt"] == 0.55


# --- retention --------------------------------------------------------------

def test_retention_clears_old_transcripts_but_keeps_metrics(db):
    sid = db.start_session(mode="meeting", persist_transcript=True)
    uid = db.record_utterance(sid, make_result())
    old = "2020-01-01T00:00:00+00:00"
    db._conn.execute("UPDATE utterances SET created_at_utc=? WHERE id=?", (old, uid))
    db._conn.commit()
    removed = db.apply_retention(RetentionPolicy(transcript_days=7, metric_days=99999))
    assert removed["transcripts_cleared"] == 1
    row = db.utterances_for(sid)[0]
    assert row["source_text"] is None
    assert row["total_latency_ms"] == 1103        # metric survives


def test_retention_deletes_expired_utterances(db):
    sid = db.start_session(mode="meeting")
    uid = db.record_utterance(sid, make_result())
    db._conn.execute("UPDATE utterances SET created_at_utc=? WHERE id=?",
                     ("2020-01-01T00:00:00+00:00", uid))
    db._conn.commit()
    assert db.apply_retention(RetentionPolicy(metric_days=30))["utterances_deleted"] == 1


def test_retention_is_safe_on_an_empty_database(db):
    assert db.apply_retention()["utterances_deleted"] == 0


# --- diagnostics ------------------------------------------------------------

def test_diagnostics_redact_secrets(db):
    class R:
        def __init__(s):
            s.name, s.status, s.duration_ms = "Providers", "pass", 1.0
            s.data = {"api_key": "sk-real-secret", "device": "cuda"}
    db.record_diagnostics([R()])
    row = db._conn.execute("SELECT detail_json FROM diagnostic_runs").fetchone()
    assert "sk-real-secret" not in row[0]
    assert "cuda" in row[0]
