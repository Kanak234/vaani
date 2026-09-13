"""SQLite schema and migrations (docs/05-DATA-MODEL.md).

Two rules are enforced structurally rather than by remembering them:

1. **No audio is ever stored in this database.** There is no BLOB column anywhere.
   Voice enrollment audio lives as files on disk, referenced by path.

2. **Transcript text is absent, not deleted, when persistence is off.** The
   `utterances` table still records timing and outcome so latency statistics keep
   working, but `source_text` and `translated_text` are simply never written. A
   privacy guarantee that depends on a later cleanup job is one bug away from
   failing; this one holds even if the cleanup never runs.
"""
from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, list[str]] = {
    1: [
        # --- consent: append-only, never updated or deleted ------------------
        """
        CREATE TABLE IF NOT EXISTS voice_consents (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id       TEXT    NOT NULL UNIQUE,
            consent_type      TEXT    NOT NULL CHECK (consent_type IN ('grant','revoke')),
            subject_label     TEXT    NOT NULL,
            is_self_attested  INTEGER NOT NULL DEFAULT 1,
            consent_text_hash TEXT    NOT NULL,
            consent_text      TEXT    NOT NULL,
            app_version       TEXT    NOT NULL,
            granted_at_utc    TEXT    NOT NULL,
            supersedes_id     INTEGER NULL REFERENCES voice_consents(id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_consents_type_time "
        "ON voice_consents(consent_type, granted_at_utc DESC)",

        # --- voice profiles ---------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS voice_profiles (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id            TEXT    NOT NULL UNIQUE,
            name                   TEXT    NOT NULL UNIQUE,
            consent_id             INTEGER NOT NULL
                                   REFERENCES voice_consents(id) ON DELETE RESTRICT,
            provider_key           TEXT    NOT NULL,
            model_id               TEXT    NOT NULL,
            reference_audio_path   TEXT    NULL,
            speech_seconds         REAL    NOT NULL,
            sample_rate            INTEGER NOT NULL DEFAULT 16000,
            quality_snr_db         REAL    NULL,
            quality_clipping_pct   REAL    NULL,
            status                 TEXT    NOT NULL DEFAULT 'pending'
                                   CHECK (status IN ('pending','ready','failed','revoked')),
            created_at_utc         TEXT    NOT NULL,
            last_used_at_utc       TEXT    NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_profiles_status ON voice_profiles(status)",

        # --- sessions ---------------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid                  TEXT    NOT NULL UNIQUE,
            mode                  TEXT    NOT NULL
                                  CHECK (mode IN ('meeting','conversation','diagnostic','takeover')),
            voice_profile_id      INTEGER NULL
                                  REFERENCES voice_profiles(id) ON DELETE SET NULL,
            source_language_mode  TEXT    NOT NULL DEFAULT 'auto',
            target_language       TEXT    NOT NULL DEFAULT 'en',
            performance_mode      TEXT    NOT NULL DEFAULT 'balanced',
            input_device_id       TEXT    NULL,
            output_device_id      TEXT    NULL,
            local_only            INTEGER NOT NULL DEFAULT 0,
            persist_transcript    INTEGER NOT NULL DEFAULT 0,
            started_at_utc        TEXT    NOT NULL,
            ended_at_utc          TEXT    NULL,
            end_reason            TEXT    NULL,
            utterance_count       INTEGER NOT NULL DEFAULT 0,
            suppressed_count      INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_sessions_started ON sessions(started_at_utc DESC)",
        # Partial index: crash recovery only ever looks for unfinished sessions.
        "CREATE INDEX IF NOT EXISTS ix_sessions_unfinished "
        "ON sessions(ended_at_utc) WHERE ended_at_utc IS NULL",

        # --- utterances -------------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS utterances (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id              INTEGER NOT NULL
                                    REFERENCES sessions(id) ON DELETE CASCADE,
            seq                     INTEGER NOT NULL,
            state                   TEXT    NOT NULL DEFAULT 'capturing',
            source_text             TEXT    NULL,
            translated_text         TEXT    NULL,
            lang_distribution       TEXT    NULL,
            stt_confidence          REAL    NULL,
            translation_confidence  REAL    NULL,
            speech_ms               INTEGER NULL,
            total_latency_ms        INTEGER NULL,
            was_suppressed          INTEGER NOT NULL DEFAULT 0,
            error_code              TEXT    NULL,
            created_at_utc          TEXT    NOT NULL,
            UNIQUE (session_id, seq)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_utterances_session ON utterances(session_id, created_at_utc)",
        "CREATE INDEX IF NOT EXISTS ix_utterances_suppressed ON utterances(was_suppressed)",

        # --- per-stage timings: the latency ledger ---------------------------
        """
        CREATE TABLE IF NOT EXISTS utterance_stages (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            utterance_id  INTEGER NOT NULL
                          REFERENCES utterances(id) ON DELETE CASCADE,
            stage         TEXT    NOT NULL,
            provider_key  TEXT    NULL,
            duration_ms   REAL    NOT NULL,
            queued_ms     REAL    NULL,
            succeeded     INTEGER NOT NULL DEFAULT 1,
            error_code    TEXT    NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_stages_utterance ON utterance_stages(utterance_id, stage)",
        "CREATE INDEX IF NOT EXISTS ix_stages_percentiles ON utterance_stages(stage, duration_ms)",

        # --- why nothing was spoken ------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS suppressions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            utterance_id    INTEGER NOT NULL
                            REFERENCES utterances(id) ON DELETE CASCADE,
            reason          TEXT    NOT NULL,
            threshold_value REAL    NULL,
            actual_value    REAL    NULL,
            created_at_utc  TEXT    NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_suppressions_reason ON suppressions(reason)",

        # --- session events ---------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS session_events (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER NOT NULL
                           REFERENCES sessions(id) ON DELETE CASCADE,
            event_type     TEXT    NOT NULL,
            detail         TEXT    NULL,
            created_at_utc TEXT    NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_events_session ON session_events(session_id, created_at_utc)",

        # --- diagnostics ------------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS diagnostic_runs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            run_uuid       TEXT    NOT NULL,
            check_name     TEXT    NOT NULL,
            status         TEXT    NOT NULL CHECK (status IN ('pass','fail','skip')),
            duration_ms    REAL    NOT NULL,
            detail_json    TEXT    NULL,
            created_at_utc TEXT    NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_diag_run ON diagnostic_runs(run_uuid)",
        "CREATE INDEX IF NOT EXISTS ix_diag_check ON diagnostic_runs(check_name, created_at_utc DESC)",

        # --- settings ---------------------------------------------------------
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key            TEXT PRIMARY KEY,
            value_json     TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        )
        """,

        # --- remembered device selections ------------------------------------
        """
        CREATE TABLE IF NOT EXISTS audio_devices (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            device_key    TEXT    NOT NULL UNIQUE,
            kind          TEXT    NOT NULL,
            display_name  TEXT    NOT NULL,
            backend       TEXT    NOT NULL DEFAULT 'auto',
            channels      INTEGER NULL,
            sample_rate   INTEGER NULL,
            is_virtual    INTEGER NOT NULL DEFAULT 0,
            is_available  INTEGER NOT NULL DEFAULT 1,
            last_seen_utc TEXT    NOT NULL
        )
        """,
    ],
}


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Bring the database up to SCHEMA_VERSION. Idempotent."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version in sorted(_MIGRATIONS):
        if version <= current:
            continue
        for statement in _MIGRATIONS[version]:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {version}")
        current = version
    conn.commit()
    return current
