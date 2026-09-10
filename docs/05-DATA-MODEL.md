# Data Model — Vaani

**Phase:** 5 · **Engine:** SQLite (WAL) at `~/.local/share/vaani/vaani.db`

---

## 1. Storage principles

1. **Raw audio is not a database concern.** No audio blob is ever stored in SQLite.
   The only audio persisted at all is voice-enrollment reference audio, which lives
   as encrypted files on disk and is referenced by path.
2. **Default to not storing.** Transcripts are opt-in and off by default.
   Untranslated audio buffers are freed as soon as an utterance completes.
3. **Every row that holds personal data has a stated retention policy** and a
   deletion path that is actually executed, not merely documented.
4. **Consent is append-only.** A consent record is never updated or deleted — a
   revocation is a *new* row. The history of what was agreed to must survive.
5. **Secrets are never in this database.** API keys live in the OS keyring; the
   database stores only a reference to the keyring entry.

## 2. Entity overview

```
voice_consents ──< voice_profiles ──< sessions ──< utterances ──< utterance_stages
                                          │            └──< suppressions
                                          ├──< session_events
audio_devices (cache)   providers ──< provider_settings   app_settings   diagnostic_runs
```

There is no `users` table. v1 is explicitly single-user (BRD C6); inventing a users
table now would add joins and foreign keys that carry no information. A
`profile_owner_label` field on `voice_profiles` records who the voice belongs to,
which is the only identity fact the product actually needs.

## 3. Tables

### 3.1 `voice_consents` — append-only consent ledger
| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | no | PK AUTOINCREMENT | |
| `consent_type` | TEXT | no | | CHECK in (`grant`,`revoke`) |
| `subject_label` | TEXT | no | | Whose voice, as typed by the user |
| `is_self_attested` | INTEGER | no | 1 | User asserts it is their own voice |
| `consent_text_hash` | TEXT | no | | SHA-256 of the exact text shown |
| `consent_text` | TEXT | no | | Full text, for audit |
| `app_version` | TEXT | no | | |
| `granted_at_utc` | TEXT | no | | ISO-8601 UTC |
| `supersedes_id` | INTEGER | yes | NULL | FK → `voice_consents(id)` |

Indexes: `(consent_type, granted_at_utc DESC)`.
**Retention:** permanent. Never deleted, even when the profile is.
**Constraint:** a `revoke` row must reference the `grant` it revokes via `supersedes_id`.

### 3.2 `voice_profiles`
| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | no | PK | |
| `name` | TEXT | no | | UNIQUE |
| `consent_id` | INTEGER | no | | FK → `voice_consents(id)` ON DELETE RESTRICT |
| `provider_key` | TEXT | no | | Which TTS backend produced it |
| `model_id` | TEXT | no | | Exact model + version |
| `embedding_path` | TEXT | yes | NULL | Encrypted file on disk |
| `reference_audio_dir` | TEXT | yes | NULL | Encrypted dir |
| `speech_seconds` | REAL | no | | Detected speech, not wall-clock |
| `sample_rate` | INTEGER | no | 16000 | |
| `quality_snr_db` | REAL | yes | NULL | Measured at enrollment |
| `quality_clipping_pct` | REAL | yes | NULL | |
| `status` | TEXT | no | `'pending'` | CHECK in (`pending`,`ready`,`failed`,`revoked`) |
| `created_at_utc` | TEXT | no | | |
| `last_used_at_utc` | TEXT | yes | NULL | |

Indexes: UNIQUE `(name)`, `(status)`.
**FK rule:** `ON DELETE RESTRICT` on `consent_id` — a profile cannot exist without the
consent record that authorised it.
**Retention:** until user deletion. Deletion must shred `embedding_path` and
`reference_audio_dir` on disk *before* removing the row, and verify absence (AC-09.6).

### 3.3 `sessions`
| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | no | PK | |
| `uuid` | TEXT | no | | UNIQUE, used in logs |
| `mode` | TEXT | no | | CHECK in (`meeting`,`conversation`,`diagnostic`) |
| `voice_profile_id` | INTEGER | yes | NULL | FK, ON DELETE SET NULL |
| `source_language_mode` | TEXT | no | `'auto'` | `auto|hi|en|hinglish` |
| `target_language` | TEXT | no | `'en'` | |
| `performance_mode` | TEXT | no | `'balanced'` | `low_latency|balanced|quality` |
| `input_device_id` | TEXT | yes | NULL | |
| `output_device_id` | TEXT | yes | NULL | |
| `local_only` | INTEGER | no | 0 | Privacy mode at session start |
| `persist_transcript` | INTEGER | no | 0 | **Off by default** |
| `started_at_utc` | TEXT | no | | |
| `ended_at_utc` | TEXT | yes | NULL | NULL ⇒ crashed or running |
| `end_reason` | TEXT | yes | NULL | `user_stop|error|crash|device_lost` |
| `utterance_count` | INTEGER | no | 0 | Denormalised counter |
| `suppressed_count` | INTEGER | no | 0 | |

Indexes: UNIQUE `(uuid)`, `(started_at_utc DESC)`, `(ended_at_utc)` for crash recovery.
**Session isolation:** context never crosses `session.id` (AC-06.5).
**Retention:** metadata 90 days default, configurable.

### 3.4 `utterances`
| Column | Type | Null | Default | Notes |
|---|---|---|---|---|
| `id` | INTEGER | no | PK | |
| `session_id` | INTEGER | no | | FK, ON DELETE CASCADE |
| `seq` | INTEGER | no | | Ordinal within session |
| `state` | TEXT | no | `'capturing'` | see state machine |
| `source_text` | TEXT | yes | NULL | **NULL unless `persist_transcript`** |
| `translated_text` | TEXT | yes | NULL | Same rule |
| `lang_distribution` | TEXT | yes | NULL | JSON `{"hi":0.6,"en":0.4}` |
| `stt_confidence` | REAL | yes | NULL | |
| `translation_confidence` | REAL | yes | NULL | |
| `speech_ms` | INTEGER | yes | NULL | Detected speech duration |
| `total_latency_ms` | INTEGER | yes | NULL | Speech-end → audio-start |
| `was_suppressed` | INTEGER | no | 0 | |
| `error_code` | TEXT | yes | NULL | |
| `created_at_utc` | TEXT | no | | |

Indexes: UNIQUE `(session_id, seq)`, `(session_id, created_at_utc)`, `(was_suppressed)`.
**Critical rule:** when `sessions.persist_transcript = 0`, `source_text` and
`translated_text` are **never written** — the row still exists for latency
statistics, but carries no content. This makes the privacy guarantee structural
rather than a matter of remembering to delete later.
**Retention:** text purged after `retention_days` (default 7); metrics kept 90 days.

### 3.5 `utterance_stages` — the latency ledger (NFR-1, AC-13.1)
| Column | Type | Null | Notes |
|---|---|---|---|
| `id` | INTEGER | no | PK |
| `utterance_id` | INTEGER | no | FK, ON DELETE CASCADE |
| `stage` | TEXT | no | `capture|vad|stt|context|translate|normalize|tts|postprocess|buffer|output` |
| `provider_key` | TEXT | yes | Which implementation ran |
| `duration_ms` | REAL | no | **Measured** |
| `queued_ms` | REAL | yes | Time spent waiting before the stage started |
| `succeeded` | INTEGER | no | |
| `error_code` | TEXT | yes | |

Indexes: `(utterance_id, stage)`, `(stage, duration_ms)` for percentile queries.
**Retention:** 90 days. Contains no content — safe to keep.

### 3.6 `suppressions` — why nothing was spoken (FR-15)
`id` PK · `utterance_id` FK CASCADE · `reason` CHECK in
(`low_stt_confidence`,`low_translation_confidence`,`unsupported_language`,
`stage_failure`,`user_manual`,`emergency_stop`,`empty_result`) · `threshold_value`
REAL null · `actual_value` REAL null · `created_at_utc`.

This is a first-class table, not a log line. "Why did it stay silent?" is a question
the user will ask often, and it must be answerable precisely.

### 3.7 `session_events`
`id` PK · `session_id` FK CASCADE · `event_type` (`start`,`pause`,`resume`,`mute`,
`unmute`,`device_change`,`provider_failover`,`emergency_stop`,`error`,`stop`) ·
`detail` TEXT (JSON, no PII) · `created_at_utc`. Index `(session_id, created_at_utc)`.

### 3.8 `audio_devices` — cache of last-seen devices
`id` PK · `device_key` UNIQUE · `kind` (`input`,`output`,`virtual_source`) ·
`display_name` · `backend` (default `pipewire`) · `channels` · `sample_rate` ·
`is_virtual` · `is_available` · `last_seen_utc`.
**Retention:** transient; rebuilt on enumeration. Persisted only to remember the
user's selection across restarts.

### 3.9 `providers` / `provider_settings`
`providers`: `id` PK · `provider_key` UNIQUE · `kind` CHECK in (`stt`,`translation`,
`tts`,`vad`) · `display_name` · `is_local` INTEGER · `requires_network` INTEGER ·
`keyring_ref` TEXT null (**a reference, never a key**) · `is_enabled` ·
`priority` INTEGER (failover order).

`provider_settings`: `id` PK · `provider_id` FK CASCADE · `key` · `value_json` ·
UNIQUE `(provider_id, key)`.

**Constraint:** a provider with `requires_network = 1` must never be selected while
the session has `local_only = 1`; enforced in code *and* asserted by a test.

### 3.10 `app_settings`
`key` TEXT PK · `value_json` TEXT · `updated_at_utc`. Single-row-per-key KV. Typed
and validated in the config layer, not by the database.

### 3.11 `diagnostic_runs`
`id` PK · `run_uuid` · `check_name` · `status` (`pass`,`fail`,`skip`) ·
`duration_ms` REAL · `detail_json` (secrets redacted) · `created_at_utc`.
Index `(run_uuid)`, `(check_name, created_at_utc DESC)`. **Retention:** 30 days.

## 4. Retention summary

| Data | Default | Configurable | Deletion trigger |
|---|---|---|---|
| Consent records | Permanent | No | Never |
| Voice embedding + reference audio | Until deleted | No | User action / revoke |
| Transcript text | **Not stored** | Yes (opt-in, 7 d) | Retention job / session delete |
| Utterance metrics | 90 days | Yes | Retention job |
| Session metadata | 90 days | Yes | Retention job |
| Diagnostics | 30 days | Yes | Retention job |
| Raw audio buffers | **Never persisted** | No | Freed at utterance end |
| API keys | OS keyring only | No | User action |

A retention job runs at startup and every 6 h; deletions are logged to
`session_events` without content.
