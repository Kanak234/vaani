"""Voice-cloning consent (FR-9, AC-09.1, AC-09.2).

Cloning a voice is the one capability in this product that could do real harm if
misused, so consent is not a checkbox stored as a boolean. It is an append-only
ledger: a revocation is a NEW record, never an update or a delete, so the history
of what was agreed to always survives.

The record captures the exact text shown to the user and its hash, so that if the
wording changes in a later version it is still possible to say precisely what was
agreed to, and when.
"""
from __future__ import annotations

import enum
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..core.errors import ErrorCode, Severity, VaaniError

#: Shown verbatim before any enrollment audio is captured. Changing this string
#: changes its hash, which is intentional -- old consents remain attributable to
#: the exact wording they were given under.
CONSENT_TEXT = """\
VOICE CLONING CONSENT

You are about to create a synthetic copy of a voice. Please read this in full.

WHAT THIS DOES
  Vaani will record your speech and build a voice profile from it. That profile
  can then generate NEW speech in that voice -- sentences the speaker never said.

WHOSE VOICE
  You may only enroll a voice you are authorised to use. In practice that means
  YOUR OWN voice. Enrolling someone else's voice without their informed consent
  is likely illegal in your jurisdiction and is not a supported use of this tool.

WHERE THE DATA GOES
  The recordings and the resulting voice profile are stored ONLY on this computer,
  encrypted at rest. They are not uploaded, not shared, and not used to train any
  model outside this machine.

HOW TO DELETE IT
  Voice Profile -> Delete removes the profile, the embedding and every recording,
  and verifies they are gone from disk. You may do this at any time, without
  explanation. Revoking this consent deletes them automatically.

WHAT WE KEEP AFTER DELETION
  Only this consent record itself, so there is a durable log of what was agreed
  to and when. It contains no audio.

By confirming, you state that the voice you are about to enroll is your own, or
that you have the explicit permission of the person whose voice it is.
"""

#: The user must type this exactly. A checkbox alone is too easy to click past for
#: a decision this consequential.
CONFIRMATION_PHRASE = "I CONSENT"


class ConsentType(enum.Enum):
    GRANT = "grant"
    REVOKE = "revoke"


@dataclass(slots=True)
class ConsentRecord:
    consent_type: str
    subject_label: str
    is_self_attested: bool
    consent_text_hash: str
    consent_text: str
    app_version: str
    granted_at_utc: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    supersedes_id: str | None = None

    @property
    def is_grant(self) -> bool:
        return self.consent_type == ConsentType.GRANT.value


def consent_text_hash(text: str = CONSENT_TEXT) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ConsentLedger:
    """Append-only JSONL ledger.

    JSONL rather than SQLite specifically because append-only is the property that
    matters here, and it is structurally guaranteed by the format rather than by
    remembering not to write an UPDATE.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def grant(self, *, subject_label: str, confirmation: str,
              app_version: str, is_self_attested: bool = True) -> ConsentRecord:
        """Record a grant. Raises unless the confirmation phrase matches exactly."""
        if confirmation.strip().upper() != CONFIRMATION_PHRASE:
            raise VaaniError(
                code=ErrorCode.VOICE_CONSENT_MISSING,
                message=f"consent not given: type {CONFIRMATION_PHRASE!r} to confirm",
                severity=Severity.FATAL,
            )
        if not subject_label.strip():
            raise VaaniError(
                code=ErrorCode.VOICE_CONSENT_MISSING,
                message="consent requires naming whose voice is being enrolled",
                severity=Severity.FATAL,
            )
        record = ConsentRecord(
            consent_type=ConsentType.GRANT.value,
            subject_label=subject_label.strip(),
            is_self_attested=is_self_attested,
            consent_text_hash=consent_text_hash(),
            consent_text=CONSENT_TEXT,
            app_version=app_version,
            granted_at_utc=datetime.now(UTC).isoformat(),
        )
        self._append(record)
        return record

    def revoke(self, grant_id: str, *, app_version: str) -> ConsentRecord:
        """Record a revocation as a NEW row referencing the grant it cancels."""
        grant = self.get(grant_id)
        if grant is None or not grant.is_grant:
            raise VaaniError(
                code=ErrorCode.VOICE_CONSENT_MISSING,
                message=f"no consent grant with id {grant_id}",
                severity=Severity.SESSION,
            )
        record = ConsentRecord(
            consent_type=ConsentType.REVOKE.value,
            subject_label=grant.subject_label,
            is_self_attested=grant.is_self_attested,
            consent_text_hash=grant.consent_text_hash,
            consent_text=grant.consent_text,
            app_version=app_version,
            granted_at_utc=datetime.now(UTC).isoformat(),
            supersedes_id=grant_id,
        )
        self._append(record)
        return record

    def _append(self, record: ConsentRecord) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())   # consent must survive a crash
        except OSError as exc:
            raise VaaniError(code=ErrorCode.STORAGE_FAILURE,
                             message=f"could not write the consent record: {exc}",
                             severity=Severity.FATAL, cause=exc) from exc

    def all_records(self) -> list[ConsentRecord]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(ConsentRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue    # a corrupt line must not hide valid ones
        return out

    def get(self, record_id: str) -> ConsentRecord | None:
        return next((r for r in self.all_records() if r.id == record_id), None)

    def active_grant(self) -> ConsentRecord | None:
        """The most recent grant that has not been revoked."""
        records = self.all_records()
        revoked = {r.supersedes_id for r in records
                   if r.consent_type == ConsentType.REVOKE.value}
        grants = [r for r in records if r.is_grant and r.id not in revoked]
        return grants[-1] if grants else None

    def has_active_consent(self) -> bool:
        return self.active_grant() is not None

    def require_active_consent(self) -> ConsentRecord:
        """Gate for every cloning operation. Enrollment must call this FIRST."""
        grant = self.active_grant()
        if grant is None:
            raise VaaniError(
                code=ErrorCode.VOICE_CONSENT_MISSING,
                message="voice cloning requires consent; none is on record",
                severity=Severity.FATAL,
            )
        return grant


def _default_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "vaani" / "consent.jsonl"
