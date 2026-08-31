"""The decision log.

Every recommendation the system makes, every red flag that fired, and every time a
nurse overrode it. This is not telemetry — it is the record that makes the system
answerable after something goes wrong, and the training signal that lets it improve.

Two properties it needs that ordinary logging does not have:

**Append-only and tamper-evident.** Each record carries the hash of the record before
it, so the log forms a chain. Editing or deleting an entry after the fact breaks every
hash downstream, which `verify` detects. Nobody can quietly revise what the system
recommended before an adverse event. This does not prevent tampering — nothing local
can — but it makes it visible, which is what an investigation needs.

**It records the input, not just the output.** A recommendation without the vitals it
was made from cannot be reviewed: the question after an incident is never only "what
did it say" but "what did it know, and was that reasonable given what it knew".

Overrides are first-class. A nurse disagreeing with the system is the system working
as designed, so an override is recorded as an ordinary event rather than an error, and
the recorded pair (what we said, what they did) is the label for the next model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


@dataclass(frozen=True)
class AuditRecord:
    event: str
    patient_id: str
    timestamp: str
    payload: dict[str, Any]
    previous_hash: str
    record_hash: str = field(default="")

    def compute_hash(self) -> str:
        body = json.dumps(
            {
                "event": self.event,
                "patient_id": self.patient_id,
                "timestamp": self.timestamp,
                "payload": self.payload,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only, hash-chained JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = json.loads(line)["record_hash"]
        return last

    def append(self, event: str, patient_id: str, payload: dict[str, Any]) -> AuditRecord:
        previous = self._last_hash()
        record = AuditRecord(
            event=event,
            patient_id=patient_id,
            timestamp=datetime.now(UTC).isoformat(),
            payload=payload,
            previous_hash=previous,
        )
        record = AuditRecord(**{**asdict(record), "record_hash": record.compute_hash()})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), default=str) + "\n")
        return record

    def records(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [AuditRecord(**json.loads(line)) for line in handle if line.strip()]

    def verify(self) -> tuple[bool, str | None]:
        """Walk the chain. Returns (intact, description of the first break)."""
        previous = GENESIS
        for position, record in enumerate(self.records()):
            if record.previous_hash != previous:
                return False, f"record {position} does not follow the one before it"
            recomputed = AuditRecord(
                event=record.event,
                patient_id=record.patient_id,
                timestamp=record.timestamp,
                payload=record.payload,
                previous_hash=record.previous_hash,
            ).compute_hash()
            if recomputed != record.record_hash:
                return False, f"record {position} has been modified since it was written"
            previous = record.record_hash
        return True, None

    def overrides(self) -> list[AuditRecord]:
        """Every time a nurse disagreed — the next model's training data."""
        return [r for r in self.records() if r.event == "override"]
