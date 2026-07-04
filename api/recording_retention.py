"""Recording retention / expiry helpers.

Recordings (and their S3 audio copies) are not stored forever. Each user has
an ``average_appointment_minutes`` setting — collected during onboarding and
editable from Settings — that estimates how long their visits typically run.
A recording's retention window is dynamic rather than a fixed timer:

    retention_minutes = average_appointment_minutes + RETENTION_BUFFER_MINUTES

...counted from the moment the recording was created. Once that window has
elapsed the recording is considered expired and is purged (S3 object + the
database row) the next time the owning user's recordings are listed or
fetched — see ``views._purge_expired_recordings``.

This module is intentionally framework-agnostic (plain dataclasses/datetimes,
no Django ORM imports) so it can be unit tested in isolation, mirroring the
``medications/scheduler.py`` pure-function pattern used elsewhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

# Extra buffer added on top of the user's average appointment length so a
# recording remains available for a reasonable grace period after the visit.
RETENTION_BUFFER_MINUTES = 120  # 2 hours

# Fallback used when a user has no PersonalizationProfile yet (should be rare).
DEFAULT_AVERAGE_APPOINTMENT_MINUTES = 30


def retention_window_minutes(average_appointment_minutes: Optional[int]) -> int:
    """Total minutes a recording stays available for, counted from creation time."""
    avg = average_appointment_minutes if average_appointment_minutes and average_appointment_minutes > 0 else DEFAULT_AVERAGE_APPOINTMENT_MINUTES
    return int(avg) + RETENTION_BUFFER_MINUTES


def compute_expires_at(created_at: datetime, average_appointment_minutes: Optional[int]) -> datetime:
    """The absolute timestamp at which a recording created at `created_at` expires."""
    return created_at + timedelta(minutes=retention_window_minutes(average_appointment_minutes))


def is_expired(
    created_at: datetime,
    average_appointment_minutes: Optional[int],
    *,
    now: Optional[datetime] = None,
) -> bool:
    """True once `now` has passed the recording's dynamic retention window."""
    now_value = now or datetime.now(timezone.utc)
    return now_value >= compute_expires_at(created_at, average_appointment_minutes)
