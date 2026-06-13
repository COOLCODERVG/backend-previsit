"""Medication-reminder scheduler.

`compute_due_reminders(now)` returns the set of `MedicationReminder` rows whose
`time_of_day` × `days_of_week` × `timezone` falls inside the current minute
window. The dispatcher (Lambda or ECS Scheduled Task — see
`infra/terraform/lambda_reminder_dispatch.tf`) runs once a minute and consumes
this output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore

from .models import Medication, MedicationReminder


@dataclass(frozen=True)
class DueReminder:
    reminder_id: int
    medication_id: int
    user_id: int
    time_of_day: str
    timezone: str
    medication_name: str  # for the dispatcher's audit log only — never sent in the push body


_WEEKDAY_TOKENS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _matches_now(reminder: MedicationReminder, now_utc: datetime, *, window_seconds: int = 60) -> bool:
    """True iff the reminder is due within `[now_utc, now_utc + window_seconds)`."""
    tz_name = (reminder.timezone or "UTC").strip() or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")

    local_now = now_utc.astimezone(tz)

    # Day-of-week filter — empty list means "every day".
    days = [str(d).lower().strip() for d in (reminder.days_of_week or []) if d]
    if days:
        weekday_token = _WEEKDAY_TOKENS[local_now.weekday()]
        if weekday_token not in days:
            return False

    try:
        hh, mm = (reminder.time_of_day or "00:00").split(":", 1)
        hour = int(hh)
        minute = int(mm)
    except Exception:
        return False

    target_local = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    target_utc = target_local.astimezone(timezone.utc)
    delta = (target_utc - now_utc).total_seconds()
    return -1 <= delta < window_seconds


def compute_due_reminders(
    now: Optional[datetime] = None,
    *,
    window_seconds: int = 60,
) -> List[DueReminder]:
    """Return reminders due in the next `window_seconds` (default 60s)."""
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

    qs = (
        MedicationReminder.objects.filter(is_active=True)
        .select_related("medication", "user")
        .filter(medication__status__in=["active", "modified"])
    )

    out: List[DueReminder] = []
    for r in qs:
        if not _matches_now(r, now_utc, window_seconds=window_seconds):
            continue
        out.append(
            DueReminder(
                reminder_id=r.id,
                medication_id=r.medication_id,
                user_id=r.user_id,
                time_of_day=r.time_of_day,
                timezone=(r.timezone or "UTC"),
                medication_name=getattr(r.medication, "name", "") or "",
            )
        )
    return out
