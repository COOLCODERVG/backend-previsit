"""Dispatch due medication reminders.

Run every minute (e.g. EventBridge Scheduler → Lambda, or an ECS Scheduled
Task). The flow is:

    1. compute_due_reminders(now) — time-of-day × days-of-week × timezone
       windowing.
    2. Load each due user's active push devices.
    3. Send a single, **PHI-free** push notification ("Time for your
       medication" + a deeplink to /medications) via the Expo Push API.
    4. Deactivate any tokens that come back as DeviceNotRegistered.

Pass ``--dry-run`` to inspect which reminders match without sending pushes.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List

from django.core.management.base import BaseCommand

from medications.scheduler import compute_due_reminders

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Compute due medication reminders and dispatch PHI-free pushes via Expo."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Don't send pushes; print due reminders only.")
        parser.add_argument("--window-seconds", type=int, default=60)

    def handle(self, *args, **options):
        from notifications.models import PushDevice
        from notifications.expo_push import send_medication_reminders

        due = compute_due_reminders(window_seconds=int(options["window_seconds"]))
        self.stdout.write(f"due_count={len(due)}")
        if not due:
            return

        # Group reminders by user so each user only receives one push per cadence.
        by_user: Dict[int, List] = {}
        for d in due:
            by_user.setdefault(d.user_id, []).append(d)

        if options["dry_run"]:
            self.stdout.write(json.dumps({"dry_run": True, "by_user": {str(k): len(v) for k, v in by_user.items()}}))
            return

        sent_total = 0
        invalid_total = 0
        for user_id, items in by_user.items():
            tokens = []
            for d in PushDevice.objects.filter(user_id=user_id, is_active=True):
                prefs = d.notification_prefs if isinstance(d.notification_prefs, dict) else {}
                if prefs.get("medication_reminders", True):
                    tokens.append(d.expo_push_token)
            if not tokens:
                continue
            result = send_medication_reminders(tokens)
            sent_total += result.sent
            if result.invalid_tokens:
                invalid_total += len(result.invalid_tokens)
                PushDevice.objects.filter(expo_push_token__in=result.invalid_tokens).update(is_active=False)
            self.stdout.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        "reminders": len(items),
                        "tokens": len(tokens),
                        "sent": result.sent,
                        "invalid": len(result.invalid_tokens),
                    }
                )
            )

        self.stdout.write(f"dispatched sent={sent_total} invalid_pruned={invalid_total}")
