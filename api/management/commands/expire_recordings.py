"""Purge recordings that are past their dynamic retention window.

Each recording's retention window is `average_appointment_minutes + 2 hours`
(see `api/recording_retention.py`). The API already purges a user's own
expired recordings opportunistically whenever they list/fetch recordings, so
this command is not required for correctness — it's here so recordings can
also be swept up server-wide on a schedule (cron / ECS scheduled task /
Lambda), the same way `medications/scheduler.py` is consumed by an external
dispatcher rather than an in-process scheduler.

Usage:
    python manage.py expire_recordings
    python manage.py expire_recordings --dry-run
"""

from django.core.management.base import BaseCommand

from api.models import Recording
from api.s3 import delete_audio_object


class Command(BaseCommand):
    help = "Delete recordings (and their S3 audio) past their dynamic retention window."

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='List recordings that would be deleted without deleting them.',
        )

    def handle(self, *args, **options):
        dry_run = options.get('dry_run', False)
        expired = [r for r in Recording.objects.all() if r.is_expired]

        if not expired:
            self.stdout.write('No expired recordings found.')
            return

        for recording in expired:
            label = f"Recording #{recording.id} (user {recording.user_id}, created {recording.created_at.isoformat()})"
            if dry_run:
                self.stdout.write(f"[dry-run] Would delete: {label}")
                continue

            if recording.audio_storage == 's3' and recording.audio_object_key:
                try:
                    delete_audio_object(object_key=recording.audio_object_key)
                except Exception as exc:  # pragma: no cover - best effort
                    self.stderr.write(f"Failed to delete S3 object for {label}: {exc}")

            recording.delete()
            self.stdout.write(f"Deleted: {label}")

        if not dry_run:
            self.stdout.write(self.style.SUCCESS(f"Purged {len(expired)} expired recording(s)."))
