from __future__ import annotations

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    """
    Converts ExportedPdf.user_id / ExportedPdf.appointment_id from plain BigIntegerField
    columns into real ForeignKeys, now that the `documents` app lives in the single unified
    application database alongside `api` (no more cross-database FK restriction).

    State-only field change for the same reason as vectors/migrations/0007: the underlying
    columns keep the exact same name (`user_id`, `appointment_id`) and storage type (bigint)
    that a ForeignKey to `api.User`/`api.Appointment` requires, so nothing needs to actually
    ALTER at the database level. This avoids `makemigrations` needing a one-off default value
    for a new NOT-NULL column (which would otherwise prompt interactively).
    """

    dependencies = [
        ("documents", "0002_rename_exported_pd_user_id_4acdfd_idx_exported_pd_user_id_2bc34d_idx"),
        ("api", "0010_personalizationprofile_profile_basics"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                migrations.RemoveField(
                    model_name="exportedpdf",
                    name="user_id",
                ),
                migrations.AddField(
                    model_name="exportedpdf",
                    name="user",
                    field=models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="exported_pdfs",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                migrations.RemoveField(
                    model_name="exportedpdf",
                    name="appointment_id",
                ),
                migrations.AddField(
                    model_name="exportedpdf",
                    name="appointment",
                    field=models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="exported_pdfs",
                        to="api.appointment",
                    ),
                ),
            ],
        ),
        migrations.RemoveIndex(
            model_name="exportedpdf",
            name="exported_pd_user_id_2bc34d_idx",
        ),
        migrations.AddIndex(
            model_name="exportedpdf",
            index=models.Index(fields=["user", "appointment", "created_at"], name="exported_pd_user_id_2bc34d_idx"),
        ),
    ]
