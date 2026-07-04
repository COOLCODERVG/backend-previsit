from __future__ import annotations

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    """
    Converts PreferenceContext.user_id / SteeringVector.context_id / SteeringVector.user_id
    from plain BigIntegerField/UUIDField columns into real ForeignKeys, now that the
    `vectors` app lives in the single unified application database alongside `api` (no more
    cross-database FK restriction).

    This is deliberately a *state-only* field change (`SeparateDatabaseAndState` with empty
    `database_operations`): the underlying database columns keep the exact same name
    (`user_id`, `context_id`) and storage type (bigint / uuid) that a ForeignKey to
    `api.User`/`vectors.PreferenceContext` requires, so there is nothing to actually ALTER
    at the database level \u2014 only Django's own model bookkeeping changes. This avoids
    `makemigrations` needing a one-off default value for a new NOT-NULL column (which would
    otherwise prompt interactively), since no new column is really being added.

    The affected indexes are renamed for real (a plain, fast, default-free DROP+CREATE on the
    identical underlying columns) so a later `makemigrations --check` has nothing left to detect.
    """

    dependencies = [
        ("vectors", "0006_alter_preferencecontext_semantic_vec"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                migrations.RemoveField(
                    model_name="preferencecontext",
                    name="user_id",
                ),
                migrations.AddField(
                    model_name="preferencecontext",
                    name="user",
                    field=models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="preference_contexts",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                migrations.RemoveField(
                    model_name="steeringvector",
                    name="context_id",
                ),
                migrations.AddField(
                    model_name="steeringvector",
                    name="context",
                    field=models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="steering_vectors",
                        to="vectors.preferencecontext",
                    ),
                ),
                migrations.RemoveField(
                    model_name="steeringvector",
                    name="user_id",
                ),
                migrations.AddField(
                    model_name="steeringvector",
                    name="user",
                    field=models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="steering_vectors",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
        migrations.RemoveIndex(
            model_name="preferencecontext",
            name="vector_pref_user_id_96ac85_idx",
        ),
        migrations.AddIndex(
            model_name="preferencecontext",
            index=models.Index(fields=["user", "created_at"], name="vector_pref_user_id_96ac85_idx"),
        ),
        migrations.RemoveIndex(
            model_name="steeringvector",
            name="vector_stee_user_id_eb0593_idx",
        ),
        migrations.AddIndex(
            model_name="steeringvector",
            index=models.Index(fields=["user", "context", "layer"], name="vector_stee_user_id_eb0593_idx"),
        ),
    ]
