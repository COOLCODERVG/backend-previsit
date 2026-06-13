"""Convert ``PreferenceContext.semantic_vec`` from JSONField to pgvector ``vector(384)``.

This migration is intentionally idempotent and conditional:

* On Postgres with the ``vector`` extension installed it ALTERs the column
  to ``vector(384)``, casting any existing JSON rows on the way through.
* On Postgres without the extension (or any other backend such as SQLite for
  local dev) it is a no-op so the migration history stays linear.

The Django model in :mod:`vectors.models` already swaps to
``pgvector.django.VectorField`` at import time when pgvector is importable, so
once this migration runs the ORM and the column type are aligned and
``CosineDistance`` annotations stop falling back to the "recent contexts"
path in :func:`api.llm_retrieval.steering_vectors_ann`.
"""

from __future__ import annotations

from django.db import migrations


def upgrade(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return

    with schema_editor.connection.cursor() as cursor:
        # Make sure the extension exists; 0002 already did this best-effort.
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            return  # extension not installable — bail without breaking history

        # Verify the extension is now present before we ALTER the column.
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if not cursor.fetchone():
            return

        # If the column is already `vector(384)` we're done.
        cursor.execute(
            """
            SELECT data_type, udt_name
              FROM information_schema.columns
             WHERE table_name = 'vector_preference_contexts'
               AND column_name = 'semantic_vec'
            """
        )
        row = cursor.fetchone()
        if row and row[1] == "vector":
            return

        # Cast jsonb -> text -> vector. If the JSON is null/empty we leave NULL.
        cursor.execute(
            """
            ALTER TABLE vector_preference_contexts
            ALTER COLUMN semantic_vec DROP DEFAULT,
            ALTER COLUMN semantic_vec TYPE vector(384)
            USING (
                CASE
                    WHEN semantic_vec IS NULL THEN NULL
                    WHEN jsonb_typeof(semantic_vec) = 'array'
                         AND jsonb_array_length(semantic_vec) = 384
                    THEN semantic_vec::text::vector(384)
                    ELSE NULL
                END
            )
            """
        )

        # Optional: an IVFFlat index on the cosine operator class. We use a
        # conservative `lists` value tuned for early-stage volumes; you can
        # `REINDEX` later once the table grows.
        try:
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS vector_pref_semantic_vec_cosine_idx
                ON vector_preference_contexts
                USING ivfflat (semantic_vec vector_cosine_ops)
                WITH (lists = 50)
                """
            )
        except Exception:
            # Older pgvector builds may lack ivfflat; the table still works
            # without an ANN index, just slower at scale.
            pass


def downgrade(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        try:
            cursor.execute("DROP INDEX IF EXISTS vector_pref_semantic_vec_cosine_idx")
        except Exception:
            pass
        try:
            cursor.execute(
                """
                ALTER TABLE vector_preference_contexts
                ALTER COLUMN semantic_vec TYPE jsonb
                USING to_jsonb(semantic_vec::float8[])
                """
            )
        except Exception:
            pass


class Migration(migrations.Migration):
    dependencies = [
        ("vectors", "0003_preferencecontext_representation_meta"),
    ]

    operations = [
        migrations.RunPython(upgrade, reverse_code=downgrade),
    ]
