from __future__ import annotations

import logging
import os
import uuid
from django.db import models

try:
    from pgvector.django import VectorField
    _VECTOR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    VectorField = None
    _VECTOR_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

_REQUIRE_PGVECTOR = (os.environ.get("REQUIRE_PGVECTOR") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if VectorField is None:
    msg = "pgvector unavailable; semantic_vec will use JSONField fallback and ANN retrieval will be disabled"
    if _REQUIRE_PGVECTOR:
        raise RuntimeError(f"{msg}. Set up pgvector/psycopg correctly.") from _VECTOR_IMPORT_ERROR
    logger.warning(msg)


class PreferenceContext(models.Model):
    """
    Stored in the `vectors` DB. No cross-db FKs.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.BigIntegerField(db_index=True)
    source = models.CharField(max_length=30, default="signup")  # signup|med_update|visit_extract|manual
    content = models.TextField()

    # Dense embedding for ANN search (sentence-transformer).
    if VectorField:
        semantic_vec = VectorField(dimensions=384, null=True)
    else:
        semantic_vec = models.JSONField(default=list, blank=True)

    # Optional UMAP/PCA / pipeline provenance (RDS #2); does not replace semantic_vec or steering_vec.
    representation_meta = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vector_preference_contexts"
        indexes = [
            models.Index(fields=["user_id", "created_at"]),
        ]


class SteeringVector(models.Model):
    """
    Steering vectors live in the LLM residual stream space and are injected at inference time.
    Stored separately so we can support multiple layers/variants per context.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    context_id = models.UUIDField(db_index=True)
    user_id = models.BigIntegerField(db_index=True)
    layer = models.IntegerField(default=0)
    # Store as JSON list of floats to avoid coupling to a specific tensor format.
    steering_vec = models.JSONField(default=list, blank=True)
    norm = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vector_steering_vectors"
        indexes = [
            models.Index(fields=["user_id", "context_id", "layer"]),
        ]

