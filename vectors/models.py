from __future__ import annotations

import logging
import os
import uuid
from django.conf import settings
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
    A user's embedded preference/personalization context, stored as a table in
    the single unified application database (alongside every other model).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="preference_contexts",
        db_index=True,
    )
    source = models.CharField(max_length=30, default="signup")  # signup|med_update|visit_extract|manual
    content = models.TextField()

    # Dense embedding for ANN search (sentence-transformer).
    if VectorField:
        semantic_vec = VectorField(dimensions=384, null=True)
    else:
        semantic_vec = models.JSONField(default=list, blank=True)

    # Optional UMAP/PCA / pipeline provenance from an offline worker.
    representation_meta = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vector_preference_contexts"
        indexes = [
            models.Index(fields=["user", "created_at"]),
        ]


class SteeringVector(models.Model):
    """
    Steering vectors live in the LLM residual stream space and are injected at inference time.
    Stored separately so we can support multiple layers/variants per context.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    context = models.ForeignKey(
        PreferenceContext,
        on_delete=models.CASCADE,
        related_name="steering_vectors",
        db_index=True,
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="steering_vectors",
        db_index=True,
    )
    layer = models.IntegerField(default=0)
    # Store as JSON list of floats to avoid coupling to a specific tensor format.
    steering_vec = models.JSONField(default=list, blank=True)
    norm = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vector_steering_vectors"
        indexes = [
            models.Index(fields=["user", "context", "layer"]),
        ]

