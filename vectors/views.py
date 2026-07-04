from __future__ import annotations

import logging

from django.db import connection
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from api.llm_retrieval import encode_query_text

from .ml_bridge import local_ann_search
from .models import PreferenceContext, SteeringVector
from .serializers import (
    VectorSearchSerializer,
    PreferenceContextUpsertSerializer,
    SteeringVectorUpsertSerializer,
)

logger = logging.getLogger(__name__)


def _results_for_contexts(*, user_id: int, contexts: list, distances: list | None = None) -> list:
    results = []
    for i, ctx in enumerate(contexts):
        dist = None
        if distances is not None and i < len(distances):
            dist = distances[i]
        steer = list(
            SteeringVector.objects
            .filter(user_id=user_id, context_id=ctx.id)
            .order_by("layer", "-created_at")[:3]
        )
        results.append(
            {
                "context": {
                    "id": str(ctx.id),
                    "source": ctx.source,
                    "content": ctx.content,
                    "created_at": ctx.created_at,
                    "distance": float(dist) if dist is not None else None,
                },
                "steering_vectors": [
                    {"id": str(v.id), "layer": v.layer, "norm": v.norm, "steering_vec": v.steering_vec}
                    for v in steer
                ],
            }
        )
    return results


def _recent_contexts_response(*, user_id: int, top_k: int, note: str) -> Response:
    qs = (
        PreferenceContext.objects
        .filter(user_id=user_id)
        .order_by("-created_at")[:top_k]
    )
    contexts = list(qs)
    logger.info("vector_search mode=recent user_id=%s contexts=%s reason=%s", user_id, len(contexts), note)
    return Response({"results": _results_for_contexts(user_id=user_id, contexts=contexts, distances=None), "note": note})


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def search_preference_contexts(request):
    """
    ANN search on semantic_vec (pgvector cosine distance) when a query vector is available.

    Pass `query_vector` and/or `query_text`. Text is encoded via EMBEDDING_SERVICE_URL,
    LLM_INFERENCE_URL `/v1/embed`, or the local `machinelearning` embedder (see `vectors.ml_bridge`)
    when a vector is not sent.
    """
    s = VectorSearchSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    d = s.validated_data

    user_id = int(request.user.id)
    top_k = int(d["top_k"])
    qv = d.get("query_vector")
    qt = (d.get("query_text") or "").strip()
    if qv is None and qt:
        qv = encode_query_text(qt)

    note = None
    if qv is None and qt:
        note = "embedding_unavailable_fell_back_to_recent"

    if connection.vendor != "postgresql":
        # Non-Postgres backend (e.g. local SQLite dev): rank via the local FAISS bridge
        # instead of straight "most recent" whenever we have a query vector to work with.
        if qv is not None:
            candidates = [
                (str(ctx.id), ctx.semantic_vec)
                for ctx in PreferenceContext.objects.filter(user_id=user_id).exclude(semantic_vec=None)
                if ctx.semantic_vec
            ]
            ranked_ids = local_ann_search(query_vector=qv, candidates=candidates, top_k=top_k)
            if ranked_ids:
                by_id = {str(c.id): c for c in PreferenceContext.objects.filter(id__in=ranked_ids)}
                contexts = [by_id[i] for i in ranked_ids if i in by_id]
                logger.info("vector_search mode=local_ann user_id=%s contexts=%s", user_id, len(contexts))
                return Response({"results": _results_for_contexts(user_id=user_id, contexts=contexts, distances=None)})
        return _recent_contexts_response(
            user_id=user_id,
            top_k=top_k,
            note=note or f"non_postgres_backend:{connection.vendor}; returned most recent contexts",
        )

    try:
        from pgvector.django import CosineDistance  # type: ignore

        if qv is not None:
            qs = (
                PreferenceContext.objects
                .filter(user_id=user_id)
                .exclude(semantic_vec=None)
                .annotate(distance=CosineDistance("semantic_vec", qv))
                .order_by("distance")[:top_k]
            )
            contexts = list(qs)
            dists = [float(getattr(c, "distance", 0.0)) for c in contexts]
            logger.info("vector_search mode=ann user_id=%s contexts=%s", user_id, len(contexts))
            payload = {"results": _results_for_contexts(user_id=user_id, contexts=contexts, distances=dists)}
            if note:
                payload["note"] = note
            return Response(payload)
    except Exception:
        logger.warning("vector_search ANN unavailable; falling back to recent", exc_info=True)

    return _recent_contexts_response(
        user_id=user_id,
        top_k=top_k,
        note=note or "pgvector_not_active_or_no_query_vector; returned most recent contexts",
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upsert_preference_context(request):
    """
    Store a preference context and optional semantic embedding (sentence-transformer).
    Optional `representation_meta` holds UMAP/PCA pipeline outputs from an offline worker.
    """
    s = PreferenceContextUpsertSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    d = s.validated_data

    meta = d.get("representation_meta")
    if meta is None:
        meta = {}

    ctx = PreferenceContext.objects.create(
        user_id=int(request.user.id),
        source=(d.get("source") or "signup").strip() or "signup",
        content=d["content"],
        semantic_vec=d.get("semantic_vec") if "semantic_vec" in d else None,
        representation_meta=meta if isinstance(meta, dict) else {},
    )
    return Response({"context_id": str(ctx.id), "created_at": ctx.created_at}, status=201)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def add_steering_vector(request):
    """
    Store steering vectors derived from representation editing (PCA + mid/late-layer activations).
    """
    s = SteeringVectorUpsertSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    d = s.validated_data

    vec = d["steering_vec"]
    norm = float(d.get("norm") or 0.0)
    if not norm:
        norm = sum((float(x) * float(x) for x in vec)) ** 0.5

    item = SteeringVector.objects.create(
        user_id=int(request.user.id),
        context_id=d["context_id"],
        layer=int(d["layer"]),
        steering_vec=vec,
        norm=norm,
    )
    return Response({"steering_vector_id": str(item.id), "norm": item.norm}, status=201)
