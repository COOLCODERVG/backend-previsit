from __future__ import annotations

"""
Inference-time retrieval for representation editing:

1) Encode user query with a sentence-transformer (external embedding service).
2) ANN search on PreferenceContext.semantic_vec (pgvector cosine distance).
3) Load matching SteeringVector rows (residual-stream space) for injection at the LLM service.

Falls back to "most recent contexts" when pgvector or embeddings are unavailable.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from django.db import connection

from vectors.ml_bridge import embed_text as _local_embed_text, local_ann_search as _local_ann_search
from vectors.models import PreferenceContext, SteeringVector

logger = logging.getLogger(__name__)

DEFAULT_LLM_INFERENCE_URL = "http://127.0.0.1:11434"


def _embedding_service_url() -> str:
    return (os.environ.get("EMBEDDING_SERVICE_URL") or "").strip().rstrip("/")


def _llm_base_url() -> str:
    return (os.environ.get("LLM_INFERENCE_URL") or DEFAULT_LLM_INFERENCE_URL).strip().rstrip("/")


def encode_query_text(text: str, *, timeout_seconds: float = 30.0) -> Optional[List[float]]:
    """
    Produce a dense vector for ANN search. Configure one of:
      - EMBEDDING_SERVICE_URL (preferred): POST {url}/v1/embed
      - LLM_INFERENCE_URL: POST {url}/v1/embed

    Request body: {"text": "..."} (and "input" duplicated for common servers).
    Response: {"embedding": [...]} or {"vector": [...]} or API-compatible {"data":[{"embedding":...}]}

    Falls back to the local `machinelearning` sentence-transformer embedder (see
    `vectors.ml_bridge`) when neither HTTP service is reachable, so embedding still
    works offline / in local dev without the dedicated embedding container running.
    """
    text = (text or "").strip()
    if not text:
        return None

    bases = []
    u = _embedding_service_url()
    if u:
        bases.append(u)
    llm = _llm_base_url()
    if llm and llm not in bases:
        bases.append(llm)

    payload = {"text": text[:8000], "input": text[:8000]}
    for base in bases:
        try:
            resp = requests.post(f"{base}/v1/embed", json=payload, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.debug("embed request failed for %s", base, exc_info=True)
            continue

        if not isinstance(data, dict):
            continue
        emb = data.get("embedding")
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb]
        vec = data.get("vector")
        if isinstance(vec, list) and vec:
            return [float(x) for x in vec]
        data_list = data.get("data")
        if isinstance(data_list, list) and data_list:
            first = data_list[0]
            if isinstance(first, dict):
                inner = first.get("embedding")
                if isinstance(inner, list) and inner:
                    return [float(x) for x in inner]

    local_vec = _local_embed_text(text)
    if local_vec:
        logger.info("encode_query_text: used local machinelearning embedder fallback")
        return local_vec

    return None


def _steering_dicts_for_contexts(user_id: int, contexts: List[PreferenceContext], per_context: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ctx in contexts:
        steer = list(
            SteeringVector.objects
            .filter(user_id=user_id, context_id=ctx.id)
            .order_by("-norm", "layer", "-created_at")[:per_context]
        )
        for v in steer:
            if v.steering_vec:
                out.append(
                    {
                        "layer": int(v.layer),
                        "vector": v.steering_vec,
                        "norm": float(v.norm),
                        "context_id": str(v.context_id),
                    }
                )
    return out


def steering_vectors_recent(
    *,
    user_id: int,
    top_k: int = 5,
    per_context: int = 3,
    reason: str = "fallback_recent",
) -> List[Dict[str, Any]]:
    contexts = list(
        PreferenceContext.objects
        .filter(user_id=user_id)
        .order_by("-created_at")[:top_k]
    )
    out = _steering_dicts_for_contexts(user_id, contexts, per_context=per_context)
    logger.info(
        "steering_retrieval mode=recent reason=%s user_id=%s contexts=%s vectors=%s",
        reason,
        user_id,
        len(contexts),
        len(out),
    )
    return out


def _steering_vectors_local_ann(
    *, user_id: int, query_vector: List[float], top_k: int, per_context: int, reason: str
) -> List[Dict[str, Any]]:
    """ANN over the user's own contexts via the local FAISS bridge (see `vectors.ml_bridge`),
    used when the active DB connection isn't Postgres/pgvector. Ranks in-memory since one
    user's context set is small (dozens of rows at most)."""
    candidates = [
        (str(ctx.id), ctx.semantic_vec)
        for ctx in PreferenceContext.objects.filter(user_id=user_id).exclude(semantic_vec=None)
        if ctx.semantic_vec
    ]
    ranked_ids = _local_ann_search(query_vector=query_vector, candidates=candidates, top_k=top_k)
    if not ranked_ids:
        return steering_vectors_recent(user_id=user_id, top_k=top_k, per_context=per_context, reason=reason)

    by_id = {str(c.id): c for c in PreferenceContext.objects.filter(id__in=ranked_ids)}
    contexts = [by_id[i] for i in ranked_ids if i in by_id]
    out = _steering_dicts_for_contexts(user_id, contexts, per_context=per_context)
    logger.info(
        "steering_retrieval mode=local_ann reason=%s user_id=%s contexts=%s vectors=%s",
        reason,
        user_id,
        len(contexts),
        len(out),
    )
    return out


def steering_vectors_ann(*, user_id: int, query_vector: List[float], top_k: int = 5, per_context: int = 3) -> List[Dict[str, Any]]:
    if connection.vendor != "postgresql":
        return _steering_vectors_local_ann(
            user_id=user_id,
            query_vector=query_vector,
            top_k=top_k,
            per_context=per_context,
            reason=f"non_postgres_backend:{connection.vendor}",
        )
    try:
        from pgvector.django import CosineDistance  # type: ignore

        semantic_field = PreferenceContext._meta.get_field("semantic_vec")
        if semantic_field.__class__.__name__ != "VectorField":
            return _steering_vectors_local_ann(
                user_id=user_id,
                query_vector=query_vector,
                top_k=top_k,
                per_context=per_context,
                reason="semantic_vec_not_vector_field",
            )

        qs = (
            PreferenceContext.objects
            .filter(user_id=user_id)
            .exclude(semantic_vec=None)
            .annotate(distance=CosineDistance("semantic_vec", query_vector))
            .order_by("distance")[:top_k]
        )
        contexts = list(qs)
        if contexts:
            out = _steering_dicts_for_contexts(user_id, contexts, per_context=per_context)
            logger.info(
                "steering_retrieval mode=ann user_id=%s contexts=%s vectors=%s dim=%s",
                user_id,
                len(contexts),
                len(out),
                len(query_vector),
            )
            return out
        return steering_vectors_recent(
            user_id=user_id,
            top_k=top_k,
            per_context=per_context,
            reason="ann_empty_result",
        )
    except Exception:
        logger.warning("ANN steering retrieval unavailable; falling back to recent", exc_info=True)

    return steering_vectors_recent(
        user_id=user_id,
        top_k=top_k,
        per_context=per_context,
        reason="ann_exception",
    )


def retrieve_steering_for_inference(
    *,
    user_id: int,
    query_text: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Prefer ANN when a query vector is available (directly or via encode_query_text).
    Otherwise use recent preference contexts (same ordering as pre-retrieval stub).
    """
    if query_vector:
        logger.info("steering_retrieval request=user_provided_vector user_id=%s top_k=%s", user_id, top_k)
        return steering_vectors_ann(user_id=user_id, query_vector=query_vector, top_k=top_k)

    q = (query_text or "").strip()
    if q:
        vec = encode_query_text(q)
        if vec:
            logger.info("steering_retrieval request=encoded_query user_id=%s top_k=%s", user_id, top_k)
            return steering_vectors_ann(user_id=user_id, query_vector=vec, top_k=top_k)
        logger.warning("steering_retrieval embedding_unavailable user_id=%s; falling back to recent", user_id)

    return steering_vectors_recent(user_id=user_id, top_k=top_k, reason="no_query_vector")


def build_pdf_guidance_retrieval_query(deidentified: Dict[str, Any]) -> str:
    """Concatenate de-identified signals into one string for semantic retrieval."""
    parts: List[str] = []
    pers = (deidentified or {}).get("personalization") or {}
    for key in ("main_reason", "biggest_concern", "condition_stage", "appointment_outcome"):
        v = pers.get(key)
        if v:
            parts.append(str(v))
    tq = (deidentified or {}).get("top_questions") or []
    if isinstance(tq, list):
        parts.extend(str(x) for x in tq[:4] if x)
    ts = (deidentified or {}).get("top_symptoms") or []
    if isinstance(ts, list):
        for item in ts[:4]:
            if isinstance(item, dict) and item.get("name"):
                parts.append(str(item.get("name")))
    ml = pers.get("ml_preferences") or []
    if isinstance(ml, list):
        for item in ml[:6]:
            if not isinstance(item, dict):
                continue
            q = item.get("question")
            a = item.get("answer")
            if q:
                parts.append(str(q))
            if a:
                parts.append(str(a))
    return "\n".join(parts).strip()
