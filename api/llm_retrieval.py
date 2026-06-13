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

from vectors.models import PreferenceContext, SteeringVector

logger = logging.getLogger(__name__)


def _embedding_service_url() -> str:
    return (os.environ.get("EMBEDDING_SERVICE_URL") or "").strip().rstrip("/")


def _llm_base_url() -> str:
    return (os.environ.get("LLM_INFERENCE_URL") or "").strip().rstrip("/")


def encode_query_text(text: str, *, timeout_seconds: float = 30.0) -> Optional[List[float]]:
    """
    Produce a dense vector for ANN search. Configure one of:
      - EMBEDDING_SERVICE_URL (preferred): POST {url}/v1/embed
      - LLM_INFERENCE_URL: POST {url}/v1/embed

    Request body: {"text": "..."} (and "input" duplicated for common servers).
    Response: {"embedding": [...]} or {"vector": [...]} or OpenAI-style {"data":[{"embedding":...}]}
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

    return None


def _steering_dicts_for_contexts(user_id: int, contexts: List[PreferenceContext], per_context: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ctx in contexts:
        steer = list(
            SteeringVector.objects.using("vectors")
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


def steering_vectors_recent(*, user_id: int, top_k: int = 5, per_context: int = 3) -> List[Dict[str, Any]]:
    contexts = list(
        PreferenceContext.objects.using("vectors")
        .filter(user_id=user_id)
        .order_by("-created_at")[:top_k]
    )
    return _steering_dicts_for_contexts(user_id, contexts, per_context=per_context)


def steering_vectors_ann(*, user_id: int, query_vector: List[float], top_k: int = 5, per_context: int = 3) -> List[Dict[str, Any]]:
    try:
        from pgvector.django import CosineDistance  # type: ignore

        qs = (
            PreferenceContext.objects.using("vectors")
            .filter(user_id=user_id)
            .exclude(semantic_vec=None)
            .annotate(distance=CosineDistance("semantic_vec", query_vector))
            .order_by("distance")[:top_k]
        )
        contexts = list(qs)
        if contexts:
            return _steering_dicts_for_contexts(user_id, contexts, per_context=per_context)
    except Exception:
        logger.debug("ANN steering retrieval unavailable", exc_info=True)

    return steering_vectors_recent(user_id=user_id, top_k=top_k, per_context=per_context)


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
        return steering_vectors_ann(user_id=user_id, query_vector=query_vector, top_k=top_k)

    q = (query_text or "").strip()
    if q:
        vec = encode_query_text(q)
        if vec:
            return steering_vectors_ann(user_id=user_id, query_vector=vec, top_k=top_k)

    return steering_vectors_recent(user_id=user_id, top_k=top_k)


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
    return "\n".join(parts).strip()
