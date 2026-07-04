from __future__ import annotations

"""
HTTP bridge to the `services/embedding` ML API -- embeddings, ANN search, RAG
retrieval, PCA/UMAP dimensionality reduction, and clustering.

Per the "models are called via an API, not locally downloaded" requirement,
the Django backend NEVER imports the top-level `machinelearning/` package or
loads a sentence-transformer/FAISS index in-process. Every function here is a
thin HTTP client around the dedicated embedding/ML service
(`EMBEDDING_SERVICE_URL`), which is itself a wrapper around `machinelearning/`
-- the single source of truth for all of this logic (see
`services/embedding/app/main.py`). No embedding/ANN/PCA/UMAP/clustering logic
is duplicated in this file; it only translates Django-side calls into HTTP
requests and back.

Every function degrades gracefully (returns None when the service is
unreachable or unconfigured) so callers can fall back to "most recent
contexts" behavior already present in `api.llm_retrieval` instead of raising.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


def _embedding_service_url() -> str:
    return (os.environ.get("EMBEDDING_SERVICE_URL") or "").strip().rstrip("/")


def ml_bridge_available() -> bool:
    """True once an embedding/ML service base URL is configured.

    This doesn't itself guarantee the service is reachable right now --
    individual calls below still degrade to `None`/empty on connection
    errors or timeouts so callers can fall back gracefully.
    """
    return bool(_embedding_service_url())


def _post(path: str, payload: Dict[str, Any], *, timeout: float = _DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    base = _embedding_service_url()
    if not base:
        return None
    try:
        resp = requests.post(f"{base}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.warning("embedding service call failed: POST %s", path, exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #

def embed_texts(texts: Sequence[str]) -> Optional[List[List[float]]]:
    """
    Embed one or more strings via the embedding service's `/v1/embed`
    (backed by `machinelearning/embeddings/embedder.py`, all-MiniLM-L6-v2,
    384-dim -- matching `PreferenceContext.semantic_vec`'s pgvector dimension
    exactly). Requests are served through the service's embedding cache
    (`machinelearning/embeddings/cache.py`) so repeated text (e.g. re-syncing
    unchanged personalization answers) doesn't re-hit the model.

    Returns None when the service is unreachable/unconfigured so callers can
    fall back to an alternate embedding source or "most recent contexts"
    ordering.
    """
    items = [str(t or "") for t in texts]
    if not any(items):
        return None
    body = _post("/v1/embed", {"texts": items})
    if not body:
        return None
    vecs = body.get("embeddings")
    if isinstance(vecs, list) and vecs:
        return vecs
    return None


def embed_text(text: str) -> Optional[List[float]]:
    """Single-string convenience wrapper around `embed_texts`."""
    out = embed_texts([text])
    return out[0] if out else None


# --------------------------------------------------------------------------- #
# ANN / vector search                                                        #
# --------------------------------------------------------------------------- #

def local_ann_search(
    *,
    query_vector: Sequence[float],
    candidates: Sequence[Tuple[str, Sequence[float]]],
    top_k: int = 5,
) -> Optional[List[str]]:
    """
    Rank `candidates` (id, embedding) pairs against `query_vector` via the
    embedding service's `/v1/ann_search` (FAISS flat index -- see
    `machinelearning/embeddings/vector_db.py`). Used when the active Django DB
    connection isn't Postgres/pgvector (e.g. local SQLite dev), so in-database
    cosine-distance ANN can't run -- see `api.llm_retrieval.steering_vectors_ann`.

    Returns ids in ranked (most-similar-first) order, or None when the
    service is unreachable or there are no usable candidates, so callers can
    fall back gracefully to "most recent contexts".
    """
    rows = [(cid, vec) for cid, vec in candidates if vec]
    if not rows or not query_vector:
        return None
    body = _post(
        "/v1/ann_search",
        {
            "query_vector": list(query_vector),
            "candidates": [{"id": cid, "vector": list(vec)} for cid, vec in rows],
            "top_k": top_k,
        },
    )
    if not body:
        return None
    hits = body.get("hits")
    if not isinstance(hits, list):
        return None
    return [h["id"] for h in hits if isinstance(h, dict) and "id" in h]


# --------------------------------------------------------------------------- #
# Dimensionality reduction                                                    #
# --------------------------------------------------------------------------- #

def reduce_pca(
    vectors: Sequence[Sequence[float]],
    *,
    n_components: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    PCA via the embedding service's `/v1/reduce/pca`
    (`machinelearning/utils/pca.py`, no reimplementation in Django).

    Returns the raw response dict on success:
      `components`                 -- (n_components, dim) leading directions
      `explained_variance_ratio`   -- per-component variance ratio
      `projected`                  -- (N, n_components) coordinates
      `reconstructed`              -- (N, dim) denoised/steering vectors
      `mean`                       -- (dim,) fitted mean
    or None if the service is unreachable/unconfigured.
    """
    if not vectors:
        return None
    return _post("/v1/reduce/pca", {"vectors": [list(v) for v in vectors], "n_components": n_components})


def reduce_umap(
    vectors: Sequence[Sequence[float]],
    *,
    n_components: int = 10,
    check_clusters: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    UMAP via the embedding service's `/v1/reduce/umap`
    (`machinelearning/utils/umap_reduce.py`). When `check_clusters=True`, the
    service also runs a K-Means + silhouette pass on the UMAP output
    (`machinelearning/utils/clustering.py`) and returns `silhouette`/`meaningful`
    alongside `reduced`, replacing the old inline cluster-quality check.
    """
    if not vectors:
        return None
    return _post(
        "/v1/reduce/umap",
        {"vectors": [list(v) for v in vectors], "n_components": n_components, "check_clusters": check_clusters},
    )


# --------------------------------------------------------------------------- #
# Clustering                                                                  #
# --------------------------------------------------------------------------- #

def cluster_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    method: str = "kmeans",
    n_clusters: int = 3,
    eps: float = 0.5,
    min_samples: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    K-Means/DBSCAN clustering via the embedding service's `/v1/cluster`
    (`machinelearning/utils/clustering.py`). Used for user-preference
    clustering / semantic grouping / topic discovery over a set of
    `PreferenceContext.semantic_vec` rows (see
    `vectors.management.commands.cluster_user_preferences`).
    """
    if not vectors:
        return None
    return _post(
        "/v1/cluster",
        {
            "vectors": [list(v) for v in vectors],
            "method": method,
            "n_clusters": n_clusters,
            "eps": eps,
            "min_samples": min_samples,
        },
    )


# --------------------------------------------------------------------------- #
# RAG (knowledge-base retrieval)                                              #
# --------------------------------------------------------------------------- #

def rag_query(query: str, *, top_k: int = 5, fetch_k: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Full RAG pipeline (multi-stage retrieval, hybrid vector+keyword scoring,
    cross-encoder rerank, dedup, citation tracking) over the knowledge-base
    index via the embedding service's `/v1/rag/query`
    (`machinelearning/rag/pipeline.py`).

    Returns `{"query", "context_text", "citations", "chunks"}` on success, or
    None if the service is unreachable/unconfigured -- see
    `api.rag_retrieval.retrieve_rag_context` for the calling convention used
    at LLM-inference time.
    """
    query = (query or "").strip()
    if not query:
        return None
    payload: Dict[str, Any] = {"query": query, "top_k": top_k}
    if fetch_k is not None:
        payload["fetch_k"] = fetch_k
    return _post("/v1/rag/query", payload)

