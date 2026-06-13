"""Representation-editing steering workflow.

End-to-end pipeline that turns one user's preference contexts into the two
embedding spaces described in the construction overview:

* **Embedding space #1 — semantic_vec.** A 384-dim sentence-transformer
  embedding optimized for cosine similarity in pgvector.
* **Embedding space #2 — steering_vec.** A vector pulled from the LLM's
  residual stream at a configurable mid/late layer (default 12 of 16),
  PCA-reduced to its dominant direction across the user's contexts.

UMAP is used as a sanity check that the contexts cluster meaningfully — if
the silhouette score is below a threshold we still write the data but record
the warning in `representation_meta` so the worker can downgrade the
steering vector's confidence.

This module talks to two HTTP services:
  * EMBEDDING_SERVICE_URL (`/v1/embed`)        — the dedicated embedding tier.
  * LLM_INFERENCE_URL    (`/v1/extract_activations`) — the LLM service.

If either is unreachable the worker raises so the calling job can retry. We
deliberately avoid a "silently fall back to zeros" path because that would
poison ANN search.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

def _embedding_url() -> str:
    return (os.environ.get("EMBEDDING_SERVICE_URL") or "").strip().rstrip("/")


def _llm_url() -> str:
    return (os.environ.get("LLM_INFERENCE_URL") or "").strip().rstrip("/")


DEFAULT_LAYER = int(os.environ.get("LLM_DEFAULT_EXTRACT_LAYER", "12"))
UMAP_MIN_SILHOUETTE = float(os.environ.get("STEERING_UMAP_MIN_SILHOUETTE", "0.05"))
PCA_MAX_COMPONENTS = int(os.environ.get("STEERING_PCA_MAX_COMPONENTS", "1"))


# --------------------------------------------------------------------------- #
# Data shapes                                                                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SteeringArtifacts:
    semantic_vec: List[float]
    steering_vec: List[float]
    layer: int


@dataclass
class DerivationResult:
    """One row's worth of artifacts plus pipeline diagnostics."""

    semantic_vec: List[float]
    steering_vec: List[float]
    layer: int
    norm: float
    pca_explained_variance: float = 0.0
    umap_silhouette: float = 0.0
    umap_meaningful: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# HTTP helpers                                                                #
# --------------------------------------------------------------------------- #

def encode_semantic(text: str, *, timeout: float = 30.0) -> List[float]:
    """Call the embedding service for the dense MiniLM-384 representation."""
    base = _embedding_url() or _llm_url()
    if not base:
        raise RuntimeError("Neither EMBEDDING_SERVICE_URL nor LLM_INFERENCE_URL is set")
    payload = {"text": text[:8000]}
    resp = requests.post(f"{base}/v1/embed", json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    vec = body.get("embedding") or body.get("vector")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError(f"Embedding service returned no vector: {body!r}")
    return [float(x) for x in vec]


def extract_residual_activations(
    text: str,
    *,
    layer: Optional[int] = None,
    pool: str = "mean",
    timeout: float = 60.0,
) -> tuple[List[float], int]:
    """Ask the LLM service for the user-prompt's mid/late-layer activations."""
    base = _llm_url()
    if not base:
        raise RuntimeError("LLM_INFERENCE_URL is not set")
    payload = {"text": text[:4000], "layer": int(layer if layer is not None else DEFAULT_LAYER), "pool": pool}
    resp = requests.post(f"{base}/v1/extract_activations", json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    vec = body.get("vector")
    out_layer = int(body.get("layer", payload["layer"]))
    if not isinstance(vec, list) or not vec:
        raise RuntimeError(f"Activation extraction returned no vector: {body!r}")
    return [float(x) for x in vec], out_layer


# --------------------------------------------------------------------------- #
# PCA / UMAP                                                                  #
# --------------------------------------------------------------------------- #

def derive_pca_direction(
    activations: Sequence[Sequence[float]],
    *,
    n_components: int = 1,
) -> tuple[List[float], float]:
    """Return the leading principal direction (unit-norm) and explained variance ratio.

    `activations` is a [n_contexts, hidden_dim] matrix. When n_contexts == 1
    we return the single vector (normalized). When all rows are identical the
    explained variance is 1.0 and we return that direction.
    """
    import numpy as np

    arr = np.asarray(activations, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("activations must be a non-empty 2D matrix")

    if arr.shape[0] == 1:
        v = arr[0]
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist(), 1.0

    # Center the rows; PCA is meaningful only on centered data.
    centered = arr - arr.mean(axis=0, keepdims=True)
    if not np.any(centered):
        v = arr[0]
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist(), 1.0

    from sklearn.decomposition import PCA

    k = max(1, min(int(n_components), centered.shape[0] - 1, centered.shape[1]))
    pca = PCA(n_components=k)
    pca.fit(centered)
    direction = pca.components_[0]
    n = float(np.linalg.norm(direction)) or 1.0
    return (direction / n).tolist(), float(pca.explained_variance_ratio_[0])


def umap_cluster_quality(activations: Sequence[Sequence[float]]) -> tuple[float, bool]:
    """Run UMAP + silhouette score as a sanity check on cluster meaningfulness.

    Returns (silhouette_score, meaningful_bool). With fewer than 4 contexts
    the test is degenerate; we return 0.0 / False without raising so the
    worker can still persist what it has.
    """
    import numpy as np

    arr = np.asarray(activations, dtype=np.float32)
    if arr.shape[0] < 4:
        return 0.0, False
    try:
        import umap
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as exc:  # pragma: no cover
        logger.warning("UMAP/sklearn unavailable, skipping cluster check: %s", exc)
        return 0.0, False

    try:
        n_neighbors = min(15, max(2, arr.shape[0] - 1))
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
        emb = reducer.fit_transform(arr)
        n_clusters = max(2, min(5, arr.shape[0] // 2))
        labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(emb)
        if len(set(labels)) < 2:
            return 0.0, False
        score = float(silhouette_score(emb, labels))
        return score, score >= UMAP_MIN_SILHOUETTE
    except Exception as exc:  # pragma: no cover
        logger.info("UMAP sanity check failed: %s", exc)
        return 0.0, False


# --------------------------------------------------------------------------- #
# Top-level: derive artifacts for a user                                      #
# --------------------------------------------------------------------------- #

def derive_user_steering(
    contexts: Sequence[str],
    *,
    layer: Optional[int] = None,
) -> List[DerivationResult]:
    """Run the full pipeline for one user.

    For each input context we compute its semantic vector and its raw
    activation. We then PCA across all activations to get the user's
    *dominant* steering direction; that single direction is reused as the
    steering_vec for every context (so retrieval at inference time always
    returns the same direction regardless of which context matched). The
    per-context explained variance / UMAP score live on each row's
    diagnostics so the worker can decide whether to apply or quarantine.
    """
    import numpy as np

    if not contexts:
        return []

    semantic_vecs: List[List[float]] = []
    activations: List[List[float]] = []
    chosen_layer = int(layer if layer is not None else DEFAULT_LAYER)

    for c in contexts:
        text = (c or "").strip()
        if not text:
            continue
        semantic_vecs.append(encode_semantic(text))
        vec, out_layer = extract_residual_activations(text, layer=chosen_layer)
        activations.append(vec)
        chosen_layer = out_layer  # the service may snap to its actual layer

    if not activations:
        return []

    direction, explained = derive_pca_direction(activations, n_components=PCA_MAX_COMPONENTS)
    silhouette, meaningful = umap_cluster_quality(activations)

    direction_arr = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction_arr))

    results: List[DerivationResult] = []
    for sv in semantic_vecs:
        results.append(
            DerivationResult(
                semantic_vec=sv,
                steering_vec=direction,
                layer=chosen_layer,
                norm=norm,
                pca_explained_variance=explained,
                umap_silhouette=silhouette,
                umap_meaningful=meaningful,
                diagnostics={
                    "n_contexts": len(activations),
                    "extract_layer": chosen_layer,
                    "umap_threshold": UMAP_MIN_SILHOUETTE,
                },
            )
        )
    return results
