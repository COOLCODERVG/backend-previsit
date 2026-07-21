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
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

from . import ml_bridge

logger = logging.getLogger(__name__)

DEFAULT_LLM_INFERENCE_URL = "http://127.0.0.1:11434"


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

def _embedding_url() -> str:
    return (os.environ.get("EMBEDDING_SERVICE_URL") or "").strip().rstrip("/")


def _llm_url() -> str:
    return (os.environ.get("LLM_INFERENCE_URL") or DEFAULT_LLM_INFERENCE_URL).strip().rstrip("/")


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
#                                                                              #
# Both are delegated to the embedding service's /v1/reduce/pca and           #
# /v1/reduce/umap endpoints (see vectors.ml_bridge), which are themselves    #
# thin wrappers around machinelearning/utils/pca.py and                     #
# utils/umap_reduce.py -- the single source of truth. Nothing here           #
# reimplements sklearn/UMAP; Django only makes the HTTP call and applies a   #
# couple of degenerate-input guards (single context / all-identical         #
# contexts) that are cheap enough to stay local and avoid a pointless        #
# round-trip for trivial inputs.                                             #
# --------------------------------------------------------------------------- #

def _l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
    return [float(x) / norm for x in vec]


def derive_pca_direction(
    activations: Sequence[Sequence[float]],
    *,
    n_components: int = 1,
) -> tuple[List[float], float]:
    """Return the leading principal direction (unit-norm) and explained variance ratio.

    `activations` is a [n_contexts, hidden_dim] matrix. When n_contexts == 1
    we return the single vector (normalized). When all rows are identical the
    explained variance is 1.0 and we return that direction. Otherwise this
    calls the embedding service's PCA endpoint
    (`machinelearning/utils/pca.py::fit_pca`) and takes its leading component.
    """
    arr = [list(row) for row in activations]
    if not arr:
        raise ValueError("activations must be a non-empty 2D matrix")

    if len(arr) == 1:
        return _l2_normalize(arr[0]), 1.0

    first = arr[0]
    if all(row == first for row in arr[1:]):
        return _l2_normalize(first), 1.0

    result = ml_bridge.reduce_pca(arr, n_components=n_components)
    if not result:
        raise RuntimeError(
            "EMBEDDING_SERVICE_URL is unreachable/unconfigured -- cannot derive PCA steering direction"
        )
    components = result.get("components") or []
    if not components:
        raise RuntimeError(f"PCA endpoint returned no components: {result!r}")
    variances = result.get("explained_variance_ratio") or [0.0]
    return _l2_normalize(components[0]), float(variances[0])


def umap_cluster_quality(activations: Sequence[Sequence[float]]) -> tuple[float, bool]:
    """Run UMAP + silhouette score as a sanity check on cluster meaningfulness.

    Returns (silhouette_score, meaningful_bool). With fewer than 4 contexts
    the test is degenerate; we return 0.0 / False without raising so the
    worker can still persist what it has. Delegates the actual UMAP
    projection + K-Means/silhouette check to the embedding service
    (`machinelearning/utils/umap_reduce.py` + `utils/clustering.py`) via
    `vectors.ml_bridge.reduce_umap(..., check_clusters=True)`; the meaningful
    threshold itself stays local so it remains tunable via
    `STEERING_UMAP_MIN_SILHOUETTE` independently of the service's own default.

    Note: the shared `umap_reduce` helper requires >= 10 samples (a
    deliberate guardrail in the canonical implementation); with 4-9 contexts
    this will therefore also return 0.0/False rather than attempting a
    lower-confidence fit. That's an intentional trade-off of reusing the
    single source of truth's stricter guard instead of a laxer duplicate.
    """
    if len(activations) < 4:
        return 0.0, False

    result = ml_bridge.reduce_umap(activations, n_components=2, check_clusters=True)
    if not result:
        logger.info("embedding service unavailable/rejected UMAP request; skipping cluster check")
        return 0.0, False

    score = float(result.get("silhouette", 0.0))
    return score, score >= UMAP_MIN_SILHOUETTE


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

    norm = math.sqrt(sum(float(x) * float(x) for x in direction))

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
