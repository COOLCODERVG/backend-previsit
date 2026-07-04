from __future__ import annotations

"""
Knowledge-base RAG retrieval for LLM-inference context assembly.

Thin wrapper around `vectors.ml_bridge.rag_query` (itself an HTTP call to the
embedding service's `/v1/rag/query`, backed end-to-end by
`machinelearning/rag/pipeline.py` -- multi-stage retrieval, hybrid
vector+keyword scoring, cross-encoder rerank, dedup, and citation tracking)
so callers assembling an LLM payload can attach retrieved, cited
knowledge-base context without knowing anything about the RAG service's
transport details.

This is what makes the "Full RAG pipeline... provided as context to the LLM
during inference" requirement concrete: `retrieve_rag_context` is called from
`api.utils.generate_visit_one_pager` / `generate_llm_pdf_guidance`, and its
output is merged directly into the payload sent to the LLM service, which
surfaces it to the model alongside the personalization steering vectors.
"""

import logging
from typing import Any, Dict

from vectors.ml_bridge import rag_query as _rag_query

logger = logging.getLogger(__name__)


def retrieve_rag_context(query: str, *, top_k: int = 5) -> Dict[str, Any]:
    """
    Retrieve knowledge-base context for `query` via the full RAG pipeline.

    Returns `{"rag_context": str, "rag_citations": [...]}`. Always returns a
    dict (never None/raises) so call sites can unconditionally merge it into
    an LLM payload -- an unreachable/unconfigured embedding service just
    yields an empty context and no citations rather than breaking generation.
    """
    query = (query or "").strip()
    if not query:
        return {"rag_context": "", "rag_citations": []}

    result = _rag_query(query, top_k=top_k)
    if not result:
        logger.info(
            "rag_retrieval unavailable for query=%r; continuing without knowledge-base context",
            query[:80],
        )
        return {"rag_context": "", "rag_citations": []}

    return {
        "rag_context": result.get("context_text") or "",
        "rag_citations": result.get("citations") or [],
    }
