from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

# Default matches NeuraVia spec (host should pin exact revision in inference image).
DEFAULT_LLM_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_LLM_INFERENCE_URL = "http://127.0.0.1:11434"
logger = logging.getLogger(__name__)

VISIT_ONE_PAGER_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json",
    "schema": {
        "headline": "string",
        "focus_summary": "string",
        "key_takeaways": "array",
        "action_items": "array",
        "questions_for_doctor": "array",
        "caregiver_blurb": "string",
    },
}

PDF_GUIDANCE_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json",
    "schema": {
        "tone": "string",
        "focus_header": "string",
        "focus_summary": "string",
        "discussion_points": "array",
        "clinician_questions": "array",
        "formatting_hints": "object",
    },
}

VOICE_TRANSCRIPT_EXTRACTION_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json",
    "schema": {
        "medications": [
            {
                "name": "string",
                "dose": "string",
                "frequency": "string",
                "route": "string",
                "rxnorm_code": "string",
                "confidence": "number",
                "text_span": "string",
            }
        ],
        "symptoms": [
            {
                "name": "string",
                "severity": "string",
                "confidence": "number",
                "text_span": "string",
            }
        ],
        "conditions": [
            {
                "name": "string",
                "icd10_code": "string",
                "confidence": "number",
                "text_span": "string",
            }
        ],
        "instructions": [
            {
                "text": "string",
                "confidence": "number",
            }
        ],
        "notes": "string",
        "codes": {
            "rxnorm": ["string"],
            "icd10": ["string"],
        },
    },
}


def _inference_url() -> str:
    return (os.environ.get("LLM_INFERENCE_URL") or DEFAULT_LLM_INFERENCE_URL).strip().rstrip("/")


def _model_id() -> str:
    return (os.environ.get("LLM_MODEL_ID") or DEFAULT_LLM_MODEL_ID).strip()


def build_inference_request(
    *,
    task: str,
    input_payload: Dict[str, Any],
    steering_vectors: List[Dict[str, Any]],
    generation: Optional[Dict[str, Any]] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    JSON contract for the internal Llama 3.2 1B Instruct service with representation editing.

    The inference container is expected to:
      - keep base weights frozen
      - inject `steering[].vector` into the residual stream at `steering[].layer`
      - use `input` / `task` to build prompts server-side (PHI should already be minimized upstream)
    """
    gen = generation
    if gen is None:
        gen = {"temperature": 0.3, "max_tokens": 700}

    body: Dict[str, Any] = {
        "model": _model_id(),
        "personalization": {
            "method": "representation_editing",
            "llm": "llama-3.2-1b-instruct",
            "weights_frozen": True,
            "steering_injection": "residual_stream",
        },
        "task": task,
        "input": input_payload,
        "steering": steering_vectors,
        "generation": gen,
    }
    if response_format is not None:
        body["response_format"] = response_format
    return body


def _invoke_via_http(*, url: str, body: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    """Transport: the self-hosted HTTP inference service (`services/llm`).

    Sends the same JSON contract produced by `build_inference_request` to
    the local LLM inference service at `/v1/generate`.
    """
    resp = requests.post(f"{url}/v1/generate", json=body, timeout=timeout_seconds)
    resp.raise_for_status()
    return resp.json()


def call_llama_inference(
    *,
    task: str,
    input_payload: Dict[str, Any],
    steering_vectors: List[Dict[str, Any]],
    generation: Optional[Dict[str, Any]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 30,
) -> Optional[Dict[str, Any]]:
    url = _inference_url()
    if not url:
        logger.warning("LLM inference skipped: LLM_INFERENCE_URL is not configured")
        return None

    body = build_inference_request(
        task=task,
        input_payload=input_payload,
        steering_vectors=steering_vectors,
        generation=generation,
        response_format=response_format,
    )
    try:
        logger.info(
            "llm_inference request task=%s model=%s steering_vectors=%s response_format=%s transport=%s",
            task,
            body.get("model"),
            len(steering_vectors or []),
            bool(response_format),
            "http",
        )
        data = _invoke_via_http(url=url, body=body, timeout=timeout_seconds)
    except Exception:
        logger.exception(
            "llm_inference request failed task=%s transport=%s target=%s",
            task,
            "http",
            url,
        )
        return None

    if not isinstance(data, dict):
        return None

    out = data.get("output")
    if isinstance(out, dict):
        return out
    if isinstance(out, str):
        try:
            return json.loads(out)
        except Exception:
            return {"text": out}

    # Some gateways return the object at top level
    if task in ("pdf_guidance", "visit_one_pager") and (
        "tone" in data or "headline" in data or "focus_summary" in data
    ):
        return data
    if "text" in data or "content" in data:
        return data
    return data


def call_llama_visit_one_pager(
    *,
    deidentified_payload: Dict[str, Any],
    steering_vectors: List[Dict[str, Any]],
    view_mode: str = "standard",
    timeout_seconds: int = 35,
) -> Optional[Dict[str, Any]]:
    payload = {**deidentified_payload, "view_mode": view_mode}
    return call_llama_inference(
        task="visit_one_pager",
        input_payload=payload,
        steering_vectors=steering_vectors,
        generation={"temperature": 0.35, "max_tokens": 900},
        response_format=VISIT_ONE_PAGER_RESPONSE_FORMAT,
        timeout_seconds=timeout_seconds,
    )


def call_llama_pdf_guidance(
    *,
    deidentified_payload: Dict[str, Any],
    steering_vectors: List[Dict[str, Any]],
    timeout_seconds: int = 20,
) -> Optional[Dict[str, Any]]:
    return call_llama_inference(
        task="pdf_guidance",
        input_payload=deidentified_payload,
        steering_vectors=steering_vectors,
        generation={"temperature": 0.3, "max_tokens": 700},
        response_format=PDF_GUIDANCE_RESPONSE_FORMAT,
        timeout_seconds=timeout_seconds,
    )


def call_llama_transcript_extraction(
    *,
    transcript: str,
    steering_vectors: List[Dict[str, Any]],
    timeout_seconds: int = 45,
) -> Optional[Dict[str, Any]]:
    return call_llama_inference(
        task="voice_transcript_extraction",
        input_payload={"transcript": transcript},
        steering_vectors=steering_vectors,
        generation={"temperature": 0.2, "max_tokens": 900},
        response_format=VOICE_TRANSCRIPT_EXTRACTION_RESPONSE_FORMAT,
        timeout_seconds=timeout_seconds,
    )
