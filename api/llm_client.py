from __future__ import annotations

import json
import logging
import os
import re
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

VISIT_SUMMARY_RESPONSE_FORMAT: Dict[str, Any] = {
    "type": "json",
    "schema": {
        "summary": "string",
        "action_items": ["string"],
        "medication_changes": [
            {
                "name": "string",
                "dosage": "string",
                "frequency": "string",
                "duration": "string",
                "change_type": "string",
            }
        ],
        "tests_ordered": ["string"],
        "follow_ups": ["string"],
        "upcoming_appointments": ["string"],
        "doctor_instructions": ["string"],
        "lifestyle_recommendations": ["string"],
        "warnings": ["string"],
        "questions_for_next_visit": ["string"],
    },
}

# Minimal required-field sets used to validate that the model actually returned
# a usable object for the schema, rather than an empty/partial/garbled blob.
# Keyed by `task` (matches the `task` argument passed to `call_llama_inference`).
REQUIRED_FIELDS_BY_TASK: Dict[str, List[str]] = {
    "visit_one_pager": ["headline", "focus_summary"],
    "pdf_guidance": ["focus_summary"],
    "voice_transcript_extraction": [],
    "visit_summary": ["summary"],
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


def _serialize_prompt_for_ollama(body: Dict[str, Any], *, correction_note: Optional[str] = None) -> str:
    """Convert the existing structured `build_inference_request()` body into a
    single text prompt that a plain-text completion API (Ollama `/api/generate`)
    can consume, while preserving the same task/input/steering/schema contract.

    When `correction_note` is provided (used on the one automatic retry after an
    invalid/unparseable first response), the prompt is amended with an explicit
    instruction to fix the previous mistake.
    """
    task = body.get("task") or "general"
    input_payload = body.get("input") or {}
    steering_vectors = body.get("steering") or []
    response_format = body.get("response_format")

    lines: List[str] = [
        "You are the SyniVia clinical visit-prep assistant.",
        f"Task: {task}",
    ]

    def _truncate_long_strings(obj: Any, max_len: int = 2000) -> Any:
        """Recursively truncate long string values in nested dicts/lists to keep
        Ollama prompts CPU-friendly and avoid sending excessive payloads.
        """
        if isinstance(obj, str):
            if len(obj) > max_len:
                return obj[:max_len].rstrip() + '...'
            return obj
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # Truncate known large fields aggressively
                if k in ("transcript", "transcript_dump", "voice_transcript", "combined_text") and isinstance(v, str):
                    out[k] = v[:max_len].rstrip() + ("..." if len(v) > max_len else "")
                else:
                    out[k] = _truncate_long_strings(v, max_len=max_len)
            return out
        if isinstance(obj, list):
            return [_truncate_long_strings(x, max_len=max_len) for x in obj]
        return obj

    if steering_vectors:
        lines.append(
            f"(Personalization context: {len(steering_vectors)} steering vector(s) "
            "supplied by the upstream personalization system; reflect their guidance "
            "implicitly in tone and emphasis.)"
        )

    # Truncate huge fields to keep the prompt small for CPU inference.
    try:
        safe_input = _truncate_long_strings(input_payload, max_len=2000)
        lines.append("Input data (JSON):")
        lines.append(json.dumps(safe_input, ensure_ascii=False, default=str))
    except Exception:
        lines.append(str(input_payload))

    if isinstance(response_format, dict) and response_format.get("schema"):
        schema = response_format["schema"]
        lines.append(
            "STRICT OUTPUT RULES:\n"
            "1. Output ONLY a single valid JSON object. Nothing else.\n"
            "2. Do NOT use markdown, code fences, backticks, or any commentary.\n"
            "3. Do NOT explain your reasoning before or after the JSON.\n"
            "4. Do NOT include any text outside the { } braces.\n"
            "5. Follow this schema EXACTLY (same keys, same value types). Use empty "
            'strings/arrays for anything unknown — never omit a key:\n'
            f"{json.dumps(schema, ensure_ascii=False)}\n"
            "Your entire response must start with '{' and end with '}'."
        )
    else:
        lines.append("Respond with a concise, helpful answer.")

    if correction_note:
        lines.append(
            "IMPORTANT CORRECTION: Your previous response was invalid "
            f"({correction_note}). Re-read the STRICT OUTPUT RULES above and "
            "respond again with ONLY the corrected JSON object."
        )

    return "\n\n".join(lines)


def _parse_llm_json_output(text: Optional[str]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Best-effort parse of a model's raw text response into a JSON dict.

    Handles: plain JSON, JSON wrapped in ```json ... ``` code fences, and JSON
    embedded within surrounding prose (extracts the first balanced {...} block).

    Returns a `(parsed, failure_reason)` tuple. `failure_reason` is `None` on
    success, otherwise a short human-readable string describing why parsing
    failed (the caller is responsible for logging it alongside the raw text —
    this function never logs directly since it has no task/endpoint context).
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response"

    cleaned = text.strip()

    fence_match = re.search(r"```[a-zA-Z]*\s*\n?(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, None
        return None, "parsed_json_not_object"
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
            return None, "embedded_json_not_object"
        except Exception as exc:
            return None, f"embedded_json_decode_error:{exc}"

    return None, "no_json_found_in_response"


def _invoke_via_http(
    *,
    url: str,
    body: Dict[str, Any],
    timeout_seconds: int,
    correction_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Transport: local Ollama server running the `synivia-model` model.

    Ollama exposes a plain-text completion API at `POST /api/generate`
    (`{"model", "prompt", "stream": false}` -> `{"response": "..."}`), which is
    different from the original custom `/v1/generate` JSON-in/JSON-out contract
    `build_inference_request()` was designed for. We preserve
    `build_inference_request()`'s structured body unchanged and simply
    serialize it into a single prompt here, then parse Ollama's text response
    back into a dict so the rest of the pipeline (schema-shaped dicts) is
    unaffected.

    Also forwards `generation.temperature` / `generation.max_tokens` as Ollama's
    `options.temperature` / `options.num_predict`, and sets `"format": "json"` so
    Ollama constrains its output to valid JSON at the sampling level (in addition
    to the strict-JSON prompt instructions).
    """
    prompt = _serialize_prompt_for_ollama(body, correction_note=correction_note)
    generation = body.get("generation") if isinstance(body.get("generation"), dict) else {}
    temperature = generation.get("temperature", 0.1)
    max_tokens = generation.get("max_tokens")

    options: Dict[str, Any] = {"temperature": temperature}
    if isinstance(max_tokens, (int, float)) and max_tokens > 0:
        options["num_predict"] = int(max_tokens)

    ollama_payload: Dict[str, Any] = {
        "model": body.get("model") or _model_id(),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": options,
    }

    resp = requests.post(f"{url}/api/generate", json=ollama_payload, timeout=timeout_seconds)
    resp.raise_for_status()
    data = resp.json()

    raw_text = data.get("response") if isinstance(data, dict) else None
    parsed, parse_failure_reason = _parse_llm_json_output(raw_text)

    return {
        "output": parsed,
        "raw_text": raw_text,
        "parse_failure_reason": parse_failure_reason,
    }


def _validate_required_fields(task: str, parsed: Optional[Dict[str, Any]]) -> Optional[str]:
    """Returns None if `parsed` satisfies the required fields for `task`,
    otherwise a short description of what's missing/invalid.
    """
    if not isinstance(parsed, dict):
        return "output_not_a_dict"

    required = REQUIRED_FIELDS_BY_TASK.get(task, [])
    missing = []
    for field in required:
        val = parsed.get(field, None)
        if val is None:
            missing.append(field)
            continue
        # Treat empty strings/empty lists as missing
        if isinstance(val, str) and not val.strip():
            missing.append(field)
            continue
        if isinstance(val, (list, dict)) and len(val) == 0:
            missing.append(field)
            continue

    if missing:
        return f"missing_or_empty_fields:{','.join(missing)}"
    return None


def _attempt_llm_call(
    *,
    url: str,
    body: Dict[str, Any],
    task: str,
    timeout_seconds: int,
    correction_note: Optional[str] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Single HTTP round-trip to Ollama. Returns `(parsed_output, failure_reason)`.

    `failure_reason` is None only when a dict was successfully parsed AND it
    passes required-field validation for `task`. All failure paths are logged
    here with full context (endpoint, task, model, status code / error, and
    raw response text where available) so nothing fails silently.
    """
    endpoint = f"{url}/api/generate"
    try:
        data = _invoke_via_http(
            url=url,
            body=body,
            timeout_seconds=timeout_seconds,
            correction_note=correction_note,
        )
    except requests.exceptions.Timeout as exc:
        logger.error(
            "llm_inference timeout task=%s model=%s endpoint=%s timeout_seconds=%s error=%s",
            task, body.get("model"), endpoint, timeout_seconds, str(exc),
        )
        return None, f"timeout:{exc}"
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        response_text = None
        if exc.response is not None:
            try:
                response_text = exc.response.text[:2000]
            except Exception:
                response_text = None
        logger.error(
            "llm_inference http_error task=%s model=%s endpoint=%s status_code=%s response_text=%s",
            task, body.get("model"), endpoint, status_code, response_text,
        )
        return None, f"http_error:{status_code}"
    except requests.exceptions.RequestException as exc:
        logger.error(
            "llm_inference connection_error task=%s model=%s endpoint=%s error=%s",
            task, body.get("model"), endpoint, str(exc),
        )
        return None, f"connection_error:{exc}"
    except Exception:
        logger.exception(
            "llm_inference unexpected_error task=%s model=%s endpoint=%s",
            task, body.get("model"), endpoint,
        )
        return None, "unexpected_error"

    parsed = data.get("output")
    parse_failure_reason = data.get("parse_failure_reason")
    raw_text = data.get("raw_text")

    if parse_failure_reason:
        logger.error(
            "llm_inference json_parse_error task=%s model=%s endpoint=%s reason=%s raw_response=%s",
            task, body.get("model"), endpoint, parse_failure_reason,
            (raw_text or "")[:2000],
        )
        return None, f"json_parse_error:{parse_failure_reason}"

    validation_failure = _validate_required_fields(task, parsed)
    if validation_failure:
        logger.error(
            "llm_inference validation_error task=%s model=%s endpoint=%s reason=%s raw_response=%s",
            task, body.get("model"), endpoint, validation_failure,
            (raw_text or "")[:2000],
        )
        return None, f"validation_error:{validation_failure}"

    return parsed, None


def call_llama_inference(
    *,
    task: str,
    input_payload: Dict[str, Any],
    steering_vectors: List[Dict[str, Any]],
    generation: Optional[Dict[str, Any]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 60,
) -> Optional[Dict[str, Any]]:
    """Calls the local Ollama model, validates the parsed JSON output against
    the task's required fields, and retries ONCE with an explicit correction
    prompt if the first attempt fails (bad HTTP, unparseable JSON, or missing
    required fields). If the retry also fails, returns None and the caller
    (e.g. `generate_visit_one_pager`) falls back to templated content — but
    every failure is logged loudly with full context, so this is never silent.
    """
    url = _inference_url()
    if not url:
        logger.warning("llm_inference skipped task=%s reason=LLM_INFERENCE_URL_not_configured", task)
        return None

    body = build_inference_request(
        task=task,
        input_payload=input_payload,
        steering_vectors=steering_vectors,
        generation=generation,
        response_format=response_format,
    )

    logger.info(
        "llm_inference request task=%s model=%s steering_vectors=%s response_format=%s transport=http endpoint=%s",
        task,
        body.get("model"),
        len(steering_vectors or []),
        bool(response_format),
        f"{url}/api/generate",
    )

    parsed, failure_reason = _attempt_llm_call(
        url=url, body=body, task=task, timeout_seconds=timeout_seconds,
    )

    if parsed is not None:
        return parsed

    logger.warning(
        "llm_inference first_attempt_failed task=%s model=%s reason=%s — retrying once with correction prompt",
        task, body.get("model"), failure_reason,
    )

    parsed, retry_failure_reason = _attempt_llm_call(
        url=url,
        body=body,
        task=task,
        timeout_seconds=timeout_seconds,
        correction_note=failure_reason or "invalid_or_missing_json",
    )

    if parsed is not None:
        logger.info("llm_inference retry_succeeded task=%s model=%s", task, body.get("model"))
        return parsed

    logger.error(
        "llm_inference giving_up task=%s model=%s first_failure=%s retry_failure=%s — falling back to templated content",
        task, body.get("model"), failure_reason, retry_failure_reason,
    )
    return None


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
        generation={"temperature": 0.1, "max_tokens": 900},
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
        generation={"temperature": 0.1, "max_tokens": 700},
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
        generation={"temperature": 0.1, "max_tokens": 900},
        response_format=VOICE_TRANSCRIPT_EXTRACTION_RESPONSE_FORMAT,
        timeout_seconds=timeout_seconds,
    )


def call_llama_visit_summary(
    *,
    transcript: str,
    steering_vectors: Optional[List[Dict[str, Any]]] = None,
    timeout_seconds: int = 45,
) -> Optional[Dict[str, Any]]:
    """Generate a structured post-visit summary from a raw visit transcript.

    Unlike `call_llama_transcript_extraction` (which pulls out clinical
    entities to merge into Symptom/Note rows), this produces a
    patient-facing follow-up brief: a plain-language summary plus explicit
    action items, medication changes, ordered tests, follow-ups/deadlines,
    doctor instructions, lifestyle recommendations, warnings, and questions
    to bring to the next visit.
    """
    return call_llama_inference(
        task="visit_summary",
        input_payload={"transcript": transcript},
        steering_vectors=steering_vectors or [],
        generation={"temperature": 0.1, "max_tokens": 1100},
        response_format=VISIT_SUMMARY_RESPONSE_FORMAT,
        timeout_seconds=timeout_seconds,
    )
