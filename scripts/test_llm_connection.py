#!/usr/bin/env python
"""
Quick end-to-end sanity check for the Django -> Ollama LLM integration.

Verifies:
  1. Django can import and use `api.llm_client` (settings load correctly).
  2. Ollama is reachable at LLM_INFERENCE_URL (default http://127.0.0.1:11434).
  3. A raw `/api/generate` call returns a response.
  4. `call_llama_visit_one_pager()` returns a valid dict with the required
     fields populated (i.e. what `generate_visit_one_pager()` needs to set
     `source="llm"` instead of `source="fallback"`).

Usage (from the `backend/` directory, with the project virtualenv active):

    python scripts/test_llm_connection.py

Exit code is 0 only if every check passes.
"""
from __future__ import annotations

import os
import sys

# Make sure `api` is importable when run as `python scripts/test_llm_connection.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "previsit.settings_force_sqlite")

import django  # noqa: E402

django.setup()

import requests  # noqa: E402

from api.llm_client import (  # noqa: E402
    _inference_url,
    _model_id,
    call_llama_pdf_guidance,
    call_llama_visit_one_pager,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    print(f"[{status}] {label}" + (f" \u2014 {detail}" if detail else ""))
    return ok


def main() -> int:
    url = _inference_url()
    model = _model_id()
    print(f"LLM_INFERENCE_URL = {url}")
    print(f"LLM_MODEL_ID      = {model}")
    print("-" * 60)

    all_ok = True

    # 1. Raw connectivity check against Ollama's /api/tags
    try:
        tags_resp = requests.get(f"{url}/api/tags", timeout=5)
        tags_resp.raise_for_status()
        models = [m.get("name") for m in tags_resp.json().get("models", [])]
        model_present = any(model in (m or "") for m in models)
        all_ok &= check(
            "Ollama reachable at /api/tags",
            True,
            f"models available: {models}",
        )
        all_ok &= check(
            f"Model '{model}' is present in Ollama",
            model_present,
            "" if model_present else "run `ollama pull` / check LLM_MODEL_ID",
        )
    except Exception as exc:
        all_ok &= check("Ollama reachable at /api/tags", False, str(exc))
        print("\nOllama does not appear to be reachable. Aborting remaining checks.")
        return 1

    # 2. Raw /api/generate smoke test (mirrors the curl example in the runbook)
    try:
        gen_resp = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": "Return JSON with headline and focus_summary",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            },
            timeout=30,
        )
        gen_resp.raise_for_status()
        raw = gen_resp.json().get("response", "")
        all_ok &= check(
            "Ollama /api/generate raw smoke test",
            bool(raw),
            f"raw response (truncated): {raw[:200]!r}",
        )
    except Exception as exc:
        all_ok &= check("Ollama /api/generate raw smoke test", False, str(exc))

    # 3. Full pipeline: call_llama_visit_one_pager() via api/llm_client.py
    sample_payload = {
        "appointment": {"doctor_name": "Dr. Test", "specialty": "General"},
        "personalization_profile": {"main_reason": "Follow-up on fatigue and joint pain"},
        "symptoms": [{"name": "Fatigue", "severity": 6}],
        "questions": [{"text": "Could this be related to my medication?", "is_answered": False}],
        "llm_input_coverage": {"warnings": []},
    }
    one_pager = call_llama_visit_one_pager(
        deidentified_payload=sample_payload,
        steering_vectors=[],
        view_mode="standard",
        timeout_seconds=35,
    )
    source = "llm" if isinstance(one_pager, dict) and one_pager.get("headline") else "fallback"
    all_ok &= check(
        "call_llama_visit_one_pager() returns usable dict",
        source == "llm",
        f"source={source} output={one_pager}",
    )

    # 4. PDF guidance path (used by export-pdf's ai_guidance)
    guidance = call_llama_pdf_guidance(
        deidentified_payload=sample_payload,
        steering_vectors=[],
        timeout_seconds=20,
    )
    all_ok &= check(
        "call_llama_pdf_guidance() returns usable dict",
        isinstance(guidance, dict) and bool(guidance.get("focus_summary")),
        f"output={guidance}",
    )

    print("-" * 60)
    if all_ok:
        print("All checks passed \u2014 expect source=\"llm\" from /api/appointments/{id}/generate-one-pager")
    else:
        print("One or more checks FAILED \u2014 check Django logs for detailed llm_inference error lines")
        print("(look for: llm_inference http_error / connection_error / json_parse_error / validation_error)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
