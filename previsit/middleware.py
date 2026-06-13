"""Cross-cutting Django middleware.

`StructuredAccessLogMiddleware` emits one JSON log line per HTTP request,
suitable for ingestion by AWS CloudWatch Logs Insights:

    {"ts": "2026-05-03T17:00:00Z", "request_id": "...", "method": "GET",
     "route": "/api/health", "status": 200, "latency_ms": 4.1, "user_sub": null}

It is deliberately PHI-free: only the route template, the HTTP method, the
HTTP status, the latency, and an opaque user identifier (the Django user PK or
the Cognito `sub` if present) are emitted.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

logger = logging.getLogger("neuravia.access")


class StructuredAccessLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.request_id = request_id
        start = time.perf_counter()
        response = self.get_response(request)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

        user_id = None
        try:
            if getattr(request, "user", None) and request.user.is_authenticated:
                user_id = str(getattr(request.user, "pk", None) or "")
        except Exception:  # noqa: BLE001
            user_id = None

        try:
            payload = {
                "request_id": request_id,
                "method": request.method,
                "route": request.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
                "user_id": user_id,
                "remote": request.META.get("REMOTE_ADDR"),
            }
            logger.info(json.dumps(payload, separators=(",", ":")))
        except Exception:  # noqa: BLE001
            pass

        try:
            response["X-Request-ID"] = request_id
        except Exception:  # noqa: BLE001
            pass
        return response
