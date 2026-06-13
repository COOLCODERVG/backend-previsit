"""Expo Push API client (HIPAA-safe wrapper).

Hard rule: the message body and title must never contain PHI. The mobile app
fetches the actual medication name *after* the user opens the app over an
authenticated session.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
DEFAULT_TITLE = "NeuraVia"
DEFAULT_BODY = "Time for your medication"
DEFAULT_DEEP_LINK = "neuraviapre:///medications"
SUMMARY_BODY = "Your visit summary is ready to review."
SUMMARY_DEEP_LINK_TEMPLATE = "neuraviapre:///appointment/{appointment_id}/summary"


@dataclass
class PushResult:
    sent: int
    invalid_tokens: List[str]
    raw: dict


def _headers() -> dict:
    h = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "Content-Type": "application/json",
    }
    token = (os.environ.get("EXPO_ACCESS_TOKEN") or "").strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def send_medication_reminders(
    expo_tokens: Iterable[str],
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    deep_link: Optional[str] = None,
    timeout: float = 10.0,
) -> PushResult:
    tokens = [t for t in {t.strip() for t in expo_tokens if t} if t.startswith("ExponentPushToken[")]
    if not tokens:
        return PushResult(sent=0, invalid_tokens=[], raw={})

    payload = [
        {
            "to": t,
            "title": title or DEFAULT_TITLE,
            "body": body or DEFAULT_BODY,
            "sound": "default",
            "priority": "high",
            "channelId": "medication-reminders",
            "data": {"deeplink": deep_link or DEFAULT_DEEP_LINK, "kind": "medication_reminder"},
        }
        for t in tokens
    ]

    try:
        resp = requests.post(EXPO_PUSH_URL, json=payload, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Expo push failed: %s", exc)
        return PushResult(sent=0, invalid_tokens=[], raw={"error": str(exc)})

    invalid: List[str] = []
    receipts = data.get("data") if isinstance(data, dict) else None
    if isinstance(receipts, list):
        for token, item in zip(tokens, receipts):
            if not isinstance(item, dict):
                continue
            if item.get("status") == "error":
                err = (item.get("details") or {}).get("error", "")
                if err in {"DeviceNotRegistered", "InvalidCredentials"}:
                    invalid.append(token)
    sent = max(0, len(tokens) - len(invalid))
    return PushResult(sent=sent, invalid_tokens=invalid, raw=data if isinstance(data, dict) else {})


def send_summary_ready(
    expo_tokens: Iterable[str],
    *,
    appointment_id: int,
    title: Optional[str] = None,
    body: Optional[str] = None,
    timeout: float = 10.0,
) -> PushResult:
    """PHI-free push when transcription / summary is ready."""
    tokens = [t for t in {t.strip() for t in expo_tokens if t} if t.startswith("ExponentPushToken[")]
    if not tokens:
        return PushResult(sent=0, invalid_tokens=[], raw={})

    deep_link = SUMMARY_DEEP_LINK_TEMPLATE.format(appointment_id=appointment_id)
    payload = [
        {
            "to": t,
            "title": title or DEFAULT_TITLE,
            "body": body or SUMMARY_BODY,
            "sound": "default",
            "priority": "high",
            "data": {
                "deeplink": deep_link,
                "kind": "summary_ready",
                "appointmentId": appointment_id,
            },
        }
        for t in tokens
    ]

    try:
        resp = requests.post(EXPO_PUSH_URL, json=payload, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Expo summary push failed: %s", exc)
        return PushResult(sent=0, invalid_tokens=[], raw={"error": str(exc)})

    invalid: List[str] = []
    receipts = data.get("data") if isinstance(data, dict) else None
    if isinstance(receipts, list):
        for token, item in zip(tokens, receipts):
            if not isinstance(item, dict):
                continue
            if item.get("status") == "error":
                err = (item.get("details") or {}).get("error", "")
                if err in {"DeviceNotRegistered", "InvalidCredentials"}:
                    invalid.append(token)
    sent = max(0, len(tokens) - len(invalid))
    return PushResult(sent=sent, invalid_tokens=invalid, raw=data if isinstance(data, dict) else {})
