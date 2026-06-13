"""User-targeted push dispatch helpers."""

from __future__ import annotations

from typing import List

from .models import PushDevice


def _tokens_for_user(user_id: int, pref_key: str) -> List[str]:
    tokens: List[str] = []
    for device in PushDevice.objects.filter(user_id=user_id, is_active=True):
        prefs = device.notification_prefs if isinstance(device.notification_prefs, dict) else {}
        if prefs.get(pref_key, True):
            tokens.append(device.expo_push_token)
    return tokens


def push_summary_ready_for_user(user_id: int, appointment_id: int) -> int:
    """Send PHI-free summary-ready push if user opted in. Returns sent count."""
    from .expo_push import send_summary_ready

    tokens = _tokens_for_user(user_id, "summary_reminders")
    if not tokens:
        return 0
    result = send_summary_ready(tokens, appointment_id=appointment_id)
    if result.invalid_tokens:
        PushDevice.objects.filter(expo_push_token__in=result.invalid_tokens).update(is_active=False)
    return result.sent
