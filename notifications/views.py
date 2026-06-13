from __future__ import annotations

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import PushDevice

DEFAULT_NOTIFICATION_PREFS = {
    "medication_reminders": True,
    "appointment_reminders": True,
    "record_reminders": True,
    "summary_reminders": True,
}


def _merge_notification_prefs(existing: dict | None, incoming: dict | None) -> dict:
    base = {**DEFAULT_NOTIFICATION_PREFS, **(existing or {})}
    if not isinstance(incoming, dict):
        return base
    for key, default in DEFAULT_NOTIFICATION_PREFS.items():
        if key in incoming:
            base[key] = bool(incoming[key])
    return base


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def push_devices_view(request):
    """Register an Expo push token for the current user.

    POST body: {"expo_push_token": "ExponentPushToken[...]", "platform": "ios|android|web", "device_id": "..."}
    The same token can be POSTed repeatedly; we upsert on `expo_push_token`.
    """
    if request.method == "GET":
        qs = PushDevice.objects.filter(user=request.user, is_active=True).order_by("-last_seen_at")
        return Response([
            {
                "id": d.id,
                "expo_push_token": d.expo_push_token,
                "platform": d.platform,
                "device_id": d.device_id,
                "last_seen_at": d.last_seen_at,
                "notification_prefs": _merge_notification_prefs(d.notification_prefs, None),
            }
            for d in qs
        ])

    body = request.data if isinstance(request.data, dict) else {}
    token = (body.get("expo_push_token") or "").strip()
    if not token or not token.startswith("ExponentPushToken["):
        return Response({"detail": "Invalid expo_push_token"}, status=400)

    platform = (body.get("platform") or "unknown").strip().lower()
    if platform not in {"ios", "android", "web", "unknown"}:
        platform = "unknown"
    device_id = (body.get("device_id") or "").strip()[:128]
    incoming_prefs = body.get("notification_prefs")
    try:
        existing = PushDevice.objects.get(expo_push_token=token)
        merged_prefs = _merge_notification_prefs(existing.notification_prefs, incoming_prefs)
    except PushDevice.DoesNotExist:
        merged_prefs = _merge_notification_prefs({}, incoming_prefs)

    device, _created = PushDevice.objects.update_or_create(
        expo_push_token=token,
        defaults={
            "user": request.user,
            "platform": platform,
            "device_id": device_id,
            "is_active": True,
            "notification_prefs": merged_prefs,
        },
    )
    return Response({
        "id": device.id,
        "expo_push_token": device.expo_push_token,
        "platform": device.platform,
        "is_active": device.is_active,
        "notification_prefs": device.notification_prefs,
    }, status=201)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def push_device_detail_view(request, token: str):
    qs = PushDevice.objects.filter(user=request.user, expo_push_token=token)
    if not qs.exists():
        return Response({"detail": "Token not found"}, status=404)
    qs.update(is_active=False)
    return Response({"message": "Token deactivated"})
