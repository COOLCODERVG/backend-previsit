from __future__ import annotations

from django.conf import settings
from django.db import models


class PushDevice(models.Model):
    """An Expo Push token registered by a single device.

    The token is the only PII-adjacent value here; we never store the device
    name or geo. Tokens are scoped to the owning user and are removed on
    logout / uninstall (the Expo Push API surfaces ``DeviceNotRegistered``
    receipts that the dispatcher uses to prune stale rows).
    """

    PLATFORM_CHOICES = [
        ("ios", "iOS"),
        ("android", "Android"),
        ("web", "Web"),
        ("unknown", "Unknown"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="push_devices",
    )
    expo_push_token = models.CharField(max_length=255, unique=True)
    platform = models.CharField(max_length=12, choices=PLATFORM_CHOICES, default="unknown")
    device_id = models.CharField(max_length=128, blank=True, default="")
    is_active = models.BooleanField(default=True)
    notification_prefs = models.JSONField(default=dict, blank=True)
    last_seen_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "push_devices"
        indexes = [
            models.Index(fields=["user", "is_active"]),
        ]
