from __future__ import annotations

from django.db import models
from django.conf import settings


class Medication(models.Model):
    STATUS_CHOICES = [
        ("active", "Active"),
        ("modified", "Modified"),
        ("discontinued", "Discontinued"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="medications")
    name = models.CharField(max_length=255)
    rxnorm_code = models.CharField(max_length=50, blank=True, default="")
    dose = models.CharField(max_length=120, blank=True, default="")
    frequency = models.CharField(max_length=120, blank=True, default="")  # e.g. "2x/day", "q8h"
    route = models.CharField(max_length=80, blank=True, default="")  # oral, iv, etc
    start_date = models.CharField(max_length=10, blank=True, default="")  # YYYY-MM-DD
    end_date = models.CharField(max_length=10, blank=True, default="")  # YYYY-MM-DD
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")

    source = models.CharField(max_length=30, default="manual")  # manual|voice_extract|import
    verification = models.CharField(max_length=30, default="unverified")  # unverified|verified|rejected
    confidence = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "medications"
        ordering = ["-updated_at"]


class MedicationEvent(models.Model):
    EVENT_CHOICES = [
        ("created", "Created"),
        ("modified", "Modified"),
        ("discontinued", "Discontinued"),
        ("verified", "Verified"),
    ]

    medication = models.ForeignKey(Medication, on_delete=models.CASCADE, related_name="events")
    event_type = models.CharField(max_length=20, choices=EVENT_CHOICES)
    payload = models.JSONField(default=dict, blank=True)
    occurred_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "medication_events"
        ordering = ["-occurred_at"]


class MedicationReminder(models.Model):
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE, related_name="reminders")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="medication_reminders")

    time_of_day = models.CharField(max_length=5)  # HH:MM (local)
    timezone = models.CharField(max_length=64, default="UTC")
    days_of_week = models.JSONField(default=list, blank=True)  # ["mon","tue",...]
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "medication_reminders"
        ordering = ["time_of_day"]

