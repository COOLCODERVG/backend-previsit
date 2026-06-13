from __future__ import annotations

from rest_framework import serializers

from .models import Medication, MedicationReminder


class MedicationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Medication
        fields = [
            "id",
            "name",
            "rxnorm_code",
            "dose",
            "frequency",
            "route",
            "start_date",
            "end_date",
            "status",
            "source",
            "verification",
            "confidence",
            "created_at",
            "updated_at",
        ]


class MedicationCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    rxnorm_code = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    dose = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    frequency = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    route = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    start_date = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    end_date = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    status = serializers.ChoiceField(choices=[c[0] for c in Medication.STATUS_CHOICES], required=False)


class MedicationReminderSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationReminder
        fields = ["id", "medication_id", "time_of_day", "timezone", "days_of_week", "is_active", "created_at", "updated_at"]


class MedicationReminderCreateSerializer(serializers.Serializer):
    medication_id = serializers.IntegerField()
    time_of_day = serializers.CharField(max_length=5)
    timezone = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    days_of_week = serializers.ListField(child=serializers.CharField(), required=False)
    is_active = serializers.BooleanField(default=True)

