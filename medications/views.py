from __future__ import annotations

from rest_framework.decorators import api_view
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
from rest_framework.response import Response

from .models import Medication, MedicationEvent, MedicationReminder
from .serializers import (
    MedicationSerializer,
    MedicationCreateSerializer,
    MedicationReminderSerializer,
    MedicationReminderCreateSerializer,
)


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def medications_view(request):
    if request.method == "GET":
        qs = Medication.objects.filter(user=request.user)
        status_q = request.query_params.get("status")
        if status_q:
            qs = qs.filter(status=status_q)
        return Response(MedicationSerializer(qs, many=True).data)

    s = MedicationCreateSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    d = s.validated_data
    med = Medication.objects.create(
        user=request.user,
        name=d["name"],
        rxnorm_code=(d.get("rxnorm_code") or "").strip(),
        dose=(d.get("dose") or "").strip(),
        frequency=(d.get("frequency") or "").strip(),
        route=(d.get("route") or "").strip(),
        start_date=(d.get("start_date") or "").strip(),
        end_date=(d.get("end_date") or "").strip(),
        status=d.get("status") or "active",
        source="manual",
        verification="unverified",
        confidence=0.0,
    )
    MedicationEvent.objects.create(medication=med, event_type="created", payload={"source": "manual"})
    return Response(MedicationSerializer(med).data, status=201)


@api_view(["PUT", "DELETE"])
@permission_classes([IsAuthenticated])
def medication_detail_view(request, pk: int):
    try:
        med = Medication.objects.get(pk=pk, user=request.user)
    except Medication.DoesNotExist:
        return Response({"detail": "Medication not found"}, status=404)

    if request.method == "PUT":
        changed = {}
        for field in ["name", "rxnorm_code", "dose", "frequency", "route", "start_date", "end_date", "status", "verification"]:
            if field in request.data:
                val = request.data[field]
                setattr(med, field, val)
                changed[field] = val
        if changed:
            if "status" in changed and changed["status"] == "discontinued":
                MedicationEvent.objects.create(medication=med, event_type="discontinued", payload=changed)
            else:
                MedicationEvent.objects.create(medication=med, event_type="modified", payload=changed)
            med.save()
        return Response(MedicationSerializer(med).data)

    med.delete()
    return Response({"message": "Medication deleted successfully"})


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def medication_reminders_view(request):
    if request.method == "GET":
        qs = MedicationReminder.objects.filter(user=request.user).select_related("medication")
        return Response(MedicationReminderSerializer(qs, many=True).data)

    s = MedicationReminderCreateSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    d = s.validated_data
    try:
        med = Medication.objects.get(pk=d["medication_id"], user=request.user)
    except Medication.DoesNotExist:
        return Response({"detail": "Medication not found"}, status=404)

    reminder = MedicationReminder.objects.create(
        medication=med,
        user=request.user,
        time_of_day=d["time_of_day"],
        timezone=(d.get("timezone") or "UTC").strip() or "UTC",
        days_of_week=d.get("days_of_week") or [],
        is_active=bool(d.get("is_active", True)),
    )
    return Response(MedicationReminderSerializer(reminder).data, status=201)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def medication_reminder_detail_view(request, pk: int):
    try:
        reminder = MedicationReminder.objects.get(pk=pk, user=request.user)
    except MedicationReminder.DoesNotExist:
        return Response({"detail": "Reminder not found"}, status=404)
    reminder.delete()
    return Response({"message": "Reminder deleted successfully"})

