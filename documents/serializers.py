from __future__ import annotations

from rest_framework import serializers
from .models import ExportedPdf


class ExportedPdfSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExportedPdf
        fields = [
            "id",
            "user_id",
            "appointment_id",
            "filename",
            "sha256",
            "size_bytes",
            "created_at",
        ]

