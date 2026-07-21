from __future__ import annotations

from django.conf import settings
from django.db import models


class ExportedPdf(models.Model):
    """
    Exported visit-summary PDF metadata, stored as a table in the single
    unified application database (alongside every other model).
    """

    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="exported_pdfs",
        db_index=True,
    )
    appointment = models.ForeignKey(
        "api.Appointment",
        on_delete=models.CASCADE,
        related_name="exported_pdfs",
        db_index=True,
    )

    filename = models.CharField(max_length=255)

    sha256 = models.CharField(max_length=64, blank=True, default="")
    size_bytes = models.BigIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "exported_pdfs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "appointment", "created_at"]),
        ]

