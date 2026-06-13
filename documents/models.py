from __future__ import annotations

from django.db import models


class ExportedPdf(models.Model):
    """
    Stored in the `documents` DB (RDS #3). No cross-db foreign keys.
    """

    id = models.BigAutoField(primary_key=True)
    user_id = models.BigIntegerField(db_index=True)
    appointment_id = models.BigIntegerField(db_index=True)

    filename = models.CharField(max_length=255)
    s3_bucket = models.CharField(max_length=255, blank=True, default="")
    s3_key = models.CharField(max_length=1024, blank=True, default="")
    storage = models.CharField(max_length=20, default="local")  # local|s3

    sha256 = models.CharField(max_length=64, blank=True, default="")
    size_bytes = models.BigIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "exported_pdfs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user_id", "appointment_id", "created_at"]),
        ]

