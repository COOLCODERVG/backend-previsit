from __future__ import annotations

import os
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import ExportedPdf
from .serializers import ExportedPdfSerializer

try:
    from api.s3 import presign_get_docs
except Exception:  # pragma: no cover
    presign_get_docs = None


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_exports(request):
    appointment_id = request.query_params.get("appointment_id")
    qs = ExportedPdf.objects.using("documents").filter(user_id=request.user.id)
    if appointment_id:
        qs = qs.filter(appointment_id=int(appointment_id))
    data = ExportedPdfSerializer(qs[:50], many=True).data
    return Response(data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def export_download_url(request, pk: int):
    try:
        item = ExportedPdf.objects.using("documents").get(pk=pk, user_id=request.user.id)
    except ExportedPdf.DoesNotExist:
        return Response({"detail": "Export not found"}, status=404)

    if item.storage == "s3":
        if not presign_get_docs:
            return Response({"detail": "S3 helper unavailable"}, status=500)
        url = presign_get_docs(key=item.s3_key, expires_seconds=900)
        return Response({"download_url": url, "filename": item.filename})

    # Local dev fallback (same folder used by api.utils.persist_export_pdf)
    from pathlib import Path
    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "exports" / item.s3_key
    if not file_path.exists():
        return Response({"detail": "Local export file not found"}, status=404)
    return Response({"download_path": str(file_path), "filename": item.filename})

