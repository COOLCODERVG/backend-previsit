from __future__ import annotations

"""
S3 access for NeuraVia.

Single module layout (by concern):
  1) Low-level: boto client, SSE defaults, put/get, presign, s3:// parsing
  2) Audio bucket (recordings, presigned PUT/GET)
  3) Docs bucket (exported PDFs)

You still use *multiple S3 buckets* in AWS (audio / transcripts / docs); this file is only
the Python packaging — one import surface instead of s3_core + s3 + s3_docs.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import boto3

# ---------------------------------------------------------------------------
# Low-level
# ---------------------------------------------------------------------------


def client():
    region = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip() or None
    return boto3.client("s3", region_name=region)


def sse_put_params() -> Dict[str, Any]:
    """
    Server-side encryption params for PutObject.

    Env:
      - AWS_S3_SSE: unset|AES256|aws:kms (default: aws:kms if AWS_S3_KMS_KEY_ID set else AES256)
      - AWS_S3_KMS_KEY_ID: optional KMS key id/arn for SSE-KMS
    """
    sse = (os.environ.get("AWS_S3_SSE") or "").strip().lower()
    kms_key = (os.environ.get("AWS_S3_KMS_KEY_ID") or "").strip()
    if not sse:
        sse = "aws:kms" if kms_key else "AES256"

    params: Dict[str, Any] = {"ServerSideEncryption": sse}
    if sse == "aws:kms" and kms_key:
        params["SSEKMSKeyId"] = kms_key
    return params


def put_bytes(*, bucket: str, key: str, body: bytes, content_type: str) -> Tuple[str, int]:
    s3 = client()
    extra = sse_put_params()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type, **extra)
    digest = hashlib.sha256(body).hexdigest()
    return digest, len(body)


def get_bytes(*, bucket: str, key: str) -> bytes:
    s3 = client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def presign_put(*, bucket: str, key: str, content_type: str, expires_seconds: int = 900) -> Tuple[str, Dict[str, str]]:
    s3 = client()
    params: Dict[str, Any] = {"Bucket": bucket, "Key": key, "ContentType": content_type}
    params.update(sse_put_params())
    url = s3.generate_presigned_url("put_object", Params=params, ExpiresIn=expires_seconds)

    headers: Dict[str, str] = {"Content-Type": content_type}
    sse = params.get("ServerSideEncryption")
    if sse:
        headers["x-amz-server-side-encryption"] = str(sse)
    kms_key = params.get("SSEKMSKeyId")
    if kms_key:
        headers["x-amz-server-side-encryption-aws-kms-key-id"] = str(kms_key)
    return url, headers


def presign_get(*, bucket: str, key: str, expires_seconds: int = 900) -> str:
    s3 = client()
    return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_seconds)


def delete_object(*, bucket: str, key: str) -> None:
    """Permanently delete a single object. Used by the recording-retention purge."""
    s3 = client()
    s3.delete_object(Bucket=bucket, Key=key)


def parse_s3_uri(uri: str) -> Optional[Tuple[str, str]]:
    if not uri or not uri.startswith("s3://"):
        return None
    rest = uri[len("s3://") :]
    if "/" not in rest:
        return None
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        return None
    return bucket, key


# ---------------------------------------------------------------------------
# Audio bucket
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PresignedPut:
    upload_url: str
    headers: Dict[str, str]


def audio_bucket() -> str:
    bucket = os.environ.get("S3_AUDIO_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("S3_AUDIO_BUCKET is not configured")
    return bucket


def presign_put_audio(*, object_key: str, content_type: str, expires_seconds: int = 900) -> PresignedPut:
    bucket = audio_bucket()
    url, headers = presign_put(bucket=bucket, key=object_key, content_type=content_type, expires_seconds=expires_seconds)
    return PresignedPut(upload_url=url, headers=headers)


def presign_get_audio(*, object_key: str, expires_seconds: int = 900) -> str:
    return presign_get(bucket=audio_bucket(), key=object_key, expires_seconds=expires_seconds)


def delete_audio_object(*, object_key: str) -> None:
    """Permanently delete an expired recording's audio object from the audio bucket."""
    delete_object(bucket=audio_bucket(), key=object_key)


def list_audio_objects(*, prefix: str, max_items: int = 200) -> list[Dict[str, Any]]:
    """
    List objects in the audio bucket under the given key prefix (e.g. ``audio/u42/``).
    Returns lightweight metadata; presigned URLs are generated separately on demand.
    """
    s3 = client()
    paginator = s3.get_paginator("list_objects_v2")
    out: list[Dict[str, Any]] = []
    remaining = max_items
    for page in paginator.paginate(Bucket=audio_bucket(), Prefix=prefix, PaginationConfig={"PageSize": min(max_items, 1000)}):
        for obj in page.get("Contents") or []:
            if remaining <= 0:
                return out
            out.append({
                "object_key": obj.get("Key"),
                "size_bytes": int(obj.get("Size") or 0),
                "etag": (obj.get("ETag") or "").strip('"'),
                "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
                "storage_class": obj.get("StorageClass"),
            })
            remaining -= 1
    return out


# ---------------------------------------------------------------------------
# Docs bucket (PDF exports, etc.)
# ---------------------------------------------------------------------------


def docs_bucket() -> str:
    bucket = os.environ.get("S3_DOCS_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("S3_DOCS_BUCKET is not configured")
    return bucket


def put_pdf_bytes(*, key: str, pdf_bytes: bytes, content_type: str = "application/pdf") -> None:
    put_bytes(bucket=docs_bucket(), key=key, body=pdf_bytes, content_type=content_type)


def presign_get_docs(*, key: str, expires_seconds: int = 900) -> str:
    """Pre-signed GET for an object in the docs/export bucket."""
    return presign_get(bucket=docs_bucket(), key=key, expires_seconds=expires_seconds)
