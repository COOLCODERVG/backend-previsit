from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import boto3


@dataclass(frozen=True)
class TranscribeStartResult:
    job_name: str
    status: str


def _client():
    region = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip() or None
    return boto3.client("transcribe", region_name=region)


def _output_bucket() -> str:
    bucket = os.environ.get("S3_TRANSCRIPTS_BUCKET", "").strip() or os.environ.get("S3_AUDIO_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("S3_TRANSCRIPTS_BUCKET (or S3_AUDIO_BUCKET) is not configured")
    return bucket


def start_medical_job(
    *,
    job_name: str,
    media_s3_uri: str,
    language_code: str = "en-US",
    specialty: str = "PRIMARYCARE",
    transcription_type: str = "CONVERSATION",
    output_key: Optional[str] = None,
) -> TranscribeStartResult:
    client = _client()
    kwargs = {
        "MedicalTranscriptionJobName": job_name,
        "LanguageCode": language_code,
        "Media": {"MediaFileUri": media_s3_uri},
        "Specialty": specialty,
        "Type": transcription_type,
        "OutputBucketName": _output_bucket(),
    }
    if output_key:
        kwargs["OutputKey"] = output_key
    resp = client.start_medical_transcription_job(**kwargs)
    status = (
        ((resp.get("MedicalTranscriptionJob") or {}).get("TranscriptionJobStatus"))
        or "UNKNOWN"
    )
    return TranscribeStartResult(job_name=job_name, status=status)


def get_medical_job_status(*, job_name: str) -> dict:
    client = _client()
    resp = client.get_medical_transcription_job(MedicalTranscriptionJobName=job_name)
    return resp.get("MedicalTranscriptionJob") or {}


def wait_for_job(*, job_name: str, timeout_seconds: int = 60, poll_seconds: int = 3) -> dict:
    deadline = time.time() + timeout_seconds
    last = {}
    while time.time() < deadline:
        last = get_medical_job_status(job_name=job_name)
        status = (last.get("TranscriptionJobStatus") or "").upper()
        if status in {"COMPLETED", "FAILED"}:
            return last
        time.sleep(poll_seconds)
    return last

