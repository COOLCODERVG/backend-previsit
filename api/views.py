from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from django.http import FileResponse
from django.utils import timezone
from pathlib import Path
from datetime import timedelta
import base64
import json
import logging
import urllib.request

from .models import User, Appointment, Symptom, Feeling, Question, Note, Recording, PersonalizationProfile
from .serializers import (
    UserSerializer, AppointmentSerializer, AppointmentCreateSerializer,
    SymptomSerializer, SymptomCreateSerializer,
    FeelingSerializer, FeelingCreateSerializer,
    QuestionSerializer, QuestionCreateSerializer,
    NoteSerializer, NoteCreateSerializer,
    RecordingSerializer, RecordingDetailSerializer, RecordingCreateSerializer,
    AudioUploadInitSerializer,
    PersonalizationProfileSerializer, PersonalizationProfileUpdateSerializer,
)
from .utils import (
    assess_llm_input_coverage,
    generate_llm_pdf_guidance,
    generate_visit_one_pager,
    persist_export_pdf,
)
from .pdf_renderer import render_summary_html, html_to_pdf_bytes
from .object_store import resolve_local_store, sniff_audio_content_type
from .s3 import (
    docs_bucket,
    delete_audio_object,
    get_bytes,
    list_audio_objects,
    parse_s3_uri,
    presign_get_audio,
    presign_get_docs,
    presign_put_audio,
    put_bytes,
    put_pdf_bytes,
)
from .transcribe_medical import start_medical_job, get_medical_job_status, wait_for_job
import os
import uuid
from .llm_client import PDF_GUIDANCE_RESPONSE_FORMAT, call_llama_inference
from .llm_retrieval import retrieve_steering_for_inference
from vectors.personalization_sync import get_user_vector_status, sync_user_preference_vectors

logger = logging.getLogger(__name__)


def _strip_data_url_prefix(b64: str) -> str:
    s = (b64 or "").strip()
    if s.startswith("data:") and "base64," in s:
        s = s.split("base64,", 1)[-1]
    return s.strip()


def _decode_audio_base64(b64: str) -> bytes:
    raw = _strip_data_url_prefix(b64)
    pad = (-len(raw)) % 4
    if pad:
        raw += "=" * pad
    return base64.b64decode(raw, validate=False)


def _transcript_text_from_aws_json(data: dict) -> str:
    results = data.get("results")
    if isinstance(results, dict):
        transcripts = results.get("transcripts")
        if isinstance(transcripts, list) and transcripts:
            t0 = transcripts[0]
            if isinstance(t0, dict):
                tx = (t0.get("transcript") or "").strip()
                if tx:
                    return tx
        items = results.get("items")
        if isinstance(items, list):
            parts = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                alts = it.get("alternatives")
                if isinstance(alts, list) and alts and isinstance(alts[0], dict):
                    c = (alts[0].get("content") or "").strip()
                    if c:
                        parts.append(c)
            return " ".join(parts).strip()
    return ""


def _load_transcript_bytes_from_uri(uri: str) -> bytes:
    parsed = parse_s3_uri(uri)
    if parsed:
        bucket, key = parsed
        return get_bytes(bucket=bucket, key=key)
    req = urllib.request.Request(uri, headers={"User-Agent": "SyniVia/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


# ============== Auth Views ==============

@api_view(['POST'])
@permission_classes([AllowAny])
def register_view(request):
    name = request.data.get('name', '').strip()
    email = (request.data.get('email', '') or '').lower().strip()
    password = request.data.get('password', '')

    if not email or not password or len(password) < 8:
        return Response({'detail': 'Email and password (min 8 chars) required'}, status=400)

    if User.objects.filter(email=email).exists():
        return Response({'detail': 'Email already registered'}, status=400)

    user = User.objects.create_user(email=email, password=password, name=name, role='user')
    profile = PersonalizationProfile.objects.create(user=user)
    refresh = RefreshToken.for_user(user)

    return Response({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'role': user.role,
        'personalization_completed': profile.is_completed,
        'access_token': str(refresh.access_token),
        'refresh_token': str(refresh),
        'token_type': 'Bearer',
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    email = (request.data.get('email', '') or '').lower().strip()
    password = request.data.get('password', '')

    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        return Response({'detail': 'Incorrect username or password'}, status=401)
    except Exception as e:
        # Log database or other errors for debugging, but return generic auth error to user
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("Login error for email=%s", email)
        return Response({'detail': 'Incorrect username or password'}, status=401)

    if not user.check_password(password):
        return Response({'detail': 'Incorrect username or password'}, status=401)

    profile, _ = PersonalizationProfile.objects.get_or_create(user=user)
    refresh = RefreshToken.for_user(user)

    return Response({
        'id': user.id,
        'name': user.name,
        'email': user.email,
        'role': user.role,
        'personalization_completed': profile.is_completed,
        'access_token': str(refresh.access_token),
        'refresh_token': str(refresh),
        'token_type': 'Bearer',
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def logout_view(request):
    # Best-effort revocation of the legacy SimpleJWT refresh token.
    # Cognito sessions are revoked separately via AdminUserGlobalSignOut from
    # the management UI; the mobile app additionally clears its secure store.
    # Accept refresh token in the request body (preferred) or Authorization header.
    refresh = (request.data.get('refresh') or '').strip()
    if not refresh:
        auth = request.META.get('HTTP_AUTHORIZATION', '')
        if auth and auth.lower().startswith('bearer '):
            # If a refresh token was passed as a bearer token, treat it as such.
            refresh = auth.split(None, 1)[1].strip()
    blacklisted = False
    if refresh:
        try:
            RefreshToken(refresh).blacklist()
            blacklisted = True
        except Exception as exc:  # noqa: BLE001
            logger.info("logout: refresh token blacklist failed: %s", exc)
    return Response({'message': 'Logged out successfully', 'blacklisted': blacklisted})


@api_view(['GET', 'PATCH', 'DELETE'])
def me_view(request):
    if request.method == 'DELETE':
        # Deletes the Django user and related rows (CASCADE). Cognito users may
        # still exist until separately removed from the user pool.
        request.user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    if request.method == 'PATCH':
        name = request.data.get('name')
        if isinstance(name, str) and name.strip():
            request.user.name = name.strip()
            request.user.save(update_fields=['name'])

    profile, _ = PersonalizationProfile.objects.get_or_create(user=request.user)
    return Response({
        'id': request.user.id,
        'name': request.user.name,
        'email': request.user.email,
        'role': request.user.role,
        'personalization_completed': profile.is_completed,
    })


@api_view(['GET', 'PUT'])
def personalization_view(request):
    profile, _ = PersonalizationProfile.objects.get_or_create(user=request.user)

    if request.method == 'GET':
        return Response(PersonalizationProfileSerializer(profile).data)

    serializer = PersonalizationProfileUpdateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data

    profile.main_reason = d['main_reason'].strip()
    profile.condition_stage = d['condition_stage']
    profile.biggest_concern = d['biggest_concern'].strip()
    profile.prepared_items = d['prepared_items']
    profile.appointment_outcome = d['appointment_outcome']
    profile.family_history = (d['family_history'] or '').strip()
    profile.age = (d.get('age') or '').strip()
    profile.gender = (d.get('gender') or '').strip()
    profile.region = (d.get('region') or '').strip()
    profile.ml_preferences = d.get('ml_preferences') or []
    profile.average_appointment_minutes = d.get('average_appointment_minutes') or 30
    profile.is_completed = True
    profile.save()

    try:
        sync_user_preference_vectors(user_id=int(request.user.id))
    except Exception:
        logger.exception("personalization vector sync failed for user %s", request.user.id)

    return Response(PersonalizationProfileSerializer(profile).data)


@api_view(['GET'])
def personalization_ml_status_view(request):
    profile = PersonalizationProfile.objects.filter(user=request.user).first()
    status_payload = {
        'profile_completed': bool(profile and profile.is_completed),
        'profile_updated_at': profile.updated_at if profile else None,
    }
    try:
        status_payload['vector_status'] = get_user_vector_status(user_id=int(request.user.id))
    except Exception as exc:
        logger.exception("vector status lookup failed for user %s", request.user.id)
        status_payload['vector_status'] = {'enabled': False, 'error': str(exc)}
    return Response(status_payload)


@api_view(['POST'])
def personalization_ml_refresh_view(request):
    body = request.data if isinstance(request.data, dict) else {}
    reason = str(body.get('reason') or '').strip()
    extra_contexts = []
    if reason:
        extra_contexts.append(f"Refresh reason: {reason}")

    try:
        result = sync_user_preference_vectors(
            user_id=int(request.user.id),
            extra_contexts=extra_contexts,
        )
        return Response({'ok': True, 'sync': result})
    except Exception as exc:
        logger.exception("manual vector refresh failed for user %s", request.user.id)
        return Response({'ok': False, 'detail': 'vector_refresh_failed', 'error': str(exc)}, status=503)


@api_view(['GET'])
@permission_classes([AllowAny])
def oidc_config_view(request):
    """Expose Cognito hosted-UI parameters to the mobile client.

    The Expo app reads these on startup so the bundled binary doesn't have to
    hardcode the user-pool id / domain (which differ between dev and prod).
    """
    pool_id = (os.environ.get('COGNITO_USER_POOL_ID') or '').strip()
    if not pool_id:
        return Response({'enabled': False})
    region = (os.environ.get('AWS_REGION') or 'us-east-1').strip()
    issuer = (os.environ.get('COGNITO_ISSUER') or
              f'https://cognito-idp.{region}.amazonaws.com/{pool_id}').strip()
    return Response({
        'enabled': True,
        'issuer': issuer,
        'jwks_url': (os.environ.get('COGNITO_JWKS_URL') or
                     f'{issuer}/.well-known/jwks.json'),
        'client_id': (os.environ.get('COGNITO_APP_CLIENT_ID') or '').strip(),
        'domain': (os.environ.get('COGNITO_DOMAIN') or '').strip(),
        'audience': (os.environ.get('COGNITO_AUDIENCE') or
                     os.environ.get('COGNITO_APP_CLIENT_ID') or '').strip(),
        'scopes': ['openid', 'email', 'profile'],
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def refresh_view(request):
    token = request.data.get('refresh')
    if not token:
        return Response({'detail': 'No refresh token'}, status=401)
    try:
        refresh = RefreshToken(token)
        return Response({
            'message': 'Token refreshed',
            'access_token': str(refresh.access_token),
        })
    except Exception:
        return Response({'detail': 'Invalid refresh token'}, status=401)


def _normalize_instruction_text(inst):
    if isinstance(inst, dict):
        return (inst.get('text') or '').strip()
    if isinstance(inst, str):
        return inst.strip()
    return ''


@api_view(['GET'])
def action_plan_view(request):
    """Aggregate deduplicated provider instructions from recent extracted recordings."""
    since = timezone.now() - timedelta(days=30)
    recordings = (
        Recording.objects.filter(user=request.user, status='extracted', updated_at__gte=since)
        .select_related('appointment')
        .order_by('-updated_at')
    )
    candidates = []
    for rec in recordings:
        entities = rec.extracted_entities
        if not isinstance(entities, dict):
            continue
        for inst in entities.get('instructions') or []:
            text = _normalize_instruction_text(inst)
            if not text:
                continue
            candidates.append((rec.updated_at, text, rec.appointment_id, rec.id))
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    items = []
    for updated_at, text, appointment_id, recording_id in candidates:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({
            'text': text,
            'appointment_id': appointment_id,
            'recording_id': recording_id,
            'updated_at': updated_at.isoformat(),
        })
        if len(items) >= 25:
            break
    return Response({'items': items})


# ============== Health Check ==============

@api_view(['GET'])
@permission_classes([AllowAny])
def root_view(request):
    return Response({'message': 'SyniVia API is running (Django)', 'version': '2.0.0'})


@api_view(['GET'])
@permission_classes([AllowAny])
def health_view(request):
    return Response({'status': 'healthy', 'database': 'sqlite3'})


# ============== Appointment Views ==============

@api_view(['GET', 'POST'])
def appointments_view(request):
    if request.method == 'GET':
        qs = Appointment.objects.filter(user=request.user)
        completed = request.query_params.get('completed')
        if completed is not None:
            qs = qs.filter(is_completed=completed.lower() == 'true')
        return Response(AppointmentSerializer(qs, many=True).data)

    # POST
    serializer = AppointmentCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    apt = Appointment.objects.create(user=request.user, **serializer.validated_data)
    return Response(AppointmentSerializer(apt).data, status=201)


@api_view(['GET', 'PUT', 'DELETE'])
def appointment_detail_view(request, pk):
    try:
        apt = Appointment.objects.get(pk=pk, user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)

    if request.method == 'GET':
        return Response(AppointmentSerializer(apt).data)

    if request.method == 'PUT':
        for field in ['doctor_name', 'specialty', 'location', 'appointment_date', 'appointment_time', 'notes']:
            if field in request.data:
                setattr(apt, field, request.data[field])
        if 'is_completed' in request.data:
            apt.is_completed = request.data['is_completed']
        apt.save()
        return Response(AppointmentSerializer(apt).data)

    # DELETE
    apt.delete()
    return Response({'message': 'Appointment deleted successfully'})


# ============== Symptom Views ==============

@api_view(['GET', 'POST'])
def symptoms_view(request):
    if request.method == 'GET':
        qs = Symptom.objects.filter(user=request.user)
        apt_id = request.query_params.get('appointment_id')
        if apt_id:
            qs = qs.filter(appointment_id=apt_id)
        data = []
        for s in qs:
            d = SymptomSerializer(s).data
            d['appointment_id'] = s.appointment_id
            data.append(d)
        return Response(data)

    # POST
    serializer = SymptomCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data
    try:
        apt = Appointment.objects.get(pk=d['appointment_id'], user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)
    symptom = Symptom.objects.create(
        user=request.user, appointment=apt,
        name=d['name'], severity=d['severity'],
        timing=d.get('timing'), duration=d.get('duration'),
        is_new=d.get('is_new', True), is_worsening=d.get('is_worsening', False),
        notes=d.get('notes'),
    )
    resp = SymptomSerializer(symptom).data
    resp['appointment_id'] = symptom.appointment_id
    return Response(resp, status=201)


@api_view(['PUT', 'DELETE'])
def symptom_detail_view(request, pk):
    try:
        symptom = Symptom.objects.get(pk=pk, user=request.user)
    except Symptom.DoesNotExist:
        return Response({'detail': 'Symptom not found'}, status=404)

    if request.method == 'PUT':
        for field in ['name', 'severity', 'timing', 'duration', 'is_new', 'is_worsening', 'notes']:
            if field in request.data:
                setattr(symptom, field, request.data[field])
        symptom.save()
        resp = SymptomSerializer(symptom).data
        resp['appointment_id'] = symptom.appointment_id
        return Response(resp)

    symptom.delete()
    return Response({'message': 'Symptom deleted successfully'})


# ============== Feeling Views ==============

@api_view(['GET', 'POST'])
def feelings_view(request):
    if request.method == 'GET':
        qs = Feeling.objects.filter(user=request.user)
        start = request.query_params.get('start_date')
        end = request.query_params.get('end_date')
        if start:
            qs = qs.filter(date__gte=start)
        if end:
            qs = qs.filter(date__lte=end)
        return Response(FeelingSerializer(qs[:30], many=True).data)

    # POST
    serializer = FeelingCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data
    feeling_date = d.get('date') or timezone.now().strftime('%Y-%m-%d')
    feeling = Feeling.objects.create(
        user=request.user,
        health_score=d['health_score'],
        energy_level=d['energy_level'],
        mood=d.get('mood'),
        notes=d.get('notes'),
        date=feeling_date,
    )
    return Response(FeelingSerializer(feeling).data, status=201)


@api_view(['GET'])
def feeling_by_date_view(request, date_str):
    feeling = Feeling.objects.filter(user=request.user, date=date_str).first()
    if feeling:
        return Response(FeelingSerializer(feeling).data)
    return Response(None, status=200)


@api_view(['PUT', 'DELETE'])
def feeling_detail_view(request, pk):
    try:
        feeling = Feeling.objects.get(pk=pk, user=request.user)
    except Feeling.DoesNotExist:
        return Response({'detail': 'Feeling not found'}, status=404)

    if request.method == 'PUT':
        for field in ['health_score', 'energy_level', 'mood', 'notes']:
            if field in request.data:
                setattr(feeling, field, request.data[field])
        feeling.save()
        return Response(FeelingSerializer(feeling).data)

    feeling.delete()
    return Response({'message': 'Feeling deleted successfully'})


# ============== Question Views ==============

@api_view(['GET', 'POST'])
def questions_view(request):
    if request.method == 'GET':
        qs = Question.objects.filter(user=request.user)
        apt_id = request.query_params.get('appointment_id')
        if apt_id:
            qs = qs.filter(appointment_id=apt_id)
        answered = request.query_params.get('answered')
        if answered is not None:
            qs = qs.filter(is_answered=answered.lower() == 'true')
        data = []
        for q in qs:
            d = QuestionSerializer(q).data
            d['appointment_id'] = q.appointment_id
            data.append(d)
        return Response(data)

    # POST
    serializer = QuestionCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data
    try:
        apt = Appointment.objects.get(pk=d['appointment_id'], user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)
    question = Question.objects.create(
        user=request.user, appointment=apt,
        text=d['text'], priority=d.get('priority', 1),
        is_answered=d.get('is_answered', False),
        answer_notes=d.get('answer_notes'),
    )
    resp = QuestionSerializer(question).data
    resp['appointment_id'] = question.appointment_id
    return Response(resp, status=201)


@api_view(['PUT', 'DELETE'])
def question_detail_view(request, pk):
    try:
        question = Question.objects.get(pk=pk, user=request.user)
    except Question.DoesNotExist:
        return Response({'detail': 'Question not found'}, status=404)

    if request.method == 'PUT':
        for field in ['text', 'priority', 'is_answered', 'answer_notes']:
            if field in request.data:
                setattr(question, field, request.data[field])
        question.save()
        resp = QuestionSerializer(question).data
        resp['appointment_id'] = question.appointment_id
        return Response(resp)

    question.delete()
    return Response({'message': 'Question deleted successfully'})


# ============== Note Views ==============

@api_view(['GET', 'POST'])
def notes_view(request):
    if request.method == 'GET':
        qs = Note.objects.filter(user=request.user)
        apt_id = request.query_params.get('appointment_id')
        if apt_id:
            qs = qs.filter(appointment_id=apt_id)
        category = request.query_params.get('category')
        if category:
            qs = qs.filter(category=category)
        data = []
        for n in qs:
            d = NoteSerializer(n).data
            d['appointment_id'] = n.appointment_id
            data.append(d)
        return Response(data)

    # POST
    serializer = NoteCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data
    try:
        apt = Appointment.objects.get(pk=d['appointment_id'], user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)
    note = Note.objects.create(
        user=request.user, appointment=apt,
        title=d['title'], content=d['content'],
        category=d.get('category', 'general'),
    )
    resp = NoteSerializer(note).data
    resp['appointment_id'] = note.appointment_id
    return Response(resp, status=201)


@api_view(['PUT', 'DELETE'])
def note_detail_view(request, pk):
    try:
        note = Note.objects.get(pk=pk, user=request.user)
    except Note.DoesNotExist:
        return Response({'detail': 'Note not found'}, status=404)

    if request.method == 'PUT':
        for field in ['title', 'content', 'category']:
            if field in request.data:
                setattr(note, field, request.data[field])
        note.save()
        resp = NoteSerializer(note).data
        resp['appointment_id'] = note.appointment_id
        return Response(resp)

    note.delete()
    return Response({'message': 'Note deleted successfully'})


# ============== Recording Views ==============

def _presign_recording_audio(recording) -> str | None:
    """Best-effort presigned GET for an S3-backed recording. Returns None on failure."""
    if recording.audio_storage != 's3' or not recording.audio_object_key:
        return None
    try:
        return presign_get_audio(object_key=recording.audio_object_key, expires_seconds=900)
    except Exception:
        return None


def _purge_expired_recordings(user) -> None:
    """
    Delete recordings past their dynamic retention window (average appointment
    length + a 2-hour buffer — see `recording_retention.py`).

    Runs opportunistically whenever this user's recordings are listed/fetched,
    so no separate scheduled job is required for the recordings to actually
    become unavailable once they expire. Deletes the S3 audio object (when
    present) before removing the database row.
    """
    for recording in Recording.objects.filter(user=user):
        if not recording.is_expired:
            continue
        if recording.audio_storage == 's3' and recording.audio_object_key:
            try:
                delete_audio_object(object_key=recording.audio_object_key)
            except Exception:
                logger.exception("Failed to delete expired S3 audio object %s", recording.audio_object_key)
        recording.delete()


@api_view(['GET', 'POST'])
def recordings_view(request):
    if request.method == 'GET':
        _purge_expired_recordings(request.user)
        qs = Recording.objects.filter(user=request.user)
        apt_id = request.query_params.get('appointment_id')
        if apt_id:
            qs = qs.filter(appointment_id=apt_id)
        data = []
        for r in qs:
            d = RecordingSerializer(r).data
            d['appointment_id'] = r.appointment_id
            d['audio_download_url'] = _presign_recording_audio(r)
            data.append(d)
        return Response(data)

    # POST
    serializer = RecordingCreateSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data
    try:
        apt = Appointment.objects.get(pk=d['appointment_id'], user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)
    recording = Recording.objects.create(
        user=request.user, appointment=apt,
        title=d['title'], duration_seconds=d.get('duration_seconds', 0),
    )

    # Backwards-compatible dev path: accept base64 audio, but store as object.
    audio_b64 = (d.get('audio_base64') or '').strip()
    if audio_b64:
        object_key = f"audio/u{request.user.id}/apt{apt.id}/rec{recording.id}.bin"
        content_type = sniff_audio_content_type(
            recording.title,
            provided=(d.get('audio_content_type') or '').strip() or None,
        )
        audio_bucket_name = os.environ.get("S3_AUDIO_BUCKET", "").strip()
        if audio_bucket_name:
            try:
                body = _decode_audio_base64(audio_b64)
                digest, size = put_bytes(
                    bucket=audio_bucket_name,
                    key=object_key,
                    body=body,
                    content_type=content_type,
                )
                recording.audio_storage = "s3"
                recording.audio_object_key = object_key
                recording.audio_sha256 = digest
                recording.audio_size_bytes = size
                recording.audio_content_type = content_type
                recording.status = "uploaded"
            except Exception:
                logger.exception("S3 audio upload failed; falling back to local object store")
                store = resolve_local_store()
                put = store.put_base64(object_key=object_key, content_base64=audio_b64)
                recording.audio_storage = "local"
                recording.audio_object_key = put.object_key
                recording.audio_sha256 = put.sha256
                recording.audio_size_bytes = put.size_bytes
                recording.audio_content_type = content_type
                recording.status = "uploaded"
        else:
            store = resolve_local_store()
            put = store.put_base64(object_key=object_key, content_base64=audio_b64)
            recording.audio_storage = "local"
            recording.audio_object_key = put.object_key
            recording.audio_sha256 = put.sha256
            recording.audio_size_bytes = put.size_bytes
            recording.audio_content_type = content_type
            recording.status = "uploaded"
        recording.save()

    resp = RecordingSerializer(recording).data
    resp['appointment_id'] = recording.appointment_id
    return Response(resp, status=201)


@api_view(['GET', 'DELETE'])
def recording_detail_view(request, pk):
    try:
        recording = Recording.objects.get(pk=pk, user=request.user)
    except Recording.DoesNotExist:
        return Response({'detail': 'Recording not found'}, status=404)

    if request.method == 'GET':
        if recording.is_expired:
            if recording.audio_storage == 's3' and recording.audio_object_key:
                try:
                    delete_audio_object(object_key=recording.audio_object_key)
                except Exception:
                    logger.exception("Failed to delete expired S3 audio object %s", recording.audio_object_key)
            recording.delete()
            return Response({'detail': 'Recording has expired and was removed'}, status=404)
        resp = RecordingDetailSerializer(recording).data
        resp['appointment_id'] = recording.appointment_id
        resp['audio_download_url'] = _presign_recording_audio(recording)
        return Response(resp)

    recording.delete()
    return Response({'message': 'Recording deleted successfully'})


# ============== Audio Upload + Transcription (NeuraVia) ==============

@api_view(['POST'])
def voice_dump_transcribe_view(request):
    """
    Transcribe a short ad-hoc voice note for text fields.
    Request body:
      - audio_base64: base64-encoded audio bytes
      - audio_content_type: optional, defaults to audio/mp4
    """
    body = request.data if isinstance(request.data, dict) else {}
    audio_base64 = str(body.get('audio_base64') or '').strip()
    if not audio_base64:
        return Response({'detail': 'audio_base64 is required'}, status=400)

    content_type = (str(body.get('audio_content_type') or '').strip() or 'audio/mp4')
    bucket = os.environ.get('S3_AUDIO_BUCKET', '').strip()
    if not bucket:
        return Response({'detail': 'S3_AUDIO_BUCKET is not configured'}, status=500)

    try:
        audio_bytes = _decode_audio_base64(audio_base64)
    except Exception:
        return Response({'detail': 'Invalid audio_base64 payload'}, status=400)

    if not audio_bytes:
        return Response({'detail': 'Audio payload is empty'}, status=400)

    object_key = f"audio/voice-dump/u{request.user.id}/{uuid.uuid4().hex}.bin"
    try:
        put_bytes(bucket=bucket, key=object_key, body=audio_bytes, content_type=content_type)
    except Exception:
        logger.exception("voice-dump: failed to upload audio")
        return Response({'detail': 'Failed to upload voice note'}, status=500)

    media_uri = f"s3://{bucket}/{object_key}"
    job_name = f"nv_vdump_u{request.user.id}_{uuid.uuid4().hex[:10]}"

    try:
        start_medical_job(job_name=job_name, media_s3_uri=media_uri)
        job = wait_for_job(job_name=job_name, timeout_seconds=45, poll_seconds=3)
    except Exception:
        logger.exception("voice-dump: failed to transcribe")
        return Response({'detail': 'Failed to start transcription'}, status=502)

    status_value = (job.get('TranscriptionJobStatus') or '').lower()
    if status_value != 'completed':
        return Response(
            {
                'status': status_value or 'failed',
                'transcript': '',
                'detail': 'Transcription did not complete in time',
            },
            status=202,
        )

    uri = ((job.get('Transcript') or {}).get('TranscriptFileUri') or '').strip()
    if not uri:
        return Response({'status': 'completed', 'transcript': ''})

    try:
        raw = _load_transcript_bytes_from_uri(uri)
        data = json.loads(raw.decode('utf-8'))
        transcript = _transcript_text_from_aws_json(data)
    except Exception:
        logger.exception("voice-dump: transcript fetch/parse failed")
        return Response({'detail': 'Failed to parse transcript'}, status=502)

    return Response({'status': 'completed', 'transcript': transcript or ''})

@api_view(['POST'])
def audio_upload_init_view(request):
    """
    Returns a pre-signed S3 PUT URL for direct-to-S3 uploads.
    Creates a Recording row in `created` state and returns its id + object key.
    """
    serializer = AudioUploadInitSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    d = serializer.validated_data

    try:
        apt = Appointment.objects.get(pk=d['appointment_id'], user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)

    content_type = (d.get('content_type') or '').strip() or 'application/octet-stream'
    object_key = f"audio/u{request.user.id}/apt{apt.id}/{uuid.uuid4().hex}.bin"

    recording = Recording.objects.create(
        user=request.user,
        appointment=apt,
        title=d['title'],
        duration_seconds=d.get('duration_seconds', 0),
        status='created',
        audio_storage='s3',
        audio_object_key=object_key,
        audio_content_type=content_type,
        audio_size_bytes=int(d.get('size_bytes') or 0),
    )

    try:
        presigned = presign_put_audio(object_key=object_key, content_type=content_type, expires_seconds=900)
    except RuntimeError as exc:
        # If S3 isn't configured, keep the recording but inform the client.
        return Response({'detail': str(exc)}, status=500)

    return Response({
        'recording_id': recording.id,
        'appointment_id': apt.id,
        'object_key': object_key,
        'upload_url': presigned.upload_url,
        'headers': presigned.headers,
        'expires_seconds': 900,
    }, status=201)


@api_view(['GET'])
def audio_objects_list_view(request):
    """
    Lists raw S3 objects under the caller's audio prefix (``audio/u<user_id>/``).
    Optional ?appointment_id=<n> filters to a specific appointment subprefix.
    Each item includes a short-lived presigned GET URL.

    This is independent of the Recording table — useful for verifying what's
    actually in the bucket and for orphaned-file cleanup.
    """
    appointment_id = (request.query_params.get('appointment_id') or '').strip()
    prefix = f"audio/u{request.user.id}/"
    if appointment_id and appointment_id.isdigit():
        prefix = f"audio/u{request.user.id}/apt{int(appointment_id)}/"

    try:
        items = list_audio_objects(prefix=prefix, max_items=200)
    except RuntimeError as exc:
        return Response({'detail': str(exc)}, status=500)
    except Exception:
        logger.exception("Failed to list S3 audio objects under %s", prefix)
        return Response({'detail': 'Failed to list audio objects'}, status=500)

    enriched = []
    for it in items:
        url = None
        try:
            url = presign_get_audio(object_key=it['object_key'], expires_seconds=900)
        except Exception:
            url = None
        enriched.append({**it, 'download_url': url})

    return Response({'prefix': prefix, 'count': len(enriched), 'items': enriched})


@api_view(['POST'])
def audio_transcribe_start_view(request, pk):
    """
    Starts AWS Transcribe Medical for an uploaded audio object.
    Stores job metadata in Recording.transcript_json.
    """
    try:
        recording = Recording.objects.get(pk=pk, user=request.user)
    except Recording.DoesNotExist:
        return Response({'detail': 'Recording not found'}, status=404)

    bucket = os.environ.get("S3_AUDIO_BUCKET", "").strip()
    if not bucket:
        return Response({'detail': 'S3_AUDIO_BUCKET is not configured'}, status=500)

    if recording.audio_storage == "s3" and recording.audio_object_key:
        media_uri = f"s3://{bucket}/{recording.audio_object_key}"
    elif recording.audio_storage == "local" and recording.audio_object_key:
        # Promote local/dev audio to S3 so Transcribe can read a consistent s3:// URI.
        store = resolve_local_store()
        try:
            body, meta = store.read_bytes(
                recording.audio_object_key,
                content_type=(recording.audio_content_type or "application/octet-stream"),
            )
        except Exception:
            return Response({'detail': 'Recording audio object not found'}, status=400)
        ct = (recording.audio_content_type or meta.content_type or "application/octet-stream").strip()
        s3_key = f"audio/u{request.user.id}/apt{recording.appointment_id}/rec{recording.id}_{uuid.uuid4().hex[:8]}.bin"
        try:
            digest, size = put_bytes(bucket=bucket, key=s3_key, body=body, content_type=ct)
        except Exception:
            return Response({'detail': 'Failed to upload audio to S3 for transcription'}, status=500)
        recording.audio_storage = "s3"
        recording.audio_object_key = s3_key
        recording.audio_content_type = ct
        recording.audio_sha256 = digest or recording.audio_sha256
        recording.audio_size_bytes = size or recording.audio_size_bytes
        recording.save()
        media_uri = f"s3://{bucket}/{s3_key}"
    else:
        return Response({'detail': 'Recording has no audio object to transcribe'}, status=400)

    job_name = f"nv_u{request.user.id}_r{recording.id}_{uuid.uuid4().hex[:10]}"

    try:
        start = start_medical_job(job_name=job_name, media_s3_uri=media_uri)
    except Exception as exc:
        recording.status = 'failed'
        recording.transcript_json = {'error': str(exc)}
        recording.save()
        return Response({'detail': 'Failed to start transcription'}, status=500)

    recording.status = 'transcribing'
    recording.transcript_json = {
        **(recording.transcript_json or {}),
        'provider': 'aws_transcribe_medical',
        'job_name': start.job_name,
        'status': start.status,
        'media_s3_uri': media_uri,
    }
    recording.save()

    return Response({'recording_id': recording.id, 'job_name': start.job_name, 'status': start.status})


@api_view(['GET'])
def audio_transcribe_status_view(request, pk):
    """
    Returns AWS Transcribe Medical job status and stores terminal results.
    """
    try:
        recording = Recording.objects.get(pk=pk, user=request.user)
    except Recording.DoesNotExist:
        return Response({'detail': 'Recording not found'}, status=404)

    job_name = (recording.transcript_json or {}).get('job_name')
    if not job_name:
        return Response({'detail': 'No transcription job started for this recording'}, status=400)

    try:
        job = get_medical_job_status(job_name=job_name)
    except Exception:
        return Response({'detail': 'Failed to fetch job status'}, status=500)

    status_value = (job.get('TranscriptionJobStatus') or '').lower()
    prior_status = recording.status
    recording.transcript_json = {**(recording.transcript_json or {}), 'aws_job': job}
    if status_value == 'completed':
        recording.status = 'transcribed'
        uri = (job.get('Transcript') or {}).get("TranscriptFileUri")
        if uri and not (recording.transcript_text or "").strip():
            try:
                raw = _load_transcript_bytes_from_uri(uri)
                data = json.loads(raw.decode("utf-8"))
                text = _transcript_text_from_aws_json(data)
                if text:
                    recording.transcript_text = text
                tj = recording.transcript_json or {}
                tj["transcript_file_uri"] = uri
                tj["transcript_json_preview"] = data if isinstance(data, dict) else {"raw": True}
                recording.transcript_json = tj
            except Exception:
                logger.exception("Failed to load transcript artifact from %s", uri)
    elif status_value == 'failed':
        recording.status = 'failed'
    recording.save()

    if status_value == 'completed' and prior_status != 'transcribed':
        try:
            from notifications.dispatch import push_summary_ready_for_user

            push_summary_ready_for_user(request.user.id, recording.appointment_id)
        except Exception:
            logger.exception("summary_ready push failed for recording %s", recording.id)

    return Response({'recording_id': recording.id, 'status': job.get('TranscriptionJobStatus'), 'job': job})


@api_view(['POST'])
def audio_extract_entities_view(request, pk):
    """Extract structured clinical data from a transcribed recording.

    Pipeline:
      1. Pull `Recording.transcript_text` (must already exist; the
         transcription endpoint produces it).
      2. Run `clinical_extraction.extract_from_transcript`, which calls
         AWS Comprehend Medical (DetectEntitiesV2 + InferRxNorm + InferICD10CM).
      3. Store the structured output on `Recording.extracted_entities`.
      4. Reconcile detected medications into the user's `Medication` table
         (high-confidence rows auto-confirmed, others queued for review).
      5. If anything changed, asynchronously kick off the steering-vector
         derivation worker so the LLM is immediately biased toward the new
         clinical state.

    The caller may POST a body with ``override_payload`` to bypass Comprehend
    Medical (useful in tests or when an external worker has already extracted
    the entities).
    """
    try:
        recording = Recording.objects.get(pk=pk, user=request.user)
    except Recording.DoesNotExist:
        return Response({'detail': 'Recording not found'}, status=404)

    if recording.status not in {'transcribed', 'extracted'}:
        return Response({'detail': 'Recording must be transcribed before extraction'}, status=400)

    body = request.data if isinstance(request.data, dict) else {}

    payload: dict
    medications_for_reconcile = []
    if isinstance(body.get('override_payload'), dict):
        payload = body['override_payload']
        # Allow clients/workers to pass a pre-extracted shape but still
        # reconcile its medications (mirrors the prior stub behavior).
        try:
            from .clinical_extraction import MedicationCandidate
            for m in payload.get('medications') or []:
                if not isinstance(m, dict):
                    continue
                medications_for_reconcile.append(MedicationCandidate(
                    name=str(m.get('name') or '').strip(),
                    rxnorm_code=str(m.get('rxnorm_code') or '').strip(),
                    rxnorm_description=str(m.get('rxnorm_description') or '').strip(),
                    dose=str(m.get('dose') or '').strip(),
                    frequency=str(m.get('frequency') or '').strip(),
                    route=str(m.get('route') or '').strip(),
                    confidence=float(m.get('confidence') or 0.0),
                    text_span=str(m.get('text_span') or m.get('name') or '').strip(),
                ))
        except Exception:
            logger.exception("override_payload medication parse failed")
    else:
        from .clinical_extraction import extract_from_transcript
        try:
            result = extract_from_transcript(recording.transcript_text or '')
        except Exception as exc:
            logger.exception("Comprehend Medical extraction failed: %s", exc)
            return Response({'detail': 'Clinical extraction failed', 'error': str(exc)}, status=502)
        payload = result.to_payload()
        medications_for_reconcile = result.medications

    recording.status = 'extracted'
    recording.extracted_entities = payload
    recording.save()

    delta_summary = 'no changes'
    if medications_for_reconcile:
        try:
            from .clinical_extraction import reconcile_from_voice
            delta = reconcile_from_voice(user=request.user, candidates=medications_for_reconcile)
            delta_summary = delta.summary()
            if not delta.is_empty():
                # Rebuild user vectors to include updated medication state.
                try:
                    sync_user_preference_vectors(
                        user_id=int(request.user.id),
                        extra_contexts=[
                            'Medication update from appointment recording: ' + delta.summary()
                        ],
                    )
                except Exception:
                    logger.exception("vector sync after extraction failed")
        except Exception:
            logger.exception("Medication reconciliation failed")

    return Response({
        'recording_id': recording.id,
        'status': recording.status,
        'extracted_entities': recording.extracted_entities,
        'reconciliation': delta_summary,
    })


# ============== Summary View ==============

def build_summary_payload(user, apt):
    _purge_expired_recordings(user)
    symptoms = Symptom.objects.filter(appointment=apt, user=user)
    questions = Question.objects.filter(appointment=apt, user=user)
    notes = Note.objects.filter(appointment=apt, user=user)
    recordings = Recording.objects.filter(appointment=apt, user=user)

    two_weeks_ago = (timezone.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    feelings = Feeling.objects.filter(user=user, date__gte=two_weeks_ago)
    personalization_profile = PersonalizationProfile.objects.filter(user=user).first()

    sym_data = []
    for s in symptoms:
        d = SymptomSerializer(s).data
        d['appointment_id'] = s.appointment_id
        sym_data.append(d)

    q_data = []
    for q in questions:
        d = QuestionSerializer(q).data
        d['appointment_id'] = q.appointment_id
        q_data.append(d)

    n_data = []
    for n in notes:
        d = NoteSerializer(n).data
        d['appointment_id'] = n.appointment_id
        n_data.append(d)

    r_data = []
    for r in recordings:
        d = RecordingSerializer(r).data
        d['appointment_id'] = r.appointment_id
        r_data.append(d)

    transcript_snippets = []
    for r in recordings:
        tx = (getattr(r, 'transcript_text', None) or '').strip()
        if tx:
            transcript_snippets.append(tx[:2000])

    cache = apt.visit_summary_cache if isinstance(getattr(apt, 'visit_summary_cache', None), dict) else {}
    partial_payload = {
        'appointment': AppointmentSerializer(apt).data,
        'symptoms': sym_data,
        'questions': q_data,
        'notes': n_data,
        'feelings': FeelingSerializer(feelings, many=True).data,
        'recordings': r_data,
        'personalization_profile': (
            PersonalizationProfileSerializer(personalization_profile).data
            if personalization_profile
            else None
        ),
        'transcript_snippets': transcript_snippets,
    }

    return {
        **partial_payload,
        'llm_input_coverage': assess_llm_input_coverage(partial_payload),
        'one_pager': cache.get('one_pager'),
        'action_items': cache.get('action_items') or [],
        'one_pager_view_mode': cache.get('view_mode') or 'standard',
        'one_pager_source': cache.get('source') or '',
    }


@api_view(['POST'])
def generate_one_pager_view(request, pk):
    try:
        apt = Appointment.objects.get(pk=pk, user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)

    body = request.data if isinstance(request.data, dict) else {}
    view_mode = str(body.get('view_mode') or 'standard').strip().lower()
    if view_mode not in {'standard', 'simplified', 'caregiver'}:
        view_mode = 'standard'
    force = bool(body.get('force', False))

    summary_payload = build_summary_payload(request.user, apt)
    llm_coverage = summary_payload.get('llm_input_coverage') or assess_llm_input_coverage(summary_payload)
    llm_warnings = llm_coverage.get('warnings') or []
    cache = apt.visit_summary_cache if isinstance(apt.visit_summary_cache, dict) else {}

    if (
        not force
        and cache.get('one_pager')
        and (cache.get('view_mode') or 'standard') == view_mode
    ):
        return Response({
            'one_pager': cache.get('one_pager'),
            'action_items': cache.get('action_items') or [],
            'view_mode': view_mode,
            'source': cache.get('source') or 'cache',
            'generated_at': cache.get('generated_at'),
            'llm_input_coverage': llm_coverage,
            'llm_context_warnings': llm_warnings,
        })

    one_pager, source = generate_visit_one_pager(
        summary_payload,
        view_mode=view_mode,
        user_id=request.user.id,
    )
    action_items = one_pager.get('action_items') or []
    apt.visit_summary_cache = {
        'one_pager': one_pager,
        'action_items': action_items,
        'view_mode': view_mode,
        'source': source,
        'generated_at': timezone.now().isoformat(),
    }
    apt.save(update_fields=['visit_summary_cache', 'updated_at'])

    return Response({
        'one_pager': one_pager,
        'action_items': action_items,
        'view_mode': view_mode,
        'source': source,
        'generated_at': apt.visit_summary_cache.get('generated_at'),
        'llm_input_coverage': llm_coverage,
        'llm_context_warnings': llm_warnings,
    })


@api_view(['GET'])
def summary_view(request, pk):
    try:
        apt = Appointment.objects.get(pk=pk, user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)

    return Response(build_summary_payload(request.user, apt))


@api_view(['POST'])
def export_summary_pdf_view(request, pk):
    try:
        apt = Appointment.objects.get(pk=pk, user=request.user)
    except Appointment.DoesNotExist:
        return Response({'detail': 'Appointment not found'}, status=404)

    summary_payload = build_summary_payload(request.user, apt)
    export_preferences = request.data if isinstance(request.data, dict) else {}
    use_ai_personalization = bool(export_preferences.get('use_ai_personalization', True))

    ai_guidance = None
    if use_ai_personalization:
        ai_guidance = generate_llm_pdf_guidance(summary_payload, export_preferences, user_id=request.user.id)

    try:
        html = render_summary_html(summary_payload, ai_guidance=ai_guidance)
        pdf_bytes = html_to_pdf_bytes(html)
        doctor = (summary_payload.get("appointment") or {}).get("doctor_name") or "appointment"
        apt_date = (summary_payload.get("appointment") or {}).get("appointment_date") or timezone.now().strftime("%Y-%m-%d")
        filename = f"SyniVia_{doctor.replace(' ', '_')}_{apt_date}.pdf"
    except RuntimeError as exc:
        return Response({'detail': str(exc)}, status=500)

    encoded = base64.b64encode(pdf_bytes).decode('utf-8')

    storage = "local"
    object_key = ""
    bucket = ""
    download_url = None

    # Preferred: S3 storage (HIPAA aligned, SSE-KMS).
    try:
        object_key = f"pdf/u{request.user.id}/apt{apt.id}/{uuid.uuid4().hex}.pdf"
        put_pdf_bytes(key=object_key, pdf_bytes=pdf_bytes)
        bucket = docs_bucket()
        storage = "s3"
        download_url = presign_get_docs(key=object_key, expires_seconds=900)
    except Exception:
        # Dev fallback: persist locally and expose existing download endpoint.
        file_id = persist_export_pdf(pdf_bytes, request.user.id, filename)
        object_key = file_id
        storage = "local"
        download_url = request.build_absolute_uri(f'/api/exports/{file_id}')

    # Store export metadata (single unified database — documents app tables).
    try:
        from documents.models import ExportedPdf
        ExportedPdf.objects.create(
            user_id=request.user.id,
            appointment_id=apt.id,
            filename=filename,
            storage=storage,
            s3_bucket=bucket,
            s3_key=object_key,
            sha256="",
            size_bytes=len(pdf_bytes),
        )
    except Exception:
        pass

    return Response({
        'filename': filename,
        'mime_type': 'application/pdf',
        'content_base64': encoded,
        'download_url': download_url,
        'personalization_applied': bool(summary_payload.get('personalization_profile')),
        'ai_personalization_requested': use_ai_personalization,
        'ai_personalization_used': bool(ai_guidance),
    })


@api_view(['GET'])
def export_file_download_view(request, file_id):
    expected_prefix = f'u{request.user.id}_'
    if not file_id.startswith(expected_prefix):
        return Response({'detail': 'Export file not found'}, status=404)

    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / 'exports' / file_id
    if not file_path.exists() or not file_path.is_file():
        return Response({'detail': 'Export file not found'}, status=404)

    response = FileResponse(open(file_path, 'rb'), content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{file_id.split("_", 2)[-1]}"'
    return response


# ============== LLM Generation (NeuraVia) ==============

@api_view(['POST'])
def llm_generate_view(request):
    """
    Personalized generation with representation editing:
    - Optional ANN retrieval: encode `retrieval_query` / `query` via embedding service, pgvector ANN,
      then attach top-k steering vectors for the matching preference contexts.
    - Tasks: `general`, `pdf_guidance`, `medication_update` (inference server interprets `task`).
    """
    payload = request.data if isinstance(request.data, dict) else {}
    user_prompt = str(payload.get("prompt") or "").strip()
    deidentified = payload.get("deidentified_input")
    if not deidentified:
        deidentified = {"prompt": user_prompt[:1000]}

    task = str(payload.get("task") or "general").strip() or "general"
    top_k = int(payload.get("top_k") or 5)
    retrieval_query = str(
        payload.get("retrieval_query")
        or payload.get("query")
        or user_prompt
        or ""
    ).strip()
    query_vector = payload.get("query_vector")
    if isinstance(query_vector, list) and query_vector:
        qv = [float(x) for x in query_vector]
    else:
        qv = None

    steering_vectors = retrieve_steering_for_inference(
        user_id=int(request.user.id),
        query_text=retrieval_query or None,
        query_vector=qv,
        top_k=top_k,
    )

    generation = payload.get("generation") if isinstance(payload.get("generation"), dict) else None
    response_format = payload.get("response_format") if isinstance(payload.get("response_format"), dict) else None
    if task == "pdf_guidance" and response_format is None:
        response_format = PDF_GUIDANCE_RESPONSE_FORMAT

    out = call_llama_inference(
        task=task,
        input_payload=deidentified if isinstance(deidentified, dict) else {"prompt": user_prompt[:1000]},
        steering_vectors=steering_vectors,
        generation=generation,
        response_format=response_format,
        timeout_seconds=30,
    )
    if not out:
        return Response({"detail": "LLM inference unavailable (check LLM_INFERENCE_URL)"}, status=503)

    return Response(
        {
            "output": out,
            "steering_used": len(steering_vectors),
            "task": task,
            "retrieval_query": retrieval_query or None,
        }
    )
