from rest_framework.views import exception_handler
from rest_framework.response import Response
from io import BytesIO
from datetime import datetime
from pathlib import Path
import logging
import os
import re
import json
import uuid

import requests

from .llm_client import call_llama_pdf_guidance, call_llama_visit_one_pager
from .llm_retrieval import build_pdf_guidance_retrieval_query, retrieve_steering_for_inference
from .rag_retrieval import retrieve_rag_context

logger = logging.getLogger(__name__)


def custom_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is not None:
        # Normalize error format to match frontend expectations
        if 'detail' not in response.data:
            errors = []
            for field, messages in response.data.items():
                if isinstance(messages, list):
                    errors.extend([f"{field}: {m}" for m in messages])
                else:
                    errors.append(f"{field}: {messages}")
            response.data = {'detail': '; '.join(errors) if errors else 'Validation error'}
    return response


def persist_export_pdf(pdf_bytes, user_id, filename):
    base_dir = Path(__file__).resolve().parents[1]  # backend/
    export_dir = base_dir / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)

    safe_name = ''.join(ch for ch in (filename or 'summary.pdf') if ch.isalnum() or ch in ('-', '_', '.'))
    if not safe_name.lower().endswith('.pdf'):
        safe_name += '.pdf'

    file_id = f"u{user_id}_{uuid.uuid4().hex}_{safe_name}"
    file_path = export_dir / file_id
    file_path.write_bytes(pdf_bytes)
    return file_id


# Caps for de-identified LLM payloads (higher than legacy 6/5/1200 limits).
_LLM_MAX_SYMPTOMS = 20
_LLM_MAX_QUESTIONS = 20
_LLM_MAX_NOTES = 15
_LLM_MAX_FEELINGS = 14
_LLM_MAX_MEDICATIONS = 15
_LLM_TRANSCRIPT_PER_RECORDING = 2000
_LLM_TRANSCRIPT_TOTAL = 4000


def assess_llm_input_coverage(summary_payload):
    """Report whether visit prep data is available for LLM context (esp. voice)."""
    recordings = summary_payload.get('recordings') or []
    snippets = summary_payload.get('transcript_snippets') or []
    total = len(recordings)
    with_tx = 0
    pending = []
    for rec in recordings:
        tx = (rec.get('transcript_text') or '').strip()
        if tx:
            with_tx += 1
        else:
            pending.append({
                'id': rec.get('id'),
                'status': rec.get('status') or 'unknown',
                'title': _sanitize_for_llm(rec.get('title'), 80) or None,
            })

    warnings = []
    if total and with_tx == 0:
        warnings.append(
            'You have voice recordings but none are transcribed yet. '
            'The AI summary cannot use your recording content until transcription completes.'
        )
    elif pending:
        warnings.append(
            f'{len(pending)} of {total} recording(s) are not transcribed yet; '
            'those clips are omitted from the AI summary.'
        )

    return {
        'recording_count': total,
        'recordings_with_transcript': with_tx,
        'recordings_pending_transcript': len(pending),
        'has_transcript_snippets': bool(snippets),
        'pending_recordings': pending[:10],
        'warnings': warnings,
    }


def _sanitize_for_llm(text, max_len=280):
    value = str(text or '').strip()
    if not value:
        return ''

    # Remove direct identifiers and common contact signatures.
    value = re.sub(r'\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b', '[redacted-email]', value)
    value = re.sub(r'\+?\d[\d\s().-]{7,}\d', '[redacted-phone]', value)
    value = re.sub(r'https?://\S+|www\.\S+', '[redacted-link]', value)
    value = re.sub(r'@\w+', '[redacted-handle]', value)
    value = re.sub(r'\b\d{6,}\b', '[redacted-id]', value)

    if max_len is not None and len(value) > max_len:
        value = value[:max_len].rstrip() + '...'
    return value


def _build_llm_transcript_dump(summary_payload):
    recordings = []
    combined_text = []

    for rec in summary_payload.get('recordings') or []:
        tx = (rec.get('transcript_text') or '').strip()
        if not tx:
            continue

        sanitized = _sanitize_for_llm(tx, None)
        recordings.append({
            'recording_id': rec.get('id'),
            'status': rec.get('status') or '',
            'duration_seconds': rec.get('duration_seconds'),
            'title': _sanitize_for_llm(rec.get('title'), 80),
            'transcript_text': sanitized,
        })
        combined_text.append(sanitized)

    return {
        'recordings': recordings,
        'combined_text': ' '.join(combined_text),
    }


def _build_llm_user_preferences(export_preferences):
    safe_preferences = export_preferences or {}
    return {
        'layout_style': safe_preferences.get('layout_style', 'detailed'),
        'font_size': safe_preferences.get('font_size', 'normal'),
        'date_format': safe_preferences.get('date_format', 'long'),
        'include_personalization': bool(safe_preferences.get('include_personalization', True)),
        'include_sections': safe_preferences.get('include_sections') or {},
    }


def _build_llm_transcript_bundle(summary_payload):
    """Per-recording excerpts (de-identified) plus a combined excerpt for the prompt."""
    recordings = summary_payload.get('recordings') or []
    snippets = summary_payload.get('transcript_snippets') or []
    per_recording = []
    combined_parts = []

    for idx, rec in enumerate(recordings):
        tx = (rec.get('transcript_text') or '').strip()
        if not tx and idx < len(snippets):
            tx = (snippets[idx] or '').strip()
        if not tx:
            continue
        excerpt = _sanitize_for_llm(tx, _LLM_TRANSCRIPT_PER_RECORDING)
        per_recording.append({
            'recording_id': rec.get('id'),
            'status': rec.get('status') or '',
            'duration_seconds': rec.get('duration_seconds'),
            'has_transcript': True,
            'excerpt': excerpt,
        })
        combined_parts.append(excerpt)

    if not combined_parts and snippets:
        combined_parts = [
            _sanitize_for_llm(s, _LLM_TRANSCRIPT_PER_RECORDING) for s in snippets if s
        ]

    combined = _sanitize_for_llm(' '.join(combined_parts), _LLM_TRANSCRIPT_TOTAL)
    return per_recording, combined


def _build_recording_llm_metadata(summary_payload):
    """Recording metadata for the LLM (no audio); includes extraction hints when present."""
    rows = []
    for rec in summary_payload.get('recordings') or []:
        entities = rec.get('extracted_entities') if isinstance(rec.get('extracted_entities'), dict) else {}
        instructions = entities.get('provider_instructions') or entities.get('instructions') or []
        if isinstance(instructions, list):
            instructions = [
                _sanitize_for_llm(x, 160) for x in instructions[:5] if x
            ]
        rows.append({
            'recording_id': rec.get('id'),
            'status': rec.get('status') or '',
            'duration_seconds': rec.get('duration_seconds'),
            'has_transcript': bool((rec.get('transcript_text') or '').strip()),
            'title': _sanitize_for_llm(rec.get('title'), 80),
            'provider_instruction_hints': instructions,
        })
    return rows[:15]


def _build_voice_appointment_fields(summary_payload):
    """Extract structured appointment fields from voice dump recordings.
    
    Returns a dict with extracted entities from voice transcripts:
    {
        'medications': [...],
        'symptoms': [...],
        'conditions': [...],
        'instructions': [...],
        'recording_ids': [...]
    }
    
    This represents the voice dump as structured appointment data rather than
    unstructured notes, ensuring the LLM treats voice input as clinical facts.
    """
    medications = []
    symptoms = []
    conditions = []
    instructions = []
    recording_ids = []
    
    for rec in summary_payload.get('recordings') or []:
        entities = rec.get('extracted_entities') if isinstance(rec.get('extracted_entities'), dict) else {}
        if not entities:
            continue
            
        recording_ids.append(rec.get('id'))
        
        # Extract medications (already structured by clinical_extraction.py)
        for med in entities.get('medications') or []:
            if not isinstance(med, dict):
                continue
            medications.append({
                'name': _sanitize_for_llm(med.get('name'), 80),
                'dose': _sanitize_for_llm(med.get('dose'), 60),
                'frequency': _sanitize_for_llm(med.get('frequency'), 60),
                'route': _sanitize_for_llm(med.get('route'), 40),
                'rxnorm_code': med.get('rxnorm_code', ''),
                'confidence': float(med.get('confidence') or 0.0),
            })
        
        # Extract symptoms
        for sym in entities.get('symptoms') or []:
            if not isinstance(sym, dict):
                continue
            symptoms.append({
                'name': _sanitize_for_llm(sym.get('name'), 80),
                'severity': _sanitize_for_llm(sym.get('severity'), 20),
                'confidence': float(sym.get('confidence') or 0.0),
            })
        
        # Extract conditions
        for cond in entities.get('conditions') or []:
            if not isinstance(cond, dict):
                continue
            conditions.append({
                'name': _sanitize_for_llm(cond.get('name'), 80),
                'icd10_code': cond.get('icd10_code', ''),
                'confidence': float(cond.get('confidence') or 0.0),
            })
        
        # Extract instructions/procedures
        for instr in entities.get('instructions') or []:
            if not isinstance(instr, dict):
                continue
            instructions.append({
                'text': _sanitize_for_llm(instr.get('text'), 160),
                'confidence': float(instr.get('confidence') or 0.0),
            })
    
    # Cap results
    return {
        'medications': medications[:_LLM_MAX_MEDICATIONS],
        'symptoms': symptoms[:_LLM_MAX_SYMPTOMS],
        'conditions': conditions[:10],
        'instructions': instructions[:10],
        'recording_ids': recording_ids,
        'has_structured_data': bool(recording_ids),
    }


def _fetch_active_medications_for_llm(user_id):
    if not user_id:
        return []
    try:
        from medications.models import Medication

        meds = Medication.objects.filter(user_id=user_id, status='active').order_by('-updated_at')[:_LLM_MAX_MEDICATIONS]
        return [
            {
                'name': _sanitize_for_llm(m.name, 120),
                'dose': _sanitize_for_llm(m.dose, 80),
                'frequency': _sanitize_for_llm(m.frequency, 80),
                'route': _sanitize_for_llm(m.route, 40),
            }
            for m in meds
        ]
    except Exception:
        return []


def _build_deidentified_llm_payload(summary_payload, export_preferences=None, user_id=None):
    preferences = export_preferences or {}
    appointment = summary_payload.get('appointment') or {}
    symptoms = summary_payload.get('symptoms') or []
    questions = summary_payload.get('questions') or []
    notes = summary_payload.get('notes') or []
    feelings = summary_payload.get('feelings') or []
    personalization = summary_payload.get('personalization_profile') or {}

    def symptom_items():
        ranked = sorted(
            symptoms,
            key=lambda s: (not bool(s.get('is_worsening')), -int(s.get('severity') or 0)),
        )
        return [
            {
                'name': _sanitize_for_llm(item.get('name'), 80),
                'severity': int(item.get('severity') or 0),
                'is_new': bool(item.get('is_new')),
                'is_worsening': bool(item.get('is_worsening')),
                'notes': _sanitize_for_llm(item.get('notes'), 280),
            }
            for item in ranked[:_LLM_MAX_SYMPTOMS]
        ]

    def question_items():
        ranked = sorted(questions, key=lambda q: (bool(q.get('is_answered')), -int(q.get('priority') or 0)))
        return [
            {
                'text': _sanitize_for_llm(q.get('text'), 280),
                'is_answered': bool(q.get('is_answered')),
                'priority': int(q.get('priority') or 0),
            }
            for q in ranked[:_LLM_MAX_QUESTIONS]
        ]

    def feeling_items():
        recent = sorted(feelings, key=lambda f: f.get('date') or '', reverse=True)
        return [
            {
                'date': f.get('date') or '',
                'mood': _sanitize_for_llm(f.get('mood'), 40),
                'health_score': f.get('health_score'),
                'energy_level': f.get('energy_level'),
                'notes': _sanitize_for_llm(f.get('notes'), 200),
            }
            for f in recent[:_LLM_MAX_FEELINGS]
        ]

    avg_health = None
    avg_energy = None
    if feelings:
        avg_health = round(sum((f.get('health_score') or 0) for f in feelings) / len(feelings), 1)
        avg_energy = round(sum((f.get('energy_level') or 0) for f in feelings) / len(feelings), 1)

    transcript_rows, transcript_combined = _build_llm_transcript_bundle(summary_payload)
    transcript_dump = _build_llm_transcript_dump(summary_payload)
    voice_appointment_fields = _build_voice_appointment_fields(summary_payload)
    family_history = _sanitize_for_llm(
        (personalization.get('family_history') if personalization else None) or '',
        500,
    )

    visit_data = {
        'context': {
            # Explicitly no patient name/email/id, no doctor name, no location.
            'appointment_specialty': _sanitize_for_llm(appointment.get('specialty'), 80),
            'has_appointment_notes': bool(appointment.get('notes')),
            'appointment_notes': _sanitize_for_llm(appointment.get('notes'), 400),
        },
        'personalization': {
            'main_reason': _sanitize_for_llm(personalization.get('main_reason'), 280),
            'condition_stage': personalization.get('condition_stage') or '',
            'biggest_concern': _sanitize_for_llm(personalization.get('biggest_concern'), 280),
            'prepared_items': [str(v) for v in (personalization.get('prepared_items') or [])],
            'appointment_outcome': personalization.get('appointment_outcome') or '',
            'family_history': family_history,
            'ml_preferences': [
                {
                    'question': _sanitize_for_llm((item or {}).get('question'), 220),
                    'answer': _sanitize_for_llm((item or {}).get('answer'), 280),
                }
                for item in (personalization.get('ml_preferences') or [])
                if isinstance(item, dict)
            ][:20],
        },
        'signals': {
            'symptom_count': len(symptoms),
            'worsening_count': len([s for s in symptoms if s.get('is_worsening')]),
            'question_count': len(questions),
            'unanswered_question_count': len([q for q in questions if not q.get('is_answered')]),
            'note_count': len(notes),
            'wellbeing_entries': len(feelings),
            'avg_health_score': avg_health,
            'avg_energy_level': avg_energy,
            'recording_count': len(summary_payload.get('recordings') or []),
            'recordings_with_transcript': len(transcript_rows),
            'has_voice_appointment_fields': voice_appointment_fields.get('has_structured_data', False),
        },
        'symptoms': symptom_items(),
        'top_symptoms': symptom_items(),
        'questions': question_items(),
        'top_questions': [
            q['text'] for q in question_items() if not q.get('is_answered')
        ][:12],
        'notes': [
            {
                'title': _sanitize_for_llm(n.get('title'), 100),
                'content': _sanitize_for_llm(n.get('content'), 400),
            }
            for n in notes[:_LLM_MAX_NOTES]
        ],
        'top_notes': [
            {
                'title': _sanitize_for_llm(n.get('title'), 100),
                'content': _sanitize_for_llm(n.get('content'), 400),
            }
            for n in notes[:_LLM_MAX_NOTES]
        ],
        'feelings': feeling_items(),
        'medications': _fetch_active_medications_for_llm(user_id),
        'recordings': _build_recording_llm_metadata(summary_payload),
        'transcripts': transcript_rows,
        'transcript_dump': transcript_dump,
        'voice_appointment_fields': voice_appointment_fields,
    }
    if transcript_combined:
        visit_data['transcript_excerpt'] = transcript_combined

    deidentified = {
        'user_preferences': _build_llm_user_preferences(preferences),
        'visit_data': visit_data,
        'voice_transcript': transcript_dump['combined_text'],
        'transcript_dump': transcript_dump,
        'voice_appointment_fields': voice_appointment_fields,
        **visit_data,
    }
    return deidentified


def _normalize_one_pager(raw, summary_payload, view_mode="standard"):
    appointment = summary_payload.get("appointment") or {}
    personalization = summary_payload.get("personalization_profile") or {}
    symptoms = summary_payload.get("symptoms") or []
    questions = [q for q in (summary_payload.get("questions") or []) if not q.get("is_answered")]

    if not isinstance(raw, dict):
        raw = {}

    headline = _sanitize_for_llm(
        raw.get("headline") or f"Visit prep: {appointment.get('specialty') or 'your appointment'}",
        120,
    )
    focus = _sanitize_for_llm(
        raw.get("focus_summary") or personalization.get("main_reason") or "Your visit at a glance.",
        500,
    )
    takeaways = [_sanitize_for_llm(v, 200) for v in (raw.get("key_takeaways") or [])[:6] if v]
    if not takeaways and symptoms:
        takeaways = [
            _sanitize_for_llm(f"{s.get('name')} (severity {s.get('severity')}/10)", 200)
            for s in symptoms[:3]
        ]

    action_items = [_sanitize_for_llm(v, 200) for v in (raw.get("action_items") or [])[:10] if v]
    if not action_items and questions:
        action_items = [_sanitize_for_llm(f"Ask: {q.get('text')}", 200) for q in questions[:3]]

    doctor_qs = [_sanitize_for_llm(v, 200) for v in (raw.get("questions_for_doctor") or [])[:6] if v]
    if not doctor_qs and questions:
        doctor_qs = [_sanitize_for_llm(q.get("text"), 200) for q in questions[:5]]

    caregiver = _sanitize_for_llm(
        raw.get("caregiver_blurb")
        or f"Upcoming visit focus: {personalization.get('main_reason') or focus[:120]}.",
        400,
    )

    if view_mode == "simplified":
        takeaways = takeaways[:3]
        action_items = action_items[:4]
        focus = focus[:280]
    elif view_mode == "caregiver":
        takeaways = takeaways[:4]

    return {
        "headline": headline,
        "focus_summary": focus,
        "key_takeaways": takeaways,
        "action_items": action_items,
        "questions_for_doctor": doctor_qs,
        "caregiver_blurb": caregiver,
        "view_mode": view_mode,
    }


def generate_visit_one_pager(summary_payload, *, view_mode="standard", user_id=None):
    # Full internal context — used ONLY for building the retrieval query and
    # steering-vector lookup (personalization influences generation via these
    # signals). It is never sent to the model as the prompt input anymore.
    internal_context = _build_deidentified_llm_payload(
        summary_payload,
        {"layout_style": "detailed"},
        user_id=user_id,
    )

    retrieval_query = None
    try:
        retrieval_query = build_pdf_guidance_retrieval_query(internal_context)
    except Exception:
        logger.exception("one_pager retrieval query build failed for user=%s", user_id)

    steering_vectors = []
    try:
        if user_id and retrieval_query:
            steering_vectors = retrieve_steering_for_inference(
                user_id=int(user_id),
                query_text=retrieval_query,
                top_k=5,
            )
    except Exception:
        logger.exception("one_pager steering retrieval failed for user=%s", user_id)
        steering_vectors = []

    # Minimal payload actually sent to the LLM: appointment notes, symptoms,
    # questions, and a compact preferences object — never the raw DB blob.
    lean_input = build_lean_llm_generation_input(summary_payload)
    lean_input["view_mode"] = view_mode

    # Knowledge-base RAG context (machinelearning/rag/pipeline.py via the embedding
    # service) -- independent of personalization steering and of user_id, since the
    # knowledge base (appointment prep FAQ/medications/notes) isn't user-specific.
    try:
        rag = retrieve_rag_context(retrieval_query or "", top_k=5)
        if rag.get("rag_context"):
            lean_input["rag_context"] = rag["rag_context"]
            lean_input["rag_citations"] = rag["rag_citations"]
    except Exception:
        logger.exception("one_pager RAG retrieval failed for user=%s", user_id)

    try:
        raw = call_llama_visit_one_pager(
            deidentified_payload=lean_input,
            steering_vectors=steering_vectors,
            view_mode=view_mode,
            timeout_seconds=60,
        )
    except Exception:
        logger.exception("one_pager LLM call raised unexpected exception for user=%s", user_id)

        raw = None

    source = "llm" if isinstance(raw, dict) and raw.get("headline") else "fallback"
    normalized = _normalize_one_pager(raw if isinstance(raw, dict) else None, summary_payload, view_mode)
    return normalized, source


_VAGUE_QUESTION_MARKERS = {
    'nope', 'no', 'none', 'n/a', 'na', 'nothing', 'no questions', 'not sure',
    'idk', "dont know", "don't know", 'no q', 'no qs', '-', '.',
}


def is_vague_text(text):
    """True when a free-text field is effectively empty/non-informative
    (e.g. "Nope", "N/A", "-"). Used to avoid generating hollow PDF sections
    from placeholder answers."""
    value = str(text or '').strip().lower().strip('.! ')
    if len(value) < 3:
        return True
    return value in _VAGUE_QUESTION_MARKERS


def _infer_communication_style(personalization):
    """Best-effort communication-style preference from onboarding Q&A pairs
    (no dedicated model field exists yet, so this scans ml_preferences)."""
    for item in (personalization.get('ml_preferences') or []):
        if not isinstance(item, dict):
            continue
        question = str(item.get('question') or '').lower()
        if 'communicat' in question or 'style' in question or 'explain' in question:
            answer = str(item.get('answer') or '').strip()
            if answer:
                return _sanitize_for_llm(answer, 120)
    return 'plain-language, empathetic'


_APPOINTMENT_OUTCOME_LABELS = {
    'clear_diagnosis': 'Wants a clear diagnosis',
    'next_steps_plan': 'Wants next steps or a treatment plan',
    'tests_or_referrals': 'Wants tests or referrals',
    'heard_understood': 'Wants to feel heard and understood',
}


def _note_sentiment(content):
    text = str(content or '').lower()
    negative_markers = ('bad', 'worried', 'anxious', 'scared', 'stressed', 'worse', 'terrible', 'awful', 'pain', 'sad')
    positive_markers = ('good', 'fine', 'better', 'great', 'improving', 'okay', 'ok', 'well')
    if any(m in text for m in negative_markers):
        return 'negative'
    if any(m in text for m in positive_markers):
        return 'positive'
    return 'neutral'


def build_llm_context(summary_payload, export_preferences=None):
    """Builds the structured LLM context object (see PART 1 of the export spec):

    {
      appointment: {doctor_name, specialty, date, time, purpose},
      patient_preferences: {communication_style, concerns, goals, preparation_preferences},
      symptoms: [{name, severity, duration, description}],
      questions_for_provider: [{question, priority}],
      notes: [{content, category, sentiment}]
    }

    This is a structured, purpose-built object -- distinct from the legacy
    de-identified `_build_deidentified_llm_payload` blob -- so the LLM prompt
    receives clearly-labelled, personalized fields instead of a raw DB dump.
    """
    appointment = summary_payload.get('appointment') or {}
    personalization = summary_payload.get('personalization_profile') or {}
    symptoms = summary_payload.get('symptoms') or []
    questions = summary_payload.get('questions') or []
    notes = summary_payload.get('notes') or []

    purpose = _sanitize_for_llm(appointment.get('notes'), 300) or _sanitize_for_llm(
        personalization.get('main_reason'), 300
    )

    context = {
        'appointment': {
            'doctor_name': _sanitize_for_llm(appointment.get('doctor_name'), 120),
            'specialty': _sanitize_for_llm(appointment.get('specialty'), 80),
            'date': appointment.get('appointment_date') or '',
            'time': appointment.get('appointment_time') or '',
            'purpose': purpose,
        },
        'patient_preferences': {
            'communication_style': _infer_communication_style(personalization),
            'concerns': _sanitize_for_llm(personalization.get('biggest_concern'), 280),
            'goals': _APPOINTMENT_OUTCOME_LABELS.get(
                personalization.get('appointment_outcome') or '', ''
            ) or _sanitize_for_llm(personalization.get('main_reason'), 200),
            'preparation_preferences': [
                str(v) for v in (personalization.get('prepared_items') or [])
            ],
        },
        'symptoms': [
            {
                'name': _sanitize_for_llm(s.get('name'), 80),
                'severity': s.get('severity'),
                'duration': _sanitize_for_llm(s.get('duration'), 60),
                'description': _sanitize_for_llm(s.get('notes'), 280),
            }
            for s in symptoms[:_LLM_MAX_SYMPTOMS]
        ],
        'questions_for_provider': [
            {
                'question': _sanitize_for_llm(q.get('text'), 280),
                'priority': q.get('priority') or 0,
            }
            for q in questions
            if not q.get('is_answered') and not is_vague_text(q.get('text'))
        ][:_LLM_MAX_QUESTIONS],
        'notes': [
            {
                'content': _sanitize_for_llm(n.get('content'), 400),
                'category': n.get('category') or 'general',
                'sentiment': _note_sentiment(n.get('content')),
            }
            for n in notes[:_LLM_MAX_NOTES]
        ],
    }
    return context


def build_lean_llm_generation_input(summary_payload, *, tone=None):
    """Builds the MINIMAL input payload actually sent to the LLM for content
    generation (visit one-pager / PDF guidance).

    Personalization/profile/DB data still INFLUENCES generation (it's read
    here and folded into `preferences`, plus steering vectors and RAG context
    are retrieved separately using the full internal context), but none of
    the raw DB objects (personalization profile, ml_preferences, signals,
    full symptom/question/note records, transcripts, etc.) are put in front
    of the model. The model only ever sees:

        {
          "appointment_notes": "...",
          "symptoms": [{"name": "...", "severity": 5, "new": true}, ...],
          "questions": ["..."],
          "notes": [{"content": "...", "category": "medication"}, ...],
          "preferences": {"tone": "...", "concerns": "...", "goals": "..."}
        }

    Symptoms are sorted by severity (descending) so the model's "primary
    concern" framing lines up with what the PDF itself highlights first.
    `notes` are capped and trimmed short (grounding for executive-summary
    bullets like "Medication use was mentioned" without pulling in the full
    patient note text) — this keeps the prompt small (faster generation) and
    removes any incentive for the model to echo the input structure back.
    """
    context = build_llm_context(summary_payload)
    appointment = context.get('appointment') or {}
    prefs = context.get('patient_preferences') or {}

    symptoms_sorted = sorted(
        (context.get('symptoms') or []),
        key=lambda s: -(int(s.get('severity') or 0)),
    )

    return {
        'appointment_notes': appointment.get('purpose') or '',
        'symptoms': [
            {
                'name': s.get('name'),
                'severity': s.get('severity'),
                'new': bool(s.get('is_new')),
            }
            for s in symptoms_sorted
            if s.get('name')
        ][:8],
        'questions': [
            q.get('question') for q in (context.get('questions_for_provider') or []) if q.get('question')
        ][:8],
        'notes': [
            {'content': (n.get('content') or '')[:140], 'category': n.get('category') or 'general'}
            for n in (context.get('notes') or [])
            if n.get('content')
        ][:6],
        'preferences': {
            'tone': tone or prefs.get('communication_style') or 'concise',
            'concerns': prefs.get('concerns') or '',
            'goals': prefs.get('goals') or '',
        },
    }


def _log_llm_context_debug(llm_context, summary_payload, user_id=None):
    """Non-sensitive logging: presence/shape only, never raw field values."""
    personalization = summary_payload.get('personalization_profile')
    appointment = llm_context.get('appointment') or {}
    prefs = llm_context.get('patient_preferences') or {}

    logger.info(
        "pdf_guidance context_check user=%s personalization_present=%s "
        "personalization_fields=%s appointment_context_present=%s "
        "symptoms_included=%s questions_included=%s notes_included=%s",
        user_id,
        bool(personalization),
        sorted([k for k, v in prefs.items() if v]) if personalization else [],
        bool(appointment.get('doctor_name') or appointment.get('specialty') or appointment.get('date')),
        len(llm_context.get('symptoms') or []),
        len(llm_context.get('questions_for_provider') or []),
        len(llm_context.get('notes') or []),
    )


def generate_llm_pdf_guidance(summary_payload, export_preferences=None, user_id=None):
    # Full internal context — used ONLY for building the retrieval query and
    # steering-vector lookup / debug logging. Never sent to the model directly.
    internal_context = _build_deidentified_llm_payload(
        summary_payload,
        export_preferences,
        user_id=user_id,
    )

    # Structured, purpose-built context (PART 1): explicit appointment/
    # personalization/symptoms/questions/notes object — used for retrieval
    # query building and debug logging only (not sent to the model as-is).
    llm_context = build_llm_context(summary_payload, export_preferences)
    _log_llm_context_debug(llm_context, summary_payload, user_id=user_id)
    internal_context['structured_context'] = llm_context

    retrieval_query = None
    try:
        retrieval_query = build_pdf_guidance_retrieval_query(internal_context)
    except Exception:
        logger.exception("pdf_guidance retrieval query build failed for user=%s", user_id)

    steering_vectors = []
    try:
        if user_id and retrieval_query:
            steering_vectors = retrieve_steering_for_inference(
                user_id=int(user_id),
                query_text=retrieval_query,
                top_k=5,
            )
    except Exception:
        logger.exception("pdf_guidance steering retrieval failed for user=%s", user_id)
        steering_vectors = []

    # Minimal payload actually sent to the LLM: appointment notes, symptoms,
    # questions, and a compact preferences object — never the raw DB blob.
    tone_pref = (export_preferences or {}).get('tone') if isinstance(export_preferences, dict) else None
    lean_input = build_lean_llm_generation_input(summary_payload, tone=tone_pref)

    try:
        rag = retrieve_rag_context(retrieval_query or "", top_k=5)
        if rag.get("rag_context"):
            lean_input["rag_context"] = rag["rag_context"]
            lean_input["rag_citations"] = rag["rag_citations"]
    except Exception:
        logger.exception("pdf_guidance RAG retrieval failed for user=%s", user_id)

    try:
        guidance = call_llama_pdf_guidance(
            deidentified_payload=lean_input,
            steering_vectors=steering_vectors,
            timeout_seconds=60,
        )

    except Exception:
        logger.exception("pdf_guidance LLM call raised unexpected exception for user=%s", user_id)
        guidance = None

    if not isinstance(guidance, dict):
        return None

    # `suggest_questions` lets the user (via export_preferences / personalization
    # settings) opt out of LLM-suggested discussion topics in the "Questions
    # for Provider" section. Defaults on.
    suggest_questions = True
    if isinstance(export_preferences, dict) and 'suggest_questions' in export_preferences:
        suggest_questions = bool(export_preferences.get('suggest_questions'))

    return {
        'tone': _sanitize_for_llm(guidance.get('tone'), 80),
        'primary_goal': _sanitize_for_llm(guidance.get('primary_goal'), 300),
        'executive_summary': [
            _sanitize_for_llm(v, 160) for v in (guidance.get('executive_summary') or [])[:5] if v
        ],
        'visit_prep_topics': [
            _sanitize_for_llm(v, 160) for v in (guidance.get('visit_prep_topics') or [])[:5] if v
        ],
        'suggested_questions': [
            _sanitize_for_llm(v, 160) for v in (guidance.get('suggested_questions') or [])[:5] if v
        ] if suggest_questions else [],
        'formatting_hints': {
            'include_bullets': bool((guidance.get('formatting_hints') or {}).get('include_bullets', True)),
            'emphasis_words': [
                _sanitize_for_llm(v, 40)
                for v in ((guidance.get('formatting_hints') or {}).get('emphasis_words') or [])[:6]
            ],
        },
    }


def build_custom_summary_pdf(summary_payload, export_preferences=None, ai_guidance=None):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise RuntimeError('PDF generation dependency missing: reportlab') from exc

    preferences = export_preferences or {}

    layout_style = preferences.get('layout_style', 'detailed')
    font_size_key = preferences.get('font_size', 'normal')
    include_personalization = preferences.get('include_personalization', True)
    include_sections = preferences.get('include_sections') or {}
    date_format_pref = preferences.get('date_format', 'long')

    font_size = {'small': 9, 'normal': 10, 'large': 11}.get(font_size_key, 10)
    line_height = 14 if layout_style == 'compact' else 16

    appointment = summary_payload.get('appointment') or {}
    symptoms = summary_payload.get('symptoms') or []
    questions = summary_payload.get('questions') or []
    notes = summary_payload.get('notes') or []
    feelings = summary_payload.get('feelings') or []
    personalization = summary_payload.get('personalization_profile') or {}

    appointment_outcome = personalization.get('appointment_outcome', '')
    section_order_by_outcome = {
        'clear_diagnosis': ['focus', 'symptoms', 'questions', 'wellbeing', 'notes'],
        'next_steps_plan': ['focus', 'questions', 'symptoms', 'notes', 'wellbeing'],
        'tests_or_referrals': ['focus', 'symptoms', 'notes', 'questions', 'wellbeing'],
        'heard_understood': ['focus', 'notes', 'symptoms', 'questions', 'wellbeing'],
    }
    ordered_sections = section_order_by_outcome.get(
        appointment_outcome,
        ['focus', 'symptoms', 'questions', 'wellbeing', 'notes'],
    )

    def allows(section_name):
        if section_name not in include_sections:
            return True
        return bool(include_sections.get(section_name))

    def format_date(value):
        if not value:
            return '—'
        try:
            dt = datetime.strptime(value, '%Y-%m-%d')
            if date_format_pref == 'short':
                return dt.strftime('%m/%d/%Y')
            return dt.strftime('%B %d, %Y')
        except Exception:
            return value

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    page_width, page_height = letter
    margin_x = 54
    margin_top = page_height - 54
    margin_bottom = 54
    content_width = page_width - (margin_x * 2)
    y = margin_top

    def ensure_space(required_height):
        nonlocal y
        if y - required_height < margin_bottom:
            draw_footer()
            pdf.showPage()
            y = margin_top
            pdf.setFont('Helvetica', font_size)

    def wrap_text(text, max_width, font_name='Helvetica', size=10):
        text = str(text or '').strip()
        if not text:
            return ['—']
        words = text.split()
        lines = []
        current = ''
        for word in words:
            candidate = f"{current} {word}".strip()
            if pdfmetrics.stringWidth(candidate, font_name, size) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or ['—']

    def draw_title(text):
        nonlocal y
        ensure_space(32)
        pdf.setFont('Helvetica-Bold', 16)
        pdf.drawString(margin_x, y, text)
        y -= 22

    def draw_subtitle(text):
        nonlocal y
        ensure_space(18)
        pdf.setFont('Helvetica', font_size)
        pdf.drawString(margin_x, y, text)
        y -= line_height

    def draw_section(title):
        nonlocal y
        ensure_space(24)
        pdf.setFont('Helvetica-Bold', 11)
        pdf.drawString(margin_x, y, title.upper())
        y -= 8
        pdf.line(margin_x, y, margin_x + content_width, y)
        y -= 12

    def draw_paragraph(text, bullet=False, italic=False):
        nonlocal y
        prefix = '• ' if bullet else ''
        font_name = 'Helvetica-Oblique' if italic else 'Helvetica'
        lines = wrap_text(f"{prefix}{text}", content_width, font_name, font_size)
        ensure_space(len(lines) * line_height + 4)
        pdf.setFont(font_name, font_size)
        for line in lines:
            pdf.drawString(margin_x, y, line)
            y -= line_height

    def draw_footer():
        pdf.setFont('Helvetica', 8)
        pdf.line(margin_x, 40, margin_x + content_width, 40)
        pdf.drawString(margin_x, 28, 'Generated by SyniVia · Not medical advice · SyniVia not liable for errors/omissions')
        if ai_guidance:
            pdf.drawString(margin_x, 18, 'Personalized using SyniVia with de-identified inputs')

    draw_title('SYNIVIA SUMMARY PDF')
    draw_subtitle(f"Doctor: {appointment.get('doctor_name') or '—'}")
    if appointment.get('specialty'):
        draw_subtitle(f"Specialty: {appointment.get('specialty')}")
    draw_subtitle(f"Appointment Date: {format_date(appointment.get('appointment_date'))}")
    draw_subtitle(f"Appointment Time: {appointment.get('appointment_time') or '—'}")
    if appointment.get('location'):
        draw_subtitle(f"Location: {appointment.get('location')}")
    y -= 6

    if include_personalization and allows('personalization'):
        draw_section('Personalization Profile')
        draw_paragraph(f"Main health focus: {personalization.get('main_reason') or '—'}")
        draw_paragraph(f"Condition stage: {personalization.get('condition_stage') or '—'}")
        draw_paragraph(f"Biggest concern: {personalization.get('biggest_concern') or '—'}")
        prepared = personalization.get('prepared_items') or []
        prepared_text = ', '.join(prepared) if prepared else '—'
        draw_paragraph(f"Prepared items: {prepared_text}")
        draw_paragraph(f"Desired appointment outcome: {personalization.get('appointment_outcome') or '—'}")
        y -= 4

    if ai_guidance and allows('focus'):
        draw_section(ai_guidance.get('focus_header') or 'Personalized Focus Brief')
        draw_paragraph(ai_guidance.get('focus_summary') or 'No AI focus summary generated.', italic=not bool(ai_guidance.get('focus_summary')))

        for item in (ai_guidance.get('discussion_points') or []):
            draw_paragraph(item, bullet=True)

        if ai_guidance.get('clinician_questions'):
            y -= 2
            draw_paragraph('Suggested clinician-facing clarifications:', italic=True)
            for q in ai_guidance.get('clinician_questions'):
                draw_paragraph(q, bullet=True)

        y -= 4

    for section in ordered_sections:
        if not allows(section):
            continue

        if section == 'focus':
            draw_section('Visit Focus')
            focus = personalization.get('main_reason') or appointment.get('notes') or 'No focus provided.'
            draw_paragraph(focus)
            concern = personalization.get('biggest_concern')
            if concern:
                draw_paragraph(f"Priority concern: {concern}", italic=True)

        elif section == 'symptoms':
            draw_section(f"Symptoms ({len(symptoms)})")
            if not symptoms:
                draw_paragraph('No symptoms recorded.', italic=True)
            else:
                sorted_symptoms = sorted(
                    symptoms,
                    key=lambda s: (not bool(s.get('is_worsening')), -int(s.get('severity') or 0)),
                )
                items = sorted_symptoms[:6] if layout_style == 'compact' else sorted_symptoms
                for symptom in items:
                    name = symptom.get('name') or 'Unnamed symptom'
                    severity = symptom.get('severity') or '—'
                    flags = []
                    if symptom.get('is_new'):
                        flags.append('new')
                    if symptom.get('is_worsening'):
                        flags.append('worsening')
                    flag_text = f" [{', '.join(flags)}]" if flags else ''
                    draw_paragraph(f"{name} — severity {severity}/10{flag_text}", bullet=True)
                    if layout_style != 'compact' and symptom.get('notes'):
                        draw_paragraph(f"Notes: {symptom.get('notes')}")

        elif section == 'questions':
            unanswered = [q for q in questions if not q.get('is_answered')]
            draw_section(f"Patient Questions ({len(unanswered)})")
            if not unanswered:
                draw_paragraph('No pending questions.', italic=True)
            else:
                items = unanswered[:6] if layout_style == 'compact' else unanswered
                for q in items:
                    draw_paragraph(q.get('text') or '—', bullet=True)

        elif section == 'wellbeing':
            draw_section('Recent Wellbeing')
            if not feelings:
                draw_paragraph('No recent wellbeing check-ins.', italic=True)
            else:
                avg_health = round(sum((f.get('health_score') or 0) for f in feelings) / len(feelings), 1)
                avg_energy = round(sum((f.get('energy_level') or 0) for f in feelings) / len(feelings), 1)
                draw_paragraph(f"Average health score: {avg_health}/10")
                draw_paragraph(f"Average energy level: {avg_energy}/10")

        elif section == 'notes':
            draw_section(f"Notes ({len(notes)})")
            if not notes:
                draw_paragraph('No notes available.', italic=True)
            else:
                items = notes[:4] if layout_style == 'compact' else notes
                for n in items:
                    draw_paragraph(f"{n.get('title') or 'Note'}: {n.get('content') or '—'}", bullet=True)

        y -= 4

    draw_footer()
    pdf.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
