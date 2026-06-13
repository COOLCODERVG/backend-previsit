"""Unit tests for de-identified LLM summary payloads (no live API server)."""

import os

import django

# Force SQLite so pytest works without psycopg/postgres (even if backend/.env sets CORE_DATABASE_URL).
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'previsit.settings_force_sqlite')
django.setup()

from api.utils import (  # noqa: E402
    _build_deidentified_llm_payload,
    assess_llm_input_coverage,
)


def _sample_summary():
    return {
        'appointment': {'specialty': 'Cardiology', 'notes': 'Bring prior labs'},
        'symptoms': [
            {'name': 'Fatigue', 'severity': 7, 'is_new': True, 'is_worsening': True, 'notes': 'Worse at night'},
            {'name': 'Joint pain', 'severity': 5, 'is_new': False, 'is_worsening': False, 'notes': ''},
        ],
        'questions': [
            {'text': 'Should I change meds?', 'is_answered': False, 'priority': 2},
            {'text': 'Old question', 'is_answered': True, 'priority': 1},
        ],
        'notes': [{'title': 'Sleep', 'content': 'Only 5 hours most nights'}],
        'feelings': [
            {'date': '2026-05-10', 'mood': 'tired', 'health_score': 4, 'energy_level': 3, 'notes': 'Low day'},
        ],
        'recordings': [
            {'id': 1, 'status': 'uploaded', 'duration_seconds': 42, 'title': 'Visit note', 'transcript_text': ''},
            {'id': 2, 'status': 'transcribed', 'duration_seconds': 90, 'title': 'Symptoms', 'transcript_text': 'Mentioned fatigue and joint stiffness.'},
        ],
        'transcript_snippets': ['Mentioned fatigue and joint stiffness.'],
        'personalization_profile': {
            'main_reason': 'Follow-up fatigue',
            'family_history': 'Mother had RA',
        },
    }


def test_assess_llm_input_coverage_warns_on_missing_transcripts():
    coverage = assess_llm_input_coverage(_sample_summary())
    assert coverage['recording_count'] == 2
    assert coverage['recordings_with_transcript'] == 1
    assert coverage['recordings_pending_transcript'] == 1
    assert coverage['warnings']


def test_build_deidentified_llm_payload_includes_extended_fields():
    payload = _build_deidentified_llm_payload(_sample_summary(), user_id=None)
    assert len(payload['symptoms']) == 2
    assert len(payload['questions']) == 2
    assert any(q['is_answered'] for q in payload['questions'])
    assert payload['feelings']
    assert payload['personalization']['family_history']
    assert payload['transcript_excerpt']
    assert payload['recordings']
    assert payload['transcripts']
