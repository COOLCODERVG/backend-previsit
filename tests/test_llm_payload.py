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


def _sample_summary_with_voice_entities():
    """Sample summary that includes extracted_entities from voice recordings."""
    return {
        'appointment': {'specialty': 'Rheumatology', 'notes': 'Joint pain follow-up'},
        'symptoms': [
            {'name': 'Fatigue', 'severity': 6, 'is_new': False, 'is_worsening': True, 'notes': 'Morning fatigue'},
        ],
        'questions': [
            {'text': 'Medication side effects?', 'is_answered': False, 'priority': 1},
        ],
        'notes': [],
        'feelings': [],
        'recordings': [
            {
                'id': 1,
                'status': 'extracted',
                'duration_seconds': 120,
                'title': 'Voice Dump',
                'transcript_text': 'I have been experiencing severe joint pain especially in my hands and wrists. Taking ibuprofen 400mg twice daily. Also noticed some swelling.',
                'extracted_entities': {
                    'medications': [
                        {
                            'name': 'ibuprofen',
                            'dose': '400mg',
                            'frequency': 'twice daily',
                            'route': 'oral',
                            'rxnorm_code': '5640',
                            'confidence': 0.92,
                        }
                    ],
                    'symptoms': [
                        {
                            'name': 'joint pain',
                            'severity': 'severe',
                            'confidence': 0.88,
                        },
                        {
                            'name': 'swelling',
                            'confidence': 0.85,
                        }
                    ],
                    'conditions': [
                        {
                            'name': 'joint pain',
                            'icd10_code': 'M25.5',
                            'confidence': 0.87,
                        }
                    ],
                    'instructions': [
                        {
                            'text': 'Monitor swelling progression',
                            'confidence': 0.78,
                        }
                    ],
                }
            }
        ],
        'transcript_snippets': [],
        'personalization_profile': {
            'main_reason': 'Manage joint inflammation',
            'family_history': '',
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
    assert payload['transcript_dump']
    assert 'combined_text' in payload['transcript_dump']
    assert payload['user_preferences']
    assert payload['voice_transcript'] == payload['transcript_dump']['combined_text']
    assert payload['visit_data']['personalization']['main_reason'] == 'Follow-up fatigue'


def test_build_deidentified_llm_payload_drops_unsafe_preferences():
    payload = _build_deidentified_llm_payload(_sample_summary(), {'patient_name': 'Jane Doe', 'layout_style': 'compact'}, user_id=None)
    assert payload['user_preferences']['layout_style'] == 'compact'
    assert 'patient_name' not in payload['user_preferences']


def test_build_deidentified_llm_payload_with_voice_appointment_fields():
    """Verify voice extracted_entities are transformed into structured voice_appointment_fields."""
    payload = _build_deidentified_llm_payload(_sample_summary_with_voice_entities(), user_id=None)
    
    # Check that voice_appointment_fields exists at top level
    assert 'voice_appointment_fields' in payload
    vaf = payload['voice_appointment_fields']
    
    # Verify structure
    assert vaf['has_structured_data'] is True
    assert len(vaf['recording_ids']) == 1
    assert vaf['recording_ids'][0] == 1
    
    # Verify medications are extracted and de-identified
    assert len(vaf['medications']) == 1
    med = vaf['medications'][0]
    assert med['name'] == 'ibuprofen'
    assert med['dose'] == '400mg'
    assert med['frequency'] == 'twice daily'
    assert 'confidence' in med
    assert med['confidence'] == 0.92
    
    # Verify symptoms are extracted
    assert len(vaf['symptoms']) == 2
    assert vaf['symptoms'][0]['name'] == 'joint pain'
    assert vaf['symptoms'][0]['severity'] == 'severe'
    
    # Verify conditions are extracted
    assert len(vaf['conditions']) == 1
    assert vaf['conditions'][0]['name'] == 'joint pain'
    assert vaf['conditions'][0]['icd10_code'] == 'M25.5'
    
    # Verify instructions are extracted
    assert len(vaf['instructions']) == 1
    assert 'Monitor' in vaf['instructions'][0]['text']
    
    # Verify voice_appointment_fields is also in visit_data
    assert 'voice_appointment_fields' in payload['visit_data']
    
    # Verify signal for LLM that structured data is available
    assert payload['visit_data']['signals']['has_voice_appointment_fields'] is True


def test_build_deidentified_llm_payload_without_voice_entities():
    """Verify empty voice_appointment_fields when no extracted_entities present."""
    payload = _build_deidentified_llm_payload(_sample_summary(), user_id=None)
    
    # Should still have voice_appointment_fields but with no data
    assert 'voice_appointment_fields' in payload
    vaf = payload['voice_appointment_fields']
    assert vaf['has_structured_data'] is False
    assert len(vaf['medications']) == 0
    assert len(vaf['symptoms']) == 0
    assert len(vaf['conditions']) == 0
    assert len(vaf['instructions']) == 0
