from django.urls import path
from . import views

urlpatterns = [
    # Health
    path('', views.root_view),
    path('health', views.health_view),

    # Auth
    path('auth/register', views.register_view),
    path('auth/login', views.login_view),
    path('auth/logout', views.logout_view),
    path('auth/me', views.me_view),
    path('me/action-plan', views.action_plan_view),
    path('auth/refresh', views.refresh_view),
    path('auth/oidc-config', views.oidc_config_view),
    path('personalization', views.personalization_view),
    path('personalization/ml-status', views.personalization_ml_status_view),
    path('personalization/ml-refresh', views.personalization_ml_refresh_view),

    # Appointments
    path('appointments', views.appointments_view),
    path('appointments/<int:pk>', views.appointment_detail_view),
    path('appointments/<int:pk>/summary', views.summary_view),
    path('appointments/<int:pk>/generate-one-pager', views.generate_one_pager_view),
    path('appointments/<int:pk>/export-pdf', views.export_summary_pdf_view),
    path('exports/<str:file_id>', views.export_file_download_view),
    path('llm/generate', views.llm_generate_view),

    # Symptoms
    path('symptoms', views.symptoms_view),
    path('symptoms/<int:pk>', views.symptom_detail_view),

    # Feelings
    path('feelings', views.feelings_view),
    path('feelings/date/<str:date_str>', views.feeling_by_date_view),
    path('feelings/<int:pk>', views.feeling_detail_view),

    # Questions
    path('questions', views.questions_view),
    path('questions/<int:pk>', views.question_detail_view),

    # Notes
    path('notes', views.notes_view),
    path('notes/<int:pk>', views.note_detail_view),

    # Recordings
    path('recordings', views.recordings_view),
    path('recordings/<int:pk>', views.recording_detail_view),

    # Audio pipeline (direct-to-S3 + transcription)
    path('audio/uploads', views.audio_upload_init_view),
    path('audio/objects', views.audio_objects_list_view),
    # Deprecated: server-side transcription endpoints removed in favor of on-device recognition
    # Clients should use on-device `expo-speech-recognition` and may POST transcripts to
    # `/audio/transcript` to persist transcripts or request server-side extraction.
    path('audio/transcript', views.audio_receive_transcript_view),
    path('audio/<int:pk>/extract', views.audio_extract_entities_view),
]
