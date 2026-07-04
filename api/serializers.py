from rest_framework import serializers
from .models import User, Appointment, Symptom, Feeling, Question, Note, Recording, PersonalizationProfile


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'name', 'email', 'role']


class AppointmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Appointment
        fields = ['id', 'doctor_name', 'specialty', 'location', 'appointment_date',
                  'appointment_time', 'notes', 'is_completed', 'created_at', 'updated_at']


class AppointmentCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Appointment
        fields = ['doctor_name', 'specialty', 'location', 'appointment_date',
                  'appointment_time', 'notes']


class SymptomSerializer(serializers.ModelSerializer):
    appointment_id = serializers.IntegerField(source='appointment.id', read_only=True)

    class Meta:
        model = Symptom
        fields = ['id', 'appointment_id', 'name', 'severity', 'timing', 'duration',
                  'is_new', 'is_worsening', 'notes', 'created_at', 'updated_at']


class SymptomCreateSerializer(serializers.Serializer):
    appointment_id = serializers.IntegerField()
    name = serializers.CharField(max_length=255)
    severity = serializers.IntegerField(min_value=1, max_value=10)
    timing = serializers.CharField(max_length=255, required=False, allow_null=True, allow_blank=True)
    duration = serializers.CharField(max_length=255, required=False, allow_null=True, allow_blank=True)
    is_new = serializers.BooleanField(default=True)
    is_worsening = serializers.BooleanField(default=False)
    notes = serializers.CharField(required=False, allow_null=True, allow_blank=True)


class FeelingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feeling
        fields = ['id', 'health_score', 'energy_level', 'mood', 'notes', 'date',
                  'created_at', 'updated_at']


class FeelingCreateSerializer(serializers.Serializer):
    health_score = serializers.IntegerField(min_value=1, max_value=10)
    energy_level = serializers.IntegerField(min_value=1, max_value=10)
    mood = serializers.CharField(max_length=50, required=False, allow_null=True, allow_blank=True)
    notes = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    date = serializers.CharField(max_length=10, required=False, allow_null=True)


class QuestionSerializer(serializers.ModelSerializer):
    appointment_id = serializers.IntegerField(source='appointment.id', read_only=True)

    class Meta:
        model = Question
        fields = ['id', 'appointment_id', 'text', 'priority', 'is_answered',
                  'answer_notes', 'created_at', 'updated_at']


class QuestionCreateSerializer(serializers.Serializer):
    appointment_id = serializers.IntegerField()
    text = serializers.CharField()
    priority = serializers.IntegerField(default=1)
    is_answered = serializers.BooleanField(default=False)
    answer_notes = serializers.CharField(required=False, allow_null=True, allow_blank=True)


class NoteSerializer(serializers.ModelSerializer):
    appointment_id = serializers.IntegerField(source='appointment.id', read_only=True)

    class Meta:
        model = Note
        fields = ['id', 'appointment_id', 'title', 'content', 'category',
                  'created_at', 'updated_at']


class NoteCreateSerializer(serializers.Serializer):
    appointment_id = serializers.IntegerField()
    title = serializers.CharField(max_length=255)
    content = serializers.CharField()
    category = serializers.CharField(max_length=50, default='general')


class RecordingSerializer(serializers.ModelSerializer):
    appointment_id = serializers.IntegerField(source='appointment.id', read_only=True)
    expires_at = serializers.DateTimeField(read_only=True)
    is_expired = serializers.BooleanField(read_only=True)

    class Meta:
        model = Recording
        fields = [
            'id',
            'appointment_id',
            'title',
            'duration_seconds',
            'status',
            'audio_storage',
            'audio_object_key',
            'audio_content_type',
            'audio_size_bytes',
            'created_at',
            'updated_at',
            'expires_at',
            'is_expired',
        ]


class RecordingDetailSerializer(serializers.ModelSerializer):
    appointment_id = serializers.IntegerField(source='appointment.id', read_only=True)
    expires_at = serializers.DateTimeField(read_only=True)
    is_expired = serializers.BooleanField(read_only=True)

    class Meta:
        model = Recording
        fields = [
            'id',
            'appointment_id',
            'title',
            'duration_seconds',
            'status',
            'audio_storage',
            'audio_object_key',
            'audio_content_type',
            'audio_size_bytes',
            'audio_sha256',
            'transcript_text',
            'transcript_json',
            'extracted_entities',
            'created_at',
            'updated_at',
            'expires_at',
            'is_expired',
        ]


class RecordingCreateSerializer(serializers.Serializer):
    appointment_id = serializers.IntegerField()
    title = serializers.CharField(max_length=255)
    duration_seconds = serializers.IntegerField(default=0)
    # Backwards-compatible for local dev: allow base64 uploads, but do not store in DB.
    audio_base64 = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    audio_content_type = serializers.CharField(required=False, allow_null=True, allow_blank=True)


class AudioUploadInitSerializer(serializers.Serializer):
    appointment_id = serializers.IntegerField()
    title = serializers.CharField(max_length=255)
    duration_seconds = serializers.IntegerField(default=0)
    content_type = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    size_bytes = serializers.IntegerField(required=False, allow_null=True)


class PersonalizationProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonalizationProfile
        fields = [
            'main_reason',
            'condition_stage',
            'biggest_concern',
            'prepared_items',
            'appointment_outcome',
            'family_history',
            'age',
            'gender',
            'region',
            'ml_preferences',
            'average_appointment_minutes',
            'is_completed',
            'updated_at',
        ]


class PersonalizationProfileUpdateSerializer(serializers.Serializer):
    PREPARED_ITEM_CHOICES = {'notes', 'test_results_labs', 'medications_list', 'nothing_yet'}

    main_reason = serializers.CharField(required=False, allow_blank=True, default='')
    condition_stage = serializers.ChoiceField(
        choices=[choice[0] for choice in PersonalizationProfile.CONDITION_STAGE_CHOICES],
        required=False,
        default='not_sure',
    )
    biggest_concern = serializers.CharField(required=False, allow_blank=True, default='')
    prepared_items = serializers.ListField(child=serializers.CharField(), required=False, allow_empty=True, default=list)
    appointment_outcome = serializers.ChoiceField(
        choices=[choice[0] for choice in PersonalizationProfile.APPOINTMENT_OUTCOME_CHOICES],
        required=False,
        default='heard_understood',
    )
    family_history = serializers.CharField(required=False, allow_blank=True, default='')
    age = serializers.CharField(required=False, allow_blank=True, default='')
    gender = serializers.CharField(required=False, allow_blank=True, default='')
    region = serializers.CharField(required=False, allow_blank=True, default='')
    ml_preferences = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    average_appointment_minutes = serializers.IntegerField(required=False, min_value=5, max_value=240, default=30)

    def validate_ml_preferences(self, value):
        cleaned = []
        for item in value or []:
            if not isinstance(item, dict):
                continue
            q = str(item.get('question') or '').strip()
            a = str(item.get('answer') or '').strip()
            if q:
                cleaned.append({'question': q, 'answer': a})
        return cleaned

    def validate_prepared_items(self, value):
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            return ['nothing_yet']

        invalid = [item for item in cleaned if item not in self.PREPARED_ITEM_CHOICES]
        if invalid:
            raise serializers.ValidationError(f'Invalid prepared item(s): {", ".join(invalid)}')

        unique_items = list(dict.fromkeys(cleaned))
        if 'nothing_yet' in unique_items and len(unique_items) > 1:
            raise serializers.ValidationError('"nothing_yet" cannot be combined with other prepared items')

        return unique_items

    def validate(self, attrs):
        prefs = attrs.get('ml_preferences') or []
        # If ML onboarding answers are present, legacy fields can be soft-defaulted.
        if prefs:
            attrs.setdefault('main_reason', '')
            attrs.setdefault('condition_stage', 'not_sure')
            attrs.setdefault('biggest_concern', '')
            attrs.setdefault('prepared_items', ['nothing_yet'])
            attrs.setdefault('appointment_outcome', 'heard_understood')
        return attrs
