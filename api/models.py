from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

from .recording_retention import compute_expires_at, is_expired as _recording_is_expired


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('Email is required')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('role', 'admin')
        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser):
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=255, default='')
    role = models.CharField(max_length=20, default='user')
    created_at = models.DateTimeField(auto_now_add=True)

    objects = UserManager()
    USERNAME_FIELD = 'email'

    class Meta:
        db_table = 'users'


class PersonalizationProfile(models.Model):
    CONDITION_STAGE_CHOICES = [
        ('new', 'New'),
        ('ongoing', 'Ongoing'),
        ('not_sure', 'Not sure'),
    ]

    APPOINTMENT_OUTCOME_CHOICES = [
        ('clear_diagnosis', 'Clear diagnosis'),
        ('next_steps_plan', 'Next steps / treatment plan'),
        ('tests_or_referrals', 'Tests or referrals'),
        ('heard_understood', 'Just to be heard / understood'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='personalization')
    main_reason = models.TextField(blank=True, default='')
    condition_stage = models.CharField(max_length=20, choices=CONDITION_STAGE_CHOICES, blank=True, default='')
    biggest_concern = models.TextField(blank=True, default='')
    prepared_items = models.JSONField(default=list, blank=True)
    appointment_outcome = models.CharField(
        max_length=30,
        choices=APPOINTMENT_OUTCOME_CHOICES,
        blank=True,
        default='',
    )
    family_history = models.TextField(blank=True, default='')
    # Profile basics (Settings > Profile — Instagram-style structured layout).
    age = models.CharField(max_length=10, blank=True, default='')
    gender = models.CharField(max_length=30, blank=True, default='')
    # IANA timezone identifier (e.g. "America/New_York") representing the user's region.
    region = models.CharField(max_length=64, blank=True, default='')
    # Question/answer rows from ML onboarding (representation-editing preferences).
    # Shape: [{"question": str, "answer": str}, ...]
    ml_preferences = models.JSONField(default=list, blank=True)
    # User-reported average appointment length in minutes (collected during onboarding).
    # Used to compute a dynamic recording-retention window (average duration + buffer)
    # instead of a fixed expiry timer.
    average_appointment_minutes = models.PositiveIntegerField(default=30)
    is_completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'personalization_profiles'


class Appointment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='appointments')
    doctor_name = models.CharField(max_length=255)
    specialty = models.CharField(max_length=255, blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    appointment_date = models.CharField(max_length=10)  # YYYY-MM-DD
    appointment_time = models.CharField(max_length=5)    # HH:MM
    notes = models.TextField(blank=True, null=True)
    is_completed = models.BooleanField(default=False)
    visit_summary_cache = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'appointments'
        ordering = ['appointment_date']


class Symptom(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='symptoms')
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='symptoms')
    name = models.CharField(max_length=255)
    severity = models.IntegerField()  # 1-10
    timing = models.CharField(max_length=255, blank=True, null=True)
    duration = models.CharField(max_length=255, blank=True, null=True)
    is_new = models.BooleanField(default=True)
    is_worsening = models.BooleanField(default=False)
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'symptoms'
        ordering = ['-created_at']


class Feeling(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feelings')
    health_score = models.IntegerField()  # 1-10
    energy_level = models.IntegerField()  # 1-10
    mood = models.CharField(max_length=50, blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    date = models.CharField(max_length=10)  # YYYY-MM-DD
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'feelings'
        ordering = ['-date']


class Question(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='questions')
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='questions')
    text = models.TextField()
    priority = models.IntegerField(default=1)
    is_answered = models.BooleanField(default=False)
    answer_notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'questions'
        ordering = ['priority']


class Note(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notes')
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='appointment_notes')
    title = models.CharField(max_length=255)
    content = models.TextField()
    category = models.CharField(max_length=50, default='general')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'notes'
        ordering = ['-created_at']


class Recording(models.Model):
    STATUS_CHOICES = [
        ('created', 'created'),
        ('uploaded', 'uploaded'),
        ('transcribing', 'transcribing'),
        ('transcribed', 'transcribed'),
        ('extracting', 'extracting'),
        ('extracted', 'extracted'),
        ('failed', 'failed'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recordings')
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='recordings')
    title = models.CharField(max_length=255)
    duration_seconds = models.IntegerField(default=0)
    audio_object_key = models.CharField(max_length=1024, blank=True, default='')
    audio_content_type = models.CharField(max_length=120, blank=True, default='')
    audio_size_bytes = models.BigIntegerField(default=0)
    audio_sha256 = models.CharField(max_length=64, blank=True, default='')
    status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default='created',
        help_text='created|uploaded|transcribing|transcribed|extracting|extracted|failed',
    )
    transcript_text = models.TextField(blank=True, default='')
    transcript_json = models.JSONField(blank=True, default=dict)
    extracted_entities = models.JSONField(blank=True, default=dict)
    # Backwards-compatible local/dev fallback for inline upload payloads.
    audio_base64 = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'recordings'
        ordering = ['-created_at']

    @property
    def _average_appointment_minutes(self) -> int:
        """The recording owner's typical appointment length, used to size the
        dynamic retention window. Falls back to 30 minutes if the user has no
        PersonalizationProfile yet."""
        try:
            return self.user.personalization.average_appointment_minutes
        except PersonalizationProfile.DoesNotExist:
            return 30

    @property
    def expires_at(self):
        """Timestamp after which this recording is purged (audio + row)."""
        return compute_expires_at(self.created_at, self._average_appointment_minutes)

    @property
    def is_expired(self) -> bool:
        return _recording_is_expired(self.created_at, self._average_appointment_minutes)
