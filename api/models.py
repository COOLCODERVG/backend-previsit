from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager


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
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recordings')
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='recordings')
    title = models.CharField(max_length=255)
    duration_seconds = models.IntegerField(default=0)
    audio_base64 = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'recordings'
        ordering = ['-created_at']
