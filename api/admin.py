from django.contrib import admin
from .models import User, Appointment, Symptom, Feeling, Question, Note, Recording

admin.site.register(User)
admin.site.register(Appointment)
admin.site.register(Symptom)
admin.site.register(Feeling)
admin.site.register(Question)
admin.site.register(Note)
admin.site.register(Recording)
