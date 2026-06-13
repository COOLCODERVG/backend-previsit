from django.urls import path
from . import views


urlpatterns = [
    path("", views.medications_view),
    path("<int:pk>", views.medication_detail_view),
    path("reminders", views.medication_reminders_view),
    path("reminders/<int:pk>", views.medication_reminder_detail_view),
]

