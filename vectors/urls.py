from django.urls import path
from . import views


urlpatterns = [
    path("contexts", views.upsert_preference_context),
    path("steering", views.add_steering_vector),
    path("search", views.search_preference_contexts),
]

