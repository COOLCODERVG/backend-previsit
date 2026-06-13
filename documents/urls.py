from django.urls import path
from . import views


urlpatterns = [
    path("exports", views.list_exports),
    path("exports/<int:pk>/download", views.export_download_url),
]

