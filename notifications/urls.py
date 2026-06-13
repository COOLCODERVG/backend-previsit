from django.urls import path
from . import views

urlpatterns = [
    path("devices", views.push_devices_view),
    path("devices/<str:token>", views.push_device_detail_view),
]
