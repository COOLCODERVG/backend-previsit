from django.urls import path, include

urlpatterns = [
    path('api/', include('api.urls')),
    path('api/vectors/', include('vectors.urls')),
    path('api/medications/', include('medications.urls')),
    path('api/documents/', include('documents.urls')),
    path('api/push/', include('notifications.urls')),
]
