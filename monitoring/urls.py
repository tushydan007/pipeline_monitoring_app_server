from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    PipelineViewSet,
    SatelliteImageViewSet,
    AnalysisViewSet,
    AnomalyViewSet,
    NotificationViewSet,
    UserDeviceViewSet,
    UserViewSet
)

router = DefaultRouter()
router.register(r'pipelines', PipelineViewSet, basename='pipeline')
router.register(r'satellite-images', SatelliteImageViewSet, basename='satelliteimage')
router.register(r'analyses', AnalysisViewSet, basename='analysis')
router.register(r'anomalies', AnomalyViewSet, basename='anomaly')
router.register(r'notifications', NotificationViewSet, basename='notification')
router.register(r'devices', UserDeviceViewSet, basename='device')
router.register(r'users', UserViewSet, basename='user')

urlpatterns = [
    path('', include(router.urls)),
]

