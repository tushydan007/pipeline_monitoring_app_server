from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, UserProfileViewSet, UserActivityViewSet

router = DefaultRouter()
router.register(r"users", UserViewSet, basename="user")
router.register(r"profiles", UserProfileViewSet, basename="profile")
router.register(r"activities", UserActivityViewSet, basename="activity")

urlpatterns = [
    path("", include(router.urls)),
]
