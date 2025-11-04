from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from djoser.views import UserViewSet as DjoserUserViewSet
from django.contrib.auth.models import User
from .models import UserProfile, UserPreference, UserActivity
from .serializers import (
    UserSerializer,
    UserProfileSerializer,
    UserPreferenceSerializer,
    UserActivitySerializer
)


class UserViewSet(DjoserUserViewSet):
    """
    Extended Djoser UserViewSet with additional user management endpoints
    All endpoints follow Djoser conventions and use custom serializers with profile data
    """
    
    def get_serializer(self, *args, **kwargs):
        """Override to skip serializer for DELETE requests"""
        if self.action == 'me' and hasattr(self.request, 'method') and self.request.method == 'DELETE':
            # Skip serializer validation for DELETE - we handle it directly in the action
            # Return a dummy serializer that doesn't validate
            class DummySerializer:
                def __init__(self, *args, **kwargs):
                    pass
                def is_valid(self, *args, **kwargs):
                    return True
            return DummySerializer()
        return super().get_serializer(*args, **kwargs)
    
    @action(
        detail=False,
        methods=['get', 'put', 'patch', 'delete'],
        permission_classes=[IsAuthenticated],
        url_path='me',
        url_name='me'
    )
    def me(self, request):
        """
        Get, update, or delete current user profile
        This is already provided by Djoser, but we're ensuring it works correctly
        """
        if request.method == 'GET':
            serializer = self.get_serializer(request.user)
            return Response(serializer.data, status=status.HTTP_200_OK)
        elif request.method in ['PUT', 'PATCH']:
            # Get the user instance
            user = request.user
            
            # Update User model fields directly
            if 'username' in request.data:
                user.username = request.data['username']
            if 'email' in request.data:
                # Validate email uniqueness
                existing_user = User.objects.filter(email__iexact=request.data['email']).exclude(pk=user.pk).first()
                if existing_user:
                    return Response(
                        {'email': ['A user with this email already exists.']},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                user.email = request.data['email'].lower()
            if 'first_name' in request.data:
                user.first_name = request.data['first_name']
            if 'last_name' in request.data:
                user.last_name = request.data['last_name']
            
            # Save the user
            user.save()
            
            # Return updated serializer data
            serializer = self.get_serializer(user)
            return Response(serializer.data, status=status.HTTP_200_OK)
        elif request.method == 'DELETE':
            # Delete current user account
            # Bypass Djoser's serializer validation by handling DELETE directly
            try:
                user = request.user
                username = user.username
                user_email = user.email
                
                # Log the deletion activity before deleting
                try:
                    UserActivity.objects.create(
                        user=user,
                        activity_type='account_deletion',
                        ip_address=self._get_client_ip(request),
                        user_agent=request.META.get('HTTP_USER_AGENT', ''),
                        metadata={'username': username, 'email': user_email}
                    )
                except Exception:
                    pass  # Don't fail deletion if logging fails
                
                # Delete the user (CASCADE will handle related objects)
                user.delete()
                
                return Response(
                    status=status.HTTP_204_NO_CONTENT
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return Response(
                    {'detail': f'Failed to delete account: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

    @action(
        detail=False,
        methods=['get', 'patch'],
        permission_classes=[IsAuthenticated],
        url_path='me/profile',
        url_name='me-profile'
    )
    def me_profile(self, request):
        """
        Get or update current user's profile information
        """
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        if request.method == 'GET':
            serializer = UserProfileSerializer(profile)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        elif request.method == 'PATCH':
            serializer = UserProfileSerializer(
                profile,
                data=request.data,
                partial=True
            )
            serializer.is_valid(raise_exception=True)
            serializer.save()
            
            # Log activity
            UserActivity.objects.create(
                user=request.user,
                activity_type='profile_update',
                ip_address=self._get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                metadata={'updated_fields': list(request.data.keys())}
            )
            
            return Response(serializer.data, status=status.HTTP_200_OK)

    @action(
        detail=False,
        methods=['get', 'patch'],
        permission_classes=[IsAuthenticated],
        url_path='me/preferences',
        url_name='me-preferences'
    )
    def me_preferences(self, request):
        """
        Get or update current user's preferences
        """
        preferences, created = UserPreference.objects.get_or_create(user=request.user)
        
        if request.method == 'GET':
            serializer = UserPreferenceSerializer(preferences)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        elif request.method == 'PATCH':
            serializer = UserPreferenceSerializer(
                preferences,
                data=request.data,
                partial=True
            )
            serializer.is_valid(raise_exception=True)
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)

    @action(
        detail=False,
        methods=['get'],
        permission_classes=[IsAuthenticated],
        url_path='me/activities',
        url_name='me-activities'
    )
    def me_activities(self, request):
        """
        Get current user's activity history
        """
        activities = UserActivity.objects.filter(user=request.user)[:50]
        serializer = UserActivitySerializer(activities, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


    def _get_client_ip(self, request):
        """Get client IP address from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class UserProfileViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing user profiles (read-only)
    """
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Only return the current user's profile"""
        return UserProfile.objects.filter(user=self.request.user)


class UserActivityViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing user activities (read-only)
    """
    queryset = UserActivity.objects.all()
    serializer_class = UserActivitySerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Only return the current user's activities"""
        queryset = UserActivity.objects.filter(user=self.request.user)
        
        # Filter by activity type if provided
        activity_type = self.request.query_params.get('activity_type', None)
        if activity_type:
            queryset = queryset.filter(activity_type=activity_type)
        
        return queryset.order_by('-created_at')
