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
    
    def perform_create(self, serializer):
        """
        Override perform_create to ensure username is always present before saving.
        This is called by Djoser's create() method.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.error("=== CUSTOM UserViewSet.perform_create() CALLED ===")
        
        # Get validated_data from serializer
        if hasattr(serializer, 'validated_data') and serializer.validated_data:
            validated_data = serializer.validated_data
            logger.error(f"validated_data keys: {list(validated_data.keys())}")
            
            # Ensure username is present
            email = validated_data.get('email', '')
            if not validated_data.get('username') or validated_data.get('username', '').strip() == '':
                # Generate username from email if missing
                username = email.split('@')[0]
                base_username = username
                counter = 1
                while User.objects.filter(username=username).exists():
                    username = f"{base_username}{counter}"
                    counter += 1
                validated_data['username'] = username
                serializer.validated_data['username'] = username
                logger.error(f"Generated username in ViewSet: {username}")
        
        # Call parent's perform_create which will call serializer.save()
        return super().perform_create(serializer)
    
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
            try:
                # Get the user instance - refresh from DB to ensure we have the latest
                user = User.objects.get(pk=request.user.pk)
                
                # Track what fields are being updated
                updated_fields = []
                
                # Update User model fields directly
                if 'username' in request.data:
                    new_username = request.data['username'].strip()
                    if new_username and new_username != user.username:
                        # Validate username uniqueness
                        existing_user = User.objects.filter(username__iexact=new_username).exclude(pk=user.pk).first()
                        if existing_user:
                            return Response(
                                {'username': ['A user with this username already exists.']},
                                status=status.HTTP_400_BAD_REQUEST
                            )
                        user.username = new_username
                        updated_fields.append('username')
                
                if 'email' in request.data:
                    new_email = request.data['email'].strip().lower()
                    if new_email and new_email != user.email:
                        # Validate email uniqueness
                        existing_user = User.objects.filter(email__iexact=new_email).exclude(pk=user.pk).first()
                        if existing_user:
                            return Response(
                                {'email': ['A user with this email already exists.']},
                                status=status.HTTP_400_BAD_REQUEST
                            )
                        user.email = new_email
                        updated_fields.append('email')
                
                if 'first_name' in request.data:
                    new_first_name = request.data['first_name'].strip() if request.data['first_name'] else ''
                    if new_first_name != user.first_name:
                        user.first_name = new_first_name
                        updated_fields.append('first_name')
                
                if 'last_name' in request.data:
                    new_last_name = request.data['last_name'].strip() if request.data['last_name'] else ''
                    if new_last_name != user.last_name:
                        user.last_name = new_last_name
                        updated_fields.append('last_name')
                
                # Only save if there are actual changes
                if updated_fields:
                    # Log before save
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Updating user {user.pk} fields: {updated_fields}")
                    logger.info(f"User before save - username: {user.username}, email: {user.email}, first_name: {user.first_name}, last_name: {user.last_name}")
                    
                    # Full clean and save with validation
                    user.full_clean()
                    user.save(update_fields=updated_fields)
                    
                    # Refresh from database to ensure we have the latest data
                    user.refresh_from_db()
                    
                    # Log after save
                    logger.info(f"User after save - username: {user.username}, email: {user.email}, first_name: {user.first_name}, last_name: {user.last_name}")
                    
                    # Verify the save worked by querying from DB
                    user_from_db = User.objects.get(pk=user.pk)
                    logger.info(f"User from DB - username: {user_from_db.username}, email: {user_from_db.email}, first_name: {user_from_db.first_name}, last_name: {user_from_db.last_name}")
                    
                    # Use the freshly queried user for serialization
                    user = user_from_db
                    
                    # Log the activity
                    try:
                        UserActivity.objects.create(
                            user=user,
                            activity_type='profile_update',
                            ip_address=self._get_client_ip(request),
                            user_agent=request.META.get('HTTP_USER_AGENT', ''),
                            metadata={'updated_fields': updated_fields}
                        )
                    except Exception:
                        pass  # Don't fail update if logging fails
                else:
                    # No fields were updated, but still return success
                    pass
                
                # Refresh user instance one more time before serializing
                user.refresh_from_db()
                
                # Return updated serializer data
                serializer = self.get_serializer(user)
                response_data = serializer.data
                
                # Log response data for debugging
                logger.info(f"Response data keys: {list(response_data.keys())}")
                logger.info(f"Response data - username: {response_data.get('username')}, email: {response_data.get('email')}, first_name: {response_data.get('first_name')}, last_name: {response_data.get('last_name')}")
                
                return Response(response_data, status=status.HTTP_200_OK)
                
            except Exception as e:
                import traceback
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error updating user profile: {str(e)}")
                traceback.print_exc()
                
                # Return detailed error message
                error_message = str(e)
                if 'unique constraint' in error_message.lower() or 'duplicate' in error_message.lower():
                    if 'username' in error_message.lower():
                        return Response(
                            {'username': ['A user with this username already exists.']},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    elif 'email' in error_message.lower():
                        return Response(
                            {'email': ['A user with this email already exists.']},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                
                return Response(
                    {'detail': f'Failed to update profile: {error_message}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
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
