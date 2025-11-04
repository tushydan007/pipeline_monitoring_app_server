from rest_framework import serializers
from django.contrib.auth.models import User
from djoser.serializers import UserSerializer as DjoserUserSerializer, UserCreateSerializer as DjoserUserCreateSerializer
from .models import UserProfile, UserPreference, UserActivity


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for UserProfile"""
    class Meta:
        model = UserProfile
        fields = [
            'phone_number',
            'company',
            'job_title',
            'address',
            'city',
            'country',
            'postal_code',
            'timezone',
            'preferred_language',
            'bio',
            'avatar',
            'email_notifications',
            'push_notifications',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']


class UserPreferenceSerializer(serializers.ModelSerializer):
    """Serializer for UserPreference"""
    class Meta:
        model = UserPreference
        fields = [
            'theme',
            'items_per_page',
            'dashboard_layout',
            'notification_preferences',
            'map_preferences',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']


class UserActivitySerializer(serializers.ModelSerializer):
    """Serializer for UserActivity"""
    activity_type_display = serializers.CharField(source='get_activity_type_display', read_only=True)

    class Meta:
        model = UserActivity
        fields = [
            'id',
            'activity_type',
            'activity_type_display',
            'ip_address',
            'user_agent',
            'metadata',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class UserSerializer(DjoserUserSerializer):
    """Extended User serializer with profile and preferences"""
    profile = UserProfileSerializer(read_only=True)
    preferences = UserPreferenceSerializer(read_only=True)
    
    # Explicitly override User model fields to ensure they're included and writable
    username = serializers.CharField(required=False)
    email = serializers.EmailField(required=False)
    first_name = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    last_name = serializers.CharField(required=False, allow_blank=True, allow_null=True)

    class Meta(DjoserUserSerializer.Meta):
        # Explicitly define all fields to ensure they're included in response
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'is_active',
            'date_joined',
            'profile',
            'preferences',
        ]
        # Only id, date_joined, profile, and preferences are read-only
        read_only_fields = ['id', 'date_joined', 'profile', 'preferences']


class UserCreateSerializer(DjoserUserCreateSerializer):
    """Extended User creation serializer with email uniqueness validation"""
    email = serializers.EmailField(required=True)
    
    class Meta(DjoserUserCreateSerializer.Meta):
        fields = DjoserUserCreateSerializer.Meta.fields
    
    def validate_email(self, value):
        """Ensure email is unique"""
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value.lower()  # Store emails in lowercase

