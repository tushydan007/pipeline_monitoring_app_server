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
    profile = UserProfileSerializer(read_only=True, required=False, allow_null=True)
    preferences = UserPreferenceSerializer(read_only=True, required=False, allow_null=True)
    
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
    
    def to_representation(self, instance):
        """Override to safely handle profile and preferences that might not exist"""
        representation = super().to_representation(instance)
        
        # Safely get profile if it exists
        try:
            if hasattr(instance, 'profile') and instance.profile:
                representation['profile'] = UserProfileSerializer(instance.profile).data
            else:
                representation['profile'] = None
        except Exception:
            representation['profile'] = None
        
        # Safely get preferences if it exists
        try:
            if hasattr(instance, 'preferences') and instance.preferences:
                representation['preferences'] = UserPreferenceSerializer(instance.preferences).data
            else:
                representation['preferences'] = None
        except Exception:
            representation['preferences'] = None
        
        return representation


class UserCreateSerializer(DjoserUserCreateSerializer):
    """Extended User creation serializer with email uniqueness validation"""
    email = serializers.EmailField(required=True)
    username = serializers.CharField(required=False, allow_blank=True)
    
    class Meta(DjoserUserCreateSerializer.Meta):
        fields = DjoserUserCreateSerializer.Meta.fields
    
    def validate_email(self, value):
        """Ensure email is unique"""
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value.lower()  # Store emails in lowercase
    
    def validate(self, attrs):
        """Ensure username is provided or generate from email"""
        attrs = super().validate(attrs)
        email = attrs.get('email', '')
        
        # If username is not provided or is empty, generate it from email
        if not attrs.get('username') or attrs.get('username', '').strip() == '':
            # Use email prefix as username (before @)
            username = email.split('@')[0]
            # Ensure username is unique by appending numbers if needed
            base_username = username
            counter = 1
            while User.objects.filter(username=username).exists():
                username = f"{base_username}{counter}"
                counter += 1
            attrs['username'] = username
        
        return attrs
    
    def save(self, **kwargs):
        """
        Override save() to ensure username is always present.
        This is the entry point that DRF calls, so we intercept here.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.error("=== CUSTOM UserCreateSerializer.save() CALLED ===")
        
        # Get validated_data from the serializer
        if not self.validated_data:
            # If not validated yet, validate first
            self.is_valid(raise_exception=True)
        
        validated_data = self.validated_data.copy()
        logger.error(f"validated_data keys: {list(validated_data.keys())}")
        logger.error(f"validated_data: {validated_data}")
        
        # Ensure username is in validated_data (defensive check - triple check!)
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
            logger.error(f"Generated username: {username}")
        
        # Log for debugging
        logger.error(f"Final validated_data before create_user: username={validated_data.get('username')}, email={validated_data.get('email')}")
        
        # Ensure username is present one more time before calling create_user
        if 'username' not in validated_data or not validated_data['username']:
            logger.error("ERROR: Username is still missing after generation!")
            raise ValueError("Username is required but was not provided or generated")
        
        # Update self.validated_data with username
        self.validated_data['username'] = validated_data['username']
        
        # Call parent's save() which will call create() with the updated validated_data
        # But we'll override create() to ensure it uses our logic
        return super().save(**kwargs)
    
    def create(self, validated_data):
        """
        Override create() to ensure username is always present.
        This is called by the parent's save() method.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.error("=== CUSTOM UserCreateSerializer.create() CALLED ===")
        
        # Ensure username is present (defensive check)
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
            logger.error(f"Generated username in create(): {username}")
        
        # Create the user using Django's UserManager directly
        try:
            user = User.objects.create_user(**validated_data)
            logger.error(f"User created successfully: {user.username}")
        except Exception as e:
            logger.error(f"ERROR creating user: {str(e)}")
            logger.error(f"validated_data passed to create_user: {validated_data}")
            raise
        
        # Ensure profile and preferences are created (signals should handle this, but be defensive)
        try:
            UserProfile.objects.get_or_create(user=user)
        except Exception:
            pass  # Profile creation might fail, but don't fail user creation
        
        try:
            UserPreference.objects.get_or_create(user=user)
        except Exception:
            pass  # Preferences creation might fail, but don't fail user creation
        
        return user

