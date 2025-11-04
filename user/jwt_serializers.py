from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    Custom JWT token serializer that allows login with either username or email
    """
    username_field = 'username'

    def validate(self, attrs):
        """
        Custom validation to allow login with username or email
        """
        from user.backends import EmailOrUsernameModelBackend
        
        # Get the username field (which can be username or email)
        username = attrs.get(self.username_field)
        password = attrs.get('password')

        if username is None or password is None:
            raise serializers.ValidationError(
                'Must include "username" and "password".'
            )

        # Try to authenticate with email or username
        backend = EmailOrUsernameModelBackend()
        user = backend.authenticate(
            request=self.context['request'],
            username=username,
            password=password
        )

        if user is None:
            raise serializers.ValidationError(
                'No active account found with the given credentials'
            )

        if not user.is_active:
            raise serializers.ValidationError('User account is disabled.')

        # Get the refresh and access tokens
        refresh = self.get_token(user)

        data = {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        }

        return data

