from django.db import models
from django.contrib.auth.models import User


class UserProfile(models.Model):
    """
    Extended user profile model to store additional user information
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile',
        primary_key=True
    )
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    company = models.CharField(max_length=255, blank=True, null=True)
    job_title = models.CharField(max_length=255, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    postal_code = models.CharField(max_length=20, blank=True, null=True)
    timezone = models.CharField(max_length=50, default='UTC')
    preferred_language = models.CharField(max_length=10, default='en')
    bio = models.TextField(blank=True, null=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    email_notifications = models.BooleanField(default=True)
    push_notifications = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
        ordering = ['-created_at']

    def __str__(self):
        return f"Profile of {self.user.username}"


class UserActivity(models.Model):
    """
    Model to track user activity and login history
    """
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='activities'
    )
    activity_type = models.CharField(
        max_length=50,
        choices=[
            ('login', 'Login'),
            ('logout', 'Logout'),
            ('password_change', 'Password Change'),
            ('profile_update', 'Profile Update'),
            ('pipeline_create', 'Pipeline Created'),
            ('image_upload', 'Image Uploaded'),
            ('analysis_run', 'Analysis Run'),
        ]
    )
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'User Activity'
        verbose_name_plural = 'User Activities'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'activity_type']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.get_activity_type_display()} - {self.created_at}"


class UserPreference(models.Model):
    """
    Model to store user preferences and settings
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='preferences',
        primary_key=True
    )
    theme = models.CharField(
        max_length=10,
        choices=[
            ('light', 'Light'),
            ('dark', 'Dark'),
            ('auto', 'Auto'),
        ],
        default='auto'
    )
    items_per_page = models.IntegerField(default=20)
    dashboard_layout = models.JSONField(default=dict, blank=True)
    notification_preferences = models.JSONField(
        default=dict,
        blank=True,
        help_text="Stores notification preferences for different event types"
    )
    map_preferences = models.JSONField(
        default=dict,
        blank=True,
        help_text="Stores map display preferences"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'User Preference'
        verbose_name_plural = 'User Preferences'

    def __str__(self):
        return f"Preferences of {self.user.username}"
