"""
Signal handlers for user app
These signals are imported in apps.py to ensure they are registered
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import UserProfile, UserPreference


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """
    Signal to automatically create a UserProfile when a User is created
    """
    if created:
        UserProfile.objects.get_or_create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """
    Signal to save the UserProfile when User is saved
    """
    if hasattr(instance, 'profile'):
        instance.profile.save()
    else:
        UserProfile.objects.get_or_create(user=instance)


@receiver(post_save, sender=User)
def create_user_preferences(sender, instance, created, **kwargs):
    """
    Signal to automatically create UserPreferences when a User is created
    """
    if created:
        UserPreference.objects.get_or_create(user=instance)

