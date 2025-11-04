from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from django.utils.html import format_html
from .models import UserProfile, UserActivity, UserPreference


class UserProfileInline(admin.StackedInline):
    """Inline admin for UserProfile"""
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'
    fk_name = 'user'
    fields = (
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
    )


class UserPreferenceInline(admin.StackedInline):
    """Inline admin for UserPreference"""
    model = UserPreference
    can_delete = False
    verbose_name_plural = 'Preferences'
    fk_name = 'user'


class CustomUserAdmin(BaseUserAdmin):
    """Extended User admin with profile and preferences"""
    inlines = (UserProfileInline, UserPreferenceInline)
    list_display = (
        'username',
        'email',
        'first_name',
        'last_name',
        'is_staff',
        'is_active',
        'date_joined',
        'last_login',
        'profile_info',
    )
    list_filter = (
        'is_staff',
        'is_superuser',
        'is_active',
        'date_joined',
        'last_login',
    )
    search_fields = (
        'username',
        'email',
        'first_name',
        'last_name',
        'profile__company',
    )

    def profile_info(self, obj):
        """Display profile information"""
        if hasattr(obj, 'profile'):
            profile = obj.profile
            return format_html(
                '<div>Company: {}<br>Phone: {}</div>',
                profile.company or 'N/A',
                profile.phone_number or 'N/A'
            )
        return 'No Profile'
    profile_info.short_description = 'Profile Info'

    def get_inline_instances(self, request, obj=None):
        """Only show inlines when editing an existing user"""
        if not obj:
            return []
        return super().get_inline_instances(request, obj)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """Admin for UserProfile"""
    list_display = (
        'user',
        'company',
        'phone_number',
        'city',
        'country',
        'email_notifications',
        'push_notifications',
        'created_at',
    )
    list_filter = (
        'email_notifications',
        'push_notifications',
        'country',
        'timezone',
        'created_at',
    )
    search_fields = (
        'user__username',
        'user__email',
        'company',
        'phone_number',
        'city',
        'country',
    )
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Contact Information', {
            'fields': (
                'phone_number',
                'company',
                'job_title',
                'address',
                'city',
                'country',
                'postal_code',
            )
        }),
        ('Preferences', {
            'fields': (
                'timezone',
                'preferred_language',
                'email_notifications',
                'push_notifications',
            )
        }),
        ('Additional Information', {
            'fields': (
                'bio',
                'avatar',
            )
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    """Admin for UserActivity"""
    list_display = (
        'user',
        'activity_type',
        'ip_address',
        'created_at',
    )
    list_filter = (
        'activity_type',
        'created_at',
    )
    search_fields = (
        'user__username',
        'user__email',
        'ip_address',
        'activity_type',
    )
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

    def has_add_permission(self, request):
        """Disable manual creation of activities"""
        return False

    def has_change_permission(self, request, obj=None):
        """Disable editing of activities"""
        return False


@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
    """Admin for UserPreference"""
    list_display = (
        'user',
        'theme',
        'items_per_page',
        'updated_at',
    )
    list_filter = (
        'theme',
        'items_per_page',
        'updated_at',
    )
    search_fields = (
        'user__username',
        'user__email',
    )
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        ('User', {
            'fields': ('user',)
        }),
        ('Display Preferences', {
            'fields': (
                'theme',
                'items_per_page',
            )
        }),
        ('Advanced Preferences', {
            'fields': (
                'dashboard_layout',
                'notification_preferences',
                'map_preferences',
            ),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )


# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)
