from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import (
    Pipeline,
    SatelliteImage,
    Analysis,
    Anomaly,
    Notification,
    UserDevice
)
from .tasks import convert_to_cog_task, run_analysis_task
from .utils import extract_geotiff_bbox


@admin.register(Pipeline)
class PipelineAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'status', 'length_km', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = ['id', 'created_at', 'updated_at', 'geojson_file_link']
    date_hierarchy = 'created_at'
    list_per_page = 20
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'name', 'description', 'status')
        }),
        ('GeoJSON File', {
            'fields': ('geojson_file', 'geojson_file_link'),
            'description': 'Upload a GeoJSON file (.json or .geojson) containing the pipeline route geometry.'
        }),
        ('Pipeline Information', {
            'fields': ('length_km',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    
    def geojson_file_link(self, obj):
        """Display a link to download the uploaded GeoJSON file"""
        if obj.pk and obj.geojson_file:
            return format_html(
                '<a href="{}" target="_blank">Download GeoJSON File</a>',
                obj.geojson_file.url
            )
        return 'No file uploaded yet'
    geojson_file_link.short_description = 'GeoJSON File'
    
    def save_model(self, request, obj, form, change):
        """Save the model and ensure user is set"""
        if not change:  # New object
            obj.user = request.user
        super().save_model(request, obj, form, change)


@admin.register(SatelliteImage)
class SatelliteImageAdmin(admin.ModelAdmin):
    list_display = [
        'name',
        'user',
        'pipeline',
        'image_type',
        'acquisition_date',
        'cog_status',
        'created_at'
    ]
    list_filter = [
        'image_type',
        'is_cog_converted',
        'conversion_status',
        'acquisition_date',
        'created_at'
    ]
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = [
        'id',
        'created_at',
        'updated_at',
        'cog_status_display',
        'bbox_info'
    ]
    date_hierarchy = 'acquisition_date'
    list_per_page = 20
    actions = ['convert_to_cog', 'run_analysis_action']

    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'pipeline', 'name', 'description')
        }),
        ('Image Files', {
            'fields': ('original_tiff', 'cog_tiff', 'image_type', 'acquisition_date')
        }),
        ('COG Conversion', {
            'fields': ('is_cog_converted', 'conversion_status', 'cog_status_display')
        }),
        ('Geospatial Info', {
            'fields': ('bbox_info', 'bbox_minx', 'bbox_miny', 'bbox_maxx', 'bbox_maxy')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )

    def cog_status(self, obj):
        if obj.is_cog_converted:
            color = 'green'
            text = '✓ COG Ready'
        elif obj.conversion_status == 'processing':
            color = 'orange'
            text = 'Processing...'
        elif obj.conversion_status == 'failed':
            color = 'red'
            text = 'Failed'
        else:
            color = 'gray'
            text = 'Pending'
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            text
        )
    cog_status.short_description = 'COG Status'

    def cog_status_display(self, obj):
        return self.cog_status(obj)
    cog_status_display.short_description = 'COG Conversion Status'

    def bbox_info(self, obj):
        if all([obj.bbox_minx, obj.bbox_miny, obj.bbox_maxx, obj.bbox_maxy]):
            return format_html(
                '<div>Min: ({:.6f}, {:.6f})<br>Max: ({:.6f}, {:.6f})</div>',
                obj.bbox_minx, obj.bbox_miny, obj.bbox_maxx, obj.bbox_maxy
            )
        return 'Not available'
    bbox_info.short_description = 'Bounding Box'

    def save_model(self, request, obj, form, change):
        """Save the model and extract bbox if original_tiff is uploaded"""
        # Set user if new object
        if not change:
            obj.user = request.user
        
        # Save first to ensure file is available
        super().save_model(request, obj, form, change)
        
        # Extract bbox from original_tiff if available and bbox not already set
        if obj.original_tiff and not all([obj.bbox_minx, obj.bbox_miny, obj.bbox_maxx, obj.bbox_maxy]):
            try:
                tiff_path = obj.original_tiff.path
                bbox = extract_geotiff_bbox(tiff_path)
                if bbox:
                    obj.bbox_minx = bbox['minx']
                    obj.bbox_miny = bbox['miny']
                    obj.bbox_maxx = bbox['maxx']
                    obj.bbox_maxy = bbox['maxy']
                    obj.save(update_fields=['bbox_minx', 'bbox_miny', 'bbox_maxx', 'bbox_maxy'])
            except Exception as e:
                # Log error but don't fail the save
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to extract bbox for image {obj.id}: {str(e)}")

    @admin.action(description='Convert selected images to Cloud Optimized GeoTIFF')
    def convert_to_cog(self, request, queryset):
        count = 0
        for image in queryset:
            if not image.is_cog_converted:
                convert_to_cog_task.delay(str(image.id))
                count += 1
        self.message_user(
            request,
            f'{count} image(s) queued for COG conversion.'
        )

    @admin.action(description='Run analysis on selected images')
    def run_analysis_action(self, request, queryset):
        count = 0
        for image in queryset:
            if image.is_cog_converted:
                run_analysis_task.delay(str(image.id))
                count += 1
            else:
                self.message_user(
                    request,
                    f'Image {image.name} must be converted to COG first.',
                    level='warning'
                )
        self.message_user(
            request,
            f'Analysis queued for {count} image(s).'
        )


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = [
        'analysis_type_display',
        'satellite_image',
        'user',
        'status',
        'severity',
        'confidence_score',
        'created_at'
    ]
    list_filter = [
        'analysis_type',
        'status',
        'severity',
        'created_at'
    ]
    search_fields = [
        'satellite_image__name',
        'user__username',
        'pipeline__name'
    ]
    readonly_fields = [
        'id',
        'created_at',
        'updated_at',
        'processing_info'
    ]
    date_hierarchy = 'created_at'
    list_per_page = 20

    def analysis_type_display(self, obj):
        return obj.get_analysis_type_display()
    analysis_type_display.short_description = 'Analysis Type'

    def processing_info(self, obj):
        if obj.processing_time_seconds:
            return format_html(
                '<div>Processing Time: {:.2f} seconds</div>',
                obj.processing_time_seconds
            )
        return 'N/A'
    processing_info.short_description = 'Processing Information'


@admin.register(Anomaly)
class AnomalyAdmin(admin.ModelAdmin):
    list_display = [
        'anomaly_type_display',
        'severity',
        'user',
        'location',
        'confidence_score',
        'is_resolved',
        'created_at'
    ]
    list_filter = [
        'anomaly_type',
        'severity',
        'is_verified',
        'is_resolved',
        'created_at'
    ]
    search_fields = [
        'description',
        'user__username',
        'analysis__satellite_image__name'
    ]
    readonly_fields = ['id', 'created_at', 'updated_at']
    date_hierarchy = 'created_at'
    list_per_page = 20
    actions = ['mark_as_resolved', 'mark_as_unresolved']

    def anomaly_type_display(self, obj):
        return obj.get_anomaly_type_display()
    anomaly_type_display.short_description = 'Anomaly Type'

    def location(self, obj):
        return format_html(
            '({:.6f}, {:.6f})',
            obj.location_lat,
            obj.location_lon
        )
    location.short_description = 'Location'

    @admin.action(description='Mark selected anomalies as resolved')
    def mark_as_resolved(self, request, queryset):
        from django.utils import timezone
        queryset.update(is_resolved=True, resolved_at=timezone.now())
        self.message_user(
            request,
            f'{queryset.count()} anomaly(ies) marked as resolved.'
        )

    @admin.action(description='Mark selected anomalies as unresolved')
    def mark_as_unresolved(self, request, queryset):
        queryset.update(is_resolved=False, resolved_at=None)
        self.message_user(
            request,
            f'{queryset.count()} anomaly(ies) marked as unresolved.'
        )


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = [
        'title',
        'user',
        'notification_type',
        'is_read',
        'is_sent',
        'created_at'
    ]
    list_filter = [
        'notification_type',
        'is_read',
        'is_sent',
        'created_at'
    ]
    search_fields = ['title', 'message', 'user__username']
    readonly_fields = ['id', 'created_at']
    date_hierarchy = 'created_at'
    list_per_page = 20


@admin.register(UserDevice)
class UserDeviceAdmin(admin.ModelAdmin):
    list_display = ['user', 'device_type', 'is_active', 'last_used_at', 'created_at']
    list_filter = ['device_type', 'is_active', 'created_at']
    search_fields = ['user__username', 'device_token']
    readonly_fields = ['id', 'created_at']
    list_per_page = 20
