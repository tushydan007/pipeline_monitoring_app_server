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
    UserDevice,
)
from .tasks import convert_to_cog_task, run_analysis_task
from .utils import extract_geotiff_bbox
from .models import LegendCategory, MappedObject


@admin.register(LegendCategory)
class LegendCategoryAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "color_preview",
        "category_type",
        "icon",
        "user",
        "created_at",
    ]
    list_filter = ["category_type", "created_at"]
    search_fields = ["name", "description", "user__username"]
    readonly_fields = ["id", "created_at", "updated_at", "color_preview_large"]
    date_hierarchy = "created_at"
    list_per_page = 20

    fieldsets = (
        ("Basic Information", {"fields": ("id", "user", "name", "description")}),
        (
            "Visual Properties",
            {
                "fields": ("color", "color_preview_large", "icon", "category_type"),
                "description": "Configure how this legend appears on the map",
            },
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def color_preview(self, obj):
        """Display color as a small colored box"""
        return format_html(
            '<div style="width: 30px; height: 20px; background-color: {}; border: 1px solid #000; border-radius: 3px;"></div>',
            obj.color,
        )

    color_preview.short_description = "Color"

    def color_preview_large(self, obj):
        """Display larger color preview"""
        return format_html(
            '<div style="width: 100px; height: 50px; background-color: {}; border: 2px solid #000; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; text-shadow: 1px 1px 2px black;">{}</div>',
            obj.color,
            obj.color,
        )

    color_preview_large.short_description = "Color Preview"

    def save_model(self, request, obj, form, change):
        """Set user if creating new object"""
        if not change:
            obj.user = request.user
        super().save_model(request, obj, form, change)


@admin.register(Pipeline)
class PipelineAdmin(admin.ModelAdmin):
    list_display = ["name", "user", "status", "length_km", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["name", "description", "user__username"]
    readonly_fields = ["id", "created_at", "updated_at", "geojson_file_link"]
    date_hierarchy = "created_at"
    list_per_page = 20

    fieldsets = (
        (
            "Basic Information",
            {"fields": ("id", "user", "name", "description", "status")},
        ),
        (
            "GeoJSON File",
            {
                "fields": ("geojson_file", "geojson_file_link"),
                "description": "Upload a GeoJSON file (.json or .geojson) containing the pipeline route geometry.",
            },
        ),
        ("Pipeline Information", {"fields": ("length_km",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def geojson_file_link(self, obj):
        """Display a link to download the uploaded GeoJSON file"""
        if obj.pk and obj.geojson_file:
            return format_html(
                '<a href="{}" target="_blank">Download GeoJSON File</a>',
                obj.geojson_file.url,
            )
        return "No file uploaded yet"

    geojson_file_link.short_description = "GeoJSON File"

    def save_model(self, request, obj, form, change):
        """Save the model and ensure user is set"""
        if not change:  # New object
            obj.user = request.user
        super().save_model(request, obj, form, change)


@admin.register(SatelliteImage)
class SatelliteImageAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "user",
        "pipeline",
        "image_type",
        "acquisition_date",
        "cog_status",
        "created_at",
    ]
    list_filter = [
        "image_type",
        "is_cog_converted",
        "conversion_status",
        "acquisition_date",
        "created_at",
    ]
    search_fields = ["name", "description", "user__username"]
    readonly_fields = [
        "id",
        "created_at",
        "updated_at",
        "cog_status_display",
        "bbox_info",
    ]
    date_hierarchy = "acquisition_date"
    list_per_page = 20
    actions = ["convert_to_cog", "run_analysis_action"]

    fieldsets = (
        (
            "Basic Information",
            {"fields": ("id", "user", "pipeline", "name", "description")},
        ),
        (
            "Image Files",
            {"fields": ("original_tiff", "cog_tiff", "image_type", "acquisition_date")},
        ),
        (
            "COG Conversion",
            {"fields": ("is_cog_converted", "conversion_status", "cog_status_display")},
        ),
        (
            "Geospatial Info",
            {
                "fields": (
                    "bbox_info",
                    "bbox_minx",
                    "bbox_miny",
                    "bbox_maxx",
                    "bbox_maxy",
                )
            },
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def cog_status(self, obj):
        if obj.is_cog_converted:
            color = "green"
            text = "✓ COG Ready"
        elif obj.conversion_status == "processing":
            color = "orange"
            text = "Processing..."
        elif obj.conversion_status == "failed":
            color = "red"
            text = "Failed"
        else:
            color = "gray"
            text = "Pending"
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>', color, text
        )

    cog_status.short_description = "COG Status"

    def cog_status_display(self, obj):
        return self.cog_status(obj)

    cog_status_display.short_description = "COG Conversion Status"

    def bbox_info(self, obj):
        if all([obj.bbox_minx, obj.bbox_miny, obj.bbox_maxx, obj.bbox_maxy]):
            return format_html(
                "<div>Min: ({:.6f}, {:.6f})<br>Max: ({:.6f}, {:.6f})</div>",
                obj.bbox_minx,
                obj.bbox_miny,
                obj.bbox_maxx,
                obj.bbox_maxy,
            )
        return "Not available"

    bbox_info.short_description = "Bounding Box"

    def save_model(self, request, obj, form, change):
        """Save the model and extract bbox if original_tiff is uploaded"""
        # Set user if new object
        if not change:
            obj.user = request.user

        # Save first to ensure file is available
        super().save_model(request, obj, form, change)

        # Extract bbox from original_tiff if available and bbox not already set
        if obj.original_tiff and not all(
            [obj.bbox_minx, obj.bbox_miny, obj.bbox_maxx, obj.bbox_maxy]
        ):
            try:
                tiff_path = obj.original_tiff.path
                bbox = extract_geotiff_bbox(tiff_path)
                if bbox:
                    obj.bbox_minx = bbox["minx"]
                    obj.bbox_miny = bbox["miny"]
                    obj.bbox_maxx = bbox["maxx"]
                    obj.bbox_maxy = bbox["maxy"]
                    obj.save(
                        update_fields=[
                            "bbox_minx",
                            "bbox_miny",
                            "bbox_maxx",
                            "bbox_maxy",
                        ]
                    )
            except Exception as e:
                # Log error but don't fail the save
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Failed to extract bbox for image {obj.id}: {str(e)}")

    @admin.action(description="Convert selected images to Cloud Optimized GeoTIFF")
    def convert_to_cog(self, request, queryset):
        count = 0
        for image in queryset:
            if not image.is_cog_converted:
                convert_to_cog_task.delay(str(image.id))
                count += 1
        self.message_user(request, f"{count} image(s) queued for COG conversion.")

    @admin.action(description="Run analysis on selected images")
    def run_analysis_action(self, request, queryset):
        count = 0
        for image in queryset:
            if image.is_cog_converted:
                run_analysis_task.delay(str(image.id))
                count += 1
            else:
                self.message_user(
                    request,
                    f"Image {image.name} must be converted to COG first.",
                    level="warning",
                )
        self.message_user(request, f"Analysis queued for {count} image(s).")


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = [
        "analysis_type_display",
        "satellite_image",
        "user",
        "status",
        "severity",
        "confidence_score",
        "created_at",
    ]
    list_filter = ["analysis_type", "status", "severity", "created_at"]
    search_fields = ["satellite_image__name", "user__username", "pipeline__name"]
    readonly_fields = [
        "id",
        "created_at",
        "updated_at",
        "processing_info",
        "analysis_summary",
    ]
    date_hierarchy = "created_at"
    list_per_page = 20

    fieldsets = (
        (
            "Basic Information",
            {"fields": ("id", "user", "satellite_image", "pipeline")},
        ),
        (
            "Analysis Details",
            {
                "fields": ("analysis_type", "status", "confidence_score", "severity"),
                "description": "SAR-based analysis configuration and results",
            },
        ),
        (
            "Results",
            {
                "fields": ("results_json", "metadata", "analysis_summary"),
                "classes": ("collapse",),
            },
        ),
        (
            "Processing",
            {
                "fields": (
                    "processing_time_seconds",
                    "processing_info",
                    "error_message",
                ),
            },
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def analysis_type_display(self, obj):
        # Color code by analysis family
        sar_insar = [
            "insar",
            "dinsar",
            "psinsar",
            "sbas_insar",
            "phase_unwrap",
            "atmospheric_corr",
        ]
        polarimetric = ["polarimetric_decomp", "polarimetric_scatter"]
        texture = ["glcm_texture", "texture_shape"]

        if obj.analysis_type in sar_insar:
            color = "#2196F3"  # Blue for InSAR family
        elif obj.analysis_type in polarimetric:
            color = "#9C27B0"  # Purple for polarimetric
        elif obj.analysis_type in texture:
            color = "#FF9800"  # Orange for texture
        elif obj.analysis_type == "deep_learning":
            color = "#4CAF50"  # Green for AI
        else:
            color = "#607D8B"  # Gray for others

        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.get_analysis_type_display(),
        )

    analysis_type_display.short_description = "Analysis Type"
    analysis_type_display.admin_order_field = "analysis_type"

    def analysis_summary(self, obj):
        """Display a summary of analysis results"""
        if not obj.results_json:
            return "No results available"

        anomaly_count = obj.anomalies.count()
        results = obj.results_json

        html = f'<div style="font-family: monospace;">'
        html += f"<strong>Anomalies Detected:</strong> {anomaly_count}<br>"

        # Display key metrics based on analysis type
        if "anomaly_count" in results:
            html += f'<strong>Analysis Count:</strong> {results["anomaly_count"]}<br>'

        if "mean_backscatter" in results:
            html += f'<strong>Mean Backscatter:</strong> {results["mean_backscatter"]:.3f}<br>'

        if "mean_coherence" in results:
            html += (
                f'<strong>Mean Coherence:</strong> {results["mean_coherence"]:.3f}<br>'
            )

        if "mean_displacement" in results:
            html += f'<strong>Mean Displacement:</strong> {results["mean_displacement"]:.3f}<br>'

        if "contrast" in results:
            html += f'<strong>GLCM Contrast:</strong> {results["contrast"]:.3f}<br>'

        html += "</div>"
        return format_html(html)

    analysis_summary.short_description = "Analysis Summary"

    def processing_info(self, obj):
        if obj.processing_time_seconds:
            return format_html(
                "<div><strong>Processing Time:</strong> {:.2f} seconds<br>"
                "<strong>Status:</strong> {}<br>"
                "<strong>Confidence:</strong> {:.1%}</div>",
                obj.processing_time_seconds,
                obj.get_status_display(),
                obj.confidence_score or 0,
            )
        return "N/A"

    processing_info.short_description = "Processing Information"


@admin.register(Anomaly)
class AnomalyAdmin(admin.ModelAdmin):
    list_display = [
        "anomaly_type_display",
        "severity",
        "user",
        "location",
        "confidence_score",
        "is_resolved",
        "created_at",
    ]
    list_filter = [
        "anomaly_type",
        "severity",
        "is_verified",
        "is_resolved",
        "created_at",
    ]
    search_fields = ["description", "user__username", "analysis__satellite_image__name"]
    readonly_fields = ["id", "created_at", "updated_at", "anomaly_details"]
    date_hierarchy = "created_at"
    list_per_page = 20
    actions = ["mark_as_resolved", "mark_as_unresolved", "mark_as_verified"]

    fieldsets = (
        (
            "Basic Information",
            {"fields": ("id", "analysis", "user", "anomaly_type", "severity")},
        ),
        ("Location", {"fields": ("location_lat", "location_lon", "area_m2")}),
        (
            "Details",
            {
                "fields": (
                    "description",
                    "confidence_score",
                    "metadata",
                    "anomaly_details",
                )
            },
        ),
        (
            "Status",
            {
                "fields": (
                    "is_verified",
                    "verified_by",
                    "verified_at",
                    "is_resolved",
                    "resolved_at",
                )
            },
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def anomaly_type_display(self, obj):
        # Color code by severity
        colors = {
            "critical": "#F44336",
            "high": "#FF9800",
            "medium": "#FFC107",
            "low": "#4CAF50",
        }
        color = colors.get(obj.severity, "#9E9E9E")

        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.get_anomaly_type_display(),
        )

    anomaly_type_display.short_description = "Anomaly Type"
    anomaly_type_display.admin_order_field = "anomaly_type"

    def location(self, obj):
        if obj.location_lat is None or obj.location_lon is None:
            return "No coordinates yet"
        
        # Use % formatting (what format_html expects) and force float formatting safely
        lat = float(obj.location_lat)
        lon = float(obj.location_lon)
        
        link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        display_text = "({:.6f}, {:.6f})".format(lat, lon)
        
        return format_html(
            '<a href="{}" target="_blank" style="color: #1a0dab; text-decoration: underline;">{}</a>',
            link,
            display_text
        )

    location.short_description = "Location (click to view on map)"
    location.admin_order_field = "location_lat"  # optional: allows sorting

    def anomaly_details(self, obj):
        """Display detailed information about the anomaly"""
        html = '<div style="font-family: monospace;">'
        html += f"<strong>Analysis Type:</strong> {obj.analysis.get_analysis_type_display()}<br>"
        html += f"<strong>Confidence:</strong> {obj.confidence_score:.1%}<br>"

        if obj.area_m2:
            html += f"<strong>Area:</strong> {obj.area_m2:.1f} m²<br>"

        if obj.metadata:
            html += "<br><strong>Technical Metadata:</strong><br>"
            for key, value in obj.metadata.items():
                if isinstance(value, (int, float)):
                    html += f"&nbsp;&nbsp;{key}: {value:.3f}<br>"
                else:
                    html += f"&nbsp;&nbsp;{key}: {value}<br>"

        html += "</div>"
        return format_html(html)

    anomaly_details.short_description = "Detailed Information"

    @admin.action(description="Mark selected anomalies as resolved")
    def mark_as_resolved(self, request, queryset):
        from django.utils import timezone

        queryset.update(is_resolved=True, resolved_at=timezone.now())
        self.message_user(
            request, f"{queryset.count()} anomaly(ies) marked as resolved."
        )

    @admin.action(description="Mark selected anomalies as unresolved")
    def mark_as_unresolved(self, request, queryset):
        queryset.update(is_resolved=False, resolved_at=None)
        self.message_user(
            request, f"{queryset.count()} anomaly(ies) marked as unresolved."
        )

    @admin.action(description="Mark selected anomalies as verified")
    def mark_as_verified(self, request, queryset):
        from django.utils import timezone

        queryset.update(
            is_verified=True, verified_by=request.user, verified_at=timezone.now()
        )
        self.message_user(
            request, f"{queryset.count()} anomaly(ies) marked as verified."
        )


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "user",
        "notification_type",
        "is_read",
        "is_sent",
        "created_at",
    ]
    list_filter = ["notification_type", "is_read", "is_sent", "created_at"]
    search_fields = ["title", "message", "user__username"]
    readonly_fields = ["id", "created_at"]
    date_hierarchy = "created_at"
    list_per_page = 20


@admin.register(UserDevice)
class UserDeviceAdmin(admin.ModelAdmin):
    list_display = ["user", "device_type", "is_active", "last_used_at", "created_at"]
    list_filter = ["device_type", "is_active", "created_at"]
    search_fields = ["user__username", "device_token"]
    readonly_fields = ["id", "created_at"]
    list_per_page = 20


@admin.register(MappedObject)
class MappedObjectAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "object_type_colored",
        "satellite_image",
        "legend_display",
        "area_display",
        "identified_by",
        "is_verified",
        "created_at",
    ]
    list_filter = [
        "object_type",
        "identified_by",
        "is_verified",
        "legend_category__category_type",
        "created_at",
    ]
    search_fields = ["name", "description", "user__username", "satellite_image__name"]
    readonly_fields = [
        "id",
        "created_at",
        "updated_at",
        "geojson_preview",
        "location_map",
    ]
    date_hierarchy = "created_at"
    list_per_page = 20
    actions = ["mark_as_verified", "mark_as_unverified"]

    fieldsets = (
        ("Basic Information", {"fields": ("id", "user", "name", "description")}),
        ("Association", {"fields": ("satellite_image", "pipeline", "legend_category")}),
        (
            "Object Details",
            {
                "fields": (
                    "object_type",
                    "geojson_file",
                    "geojson_preview",
                    "area_m2",
                    "perimeter_m",
                    "centroid_lat",
                    "centroid_lon",
                    "location_map",
                )
            },
        ),
        (
            "Identification",
            {
                "fields": (
                    "identified_by",
                    "confidence_score",
                    "is_verified",
                    "verified_by",
                    "verified_at",
                )
            },
        ),
        ("Additional Data", {"fields": ("metadata",), "classes": ("collapse",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def object_type_colored(self, obj):
        """Display object type with color coding"""
        color_map = {
            "oil_spill": "#FF0000",
            "encroachment": "#FF6600",
            "building": "#666666",
            "vehicle": "#0066FF",
            "infrastructure": "#9933FF",
            "vegetation": "#00CC00",
            "water_body": "#0099CC",
            "unknown": "#CCCCCC",
        }
        color = color_map.get(obj.object_type, "#CCCCCC")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 8px; border-radius: 3px; font-weight: bold;">{}</span>',
            color,
            obj.get_object_type_display(),
        )

    object_type_colored.short_description = "Object Type"
    object_type_colored.admin_order_field = "object_type"

    def legend_display(self, obj):
        """Display legend category with color"""
        if obj.legend_category:
            return format_html(
                '<div style="display: flex; align-items: center; gap: 5px;">'
                '<div style="width: 20px; height: 20px; background-color: {}; border: 1px solid #000; border-radius: 3px;"></div>'
                "<span>{}</span>"
                "</div>",
                obj.legend_category.color,
                obj.legend_category.name,
            )
        return "No legend"

    legend_display.short_description = "Legend"

    def area_display(self, obj):
        """Display area in user-friendly format"""
        if obj.area_m2:
            if obj.area_m2 > 10000:
                return f"{obj.area_m2 / 10000:.2f} ha"
            return f"{obj.area_m2:.2f} m²"
        return "N/A"

    area_display.short_description = "Area"
    area_display.admin_order_field = "area_m2"

    def geojson_preview(self, obj):
        """Display GeoJSON file preview link"""
        if obj.geojson_file:
            return format_html(
                '<a href="{}" target="_blank" class="button">View GeoJSON File</a>',
                obj.geojson_file.url,
            )
        return "No file"

    geojson_preview.short_description = "GeoJSON File"

    def location_map(self, obj):
        if obj.centroid_lat is not None and obj.centroid_lon is not None:
            lat = float(obj.centroid_lat)
            lon = float(obj.centroid_lon)
            link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            display = "View on Map ({:.6f}, {:.6f})".format(lat, lon)
            return format_html(
                '<a href="{}" target="_blank" style="color: #1a0dab;">{}</a>',
                link,
                display
            )
        return "No coordinates"

    location_map.short_description = "Location"

    @admin.action(description="Mark selected objects as verified")
    def mark_as_verified(self, request, queryset):
        from django.utils import timezone

        queryset.update(
            is_verified=True, verified_by=request.user, verified_at=timezone.now()
        )
        self.message_user(request, f"{queryset.count()} object(s) marked as verified.")

    @admin.action(description="Mark selected objects as unverified")
    def mark_as_unverified(self, request, queryset):
        queryset.update(is_verified=False, verified_by=None, verified_at=None)
        self.message_user(
            request, f"{queryset.count()} object(s) marked as unverified."
        )

    def save_model(self, request, obj, form, change):
        """Set user if creating new object and calculate geometry properties"""
        if not change:
            obj.user = request.user

        # Save the object first to ensure the file is properly saved
        super().save_model(request, obj, form, change)

        # Now try to extract geometry properties from GeoJSON
        # Only do this if we have a geojson_file and the geometry fields are not set
        if obj.geojson_file and not all(
            [obj.centroid_lat, obj.centroid_lon, obj.area_m2]
        ):
            try:
                import json
                from io import BytesIO

                # Read the file content into memory
                geojson_file = obj.geojson_file

                # Check if file is already closed or doesn't exist
                if not geojson_file:
                    return

                # Read file content safely
                try:
                    # Seek to beginning if possible
                    geojson_file.seek(0)
                    file_content = geojson_file.read()
                except (ValueError, AttributeError):
                    # File might be closed, try to reopen it
                    geojson_file.open("rb")
                    file_content = geojson_file.read()
                    geojson_file.close()

                # Parse the JSON
                if isinstance(file_content, bytes):
                    geojson_data = json.loads(file_content.decode("utf-8"))
                else:
                    geojson_data = json.loads(file_content)

                # Calculate centroid and area
                from .utils import calculate_geojson_properties

                properties = calculate_geojson_properties(geojson_data)

                if properties:
                    obj.centroid_lat = properties.get("centroid_lat")
                    obj.centroid_lon = properties.get("centroid_lon")
                    obj.area_m2 = properties.get("area_m2")
                    obj.perimeter_m = properties.get("perimeter_m")

                    # Save again with the calculated properties
                    obj.save(
                        update_fields=[
                            "centroid_lat",
                            "centroid_lon",
                            "area_m2",
                            "perimeter_m",
                        ]
                    )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error calculating GeoJSON properties: {str(e)}")
                # Don't fail the save operation, just log the error
                pass
