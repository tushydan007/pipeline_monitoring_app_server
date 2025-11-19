
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
import uuid


class LegendCategory(models.Model):
    """Model for legend categories used in GeoJSON object classification"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="legend_categories"
    )
    name = models.CharField(
        max_length=100,
        help_text='Category name (e.g., "Oil Spill", "Building", "Vegetation")',
    )
    color = models.CharField(
        max_length=7,
        default="#FF0000",
        help_text="Hex color code for this category (e.g., #FF0000 for red)",
    )
    icon = models.CharField(
        max_length=50, blank=True, help_text="Icon name or emoji for this category"
    )
    description = models.TextField(blank=True)
    category_type = models.CharField(
        max_length=50,
        choices=[
            ("oil_spill", "Oil Spill"),
            ("encroachment", "Pipeline Encroachment"),
            ("infrastructure", "Infrastructure"),
            ("vegetation", "Vegetation"),
            ("water_body", "Water Body"),
            ("building", "Building"),
            ("vehicle", "Vehicle"),
            ("other", "Other"),
        ],
        default="other",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "Legend Category"
        verbose_name_plural = "Legend Categories"
        unique_together = ["user", "name"]

    def __str__(self):
        return f"{self.name} ({self.color})"


class MappedObject(models.Model):
    """Model for mapped objects from GIS team with GeoJSON data"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="mapped_objects"
    )
    satellite_image = models.ForeignKey(
        "SatelliteImage",
        on_delete=models.CASCADE,
        related_name="mapped_objects",
        help_text="The satellite image this object was mapped on",
    )
    pipeline = models.ForeignKey(
        "Pipeline",
        on_delete=models.CASCADE,
        related_name="mapped_objects",
        null=True,
        blank=True,
        help_text="Associated pipeline (optional)",
    )
    name = models.CharField(
        max_length=255, help_text="Name/identifier for this mapped object"
    )
    description = models.TextField(blank=True)
    geojson_file = models.FileField(
        upload_to="mapped_objects/geojson/",
        validators=[FileExtensionValidator(allowed_extensions=["json", "geojson"])],
        help_text="Upload a GeoJSON file containing the mapped object geometry",
    )
    legend_category = models.ForeignKey(
        LegendCategory,
        on_delete=models.SET_NULL,
        null=True,
        related_name="mapped_objects",
        help_text="Legend category for this object",
    )
    object_type = models.CharField(
        max_length=50,
        choices=[
            ("oil_spill", "Oil Spill"),
            ("encroachment", "Pipeline Encroachment"),
            ("building", "Building"),
            ("vehicle", "Vehicle"),
            ("infrastructure", "Infrastructure"),
            ("vegetation", "Vegetation"),
            ("water_body", "Water Body"),
            ("unknown", "Unknown/Other"),
        ],
        default="unknown",
    )
    area_m2 = models.FloatField(
        null=True, blank=True, help_text="Area in square meters"
    )
    perimeter_m = models.FloatField(
        null=True, blank=True, help_text="Perimeter in meters"
    )
    centroid_lat = models.FloatField(null=True, blank=True)
    centroid_lon = models.FloatField(null=True, blank=True)
    confidence_score = models.FloatField(
        null=True, blank=True, help_text="Confidence score (0-1) if identified by AI"
    )
    identified_by = models.CharField(
        max_length=20,
        choices=[
            ("manual", "Manual Mapping"),
            ("ai", "AI Detection"),
            ("hybrid", "Hybrid (AI + Manual)"),
        ],
        default="manual",
    )
    metadata = models.JSONField(
        default=dict, blank=True, help_text="Additional metadata"
    )
    is_verified = models.BooleanField(default=False, help_text="Verified by expert")
    verified_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="verified_objects",
    )
    verified_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Mapped Object"
        verbose_name_plural = "Mapped Objects"
        indexes = [
            models.Index(fields=["user", "satellite_image"]),
            models.Index(fields=["object_type", "created_at"]),
        ]

    def __str__(self):
        return f"{self.name} - {self.get_object_type_display()}"


class Pipeline(models.Model):
    """Model for pipeline routes stored as GeoJSON"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="pipelines")
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    geojson_file = models.FileField(
        upload_to="pipelines/geojson/",
        validators=[FileExtensionValidator(allowed_extensions=["json", "geojson"])],
        help_text="Upload a GeoJSON file (.json or .geojson) containing the pipeline route geometry.",
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ("active", "Active"),
            ("inactive", "Inactive"),
            ("maintenance", "Maintenance"),
        ],
        default="active",
    )
    length_km = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Pipeline"
        verbose_name_plural = "Pipelines"

    def __str__(self):
        return f"{self.name} - {self.user.username}"


class SatelliteImage(models.Model):
    """Model for satellite imagery in TIFF format"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="satellite_images"
    )
    pipeline = models.ForeignKey(
        Pipeline,
        on_delete=models.CASCADE,
        related_name="satellite_images",
        null=True,
        blank=True,
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    original_tiff = models.FileField(
        upload_to="satellite/original/",
        validators=[FileExtensionValidator(allowed_extensions=["tif", "tiff"])],
    )
    cog_tiff = models.FileField(
        upload_to="satellite/cog/",
        null=True,
        blank=True,
        validators=[FileExtensionValidator(allowed_extensions=["tif", "tiff"])],
    )
    acquisition_date = models.DateTimeField()
    image_type = models.CharField(
        max_length=50,
        choices=[
            ("optical", "Optical"),
            ("sar", "SAR"),
            ("thermal", "Thermal"),
            ("multispectral", "Multispectral"),
        ],
        default="optical",
    )
    is_cog_converted = models.BooleanField(default=False)
    conversion_status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
    )
    bbox_minx = models.FloatField(null=True, blank=True)
    bbox_miny = models.FloatField(null=True, blank=True)
    bbox_maxx = models.FloatField(null=True, blank=True)
    bbox_maxy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-acquisition_date"]
        verbose_name = "Satellite Image"
        verbose_name_plural = "Satellite Images"

    def __str__(self):
        return f"{self.name} - {self.user.username}"


class Analysis(models.Model):
    """Model for pipeline monitoring analysis results"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="analyses")
    satellite_image = models.ForeignKey(
        SatelliteImage, on_delete=models.CASCADE, related_name="analyses"
    )
    pipeline = models.ForeignKey(
        Pipeline,
        on_delete=models.CASCADE,
        related_name="analyses",
        null=True,
        blank=True,
    )
    analysis_type = models.CharField(
        max_length=50,
        choices=[
            ("backscatter_threshold", "Backscatter Thresholding"),
            ("glcm_texture", "GLCM Texture Analysis"),
            ("polarimetric_decomp", "Polarimetric Decomposition"),
            ("time_series", "Time-series Tracking"),
            ("insar", "InSAR"),
            ("dinsar", "DInSAR"),
            ("psinsar", "PS-InSAR"),
            ("sbas_insar", "SBAS-InSAR"),
            ("phase_unwrap", "Phase Unwrapping"),
            ("atmospheric_corr", "Atmospheric Correction"),
            ("cfar", "CFAR Detection"),
            ("polarimetric_scatter", "Polarimetric Scattering Analysis"),
            ("texture_shape", "Texture/Shape Metrics"),
            ("coherence_change", "Coherence Change Detection"),
            ("deep_learning", "Deep Learning Analysis"),
            ("object_identification", "Object Identification"),  # NEW
            ("oil_spill_detection", "Oil Spill Detection"),  # NEW
            ("pipeline_encroachment", "Pipeline Encroachment"),  # NEW
        ],
    )
    mapped_objects = models.ManyToManyField(
        MappedObject,
        blank=True,
        related_name="analyses",
        help_text="Mapped objects associated with this analysis",
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
    )
    confidence_score = models.FloatField(
        null=True, blank=True, help_text="Confidence score (0-1)"
    )
    severity = models.CharField(
        max_length=20,
        choices=[
            ("low", "Low"),
            ("medium", "Medium"),
            ("high", "High"),
            ("critical", "Critical"),
        ],
        null=True,
        blank=True,
    )
    results_json = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    processing_time_seconds = models.FloatField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Analysis"
        verbose_name_plural = "Analyses"
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["satellite_image", "analysis_type"]),
        ]

    def get_analysis_summary(self):
        """Get user-friendly summary for dashboard display"""
        summary = {
            "header": self.get_analysis_type_display(),
            "date": self.created_at,
            "confidence": self.confidence_score,
            "severity": self.severity,
            "location": None,
            "coordinates": None,
            "legends": [],
            "specific_data": {},
        }

        # Get associated mapped objects with their legends
        if self.mapped_objects.exists():
            legends = []
            for obj in self.mapped_objects.all():
                if obj.legend_category:
                    legends.append(
                        {
                            "name": obj.legend_category.name,
                            "color": obj.legend_category.color,
                            "icon": obj.legend_category.icon,
                            "type": obj.legend_category.category_type,
                        }
                    )
                if obj.centroid_lat and obj.centroid_lon:
                    summary["coordinates"] = {
                        "lat": obj.centroid_lat,
                        "lon": obj.centroid_lon,
                    }
            summary["legends"] = legends

        # Analysis type specific data
        if self.analysis_type in ["oil_spill_detection", "object_identification"]:
            oil_spills = self.mapped_objects.filter(object_type="oil_spill")
            if oil_spills.exists():
                total_extent = sum(obj.area_m2 or 0 for obj in oil_spills)
                summary["specific_data"] = {
                    "spill_extent": f"{total_extent:.2f} m²",
                    "num_spills": oil_spills.count(),
                }

        elif self.analysis_type == "pipeline_encroachment":
            encroachments = self.mapped_objects.filter(object_type="encroachment")
            summary["specific_data"] = {
                "num_encroachments": encroachments.count(),
                "total_area": sum(obj.area_m2 or 0 for obj in encroachments),
            }

        elif self.analysis_type == "object_identification":
            objects_by_type = {}
            for obj in self.mapped_objects.all():
                obj_type = obj.get_object_type_display()
                objects_by_type[obj_type] = objects_by_type.get(obj_type, 0) + 1
            summary["specific_data"] = {
                "objects_identified": objects_by_type,
                "total_objects": self.mapped_objects.count(),
            }

        return summary

    def __str__(self):
        return f"{self.get_analysis_type_display()} - {self.satellite_image.name}"

    # Analysis.get_analysis_summary = get_analysis_summary


class Anomaly(models.Model):
    """Model for detected anomalies in pipeline monitoring"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis = models.ForeignKey(
        Analysis, on_delete=models.CASCADE, related_name="anomalies"
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="anomalies")
    anomaly_type = models.CharField(
        max_length=50,
        choices=[
            ("displacement", "Ground Displacement"),
            ("subsidence", "Subsidence"),
            ("deformation", "Surface Deformation"),
            ("scattering_change", "Scattering Pattern Change"),
            ("coherence_loss", "Coherence Loss"),
            ("texture_anomaly", "Texture Anomaly"),
            ("phase_anomaly", "Phase Anomaly"),
            ("intensity_change", "Intensity Change"),
            ("structural_change", "Structural Change"),
            ("temporal_anomaly", "Temporal Anomaly"),
        ],
    )
    severity = models.CharField(
        max_length=20,
        choices=[
            ("low", "Low"),
            ("medium", "Medium"),
            ("high", "High"),
            ("critical", "Critical"),
        ],
    )
    location_lat = models.FloatField()
    location_lon = models.FloatField()
    area_m2 = models.FloatField(null=True, blank=True)
    description = models.TextField(blank=True)
    confidence_score = models.FloatField(help_text="Confidence score (0-1)")
    is_verified = models.BooleanField(default=False)
    is_resolved = models.BooleanField(default=False)
    verified_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="verified_anomalies",
    )
    verified_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-severity", "-created_at"]
        verbose_name = "Anomaly"
        verbose_name_plural = "Anomalies"
        indexes = [
            models.Index(fields=["user", "is_resolved"]),
            models.Index(fields=["severity", "created_at"]),
        ]

    def __str__(self):
        return f"{self.get_anomaly_type_display()} - {self.get_severity_display()}"


class Notification(models.Model):
    """Model for user notifications (email and push)"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="notifications"
    )
    anomaly = models.ForeignKey(
        Anomaly,
        on_delete=models.CASCADE,
        related_name="notifications",
        null=True,
        blank=True,
    )
    notification_type = models.CharField(
        max_length=20,
        choices=[
            ("email", "Email"),
            ("push", "Push Notification"),
            ("both", "Both"),
        ],
        default="both",
    )
    title = models.CharField(max_length=255)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    is_sent = models.BooleanField(default=False)
    sent_at = models.DateTimeField(null=True, blank=True)
    read_at = models.DateTimeField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Notification"
        verbose_name_plural = "Notifications"
        indexes = [
            models.Index(fields=["user", "is_read"]),
        ]

    def __str__(self):
        return f"{self.title} - {self.user.username}"


class UserDevice(models.Model):
    """Model for storing user devices for push notifications"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="devices")
    device_token = models.CharField(max_length=500, unique=True)
    device_type = models.CharField(
        max_length=20,
        choices=[
            ("web", "Web"),
            ("ios", "iOS"),
            ("android", "Android"),
        ],
    )
    is_active = models.BooleanField(default=True)
    last_used_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "User Device"
        verbose_name_plural = "User Devices"

    def __str__(self):
        return f"{self.user.username} - {self.device_type}"
