from rest_framework import serializers
from django.contrib.auth.models import User
import json
from .models import (
    Pipeline,
    SatelliteImage,
    Analysis,
    Anomaly,
    Notification,
    UserDevice,
    LegendCategory,
    MappedObject,
)


class LegendCategorySerializer(serializers.ModelSerializer):
    category_type_display = serializers.CharField(
        source="get_category_type_display", read_only=True
    )

    class Meta:
        model = LegendCategory
        fields = [
            "id",
            "name",
            "color",
            "icon",
            "description",
            "category_type",
            "category_type_display",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class MappedObjectSerializer(serializers.ModelSerializer):
    object_type_display = serializers.CharField(
        source="get_object_type_display", read_only=True
    )
    identified_by_display = serializers.CharField(
        source="get_identified_by_display", read_only=True
    )
    legend_category_data = LegendCategorySerializer(
        source="legend_category", read_only=True
    )
    geojson_url = serializers.SerializerMethodField()
    geojson_data = serializers.SerializerMethodField()

    class Meta:
        model = MappedObject
        fields = [
            "id",
            "name",
            "description",
            "satellite_image",
            "pipeline",
            "geojson_file",
            "geojson_url",
            "geojson_data",
            "legend_category",
            "legend_category_data",
            "object_type",
            "object_type_display",
            "area_m2",
            "perimeter_m",
            "centroid_lat",
            "centroid_lon",
            "confidence_score",
            "identified_by",
            "identified_by_display",
            "is_verified",
            "metadata",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "area_m2",
            "perimeter_m",
            "centroid_lat",
            "centroid_lon",
            "created_at",
        ]

    def get_geojson_url(self, obj):
        if obj.geojson_file:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.geojson_file.url)
        return None

    def get_geojson_data(self, obj):
        """Return parsed GeoJSON data"""
        if obj.geojson_file:
            try:
                with obj.geojson_file.open("r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name"]
        read_only_fields = ["id"]


class PipelineSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    satellite_images_count = serializers.SerializerMethodField()

    class Meta:
        model = Pipeline
        fields = [
            "id",
            "user",
            "name",
            "description",
            "geojson_file",
            "status",
            "length_km",
            "satellite_images_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "user", "created_at", "updated_at"]

    def get_satellite_images_count(self, obj):
        return obj.satellite_images.count()


class PipelineCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pipeline
        fields = ["name", "description", "geojson_file", "status", "length_km"]


class SatelliteImageSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    pipeline_name = serializers.CharField(source="pipeline.name", read_only=True)
    cog_url = serializers.SerializerMethodField()
    original_url = serializers.SerializerMethodField()
    bbox = serializers.SerializerMethodField()
    analyses_count = serializers.SerializerMethodField()
    mapped_objects_data = MappedObjectSerializer(
        source="mapped_objects", many=True, read_only=True
    )
    grouped_analysis_results = serializers.SerializerMethodField()

    class Meta:
        model = SatelliteImage
        fields = [
            "id",
            "user",
            "pipeline",
            "pipeline_name",
            "name",
            "description",
            "original_tiff",
            "original_url",
            "cog_tiff",
            "cog_url",
            "acquisition_date",
            "image_type",
            "is_cog_converted",
            "conversion_status",
            "bbox",
            "bbox_minx",
            "bbox_miny",
            "bbox_maxx",
            "bbox_maxy",
            "analyses_count",
            "mapped_objects_data",
            "grouped_analysis_results",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "cog_tiff",
            "is_cog_converted",
            "conversion_status",
            "bbox_minx",
            "bbox_miny",
            "bbox_maxx",
            "bbox_maxy",
            "created_at",
            "updated_at",
        ]

    def get_cog_url(self, obj):
        if obj.cog_tiff:
            try:
                request = self.context.get("request")
                if request and obj.cog_tiff.name:
                    return request.build_absolute_uri(obj.cog_tiff.url)
            except (ValueError, AttributeError):
                pass
        return None

    def get_original_url(self, obj):
        if obj.original_tiff:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.original_tiff.url)
        return None

    def get_bbox(self, obj):
        if all([obj.bbox_minx, obj.bbox_miny, obj.bbox_maxx, obj.bbox_maxy]):
            return {
                "minx": obj.bbox_minx,
                "miny": obj.bbox_miny,
                "maxx": obj.bbox_maxx,
                "maxy": obj.bbox_maxy,
            }
        return None

    def get_analyses_count(self, obj):
        return obj.analyses.count()

    def get_grouped_analysis_results(self, obj):
        """Get grouped analysis results by category"""
        analyses = obj.analyses.filter(status="completed")
        
        result = {
            "oil_spill_detection": None,
            "pipeline_encroachment": None,
            "object_detection": None,
        }

        # Group analyses by category
        oil_spill_analyses = analyses.filter(analysis_type__in=Analysis.OIL_SPILL_TYPES)
        encroachment_analyses = analyses.filter(analysis_type__in=Analysis.ENCROACHMENT_TYPES)
        object_detection_analyses = analyses.filter(analysis_type__in=Analysis.OBJECT_DETECTION_TYPES)

        # Process Oil Spill Detection
        if oil_spill_analyses.exists():
            oil_spill_anomalies = Anomaly.objects.filter(
                analysis__in=oil_spill_analyses,
                is_resolved=False
            )
            
            # Get all anomalies with locations
            spill_locations = []
            total_area = 0
            legends = set()
            
            for anomaly in oil_spill_anomalies:
                spill_locations.append({
                    "lat": anomaly.location_lat,
                    "lon": anomaly.location_lon,
                    "area_m2": anomaly.area_m2,
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence_score,
                })
                if anomaly.area_m2:
                    total_area += anomaly.area_m2
            
            # Get mapped objects related to oil spills
            oil_spill_objects = obj.mapped_objects.filter(
                object_type="oil_spill"
            )
            
            for spill_obj in oil_spill_objects:
                if spill_obj.legend_category:
                    legends.add(json.dumps({
                        "name": spill_obj.legend_category.name,
                        "color": spill_obj.legend_category.color,
                        "icon": spill_obj.legend_category.icon,
                    }))
            
            result["oil_spill_detection"] = {
                "detected": oil_spill_anomalies.count() > 0 or oil_spill_objects.exists(),
                "date": obj.acquisition_date.isoformat(),
                "num_spills": oil_spill_anomalies.count(),
                "total_area_m2": total_area,
                "spill_extent": f"{total_area:.2f} m²" if total_area > 0 else "N/A",
                "locations": spill_locations,
                "legends": [json.loads(l) for l in legends],
                "confidence_score": sum(a.confidence_score for a in oil_spill_analyses if a.confidence_score) / len(oil_spill_analyses) if oil_spill_analyses else 0,
                "severity": self._get_highest_severity(oil_spill_anomalies),
            }

        # Process Pipeline Encroachment
        if encroachment_analyses.exists():
            encroachment_anomalies = Anomaly.objects.filter(
                analysis__in=encroachment_analyses,
                is_resolved=False
            )
            
            encroachment_locations = []
            legends = set()
            
            for anomaly in encroachment_anomalies:
                encroachment_locations.append({
                    "lat": anomaly.location_lat,
                    "lon": anomaly.location_lon,
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence_score,
                    "type": anomaly.anomaly_type,
                })
            
            # Get mapped objects related to encroachments
            encroachment_objects = obj.mapped_objects.filter(
                object_type="encroachment"
            )
            
            for enc_obj in encroachment_objects:
                if enc_obj.legend_category:
                    legends.add(json.dumps({
                        "name": enc_obj.legend_category.name,
                        "color": enc_obj.legend_category.color,
                        "icon": enc_obj.legend_category.icon,
                    }))
            
            result["pipeline_encroachment"] = {
                "detected": encroachment_anomalies.count() > 0 or encroachment_objects.exists(),
                "date": obj.acquisition_date.isoformat(),
                "num_encroachments": encroachment_anomalies.count(),
                "locations": encroachment_locations,
                "legends": [json.loads(l) for l in legends],
                "confidence_score": sum(a.confidence_score for a in encroachment_analyses if a.confidence_score) / len(encroachment_analyses) if encroachment_analyses else 0,
                "severity": self._get_highest_severity(encroachment_anomalies),
            }

        # Process Object Detection
        if object_detection_analyses.exists():
            # Get all mapped objects for this image
            detected_objects = obj.mapped_objects.all()
            
            objects_by_type = {}
            legends = set()
            all_locations = []
            
            for obj_item in detected_objects:
                obj_type = obj_item.object_type
                if obj_type not in objects_by_type:
                    objects_by_type[obj_type] = 0
                objects_by_type[obj_type] += 1
                
                if obj_item.centroid_lat and obj_item.centroid_lon:
                    all_locations.append({
                        "lat": obj_item.centroid_lat,
                        "lon": obj_item.centroid_lon,
                        "type": obj_item.get_object_type_display(),
                        "name": obj_item.name,
                    })
                
                if obj_item.legend_category:
                    legends.add(json.dumps({
                        "name": obj_item.legend_category.name,
                        "color": obj_item.legend_category.color,
                        "icon": obj_item.legend_category.icon,
                    }))
            
            result["object_detection"] = {
                "detected": detected_objects.exists(),
                "date": obj.acquisition_date.isoformat(),
                "total_objects": detected_objects.count(),
                "objects_by_type": objects_by_type,
                "locations": all_locations,
                "legends": [json.loads(l) for l in legends],
                "confidence_score": sum(a.confidence_score for a in object_detection_analyses if a.confidence_score) / len(object_detection_analyses) if object_detection_analyses else 0,
            }

        return result

    def _get_highest_severity(self, anomalies):
        """Get the highest severity level from anomalies"""
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        highest = "low"
        highest_val = 0
        
        for anomaly in anomalies:
            val = severity_order.get(anomaly.severity, 0)
            if val > highest_val:
                highest_val = val
                highest = anomaly.severity
        
        return highest


class SatelliteImageCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SatelliteImage
        fields = [
            "name",
            "description",
            "original_tiff",
            "pipeline",
            "acquisition_date",
            "image_type",
        ]


class AnalysisSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    satellite_image_name = serializers.CharField(
        source="satellite_image.name", read_only=True
    )
    pipeline_name = serializers.CharField(source="pipeline.name", read_only=True)
    analysis_type_display = serializers.CharField(
        source="get_analysis_type_display", read_only=True
    )
    severity_display = serializers.CharField(
        source="get_severity_display", read_only=True
    )
    anomalies_count = serializers.SerializerMethodField()
    analysis_category = serializers.SerializerMethodField()

    class Meta:
        model = Analysis
        fields = [
            "id",
            "user",
            "satellite_image",
            "satellite_image_name",
            "pipeline",
            "pipeline_name",
            "analysis_type",
            "analysis_type_display",
            "analysis_category",
            "status",
            "confidence_score",
            "severity",
            "severity_display",
            "results_json",
            "metadata",
            "processing_time_seconds",
            "error_message",
            "anomalies_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "status",
            "confidence_score",
            "severity",
            "results_json",
            "metadata",
            "processing_time_seconds",
            "error_message",
            "created_at",
            "updated_at",
        ]

    def get_anomalies_count(self, obj):
        return obj.anomalies.count()

    def get_analysis_category(self, obj):
        return obj.get_analysis_category()


class AnomalySerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    analysis_type = serializers.CharField(
        source="analysis.analysis_type", read_only=True
    )
    satellite_image_name = serializers.CharField(
        source="analysis.satellite_image.name", read_only=True
    )
    anomaly_type_display = serializers.CharField(
        source="get_anomaly_type_display", read_only=True
    )
    severity_display = serializers.CharField(
        source="get_severity_display", read_only=True
    )
    verified_by_username = serializers.CharField(
        source="verified_by.username", read_only=True
    )

    class Meta:
        model = Anomaly
        fields = [
            "id",
            "analysis",
            "user",
            "analysis_type",
            "satellite_image_name",
            "anomaly_type",
            "anomaly_type_display",
            "severity",
            "severity_display",
            "location_lat",
            "location_lon",
            "area_m2",
            "description",
            "confidence_score",
            "is_verified",
            "is_resolved",
            "verified_by",
            "verified_by_username",
            "verified_at",
            "resolved_at",
            "metadata",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "verified_by",
            "verified_at",
            "resolved_at",
            "created_at",
            "updated_at",
        ]


class NotificationSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    anomaly_type = serializers.CharField(source="anomaly.anomaly_type", read_only=True)
    notification_type_display = serializers.CharField(
        source="get_notification_type_display", read_only=True
    )

    class Meta:
        model = Notification
        fields = [
            "id",
            "user",
            "anomaly",
            "anomaly_type",
            "notification_type",
            "notification_type_display",
            "title",
            "message",
            "is_read",
            "is_sent",
            "sent_at",
            "read_at",
            "metadata",
            "created_at",
        ]
        read_only_fields = ["id", "user", "is_sent", "sent_at", "read_at", "created_at"]


class UserDeviceSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    device_type_display = serializers.CharField(
        source="get_device_type_display", read_only=True
    )

    class Meta:
        model = UserDevice
        fields = [
            "id",
            "user",
            "device_token",
            "device_type",
            "device_type_display",
            "is_active",
            "last_used_at",
            "created_at",
        ]
        read_only_fields = ["id", "user", "last_used_at", "created_at"]