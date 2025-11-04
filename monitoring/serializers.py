from rest_framework import serializers
from django.contrib.auth.models import User
from .models import (
    Pipeline,
    SatelliteImage,
    Analysis,
    Anomaly,
    Notification,
    UserDevice
)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']


class PipelineSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    satellite_images_count = serializers.SerializerMethodField()

    class Meta:
        model = Pipeline
        fields = [
            'id',
            'user',
            'name',
            'description',
            'geojson_file',
            'status',
            'length_km',
            'satellite_images_count',
            'created_at',
            'updated_at'
        ]
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']

    def get_satellite_images_count(self, obj):
        return obj.satellite_images.count()


class PipelineCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pipeline
        fields = ['name', 'description', 'geojson_file', 'status', 'length_km']


class SatelliteImageSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    pipeline_name = serializers.CharField(source='pipeline.name', read_only=True)
    cog_url = serializers.SerializerMethodField()
    original_url = serializers.SerializerMethodField()
    bbox = serializers.SerializerMethodField()
    analyses_count = serializers.SerializerMethodField()

    class Meta:
        model = SatelliteImage
        fields = [
            'id',
            'user',
            'pipeline',
            'pipeline_name',
            'name',
            'description',
            'original_tiff',
            'original_url',
            'cog_tiff',
            'cog_url',
            'acquisition_date',
            'image_type',
            'is_cog_converted',
            'conversion_status',
            'bbox',
            'bbox_minx',
            'bbox_miny',
            'bbox_maxx',
            'bbox_maxy',
            'analyses_count',
            'created_at',
            'updated_at'
        ]
        read_only_fields = [
            'id',
            'user',
            'cog_tiff',
            'is_cog_converted',
            'conversion_status',
            'bbox_minx',
            'bbox_miny',
            'bbox_maxx',
            'bbox_maxy',
            'created_at',
            'updated_at'
        ]

    def get_cog_url(self, obj):
        if obj.cog_tiff:
            try:
                request = self.context.get('request')
                if request and obj.cog_tiff.name:
                    return request.build_absolute_uri(obj.cog_tiff.url)
            except (ValueError, AttributeError):
                # File field might be empty or file doesn't exist
                pass
        return None

    def get_original_url(self, obj):
        if obj.original_tiff:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.original_tiff.url)
        return None

    def get_bbox(self, obj):
        if all([obj.bbox_minx, obj.bbox_miny, obj.bbox_maxx, obj.bbox_maxy]):
            return {
                'minx': obj.bbox_minx,
                'miny': obj.bbox_miny,
                'maxx': obj.bbox_maxx,
                'maxy': obj.bbox_maxy
            }
        return None

    def get_analyses_count(self, obj):
        return obj.analyses.count()


class SatelliteImageCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SatelliteImage
        fields = [
            'name',
            'description',
            'original_tiff',
            'pipeline',
            'acquisition_date',
            'image_type'
        ]


class AnalysisSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    satellite_image_name = serializers.CharField(source='satellite_image.name', read_only=True)
    pipeline_name = serializers.CharField(source='pipeline.name', read_only=True)
    analysis_type_display = serializers.CharField(source='get_analysis_type_display', read_only=True)
    severity_display = serializers.CharField(source='get_severity_display', read_only=True)
    anomalies_count = serializers.SerializerMethodField()

    class Meta:
        model = Analysis
        fields = [
            'id',
            'user',
            'satellite_image',
            'satellite_image_name',
            'pipeline',
            'pipeline_name',
            'analysis_type',
            'analysis_type_display',
            'status',
            'confidence_score',
            'severity',
            'severity_display',
            'results_json',
            'metadata',
            'processing_time_seconds',
            'error_message',
            'anomalies_count',
            'created_at',
            'updated_at'
        ]
        read_only_fields = [
            'id',
            'user',
            'status',
            'confidence_score',
            'severity',
            'results_json',
            'metadata',
            'processing_time_seconds',
            'error_message',
            'created_at',
            'updated_at'
        ]

    def get_anomalies_count(self, obj):
        return obj.anomalies.count()


class AnomalySerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    analysis_type = serializers.CharField(source='analysis.analysis_type', read_only=True)
    satellite_image_name = serializers.CharField(
        source='analysis.satellite_image.name',
        read_only=True
    )
    anomaly_type_display = serializers.CharField(source='get_anomaly_type_display', read_only=True)
    severity_display = serializers.CharField(source='get_severity_display', read_only=True)
    verified_by_username = serializers.CharField(source='verified_by.username', read_only=True)

    class Meta:
        model = Anomaly
        fields = [
            'id',
            'analysis',
            'user',
            'analysis_type',
            'satellite_image_name',
            'anomaly_type',
            'anomaly_type_display',
            'severity',
            'severity_display',
            'location_lat',
            'location_lon',
            'area_m2',
            'description',
            'confidence_score',
            'is_verified',
            'is_resolved',
            'verified_by',
            'verified_by_username',
            'verified_at',
            'resolved_at',
            'metadata',
            'created_at',
            'updated_at'
        ]
        read_only_fields = [
            'id',
            'user',
            'verified_by',
            'verified_at',
            'resolved_at',
            'created_at',
            'updated_at'
        ]


class NotificationSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    anomaly_type = serializers.CharField(source='anomaly.anomaly_type', read_only=True)
    notification_type_display = serializers.CharField(
        source='get_notification_type_display',
        read_only=True
    )

    class Meta:
        model = Notification
        fields = [
            'id',
            'user',
            'anomaly',
            'anomaly_type',
            'notification_type',
            'notification_type_display',
            'title',
            'message',
            'is_read',
            'is_sent',
            'sent_at',
            'read_at',
            'metadata',
            'created_at'
        ]
        read_only_fields = [
            'id',
            'user',
            'is_sent',
            'sent_at',
            'read_at',
            'created_at'
        ]


class UserDeviceSerializer(serializers.ModelSerializer):
    user = serializers.StringRelatedField(read_only=True)
    device_type_display = serializers.CharField(source='get_device_type_display', read_only=True)

    class Meta:
        model = UserDevice
        fields = [
            'id',
            'user',
            'device_token',
            'device_type',
            'device_type_display',
            'is_active',
            'last_used_at',
            'created_at'
        ]
        read_only_fields = ['id', 'user', 'last_used_at', 'created_at']

