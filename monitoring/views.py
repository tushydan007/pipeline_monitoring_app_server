from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from django.db.models import Q
from django.http import FileResponse
from django.utils import timezone
from .models import (
    Pipeline,
    SatelliteImage,
    Analysis,
    Anomaly,
    Notification,
    UserDevice
)
from .serializers import (
    PipelineSerializer,
    PipelineCreateSerializer,
    SatelliteImageSerializer,
    SatelliteImageCreateSerializer,
    AnalysisSerializer,
    AnomalySerializer,
    NotificationSerializer,
    UserDeviceSerializer,
    UserSerializer
)
from .tasks import convert_to_cog_task, run_analysis_task
from .utils import extract_geotiff_bbox


class PipelineViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing pipelines
    """
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'description']
    ordering_fields = ['created_at', 'name', 'status']
    ordering = ['-created_at']

    def get_queryset(self):
        return Pipeline.objects.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action == 'create':
            return PipelineCreateSerializer
        return PipelineSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['get'])
    def geojson(self, request, pk=None):
        """Retrieve GeoJSON file"""
        import json
        pipeline = self.get_object()
        if pipeline.geojson_file:
            try:
                # Read and parse the GeoJSON file
                with pipeline.geojson_file.open('r') as f:
                    geojson_data = json.load(f)
                return Response(geojson_data, content_type='application/json')
            except json.JSONDecodeError:
                return Response(
                    {'error': 'Invalid GeoJSON file format'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            except Exception as e:
                return Response(
                    {'error': f'Error reading GeoJSON file: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        return Response(
            {'error': 'GeoJSON file not found'},
            status=status.HTTP_404_NOT_FOUND
        )


class SatelliteImageViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing satellite images
    """
    permission_classes = [IsAuthenticated]
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['name', 'description']
    ordering_fields = ['acquisition_date', 'created_at', 'name']
    ordering = ['-acquisition_date']

    def get_queryset(self):
        queryset = SatelliteImage.objects.filter(user=self.request.user)
        pipeline_id = self.request.query_params.get('pipeline', None)
        if pipeline_id:
            queryset = queryset.filter(pipeline_id=pipeline_id)
        return queryset

    def get_serializer_class(self):
        if self.action == 'create':
            return SatelliteImageCreateSerializer
        return SatelliteImageSerializer

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

    def perform_create(self, serializer):
        image = serializer.save(user=self.request.user)
        
        # Extract bbox from original_tiff if available
        if image.original_tiff and not all([image.bbox_minx, image.bbox_miny, image.bbox_maxx, image.bbox_maxy]):
            try:
                tiff_path = image.original_tiff.path
                bbox = extract_geotiff_bbox(tiff_path)
                if bbox:
                    image.bbox_minx = bbox['minx']
                    image.bbox_miny = bbox['miny']
                    image.bbox_maxx = bbox['maxx']
                    image.bbox_maxy = bbox['maxy']
                    image.save(update_fields=['bbox_minx', 'bbox_miny', 'bbox_maxx', 'bbox_maxy'])
            except Exception as e:
                # Log error but don't fail the create
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to extract bbox for image {image.id}: {str(e)}")

    @action(detail=True, methods=['post'])
    def convert_to_cog(self, request, pk=None):
        """Trigger COG conversion"""
        image = self.get_object()
        if image.is_cog_converted:
            return Response(
                {'message': 'Image already converted to COG'},
                status=status.HTTP_400_BAD_REQUEST
            )
        convert_to_cog_task.delay(str(image.id))
        return Response({'message': 'COG conversion queued'})

    @action(detail=True, methods=['post'])
    def run_analysis(self, request, pk=None):
        """Trigger analysis"""
        image = self.get_object()
        if not image.is_cog_converted:
            return Response(
                {'message': 'Image must be converted to COG first'},
                status=status.HTTP_400_BAD_REQUEST
            )
        run_analysis_task.delay(str(image.id))
        return Response({'message': 'Analysis queued'})

    @action(detail=True, methods=['post'])
    def extract_bbox(self, request, pk=None):
        """Extract and update bounding box from image file"""
        from .utils import extract_geotiff_bbox
        
        image = self.get_object()
        
        # Try to use COG if available, otherwise use original
        tiff_path = None
        if image.is_cog_converted and image.cog_tiff:
            try:
                tiff_path = image.cog_tiff.path
            except (ValueError, AttributeError):
                pass
        
        if not tiff_path and image.original_tiff:
            try:
                tiff_path = image.original_tiff.path
            except (ValueError, AttributeError):
                pass
        
        if not tiff_path:
            return Response(
                {'error': 'No image file available'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        bbox = extract_geotiff_bbox(tiff_path)
        if bbox:
            image.bbox_minx = bbox['minx']
            image.bbox_miny = bbox['miny']
            image.bbox_maxx = bbox['maxx']
            image.bbox_maxy = bbox['maxy']
            image.save(update_fields=['bbox_minx', 'bbox_miny', 'bbox_maxx', 'bbox_maxy'])
            
            serializer = self.get_serializer(image)
            return Response(serializer.data)
        else:
            return Response(
                {'error': 'Could not extract bounding box from image'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get'])
    def display_image(self, request, pk=None):
        """Serve satellite image as PNG for display in browser"""
        try:
            from PIL import Image as PILImage
        except ImportError:
            return Response(
                {'error': 'PIL/Pillow is required for image processing'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        from .utils import extract_geotiff_bbox
        import rasterio
        import numpy as np
        from io import BytesIO
        
        image = self.get_object()
        
        # Try to use COG if available, otherwise use original
        tiff_path = None
        if image.is_cog_converted and image.cog_tiff:
            try:
                tiff_path = image.cog_tiff.path
            except ValueError:
                # File field might be empty or path doesn't exist
                tiff_path = None
        
        if not tiff_path and image.original_tiff:
            try:
                tiff_path = image.original_tiff.path
            except ValueError:
                pass
        
        if not tiff_path:
            return Response(
                {'error': 'No image file available'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Update bbox if missing
        if not all([image.bbox_minx, image.bbox_miny, image.bbox_maxx, image.bbox_maxy]):
            bbox = extract_geotiff_bbox(tiff_path)
            if bbox:
                image.bbox_minx = bbox['minx']
                image.bbox_miny = bbox['miny']
                image.bbox_maxx = bbox['maxx']
                image.bbox_maxy = bbox['maxy']
                image.save(update_fields=['bbox_minx', 'bbox_miny', 'bbox_maxx', 'bbox_maxy'])
        
        try:
            with rasterio.open(tiff_path) as src:
                # Read image data
                if src.count >= 3:
                    # RGB image
                    red = src.read(1)
                    green = src.read(2)
                    blue = src.read(3)
                    # Normalize to 0-255
                    red_norm = ((red - red.min()) / (red.max() - red.min() + 1e-8) * 255).astype(np.uint8)
                    green_norm = ((green - green.min()) / (green.max() - green.min() + 1e-8) * 255).astype(np.uint8)
                    blue_norm = ((blue - blue.min()) / (blue.max() - blue.min() + 1e-8) * 255).astype(np.uint8)
                    rgb_array = np.dstack([red_norm, green_norm, blue_norm])
                elif src.count == 1:
                    # Grayscale image - convert to RGB
                    gray = src.read(1)
                    gray_norm = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np.uint8)
                    rgb_array = np.dstack([gray_norm, gray_norm, gray_norm])
                else:
                    # Use first band
                    band = src.read(1)
                    band_norm = ((band - band.min()) / (band.max() - band.min() + 1e-8) * 255).astype(np.uint8)
                    rgb_array = np.dstack([band_norm, band_norm, band_norm])
                
                # Resize if too large (max 2000px on longest side)
                max_dimension = 2000
                height, width = rgb_array.shape[:2]
                if width > max_dimension or height > max_dimension:
                    scale = max_dimension / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    rgb_array = np.array(PILImage.fromarray(rgb_array).resize((new_width, new_height), PILImage.Resampling.LANCZOS))
                
                # Convert to PIL Image and then to PNG
                img = PILImage.fromarray(rgb_array)
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                
                from django.http import HttpResponse
                response = HttpResponse(buffer.read(), content_type='image/png')
                response['Content-Disposition'] = f'inline; filename="{image.name}.png"'
                return response
                
        except Exception as e:
            return Response(
                {'error': f'Error processing image: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AnalysisViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing analyses (read-only)
    """
    permission_classes = [IsAuthenticated]
    serializer_class = AnalysisSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['satellite_image__name', 'pipeline__name']
    ordering_fields = ['created_at', 'confidence_score', 'severity']
    ordering = ['-created_at']

    def get_queryset(self):
        queryset = Analysis.objects.filter(user=self.request.user)
        satellite_image_id = self.request.query_params.get('satellite_image', None)
        pipeline_id = self.request.query_params.get('pipeline', None)
        analysis_type = self.request.query_params.get('analysis_type', None)
        severity = self.request.query_params.get('severity', None)

        if satellite_image_id:
            queryset = queryset.filter(satellite_image_id=satellite_image_id)
        if pipeline_id:
            queryset = queryset.filter(pipeline_id=pipeline_id)
        if analysis_type:
            queryset = queryset.filter(analysis_type=analysis_type)
        if severity:
            queryset = queryset.filter(severity=severity)

        return queryset


class AnomalyViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing anomalies
    """
    permission_classes = [IsAuthenticated]
    serializer_class = AnomalySerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['description', 'anomaly_type']
    ordering_fields = ['created_at', 'severity', 'confidence_score']
    ordering = ['-severity', '-created_at']

    def get_queryset(self):
        queryset = Anomaly.objects.filter(user=self.request.user)
        is_resolved = self.request.query_params.get('is_resolved', None)
        severity = self.request.query_params.get('severity', None)
        anomaly_type = self.request.query_params.get('anomaly_type', None)

        if is_resolved is not None:
            queryset = queryset.filter(is_resolved=is_resolved.lower() == 'true')
        if severity:
            queryset = queryset.filter(severity=severity)
        if anomaly_type:
            queryset = queryset.filter(anomaly_type=anomaly_type)

        return queryset

    @action(detail=True, methods=['post'])
    def mark_resolved(self, request, pk=None):
        """Mark anomaly as resolved"""
        anomaly = self.get_object()
        anomaly.is_resolved = True
        from django.utils import timezone
        anomaly.resolved_at = timezone.now()
        anomaly.save()
        return Response({'message': 'Anomaly marked as resolved'})

    @action(detail=True, methods=['post'])
    def mark_unresolved(self, request, pk=None):
        """Mark anomaly as unresolved"""
        anomaly = self.get_object()
        anomaly.is_resolved = False
        anomaly.resolved_at = None
        anomaly.save()
        return Response({'message': 'Anomaly marked as unresolved'})


class NotificationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing notifications
    """
    permission_classes = [IsAuthenticated]
    serializer_class = NotificationSerializer
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['created_at', 'is_read']
    ordering = ['-created_at']

    def get_queryset(self):
        queryset = Notification.objects.filter(user=self.request.user)
        is_read = self.request.query_params.get('is_read', None)
        if is_read is not None:
            queryset = queryset.filter(is_read=is_read.lower() == 'true')
        return queryset

    @action(detail=True, methods=['post'])
    def mark_read(self, request, pk=None):
        """Mark notification as read"""
        notification = self.get_object()
        notification.is_read = True
        from django.utils import timezone
        notification.read_at = timezone.now()
        notification.save()
        return Response({'message': 'Notification marked as read'})

    @action(detail=False, methods=['post'])
    def mark_all_read(self, request):
        """Mark all notifications as read"""
        count = Notification.objects.filter(
            user=request.user,
            is_read=False
        ).update(
            is_read=True,
            read_at=timezone.now()
        )
        from django.utils import timezone
        return Response({'message': f'{count} notification(s) marked as read'})

    @action(detail=False, methods=['get'])
    def unread_count(self, request):
        """Get count of unread notifications"""
        count = Notification.objects.filter(
            user=request.user,
            is_read=False
        ).count()
        return Response({'unread_count': count})


class UserDeviceViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing user devices for push notifications
    """
    permission_classes = [IsAuthenticated]
    serializer_class = UserDeviceSerializer

    def get_queryset(self):
        return UserDevice.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing user information
    """
    permission_classes = [IsAuthenticated]
    serializer_class = UserSerializer

    def get_queryset(self):
        return User.objects.filter(id=self.request.user.id)

    @action(detail=False, methods=['get'])
    def me(self, request):
        """Get current user information"""
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)
