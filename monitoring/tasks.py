from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
import os
import json
from typing import Dict, Any
from .models import SatelliteImage, Analysis, Anomaly, Notification
from .utils import convert_to_cog, extract_geotiff_bbox, run_pipeline_analysis


@shared_task(bind=True, max_retries=3)
def convert_to_cog_task(self, image_id: str):
    """Convert satellite image to Cloud Optimized GeoTIFF"""
    try:
        image = SatelliteImage.objects.get(id=image_id)
        image.conversion_status = 'processing'
        image.save()

        cog_path = convert_to_cog(image.original_tiff.path)
        
        # Update the model with COG file
        image.cog_tiff.name = cog_path.replace(settings.MEDIA_ROOT + '/', '')
        image.is_cog_converted = True
        image.conversion_status = 'completed'

        # Extract bounding box
        bbox = extract_geotiff_bbox(cog_path)
        if bbox:
            image.bbox_minx = bbox['minx']
            image.bbox_miny = bbox['miny']
            image.bbox_maxx = bbox['maxx']
            image.bbox_maxy = bbox['maxy']

        image.save()
        return {'status': 'success', 'image_id': str(image_id)}

    except SatelliteImage.DoesNotExist:
        return {'status': 'error', 'message': 'Image not found'}
    except Exception as exc:
        image = SatelliteImage.objects.get(id=image_id)
        image.conversion_status = 'failed'
        image.save()
        raise self.retry(exc=exc, countdown=60)


@shared_task(bind=True, max_retries=3)
def run_analysis_task(self, image_id: str):
    """Run comprehensive pipeline monitoring analysis"""
    try:
        image = SatelliteImage.objects.get(id=image_id)
        if not image.is_cog_converted:
            return {
                'status': 'error',
                'message': 'Image must be converted to COG first'
            }

        start_time = timezone.now()

        # Run analysis
        results = run_pipeline_analysis(image.cog_tiff.path, image)

        # Create Analysis objects for each analysis type
        analysis_objects = []
        for analysis_type, result_data in results.items():
            analysis = Analysis.objects.create(
                user=image.user,
                satellite_image=image,
                pipeline=image.pipeline,
                analysis_type=analysis_type,
                status='completed',
                confidence_score=result_data.get('confidence_score'),
                severity=result_data.get('severity'),
                results_json=result_data.get('results', {}),
                metadata=result_data.get('metadata', {}),
                processing_time_seconds=result_data.get('processing_time', 0),
            )
            analysis_objects.append(analysis)

            # Create anomalies if detected
            if result_data.get('anomalies'):
                for anomaly_data in result_data['anomalies']:
                    anomaly = Anomaly.objects.create(
                        analysis=analysis,
                        user=image.user,
                        anomaly_type=anomaly_data['type'],
                        severity=anomaly_data['severity'],
                        location_lat=anomaly_data['location'][0],
                        location_lon=anomaly_data['location'][1],
                        area_m2=anomaly_data.get('area_m2'),
                        description=anomaly_data.get('description', ''),
                        confidence_score=anomaly_data['confidence'],
                        metadata=anomaly_data.get('metadata', {}),
                    )

                    # Create notification for critical and high severity anomalies
                    if anomaly.severity in ['critical', 'high']:
                        Notification.objects.create(
                            user=image.user,
                            anomaly=anomaly,
                            notification_type='both',
                            title=f'{anomaly.get_anomaly_type_display()} Detected',
                            message=(
                                f'A {anomaly.get_severity_display().lower()} severity '
                                f'{anomaly.get_anomaly_type_display()} has been detected '
                                f'at location ({anomaly.location_lat:.6f}, {anomaly.location_lon:.6f}).'
                            ),
                            metadata={
                                'anomaly_id': str(anomaly.id),
                                'analysis_type': analysis_type,
                            }
                        )

        # Send email notification if critical anomalies found
        critical_anomalies = Anomaly.objects.filter(
            analysis__in=analysis_objects,
            severity='critical'
        )
        if critical_anomalies.exists():
            send_analysis_notification_email.delay(str(image.user.id), len(critical_anomalies))

        processing_time = (timezone.now() - start_time).total_seconds()

        return {
            'status': 'success',
            'image_id': str(image_id),
            'analyses_created': len(analysis_objects),
            'processing_time': processing_time
        }

    except SatelliteImage.DoesNotExist:
        return {'status': 'error', 'message': 'Image not found'}
    except Exception as exc:
        # Update analysis status to failed
        Analysis.objects.filter(
            satellite_image_id=image_id,
            status='processing'
        ).update(status='failed', error_message=str(exc))
        raise self.retry(exc=exc, countdown=120)


@shared_task
def send_analysis_notification_email(user_id: str, anomaly_count: int):
    """Send email notification about analysis results"""
    try:
        from django.contrib.auth.models import User
        user = User.objects.get(id=user_id)
        
        subject = f'Critical Anomalies Detected - {anomaly_count} Alert(s)'
        message = (
            f'Dear {user.username},\n\n'
            f'Your pipeline monitoring analysis has completed and detected '
            f'{anomaly_count} critical anomaly/anomalies.\n\n'
            f'Please log in to your dashboard to review the details.\n\n'
            f'Best regards,\n'
            f'Pipeline Monitoring System'
        )
        
        send_mail(
            subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [user.email],
            fail_silently=False,
        )
        return {'status': 'success', 'user_id': user_id}
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}


@shared_task
def send_push_notification(user_id: str, notification_id: str):
    """Send push notification to user devices"""
    try:
        from django.contrib.auth.models import User
        from .models import UserDevice
        from django.conf import settings
        
        notification = Notification.objects.get(id=notification_id)
        devices = UserDevice.objects.filter(user_id=user_id, is_active=True)
        
        # In a real implementation, you would use Firebase Cloud Messaging
        # or similar service here
        # For now, we'll just mark it as sent
        notification.is_sent = True
        notification.sent_at = timezone.now()
        notification.save()
        
        return {'status': 'success', 'devices_notified': devices.count()}
    except Exception as exc:
        return {'status': 'error', 'message': str(exc)}

